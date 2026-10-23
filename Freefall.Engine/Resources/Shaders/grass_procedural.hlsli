// ═══════════════════════════════════════════════════════════════════════════
// Procedural decorators — cards as blades and plants (shared by grass_blades.fx and grass_plants.fx)
//
// CS_SpawnInstances routes the instances in a variant's procedural distance band (DecoratorVariant.Bands: Blades,
// Flowers, Spikes or Ferns; each switching at its own point within ±15% of the band's end) into a buffer of their own.
// Both effects draw that buffer with the same AS; each takes only its own kinds. Two pipelines because the plants'
// textured parts are alpha-tested, and a shader that can discard costs the blades their early depth test.
// ═══════════════════════════════════════════════════════════════════════════
cbuffer PushConstants : register(b3)
{
    uint DecoratorSlotsIdx;     // 0
    uint BladeInstanceSRV;      // 1: DecoInstance, padded to whole AS groups with zero-scale entries
    uint BladeColorsSRV;        // 2: float4 per variant x COLORS_PER_VARIANT (rgb, coverage)
    uint BladesPerClump;        // 3
    uint CascadeBufferSRVIdx;   // 4: shadow pass only
};

#include "common.fx"
#include "ground_shape.hlsli"

// ─── GPU structs (must match grass_compute.hlsl and TerrainRenderer.cs) ─────

struct DecoratorSlot
{
    float Density;
    float MinH, MaxH;
    float MinW, MaxW;
    uint LODCount;
    uint LODTableOffset;
    float Rot00, Rot01, Rot02;
    float Rot10, Rot11, Rot12;
    float Rot20, Rot21, Rot22;
    float SlopeBias;
    uint Seed;
    uint BandModes;
    uint Mode;
    uint TextureIdx;
    float3 HealthyColor;
    float3 DryColor;
    float NoiseSpread;
    uint BandEnds01;
    uint BandEnds23;
    float ClusterScale;
    float ClusterAmount;
    uint AlphaClip;
    float HeightNoiseScale;
    float HeightNoiseAmount;
    uint PlantPartsIdx;         // Flowers / Spikes / Ferns: parts atlas (Tools/BlenderKits/build_flowers.py), 0 = none
};

struct DecoInstance
{
    float3 Position;
    float  Rotation;
    float3 TerrainNormal;
    float  FadeFactor;
    float2 Scale;           // (width, height) of the card this instance stands in for
    uint   GroundTint;      // what the ground at the root does to the colour (ground_shape.hlsli PackGroundTint)
    float  Kept;            // share of the plants here the spawn's distance thinning kept: survivors grow by 1 / it
    uint   SlotIdx;
    uint   LOD;
    float  InstanceSeed;
    uint   Mode;
};

// ─── Constants ─────────────────────────────────────────────────────────────

#define CLUMPS_PER_AS 32        // = BLADE_CLUMPS_PER_AS in grass_compute.hlsl
#define BLADE_COLOR_BANDS 4     // = grass_compute.hlsl
#define COLORS_PER_VARIANT 6    // = grass_compute.hlsl: 4 height bands, foliage, bloom
#define COLOR_FOLIAGE 4
#define COLOR_BLOOM 5
#define MS_THREADS 128          // each thread writes up to two vertices and two triangles
#define MAX_VERTS 256
#define MAX_PRIMS 256

#define MODE_BLADES  3
#define MODE_FLOWERS 4
#define MODE_SPIKES  5
#define MODE_FERNS   6

SamplerState ClampSampler : register(s2);

// ─── Helpers ───────────────────────────────────────────────────────────────

uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float Rand(inout uint h)
{
    h = pcg(h);
    return float(h) / 4294967295.0;
}

// The scene wind's sway (GrassWind, common.fx) at height fraction t of a part, with a per-part phase so a clump does
// not move as one. 'push' = WindPush at the instance, looked up once per mesh group: per vertex, its gust noise was
// most of the procedural passes' cost.
float3 Wind(float push, float t, float height, float phase)
{
    return GrassSway(push, phase) * (t * t * height);
}

// Triangle 'p' of a strip of 'segs' quads (left/right vertex pairs up from 'baseVertex') closed by a tip vertex
uint3 StripTriangle(uint baseVertex, uint p, uint segs)
{
    if (p >= 2 * (segs - 1))
        return baseVertex + uint3(2 * (segs - 1), 2 * (segs - 1) + 1, 2 * segs);
    uint k = p / 2;
    uint l0 = baseVertex + 2 * k;
    return (p & 1) ? uint3(l0 + 1, l0 + 3, l0 + 2) : uint3(l0, l0 + 1, l0 + 2);
}

// ─── Amplification Shaders ─────────────────────────────────────────────────
// One thread per instance. An instance of this effect's kinds (PROCEDURAL_KINDS, defined by the effect) goes to mesh
// groups, the rest (the other effect's kinds, the padding) nowhere. The effect cuts each instance into units (PackClass:
// a clump's blades, a whole plant) and says how many units of a class fill a mesh group; the units of all instances
// of a class run on as one stream, cut into groups. Small groups and a small payload are what made the blades cheap:
// a group per clump, each declaring room for 256 vertices, kept few groups in flight, and the pass cost the same with
// 8 blades a clump as with 16; declaring 128 halved it (2026-10-23, 17k clumps: 0.26 -> 0.12 ms). A payload naming
// every mesh group (800 bytes for 160 groups, then 416 for 64) cost as much again: the groups are worked out in the MS
// from one word per class, and the instances are 5-bit offsets. Until the same day every instance also launched a
// group in both pipelines and the MS output nothing for the other kinds.

#define MAX_PACK_CLASSES 8
#define LANES_PER_WORD 6        // 5-bit instance offsets

struct ProceduralPayload
{
    uint Base;                              // the AS group's first instance
    uint Lanes[(CLUMPS_PER_AS + LANES_PER_WORD - 1) / LANES_PER_WORD];   // its instances a class after another, as offsets from Base
    uint Classes[MAX_PACK_CLASSES];         // per class: first mesh group | first instance << 8 | instances << 16 | units per group << 24
    uint Units[MAX_PACK_CLASSES / 4];       // per class, a byte: units per instance
};

// The effect's packing: an instance of class 'cls' (0 .. MAX_PACK_CLASSES-1) is 'units' units, 'perGroup' of which
// fill a mesh group; both the same for every instance of a class
uint PackClass(DecoInstance di, out uint units, out uint perGroup);

groupshared ProceduralPayload s_Payload;
groupshared uint s_ClassCount[MAX_PACK_CLASSES];        // instances per class
groupshared uint s_ClassUnits[MAX_PACK_CLASSES];
groupshared uint s_ClassPerGroup[MAX_PACK_CLASSES];
groupshared uint s_Lanes[(CLUMPS_PER_AS + LANES_PER_WORD - 1) / LANES_PER_WORD];
groupshared uint s_GroupCount;

// A mesh group of the payload: its class, the first of its units in the class' stream, how many
void GroupRun(ProceduralPayload p, uint group, out uint cls, out uint firstUnit, out uint units)
{
    cls = 0;
    firstUnit = 0;
    units = 0;
    for (uint c = 0; c < MAX_PACK_CLASSES; c++)
    {
        uint info = p.Classes[c];
        uint firstGroup = info & 0xFF;
        uint perGroup = info >> 24;
        uint total = ((info >> 16) & 0xFF) * ((p.Units[c / 4] >> (8 * (c % 4))) & 0xFF);
        if (perGroup > 0 && group >= firstGroup && group < firstGroup + (total + perGroup - 1) / perGroup)
        {
            cls = c;
            firstUnit = (group - firstGroup) * perGroup;
            units = min(perGroup, total - firstUnit);
        }
    }
}

// The k-th instance of class 'cls' in the payload
uint PayloadInstance(ProceduralPayload p, uint cls, uint k)
{
    uint i = ((p.Classes[cls] >> 8) & 0xFF) + k;
    return p.Base + ((p.Lanes[i / LANES_PER_WORD] >> (5 * (i % LANES_PER_WORD))) & 31);
}

void PackAndDispatch(uint gtid, uint base, bool take, DecoInstance di)
{
    if (gtid < MAX_PACK_CLASSES) s_ClassCount[gtid] = 0;
    if (gtid < (CLUMPS_PER_AS + LANES_PER_WORD - 1) / LANES_PER_WORD) s_Lanes[gtid] = 0;
    GroupMemoryBarrierWithGroupSync();

    uint cls = 0, slot = 0;
    if (take)
    {
        uint units, perGroup;
        cls = min(PackClass(di, units, perGroup), MAX_PACK_CLASSES - 1u);
        s_ClassUnits[cls] = units;                              // the same for every instance of a class
        s_ClassPerGroup[cls] = max(perGroup, 1u);
        InterlockedAdd(s_ClassCount[cls], 1, slot);
    }
    GroupMemoryBarrierWithGroupSync();

    // The classes one after another, each class' stream of units cut into groups
    if (gtid == 0)
    {
        uint start = 0, groups = 0;
        s_Payload.Base = base;
        s_Payload.Units[0] = 0;
        s_Payload.Units[1] = 0;
        for (uint c = 0; c < MAX_PACK_CLASSES; c++)
        {
            uint n = s_ClassCount[c];
            uint units = n > 0 ? s_ClassUnits[c] : 0;
            uint perGroup = n > 0 ? s_ClassPerGroup[c] : 0;
            s_Payload.Classes[c] = groups | (start << 8) | (n << 16) | (perGroup << 24);
            s_Payload.Units[c / 4] |= units << (8 * (c % 4));
            if (n > 0) groups += (n * units + perGroup - 1) / perGroup;
            start += n;
        }
        s_GroupCount = groups;
    }
    GroupMemoryBarrierWithGroupSync();

    if (take)
    {
        uint i = ((s_Payload.Classes[cls] >> 8) & 0xFF) + slot;
        InterlockedOr(s_Lanes[i / LANES_PER_WORD], gtid << (5 * (i % LANES_PER_WORD)));
    }
    GroupMemoryBarrierWithGroupSync();
    if (gtid < (CLUMPS_PER_AS + LANES_PER_WORD - 1) / LANES_PER_WORD) s_Payload.Lanes[gtid] = s_Lanes[gtid];
    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(s_GroupCount, 1, 1, s_Payload);
}

[numthreads(CLUMPS_PER_AS, 1, 1)]
void AS(uint gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint base = gid.x * CLUMPS_PER_AS;
    StructuredBuffer<DecoInstance> instances = ResourceDescriptorHeap[BladeInstanceSRV];
    DecoInstance di = instances[base + gtid];
    PackAndDispatch(gtid, base, di.Scale.y > 0 && PROCEDURAL_KINDS(di.Mode), di);
}

// The camera's instances reach out to the decoration radius, but these decorators cast into cascade 0 only: an
// instance whose root, widened by its size and sway, falls inside cascade 0's square is taken; the rest are not (as
// grass.fx AS_Shadow does for the cards). Until 2026-10-16 every instance of the prepass launched a group in both
// pipelines and the rasterizer clipped them.
[numthreads(CLUMPS_PER_AS, 1, 1)]
void AS_Shadow(uint gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint base = gid.x * CLUMPS_PER_AS;
    StructuredBuffer<DecoInstance> instances = ResourceDescriptorHeap[BladeInstanceSRV];
    DecoInstance di = instances[base + gtid];
    bool take = di.Scale.y > 0 && PROCEDURAL_KINDS(di.Mode);
    if (take)
    {
        StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
        row_major float4x4 vp = cascades[0].VP;
        float4 p = mul(float4(di.Position, 1.0), vp);
        p.xy /= max(p.w, 1e-6);   // orthographic: w = 1
        // Half the width, the height and the wind's sway, with room to spare; a world offset of 'reach' moves
        // clip x/y by at most reach times the length of that column
        float reach = 0.5 * di.Scale.x + 1.3 * di.Scale.y;
        float2 margin = reach * float2(length(float3(vp._11, vp._21, vp._31)), length(float3(vp._12, vp._22, vp._32)));
        take = all(abs(p.xy) <= 1.0 + margin);
    }
    PackAndDispatch(gtid, base, take, di);
}
