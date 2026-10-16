// ═══════════════════════════════════════════════════════════════════════════
// Procedural decorators — what cards become near the camera (shared by grass_blades.fx and grass_plants.fx)
//
// CS_SpawnInstances routes the instances in a variant's procedural distance band (DecoratorVariant.Bands: Blades,
// Flowers or Spikes; each switching at its own point within ±15% of the band's end) into a buffer of their own.
// Both effects draw that buffer with the same AS (one MS group per instance); each MS builds its own kinds and
// outputs nothing for the rest. Two pipelines because the plants' textured parts are alpha-tested, and a shader
// that can discard costs the blades their early depth test.
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
    uint   _groundTintPad;
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

// The scene wind's sway (GrassWind, common.fx) at height fraction t of a part rooted at 'root', with a per-part
// phase so a clump does not move as one
float3 Wind(float3 root, float t, float height, float phase)
{
    return GrassWind(root, t, height, phase);
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

// ─── Amplification Shader ──────────────────────────────────────────────────

struct ProceduralPayload
{
    uint BaseInstance;
};

groupshared ProceduralPayload s_Payload;

// One instance per MS group. The instance buffer is padded to whole AS groups, so every group launches all of them.
[numthreads(1, 1, 1)]
void AS(uint3 gid : SV_GroupID)
{
    s_Payload.BaseInstance = gid.x * CLUMPS_PER_AS;
    DispatchMesh(CLUMPS_PER_AS, 1, 1, s_Payload);
}

// ─── Shadow Amplification Shader ───────────────────────────────────────────
// The camera's instances reach out to the procedural bands' ends, but these decorators cast into cascade 0 only,
// and each of the two pipelines builds only its own kinds. One thread per instance: an instance of this effect's
// kinds (PROCEDURAL_SHADOW_MODE, defined by the effect) whose root, widened by its size and sway, falls inside
// cascade 0's square gets a mesh group; the rest get none (as grass.fx AS_Shadow does for the cards). Until
// 2026-10-16 every instance of the prepass launched a group in both pipelines and the rasterizer clipped them.

struct ProceduralShadowPayload
{
    uint Instances[CLUMPS_PER_AS];
};

groupshared ProceduralShadowPayload s_ShadowPayload;
groupshared uint s_ShadowCount;

[numthreads(CLUMPS_PER_AS, 1, 1)]
void AS_Shadow(uint gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    if (gtid == 0) s_ShadowCount = 0;
    GroupMemoryBarrierWithGroupSync();

    uint inst = gid.x * CLUMPS_PER_AS + gtid;
    StructuredBuffer<DecoInstance> instances = ResourceDescriptorHeap[BladeInstanceSRV];
    DecoInstance di = instances[inst];
    if (di.Scale.y > 0 && PROCEDURAL_SHADOW_MODE(di.Mode))
    {
        StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
        row_major float4x4 vp = cascades[0].VP;
        float4 p = mul(float4(di.Position, 1.0), vp);
        p.xy /= max(p.w, 1e-6);   // orthographic: w = 1
        // Half the width, the height and the wind's sway, with room to spare; a world offset of 'reach' moves
        // clip x/y by at most reach times the length of that column
        float reach = 0.5 * di.Scale.x + 1.3 * di.Scale.y;
        float2 margin = reach * float2(length(float3(vp._11, vp._21, vp._31)), length(float3(vp._12, vp._22, vp._32)));
        if (all(abs(p.xy) <= 1.0 + margin))
        {
            uint slot;
            InterlockedAdd(s_ShadowCount, 1, slot);
            s_ShadowPayload.Instances[slot] = inst;
        }
    }

    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(s_ShadowCount, 1, 1, s_ShadowPayload);
}
