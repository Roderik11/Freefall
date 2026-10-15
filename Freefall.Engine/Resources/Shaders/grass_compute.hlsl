// ═══════════════════════════════════════════════════════════════════════════
// Grass/Decorator Compute Prepass
//
// Pipeline:
//   0. CS_BakeTerrainNormals: heightmap → R16G16_SNORM normal map (one-time)
//   1. CS_SpawnInstances:     single-pass: camera-centered grid → instance placement + cull → append.
//                             The control texture names decorators (coverage only); each decorator is a
//                             group of variants, and density + clumping are evaluated per variant here.
//                             Card instances near the camera go to a separate buffer, drawn as blade clumps.
//   2. CS_BuildDrawArgs:      reads append counters → writes DispatchMesh indirect args (cards, blade clumps)
//   3. CS_BladeColors:        per card variant, blade colour root → tip from its texture (when variants change)
//
// Output: StructuredBuffer<DecoInstance> consumed by the lean AS/MS in grass.fx and grass_blades.fx
// ═══════════════════════════════════════════════════════════════════════════

#pragma kernel CS_BakeTerrainNormals
#pragma kernel CS_SpawnInstances
#pragma kernel CS_BuildDrawArgs
#pragma kernel CS_BinMeshInstances
#pragma kernel CS_BladeColors

#include "decoration_noise.hlsli"   // DecoClusterMask, DecoHeightFactor

// Push constants (root parameter 0, register b3) — bindless indices only
cbuffer PushConstants : register(b3)
{
    uint DecoratorSlotsIdx;         // slot 0  — SRV
    uint LODTableIdx;               // slot 1  — SRV
    uint MeshRegistryIdx;           // slot 2  — SRV
    uint HeightmapIdx;              // slot 3  — SRV
    uint DecoControlIdx;            // slot 4  — SRV
    uint BakedNormalIdx;            // slot 5  — SRV
    uint BakedNormalUAVIdx;         // slot 6  — UAV (bake kernel)
    uint DecoMapsIdx;               // slot 7  — SRV
    uint DecoInstanceUAVIdx;        // slot 8  — UAV (spawn kernel)
    uint InstanceCounterUAVIdx;     // slot 9  — UAV
    uint DispatchArgsUAVIdx;        // slot 10 — UAV (build args kernel)
    uint MeshDecoInstanceUAVIdx;    // slot 11 — UAV (spawn), SRV (binning)
    uint SortedMeshInstanceUAVIdx;  // slot 12 — UAV (binning kernel)
    uint MeshDrawArgsUAVIdx;        // slot 13 — UAV (binning kernel)
    uint MeshDrawCountUAVIdx;       // slot 14 — UAV (binning kernel)
    uint DrawSortedSRVIdx;          // slot 15 — SRV: → draw cmd slot 4
    uint DrawSlotsSRVIdx;           // slot 16 — SRV: → draw cmd slot 5
    uint DrawLODSRVIdx;             // slot 17 — SRV: → draw cmd slot 6
    uint DrawMeshRegSRVIdx;         // slot 18 — SRV: → draw cmd slot 7
    uint DrawMaterialsSRVIdx;       // slot 19 — SRV: → draw cmd slot 14
    uint DecoratorGroupsIdx;        // slot 20 — SRV: StructuredBuffer<DecoratorGroup> (spawn kernel)
    uint BladeInstanceUAVIdx;       // slot 21 — UAV: card instances near the camera, drawn as blade clumps
    uint BladeColorsUAVIdx;         // slot 22 — UAV: per card variant, blade colour root → tip (CS_BladeColors)
};

// Camera frustum in terrain-local space (root slot 1 → register b0, uploaded by the terrain quadtree pass).
// Normals point out: a point is outside a plane when dot(n, p) + w > 0.
cbuffer FrustumPlanes : register(b0)
{
    float4 Planes[6];
};

// Hi-Z occlusion parameters (root slot 2 → register b1, shared with terrain_quadtree.hlsl)
cbuffer HiZParams : register(b1)
{
    row_major float4x4 OcclusionProjection;
    uint HiZSrvIdx;
    float2 HiZSize;
    uint HiZMipCount;
    float NearPlane;
    uint CullStatsUAVIdx;
    uint FrustumDebugMode;
    float _hiZPad;
};

// Hi-Z occlusion test: project bounding sphere, pick mip, sample depth pyramid.
// Returns true if the sphere is FULLY behind solid geometry (should be culled).
bool IsCellOccluded(float3 worldCenter, float worldRadius)
{
    if (HiZSrvIdx == 0) return false;

    float4 clipCenter = mul(float4(worldCenter, 1.0), OcclusionProjection);
    float3 ndc = clipCenter.xyz / clipCenter.w;
    float2 uv = ndc.xy * float2(0.5, -0.5) + 0.5;

    if (any(uv < 0.0) || any(uv > 1.0)) return false;

    Texture2D<float> hiZ = ResourceDescriptorHeap[HiZSrvIdx];
    float2 mip0Size = HiZSize;

    // Project sphere radius to screen pixels for mip selection.
    // For standard perspective, _m00 * mip0Size.x == _m11 * mip0Size.y,
    // so vertical-only calculation is sufficient.
    float projScale = abs(OcclusionProjection._m11);
    projScale = projScale < 0.001 ? 1.0 : projScale;
    float screenRadius = (worldRadius * projScale / clipCenter.w) * mip0Size.y * 0.5;

    // Conservative +1 mip bias to widen the 4-tap footprint
    float mipLevel = ceil(log2(max(screenRadius*2, 1.0)));
    mipLevel = min(mipLevel, float(HiZMipCount) - 1.0f);

    uint mip = (uint)mipLevel;
    float mipW, mipH, unused;
    hiZ.GetDimensions(mip, mipW, mipH, unused);
    float2 mipSize = float2(mipW, mipH);

    float2 texCoordFloat = uv * mipSize - 0.5;
    int2 baseCoord = int2(texCoordFloat);
    int2 maxCoord = int2(mipSize) - 1;

    float d0 = hiZ.Load(int3(clamp(baseCoord,             int2(0,0), maxCoord), mip));
    float d1 = hiZ.Load(int3(clamp(baseCoord + int2(1,0), int2(0,0), maxCoord), mip));
    float d2 = hiZ.Load(int3(clamp(baseCoord + int2(0,1), int2(0,0), maxCoord), mip));
    float d3 = hiZ.Load(int3(clamp(baseCoord + int2(1,1), int2(0,0), maxCoord), mip));

    float sampledDepth = max(max(d0, d1), max(d2, d3));
    float sphereNearestDepth = clipCenter.w - worldRadius;
    return sphereNearestDepth > sampledDepth;
}

// ─── Per-dispatch parameters ───────────────────────────────────────────────
cbuffer DecoParams : register(b4)
{
    float2 TerrainSize;
    float  MaxHeight;
    float  DecoRadius;
    float3 CamPos;
    float  TileSize;
    float3 TerrainOrigin;
    float  DecorationDensity;
    float  DecoReach;           // farthest any decorator reaches from its root (m)
    float  _decoPad;
    uint   ControlWidth;
    uint   ControlHeight;
    uint   SlotCount;
    int    BaseTileX;
    int    BaseTileZ;
    uint   MaxInstances;
    uint2  HeightmapSize;
    float3 ShadowSweep;         // light direction x longest decorator shadow (0 = shadows off)
    uint   GridSize;            // cells per side of the dispatch (odd: the camera's cell in the middle)
    uint   MaxBladeInstances;
};

#define BLADE_CLUMPS_PER_AS 32  // grass_blades.fx: one AS group launches this many clump MS groups
#define BLADE_COLOR_BANDS 4     // grass_blades.fx: blade colour bands, root to tip

// True when a sphere (terrain-local), or the shadow it throws along ShadowSweep, can reach the camera frustum.
bool CellInFrustum(float3 center, float radius)
{
    [unroll]
    for (uint i = 0; i < 6; i++)
    {
        float d = dot(Planes[i].xyz, center) + Planes[i].w;
        float dSwept = d + dot(Planes[i].xyz, ShadowSweep);
        if (min(d, dSwept) > radius)
            return false;
    }
    return true;
}

// ─── GPU structs (must match grass.fx and TerrainRenderer.cs) ──────────────

// One TerrainDecorator: the slot index stored in the control texture. Its variants are a run of
// DecoratorSlots.
struct DecoratorGroup
{
    uint VariantOffset;
    uint VariantCount;
};

// One decorator variant. Instances carry the index of this record (DecoInstance.SlotIdx).
struct DecoratorSlot
{
    float Density;
    float MinH, MaxH;
    float MinW, MaxW;
    uint LODCount;
    uint LODTableOffset;
    // Root rotation: 9 floats instead of float3x3 to match C# struct packing
    float Rot00, Rot01, Rot02;
    float Rot10, Rot11, Rot12;
    float Rot20, Rot21, Rot22;
    float SlopeBias;
    uint Seed;                  // decorrelates this variant's scatter and noise fields from the others
    uint BandModes;             // distance bands near → far: one mode per byte, 0xFF = no band (Mode out to DecoRadius)
    uint Mode;
    uint TextureIdx;
    float3 HealthyColor;
    float3 DryColor;
    float NoiseSpread;
    uint BandEnds01;            // far ends of bands 0, 1 (m, two halves)
    uint BandEnds23;            // bands 2, 3
    float ClusterScale;         // world size of density clumps in m (0 = off)
    float ClusterAmount;        // 0 = uniform, 1 = full clumps + bare gaps
    uint AlphaClip;             // mesh mode: 1 = alpha-test albedo (foliage), 0 = opaque (rocks, pebbles)
    float HeightNoiseScale;     // world size of short vs. lush patches in m (0 = off)
    float HeightNoiseAmount;    // 0 = none, 1 = 0.4x .. 1.4x height
};

struct LODEntry { uint MeshPartId; float MaxDistance; uint MaterialId; uint _pad; };

struct MeshPartEntry
{
    uint PosBufferIdx, NormBufferIdx, UVBufferIdx, IndexBufferIdx;
    uint BaseIndex, VertexCount;
    uint BoneWeightsBufferIdx, NumBones;
    float BoundsCenterX, BoundsCenterY, BoundsCenterZ, BoundsRadius;
    uint Reserved4, Reserved5, Reserved6, Reserved7, Reserved8, Reserved9;
};

// Per-instance output — consumed by the lean MS in grass.fx
struct DecoInstance
{
    float3 Position;        // world-space base position
    float  Rotation;        // Y-axis rotation angle (radians)
    float3 TerrainNormal;   // pre-sampled from baked normal map
    float  FadeFactor;      // distance fade [0-1]
    float2 Scale;           // (width, height) after density + fade bias
    float2 TerrainUV;       // for ground color sampling in PS
    uint   SlotIdx;         // decorator variant index (DecoratorSlot)
    uint   LOD;             // selected LOD index
    float  InstanceSeed;    // for color variation in PS
    uint   Mode;            // how this instance is drawn (MODE_*): its distance band's mode, else the variant's
};  // 64 bytes — cache line aligned



#define MODE_MESH      0
#define MODE_BILLBOARD 1
#define MODE_CROSS     2
#define MODE_BLADES    3
#define MAX_BANDS      4

// ─── Distance bands (DecoratorVariant.Bands) ──────────────────────────────

uint BandMode(DecoratorSlot s, uint band) { return (s.BandModes >> (8 * band)) & 0xFF; }

// Far end of a band (m), packed as halves two per uint
float BandEnd(DecoratorSlot s, uint band)
{
    uint packed = band < 2 ? s.BandEnds01 : s.BandEnds23;
    return f16tof32((packed >> (16 * (band & 1))) & 0xFFFF);
}

uint BandCount(DecoratorSlot s)
{
    uint n = 0;
    while (n < MAX_BANDS && BandMode(s, n) != 0xFF) n++;
    return n;
}

// How far the variant is drawn: the end of its last band, never past the decoration radius
float VariantRange(DecoratorSlot s)
{
    uint n = BandCount(s);
    return n > 0 ? min(DecoRadius, BandEnd(s, n - 1)) : DecoRadius;
}

// ─── Samplers ──────────────────────────────────────────────────────────────

SamplerState HeightSampler : register(s0);
SamplerState ClampSampler  : register(s2);

// ─── PCG Hash ──────────────────────────────────────────────────────────────

uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint pcg2d(uint2 v)
{
    v = v * uint2(1664525u, 1013904223u) + uint2(1013904223u, 1664525u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v.x ^= v.x >> 16u;
    v.y ^= v.y >> 16u;
    v.x += v.y * 1664525u;
    return v.x ^ v.y;
}

float hash21(float2 p)
{
    uint2 ip = uint2(asuint(p.x), asuint(p.y));
    return float(pcg2d(ip)) / 4294967295.0;
}

float2 hash22(float2 p)
{
    uint2 ip = uint2(asuint(p.x), asuint(p.y));
    uint h0 = pcg2d(ip);
    uint h1 = pcg(h0);
    return float2(float(h0), float(h1)) / 4294967295.0;
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage 0: CS_BakeTerrainNormals
//
// One-time prepass: heightmap → R16G16_SNORM normal map.
// Dispatched once (or when heightmap changes).
// Stores only XZ components; Y is reconstructed as sqrt(1 - x² - z²).
// ═══════════════════════════════════════════════════════════════════════════

[numthreads(8, 8, 1)]
void CS_BakeTerrainNormals(uint3 dtid : SV_DispatchThreadID)
{
    Texture2D heightTex = ResourceDescriptorHeap[HeightmapIdx];
    RWTexture2D<float2> normalOut = ResourceDescriptorHeap[BakedNormalUAVIdx];

    uint hmW = HeightmapSize.x;
    uint hmH = HeightmapSize.y;
    if (dtid.x >= hmW || dtid.y >= hmH) return;

    float2 uv = (float2(dtid.xy) + 0.5) / float2(hmW, hmH);
    float texelStep = 1.0 / float(hmW);

    float hL = heightTex.SampleLevel(HeightSampler, uv + float2(-texelStep, 0), 0).r;
    float hR = heightTex.SampleLevel(HeightSampler, uv + float2( texelStep, 0), 0).r;
    float hD = heightTex.SampleLevel(HeightSampler, uv + float2(0, -texelStep), 0).r;
    float hU = heightTex.SampleLevel(HeightSampler, uv + float2(0,  texelStep), 0).r;

    // Central difference spans two texels (hL-hR, hD-hU)
    float texelWorldSize = TerrainSize.x * texelStep;
    float heightScale = MaxHeight / (2.0 * texelWorldSize);

    float3 n = normalize(float3(
        (hL - hR) * heightScale,
        1.0,
        (hD - hU) * heightScale
    ));

    // Store XZ in R16G16_SNORM; Y is reconstructed as sqrt(1 - x² - z²)
    normalOut[dtid.xy] = float2(n.x, n.z);
}

// ═══════════════════════════════════════════════════════════════════════════
// CS_SpawnInstances — Single-pass instance spawning
//
// Dispatch(N, N, 1) where N = GridSize = cells per side around the camera, walked as a spiral from its cell.
// Each group = one cell, culled as a whole against the camera frustum and Hi-Z. [8,8,1] = 64 threads
// cooperate per cell.
// Thread 0 turns the cell's control data into a list of (variant, instance count) entries in
// groupshared; all threads read from there. Doing the per-variant clump noise once per cell rather
// than once per thread is what keeps it cheap.
// ═══════════════════════════════════════════════════════════════════════════

#define MAX_CELL_ENTRIES 64     // 8 decorators per texel x 8 variants per decorator
#define MAX_VARIANTS 8

groupshared uint  gs_entrySlot[MAX_CELL_ENTRIES];    // variant index (DecoratorSlot)
groupshared uint  gs_entryCount[MAX_CELL_ENTRIES];   // instances to spawn in this cell
groupshared float gs_entryWeight[MAX_CELL_ENTRIES];  // cubed cell weight, 0..1
groupshared uint  gs_entries;

// The i-th cell of a square spiral around the centre cell: ring k holds the 8k cells at Chebyshev distance k.
int2 SpiralCell(uint i)
{
    if (i == 0) return int2(0, 0);
    uint k = (uint)((sqrt((float)i) + 1.0) * 0.5);
    if ((2 * k + 1) * (2 * k + 1) <= i) k++;            // float rounding at ring boundaries
    if ((2 * k - 1) * (2 * k - 1) > i) k--;
    uint j = i - (2 * k - 1) * (2 * k - 1);             // 0 .. 8k-1 along the ring
    uint side = j / (2 * k);
    int t = int(j % (2 * k));
    int r = int(k);
    if (side == 0) return int2(-r + t, -r);
    if (side == 1) return int2(r, -r + t);
    if (side == 2) return int2(r - t, r);
    return int2(-r, r - t);
}

[numthreads(8, 8, 1)]
void CS_SpawnInstances(uint3 gid : SV_GroupID, uint3 gtid3 : SV_GroupThreadID)
{
    uint flatThread = gtid3.y * 8 + gtid3.x;  // 0..63

    // Near cells first: groups launch in order, so the n-th group takes the n-th cell of a spiral around the
    // camera's cell and the instances come out roughly near-to-far. Alpha-tested grass gets little early-Z, but
    // far-to-near still cost ~7% more on Eldermoor (2026-10-15); the plain grid was far-to-near looking south.
    int2 spiral = SpiralCell(gid.y * GridSize + gid.x);
    int cellX = BaseTileX + int(GridSize / 2) + spiral.x;
    int cellZ = BaseTileZ + int(GridSize / 2) + spiral.y;

    // Control texture (dimensions from cbuffer)
    Texture2DArray<uint4> controlTex = ResourceDescriptorHeap[DecoControlIdx];

    // Control map texel — same Y-flip as gputerrain.fx debug overlay
    int cx = cellX;
    int cy = int(ControlHeight) - 1 - cellZ;

    // Check tile validity — no early return before the barrier
    float ts = TileSize;
    float tileWorldX = TerrainOrigin.x + (float(cellX) + 0.5) * ts;
    float tileWorldZ = TerrainOrigin.y + (float(cellZ) + 0.5) * ts;
    float3 camPos = CamPos;
    float dx2 = tileWorldX - camPos.x;
    float dz2 = tileWorldZ - camPos.z;
    float maxR = DecoRadius + ts;

    bool tileValid = (cx >= 0 && cx < int(ControlWidth) && cy >= 0 && cy < int(ControlHeight))
                  && (dx2 * dx2 + dz2 * dz2 <= maxR * maxR);

    // ── Thread 0 reads BAKED control texture and populates groupshared ──
    // The baked control texture (RGBA16_UINT, 2 slices) was built by decoration_prepass.hlsl.
    // Each channel packs (decoratorSlot << 8) | coverage. Up to 8 decorators per texel.
    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    if (flatThread == 0)
    {
        gs_entries = 0;

        // Cull the cell (thread 0 only) against the camera frustum, then against last frame's Hi-Z. Both take
        // terrain-local positions. The frustum sphere sits on the ground at the cell centre and holds the largest
        // decorator; two cells of slack cover the slope across the cell.
        if (tileValid)
        {
            Texture2D heightTex2 = ResourceDescriptorHeap[HeightmapIdx];
            float2 cellLocal = (float2(cellX, cellZ) + 0.5) * ts;
            float h = heightTex2.SampleLevel(HeightSampler, cellLocal / TerrainSize, 0).r * MaxHeight;
            if (!CellInFrustum(float3(cellLocal.x, h + 0.5 * DecoReach, cellLocal.y), 2.0 * ts + DecoReach))
                tileValid = false;
            else if (HiZSrvIdx != 0 && IsCellOccluded(float3(cellLocal.x, h, cellLocal.y), ts * 0.707 + MaxHeight * 0.05))
                tileValid = false;
        }

        if (tileValid)
        {
            // Load baked control data (RGBA16_UINT is uncompressed — Load() is valid)
            uint4 packed0 = controlTex.Load(int4(cx, cy, 0, 0));
            uint4 packed1 = controlTex.Load(int4(cx, cy, 1, 0));

            StructuredBuffer<DecoratorGroup> groups = ResourceDescriptorHeap[DecoratorGroupsIdx];

            // Cell centre in the frame the clump noise is defined in (terrain-local metres, Z measured
            // from the far edge like the control texture rows)
            float2 clumpPos = float2((float(cellX) + 0.5) * ts, TerrainSize.y - (float(cellZ) + 0.5) * ts);

            uint entries = 0;
            for (uint ch = 0; ch < 8; ch++)
            {
                uint packed = (ch < 4) ? packed0[ch] : packed1[ch - 4];
                uint groupIdx = packed >> 8;
                uint coverage = packed & 0xFF;

                if (groupIdx == 255 || coverage == 0) continue;

                DecoratorGroup group = groups[groupIdx];
                uint variantCount = min(group.VariantCount, (uint)MAX_VARIANTS);

                for (uint v = 0; v < variantCount; v++)
                {
                    uint variantIdx = group.VariantOffset + v;
                    DecoratorSlot variant = slots[variantIdx];

                    // A cell past the variant's range (the end of its last band) spawns nothing of it
                    float variantReach = VariantRange(variant) + ts;
                    if (dx2 * dx2 + dz2 * dz2 > variantReach * variantReach) continue;

                    // Coverage is shared by the decorator; natural patchiness is the variant's own
                    float w = float(coverage) / 255.0;
                    if (variant.ClusterScale > 0 && variant.ClusterAmount > 0)
                        w *= lerp(1.0, DecoClusterMask(clumpPos, variant.ClusterScale, variant.Seed), saturate(variant.ClusterAmount));

                    w = w * w * w; // aggressive power curve: crush low weights
                    uint count = min(64u, (uint)(w * 64.0 * variant.Density * DecorationDensity + 0.5));
                    if (count == 0) continue;

                    gs_entrySlot[entries] = variantIdx;
                    gs_entryCount[entries] = count;
                    gs_entryWeight[entries] = w;
                    entries++;
                }
            }
            gs_entries = entries;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (gs_entries == 0) return;

    StructuredBuffer<LODEntry>      lodTbl = ResourceDescriptorHeap[LODTableIdx];

    float tileOriginX = TerrainOrigin.x + float(cellX) * ts;
    float tileOriginZ = TerrainOrigin.y + float(cellZ) * ts;

    RWStructuredBuffer<DecoInstance> outBuffer = ResourceDescriptorHeap[DecoInstanceUAVIdx];
    RWByteAddressBuffer instCounter = ResourceDescriptorHeap[InstanceCounterUAVIdx];
    uint maxInst = MaxInstances;

    Texture2D heightTex = ResourceDescriptorHeap[HeightmapIdx];
    Texture2D<float2> bakedNormals = ResourceDescriptorHeap[BakedNormalIdx];

    // Iterate over the cell's entries (dynamic loop — all threads share the list, no divergence)
    uint entryCount = min(gs_entries, (uint)MAX_CELL_ENTRIES);
    for (uint ci = 0; ci < entryCount; ci++)
    {
        uint slotIdx = gs_entrySlot[ci];
        uint decoInstCount = gs_entryCount[ci];
        float cellWeight = gs_entryWeight[ci];

        DecoratorSlot slot = slots[slotIdx];
        uint bandCount = BandCount(slot);
        float range = VariantRange(slot);

        // Stride loop: 64 threads cooperate to spawn decoInstCount instances
        for (uint instanceIdx = flatThread; instanceIdx < decoInstCount; instanceIdx += 64)
        {
            // Per-instance jitter: seed from integer cell coords to avoid float hash correlation
            uint cellSeed = pcg(uint(cellX) * 1664525u ^ uint(cellZ) * 1013904223u);
            uint instSeed = pcg(cellSeed + instanceIdx * 747796405u + slot.Seed * 2891336453u);
            float rngX = float(instSeed) / 4294967295.0;
            float rngY = float(pcg(instSeed)) / 4294967295.0;
            float2 tileOrig = float2(tileOriginX, tileOriginZ);
            float overshoot = 0;// 0.5f * normalizedWeight;
            float wx = tileOrig.x + (rngX * (1.0 + 2.0 * overshoot) - overshoot) * ts;
            float wz = tileOrig.y + (rngY * (1.0 + 2.0 * overshoot) - overshoot) * ts;

            // Terrain UV
            float2 texelUV = float2(
                (wx - TerrainOrigin.x) / TerrainSize.x,
                (wz - TerrainOrigin.y) / TerrainSize.y
            );
            if (texelUV.x < 0 || texelUV.x > 1 || texelUV.y < 0 || texelUV.y > 1) continue;

            // ── Heightmap sample ──
            float h = heightTex.SampleLevel(HeightSampler, texelUV, 0).r;
            float3 instancePos = float3(wx, TerrainOrigin.z + h * MaxHeight, wz);

            // Distance cull (squared — avoids sqrt for rejected instances)
            float3 toCam = instancePos - camPos;
            float camDistSq = dot(toCam, toCam);
            if (camDistSq >= range * range) continue;

            // ── Terrain normal (R16G16_SNORM — already [-1,1]) ──
            float2 nxz = bakedNormals.SampleLevel(ClampSampler, texelUV, 0).rg;
            float ny = sqrt(max(0.0, 1.0 - nxz.x * nxz.x - nxz.y * nxz.y));
            float3 terrainNormal = normalize(float3(nxz.x, ny, nxz.y));

            // ── LOD selection ──
            uint lod = 0;
            // DIAGNOSTIC: skip LOD culling to test if all 8 slots render
            //uint maxLod = min(slot.LODCount, 8u);
            //for (uint il = 0; il < maxLod; il++)
            //{
            //    if (camDist > lodTbl[slot.LODTableOffset + il].MaxDistance)
            //        lod = il + 1;
            //}
            //if (lod >= slot.LODCount) continue;

            // ── Per-instance scale & rotation ──
            uint scaleSeed = pcg(instSeed + 337u);
            float densityBias = lerp(0.3, 1.0, cellWeight);
            float randH = float(scaleSeed) / 4294967295.0;
            float randW = float(pcg(scaleSeed + 779u)) / 4294967295.0;
            float scaleH = lerp(slot.MinH, slot.MaxH, randH * densityBias);
            float scaleW = lerp(slot.MinW, slot.MaxW, randW * densityBias);

            // Patch-scale height variation, independent of density (lush hollows vs. short grazed patches).
            // Density already couples to height via densityBias (clump fringes stay short).
            if (slot.HeightNoiseScale > 0 && slot.HeightNoiseAmount > 0)
                scaleH *= DecoHeightFactor(texelUV * TerrainSize, slot.HeightNoiseScale, saturate(slot.HeightNoiseAmount), slot.Seed);

            // Distance fade: smoothly shrink in the last 25% of range
            float fadeStart = range * 0.85;
            float camDist = sqrt(camDistSq);
            float fadeFactor = smoothstep(0.0, 1.0, (range - camDist) / (range - fadeStart));
            scaleH *= fadeFactor;
            scaleW *= fadeFactor;

            float instanceRot = float(pcg(instSeed + 731u)) / 4294967295.0 * 6.2831853;
            float instanceSeedVal = float(pcg(instSeed + 991u)) / 4294967295.0;

            // ── Append to output ──
            DecoInstance di;
            di.Position = instancePos;
            di.Rotation = instanceRot;
            di.TerrainNormal = terrainNormal;
            di.FadeFactor = fadeFactor;
            di.Scale = float2(scaleW, scaleH);
            di.TerrainUV = texelUV;
            di.SlotIdx = slotIdx;
            di.LOD = lod;
            di.InstanceSeed = instanceSeedVal;

            // How this instance is drawn: the first of the variant's distance bands it is inside. Each instance
            // switches at its own point within ±15% of a band's end, so a hand-over has no ring; the last band
            // ends at the variant's range (the fade above).
            uint mode = slot.Mode;
            if (bandCount > 0)
            {
                float switchAt = 0.85 + 0.3 * frac(instanceSeedVal * 13.37);
                mode = BandMode(slot, bandCount - 1);
                for (uint band = 0; band + 1 < bandCount; band++)
                {
                    if (camDist < BandEnd(slot, band) * switchAt)
                    {
                        mode = BandMode(slot, band);
                        break;
                    }
                }
            }
            di.Mode = mode;

            if (mode == MODE_MESH)
            {
                // Mesh instances → separate buffer, counter at offset 4
                uint outIdx;
                instCounter.InterlockedAdd(4, 1, outIdx);
                if (outIdx >= maxInst) continue;
                RWStructuredBuffer<DecoInstance> meshBuffer = ResourceDescriptorHeap[MeshDecoInstanceUAVIdx];
                meshBuffer[outIdx] = di;
            }
            else if (mode == MODE_BLADES)
            {
                // Clumps of procedural blades (grass_blades.fx) → own buffer, counter at offset 8
                uint bladeIdx;
                instCounter.InterlockedAdd(8, 1, bladeIdx);
                if (bladeIdx >= MaxBladeInstances) continue;
                RWStructuredBuffer<DecoInstance> bladeBuffer = ResourceDescriptorHeap[BladeInstanceUAVIdx];
                bladeBuffer[bladeIdx] = di;
            }
            else
            {
                // Billboard/Cross → main buffer, counter at offset 0
                uint outIdx;
                instCounter.InterlockedAdd(0, 1, outIdx);
                if (outIdx >= maxInst) continue;
                outBuffer[outIdx] = di;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage 3: CS_BuildDrawArgs
//
// Single-thread: reads instance counter, writes DispatchMesh indirect args.
// ═══════════════════════════════════════════════════════════════════════════

[numthreads(1, 1, 1)]
void CS_BuildDrawArgs(uint3 dtid : SV_DispatchThreadID)
{
    RWByteAddressBuffer instCounter = ResourceDescriptorHeap[InstanceCounterUAVIdx];
    uint instanceCount = instCounter.Load(0);
    instanceCount = min(instanceCount, MaxInstances);

    // Write DispatchMesh indirect args: (groupsX, groupsY, groupsZ)
    // Each MS group handles up to 16 instances (128 threads / 8 verts per instance)
    uint groupsX = (instanceCount + 15) / 16;

    RWByteAddressBuffer argsBuffer = ResourceDescriptorHeap[DispatchArgsUAVIdx];
    argsBuffer.Store(0, groupsX);
    argsBuffer.Store(4, 1u);
    argsBuffer.Store(8, 1u);
    // Store mesh instance count at offset 12 for CS_BinMeshInstances
    uint meshCount = instCounter.Load(4);
    meshCount = min(meshCount, MaxInstances);
    argsBuffer.Store(12, meshCount);

    // Blade clumps: DispatchMesh args at offset 16. Every AS group launches BLADE_CLUMPS_PER_AS clumps, so the
    // last group's tail is padded with empty instances (zero scale), which the MS skips.
    uint bladeCount = min(instCounter.Load(8), MaxBladeInstances);
    uint bladeGroups = (bladeCount + BLADE_CLUMPS_PER_AS - 1) / BLADE_CLUMPS_PER_AS;
    argsBuffer.Store(16, bladeGroups);
    argsBuffer.Store(20, 1u);
    argsBuffer.Store(24, 1u);
    RWStructuredBuffer<DecoInstance> blades = ResourceDescriptorHeap[BladeInstanceUAVIdx];
    for (uint pad = bladeCount; pad < bladeGroups * BLADE_CLUMPS_PER_AS; pad++)
        blades[pad] = (DecoInstance)0;
}

// ═══════════════════════════════════════════════════════════════════════════
// CS_BladeColors
//
// One group per decorator variant: the colour its blade clumps take, root to tip, from its card texture — the
// alpha-weighted mean of the texels in BLADE_COLOR_BANDS height bands (w = coverage). Read at full resolution on a
// 128 x 128 grid: the transparent texels of the grass cards are near-black, so any mip would darken the mean.
// Runs when the variants change.
// ═══════════════════════════════════════════════════════════════════════════

groupshared float4 gs_bandSum[BLADE_COLOR_BANDS][64];

[numthreads(64, 1, 1)]
void CS_BladeColors(uint3 gid : SV_GroupID, uint gtid : SV_GroupIndex)
{
    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    DecoratorSlot slot = slots[gid.x];

    float4 sums[BLADE_COLOR_BANDS];
    [unroll] for (uint b = 0; b < BLADE_COLOR_BANDS; b++) sums[b] = 0;

    if (slot.Mode != MODE_MESH && slot.TextureIdx != 0)
    {
        Texture2D tex = ResourceDescriptorHeap[slot.TextureIdx];
        uint w, h, mips;
        tex.GetDimensions(0, w, h, mips);
        for (uint row = gtid; row < 128; row += 64)
        {
            uint y = (row * h + h / 2) / 128;                               // row 0 = top of the card = tips
            uint band = min(BLADE_COLOR_BANDS - 1, (127 - row) * BLADE_COLOR_BANDS / 128);
            for (uint col = 0; col < 128; col++)
            {
                float4 c = tex.Load(int3((col * w + w / 2) / 128, y, 0));
                [unroll] for (uint b2 = 0; b2 < BLADE_COLOR_BANDS; b2++)
                    if (b2 == band) sums[b2] += float4(c.rgb * c.a, c.a);
            }
        }
    }

    [unroll] for (uint b3 = 0; b3 < BLADE_COLOR_BANDS; b3++) gs_bandSum[b3][gtid] = sums[b3];
    GroupMemoryBarrierWithGroupSync();

    if (gtid < BLADE_COLOR_BANDS)
    {
        float4 s = 0;
        for (uint t = 0; t < 64; t++) s += gs_bandSum[gtid][t];
        RWStructuredBuffer<float4> colors = ResourceDescriptorHeap[BladeColorsUAVIdx];
        colors[gid.x * BLADE_COLOR_BANDS + gtid] = float4(s.a > 0 ? s.rgb / s.a : 0.0.xxx, s.a / (128.0 * 128.0 / BLADE_COLOR_BANDS));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage 4: CS_BinMeshInstances
//
// Groups mesh-mode instances by meshPartId, writes sorted instances and
// builds per-mesh-type DrawInstanced indirect args for BindlessCommandSignature.
//
// Input:  unsorted MeshDecoInstance buffer + mesh instance count
// Output: SortedMeshInstance buffer + BindlessDrawCommand args buffer + draw count
//
// BindlessCommandSignature command layout (72 bytes):
//   [0..55]  14 uint root constants (slots 2-15 of push constants)
//   [56..71] DrawInstancedArguments { VertexCount, InstanceCount, StartVertex, StartInstance }
//
// Push constants (reuses same register b3):
//   MeshDecoInstanceUAVIdx (11) = unsorted mesh instance (UAV in spawn, SRV in binning)
//   Binning-specific slots set per-dispatch from C#:
//     DispatchArgsUAVIdx (10)    = reused: read mesh count from offset 12
//     SortedMeshInstanceUAVIdx   = UAV for sorted output
//     MeshDrawArgsUAVIdx         = UAV for bindless draw commands
//     MeshDrawCountUAVIdx        = UAV for draw count (1 uint)
// ═══════════════════════════════════════════════════════════════════════════

// Binning push constant slots (12-19) are named fields in cbuffer PushConstants above.

#define MAX_MESH_TYPES 32

groupshared uint gs_typeCounts[MAX_MESH_TYPES];
groupshared uint gs_typePartId[MAX_MESH_TYPES];
groupshared uint gs_typeBaseOffset[MAX_MESH_TYPES];
groupshared uint gs_numTypes;
groupshared uint gs_meshCount;

[numthreads(64, 1, 1)]
void CS_BinMeshInstances(uint gtid : SV_GroupThreadID)
{
    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    StructuredBuffer<LODEntry> lodTbl = ResourceDescriptorHeap[LODTableIdx];
    StructuredBuffer<MeshPartEntry> meshReg = ResourceDescriptorHeap[MeshRegistryIdx];

    // Read mesh instance count from argsBuffer offset 12 (written by CS_BuildDrawArgs)
    RWByteAddressBuffer argsBuffer = ResourceDescriptorHeap[DispatchArgsUAVIdx];
    if (gtid == 0)
    {
        gs_meshCount = argsBuffer.Load(12);
        gs_numTypes = 0;
        for (uint i = 0; i < MAX_MESH_TYPES; i++)
        {
            gs_typeCounts[i] = 0;
            gs_typePartId[i] = 0xFFFFFFFF;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint meshCount = gs_meshCount;
    if (meshCount == 0) return;

    StructuredBuffer<DecoInstance> meshInstances = ResourceDescriptorHeap[MeshDecoInstanceUAVIdx];

    // Pass 1: Count instances per meshPartId
    // (single workgroup iterates all instances — fine for terrain decorator counts)
    for (uint i = gtid; i < meshCount; i += 64)
    {
        DecoInstance di = meshInstances[i];
        DecoratorSlot slot = slots[di.SlotIdx];
        LODEntry lod = lodTbl[slot.LODTableOffset + di.LOD];
        uint partId = lod.MeshPartId;

        // Find or register this meshPartId
        uint typeIdx = 0xFFFFFFFF;
        for (uint t = 0; t < MAX_MESH_TYPES; t++)
        {
            uint expected = 0xFFFFFFFF;
            // Try to claim this slot for our partId
            InterlockedCompareExchange(gs_typePartId[t], expected, partId, expected);
            if (expected == 0xFFFFFFFF || expected == partId)
            {
                // We either claimed it or it was already ours
                if (gs_typePartId[t] == partId)
                {
                    typeIdx = t;
                    break;
                }
            }
        }

        if (typeIdx < MAX_MESH_TYPES)
        {
            uint dummy;
            InterlockedAdd(gs_typeCounts[typeIdx], 1, dummy);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Count how many unique types and compute prefix sum (single thread)
    if (gtid == 0)
    {
        uint numTypes = 0;
        uint offset = 0;
        for (uint t = 0; t < MAX_MESH_TYPES; t++)
        {
            if (gs_typePartId[t] == 0xFFFFFFFF) continue;
            gs_typeBaseOffset[t] = offset;
            offset += gs_typeCounts[t];
            gs_typeCounts[t] = 0; // reset for pass 2 scatter
            numTypes++;
        }
        gs_numTypes = numTypes;
    }
    GroupMemoryBarrierWithGroupSync();

    // Pass 2: Scatter instances into sorted buffer
    RWStructuredBuffer<DecoInstance> sortedBuffer = ResourceDescriptorHeap[SortedMeshInstanceUAVIdx];

    for (uint j = gtid; j < meshCount; j += 64)
    {
        DecoInstance di = meshInstances[j];
        DecoratorSlot slot = slots[di.SlotIdx];
        LODEntry lod = lodTbl[slot.LODTableOffset + di.LOD];
        uint partId = lod.MeshPartId;

        // Find type index
        uint typeIdx = 0;
        for (uint t = 0; t < MAX_MESH_TYPES; t++)
        {
            if (gs_typePartId[t] == partId) { typeIdx = t; break; }
        }

        uint localOffset;
        InterlockedAdd(gs_typeCounts[typeIdx], 1, localOffset);
        sortedBuffer[gs_typeBaseOffset[typeIdx] + localOffset] = di;
    }
    GroupMemoryBarrierWithGroupSync();

    // Pass 3: Build per-type BindlessDrawCommand indirect args (single thread)
    // BindlessCommandSignature: 14 root constants (56 bytes) + DrawInstancedArgs (16 bytes) = 72 bytes
    if (gtid == 0)
    {
        RWByteAddressBuffer drawArgsBuffer = ResourceDescriptorHeap[MeshDrawArgsUAVIdx];
        RWByteAddressBuffer drawCountBuffer = ResourceDescriptorHeap[MeshDrawCountUAVIdx];

        uint drawIdx = 0;
        for (uint t = 0; t < MAX_MESH_TYPES; t++)
        {
            if (gs_typePartId[t] == 0xFFFFFFFF) continue;

            uint partId = gs_typePartId[t];
            MeshPartEntry part = meshReg[partId];
            uint baseOffset = gs_typeBaseOffset[t];
            uint instanceCount = gs_typeCounts[t];

            // Write 14 root constants (slots 2-15 of push constants)
            // These map to grass_mesh.fx push constant layout
            uint cmdOffset = drawIdx * 72;
            drawArgsBuffer.Store(cmdOffset + 0,  partId);                // slot 2: MeshPartId
            drawArgsBuffer.Store(cmdOffset + 4,  baseOffset);           // slot 3: InstanceBaseOffset
            drawArgsBuffer.Store(cmdOffset + 8,  DrawSortedSRVIdx);     // slot 4: SortedInstancesIdx
            drawArgsBuffer.Store(cmdOffset + 12, DrawSlotsSRVIdx);      // slot 5: DecoratorSlotsIdx
            drawArgsBuffer.Store(cmdOffset + 16, DrawLODSRVIdx);        // slot 6: LODTableIdx
            drawArgsBuffer.Store(cmdOffset + 20, DrawMeshRegSRVIdx);    // slot 7: MeshRegistryIdx
            drawArgsBuffer.Store(cmdOffset + 24, 0u);                   // slot 8
            drawArgsBuffer.Store(cmdOffset + 28, 0u);                   // slot 9
            drawArgsBuffer.Store(cmdOffset + 32, 0u);                   // slot 10
            drawArgsBuffer.Store(cmdOffset + 36, 0u);                   // slot 11
            drawArgsBuffer.Store(cmdOffset + 40, 0u);                   // slot 12
            drawArgsBuffer.Store(cmdOffset + 44, 0u);                   // slot 13
            drawArgsBuffer.Store(cmdOffset + 48, DrawMaterialsSRVIdx);  // slot 14: MaterialsIdx
            drawArgsBuffer.Store(cmdOffset + 52, 0u);                   // slot 15

            // DrawInstancedArguments: { VertexCount, InstanceCount, StartVertex, StartInstance }
            drawArgsBuffer.Store(cmdOffset + 56, part.VertexCount);  // VertexCountPerInstance (= numIndices)
            drawArgsBuffer.Store(cmdOffset + 60, instanceCount);     // InstanceCount
            drawArgsBuffer.Store(cmdOffset + 64, 0u);                // StartVertexLocation
            drawArgsBuffer.Store(cmdOffset + 68, 0u);                // StartInstanceLocation

            drawIdx++;
        }
        drawCountBuffer.Store(0, drawIdx);
    }
}
