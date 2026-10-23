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
#pragma kernel CS_DecoratorTints

#include "decoration_noise.hlsli"   // DecoClusterMask, DecoHeightFactor
#include "ground_shape.hlsli"       // GroundShape, GroundShapeTint, PackGroundTint
#include "terrain_holes.hlsli"      // TerrainHoleAt
#include "weather_cover.hlsli"      // ShelterTexels: how much the cover map keeps the snow off a plant's root

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
    uint DecoratorTintsUAVIdx;      // slot 23 — UAV: per decorator, the colour the ground under it takes (CS_DecoratorTints)
    uint SplatControlIdx;           // slot 24 — SRV: the terrain's layer weights (Texture2DArray, 4 layers per slice)
    uint LayerPlantTintIdx;         // slot 25 — SRV: StructuredBuffer<float4>, TerrainLayer.PlantTint per channel
    uint SplatLayerCount;           // slot 26 — layers in the palette (0 = no plant tint)
    uint HoleTexIdx;                // slot 27 — SRV: R8 hole mask (HoleStamp, terrain_holes.hlsli), 0 = no holes
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
    // The snow lying on the ground buries what grows in it (SnowBurial): Weather.SnowCover, and the cover map's two
    // finest levels (ShelterMap: left, top, size; its map, 0 = not drawn) for where crowns and roofs keep it off
    float  SnowCover;
    float  ShelterMinY;
    float  ShelterRangeY;
    float3 ShelterLevel0;
    uint   ShelterMap0;
    float3 ShelterLevel1;
    uint   ShelterMap1;
};

#define BLADE_CLUMPS_PER_AS 32  // grass_blades.fx: one AS group launches this many clump MS groups
#define BLADE_COLOR_BANDS 4     // grass_blades.fx: blade colour bands, root to tip
#define COLORS_PER_VARIANT 6    // CS_BladeColors: the height bands, then foliage and bloom

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
    uint PlantPartsIdx;         // Flowers / Spikes: parts atlas (grass_plants.fx), 0 = none
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
    uint   GroundTint;      // what the ground at the root does to the plant's colour (PackGroundTint): layer, shape
    float  Kept;            // share of the plants here the distance thinning kept (1 = all): the survivors grow by 1 / it
    uint   SlotIdx;         // decorator variant index (DecoratorSlot)
    uint   LOD;             // selected LOD index
    float  InstanceSeed;    // for color variation in PS
    uint   Mode;            // how this instance is drawn (MODE_*): its distance band's mode, else the variant's
};  // 64 bytes — cache line aligned



#define MODE_MESH      0
#define MODE_BILLBOARD 1
#define MODE_CROSS     2
#define MODE_BLADES    3        // MODE_BLADES and up: procedural, drawn by grass_blades.fx
#define MODE_FLOWERS   4
#define MODE_SPIKES    5
#define MODE_FERNS     6
#define MAX_BANDS      4

// Distance thinning of the procedural plants (CS_SpawnInstances): how many of one variant's plants a steradian of view
// holds before they thin out, and the flattest view of the ground counted (a grazing view would keep next to nothing)
#define THIN_PLANTS_PER_SR 20000.0
#define THIN_MIN_COS 0.02

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

// ─── Distance thinning (CS_SpawnInstances) ─────────────────────────────────

bool ThinningMode(uint mode) { return mode >= MODE_BLADES && mode <= MODE_SPIKES; }

// Whether every instance of the variant is a kind the thinning takes (whatever band it is in)
bool AlwaysThins(DecoratorSlot s)
{
    uint n = BandCount(s);
    if (n == 0) return ThinningMode(s.Mode);
    for (uint b = 0; b < n; b++)
        if (!ThinningMode(BandMode(s, b))) return false;
    return true;
}

// The share of a variant's plants kept at squared distance 'distSq' on ground facing the view, before saturating:
// THIN_PLANTS_PER_SR over the plants a steradian there holds. 'weight' = the cell's highest weight (cubed coverage x
// clumps): the same for every plant of the cell, so a cell can tell how many of its candidates any point keeps.
float ThinFacing(float weight, DecoratorSlot s, float distSq)
{
    float density = weight * 64.0 * s.Density * DecorationDensity / (TileSize * TileSize);   // plants per m²
    return THIN_PLANTS_PER_SR / max(density * distSq, 1e-6);
}

// Every variant thins out over the last 30% of its range: the share kept falls linearly to nothing
float EndKeep(DecoratorSlot s, float dist)
{
    float range = VariantRange(s);
    return saturate((range - dist) / (range * 0.3));
}

// ─── Samplers ──────────────────────────────────────────────────────────────

SamplerState HeightSampler : register(s0);
SamplerState ClampSampler  : register(s2);

// ─── Snow ──────────────────────────────────────────────────────────────────

// How much of a plant's height the snow lying at its root buries, 0..1 (`vary` 0..1: this plant's own share, so they
// do not all end at one line): the snow cover as far as the fall reaches the ground there (the cover map's share over
// the root, as the ground's own snow takes it: a crown lets half through, a roof none). Grass and flowers stood
// coloured in the snow out to the decoration's radius, and the white ground past it drew a line across the land (user,
// 2026-10-23)
float SnowBurial(float3 root, float vary)
{
    if (SnowCover <= 0.0)
        return 0.0;
    WeatherShelter s = (WeatherShelter)0;
    s.MinY = ShelterMinY;
    s.RangeY = ShelterRangeY;
    ShelterLevel l;
    l.Left = ShelterLevel0.x;
    l.Top = ShelterLevel0.y;
    l.Size = ShelterLevel0.z;
    l.Map = ShelterMap0;
    if (l.Size <= 0.0 || ShelterInside(l, root.xz) <= 0.0)
    {
        l.Left = ShelterLevel1.x;
        l.Top = ShelterLevel1.y;
        l.Size = ShelterLevel1.z;
        l.Map = ShelterMap1;
    }
    float covered = 0.0;
    if (l.Size > 0.0 && l.Map != 0u)
    {
        float4 tops, shares;
        float2 f;
        ShelterTexels(s, l, ClampSampler, root.xz, tops, shares, f);
        covered = WeatherBilinear(shares * step(root.y + 0.5, tops), f);
    }
    float lying = SnowCover * (1.0 - covered);
    return smoothstep(0.15, 0.85, lying * lerp(0.8, 1.2, vary));
}

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

#define MAX_DECORATORS 32       // TerrainBaker.MaxDecorators: decorator indices in the control texture

groupshared uint  gs_entrySlot[MAX_CELL_ENTRIES];    // variant index (DecoratorSlot)
groupshared uint  gs_entryStart[MAX_CELL_ENTRIES + 1]; // first candidate of each entry in the cell's list; [entries] = total
groupshared uint  gs_entryGroup[MAX_CELL_ENTRIES];   // its decorator (row of gs_cov)
groupshared float gs_entryKeep[MAX_CELL_ENTRIES];    // share of the candidates kept at the highest weight (expected / count)
groupshared float gs_entryInvMax[MAX_CELL_ENTRIES];  // 1 / that weight: a survivor is kept with probability weight / it
groupshared float gs_entryEven[MAX_CELL_ENTRIES];    // >= 0: coverage, even around the cell; -1 bilinear; -2 + edge offset
groupshared uint  gs_entryCount[MAX_CELL_ENTRIES];   // candidates the cell has (the thinning ranks them); it tries the first of them
groupshared uint  gs_entryThins[MAX_CELL_ENTRIES];   // 1: every kind the variant draws thins with distance (AlwaysThins)
groupshared uint  gs_entries;
groupshared uint  gs_cellValid;
groupshared uint  gs_groupMask;                      // bit N: decorator N covers one of the 3x3 texels
groupshared float gs_cov[MAX_DECORATORS][9];         // coverage 0..1 per decorator at the 3x3 texel centres, row = z offset

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

// The layers' plant tints (TerrainLayer.PlantTint) where they show, weighted as the terrain surface weights the
// layers: a later channel covers the ones before it (gputerrain.fx, without its height blend). Ground no layer
// covers tints nothing. controlUV: control-array space (Y flipped against terrain UV).
float3 LayerPlantTint(float2 controlUV)
{
    Texture2DArray controlMaps = ResourceDescriptorHeap[SplatControlIdx];
    StructuredBuffer<float4> plantTints = ResourceDescriptorHeap[LayerPlantTintIdx];
    uint layerCount = min(SplatLayerCount, 32u);
    float3 tint = 0;
    float remaining = 1.0;
    for (int slice = int((layerCount + 3) / 4) - 1; slice >= 0; slice--)
    {
        float4 w = controlMaps.SampleLevel(ClampSampler, float3(controlUV, slice), 0);
        for (int c = 3; c >= 0; c--)
        {
            uint layer = uint(slice) * 4 + uint(c);
            if (layer >= layerCount) continue;
            tint += plantTints[layer].rgb * (w[c] * remaining);
            remaining *= saturate(1.0 - w[c]);
        }
    }
    return tint + remaining.xxx;
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
        gs_cellValid = tileValid ? 1u : 0u;
        gs_groupMask = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (gs_cellValid == 0) return;      // the same for the whole group: culled cells skip the two barriers below

    // The cell's 3x3 control texels, one thread each. An instance looks its coverage up bilinearly between the
    // texel centres around it, so a patch of decorators ends in a ramp instead of a staircase of 1 m squares that
    // are each all in or all out. Each thread owns one column of the per-decorator table (clears it, then writes its
    // texel's decorators), so nothing is merged serially: a thread-0 merge of the 72 channels cost as much as the old
    // spawn did. (RGBA16_UINT is uncompressed: Load is valid; edges clamp.)
    if (flatThread < 9)
    {
        for (uint z = 0; z < MAX_DECORATORS; z++) gs_cov[z][flatThread] = 0;
        int2 d = int2(flatThread % 3, flatThread / 3) - 1;                 // cell offset in x, z
        int2 texel = clamp(int2(cx + d.x, cy - d.y), int2(0, 0), int2(int(ControlWidth) - 1, int(ControlHeight) - 1));
        uint4 s0 = controlTex.Load(int4(texel, 0, 0));
        uint4 s1 = controlTex.Load(int4(texel, 1, 0));
        [unroll] for (uint ch = 0; ch < 8; ch++)
        {
            uint packed = ch < 4 ? s0[ch] : s1[ch - 4];
            uint groupIdx = packed >> 8;
            uint coverage = packed & 0xFF;
            if (groupIdx >= MAX_DECORATORS || coverage == 0) continue;
            gs_cov[groupIdx][flatThread] = float(coverage) / 255.0;
            InterlockedOr(gs_groupMask, 1u << groupIdx);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Thread 0: per decorator present and per variant, how many candidates the cell tries. The count is made for the
    // highest weight any point of the cell can look up; each candidate is then kept with probability (its own weight /
    // that), see below.
    if (flatThread == 0)
    {
        StructuredBuffer<DecoratorGroup> groups = ResourceDescriptorHeap[DecoratorGroupsIdx];

        // Cell centre in the frame the clump noise is defined in (terrain-local metres, Z measured from the far edge
        // like the control texture rows)
        float2 clumpPos = float2((float(cellX) + 0.5) * ts, TerrainSize.y - (float(cellZ) + 0.5) * ts);

        uint entries = 0;
        uint candidates = 0;
        uint present = gs_groupMask;
        while (present != 0)
        {
            uint gk = firstbitlow(present);
            present &= present - 1;

            float coverMax = 0, coverMin = 1;
            for (uint gt = 0; gt < 9; gt++)
            {
                coverMax = max(coverMax, gs_cov[gk][gt]);
                coverMin = min(coverMin, gs_cov[gk][gt]);
            }

            DecoratorGroup group = groups[gk];
            uint variantCount = min(group.VariantCount, (uint)MAX_VARIANTS);
            for (uint v = 0; v < variantCount && entries < MAX_CELL_ENTRIES; v++)
            {
                uint variantIdx = group.VariantOffset + v;
                DecoratorSlot variant = slots[variantIdx];

                // A cell past the variant's range (the end of its last band) spawns nothing of it
                float variantReach = VariantRange(variant) + ts;
                if (dx2 * dx2 + dz2 * dz2 > variantReach * variantReach) continue;

                // Clumps are evaluated per instance; for the count, the cell centre's plus a margin for how far the
                // field moves within a cell (an underestimate only thins a steep clump edge slightly)
                float clumpMax = 1.0;
                if (variant.ClusterScale > 0 && variant.ClusterAmount > 0)
                    clumpMax = lerp(1.0, saturate(DecoClusterMask(clumpPos, variant.ClusterScale, variant.Seed) + 0.35), saturate(variant.ClusterAmount));

                float wMax = coverMax * clumpMax;
                wMax = wMax * wMax * wMax;          // density = weight cubed: thin coverage thins out fast
                float expected = wMax * 64.0 * variant.Density * DecorationDensity;
                uint count = min(64u, (uint)ceil(expected - 0.001));
                if (count == 0) continue;

                // The distance thinning keeps the lowest ranks, and a candidate's rank grows with its index: the cell
                // tries only as many as its nearest point (horizontally: never farther than in 3D) can keep
                uint tried = count;
                bool thins = AlwaysThins(variant);
                gs_entryThins[entries] = thins ? 1u : 0u;
                if (thins)
                {
                    float nearest = max(0.0, sqrt(dx2 * dx2 + dz2 * dz2) - 0.71 * ts);
                    float keepMax = saturate(ThinFacing(wMax, variant, nearest * nearest)) * EndKeep(variant, nearest);
                    tried = min(count, (uint)ceil(keepMax * count));
                    if (tried == 0) continue;
                }

                gs_entrySlot[entries] = variantIdx;
                gs_entryStart[entries] = candidates;
                gs_entryCount[entries] = count;
                candidates += tried;
                gs_entryGroup[entries] = gk;
                // How much lookup the instances need: none inside a patch (coverage within 0.03 everywhere around, the
                // mean of the extremes), plain bilinear where it changes gently, and the edge offset (4 noises per
                // instance) only across a real edge. Offset everywhere doubled the spawn's cost on Eldermoor.
                float spread = coverMax - coverMin;
                gs_entryEven[entries] = spread < 0.03 ? 0.5 * (coverMax + coverMin) : (spread < 0.2 ? -1.0 : -2.0);
                gs_entryKeep[entries] = min(expected, 64.0) / float(count);
                gs_entryInvMax[entries] = 1.0 / wMax;
                entries++;
            }
        }
        gs_entries = entries;
        gs_entryStart[entries] = candidates;
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

    // All of the cell's candidates as one list, 64 threads over it. One pass per variant left most threads idle in
    // each (a sparse variant has a candidate or two) while every pass still paid a full round of texture latency.
    uint entryCount = min(gs_entries, (uint)MAX_CELL_ENTRIES);
    uint totalCandidates = gs_entryStart[entryCount];
    for (uint cand = flatThread; cand < totalCandidates; cand += 64)
    {
        uint ci = 0;
        while (ci + 1 < entryCount && gs_entryStart[ci + 1] <= cand) ci++;
        uint instanceIdx = cand - gs_entryStart[ci];
        uint slotIdx = gs_entrySlot[ci];
        uint coverGroup = gs_entryGroup[ci];
        float keepShare = gs_entryKeep[ci];
        float invMaxWeight = gs_entryInvMax[ci];
        float evenCover = gs_entryEven[ci];

        DecoratorSlot slot = slots[slotIdx];
        uint bandCount = BandCount(slot);
        float range = VariantRange(slot);

        {
            // Per-instance jitter: seed from integer cell coords to avoid float hash correlation
            uint cellSeed = pcg(uint(cellX) * 1664525u ^ uint(cellZ) * 1013904223u);
            uint instSeed = pcg(cellSeed + instanceIdx * 747796405u + slot.Seed * 2891336453u);
            // First half of the keep test (the count was rounded up): before any noise is evaluated, so the extra
            // candidate a sparse variant gets in every cell costs next to nothing
            if (float(pcg(instSeed + 5099u)) / 4294967295.0 >= keepShare) continue;
            float rngX = float(instSeed) / 4294967295.0;
            float rngY = float(pcg(instSeed)) / 4294967295.0;
            float2 tileOrig = float2(tileOriginX, tileOriginZ);
            float overshoot = 0;// 0.5f * normalizedWeight;
            float wx = tileOrig.x + (rngX * (1.0 + 2.0 * overshoot) - overshoot) * ts;
            float wz = tileOrig.y + (rngY * (1.0 + 2.0 * overshoot) - overshoot) * ts;

            // The candidate's rank in the distance thinning (below): its index in the cell's list, jittered. Where
            // every kind the variant draws thins, the share its spot keeps on ground facing the view (the most it can
            // keep) is known before any noise or texture is read.
            float rank = (float(instanceIdx) + float(pcg(instSeed + 4243u)) / 4294967295.0) / float(gs_entryCount[ci]);
            if (gs_entryThins[ci] != 0)
            {
                float2 flat = float2(wx, wz) - camPos.xz;
                float flatSq = dot(flat, flat);
                if (rank >= saturate(ThinFacing(1.0 / invMaxWeight, slot, flatSq)) * EndKeep(slot, sqrt(flatSq))) continue;
            }

            // Terrain UV
            float2 texelUV = float2(
                (wx - TerrainOrigin.x) / TerrainSize.x,
                (wz - TerrainOrigin.y) / TerrainSize.y
            );
            if (texelUV.x < 0 || texelUV.x > 1 || texelUV.y < 0 || texelUV.y > 1) continue;

            // Weight here: the decorator's coverage bilinear between the 3x3 texel centres (grid 0..2 = the centres of
            // cell offsets -1..1), looked up a little off the instance (DecoEdgeWarp, also in the ground tint) so a
            // patch edge wobbles; times the variant's clumps here; cubed. Kept with probability weight / the cell's
            // highest weight, which (with the first half above) makes the density weight-cubed x Density everywhere,
            // down to a ragged, thinning edge.
            float2 localXZ = float2(wx - TerrainOrigin.x, wz - TerrainOrigin.y);
            float w = evenCover;
            if (w < 0)
            {
                float2 offset = evenCover < -1.5 ? DecoEdgeWarp(localXZ) / ts : 0.0.xx;
                float2 g = clamp(float2(rngX, rngY) + offset + 0.5, 0.0, 2.0);
                uint2 g0 = min(uint2(g), uint2(1, 1));
                float2 gf = g - float2(g0);
                uint gi = g0.y * 3 + g0.x;
                w = lerp(lerp(gs_cov[coverGroup][gi], gs_cov[coverGroup][gi + 1], gf.x),
                         lerp(gs_cov[coverGroup][gi + 3], gs_cov[coverGroup][gi + 4], gf.x), gf.y);
            }
            if (slot.ClusterScale > 0 && slot.ClusterAmount > 0)
                w *= lerp(1.0, DecoClusterMask(float2(localXZ.x, TerrainSize.y - localXZ.y), slot.ClusterScale, slot.Seed), saturate(slot.ClusterAmount));
            float weight = w * w * w;
            if (float(pcg(instSeed + 6211u)) / 4294967295.0 >= weight * invMaxWeight) continue;

            // Nothing grows where a hole opens the terrain (the edge the terrain is cut at)
            if (TerrainHoleAt(HoleTexIdx, ClampSampler, localXZ, TerrainSize)) continue;

            // ── Heightmap sample ──
            float h = heightTex.SampleLevel(HeightSampler, texelUV, 0).r;
            float3 instancePos = float3(wx, TerrainOrigin.z + h * MaxHeight, wz);

            // Distance cull (squared — avoids sqrt for rejected instances)
            float3 toCam = instancePos - camPos;
            float camDistSq = dot(toCam, toCam);
            if (camDistSq >= range * range) continue;
            float camDist = sqrt(camDistSq);
            float instanceSeedVal = float(pcg(instSeed + 991u)) / 4294967295.0;

            // How this instance is drawn: the first of the variant's distance bands it is inside. Each instance
            // switches at its own point within ±15% of a band's end, so a hand-over has no ring; the last band
            // ends at the variant's range (the thinning below).
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

            // Thinning. Procedural plants (blades, flowers, spikes) are drawn in full until they would crowd the
            // view: past THIN_PLANTS_PER_SR of a variant's plants per steradian (the cell's density times the solid
            // angle a square metre of ground fills, which falls with the square of the distance and with how flat
            // the view meets the ground) the share kept falls to that, and the survivors grow to cover for the ones
            // dropped (grass_blades.fx, grass_plants.fx: di.Kept). The meadow keeps its coverage and colour while
            // its plants get fewer and bigger, so nothing has to switch to a card (Cook et al. 2007, "Stochastic
            // Simplification of Aggregate Detail"). Every variant also thins out over the last 30% of its range
            // (density falls linearly to nothing, into the sward). Each plant has a rank: it is there while the
            // share kept is above it and shrinks away over the last quarter of the share before it leaves, so no
            // plant pops and no two leave at the same distance. (Until 2026-10-23 grass switched to cross cards at
            // 15 m and flowers at 20 m: the cards read as different plants. Instances dropped out at random in the
            // last 30%, popping, and all survivors shrank in the last 8%.)
            bool thins = ThinningMode(mode);
            float thinFacing = thins ? ThinFacing(1.0 / invMaxWeight, slot, camDistSq) : 1.0;   // ground facing the view
            float endKeep = EndKeep(slot, camDist);
            if (rank >= saturate(thinFacing) * endKeep) continue;      // before the normal: ground at an angle keeps less

            // ── Terrain normal (R16G16_SNORM — already [-1,1]) ──
            float2 nxz = bakedNormals.SampleLevel(ClampSampler, texelUV, 0).rg;
            float ny = sqrt(max(0.0, 1.0 - nxz.x * nxz.x - nxz.y * nxz.y));
            float3 terrainNormal = normalize(float3(nxz.x, ny, nxz.y));

            float thin = thins ? saturate(thinFacing * max(abs(dot(toCam, terrainNormal)) / camDist, THIN_MIN_COS)) : 1.0;
            float keep = thin * endKeep;
            if (rank >= keep) continue;
            float fadeFactor = smoothstep(0.0, 1.0, (keep - rank) / (0.25 * keep));

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
            float densityBias = lerp(0.3, 1.0, weight);       // thin grass at a patch edge is short grass too
            float randH = float(scaleSeed) / 4294967295.0;
            float randW = float(pcg(scaleSeed + 779u)) / 4294967295.0;
            float scaleH = lerp(slot.MinH, slot.MaxH, randH * densityBias);
            float scaleW = lerp(slot.MinW, slot.MaxW, randW * densityBias);

            // Patch-scale height variation, independent of density (lush hollows vs. short grazed patches).
            // Density already couples to height via densityBias (clump fringes stay short).
            if (slot.HeightNoiseScale > 0 && slot.HeightNoiseAmount > 0)
                scaleH *= DecoHeightFactor(texelUV * TerrainSize, slot.HeightNoiseScale, saturate(slot.HeightNoiseAmount), slot.Seed);

            // A plant the thinning is about to drop shrinks away instead of popping
            scaleH *= fadeFactor;
            scaleW *= fadeFactor;

            // What the ground here does to the plant's colour: the layers' plant tints (dry grass yellows what grows
            // on it), then the shape of the land as the terrain's own colour takes it (ground_shape.hlsli), so the
            // pattern of the bare ground past the decoration radius runs on through the sward
            float3 groundTint = 1.0.xxx;
            if (SplatLayerCount > 0 && SplatControlIdx != 0 && LayerPlantTintIdx != 0)
                groundTint = LayerPlantTint(float2(texelUV.x, 1.0 - texelUV.y));
            groundTint *= GroundShapeTint(GroundShape(heightTex, ClampSampler, texelUV, 1.0 / float(HeightmapSize.x), MaxHeight),
                                          smoothstep(0.7, 0.9, terrainNormal.y));
            groundTint *= SwardGrain(localXZ);       // the sward's grain, as the ground under it from afar

            float instanceRot = float(pcg(instSeed + 731u)) / 4294967295.0 * 6.2831853;

            // ── Append to output ──
            DecoInstance di;
            di.Position = instancePos;
            di.Rotation = instanceRot;
            di.TerrainNormal = terrainNormal;
            di.FadeFactor = fadeFactor;
            di.Scale = float2(scaleW, scaleH);
            di.GroundTint = PackGroundTint(groundTint);
            di.Kept = thin;
            di.SlotIdx = slotIdx;
            di.LOD = lod;
            di.InstanceSeed = instanceSeedVal;
            di.Mode = mode;

            // Under the snow: as much shorter as it buries them, gone once it nearly does. Not the meshes: stones and
            // bushes stand out of it
            if (mode != MODE_MESH && SnowCover > 0.0)
            {
                float buried = SnowBurial(instancePos, frac(instanceSeedVal * 7.31));
                if (buried > 0.92) continue;
                di.Scale.y *= 1.0 - buried;
            }

            if (mode == MODE_MESH)
            {
                // Mesh instances → separate buffer, counter at offset 4
                uint outIdx;
                instCounter.InterlockedAdd(4, 1, outIdx);
                if (outIdx >= maxInst) continue;
                RWStructuredBuffer<DecoInstance> meshBuffer = ResourceDescriptorHeap[MeshDecoInstanceUAVIdx];
                meshBuffer[outIdx] = di;
            }
            else if (mode >= MODE_BLADES)
            {
                // Procedural (grass_blades.fx: blade clumps, flowers, spikes) → own buffer, counter at offset 8
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
// One group per decorator variant: the colours its procedural stand-ins (grass_blades.fx) take, from its card
// texture. COLORS_PER_VARIANT float4 (rgb, coverage):
//   0..3  alpha-weighted mean of all texels in BLADE_COLOR_BANDS height bands, root → tip (grass blades)
//   4     foliage: the green-dominant texels (stems and leaves of flowers)
//   5     bloom: the other texels, weighted by brightness² so dark stems and shading barely count
// Read at full resolution on a 128 x 128 grid: the transparent texels of the cards are near-black, so any mip
// would darken the means. Runs when the variants change.
// ═══════════════════════════════════════════════════════════════════════════

groupshared float4 gs_colorSum[COLORS_PER_VARIANT][64];   // (rgb x weight, weight)
groupshared float  gs_coverage[COLORS_PER_VARIANT][64];

[numthreads(64, 1, 1)]
void CS_BladeColors(uint3 gid : SV_GroupID, uint gtid : SV_GroupIndex)
{
    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    DecoratorSlot slot = slots[gid.x];

    float4 sums[COLORS_PER_VARIANT];
    float coverage[COLORS_PER_VARIANT];
    [unroll] for (uint b = 0; b < COLORS_PER_VARIANT; b++) { sums[b] = 0; coverage[b] = 0; }

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
                {
                    if (b2 == band)
                    {
                        sums[b2] += float4(c.rgb * c.a, c.a);
                        coverage[b2] += c.a;
                    }
                }

                float brightest = max(c.r, max(c.g, c.b));
                if (c.g > max(c.r, c.b) * 1.05)
                {
                    sums[4] += float4(c.rgb * c.a, c.a);
                    coverage[4] += c.a;
                }
                else
                {
                    float wBloom = c.a * brightest * brightest;
                    sums[5] += float4(c.rgb * wBloom, wBloom);
                    coverage[5] += c.a;
                }
            }
        }
    }

    [unroll] for (uint b3 = 0; b3 < COLORS_PER_VARIANT; b3++)
    {
        gs_colorSum[b3][gtid] = sums[b3];
        gs_coverage[b3][gtid] = coverage[b3];
    }
    GroupMemoryBarrierWithGroupSync();

    if (gtid < COLORS_PER_VARIANT)
    {
        float4 s = 0;
        float cover = 0;
        for (uint t = 0; t < 64; t++)
        {
            s += gs_colorSum[gtid][t];
            cover += gs_coverage[gtid][t];
        }
        float samples = 128.0 * 128.0 / (gtid < BLADE_COLOR_BANDS ? BLADE_COLOR_BANDS : 1);
        RWStructuredBuffer<float4> colors = ResourceDescriptorHeap[BladeColorsUAVIdx];
        colors[gid.x * COLORS_PER_VARIANT + gtid] = float4(s.a > 0 ? s.rgb / s.a : 0.0.xxx, cover / samples);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CS_DecoratorTints
//
// One thread per decorator: the colour the ground under it takes (gputerrain.fx), as its plants look from afar —
// each card variant's mean colour (CS_BladeColors' height bands, as the card PS shades it: x 0.8 average
// brightness x HealthyColor), weighted by the opaque area it puts on a square metre (one control cell) of fully
// covered ground: 64 x Density x DecorationDensity instances times the leaf area of one. A card is two quads of mean
// W x H x the texture's coverage; a fern (Ferns band) FERN_FRONDS fronds of its length (instance height) and the
// atlas frond's width, x the frond's coverage, seen from above. Mesh variants do not count.
// Two float4 per decorator:
//   [2g]     rgb = colour, a = the weight of its unclumped variants (gputerrain.fx saturates the sum)
//   [2g + 1] the clumps of its heaviest strongly clumped variant (ClusterAmount > 0.5): scale, seed x 2 + amount,
//            its range, weight. The terrain evaluates the same clump field (DecoClusterMask) as the spawn, so from
//            afar the ground shows the patches the plants grow in instead of an even wash, and takes over exactly as
//            the spawn thins the instances out (the last 30% of the range): no visible band where the plants end.
//            Weakly clumped variants (grass, 0.3..0.45) count as even.
// Runs after CS_BladeColors, when the variants change. (Until 2026-10-15 the colour was also halved, for vegetation
// then lit at 0.4x by a 1.14-for-pi in light_directional_cs.hlsl; with that fixed, the ground past the cards
// rendered half as bright as the cards before it. The fern area was first taken as two cards: ~5x too little.)
// ═══════════════════════════════════════════════════════════════════════════

// = grass_plants.fx: fronds per fern, the atlas frond's half-width (u) and its squeeze. Mean frond area: lengths
// lerp(1.1, 0.55, young) x the height, squared and averaged = 0.706 x H^2; seen from above at ~0.8 of it.
#define FERN_FRONDS_TINT 10.0
#define FERN_FROND_HALF_U 0.245
#define FERN_FROND_WIDTH 0.85

[numthreads(64, 1, 1)]
void CS_DecoratorTints(uint3 dtid : SV_DispatchThreadID)
{
    StructuredBuffer<DecoratorGroup> groups = ResourceDescriptorHeap[DecoratorGroupsIdx];
    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    RWStructuredBuffer<float4> colors = ResourceDescriptorHeap[BladeColorsUAVIdx];
    RWStructuredBuffer<float4> tints = ResourceDescriptorHeap[DecoratorTintsUAVIdx];

    uint groupCount, stride;
    groups.GetDimensions(groupCount, stride);
    if (dtid.x >= groupCount) return;

    DecoratorGroup g = groups[dtid.x];
    float3 colorSum = 0;
    float weightSum = 0;
    float evenWeight = 0;
    float4 clumps = 0;          // scale, seed x 2 + amount, range, weight of the heaviest strongly clumped variant
    for (uint v = 0; v < g.VariantCount; v++)
    {
        uint slotIdx = g.VariantOffset + v;
        DecoratorSlot slot = slots[slotIdx];
        if (slot.Mode == MODE_MESH || slot.TextureIdx == 0) continue;

        float3 bandColor = 0;
        float coverage = 0;
        [unroll] for (uint b = 0; b < BLADE_COLOR_BANDS; b++)
        {
            float4 c = colors[slotIdx * COLORS_PER_VARIANT + b];
            bandColor += c.rgb * c.a;
            coverage += c.a;
        }
        if (coverage <= 0) continue;
        float3 cardColor = bandColor / coverage * 0.8 * slot.HealthyColor;
        coverage /= BLADE_COLOR_BANDS;      // the texture's opaque share

        float meanH = 0.5 * (slot.MinH + slot.MaxH);
        float area = 2.0 * 0.5 * (slot.MinW + slot.MaxW) * meanH * coverage;
        if (slot.Mode == MODE_FERNS || BandMode(slot, 0) == MODE_FERNS)
        {
            // A fern crown shades itself (fronds over fronds, darker towards the crown in grass_plants.fx): as flat
            // ground at the cards' 0.8 its patches were brighter and greener than the ferns up close
            cardColor *= 0.6;
            float frondWidth = 2.0 * FERN_FROND_HALF_U / 0.98 * FERN_FROND_WIDTH;     // x the frond's length
            float frondCoverage = saturate(coverage / (2.0 * FERN_FROND_HALF_U * 0.98));
            area = FERN_FRONDS_TINT * 0.706 * meanH * meanH * frondWidth * frondCoverage * 0.8;
        }
        float weight = 64.0 * slot.Density * DecorationDensity * area;
        colorSum += cardColor * weight;
        weightSum += weight;
        if (slot.ClusterScale > 0 && slot.ClusterAmount > 0.5)
        {
            if (weight > clumps.w)
            {
                evenWeight += clumps.w;     // a lighter clumped variant counts as even
                clumps = float4(slot.ClusterScale, float(slot.Seed) * 2.0 + saturate(slot.ClusterAmount), VariantRange(slot), weight);
            }
            else
                evenWeight += weight;
        }
        else
            evenWeight += weight;
    }
    tints[dtid.x * 2] = float4(weightSum > 0 ? colorSum / weightSum : 0.0.xxx, evenWeight);
    tints[dtid.x * 2 + 1] = clumps;
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
