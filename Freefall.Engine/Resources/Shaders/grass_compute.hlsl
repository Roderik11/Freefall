// ═══════════════════════════════════════════════════════════════════════════
// Grass/Decorator Compute Prepass
//
// Pipeline:
//   0. CS_BakeTerrainNormals: heightmap → R16G16_SNORM normal map (one-time)
//   1. CS_SpawnInstances:     single-pass: camera-centered grid → instance placement + cull → append
//   2. CS_BuildDrawArgs:      reads append counter → writes DispatchMesh indirect args
//
// Output: StructuredBuffer<DecoInstance> consumed by the lean AS/MS in grass.fx
// ═══════════════════════════════════════════════════════════════════════════

#pragma kernel CS_BakeTerrainNormals
#pragma kernel CS_SpawnInstances
#pragma kernel CS_BuildDrawArgs
#pragma kernel CS_BinMeshInstances

// Push constants (same layout as grass.fx common.fx)
struct PushConstantsData { uint4 indices[8]; };
ConstantBuffer<PushConstantsData> PushConstants : register(b3);
#define GET_INDEX(i) PushConstants.indices[i/4][i%4]

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
    float w, h, levels;
    hiZ.GetDimensions(0, w, h, levels);
    float2 mip0Size = float2(w, h);

    float projScale = OcclusionProjection._m11;
    projScale = abs(projScale) < 0.001 ? 1.0 : projScale;
    float projRadius = (worldRadius * projScale) / clipCenter.w;
    float screenRadius = projRadius * mip0Size.y * 0.5;

    float mipLevel = ceil(log2(max(screenRadius * 2.0, 1.0)));
    mipLevel = min(mipLevel, levels - 1.0f);

    uint mip = (uint)mipLevel;
    float2 mipSize = max(float2(1,1), mip0Size / (float)(1u << mip));

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

// ─── Push constant slots ───────────────────────────────────────────────────
// All names must follow XxxIdx = GET_INDEX(N) for FXParser binding discovery
#define DecoratorSlotsIdx   GET_INDEX(0)
#define LODTableIdx         GET_INDEX(1)
#define MeshRegistryIdx     GET_INDEX(2)
#define HeightmapIdx        GET_INDEX(3)
#define TerrainSizeXIdx     GET_INDEX(4)
#define TerrainSizeYIdx     GET_INDEX(5)
#define MaxHeightIdx        GET_INDEX(6)
#define CamPosXIdx          GET_INDEX(7)
#define CamPosYIdx          GET_INDEX(8)
#define CamPosZIdx          GET_INDEX(9)
#define TerrainOriginXIdx   GET_INDEX(10)
#define TerrainOriginYIdx   GET_INDEX(11)
#define TerrainOriginZIdx   GET_INDEX(12)
#define DecoControlIdx      GET_INDEX(13)
#define DecoRadiusIdx       GET_INDEX(14)
#define TileSizeIdx         GET_INDEX(15)
#define ControlWidthIdx     GET_INDEX(16)
#define ControlHeightIdx    GET_INDEX(17)
#define SlotCountIdx        GET_INDEX(18)

// CS-specific UAV/SRV slots
#define BaseTileXIdx        GET_INDEX(19)   // int as uint: camera-centered grid base X
#define BaseTileZIdx        GET_INDEX(20)   // int as uint: camera-centered grid base Z
#define DecoInstanceIdx     GET_INDEX(21)
#define InstanceCounterIdx  GET_INDEX(22)
#define MaxInstancesIdx     GET_INDEX(23)
#define DispatchArgsIdx     GET_INDEX(24)
#define DecorationDensityIdx GET_INDEX(25)
#define BakedNormalIdx      GET_INDEX(26)   // SRV for baked normal map
#define BakedNormalUAVIdx   GET_INDEX(27)   // UAV for baked normal map
#define DecoMapsIdx         GET_INDEX(28)   // SRV for density map Texture2DArray
#define CamFwdXIdx          GET_INDEX(29)   // Camera forward direction (normalized XZ)
#define CamFwdZIdx          GET_INDEX(30)
#define MeshDecoInstanceIdx GET_INDEX(31)   // UAV: mesh-mode instance output buffer

// Convenience accessors for float params
#define TerrainSizeX        asfloat(TerrainSizeXIdx)
#define TerrainSizeY        asfloat(TerrainSizeYIdx)
#define MaxHeight           asfloat(MaxHeightIdx)
#define CamPosX             asfloat(CamPosXIdx)
#define CamPosY             asfloat(CamPosYIdx)
#define CamPosZ             asfloat(CamPosZIdx)
#define TerrainOriginX      asfloat(TerrainOriginXIdx)
#define TerrainOriginY      asfloat(TerrainOriginYIdx)
#define TerrainOriginZ      asfloat(TerrainOriginZIdx)
#define DecoRadius          asfloat(DecoRadiusIdx)
#define TileSize            asfloat(TileSizeIdx)
#define DecorationDensity   asfloat(DecorationDensityIdx)
#define CamFwdX             asfloat(CamFwdXIdx)
#define CamFwdZ             asfloat(CamFwdZIdx)

// ─── GPU structs (must match grass.fx and TerrainRenderer.cs) ──────────────

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
    uint DecoMapSlice;
    uint _pad0;
    uint Mode;
    uint TextureIdx;
    float3 HealthyColor;
    float3 DryColor;
    float NoiseSpread;
    uint _colorPad;
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
    uint   SlotIdx;         // decorator slot index
    uint   LOD;             // selected LOD index
    float  InstanceSeed;    // for color variation in PS
    uint   _pad;
};  // 64 bytes — cache line aligned



#define MODE_MESH      0
#define MODE_BILLBOARD 1
#define MODE_CROSS     2

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

    uint hmW, hmH;
    heightTex.GetDimensions(hmW, hmH);
    if (dtid.x >= hmW || dtid.y >= hmH) return;

    float2 uv = (float2(dtid.xy) + 0.5) / float2(hmW, hmH);
    float texelStep = 1.0 / float(hmW);

    float hL = heightTex.SampleLevel(HeightSampler, uv + float2(-texelStep, 0), 0).r;
    float hR = heightTex.SampleLevel(HeightSampler, uv + float2( texelStep, 0), 0).r;
    float hD = heightTex.SampleLevel(HeightSampler, uv + float2(0, -texelStep), 0).r;
    float hU = heightTex.SampleLevel(HeightSampler, uv + float2(0,  texelStep), 0).r;

    float texelWorldSize = TerrainSizeX * texelStep;
    float heightScale = MaxHeight / texelWorldSize;

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
// Dispatch(N, N, 1) where N = cells per side around camera.
// Each group = one cell. [8,8,1] = 64 threads cooperate per cell.
// Thread 0 loads control data into groupshared; all threads read from there.
// ═══════════════════════════════════════════════════════════════════════════

groupshared uint gs_ctrl[8];
groupshared bool gs_tileActive;

[numthreads(8, 8, 1)]
void CS_SpawnInstances(uint3 gid : SV_GroupID, uint3 gtid3 : SV_GroupThreadID)
{
    uint flatThread = gtid3.y * 8 + gtid3.x;  // 0..63

    // Map group to world-space cell via base offset (camera-centered grid)
    int cellX = asint(BaseTileXIdx) + int(gid.x);
    int cellZ = asint(BaseTileZIdx) + int(gid.y);

    // Get control texture dimensions directly (match gputerrain.fx approach)
    Texture2DArray<uint4> controlTex = ResourceDescriptorHeap[DecoControlIdx];
    uint ctrlW, ctrlH, ctrlSlices;
    controlTex.GetDimensions(ctrlW, ctrlH, ctrlSlices);

    // Control map texel — same Y-flip as gputerrain.fx debug overlay
    int cx = cellX;
    int cy = int(ctrlH) - 1 - cellZ;

    // Check tile validity — no early return before the barrier
    float ts = TileSize;
    float tileWorldX = TerrainOriginX + (float(cellX) + 0.5) * ts;
    float tileWorldZ = TerrainOriginY + (float(cellZ) + 0.5) * ts;
    float3 camPos = float3(CamPosX, CamPosY, CamPosZ);
    float dx2 = tileWorldX - camPos.x;
    float dz2 = tileWorldZ - camPos.z;
    float maxR = DecoRadius + ts;

    bool tileValid = (cx >= 0 && cx < int(ctrlW) && cy >= 0 && cy < int(ctrlH))
                  && (dx2 * dx2 + dz2 * dz2 <= maxR * maxR);

    // Half-space cull: reject tiles behind the camera.
    // dot > -0.2 keeps ~100° behind-camera margin for scatter overshoot.
    if (tileValid)
    {
        float len = sqrt(dx2 * dx2 + dz2 * dz2);
        if (len > ts)   // skip tiles directly under the camera
            tileValid = (dx2 * CamFwdX + dz2 * CamFwdZ) / len > -0.2;
    }

    // ── Thread 0 reads BAKED control texture and populates groupshared ──
    // The baked control texture (RGBA16_UINT, 2 slices) was built by decoration_prepass.hlsl.
    // Each channel packs (slotIndex << 8) | weight. Up to 8 slots per texel.
    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    if (flatThread == 0)
    {
        gs_tileActive = false;
        [unroll] for (uint k = 0; k < 8; k++) gs_ctrl[k] = 0;

        // Hi-Z occlusion cull (thread 0 only): skip cells fully behind solid geometry
        if (tileValid && HiZSrvIdx != 0)
        {
            Texture2D heightTex2 = ResourceDescriptorHeap[HeightmapIdx];
            float2 cellUV = float2(
                (tileWorldX - TerrainOriginX) / TerrainSizeX,
                (tileWorldZ - TerrainOriginY) / TerrainSizeY
            );
            float h = heightTex2.SampleLevel(HeightSampler, cellUV, 2).r * MaxHeight;
            float3 sphereCenter = float3(tileWorldX, h, tileWorldZ);
            float sphereRadius = ts * 0.707 + MaxHeight * 0.05;
            if (IsCellOccluded(sphereCenter, sphereRadius))
                tileValid = false;
        }

        if (tileValid)
        {
            // Load baked control data (RGBA16_UINT is uncompressed — Load() is valid)
            uint4 packed0 = controlTex.Load(int4(cx, cy, 0, 0));
            uint4 packed1 = controlTex.Load(int4(cx, cy, 1, 0));

            uint activeIdx = 0;
            [unroll] for (uint ch = 0; ch < 8; ch++)
            {
                uint packed = (ch < 4) ? packed0[ch] : packed1[ch - 4];
                uint slotIdx = packed >> 8;
                uint weight  = packed & 0xFF;

                if (slotIdx == 255 || weight == 0) continue;

                DecoratorSlot slot = slots[slotIdx];

                gs_ctrl[activeIdx] = packed;
                activeIdx++;
            }
            gs_tileActive = (activeIdx > 0);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (!gs_tileActive) return;

    StructuredBuffer<LODEntry>      lodTbl = ResourceDescriptorHeap[LODTableIdx];

    float range = DecoRadius;
    float tileOriginX = TerrainOriginX + float(cellX) * ts;
    float tileOriginZ = TerrainOriginY + float(cellZ) * ts;

    RWStructuredBuffer<DecoInstance> outBuffer = ResourceDescriptorHeap[DecoInstanceIdx];
    RWByteAddressBuffer instCounter = ResourceDescriptorHeap[InstanceCounterIdx];
    uint maxInst = MaxInstancesIdx;

    Texture2D heightTex = ResourceDescriptorHeap[HeightmapIdx];
    Texture2D<float2> bakedNormals = ResourceDescriptorHeap[BakedNormalIdx];

    // Iterate over collected decorators (from groupshared)
    [unroll] for (uint ci = 0; ci < 8; ci++)
    {
        uint packed = gs_ctrl[ci];
        uint slotIdx = packed >> 8;
        uint weight  = packed & 0xFF;
        if (slotIdx == 255 || weight == 0) break;

        DecoratorSlot slot = slots[slotIdx];

        // Weight is 0-255 from density map; normalize to 0-1, then scale by max instances per tile
        float normalizedWeight = float(weight) / 255.0;
        uint decoInstCount = max(1u, min(64u, (uint)(normalizedWeight * 64.0 * slot.Density * DecorationDensity + 0.5)));

        // Stride loop: 64 threads cooperate to spawn decoInstCount instances
        for (uint instanceIdx = flatThread; instanceIdx < decoInstCount; instanceIdx += 64)
        {
            // Scatter beyond tile boundaries: overshoot scales with weight.
            // At max weight: range = (1 + 2*1) = 3× tile size. Less weight → less overshoot.
            float2 instanceSeed = float2(float(instanceIdx) * 0.7123 + float(slotIdx) * 3.917,
                                          float(instanceIdx) * 1.3147 + float(slotIdx) * 7.213);
            float2 tileOrig = float2(tileOriginX, tileOriginZ);
            float2 rng = hash22(instanceSeed + tileOrig);
            float overshoot = 0;// 0.5f * normalizedWeight;
            float wx = tileOrig.x + (rng.x * (1.0 + 2.0 * overshoot) - overshoot) * ts;
            float wz = tileOrig.y + (rng.y * (1.0 + 2.0 * overshoot) - overshoot) * ts;

            // Terrain UV
            float2 texelUV = float2(
                (wx - TerrainOriginX) / TerrainSizeX,
                (wz - TerrainOriginY) / TerrainSizeY
            );
            if (texelUV.x < 0 || texelUV.x > 1 || texelUV.y < 0 || texelUV.y > 1) continue;

            // ── Heightmap sample ──
            float h = heightTex.SampleLevel(HeightSampler, texelUV, 0).r;
            float3 instancePos = float3(wx, TerrainOriginZ + h * MaxHeight, wz);

            // Distance cull
            float camDist = distance(instancePos, camPos);
            if (camDist >= range) continue;

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
            float cellWeight = weight / 255.0;
            float2 seed = instanceSeed + tileOrig * 0.0137;
            float densityBias = lerp(0.3, 1.0, cellWeight);
            float randH = hash21(seed + 33.7);
            float randW = hash21(seed.yx + 77.9);
            float scaleH = lerp(slot.MinH, slot.MaxH, randH * densityBias);
            float scaleW = lerp(slot.MinW, slot.MaxW, randW * densityBias);

            // Distance fade: smoothly shrink in the last 25% of range
            float fadeStart = range * 0.85;
            float fadeFactor = smoothstep(0.0, 1.0, (range - camDist) / (range - fadeStart));
            scaleH *= fadeFactor;
            scaleW *= fadeFactor;

            float instanceRot = hash21(float2(wx * 7.3, wz * 31.7)) * 6.2831853;
            float instanceSeedVal = hash21(seed + 99.1);

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
            di._pad = 0;

            if (slot.Mode == MODE_MESH)
            {
                // Mesh instances → separate buffer, counter at offset 4
                uint outIdx;
                instCounter.InterlockedAdd(4, 1, outIdx);
                if (outIdx >= maxInst) continue;
                RWStructuredBuffer<DecoInstance> meshBuffer = ResourceDescriptorHeap[MeshDecoInstanceIdx];
                meshBuffer[outIdx] = di;
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
    RWByteAddressBuffer instCounter = ResourceDescriptorHeap[InstanceCounterIdx];
    uint instanceCount = instCounter.Load(0);
    instanceCount = min(instanceCount, MaxInstancesIdx);

    // Write DispatchMesh indirect args: (groupsX, groupsY, groupsZ)
    // Each MS group handles up to 16 instances (128 threads / 8 verts per instance)
    uint groupsX = (instanceCount + 15) / 16;

    RWByteAddressBuffer argsBuffer = ResourceDescriptorHeap[DispatchArgsIdx];
    argsBuffer.Store(0, groupsX);
    argsBuffer.Store(4, 1u);
    argsBuffer.Store(8, 1u);
    // Store mesh instance count at offset 12 for CS_BinMeshInstances
    uint meshCount = instCounter.Load(4);
    meshCount = min(meshCount, MaxInstancesIdx);
    argsBuffer.Store(12, meshCount);
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
//   MeshDecoInstanceIdx (31) = unsorted mesh instance SRV
//   Binning-specific slots set per-dispatch from C#:
//     DispatchArgsIdx (24)   = reused: read mesh count from offset 12
//     SortedMeshInstanceIdx  = new: UAV for sorted output
//     MeshDrawArgsIdx        = new: UAV for bindless draw commands
//     MeshDrawCountIdx       = new: UAV for draw count (1 uint)
// ═══════════════════════════════════════════════════════════════════════════

// Additional push constant slots for binning (set per-dispatch)
// Reuse existing slots that aren't needed during binning:
// BaseTileX(19), BaseTileZ(20) are only used by CS_SpawnInstances
#define SortedMeshInstanceIdx GET_INDEX(19)
#define MeshDrawArgsIdx       GET_INDEX(20)
#define MeshDrawCountIdx      GET_INDEX(22)  // reuse InstanceCounterIdx slot

// SRV indices to embed in each draw command's root constants (slots 4-15 of grass_mesh.fx)
// Reuse spawn-only slots for passing these into the binning kernel:
#define DrawSortedSRVIdx      GET_INDEX(7)   // -> draw cmd slot 4: SortedInstancesIdx
#define DrawSlotsSRVIdx       GET_INDEX(8)   // -> draw cmd slot 5: DecoratorSlotsIdx
#define DrawLODSRVIdx         GET_INDEX(9)   // -> draw cmd slot 6: LODTableIdx
#define DrawMeshRegSRVIdx     GET_INDEX(10)  // -> draw cmd slot 7: MeshRegistryIdx
#define DrawMaterialsSRVIdx   GET_INDEX(11)  // -> draw cmd slot 14: MaterialsIdx

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
    RWByteAddressBuffer argsBuffer = ResourceDescriptorHeap[DispatchArgsIdx];
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

    StructuredBuffer<DecoInstance> meshInstances = ResourceDescriptorHeap[MeshDecoInstanceIdx];

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
    RWStructuredBuffer<DecoInstance> sortedBuffer = ResourceDescriptorHeap[SortedMeshInstanceIdx];

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
        RWByteAddressBuffer drawArgsBuffer = ResourceDescriptorHeap[MeshDrawArgsIdx];
        RWByteAddressBuffer drawCountBuffer = ResourceDescriptorHeap[MeshDrawCountIdx];

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
