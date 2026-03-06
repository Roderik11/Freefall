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

// Push constants (same layout as grass.fx common.fx)
struct PushConstantsData { uint4 indices[8]; };
ConstantBuffer<PushConstantsData> PushConstants : register(b3);
#define GET_INDEX(i) PushConstants.indices[i/4][i%4]

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

    // ── Thread 0 reads BAKED control texture and populates groupshared ──
    // The baked control texture (RGBA16_UINT, 2 slices) was built by decoration_prepass.hlsl.
    // Each channel packs (slotIndex << 8) | weight. Up to 8 slots per texel.
    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    if (flatThread == 0)
    {
        gs_tileActive = false;
        [unroll] for (uint k = 0; k < 8; k++) gs_ctrl[k] = 0;

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

                // Filter: only billboard/cross modes for this pipeline
                DecoratorSlot slot = slots[slotIdx];
                if (slot.Mode == MODE_MESH) continue;

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

        // Skip mesh-mode decorators — rendered by separate VS/PS pipeline
        if (slot.Mode == MODE_MESH) continue;

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
            float overshoot = 0.f * normalizedWeight;
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

            // ── Terrain normal ──
            float2 nxz = bakedNormals.SampleLevel(ClampSampler, texelUV, 0);
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
            float fadeStart = range * 0.75;
            float fadeFactor = smoothstep(0.0, 1.0, (range - camDist) / (range - fadeStart));
            scaleH *= fadeFactor;
            scaleW *= fadeFactor;

            float instanceRot = hash21(float2(wx * 7.3, wz * 31.7)) * 6.2831853;
            float instanceSeedVal = hash21(seed + 99.1);

            // ── Append to output ──
            uint outIdx;
            instCounter.InterlockedAdd(0, 1, outIdx);
            if (outIdx >= maxInst) continue;

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
            outBuffer[outIdx] = di;
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
}
