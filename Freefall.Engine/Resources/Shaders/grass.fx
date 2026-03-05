// ═══════════════════════════════════════════════════════════════════════════
// Ground Coverage Shader — Amplification → Mesh → Pixel
// GPU-driven decoration instancing. AS does tile culling, MS generates
// instances procedurally. Each vertex thread computes its own instance data
// for maximum GPU utilization (no groupshared, no barriers).
// Root rotation matrix is precomputed on CPU — zero GPU trig for rotation.
// ═══════════════════════════════════════════════════════════════════════════
#include "common.fx"

#define ChannelHeadersIdx   GET_INDEX(0)
#define DecoratorSlotsIdx   GET_INDEX(1)
#define LODTableIdx         GET_INDEX(2)
#define MeshRegistryIdx     GET_INDEX(3)
#define HeightmapIdx        GET_INDEX(4)
#define TerrainSizeX        asfloat(GET_INDEX(5))
#define TerrainSizeY        asfloat(GET_INDEX(6))
#define MaxHeight           asfloat(GET_INDEX(7))
#define CamPosX             asfloat(GET_INDEX(8))
#define CamPosY             asfloat(GET_INDEX(9))
#define CamPosZ             asfloat(GET_INDEX(10))
#define TerrainOriginX      asfloat(GET_INDEX(11))
#define TerrainOriginY      asfloat(GET_INDEX(12))
#define TerrainOriginZ      asfloat(GET_INDEX(13))
#define MaterialsBufferIdx  GET_INDEX(14)
#define CellSize            asfloat(GET_INDEX(15))
#define DecoRadius          asfloat(GET_INDEX(16))
#define WindTime            asfloat(GET_INDEX(17))
#define DecoControlIdx      GET_INDEX(18)
#define BakedAlbedoIdx      GET_INDEX(19)
#define CascadeBufferSRVIdx  GET_INDEX(20)
#define ShadowCascadeCount  GET_INDEX(21)
#define ShadowHiZIdx        GET_INDEX(22)



// ─── GPU structs ───────────────────────────────────────────────────────────

struct ChannelHeader { uint StartIndex; uint Count; };

struct DecoratorSlot
{
    float Density;
    float MinH, MaxH;
    float MinW, MaxW;
    uint LODCount;
    uint LODTableOffset;
    // Precomputed root rotation matrix (CPU-computed, zero GPU trig)
    float3x3 RootMat;
    float SlopeBias;
    uint DecoMapSlice;      // slice in DecoMaps Texture2DArray (0xFFFFFFFF = none)
    uint _pad0;             // was ControlChannel, now unused (always sample R)
    uint Mode;              // 0=Mesh, 1=Billboard, 2=Cross
    uint TextureIdx;        // bindless texture index (billboard/cross)
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

// ─── Constants ─────────────────────────────────────────────────────────────

#define TILE_SIZE 8
#define CELLS_PER_TILE 64
#define MS_MAX_THREADS 128
#define MAX_VERTS 256
#define MAX_PRIMS 256

#define MODE_MESH      0
#define MODE_BILLBOARD 1
#define MODE_CROSS     2

#define BILLBOARD_VERTS 6
#define CROSS_VERTS    12

SamplerState HeightSampler : register(s0);
SamplerState ClampSampler  : register(s2);  // linear clamp (matches gputerrain.fx sampHeightFilter)

// ─── Hash ──────────────────────────────────────────────────────────────────

float hash21(float2 p)
{
    p = frac(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return frac(p.x * p.y);
}

float2 hash22(float2 p)
{
    float3 a = frac(p.xyx * float3(123.34, 456.21, 789.01));
    a += dot(a, a + 45.32);
    return frac(float2(a.x * a.y, a.y * a.z));
}

// ─── Amplification Shader ──────────────────────────────────────────────────

struct ASPayload
{
    float2 TileOrigin;
    uint   ActiveCount;           // number of active decorators in this tile (0-8)
    uint   GroupsPerDecorator;    // MS groups needed per decorator (worst-case across active)
    uint   SlotIndices0;          // packed 4×8-bit slot indices (slots 0-3)
    uint   SlotIndices1;          // packed 4×8-bit slot indices (slots 4-7)
    uint   Weights0;              // packed 4×8-bit weights (decorators 0-3)
    uint   Weights1;              // packed 4×8-bit weights (decorators 4-7)
    uint   InstanceCounts0;       // packed 4×8-bit instance counts (decorators 0-3)
    uint   InstanceCounts1;       // packed 4×8-bit instance counts (decorators 4-7)
};

// Pack/unpack helpers for 8-bit values in a uint
uint Pack4x8(uint a, uint b, uint c, uint d)
{
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}
uint Unpack8(uint packed0, uint packed1, uint which)
{
    uint packed = which < 4 ? packed0 : packed1;
    uint shift = (which % 4) * 8;
    return (packed >> shift) & 0xFF;
}

groupshared ASPayload s_Payload;

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void AS(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint gidx : SV_GroupIndex)
{
    float3 camPos = float3(CamPosX, CamPosY, CamPosZ);
    float cs = CellSize;
    float range = DecoRadius;
    float tileSize = cs * TILE_SIZE;

    float tileSnapX = floor(camPos.x / tileSize) * tileSize - range;
    float tileSnapZ = floor(camPos.z / tileSize) * tileSize - range;
    float2 tileOrigin = float2(
        tileSnapX + gid.x * tileSize,
        tileSnapZ + gid.y * tileSize
    );

    float2 tileCenter = tileOrigin + tileSize * 0.5;
    float tileDist = distance(float2(camPos.x, camPos.z), tileCenter);
    bool tileAlive = tileDist < (range + tileSize);

    // Frustum culling: test tile AABB against 4 side frustum planes
    // Extract planes from ViewProjection (row-major): plane = col ± col
    if (tileAlive)
    {
        // Tile AABB: XZ from tileOrigin, Y from terrain bounds (generous range)
        float minY = TerrainOriginZ;                   // terrain base
        float maxY = TerrainOriginZ + MaxHeight + 2.0;  // terrain peak + grass height margin

        float3 bboxMin = float3(tileOrigin.x, minY, tileOrigin.y);
        float3 bboxMax = float3(tileOrigin.x + tileSize, maxY, tileOrigin.y + tileSize);

        // Extract 4 frustum planes from ViewProjection
        // With row-vector mul (mul(pos, VP)) and row_major, planes come from COLUMNS
        float4 c0 = float4(ViewProjection[0][0], ViewProjection[1][0], ViewProjection[2][0], ViewProjection[3][0]);
        float4 c1 = float4(ViewProjection[0][1], ViewProjection[1][1], ViewProjection[2][1], ViewProjection[3][1]);
        float4 c3 = float4(ViewProjection[0][3], ViewProjection[1][3], ViewProjection[2][3], ViewProjection[3][3]);

        float4 planes[4];
        planes[0] = c3 + c0;  // left
        planes[1] = c3 - c0;  // right
        planes[2] = c3 + c1;  // bottom
        planes[3] = c3 - c1;  // top

        [unroll] for (uint pi = 0; pi < 4; pi++)
        {
            float4 p = planes[pi];
            // Test the "most positive" corner of the AABB against the plane
            float3 pv = float3(
                p.x > 0 ? bboxMax.x : bboxMin.x,
                p.y > 0 ? bboxMax.y : bboxMin.y,
                p.z > 0 ? bboxMax.z : bboxMin.z
            );
            if (dot(p.xyz, pv) + p.w < 0)
            {
                tileAlive = false;
                break;
            }
        }
    }

    // ── Thread 0: sample CONTROL once at tile center, build active list ──
    StructuredBuffer<DecoratorSlot> slots   = ResourceDescriptorHeap[DecoratorSlotsIdx];
    StructuredBuffer<LODEntry>      lodTbl  = ResourceDescriptorHeap[LODTableIdx];
    StructuredBuffer<MeshPartEntry> registry = ResourceDescriptorHeap[MeshRegistryIdx];

    if (gidx == 0)
    {
        uint activeCount = 0;
        uint activeSlots[8];
        uint activeWeights[8];
        uint activeCounts[8];
        uint vpi = 3;

        [unroll] for (uint k = 0; k < 8; k++)
        {
            activeSlots[k] = 255;
            activeWeights[k] = 0;
            activeCounts[k] = 0;
        }

        if (tileAlive && DecoControlIdx != 0)
        {
            // Tile center UV
            float2 texelUV = float2(
                (tileCenter.x - TerrainOriginX) / TerrainSizeX,
                (tileCenter.y - TerrainOriginY) / TerrainSizeY
            );

            if (texelUV.x >= 0 && texelUV.x <= 1 && texelUV.y >= 0 && texelUV.y <= 1)
            {
                Texture2DArray<uint4> controlTex = ResourceDescriptorHeap[DecoControlIdx];
                uint ctrlW, ctrlH, ctrlSlices;
                controlTex.GetDimensions(ctrlW, ctrlH, ctrlSlices);

                float2 ctrlUV = float2(texelUV.x, 1.0 - texelUV.y);
                int2 texel = int2(ctrlUV * float2(ctrlW, ctrlH));

                uint4 ctrl0 = controlTex.Load(int4(texel, 0, 0));
                uint4 ctrl1 = controlTex.Load(int4(texel, 1, 0));

                // Unpack CONTROL: up to 8 (slotIndex, weight) pairs
                uint4 allCtrl[2] = { ctrl0, ctrl1 };
                [unroll] for (uint ci = 0; ci < 8; ci++)
                {
                    uint packed = allCtrl[ci / 4][ci % 4];
                    uint slotIdx = packed >> 8;
                    uint weight  = packed & 0xFF;
                    if (slotIdx == 255 || weight == 0) break;

                    activeSlots[activeCount] = slotIdx;
                    activeWeights[activeCount] = weight;

                    // Instance count: density × tile area × weight/255, up to 255 (8-bit payload)
                    // Can exceed 64 — multiple instances per cell at high density
                    DecoratorSlot s = slots[slotIdx];
                    float tileArea = tileSize * tileSize;
                    uint maxInst = max(1u, min(255u, (uint)(s.Density * tileArea + 0.5)));
                    activeCounts[activeCount] = max(1u, min(255u, (maxInst * weight + 127) / 255));

                    // Vertex count for groupsPerDeco calculation
                    if (s.Mode == MODE_BILLBOARD)
                        vpi = max(vpi, BILLBOARD_VERTS);
                    else if (s.Mode == MODE_CROSS)
                        vpi = max(vpi, CROSS_VERTS);
                    else if (s.LODCount > 0 && s.LODCount <= 8)
                    {
                        uint vc = registry[lodTbl[s.LODTableOffset].MeshPartId].VertexCount;
                        vpi = max(vpi, vc);
                    }
                    activeCount++;
                }
            }
        }
        vpi = clamp(vpi, 3, 128);

        // groupsPerDeco from max instance count across active decorators
        uint maxCount = 0;
        [unroll] for (uint mi = 0; mi < 8; mi++)
            maxCount = max(maxCount, activeCounts[mi]);

        uint ipg = max(1, min(MS_MAX_THREADS / vpi, MAX_VERTS / vpi));
        uint groupsPerDeco = (maxCount + ipg - 1) / ipg;

        s_Payload.TileOrigin = tileOrigin;
        s_Payload.ActiveCount = activeCount;
        s_Payload.GroupsPerDecorator = groupsPerDeco;
        s_Payload.SlotIndices0 = Pack4x8(activeSlots[0], activeSlots[1], activeSlots[2], activeSlots[3]);
        s_Payload.SlotIndices1 = Pack4x8(activeSlots[4], activeSlots[5], activeSlots[6], activeSlots[7]);
        s_Payload.Weights0 = Pack4x8(activeWeights[0], activeWeights[1], activeWeights[2], activeWeights[3]);
        s_Payload.Weights1 = Pack4x8(activeWeights[4], activeWeights[5], activeWeights[6], activeWeights[7]);
        s_Payload.InstanceCounts0 = Pack4x8(activeCounts[0], activeCounts[1], activeCounts[2], activeCounts[3]);
        s_Payload.InstanceCounts1 = Pack4x8(activeCounts[4], activeCounts[5], activeCounts[6], activeCounts[7]);
    }
    GroupMemoryBarrierWithGroupSync();

    uint totalGroups = s_Payload.ActiveCount * s_Payload.GroupsPerDecorator;
    DispatchMesh(totalGroups, 1, 1, s_Payload);
}

// ─── Mesh Shader ───────────────────────────────────────────────────────────
//
// No groupshared, no barriers. Each vertex thread computes its own instance
// data independently. This maintains 100% GPU utilization and allows maximum
// occupancy for latency hiding. The "redundant" per-instance work across
// same-instance vertices is cheaper than barrier + underutilization.

struct MSOutput
{
    float4 Position : SV_POSITION;
    float3 Normal   : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 WorldPos : TEXCOORD1;
    float  Depth    : TEXCOORD2;
    nointerpolation uint MaterialId      : TEXCOORD3;
    nointerpolation uint TextureOverride  : TEXCOORD4;  // >0: use directly (billboard/cross)
    float  HeightFrac : TEXCOORD5;  // 0=base, 1=top — for vertex AO
    float2 TerrainUV  : TEXCOORD6;  // UV in terrain space for ground color sampling
    nointerpolation float InstanceSeed : TEXCOORD7;  // per-instance hash for color variation
};

[outputtopology("triangle")]
[numthreads(MS_MAX_THREADS, 1, 1)]
void MS(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    in payload ASPayload payload,
    out vertices MSOutput verts[MAX_VERTS],
    out indices uint3 tris[MAX_PRIMS])
{
    // ── Decompose group index into decorator + cell group ──
    uint groupsPerDeco = payload.GroupsPerDecorator;
    uint decoIdx = gid.x / groupsPerDeco;          // which active decorator (0..ActiveCount-1)
    uint groupInDeco = gid.x % groupsPerDeco;       // which cell group within that decorator

    // Look up the actual slot index and instance count from the payload
    uint slotIdx = Unpack8(payload.SlotIndices0, payload.SlotIndices1, decoIdx);
    uint decoInstCount = Unpack8(payload.InstanceCounts0, payload.InstanceCounts1, decoIdx);

    // Read slot data for this decorator
    StructuredBuffer<DecoratorSlot> msSlots   = ResourceDescriptorHeap[DecoratorSlotsIdx];
    StructuredBuffer<LODEntry>      msLodTbl  = ResourceDescriptorHeap[LODTableIdx];
    StructuredBuffer<MeshPartEntry> msRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    DecoratorSlot slot = msSlots[slotIdx];

    // Determine exact vertex count for THIS decorator (not max across all)
    uint vertsPerInst = 3;
    if (slot.Mode == MODE_BILLBOARD)
        vertsPerInst = BILLBOARD_VERTS;
    else if (slot.Mode == MODE_CROSS)
        vertsPerInst = CROSS_VERTS;
    else if (slot.LODCount > 0 && slot.LODCount <= 8)
        vertsPerInst = msRegistry[msLodTbl[slot.LODTableOffset].MeshPartId].VertexCount;
    vertsPerInst = clamp(vertsPerInst, 3, 128);

    uint trisPerInst = vertsPerInst / 3;
    uint ipg = max(1, min(MS_MAX_THREADS / vertsPerInst, MAX_VERTS / vertsPerInst));

    uint baseInstance = groupInDeco * ipg;
    // Clamp to this decorator's instance count (weight-proportional, not full 64)
    uint batchCount = min(ipg, decoInstCount - min(baseInstance, decoInstCount));

    uint totalVerts = batchCount * vertsPerInst;
    uint totalTris = batchCount * trisPerInst;
    SetMeshOutputCounts(totalVerts, totalTris);

    if (gtid >= totalVerts) return;

    uint instInBatch = gtid / vertsPerInst;
    uint vertInInst = gtid % vertsPerInst;
    uint instanceIdx = baseInstance + instInBatch;

    // ── Per-instance data ──

    // Fully random placement: hash instance index → position within tile.
    // No cell grid = no grid artifacts.
    float2 instanceSeed = float2(float(instanceIdx) * 0.7123 + float(slotIdx) * 3.917,
                                  float(instanceIdx) * 1.3147 + float(slotIdx) * 7.213);
    float2 rng = hash22(instanceSeed + payload.TileOrigin);  // [0,1] × [0,1]
    float cs = CellSize;
    float tileWorldSize = cs * TILE_SIZE;  // full tile extent
    float wx = payload.TileOrigin.x + rng.x * tileWorldSize;
    float wz = payload.TileOrigin.y + rng.y * tileWorldSize;
    float2 seed = instanceSeed + payload.TileOrigin * 0.0137;  // include tile position for unique scale/rotation per tile

    // Terrain bounds
    float2 texelUV = float2(
        (wx - TerrainOriginX) / TerrainSizeX,
        (wz - TerrainOriginY) / TerrainSizeY
    );
    bool alive = texelUV.x >= 0 && texelUV.x <= 1 && texelUV.y >= 0 && texelUV.y <= 1;

    // Per-cell CONTROL presence check: verify this decorator is active at this cell's position
    // (AS sampled at tile center for dispatch; MS refines per-cell for smooth boundaries)
    if (alive && DecoControlIdx != 0)
    {
        Texture2DArray<uint4> controlTex = ResourceDescriptorHeap[DecoControlIdx];
        uint ctrlW, ctrlH, ctrlSlices;
        controlTex.GetDimensions(ctrlW, ctrlH, ctrlSlices);
        float2 ctrlUV = float2(texelUV.x, 1.0 - texelUV.y);
        int2 ctrlTexel = int2(ctrlUV * float2(ctrlW, ctrlH));

        uint4 ctrl0 = controlTex.Load(int4(ctrlTexel, 0, 0));
        uint4 ctrl1 = controlTex.Load(int4(ctrlTexel, 1, 0));

        bool found = false;
        uint4 allCtrl[2] = { ctrl0, ctrl1 };
        [unroll] for (uint ci = 0; ci < 8; ci++)
        {
            uint packed = allCtrl[ci / 4][ci % 4];
            uint idx = packed >> 8;
            if (idx == 255) break;
            if (idx == slotIdx) { found = true; break; }
        }
        if (!found) alive = false;
    }

    // Heightmap & position
    float3 camPos = float3(CamPosX, CamPosY, CamPosZ);
    float h = 0;
    float3 terrainNormal = float3(0, 1, 0);
    if (alive)
    {
        Texture2D heightTex = ResourceDescriptorHeap[HeightmapIdx];
        h = heightTex.SampleLevel(HeightSampler, texelUV, 0).r;

        // Terrain normal — match terrain.fx GetNormal() exactly
        uint hmW, hmH;
        heightTex.GetDimensions(hmW, hmH);
        float texelStep = 1.0 / float(hmW);
        float hL = heightTex.SampleLevel(HeightSampler, texelUV + float2(-texelStep, 0), 0).r;
        float hR = heightTex.SampleLevel(HeightSampler, texelUV + float2( texelStep, 0), 0).r;
        float hD = heightTex.SampleLevel(HeightSampler, texelUV + float2(0, -texelStep), 0).r;
        float hU = heightTex.SampleLevel(HeightSampler, texelUV + float2(0,  texelStep), 0).r;
        float texelWorldSize = TerrainSizeX * texelStep;
        float heightScale = MaxHeight / texelWorldSize;
        terrainNormal = normalize(float3(
            (hL - hR) * heightScale,
            1.0,
            (hD - hU) * heightScale
        ));
    }

    float3 instancePos = float3(wx, TerrainOriginZ + h * MaxHeight, wz);

    if (alive && distance(instancePos, camPos) >= DecoRadius) alive = false;

    // LOD
    float camDist = distance(instancePos, camPos);
    uint lod = 0;
    if (alive)
    {
        uint maxLod = min(slot.LODCount, 8);
        for (uint il = 0; il < maxLod; il++)
        {
            if (camDist > msLodTbl[slot.LODTableOffset + il].MaxDistance)
                lod = il + 1;
        }
        if (lod >= slot.LODCount) alive = false;
    }

    // Dead instance: degenerate triangle (clipped for free)
    if (!alive)
    {
        MSOutput o = (MSOutput)0;
        o.Position = float4(0, 0, -1, 0);
        verts[gtid] = o;
        if (vertInInst % 3 == 0)
        {
            uint ti = instInBatch * trisPerInst + vertInInst / 3;
            uint bv = instInBatch * vertsPerInst + vertInInst;
            tris[ti] = uint3(bv, bv + 1, bv + 2);
        }
        return;
    }

    // ── Per-instance scale & rotation ──

    float scaleH = lerp(slot.MinH, slot.MaxH, hash21(seed + 33.7));
    float scaleW = lerp(slot.MinW, slot.MaxW, hash21(seed.yx + 77.9));
    
    // Distance fade: smoothly shrink to zero in the last 25% of range
    float fadeStart = DecoRadius * 0.75;
    float fadeFactor = smoothstep(0.0, 1.0, (DecoRadius - camDist) / (DecoRadius - fadeStart));
    scaleH *= fadeFactor;
    scaleW *= fadeFactor;
    float instanceRot = hash21(seed + 51.3) * 6.2831853;
    float cosR = cos(instanceRot), sinR = sin(instanceRot);

    // LOD entry (Mesh mode) or material from texture (Billboard/Cross)
    uint materialId = 0;
    if (slot.Mode == MODE_MESH)
    {
        LODEntry lodEntry = msLodTbl[slot.LODTableOffset + lod];
        materialId = lodEntry.MaterialId;
    }
    else if (slot.LODCount > 0)
    {
        // Billboard/Cross: LODTable[0].MaterialId holds the billboard material
        materialId = msLodTbl[slot.LODTableOffset].MaterialId;
    }

    // ── Billboard / Cross: procedural geometry ──

    if (slot.Mode == MODE_BILLBOARD || slot.Mode == MODE_CROSS)
    {
        // Billboard: 1 quad (6 verts) — camera facing
        // Cross: 2 quads (12 verts) — fixed 90° intersection
        uint totalQuadVerts = (slot.Mode == MODE_BILLBOARD) ? BILLBOARD_VERTS : CROSS_VERTS;

        if (vertInInst >= totalQuadVerts)
        {
            MSOutput o = (MSOutput)0;
            o.Position = float4(0, 0, -1, 0);
            verts[gtid] = o;
            if (vertInInst % 3 == 0)
            {
                uint ti = instInBatch * trisPerInst + vertInInst / 3;
                uint bv = instInBatch * vertsPerInst + vertInInst;
                tris[ti] = uint3(bv, bv + 1, bv + 2);
            }
            return;
        }

        // Quad vertex layout (triangle list, 6 verts per quad):
        // v0(0,0)  v1(1,0)  v2(1,1)  |  v3(0,0)  v4(1,1)  v5(0,1)
        static const float2 quadUV[6] = {
            float2(0,1), float2(1,1), float2(1,0),
            float2(0,1), float2(1,0), float2(0,0)
        };
        // Local offsets: X = [-0.5..0.5] * width, Y = [0..1] * height
        static const float2 quadPos[6] = {
            float2(-0.5, 0.0), float2(0.5, 0.0), float2(0.5, 1.0),
            float2(-0.5, 0.0), float2(0.5, 1.0), float2(-0.5, 1.0)
        };

        uint quadIdx = vertInInst / 6;    // which quad (0 or 1)
        uint vertInQuad = vertInInst % 6;  // which vert in quad

        float2 lp = quadPos[vertInQuad];
        float2 uv = quadUV[vertInQuad];

        float3 worldPos;
        float3 worldNorm;

        // Slope-aligned up vector: blend between vertical and terrain normal
        float3 slopeUp = normalize(lerp(float3(0, 1, 0), terrainNormal, slot.SlopeBias));

        if (slot.Mode == MODE_BILLBOARD)
        {
            // Camera-facing billboard
            float3 toCamera = normalize(float3(camPos.x - instancePos.x, 0, camPos.z - instancePos.z));
            float3 right = normalize(cross(slopeUp, toCamera));
            worldPos = instancePos + right * lp.x * scaleW + slopeUp * lp.y * scaleH;
            worldNorm = toCamera;
        }
        else // MODE_CROSS
        {
            // Two quads at 90° with instance Y-rotation
            float3 right;
            if (quadIdx == 0)
                right = float3(cosR, 0, sinR);
            else
                right = float3(-sinR, 0, cosR);
            // Re-orthogonalize right against slope up
            right = normalize(right - slopeUp * dot(right, slopeUp));

            worldPos = instancePos + right * lp.x * scaleW + slopeUp * lp.y * scaleH;
            worldNorm = (quadIdx == 0)
                ? float3(-sinR, 0, cosR)
                : float3(-cosR, 0, -sinR);
        }

        // Wind sway: two overlapping sine waves for organic motion
        // HeightFrac² keeps the base planted, tips sway most
        float windInfluence = lp.y * lp.y;
        float windWave1 = sin(WindTime * 2.1 + worldPos.x * 0.8 + worldPos.z * 0.6) * 0.12;
        float windWave2 = sin(WindTime * 1.4 + worldPos.x * 0.5 - worldPos.z * 0.9) * 0.08;
        worldPos.x += (windWave1 + windWave2) * windInfluence * scaleH;
        worldPos.z += (windWave2 - windWave1 * 0.5) * windInfluence * scaleH;

        MSOutput o;
        float4 wp = float4(worldPos, 1.0);
        o.WorldPos = wp;
        o.Position = mul(wp, ViewProjection);
        // Terrain normal + subtle per-instance perturbation (terrain normal dominates)
        float2 noiseSeed = seed + float(slotIdx) * 7.3;
        float3 normalJitter = float3(
            hash21(noiseSeed + 11.1) - 0.5,
            hash21(noiseSeed + 33.3) * 0.15,
            hash21(noiseSeed + 22.2) - 0.5
        ) * 0.8;
        o.Normal = normalize(terrainNormal + normalJitter);
        o.TexCoord = uv;
        o.Depth = o.Position.w;
        o.MaterialId = materialId;
        o.TextureOverride = slot.TextureIdx;
        o.HeightFrac = saturate(lp.y);  // 0 at base, 1 at top
        o.TerrainUV = texelUV;
        o.InstanceSeed = float(instanceIdx) / 64.0;
        verts[gtid] = o;

        if (vertInInst % 3 == 0)
        {
            uint ti = instInBatch * trisPerInst + vertInInst / 3;
            uint bv = instInBatch * vertsPerInst + vertInInst;
            tris[ti] = uint3(bv, bv + 1, bv + 2);
        }
        return;
    }

    // ── Mesh mode: existing registry-based geometry ──

    LODEntry lodEntry = msLodTbl[slot.LODTableOffset + lod];
    uint meshPartId = lodEntry.MeshPartId;

    // Root rotation: read precomputed matrix from slot (zero trig)
    float3x3 rootMat = slot.RootMat;

    // Instance Y-rotation + non-uniform scale (W for XZ, H for Y)
    float3x3 instanceMat = float3x3(
        cosR * scaleW, 0, sinR * scaleW,
        0, scaleH, 0,
        -sinR * scaleW, 0, cosR * scaleW
    );

    // Slope alignment: rotate from (0,1,0) toward terrain normal based on SlopeBias
    float3 slopeUp = normalize(lerp(float3(0, 1, 0), terrainNormal, slot.SlopeBias));
    float3 axis = cross(float3(0, 1, 0), slopeUp);
    float sinA = length(axis);
    float cosA = dot(float3(0, 1, 0), slopeUp);
    float3x3 slopeMat = float3x3(1,0,0, 0,1,0, 0,0,1); // identity default
    if (sinA > 0.001) // only rotate if there's a meaningful slope
    {
        axis = axis / sinA; // normalize
        float3x3 K = float3x3(0, -axis.z, axis.y, axis.z, 0, -axis.x, -axis.y, axis.x, 0);
        float3x3 I = float3x3(1,0,0, 0,1,0, 0,0,1);
        slopeMat = I + K * sinA + mul(K, K) * (1.0 - cosA);
    }

    float3x3 finalMat = mul(mul(rootMat, instanceMat), slopeMat);

    // ── Vertex fetch & transform ──

    MeshPartEntry part = msRegistry[meshPartId];

    // If this mesh has fewer vertices than the payload's max, degenerate extra threads
    if (vertInInst >= part.VertexCount)
    {
        MSOutput o = (MSOutput)0;
        o.Position = float4(0, 0, -1, 0);
        verts[gtid] = o;
        if (vertInInst % 3 == 0)
        {
            uint ti = instInBatch * trisPerInst + vertInInst / 3;
            uint bv = instInBatch * vertsPerInst + vertInInst;
            tris[ti] = uint3(bv, bv + 1, bv + 2);
        }
        return;
    }

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float3> normals   = ResourceDescriptorHeap[part.NormBufferIdx];
    StructuredBuffer<float2> uvs       = ResourceDescriptorHeap[part.UVBufferIdx];
    StructuredBuffer<uint>   indices   = ResourceDescriptorHeap[part.IndexBufferIdx];

    uint vertIdx = indices[vertInInst + part.BaseIndex];
    float3 pos = positions[vertIdx];
    float3 norm = normals[vertIdx];
    float2 uv = uvs[vertIdx];

    float3 worldPos = mul(pos, finalMat) + instancePos;
    float3 worldNorm = normalize(mul(norm, finalMat));

    // Wind sway for mesh decorators
    float localY = mul(pos, rootMat).y;
    float heightFrac = saturate(localY / max(scaleH, 0.01));
    float windInfluence = heightFrac * heightFrac;
    float windWave1 = sin(WindTime * 2.1 + worldPos.x * 0.8 + worldPos.z * 0.6) * 0.12;
    float windWave2 = sin(WindTime * 1.4 + worldPos.x * 0.5 - worldPos.z * 0.9) * 0.08;
    worldPos.x += (windWave1 + windWave2) * windInfluence * scaleH;
    worldPos.z += (windWave2 - windWave1 * 0.5) * windInfluence * scaleH;

    MSOutput o;
    float4 wp = float4(worldPos, 1.0);
    o.WorldPos = wp;
    o.Position = mul(wp, ViewProjection);
    // Full meshes keep geometric normals for proper 3D shading
    o.Normal = worldNorm;
    o.TexCoord = float2(uv.x, 1 - uv.y);
    o.Depth = o.Position.w;
    o.MaterialId = materialId;
    o.TextureOverride = 0;
    // Approximate height fraction from local-space Y after root rotation
    o.HeightFrac = saturate(localY / max(scaleH, 0.01));
    o.TerrainUV = texelUV;
    o.InstanceSeed = hash21(seed + 99.1);
    verts[gtid] = o;

    if (vertInInst % 3 == 0)
    {
        uint ti = instInBatch * trisPerInst + vertInInst / 3;
        uint bv = instInBatch * vertsPerInst + vertInInst;
        tris[ti] = uint3(bv, bv + 1, bv + 2);
    }
}

// ─── Pixel Shader ──────────────────────────────────────────────────────────

struct PSOutput
{
    float4 Albedo : SV_Target0;
    float4 Normal : SV_Target1;
    float4 Data   : SV_Target2;
    float  Depth  : SV_Target3;
};

PSOutput PS(MSOutput input)
{
    PSOutput output;

    float3 baseColor;
    float alpha = 1.0;

    if (input.TextureOverride > 0)
    {
        // Billboard/Cross: sample texture directly (no material indirection)
        Texture2D texOverride = ResourceDescriptorHeap[input.TextureOverride];
        float4 texColor = texOverride.Sample(HeightSampler, input.TexCoord);
        baseColor = texColor.rgb;
        alpha = texColor.a;
    }
    else
    {
        StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsBufferIdx];
        MaterialData mat = materials[input.MaterialId];

        if (mat.AlbedoIdx > 0)
        {
            Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
            float4 texColor = albedoTex.Sample(HeightSampler, input.TexCoord);
            baseColor = texColor.rgb;
            alpha = texColor.a;
        }
        else
        {
            baseColor = float3(0.15, 0.35, 0.08);
        }
    }

    // Distance-based alpha clip: mipmaps average alpha down at distance.
    // Lower the threshold to preserve coverage instead of eating the geometry.
    float dist = length(input.WorldPos.xyz);
    float alphaThreshold = lerp(0.5, 0.15, saturate((dist - 20.0) / 30.0));
    clip(alpha - alphaThreshold);

    // Sample baked terrain albedo for ground color blending
    Texture2D BakedAlbedo = ResourceDescriptorHeap[BakedAlbedoIdx];
    float3 groundColor = BakedAlbedo.SampleLevel(ClampSampler, input.TerrainUV, 0).rgb;

    // Ground color blend: at base, shift toward terrain color
    float aoFactor = pow(input.HeightFrac, 0.4);

    // Per-instance brightness variation — only affects texture, not ground
    float hashA = frac(input.InstanceSeed * 17.31);
    float brightVar = 0.2 + 0.6 * hashA;

    float3 variedBase = baseColor * brightVar;
    float3 aoColor = lerp(groundColor * 0.75, variedBase, aoFactor);

    output.Albedo = float4(aoColor, 1.0);

    // Per-pixel normals: blend from terrain normal (base) to upward (tip)
    // This creates natural top-bright / base-dark variation across each instance
    float3 baseNorm = normalize(input.Normal);
    float3 tipNorm = normalize(lerp(baseNorm, float3(0, 1, 0), 0.6));
    float3 heightNorm = normalize(lerp(baseNorm, tipNorm, input.HeightFrac));
    
    // Alpha gradient micro-detail on top of the height blend
    float dAlphaX = ddx(alpha);
    float dAlphaY = ddy(alpha);
    float bladeX = (input.TexCoord.x - 0.5) * 2.0;
    float3 perturbedNormal = normalize(heightNorm + float3(
        dAlphaX * 1.0 + bladeX * 0.2,
        0,
        dAlphaY * 1.0
    ));
    output.Normal = float4(perturbedNormal, 1.0);

    // Data channel: R=roughness, G=metallic, B=ao, A=flags
    // Flags: 0=unlit/skybox, 0.5=vegetation, 1.0=standard PBR
    output.Data = float4(0.9, 0.0, 1.0, 0.5);
    output.Depth = input.Depth;
    return output;
}

// ─── Shadow Pixel Shader ───────────────────────────────────────────────────
// Depth-only: alpha test, hardware writes depth. No MRT output.

void PS_Shadow(MSOutput input)
{
    if (input.TextureOverride > 0)
    {
        Texture2D texOverride = ResourceDescriptorHeap[input.TextureOverride];
        float alpha = texOverride.Sample(HeightSampler, input.TexCoord).a;
        clip(alpha - 0.25);
    }
    else
    {
        StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsBufferIdx];
        MaterialData mat = materials[input.MaterialId];

        if (mat.AlbedoIdx > 0)
        {
            Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
            float alpha = albedoTex.Sample(HeightSampler, input.TexCoord).a;
            clip(alpha - 0.25);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Single-pass shadow rendering — AS tests all cascade frustums, MS outputs
// SV_RenderTargetArrayIndex per cascade. One DispatchMesh for all cascades.
// ═══════════════════════════════════════════════════════════════════════════

// Shadow-specific push constants (slots 20-21) — defines at top of file via CascadeBufferSRVIdx/ShadowCascadeCount

// Shadow AS payload — adds cascade index for MS routing
struct ShadowASPayload
{
    float2 TileOrigin;
    uint   ActiveCount;
    uint   GroupsPerDecorator;
    uint   SlotIndices0;
    uint   SlotIndices1;
    uint   Weights0;
    uint   Weights1;
    uint   InstanceCounts0;
    uint   InstanceCounts1;
    uint   CascadeIdx;          // which cascade this batch of MS groups renders to
};

groupshared ShadowASPayload s_ShadowPayload;

// Test tile AABB against frustum planes extracted from a VP matrix.
// Returns true if tile is (potentially) visible in this cascade.
bool TestTileFrustum(float2 tileOrigin, float tileSize, row_major float4x4 vp)
{
    float minY = TerrainOriginZ;
    float maxY = TerrainOriginZ + MaxHeight + 2.0;
    float3 bboxMin = float3(tileOrigin.x, minY, tileOrigin.y);
    float3 bboxMax = float3(tileOrigin.x + tileSize, maxY, tileOrigin.y + tileSize);

    // Extract 4 side frustum planes from VP (row-vector convention)
    float4 c0 = float4(vp[0][0], vp[1][0], vp[2][0], vp[3][0]);
    float4 c1 = float4(vp[0][1], vp[1][1], vp[2][1], vp[3][1]);
    float4 c3 = float4(vp[0][3], vp[1][3], vp[2][3], vp[3][3]);

    float4 planes[4];
    planes[0] = c3 + c0;  // left
    planes[1] = c3 - c0;  // right
    planes[2] = c3 + c1;  // bottom
    planes[3] = c3 - c1;  // top

    [unroll] for (uint pi = 0; pi < 4; pi++)
    {
        float4 p = planes[pi];
        float3 pv = float3(
            p.x > 0 ? bboxMax.x : bboxMin.x,
            p.y > 0 ? bboxMax.y : bboxMin.y,
            p.z > 0 ? bboxMax.z : bboxMin.z
        );
        if (dot(p.xyz, pv) + p.w < 0)
            return false;
    }
    return true;
}

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void AS_Shadow(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint gidx : SV_GroupIndex)
{
    float3 camPos = float3(CamPosX, CamPosY, CamPosZ);
    float cs = CellSize;
    float range = DecoRadius;
    float tileSize = cs * TILE_SIZE;

    float tileSnapX = floor(camPos.x / tileSize) * tileSize - range;
    float tileSnapZ = floor(camPos.z / tileSize) * tileSize - range;
    float2 tileOrigin = float2(
        tileSnapX + gid.x * tileSize,
        tileSnapZ + gid.y * tileSize
    );

    float2 tileCenter = tileOrigin + tileSize * 0.5;
    float tileDist = distance(float2(camPos.x, camPos.z), tileCenter);
    bool tileAlive = tileDist < (range + tileSize);

    // Read cascade VPs from unified buffer
    StructuredBuffer<CascadeData> shadowCascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    uint cascadeCount = ShadowCascadeCount;

    // Test against all cascade frustums to build cascade mask
    uint cascadeMask = 0;
    if (tileAlive)
    {
        for (uint ci = 0; ci < cascadeCount; ci++)
        {
            if (TestTileFrustum(tileOrigin, tileSize, shadowCascades[ci].VP))
                cascadeMask |= (1u << ci);
        }
        if (cascadeMask == 0)
            tileAlive = false;
    }

    // Thread 0: build active decorator list (same as AS but with distance check disabled for shadows)
    StructuredBuffer<DecoratorSlot> slots   = ResourceDescriptorHeap[DecoratorSlotsIdx];
    StructuredBuffer<LODEntry>      lodTbl  = ResourceDescriptorHeap[LODTableIdx];
    StructuredBuffer<MeshPartEntry> registry = ResourceDescriptorHeap[MeshRegistryIdx];

    uint activeCount = 0;
    uint activeSlots[8];
    uint activeWeights[8];
    uint activeCounts[8];
    uint vpi = 3;
    uint totalMSGroups = 0;

    [unroll] for (uint k = 0; k < 8; k++)
    {
        activeSlots[k] = 255;
        activeWeights[k] = 0;
        activeCounts[k] = 0;
    }

    if (gidx == 0 && tileAlive && DecoControlIdx != 0)
    {
        float2 texelUV = float2(
            (tileCenter.x - TerrainOriginX) / TerrainSizeX,
            (tileCenter.y - TerrainOriginY) / TerrainSizeY
        );

        if (texelUV.x >= 0 && texelUV.x <= 1 && texelUV.y >= 0 && texelUV.y <= 1)
        {
            Texture2DArray<uint4> controlTex = ResourceDescriptorHeap[DecoControlIdx];
            uint ctrlW, ctrlH, ctrlSlices;
            controlTex.GetDimensions(ctrlW, ctrlH, ctrlSlices);

            float2 ctrlUV = float2(texelUV.x, 1.0 - texelUV.y);
            int2 texel = int2(ctrlUV * float2(ctrlW, ctrlH));

            uint4 ctrl0 = controlTex.Load(int4(texel, 0, 0));
            uint4 ctrl1 = controlTex.Load(int4(texel, 1, 0));

            uint4 allCtrl[2] = { ctrl0, ctrl1 };
            [unroll] for (uint ci = 0; ci < 8; ci++)
            {
                uint packed = allCtrl[ci / 4][ci % 4];
                uint slotIdx = packed >> 8;
                uint weight  = packed & 0xFF;
                if (slotIdx == 255 || weight == 0) break;

                activeSlots[activeCount] = slotIdx;
                activeWeights[activeCount] = weight;

                DecoratorSlot s = slots[slotIdx];
                float tileArea = tileSize * tileSize;
                uint maxInst = max(1u, min(255u, (uint)(s.Density * tileArea + 0.5)));
                activeCounts[activeCount] = max(1u, min(255u, (maxInst * weight + 127) / 255));

                if (s.Mode == MODE_BILLBOARD)
                    vpi = max(vpi, BILLBOARD_VERTS);
                else if (s.Mode == MODE_CROSS)
                    vpi = max(vpi, CROSS_VERTS);
                else if (s.LODCount > 0 && s.LODCount <= 8)
                {
                    uint vc = registry[lodTbl[s.LODTableOffset].MeshPartId].VertexCount;
                    vpi = max(vpi, vc);
                }
                activeCount++;
            }
        }
    }
    vpi = clamp(vpi, 3, 128);

    // Compute MS groups per decorator
    uint maxCount = 0;
    [unroll] for (uint mi = 0; mi < 8; mi++)
        maxCount = max(maxCount, activeCounts[mi]);

    uint ipg = max(1, min(MS_MAX_THREADS / vpi, MAX_VERTS / vpi));
    uint groupsPerDeco = (maxCount + ipg - 1) / ipg;

    GroupMemoryBarrierWithGroupSync();

    // Expand across cascades: dispatch totalMSGroups per visible cascade
    totalMSGroups = activeCount * groupsPerDeco;

    // Count visible cascades
    uint visibleCascades = 0;
    for (uint vc2 = 0; vc2 < cascadeCount; vc2++)
        visibleCascades += (cascadeMask >> vc2) & 1u;

    // Dead tile or no work: dispatch 0 MS groups (single DispatchMesh required by DXC)
    if (!tileAlive || activeCount == 0 || visibleCascades == 0)
    {
        totalMSGroups = 0;
        visibleCascades = 1; // Y must be >= 1 for valid DispatchMesh
    }

    // For cascade expansion: we encode the cascade index into the Y dimension of DispatchMesh.
    // Each Y slice = one cascade. MS reads gid.y as cascadeIdx.
    if (gidx == 0)
    {
        s_ShadowPayload.TileOrigin = tileOrigin;
        s_ShadowPayload.ActiveCount = activeCount;
        s_ShadowPayload.GroupsPerDecorator = groupsPerDeco;
        s_ShadowPayload.SlotIndices0 = Pack4x8(activeSlots[0], activeSlots[1], activeSlots[2], activeSlots[3]);
        s_ShadowPayload.SlotIndices1 = Pack4x8(activeSlots[4], activeSlots[5], activeSlots[6], activeSlots[7]);
        s_ShadowPayload.Weights0 = Pack4x8(activeWeights[0], activeWeights[1], activeWeights[2], activeWeights[3]);
        s_ShadowPayload.Weights1 = Pack4x8(activeWeights[4], activeWeights[5], activeWeights[6], activeWeights[7]);
        s_ShadowPayload.InstanceCounts0 = Pack4x8(activeCounts[0], activeCounts[1], activeCounts[2], activeCounts[3]);
        s_ShadowPayload.InstanceCounts1 = Pack4x8(activeCounts[4], activeCounts[5], activeCounts[6], activeCounts[7]);
        s_ShadowPayload.CascadeIdx = cascadeMask;
    }
    GroupMemoryBarrierWithGroupSync();

    DispatchMesh(totalMSGroups, visibleCascades, 1, s_ShadowPayload);
}

// Shadow MS per-vertex output
struct ShadowMSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    nointerpolation uint MaterialId      : TEXCOORD3;
    nointerpolation uint TextureOverride  : TEXCOORD4;
};

// Per-primitive attributes (SV_RenderTargetArrayIndex must be per-primitive in mesh shaders)
struct ShadowPrimAttribs
{
    uint RTIndex : SV_RenderTargetArrayIndex;
};

// Shadow mesh shader — projects with cascade-specific VP
[outputtopology("triangle")]
[numthreads(MS_MAX_THREADS, 1, 1)]
void MS_Shadow(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    in payload ShadowASPayload payload,
    out vertices ShadowMSOutput verts[MAX_VERTS],
    out indices uint3 tris[MAX_PRIMS],
    out primitives ShadowPrimAttribs primAttribs[MAX_PRIMS])
{
    // gid.x = decorator group index, gid.y = cascade slice within visible set
    uint groupsPerDeco = payload.GroupsPerDecorator;
    uint decoIdx = gid.x / groupsPerDeco;
    uint groupInDeco = gid.x % groupsPerDeco;

    // Decode which actual cascade index from the mask + gid.y
    uint cascadeMask = payload.CascadeIdx;
    uint cascadeIdx = 0;
    {
        uint count = 0;
        for (uint ci = 0; ci < 8; ci++)
        {
            if ((cascadeMask >> ci) & 1u)
            {
                if (count == gid.y)
                {
                    cascadeIdx = ci;
                    break;
                }
                count++;
            }
        }
    }

    // Read cascade VP for projection
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    row_major float4x4 cascadeVP = cascades[cascadeIdx].VP;

    uint slotIdx = Unpack8(payload.SlotIndices0, payload.SlotIndices1, decoIdx);
    uint decoInstCount = Unpack8(payload.InstanceCounts0, payload.InstanceCounts1, decoIdx);

    StructuredBuffer<DecoratorSlot> msSlots   = ResourceDescriptorHeap[DecoratorSlotsIdx];
    StructuredBuffer<LODEntry>      msLodTbl  = ResourceDescriptorHeap[LODTableIdx];
    StructuredBuffer<MeshPartEntry> msRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    DecoratorSlot slot = msSlots[slotIdx];

    uint vertsPerInst = 3;
    if (slot.Mode == MODE_BILLBOARD)
        vertsPerInst = BILLBOARD_VERTS;
    else if (slot.Mode == MODE_CROSS)
        vertsPerInst = CROSS_VERTS;
    else if (slot.LODCount > 0 && slot.LODCount <= 8)
        vertsPerInst = msRegistry[msLodTbl[slot.LODTableOffset].MeshPartId].VertexCount;
    vertsPerInst = clamp(vertsPerInst, 3, 128);

    uint trisPerInst = vertsPerInst / 3;
    uint ipg = max(1, min(MS_MAX_THREADS / vertsPerInst, MAX_VERTS / vertsPerInst));

    uint baseInstance = groupInDeco * ipg;
    uint batchCount = min(ipg, decoInstCount - min(baseInstance, decoInstCount));

    uint totalVerts = batchCount * vertsPerInst;
    uint totalTris = batchCount * trisPerInst;
    SetMeshOutputCounts(totalVerts, totalTris);

    if (gtid >= totalVerts) return;

    uint instInBatch = gtid / vertsPerInst;
    uint vertInInst = gtid % vertsPerInst;
    uint instanceIdx = baseInstance + instInBatch;

    // Per-instance placement (same hash as opaque MS)
    float2 instanceSeed = float2(float(instanceIdx) * 0.7123 + float(slotIdx) * 3.917,
                                  float(instanceIdx) * 1.3147 + float(slotIdx) * 7.213);
    float2 rng = hash22(instanceSeed + payload.TileOrigin);
    float cs = CellSize;
    float tileWorldSize = cs * TILE_SIZE;
    float wx = payload.TileOrigin.x + rng.x * tileWorldSize;
    float wz = payload.TileOrigin.y + rng.y * tileWorldSize;
    float2 seed = instanceSeed + payload.TileOrigin * 0.0137;  // include tile position for unique scale/rotation per tile

    float2 texelUV = float2(
        (wx - TerrainOriginX) / TerrainSizeX,
        (wz - TerrainOriginY) / TerrainSizeY
    );
    bool alive = texelUV.x >= 0 && texelUV.x <= 1 && texelUV.y >= 0 && texelUV.y <= 1;

    // Per-cell control check
    if (alive && DecoControlIdx != 0)
    {
        Texture2DArray<uint4> controlTex = ResourceDescriptorHeap[DecoControlIdx];
        uint ctrlW, ctrlH, ctrlSlices;
        controlTex.GetDimensions(ctrlW, ctrlH, ctrlSlices);
        float2 ctrlUV = float2(texelUV.x, 1.0 - texelUV.y);
        int2 ctrlTexel = int2(ctrlUV * float2(ctrlW, ctrlH));

        uint4 ctrl0 = controlTex.Load(int4(ctrlTexel, 0, 0));
        uint4 ctrl1 = controlTex.Load(int4(ctrlTexel, 1, 0));

        bool found = false;
        uint4 allCtrl[2] = { ctrl0, ctrl1 };
        [unroll] for (uint ci = 0; ci < 8; ci++)
        {
            uint packed = allCtrl[ci / 4][ci % 4];
            uint idx = packed >> 8;
            if (idx == 255) break;
            if (idx == slotIdx) { found = true; break; }
        }
        if (!found) alive = false;
    }

    // Heightmap & position
    float3 camPos = float3(CamPosX, CamPosY, CamPosZ);
    float h = 0;
    float3 terrainNormal = float3(0, 1, 0);
    if (alive)
    {
        Texture2D heightTex = ResourceDescriptorHeap[HeightmapIdx];
        h = heightTex.SampleLevel(HeightSampler, texelUV, 0).r;

        uint hmW, hmH;
        heightTex.GetDimensions(hmW, hmH);
        float texelStep = 1.0 / float(hmW);
        float hL = heightTex.SampleLevel(HeightSampler, texelUV + float2(-texelStep, 0), 0).r;
        float hR = heightTex.SampleLevel(HeightSampler, texelUV + float2( texelStep, 0), 0).r;
        float hD = heightTex.SampleLevel(HeightSampler, texelUV + float2(0, -texelStep), 0).r;
        float hU = heightTex.SampleLevel(HeightSampler, texelUV + float2(0,  texelStep), 0).r;
        float texelWorldSize = TerrainSizeX * texelStep;
        float heightScale = MaxHeight / texelWorldSize;
        terrainNormal = normalize(float3(
            (hL - hR) * heightScale,
            1.0,
            (hD - hU) * heightScale
        ));
    }

    float3 instancePos = float3(wx, TerrainOriginZ + h * MaxHeight, wz);
    if (alive && distance(instancePos, camPos) >= DecoRadius) alive = false;

    // Shadow Hi-Z occlusion: skip grass behind existing shadow geometry
    // Use conservative bias to account for grass height extending above base
    if (alive && ShadowHiZIdx != 0)
    {
        // Test a point above the base to account for grass height
        float grassMaxH = slot.MaxH;
        float3 testPos = instancePos + float3(0, grassMaxH, 0);
        float4 clipCenter = mul(float4(testPos, 1.0), cascadeVP);
        float3 ndc = clipCenter.xyz / clipCenter.w;
        float2 uv = ndc.xy * float2(0.5, -0.5) + 0.5;
        if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0)
        {
            Texture2DArray<float> shadowHiZ = ResourceDescriptorHeap[ShadowHiZIdx];
            float w, hh, elems, levels;
            shadowHiZ.GetDimensions(0, w, hh, elems, levels);
            // Coarse mip for conservative coverage
            uint mip = min(4, (uint)(levels - 1));
            float2 mipSize = max(float2(1,1), float2(w, hh) / (float)(1u << mip));
            int2 coord = int2(uv * mipSize);
            coord = clamp(coord, int2(0,0), int2(mipSize) - 1);
            float pyramidDepth = shadowHiZ.Load(int4(coord, cascadeIdx, mip));
            // Also test base position depth (nearest point to light)
            float4 clipBase = mul(float4(instancePos, 1.0), cascadeVP);
            float baseDepth = clipBase.z / clipBase.w;
            // Only cull if even the base (nearest) is behind the farthest existing geometry
            if (baseDepth > pyramidDepth) alive = false;
        }
    }

    // LOD — for shadows, always use LOD 0 (highest detail in shadow cascade range)
    uint lod = 0;
    if (alive)
    {
        float camDist = distance(instancePos, camPos);
        uint maxLod = min(slot.LODCount, 8);
        for (uint il = 0; il < maxLod; il++)
        {
            if (camDist > msLodTbl[slot.LODTableOffset + il].MaxDistance)
                lod = il + 1;
        }
        if (lod >= slot.LODCount) alive = false;
    }

    // Dead: degenerate triangle
    if (!alive)
    {
        ShadowMSOutput o = (ShadowMSOutput)0;
        o.Position = float4(0, 0, -1, 0);
        verts[gtid] = o;
        if (vertInInst % 3 == 0)
        {
            uint ti = instInBatch * trisPerInst + vertInInst / 3;
            uint bv = instInBatch * vertsPerInst + vertInInst;
            tris[ti] = uint3(bv, bv + 1, bv + 2);
            primAttribs[ti].RTIndex = cascadeIdx;
        }
        return;
    }

    // Scale & rotation
    float scaleH = lerp(slot.MinH, slot.MaxH, hash21(seed + 33.7));
    float scaleW = lerp(slot.MinW, slot.MaxW, hash21(seed.yx + 77.9));

    // Distance fade
    float camDist = distance(instancePos, camPos);
    float fadeStart = DecoRadius * 0.75;
    float fadeFactor = smoothstep(0.0, 1.0, (DecoRadius - camDist) / (DecoRadius - fadeStart));
    scaleH *= fadeFactor;
    scaleW *= fadeFactor;
    float instanceRot = hash21(seed + 51.3) * 6.2831853;
    float cosR = cos(instanceRot), sinR = sin(instanceRot);

    uint materialId = 0;
    if (slot.Mode == MODE_MESH)
    {
        LODEntry lodEntry = msLodTbl[slot.LODTableOffset + lod];
        materialId = lodEntry.MaterialId;
    }
    else if (slot.LODCount > 0)
        materialId = msLodTbl[slot.LODTableOffset].MaterialId;

    // Billboard / Cross
    if (slot.Mode == MODE_BILLBOARD || slot.Mode == MODE_CROSS)
    {
        uint totalQuadVerts = (slot.Mode == MODE_BILLBOARD) ? BILLBOARD_VERTS : CROSS_VERTS;
        if (vertInInst >= totalQuadVerts)
        {
            ShadowMSOutput o = (ShadowMSOutput)0;
            o.Position = float4(0, 0, -1, 0);
            verts[gtid] = o;
            if (vertInInst % 3 == 0)
            {
                uint ti = instInBatch * trisPerInst + vertInInst / 3;
                uint bv = instInBatch * vertsPerInst + vertInInst;
                tris[ti] = uint3(bv, bv + 1, bv + 2);
                primAttribs[ti].RTIndex = cascadeIdx;
            }
            return;
        }

        static const float2 quadPos[6] = {
            float2(-0.5, 0.0), float2(0.5, 0.0), float2(0.5, 1.0),
            float2(-0.5, 0.0), float2(0.5, 1.0), float2(-0.5, 1.0)
        };
        static const float2 quadUV[6] = {
            float2(0,1), float2(1,1), float2(1,0),
            float2(0,1), float2(1,0), float2(0,0)
        };

        uint quadIdx = vertInInst / 6;
        uint vertInQuad = vertInInst % 6;
        float2 lp = quadPos[vertInQuad];
        float2 uv = quadUV[vertInQuad];

        float3 slopeUp = normalize(lerp(float3(0, 1, 0), terrainNormal, slot.SlopeBias));
        float3 worldPos;
        if (slot.Mode == MODE_BILLBOARD)
        {
            float3 toCamera = normalize(float3(camPos.x - instancePos.x, 0, camPos.z - instancePos.z));
            float3 right = normalize(cross(slopeUp, toCamera));
            worldPos = instancePos + right * lp.x * scaleW + slopeUp * lp.y * scaleH;
        }
        else
        {
            float3 right;
            if (quadIdx == 0) right = float3(cosR, 0, sinR);
            else right = float3(-sinR, 0, cosR);
            right = normalize(right - slopeUp * dot(right, slopeUp));
            worldPos = instancePos + right * lp.x * scaleW + slopeUp * lp.y * scaleH;
        }

        // Wind
        float windInfluence = lp.y * lp.y;
        float windWave1 = sin(WindTime * 2.1 + worldPos.x * 0.8 + worldPos.z * 0.6) * 0.12;
        float windWave2 = sin(WindTime * 1.4 + worldPos.x * 0.5 - worldPos.z * 0.9) * 0.08;
        worldPos.x += (windWave1 + windWave2) * windInfluence * scaleH;
        worldPos.z += (windWave2 - windWave1 * 0.5) * windInfluence * scaleH;

        ShadowMSOutput o;
        o.Position = mul(float4(worldPos, 1.0), cascadeVP);
        o.TexCoord = uv;
        o.MaterialId = materialId;
        o.TextureOverride = slot.TextureIdx;
        verts[gtid] = o;

        if (vertInInst % 3 == 0)
        {
            uint ti = instInBatch * trisPerInst + vertInInst / 3;
            uint bv = instInBatch * vertsPerInst + vertInInst;
            tris[ti] = uint3(bv, bv + 1, bv + 2);
            primAttribs[ti].RTIndex = cascadeIdx;
        }
        return;
    }

    // Mesh mode
    LODEntry lodEntry = msLodTbl[slot.LODTableOffset + lod];
    uint meshPartId = lodEntry.MeshPartId;

    float3x3 rootMat = slot.RootMat;
    float3x3 instanceMat = float3x3(
        cosR * scaleW, 0, sinR * scaleW,
        0, scaleH, 0,
        -sinR * scaleW, 0, cosR * scaleW
    );

    float3 slopeUp = normalize(lerp(float3(0, 1, 0), terrainNormal, slot.SlopeBias));
    float3 axis = cross(float3(0, 1, 0), slopeUp);
    float sinA = length(axis);
    float cosA = dot(float3(0, 1, 0), slopeUp);
    float3x3 slopeMat = float3x3(1,0,0, 0,1,0, 0,0,1);
    if (sinA > 0.001)
    {
        axis = axis / sinA;
        float3x3 K = float3x3(0, -axis.z, axis.y, axis.z, 0, -axis.x, -axis.y, axis.x, 0);
        float3x3 I = float3x3(1,0,0, 0,1,0, 0,0,1);
        slopeMat = I + K * sinA + mul(K, K) * (1.0 - cosA);
    }

    float3x3 finalMat = mul(mul(rootMat, instanceMat), slopeMat);

    MeshPartEntry part = msRegistry[meshPartId];
    if (vertInInst >= part.VertexCount)
    {
        ShadowMSOutput o = (ShadowMSOutput)0;
        o.Position = float4(0, 0, -1, 0);
        verts[gtid] = o;
        if (vertInInst % 3 == 0)
        {
            uint ti = instInBatch * trisPerInst + vertInInst / 3;
            uint bv = instInBatch * vertsPerInst + vertInInst;
            tris[ti] = uint3(bv, bv + 1, bv + 2);
            primAttribs[ti].RTIndex = cascadeIdx;
        }
        return;
    }

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float2> uvs       = ResourceDescriptorHeap[part.UVBufferIdx];
    StructuredBuffer<uint>   indices   = ResourceDescriptorHeap[part.IndexBufferIdx];

    uint vertIdx = indices[vertInInst + part.BaseIndex];
    float3 pos = positions[vertIdx];
    float2 uv = uvs[vertIdx];

    float3 worldPos = mul(pos, finalMat) + instancePos;

    // Wind
    float localY = mul(pos, rootMat).y;
    float heightFrac = saturate(localY / max(scaleH, 0.01));
    float windInfluence = heightFrac * heightFrac;
    float windWave1 = sin(WindTime * 2.1 + worldPos.x * 0.8 + worldPos.z * 0.6) * 0.12;
    float windWave2 = sin(WindTime * 1.4 + worldPos.x * 0.5 - worldPos.z * 0.9) * 0.08;
    worldPos.x += (windWave1 + windWave2) * windInfluence * scaleH;
    worldPos.z += (windWave2 - windWave1 * 0.5) * windInfluence * scaleH;

    ShadowMSOutput o;
    o.Position = mul(float4(worldPos, 1.0), cascadeVP);
    o.TexCoord = float2(uv.x, 1 - uv.y);
    o.MaterialId = materialId;
    o.TextureOverride = 0;
    verts[gtid] = o;

    if (vertInInst % 3 == 0)
    {
        uint ti = instInBatch * trisPerInst + vertInInst / 3;
        uint bv = instInBatch * vertsPerInst + vertInInst;
        tris[ti] = uint3(bv, bv + 1, bv + 2);
        primAttribs[ti].RTIndex = cascadeIdx;
    }
}

// Single-pass shadow PS — same alpha test, different input struct
void PS_Shadow_SP(ShadowMSOutput input)
{
    if (input.TextureOverride > 0)
    {
        Texture2D texOverride = ResourceDescriptorHeap[input.TextureOverride];
        float alpha = texOverride.SampleLevel(HeightSampler, input.TexCoord, 0).a;
        clip(alpha - 0.25);
    }
    else
    {
        StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsBufferIdx];
        MaterialData mat = materials[input.MaterialId];

        if (mat.AlbedoIdx > 0)
        {
            Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
            float alpha = albedoTex.SampleLevel(HeightSampler, input.TexCoord, 0).a;
            clip(alpha - 0.25);
        }
    }
}

// ─── Technique ─────────────────────────────────────────────────────────────

// @RenderState(RenderTargets=4, CullMode=None)
technique11 GroundCoverage
{
    pass Opaque
    {
        SetAmplificationShader(CompileShader(as_6_5, AS()));
        SetMeshShader(CompileShader(ms_6_5, MS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }

    pass Shadow
    {
        SetAmplificationShader(CompileShader(as_6_6, AS_Shadow()));
        SetMeshShader(CompileShader(ms_6_6, MS_Shadow()));
        SetPixelShader(CompileShader(ps_6_6, PS_Shadow_SP()));
    }
}
