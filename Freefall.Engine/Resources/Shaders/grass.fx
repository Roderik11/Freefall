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
#define TotalDensity        asfloat(GET_INDEX(17))
#define ControlMapsIdx      GET_INDEX(18)

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
    uint ControlMapIndex;   // array slice in ControlMaps Texture2DArray
    uint ControlChannel;    // 0=R, 1=G, 2=B, 3=A
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
    uint   InstanceCount;
    uint   VertsPerInstance;
};

groupshared ASPayload s_Payload;

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void AS(uint3 gid : SV_GroupID, uint gidx : SV_GroupIndex)
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

    // Read vert count from first decoration's mesh
    StructuredBuffer<ChannelHeader> headers = ResourceDescriptorHeap[ChannelHeadersIdx];
    StructuredBuffer<DecoratorSlot> slots   = ResourceDescriptorHeap[DecoratorSlotsIdx];
    StructuredBuffer<LODEntry>      lodTbl  = ResourceDescriptorHeap[LODTableIdx];
    StructuredBuffer<MeshPartEntry> registry = ResourceDescriptorHeap[MeshRegistryIdx];

    uint vpi = 3;
    ChannelHeader asHdr = headers[0];
    uint hdrCount = min(asHdr.Count, 32); // cap for safety
    for (uint si = 0; si < hdrCount; si++)
    {
        DecoratorSlot s = slots[asHdr.StartIndex + si];
        if (s.LODCount > 0 && s.LODCount <= 8)
        {
            uint vc = registry[lodTbl[s.LODTableOffset].MeshPartId].VertexCount;
            vpi = max(vpi, vc);
        }
    }
    vpi = clamp(vpi, 3, 128);

    if (gidx == 0)
    {
        s_Payload.TileOrigin = tileOrigin;
        s_Payload.InstanceCount = tileAlive ? CELLS_PER_TILE : 0;
        s_Payload.VertsPerInstance = vpi;
    }
    GroupMemoryBarrierWithGroupSync();

    uint ipg = max(1, min(MS_MAX_THREADS / vpi, MAX_VERTS / vpi));
    uint msGroups = tileAlive ? ((CELLS_PER_TILE + ipg - 1) / ipg) : 0;
    DispatchMesh(msGroups, 1, 1, s_Payload);
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
    nointerpolation uint MaterialId : TEXCOORD3;
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
    uint vertsPerInst = payload.VertsPerInstance;
    uint trisPerInst = vertsPerInst / 3;
    uint ipg = max(1, min(MS_MAX_THREADS / vertsPerInst, MAX_VERTS / vertsPerInst));

    uint baseInstance = gid.x * ipg;
    uint batchCount = min(ipg, payload.InstanceCount - min(baseInstance, payload.InstanceCount));

    uint totalVerts = batchCount * vertsPerInst;
    uint totalTris = batchCount * trisPerInst;
    SetMeshOutputCounts(totalVerts, totalTris);

    if (gtid >= totalVerts) return;

    uint instInBatch = gtid / vertsPerInst;
    uint vertInInst = gtid % vertsPerInst;
    uint instanceIdx = baseInstance + instInBatch;

    // ── Per-instance data (computed by every vertex thread — no barrier needed) ──

    uint cellX = instanceIdx % TILE_SIZE;
    uint cellZ = instanceIdx / TILE_SIZE;

    float cs = CellSize;
    float wx = payload.TileOrigin.x + (cellX + 0.5) * cs;
    float wz = payload.TileOrigin.y + (cellZ + 0.5) * cs;

    float2 seed = float2(floor(wx / cs), floor(wz / cs));

    // Terrain bounds
    float2 texelUV = float2(
        (wx - TerrainOriginX) / TerrainSizeX,
        (wz - TerrainOriginY) / TerrainSizeY
    );
    bool alive = texelUV.x >= 0 && texelUV.x <= 1 && texelUV.y >= 0 && texelUV.y <= 1;

    // Density-weighted slot selection with control map gating.
    // Each slot's effective weight = density × controlMapWeight.
    // Slots not painted at this location get zero weight → never picked.
    StructuredBuffer<ChannelHeader> msHeaders = ResourceDescriptorHeap[ChannelHeadersIdx];
    StructuredBuffer<DecoratorSlot> msSlots   = ResourceDescriptorHeap[DecoratorSlotsIdx];
    StructuredBuffer<LODEntry>      msLodTbl  = ResourceDescriptorHeap[LODTableIdx];

    ChannelHeader hdr = msHeaders[0];
    if (hdr.Count == 0) alive = false;

    uint slotIdx = 0;
    DecoratorSlot slot = (DecoratorSlot)0;
    if (alive)
    {
        // First pass: density only — control map does NOT affect spawning
        float effectiveTotal = 0;
        for (uint s = 0; s < hdr.Count && s < 32; s++)
        {
            DecoratorSlot candidate = msSlots[hdr.StartIndex + s];
            effectiveTotal += candidate.Density;
        }

        if (effectiveTotal < 0.01) alive = false;

        // Second pass: density-weighted pick
        if (alive)
        {
            float rnd = hash21(seed) * effectiveTotal;
            float cumulative = 0;
            uint picked = 0;
            for (uint s2 = 0; s2 < hdr.Count && s2 < 32; s2++)
            {
                DecoratorSlot candidate = msSlots[hdr.StartIndex + s2];
                cumulative += candidate.Density;
                if (rnd < cumulative) { picked = s2; break; }
                picked = s2;
            }
            slotIdx = hdr.StartIndex + picked;
            slot = msSlots[slotIdx];
        }
    }

    // Heightmap & position
    float3 camPos = float3(CamPosX, CamPosY, CamPosZ);
    float h = 0;
    if (alive)
    {
        Texture2D heightTex = ResourceDescriptorHeap[HeightmapIdx];
        h = heightTex.SampleLevel(HeightSampler, texelUV, 0).r;
    }

    float2 jitter = hash22(seed + 73.1) - 0.5;  // [-0.5, 0.5] keeps instance within cell
    float3 instancePos = float3(wx + jitter.x * cs, TerrainOriginZ + h * MaxHeight, wz + jitter.y * cs);

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

    // Control map: scales instance size, smoothstep cull if too small
    float ctrlScale = 1.0;
    if (alive && ControlMapsIdx != 0)
    {
        Texture2DArray controlMaps = ResourceDescriptorHeap[ControlMapsIdx];
        float2 ctrlUV = float2(texelUV.x, 1.0 - texelUV.y);
        float4 ctrl = controlMaps.SampleLevel(HeightSampler, float3(ctrlUV, slot.ControlMapIndex), 0);
        float4 mask = float4(
            slot.ControlChannel == 0 ? 1 : 0,
            slot.ControlChannel == 1 ? 1 : 0,
            slot.ControlChannel == 2 ? 1 : 0,
            slot.ControlChannel == 3 ? 1 : 0);
        ctrlScale = dot(ctrl, mask);
        ctrlScale = smoothstep(0.15, 0.5, ctrlScale);
        if (ctrlScale < 0.01) alive = false;
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

    // ── Transform (root rotation precomputed on CPU, only instance Y-rot on GPU) ──

    LODEntry lodEntry = msLodTbl[slot.LODTableOffset + lod];
    uint meshPartId = lodEntry.MeshPartId;
    uint materialId = lodEntry.MaterialId;

    float instanceScale = lerp(slot.MinH, slot.MaxH, hash21(seed + 33.7)) * ctrlScale;
    float instanceRot = hash21(seed + 51.3) * 6.2831853;

    // Root rotation: read precomputed matrix from slot (zero trig)
    float3x3 rootMat = slot.RootMat;

    // Instance Y-rotation + scale (only 2 trig calls)
    float cosR = cos(instanceRot), sinR = sin(instanceRot);
    float3x3 instanceMat = float3x3(
        cosR * instanceScale, 0, sinR * instanceScale,
        0, instanceScale, 0,
        -sinR * instanceScale, 0, cosR * instanceScale
    );

    float3x3 finalMat = mul(rootMat, instanceMat);

    // ── Vertex fetch & transform ──

    StructuredBuffer<MeshPartEntry> msRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
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

    MSOutput o;
    float4 wp = float4(worldPos, 1.0);
    o.WorldPos = wp;
    o.Position = mul(wp, ViewProjection);
    o.Normal = worldNorm;
    o.TexCoord = float2(uv.x, 1 - uv.y);
    o.Depth = o.Position.w;
    o.MaterialId = materialId;
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
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsBufferIdx];
    MaterialData mat = materials[input.MaterialId];

    float3 baseColor;
    float alpha = 1.0;
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

    clip(alpha - 0.5);

    output.Albedo = float4(baseColor, 1.0);
    output.Normal = float4(normalize(input.Normal), 1.0);
    output.Data = float4(0.9, 0.0, 1.0, 1.0);
    output.Depth = input.Depth;
    return output;
}

// ─── Shadow Pixel Shader ───────────────────────────────────────────────────
// Depth-only: alpha test, hardware writes depth. No MRT output.

void PS_Shadow(MSOutput input)
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

    // @RenderState(RenderTargets=0, CullMode=None)
    pass Shadow
    {
        SetAmplificationShader(CompileShader(as_6_5, AS()));
        SetMeshShader(CompileShader(ms_6_5, MS()));
        SetPixelShader(CompileShader(ps_6_6, PS_Shadow()));
    }
}
