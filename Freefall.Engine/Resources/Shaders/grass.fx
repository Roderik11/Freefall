// ═══════════════════════════════════════════════════════════════════════════
// Ground Coverage Shader — Lean Amplification → Mesh → Pixel
// Instance data is pre-computed by CS_SpawnInstances (grass_compute.hlsl).
// AS does trivial grouping, MS expands billboard/cross quads from DecoInstance.
// Mesh-mode decorators are rendered by a separate VS/PS pipeline.
// ═══════════════════════════════════════════════════════════════════════════
cbuffer PushConstants : register(b3)
{
    uint DecoratorSlotsIdx;     // 0
    uint LODTableIdx;           // 1
    uint MeshRegistryIdx;       // 2
    uint DecoInstanceSRV;       // 3
    uint InstanceCount;         // 4
    uint MaterialsBufferIdx;    // 5
    uint BakedAlbedoIdx;        // 6
    uint _reserved7;            // 7: was WindTime (now in SceneConstants)
    uint _reserved8;            // 8: was CamPosX
    uint _reserved9;            // 9: was CamPosY
    uint _reserved10;           // 10: was CamPosZ
    uint CascadeBufferSRVIdx;   // 11: Shadow-only
    uint ShadowCascadeCount;    // 12
};

#include "common.fx"

// ─── GPU structs ───────────────────────────────────────────────────────────

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

// Per-instance data — produced by CS_SpawnInstances (grass_compute.hlsl)
struct DecoInstance
{
    float3 Position;
    float  Rotation;
    float3 TerrainNormal;
    float  FadeFactor;
    float2 Scale;
    float2 TerrainUV;
    uint   SlotIdx;
    uint   LOD;
    float  InstanceSeed;
    uint   _pad;
};  // 64 bytes


// ─── Constants ─────────────────────────────────────────────────────────────

#define INSTANCES_PER_GROUP 16
#define VERTS_PER_INSTANCE 8     // 2 quads max (cross); billboard uses 4, second degenerate
#define TRIS_PER_INSTANCE  4
#define MS_MAX_THREADS 128        // 16 instances × 8 verts = 128
#define MAX_VERTS 128
#define MAX_PRIMS 64

#define MODE_MESH      0
#define MODE_BILLBOARD 1
#define MODE_CROSS     2

SamplerState HeightSampler : register(s0);
SamplerState ClampSampler  : register(s2);

// ─── Amplification Shader ──────────────────────────────────────────────────
// Trivial instance grouping — billboard/cross only.
// Mesh mode decorators are rendered via a separate VS/PS pipeline.

struct ASPayload
{
    uint BaseInstance;
    uint BatchCount;
};

groupshared ASPayload s_Payload;

[numthreads(1, 1, 1)]
void AS(uint3 gid : SV_GroupID)
{
    uint totalInstances = InstanceCount;
    uint baseInst = gid.x * INSTANCES_PER_GROUP;
    uint batchCount = min(INSTANCES_PER_GROUP, totalInstances - min(baseInst, totalInstances));

    s_Payload.BaseInstance = baseInst;
    s_Payload.BatchCount = batchCount;

    uint groups = (batchCount > 0) ? 1u : 0u;
    DispatchMesh(groups, 1, 1, s_Payload);
}

// ─── Mesh Shader ───────────────────────────────────────────────────────────
// Reads DecoInstance, expands billboard/cross quads. No texture samples.

struct MSOutput
{
    float4 Position : SV_POSITION;
    float3 Normal   : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 WorldPos : TEXCOORD1;
    float  Depth    : TEXCOORD2;
    nointerpolation uint MaterialId      : TEXCOORD3;
    nointerpolation uint TextureOverride  : TEXCOORD4;
    float  HeightFrac : TEXCOORD5;
    float2 TerrainUV  : TEXCOORD6;
    nointerpolation float InstanceSeed : TEXCOORD7;
    nointerpolation uint SlotIdx : TEXCOORD8;
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
    uint batchCount = payload.BatchCount;
    uint totalV = min(batchCount * VERTS_PER_INSTANCE, MAX_VERTS);
    uint totalT = min(batchCount * TRIS_PER_INSTANCE, MAX_PRIMS);
    SetMeshOutputCounts(totalV, totalT);

    if (gtid >= totalV) return;

    uint instInBatch = gtid / VERTS_PER_INSTANCE;
    uint vertInInst  = gtid % VERTS_PER_INSTANCE;  // 0-7
    uint quadIdx     = vertInInst / 4;               // 0 = first quad, 1 = second quad
    uint vertInQuad  = vertInInst % 4;               // 0-3 within each quad
    uint globalInst  = payload.BaseInstance + instInBatch;

    StructuredBuffer<DecoInstance>   instances = ResourceDescriptorHeap[DecoInstanceSRV];
    StructuredBuffer<DecoratorSlot> slots     = ResourceDescriptorHeap[DecoratorSlotsIdx];
    StructuredBuffer<LODEntry>      lodTbl    = ResourceDescriptorHeap[LODTableIdx];
    float3 camPos = CamPos;

    // Degenerate output helper
    bool degenerate = (instInBatch >= batchCount || globalInst >= InstanceCount);

    DecoInstance di;
    DecoratorSlot slot;
    LODEntry lodEntry;
    if (!degenerate)
    {
        di = instances[globalInst];
        slot = slots[di.SlotIdx];
        if (slot.Mode == MODE_MESH) degenerate = true;
        // Billboard: second quad (quadIdx==1) is degenerate
        if (slot.Mode == MODE_BILLBOARD && quadIdx == 1) degenerate = true;
    }

    if (degenerate)
    {
        MSOutput o = (MSOutput)0;
        o.Position = float4(0, 0, -1, 0);
        verts[gtid] = o;
        if (vertInQuad < 2)
        {
            uint ti = instInBatch * TRIS_PER_INSTANCE + quadIdx * 2 + vertInQuad;
            uint bv = instInBatch * VERTS_PER_INSTANCE + quadIdx * 4;
            if (vertInQuad == 0) tris[ti] = uint3(bv, bv+1, bv+2);
            else                 tris[ti] = uint3(bv, bv+2, bv+3);
        }
        return;
    }

    float scaleW = di.Scale.x;
    float scaleH = di.Scale.y;
    float cosR = cos(di.Rotation);
    float sinR = sin(di.Rotation);

    lodEntry = lodTbl[slot.LODTableOffset + di.LOD];

    static const float2 quadPos[4] = {
        float2(-0.5, 0.0), float2(0.5, 0.0), float2(0.5, 1.0), float2(-0.5, 1.0)
    };
    static const float2 quadUV[4] = {
        float2(0, 1), float2(1, 1), float2(1, 0), float2(0, 0)
    };

    float2 lp = quadPos[vertInQuad];
    float2 uv = quadUV[vertInQuad];
    // Random X-flip for visual variety (seed bit → 50% chance)
    if (frac(di.InstanceSeed * 7.77) > 0.5) uv.x = 1.0 - uv.x;
    float3 worldPos, worldNorm;

    float3 slopeUp = normalize(lerp(float3(0, 1, 0), di.TerrainNormal, abs(slot.SlopeBias)));

    if (slot.Mode == MODE_BILLBOARD)
    {
        // quadIdx == 0 only (quadIdx == 1 is degenerate for billboard)
        float3 toCamera = normalize(float3(camPos.x - di.Position.x, 0, camPos.z - di.Position.z));
        float3 right = normalize(cross(slopeUp, toCamera));
        worldPos = di.Position + right * lp.x * scaleW + slopeUp * lp.y * scaleH;
        worldNorm = di.TerrainNormal;
    }
    else // MODE_CROSS — two perpendicular quads
    {
        // Quad 0: along rotation axis. Quad 1: perpendicular (rotated 90°).
        float cR = (quadIdx == 0) ? cosR : -sinR;
        float sR = (quadIdx == 0) ? sinR :  cosR;
        float3 right = float3(cR, 0, sR);
        right = normalize(right - slopeUp * dot(right, slopeUp));
        worldPos = di.Position + right * lp.x * scaleW + slopeUp * lp.y * scaleH;
        worldNorm = di.TerrainNormal;
    }

    // Wind
    float windInfluence = lp.y * lp.y;
    float windWave1 = sin(Time * 2.1 + worldPos.x * 0.8 + worldPos.z * 0.6) * 0.12;
    float windWave2 = sin(Time * 1.4 + worldPos.x * 0.5 - worldPos.z * 0.9) * 0.08;
    worldPos.x += (windWave1 + windWave2) * windInfluence * scaleH;
    worldPos.z += (windWave2 - windWave1 * 0.5) * windInfluence * scaleH;

    MSOutput o;
    float4 wp = float4(worldPos, 1.0);
    o.WorldPos = wp;
    o.Position = mul(wp, ViewProjection);
    o.Normal = worldNorm;
    o.TexCoord = uv;
    o.Depth = o.Position.w;
    o.MaterialId = lodEntry.MaterialId;
    o.TextureOverride = slot.TextureIdx;
    o.HeightFrac = saturate(lp.y);
    o.TerrainUV = di.TerrainUV;
    o.InstanceSeed = di.InstanceSeed;
    o.SlotIdx = di.SlotIdx;
    verts[gtid] = o;

    if (vertInQuad < 2)
    {
        uint ti = instInBatch * TRIS_PER_INSTANCE + quadIdx * 2 + vertInQuad;
        uint bv = instInBatch * VERTS_PER_INSTANCE + quadIdx * 4;
        if (vertInQuad == 0) tris[ti] = uint3(bv, bv+1, bv+2);
        else                 tris[ti] = uint3(bv, bv+2, bv+3);
    }
}

// ─── Pixel Shader ──────────────────────────────────────────────────────────

struct PSOutput
{
    float4 Albedo : SV_Target0;
    float4 Normal : SV_Target1;
    float4 Data   : SV_Target2;
    float  Depth  : SV_Target3;
    uint   EntityId : SV_Target4;
};

PSOutput PS(MSOutput input)
{
    PSOutput output;

    float dist = input.Depth;
    float3 baseColor;
    float alpha = 1.0;

    if (input.TextureOverride > 0)
    {
        // ── Billboard / Cross path ──
        Texture2D texOverride = ResourceDescriptorHeap[input.TextureOverride];
        float4 texColor = texOverride.Sample(ClampSampler, input.TexCoord);
        baseColor = texColor.rgb;
        alpha = texColor.a;

        // Distance-based alpha threshold: tight clip up close, loose at distance
        //float alphaThreshold = lerp(0.25, 0.05, saturate(dist / 80.0));
        clip(alpha - 0.15);

        // Ground color fade
        //Texture2D BakedAlbedo = ResourceDescriptorHeap[BakedAlbedoIdx];
        //float3 groundColor = BakedAlbedo.SampleLevel(ClampSampler, input.TerrainUV, 0).rgb;
        float aoFactor = saturate(input.HeightFrac / 0.33);
        float brightVar = 0.6 + 0.4 * frac(input.InstanceSeed * 17.31);
        float3 variedBase = baseColor * brightVar;

        // Apply decorator color tint
        StructuredBuffer<DecoratorSlot> psSlots = ResourceDescriptorHeap[DecoratorSlotsIdx];
        DecoratorSlot psSlot = psSlots[input.SlotIdx];
        variedBase *= psSlot.HealthyColor;

        output.Albedo = float4(variedBase, 1.0);
        output.Normal = float4(normalize(input.Normal), 1.0); // terrain normal
    }
    else
    {
        // ── Mesh path (future — uses material albedo) ──
        StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsBufferIdx];
        MaterialData mat = materials[input.MaterialId];

        if (mat.AlbedoIdx > 0)
        {
            Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
            float4 texColor = albedoTex.Sample(ClampSampler, input.TexCoord);
            baseColor = texColor.rgb;
            alpha = texColor.a;
            clip(alpha - 0.25);
        }
        else
        {
            baseColor = float3(0.5, 0.5, 0.5);
        }

        output.Albedo = float4(baseColor, 1.0);
        output.Normal = float4(normalize(input.Normal), 1.0);
    }

    output.Data = float4(0.95, 0.0, 1.0, 0.5);   // roughness=matte, metallic=0, ao=1, vegetation lighting
    output.Depth = dist;
    output.EntityId = 0;

    return output;
}

// ─── Shadow Rendering ──────────────────────────────────────────────────────

struct ShadowASPayload
{
    uint BaseInstance;
    uint BatchCount;
};

groupshared ShadowASPayload s_ShadowPayload;

[numthreads(1, 1, 1)]
void AS_Shadow(uint3 gid : SV_GroupID)
{
    uint totalInstances = InstanceCount;
    uint baseInst = gid.x * INSTANCES_PER_GROUP;
    uint batchCount = min(INSTANCES_PER_GROUP, totalInstances - min(baseInst, totalInstances));

    s_ShadowPayload.BaseInstance = baseInst;
    s_ShadowPayload.BatchCount = batchCount;

    uint cascadeCount = max(1u, ShadowCascadeCount);
    uint groups = (batchCount > 0) ? 1u : 0u;
    DispatchMesh(groups, cascadeCount, 1, s_ShadowPayload);
}

// Shadow MS output
struct ShadowMSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    float  Depth    : TEXCOORD1;
    nointerpolation uint MaterialId      : TEXCOORD3;
    nointerpolation uint TextureOverride  : TEXCOORD4;
};

struct ShadowPrimAttribs
{
    uint RTIndex : SV_RenderTargetArrayIndex;
};

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
    uint cascadeIdx = gid.y;
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    row_major float4x4 cascadeVP = cascades[cascadeIdx].VP;

    uint batchCount = payload.BatchCount;
    uint totalV = min(batchCount * VERTS_PER_INSTANCE, MAX_VERTS);
    uint totalT = min(batchCount * TRIS_PER_INSTANCE, MAX_PRIMS);
    SetMeshOutputCounts(totalV, totalT);

    if (gtid >= totalV) return;

    uint instInBatch = gtid / VERTS_PER_INSTANCE;
    uint vertInInst  = gtid % VERTS_PER_INSTANCE;
    uint quadIdx     = vertInInst / 4;
    uint vertInQuad  = vertInInst % 4;
    uint globalInst  = payload.BaseInstance + instInBatch;

    StructuredBuffer<DecoInstance>   instances = ResourceDescriptorHeap[DecoInstanceSRV];
    StructuredBuffer<DecoratorSlot> slots     = ResourceDescriptorHeap[DecoratorSlotsIdx];
    StructuredBuffer<LODEntry>      lodTbl    = ResourceDescriptorHeap[LODTableIdx];
    float3 camPos = CamPos;

    bool degenerate = (instInBatch >= batchCount || globalInst >= InstanceCount);

    DecoInstance di;
    DecoratorSlot slot;
    LODEntry lodEntry;
    if (!degenerate)
    {
        di = instances[globalInst];
        slot = slots[di.SlotIdx];
        if (slot.Mode == MODE_MESH) degenerate = true;
        if (slot.Mode == MODE_BILLBOARD && quadIdx == 1) degenerate = true;
    }

    if (degenerate)
    {
        ShadowMSOutput o = (ShadowMSOutput)0;
        o.Position = float4(0, 0, -1, 0);
        verts[gtid] = o;
        if (vertInQuad < 2)
        {
            uint ti = instInBatch * TRIS_PER_INSTANCE + quadIdx * 2 + vertInQuad;
            uint bv = instInBatch * VERTS_PER_INSTANCE + quadIdx * 4;
            if (vertInQuad == 0) tris[ti] = uint3(bv, bv+1, bv+2);
            else                 tris[ti] = uint3(bv, bv+2, bv+3);
            primAttribs[ti].RTIndex = cascadeIdx;
        }
        return;
    }

    float scaleW = di.Scale.x;
    float scaleH = di.Scale.y;
    float cosR = cos(di.Rotation);
    float sinR = sin(di.Rotation);
    lodEntry = lodTbl[slot.LODTableOffset + di.LOD];

    static const float2 quadPos[4] = {
        float2(-0.5, 0.0), float2(0.5, 0.0), float2(0.5, 1.0), float2(-0.5, 1.0)
    };
    static const float2 quadUV[4] = {
        float2(0, 1), float2(1, 1), float2(1, 0), float2(0, 0)
    };

    float2 lp = quadPos[vertInQuad];
    float2 uv = quadUV[vertInQuad];
    if (frac(di.InstanceSeed * 7.77) > 0.5) uv.x = 1.0 - uv.x;
    float3 slopeUp = normalize(lerp(float3(0, 1, 0), di.TerrainNormal, abs(slot.SlopeBias)));
    float3 worldPos;

    if (slot.Mode == MODE_BILLBOARD)
    {
        float3 toCamera = normalize(float3(camPos.x - di.Position.x, 0, camPos.z - di.Position.z));
        float3 right = normalize(cross(slopeUp, toCamera));
        worldPos = di.Position + right * lp.x * scaleW + slopeUp * lp.y * scaleH;
    }
    else // MODE_CROSS — two perpendicular quads
    {
        float cR = (quadIdx == 0) ? cosR : -sinR;
        float sR = (quadIdx == 0) ? sinR :  cosR;
        float3 right = float3(cR, 0, sR);
        right = normalize(right - slopeUp * dot(right, slopeUp));
        worldPos = di.Position + right * lp.x * scaleW + slopeUp * lp.y * scaleH;
    }

    // Wind
    float windInfluence = lp.y * lp.y;
    float windWave1 = sin(Time * 2.1 + worldPos.x * 0.8 + worldPos.z * 0.6) * 0.12;
    float windWave2 = sin(Time * 1.4 + worldPos.x * 0.5 - worldPos.z * 0.9) * 0.08;
    worldPos.x += (windWave1 + windWave2) * windInfluence * scaleH;
    worldPos.z += (windWave2 - windWave1 * 0.5) * windInfluence * scaleH;

    ShadowMSOutput o;
    o.Position = mul(float4(worldPos, 1.0), cascadeVP);
    o.TexCoord = uv;
    o.Depth = o.Position.w;
    o.MaterialId = lodEntry.MaterialId;
    o.TextureOverride = slot.TextureIdx;
    verts[gtid] = o;

    if (vertInQuad < 2)
    {
        uint ti = instInBatch * TRIS_PER_INSTANCE + quadIdx * 2 + vertInQuad;
        uint bv = instInBatch * VERTS_PER_INSTANCE + quadIdx * 4;
        if (vertInQuad == 0) tris[ti] = uint3(bv, bv+1, bv+2);
        else                 tris[ti] = uint3(bv, bv+2, bv+3);
        primAttribs[ti].RTIndex = cascadeIdx;
    }
}

// ─── Shadow Pixel Shader ───────────────────────────────────────────────────

void PS_Shadow_SP(ShadowMSOutput input)
{
    if (input.TextureOverride > 0)
    {
        Texture2D texOverride = ResourceDescriptorHeap[input.TextureOverride];
        float alpha = texOverride.SampleLevel(HeightSampler, input.TexCoord, 0).a;
        clip(alpha - 0.15);
    }
    else
    {
        StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsBufferIdx];
        MaterialData mat = materials[input.MaterialId];

        if (mat.AlbedoIdx > 0)
        {
            Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
            float alpha = albedoTex.SampleLevel(HeightSampler, input.TexCoord, 0).a;

            // Distance-based alpha threshold: tight clip up close, loose at distance
            // so mip-averaged alpha still passes at mid/far range
            //float alphaThreshold = lerp(0.15, 0.015, saturate(input.Depth / 60.0));
            //clip(alpha - alphaThreshold);

            clip(alpha - 0.15);
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
