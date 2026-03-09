// grass_mesh.fx — VS/PS for mesh-mode terrain decorators (rocks, pebbles, flowers)
// Rendered via BindlessCommandSignature ExecuteIndirect (per-mesh-type batched draws)
// @RenderState(RenderTargets=4)

cbuffer PushConstants : register(b3)
{
    uint _reserved0;
    uint _reserved1;
    uint MeshPartIdIdx;         // 2: MeshPartEntry index in MeshRegistry
    uint InstanceBaseIdx;       // 3: base offset in sorted instance buffer
    uint SortedInstancesIdx;    // 4: SRV: sorted DecoInstance buffer
    uint DecoratorSlotsIdx;     // 5: SRV: DecoratorSlot buffer
    uint LODTableIdx;           // 6: SRV: LODEntry buffer
    uint MeshRegistryIdx;       // 7: SRV: MeshPartEntry buffer
    uint CascadeBufferSRVIdx;   // 8: SRV: CascadeData for shadow pass
    uint _reserved9;
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint MaterialsIdx;          // 14: from common.fx
    uint GlobalTransformBufferIdx; // 15
};

#include "common.fx"

inline MaterialData GetMaterial(uint id)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    return materials[id];
}
#define GET_MATERIAL(id) GetMaterial(id)

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
};

// ─── VS/PS Structures ──────────────────────────────────────────────────────

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 WorldPos : TEXCOORD1;
    nointerpolation uint MaterialID : TEXCOORD2;
    float Depth : TEXCOORD3;
    nointerpolation uint SlotIdx : TEXCOORD4;
};

struct PSOutput
{
    float4 Albedo : SV_Target0;
    float4 Normal : SV_Target1;
    float4 Data : SV_Target2;
    float  Depth : SV_Target3;
};

SamplerState Sampler : register(s0);

// ─── Vertex Shader ─────────────────────────────────────────────────────────

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output = (VSOutput)0;

    // Per-draw root constants from BindlessCommandSignature
    uint meshPartId = MeshPartIdIdx;
    uint instanceBase = InstanceBaseIdx;

    // Fetch mesh part metadata
    StructuredBuffer<MeshPartEntry> meshReg = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshReg[meshPartId];

    // Vertex pulling: index buffer → vertex index → position/normal/UV
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[part.IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + part.BaseIndex];

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[part.NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];

    float3 localPos = positions[vertexID];
    float3 localNorm = normals[vertexID];
    float2 uv = uvs[vertexID];

    // Instance data from sorted buffer
    StructuredBuffer<DecoInstance> instances = ResourceDescriptorHeap[SortedInstancesIdx];
    DecoInstance di = instances[instanceBase + instanceID];

    // Decorator slot
    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    DecoratorSlot slot = slots[di.SlotIdx];

    // Step 1: scale in mesh-native space (W, H, W)
    float3 lp = localPos * float3(di.Scale.x, di.Scale.y, di.Scale.x);
    float3 ln = localNorm;

    // Step 2: rootMat corrects mesh orientation (e.g. Z-up → Y-up)
    float3x3 rootMat = float3x3(
        slot.Rot00, slot.Rot01, slot.Rot02,
        slot.Rot10, slot.Rot11, slot.Rot12,
        slot.Rot20, slot.Rot21, slot.Rot22
    );
    lp = mul(lp, rootMat);
    ln = mul(ln, rootMat);

    // Step 3: slope-aligned world frame with Y rotation
    float cosR = cos(di.Rotation);
    float sinR = sin(di.Rotation);
    float3 slopeUp = normalize(lerp(float3(0, 1, 0), di.TerrainNormal, abs(slot.SlopeBias)));
    float3 right = float3(cosR, 0, sinR);
    right = normalize(right - slopeUp * dot(right, slopeUp));
    float3 fwd = normalize(cross(right, slopeUp));

    // Step 4: place in world
    float3 worldPos = right * lp.x + slopeUp * lp.y + fwd * lp.z + di.Position;
    float3 worldNorm = normalize(right * ln.x + slopeUp * ln.y + fwd * ln.z);

    output.WorldPos = float4(worldPos, 1.0);
    output.Position = mul(float4(worldPos, 1.0), ViewProjection);
    output.Normal = worldNorm;
    output.TexCoord = uv;
    output.TexCoord.y = 1 - output.TexCoord.y; // OpenGL → DX UV flip
    output.Depth = output.Position.w;
    output.SlotIdx = di.SlotIdx;

    // Material from LOD table
    StructuredBuffer<LODEntry> lodTbl = ResourceDescriptorHeap[LODTableIdx];
    LODEntry lod = lodTbl[slot.LODTableOffset + di.LOD];
    output.MaterialID = lod.MaterialId;

    return output;
}

// ─── Pixel Shader ──────────────────────────────────────────────────────────

PSOutput PS(VSOutput input)
{
    PSOutput output;

    MaterialData mat = GET_MATERIAL(input.MaterialID);

    // Sample albedo
    float3 baseColor = float3(0.5, 0.5, 0.5);
    float alpha = 1.0;

    if (mat.AlbedoIdx > 0)
    {
        Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
        float4 texColor = albedoTex.Sample(Sampler, input.TexCoord);
        baseColor = texColor.rgb;
        alpha = texColor.a;
        clip(alpha - 0.25);
    }

    // Apply decorator color tint
    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    DecoratorSlot slot = slots[input.SlotIdx];
    if (slot.HealthyColor.x + slot.HealthyColor.y + slot.HealthyColor.z > 0.01)
        baseColor *= slot.HealthyColor;

    // Normal mapping
    float3 N = normalize(input.Normal);

    if (mat.NormalIdx != 0)
    {
        // Cotangent frame from screen-space derivatives
        float3 dp1 = ddx(input.WorldPos.xyz);
        float3 dp2 = ddy(input.WorldPos.xyz);
        float2 duv1 = ddx(input.TexCoord);
        float2 duv2 = ddy(input.TexCoord);

        float3 dp2perp = cross(dp2, N);
        float3 dp1perp = cross(N, dp1);
        float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
        float3 B = dp2perp * duv1.y + dp1perp * duv2.y;
        float invmax = rsqrt(max(dot(T, T), dot(B, B)));
        float3x3 TBN = float3x3(T * invmax, B * invmax, N);

        Texture2D normalTex = ResourceDescriptorHeap[mat.NormalIdx];
        float2 nXY = normalTex.Sample(Sampler, input.TexCoord).rg * 2.0 - 1.0;
        nXY.y = -nXY.y;
        float3 texNormal = float3(nXY, sqrt(max(0.001, 1.0 - dot(nXY, nXY))));
        N = normalize(mul(texNormal, TBN));
    }

    output.Albedo = float4(baseColor, 1.0);
    output.Normal = float4(N, 1.0);
    output.Data = float4(0.85, 0.0, 1.0, 0.5); // roughness, metallic, ao, vegetation lighting flag
    output.Depth = input.Depth;

    return output;
}


// Shadow structures

struct ShadowVSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    nointerpolation uint MaterialID : TEXCOORD1;
    nointerpolation uint RTIndex : SV_RenderTargetArrayIndex;
};

ShadowVSOutput VS_Shadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    ShadowVSOutput output = (ShadowVSOutput)0;

    uint meshPartId = MeshPartIdIdx;
    uint instanceBase = InstanceBaseIdx;

    StructuredBuffer<MeshPartEntry> meshReg = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshReg[meshPartId];

    StructuredBuffer<uint> indices = ResourceDescriptorHeap[part.IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + part.BaseIndex];

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];

    float3 localPos = positions[vertexID];
    float2 uv = uvs[vertexID];

    StructuredBuffer<DecoInstance> instances = ResourceDescriptorHeap[SortedInstancesIdx];
    DecoInstance di = instances[instanceBase + instanceID];

    StructuredBuffer<DecoratorSlot> slots = ResourceDescriptorHeap[DecoratorSlotsIdx];
    DecoratorSlot slot = slots[di.SlotIdx];

    float3 lp = localPos * float3(di.Scale.x, di.Scale.y, di.Scale.x);
    float3x3 rootMat = float3x3(
        slot.Rot00, slot.Rot01, slot.Rot02,
        slot.Rot10, slot.Rot11, slot.Rot12,
        slot.Rot20, slot.Rot21, slot.Rot22
    );
    lp = mul(lp, rootMat);

    float cosR = cos(di.Rotation);
    float sinR = sin(di.Rotation);
    float3 slopeUp = normalize(lerp(float3(0, 1, 0), di.TerrainNormal, abs(slot.SlopeBias)));
    float3 right = float3(cosR, 0, sinR);
    right = normalize(right - slopeUp * dot(right, slopeUp));
    float3 fwd = normalize(cross(right, slopeUp));
    float3 worldPos = right * lp.x + slopeUp * lp.y + fwd * lp.z + di.Position;

    // Shadow cascade selection — use cascade 0 for now (single cascade support)
    // TODO: multi-cascade expansion like gbuffer.fx
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    output.Position = mul(float4(worldPos, 1.0), cascades[0].VP);
    output.RTIndex = 0;
    output.TexCoord = float2(uv.x, 1 - uv.y);

    StructuredBuffer<LODEntry> lodTbl = ResourceDescriptorHeap[LODTableIdx];
    LODEntry lod = lodTbl[slot.LODTableOffset + di.LOD];
    output.MaterialID = lod.MaterialId;

    return output;
}

void PS_Shadow(ShadowVSOutput input)
{
    MaterialData mat = GET_MATERIAL(input.MaterialID);
    if (mat.AlbedoIdx > 0)
    {
        Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
        float alpha = albedoTex.Sample(Sampler, input.TexCoord).a;
        clip(alpha - 0.25f);
    }
}

// ─── Techniques ────────────────────────────────────────────────────────────

technique11 GrassMesh
{
    pass Opaque
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }

    pass Shadow
    {
        SetVertexShader(CompileShader(vs_6_6, VS_Shadow()));
        SetPixelShader(CompileShader(ps_6_6, PS_Shadow()));
    }
}
