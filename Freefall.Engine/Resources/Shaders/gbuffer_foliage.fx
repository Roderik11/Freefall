#include "common.fx"
// @RenderState(RenderTargets=4, CullMode=None)

// Foliage GBuffer shader — based on gbuffer.fx with:
// - Two-sided rendering (CullMode=None)
// - Backface normal flip via SV_IsFrontFace
// - Subsurface scattering approximation
// - Vegetation flag (data.a = 0.5)
// - Wind flutter animation
// - Distance-based alpha threshold

// Unified Push Constant Layout (Slots 2-11) — MUST match gbuffer.fx exactly
#define DescriptorBufIdx GET_INDEX(2)
#define Reserved0Idx GET_INDEX(3)
#define SortedIndicesIdx GET_INDEX(4)
#define BoneWeightsIdx GET_INDEX(5)
#define BonesIdx GET_INDEX(6)
#define IndexBufferIdx GET_INDEX(7)
#define BaseIndex GET_INDEX(8)
#define PosBufferIdx GET_INDEX(9)
#define NormBufferIdx GET_INDEX(10)
#define UVBufferIdx GET_INDEX(11)
#define NumBones GET_INDEX(12)
#define InstanceBaseOffset GET_INDEX(13)
#define DebugMode GET_INDEX(16)

struct InstanceDescriptor
{
    uint TransformSlot;
    uint MaterialId;
    uint CustomDataIdx;
};

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 WorldPos : TEXCOORD1;
    nointerpolation uint MaterialID : TEXCOORD2;
    float Depth : TEXCOORD3;
};

// Wind animation helpers
float3 WindDisplacement(float3 worldPos, float strength)
{
    float phase = Time + worldPos.x * 0.15 + worldPos.z * 0.12;
    float sway = sin(phase * 1.2) * 0.3 + sin(phase * 2.7) * 0.15;
    
    // High-frequency flutter for leaves
    float flutter = sin(phase * 5.3 + worldPos.y * 3.0) * 0.08
                  + sin(phase * 7.1 + worldPos.z * 4.0) * 0.05;
    
    return float3(sway + flutter, flutter * 0.3, sway * 0.5 + flutter) * strength;
}

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[UVBufferIdx];

    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;

    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    float3 pos = positions[vertexID];
    float3 norm = normals[vertexID];
    float2 uv = uvs[vertexID];

    float4 worldPos = mul(float4(pos, 1.0f), World);
    
    // Wind flutter — scale by height above ground (local Y)
    float windWeight = saturate(pos.y * 0.3); // higher vertices move more
    worldPos.xyz += WindDisplacement(worldPos.xyz, windWeight * 0.15);
    
    output.WorldPos = worldPos;
    output.Position = mul(mul(worldPos, View), Projection);
    
    // Canopy shape: mostly upward with subtle outward dome from tree origin
    float3 treeOrigin = float3(World._41, World._42, World._43);
    float3 outward = normalize(worldPos.xyz - treeOrigin);
    output.Normal = normalize(lerp(float3(0, 1, 0), outward, 0.25));
    output.TexCoord = uv;
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = desc.MaterialId;
    output.Depth = output.Position.w;
    return output;
}

struct PSOutput
{
    float4 Albedo : SV_Target0;
    float4 Normal : SV_Target1;
    float4 Data : SV_Target2;
    float  Depth : SV_Target3;
};

SamplerState Sampler : register(s0);

PSOutput PS(VSOutput input, bool isFrontFace : SV_IsFrontFace)
{
    PSOutput output;

    MaterialData mat = GET_MATERIAL(input.MaterialID);
    Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];

    float4 color = albedoTex.Sample(Sampler, input.TexCoord);
    
    // Distance-based alpha threshold — preserve coverage at distance
    float dist = length(input.WorldPos.xyz);
    float alphaThreshold = lerp(0.5, 0.15, saturate((dist - 30.0) / 50.0));
    clip(color.a - alphaThreshold);

    // Uniform upward normal for foliage — no backface flip needed
    // Both sides of each leaf card should receive equal lighting
    float3 N = float3(0, 1, 0);

    // Subsurface scattering hint: store translucency in albedo alpha
    // The directional light shader can use this for wrap lighting
    float translucency = 0.0;
    if (mat.AOIdx != 0)
    {
        // If a translucency map (_t.dds) is bound, use it
        Texture2D tTex = ResourceDescriptorHeap[mat.AOIdx];
        translucency = tTex.Sample(Sampler, input.TexCoord).r;
    }

    output.Albedo = float4(color.rgb, 1.0);
    output.Normal = float4(N, translucency); // store translucency in normal.w
    output.Data = float4(0.7, 0.0, 1.0, 0.5); // roughness=0.7, metal=0, ao=1, flag=vegetation
    output.Depth = input.Depth;
    return output;
}

// Shadow pass
struct ShadowVSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    nointerpolation uint MaterialID : TEXCOORD1;
};

ShadowVSOutput VS_Shadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    ShadowVSOutput output;

    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[UVBufferIdx];

    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    float3 pos = positions[vertexID];
    float2 uv = uvs[vertexID];

    float4 worldPos = mul(float4(pos, 1.0f), World);
    
    // Match wind displacement from VS
    float windWeight = saturate(pos.y * 0.3);
    worldPos.xyz += WindDisplacement(worldPos.xyz, windWeight * 0.15);
    
    output.Position = mul(mul(worldPos, View), Projection);
    output.TexCoord = uv;
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = desc.MaterialId;
    return output;
}

void PS_Shadow(ShadowVSOutput input)
{
    MaterialData mat = GET_MATERIAL(input.MaterialID);
    Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
    float alpha = albedoTex.Sample(Sampler, input.TexCoord).a;
    clip(alpha - 0.25f);
}

technique11 GBuffer
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
