cbuffer PushConstants : register(b3)
{
    uint _reserved0;
    uint _reserved1;
    uint DescriptorBufIdx;      // 2
    uint _reserved3;            // 3
    uint SortedIndicesIdx;      // 4
    uint BoneWeightsIdx;        // 5
    uint BonesIdx;              // 6
    uint IndexBufferIdx;        // 7
    uint BaseIndex;             // 8
    uint PosBufferIdx;          // 9
    uint NormBufferIdx;         // 10
    uint UVBufferIdx;           // 11
    uint NumBones;              // 12
    uint InstanceBaseOffset;    // 13
    uint MaterialsIdx;          // 14
    uint GlobalTransformBufferIdx; // 15
    uint DebugMode;             // 16
    uint _reserved17;
    uint _reserved18;
    uint _reserved19;
    uint ExpansionBufferIdx;    // 20
    uint CascadeBufferSRVIdx;   // 21
};

#include "common.fx"
// @RenderState(RenderTargets=5, CullMode=None)

// Foliage GBuffer shader — based on gbuffer.fx with:
// - Two-sided rendering (CullMode=None)
// - Backface normal flip via SV_IsFrontFace
// - Subsurface scattering approximation
// - Vegetation flag (data.a = 0.5)
// - Wind flutter animation
// - Distance-based alpha threshold

inline MaterialData GetMaterial(uint id)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    return materials[id];
}
#define GET_MATERIAL(id) GetMaterial(id)



struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 WorldPos : TEXCOORD1;
    nointerpolation uint MaterialID : TEXCOORD2;
    float Depth : TEXCOORD3;
    nointerpolation uint TransformSlot : TEXCOORD4;
    nointerpolation uint MeshPartIdx : TEXCOORD5;
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
    output.TransformSlot = desc.TransformSlot;
    output.MeshPartIdx = desc.MeshPartIdx;
    return output;
}

struct PSOutput
{
    float4 Albedo : SV_Target0;
    float4 Normal : SV_Target1;
    float4 Data : SV_Target2;
    float  Depth : SV_Target3;
    uint   EntityId : SV_Target4;
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

    // Use interpolated dome normal from VS — provides directional response
    // while still being smoothed enough to avoid harsh self-shadowing
    // No backface flip: both sides of each leaf card should receive equal lighting
    float3 N = normalize(input.Normal);

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
    output.EntityId = (input.TransformSlot << 8u) | (input.MeshPartIdx & 0xFFu);
    return output;
}


// Shadow pass
struct ShadowVSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    nointerpolation uint MaterialID : TEXCOORD1;
    nointerpolation uint RTIndex : SV_RenderTargetArrayIndex;
};

ShadowVSOutput VS_Shadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    ShadowVSOutput output;
    output.Position = float4(0, 0, 0, 1);
    output.TexCoord = float2(0, 0);
    output.MaterialID = 0;
    output.RTIndex = 0;

    // Read expansion entry: bits 30-31 = cascadeIdx, bits 0-29 = instance index
    StructuredBuffer<uint> expansion = ResourceDescriptorHeap[ExpansionBufferIdx];
    uint entry = expansion[InstanceBaseOffset + instanceID];
    uint cascadeIdx = entry >> 30;
    uint idx = entry & 0x3FFFFFFFu;
    output.RTIndex = cascadeIdx;

    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[UVBufferIdx];

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    float3 pos = positions[vertexID];
    float4 worldPos = mul(float4(pos, 1.0f), World);

    // Match wind displacement from VS
    float windWeight = saturate(pos.y * 0.3);
    worldPos.xyz += WindDisplacement(worldPos.xyz, windWeight * 0.15);

    StructuredBuffer<CascadeData> cascadeData = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    output.Position = mul(worldPos, cascadeData[cascadeIdx].VP);
    output.TexCoord = uvs[vertexID];
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
