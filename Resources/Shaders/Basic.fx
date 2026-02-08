#include "common.fx"

struct VSOutput
{
    float4 Pos : SV_Position;
    float3 Normal : NORMAL;
    float2 UV : TEXCOORD0;
};

// Bindless indices from PushConstants (Slot 0)
#define AlbedoIdx PushConstants.indices[0]
#define NormalIdx PushConstants.indices[1]
#define PosBufferIdx PushConstants.indices[7]
#define NormBufferIdx PushConstants.indices[8]
#define UVBufferIdx PushConstants.indices[9]

row_major matrix World : register(b1); // ObjectConstants bound to Slot 2

VSOutput VS(uint vertexID : SV_VertexID)
{
    VSOutput output;

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[UVBufferIdx];

    float3 pos = positions[vertexID];
    float3 norm = normals[vertexID];
    float2 uv = uvs[vertexID];

    float4 worldPos = mul(float4(pos, 1.0f), World);
    output.Pos = mul(worldPos, View); // Note: common.fx has View/Projection
    output.Pos = mul(output.Pos, Projection);
    
    output.Normal = mul(norm, (float3x3)World);
    output.UV = uv;
    return output;
}

SamplerState LinearSampler : register(s0);

float4 PS(VSOutput input) : SV_Target
{
    float3 lightDir = normalize(float3(1.0, 1.0, -1.0));
    float NdotL = saturate(dot(normalize(input.Normal), lightDir));
    
    Texture2D AlbedoTex = ResourceDescriptorHeap[AlbedoIdx];
    float3 albedo = AlbedoTex.Sample(LinearSampler, input.UV).rgb;
    float3 diffuse = albedo * NdotL + albedo * 0.3;
    return float4(diffuse, 1.0);
}

technique11 Standard
{
    pass Opaque
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
