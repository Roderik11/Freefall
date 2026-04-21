// Minimal forward lit shader for inspector mesh preview.
// Renders outside the deferred pipeline — no InstanceDescriptor, no SortedIndices.
// Reads mesh buffers via push constants (PosBufferIdx, NormBufferIdx, etc.)

cbuffer PushConstants : register(b3)
{
    uint PosBufferIdx;        // 0
    uint NormBufferIdx;       // 1
    uint UVBufferIdx;         // 2
    uint IndexBufferIdx;      // 3
    uint BaseIndex;           // 4
    uint _pad5;               // 5
    uint _pad6;               // 6
    uint _pad7;               // 7
};

#include "common.fx"

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 Normal   : NORMAL;
    float2 TexCoord : TEXCOORD0;
};

cbuffer PreviewConstants : register(b1)
{
    row_major float4x4 World;
    float3 LightDir;
    float  _padLight;
    float3 LightColor;
    float  _padLight2;
    float3 MaterialColor;
    float  _padMat;
};

VSOutput VS(uint vertexID : SV_VertexID)
{
    VSOutput output;

    StructuredBuffer<uint>   indices   = ResourceDescriptorHeap[IndexBufferIdx];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    StructuredBuffer<float3> normals   = ResourceDescriptorHeap[NormBufferIdx];
    StructuredBuffer<float2> uvs       = ResourceDescriptorHeap[UVBufferIdx];

    uint idx = indices[vertexID + BaseIndex];
    float3 pos  = positions[idx];
    float3 norm = normals[idx];
    float2 uv   = uvs[idx];

    float4 worldPos = mul(float4(pos, 1.0f), World);
    output.Position = mul(mul(worldPos, View), Projection);
    output.Normal   = normalize(mul(norm, (float3x3)World));
    output.TexCoord = uv;

    return output;
}

float4 PS(VSOutput input) : SV_Target0
{
    float3 N = normalize(input.Normal);
    float3 L = normalize(-LightDir);

    // Simple hemisphere lighting: warm top + cool bottom
    float hemisphereBlend = N.y * 0.5 + 0.5;
    float3 ambient = lerp(float3(0.08, 0.08, 0.12), float3(0.15, 0.18, 0.22), hemisphereBlend);

    // Diffuse N.L
    float NdotL = max(dot(N, L), 0.0);
    float3 diffuse = LightColor * NdotL;

    // Rim light for depth perception
    float rim = 1.0 - saturate(dot(N, float3(0, 0, -1)));
    rim = pow(rim, 3.0) * 0.15;

    float3 color = MaterialColor * (ambient + diffuse) + rim;
    return float4(color, 1.0);
}

technique11 GBuffer
{
    pass Forward
    {
        SetRenderState(RenderTargets=1, DepthWrite=true, DepthFunc=LessEqual);
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
