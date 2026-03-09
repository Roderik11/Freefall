cbuffer PushConstants : register(b3)
{
    uint _reserved0;
    uint _reserved1;
    uint _reserved2;
    uint _reserved3;
    uint _reserved4;
    uint _reserved5;
    uint _reserved6;
    uint IndexBufferIdx;        // 7
};

#include "common.fx"
// @RenderState(RenderTargets=5, DepthTest=false, DepthWrite=false)

struct VSOutput
{
    float4 Position : SV_POSITION;
};

VSOutput VS(uint vertexID : SV_VertexID)
{
    VSOutput output;
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[IndexBufferIdx];
    output.Position = float4(positions[vertexID], 1.0f);
    return output;
}

struct PSOutput
{
    float4 Albedo : SV_Target0;
    float4 Normal : SV_Target1;
    float4 Data : SV_Target2;
    float4 Depth : SV_Target3;
    float  EntityId : SV_Target4;
};

PSOutput PS(VSOutput input)
{
    PSOutput output;
    output.Albedo = float4(0,0,0,0);
    output.Normal = float4(0,0,0,0);
    output.Data = float4(0,0,0,0);
    output.Depth = float4(0,0,0,0);
    output.EntityId = 0.0;
    return output;
}

technique11 GBuffer
{
    pass PostProcess
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
