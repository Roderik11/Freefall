#include "common.fx"
// @RenderState(RenderTargets=4, DepthTest=false, DepthWrite=false)

struct VSOutput
{
    float4 Position : SV_POSITION;
};

VSOutput VS(uint vertexID : SV_VertexID)
{
    VSOutput output;
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[GET_INDEX(7)];
    output.Position = float4(positions[vertexID], 1.0f);
    return output;
}

struct PSOutput
{
    float4 Albedo : SV_Target0;
    float4 Normal : SV_Target1;
    float4 Data : SV_Target2;
    float4 Depth : SV_Target3;
};

PSOutput PS(VSOutput input)
{
    PSOutput output;
    output.Albedo = float4(0,0,0,0);
    output.Normal = float4(0,0,0,0);
    output.Data = float4(0,0,0,0);
    output.Depth = float4(0,0,0,0);
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
