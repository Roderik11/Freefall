#include "common.fx"
// @RenderState(DepthTest=false, DepthWrite=false)

struct VSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

// Composition Pass Push Constant Layout (Slots 0-1)
// Fullscreen post-process pass combining G-Buffer and light buffer
#define AlbedoTexIdx GET_INDEX(0)          // G-Buffer albedo texture
#define LightTexIdx GET_INDEX(1)           // Accumulated light buffer
#define DataTexIdx GET_INDEX(2)            // G-Buffer data texture

VSOutput VS(uint vertexID : SV_VertexID)
{
    VSOutput output;
    
    float2 pos[4] = {
        float2(-1, 1),
        float2(1, 1),
        float2(-1, -1),
        float2(1, -1)
    };
    float2 uv[4] = {
        float2(0, 0),
        float2(1, 0),
        float2(0, 1),
        float2(1, 1)
    };

    output.Position = float4(pos[vertexID], 0.0f, 1.0f);
    output.TexCoord = uv[vertexID];
    return output;
}

SamplerState Sampler : register(s0);

float4 PS(VSOutput input) : SV_Target
{
    Texture2D AlbedoTex = ResourceDescriptorHeap[AlbedoTexIdx];
    Texture2D LightTex = ResourceDescriptorHeap[LightTexIdx];
    Texture2D DataTex = ResourceDescriptorHeap[DataTexIdx];

    float4 albedo = AlbedoTex.Sample(Sampler, input.TexCoord);
    float4 light = LightTex.Sample(Sampler, input.TexCoord);
    float4 data = DataTex.Sample(Sampler, input.TexCoord);
    
    // Ambient should be subtle but enough to see dark areas
    float3 ambient = float3(0.12, 0.13, 0.14); 
    
float3 finalColor = albedo.rgb * lerp(1.0, light.rgb + ambient, data.w);
    
    // Final Gamma Correction (Linear -> sRGB)
    finalColor = pow(abs(finalColor), 1.0f / 2.2f);
    
    return float4(finalColor, 1.0f);
}

technique11 Standard
{
    pass PostProcess
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
