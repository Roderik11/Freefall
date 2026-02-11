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
#define NormalTexIdx GET_INDEX(3)          // G-Buffer normals (for hemisphere ambient)

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

    Texture2D NormalTex = ResourceDescriptorHeap[NormalTexIdx];

    float4 albedo = AlbedoTex.Sample(Sampler, input.TexCoord);
    float4 light = LightTex.Sample(Sampler, input.TexCoord);
    float4 data = DataTex.Sample(Sampler, input.TexCoord);
    float3 normal = NormalTex.Sample(Sampler, input.TexCoord).xyz;
    
    // Hemisphere ambient: sky-facing surfaces get a cooler/brighter tint,
    // ground-facing get warmer/darker. Gives shape to fully shadowed objects.
    float ao = data.b;
    float3 skyColor    = float3(0.25, 0.28, 0.35);  // cool sky bounce
    float3 groundColor = float3(0.12, 0.11, 0.10);  // warm ground bounce
    float hemi = normal.y * 0.5 + 0.5;               // remap [-1,1] → [0,1]
    float3 ambient = lerp(groundColor, skyColor, hemi) * ao * AmbientScale;
    
    // Light buffer contains pre-lit PBR color (albedo already baked into diffuse term)
    // Additive model: ambient*albedo + light; lerp with data.w for unlit bypass (skybox)
    float3 finalColor = lerp(albedo.rgb, ambient * albedo.rgb + light.rgb, data.w);
    
    // Final Gamma Correction (Linear -> sRGB)
    finalColor = pow(abs(finalColor), 1.0f / 2.2f);
    
    // Dithering — break up color banding in smooth gradients (sky)
    // Triangular-distribution noise: ±0.5/255 in sRGB space
    float2 seed = input.TexCoord * float2(1920.0, 1080.0); // pixel coords
    float noise1 = frac(sin(dot(seed, float2(12.9898, 78.233))) * 43758.5453);
    float noise2 = frac(sin(dot(seed, float2(39.3468, 11.135))) * 23564.2365);
    float dither = (noise1 + noise2 - 1.0) / 255.0;
    finalColor += dither;
    
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
