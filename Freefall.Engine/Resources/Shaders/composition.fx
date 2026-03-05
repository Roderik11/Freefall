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

    // Pixel-exact Load — 1:1 fullscreen pass, no filtering needed
    int3 px = int3(input.Position.xy, 0);
    float4 albedo = AlbedoTex.Load(px);
    float4 light = LightTex.Load(px);
    float4 data = DataTex.Load(px);
    float3 normal = NormalTex.Load(px).xyz;
    
    // Hemisphere ambient: sky-facing surfaces get a cooler/brighter tint,
    // ground-facing get warmer/darker. Gives shape to fully shadowed objects.
    float ao = data.b;
    float3 skyColor    = float3(0.25, 0.28, 0.35);  // cool sky bounce
    float3 groundColor = float3(0.12, 0.11, 0.10);  // warm ground bounce
    float hemi = normal.y * 0.5 + 0.5;               // remap [-1,1] → [0,1]
    float3 ambient = lerp(groundColor, skyColor, hemi) * ao * AmbientScale;
    
    // data.a flags: 0=unlit (skybox), >0=lit (0.5=vegetation, 1.0=standard PBR)
    float isLit = step(0.1, data.a);
    float3 finalColor = lerp(albedo.rgb, ambient * albedo.rgb + light.rgb, isLit);
    
    // ACES Filmic Tone Mapping (Narkowicz 2015 approximation)
    // Compresses highlights, deepens shadows, adds warmth
    float3 x = finalColor;
    finalColor = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    finalColor = saturate(finalColor);
    
    // Vibrance boost: selectively saturate under-saturated pixels
    // (already-vivid colors stay put, muted terrain/rock gets pushed)
    float luma = dot(finalColor, float3(0.2126, 0.7152, 0.0722));
    float currentSat = max(finalColor.r, max(finalColor.g, finalColor.b)) - min(finalColor.r, min(finalColor.g, finalColor.b));
    float vibranceAmount = (1.0 - currentSat) * 0.15; // subtle vibrance on desaturated pixels
    finalColor = lerp(float3(luma, luma, luma), finalColor, 1.0 + vibranceAmount);
    finalColor = saturate(finalColor);
    
    // Final Gamma Correction (Linear -> sRGB)
    finalColor = pow(abs(finalColor), 1.0f / 2.2f);
    
    // Dithering — break up color banding in smooth gradients (sky)
    // Triangular-distribution noise: ±0.5/255 in sRGB space
    float2 seed = input.Position.xy; // SV_POSITION = pixel coords at any resolution
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
