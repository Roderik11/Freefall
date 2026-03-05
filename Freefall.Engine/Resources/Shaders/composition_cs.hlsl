#include "common.fx"

// Composition Compute Shader — replaces composition.fx fullscreen quad
// [numthreads(8, 8, 1)] = 64 threads per tile, standard for 2D image processing

// Push constants: input SRVs + output UAV + screen dimensions
#define AlbedoTexIdx    GET_INDEX(0)
#define LightTexIdx     GET_INDEX(1)
#define DataTexIdx      GET_INDEX(2)
#define NormalTexIdx    GET_INDEX(3)
#define OutputUAVIdx    GET_INDEX(4)
#define ScreenWidth     asuint(GET_INDEX(5))
#define ScreenHeight    asuint(GET_INDEX(6))

[numthreads(8, 8, 1)]
void CSCompose(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 px = dispatchThreadId.xy;
    
    // Bounds check — dispatch may overshoot texture dimensions
    if (px.x >= ScreenWidth || px.y >= ScreenHeight)
        return;
    
    Texture2D AlbedoTex = ResourceDescriptorHeap[AlbedoTexIdx];
    Texture2D LightTex = ResourceDescriptorHeap[LightTexIdx];
    Texture2D DataTex = ResourceDescriptorHeap[DataTexIdx];
    Texture2D NormalTex = ResourceDescriptorHeap[NormalTexIdx];
    RWTexture2D<float4> Output = ResourceDescriptorHeap[OutputUAVIdx];
    
    int3 coord = int3(px, 0);
    float4 albedo = AlbedoTex.Load(coord);
    float4 light = LightTex.Load(coord);
    float4 data = DataTex.Load(coord);
    float3 normal = NormalTex.Load(coord).xyz;
    
    // Hemisphere ambient: sky-facing surfaces get a cooler/brighter tint,
    // ground-facing get warmer/darker. Gives shape to fully shadowed objects.
    float ao = data.b;
    float3 skyColor    = float3(0.25, 0.28, 0.35);  // cool sky bounce
    float3 groundColor = float3(0.12, 0.11, 0.10);  // warm ground bounce
    float hemi = normal.y * 0.5 + 0.5;               // remap [-1,1] -> [0,1]
    float3 ambient = lerp(groundColor, skyColor, hemi) * ao * AmbientScale;
    
    // data.a flags: 0=unlit (skybox), >0=lit (0.5=vegetation, 1.0=standard PBR)
    float isLit = step(0.1, data.a);
    float3 finalColor = lerp(albedo.rgb, ambient * albedo.rgb + light.rgb, isLit);
    
    // ACES Filmic Tone Mapping (Narkowicz 2015 approximation)
    float3 x = finalColor;
    finalColor = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    finalColor = saturate(finalColor);
    
    // Vibrance boost: selectively saturate under-saturated pixels
    float luma = dot(finalColor, float3(0.2126, 0.7152, 0.0722));
    float currentSat = max(finalColor.r, max(finalColor.g, finalColor.b)) - min(finalColor.r, min(finalColor.g, finalColor.b));
    float vibranceAmount = (1.0 - currentSat) * 0.15;
    finalColor = lerp(float3(luma, luma, luma), finalColor, 1.0 + vibranceAmount);
    finalColor = saturate(finalColor);
    
    // Final Gamma Correction (Linear -> sRGB)
    finalColor = pow(abs(finalColor), 1.0f / 2.2f);
    
    // Dithering — break up color banding in smooth gradients (sky)
    float2 seed = float2(px);
    float noise1 = frac(sin(dot(seed, float2(12.9898, 78.233))) * 43758.5453);
    float noise2 = frac(sin(dot(seed, float2(39.3468, 11.135))) * 23564.2365);
    float dither = (noise1 + noise2 - 1.0) / 255.0;
    finalColor += dither;
    
    Output[px] = float4(finalColor, 1.0f);
}
