// Composition Compute Shader — replaces composition.fx fullscreen quad
// [numthreads(8, 8, 1)] = 64 threads per tile, standard for 2D image processing

#pragma kernel CSCompose

// Named push constants for ComputeShader reflection
cbuffer PushConstants : register(b3)
{
    uint AlbedoTexIdx;
    uint LightTexIdx;
    uint DataTexIdx;
    uint NormalTexIdx;
    uint OutputUAVIdx;
    uint ScreenWidthIdx;
    uint ScreenHeightIdx;
    uint DepthGBufIdx;
};

cbuffer Params : register(b4)
{
    uint SSDMTexIdx;
    uint3 _compPad;
};

#include "common.fx"
#include "sky_common.fx"

[numthreads(8, 8, 1)]
void CSCompose(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 px = dispatchThreadId.xy;
    
    // Bounds check — dispatch may overshoot texture dimensions
    if (px.x >= ScreenWidthIdx || px.y >= ScreenHeightIdx)
        return;
    
    Texture2D AlbedoTex = ResourceDescriptorHeap[AlbedoTexIdx];
    Texture2D LightTex = ResourceDescriptorHeap[LightTexIdx];
    Texture2D DataTex = ResourceDescriptorHeap[DataTexIdx];
    Texture2D NormalTex = ResourceDescriptorHeap[NormalTexIdx];
    Texture2D DepthGBuf = ResourceDescriptorHeap[DepthGBufIdx];
    RWTexture2D<float4> Output = ResourceDescriptorHeap[OutputUAVIdx];
    
    int3 coord = int3(px, 0);
    int3 displaced_coord = coord;

    // Apply SSDM displacement to GBuffer reads
    if (SSDMTexIdx != 0)
    {
        Texture2D<float2> SSDMTex = ResourceDescriptorHeap[SSDMTexIdx];
        float2 ssdmVal = SSDMTex.Load(coord);
        // (0,0) sentinel = no displacement (avoids half-precision UV jitter)
        if (any(ssdmVal != 0))
        {
            float2 myUV = (float2(px) + 0.5) / float2(ScreenWidthIdx, ScreenHeightIdx);
            float2 offset_px = (ssdmVal - myUV) * float2(ScreenWidthIdx, ScreenHeightIdx);
            displaced_coord = int3(clamp(int2(px) + int2(round(offset_px)), int2(0,0), int2(ScreenWidthIdx-1, ScreenHeightIdx-1)), 0);
        }
    }

    float4 albedo = AlbedoTex.Load(displaced_coord);
    float4 light = LightTex.Load(displaced_coord);
    float4 data = DataTex.Load(displaced_coord);
    float3 normal = NormalTex.Load(displaced_coord).xyz;
    
    // Hemisphere ambient: sky-facing surfaces get a cooler/brighter tint,
    // ground-facing get warmer/darker. Gives shape to fully shadowed objects.
    float ao = data.b;
    float3 skyColor    = GetSkyColor(float3(0, 1, 0), FogSunDirection) * 0.45;  // sky bounce from atmosphere
    float3 groundColor = float3(0.12, 0.11, 0.10);  // warm ground bounce (terrain-dependent)
    float hemi = normal.y * 0.5 + 0.5;               // remap [-1,1] -> [0,1]
    float3 ambient = lerp(groundColor, skyColor, hemi) * ao * AmbientScale;
    
    // data.a flags: 0=unlit (skybox), >0=lit (0.5=vegetation, 1.0=standard PBR)
    float isLit = step(0.1, data.a);
    float emissiveMask = albedo.a;
    float3 pbrLit = ambient * albedo.rgb + light.rgb;
    float3 emissiveGlow = albedo.rgb * 2.0; // self-illumination (HDR boost)
    float3 finalColor = lerp(albedo.rgb, lerp(pbrLit, emissiveGlow, emissiveMask), isLit);
    
    // Aerial perspective — sky-derived per-pixel fog color
    if (FogEnabled > 0 && isLit > 0)
    {
        float linearDepth = DepthGBuf.Load(coord).r;

        // View direction from screen position (for per-pixel sky color)
        float2 uv = (float2(px) + 0.5) / float2(ScreenWidthIdx, ScreenHeightIdx);
        float2 ndc = float2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));
        float4 farClip = mul(float4(ndc, 0, 1), CameraInverse); // reverse-Z: 0 = far
        float3 viewDir = normalize(farClip.xyz / farClip.w);

        // Exponential-squared extinction using linear depth as distance
        float fogFactor = 1.0 - exp(-pow(linearDepth * FogDensity, 2.0));
        fogFactor = saturate(fogFactor);

        // Inscatter: sky color along view ray (warm toward sun, cool away)
        float3 inscatter = GetSkyColor(float3(viewDir.x, max(viewDir.y, 0.01), viewDir.z), FogSunDirection);

        finalColor = lerp(finalColor, inscatter, fogFactor);
    }
    
    // Output HDR linear — tonemapping deferred to finalize pass
    Output[px] = float4(finalColor, 1.0f);
}
