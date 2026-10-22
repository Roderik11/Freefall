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
    uint AOTexIdx;
};

cbuffer Params : register(b4)
{
    uint SSDMTexIdx;
    float AODirectStrength;   // how much GTAO also darkens direct light (0 = ambient only)
    uint AmbientSpecular;     // 1: the sky's light reflected by every lit surface (EngineSettings.AmbientSpecular)
    uint SSRTexIdx;           // the screen-space reflections at half size (ssr_cs.hlsl: rgb × confidence, confidence); 0 = none
};

#include "common.fx"
#include "sky_common.fx"

// Multi-bounce AO approximation (Jimenez et al. 2016): bright surfaces
// re-light their own crevices, so occlusion is weaker on high albedo.
float3 MultiBounceAO(float visibility, float3 albedo)
{
    float3 a = 2.0404 * albedo - 0.3324;
    float3 b = -4.7951 * albedo + 0.6417;
    float3 c = 2.7552 * albedo + 0.6903;
    return max(visibility, ((visibility * a + b) * visibility + c) * visibility);
}

// The split sum's environment BRDF, Karis' fit ("Physically Based Shading on Mobile", 2014): the share of the light
// reflected from round the mirror direction, by the reflectance at normal incidence F0, the roughness and N·V
float3 EnvBRDFApprox(float3 f0, float roughness, float NoV)
{
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    float2 ab = float2(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

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
    
    // The sky's ambient light for this normal (the sky probe's SH, sky_common.fx SkyAmbient): the sky, the clouds and the
    // lit ground, brighter toward where the sky is bright. Gives shape to fully shadowed objects. × how much of the sky
    // the surface sees past the terrain and the meshes (sky_visibility_cs, in the light buffer's alpha): a cave or a room
    // gets little
    float ao = data.b;
    float3 ambient = SkyAmbient(normal) * ao * saturate(light.a);
    float occlusion = ao;

    // Screen-space AO (GTAO) — fully occludes ambient; direct light only by AODirectStrength
    // (not physically based, but keeps creases readable in sunlit areas)
    if (AOTexIdx != 0)
    {
        Texture2D<float> AOTex = ResourceDescriptorHeap[AOTexIdx];
        float gtao = AOTex.Load(displaced_coord);
        ambient *= MultiBounceAO(gtao, albedo.rgb);
        light.rgb *= lerp(1.0, gtao, AODirectStrength);
        occlusion *= gtao;
    }

    // data.a flags: 0=unlit (skybox), >0=lit (0.5=vegetation, 1.0=standard PBR)
    float isLit = step(0.1, data.a);
    float emissiveMask = albedo.a;
    float3 diffuseAlbedo = albedo.rgb;

    // The light reflected (WEATHER_PLAN stage 4: wet stone and puddles must show the sky and the town): the screen-space
    // reflections where their ray found something (ssr_cs.hlsl), else the sky probe in the mirror direction blurred by the
    // roughness × the sky the surface sees (a room, a cave reflects no sky); × the environment BRDF and the AO. Rough
    // surfaces get a faint sheen at grazing angles, metals their tinted reflection instead of a diffuse ambient. Before
    // 2026-10-22 opaque surfaces had no ambient specular at all (only glass and water): a puddle at roughness 0.04 drew as
    // a darker patch of cobbles.
    float3 specular = 0;
    if (AmbientSpecular != 0 && isLit > 0 && occlusion > 0.0)
    {
        float2 uv = (float2(px) + 0.5) / float2(ScreenWidthIdx, ScreenHeightIdx);
        float3 toPixel = normalize(posFromDepth(uv, 1.0, CameraInverse).xyz);   // reverse Z: depth 1 = the near plane
        float3 N = normalize(normal);
        float NoV = saturate(dot(N, -toPixel));
        float roughness = saturate(data.r), metal = saturate(data.g);
        float3 f0 = lerp(0.04, albedo.rgb, metal);
        float3 environment = SkyReflection(reflect(toPixel, N), roughness) * saturate(light.a);
        if (SSRTexIdx != 0)
        {
            // Blurred by the roughness, twice as far down the screen as across (a wet street draws a light as a streak
            // toward the viewer): one sharp ray a pixel gave glossy surfaces a reflection made of single pixels, hits
            // beside misses, and soaked cobbles turned to salt and pepper (user, 2026-10-23). The texture is premultiplied
            // by its confidence, so the taps average hits and misses into a soft one; mirrors (puddles) stay sharp
            Texture2D ssrTex = ResourceDescriptorHeap[SSRTexIdx];
            float4 ssr = ssrTex.SampleLevel(AtmosphereLinearClamp, uv, 0);
            float spread = saturate((roughness - 0.05) / 0.3) * 6.0;           // half-size texels
            if (spread > 0.25)
            {
                float2 texel = 2.0 / float2(ScreenWidthIdx, ScreenHeightIdx);
                [unroll]
                for (uint i = 0; i < 8; i++)
                {
                    float a = i * 2.39996323;
                    float2 o = float2(cos(a), 2.0 * sin(a)) * (spread * sqrt((i + 0.5) / 8.0));
                    ssr += ssrTex.SampleLevel(AtmosphereLinearClamp, uv + o * texel, 0);
                }
                ssr /= 9.0;
            }
            environment = environment * (1.0 - ssr.a) + ssr.rgb;
        }
        specular = environment * EnvBRDFApprox(f0, roughness, NoV) * occlusion;
        diffuseAlbedo *= 1.0 - metal;
    }
    float3 pbrLit = ambient * diffuseAlbedo + specular + light.rgb;
    // Self-illumination: where it outshines the lit surface the G-buffer holds its luminance, pre-exposed (gbuffer.fx)
    float3 emissiveGlow = albedo.rgb;
    // Unlit pixels (the sky dome) arrive pre-exposed already
    float3 finalColor = lerp(albedo.rgb, lerp(pbrLit, emissiveGlow, emissiveMask), isLit);

    // The fog (sky_common.fx AtmosphereFog); sky pixels already hold all of the air, the near fog volume takes the first
    // stretch of it over
    if (isLit > 0)
        finalColor = ApplyAerialPerspective(finalColor, float2(px) + 0.5, DepthGBuf.Load(coord).r, saturate(light.a));
    else
        finalColor = AtmosphereFogOverSky(finalColor, float2(px) + 0.5);
    
    // Output HDR linear — tonemapping deferred to finalize pass
    Output[px] = float4(finalColor, 1.0f);
}
