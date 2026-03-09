// Directional Light Compute Shader — replaces light_directional.fx fullscreen quad
// [numthreads(8, 8, 1)] = 64 threads per tile

#pragma kernel CSDirectionalLight

// Named push constants for ComputeShader reflection
#define PUSH_CONSTANTS_DEFINED
cbuffer PushConstants : register(b3)
{
    uint NormalTexIdx;
    uint DepthTexIdx;
    uint ShadowMapIdx;
    uint DepthGBufIdx;
    uint AlbedoTexIdx;
    uint DataTexIdx;
    uint LightingCascadeSRVIdx;
    uint OutputUAVIdx;
    uint ScreenWidthIdx;
    uint ScreenHeightIdx;
};

#include "common.fx"

// Light params from ObjectConstants (Slot 2, b1)
cbuffer ObjectConstants : register(b1)
{
    float3 LightColor;
    float LightIntensity;
    float3 LightDirection;
    float _pad0;
    
    row_major float4x4 LightSpaces[8];
    float4 Cascades[8];
    
    int CascadeCount;
    int DebugVisualizationMode;
    float2 _pad1;
};

// GPU-computed lighting cascade data
struct LightingCascadeData {
    row_major float4x4 VP;
    float4 Cascade;
};

float4x4 getLightSpace(int idx) {
    if (LightingCascadeSRVIdx > 0) {
        StructuredBuffer<LightingCascadeData> buf = ResourceDescriptorHeap[LightingCascadeSRVIdx];
        return buf[idx].VP;
    }
    return LightSpaces[idx];
}

float4 getCascade(int idx) {
    if (LightingCascadeSRVIdx > 0) {
        StructuredBuffer<LightingCascadeData> buf = ResourceDescriptorHeap[LightingCascadeSRVIdx];
        return buf[idx].Cascade;
    }
    return Cascades[idx];
}

SamplerState Sampler : register(s0);
SamplerState ShadowClampSampler : register(s2);
SamplerComparisonState ShadowSampler : register(s3);

// PBR helpers
float3 FresnelSchlick(float cosTheta, float3 F0, float roughness)
{
    float3 F_max = max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0);
    return F0 + (F_max - F0) * pow(1.0 - cosTheta, 5.0);
}

float4 ApplyNormalOffset(float4 worldPos, float3 normal, float3 lightDir, row_major float4x4 lightVP, float shadowMapRes)
{
    float texelWorldSize = 2.0 / (shadowMapRes * abs(lightVP._11));
    float cosTheta = saturate(dot(normal, -lightDir));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    float offsetScale = texelWorldSize * sinTheta * 0.8;
    offsetScale = min(offsetScale, 0.5);
    return worldPos + float4(normal * offsetScale, 0);
}

// Contact shadows (currently disabled but preserved for future use)
#define CONTACT_STEPS     24
#define CONTACT_MAX_DIST  0.75

float ContactShadow(float3 worldPos, float3 lightDir, float2 screenPos, float linearDepth)
{
    if (linearDepth > 100.0) return 1.0;
    float distanceFade = linearDepth > 80.0 ? (linearDepth - 80.0) / 20.0 : 0.0;
    
    Texture2D<float> linDepth = ResourceDescriptorHeap[DepthGBufIdx];
    
    float3 rayDir = normalize(-lightDir);
    float3 viewFwd = float3(View._13, View._23, View._33);
    
    float dither = InterleavedGradientNoise(screenPos);
    float stepSize = CONTACT_MAX_DIST / float(CONTACT_STEPS);
    
    for (int i = 0; i < CONTACT_STEPS; i++)
    {
        float t = (float(i) + 0.5 + dither) * stepSize;
        float3 rayPos = worldPos + rayDir * t;
        
        float4 clip = mul(float4(rayPos, 1.0), CameraRelativeVP);
        if (clip.w <= 0.0) continue;
        float2 uv = (clip.xy / clip.w) * float2(0.5, -0.5) + 0.5;
        
        if (any(uv < 0.0) || any(uv > 1.0)) break;
        
        float sceneLinear = linDepth.SampleLevel(Sampler, uv, 0).r;
        float rayLinear = dot(rayPos, viewFwd);
        float penetration = rayLinear - sceneLinear;
        
        if (penetration > 0.15 && penetration < 0.3)
            return distanceFade;
    }
    
    return 1.0;
}

[numthreads(8, 8, 1)]
void CSDirectionalLight(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 px = dispatchThreadId.xy;
    
    // Bounds check
    if (px.x >= ScreenWidthIdx || px.y >= ScreenHeightIdx)
        return;
    
    Texture2D NormalTex = ResourceDescriptorHeap[NormalTexIdx];
    Texture2D DepthTex = ResourceDescriptorHeap[DepthTexIdx];
    Texture2DArray ShadowMap = ResourceDescriptorHeap[ShadowMapIdx];
    
    int3 coord = int3(px, 0);
    float3 normal = NormalTex.Load(coord).xyz;
    float depth = DepthTex.Load(coord).r;
    
    RWTexture2D<float4> Output = ResourceDescriptorHeap[OutputUAVIdx];
    
    // Skip pixels at min depth (sky) — reverse depth: far=0
    if (depth <= 0.0f)
    {
        Output[px] = float4(0, 0, 0, 0);
        return;
    }
    
    // Reconstruct world position from depth — compute UV from pixel coords
    float2 texCoord = (float2(px) + 0.5) / float2(ScreenWidthIdx, ScreenHeightIdx);
    float4 worldPos = posFromDepth(texCoord, depth, CameraInverse);
    
    // Sample Data texture ONCE
    Texture2D DataTex = ResourceDescriptorHeap[DataTexIdx];
    float4 data = DataTex.Load(coord);
    bool isVegetation = (data.a > 0.3 && data.a < 0.7);
    
    // Compute NdotL
    float3 L = normalize(-LightDirection);
    float rawNdotL = dot(normal, L);
    float NdotL = max(rawNdotL, 0.0);
    
    if (NdotL <= 0.0 && !isVegetation && DebugVisualizationMode == 0)
    {
        Output[px] = float4(0, 0, 0, 0);
        return;
    }
    
    // View-space depth for cascade selection
    float viewDepth = dot(worldPos.xyz, float3(View._13, View._23, View._33));
    
    // Select cascade
    int cascadeIndex = CascadeCount - 1;
    for (int i = 0; i < CascadeCount; i++)
    {
        if (viewDepth < getCascade(i).y)
        {
            cascadeIndex = i;
            break;
        }
    }
    
    // Shadow calculation with cascade blending
    float shadowFactor = 1.0f;
    float4 lightSpacePos = float4(0,0,0,1);
    float2 shadowUV = float2(0,0);
    float sampledDepth = 0;
    
    if (viewDepth > getCascade(CascadeCount - 1).y)
        shadowFactor = 1.0f;
    else if (ShadowMapIdx > 0)
    {
        lightSpacePos = mul(worldPos, getLightSpace(cascadeIndex));
        lightSpacePos /= lightSpacePos.w;
        
        shadowUV = lightSpacePos.xy * 0.5f + 0.5f;
        shadowUV.y = 1.0f - shadowUV.y;
        
        if (shadowUV.x >= 0.0f && shadowUV.x <= 1.0f && shadowUV.y >= 0.0f && shadowUV.y <= 1.0f)
        {
            sampledDepth = ShadowMap.SampleLevel(ShadowClampSampler, float3(shadowUV, cascadeIndex), 0).r;
            float zScale = abs(getLightSpace(cascadeIndex)._33);
            shadowFactor = GetShadowFactor(ShadowMap, ShadowSampler, shadowUV, lightSpacePos.z, cascadeIndex, normal, LightDirection, zScale, float2(px));
            
            // Cascade blending
            if (cascadeIndex < CascadeCount - 1)
            {
                float cascadeRange = getCascade(cascadeIndex).y - getCascade(cascadeIndex).x;
                float blendZone = cascadeRange * 0.1f;
                float distToEdge = getCascade(cascadeIndex).y - viewDepth;
                
                if (distToEdge < blendZone)
                {
                    float blendFactor = distToEdge / blendZone;
                    
                    int nextCascade = cascadeIndex + 1;
                    float4 nextLSP = mul(worldPos, getLightSpace(nextCascade));
                    nextLSP /= nextLSP.w;
                    float2 nextUV = nextLSP.xy * 0.5f + 0.5f;
                    nextUV.y = 1.0f - nextUV.y;
                    
                    if (nextUV.x >= 0.0f && nextUV.x <= 1.0f && nextUV.y >= 0.0f && nextUV.y <= 1.0f)
                    {
                        float nextZScale = abs(getLightSpace(nextCascade)._33);
                        float nextShadow = GetShadowFactor(ShadowMap, ShadowSampler, nextUV, nextLSP.z, nextCascade, normal, LightDirection, nextZScale, float2(px));
                        shadowFactor = lerp(nextShadow, shadowFactor, blendFactor);
                    }
                }
            }
        }
    }

    // Contact shadow (disabled for now)
    // float contactShadow = ContactShadow(worldPos.xyz, LightDirection, float2(px), viewDepth);
    // shadowFactor = min(shadowFactor, contactShadow);
    float contactShadow = 1.0;

    // Debug visualization modes
    if (DebugVisualizationMode == 1)
    {
        float4 cascadeColors[4] = { float4(1,0,0,1), float4(0,1,0,1), float4(0,0,1,1), float4(1,1,0,1) };
        Output[px] = cascadeColors[cascadeIndex];
        return;
    }
    if (DebugVisualizationMode == 2)
    {
        Output[px] = float4(shadowFactor.xxx, 1);
        return;
    }
    if (DebugVisualizationMode == 3)
    {
        Output[px] = float4(contactShadow.xxx, 1);
        return;
    }
    if (DebugVisualizationMode == 4)
    {
        Texture2D<float> depthViz = ResourceDescriptorHeap[DepthGBufIdx];
        float d = depthViz.Load(coord).r;
        float normalized = saturate(d / 100.0);
        Output[px] = float4(normalized, normalized, normalized, 1);
        return;
    }
    
    // PBR lighting
    Texture2D AlbedoTex = ResourceDescriptorHeap[AlbedoTexIdx];
    float3 albedo = AlbedoTex.Load(coord).rgb;
    
    if (data.a < 0.01)
    {
        Output[px] = float4(0, 0, 0, 0);
        return;
    }
    
    float rough = max(data.r, 0.04);
    float metal = saturate(data.g);
    float ao    = saturate(data.b);
    
    // View and half vectors
    float3 V = normalize(-worldPos.xyz);
    float3 H = normalize(L + V);
    float NdotV = max(dot(normal, V), 0.001);
    float NdotH = max(dot(normal, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    
    float a  = rough * rough;
    float a2 = a * a;
    
    // GGX Normal Distribution
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    float D = a2 / max(3.14159 * denom * denom, 1e-4);
    
    // Smith GGX Geometry
    float k = (rough + 1.0);
    k = (k * k) / 8.0;
    float Gv = NdotV / max(NdotV * (1.0 - k) + k, 1e-4);
    float Gl = NdotL / max(NdotL * (1.0 - k) + k, 1e-4);
    float G = Gv * Gl;
    
    // Fresnel
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metal);
    float3 F = FresnelSchlick(VdotH, F0, rough);
    
    // Specular BRDF
    float3 spec = (D * G) * F / max(4.0 * NdotL * NdotV, 1e-4);
    
    // Diffuse BRDF
    float3 kd = (1.0 - F) * (1.0 - metal);
    float3 diffuse = kd * (albedo / 3.14159);
    
    float3 lighting;
    
    if (isVegetation)
    {
        float wrap = saturate(rawNdotL * 0.5 + 0.5);
        float vegShadow = lerp(1.0, shadowFactor, 0.85);
        
        float3 radiance = LightColor * LightIntensity * 3.14159 * wrap * vegShadow;
        lighting = diffuse * radiance * ao;
        
        // SSS translucency — use Load instead of Sample for compute
        float translucency = NormalTex.Load(coord).w;
        translucency = max(translucency, 0.3);
        float backlight = max(-rawNdotL, 0.0);
        float3 sssColor = albedo * float3(1.0, 0.85, 0.6);
        lighting += sssColor * LightColor * LightIntensity * backlight * translucency * 0.4;
    }
    else
    {
        float3 radiance = LightColor * LightIntensity * 3.14159 * NdotL * shadowFactor;
        lighting = (diffuse + spec) * radiance * ao;
    }
    
    // Shadow wrap
    lighting += diffuse * LightColor * LightIntensity * NdotL * (1.0 - shadowFactor) * 0.12 * ao;
    
    Output[px] = float4(lighting, 1.0f);
}
