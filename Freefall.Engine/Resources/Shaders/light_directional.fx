cbuffer PushConstants : register(b3)
{
    uint NormalTexIdx;          // 0: G-Buffer normal texture
    uint DepthTexIdx;           // 1: G-Buffer depth texture (hardware)
    uint ShadowMapIdx;          // 2: Shadow map texture array
    uint DepthGBufIdx;          // 3: G-Buffer linear depth (R32_Float, for debug viz)
    uint AlbedoTexIdx;          // 4: G-Buffer albedo texture (for PBR)
    uint DataTexIdx;            // 5: G-Buffer data texture (roughness, metal, ao)
    uint LightingCascadeSRVIdx; // 6: SRV: GPU-computed cascade data (0 = use cbuffer)
};

#include "common.fx"
// @RenderState(DepthTest=false, DepthWrite=false, Blend=Additive)


// Light params from ObjectConstants (Slot 2, b1)
cbuffer ObjectConstants : register(b1)
{
    float3 LightColor;
    float LightIntensity;
    float3 LightDirection;
    float _pad0;
    
    // Shadow cascade data — arrays sized for MaxCascades (8), CascadeCount controls active count
    // Camera-relative VP matrices (matches posFromDepth output space)
    row_major float4x4 LightSpaces[8];
    float4 Cascades[8];                  // X=near, Y=far for each cascade
    
    int CascadeCount;
    int DebugVisualizationMode;
    float2 _pad1;
};

// GPU-computed lighting cascade data (matches cascade_compute.hlsl output)
struct LightingCascadeData {
    row_major float4x4 VP;   // Camera-relative light VP
    float4 Cascade;          // X=near, Y=far
};

// Dual-path cascade data access: when LightingCascadeSRVIdx > 0, read from GPU buffer;
// otherwise fall back to cbuffer arrays above.
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
SamplerState ShadowClampSampler : register(s2); // Linear+Clamp for shadow map sampling (debug)
SamplerComparisonState ShadowSampler : register(s3); // Comparison+Bilinear for PCF

// PBR helpers
float3 FresnelSchlick(float cosTheta, float3 F0, float roughness)
{
    // Roughness-attenuated Fresnel: rough surfaces don't get full grazing reflection
    float3 F_max = max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0);
    return F0 + (F_max - F0) * pow(1.0 - cosTheta, 5.0);
}

// Compute normal offset for shadow receiver to prevent acne.
// Shifts world position along the surface normal before light-space projection.
// Offset scales with shadow map texel size (from orthographic projection) and surface angle.
float4 ApplyNormalOffset(float4 worldPos, float3 normal, float3 lightDir, row_major float4x4 lightVP, float shadowMapRes)
{
    // Orthographic projection: world-space size of one shadow texel
    float texelWorldSize = 2.0 / (shadowMapRes * abs(lightVP._11));
    
    // Scale offset by sin(angle) — only surfaces at grazing angles need offset;
    // well-lit flat surfaces (sinTheta ≈ 0) get near-zero offset to avoid shifting shadows
    float cosTheta = saturate(dot(normal, -lightDir));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    float offsetScale = texelWorldSize * sinTheta * 0.8;
    offsetScale = min(offsetScale, 0.5); // hard clamp for large cascades
    
    return worldPos + float4(normal * offsetScale, 0);
}

// ─── Screen-Space Contact Shadows ──────────────────────────────────────────
// Ray-march toward the light using CameraRelativeVP for correct projection.
// Uses the linear depth buffer for simple, distance-independent depth comparison.

#define CONTACT_STEPS     24
#define CONTACT_MAX_DIST  0.75  // world-space max ray length (meters) — short to avoid mist around large objects

float ContactShadow(float3 worldPos, float3 lightDir, float2 screenPos, float linearDepth)
{
    // Skip distant pixels — contact shadows are a near-field effect
    // Smooth fade between 80-100m to avoid visible cutoff line
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
        
        // Project to screen using camera-relative VP
        float4 clip = mul(float4(rayPos, 1.0), CameraRelativeVP);
        if (clip.w <= 0.0) continue;
        float2 uv = (clip.xy / clip.w) * float2(0.5, -0.5) + 0.5;
        
        if (any(uv < 0.0) || any(uv > 1.0)) break;
        
        // Compare ray's linear depth vs scene's linear depth
        float sceneLinear = linDepth.SampleLevel(Sampler, uv, 0).r;
        float rayLinear = dot(rayPos, viewFwd);
        float penetration = rayLinear - sceneLinear;
        
        // Hit: ray clearly behind geometry (>15cm penetration, <30cm thickness)
        // Higher min penetration avoids stippled "mist" from grazing-angle hits
        if (penetration > 0.15 && penetration < 0.3)
            return distanceFade; // 0.0 at close range, fades to 1.0 at 80-100m
    }
    
    return 1.0;
}

struct VSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

VSOutput VS(uint vertexID : SV_VertexID)
{
    VSOutput output;
    
    // Procedural fullscreen quad from SV_VertexID (TriangleStrip, 4 verts)
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

float4 PS(VSOutput input) : SV_Target
{
    Texture2D NormalTex = ResourceDescriptorHeap[NormalTexIdx];
    Texture2D DepthTex = ResourceDescriptorHeap[DepthTexIdx];
    Texture2DArray ShadowMap = ResourceDescriptorHeap[ShadowMapIdx];
    
    // Pixel-exact Load — 1:1 fullscreen pass, no filtering needed
    int3 px = int3(input.Position.xy, 0);
    float3 normal = NormalTex.Load(px).xyz;
    float depth = DepthTex.Load(px).r;
    
    

    // Skip pixels at min depth (sky) — reverse depth: far=0
    if (depth <= 0.0f)
        return float4(0, 0, 0, 0);
    
    // Reconstruct world position from depth (needs UV coords, not pixel coords)
    float4 worldPos = posFromDepth(input.TexCoord, depth, CameraInverse);
    
    // Sample Data texture ONCE — used for vegetation flag AND PBR material below
    Texture2D DataTex = ResourceDescriptorHeap[DataTexIdx];
    float4 data = DataTex.Load(px);
    bool isVegetation = (data.a > 0.3 && data.a < 0.7);
    
    // Compute NdotL early — surfaces facing away from the light receive no
    // direct illumination, so skip shadow sampling entirely.
    float3 L = normalize(-LightDirection);
    float rawNdotL = dot(normal, L);
    float NdotL = max(rawNdotL, 0.0);
    
    // Vegetation needs wrap lighting for back-facing surfaces
    if (NdotL <= 0.0 && !isVegetation && DebugVisualizationMode == 0)
        return float4(0, 0, 0, 0);
    
    // Calculate view-space depth for cascade selection
    // worldPos is camera-relative (from zero-translation CameraInverse),
    // so use dot product with camera forward (View column 2) to get linear depth
    // without the translation component that's in the full View matrix
    float viewDepth = dot(worldPos.xyz, float3(View._13, View._23, View._33));
    
    // Select cascade based on view depth
    int cascadeIndex = CascadeCount - 1;
    for (int i = 0; i < CascadeCount; i++)
    {
        if (viewDepth < getCascade(i).y)
        {
            cascadeIndex = i;
            break;
        }
    }
    
    // Calculate shadow with cascade blending
    float shadowFactor = 1.0f;
    float4 lightSpacePos = float4(0,0,0,1);
    float2 shadowUV = float2(0,0);
    float sampledDepth = 0;
    
    // Beyond the last cascade far distance — no shadow (fully lit)
    if (viewDepth > getCascade(CascadeCount - 1).y)
        shadowFactor = 1.0f;
    else if (ShadowMapIdx > 0) // Only sample if shadow map is bound

    {
        // Transform world position to light space for current cascade
        lightSpacePos = mul(worldPos, getLightSpace(cascadeIndex));
        lightSpacePos /= lightSpacePos.w;
        
        // Convert from [-1,1] to [0,1] UV space
        shadowUV = lightSpacePos.xy * 0.5f + 0.5f;
        shadowUV.y = 1.0f - shadowUV.y; // Flip Y for D3D
        
        // Bounds check: skip shadow for pixels outside the shadow map
        if (shadowUV.x >= 0.0f && shadowUV.x <= 1.0f && shadowUV.y >= 0.0f && shadowUV.y <= 1.0f)
        {
            sampledDepth = ShadowMap.SampleLevel(ShadowClampSampler, float3(shadowUV, cascadeIndex), 0).r;
            float zScale = abs(getLightSpace(cascadeIndex)._33);
            shadowFactor = GetShadowFactor(ShadowMap, ShadowSampler, shadowUV, lightSpacePos.z, cascadeIndex, normal, LightDirection, zScale, input.Position.xy);
            
            // Cascade blending: cross-fade near cascade boundaries to eliminate seams
            // Outgoing blend: fade toward next cascade at far edge
            if (cascadeIndex < CascadeCount - 1)
            {
                float cascadeRange = getCascade(cascadeIndex).y - getCascade(cascadeIndex).x;
                float blendZone = cascadeRange * 0.1f;
                float distToEdge = getCascade(cascadeIndex).y - viewDepth;
                
                if (distToEdge < blendZone)
                {
                    float blendFactor = distToEdge / blendZone; // 1 at start of zone, 0 at boundary
                    
                    int nextCascade = cascadeIndex + 1;
                    float4 nextLSP = mul(worldPos, getLightSpace(nextCascade));
                    nextLSP /= nextLSP.w;
                    float2 nextUV = nextLSP.xy * 0.5f + 0.5f;
                    nextUV.y = 1.0f - nextUV.y;
                    
                    if (nextUV.x >= 0.0f && nextUV.x <= 1.0f && nextUV.y >= 0.0f && nextUV.y <= 1.0f)
                    {
                        float nextZScale = abs(getLightSpace(nextCascade)._33);
                        float nextShadow = GetShadowFactor(ShadowMap, ShadowSampler, nextUV, nextLSP.z, nextCascade, normal, LightDirection, nextZScale, input.Position.xy);
                        shadowFactor = lerp(nextShadow, shadowFactor, blendFactor);
                    }
                }
            }

        }
    }

    // Screen-space contact shadow: catch fine detail the cascade maps miss (WIP: needs blur/TAA)
    // float contactShadow = ContactShadow(worldPos.xyz, LightDirection, input.Position.xy, viewDepth);
    // shadowFactor = min(shadowFactor, contactShadow);
    float contactShadow = 1.0; // disabled for now

    // Debug visualization modes
    if (DebugVisualizationMode == 1)
    {
        // Cascade index as color: R=0, G=1, B=2, Y=3
        float4 cascadeColors[4] = { float4(1,0,0,1), float4(0,1,0,1), float4(0,0,1,1), float4(1,1,0,1) };
        return cascadeColors[cascadeIndex];
    }
    if (DebugVisualizationMode == 2)
    {
        // Raw shadow factor as grayscale
        return float4(shadowFactor.xxx, 1);
    }
    if (DebugVisualizationMode == 3)
    {
        // Contact shadow only (white = lit, black = shadowed)
        return float4(contactShadow.xxx, 1);
    }
    if (DebugVisualizationMode == 4)
    {
        // Visualize linear depth buffer (DepthGBufIdx)
        Texture2D<float> depthViz = ResourceDescriptorHeap[DepthGBufIdx];
        float d = depthViz.Load(px).r;
        float normalized = saturate(d / 100.0); // Map [0..100m] to [0..1]
        return float4(normalized, normalized, normalized, 1);
    }
    
    // PBR lighting (data already sampled above)
    Texture2D AlbedoTex = ResourceDescriptorHeap[AlbedoTexIdx];
    float3 albedo = AlbedoTex.Load(px).rgb;
    
    // Self-lit check: Data.a == 0 means pixel is already fully lit (e.g. ocean)
    // Skip deferred lighting to avoid double-illumination
    if (data.a < 0.01)
        return float4(0, 0, 0, 0);
    
    float rough = max(data.r, 0.04);
    float metal = saturate(data.g);
    float ao    = saturate(data.b);
    
    // View and half vectors (worldPos is camera-relative, camera at origin)
    float3 V = normalize(-worldPos.xyz);
    float3 H = normalize(L + V);
    float NdotV = max(dot(normal, V), 0.001);
    float NdotH = max(dot(normal, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    
    float a  = rough * rough;
    float a2 = a * a;
    
    // GGX Normal Distribution Function
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    float D = a2 / max(3.14159 * denom * denom, 1e-4);
    
    // Smith GGX Geometry (Schlick-GGX)
    float k = (rough + 1.0);
    k = (k * k) / 8.0;
    float Gv = NdotV / max(NdotV * (1.0 - k) + k, 1e-4);
    float Gl = NdotL / max(NdotL * (1.0 - k) + k, 1e-4);
    float G = Gv * Gl;
    
    // Fresnel (Schlick)
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metal);
    float3 F = FresnelSchlick(VdotH, F0, rough);
    
    // Specular BRDF
    float3 spec = (D * G) * F / max(4.0 * NdotL * NdotV, 1e-4);
    
    // Diffuse BRDF (energy-conserving Lambert)
    float3 kd = (1.0 - F) * (1.0 - metal);
    float3 diffuse = kd * (albedo / 3.14159);
    
    // Pre-multiply by PI to compensate for energy-conserving Lambert (/PI in diffuse BRDF).
    // Convention: LightIntensity=1 → full diffuse albedo brightness (standard in UE/Unity).
    float3 lighting;
    
    if (isVegetation)
    {
        // Wrap lighting: smooth linear gradient, compressed range
        // NdotL=-1→0.2, NdotL=0→0.5, NdotL=1→0.8
        float wrap = saturate(rawNdotL * 0.3 + 0.5);
        
        // Attenuate self-shadowing: leaf-on-leaf shadows are noisy, keep them subtle
        float vegShadow = lerp(1.0, shadowFactor, 0.85);
        
        float3 radiance = LightColor * LightIntensity * 3.14159 * wrap * vegShadow;
        lighting = diffuse * radiance * ao;
        
        // SSS: backlit translucency — light passes THROUGH leaves
        float translucency = NormalTex.Sample(Sampler, input.TexCoord).w;
        translucency = max(translucency, 0.3); // moderate minimum for all foliage
        float backlight = max(-rawNdotL, 0.0);
        float3 sssColor = albedo * float3(1.0, 0.85, 0.6);
        lighting += sssColor * LightColor * LightIntensity * backlight * translucency * 0.4;
    }
    else
    {
        float3 radiance = LightColor * LightIntensity * 3.14159 * NdotL * shadowFactor;
        lighting = (diffuse + spec) * radiance * ao;
    }
    
    // Shadow wrap: bleed a small fraction of diffuse NdotL through shadows
    // Gives cylindrical/rounded surfaces (tree trunks, pillars) visible shape in shadow
    lighting += diffuse * LightColor * LightIntensity * NdotL * (1.0 - shadowFactor) * 0.12 * ao;
    
    return float4(lighting, 1.0f);
}

technique11 Standard
{
    pass Light
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
