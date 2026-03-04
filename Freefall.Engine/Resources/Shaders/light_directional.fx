#include "common.fx"
// @RenderState(DepthTest=false, DepthWrite=false, Blend=Additive)

// Light Pass Push Constant Layout
// Slots 0-5: Light-specific textures (procedural fullscreen quad, no vertex buffers)
#define NormalTexIdx GET_INDEX(0)   // G-Buffer normal texture
#define DepthTexIdx GET_INDEX(1)    // G-Buffer depth texture (hardware)
#define ShadowMapIdx GET_INDEX(2)   // Shadow map texture array
#define DepthGBufIdx GET_INDEX(3)   // G-Buffer linear depth (R32_Float, for debug viz)
#define AlbedoTexIdx GET_INDEX(4)   // G-Buffer albedo texture (for PBR)
#define DataTexIdx GET_INDEX(5)     // G-Buffer data texture (roughness, metal, ao)
#define LightingCascadeSRVIdx GET_INDEX(6)  // SRV: GPU-computed cascade data (0 = use cbuffer)


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
    
    float3 normal = NormalTex.Sample(Sampler, input.TexCoord).xyz;
    float depth = DepthTex.Sample(Sampler, input.TexCoord).r;
    
    // Mode 3: Visualize GBuffer linear depth (normalized to visible range)
    if (DebugVisualizationMode == 3)
    {
        Texture2D DepthGBuf = ResourceDescriptorHeap[DepthGBufIdx];
        float gbufDepth = DepthGBuf.Sample(Sampler, input.TexCoord).r;
        float normalized = saturate(gbufDepth / 500.0); // Map [0..500] to [0..1]
        return float4(normalized, normalized, normalized, 1);
    }
    
    // Skip pixels at min depth (sky) — reverse depth: far=0
    if (depth <= 0.0f)
        return float4(0, 0, 0, 0);
    
    // Reconstruct world position from depth
    float4 worldPos = posFromDepth(input.TexCoord, depth, CameraInverse);
    
    // Vegetation flag: data.a = 0.5
    Texture2D DataTexEarly = ResourceDescriptorHeap[DataTexIdx];
    float dataA = DataTexEarly.Sample(Sampler, input.TexCoord).a;
    bool isVegetation = (dataA > 0.3 && dataA < 0.7);
    
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
    
    // PBR lighting
    Texture2D AlbedoTex = ResourceDescriptorHeap[AlbedoTexIdx];
    Texture2D DataTex = ResourceDescriptorHeap[DataTexIdx];
    float3 albedo = AlbedoTex.Sample(Sampler, input.TexCoord).rgb;
    float4 data = DataTex.Sample(Sampler, input.TexCoord);
    
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
