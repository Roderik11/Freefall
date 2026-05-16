// Forward-rendered transparent shader (glass, windows, etc.)
// Same vertex pipeline as gbuffer.fx, but does inline PBR lighting
// in the pixel shader and alpha-blends onto the Composite buffer.
//
// Runs in RenderPass.Forward with:
//   - Depth test ON, depth write OFF (draws behind opaque, doesn't block)
//   - Alpha blending enabled
//   - 2 render targets: Composite (R8G8B8A8_UNorm) + EntityId (R32_UInt)

cbuffer PushConstants : register(b3)
{
    // Slots 0-1: Forward lighting data
    uint ShadowMapIdx;          // 0: Shadow cascade array SRV
    uint CompositeSRVIdx;       // 1: CompositeSnapshot SRV for refraction
    // Slots 2-13: Mesh draw data (matches gbuffer.fx exactly)
    uint DescriptorBufIdx;      // 2
    uint _reserved3;            // 3
    uint SortedIndicesIdx;      // 4
    uint BoneWeightsIdx;        // 5
    uint BonesIdx;              // 6
    uint IndexBufferIdx;        // 7
    uint BaseIndex;             // 8
    uint PosBufferIdx;          // 9
    uint NormBufferIdx;         // 10
    uint UVBufferIdx;           // 11
    uint NumBones;              // 12
    uint InstanceBaseOffset;    // 13
    // Slots 14-15: Global bindless buffers
    uint MaterialsIdx;          // 14
    uint GlobalTransformBufferIdx; // 15
    // Slot 16: Debug
    uint DebugMode;             // 16
};

#include "common.fx"
#include "sky_common.fx"
// @RenderState(RenderTargets=2, DepthWrite=false, Blend=AlphaBlend, CullMode=None)

// Light params from ObjectConstants
cbuffer ObjectConstants : register(b1)
{
    float3 LightColor;
    float LightIntensity;
    float3 LightDirection;
    float _pad0;

    row_major float4x4 LightSpaces[8];
    float4 Cascades[8];

    int CascadeCount;
    int _debugMode;
    float2 _pad1;
};

inline MaterialData GetMaterial(uint id)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    return materials[id];
}
#define GET_MATERIAL(id) GetMaterial(id)

SamplerState Sampler : register(s0);
SamplerComparisonState ShadowSampler : register(s3);

// ═══════════════════════════════════════════════════════════════════════
// VERTEX SHADER — identical to gbuffer.fx
// ═══════════════════════════════════════════════════════════════════════

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 WorldPos : TEXCOORD1;        // camera-relative world position
    nointerpolation uint MaterialID : TEXCOORD2;
    float Depth : TEXCOORD3;
    nointerpolation uint TransformSlot : TEXCOORD4;
    nointerpolation uint MeshPartIdx : TEXCOORD5;
    float3 AbsWorldPos : TEXCOORD6;     // absolute world position (for shadow lookup)
};

struct PSOutput
{
    float4 Color : SV_Target0;      // Composite buffer
    uint   EntityId : SV_Target1;   // EntityId buffer
};

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];

    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;

    InstanceDescriptor desc = descriptors[idx];
    uint slot = desc.TransformSlot;
    uint materialID = desc.MaterialId;

    row_major matrix World = globalTransforms[slot];

    // Negative-scale fix
    float3x3 W3 = (float3x3)World;
    float det = determinant(W3);

    uint fetchID = primitiveVertexID;
    if (det < 0.0f)
    {
        uint triLocal = primitiveVertexID % 3;
        uint triBase = primitiveVertexID - triLocal;
        fetchID = triBase + (triLocal == 1u ? 2u : (triLocal == 2u ? 1u : 0u));
    }

    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[fetchID + BaseIndex];

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[UVBufferIdx];

    float3 pos = positions[vertexID];
    float3 norm = normals[vertexID];
    float2 uv = uvs[vertexID];

    float4 worldPos = mul(float4(pos, 1.0f), World);
    output.WorldPos = worldPos;
    output.AbsWorldPos = worldPos.xyz + CamPos;
    output.Position = mul(mul(worldPos, View), Projection);
    output.Normal = mul(norm, W3);
    output.TexCoord = uv;
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = materialID;
    output.Depth = output.Position.w;
    output.TransformSlot = slot;
    output.MeshPartIdx = desc.MeshPartIdx;
    return output;
}

// ═══════════════════════════════════════════════════════════════════════
// PBR HELPERS
// ═══════════════════════════════════════════════════════════════════════

float3 FresnelSchlick(float cosTheta, float3 F0, float roughness)
{
    float3 F_max = max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0);
    return F0 + (F_max - F0) * pow(1.0 - cosTheta, 5.0);
}

// ═══════════════════════════════════════════════════════════════════════
// PIXEL SHADER — inline PBR lighting for transparent surfaces
// ═══════════════════════════════════════════════════════════════════════

PSOutput PS(VSOutput input)
{
    PSOutput output;

    MaterialData mat = GET_MATERIAL(input.MaterialID);
    Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];

    float4 color = albedoTex.Sample(Sampler, input.TexCoord);

    // Alpha from albedo texture drives transparency (glass typically has low alpha)
    float alpha = color.a;

    // PBR material properties
    float roughness = 0.05;  // glass default: very smooth
    float metal = 0.0;
    float ao = 1.0;

    if (mat.RoughnessIdx != 0) { Texture2D rTex = ResourceDescriptorHeap[mat.RoughnessIdx]; roughness = 1.0 - rTex.Sample(Sampler, input.TexCoord).r; }
    if (mat.MetallicIdx  != 0) { Texture2D mTex = ResourceDescriptorHeap[mat.MetallicIdx];  metal     = mTex.Sample(Sampler, input.TexCoord).r; }
    if (mat.AOIdx        != 0) { Texture2D aTex = ResourceDescriptorHeap[mat.AOIdx];        ao        = aTex.Sample(Sampler, input.TexCoord).r; }

    // Emissive: sample texture, apply tint/intensity
    float3 emissive = float3(0, 0, 0);
    if (mat.EmissiveIdx != 0)
    {
        Texture2D eTex = ResourceDescriptorHeap[mat.EmissiveIdx];
        emissive = eTex.Sample(Sampler, input.TexCoord).rgb * mat.EmissiveColor * mat.EmissiveIntensity;
    }

    // Normal mapping via cotangent frame (same as gbuffer.fx)
    float3 N = normalize(input.Normal);

    // Flip normal for back faces (CullMode=None) — ensures correct lighting
    // from both sides. Without this, rotated glass gets NdotV≈0 → full Fresnel → dark.
    float3 V = normalize(-input.WorldPos.xyz);
    if (dot(N, V) < 0.0)
        N = -N;

    float3 dp1 = ddx(input.WorldPos.xyz);
    float3 dp2 = ddy(input.WorldPos.xyz);
    float2 duv1 = ddx(input.TexCoord);
    float2 duv2 = ddy(input.TexCoord);

    float3 dp2perp = cross(dp2, N);
    float3 dp1perp = cross(N, dp1);
    float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    float handedness = dot(cross(T, B), N) < 0.0 ? -1.0 : 1.0;
    T *= handedness;

    float invmax = rsqrt(max(dot(T, T), dot(B, B)));
    float3x3 TBN = float3x3(T * invmax, B * invmax, N);

    if (mat.NormalIdx != 0)
    {
        Texture2D normalTex = ResourceDescriptorHeap[mat.NormalIdx];
        float2 nXY = normalTex.Sample(Sampler, input.TexCoord).rg * 2.0 - 1.0;
        nXY.y = -nXY.y;
        float3 texNormal = float3(nXY, sqrt(max(0.001, 1.0 - dot(nXY, nXY))));
        N = normalize(mul(texNormal, TBN));
    }

    // ── Directional light PBR ──
    float3 L = normalize(-LightDirection);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    float3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    float rough = max(roughness, 0.04);
    float a = rough * rough;
    float a2 = a * a;

    // GGX NDF
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    float D = a2 / max(3.14159 * denom * denom, 1e-4);

    // Smith GGX Geometry
    float k = (rough + 1.0);
    k = (k * k) / 8.0;
    float Gv = NdotV / max(NdotV * (1.0 - k) + k, 1e-4);
    float Gl = NdotL / max(NdotL * (1.0 - k) + k, 1e-4);
    float G = Gv * Gl;

    // Fresnel
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), color.rgb, metal);
    float3 F = FresnelSchlick(VdotH, F0, rough);

    // Specular BRDF
    float3 spec = (D * G) * F / max(4.0 * NdotL * NdotV, 1e-4);

    // Diffuse BRDF
    float3 kd = (1.0 - F) * (1.0 - metal);
    float3 diffuse = kd * (color.rgb / 3.14159);

    // ── Shadow cascade lookup (camera-relative) ──
    float shadowFactor = 1.0;
    float viewDepth = dot(input.WorldPos.xyz, float3(View._13, View._23, View._33));

    if (ShadowMapIdx > 0 && viewDepth <= Cascades[CascadeCount - 1].y)
    {
        Texture2DArray ShadowMap = ResourceDescriptorHeap[ShadowMapIdx];

        int cascadeIndex = CascadeCount - 1;
        for (int ci = 0; ci < CascadeCount; ci++)
        {
            if (viewDepth < Cascades[ci].y)
            {
                cascadeIndex = ci;
                break;
            }
        }

        // Camera-relative worldPos → light-space via camera-relative LightSpaces[]
        float4 lsPos = mul(input.WorldPos, LightSpaces[cascadeIndex]);
        lsPos /= lsPos.w;

        float2 shadowUV = lsPos.xy * 0.5 + 0.5;
        shadowUV.y = 1.0 - shadowUV.y;

        if (all(shadowUV >= 0.0) && all(shadowUV <= 1.0))
        {
            float zScale = abs(LightSpaces[cascadeIndex]._33);
            shadowFactor = GetShadowFactor(ShadowMap, ShadowSampler, shadowUV, lsPos.z,
                cascadeIndex, N, LightDirection, zScale, input.Position.xy);
        }
    }

    // ── Combine lighting ──
    float3 radiance = LightColor * LightIntensity * 3.14159 * NdotL * shadowFactor;
    float3 lighting = (diffuse + spec) * radiance * ao;

    // Hemisphere ambient
    float hemi = N.y * 0.5 + 0.5;
    float3 skyCol = float3(0.25, 0.28, 0.35);
    float3 gndCol = float3(0.12, 0.11, 0.10);
    float3 ambient = lerp(gndCol, skyCol, hemi) * ao * AmbientScale;
    lighting += ambient * color.rgb;

    // Emissive adds directly to lighting (self-illumination)
    lighting += emissive * (1 - alpha);

    // ── Refraction: sample scene behind through glass ──
    float3 refracted = float3(0, 0, 0);
    bool hasRefraction = false;
    if (CompositeSRVIdx > 0)
    {
        Texture2D<float4> compositeSnap = ResourceDescriptorHeap[CompositeSRVIdx];
        // Compute proper screen UV from SV_POSITION
        uint2 dims;
        compositeSnap.GetDimensions(dims.x, dims.y);
        float2 screenUV = input.Position.xy / float2(dims);
        // Normal-based refraction offset
        float2 refrOffset = N.xz * 0.02 * (1.0 - alpha);
        screenUV = saturate(screenUV + refrOffset);
        refracted = compositeSnap.SampleLevel(Sampler, screenUV, 0).rgb;
        hasRefraction = true;
    }

    // ── Glass Fresnel: more reflective at glancing angles ──
    float glassFresnel = pow(1.0 - NdotV, 4.0);
    glassFresnel = lerp(0.04, 1.0, glassFresnel);

    // Environment reflection (sky)
    float3 reflectDir = reflect(-V, N);
    reflectDir.y = abs(reflectDir.y);
    float3 envReflect = GetSkyColor(reflectDir, FogSunDirection) * 0.5;

    float3 finalColor;
    float finalAlpha;

    if (hasRefraction)
    {
        // Refraction path: alpha controls glass opacity.
        // alpha=0 → fully transparent (show scene behind), alpha=1 → fully opaque (show lit surface)
        // The composite snapshot is already tonemapped+gamma'd, so we apply the same to the glass
        // lighting before mixing.
        float3 glassLit = lighting;
        
        // Tonemapping + gamma on the glass lighting component only
        float3 x = glassLit;
        glassLit = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
        glassLit = saturate(glassLit);
        glassLit = pow(abs(glassLit), 1.0 / 2.2);
        
        // Tint scene-behind by glass color
        float3 tintedScene = refracted * lerp(float3(1,1,1), color.rgb, 0.3);
        
        // Mix: alpha drives opacity (scene-behind vs glass surface)
        finalColor = lerp(tintedScene, glassLit, alpha);
        
        // Fresnel reflection on top (increases opacity at glancing angles)
        finalColor = lerp(finalColor, envReflect, glassFresnel * 0.5);
        
        // Specular highlights always on top
        float3 specTM = spec * radiance;
        specTM = (specTM * (2.51 * specTM + 0.03)) / (specTM * (2.43 * specTM + 0.59) + 0.14);
        finalColor += saturate(specTM);
        
        // Fog
        if (FogEnabled > 0)
        {
            float3 fogColor = GetSkyColor(float3(0, 0.01, 1), FogSunDirection);
            finalColor = FOG(finalColor, input.Depth, fogColor);
        }
        
        // Output alpha for multi-layer transparency:
        // The internal lerp already mixed refracted scene with glass lighting using alpha.
        // Output alpha controls how much this pixel overwrites previously-drawn transparent layers.
        // At alpha=0.5: this glass overwrites 50% of the RT (which may contain earlier transparent draws).
        // Fresnel increases opacity at glancing angles.
       // finalAlpha = saturate(alpha + glassFresnel * 0.3);
        finalAlpha = saturate(alpha);
    }
    else
    {
        // Fallback: no composite snapshot available, use alpha blending
        finalColor = lerp(envReflect, lighting, 1.0 - glassFresnel);
        finalColor += spec * radiance;
        
        // Fog
        if (FogEnabled > 0)
        {
            float3 fogColor = GetSkyColor(float3(0, 0.01, 1), FogSunDirection);
            finalColor = FOG(finalColor, input.Depth, fogColor);
        }
        
        // Tonemapping + gamma
        float3 x = finalColor;
        finalColor = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
        finalColor = saturate(finalColor);
        finalColor = pow(abs(finalColor), 1.0 / 2.2);
        
        finalAlpha = alpha;
    }

    output.Color = float4(finalColor, finalAlpha);
    output.EntityId = (input.TransformSlot << 8u) | (input.MeshPartIdx & 0xFFu);
    return output;
}

technique11 GBuffer
{
    pass Forward
    {
        SetRenderState(RenderTargets=2, DepthWrite=false, Blend=AlphaBlend);
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
