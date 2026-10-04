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
    // Slots 2-3: PER-DRAW (command signature writes these)
    uint MeshPartId;                // 2: Index into MeshRegistry
    uint InstanceBaseOffset;        // 3: Base offset for instance ID (per-command)
    // Slots 4-8: PER-BATCH (set before ExecuteIndirect)
    uint DescriptorBufIdx;          // 4: StructuredBuffer<InstanceDescriptor>
    uint SortedIndicesIdx;          // 5: StructuredBuffer<uint> - sorted draw order indices
    uint MeshRegistryIdx;           // 6: StructuredBuffer<MeshPartEntry>
    uint MaterialsIdx;              // 7: Index to materials buffer
    uint GlobalTransformBufferIdx;  // 8: Index to global TransformBuffer
    // Slots 9-15: Reserved
    uint _reserved9;
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint _reserved15;
    // Slot 16: Debug
    uint DebugMode;             // 16: Debug visualization mode
};

#include "common.fx"
#include "sky_common.fx"

// Glass effects enabled for every material using this shader (bit mask, see "Feature mask" in PS):
// 1 = specular, 2 = refraction, 4 = sky reflection. 0 = base only: a plain alpha-blended lit surface, the
// known-good baseline to fall back to when bisecting (a material can add bits through DetailTiling.x).
// 7 validated 2026-10-03 (no bright panes in shade) after the reflection fixes below.
// 8 = damp the sky reflection on panes in the sun's shadow (heuristic), down to GLASS_SHADE_REFLECTION.
//     Off by default: shaded windows really do reflect the sky, and the gate cuts the reflection along shadow
//     edges. Use 15u if shaded narrow streets read too bright.
#define GLASS_FX_DEFAULT 7u
#define GLASS_SHADE_REFLECTION 0.25
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

    // Look up mesh buffer indices from MeshRegistry
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];

    StructuredBuffer<uint> indices = ResourceDescriptorHeap[part.IndexBufferIdx];
    uint vertexID = indices[fetchID + part.BaseIndex];

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[part.NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];

    float3 pos = positions[vertexID];
    float3 norm = normals[vertexID];
    float2 uv = uvs[vertexID];

    // World is an absolute transform and View carries the camera translation (same as gbuffer.fx).
    // The pixel shader works camera-relative (V = -WorldPos, cascade depth, LightSpaces lookup), so hand it
    // worldPos - CamPos; passing the absolute position made V point at the world origin, which flipped the
    // normal of every pane facing away from the origin (lit when facing away from the sun, unlit facing it).
    float4 worldPos = mul(float4(pos, 1.0f), World);
    output.WorldPos = float4(worldPos.xyz - CamPos, 1.0f);
    output.AbsWorldPos = worldPos.xyz;
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
    float3 Ng = N;   // geometric (pane) normal, before normal mapping — used for the glass reflection

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
    // Separate diffuse and specular: for transparent surfaces, specular is added
    // separately with Reinhard rolloff to prevent GGX peaks (500+ for smooth glass)
    // from dominating the alpha-blended output.
    float3 radiance = LightColor * LightIntensity * 3.14159 * NdotL * shadowFactor;
    float3 diffuseLighting = diffuse * radiance * ao;
    float3 specLighting = spec * radiance;

    // Hemisphere ambient
    float hemi = N.y * 0.5 + 0.5;
    float3 skyCol = float3(0.25, 0.28, 0.35);
    float3 gndCol = float3(0.12, 0.11, 0.10);
    float3 ambient = lerp(gndCol, skyCol, hemi) * ao * AmbientScale;
    float3 lighting = diffuseLighting + ambient * color.rgb;

    // Emissive adds directly to lighting (self-illumination)
    lighting += emissive * (1 - alpha);

    // ── Feature mask ──
    // The base is a lit surface alpha-blended by the albedo's alpha (known good). Effects are layered on top,
    // each behind one bit, so they can be switched live per material while they are being validated:
    // material DetailTiling.x (unused by this shader otherwise) = sum of
    //   1 = specular highlight   2 = refraction (scene snapshot tinted by the glass)   4 = sky reflection (Fresnel)
    uint glassFx = GLASS_FX_DEFAULT | (uint)(mat.DetailTiling.x + 0.5);

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

    // ── Glass Fresnel (Schlick): more reflective at glancing angles ──
    // From the pane's geometric normal: with the normal-mapped N the bumps of the lead cames swing NdotV
    // towards 0 and the cames flare with sky colour.
    float glassFresnel = lerp(0.04, 1.0, pow(1.0 - saturate(dot(Ng, V)), 5.0));
    // Only the smooth panes mirror their surroundings; rough parts of the texture (lead) do not.
    float gloss = saturate(1.0 - roughness);
    glassFresnel *= gloss * gloss;

    // Environment reflection: the sky above the horizon, a dark ground tone below it (a pane seen from above
    // reflects the street, not the sky — mirroring the direction upwards made such panes glow).
    float3 reflectDir = reflect(-V, Ng);
    float3 skyReflect = GetSkyColor(normalize(float3(reflectDir.x, max(reflectDir.y, 0.02), reflectDir.z)), FogSunDirection);
    float3 envReflect = lerp(gndCol * AmbientScale, skyReflect, smoothstep(-0.15, 0.1, reflectDir.y));

    // Base: the lit glass surface; the blend state mixes it over what is behind with alpha.
    // (Both lighting and refracted are linear HDR — finalize handles tonemapping.)
    float3 finalColor = lighting;
    float finalAlpha = saturate(alpha);

    // 2: refraction — replace the base by "scene behind, tinted by the glass" mixed with the lit surface.
    //    Output alpha still controls how much this pixel overwrites previously drawn transparent layers.
    if ((glassFx & 2u) != 0 && hasRefraction)
    {
        float3 tintedScene = refracted * lerp(float3(1,1,1), color.rgb, 0.3);
        finalColor = lerp(tintedScene, lighting, alpha);
    }

    // Surface terms. They belong to the glass surface, so the pane's transparency must not dim them:
    //   pixel = (finalColor·a + behind·(1−a))·(1−F) + F·env + spec
    // which the AlphaBlend state (src·A + dst·(1−A)) produces with
    //   A = 1 − (1−a)(1−F),   src = (finalColor·a·(1−F) + F·env + spec) / A
    // (Writing them into finalColor instead scaled them by a ≈ 0.3 a second time — together with the old
    // halvings the reflection ended up at ~0.2 % of the sky: invisible.)
    float Fr = 0.0;                                  // 4: Fresnel sky reflection, stronger at glancing angles
    if ((glassFx & 4u) != 0)
        Fr = glassFresnel;
    // 8: damp the sky reflection on panes the sun does not reach (shadow map at the pane). A heuristic, not
    //    reflection occlusion: shade usually means "narrow street, little sky to mirror", where an unoccluded sky
    //    reflection reads as a glow. Wrong for a shaded pane under open sky (it would really show sky) and for a
    //    sunlit pane facing a wall. A floor keeps shaded panes from going completely dead.
    if ((glassFx & 8u) != 0)
        Fr *= lerp(GLASS_SHADE_REFLECTION, 1.0, shadowFactor * step(0.0, dot(Ng, L)));
    float3 surface = Fr * envReflect;
    if ((glassFx & 1u) != 0)                         // 1: specular highlight — Reinhard rolloff against HDR blowout
        surface += specLighting / (1.0 + specLighting);

    float outAlpha = 1.0 - (1.0 - finalAlpha) * (1.0 - Fr);
    finalColor = (finalColor * finalAlpha * (1.0 - Fr) + surface) / max(outAlpha, 1e-3);
    finalAlpha = outAlpha;

    // Aerial perspective fog — input.Depth is linear view-space Z
    if (FogEnabled > 0)
    {
        float fogFactor = saturate(1.0 - exp(-pow(input.Depth * FogDensity, 2.0)));
        float3 viewDir = normalize(input.WorldPos.xyz);
        float3 inscatter = GetSkyColor(float3(viewDir.x, max(viewDir.y, 0.01), viewDir.z), FogSunDirection);
        finalColor = lerp(finalColor, inscatter, fogFactor);
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
