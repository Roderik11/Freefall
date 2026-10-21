// Point Light Compute Shader — PBR lighting of the opaque surfaces by this frame's visible point lights
//
// CSPointLight: each 8×8 group takes the lights the light clusters give its tile and depth range (light_clusters.hlsli),
// tests them against its own frustum and depths, and every pixel shades with those. Since 2026-10-20; before, each
// group tested every visible light against its frustum (no depth bounds, at most 128 a tile) and skipped those a Hi-Z
// pass had flagged.

#pragma kernel CSPointLight

cbuffer PushConstants : register(b3)
{
    uint NormalTexIdx;
    uint DepthTexIdx;
    uint AlbedoTexIdx;
    uint DataTexIdx;
    uint OutputUAVIdx;
    uint ScreenWidthIdx;
    uint ScreenHeightIdx;
    uint LocalShadowAtlasIdx;   // Texture2DArray: point light shadow atlas (0 = none rendered this frame)
    uint LocalShadowViewsIdx;   // StructuredBuffer<LocalShadowView>
    uint LocalShadowLightsIdx;  // StructuredBuffer<LocalShadowLight>
    // Screen-space shadows of the top lights (DeferredRenderer.TracePointLightScreenShadows): the registry slot each
    // mask was traced for (0xFFFFFFFF = none) and the mask (R8, 1 = lit)
    uint ScreenShadowSlot0;
    uint ScreenShadowSlot1;
    uint ScreenShadowSlot2;
    uint ScreenShadowSlot3;
    uint ScreenShadowTex0Idx;
    uint ScreenShadowTex1Idx;
    uint ScreenShadowTex2Idx;
    uint ScreenShadowTex3Idx;
};

#include "common.fx"
#include "point_lights.hlsli"
#include "light_clusters.hlsli"

SamplerComparisonState ShadowSampler : register(s3); // LessEqual, linear: standard Z

// ── Point light shadows (LocalShadows.cs) ─────────────────────────────────
// 1 = lit. The receiver is pushed toward the light and off the surface by about a texel of the face it
// falls in (a perspective face's texels grow with distance, so a fixed depth bias would be 2 cm at 1 m and
// half a metre at 5 m); the face is chosen from the pushed point, whose projection then lies inside the
// face's 90° and so inside its tile's guard band. 3x3 bilinear compare taps.
float PointLightShadow(PointLightData light, float3 worldPos, float3 N, float3 L)
{
    if (light.ShadowLight == LOCAL_SHADOW_NO_VIEW || LocalShadowAtlasIdx == 0)
        return 1.0;

    StructuredBuffer<LocalShadowLight> shadowLights = ResourceDescriptorHeap[LocalShadowLightsIdx];
    StructuredBuffer<LocalShadowView> views = ResourceDescriptorHeap[LocalShadowViewsIdx];

    // Fading in after being picked, or out after being dropped from the budget
    float strength = shadowLights[light.ShadowLight].Strength;
    if (strength <= 0.0)
        return 1.0;

    float3 fromLight = worldPos - light.Position;
    float axisDistance = max(abs(fromLight.x), max(abs(fromLight.y), abs(fromLight.z)));
    uint v0 = shadowLights[light.ShadowLight].FaceView[LocalShadowFace(fromLight)];
    if (v0 == LOCAL_SHADOW_NO_VIEW)
        return 1.0;
    float texel = views[v0].TexelPerMetre * axisDistance;

    float3 biased = worldPos + N * (texel * 1.0) + L * (texel * 1.0);
    uint v = shadowLights[light.ShadowLight].FaceView[LocalShadowFace(biased - light.Position)];
    if (v == LOCAL_SHADOW_NO_VIEW)
        return 1.0;
    LocalShadowView view = views[v];

    float4 clip = mul(float4(biased, 1.0), view.SampleVP);
    float3 ndc = clip.xyz / clip.w;
    float2 uv = ndc.xy * float2(0.5, -0.5) + 0.5;

    // Stay inside the tile: the taps reach 2 texels out
    float tileTexels = view.Tile.x * LOCAL_SHADOW_PAGE_SIZE;
    float inset = 2.0 / tileTexels;
    uv = clamp(uv, inset, 1.0 - inset);
    float2 atlasUV = uv * view.Tile.xy + view.Tile.zw;

    Texture2DArray<float> atlas = ResourceDescriptorHeap[LocalShadowAtlasIdx];
    float t = 1.0 / LOCAL_SHADOW_PAGE_SIZE;
    float lit = 0;
    [unroll]
    for (int y = -1; y <= 1; y++)
    {
        [unroll]
        for (int x = -1; x <= 1; x++)
            lit += atlas.SampleCmpLevelZero(ShadowSampler, float3(atlasUV + float2(x, y) * t, view.Page), ndc.z);
    }
    return lerp(1.0, lit / 9.0, strength);
}

// 1 = lit. The screen-space shadow of a light among the top ones, faded with its atlas shadow (picked, dropped,
// outshone by daylight): what the atlas cannot resolve, grass blades and flowers, at the pixel.
float PointLightScreenShadow(PointLightData light, uint2 px)
{
    if (light.ShadowLight == POINT_LIGHT_NO_SHADOW || LocalShadowAtlasIdx == 0)
        return 1.0;
    uint mask = light.Slot == ScreenShadowSlot0 ? ScreenShadowTex0Idx
              : light.Slot == ScreenShadowSlot1 ? ScreenShadowTex1Idx
              : light.Slot == ScreenShadowSlot2 ? ScreenShadowTex2Idx
              : light.Slot == ScreenShadowSlot3 ? ScreenShadowTex3Idx : 0;
    if (mask == 0)
        return 1.0;
    StructuredBuffer<LocalShadowLight> shadowLights = ResourceDescriptorHeap[LocalShadowLightsIdx];
    Texture2D<float> maskTex = ResourceDescriptorHeap[mask];
    return lerp(1.0, maskTex.Load(int3(px, 0)), shadowLights[light.ShadowLight].Strength);
}

// PBR helpers
float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// ── Tiles ──────────────────────────────────────────────────────────────────
#define TILE_SIZE 8

// Build 4 side planes for a screen-space tile, in camera-relative view space.
// Each plane normal points INWARD (toward tile center).
// Uses the inverse projection to unproject tile corners to view-space rays.
void BuildTileFrustumPlanes(
    uint2 tileCoord, float2 screenSize,
    float4x4 invProj,
    out float4 planes[4])
{
    // Tile bounds in NDC [-1,1] (Y flipped for D3D)
    float2 tileMin = float2(tileCoord) * TILE_SIZE / screenSize;
    float2 tileMax = float2(tileCoord + 1) * TILE_SIZE / screenSize;

    // NDC: X [-1,1], Y [1,-1] (top=+1 in NDC, bottom=-1)
    float ndcL = tileMin.x * 2.0 - 1.0;
    float ndcR = tileMax.x * 2.0 - 1.0;
    float ndcT = 1.0 - tileMin.y * 2.0;  // top edge (smaller y = higher NDC)
    float ndcB = 1.0 - tileMax.y * 2.0;  // bottom edge

    // Unproject 4 corner rays to view space (at z=1 plane)
    // We only need direction, so w divide gives us the view-space ray
    float4 tl = mul(float4(ndcL, ndcT, 1, 1), invProj);
    float4 tr = mul(float4(ndcR, ndcT, 1, 1), invProj);
    float4 bl = mul(float4(ndcL, ndcB, 1, 1), invProj);
    float4 br = mul(float4(ndcR, ndcB, 1, 1), invProj);

    float3 vtl = tl.xyz / tl.w;
    float3 vtr = tr.xyz / tr.w;
    float3 vbl = bl.xyz / bl.w;
    float3 vbr = br.xyz / br.w;

    // 4 side planes: normal = cross product of two edge rays, pointing inward
    // Left plane:   origin → tl, origin → bl (normal points right/inward)
    // Right plane:  origin → br, origin → tr (normal points left/inward)
    // Top plane:    origin → tr, origin → tl (normal points down/inward)
    // Bottom plane: origin → bl, origin → br (normal points up/inward)
    planes[0] = float4(normalize(cross(vtl, vbl)), 0); // left
    planes[1] = float4(normalize(cross(vbr, vtr)), 0); // right
    planes[2] = float4(normalize(cross(vtr, vtl)), 0); // top
    planes[3] = float4(normalize(cross(vbl, vbr)), 0); // bottom
}

// Test sphere against 4 tile frustum planes (view space)
bool SphereFrustumTest(float3 center, float radius, float4 planes[4])
{
    [unroll]
    for (int i = 0; i < 4; i++)
    {
        if (dot(planes[i].xyz, center) < -radius)
            return false;
    }
    return true;
}

// ── A pixel's surface, and one light on it ──────────────────────────────────
struct PointSurface
{
    float3 Pos;             // camera-relative
    float3 N;
    float3 V;
    float NdotV;
    float3 F0;
    float3 DiffuseAlbedo;
    float A2;
    float K;
    float Gv;
    float Metal;
    float AO;
    bool Vegetation;
    uint2 Px;
};

// The G-buffer at a pixel; false for the sky
bool LoadPointSurface(uint2 px, float2 screenSize, out PointSurface s)
{
    s = (PointSurface)0;
    Texture2D NormalTex = ResourceDescriptorHeap[NormalTexIdx];
    Texture2D DepthTex  = ResourceDescriptorHeap[DepthTexIdx];
    int3 coord = int3(px, 0);
    float depth = DepthTex.Load(coord).r;
    // Sky pixels: reverse depth, far = 0
    if (depth <= 0.0f)
        return false;

    // World position from depth (camera-relative, zero translation)
    float2 texCoord = (float2(px) + 0.5) / screenSize;
    s.Pos = posFromDepth(texCoord, depth, CameraInverse).xyz;
    s.N = NormalTex.Load(coord).xyz;
    s.Px = px;

    Texture2D AlbedoTex = ResourceDescriptorHeap[AlbedoTexIdx];
    Texture2D DataTex   = ResourceDescriptorHeap[DataTexIdx];
    float3 albedo   = AlbedoTex.Load(coord).rgb;
    float4 dataFull = DataTex.Load(coord);
    s.Vegetation = (dataFull.a > 0.3 && dataFull.a < 0.7);

    float rough = max(dataFull.r, 0.04);
    s.Metal = saturate(dataFull.g);
    s.AO    = saturate(dataFull.b);

    // View vector (the camera at the origin)
    s.V = normalize(-s.Pos);
    s.NdotV = max(dot(s.N, s.V), 0.001);
    s.F0 = lerp(float3(0.04, 0.04, 0.04), albedo, s.Metal);
    float a = rough * rough;
    s.A2 = a * a;
    // Smith GGX geometry term (view component, shared across lights)
    s.K = (rough + 1.0) * (rough + 1.0) / 8.0;
    s.Gv = s.NdotV / max(s.NdotV * (1.0 - s.K) + s.K, 1e-4);
    s.DiffuseAlbedo = albedo / 3.14159;
    return true;
}

// One light's radiance off the surface (physical, not yet pre-exposed); 0 past its range or behind the surface
float3 ShadePointLight(PointLightData light, PointSurface s)
{
    // Direction from surface to light
    float3 toLight = light.Position - s.Pos;
    float dist = length(toLight);
    if (dist > light.Range)
        return 0;
    float3 L = toLight / dist;

    float rawNdotL = dot(s.N, L);
    float NdotL = max(rawNdotL, 0.0);
    if (NdotL <= 0.0 && !s.Vegetation)
        return 0;

    // Inverse square (Intensity is candela: lux at 1 m), with a smooth window to nothing at Range (Karis 2013:
    // (1 - (d/R)^4)^2) and no closer than 10 cm
    float r4 = dist / light.Range;
    r4 *= r4;
    r4 *= r4;
    float window = saturate(1.0 - r4);
    float attenuation = window * window * PointLightInner(light.InnerRadius, dist * dist) / max(dist * dist, 0.01);

    // Shadow: leaves lit from behind push off their side facing the light
    float shadow = PointLightShadow(light, s.Pos, rawNdotL < 0.0 ? -s.N : s.N, L);
    shadow = min(shadow, PointLightScreenShadow(light, s.Px));
    if (shadow <= 0.0)
        return 0;
    attenuation *= shadow;

    float3 H = normalize(L + s.V);
    float NdotH = max(dot(s.N, H), 0.0);
    float VdotH = max(dot(s.V, H), 0.0);

    // GGX normal distribution, Smith GGX geometry (light component), Schlick Fresnel
    float denom = (NdotH * NdotH) * (s.A2 - 1.0) + 1.0;
    float D = s.A2 / max(3.14159 * denom * denom, 1e-4);
    float Gl = NdotL / max(NdotL * (1.0 - s.K) + s.K, 1e-4);
    float G = s.Gv * Gl;
    float3 F = FresnelSchlick(VdotH, s.F0);
    float3 spec = (D * G) * F / max(4.0 * NdotL * s.NdotV, 1e-4);

    // Diffuse (energy-conserving Lambert)
    float3 kd = (1.0 - F) * (1.0 - s.Metal);
    float3 diffuse = kd * s.DiffuseAlbedo;

    if (s.Vegetation)
    {
        float wrapNdotL = rawNdotL * 0.5 + 0.5;
        float stdNdotL = max(rawNdotL, 0.0);
        float sunFacing = saturate(rawNdotL * 2.0);
        float vegNdotL = lerp(stdNdotL, wrapNdotL, sunFacing);
        return diffuse * (light.Color * light.Intensity * vegNdotL * attenuation) * s.AO;
    }
    return (diffuse + spec) * (light.Color * light.Intensity * NdotL * attenuation) * s.AO;
}

// ── On the light clusters (light_clusters.hlsli) ─────────────────────────────
// Each 8×8 group lies in one 32-pixel cluster tile. It takes the view depths its pixels span, the sorted lights the
// z-bins of those depths hold and the tile's mask lets through, and keeps (as a bitmask over the sorted list, no cap)
// those whose sphere its own frustum and depth range see: a light behind every pixel of the group, or in front of all
// of them, is left out (what the Hi-Z occlusion test did per light, now per group). Every pixel walks the same bits.
groupshared uint g_MinDepth;
groupshared uint g_MaxDepth;
groupshared uint g_First;
groupshared uint g_Last;
groupshared uint g_GroupLights[LIGHT_CLUSTER_WORDS];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void CSPointLight(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uint3 groupId          : SV_GroupID,
    uint  groupIndex       : SV_GroupIndex)
{
    uint2 px = dispatchThreadId.xy;
    float2 screenSize = float2(ScreenWidthIdx, ScreenHeightIdx);
    float3 forward = float3(CameraRelativeVP._14, CameraRelativeVP._24, CameraRelativeVP._34);

    PointSurface s;
    bool lit = px.x < ScreenWidthIdx && px.y < ScreenHeightIdx && LoadPointSurface(px, screenSize, s);
    float viewDepth = lit ? dot(s.Pos, forward) + CameraRelativeVP._44 : 0.0;

    if (groupIndex == 0)
    {
        g_MinDepth = 0x7F7FFFFFu;
        g_MaxDepth = 0u;
        g_First = 0xFFFFu;
        g_Last = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    // Positive floats sort as their bits do
    if (lit)
    {
        InterlockedMin(g_MinDepth, asuint(max(viewDepth, 0.0)));
        InterlockedMax(g_MaxDepth, asuint(max(viewDepth, 0.0)));
    }
    GroupMemoryBarrierWithGroupSync();

    // From here on every return is the whole group's: the values decided on are in group shared memory
    if (g_MaxDepth == 0u || LightClustersIdx == 0 || ClusterLightsIdx == 0)
        return;
    ByteAddressBuffer clusters = ResourceDescriptorHeap[LightClustersIdx];
    if (clusters.Load(LC_COUNT * 4) == 0)
        return;
    float minDepth = asfloat(g_MinDepth), maxDepth = asfloat(g_MaxDepth);

    // The first and last sorted light of any z-bin the group's depths span
    uint b1 = LightZBin(maxDepth);
    for (uint b = LightZBin(minDepth) + groupIndex; b <= b1; b += TILE_SIZE * TILE_SIZE)
    {
        uint range = clusters.Load((LC_BINS + b) * 4);
        uint first = range & 0xFFFFu, last = range >> 16;
        if (first <= last)
        {
            InterlockedMin(g_First, first);
            InterlockedMax(g_Last, last);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint firstLight = g_First, lastLight = g_Last;
    if (firstLight > lastLight)
        return;
    uint w0 = firstLight >> 5, w1 = lastLight >> 5;

    // The group's cluster tile (by the group's centre: the cluster grid was sized to the view, as this pass is)
    uint2 size = uint2(clusters.Load(LC_WIDTH * 4), clusters.Load(LC_HEIGHT * 4));
    float2 centre = (float2(groupId.xy * TILE_SIZE) + 0.5 * TILE_SIZE) / screenSize;
    uint2 tile = min(uint2(centre * float2(size)), size - 1u) >> LIGHT_CLUSTER_TILE_SHIFT;
    uint maskBase = (LC_MASKS + (tile.y * clusters.Load(LC_TILES_X * 4) + tile.x) * LIGHT_CLUSTER_WORDS) * 4;

    float4 planes[4];
    BuildTileFrustumPlanes(groupId.xy, screenSize, CameraInverse, planes);
    StructuredBuffer<PointLightData> lights = ResourceDescriptorHeap[ClusterLightsIdx];
    for (uint w = w0 + groupIndex; w <= w1; w += TILE_SIZE * TILE_SIZE)
    {
        uint bits = clusters.Load(maskBase + w * 4);
        if (w == w0)
            bits &= 0xFFFFFFFFu << (firstLight & 31u);
        if (w == w1)
            bits &= LightClusterLastMask(lastLight);
        uint kept = 0;
        while (bits != 0)
        {
            uint bit = firstbitlow(bits);
            bits &= bits - 1u;
            PointLightData light = lights[w * 32u + bit];
            float z = dot(light.Position, forward) + CameraRelativeVP._44;
            if (z + light.Range < minDepth || z - light.Range > maxDepth)
                continue;
            if (SphereFrustumTest(light.Position, light.Range, planes))
                kept |= 1u << bit;
        }
        g_GroupLights[w] = kept;
    }
    GroupMemoryBarrierWithGroupSync();

    if (!lit)
        return;
    float3 totalLighting = 0;
    for (uint w2 = w0; w2 <= w1; w2++)
    {
        uint bits = g_GroupLights[w2];
        while (bits != 0)
        {
            uint bit = firstbitlow(bits);
            bits &= bits - 1u;
            totalLighting += ShadePointLight(lights[w2 * 32u + bit], s);
        }
    }

    // Additive — directional light already wrote to this UAV
    if (any(totalLighting > 0))
    {
        RWTexture2D<float4> Output = ResourceDescriptorHeap[OutputUAVIdx];
        Output[px] += float4(totalLighting * Exposure, 0);   // radiance (cd/m²), pre-exposed
    }
}
