// Point Light Compute Shader — tiled culling + PBR lighting
// [numthreads(8, 8, 1)] = 64 threads per tile
//
// Phase 1: Tile builds 4 frustum planes from its screen-space bounds.
//          All 64 threads cooperatively test lights against the tile frustum,
//          building a compact per-tile light list in groupshared memory.
// Phase 2: Each thread loads its pixel's G-buffer and loops only over
//          the tile's accepted lights.

#pragma kernel CSPointLight

cbuffer PushConstants : register(b3)
{
    uint NormalTexIdx;
    uint DepthTexIdx;
    uint AlbedoTexIdx;
    uint DataTexIdx;
    uint OutputUAVIdx;
    uint LightDataIdx;      // StructuredBuffer<PointLightData>
    uint LightCountIdx;     // raw uint — number of active lights
    uint ScreenWidthIdx;
    uint ScreenHeightIdx;
};

#include "common.fx"

struct PointLightData
{
    float3 Color;
    float Intensity;
    float3 Position; // camera-relative
    float Range;
};

// PBR helpers
float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// ── Tile culling shared memory ────────────────────────────────────────────
#define TILE_SIZE 8
#define MAX_LIGHTS_PER_TILE 64

groupshared uint g_TileLightCount;
groupshared uint g_TileLightIndices[MAX_LIGHTS_PER_TILE];

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

// ── Main kernel ───────────────────────────────────────────────────────────
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void CSPointLight(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uint3 groupId          : SV_GroupID,
    uint  groupIndex       : SV_GroupIndex)
{
    uint2 px = dispatchThreadId.xy;
    float2 screenSize = float2(ScreenWidthIdx, ScreenHeightIdx);
    uint lightCount = LightCountIdx;

    // ── Phase 1: Tile-level light culling ─────────────────────────────────
    if (groupIndex == 0)
        g_TileLightCount = 0;

    GroupMemoryBarrierWithGroupSync();

    if (lightCount > 0)
    {
        // Build tile frustum planes (all threads compute the same result,
        // but it's cheaper than branching + LDS broadcast for 4 planes)
        // Planes are in VIEW SPACE — light positions need to be transformed.
        // Actually, CameraInverse maps NDC→camera-relative world space.
        // Our light positions are already camera-relative world space.
        // So we build planes in camera-relative world space using CameraInverse.
        float4 tilePlanes[4];
        BuildTileFrustumPlanes(groupId.xy, screenSize, CameraInverse, tilePlanes);

        // Each thread tests a subset of lights (round-robin)
        StructuredBuffer<PointLightData> lightData = ResourceDescriptorHeap[LightDataIdx];

        for (uint li = groupIndex; li < lightCount; li += (TILE_SIZE * TILE_SIZE))
        {
            PointLightData light = lightData[li];

            if (SphereFrustumTest(light.Position, light.Range, tilePlanes))
            {
                uint slot;
                InterlockedAdd(g_TileLightCount, 1, slot);
                if (slot < MAX_LIGHTS_PER_TILE)
                    g_TileLightIndices[slot] = li;
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // ── Phase 2: Per-pixel PBR lighting ───────────────────────────────────
    if (px.x >= ScreenWidthIdx || px.y >= ScreenHeightIdx)
        return;

    uint tileLightCount = min(g_TileLightCount, MAX_LIGHTS_PER_TILE);
    if (tileLightCount == 0)
        return;

    // Load G-Buffer
    Texture2D NormalTex = ResourceDescriptorHeap[NormalTexIdx];
    Texture2D DepthTex  = ResourceDescriptorHeap[DepthTexIdx];

    int3 coord = int3(px, 0);
    float3 normal = NormalTex.Load(coord).xyz;
    float depth = DepthTex.Load(coord).r;

    // Skip sky pixels — reverse depth: far=0
    if (depth <= 0.0f)
        return;

    // Reconstruct world position from depth (camera-relative, zero-translation)
    float2 texCoord = (float2(px) + 0.5) / screenSize;
    float4 worldPos = posFromDepth(texCoord, depth, CameraInverse);

    // Load material data
    Texture2D AlbedoTex = ResourceDescriptorHeap[AlbedoTexIdx];
    Texture2D DataTex   = ResourceDescriptorHeap[DataTexIdx];

    float3 albedo   = AlbedoTex.Load(coord).rgb;
    float4 dataFull = DataTex.Load(coord);
    bool isVegetation = (dataFull.a > 0.3 && dataFull.a < 0.7);

    float rough = max(dataFull.r, 0.04);
    float metal = saturate(dataFull.g);
    float ao    = saturate(dataFull.b);

    // View vector (worldPos is camera-relative, camera at origin)
    float3 V = normalize(-worldPos.xyz);
    float NdotV = max(dot(normal, V), 0.001);

    // Metallic F0
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metal);

    float a  = rough * rough;
    float a2 = a * a;

    // Smith GGX geometry term (view component, shared across lights)
    float k = (rough + 1.0);
    k = (k * k) / 8.0;
    float Gv = NdotV / max(NdotV * (1.0 - k) + k, 1e-4);

    // Diffuse coefficient
    float3 diffuseAlbedo = albedo / 3.14159;

    // Light buffer (re-fetch for phase 2)
    StructuredBuffer<PointLightData> lightData = ResourceDescriptorHeap[LightDataIdx];

    // Accumulate lighting from tile's lights only
    float3 totalLighting = 0;

    for (uint ti = 0; ti < tileLightCount; ti++)
    {
        PointLightData light = lightData[g_TileLightIndices[ti]];

        // Direction from surface to light
        float3 toLight = light.Position - worldPos.xyz;
        float dist = length(toLight);

        // Per-pixel range check (tile test is conservative)
        if (dist > light.Range)
            continue;

        float3 L = toLight / dist;

        // N·L
        float rawNdotL = dot(normal, L);
        float NdotL = max(rawNdotL, 0.0);

        if (NdotL <= 0.0 && !isVegetation)
            continue;

        // Attenuation: smooth quadratic falloff
        float attenuation = saturate(1.0 - (dist / light.Range));
        attenuation *= attenuation;

        // Half vector
        float3 H = normalize(L + V);
        float NdotH = max(dot(normal, H), 0.0);
        float VdotH = max(dot(V, H), 0.0);

        // GGX Normal Distribution Function
        float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
        float D = a2 / max(3.14159 * denom * denom, 1e-4);

        // Smith GGX Geometry (light component)
        float Gl = NdotL / max(NdotL * (1.0 - k) + k, 1e-4);
        float G = Gv * Gl;

        // Fresnel (Schlick)
        float3 F = FresnelSchlick(VdotH, F0);

        // Specular BRDF
        float3 spec = (D * G) * F / max(4.0 * NdotL * NdotV, 1e-4);

        // Diffuse BRDF (energy-conserving Lambert)
        float3 kd = (1.0 - F) * (1.0 - metal);
        float3 diffuse = kd * diffuseAlbedo;

        // Final lighting
        if (isVegetation)
        {
            float wrapNdotL = rawNdotL * 0.5 + 0.5;
            float stdNdotL = max(rawNdotL, 0.0);
            float sunFacing = saturate(rawNdotL * 2.0);
            float vegNdotL = lerp(stdNdotL, wrapNdotL, sunFacing);
            float3 radiance = light.Color * light.Intensity * vegNdotL * attenuation;
            totalLighting += diffuse * radiance * ao;
        }
        else
        {
            float3 radiance = light.Color * light.Intensity * NdotL * attenuation;
            totalLighting += (diffuse + spec) * radiance * ao;
        }
    }

    // Additive — directional light already wrote to this UAV
    if (any(totalLighting > 0))
    {
        RWTexture2D<float4> Output = ResourceDescriptorHeap[OutputUAVIdx];
        Output[px] += float4(totalLighting, 0);
    }
}
