// Sky visibility of the opaque surfaces (ATMOSPHERE_PLAN stage 4): how much of the sky's light reaches each pixel's
// surface past the terrain (sky_visibility.hlsli) and the meshes round the camera (sky_capture.hlsli), 0..1, into the
// red of an R8G8 texture; green: how open the sky toward the directional light is past the meshes (the clouds' forward
// light). light_directional_cs reads it for the shadow wrap and hands red on to the composition (ambient, aerial
// perspective) in the light buffer's alpha. A pass of its own: inside the directional light's shader the 16 capture
// views' loads and the unrolled directions cost 0.6-0.8 ms at 2148×1249; here the views sit in group shared memory.
// Docs: .agent/knowledge/atmospheric_effects/physical_atmosphere.md §10-11.

#pragma kernel CSSkyVisibility

cbuffer PushConstants : register(b3)
{
    uint NormalTexIdx;
    uint DepthTexIdx;       // hardware depth (reverse Z: 0 = sky)
    uint OutputUAVIdx;      // R8G8_UNORM
    uint ScreenWidthIdx;
    uint ScreenHeightIdx;
};

#include "common.fx"
#include "sky_common.fx"

groupshared SkyCaptureView gViews[SKY_CAPTURE_VIEWS];

[numthreads(8, 8, 1)]
void CSSkyVisibility(uint3 dispatchThreadId : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    // The capture views, once per group
    bool meshes = false;
    uint depthsIdx = 0;
    if (AtmosphereIdx != 0)
    {
        AtmosphereParams A = LoadAtmosphere();
        meshes = A.SkyCaptureViews != 0 && A.SkyCaptureDepth != 0 && A.SkyCaptureCount == SKY_CAPTURE_VIEWS;
        depthsIdx = A.SkyCaptureDepth;
        if (meshes && groupIndex < SKY_CAPTURE_VIEWS)
        {
            StructuredBuffer<SkyCaptureView> views = ResourceDescriptorHeap[A.SkyCaptureViews];
            gViews[groupIndex] = views[groupIndex];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint2 px = dispatchThreadId.xy;
    if (px.x >= ScreenWidthIdx || px.y >= ScreenHeightIdx)
        return;

    RWTexture2D<unorm float2> Output = ResourceDescriptorHeap[OutputUAVIdx];
    Texture2D DepthTex = ResourceDescriptorHeap[DepthTexIdx];
    float depth = DepthTex.Load(int3(px, 0)).r;
    if (depth <= 0.0)
    {
        Output[px] = 1.0;   // the sky dome
        return;
    }
    Texture2D NormalTex = ResourceDescriptorHeap[NormalTexIdx];
    float3 normal = NormalTex.Load(int3(px, 0)).xyz;
    float2 texCoord = (float2(px) + 0.5) / float2(ScreenWidthIdx, ScreenHeightIdx);
    float3 worldPos = posFromDepth(texCoord, depth, CameraInverse).xyz + CamPos;

    // y: how open the sky is toward the directional light's side past the meshes (the directions nearest it): the soft
    // shadow of the light the clouds throw on forward round the sun (light_directional_cs.hlsl). The light is the sun
    // while it is up, then the brightest moon.
    float meshVisibility = 1.0, lightSide = 1.0;
    if (meshes)
    {
        AtmosphereParams A = LoadAtmosphere();
        float3 toLight = A.SunDirection.y > 0.0 ? A.SunDirection : A.MoonDirection;
        Texture2DArray<float> depths = ResourceDescriptorHeap[depthsIdx];
        float width, height, slices;
        depths.GetDimensions(width, height, slices);
        float lit = 0, total = 0, sideLit = 0, sideTotal = 0;
        [unroll] for (uint i = 0; i < SKY_CAPTURE_VIEWS; i++)
        {
            float open = SkyCaptureViewOpen(gViews[i], i, depths, float2(width, height), AtmosphereLinearClamp, worldPos, normal);
            float w = SkyCaptureViewWeight(gViews[i], normal);
            lit += w * open;
            total += w;
            float ws = SkyCaptureViewLightWeight(gViews[i], toLight);
            sideLit += ws * open;
            sideTotal += ws;
        }
        meshVisibility = total > 0.0 ? lit / total : 1.0;
        lightSide = sideTotal > 1e-4 ? sideLit / sideTotal : meshVisibility;
    }
    Output[px] = float2(SkyVisibilityWith(worldPos, normal, meshVisibility), lightSide);
}
