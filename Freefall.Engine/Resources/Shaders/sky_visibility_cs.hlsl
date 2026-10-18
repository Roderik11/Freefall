// Sky visibility of the opaque surfaces (ATMOSPHERE_PLAN stage 4): how much of the sky's light reaches each pixel's
// surface past the terrain (sky_visibility.hlsli) and the meshes round the camera (sky_capture.hlsli), 0..1, into an
// R8 texture. light_directional_cs reads it for the shadow wrap and hands it on to the composition (ambient, aerial
// perspective) in the light buffer's alpha. A pass of its own: inside the directional light's shader the 16 capture
// views' loads and the unrolled directions cost 0.6-0.8 ms at 2148×1249; here the views sit in group shared memory.
// Docs: .agent/knowledge/atmospheric_effects/physical_atmosphere.md §10-11.

#pragma kernel CSSkyVisibility

cbuffer PushConstants : register(b3)
{
    uint NormalTexIdx;
    uint DepthTexIdx;       // hardware depth (reverse Z: 0 = sky)
    uint OutputUAVIdx;      // R8_UNORM
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

    RWTexture2D<unorm float> Output = ResourceDescriptorHeap[OutputUAVIdx];
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

    float meshVisibility = 1.0;
    if (meshes)
    {
        Texture2DArray<float> depths = ResourceDescriptorHeap[depthsIdx];
        float width, height, slices;
        depths.GetDimensions(width, height, slices);
        float lit = 0, total = 0;
        [unroll] for (uint i = 0; i < SKY_CAPTURE_VIEWS; i++)
            lit += SkyCaptureViewLight(gViews[i], i, depths, float2(width, height), AtmosphereLinearClamp, worldPos, normal, total);
        meshVisibility = total > 0.0 ? lit / total : 1.0;
    }
    Output[px] = SkyVisibilityWith(worldPos, normal, meshVisibility);
}
