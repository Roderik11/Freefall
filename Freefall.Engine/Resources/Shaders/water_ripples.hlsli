// The ripples round the camera on a water surface (Graphics/WaterRipples.cs): their height and slope at a place, from
// the height field the ripple step keeps (water_ripples_cs.hlsl). water.fx and ocean.fx add the slope to their own.
// Docs: .agent/knowledge/ocean_system/underwater.md "Ripples".
#ifndef WATER_RIPPLES_HLSLI
#define WATER_RIPPLES_HLSLI

#define RIPPLE_SIZE 512.0
#define RIPPLE_SLOPE 1.0        // the slope as the height field has it, read as a ripple's

// srv: WaterRipples.Srv (0: still water); origin: the grid's corner (world xz); span: metres it covers.
// Returns the slope (rise per metre over x and z); height: the ripple's height there (m)
float2 RippleSlope(uint srv, float2 origin, float span, float2 xz, out float height)
{
    height = 0.0;
    if (srv == 0)
        return 0;
    float2 uv = (xz - origin) / span;
    if (any(uv <= 0.0) || any(uv >= 1.0))
        return 0;
    Texture2D<float> ripples = ResourceDescriptorHeap[srv];
    float e = 1.0 / RIPPLE_SIZE;
    height = ripples.SampleLevel(AtmosphereLinearClamp, uv, 0);
    float hx = ripples.SampleLevel(AtmosphereLinearClamp, uv + float2(e, 0), 0) - ripples.SampleLevel(AtmosphereLinearClamp, uv - float2(e, 0), 0);
    float hz = ripples.SampleLevel(AtmosphereLinearClamp, uv + float2(0, e), 0) - ripples.SampleLevel(AtmosphereLinearClamp, uv - float2(0, e), 0);
    return float2(hx, hz) * (RIPPLE_SIZE / (2.0 * span)) * RIPPLE_SLOPE;
}

#endif
