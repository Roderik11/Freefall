// What the water's screen-space reflections read (Graphics/WaterReflections.cs, water.fx WaterScreenReflection), made
// once a frame after the composite is snapshot and before the forward pass draws the water:
//
// - the nearest-depth pyramid: half size and down, each texel the nearest of the pixels it covers as 1 / view depth
//   (0 = the sky). 1 / depth is affine on the screen, so a ray's value along its line on the screen is a straight line
//   and meets a cell's nearest depth at one place: the trace skips whole cells the ray passes in front of, and finds the
//   first pixel it goes behind without stepping over thin things (a post, a branch) as fixed steps do;
// - the scene's colour pyramid: the composite snapshot averaged to half size and down, read trilinearly by the blur of
//   a rippled reflection (a footprint of up to ~100 pixels in a few taps).
//
// Every mip is read through its own SRV while the next is written through its UAV (WaterReflections moves each mip to
// shader-read as soon as it is written). Mip sizes are D3D12's: half the one above, rounded down.

#pragma kernel CSNearestFirst
#pragma kernel CSNearest
#pragma kernel CSAverage

cbuffer PushConstants : register(b3)
{
    uint SourceIdx;     // the full-size linear depth (CSNearestFirst), the composite snapshot (first CSAverage) or the mip above
    uint OutputIdx;     // UAV of the mip written
    uint SourceWidth;   // the source's size
    uint SourceHeight;
    uint OutWidth;      // the mip's size: half the source's, rounded down
    uint OutHeight;
};

// The source texels an output texel covers: 2 × 2, and the last one of a row or column also the odd one left over, so
// every source texel is in some output texel (the trace clamps its loads to a mip's last texel)
void Covered(uint2 id, out uint2 first, out uint2 last)
{
    first = id * 2u;
    last = min(first + 1u, uint2(SourceWidth, SourceHeight) - 1u);
    if (id.x == OutWidth - 1u) last.x = SourceWidth - 1u;
    if (id.y == OutHeight - 1u) last.y = SourceHeight - 1u;
}

// Half size from the linear depth: the nearest (largest 1 / depth) of the pixels covered, the sky left out
[numthreads(8, 8, 1)]
void CSNearestFirst(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= OutWidth || id.y >= OutHeight)
        return;
    Texture2D<float> depth = ResourceDescriptorHeap[SourceIdx];
    RWTexture2D<float> output = ResourceDescriptorHeap[OutputIdx];
    uint2 first, last;
    Covered(id.xy, first, last);
    float nearest = 0.0;
    for (uint y = first.y; y <= last.y; y++)
        for (uint x = first.x; x <= last.x; x++)
        {
            float w = depth.Load(int3(x, y, 0));
            if (w > 0.0)
                nearest = max(nearest, 1.0 / w);
        }
    output[id.xy] = nearest;
}

// The next mip: the nearest of the texels above it covers
[numthreads(8, 8, 1)]
void CSNearest(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= OutWidth || id.y >= OutHeight)
        return;
    Texture2D<float> above = ResourceDescriptorHeap[SourceIdx];
    RWTexture2D<float> output = ResourceDescriptorHeap[OutputIdx];
    uint2 first, last;
    Covered(id.xy, first, last);
    float nearest = 0.0;
    for (uint y = first.y; y <= last.y; y++)
        for (uint x = first.x; x <= last.x; x++)
            nearest = max(nearest, above.Load(int3(x, y, 0)));
    output[id.xy] = nearest;
}

// No pixel brighter than this in what the water mirrors (water.fx WATER_SCENE_MAX): the sky's sun disc (up to 60000 a
// pixel, mesh_skybox.fx) would spread a blob over the blurred reflection; the sun's glint is its reflection
#define SCENE_MAX 32.0

float4 Bounded(float4 c)
{
    return float4(c.rgb * min(1.0, SCENE_MAX / max(max(c.r, max(c.g, c.b)), 1e-6)), c.a);
}

// The colour pyramid: the average of the texels above it covers (the composite's pixels, bounded, for the first mip)
[numthreads(8, 8, 1)]
void CSAverage(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= OutWidth || id.y >= OutHeight)
        return;
    Texture2D<float4> above = ResourceDescriptorHeap[SourceIdx];
    RWTexture2D<float4> output = ResourceDescriptorHeap[OutputIdx];
    uint2 first, last;
    Covered(id.xy, first, last);
    float4 sum = 0.0;
    for (uint y = first.y; y <= last.y; y++)
        for (uint x = first.x; x <= last.x; x++)
            sum += Bounded(above.Load(int3(x, y, 0)));
    uint2 n = last - first + 1u;
    output[id.xy] = sum / float(n.x * n.y);
}
