// Auto exposure metering (Graphics/AutoExposure.cs): a histogram of the scene's luminance in cd/m², from the HDR
// composite (deferred + forward + sky) divided by the exposure it was drawn with. 64 bins of log2(luminance) over
// [HISTOGRAM_MIN_LOG2, HISTOGRAM_MAX_LOG2); weighted toward the centre of the screen. Read back by the CPU, which takes
// the mean between two percentiles and adapts the exposure of the frames that follow.

#pragma kernel CSClear
#pragma kernel CSHistogram

cbuffer PushConstants : register(b3)
{
    uint InputSrvIdx;           // HDR composite (pre-exposed)
    uint HistogramUAVIdx;       // RWStructuredBuffer<uint>[64]
    uint Width;
    uint Height;
    uint InvExposureBits;       // 1 / the exposure the composite was drawn with (float bits)
};

#define BINS 64
#define HISTOGRAM_MIN_LOG2 -14.0     // ~6e-5 cd/m²: below a moonless night's ground
#define HISTOGRAM_MAX_LOG2 18.0      // ~2.6e5 cd/m²: above sunlit snow; the sun disc lands in the last bin

groupshared uint LocalBins[BINS];

[numthreads(BINS, 1, 1)]
void CSClear(uint3 id : SV_DispatchThreadID)
{
    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramUAVIdx];
    histogram[id.x] = 0;
}

// Every other pixel each way (a quarter of them): plenty for a 64-bin histogram
[numthreads(16, 16, 1)]
void CSHistogram(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    if (gi < BINS)
        LocalBins[gi] = 0;
    GroupMemoryBarrierWithGroupSync();

    uint2 px = id.xy * 2;
    if (px.x < Width && px.y < Height)
    {
        Texture2D<float4> input = ResourceDescriptorHeap[InputSrvIdx];
        float3 c = max(input.Load(int3(px, 0)).rgb, 0.0);
        float luminance = dot(c, float3(0.2126, 0.7152, 0.0722)) * asfloat(InvExposureBits);

        float t = (log2(max(luminance, 1e-10)) - HISTOGRAM_MIN_LOG2) / (HISTOGRAM_MAX_LOG2 - HISTOGRAM_MIN_LOG2);
        uint bin = (uint)clamp(t * BINS, 0.0, BINS - 1.0);

        // Centre-weighted: 4 in the middle down to 1 at the corners, so the sky at the edges of a view does not
        // swing the exposure as the camera turns
        float2 uv = (float2(px) + 0.5) / float2(Width, Height) - 0.5;
        uint weight = 1 + (uint)(3.0 * saturate(1.0 - dot(uv, uv) * 2.0));
        InterlockedAdd(LocalBins[bin], weight);
    }
    GroupMemoryBarrierWithGroupSync();

    if (gi < BINS && LocalBins[gi] > 0)
    {
        RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramUAVIdx];
        InterlockedAdd(histogram[gi], LocalBins[gi]);
    }
}
