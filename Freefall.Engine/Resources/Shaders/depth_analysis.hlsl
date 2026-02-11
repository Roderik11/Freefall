// SDSM Depth Analysis
// Pass 1: CSDepthReduce - parallel min/max reduction of linear depth buffer
// Pass 2: CSDepthHistogram - histogram + percentile-based cascade split computation
//
// Input: DepthGBuffer (R32_Float, linear view-space depth, 0 = sky/empty)
// Output: 4 cascade split distances in a small UAV buffer

cbuffer PushConstants : register(b3)
{
    uint4 Indices[8];
};

#define DepthTexIdx     Indices[0].x   // SRV: DepthGBuffer (R32_Float)
#define MinMaxUAVIdx    Indices[0].y   // UAV: RWByteAddressBuffer [minAsUint, maxAsUint]
#define HistogramUAVIdx Indices[0].z   // UAV: RWStructuredBuffer<uint> [256 bins]
#define SplitsUAVIdx    Indices[0].w   // UAV: RWStructuredBuffer<float> [4 splits]
#define TexWidth        Indices[1].x   // Depth texture width
#define TexHeight       Indices[1].y   // Depth texture height
#define NearPlane       asfloat(Indices[1].z)  // Camera near plane
#define FarPlane        asfloat(Indices[1].w)  // Max shadow distance (fallback far)

// ============================================================
// Pass 1: CSDepthReduce — find min/max depth across entire buffer
// ============================================================
groupshared uint gs_min;
groupshared uint gs_max;

[numthreads(16, 16, 1)]
void CSDepthReduce(uint3 id : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    // Initialize groupshared atomics
    if (groupIndex == 0)
    {
        gs_min = 0x7F7FFFFF; // Max float as uint
        gs_max = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (id.x < TexWidth && id.y < TexHeight)
    {
        Texture2D<float> depthTex = ResourceDescriptorHeap[DepthTexIdx];
        float depth = depthTex[id.xy];
        
        // Skip sky/empty pixels (depth <= 0)
        if (depth > 0.0f)
        {
            uint depthAsUint = asuint(depth);
            InterlockedMin(gs_min, depthAsUint);
            InterlockedMax(gs_max, depthAsUint);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    
    // One thread per group writes to global buffer
    if (groupIndex == 0)
    {
        RWByteAddressBuffer minMaxBuf = ResourceDescriptorHeap[MinMaxUAVIdx];
        // InterlockedMin/Max on raw buffer (byte address)
        minMaxBuf.InterlockedMin(0, gs_min);
        minMaxBuf.InterlockedMax(4, gs_max);
    }
}

// ============================================================
// Pass 2: CSDepthHistogram — bin depths + compute percentile splits
// ============================================================
#define HISTOGRAM_BINS 256

groupshared uint gs_histogram[HISTOGRAM_BINS];

[numthreads(16, 16, 1)]
void CSDepthHistogram(uint3 id : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    // Clear local histogram
    if (groupIndex < HISTOGRAM_BINS)
        gs_histogram[groupIndex] = 0;
    GroupMemoryBarrierWithGroupSync();
    
    // Read min/max from reduce pass
    RWByteAddressBuffer minMaxBuf = ResourceDescriptorHeap[MinMaxUAVIdx];
    float minDepth = asfloat(minMaxBuf.Load(0));
    float maxDepth = asfloat(minMaxBuf.Load(4));
    
    float range = maxDepth - minDepth;
    
    if (id.x < TexWidth && id.y < TexHeight && range > 0.0001f)
    {
        Texture2D<float> depthTex = ResourceDescriptorHeap[DepthTexIdx];
        float depth = depthTex[id.xy];
        
        if (depth > 0.0f)
        {
            float normalized = saturate((depth - minDepth) / range);
            uint bin = min((uint)(normalized * (HISTOGRAM_BINS - 1)), HISTOGRAM_BINS - 1);
            InterlockedAdd(gs_histogram[bin], 1);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    
    // Write local histogram to global
    if (groupIndex < HISTOGRAM_BINS)
    {
        RWStructuredBuffer<uint> globalHist = ResourceDescriptorHeap[HistogramUAVIdx];
        InterlockedAdd(globalHist[groupIndex], gs_histogram[groupIndex]);
    }
}

// ============================================================
// Pass 3: CSComputeSplits — prefix sum histogram + percentile splits
// Single thread group (1,1,1)
// ============================================================
groupshared uint gs_prefixSum[HISTOGRAM_BINS];

[numthreads(1, 1, 1)]
void CSComputeSplits(uint3 id : SV_DispatchThreadID)
{
    RWByteAddressBuffer minMaxBuf = ResourceDescriptorHeap[MinMaxUAVIdx];
    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramUAVIdx];
    RWStructuredBuffer<float> splits = ResourceDescriptorHeap[SplitsUAVIdx];
    
    float minDepth = asfloat(minMaxBuf.Load(0));
    float maxDepth = asfloat(minMaxBuf.Load(4));
    float range = maxDepth - minDepth;
    
    // Handle edge cases
    if (range <= 0.0001f || minDepth <= 0.0f)
    {
        // Fallback: can't compute meaningful splits
        // Write zeros to signal "no valid data"
        splits[0] = 0;
        splits[1] = 0;
        splits[2] = 0;
        splits[3] = 0;
        return;
    }
    
    // Compute total pixel count and prefix sum
    uint total = 0;
    for (uint i = 0; i < HISTOGRAM_BINS; i++)
    {
        total += histogram[i];
        gs_prefixSum[i] = total;
    }
    
    if (total == 0)
    {
        splits[0] = 0;
        splits[1] = 0;
        splits[2] = 0;
        splits[3] = 0;
        return;
    }
    
    // Find percentile boundaries: 25%, 50%, 75%, 100%
    // Each cascade handles an equal fraction of visible pixels
    float percentiles[4] = { 0.25f, 0.50f, 0.75f, 1.0f };
    
    for (uint c = 0; c < 4; c++)
    {
        uint target = (uint)(percentiles[c] * total);
        uint bin = HISTOGRAM_BINS - 1; // default to last bin
        
        for (uint b = 0; b < HISTOGRAM_BINS; b++)
        {
            if (gs_prefixSum[b] >= target)
            {
                bin = b;
                break;
            }
        }
        
        // Convert bin back to depth
        float depth = minDepth + (float(bin) / float(HISTOGRAM_BINS - 1)) * range;
        
        // Clamp: ensure cascade far >= near plane, and monotonically increasing
        depth = max(depth, NearPlane + 0.1f);
        splits[c] = depth;
    }
    
    // Ensure last cascade covers everything
    splits[3] = max(splits[3], maxDepth);
    
    // Ensure monotonically increasing
    for (uint s = 1; s < 4; s++)
    {
        if (splits[s] <= splits[s - 1])
            splits[s] = splits[s - 1] + 1.0f;
    }
}
