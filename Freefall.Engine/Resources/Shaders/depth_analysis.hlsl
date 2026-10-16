// SDSM Depth Analysis
// CSDepthBins   - one pass in 4x4 pixel blocks: nearest and farthest depth over every pixel, log-spaced depth
//                 histogram over [NearPlane, MaxShadowDistance] of one pixel per block (1/16)
// CSDepthSplits - prefix sum + 40/70/90/100 percentile splits; hands the histogram and min/max back cleared
//                 for the next frame, so nothing is cleared before CSDepthBins
// Was a min/max pass + a histogram pass over every pixel with one groupshared atomic each (0.37 ms at 3440x1249,
// 0.012 now); the splits match the old ones within ~0.5%.

#pragma kernel CSDepthBins
#pragma kernel CSDepthSplits

//
// Input: DepthGBuffer (R32_Float, linear view-space depth, 0 = sky/empty)
// Output: 4 cascade split distances in a small UAV buffer

cbuffer PushConstants : register(b3)
{
    uint DepthTexIdx;           // slot 0  — SRV: DepthGBuffer (R32_Float)
    uint MinMaxUAVIdx;          // slot 1  — UAV: RWByteAddressBuffer [minAsUint, maxAsUint]
    uint HistogramUAVIdx;       // slot 2  — UAV: RWStructuredBuffer<uint> [DEPTH_BINS bins]
    uint SplitsUAVIdx;          // slot 3  — UAV: RWStructuredBuffer<float> [4 splits]
};

cbuffer Params : register(b4)
{
    uint TexWidth;              // Depth texture width
    uint TexHeight;             // Depth texture height
    float NearPlane;            // Camera near plane
    float MaxShadowDistance;    // Ignore pixels beyond this depth
};

#define DEPTH_BINS 1024         // log-spaced: each bin ~0.8% of its depth (near 0.1 m, MaxShadowDistance 439 m)
#define SAMPLE_STEP 4           // histogram: every 4th pixel each way, 1/16 of them is plenty for percentiles
#define TILE_SAMPLES 32         // blocks per group edge: 16 threads x 2
#define TILE_PIXELS (TILE_SAMPLES * SAMPLE_STEP) // 128: GPUCuller.AnalyzeDepth dispatches ceil(size / 128) groups
#define FLOAT_MAX_BITS 0x7F7FFFFF

SamplerState PointClamp : register(s1); // static sampler 1: point clamp (only Gather uses it)

// Bins are spaced evenly in log2(depth), so every split gets the same relative precision
float LogNear() { return log2(max(NearPlane, 0.001f)); }
float BinsPerLog2() { return DEPTH_BINS / max(log2(MaxShadowDistance) - LogNear(), 0.001f); }

uint DepthToBin(float depth, float logNear, float binsPerLog2)
{
    return min((uint)max((log2(depth) - logNear) * binsPerLog2, 0.0f), DEPTH_BINS - 1);
}

float BinToDepth(float bin, float logNear, float binsPerLog2)
{
    return exp2(logNear + bin / binsPerLog2);
}

// ============================================================
// CSDepthBins — histogram of a 1/16 subsample + min/max of every pixel, one 128x128 pixel tile per group
// ============================================================
groupshared uint gs_bins[DEPTH_BINS];
groupshared uint gs_binsMin;
groupshared uint gs_binsMax;

[numthreads(16, 16, 1)]
void CSDepthBins(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID, uint groupIndex : SV_GroupIndex)
{
    for (uint b = groupIndex; b < DEPTH_BINS; b += 256)
        gs_bins[b] = 0;
    if (groupIndex == 0)
    {
        gs_binsMin = FLOAT_MAX_BITS;
        gs_binsMax = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    Texture2D<float> depthTex = ResourceDescriptorHeap[DepthTexIdx];
    float2 texelSize = 1.0f / float2(TexWidth, TexHeight);
    float logNear = LogNear();
    float binsPerLog2 = BinsPerLog2();
    float nearest = asfloat(FLOAT_MAX_BITS);
    float farthest = 0.0f;

    // 2x2 blocks of 4x4 pixels per thread, interleaved so neighbouring lanes read neighbouring blocks.
    // Min/max see all 16 pixels of a block (the last split is the farthest depth, which a subsample misses
    // on thin far geometry); the histogram takes one pixel per block.
    [unroll]
    for (uint sy = 0; sy < 2; sy++)
    {
        [unroll]
        for (uint sx = 0; sx < 2; sx++)
        {
            uint2 pixel = (groupId.xy * TILE_SAMPLES + threadId.xy + uint2(sx, sy) * 16) * SAMPLE_STEP;
            if (pixel.x >= TexWidth || pixel.y >= TexHeight)
                continue;

            // Four 2x2 quads; .w of the first is the block's top-left pixel
            float2 uv = (float2(pixel) + 1.0f) * texelSize;
            float4 quads[4];
            quads[0] = depthTex.GatherRed(PointClamp, uv);
            quads[1] = depthTex.GatherRed(PointClamp, uv + float2(2.0f, 0.0f) * texelSize);
            quads[2] = depthTex.GatherRed(PointClamp, uv + float2(0.0f, 2.0f) * texelSize);
            quads[3] = depthTex.GatherRed(PointClamp, uv + float2(2.0f, 2.0f) * texelSize);

            // Skip sky/empty pixels (depth <= 0) and distant pixels beyond shadow range
            [unroll]
            for (uint q = 0; q < 4; q++)
            {
                [unroll]
                for (uint c = 0; c < 4; c++)
                {
                    float v = quads[q][c];
                    if (v > 0.0f && v <= MaxShadowDistance)
                    {
                        nearest = min(nearest, v);
                        farthest = max(farthest, v);
                    }
                }
            }

            float depth = quads[0].w;
            if (depth > 0.0f && depth <= MaxShadowDistance)
                InterlockedAdd(gs_bins[DepthToBin(depth, logNear, binsPerLog2)], 1);
        }
    }

    // Min/max: one groupshared atomic per wave
    nearest = WaveActiveMin(nearest);
    farthest = WaveActiveMax(farthest);
    if (WaveIsFirstLane() && farthest > 0.0f)
    {
        InterlockedMin(gs_binsMin, asuint(nearest));
        InterlockedMax(gs_binsMax, asuint(farthest));
    }
    GroupMemoryBarrierWithGroupSync();

    // Merge into the global histogram: only the bins this tile filled
    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramUAVIdx];
    for (uint b2 = groupIndex; b2 < DEPTH_BINS; b2 += 256)
    {
        uint count = gs_bins[b2];
        if (count > 0)
            InterlockedAdd(histogram[b2], count);
    }

    if (groupIndex == 0 && gs_binsMax > 0)
    {
        RWByteAddressBuffer minMaxBuf = ResourceDescriptorHeap[MinMaxUAVIdx];
        minMaxBuf.InterlockedMin(0, gs_binsMin);
        minMaxBuf.InterlockedMax(4, gs_binsMax);
    }
}

// ============================================================
// CSDepthSplits — percentile splits from the histogram, then clear it and min/max for the next frame
// 256 threads, each owns a run of 4 consecutive bins; the thread whose run holds a percentile writes it
// ============================================================
#define BINS_PER_THREAD (DEPTH_BINS / 256)

groupshared uint gs_runSum[256];
groupshared float gs_split[3];

[numthreads(256, 1, 1)]
void CSDepthSplits(uint tid : SV_GroupIndex)
{
    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramUAVIdx];

    uint counts[BINS_PER_THREAD];
    uint runTotal = 0;
    [unroll]
    for (uint k = 0; k < BINS_PER_THREAD; k++)
    {
        counts[k] = histogram[tid * BINS_PER_THREAD + k];
        histogram[tid * BINS_PER_THREAD + k] = 0;
        runTotal += counts[k];
    }

    // Hillis-Steele inclusive prefix sum over the 256 runs
    gs_runSum[tid] = runTotal;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 1; s < 256; s <<= 1)
    {
        uint val = (tid >= s) ? gs_runSum[tid - s] : 0;
        GroupMemoryBarrierWithGroupSync();
        gs_runSum[tid] += val;
        GroupMemoryBarrierWithGroupSync();
    }

    uint total = gs_runSum[255];
    uint before = gs_runSum[tid] - runTotal;
    float logNear = LogNear();
    float binsPerLog2 = BinsPerLog2();

    // Near-biased percentile boundaries; the 100th is the farthest depth (below)
    const float percentiles[3] = { 0.40f, 0.70f, 0.90f };
    [unroll]
    for (uint c = 0; c < 3; c++)
    {
        // The first bin whose inclusive prefix sum reaches the target
        uint target = max((uint)(percentiles[c] * total), 1u);
        if (target <= before || target > before + runTotal)
            continue;

        uint run = before;
        bool found = false;
        [unroll]
        for (uint k2 = 0; k2 < BINS_PER_THREAD; k2++)
        {
            if (!found && run + counts[k2] >= target)
            {
                found = true;
                // Sub-bin interpolation
                float fraction = float(target - run) / float(counts[k2]);
                gs_split[c] = BinToDepth(float(tid * BINS_PER_THREAD + k2) + fraction, logNear, binsPerLog2);
            }
            run += counts[k2];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0)
    {
        RWByteAddressBuffer minMaxBuf = ResourceDescriptorHeap[MinMaxUAVIdx];
        RWStructuredBuffer<float> splits = ResourceDescriptorHeap[SplitsUAVIdx];

        float minDepth = asfloat(minMaxBuf.Load(0));
        float maxDepth = asfloat(minMaxBuf.Load(4));
        minMaxBuf.Store2(0, uint2(FLOAT_MAX_BITS, 0));

        // No valid depth data: zeros, the light falls back to its fixed splits
        if (total == 0 || maxDepth - minDepth <= 0.0001f)
        {
            splits[0] = 0; splits[1] = 0; splits[2] = 0; splits[3] = 0;
            return;
        }

        float result[4];
        [unroll]
        for (uint c2 = 0; c2 < 3; c2++)
            result[c2] = max(gs_split[c2], NearPlane + 0.1f);
        // Last cascade covers everything
        result[3] = max(maxDepth, NearPlane + 0.1f);

        // Ensure monotonically increasing
        [unroll]
        for (uint q = 1; q < 4; q++)
        {
            if (result[q] <= result[q - 1])
                result[q] = result[q - 1] + 1.0f;
        }

        [unroll]
        for (uint w = 0; w < 4; w++)
            splits[w] = result[w];
    }
}
