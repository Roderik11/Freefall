// SDSM Depth Analysis
// CSCasterReach - per visible instance of a shadow-casting batch (the camera culling's bounds and flags): the view
//                 depth its shadow can reach, max-merged. The analysis looks only that far: the farthest receiver a
//                 caster in view can shade. Pixels past it (open land, the sea, far hills, whose shadows the terrain
//                 shadow map gives) pushed the percentile splits out, so a fixed 300 m cap stood in until 2026-10-18.
// CSDepthBins   - one pass in 4x4 pixel blocks: nearest and farthest depth over every pixel, log-spaced depth
//                 histogram over [NearPlane, range] of one pixel per block (1/16)
// CSDepthSplits - prefix sum + 40/70/90/100 percentile splits, each no farther than the logarithmic split between the
//                 nearest and the farthest pixel; hands the histogram, min/max and the reach back cleared for the next
//                 frame, so nothing is cleared before CSCasterReach / CSDepthBins
// Was a min/max pass + a histogram pass over every pixel with one groupshared atomic each (0.37 ms at 3440x1249,
// 0.012 now); the splits match the old ones within ~0.5%.

#pragma kernel CSCasterReach
#pragma kernel CSDepthBins
#pragma kernel CSDepthSplits

//
// Input: DepthGBuffer (R32_Float, linear view-space depth, 0 = sky/empty)
// Output: 4 cascade split distances in a small UAV buffer

// Everything is pushed: a Params cbuffer has one copy per frame, and each view (scene, game) analyses its own camera
cbuffer PushConstants : register(b3)
{
    uint DepthTexIdx;           // slot 0  — SRV: DepthGBuffer (R32_Float)
    uint MinMaxUAVIdx;          // slot 1  — UAV: RWByteAddressBuffer [minAsUint, maxAsUint, casterReachAsUint, 0]
    uint HistogramUAVIdx;       // slot 2  — UAV: RWStructuredBuffer<uint> [DEPTH_BINS bins]
    uint SplitsUAVIdx;          // slot 3  — UAV: RWStructuredBuffer<float> [4 splits]
    uint CasterBoundsIdx;       // slot 4  — UAV: RWStructuredBuffer<InstanceBounds>, a batch's camera culling (CSCasterReach)
    uint CasterFlagsIdx;        // slot 5  — UAV: RWStructuredBuffer<uint>, its visibility flags (0 = not seen)
    uint CasterCountIdx;        // slot 6  — its instances
    uint TexWidth;              // slot 7  — depth texture width
    uint TexHeight;             // slot 8  — depth texture height
    float NearPlane;            // slot 9  — camera near plane
    float MaxShadowDistance;    // slot 10 — the ceiling: never analyse farther (EngineSettings.MaxShadowDistance)
    float MinRange;             // slot 11 — analyse at least this far: terrain bumps, grass and decorators are no instances
    float ReachPerRadius;       // slot 12 — view depth a caster's shadow reaches past its centre per metre of radius
    float CameraX;              // slot 13-15 — camera position (world)
    float CameraY;
    float CameraZ;
    float ForwardX;             // slot 16-18 — camera forward
    float ForwardY;
    float ForwardZ;
};
#define CameraPosition float3(CameraX, CameraY, CameraZ)
#define CameraForward float3(ForwardX, ForwardY, ForwardZ)

// The camera culling's bounds per instance (cull_instances.hlsl InstanceBounds)
struct InstanceBounds
{
    float3 Center;
    float Radius;
    uint Part;
};
#define INVALID_PART 0xFFFFFFFFu

#define DEPTH_BINS 1024         // log-spaced: each bin ~1% of its depth (near 0.1 m, range 2000 m)
#define SAMPLE_STEP 4           // histogram: every 4th pixel each way, 1/16 of them is plenty for percentiles
#define TILE_SAMPLES 32         // blocks per group edge: 16 threads x 2
#define TILE_PIXELS (TILE_SAMPLES * SAMPLE_STEP) // 128: GPUCuller.AnalyzeDepth dispatches ceil(size / 128) groups
#define FLOAT_MAX_BITS 0x7F7FFFFF

SamplerState PointClamp : register(s1); // static sampler 1: point clamp (only Gather uses it)

// How far this frame's analysis looks: the farthest receiver a visible caster can shade (CSCasterReach), at least
// MinRange, at most MaxShadowDistance. Both kernels read it; CSDepthSplits clears it last.
float AnalysisRange()
{
    RWByteAddressBuffer minMaxBuf = ResourceDescriptorHeap[MinMaxUAVIdx];
    float reach = asfloat(minMaxBuf.Load(8));
    return clamp(max(reach, MinRange), NearPlane + 1.0f, max(MaxShadowDistance, NearPlane + 1.0f));
}

// Bins are spaced evenly in log2(depth), so every split gets the same relative precision
float LogNear() { return log2(max(NearPlane, 0.001f)); }
float BinsPerLog2(float range) { return DEPTH_BINS / max(log2(range) - LogNear(), 0.001f); }

// ============================================================
// CSCasterReach — one thread per instance of a shadow-casting batch, after its camera culling
// ============================================================
// A visible caster's shadow falls behind it, away from the sun: at most twice its radius high (it stands on the
// ground), so up to 2r / tan(sun elevation) across the ground. Its farthest receiver in view depth: its centre's depth,
// one radius, and that shadow's length along the view (ReachPerRadius = 1 + 2 · max(0, shadow dir · forward) / tan e,
// from the CPU). Invisible instances and the parts the culling dropped add nothing.
[numthreads(64, 1, 1)]
void CSCasterReach(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint instanceIdx = dispatchThreadId.x;
    float reach = 0.0f;
    if (instanceIdx < CasterCountIdx)
    {
        RWStructuredBuffer<uint> flags = ResourceDescriptorHeap[CasterFlagsIdx];
        RWStructuredBuffer<InstanceBounds> bounds = ResourceDescriptorHeap[CasterBoundsIdx];
        InstanceBounds sphere = bounds[instanceIdx];
        if (flags[instanceIdx] != 0 && sphere.Part != INVALID_PART && sphere.Radius > 0.0f)
        {
            float depth = dot(sphere.Center - CameraPosition, CameraForward);
            reach = max(depth + sphere.Radius * ReachPerRadius, 0.0f);
        }
    }
    reach = WaveActiveMax(reach);
    if (WaveIsFirstLane() && reach > 0.0f)
    {
        RWByteAddressBuffer minMaxBuf = ResourceDescriptorHeap[MinMaxUAVIdx];
        minMaxBuf.InterlockedMax(8, asuint(min(reach, MaxShadowDistance)));
    }
}

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
    float range = AnalysisRange();
    float logNear = LogNear();
    float binsPerLog2 = BinsPerLog2(range);
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

            // Skip sky/empty pixels (depth <= 0) and pixels past the farthest receiver a caster can shade
            [unroll]
            for (uint q = 0; q < 4; q++)
            {
                [unroll]
                for (uint c = 0; c < 4; c++)
                {
                    float v = quads[q][c];
                    if (v > 0.0f && v <= range)
                    {
                        nearest = min(nearest, v);
                        farthest = max(farthest, v);
                    }
                }
            }

            float depth = quads[0].w;
            if (depth > 0.0f && depth <= range)
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
    float binsPerLog2 = BinsPerLog2(AnalysisRange());   // read by every thread before thread 0 clears it below

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
        minMaxBuf.Store3(0, uint3(FLOAT_MAX_BITS, 0, 0));

        // No valid depth data: zeros, the light falls back to its fixed splits
        if (total == 0 || maxDepth - minDepth <= 0.0001f)
        {
            splits[0] = 0; splits[1] = 0; splits[2] = 0; splits[3] = 0;
            return;
        }

        // Each percentile split no farther than the logarithmic one between the nearest and the farthest pixel: a
        // view full of distant ground (a vista over a forest that reaches a kilometre) put 40 % of the pixels past
        // 300 m and cascade 0 with them. In a room the percentiles stay the nearer ones.
        float logFrom = max(minDepth, NearPlane + 0.1f);
        float logRatio = max(maxDepth, logFrom) / logFrom;
        float result[4];
        [unroll]
        for (uint c2 = 0; c2 < 3; c2++)
            result[c2] = min(max(gs_split[c2], NearPlane + 0.1f), logFrom * pow(logRatio, (c2 + 1) / 4.0f));
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
