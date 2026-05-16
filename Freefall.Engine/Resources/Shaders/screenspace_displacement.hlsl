// Screen-Space Displacement Mapping (SSDM)
//
// Five kernels:
//   CSDisplace      — simple single-pass (bypass pyramid for debugging)
//   CSScaleA        — copy HeightBuffer * HeightScale into A[0]
//   CSBuildMip      — build A displacement mip chain (average 2x2)
//   CSRefine        — hierarchical Newton inversion with inverse annealing
//   CSDisplaceDepth — remap depth buffer through B for contact shadows

#pragma kernel CSDisplace
#pragma kernel CSScaleA
#pragma kernel CSBuildMip
#pragma kernel CSRefine
#pragma kernel CSDisplaceDepth

SamplerState BilinearClamp : register(s2);

cbuffer PushConstants : register(b3)
{
    uint SrcMipIdx;     // SRV for source
    uint DstMipIdx;     // UAV for output
    uint PrevBMipIdx;   // UAV: previous B level (0 = coarsest, use self UV)
    uint _pad;
};

cbuffer Params : register(b4)
{
    uint DstWidth;
    uint DstHeight;
    float HeightScale;
    uint MipCount;
};

// ============================================================
// Simple pass-through (no pyramid, for debugging base data)
// ============================================================
[numthreads(8, 8, 1)]
void CSDisplace(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= DstWidth || dtid.y >= DstHeight)
        return;

    Texture2D<float2> src = ResourceDescriptorHeap[SrcMipIdx];
    RWTexture2D<float2> dst = ResourceDescriptorHeap[DstMipIdx];

    float2 disp = src.Load(int3(dtid.xy, 0));
    float2 invSize = 1.0 / float2(DstWidth, DstHeight);
    float2 myUV = (float2(dtid.xy) + 0.5) * invSize;
    // Store (0,0) sentinel for zero displacement to avoid half-precision UV jitter
    dst[dtid.xy] = all(disp == 0) ? float2(0, 0) : myUV - disp * HeightScale;
}

// ============================================================
// Scale A[0]: copy HeightBuffer * HeightScale into A mip 0
// Bakes HeightScale into displacement vectors so CSRefine
// works with pre-scaled values (no per-level amplification).
// ============================================================
[numthreads(8, 8, 1)]
void CSScaleA(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= DstWidth || dtid.y >= DstHeight)
        return;

    Texture2D<float2> src = ResourceDescriptorHeap[SrcMipIdx];
    RWTexture2D<float2> dst = ResourceDescriptorHeap[DstMipIdx];

    dst[dtid.xy] = src.Load(int3(dtid.xy, 0)) * HeightScale;
}

// ============================================================
// Build A mip chain (average 2x2 blocks, no scaling)
// ============================================================
[numthreads(8, 8, 1)]
void CSBuildMip(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= DstWidth || dtid.y >= DstHeight)
        return;

    Texture2D<float2> src = ResourceDescriptorHeap[SrcMipIdx];
    RWTexture2D<float2> dst = ResourceDescriptorHeap[DstMipIdx];

    int2 base = dtid.xy * 2;
    float2 s00 = src.Load(int3(base + int2(0,0), 0));
    float2 s10 = src.Load(int3(base + int2(1,0), 0));
    float2 s01 = src.Load(int3(base + int2(0,1), 0));
    float2 s11 = src.Load(int3(base + int2(1,1), 0));

    dst[dtid.xy] = (s00 + s10 + s01 + s11) * 0.25;
}


// ============================================================
// Refine — Coarse-seeded Newton inversion
//
// For each pixel (destination), find which source pixel's
// forward displacement maps to this destination.
// Uses coarsest mip for initial estimate (smooth, no folds),
// then refines with damped Newton iterations at full res.
// ============================================================
[numthreads(8, 8, 1)]
void CSRefine(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= DstWidth || dtid.y >= DstHeight)
        return;

    Texture2D<float2> pyramidA = ResourceDescriptorHeap[SrcMipIdx];
    RWTexture2D<float2> dstB = ResourceDescriptorHeap[DstMipIdx];

    float2 invSize = 1.0 / float2(DstWidth, DstHeight);
    float2 myUV = (float2(dtid.xy) + 0.5) * invSize;

    // Early-out: pixels with zero displacement (meshes, sky) → sentinel (0,0).
    float2 myDisp = pyramidA.Load(int3(dtid.xy, 0));
    if (all(myDisp == 0))
    {
        dstB[dtid.xy] = float2(0, 0);
        return;
    }

    // Hierarchical Newton inversion: seed from identity, then ascend
    // through the mip pyramid from coarsest to finest resolution.
    // Each iteration refines the source estimate with finer displacement detail.
    float2 source = myUV;
    float maxExtent = max(DstWidth, DstHeight);
    float maxExtentSqd = maxExtent * maxExtent;
    // Step schedule: gentle at coarse mips (careful basin search),
    // aggressive at fine mips (precise snap once in the right basin).
    [unroll]
    for (int i = 3; i >= 0; i--)
    {
        float2 d_at_source = pyramidA.SampleLevel(BilinearClamp, source, i);
        float2 error = myUV - (source + d_at_source);

        float errorSq = dot(error, error) * maxExtentSqd;
        float step = 0.5 / (1.0 + i);

        if (errorSq < step * step)
            break;

        source += error * step;
    }

    dstB[dtid.xy] = source;
}

// ============================================================
// Displace Depth — remap GBuffer linear depth using B buffer
//
// For each pixel, reads depth from the SSDM source position
// so the depth buffer matches the displaced visual content.
// This enables contact shadows and CSM to see displacement.
// ============================================================
[numthreads(8, 8, 1)]
void CSDisplaceDepth(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= DstWidth || dtid.y >= DstHeight)
        return;

    Texture2D<float2> ssdmB    = ResourceDescriptorHeap[SrcMipIdx];  // B buffer (source UVs)
    Texture2D<float>  depthIn  = ResourceDescriptorHeap[PrevBMipIdx]; // GBuffer depth (input)
    RWTexture2D<float> depthOut = ResourceDescriptorHeap[DstMipIdx]; // displaced depth (output)

    float2 sourceUV = ssdmB.Load(int3(dtid.xy, 0));
    
    // Gather the 2x2 texel quad around sourceUV in a single fetch.
    // Prevents tiny Newton residuals from creating false depth pockets.
    float4 g = depthIn.GatherRed(BilinearClamp, sourceUV);
    depthOut[dtid.xy] = min(min(g.x, g.y), min(g.z, g.w));
}

