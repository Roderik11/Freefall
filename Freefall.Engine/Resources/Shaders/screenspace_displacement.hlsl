// Screen-Space Displacement Mapping (SSDM) - Lobel 2008
//
// Four kernels:
//   CSDisplace  — simple single-pass (bypass pyramid for debugging)
//   CSScaleA    — copy HeightBuffer * HeightScale into A[0]
//   CSBuildMip  — build A displacement mip chain (average 2x2)
//   CSRefine    — coarse-to-fine barycenter convergence (paper literal)

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
    dst[dtid.xy] = myUV - disp * HeightScale;
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

    // Seed from coarsest mip (smooth displacement, no fold-over)
    int coarseMip = min((int)MipCount - 1, 7);
    float2 d_coarse = pyramidA.SampleLevel(BilinearClamp, myUV, float(coarseMip));
    float2 source = myUV - d_coarse;

    // Damped Newton iterations at full resolution
    [unroll]
    for (int i = 0; i < 4; i++)
    {
        float2 d_at_source = pyramidA.SampleLevel(BilinearClamp, source, 0);
        float2 error = myUV - (source + d_at_source);

        if (dot(error, error) < dot(invSize, invSize) * 0.25)
            break;

        source += error * 0.5;
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
    
    // Sample depth at the source position (bilinear for smooth result)
    float srcDepth = depthIn.SampleLevel(BilinearClamp, sourceUV, 0);
    depthOut[dtid.xy] = srcDepth;
}
