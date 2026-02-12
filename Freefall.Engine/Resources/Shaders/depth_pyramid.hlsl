// Hi-Z Depth Pyramid Generator
// Max-downsamples each mip level from the previous one.
// Uses GBuffer depth (cleared to 0 = no occluder) so max() naturally
// keeps the farthest geometry depth without sky contamination.
// Dispatched once per mip level: mip 0 reads from the GBuffer depth,
// subsequent mips read from the previous pyramid level.

cbuffer PushConstants : register(b3)
{
    uint4 Indices[8];
};

#define InputMipIdx   Indices[0].x   // SRV: input mip level (Texture2D<float>)
#define OutputMipIdx  Indices[0].y   // UAV: output mip level (RWTexture2D<float>)
#define OutputWidth   Indices[0].z   // Output mip width
#define OutputHeight  Indices[0].w   // Output mip height

[numthreads(8, 8, 1)]
void CSDownsample(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= OutputWidth || id.y >= OutputHeight)
        return;

    Texture2D<float> inputMip = ResourceDescriptorHeap[InputMipIdx];
    RWTexture2D<float> outputMip = ResourceDescriptorHeap[OutputMipIdx];

    // Each output texel is the max of 4 input texels (farthest geometry depth)
    // Sky/empty pixels (depth=0) are treated as FLT_MAX so that any tile containing
    // sky becomes un-occludable — you can't occlude through a hole in geometry.
    uint2 srcCoord = id.xy * 2;
    
    float d0 = inputMip[srcCoord + uint2(0, 0)];
    float d1 = inputMip[srcCoord + uint2(1, 0)];
    float d2 = inputMip[srcCoord + uint2(0, 1)];
    float d3 = inputMip[srcCoord + uint2(1, 1)];

    d0 = d0 <= 0 ? 3.402823466e+38 : d0;
    d1 = d1 <= 0 ? 3.402823466e+38 : d1;
    d2 = d2 <= 0 ? 3.402823466e+38 : d2;
    d3 = d3 <= 0 ? 3.402823466e+38 : d3;

    outputMip[id.xy] = max(max(d0, d1), max(d2, d3));
}

