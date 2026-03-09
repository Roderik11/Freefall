// Ocean mipmap downsample compute shader
// Generates mip chain for Texture2DArray (displacement + slope textures)
// SM 6.6 bindless, push constants at b3

#pragma kernel CSDownsample

cbuffer PushConstants : register(b3)
{
    uint SrcMipIdx;       // UAV: source mip level
    uint DstMipIdx;       // UAV: destination mip level
    uint MipTexelSizeIdx; // destination mip width (uint)
    uint NumSlicesIdx;    // number of array slices (uint)
    uint IsRG16FIdx;      // 0 = RGBA16F, 1 = RG16F
};

[numthreads(8, 8, 1)]
void CSDownsample(uint3 dtid : SV_DispatchThreadID)
{
    uint dstSize = MipTexelSizeIdx;
    uint sliceCount = NumSlicesIdx;
    
    if (dtid.x >= dstSize || dtid.y >= dstSize)
        return;

    uint2 srcBase = dtid.xy * 2;

    for (uint slice = 0; slice < sliceCount; slice++)
    {
        uint3 dstCoord = uint3(dtid.xy, slice);
        uint3 s00 = uint3(srcBase + uint2(0, 0), slice);
        uint3 s10 = uint3(srcBase + uint2(1, 0), slice);
        uint3 s01 = uint3(srcBase + uint2(0, 1), slice);
        uint3 s11 = uint3(srcBase + uint2(1, 1), slice);

        if (IsRG16FIdx != 0)
        {
            RWTexture2DArray<float2> srcTex = ResourceDescriptorHeap[SrcMipIdx];
            RWTexture2DArray<float2> dstTex = ResourceDescriptorHeap[DstMipIdx];
            float2 avg = (srcTex[s00] + srcTex[s10] + srcTex[s01] + srcTex[s11]) * 0.25;
            dstTex[dstCoord] = avg;
        }
        else
        {
            RWTexture2DArray<float4> srcTex = ResourceDescriptorHeap[SrcMipIdx];
            RWTexture2DArray<float4> dstTex = ResourceDescriptorHeap[DstMipIdx];
            float4 avg = (srcTex[s00] + srcTex[s10] + srcTex[s01] + srcTex[s11]) * 0.25;
            dstTex[dstCoord] = avg;
        }
    }
}
