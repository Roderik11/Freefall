// Ocean mipmap downsample compute shader
// Generates mip chain for Texture2DArray (displacement + slope textures)
// SM 6.6 bindless, push constants at b3

struct PushConstantsData { uint4 indices[8]; };
#define GET_INDEX(i) PushConstants.indices[i/4][i%4]
ConstantBuffer<PushConstantsData> PushConstants : register(b3);

#define SrcMipIdx   GET_INDEX(0)   // UAV: source mip level
#define DstMipIdx   GET_INDEX(1)   // UAV: destination mip level
#define MipTexelSize GET_INDEX(2)  // destination mip width (uint)
#define NumSlices   GET_INDEX(3)   // number of array slices (uint)
#define IsRG16F     GET_INDEX(4)   // 0 = RGBA16F, 1 = RG16F

[numthreads(8, 8, 1)]
void CSDownsample(uint3 dtid : SV_DispatchThreadID)
{
    uint dstSize = MipTexelSize;
    uint sliceCount = NumSlices;
    
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

        if (IsRG16F != 0)
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
