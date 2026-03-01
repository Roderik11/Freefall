// Ocean mipmap downsample compute shader
// Generates mip chain for Texture2DArray (displacement + slope textures)
// Each dispatch: reads mip N (via UAV), writes box-filtered average to mip N+1 (via UAV)
// Both textures stay in UnorderedAccess state — no resource transitions needed.

#define GET_INDEX(n) PushConstants[n]

#define SrcMipIdx GET_INDEX(0)   // UAV: source mip level (RWTexture2DArray)
#define DstMipIdx GET_INDEX(1)   // UAV: destination mip level (RWTexture2DArray)
#define MipTexelSize GET_INDEX(2) // destination mip width (uint)
#define NumSlices GET_INDEX(3)   // number of array slices (uint)
#define IsRG16F GET_INDEX(4)     // 0 = RGBA16F, 1 = RG16F

cbuffer RootConstants : register(b0)
{
    uint PushConstants[16];
};

[numthreads(8, 8, 1)]
void CSDownsample(uint3 dtid : SV_DispatchThreadID)
{
    uint dstSize = MipTexelSize;
    uint sliceCount = NumSlices;
    
    if (dtid.x >= dstSize || dtid.y >= dstSize)
        return;

    // Source texel coords (2x2 block in the higher-res mip)
    uint2 srcBase = dtid.xy * 2;

    // Process all array slices
    for (uint slice = 0; slice < sliceCount; slice++)
    {
        if (IsRG16F != 0)
        {
            // RG16F path (slope texture) — read/write via UAV
            RWTexture2DArray<float2> srcTex = ResourceDescriptorHeap[SrcMipIdx];
            RWTexture2DArray<float2> dstTex = ResourceDescriptorHeap[DstMipIdx];
            
            float2 v00 = srcTex[uint3(srcBase + uint2(0, 0), slice)];
            float2 v10 = srcTex[uint3(srcBase + uint2(1, 0), slice)];
            float2 v01 = srcTex[uint3(srcBase + uint2(0, 1), slice)];
            float2 v11 = srcTex[uint3(srcBase + uint2(1, 1), slice)];
            
            dstTex[uint3(dtid.xy, slice)] = (v00 + v10 + v01 + v11) * 0.25;
        }
        else
        {
            // RGBA16F path (displacement texture) — read/write via UAV
            RWTexture2DArray<float4> srcTex = ResourceDescriptorHeap[SrcMipIdx];
            RWTexture2DArray<float4> dstTex = ResourceDescriptorHeap[DstMipIdx];
            
            float4 v00 = srcTex[uint3(srcBase + uint2(0, 0), slice)];
            float4 v10 = srcTex[uint3(srcBase + uint2(1, 0), slice)];
            float4 v01 = srcTex[uint3(srcBase + uint2(0, 1), slice)];
            float4 v11 = srcTex[uint3(srcBase + uint2(1, 1), slice)];
            
            dstTex[uint3(dtid.xy, slice)] = (v00 + v10 + v01 + v11) * 0.25;
        }
    }
}
