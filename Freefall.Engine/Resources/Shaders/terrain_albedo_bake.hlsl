// terrain_albedo_bake.hlsl — Bakes terrain splatmap layers into a single low-res albedo texture.
// Dispatched once when terrain splatmaps change.
// Output: 256×256 averaged terrain color (no high-frequency tiling detail).
//
// Uses push constants (b3) matching engine convention.

#pragma kernel CSBakeTerrainAlbedo


struct PushConstantsData { uint4 indices[8]; };
#define GET_INDEX(i) PushConstants.indices[i/4][i%4]
ConstantBuffer<PushConstantsData> PushConstants : register(b3);

#define ControlMapsIdx  GET_INDEX(0)   // SRV: splatmap Texture2DArray
#define DiffuseMapsIdx  GET_INDEX(1)   // SRV: diffuse layer Texture2DArray
#define OutputUAVIdx    GET_INDEX(2)   // UAV: output RWTexture2D<float4>
#define TilingBufIdx    GET_INDEX(3)   // SRV: StructuredBuffer<float4> (LayerTiling, 32 entries)

SamplerState sampData : register(s0);  // WrappedAnisotropic

[numthreads(8, 8, 1)]
void CSBakeTerrainAlbedo(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float4> OutputAlbedo = ResourceDescriptorHeap[OutputUAVIdx];

    uint outW, outH;
    OutputAlbedo.GetDimensions(outW, outH);
    if (dtid.x >= outW || dtid.y >= outH)
        return;

    // UV in terrain space [0,1]
    float2 uv = (float2(dtid.xy) + 0.5) / float2(outW, outH);

    Texture2DArray ControlMaps = ResourceDescriptorHeap[ControlMapsIdx];
    Texture2DArray DiffuseMaps = ResourceDescriptorHeap[DiffuseMapsIdx];
    StructuredBuffer<float4> Tiling = ResourceDescriptorHeap[TilingBufIdx];

    float4 color = float4(0, 0, 0, 0);

    // Mirror terrain.fx PS splatmap blending
    for (int i = 0; i < 4; ++i)
    {
        int startIndex = i * 4;
        float4 weights = ControlMaps.SampleLevel(sampData, float3(uv, i), 0);

        for (int j = 0; j < 4; ++j)
        {
            float weight = weights[j];
            if (weight <= 0)
                continue;

            int layer = startIndex + j;
            float2 texuv = uv * Tiling[layer].xy;
            // Sample at high mip for averaged color (no tiling detail)
            float4 c = DiffuseMaps.SampleLevel(sampData, float3(texuv, layer), 4.0);
            color += c * weight;
        }
    }

    OutputAlbedo[dtid.xy] = color;
}
