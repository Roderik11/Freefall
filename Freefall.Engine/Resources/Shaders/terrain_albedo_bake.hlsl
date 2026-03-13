// terrain_albedo_bake.hlsl — Bakes terrain splatmap layers into a single low-res albedo texture.
// Dispatched once when terrain splatmaps change.
// Output: 256×256 averaged terrain color (no high-frequency tiling detail).
//
// Uses push constants (b3) matching engine convention.

#pragma kernel CSBakeTerrainAlbedo


// Push constants (root parameter 0, register b3) — bindless indices only
cbuffer PushConstants : register(b3)
{
    uint ControlMapsIdx;    // slot 0 — SRV: splatmap Texture2DArray
    uint DiffuseMapsIdx;    // slot 1 — SRV: diffuse layer Texture2DArray
    uint OutputUAVIdx;      // slot 2 — UAV: output RWTexture2D<float4>
    uint TilingBufIdx;      // slot 3 — SRV: StructuredBuffer<float4> (LayerTiling, 32 entries)
};

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

    // ControlMaps: ceil(layerCount/4) RGBA slices
    uint cmW, cmH, sliceCount;
    ControlMaps.GetDimensions(cmW, cmH, sliceCount);

    for (uint i = 0; i < sliceCount; ++i)
    {
        uint startIndex = i * 4;
        float4 weights = ControlMaps.SampleLevel(sampData, float3(uv, i), 0);

        for (uint j = 0; j < 4; ++j)
        {
            float weight = weights[j];
            if (weight <= 0)
                continue;

            uint layer = startIndex + j;
            float2 texuv = uv * Tiling[layer].xy;
            // Sample at high mip for averaged color (no tiling detail)
            float4 c = DiffuseMaps.SampleLevel(sampData, float3(texuv, layer), 4.0);
            color += c * weight;
        }
    }

    OutputAlbedo[dtid.xy] = color;
}
