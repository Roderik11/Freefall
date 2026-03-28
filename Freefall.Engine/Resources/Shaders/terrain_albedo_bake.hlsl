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
    uint HeightTexIdx;      // slot 4 — SRV: baked heightmap for procedural masking
    uint AutoMaskBufIdx;    // slot 5 — SRV: StructuredBuffer<LayerAutoMask>
    float MaxHeight;        // slot 6 — terrain max height
    float HeightTexel;      // slot 7 — 1.0 / heightmap resolution
    float TerrainSizeX;     // slot 8 — terrain world X dimension
    float TerrainSizeZ;     // slot 9 — terrain world Z dimension
};

// Must match gputerrain.fx
struct LayerAutoMask
{
    float HeightMin, HeightMax;
    float SlopeMin, SlopeMax;
    float HeightBlend, SlopeBlend;
    float ProceduralWeight;
    float _pad;
};

SamplerState sampData : register(s0);  // WrappedAnisotropic
SamplerState sampHeightFilter : register(s2);  // ClampedBilinear2D

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
    StructuredBuffer<LayerAutoMask> AutoMaskBuf = ResourceDescriptorHeap[AutoMaskBufIdx];

    // Compute terrain normal from heightmap for slope masking
    Texture2D HeightTex = ResourceDescriptorHeap[HeightTexIdx];
    float heightNorm = HeightTex.SampleLevel(sampHeightFilter, uv, 0).r;

    float4 hSamples;
    hSamples[0] = HeightTex.SampleLevel(sampHeightFilter, uv + float2(0, -HeightTexel), 0).r;
    hSamples[1] = HeightTex.SampleLevel(sampHeightFilter, uv + float2(-HeightTexel, 0), 0).r;
    hSamples[2] = HeightTex.SampleLevel(sampHeightFilter, uv + float2(HeightTexel, 0), 0).r;
    hSamples[3] = HeightTex.SampleLevel(sampHeightFilter, uv + float2(0, HeightTexel), 0).r;

    float texelWorldSize = TerrainSizeX * HeightTexel;
    float hScale = MaxHeight / texelWorldSize;
    float3 terrainNormal;
    terrainNormal.z = (hSamples[0] - hSamples[3]) * hScale;
    terrainNormal.x = (hSamples[1] - hSamples[2]) * hScale;
    terrainNormal.y = 1.0;
    terrainNormal = normalize(terrainNormal);

    float slopeDeg = acos(saturate(terrainNormal.y)) * (180.0 / 3.14159265);

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

            uint layer = startIndex + j;

            // Procedural slope/height auto-mask: max(painted, procedural)
            LayerAutoMask mask = AutoMaskBuf[layer];
            if (mask.ProceduralWeight > 0)
            {
                float pmask = 1;
                pmask *= smoothstep(mask.SlopeMin - mask.SlopeBlend, mask.SlopeMin, slopeDeg);
                pmask *= smoothstep(mask.SlopeMax + mask.SlopeBlend, mask.SlopeMax, slopeDeg);
                pmask *= smoothstep(mask.HeightMin - mask.HeightBlend, mask.HeightMin, heightNorm);
                pmask *= smoothstep(mask.HeightMax + mask.HeightBlend, mask.HeightMax, heightNorm);
                weight = max(weight, pmask * mask.ProceduralWeight);
            }

            if (weight <= 0)
                continue;

            float2 texuv = uv * Tiling[layer].xy;
            // Sample at high mip for averaged color (no tiling detail)
            float4 c = DiffuseMaps.SampleLevel(sampData, float3(texuv, layer), 4.0);
            color = lerp(color, c, weight);
        }
    }

    OutputAlbedo[dtid.xy] = color;
}
