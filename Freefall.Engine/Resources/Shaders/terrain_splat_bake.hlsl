// terrain_splat_bake.hlsl — Composites the terrain's SplatStamps into its layer weights.
// SM 6.6 bindless, push constants at b3
//
// Output: the packed control array the surface shader samples: ceil(LayerCount / 4) RGBA slices,
// one channel per layer, in palette order.
//
// The stamps arrive sorted by priority and each one paints OVER what is there, whatever its layer:
// a road stamped after a forest floor shows as road. The surface shader, however, blends the channels
// as a fixed stack (channel N covers every channel below it), so a weight alone cannot say "this was
// painted last". PaintOver therefore rewrites the weights so that the stack reproduces the paint order:
// it converts to visible ("effective") weights, alpha-composites the stamp, and converts back.
// A layer that ends up completely covered keeps the weight it was painted with: nothing of it shows,
// but filters that ask "was this painted here" (ExcludeLayers) can still see it.
//
// Second output: one R8_UINT texel per control texel naming the channel that shows most there. It is
// what the CPU reads back to answer "which layer is under this point".

#pragma kernel CS_SplatBake

#include "terrain_stamp_common.hlsli"
#include "terrain_coverage.hlsli"

cbuffer PushConstants : register(b3)
{
    uint  StampBufIdx;      // slot 0 — SRV: StructuredBuffer<CoverageStamp>, sorted by priority
    uint  OutputIdx;        // slot 1 — UAV: RWTexture2DArray<float4> packed control array
    uint  HeightTexIdx;     // slot 2 — SRV: baked heightmap (0 = none yet: height/slope read as 0)
    uint  SplineBufIdx;     // slot 3 — SRV: StructuredBuffer<StampSplinePoint>
    uint  StampCount;       // slot 4
    uint  Resolution;       // slot 5
    uint  SliceCount;       // slot 6
    uint  LayerCount;       // slot 7
    float MaxHeight;        // slot 8
    float TerrainSizeX;     // slot 9
    uint  TopLayerIdx;      // slot 10 — UAV: RWTexture2D<uint>, the channel that shows most per texel
    uint  RegionX;          // slot 11 — the control texels this bake redoes: from (RegionX, RegionY), RegionW x RegionH
    uint  RegionY;          // slot 12   (the whole map, or the region a local stamp's edit touched; the rest keeps
    uint  RegionW;          // slot 13   what the last bake wrote)
    uint  RegionH;          // slot 14
};

#define SPLAT_OP_PAINT    0u
#define SPLAT_OP_REMOVE   1u
#define SPLAT_OP_MULTIPLY 2u

#define MAX_LAYERS 32

SamplerState sampHeightFilter : register(s2); // ClampedBilinear2D

// Paint 'layer' with opacity 'a' over the current result.
// Channels below 'layer' are untouched: scaling everything above them by (1 - a) and adding the
// stamp leaves their stack weight exactly where it was.
void PaintOver(inout float raw[MAX_LAYERS], uint layer, float a, uint layerCount)
{
    float coverOld = 1.0;   // product of (1 - raw) of the channels above, before the stamp
    float sumNew = 0.0;     // visible weight of the channels above, after the stamp

    for (int c = int(layerCount) - 1; c >= int(layer); c--)
    {
        float rOld = raw[c];
        float visNew = rOld * coverOld * (1.0 - a) + (uint(c) == layer ? a : 0.0);

        float room = 1.0 - sumNew;
        if (room > 1e-4)
            raw[c] = saturate(visNew / room);
        // else: completely covered from above — keep the painted weight

        coverOld *= (1.0 - rOld);
        sumNew += visNew;
    }
}

[numthreads(8, 8, 1)]
void CS_SplatBake(uint3 groupTexel : SV_DispatchThreadID)
{
    if (groupTexel.x >= RegionW || groupTexel.y >= RegionH) return;
    uint2 texel = groupTexel.xy + uint2(RegionX, RegionY);
    if (texel.x >= Resolution || texel.y >= Resolution) return;

    RWTexture2DArray<float4> output = ResourceDescriptorHeap[OutputIdx];
    StructuredBuffer<CoverageStamp> stamps = ResourceDescriptorHeap[StampBufIdx];
    StructuredBuffer<StampSplinePoint> splinePoints = ResourceDescriptorHeap[SplineBufIdx];

    float2 uv = (float2(texel) + 0.5) / float2(Resolution, Resolution);

    // The control array is Y-flipped relative to terrain UV (heightmap, stamp shapes)
    float2 terrainUV = float2(uv.x, 1.0 - uv.y);

    float heightNorm = 0, slopeDeg = 0;
    if (HeightTexIdx != 0)
    {
        Texture2D heightTex = ResourceDescriptorHeap[HeightTexIdx];
        SampleTerrainHeightSlope(heightTex, sampHeightFilter, terrainUV, MaxHeight, TerrainSizeX, heightNorm, slopeDeg);
    }

    uint layerCount = min(LayerCount, (uint)MAX_LAYERS);

    float raw[MAX_LAYERS];
    for (uint z = 0; z < MAX_LAYERS; z++) raw[z] = 0;

    for (uint i = 0; i < StampCount; i++)
    {
        CoverageStamp stamp = stamps[i];
        if (stamp.Target >= layerCount) continue;

        float weight = CoverageShapeWeight(stamp, terrainUV, splinePoints);
        if (weight <= 0) continue;

        if (stamp.Flags & COVERAGE_FILTER)
        {
            weight *= CoverageTerrainFilter(stamp, heightNorm, slopeDeg, terrainUV * TerrainSizeX);   // square terrain

            if ((stamp.RequireMask | stamp.ExcludeMask) != 0)
            {
                // What shows (required) and what was painted (excluded), from the result so far
                float required = 0, excluded = 0;
                float cover = 1.0;
                for (int c = int(layerCount) - 1; c >= 0; c--)
                {
                    uint bit = 1u << uint(c);
                    if (stamp.RequireMask & bit) required += raw[c] * cover;        // together, as in decoration_prepass
                    if (stamp.ExcludeMask & bit) excluded = max(excluded, raw[c]);
                    cover *= (1.0 - raw[c]);
                }
                if (stamp.RequireMask != 0) weight *= saturate(required);
                weight *= (1.0 - excluded);
            }

            if (weight <= 0) continue;
        }

        uint op = CoverageOp(stamp);
        if (op == SPLAT_OP_PAINT)
            PaintOver(raw, stamp.Target, saturate(weight * stamp.Strength), layerCount);
        else if (op == SPLAT_OP_REMOVE)
            raw[stamp.Target] *= 1.0 - saturate(weight * stamp.Strength);
        else
            raw[stamp.Target] = saturate(raw[stamp.Target] * lerp(1.0, stamp.Strength, weight));
    }

    // Every slice is written every bake, so nothing from an earlier bake survives
    uint sliceCount = min(SliceCount, 8u);
    for (uint s = 0; s < sliceCount; s++)
        output[uint3(texel, s)] = float4(raw[s * 4], raw[s * 4 + 1], raw[s * 4 + 2], raw[s * 4 + 3]);

    // What the ground is here, for whoever asks on the CPU (TerrainRenderer.GetLayer): the channel
    // with the greatest visible weight, stacked as the surface shader stacks them. 255 = nothing painted.
    if (TopLayerIdx != 0)
    {
        RWTexture2D<uint> topLayer = ResourceDescriptorHeap[TopLayerIdx];

        uint top = 255;
        float most = 0.0, cover = 1.0;
        for (int c = int(layerCount) - 1; c >= 0; c--)
        {
            float visible = raw[c] * cover;
            if (visible > most) { most = visible; top = uint(c); }
            cover *= (1.0 - raw[c]);
        }
        topLayer[texel] = top;
    }
}
