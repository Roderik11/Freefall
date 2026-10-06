// decoration_prepass.hlsl — Composites the terrain's DecoStamps into the decoration control texture.
// SM 6.6 bindless, push constants at b3
//
// For each texel: walks the deco stamps in priority order and builds one coverage weight per
// decorator. Add raises a decorator's coverage, Multiply scales it (one decorator, or all of them).
// Each stamp's weight is its shape times its filter: height/slope from the heightmap, and layer terms
// from the finished splat result (so "grass blades where the grass layer shows" needs no ordering).
//
// Finds the top 8 decorators by weight and packs (slot, weight) into RGBA16_UINT x 2 slices.
// Each channel: (decoratorSlot << 8) | weight. Unused entries: slot=255, weight=0.
//
// A slot is a TerrainDecorator, not one of its variants: the weight here is pure coverage. Density and
// clumping are per variant and are applied by the spawn kernel (grass_compute.hlsl).

#pragma kernel CSBuildDecoControl

#include "terrain_stamp_common.hlsli"
#include "terrain_coverage.hlsli"

cbuffer PushConstants : register(b3)
{
    uint  StampBufIdx;      // slot 0 — SRV: StructuredBuffer<CoverageStamp>, sorted by priority
    uint  ControlUAVIdx;    // slot 1 — UAV: output RWTexture2DArray<uint4>
    uint  DecoratorCount;   // slot 2 — number of decorator slots (palette size)
    uint  Resolution;       // slot 3 — control texture width/height
    uint  HeightTexIdx;     // slot 4 — SRV: baked heightmap (0 = none: height/slope read as 0)
    uint  SplineBufIdx;     // slot 5 — SRV: StructuredBuffer<StampSplinePoint>
    uint  LayerCount;       // slot 6 — number of texture layers in the packed control array
    uint  ControlMapsIdx;   // slot 7 — SRV: packed layer weights (same array the surface shader samples)
    float MaxHeight;        // slot 8
    float TerrainSizeX;     // slot 9
    uint  StampCount;       // slot 10
};

#define DECO_OP_ADD      0u
#define DECO_OP_MULTIPLY 1u

#define MAX_LAYERS     32
#define MAX_DECORATORS 32

SamplerState ClampSampler : register(s2);

[numthreads(8, 8, 1)]
void CSBuildDecoControl(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2DArray<uint4> controlTex = ResourceDescriptorHeap[ControlUAVIdx];
    StructuredBuffer<CoverageStamp> stamps = ResourceDescriptorHeap[StampBufIdx];
    StructuredBuffer<StampSplinePoint> splinePoints = ResourceDescriptorHeap[SplineBufIdx];

    if (dtid.x >= Resolution || dtid.y >= Resolution) return;

    // Control texture space is Y-flipped relative to terrain UV (heightmap, stamp shapes)
    float2 uv = (float2(dtid.xy) + 0.5) / float2(Resolution, Resolution);
    float2 terrainUV = float2(uv.x, 1.0 - uv.y);

    float heightNorm = 0, slopeDeg = 0;
    if (HeightTexIdx != 0)
    {
        Texture2D heightTex = ResourceDescriptorHeap[HeightTexIdx];
        SampleTerrainHeightSlope(heightTex, ClampSampler, terrainUV, MaxHeight, TerrainSizeX, heightNorm, slopeDeg);
    }

    // ── Layer weights from the packed control array (final splat result) ──
    // raw = the weight as painted; visible = what shows after the layers above it:
    // visible[i] = raw[i] * product(1 - raw[k]) for all k > i, as in gputerrain.fx
    uint layerCount = ControlMapsIdx != 0 ? min(LayerCount, (uint)MAX_LAYERS) : 0;

    float rawWeight[MAX_LAYERS];
    float visibleWeight[MAX_LAYERS];
    for (uint z = 0; z < MAX_LAYERS; z++) { rawWeight[z] = 0; visibleWeight[z] = 0; }

    if (layerCount > 0)
    {
        Texture2DArray controlMaps = ResourceDescriptorHeap[ControlMapsIdx];
        uint sliceCount = (layerCount + 3) / 4;
        for (uint si = 0; si < sliceCount; si++)
        {
            float4 weights = controlMaps.SampleLevel(ClampSampler, float3(uv, si), 0);
            for (uint sj = 0; sj < 4; sj++)
            {
                uint layerIdx = si * 4 + sj;
                if (layerIdx < layerCount) rawWeight[layerIdx] = weights[sj];
            }
        }

        float cover = 1.0;
        for (int li = int(layerCount) - 1; li >= 0; li--)
        {
            visibleWeight[li] = rawWeight[li] * cover;
            cover *= saturate(1.0 - rawWeight[li]);
        }
    }

    // ── Composite the stamps into one coverage weight per decorator ──
    uint decoCount = min(DecoratorCount, (uint)MAX_DECORATORS);

    float coverage[MAX_DECORATORS];
    for (uint d0 = 0; d0 < MAX_DECORATORS; d0++) coverage[d0] = 0;

    for (uint i = 0; i < StampCount; i++)
    {
        CoverageStamp stamp = stamps[i];

        float weight = CoverageShapeWeight(stamp, terrainUV, splinePoints);
        if (weight <= 0) continue;

        if (stamp.Flags & COVERAGE_FILTER)
        {
            weight *= CoverageTerrainFilter(stamp, heightNorm, slopeDeg);

            if ((stamp.RequireMask | stamp.ExcludeMask) != 0)
            {
                float required = 0, excluded = 0;
                for (uint c = 0; c < layerCount; c++)
                {
                    uint bit = 1u << c;
                    if (stamp.RequireMask & bit) required = max(required, visibleWeight[c]);
                    if (stamp.ExcludeMask & bit) excluded = max(excluded, rawWeight[c]);
                }
                if (stamp.RequireMask != 0) weight *= required;
                weight *= (1.0 - excluded);
            }

            if (weight <= 0) continue;
        }

        if (CoverageOp(stamp) == DECO_OP_ADD)
        {
            if (stamp.Target < decoCount)
                coverage[stamp.Target] = max(coverage[stamp.Target], saturate(weight * stamp.Strength));
        }
        else
        {
            float factor = lerp(1.0, stamp.Strength, weight);
            if (stamp.Target == COVERAGE_ALL_TARGETS)
            {
                for (uint d1 = 0; d1 < decoCount; d1++)
                    coverage[d1] = saturate(coverage[d1] * factor);
            }
            else if (stamp.Target < decoCount)
            {
                coverage[stamp.Target] = saturate(coverage[stamp.Target] * factor);
            }
        }
    }

    // ── Collect top 8 decorators by weight (descending) ──
    uint topIdx[8];
    uint topWt[8];
    [unroll] for (uint k = 0; k < 8; k++)
    {
        topIdx[k] = 255;
        topWt[k] = 0;
    }
    uint topCount = 0;
    for (uint d = 0; d < decoCount; d++)
    {
        uint weight = (uint)(coverage[d] * 255.0 + 0.5);
        if (weight == 0) continue;

        // Insertion sort: find position and shift down
        uint pos = topCount < 8 ? topCount : 7;
        if (topCount >= 8 && weight <= topWt[7])
            continue;  // not heavy enough to make top 8

        // Find insertion point (descending order)
        [unroll] for (uint j = 0; j < 8; j++)
        {
            if (j < topCount && weight > topWt[j])
            {
                pos = j;
                break;
            }
        }

        // Shift entries down to make room
        [unroll] for (uint j2 = 7; j2 > 0; j2--)
        {
            if (j2 > pos)
            {
                topIdx[j2] = topIdx[j2 - 1];
                topWt[j2]  = topWt[j2 - 1];
            }
        }

        topIdx[pos] = d;
        topWt[pos]  = weight;
        if (topCount < 8) topCount++;
    }

    // Pack: (decoratorSlot << 8) | weight
    uint4 packed0, packed1;
    [unroll] for (uint ch = 0; ch < 4; ch++)
    {
        packed0[ch] = (topIdx[ch] << 8) | topWt[ch];
        packed1[ch] = (topIdx[ch + 4] << 8) | topWt[ch + 4];
    }

    controlTex[uint3(dtid.xy, 0)] = packed0;
    controlTex[uint3(dtid.xy, 1)] = packed1;
}
