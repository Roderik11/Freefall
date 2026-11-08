// decoration_prepass.hlsl — Composites the terrain's DecoStamps into the decoration control map.
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
//
// Since 5c (TERRAIN_PAGES_PLAN.md, 2026-11-07) the map is baked page by page into the decoration pool (terrain_deco.hlsli):
// CSBuildDecoPage bakes one page into its slot. CSBuildDecoControl is the whole-map bake there was before, over a region,
// Y-flipped and written from its output's corner, kept only for Check Deco Pages. Both composite through DecoCoverage().

#pragma kernel CSBuildDecoPage
#pragma kernel CSBuildDecoControl

#include "terrain_stamp_common.hlsli"
#include "terrain_coverage.hlsli"
#include "terrain_paint.hlsli"   // the layer weights the layer filters read, from the paint pages

cbuffer PushConstants : register(b3)
{
    uint  StampBufIdx;      // slot 0 — SRV: StructuredBuffer<CoverageStamp>, sorted by priority
    uint  ControlUAVIdx;    // slot 1 — UAV: RWTexture2DArray<uint4>, the decoration pool (flat: the check's array)
    uint  DecoratorCount;   // slot 2 — number of decorator slots (palette size)
    uint  Resolution;       // slot 3 — texels a side of the whole map at level 0
    uint  TerrainPagesIdx;  // slot 4 — the terrain's TerrainPagesParams (terrain_pages.hlsli); 0 = none: height/slope read as 0
    uint  SplineBufIdx;     // slot 5 — SRV: StructuredBuffer<StampSplinePoint>
    uint  LayerCount;       // slot 6 — layers in the palette (the paint pages' channels)
    uint  ControlMapsIdx;   // slot 7 — unused since 5b: the layer weights come from the paint pages (terrain_paint.hlsli)
    float MaxHeight;        // slot 8
    float TerrainSizeX;     // slot 9
    uint  StampCount;       // slot 10
    uint  RegionX;          // slot 11 — flat: the texels this bake does, from (RegionX, RegionY), RegionW x RegionH
    uint  RegionY;          // slot 12
    uint  RegionW;          // slot 13
    uint  RegionH;          // slot 14
    int   PageX;            // slot 15 — page: which page, at which level
    int   PageY;            // slot 16
    uint  Level;            // slot 17
    uint  OutX;             // slot 18 — page: its slot's quarter corner in the slice (texels)
    uint  OutY;             // slot 19
    uint  OutSlice;         // slot 20 — page: the array slice of its first slice
    uint  PageShapeIdx;     // slot 21 — page: SRV StructuredBuffer<PageShape>, a stamp's segments per page (terrain_coverage.hlsli)
    uint  PageListIdx;      // slot 22 — page: SRV StructuredBuffer<uint>, the segment lists and row parities they point into
    uint  PageShapeBase;    // slot 23 — page: this page's first PageShape (one per stamp)
    uint  PageStampOffset;  // slot 24 — page: the stamps whose box reaches the page, in order, in the uint list from here
    uint  PageStampCount;   // slot 25
    uint  RowStart;         // slot 26 — page: the slot's texel rows this dispatch bakes, from RowStart (a coarse page goes
    uint  RowCount;         // slot 27     over several frames, TerrainMapResidency); 0 rows = the whole slot
};

#define DECO_OP_ADD      0u
#define DECO_OP_MULTIPLY 1u

#define MAX_LAYERS     32
#define MAX_DECORATORS 32

#define DECO_PAGE_TEXELS 256
#define DECO_BORDER      1
#define DECO_SLOT_TEXELS 258

SamplerState ClampSampler : register(s2);

// Every deco stamp at a place (terrain UV): each decorator's coverage, and the rows of the last Add stamp that holds it.
// paged: a page bake's sample, whose spline stamps walk only the page's segments (sampleRow: its row in the page)
void DecoCoverage(float2 terrainUV, uint decoCount, bool paged, uint sampleRow, out float coverage[MAX_DECORATORS],
    out uint rowIdx, out uint rowSlot)
{
    StructuredBuffer<CoverageStamp> stamps = ResourceDescriptorHeap[StampBufIdx];
    StructuredBuffer<StampSplinePoint> splinePoints = ResourceDescriptorHeap[SplineBufIdx];
    StructuredBuffer<PageShape> pageShapes = ResourceDescriptorHeap[PageShapeIdx];
    StructuredBuffer<uint> pageList = ResourceDescriptorHeap[PageListIdx];

    float heightNorm = 0, slopeDeg = 0;
    if (TerrainPagesIdx != 0)
        SampleTerrainHeightSlope(LoadTerrain(TerrainPagesIdx), terrainUV, heightNorm, slopeDeg);

    // ── Layer weights from the paint pages (final splat result) ──
    // raw = the weight as painted; visible = what shows after the layers above it:
    // visible[i] = raw[i] * product(1 - raw[k]) for all k > i, as in gputerrain.fx
    TerrainPaintParams paint = LoadPaint(LoadTerrain(TerrainPagesIdx));   // nothing (no paint) without a terrain
    uint layerCount = PaintPresent(paint) ? min(LayerCount, (uint)MAX_LAYERS) : 0;

    float rawWeight[MAX_LAYERS];
    float visibleWeight[MAX_LAYERS];
    for (uint z = 0; z < MAX_LAYERS; z++) { rawWeight[z] = 0; visibleWeight[z] = 0; }

    if (layerCount > 0)
    {
        float4 paintWeights[8];
        PaintWeights(paint, terrainUV, ClampSampler, paintWeights);
        uint sliceCount = min((layerCount + 3) / 4, 8u);
        for (uint si = 0; si < sliceCount; si++)
        {
            float4 weights = paintWeights[si];
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
    for (uint d0 = 0; d0 < MAX_DECORATORS; d0++) coverage[d0] = 0;

    // The rows here: those of the last Add stamp that holds the texel (weight over a half) for its decorator. A
    // later stamp of that decorator without rows scatters it again.
    rowIdx = 0;
    rowSlot = COVERAGE_ALL_TARGETS;

    // A page bake goes only through the stamps whose box reaches its page (the others weigh 0 there), in their order
    paged = paged && PageShapeIdx != 0;
    uint count = paged ? PageStampCount : StampCount;

    float shapeWeight = 0;   // the shape weight of the last stamp that was not COVERAGE_SAME_SHAPE
    for (uint n = 0; n < count; n++)
    {
        uint i = paged ? pageList[PageStampOffset + n] : n;
        CoverageStamp stamp = stamps[i];

        if ((stamp.Flags & COVERAGE_SAME_SHAPE) == 0)
        {
            if (paged) shapeWeight = CoverageShapeWeightPage(stamp, pageShapes[PageShapeBase + i], terrainUV, sampleRow, splinePoints, pageList);
            else shapeWeight = CoverageShapeWeight(stamp, terrainUV, splinePoints);
        }
        float weight = shapeWeight;
        if (weight <= 0) continue;

        if (stamp.Flags & COVERAGE_FILTER)
        {
            weight *= CoverageTerrainFilter(stamp, heightNorm, slopeDeg, terrainUV * TerrainSizeX);   // square terrain

            if ((stamp.RequireMask | stamp.ExcludeMask) != 0)
            {
                float required = 0, excluded = 0;
                for (uint c = 0; c < layerCount; c++)
                {
                    uint bit = 1u << c;
                    if (stamp.RequireMask & bit) required += visibleWeight[c];      // together: grass or dry grass covers it all
                    if (stamp.ExcludeMask & bit) excluded = max(excluded, rawWeight[c]);
                }
                if (stamp.RequireMask != 0) weight *= saturate(required);
                weight *= (1.0 - excluded);
            }

            if (weight <= 0) continue;
        }

        if (CoverageOp(stamp) == DECO_OP_ADD)
        {
            if (stamp.Target < decoCount)
            {
                coverage[stamp.Target] = max(coverage[stamp.Target], saturate(weight * stamp.Strength));
                if (weight * stamp.Strength >= 0.5 && (stamp.Rows != 0 || stamp.Target == rowSlot))
                {
                    rowIdx = stamp.Rows;
                    rowSlot = stamp.Rows != 0 ? stamp.Target : COVERAGE_ALL_TARGETS;
                }
            }
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
}

// The top 8 decorators by weight (descending), packed (decoratorSlot << 8) | weight; the rows' marker takes the first free
// channel, or the lightest decorator's
void PackTop8(float coverage[MAX_DECORATORS], uint decoCount, uint rowIdx, uint rowSlot, out uint4 packed0, out uint4 packed1)
{
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

    if (rowIdx != 0 && rowSlot < decoCount && coverage[rowSlot] > 0)
    {
        uint at = min(topCount, 7u);
        topIdx[at] = DECO_ROW_MARKER;
        topWt[at] = rowIdx;
    }

    [unroll] for (uint ch = 0; ch < 4; ch++)
    {
        packed0[ch] = (topIdx[ch] << 8) | topWt[ch];
        packed1[ch] = (topIdx[ch + 4] << 8) | topWt[ch + 4];
    }
}

// One page of the decoration pool, border included: texel t of the slot is level-L texel page * 256 + t - border. A level
// 0 texel is its centre's coverage; a coarser one averages each decorator's coverage over a 4 x 4 grid of points on its
// footprint (2^L metres a side) and keeps the top 8. Crop rows only at level 0 (only the spawn reads them, near).
[numthreads(8, 8, 1)]
void CSBuildDecoPage(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.y >= (RowCount != 0 ? RowCount : DECO_SLOT_TEXELS)) return;
    dtid.y += RowStart;
    if (dtid.x >= DECO_SLOT_TEXELS || dtid.y >= DECO_SLOT_TEXELS) return;
    RWTexture2DArray<uint4> controlTex = ResourceDescriptorHeap[ControlUAVIdx];

    uint decoCount = min(DecoratorCount, (uint)MAX_DECORATORS);
    float2 j = float2(int2(PageX, PageY) * DECO_PAGE_TEXELS + int2(dtid.xy) - DECO_BORDER);   // level L texel
    float invTexels = 1.0 / float(Resolution);

    float coverage[MAX_DECORATORS];
    uint rowIdx = 0, rowSlot = COVERAGE_ALL_TARGETS;
    if (Level == 0)
        DecoCoverage((j + 0.5) * invTexels, decoCount, true, dtid.y, coverage, rowIdx, rowSlot);
    else
    {
        const uint S = 4;
        float span = float(1u << Level);
        for (uint z = 0; z < MAX_DECORATORS; z++) coverage[z] = 0;
        for (uint b = 0; b < S; b++)
            for (uint a = 0; a < S; a++)
            {
                float2 c = (j + (float2(a, b) + 0.5) / S) * span;   // level 0 texels
                float sub[MAX_DECORATORS];
                uint subRow, subSlot;
                DecoCoverage(c * invTexels, decoCount, true, dtid.y * S + b, sub, subRow, subSlot);
                for (uint d = 0; d < decoCount; d++) coverage[d] += sub[d];
            }
        for (uint d2 = 0; d2 < decoCount; d2++) coverage[d2] *= 1.0 / float(S * S);
    }

    uint4 packed0, packed1;
    PackTop8(coverage, decoCount, rowIdx, rowSlot, packed0, packed1);
    uint2 at = uint2(OutX, OutY) + dtid.xy;
    controlTex[uint3(at, OutSlice)] = packed0;
    controlTex[uint3(at, OutSlice + 1)] = packed1;
}

// The whole-map bake before 5c (Y-flipped against terrain UV, row 0 at the far edge), over a region, written from the
// output's corner: only Check Deco Pages bakes it now, to compare every level 0 page with it to the texel
[numthreads(8, 8, 1)]
void CSBuildDecoControl(uint3 groupTexel : SV_DispatchThreadID)
{
    if (groupTexel.x >= RegionW || groupTexel.y >= RegionH) return;
    uint2 texel = groupTexel.xy + uint2(RegionX, RegionY);
    if (texel.x >= Resolution || texel.y >= Resolution) return;
    RWTexture2DArray<uint4> controlTex = ResourceDescriptorHeap[ControlUAVIdx];

    // Row y holds level 0 texel row Resolution - 1 - y: its place computed as the page kernel computes it, so the two
    // agree to the bit (before 5c: 1 - (y + 0.5) / Resolution, the same place within a float's last bit)
    float2 j = float2(texel.x, Resolution - 1 - texel.y);
    uint decoCount = min(DecoratorCount, (uint)MAX_DECORATORS);
    float coverage[MAX_DECORATORS];
    uint rowIdx, rowSlot;
    DecoCoverage((j + 0.5) * (1.0 / float(Resolution)), decoCount, false, 0, coverage, rowIdx, rowSlot);

    uint4 packed0, packed1;
    PackTop8(coverage, decoCount, rowIdx, rowSlot, packed0, packed1);
    controlTex[uint3(groupTexel.xy, 0)] = packed0;
    controlTex[uint3(groupTexel.xy, 1)] = packed1;
}
