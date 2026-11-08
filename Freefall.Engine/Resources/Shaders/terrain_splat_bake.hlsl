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

// Since 5b (TERRAIN_PAGES_PLAN.md, 2026-11-07) the paint is baked page by page into the paint pool (terrain_paint.hlsli):
// CS_SplatBakePage bakes one page into its slot. CS_SplatBakeFlat is the whole-map bake there was before, into a flat
// array, kept only for Check Paint Pages (every level 0 page equal to it, to the texel). Both composite through Composite().

#pragma kernel CS_SplatBakePage
#pragma kernel CS_SplatBakeFlat

#include "terrain_stamp_common.hlsli"
#include "terrain_coverage.hlsli"

cbuffer PushConstants : register(b3)
{
    uint  StampBufIdx;      // slot 0 — SRV: StructuredBuffer<CoverageStamp>, sorted by priority
    uint  OutputIdx;        // slot 1 — UAV: RWTexture2DArray<float4>, the paint pool's weights (flat: the check's array)
    uint  TerrainPagesIdx;  // slot 2 — the terrain's TerrainPagesParams (terrain_pages.hlsli); 0 = none yet: height/slope read as 0
    uint  SplineBufIdx;     // slot 3 — SRV: StructuredBuffer<StampSplinePoint>
    uint  StampCount;       // slot 4
    uint  Resolution;       // slot 5 — paint texels a side of the whole map at level 0
    uint  SliceCount;       // slot 6
    uint  LayerCount;       // slot 7
    float MaxHeight;        // slot 8
    float TerrainSizeX;     // slot 9
    uint  TopLayerIdx;      // slot 10 — UAV: RWTexture2DArray<uint>, the channel that shows most per texel (0 = none)
    uint  RegionX;          // slot 11 — flat: the texels this bake redoes, from (RegionX, RegionY), RegionW x RegionH
    uint  RegionY;          // slot 12
    uint  RegionW;          // slot 13
    uint  RegionH;          // slot 14
    int   PageX;            // slot 15 — page: which page, at which level
    int   PageY;            // slot 16
    uint  Level;            // slot 17
    uint  OutX;             // slot 18 — page: its slot's quarter corner in the slice (texels)
    uint  OutY;             // slot 19
    uint  OutSlice;         // slot 20 — page: the array slice of its first layer slice
    uint  OutTopSlice;      // slot 21 — page: its top layer's array slice
    uint  PageShapeIdx;     // slot 22 — page: SRV StructuredBuffer<PageShape>, a stamp's segments per page (terrain_coverage.hlsli)
    uint  PageListIdx;      // slot 23 — page: SRV StructuredBuffer<uint>, the segment lists and row parities they point into
    uint  PageShapeBase;    // slot 24 — page: this page's first PageShape (one per stamp)
    uint  PageStampOffset;  // slot 25 — page: the stamps whose box reaches the page, in order, in the uint list from here
    uint  PageStampCount;   // slot 26
    uint  RowStart;         // slot 27 — page: the slot's texel rows this dispatch bakes, from RowStart (a coarse page goes
    uint  RowCount;         // slot 28     over several frames, TerrainMapResidency); 0 rows = the whole slot
};

#define PAINT_PAGE_TEXELS 256
#define PAINT_BORDER      4
#define PAINT_SLOT_TEXELS 264

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

// Every stamp at a place (terrain UV), over the heights there: each layer's raw weight, as the stack holds them. paged:
// a page bake's sample, whose spline stamps walk only the page's segments (sampleRow: its row in the page)
void Composite(float2 terrainUV, uint layerCount, bool paged, uint sampleRow, out float raw[MAX_LAYERS])
{
    StructuredBuffer<CoverageStamp> stamps = ResourceDescriptorHeap[StampBufIdx];
    StructuredBuffer<StampSplinePoint> splinePoints = ResourceDescriptorHeap[SplineBufIdx];
    StructuredBuffer<PageShape> pageShapes = ResourceDescriptorHeap[PageShapeIdx];
    StructuredBuffer<uint> pageList = ResourceDescriptorHeap[PageListIdx];

    float heightNorm = 0, slopeDeg = 0;
    if (TerrainPagesIdx != 0)
        SampleTerrainHeightSlope(LoadTerrain(TerrainPagesIdx), terrainUV, heightNorm, slopeDeg);

    for (uint z = 0; z < MAX_LAYERS; z++) raw[z] = 0;

    // A page bake goes only through the stamps whose box reaches its page (the others weigh 0 there), in their order
    paged = paged && PageShapeIdx != 0;
    uint count = paged ? PageStampCount : StampCount;

    float shapeWeight = 0;   // the shape weight of the last stamp that was not COVERAGE_SAME_SHAPE
    for (uint n = 0; n < count; n++)
    {
        uint i = paged ? pageList[PageStampOffset + n] : n;
        CoverageStamp stamp = stamps[i];
        // Before the target test, so a rule after a skipped one still reads its own shape's weight
        if ((stamp.Flags & COVERAGE_SAME_SHAPE) == 0)
        {
            if (paged) shapeWeight = CoverageShapeWeightPage(stamp, pageShapes[PageShapeBase + i], terrainUV, sampleRow, splinePoints, pageList);
            else shapeWeight = CoverageShapeWeight(stamp, terrainUV, splinePoints);
        }
        if (stamp.Target >= layerCount) continue;

        float weight = shapeWeight;
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
}

// What the ground is: the channel with the greatest visible weight, stacked as the surface shader stacks them. 255 =
// nothing painted. For whoever asks on the CPU (TerrainRenderer.GetLayer).
uint TopOf(float raw[MAX_LAYERS], uint layerCount)
{
    uint top = 255;
    float most = 0.0, cover = 1.0;
    for (int c = int(layerCount) - 1; c >= 0; c--)
    {
        float visible = raw[c] * cover;
        if (visible > most) { most = visible; top = uint(c); }
        cover *= (1.0 - raw[c]);
    }
    return top;
}

// One page of the paint pool, border included: texel t of slot texels is level-L texel page * 256 + t - border. A level 0
// texel is its centre's composite; a coarser one averages the visible weights of a 4 x 4 grid of points over its
// footprint (2^L level 0 texels a side) and stacks them back into raw weights, so a road narrower than the texel still
// shows (point-sampled, thin roads broke up, stage 0). Every slice and the top layer are written every bake.
[numthreads(8, 8, 1)]
void CS_SplatBakePage(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.y >= (RowCount != 0 ? RowCount : PAINT_SLOT_TEXELS)) return;
    dtid.y += RowStart;
    if (dtid.x >= PAINT_SLOT_TEXELS || dtid.y >= PAINT_SLOT_TEXELS) return;
    RWTexture2DArray<float4> output = ResourceDescriptorHeap[OutputIdx];

    uint layerCount = min(LayerCount, (uint)MAX_LAYERS);
    float2 j = float2(int2(PageX, PageY) * PAINT_PAGE_TEXELS + int2(dtid.xy) - PAINT_BORDER);   // level L texel
    float span = float(1u << Level);
    float invTexels = 1.0 / float(Resolution);

    float raw[MAX_LAYERS];
    if (Level == 0)
        Composite((j + 0.5) * invTexels, layerCount, true, dtid.y, raw);
    else
    {
        const uint S = 4;
        float visible[MAX_LAYERS];
        for (uint z = 0; z < MAX_LAYERS; z++) visible[z] = 0;
        for (uint b = 0; b < S; b++)
            for (uint a = 0; a < S; a++)
            {
                float2 c = (j + (float2(a, b) + 0.5) / S) * span;   // level 0 texels
                float sub[MAX_LAYERS];
                Composite(c * invTexels, layerCount, true, dtid.y * S + b, sub);
                float cover = 1.0;
                for (int k = int(layerCount) - 1; k >= 0; k--)
                {
                    visible[k] += sub[k] * cover;
                    cover *= (1.0 - sub[k]);
                }
            }
        // Back into the stack: raw[k] = visible[k] / what the layers above leave
        float remaining = 1.0;
        for (int k2 = int(MAX_LAYERS) - 1; k2 >= 0; k2--)
        {
            float v = uint(k2) < layerCount ? visible[k2] / float(S * S) : 0.0;
            raw[k2] = remaining > 1e-5 ? saturate(v / remaining) : 0.0;
            remaining = max(remaining - v, 0.0);
        }
    }

    uint2 at = uint2(OutX, OutY) + dtid.xy;
    uint sliceCount = min(SliceCount, 8u);
    for (uint s = 0; s < sliceCount; s++)
        output[uint3(at, OutSlice + s)] = float4(raw[s * 4], raw[s * 4 + 1], raw[s * 4 + 2], raw[s * 4 + 3]);
    if (TopLayerIdx != 0)
    {
        RWTexture2DArray<uint> topLayer = ResourceDescriptorHeap[TopLayerIdx];
        topLayer[uint3(at, OutTopSlice)] = TopOf(raw, layerCount);
    }
}

// The whole-map bake before 5b (Y-flipped against terrain UV, row 0 at the far edge), over a region, written from the
// output's corner: only Check Paint Pages bakes it now, to compare every level 0 page with it to the texel
[numthreads(8, 8, 1)]
void CS_SplatBakeFlat(uint3 groupTexel : SV_DispatchThreadID)
{
    if (groupTexel.x >= RegionW || groupTexel.y >= RegionH) return;
    uint2 texel = groupTexel.xy + uint2(RegionX, RegionY);
    if (texel.x >= Resolution || texel.y >= Resolution) return;

    RWTexture2DArray<float4> output = ResourceDescriptorHeap[OutputIdx];
    // Row y holds level 0 texel row Resolution - 1 - y: its place computed as the page kernel computes it, so the two
    // agree to the bit (before 5b: 1 - (y + 0.5) / Resolution, the same place within a float's last bit)
    float2 j = float2(texel.x, Resolution - 1 - texel.y);
    float2 terrainUV = (j + 0.5) * (1.0 / float(Resolution));

    uint layerCount = min(LayerCount, (uint)MAX_LAYERS);
    float raw[MAX_LAYERS];
    Composite(terrainUV, layerCount, false, 0, raw);

    // Written from the output's corner: the check bakes one page's region at a time into an array of a page's size
    uint sliceCount = min(SliceCount, 8u);
    for (uint s = 0; s < sliceCount; s++)
        output[uint3(groupTexel.xy, s)] = float4(raw[s * 4], raw[s * 4 + 1], raw[s * 4 + 2], raw[s * 4 + 3]);
}
