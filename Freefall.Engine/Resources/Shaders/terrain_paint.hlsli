#ifndef TERRAIN_PAINT_HLSLI
#define TERRAIN_PAINT_HLSLI

// The terrain's paint (the layers' blend weights) as pages (TERRAIN_PAGES_PLAN.md 5b, 2026-11-07). The only place that reads
// the paint pool: the terrain's pixel shader, the decoration prepass and the plants' tint go through these functions.
//
// A page is 256 paint texels a side inside a border of 4 (264), four to a slice of one Texture2DArray RGBA8 (2 x 2,
// 528 x 528); a slot's layer slice g (4 layers each) is array slice (slot / 4) * LayerSlices + g. The page table names, per
// level 0 page, the finest page resident over it (slot | level << 24): away from the camera a coarser level's, each texel
// the average of 2^level x 2^level level 0 texels. Pages lie in terrain UV with +Z up, paint texel i's centre at
// (i + 0.5) / Texels (the whole-map array before 5b was Y-flipped). C#: Graphics.PaintPool.

#include "terrain_pages.hlsli"

#define TERRAIN_PAINT_PAGE_TEXELS  256
#define TERRAIN_PAINT_BORDER       4
#define TERRAIN_PAINT_SLOT_TEXELS  264
#define TERRAIN_PAINT_SLICE_TEXELS 528

// One per terrain and frame (PaintPool.WriteParams, C# PaintPool.Params). 64 bytes.
struct TerrainPaintParams
{
    uint   Weights;         // Texture2DArray<float4> RGBA8: the layers' raw weights, 4 a slice
    uint   TopLayer;        // Texture2DArray<uint> R8: the channel that shows most, 255 = none
    uint   PageTable;       // StructuredBuffer<uint>: per level 0 page, slot (low 24 bits) and level
    uint   PagesAcross;     // level 0 pages a side
    uint   Texels;          // paint texels a side of the whole map at level 0
    uint   LayerSlices;     // array slices of a slot's weights
    uint   LayerCount;
    uint   _pad0;
    float4 _pad1;
    float4 _pad2;
};

TerrainPaintParams LoadPaint(TerrainPagesParams t)
{
    if (t.PaintIdx == 0)
        return (TerrainPaintParams)0;
    StructuredBuffer<TerrainPaintParams> buf = ResourceDescriptorHeap[t.PaintIdx];
    return buf[0];
}

bool PaintPresent(TerrainPaintParams p) { return p.Weights != 0 && p.LayerSlices != 0; }

// Where a place (terrain UV, +Z up) lies in the paint pool
struct PaintPlace
{
    float2 uv;          // in the slice, 0..1 over TERRAIN_PAINT_SLICE_TEXELS
    uint   slice0;      // the array slice of the page's first layer slice
    uint   topSlice;    // its top layer's array slice
    float  uvPerTexel;  // slice uv per level 0 paint texel: 1 / (2^level * TERRAIN_PAINT_SLICE_TEXELS)
    uint   level;
};

PaintPlace PaintAt(TerrainPaintParams p, float2 terrainUV)
{
    float2 c = clamp(terrainUV * p.Texels, 0.0, (float)p.Texels);   // level 0 texels: texel i's centre at i + 0.5
    uint2 page0 = min(uint2(c) / TERRAIN_PAINT_PAGE_TEXELS, p.PagesAcross - 1);
    StructuredBuffer<uint> table = ResourceDescriptorHeap[p.PageTable];
    uint e = table[page0.y * p.PagesAcross + page0.x];
    uint slot = e & 0xFFFFFFu, level = e >> 24;
    float inv = 1.0 / float(1u << level);
    // Texels into the page at its level (its texel k's centre at k + 0.5), then into its quarter of the slice
    float2 local = c * inv - float2((page0 >> level) * TERRAIN_PAINT_PAGE_TEXELS);
    uint q = slot & 3u;
    float2 texel = float2(uint2(q & 1u, q >> 1) * TERRAIN_PAINT_SLOT_TEXELS) + TERRAIN_PAINT_BORDER + local;
    PaintPlace r;
    r.uv = texel / TERRAIN_PAINT_SLICE_TEXELS;
    r.slice0 = (slot >> 2) * p.LayerSlices;
    r.topSlice = slot >> 2;
    r.uvPerTexel = inv / TERRAIN_PAINT_SLICE_TEXELS;
    r.level = level;
    return r;
}

// The layers' raw weights at a place (4 a slice, up to 8 slices), bilinear. 0 without paint.
uint PaintWeights(TerrainPaintParams p, float2 terrainUV, SamplerState s, out float4 w[8])
{
    [unroll] for (uint z = 0; z < 8; z++) w[z] = float4(0, 0, 0, 0);
    if (!PaintPresent(p)) return 0;
    uint slices = min(p.LayerSlices, 8u);
    PaintPlace at = PaintAt(p, terrainUV);
    Texture2DArray<float4> pool = ResourceDescriptorHeap[p.Weights];
    [unroll] for (uint i = 0; i < 8; i++)
        if (i < slices) w[i] = pool.SampleLevel(s, float3(at.uv, at.slice0 + i), 0);
    return slices;
}

// The same with the screen's derivatives of terrain UV, for an anisotropic sampler (the terrain's pixel shader): scaled
// into the page, so the footprint is what it was on the whole-map array, but no longer than twice the border, so the
// taps stay on the page (at a grazing view a footprint runs to tens of texels; the pool has one mip, as the array had)
uint PaintWeightsGrad(TerrainPaintParams p, float2 terrainUV, float2 dUVdx, float2 dUVdy, SamplerState s, out float4 w[8])
{
    [unroll] for (uint z = 0; z < 8; z++) w[z] = float4(0, 0, 0, 0);
    if (!PaintPresent(p)) return 0;
    uint slices = min(p.LayerSlices, 8u);
    PaintPlace at = PaintAt(p, terrainUV);
    float k = p.Texels * at.uvPerTexel;
    float2 gx = dUVdx * k, gy = dUVdy * k;
    const float most = 2.0 * TERRAIN_PAINT_BORDER / TERRAIN_PAINT_SLICE_TEXELS;
    gx *= min(1.0, most / max(length(gx), 1e-12));
    gy *= min(1.0, most / max(length(gy), 1e-12));
    Texture2DArray<float4> pool = ResourceDescriptorHeap[p.Weights];
    [unroll] for (uint i = 0; i < 8; i++)
        if (i < slices) w[i] = pool.SampleGrad(s, float3(at.uv, at.slice0 + i), gx, gy);
    return slices;
}

#endif // TERRAIN_PAINT_HLSLI
