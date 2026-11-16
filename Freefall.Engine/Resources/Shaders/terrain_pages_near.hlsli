#ifndef TERRAIN_PAGES_NEAR_HLSLI
#define TERRAIN_PAGES_NEAR_HLSLI

// Several heights round one place (a normal, the ground's shape), the same to the bit as TerrainHeight01AtSample
// (terrain_pages.hlsli) for fewer reads: that takes a page-table read and four loads per height, these one gather per
// height and one table read for every height in the same level 0 page as the place the cursor was made for. Any sampler:
// a gather does not filter. Kept out of terrain_pages.hlsli, which the height bake's shader hash covers
// (TerrainBaker.ShaderHash): a change there bakes every terrain's heights again on its next load.

#include "terrain_pages.hlsli"

struct TerrainPageCursor
{
    uint2 page0;    // the level 0 page it was looked up for
    TerrainPageRef page;
};

TerrainPageCursor TerrainCursorAt(TerrainPagesParams t, float2 s)
{
    s = clamp(s, 0.0, (float)t.Cells);
    TerrainPageCursor c;
    c.page0 = min(uint2(s) / TERRAIN_PAGE_CELLS, t.PagesAcross - 1);
    c.page = TerrainPageAt(t, c.page0);
    return c;
}

// TerrainHeight01AtSample through a cursor: its page when s lies in the same level 0 page, else the table's
float TerrainHeight01Near(TerrainPagesParams t, TerrainPageCursor c, float2 s, SamplerState anySampler)
{
    if (t.HeightPool == 0) return 0;
    s = clamp(s, 0.0, (float)t.Cells);
    uint2 page0 = min(uint2(s) / TERRAIN_PAGE_CELLS, t.PagesAcross - 1);
    TerrainPageRef p = c.page;
    if (any(page0 != c.page0))
        p = TerrainPageAt(t, page0);
    float2 corner = float2((page0 >> p.level) * (uint(TERRAIN_PAGE_CELLS) << p.level));
    float2 local = (s - corner) / float(1u << p.level);
    uint2 i0 = min(uint2(local), TERRAIN_PAGE_CELLS - 1);
    float2 f = local - float2(i0);
    int3 texel = TerrainSlotTexel(p.slot, i0);

    // The 2 x 2 texels from `texel`, gathered round the corner they share: x (0, 1), y (1, 1), z (1, 0), w (0, 0)
    Texture2DArray<float> pool = ResourceDescriptorHeap[t.HeightPool];
    float4 g = pool.GatherRed(anySampler, float3((float2(texel.xy) + 1.0) / TERRAIN_SLICE_TEXELS, (float)texel.z));
    return lerp(lerp(g.w, g.z, f.x), lerp(g.x, g.y, f.x), f.y);
}

#endif // TERRAIN_PAGES_NEAR_HLSLI
