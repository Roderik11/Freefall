#ifndef TERRAIN_PAGES_HLSLI
#define TERRAIN_PAGES_HLSLI

// The terrain's heights as pages (TERRAIN_PAGES_PLAN.md stage 4a, 2026-11-07). The only place that reads the height pool:
// every shader that wants a height at a place calls these functions, so no terrain read takes a non-uniform descriptor.
//
// A page is Terrain.PageCells (256) cells a side: page p holds samples p·256 .. p·256 + 256, the last one shared with the
// next page, inside a border of 2, so 261 texels. Four pages to a slice of one Texture2DArray (2 × 2, 522 × 522), the
// pool; the page table names the slot each level 0 page lies in. Sample i of the terrain stands at world
// Origin + i · Spacing (terrain_grid.hlsli). C#: Graphics.HeightPool.

#define TERRAIN_PAGE_CELLS   256
#define TERRAIN_PAGE_BORDER  2
#define TERRAIN_PAGE_TEXELS  261
#define TERRAIN_SLICE_TEXELS 522

// One per terrain and frame (HeightPool.WriteParams, C# HeightPool.Params). 64 bytes.
struct TerrainPagesParams
{
    uint   HeightPool;      // Texture2DArray<float>, R16 normalized to MaxHeight; 0 = no terrain
    uint   PageTable;       // StructuredBuffer<uint>: per level 0 page, row by row, its slot (low 24 bits) and level
    uint   PagesAcross;     // level 0 pages a side
    uint   Cells;           // samples a side − 1
    float3 Origin;          // world position of sample (0, 0); y is the terrain's base, where a height of 0 stands
    float  Spacing;         // metres between samples (Terrain.VertexSpacing)
    float2 Size;            // metres (Terrain.TerrainSize)
    float  MaxHeight;       // metres at 1
    uint   Version;         // changes whenever the heights may have: a bake, an upload (TerrainBaker.HeightVersion)
    float4 _pad;
};

TerrainPagesParams LoadTerrain(uint terrainIdx)
{
    if (terrainIdx == 0)
        return (TerrainPagesParams)0;
    StructuredBuffer<TerrainPagesParams> buf = ResourceDescriptorHeap[terrainIdx];
    return buf[0];
}

bool TerrainPresent(TerrainPagesParams t) { return t.HeightPool != 0; }

// The pool texel (x, y, slice) of sample i, clamped to the terrain
uint3 TerrainSampleTexel(TerrainPagesParams t, int2 i)
{
    uint2 s = (uint2)clamp(i, 0, (int)t.Cells);
    uint2 page = min(s / TERRAIN_PAGE_CELLS, t.PagesAcross - 1);
    StructuredBuffer<uint> table = ResourceDescriptorHeap[t.PageTable];
    uint slot = table[page.y * t.PagesAcross + page.x] & 0xFFFFFFu;
    uint q = slot & 3u;
    uint2 local = s - page * TERRAIN_PAGE_CELLS;   // 0 .. 256 (the last page's last sample included)
    return uint3(uint2(q & 1u, q >> 1) * TERRAIN_PAGE_TEXELS + TERRAIN_PAGE_BORDER + local, slot >> 2);
}

// Sample i of the heights, normalized to MaxHeight (0 .. 1), clamped to the terrain
float TerrainSample01(TerrainPagesParams t, int2 i)
{
    if (t.HeightPool == 0) return 0;
    Texture2DArray<float> pool = ResourceDescriptorHeap[t.HeightPool];
    return pool.Load(int4(TerrainSampleTexel(t, i), 0));
}

// The height at a place given in samples (sample i at i), normalized to MaxHeight: bilinear between the four samples
// round it, weighed in full float (the same on every GPU and on the CPU; the texture filter's weights have 8 bits).
// Clamped to the terrain's edges.
float TerrainHeight01AtSample(TerrainPagesParams t, float2 s)
{
    if (t.HeightPool == 0) return 0;   // no terrain, or its pool not made yet
    s = clamp(s, 0.0, (float)t.Cells);
    // The four samples lie in one page: the cell's left sample is at most 255 into it
    uint2 page = min(uint2(s) / TERRAIN_PAGE_CELLS, t.PagesAcross - 1);
    float2 local = s - float2(page * TERRAIN_PAGE_CELLS);
    uint2 i0 = min(uint2(local), TERRAIN_PAGE_CELLS - 1);
    float2 f = local - float2(i0);

    StructuredBuffer<uint> table = ResourceDescriptorHeap[t.PageTable];
    uint slot = table[page.y * t.PagesAcross + page.x] & 0xFFFFFFu;
    uint q = slot & 3u;
    int3 texel = int3(uint2(q & 1u, q >> 1) * TERRAIN_PAGE_TEXELS + TERRAIN_PAGE_BORDER + i0, slot >> 2);

    Texture2DArray<float> pool = ResourceDescriptorHeap[t.HeightPool];
    float h00 = pool.Load(int4(texel, 0));
    float h10 = pool.Load(int4(texel + int3(1, 0, 0), 0));
    float h01 = pool.Load(int4(texel + int3(0, 1, 0), 0));
    float h11 = pool.Load(int4(texel + int3(1, 1, 0), 0));
    return lerp(lerp(h00, h10, f.x), lerp(h01, h11, f.x), f.y);
}

// The height at a place, normalized to MaxHeight (TerrainHeight01AtSample)
float TerrainHeight01(TerrainPagesParams t, float2 worldXZ)
{
    return TerrainHeight01AtSample(t, (worldXZ - t.Origin.xz) / t.Spacing);
}

// The world Y of the ground at a place
float TerrainHeight(TerrainPagesParams t, float2 worldXZ)
{
    return t.Origin.y + TerrainHeight01(t, worldXZ) * t.MaxHeight;
}

#endif // TERRAIN_PAGES_HLSLI
