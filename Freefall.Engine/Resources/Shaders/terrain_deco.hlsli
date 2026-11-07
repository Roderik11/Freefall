#ifndef TERRAIN_DECO_HLSLI
#define TERRAIN_DECO_HLSLI

// The terrain's decoration control (the top 8 decorators per texel, one texel a metre) as pages (TERRAIN_PAGES_PLAN.md 5c,
// 2026-11-07). The only place that reads the decoration pool: the spawn (grass_compute.hlsl) and the terrain's sward tint
// (gputerrain.fx) go through these functions.
//
// A page is 256 texels a side inside a border of 1 (258), four to a pair of slices of one Texture2DArray RGBA16_UINT (2 x 2,
// 516 x 516); a slot's two slices are array slices (slot / 4) * 2 and + 1, each channel (decorator slot << 8) | coverage
// (decoration_prepass.hlsl). The page table names, per level 0 page, the finest page resident over it (slot | level << 24):
// away from the camera a coarser level's, each texel each decorator's coverage averaged over its 2^level x 2^level metres.
// Pages lie in terrain UV with +Z up, texel i's centre at (i + 0.5) / Texels (the whole-map texture before 5c was
// Y-flipped). C#: Graphics.DecoPool. Its params come by push constant, not through TerrainPagesParams.

#define TERRAIN_DECO_PAGE_TEXELS  256
#define TERRAIN_DECO_BORDER       1
#define TERRAIN_DECO_SLOT_TEXELS  258

// One per terrain and frame (DecoPool.WriteParams, C# DecoPool.Params). 32 bytes.
struct TerrainDecoParams
{
    uint  Control;       // Texture2DArray<uint4> RGBA16_UINT: two slices a slot
    uint  PageTable;     // StructuredBuffer<uint>: per level 0 page, slot (low 24 bits) and level
    uint  PagesAcross;   // level 0 pages a side
    uint  Texels;        // texels a side of the whole map at level 0
    uint4 _pad;
};

TerrainDecoParams LoadDeco(uint idx)
{
    if (idx == 0)
        return (TerrainDecoParams)0;
    StructuredBuffer<TerrainDecoParams> buf = ResourceDescriptorHeap[idx];
    return buf[0];
}

bool DecoPresent(TerrainDecoParams p) { return p.Control != 0 && p.Texels != 0; }

// The page resident over a level 0 texel: its slot's first slice, its level, and where its level-L texel 0 lies in the
// slice (texel k of that level at origin + k; the border at origin - 1 and origin + 256)
struct DecoPage
{
    uint  slice;
    uint  level;
    int2  origin;
    int2  lo, hi;        // the slot's texels, border included: loads are clamped to them
};

DecoPage DecoPageAt(TerrainDecoParams p, int2 texel0)
{
    int2 t = clamp(texel0, 0, int(p.Texels) - 1);
    uint2 page0 = min(uint2(t) / TERRAIN_DECO_PAGE_TEXELS, p.PagesAcross - 1);
    StructuredBuffer<uint> table = ResourceDescriptorHeap[p.PageTable];
    uint e = table[page0.y * p.PagesAcross + page0.x];
    uint slot = e & 0xFFFFFFu;
    DecoPage pg;
    pg.level = e >> 24;
    pg.slice = (slot >> 2) * 2;
    int2 quarter = int2(uint2(slot & 1u, (slot >> 1) & 1u) * TERRAIN_DECO_SLOT_TEXELS);
    pg.origin = quarter + TERRAIN_DECO_BORDER - int2((page0 >> pg.level) * TERRAIN_DECO_PAGE_TEXELS);
    pg.lo = quarter;
    pg.hi = quarter + TERRAIN_DECO_SLOT_TEXELS - 1;
    return pg;
}

// A texel of the page at its own level (clamped to the slot, border included)
void DecoLoad(TerrainDecoParams p, DecoPage pg, int2 levelTexel, out uint4 s0, out uint4 s1)
{
    Texture2DArray<uint4> pool = ResourceDescriptorHeap[p.Control];
    int2 at = clamp(pg.origin + levelTexel, pg.lo, pg.hi);
    s0 = pool.Load(int4(at, pg.slice, 0));
    s1 = pool.Load(int4(at, pg.slice + 1, 0));
}

// The texel over a level 0 texel (terrain UV x Texels, +Z up), from the finest resident page: nothing (255 << 8 in every
// channel) without a pool
void DecoTexel(TerrainDecoParams p, int2 texel0, out uint4 s0, out uint4 s1)
{
    if (!DecoPresent(p))
    {
        s0 = s1 = (255u << 8).xxxx;
        return;
    }
    DecoPage pg = DecoPageAt(p, texel0);
    DecoLoad(p, pg, clamp(texel0, 0, int(p.Texels) - 1) >> pg.level, s0, s1);
}

#endif // TERRAIN_DECO_HLSLI
