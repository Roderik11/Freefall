// terrain_holes.hlsli — where HoleStamps open the terrain (gputerrain.fx draws nothing there, nor its shadow;
// grass_compute.hlsl spawns nothing).
// The mask is baked on the CPU (TerrainBaker.BakeHoles) with one value per height sample: sample (x, z) at terrain-local
// (x, z) * size / (res - 1), the PhysX height field's mapping, so the edge drawn here is the one the collider and the
// navmesh leave out (Terrain.HoleWeight reads it the same way). Bilinear; open at 0.5 and above.
//
// Since 2026-11-08 the shaders' copy is pages (Graphics.HolePool): only the pages a hole cuts, a 257 x 257 R8 slice each
// (the page's samples, its shared edges included), and a table per page, 0 for none or its slice + 1. A point's page is
// the one whose cells hold it, so the four samples round it are always in that page. Until then one texture held the
// whole terrain, a texel per sample, which D3D12 cannot make past 16,384 a side.

#ifndef TERRAIN_HOLES_HLSLI
#define TERRAIN_HOLES_HLSLI

#define TERRAIN_HOLE_PAGE_CELLS   256
#define TERRAIN_HOLE_PAGE_SAMPLES 257

// One per terrain and frame (HolePool.WriteParams, C# HolePool.Params). 16 bytes.
struct TerrainHoleParams
{
    uint Pool;          // Texture2DArray<float> R8_UNorm: a page a slice
    uint Table;         // StructuredBuffer<uint>: per page, 0 or slice + 1
    uint PagesAcross;
    uint Cells;         // samples a side - 1
};

// holeParamsIdx: the terrain's TerrainHoleParams, 0 = no holes
bool TerrainHoleAt(uint holeParamsIdx, SamplerState bilinearClamp, float2 terrainLocal, float2 terrainSize)
{
    if (holeParamsIdx == 0) return false;

    StructuredBuffer<TerrainHoleParams> holeParams = ResourceDescriptorHeap[holeParamsIdx];
    TerrainHoleParams p = holeParams[0];
    // The sample coordinates, clamped to the terrain as the whole map's clamping sampler held its edge
    float2 s = clamp(terrainLocal / terrainSize * float(p.Cells), 0.0, float(p.Cells));
    uint2 page = min(uint2(s) / TERRAIN_HOLE_PAGE_CELLS, p.PagesAcross - 1);
    StructuredBuffer<uint> table = ResourceDescriptorHeap[p.Table];
    uint entry = table[page.y * p.PagesAcross + page.x];
    if (entry == 0) return false;

    Texture2DArray<float> pool = ResourceDescriptorHeap[p.Pool];
    float2 local = s - float2(page * TERRAIN_HOLE_PAGE_CELLS);   // 0 .. 256: sample k's texel centre at k + 0.5
    float2 uv = (local + 0.5) / float(TERRAIN_HOLE_PAGE_SAMPLES);
    return pool.SampleLevel(bilinearClamp, float3(uv, entry - 1), 0) >= 0.5;
}

#endif // TERRAIN_HOLES_HLSLI
