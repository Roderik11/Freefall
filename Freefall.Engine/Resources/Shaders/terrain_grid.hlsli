#ifndef TERRAIN_GRID_HLSLI
#define TERRAIN_GRID_HLSLI

// Where a terrain's height map holds the height of a place (TERRAIN_PAGES_PLAN.md stage 1, 2026-11-07). One sample on
// every vertex: sample i lies at i x Terrain.VertexSpacing from the terrain's corner, so a map N samples across spans
// N - 1 cells and its first and last samples lie on the terrain's edges. The sea mask (C# SeaMask) is laid out the same
// way. Paint and decoration maps are not: their texels sit at the middle of their cells, read at the terrain UV itself.
//
// 'terrainUV' is the place in terrain UV, (world - origin) / size, the space stamps and paint use; 'samples' is the
// map's size (GetDimensions). Until this date the bake put sample i at (i + 0.5) / N and every reader sampled at the
// terrain UV, so the vertices fell between samples (up to half a sample off towards the edges).
float2 TerrainHeightUV(float2 terrainUV, float2 samples)
{
    return (terrainUV * (samples - 1.0) + 0.5) / samples;
}

// Metres between height samples, from the terrain's size and the map's
float2 TerrainSampleSpacing(float2 terrainSize, float2 samples)
{
    return terrainSize / (samples - 1.0);
}

#endif // TERRAIN_GRID_HLSLI
