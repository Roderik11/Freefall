// terrain_holes.hlsli — where HoleStamps open the terrain (gputerrain.fx draws nothing there, grass_compute.hlsl
// spawns nothing).
// The mask is baked on the CPU (TerrainBaker.BakeHoles) with one texel per height sample: texel (x, z) is the sample
// at terrain-local (x, z) * size / (res - 1), the PhysX height field's mapping, so the edge drawn here is the one the
// collider and the navmesh leave out (Terrain.HoleWeight reads it the same way). Bilinear; open at 0.5 and above.

#ifndef TERRAIN_HOLES_HLSLI
#define TERRAIN_HOLES_HLSLI

bool TerrainHoleAt(uint holeTexIdx, SamplerState bilinearClamp, float2 terrainLocal, float2 terrainSize)
{
    if (holeTexIdx == 0) return false;

    Texture2D<float> holes = ResourceDescriptorHeap[holeTexIdx];
    uint w, h;
    holes.GetDimensions(w, h);
    float2 uv = (terrainLocal / terrainSize * (float(w) - 1.0) + 0.5) / float(w);
    return holes.SampleLevel(bilinearClamp, uv, 0) >= 0.5;
}

#endif // TERRAIN_HOLES_HLSLI
