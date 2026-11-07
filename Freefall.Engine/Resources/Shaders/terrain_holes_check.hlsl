// terrain_holes_check.hlsl — Check Hole Pages (TerrainRenderer button): the hole pages (terrain_holes.hlsli TerrainHoleAt)
// against the whole-map R8 texture there was before 2026-11-08, read the old way, at Steps x Steps points per cell over the
// whole terrain. Counts the points where they disagree and where the old map is open.
// SM 6.6 bindless, push constants at b3.

#pragma kernel CS_CheckHoles

#include "terrain_holes.hlsli"

cbuffer PushConstants : register(b3)
{
    uint  HoleParamsIdx;   // slot 0 — the pool's TerrainHoleParams
    uint  FlatIdx;         // slot 1 — Texture2D<float> R8: a texel per sample of the whole terrain (built for the check)
    uint  ResultIdx;       // slot 2 — RWByteAddressBuffer: [0] points that differ, [1] points the flat map opens, [2] points
                           //   that differ where the exact weight is more than 2/256 from the edge (0.5)
    uint  Cells;           // slot 3 — samples a side - 1
    uint  Steps;           // slot 4 — points per cell a side
    float SizeX;           // slot 5
    float SizeZ;           // slot 6
};

SamplerState ClampSampler : register(s2);

// The lookup before 2026-11-08, as it was
bool FlatHoleAt(float2 terrainLocal, float2 terrainSize)
{
    Texture2D<float> holes = ResourceDescriptorHeap[FlatIdx];
    uint w, h;
    holes.GetDimensions(w, h);
    float2 uv = (terrainLocal / terrainSize * (float(w) - 1.0) + 0.5) / float(w);
    return holes.SampleLevel(ClampSampler, uv, 0) >= 0.5;
}

[numthreads(8, 8, 1)]
void CS_CheckHoles(uint3 id : SV_DispatchThreadID)
{
    uint across = Cells * Steps;
    bool inside = id.x < across && id.y < across;
    bool differ = false, open = false, far = false;
    if (inside)
    {
        // Points a little off the cell's grid, so neither sits only on samples nor only on cell middles
        float2 s = (float2(id.xy) + 0.37) / float(Steps);
        float2 size = float2(SizeX, SizeZ);
        float2 local = s / float(Cells) * size;
        bool a = FlatHoleAt(local, size);
        bool b = TerrainHoleAt(HoleParamsIdx, ClampSampler, local, size);
        differ = a != b;
        open = a;
        if (differ)
        {
            // The exact weight at the point: the four samples round it, bilinear in float from the point itself (the
            // hardware filter takes the place in a texel to 1/256, and the flat map's place came from a float near the
            // map's size)
            Texture2D<float> holes = ResourceDescriptorHeap[FlatIdx];
            int2 i0 = min(int2(floor(s)), int(Cells) - 1);
            float2 fr = s - float2(i0);
            float h00 = holes.Load(int3(i0, 0)), h10 = holes.Load(int3(i0 + int2(1, 0), 0));
            float h01 = holes.Load(int3(i0 + int2(0, 1), 0)), h11 = holes.Load(int3(i0 + int2(1, 1), 0));
            float w = lerp(lerp(h00, h10, fr.x), lerp(h01, h11, fr.x), fr.y);
            far = abs(w - 0.5) > 2.0 / 256.0;
        }
    }
    uint nDiffer = WaveActiveCountBits(differ), nOpen = WaveActiveCountBits(open), nFar = WaveActiveCountBits(far);
    if (WaveIsFirstLane())
    {
        RWByteAddressBuffer result = ResourceDescriptorHeap[ResultIdx];
        uint before;
        if (nDiffer != 0) result.InterlockedAdd(0, nDiffer, before);
        if (nOpen != 0) result.InterlockedAdd(4, nOpen, before);
        if (nFar != 0) result.InterlockedAdd(8, nFar, before);
    }
}
