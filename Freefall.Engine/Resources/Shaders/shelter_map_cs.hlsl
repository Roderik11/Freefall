// The cover map's reduce (Graphics/ShelterMap.cs): one level's top-down depth, drawn at twice the map's resolution,
// into its map: per texel the height of the highest cover over it and the share of the texel it covers, from its 2 × 2
// samples. Drawn twice per level: first the solid meshes (Merge 0, written), then the vegetation (Merge 1, merged in:
// crowns let CrownLeak of the fall through). Cover lower than MinAboveGround over the terrain is none: grass, shrubs,
// crates and benches do not keep the rain and snow off the ground, the snow buries them. Read by weather_cover.hlsli
// WeatherOpenness. Docs: .agent/knowledge/atmospheric_effects/weather.md §2.

#pragma kernel CSReduce

cbuffer PushConstants : register(b3)
{
    uint DepthIdx;              // Texture2DArray<float>: the drawn depth (slice 0), standard Z, 1 = nothing
    uint OutputUAVIdx;          // RWTexture2D<uint>: the level's map, (height << 16) | share
    uint Merge;                 // 0: the solid meshes, written; 1: the vegetation, merged in
    uint Resolution;            // the map's texels a side (the depth has twice as many)
    uint TerrainHeightmap;      // R16 heights; 0 = no terrain (nothing is dropped as low)
    float TerrainMaxHeight;
    float TerrainOriginX;
    float TerrainOriginY;
    float TerrainOriginZ;
    float TerrainSizeX;
    float TerrainSizeZ;
    float Left;                 // world x at the square's left edge
    float Top;                  // world z at its top edge (the map's v 0)
    float Size;                 // metres it spans
    float MinY;                 // the heights drawn: depth 1 at MinY, 0 at MaxY
    float MaxY;
    float MinAboveGround;       // m over the terrain that cover must reach
    float CrownLeak;            // share of the fall that comes through the vegetation
};

SamplerState LinearClamp : register(s2);

uint PackCover(float height, float share)
{
    if (share <= 0.0)
        return 0u;
    uint h = (uint)round(saturate((height - MinY) / (MaxY - MinY)) * 65535.0);
    uint s = (uint)round(saturate(share) * 65535.0);
    return (h << 16) | max(s, 1u);
}

void UnpackCover(uint v, out float height, out float share)
{
    share = (v & 0xFFFFu) / 65535.0;
    height = MinY + (v >> 16) / 65535.0 * (MaxY - MinY);
}

[numthreads(8, 8, 1)]
void CSReduce(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= Resolution))
        return;
    Texture2DArray<float> depth = ResourceDescriptorHeap[DepthIdx];
    RWTexture2D<uint> output = ResourceDescriptorHeap[OutputUAVIdx];

    float top = MinY, share = 0.0;
    [unroll]
    for (uint k = 0; k < 4; k++)
    {
        uint2 p = id.xy * 2u + uint2(k & 1u, k >> 1u);
        float d = depth.Load(int4(p, 0, 0));
        if (d >= 1.0)
            continue;
        float y = MaxY - d * (MaxY - MinY);
        // The sample's place: u along +x, v down from the top edge (−z)
        float2 uv = (float2(p) + 0.5) / (2.0 * Resolution);
        float2 xz = float2(Left + uv.x * Size, Top - uv.y * Size);
        if (TerrainHeightmap != 0)
        {
            Texture2D<float> heights = ResourceDescriptorHeap[TerrainHeightmap];
            float2 tuv = (xz - float2(TerrainOriginX, TerrainOriginZ)) / float2(TerrainSizeX, TerrainSizeZ);
            if (all(tuv >= 0.0) && all(tuv <= 1.0)
                && y < TerrainOriginY + heights.SampleLevel(LinearClamp, tuv, 0) * TerrainMaxHeight + MinAboveGround)
                continue;
        }
        top = max(top, y);
        share += 0.25;
    }

    if (Merge != 0)
    {
        // The vegetation over what the solid meshes left open: the crowns let some of the fall through
        float solidTop, solidShare;
        UnpackCover(output[id.xy], solidTop, solidShare);
        if (solidShare > 0.0)
            top = share > 0.0 ? max(top, solidTop) : solidTop;
        share = solidShare + (1.0 - solidShare) * share * (1.0 - CrownLeak);
    }
    output[id.xy] = PackCover(top, share);
}
