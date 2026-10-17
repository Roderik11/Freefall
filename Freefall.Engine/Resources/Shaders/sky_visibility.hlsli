// How much sky and sun a place gets past the terrain (ATMOSPHERE_PLAN stage 4). Self-contained, so passes without the
// scene constants (the point light shadow selection) can use it: sky_common.fx wraps it with the atmosphere's
// parameters (TerrainOpen, SkyVisibility). Docs: .agent/knowledge/atmospheric_effects/physical_atmosphere.md §10.
#ifndef SKY_VISIBILITY_HLSLI
#define SKY_VISIBILITY_HLSLI

// The terrain's baked heightmap (R16, 0 = no terrain) and where it lies
struct TerrainCover
{
    uint Heightmap;
    float MaxHeight;        // metres at 1
    float3 Origin;          // world position of the terrain's corner (y = its base)
    float2 Size;            // metres it spans in x and z
};

float TerrainHeightAt(TerrainCover t, SamplerState linearClamp, float2 xz)
{
    Texture2D<float> heights = ResourceDescriptorHeap[t.Heightmap];
    return t.Origin.y + heights.SampleLevel(linearClamp, (xz - t.Origin.xz) / t.Size, 0) * t.MaxHeight;
}

// 1 in the open, 0 four metres or more under the terrain's surface (a cave, a tunnel, under an overhang): the point,
// lifted half a metre along its normal, against the terrain height above it. Neither the sky nor the sun reaches such
// a place, whatever the shadow map holds (the camera-driven terrain shadow pass can miss the hill over a cave).
float TerrainOpenAt(TerrainCover t, SamplerState linearClamp, float3 worldPos, float3 N)
{
    if (t.Heightmap == 0)
        return 1;
    float3 p = worldPos + N * 0.5;
    return 1.0 - smoothstep(0.0, 4.0, TerrainHeightAt(t, linearClamp, p.xz) - p.y);
}

// How much of the sky's light reaches a surface past the terrain, 0..1: from the point, lifted half a metre along its
// normal into the air it faces, the highest terrain horizon in 8 directions, sampled 2, 6, 18 and 54 m out. The
// cosine-weighted sky above a horizon at angle h is cos²h = 1 / (1 + tan²h). A cave under a hill sees none of it (the
// heightmap keeps the hill's surface over a terrain hole), a valley less of it, open ground all of it. Under the
// surface there is rock straight above: TerrainOpenAt takes the rest (the horizons alone left a sliver of zenith, an 87°
// horizon passes 0.3 % of the sky, which the auto exposure lifted to a lit grey cave). worldPos is absolute. Roofs and
// tree crowns are not in the terrain: they come with a mesh height capture.
float SkyVisibilityAt(TerrainCover t, SamplerState linearClamp, float3 worldPos, float3 N)
{
    if (t.Heightmap == 0)
        return 1;
    float open = TerrainOpenAt(t, linearClamp, worldPos, N);
    if (open <= 0.0)
        return 0;
    float3 p = worldPos + N * 0.5;
    static const float distances[4] = { 2.0, 6.0, 18.0, 54.0 };
    float visible = 0;
    [unroll] for (int a = 0; a < 8; a++)
    {
        float angle = a * 0.78539816;
        float2 dir = float2(cos(angle), sin(angle));
        float rise = 0;   // tan of the highest horizon this way
        [unroll] for (int k = 0; k < 4; k++)
            rise = max(rise, (TerrainHeightAt(t, linearClamp, p.xz + dir * distances[k]) - p.y) / distances[k]);
        visible += 1.0 / (1.0 + rise * rise);
    }
    return visible * 0.125 * open;
}

#endif
