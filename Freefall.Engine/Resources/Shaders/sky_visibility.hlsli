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

// How far below the heightmap a surface may be drawn and still be the terrain's surface, not a cave: the terrain is
// drawn coarser with distance (CDLOD) and over a ridge its triangles sit metres under the heightmap. 2 m near the
// camera plus 1 % of the distance; without it ridges and the forests on them went black from afar.
float TerrainTolerance(float distanceToCamera)
{
    return 2.0 + 0.01 * distanceToCamera;
}

// The point to look from: lifted half a metre along the normal into the air the surface faces; a point within the
// tolerance under the heightmap is the terrain's own surface (or a thing standing on it), and looks from just above it
float3 SkyLookPoint(TerrainCover t, SamplerState linearClamp, float3 worldPos, float3 N, float tolerance, out float under)
{
    float3 p = worldPos + N * 0.5;
    float surface = TerrainHeightAt(t, linearClamp, p.xz);
    under = surface - p.y;
    if (under < tolerance)
        p.y = max(p.y, surface + 0.5);
    return p;
}

// 1 in the open, 0 four metres or more past the tolerance under the terrain's surface (a cave, a tunnel, under an
// overhang). Neither the sky nor the sun reaches such a place, whatever the shadow map holds (the camera-driven terrain
// shadow pass can miss the hill over a cave).
float TerrainOpenAt(TerrainCover t, SamplerState linearClamp, float3 worldPos, float3 N, float tolerance)
{
    if (t.Heightmap == 0)
        return 1;
    float under;
    SkyLookPoint(t, linearClamp, worldPos, N, tolerance, under);
    return 1.0 - smoothstep(tolerance, tolerance + 4.0, under);
}

// How much of the sky's light reaches a surface past the terrain, 0..1: from the look point the highest terrain horizon
// in 8 directions, sampled 2, 6, 18 and 54 m out. The cosine-weighted sky above a horizon at angle h is
// cos²h = 1 / (1 + tan²h). A cave under a hill sees none of it (the heightmap keeps the hill's surface over a terrain
// hole), a valley less of it, open ground all of it. Under the surface there is rock straight above: the open factor
// takes the rest (the horizons alone left a sliver of zenith, an 87° horizon passes 0.3 % of the sky, which the auto
// exposure lifted to a lit grey cave). worldPos is absolute. Roofs and tree crowns are not in the terrain: they come
// with a mesh height capture.
float SkyVisibilityAt(TerrainCover t, SamplerState linearClamp, float3 worldPos, float3 N, float tolerance)
{
    if (t.Heightmap == 0)
        return 1;
    float under;
    float3 p = SkyLookPoint(t, linearClamp, worldPos, N, tolerance, under);
    float open = 1.0 - smoothstep(tolerance, tolerance + 4.0, under);

    // In the open: the terrain's horizons
    float sky = 0;
    if (open > 0.0)
    {
        static const float distances[4] = { 2.0, 6.0, 18.0, 54.0 };
        [unroll] for (int a = 0; a < 8; a++)
        {
            float angle = a * 0.78539816;
            float2 dir = float2(cos(angle), sin(angle));
            float rise = 0;   // tan of the highest horizon this way
            [unroll] for (int k = 0; k < 4; k++)
                rise = max(rise, (TerrainHeightAt(t, linearClamp, p.xz + dir * distances[k]) - p.y) / distances[k]);
            sky += 1.0 / (1.0 + rise * rise);
        }
        sky *= 0.125;
    }

    // Under the terrain, daylight comes in only through an opening (a cave mouth): in each direction the first place
    // within 32 m where the ground is lower than the point. The opening passes about a third of that direction's sky,
    // falling off with the distance to it like light down a 4 m tunnel; past 32 m of rock the cave is dark. Without
    // this a cave mouth was a black hole from outside, and from inside the walls by the exit stayed unlit.
    float portal = 0;
    if (open < 1.0)
    {
        static const float steps[5] = { 2.0, 4.0, 8.0, 16.0, 32.0 };
        [unroll] for (int a = 0; a < 8; a++)
        {
            float angle = a * 0.78539816;
            float2 dir = float2(cos(angle), sin(angle));
            float reach = 0;  // light coming in from this way: from the nearest opening found
            [unroll] for (int k = 4; k >= 0; k--)
            {
                if (TerrainHeightAt(t, linearClamp, p.xz + dir * steps[k]) < p.y)
                    reach = 1.0 / (1.0 + (steps[k] / 4.0) * (steps[k] / 4.0));
            }
            portal += reach;
        }
        portal *= 0.125 * 0.35;
    }
    return lerp(portal, sky, open);
}

#endif
