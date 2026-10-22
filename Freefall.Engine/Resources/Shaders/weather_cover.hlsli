// What covers a place from the sky (WEATHER_PLAN.md stage 2): the terrain's baked heights over it (a cave, the hill the
// rain falls on) or a mesh over it in the sky captures' top-down slice (SkyCapture.Shelter: a roof, a crown, an
// overhang). Rain and snow stop there (particle_compute.hlsl StopUnderCover) and surfaces under it stay dry and bare
// (decals_cs.hlsl). Docs: .agent/knowledge/atmospheric_effects/weather.md.
#ifndef WEATHER_COVER_HLSLI
#define WEATHER_COVER_HLSLI

struct WeatherCover
{
    row_major float4x4 ShelterVP;   // world -> the top-down slice's clip space (orthographic, looking down)
    uint ShelterDepth;              // Texture2DArray<float>: the sky captures' depth; 0 = not drawn yet
    uint ShelterSlice;
    uint TerrainHeightmap;          // sky_visibility.hlsli TerrainCover: R16 heights; 0 = none
    float TerrainMaxHeight;
    float3 TerrainOrigin;
    float2 TerrainSize;
};

// True where something covers the place (absolute world position) more than `bias` metres over it
bool WeatherUnderCover(WeatherCover c, SamplerState linearClamp, float3 p, float bias)
{
    if (c.TerrainHeightmap != 0)
    {
        Texture2D<float> heights = ResourceDescriptorHeap[c.TerrainHeightmap];
        float2 uv = (p.xz - c.TerrainOrigin.xz) / c.TerrainSize;
        if (all(uv >= 0.0) && all(uv <= 1.0)
            && c.TerrainOrigin.y + heights.SampleLevel(linearClamp, uv, 0) * c.TerrainMaxHeight > p.y + bias)
            return true;
    }
    if (c.ShelterDepth != 0)
    {
        float4 clip = mul(float4(p, 1.0), c.ShelterVP);
        float2 uv = float2(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
        if (all(uv >= 0.0) && all(uv < 1.0) && clip.z < 1.0)
        {
            Texture2DArray<float> depths = ResourceDescriptorHeap[c.ShelterDepth];
            uint w, h, n;
            depths.GetDimensions(w, h, n);
            float top = depths.Load(int4(int2(uv * float2(w, h)), c.ShelterSlice, 0));
            // Metres under the highest mesh there: looking down, the depth grows by -VP._23 a metre
            if ((clip.z - top) / max(-c.ShelterVP._23, 1e-9) > bias)
                return true;
        }
    }
    return false;
}

// The heights (m) of the highest mesh on the four texels of the top-down slice round a place (world xz), in Gather's
// order (w the texel at (0, 0) of the quad, z (1, 0), x (0, 1), y (1, 1)), and the place's weights between them as
// bilinear filtering takes them (f). Very low where a texel holds nothing (cleared) and outside the slice. Looking down
// the depth depends on y alone: height = (depth - depth at y 0) / VP._23
float4 WeatherShelterHeights(WeatherCover c, Texture2DArray<float> depths, SamplerState anySampler, float2 size, float2 xz, out float2 f)
{
    float4 clip0 = mul(float4(xz.x, 0.0, xz.y, 1.0), c.ShelterVP);
    float2 uv = float2(clip0.x * 0.5 + 0.5, 0.5 - clip0.y * 0.5);
    f = frac(uv * size - 0.5);
    if (any(uv < 0.0) || any(uv >= 1.0))
        return float4(-1e9, -1e9, -1e9, -1e9);
    float4 d = depths.GatherRed(anySampler, float3(uv, c.ShelterSlice));
    return (d - clip0.z) / c.ShelterVP._23;
}

float WeatherBilinear(float4 v, float2 f)
{
    return lerp(lerp(v.w, v.z, f.x), lerp(v.x, v.y, f.x), f.y);
}

#define WEATHER_SPREAD 0.25     // tan of the half-angle falling rain and snow spread over (gusts, turbulence): what
                                // covers a place h metres up softens its edge over 0.25 h
#define WEATHER_MIN_RADIUS 0.2  // m: the narrowest edge, a little under a texel's width each way (low cover, a bench)
#define WEATHER_TAPS 16
#define WEATHER_SEARCH 1.5      // m round the place the blocker search looks for cover; the widest edge

// How much of the falling rain and snow reaches a surface (absolute world position), 0..1: the share of a disc of the
// top-down slice whose highest mesh lies over the place, the disc as wide as the fall spreads over the height of the
// cover found round it (the blocker search of soft shadows). Straight down: what lies on the ground must not move with
// the wind (the fall slanted by the wind moved the snow already lying and the dry strips under the eaves with every
// change of the wind, user 2026-10-22). Under a roof 3 m up the edge softens over ~1 m; a lantern's arm leaves a faint
// mark, not a hole. Every lookup is blended over its four texels (the search's heights too: point lookups made the
// disc's width jump from texel to texel), so the edge slides smoothly across the slice's 25 cm texels: a binary test
// drew them as staircases in the snow along every eave and round every post. The terrain (a cave, an overhang of the
// hill) covers outright.
float WeatherOpenness(WeatherCover c, SamplerState linearClamp, float3 p, float bias)
{
    if (c.TerrainHeightmap != 0)
    {
        Texture2D<float> heights = ResourceDescriptorHeap[c.TerrainHeightmap];
        float2 tuv = (p.xz - c.TerrainOrigin.xz) / c.TerrainSize;
        if (all(tuv >= 0.0) && all(tuv <= 1.0)
            && c.TerrainOrigin.y + heights.SampleLevel(linearClamp, tuv, 0) * c.TerrainMaxHeight > p.y + bias)
            return 0.0;
    }
    if (c.ShelterDepth == 0)
        return 1.0;
    Texture2DArray<float> depths = ResourceDescriptorHeap[c.ShelterDepth];
    uint w, h, n;
    depths.GetDimensions(w, h, n);
    float2 size = float2(w, h);
    float y = p.y + bias;

    // The cover's height over the place: the highest found at the place and on a ring round it; nothing = open
    float rise = 0.0;
    [unroll]
    for (uint k = 0; k < 9; k++)
    {
        float angle = k * 0.785398;
        float2 offset = k == 0 ? float2(0, 0) : float2(cos(angle), sin(angle)) * WEATHER_SEARCH;
        float2 f;
        float4 tops = WeatherShelterHeights(c, depths, linearClamp, size, p.xz + offset, f);
        rise = max(rise, WeatherBilinear(max(tops - y, 0.0), f));
    }
    if (rise <= 0.0)
        return 1.0;

    // The share of the disc the fall comes through, a Vogel spiral of taps (the same pattern everywhere: no shimmer)
    float radius = min(WEATHER_MIN_RADIUS + (rise + bias) * WEATHER_SPREAD, WEATHER_SEARCH);
    float covered = 0.0;
    [unroll]
    for (uint i = 0; i < WEATHER_TAPS; i++)
    {
        float a = i * 2.39996323;
        float2 offset = float2(cos(a), sin(a)) * (radius * sqrt((i + 0.5) / WEATHER_TAPS));
        float2 f;
        float4 tops = WeatherShelterHeights(c, depths, linearClamp, size, p.xz + offset, f);
        covered += WeatherBilinear(step(y, tops), f);
    }
    return 1.0 - covered / WEATHER_TAPS;
}

#endif
