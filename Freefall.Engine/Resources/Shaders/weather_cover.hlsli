// What covers a place from the sky (WEATHER_PLAN.md stage 2): the terrain's baked heights over it (a cave, the hill the
// rain falls on) or a mesh over it. The rain and snow particles stop under the meshes in the sky captures' top-down
// slice (SkyCapture.Shelter, 256 m round the camera; particle_compute.hlsl StopUnderCover); surfaces take how much of
// the fall reaches them from the cover map's three levels (Graphics/ShelterMap.cs, out to ~2 km; decals_cs.hlsl).
// Docs: .agent/knowledge/atmospheric_effects/weather.md.
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

#define SHELTER_LEVELS 3            // ShelterMap.Levels
#define SHELTER_RESOLUTION 1024.0   // ShelterMap.Resolution

// One level of the cover map (ShelterMap.LevelData, 16 bytes)
struct ShelterLevel
{
    float Left;     // world x at the square's left edge
    float Top;      // world z at its top edge (the map's v 0)
    float Size;     // metres it spans; 0 = not drawn
    uint Map;       // Texture2D<uint>: (height << 16) | share per texel
};

// What keeps the fall off a surface: the cover map's levels and the terrain (a cave, an overhang of the hill)
struct WeatherShelter
{
    ShelterLevel Levels[SHELTER_LEVELS];
    float2 Eye;                     // world xz the levels are centred on (ShelterMap.EyeFor): what they blend by
    float MinY;                     // the heights the maps' 16 bits span
    float RangeY;
    uint TerrainHeightmap;          // R16 heights; 0 = none
    float TerrainMaxHeight;
    float3 TerrainOrigin;
    float2 TerrainSize;
};

// The four texels of a level round a place (world xz), in Gather's order (w the texel at (0, 0) of the quad, z (1, 0),
// x (0, 1), y (1, 1)): the height of the highest cover over each (very low where none) and the share of the texel it
// covers, and the place's bilinear weights between them (f)
void ShelterTexels(WeatherShelter s, ShelterLevel l, SamplerState anySampler, float2 xz, out float4 tops, out float4 shares,
    out float2 f)
{
    float2 uv = float2(xz.x - l.Left, l.Top - xz.y) / l.Size;
    f = frac(uv * SHELTER_RESOLUTION - 0.5);
    tops = -1e9;
    shares = 0.0;
    if (any(uv < 0.0) || any(uv >= 1.0))
        return;
    Texture2D<uint> map = ResourceDescriptorHeap[l.Map];
    uint4 v = map.GatherRed(anySampler, uv);
    shares = float4(v & 0xFFFFu) / 65535.0;
    tops = s.MinY + float4(v >> 16) / 65535.0 * s.RangeY - (1.0 - step(1e-6, shares)) * 1e9;
}

float WeatherBilinear(float4 v, float2 f)
{
    return lerp(lerp(v.w, v.z, f.x), lerp(v.x, v.y, f.x), f.y);
}

#define WEATHER_SPREAD 0.25     // tan of the half-angle falling rain and snow spread over (gusts, turbulence): what
                                // covers a place h metres up softens its edge over 0.25 h
#define WEATHER_MIN_RADIUS 0.2  // m: the narrowest edge (low cover, a bench), at least a texel of the level
#define WEATHER_TAPS 16
#define WEATHER_SEARCH 1.5      // m round the place the blocker search looks for cover (3 texels in a coarse level)

// How much of the fall reaches a place at height y (world xz) by one level, 0..1, as soft shadows: the height of the
// cover found over it and on a ring round it (rise), then the share of a disc, as wide as the fall spreads over that
// height, that the cover lets through (its texels' shares: a far crown covers part of a 4 m texel, the vegetation's
// crowns only part of what they span). Every lookup blended over its four texels, so edges slide smoothly over them
float ShelterLevelOpenness(WeatherShelter s, ShelterLevel l, SamplerState anySampler, float2 xz, float y, float bias)
{
    float texel = l.Size / SHELTER_RESOLUTION;
    float search = max(WEATHER_SEARCH, 3.0 * texel);
    float rise = 0.0;
    [unroll]
    for (uint k = 0; k < 9; k++)
    {
        float angle = k * 0.785398;
        float2 offset = k == 0 ? float2(0, 0) : float2(cos(angle), sin(angle)) * search;
        float4 tops, shares;
        float2 f;
        ShelterTexels(s, l, anySampler, xz + offset, tops, shares, f);
        rise = max(rise, WeatherBilinear(max(tops - y, 0.0), f));
    }
    if (rise <= 0.0)
        return 1.0;

    // A Vogel spiral of taps (the same pattern everywhere: no shimmer)
    float radius = min(max(WEATHER_MIN_RADIUS, texel) + (rise + bias) * WEATHER_SPREAD, search);
    float covered = 0.0;
    [unroll]
    for (uint i = 0; i < WEATHER_TAPS; i++)
    {
        float a = i * 2.39996323;
        float2 offset = float2(cos(a), sin(a)) * (radius * sqrt((i + 0.5) / WEATHER_TAPS));
        float4 tops, shares;
        float2 f;
        ShelterTexels(s, l, anySampler, xz + offset, tops, shares, f);
        covered += WeatherBilinear(shares * step(y, tops), f);
    }
    return 1.0 - covered / WEATHER_TAPS;
}

// How far a place (world xz) lies inside a level, past the room its lookups need round it (m; ≤ 0: not inside)
float ShelterInside(ShelterLevel l, float2 xz)
{
    if (l.Size <= 0.0)
        return -1.0;
    float room = 2.0 * max(WEATHER_SEARCH, 3.0 * l.Size / SHELTER_RESOLUTION);
    return min(min(xz.x - l.Left, l.Left + l.Size - xz.x), min(l.Top - xz.y, xz.y - (l.Top - l.Size))) - room;
}

// The share of the sky over a place in the air (world xz at height y) that cover over it hides, 0..1: one lookup blended
// over its four texels in the finest level that holds it, no disc (rain and snow in the air: precipitation_layers_cs.hlsl)
float ShelterCoverAt(WeatherShelter s, SamplerState anySampler, float2 xz, float y)
{
    [unroll]
    for (uint k = 0; k < SHELTER_LEVELS; k++)
    {
        if (ShelterInside(s.Levels[k], xz) > 0.0)
        {
            float4 tops, shares;
            float2 f;
            ShelterTexels(s, s.Levels[k], anySampler, xz, tops, shares, f);
            return WeatherBilinear(shares * step(y, tops), f);
        }
    }
    return 0.0;
}

// How far from the eye a level can be read: its square is centred again every eighth of its size (ShelterMap.Steps), so
// a circle of 7/16 of its size round the eye, less the room its lookups need, always lies inside it
float ShelterReach(ShelterLevel l)
{
    return 0.4375 * l.Size - 2.0 * max(WEATHER_SEARCH, 3.0 * l.Size / SHELTER_RESOLUTION);
}

// The levels for a place (world xz): the finest that holds it, how much of it to take (w), and the next coarser one that
// holds it (SHELTER_LEVELS: none), by the place's distance from the eye: a level in full out to 70 % of its reach, then
// blended into the next over the rest, and the last into no cover. By the distance, not the level's square, which moves
// in steps with the camera: a blend over the square's outer 15 % jumped at every step and drew a square round the
// camera (user, 2026-10-23: "distance/cross fade the ground coverage map"). A level not yet drawn again for where the
// camera is now (one a frame) is passed over where it does not reach
void ShelterBlend(WeatherShelter s, float2 xz, out uint first, out uint second, out float w)
{
    float d = length(xz - s.Eye);
    first = SHELTER_LEVELS;
    second = SHELTER_LEVELS;
    w = 0.0;
    [unroll]
    for (uint k = 0; k < SHELTER_LEVELS; k++)
    {
        if (s.Levels[k].Size <= 0.0 || ShelterInside(s.Levels[k], xz) <= 0.0)
            continue;
        float reach = ShelterReach(s.Levels[k]);
        if (first == SHELTER_LEVELS)
        {
            if (d < reach)
            {
                first = k;
                w = saturate((reach - d) / (0.3 * reach));
            }
        }
        else if (second == SHELTER_LEVELS)
            second = k;
    }
}

// How much of the falling rain and snow reaches a surface (absolute world position), 0..1. Straight down: what lies on
// the ground must not move with the wind (a fall slanted by the wind moved the snow already lying with every change of
// the wind, user 2026-10-22). Blended between the levels by the distance from the eye (ShelterBlend), so the cover
// thins out with distance instead of ending at a square round the camera (one 256 m capture left bare ground under the
// trees inside it and snow under them past it, user 2026-10-23). The terrain covers outright.
float WeatherOpenness(WeatherShelter s, SamplerState linearClamp, float3 p, float bias)
{
    if (s.TerrainHeightmap != 0)
    {
        Texture2D<float> heights = ResourceDescriptorHeap[s.TerrainHeightmap];
        float2 tuv = (p.xz - s.TerrainOrigin.xz) / s.TerrainSize;
        if (all(tuv >= 0.0) && all(tuv <= 1.0)
            && s.TerrainOrigin.y + heights.SampleLevel(linearClamp, tuv, 0) * s.TerrainMaxHeight > p.y + bias)
            return 0.0;
    }
    uint first, second;
    float w;
    ShelterBlend(s, p.xz, first, second, w);
    if (first == SHELTER_LEVELS)
        return 1.0;
    float y = p.y + bias;
    float open = ShelterLevelOpenness(s, s.Levels[first], linearClamp, p.xz, y, bias);
    if (w < 1.0)
    {
        float next = second < SHELTER_LEVELS ? ShelterLevelOpenness(s, s.Levels[second], linearClamp, p.xz, y, bias) : 1.0;
        open = lerp(next, open, w);
    }
    return open;
}

#define CROWN_SHELL 1.0     // m under the top of the cover over it a leaf takes the fall in full (half a texel added)
#define CROWN_DEPTH 3.0     // ... and none (a texel added)

// By one level: how far under the highest cover over its four texels (its own crown's top, or a roof over the tree) a
// place lies, as the share of the fall it takes. The shell widens with the level's texels: a coarse level's top is the
// highest leaf over metres, and a thin shell turned far trees green again
float CrownLevelExposure(WeatherShelter s, ShelterLevel l, SamplerState anySampler, float3 p)
{
    float4 tops, shares;
    float2 f;
    ShelterTexels(s, l, anySampler, p.xz, tops, shares, f);
    float top = max(max(tops.x, tops.y), max(tops.z, tops.w));
    float texel = l.Size / SHELTER_RESOLUTION;
    return 1.0 - smoothstep(CROWN_SHELL + 0.5 * texel, CROWN_DEPTH + texel, top - p.y);
}

// How much of the fall reaches a place in a crown (absolute world position), 0..1: the snow lies on a crown's top and
// its upper flanks, not under it or inside it. The foliage's G-buffer normal (its dome normal, gbuffer_foliage.fx) faces
// up even under the crown, so by the normal alone a full cover turned whole trees white, from above and below and in to
// the trunk (user, 2026-10-23). Blended between the levels as WeatherOpenness. A tree the map holds no top for takes
// the fall in full: the far trees' cross-mesh cards are vertical and have no area seen from straight above, so the
// cover map does not hold them (nor shelter the ground under them)
float CrownExposure(WeatherShelter s, SamplerState anySampler, float3 p)
{
    uint first, second;
    float w;
    ShelterBlend(s, p.xz, first, second, w);
    if (first == SHELTER_LEVELS)
        return 1.0;
    float exposure = CrownLevelExposure(s, s.Levels[first], anySampler, p);
    if (w < 1.0)
        exposure = lerp(second < SHELTER_LEVELS ? CrownLevelExposure(s, s.Levels[second], anySampler, p) : 1.0, exposure, w);
    return exposure;
}

#endif
