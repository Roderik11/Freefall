// What rides with a moving thing (Components/WeatherFrame.cs, Graphics/WeatherFrames.cs): the space the weather's
// patterns are laid in on it, and its own cover map, drawn in its own frame once it has left the sky captures (a ship
// under way). Read by the decals (snow's pattern and shelter on a ship's deck, decals_cs.hlsl) and the lamps (what
// covers a lantern below decks under way, point_light_lamps.hlsl). The caller passes its push constants: the table
// (StructuredBuffer<uint4>: transform slot + 1, frame slot, frame index, 0; open-addressed), its length − 1, the global
// transform buffer and the frames' covers (WeatherFrames.Srv, Mask, CoversSrv; 0 = none).
// Docs: .agent/knowledge/atmospheric_effects/weather.md "Snow on what moves".
#ifndef WEATHER_FRAMES_HLSLI
#define WEATHER_FRAMES_HLSLI

#include "weather_cover.hlsli"

// Origin and unit axes in the world: a place's pattern coordinates are its offsets along them. The world's own by
// default (origin 0, the world's axes), so a surface outside any frame gets exactly what it got before
struct WeatherSpace
{
    float3 Origin;
    float3 X, Y, Z;
    uint Root;          // the frame's index in the covers; WEATHER_WORLD for the world or none
};
#define WEATHER_WORLD 0xFFFFFFFFu

WeatherSpace WorldWeatherSpace()
{
    WeatherSpace w;
    w.Origin = 0;
    w.X = float3(1, 0, 0);
    w.Y = float3(0, 1, 0);
    w.Z = float3(0, 0, 1);
    w.Root = WEATHER_WORLD;
    return w;
}

// The space for a transform slot: its frame's rigid part (its scale taken out, so a scaled frame does not stretch the
// snow) where the slot follows one, else the world's
WeatherSpace WeatherSpaceIn(uint framesIdx, uint frameMask, uint transformsIdx, uint slot)
{
    WeatherSpace w = WorldWeatherSpace();
    if (framesIdx == 0)
        return w;
    StructuredBuffer<uint4> table = ResourceDescriptorHeap[framesIdx];
    uint key = slot + 1u;
    uint at = (slot * 2654435761u) & frameMask;     // WeatherFrames.Hash
    [loop] for (uint probe = 0; probe < 32u; probe++)  // at most half full: a probe or two
    {
        uint4 e = table[at];
        if (e.x == 0u)
            return w;
        if (e.x == key)
        {
            StructuredBuffer<row_major float4x4> transforms = ResourceDescriptorHeap[transformsIdx];
            float4x4 m = transforms[e.y];               // row vectors: rows 0-2 the axes, row 3 the origin
            w.Origin = m[3].xyz;
            w.X = normalize(m[0].xyz);
            w.Y = normalize(m[1].xyz);
            w.Z = normalize(m[2].xyz);
            w.Root = e.z;
            return w;
        }
        at = (at + 1u) & frameMask;
    }
    return w;
}

// A world place's pattern coordinates in a weather space
float3 WeatherPlace(WeatherSpace w, float3 world)
{
    float3 d = world - w.Origin;
    return float3(dot(d, w.X), dot(d, w.Y), dot(d, w.Z));
}

// Whether the frame has its own cover map drawn: it has left the sky captures (WeatherFrames.SkyCapturesChanged), so
// they no longer hold it and its own map stands in for them
bool FrameCoverDrawn(uint coversIdx, WeatherSpace w)
{
    if (w.Root == WEATHER_WORLD || coversIdx == 0)
        return false;
    StructuredBuffer<float4> covers = ResourceDescriptorHeap[coversIdx];
    return covers[w.Root * 2u + 1u].z >= 0.5;
}

// How much of the fall reaches a world place on a frame by the frame's own cover map (WeatherFrames, ShelterMap
// RenderFrameCover: its meshes drawn straight down its own up, in its frame), 0..1; 1 while it has none (still in the sky
// captures: the world's map holds it). Read as a level of the world's map is, in the frame's coordinates: its own castles
// and decks over a place on it, whatever its heel or wherever it has sailed. `s` lends the lookups its sampler state only
float FrameOpennessIn(uint coversIdx, WeatherShelter s, SamplerState linearClamp, WeatherSpace w, float3 world, float bias)
{
    if (!FrameCoverDrawn(coversIdx, w))
        return 1.0;
    StructuredBuffer<float4> covers = ResourceDescriptorHeap[coversIdx];
    float4 square = covers[w.Root * 2u], heights = covers[w.Root * 2u + 1u];
    ShelterLevel l;
    l.Left = square.x;
    l.Top = square.y;
    l.Size = square.z;
    l.Map = asuint(square.w);
    WeatherShelter own = s;
    own.MinY = heights.x;
    own.RangeY = heights.y;
    float3 p = WeatherPlace(w, world);
    return ShelterLevelOpenness(own, l, linearClamp, p.xz, p.y + bias, bias);
}

#endif
