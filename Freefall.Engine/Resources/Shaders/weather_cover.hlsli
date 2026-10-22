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

#endif
