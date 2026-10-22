// The falling rain and snow past the near particles (Graphics/PrecipitationLayers.cs, after the forward pass, on the HDR
// composite): layers of streaks and flakes on spheres round the camera from 10 to 70 m. The near particles end ~12 m out
// (the emitters' boxes); with them alone the fall was specks round the camera and nothing past them (user, 2026-10-23).
//
// Each layer holds a field of cells in metres on its sphere (mapped by direction, so it holds still as the camera turns;
// a whole number of cells round it, so there is no seam), scrolled down at the fall speed over its distance and sheared
// by the wind across the view (streaks lean and drift with it). A cell holds one drop or flake or none, as many as the
// rate: the far layers hold more per pixel, each thinner and fainter (widened to a pixel and dimmed as much, as the
// particles' PIXEL_FLOOR), so the distance fills to a veil instead of lines. A layer is drawn only in front of the scene
// (the linear depth), off covered places (the cover map at the layer's point: no rain inside an arcade seen through its
// arch) and faded into the haze (Weather.HazeKm) with its distance. Rain takes the sky's light from above (a drop is a
// tiny lens: the particles' Water), snow is lit as a white speck by the sky round it.
// Docs: .agent/knowledge/atmospheric_effects/weather.md §3.

#pragma kernel CSLayers

cbuffer PushConstants : register(b3)
{
    uint CompositeUAVIdx;       // RWTexture2D<float4>: the HDR composite, pre-exposed
    uint LinearDepthIdx;        // Texture2D<float>: view depth (0 = sky)
    uint FrameIdx;              // StructuredBuffer<PrecipFrame>: [0]
};

#include "common.fx"
#include "sky_common.fx"
#include "weather_cover.hlsli"

// PrecipitationLayers.Frame (80 bytes)
struct PrecipFrame
{
    ShelterLevel Shelter[SHELTER_LEVELS];   // the cover map (WeatherShelter's levels)
    float ShelterMinY;
    float ShelterRangeY;
    float Rain;                 // Weather.Rain, Snow at the camera (0..1)
    float Snow;
    float WindAcross;           // the steady wind across the view (m/s along +azimuth at the view's middle), eased
    float _pad;
    float HazeKm;               // Weather.HazeKm
    float PixelAngle;           // radians a pixel spans (vertically, at the screen's middle)
};

SamplerState LinearClampS : register(s2);

#define LAYERS 5
#define FIRST_LAYER 11.0        // m; each layer 1.38× the last (11, 15, 21, 29, 40 m): past them the haze is the veil
#define LAYER_STEP 1.38

#define RAIN_FALL 8.0           // m/s
#define RAIN_DENSITY 40.0       // visible drops per m³ in heavy rain (rate 1)
#define RAIN_CELL_DOWN 0.5      // m: a cell's height (a streak is 12-22 cm of it)
#define RAIN_WIDTH 0.004        // m: a drop's streak
#define RAIN_ALPHA 0.3          // as the near emitter's
#define SNOW_FALL 1.0
#define SNOW_DENSITY 100.0      // flakes per m³ in heavy snow (they fall slowly: many are in the air)
#define SNOW_RADIUS 0.012
#define SNOW_ALPHA 0.8
#define MIN_CELL_PIXELS 6.0     // cells are never narrower: past that each drawn drop stands for the slab's many
#define MIN_STREAK_PIXELS 10.0  // a far streak as long as a camera's exposure smears it on the screen
#define MAX_ALPHA 0.15          // a layer's drop at most: denser, the far layers turned to static

float LayerHash(int3 c, uint seed)
{
    uint h = ((uint)c.x * 0x8DA6B343u) ^ ((uint)c.y * 0xD8163841u) ^ ((uint)c.z * 0xCB1AB31Fu) ^ (seed * 0x9E3779B9u);
    h = h * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    return (h & 0xFFFFFFu) / 16777216.0;
}

// The coverage (0..1) of one layer's field at a direction: rain streaks or snow flakes. az, el: the direction's azimuth
// and elevation; d: the layer's distance; windAcross: the wind across the view (m/s, along the camera's right, the same
// for every pixel); layer: its index
float LayerCoverage(float az, float el, float d, float windAcross, uint layer, bool snow, float rate, float pixelAngle)
{
    // The slab of air the layer stands for (half way to its neighbours) holds density × slab drops per m² of it: cells
    // that size (in heavy rain or snow; the rate decides how many hold one), never narrower than MIN_CELL_PIXELS
    float slab = d * 0.5 * (LAYER_STEP - 1.0 / LAYER_STEP);
    float perArea = (snow ? SNOW_DENSITY : RAIN_DENSITY) * slab;
    float pixelMetres = d * pixelAngle;
    float2 cell;
    if (snow)
        cell = max(sqrt(1.0 / perArea), MIN_CELL_PIXELS * pixelMetres).xx;
    else
        cell = float2(max(1.0 / (perArea * RAIN_CELL_DOWN), MIN_CELL_PIXELS * pixelMetres), RAIN_CELL_DOWN);
    float many = min(cell.x * cell.y * perArea, 2.0);   // drops a drawn one stands for (1..2)
    float fall = snow ? SNOW_FALL : RAIN_FALL;
    // Cells round the sphere, a whole number of them; down it, scrolled by the fall
    // The field in the falling air's own frame: the drops move (wind, -fall) metres a second on the sphere (x along
    // +azimuth, y up), so a place on it at time t holds the drop that was at (x - wind t, y + fall t). A streak lies along
    // that motion: across the field u = x + slant y holds still, down it v = y + fall t scrolls, and the drops slide down
    // their own lean. With the shear on the scrolled v the cells fell straight down under streaks leaning the other way
    // (user, 2026-10-23: "falling straight down ... sometimes flying against their lean"). At most 45°.
    float slant = clamp(windAcross / fall, -1.0, 1.0);
    float around = max(1.0, round(6.2831853 * d / cell.x));
    float u = az / 6.2831853 * around + slant * el * d / cell.x;
    // (the scroll wraps after 2^16 cells, an hour and more, so the cell's fraction keeps its precision)
    float v = el * d / cell.y + fmod(fall * Time / cell.y, 65536.0);

    int2 c = int2(floor(u), floor(v));
    c.x = (int)((uint)(c.x % (int)around + (int)around) % (uint)around);
    float2 f = frac(float2(u, v));
    int3 key = int3(c, (int)layer + (snow ? 64 : 0));
    if (LayerHash(key, 11u) > saturate(rate))
        return 0.0;

    if (snow)
    {
        float2 centre = 0.2 + 0.6 * float2(LayerHash(key, 12u), LayerHash(key, 13u));
        centre.x += 0.12 * sin(Time * (0.9 + LayerHash(key, 14u)) + LayerHash(key, 15u) * 6.2831853);
        float r = SNOW_RADIUS * (0.6 + 0.8 * LayerHash(key, 16u));
        float w = max(r, 1.2 * pixelMetres);
        float2 dm = (f - centre) * cell;
        return min(SNOW_ALPHA * exp(-2.0 * dot(dm, dm) / (w * w)) * (r * r) / (w * w) * many, MAX_ALPHA);
    }
    float x0 = 0.15 + 0.7 * LayerHash(key, 12u);
    float len = min(max(0.12 + 0.1 * LayerHash(key, 14u), MIN_STREAK_PIXELS * pixelMetres) / cell.y, 0.8);
    float y0 = (1.0 - len) * LayerHash(key, 13u);
    float w = max(RAIN_WIDTH, 1.2 * pixelMetres);
    float dx = (f.x - x0) * cell.x / w;
    float across = exp(-2.0 * dx * dx);
    float t = (f.y - y0) / len;
    float along = smoothstep(0.0, 0.25, t) * (1.0 - smoothstep(0.75, 1.0, t));
    return min(RAIN_ALPHA * across * along * RAIN_WIDTH / w * many, MAX_ALPHA);
}

[numthreads(8, 8, 1)]
void CSLayers(uint3 id : SV_DispatchThreadID)
{
    if (any(float2(id.xy) >= ViewportSize))
        return;
    StructuredBuffer<PrecipFrame> frames = ResourceDescriptorHeap[FrameIdx];
    PrecipFrame P = frames[0];
    Texture2D<float> linearDepth = ResourceDescriptorHeap[LinearDepthIdx];
    RWTexture2D<float4> composite = ResourceDescriptorHeap[CompositeUAVIdx];

    float sceneZ = linearDepth.Load(int3(id.xy, 0));
    if (sceneZ <= 0.0)
        sceneZ = 1e9;
    float2 uv = (float2(id.xy) + 0.5) / ViewportSize;
    float3 dir = normalize(posFromDepth(uv, 1.0, CameraInverse).xyz);   // reverse Z: depth 1 = the near plane
    float viewZPerMetre = max(mul(float4(dir, 0.0), View).z, 1e-3);
    float az = atan2(dir.x, dir.z), el = asin(clamp(dir.y, -1.0, 1.0));
    // One wind across the view for the whole screen: per direction the lattice folded where the lean changed fast
    float windAcross = P.WindAcross;

    WeatherShelter s;
    s.Levels = P.Shelter;
    s.MinY = P.ShelterMinY;
    s.RangeY = P.ShelterRangeY;
    s.TerrainHeightmap = 0;
    s.TerrainMaxHeight = 0;
    s.TerrainOrigin = 0;
    s.TerrainSize = 1;

    // The light the fall sends toward the eye: drops show the sky above them; flakes are white specks in the sky's light
    float3 rainLight = SkyAmbient(float3(0, 1, 0));
    float3 snowLight = (SkyAmbient(float3(0, 1, 0)) + SkyAmbient(float3(0, -1, 0))) * 0.5 * 0.9;

    float clear = 1.0;          // how much of the scene still shows through the layers in front of it
    float3 light = 0;
    float d = FIRST_LAYER;
    [loop]
    for (uint k = 0; k < LAYERS; k++, d *= LAYER_STEP)
    {
        float z = d * viewZPerMetre;
        if (z > sceneZ - 0.5)
            break;
        float3 place = CamPos + dir * d;
        float open = 1.0 - ShelterCoverAt(s, LinearClampS, place.xz, place.y);
        if (open <= 0.0)
            continue;
        float fade = open * exp(-P.HazeKm * d * 0.001) * saturate((sceneZ - z) / 2.0);
        float rain = P.Rain > 0.0 ? LayerCoverage(az, el, d, windAcross, k, false, P.Rain, P.PixelAngle) * fade : 0.0;
        float snow = P.Snow > 0.0 ? LayerCoverage(az, el, d, windAcross, k, true, P.Snow, P.PixelAngle) * fade : 0.0;
        light += clear * (rain * rainLight + snow * snowLight);
        clear *= (1.0 - rain) * (1.0 - snow);
    }
    if (clear >= 1.0)
        return;
    float4 c = composite[id.xy];
    composite[id.xy] = float4(c.rgb * clear + light, c.a);
}
