// The auto exposure's incident light meter (Graphics/AutoExposure.cs): the light that reaches the camera, read as a
// photographer's incident meter reads it, so the exposure follows where the camera is, not what the frame happens to
// show (the screen's histogram only nudges it). Made once per frame after the sky captures and the lamps
// (DeferredRenderer 2f), read back by the CPU three frames later. Nothing in it depends on where the camera looks.
//
// One group of 64 threads, each a tap on a disc of METER_RADIUS round the camera at its height (the first at the camera),
// so a fleck of sun under the trees or the edge of a shadow moves the meter a little at a time:
//   the sun and the moon: their light through the air (facing them), × the share of the taps they reach past the terrain
//   (a cave, the terrain's shadow toward them: terrain_light_shadow_cs.hlsl), the meshes (the sky captures nearest
//   their direction: walls, roofs, crowns) and the clouds' shadow
//   the sky: its light on level ground × the share of it the taps see past the terrain's horizons and the sky captures
//   (with SkyCaptureBounce, as the surfaces round the camera get it); the ground's bounce light by the same share of
//   what lights the open ground
//   the point lights: candela / d² of every enabled one in range (a lamp by its LampState), unoccluded
// A tap counts as far as it sees as much sky as the camera (a room's taps through the wall in the street do not).
// From those parts two readings, the same whichever way the camera looks, which AutoExposure takes half and half: the
// light on a sphere there (a body's / 4, the sky's and the ground's / 2, the lamps' / 4) and the brightness round it
// (the sky dome from the sky probe where the camera sees it, the crowns and walls round it where it does not, the
// ground below, each lit as at the camera).
//
// Not the sun's cascades: they fit the view, and the shadow pass leaves out casters that shade nothing in view, so the
// roof over a camera facing a wall was missing from them: a room's taps lit by a quarter of the noon sun, the room 2-3 EV
// too dark and darker facing the door than facing away.
// Docs: .agent/knowledge/atmospheric_effects/physical_atmosphere.md §7.1.

#pragma kernel CSMeter

cbuffer PushConstants : register(b3)
{
    uint OutputUAVIdx;          // RWStructuredBuffer<float4>[METER_ROWS]
    uint RecordsIdx;            // StructuredBuffer<PointLightRecord>: the registry; 0 = no point lights
    uint RecordCountIdx;        // registry slots to walk
    uint TransformsIdx;         // StructuredBuffer<row_major float4x4>: the global transform buffer
    uint LampsUAVIdx;           // RWStructuredBuffer<LampState> per slot; 0 = every lamp follows OpenLamps
};

cbuffer Params : register(b4)
{
    float3 MeterPosition;       // absolute world position metered: the camera
    float OpenLamps;            // Daylight.Lamps: a lamp not judged yet
};

#include "common.fx"
#include "sky_common.fx"
#include "point_lights.hlsli"

#define METER_TAPS 64
#define METER_RADIUS 2.0        // metres round the camera
#define METER_SKY_TOLERANCE 4.0 // a tap whose sky share differs from the camera's by 1/this or more does not count
#define METER_BODY_SHARPNESS 12.0   // weight of a capture direction for a body: exp(this × (cos angle − 1))
#define METER_SURROUND_ALBEDO 0.2   // the crowns, walls and ceilings round the camera
#define METER_ROWS 4            // AutoExposure.MeterRows

SamplerState MeterWrapSampler : register(s0);       // linear wrap (the cloud noise)

groupshared SkyCaptureView gViews[SKY_CAPTURE_VIEWS];
groupshared float4 gSum[3][METER_TAPS];
groupshared float gSky[METER_TAPS];

// The meshes round a tap from the sky captures: the share of the sky it sees (each direction alike, as for a level
// surface) and toward the sun and the moon (the directions nearest each, by METER_BODY_SHARPNESS: the 16 lie 35-45°
// apart). 1 without captures.
void MeshShares(float3 tap, float3 sunDir, float3 moonDir, bool captures, out float sky, out float sun, out float moon)
{
    sky = 1; sun = 1; moon = 1;
    if (!captures)
        return;
    AtmosphereParams A = LoadAtmosphere();
    Texture2DArray<float> depths = ResourceDescriptorHeap[A.SkyCaptureDepth];
    float width, height, slices;
    depths.GetDimensions(width, height, slices);

    float skyLit = 0, skyTotal = 0, sunLit = 0, sunTotal = 0, moonLit = 0, moonTotal = 0;
    [unroll] for (uint i = 0; i < SKY_CAPTURE_VIEWS; i++)
    {
        SkyCaptureView v = gViews[i];
        if (v.Valid <= 0.0)
            continue;
        float open = SkyCaptureOpenAt(v, i, depths, float2(width, height), AtmosphereLinearClamp, tap);
        skyLit += open;
        skyTotal += 1.0;
        float ws = exp(METER_BODY_SHARPNESS * (dot(v.Direction, sunDir) - 1.0));
        float wm = exp(METER_BODY_SHARPNESS * (dot(v.Direction, moonDir) - 1.0));
        sunLit += ws * open;
        sunTotal += ws;
        moonLit += wm * open;
        moonTotal += wm;
    }
    sky = skyTotal > 0.0 ? skyLit / skyTotal : 1.0;
    sun = sunTotal > 1e-6 ? sunLit / sunTotal : sky;
    moon = moonTotal > 1e-6 ? moonLit / moonTotal : sky;
}

// A body's share at a tap past the terrain (a cave; its shadow toward the body), the meshes and the clouds
float BodyVisible(float3 tap, float3 toLight, float open, float meshes, out float terrain, out float cloud)
{
    terrain = 1; cloud = 1;
    if (toLight.y <= 0.0 || open <= 0.0)
        return 0.0;
    terrain = TerrainLightShadow(tap, toLight);
    cloud = GetCloudShadow(MeterWrapSampler, tap - CamPos, toLight);
    return open * terrain * meshes * cloud;
}

float Luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

[numthreads(METER_TAPS, 1, 1)]
void CSMeter(uint gi : SV_GroupIndex)
{
    bool captures = false;
    if (AtmosphereIdx != 0)
    {
        AtmosphereParams A = LoadAtmosphere();
        captures = A.SkyCaptureViews != 0 && A.SkyCaptureDepth != 0 && A.SkyCaptureCount == SKY_CAPTURE_VIEWS;
        if (captures && gi < SKY_CAPTURE_VIEWS)
        {
            StructuredBuffer<SkyCaptureView> views = ResourceDescriptorHeap[A.SkyCaptureViews];
            gViews[gi] = views[gi];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    float4 row0 = 0, row1 = 0;
    if (AtmosphereIdx != 0)
    {
        AtmosphereParams A = LoadAtmosphere();

        // The tap: a golden-angle spiral filling the disc evenly, the first at the camera itself
        float r = METER_RADIUS * sqrt(gi / (METER_TAPS - 1.0));
        float a = gi * 2.39996323;
        float3 tap = MeterPosition + float3(cos(a) * r, 0, sin(a) * r);
        float3 up = float3(0, 1, 0);

        float open = TerrainOpen(tap, up);
        float meshSky, meshSun, meshMoon;
        MeshShares(tap, A.SunDirection, A.MoonDirection, captures, meshSky, meshSun, meshMoon);
        // The sky's light (with the bounce through doors and windows, as the surfaces get it: sky_common.fx
        // SkyVisibilityWith) and the sky itself seen from here (without it: a room's walls, not its 4 % of sky)
        float terrainOpen, portal;
        float terrainSky = SkyVisibilityParts(LoadTerrainCover(), AtmosphereLinearClamp, tap, up,
            TerrainTolerance(distance(tap, CamPos)), terrainOpen, portal);
        float skyVis = lerp(min(portal, meshSky), terrainSky * max(meshSky, A.SkyCaptureBounce), terrainOpen);
        float skySeen = lerp(min(portal, meshSky), terrainSky * meshSky, terrainOpen);
        float terrain, cloud, mt, mc;
        float sunVis = BodyVisible(tap, A.SunDirection, open, meshSun, terrain, cloud);
        float moonVis = any(A.MoonIlluminance > 0) ? BodyVisible(tap, A.MoonDirection, open, meshMoon, mt, mc) : 0.0;

        row0 = float4(sunVis, moonVis, skyVis, open);
        row1 = float4(terrain, meshSun, cloud, skySeen);
    }

    // A tap counts as far as it sees as much sky as the camera: one through a wall (a room's taps in the sunny street
    // outside) sees far more, and is left out; under a canopy the sky's share changes slowly and all count
    gSky[gi] = row0.z;
    GroupMemoryBarrierWithGroupSync();
    float weight = saturate(1.0 - abs(row0.z - gSky[0]) * METER_SKY_TOLERANCE);
    gSum[0][gi] = row0 * weight;
    gSum[1][gi] = row1 * weight;

    // The point lights round the camera: every slot, 64 at a time
    float pointSphere = 0, pointUp = 0;
    if (RecordsIdx != 0)
    {
        StructuredBuffer<PointLightRecord> records = ResourceDescriptorHeap[RecordsIdx];
        StructuredBuffer<row_major float4x4> transforms = ResourceDescriptorHeap[TransformsIdx];
        for (uint slot = gi; slot < RecordCountIdx; slot += METER_TAPS)
        {
            PointLightRecord rec = records[slot];
            if ((rec.Flags & POINT_LIGHT_ENABLED) == 0 || rec.Range <= 0.0)
                continue;
            float intensity = rec.Intensity * Luma(rec.Color);
            if ((rec.Flags & POINT_LIGHT_WHEN_DARK) != 0)
            {
                float lamp = OpenLamps;
                if (LampsUAVIdx != 0)
                {
                    RWStructuredBuffer<LampState> lamps = ResourceDescriptorHeap[LampsUAVIdx];
                    LampState s = lamps[slot];
                    if (s.Seen != 0 && s.Lamp >= 0.0)
                        lamp = s.Lamp;
                }
                intensity *= lamp;
            }
            if (intensity <= 0.0)
                continue;
            float3 d = mul(float4(0, 0, 0, 1), transforms[rec.TransformSlot]).xyz - MeterPosition;
            float dist2 = dot(d, d);
            if (dist2 >= rec.Range * rec.Range)
                continue;
            // As the lighting has it (light_point_cs.hlsl): inverse square, windowed to nothing at Range; no closer than
            // half a metre (a lantern by the camera would otherwise decide alone)
            float r4 = dist2 / (rec.Range * rec.Range);
            r4 *= r4;
            float window = saturate(1.0 - r4);
            float lux = intensity * window * window / max(dist2, 0.25);
            pointSphere += lux * 0.25;
            pointUp += lux * saturate(d.y * rsqrt(max(dist2, 1e-6)));
        }
    }
    gSum[2][gi] = float4(pointSphere, pointUp, weight, 0);
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = METER_TAPS / 2; stride > 0; stride >>= 1)
    {
        if (gi < stride)
        {
            [unroll] for (uint k = 0; k < 3; k++)
                gSum[k][gi] += gSum[k][gi + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi != 0)
        return;

    float weights = max(gSum[2][0].z, 1e-6);     // at least the camera's own tap: 1
    float4 shares = gSum[0][0] / weights;        // sun, moon, sky, open
    float4 parts = gSum[1][0] / weights;         // the sun's terrain, meshes and clouds shares; the sky seen
    float2 points = gSum[2][0].xy;               // point lights: sphere, level surface (sums)

    float sunLux = 0, moonLux = 0, sunHeight = 0, moonHeight = 0, skyLux = 0, groundLux = 0;
    if (AtmosphereIdx != 0)
    {
        AtmosphereParams A = LoadAtmosphere();
        float3 posKm = float3(0, AtmosphereViewHeight(A, MeterPosition.y), 0);
        sunLux = Luma(A.SunIlluminance * AtmTransmittanceToLight(A, posKm, A.SunDirection));
        moonLux = any(A.MoonIlluminance > 0) ? Luma(A.MoonIlluminance * AtmTransmittanceToLight(A, posKm, A.MoonDirection)) : 0.0;
        sunHeight = saturate(A.SunDirection.y);
        moonHeight = saturate(A.MoonDirection.y);
        // The ambient is E / π × Exposure (sky on level ground, the ground's bounce on a down-facing surface)
        float toLux = ATM_PI / max(Exposure, 1e-30);
        skyLux = Luma(SkyAmbientLight()) * toLux;
        groundLux = Luma(GroundBounceLight()) * toLux;
    }

    // The ground's bounce scaled as the light on the ground round the camera is to the open ground's
    float openGround = sunLux * sunHeight + moonLux * moonHeight + skyLux;
    float localGround = sunLux * sunHeight * shares.x + moonLux * moonHeight * shares.y + skyLux * shares.z;
    float bounce = openGround > 0.0 ? groundLux * localGround / openGround : 0.0;

    // The light on a sphere at the camera, lux
    float sphere = (sunLux * shares.x + moonLux * shares.y) * 0.25 + (skyLux * shares.z + bounce) * 0.5 + points.x;

    // The brightness round the camera (cd/m²), half above and half below. Above, the sky where the camera sees it (its
    // mean over the upper hemisphere from the sky probe's first two SH bands, so a sunset's bright horizon counts; not
    // the bounce's 4 %: that is light on a room's walls, not sky in view) and where it does not, the crowns, walls and
    // ceiling round it, lit by the light at the camera; below, the ground, lit as round the camera. The light at the
    // camera alone (the sphere) exposed a sunset for the land and washed the sky out (coast 18:00, 2.3 EV over the
    // screen's metering); the brightness round it alone left a mountain's shade as dark as the old metering of a sunlit
    // view (the bright sky is half of it). AutoExposure takes them half and half.
    float skyUpper = skyLux / ATM_PI;    // a sky alike all over: its mean luminance is E / π
    if (AtmosphereIdx != 0)
    {
        AtmosphereParams A = LoadAtmosphere();
        if (A.SkyProbeSH != 0)
        {
            // Stored as irradiance / π (band 0 × 1, band 1 × 2/3) and pre-exposed; ∫ over y > 0 of Y00 and Y10 (the
            // band 2 terms integrate to 0 there), / 2π
            StructuredBuffer<float4> sh = ResourceDescriptorHeap[A.SkyProbeSH];
            skyUpper = max(Luma(sh[0].rgb * 0.282095 + sh[1].rgb * 0.366452), 0.0) / max(Exposure, 1e-30);
        }
    }
    float surround = METER_SURROUND_ALBEDO * sphere / ATM_PI;
    float upper = lerp(surround, skyUpper, parts.w);
    float lower = bounce / ATM_PI;
    float adapt = 0.5 * (upper + lower);

    RWStructuredBuffer<float4> output = ResourceDescriptorHeap[OutputUAVIdx];
    output[0] = float4(adapt, sphere, points.x, weights);
    output[1] = float4(sunLux, shares.x, moonLux, shares.y);
    output[2] = float4(skyLux, shares.z, bounce, shares.w);
    output[3] = float4(parts.x, parts.y, parts.z, skyUpper);
}
