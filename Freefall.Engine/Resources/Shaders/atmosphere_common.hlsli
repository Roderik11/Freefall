// ────────────────────────────────────────────────
// Physically based atmosphere (Hillaire 2020, "A Scalable and Production Ready Sky and Atmosphere Rendering
// Technique"): the medium, the LUT parametrisations and the lookups that every pass shares. The LUTs are made by
// atmosphere_cs.hlsl every frame (Graphics/Atmosphere.cs); the parameters live in one StructuredBuffer element whose
// bindless index is SceneConstants.AtmosphereIdx.
//
// Units: lengths in km, the planet's centre at the origin, +Y up; the camera sits at (0, BottomRadius + its world
// height in km, 0), world y = 0 being sea level. Scattering and extinction per km. The LUTs hold light per unit of
// light-source illuminance (1 lux): multiply by SunIlluminance / MoonIlluminance (lux, rgb) for cd/m², then by
// Exposure (SceneConstants) for what the pixel stores.
//
// Requires: common.fx (SceneConstants) included first.
// ────────────────────────────────────────────────
#ifndef ATMOSPHERE_COMMON
#define ATMOSPHERE_COMMON

#define ATM_PI 3.14159265

#define TRANSMITTANCE_LUT_W 256
#define TRANSMITTANCE_LUT_H 64
#define MULTISCAT_LUT_RES 32
#define SKYVIEW_LUT_W 192
#define SKYVIEW_LUT_H 108
#define AP_SLICES 32        // depth slices: slice s ends at ((s + 0.5) / AP_SLICES)² × APMaxDepth (view depth); its cells
                            // across the screen follow the view (Graphics/AerialPerspectiveVolume.cs)

static const float PLANET_RADIUS_OFFSET = 0.01;   // km: keeps sample points just off the ground

// Must match Graphics/Atmosphere.cs (GpuParams), field for field
struct AtmosphereParams
{
    float3 RayleighScattering;  float BottomRadius;        // per km; km
    float3 MieScattering;       float TopRadius;
    float3 MieExtinction;       float MieG;                // Henyey-Greenstein g of the aerosols and the fog
    float3 OzoneAbsorption;     float RayleighExpScale;    // -1 / scale height (per km)
    float3 FogScattering;       float MieExpScale;         // ground fog: a thin aerosol layer near the ground
    float3 FogExtinction;       float FogExpScale;
    float3 GroundAlbedo;        float SunAngularRadius;    // radians
    float3 SunDirection;        float CloudShade;          // average sunlight the clouds let through to the ground (0..1)
    float3 SunIlluminance;      float APMaxDepth;          // lux at the top of the atmosphere (colour × lux); metres
    float3 MoonDirection;       float _pad0;
    float3 MoonIlluminance;     float AirglowLuminance;    // lux (phase and colour included); cd/m² of the night sky
    uint TransmittanceLut;      uint MultiScatLut;         uint SkyViewSunLut;     uint SkyViewMoonLut;
    uint AerialPerspective;     uint Ambient;              uint TerrainHeightmap;  float TerrainMaxHeight;   // sky visibility: the terrain's baked heightmap (R16, 0 = none); metres at 1
    float3 TerrainOrigin;       float TerrainSizeX;        // world position of the terrain's corner (y = its base); metres it spans in x
    float TerrainSizeZ;         uint SkyCaptureViews;      uint SkyCaptureDepth;   uint SkyCaptureCount;     // sky visibility past the meshes (sky_capture.hlsli; 0 = none)
    float SkyCaptureBounce;                                // the least share of the sky a place the meshes cover keeps (light bouncing in)
    uint SkyProbeCube;          uint SkyProbeSH;           float SkyProbeMips;     // sky probe (sky_probe_cs.hlsl): TextureCube, 9 × float4 SH, mip count (0 = none)
    uint TerrainSunShadow;      uint TerrainMoonShadow;    float2 TerrainShadowOrigin;   // the terrain's shadow in the air per light (terrain_light_shadow_cs.hlsl; 0 = none); the maps' corner (world xz)
    float TerrainShadowSize;                               // metres the maps span each way
};

// The global root signature's linear clamp sampler (s2)
SamplerState AtmosphereLinearClamp : register(s2);

AtmosphereParams LoadAtmosphere()
{
    StructuredBuffer<AtmosphereParams> buffer = ResourceDescriptorHeap[AtmosphereIdx];
    return buffer[0];
}

// Camera position in atmosphere space (km)
float AtmosphereViewHeight(AtmosphereParams A, float worldHeightMeters)
{
    return A.BottomRadius + max(worldHeightMeters * 0.001, 0.001);
}

// ── Geometry ──

// Nearest intersection of a ray with a sphere, -1 if none (Hillaire's version: 0 when the origin is inside)
float RaySphereIntersectNearest(float3 r0, float3 rd, float3 s0, float sR)
{
    float a = dot(rd, rd);
    float3 s0_r0 = r0 - s0;
    float b = 2.0 * dot(rd, s0_r0);
    float c = dot(s0_r0, s0_r0) - sR * sR;
    float delta = b * b - 4.0 * a * c;
    if (delta < 0.0 || a == 0.0)
        return -1.0;
    float sol0 = (-b - sqrt(delta)) / (2.0 * a);
    float sol1 = (-b + sqrt(delta)) / (2.0 * a);
    if (sol0 < 0.0 && sol1 < 0.0)
        return -1.0;
    if (sol0 < 0.0)
        return max(0.0, sol1);
    if (sol1 < 0.0)
        return max(0.0, sol0);
    return max(0.0, min(sol0, sol1));
}

// ── Phase functions ──

float AtmRayleighPhase(float cosTheta)
{
    return 3.0 / (16.0 * ATM_PI) * (1.0 + cosTheta * cosTheta);
}

// Cornette-Shanks (the HG phase made to fit Mie better), as Hillaire's sample uses
float AtmMiePhase(float g, float cosTheta)
{
    float k = 3.0 / (8.0 * ATM_PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cosTheta * cosTheta) / pow(max(1.0 + g * g - 2.0 * g * cosTheta, 1e-4), 1.5);
}

// ── The medium ──

struct AtmMedium
{
    float3 scatteringRay;
    float3 scatteringMie;      // aerosols and ground fog: share the Mie phase
    float3 scattering;
    float3 extinction;
};

AtmMedium SampleAtmMedium(float3 posKm, AtmosphereParams A)
{
    float h = max(length(posKm) - A.BottomRadius, 0.0);
    float densityRay = exp(A.RayleighExpScale * h);
    float densityMie = exp(A.MieExpScale * h);
    float densityFog = exp(A.FogExpScale * h);
    // Ozone: a tent 30 km wide round 25 km (Bruneton's profile)
    float densityOzone = saturate(h < 25.0 ? h / 15.0 - 2.0 / 3.0 : -h / 15.0 + 8.0 / 3.0);

    AtmMedium m;
    m.scatteringRay = densityRay * A.RayleighScattering;
    m.scatteringMie = densityMie * A.MieScattering + densityFog * A.FogScattering;
    m.scattering = m.scatteringRay + m.scatteringMie;
    m.extinction = max(m.scatteringRay + densityMie * A.MieExtinction + densityFog * A.FogExtinction
                       + densityOzone * A.OzoneAbsorption, 1e-7);
    return m;
}

// ── Transmittance LUT (Bruneton's parametrisation: rays that do not meet the ground, from a height to the top) ──

float2 TransmittanceLutUv(AtmosphereParams A, float viewHeight, float viewZenithCos)
{
    float H = sqrt(max(0.0, A.TopRadius * A.TopRadius - A.BottomRadius * A.BottomRadius));
    float rho = sqrt(max(0.0, viewHeight * viewHeight - A.BottomRadius * A.BottomRadius));
    float discriminant = viewHeight * viewHeight * (viewZenithCos * viewZenithCos - 1.0) + A.TopRadius * A.TopRadius;
    float d = max(0.0, -viewHeight * viewZenithCos + sqrt(max(discriminant, 0.0)));
    float dMin = A.TopRadius - viewHeight;
    float dMax = rho + H;
    return float2((d - dMin) / max(dMax - dMin, 1e-6), rho / H);
}

void TransmittanceLutParams(AtmosphereParams A, float2 uv, out float viewHeight, out float viewZenithCos)
{
    float H = sqrt(A.TopRadius * A.TopRadius - A.BottomRadius * A.BottomRadius);
    float rho = H * uv.y;
    viewHeight = sqrt(rho * rho + A.BottomRadius * A.BottomRadius);
    float dMin = A.TopRadius - viewHeight;
    float dMax = rho + H;
    float d = dMin + uv.x * (dMax - dMin);
    viewZenithCos = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * viewHeight * d);
    viewZenithCos = clamp(viewZenithCos, -1.0, 1.0);
}

// Transmittance from a point at radius r (km) towards a direction of zenith cosine mu, to the top of the
// atmosphere. Does not test the ground: use AtmTransmittanceToLight where the planet may be in the way.
float3 AtmTransmittance(AtmosphereParams A, float r, float mu)
{
    Texture2D<float4> lut = ResourceDescriptorHeap[A.TransmittanceLut];
    return lut.SampleLevel(AtmosphereLinearClamp, TransmittanceLutUv(A, r, mu), 0).rgb;
}

// Same, 0 where the planet hides the light (the earth's shadow)
float3 AtmTransmittanceToLight(AtmosphereParams A, float3 posKm, float3 lightDir)
{
    float r = length(posKm);
    float3 up = posKm / r;
    float mu = dot(up, lightDir);
    if (RaySphereIntersectNearest(posKm, lightDir, float3(0, 0, 0), A.BottomRadius) >= 0.0)
        return 0;
    return AtmTransmittance(A, r, mu);
}

// How much of a light reaches a place in the air past the terrain: 0 in the terrain's shadow toward it (the shadow height
// maps of terrain_light_shadow_cs.hlsl, one per light), soft over 4 m either side of the shadow's top. 1 without the maps,
// outside them, and for a light at or below the horizon (no map then: the earth's shadow takes over). moon: the brightest
// moon's map instead of the sun's. worldPos is absolute. Without it the air in front of a mountain the sun stood behind
// glowed orange as if the sun shone through the rock (god rays: the mountains' shafts in the haze).
float AtmTerrainLightVisibility(AtmosphereParams A, float3 worldPos, bool moon)
{
    uint map = moon ? A.TerrainMoonShadow : A.TerrainSunShadow;
    if (map == 0)
        return 1.0;
    float2 uv = (worldPos.xz - A.TerrainShadowOrigin) / A.TerrainShadowSize;
    if (any(uv < 0.0) || any(uv > 1.0))
        return 1.0;
    Texture2D<float> shadowTop = ResourceDescriptorHeap[map];
    float top = shadowTop.SampleLevel(AtmosphereLinearClamp, uv, 0);
    return smoothstep(top - 4.0, top + 4.0, worldPos.y);
}

// ── Multiple scattering LUT (Hillaire's Ψ_ms: light scattered twice or more, per unit illuminance) ──

float3 AtmMultipleScattering(AtmosphereParams A, float3 posKm, float sunZenithCos)
{
    float2 uv = saturate(float2(sunZenithCos * 0.5 + 0.5,
                                (length(posKm) - A.BottomRadius) / (A.TopRadius - A.BottomRadius)));
    uv = (uv * (MULTISCAT_LUT_RES - 1) + 0.5) / MULTISCAT_LUT_RES;
    Texture2D<float4> lut = ResourceDescriptorHeap[A.MultiScatLut];
    return lut.SampleLevel(AtmosphereLinearClamp, uv, 0).rgb;
}

// ── Sky-view LUT (latitude packed toward the horizon, longitude relative to the light's azimuth) ──

float2 SkyViewLutUv(AtmosphereParams A, bool intersectGround, float viewZenithCos, float lightViewCos, float viewHeight)
{
    float vHorizon = sqrt(max(viewHeight * viewHeight - A.BottomRadius * A.BottomRadius, 0.0));
    float cosBeta = vHorizon / viewHeight;
    float beta = acos(cosBeta);
    float zenithHorizonAngle = ATM_PI - beta;

    float2 uv;
    if (!intersectGround)
    {
        float coord = acos(viewZenithCos) / zenithHorizonAngle;
        coord = 1.0 - sqrt(saturate(1.0 - coord));
        uv.y = coord * 0.5;
    }
    else
    {
        float coord = (acos(viewZenithCos) - zenithHorizonAngle) / beta;
        uv.y = sqrt(saturate(coord)) * 0.5 + 0.5;
    }
    uv.x = sqrt(saturate(-lightViewCos * 0.5 + 0.5));

    // From unit to the texel centres
    return (uv * float2(SKYVIEW_LUT_W - 1, SKYVIEW_LUT_H - 1) + 0.5) / float2(SKYVIEW_LUT_W, SKYVIEW_LUT_H);
}

void SkyViewLutParams(AtmosphereParams A, float2 uv, float viewHeight, out float viewZenithCos, out float lightViewCos)
{
    // From the texel centres to unit
    uv = (uv * float2(SKYVIEW_LUT_W, SKYVIEW_LUT_H) - 0.5) / float2(SKYVIEW_LUT_W - 1, SKYVIEW_LUT_H - 1);

    float vHorizon = sqrt(max(viewHeight * viewHeight - A.BottomRadius * A.BottomRadius, 0.0));
    float cosBeta = vHorizon / viewHeight;
    float beta = acos(cosBeta);
    float zenithHorizonAngle = ATM_PI - beta;

    if (uv.y < 0.5)
    {
        float coord = 1.0 - 2.0 * uv.y;
        coord = 1.0 - coord * coord;
        viewZenithCos = cos(zenithHorizonAngle * coord);
    }
    else
    {
        float coord = uv.y * 2.0 - 1.0;
        viewZenithCos = cos(zenithHorizonAngle + beta * coord * coord);
    }
    float coord = uv.x * uv.x;
    lightViewCos = -(coord * 2.0 - 1.0);
}

// Cosine of the horizontal angle between a view direction and a light (the sky-view LUT's longitude)
float LightViewCos(float3 dir, float3 lightDir)
{
    float2 d = dir.xz, l = lightDir.xz;
    float len = length(d) * length(l);
    return len > 1e-6 ? clamp(dot(d, l) / len, -1.0, 1.0) : 1.0;
}

// ── Lookups for the passes ──

// Night-sky glow of the upper air (airglow): brighter toward the horizon, where the line of sight crosses more of
// the glowing layer ~90 km up (van Rhijn), and dimmed by the air below it like starlight. Colour: the blue-grey of
// a dark night, luminance 1.
float3 AtmAirglow(AtmosphereParams A, float3 dir, float3 transmittance)
{
    float sinZ2 = saturate(1.0 - dir.y * dir.y);
    float vanRhijn = rsqrt(max(1.0 - 0.945 * sinZ2, 0.05));
    return float3(0.69, 1.02, 1.65) * (A.AirglowLuminance * vanRhijn) * transmittance;
}

// Sky luminance seen from the camera in a direction (cd/m², sun + moon + airglow, no discs, no clouds)
float3 AtmosphereSkyLuminance(AtmosphereParams A, float3 dir)
{
    float viewHeight = AtmosphereViewHeight(A, CamPos.y);
    float3 pos = float3(0, viewHeight, 0);
    bool hitsGround = RaySphereIntersectNearest(pos, dir, float3(0, 0, 0), A.BottomRadius) >= 0.0;

    Texture2D<float4> sunLut = ResourceDescriptorHeap[A.SkyViewSunLut];
    float3 L = sunLut.SampleLevel(AtmosphereLinearClamp,
        SkyViewLutUv(A, hitsGround, dir.y, LightViewCos(dir, A.SunDirection), viewHeight), 0).rgb * A.SunIlluminance;

    if (any(A.MoonIlluminance > 0))
    {
        Texture2D<float4> moonLut = ResourceDescriptorHeap[A.SkyViewMoonLut];
        L += moonLut.SampleLevel(AtmosphereLinearClamp,
            SkyViewLutUv(A, hitsGround, dir.y, LightViewCos(dir, A.MoonDirection), viewHeight), 0).rgb * A.MoonIlluminance;
    }

    float3 T = hitsGround ? 0 : AtmTransmittance(A, viewHeight, dir.y);
    return L + AtmAirglow(A, dir, T);
}

// Aerial perspective between the camera and a point on screen at a view depth (metres): rgb = light scattered in
// on the way (cd/m² × Exposure, i.e. what the pixel stores), a = how much of the point's own light gets through.
// From this view's volume (SceneConstants.AerialPerspectiveIdx); a build that predates it set the shared one in the
// atmosphere's parameters instead.
float4 AtmosphereAerialPerspective(AtmosphereParams A, float2 screenUV, float viewDepth)
{
    uint volume = AerialPerspectiveIdx != 0 ? AerialPerspectiveIdx : A.AerialPerspective;
    if (volume == 0)
        return float4(0, 0, 0, 1);
    float w = sqrt(saturate(viewDepth / A.APMaxDepth));
    float slice = w * AP_SLICES;
    float weight = 1.0;
    if (slice < 0.5)
    {
        // In front of the first slice: fade toward nothing at the camera
        weight = saturate(slice * 2.0);
        w = 0.5 / AP_SLICES;
    }
    Texture3D<float4> ap = ResourceDescriptorHeap[volume];
    float4 s = ap.SampleLevel(AtmosphereLinearClamp, float3(screenUV, w), 0);
    float3 inscatter = s.rgb * weight;   // already sun + moon, × Exposure (CSAerialPerspective)
    float transmittance = 1.0 - weight * (1.0 - s.a);
    return float4(inscatter, transmittance);
}

#endif // ATMOSPHERE_COMMON
