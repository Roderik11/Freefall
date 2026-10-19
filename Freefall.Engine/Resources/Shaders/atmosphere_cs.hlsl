// The atmosphere's LUTs (Hillaire 2020), made each frame by Graphics/Atmosphere.cs:
//   CSTransmittance      256×64   transmittance from a height toward a zenith angle, to the top of the atmosphere
//   CSMultiScattering    32×32    Ψ_ms: light scattered twice or more, isotropic, per unit illuminance
//   CSSkyView            192×108  sky luminance from the camera, per unit illuminance; z = 0 sun, 1 the brightest moon
//   CSAerialPerspective  ~20 px cells × 32 slices, sized to the view: light scattered in between camera and each froxel
//                        (rgb, sun + moon, × Exposure), transmittance (a); one march per column
//   CSAmbient            1 group  sky light on an up-facing surface and the ground's bounce light (Ambient buffer)
//   CSCloudDeck          1 group  the cloud layer round the camera (Ambient buffer, element 4; sky_common.fx CloudDeck),
//                                 first: the sky-view and the AP read it
// The first two depend on the medium only (made when it changes); the rest on the camera and the lights.

#pragma kernel CSTransmittance
#pragma kernel CSMultiScattering
#pragma kernel CSSkyView
#pragma kernel CSAerialPerspective
#pragma kernel CSAmbient
#pragma kernel CSCloudDeck

cbuffer PushConstants : register(b3)
{
    uint TransmittanceUAVIdx;
    uint MultiScatUAVIdx;
    uint SkyViewSunUAVIdx;
    uint SkyViewMoonUAVIdx;
    uint APUAVIdx;
    uint AmbientUAVIdx;
};

#include "common.fx"
#include "atmosphere_common.hlsli"
#include "sky_common.fx"

SamplerState CloudWrapSampler : register(s0);       // linear wrap (the cloud noise)

struct ScatteringResult
{
    float3 L;               // scattered light reaching the start of the ray, per unit illuminance
    float3 OpticalDepth;
    float3 Transmittance;
    float3 MultiScatAs1;    // f_ms: light scattered as if the scattered light over the sphere integrated to 1
    float3 LBelow;          // the part of L scattered below the cloud layer (deck)
};

// Hillaire's IntegrateScatteredLuminance: ray-march from posKm along dir (both km / unit), light from lightDir.
//   ground:      add the light the ground (GroundAlbedo, Lambert) bounces back where the ray meets it
//   miePhase:    true = Rayleigh + Mie phases, false = isotropic (for the multiple scattering LUT)
//   multiScat:   add Ψ_ms (the multiple scattering LUT must exist)
//   tMaxMax:     stop the march here (km)
//   deck:        the cloud layer over the air: x its height (km over the ground, flat as the world's: posKm at x = 0,
//                z = 0); below it y the share of the light's beam, z the share it diffuses, w the share of Ψ_ms left
//                (CloudAirLight); (0, 1, 0, 1) = none
ScatteringResult IntegrateScatteredLuminance(AtmosphereParams A, float3 posKm, float3 dir, float3 lightDir,
                                             bool ground, float sampleCount, bool miePhase, bool multiScat,
                                             float tMaxMax, float4 deck)
{
    ScatteringResult result = (ScatteringResult)0;
    result.Transmittance = 1.0;

    float3 earthO = float3(0, 0, 0);
    float tBottom = RaySphereIntersectNearest(posKm, dir, earthO, A.BottomRadius);
    float tTop = RaySphereIntersectNearest(posKm, dir, earthO, A.TopRadius);
    float tMax = 0.0;
    if (tBottom < 0.0)
    {
        if (tTop < 0.0)
            return result;   // neither the planet nor the atmosphere: nothing
        tMax = tTop;
    }
    else
    {
        tMax = tTop > 0.0 ? min(tTop, tBottom) : tBottom;
    }
    tMax = min(tMax, tMaxMax);

    float cosTheta = dot(dir, lightDir);
    float miePhaseValue = AtmMiePhase(A.MieG, cosTheta);
    float rayleighPhaseValue = AtmRayleighPhase(cosTheta);
    const float uniformPhase = 1.0 / (4.0 * ATM_PI);

    float3 L = 0;
    float3 throughput = 1.0;
    float3 opticalDepth = 0;
    float t = 0.0;
    const float sampleSegmentT = 0.3;
    for (float s = 0.0; s < sampleCount; s += 1.0)
    {
        // Exact segment lengths, important for the multiple scattering's accuracy
        float newT = tMax * (s + sampleSegmentT) / sampleCount;
        float dt = newT - t;
        t = newT;

        float3 P = posKm + t * dir;
        AtmMedium medium = SampleAtmMedium(P, A);
        float3 sampleOpticalDepth = medium.extinction * dt;
        float3 sampleTransmittance = exp(-sampleOpticalDepth);
        opticalDepth += sampleOpticalDepth;

        float pHeight = length(P);
        float3 up = P / pHeight;
        float lightZenithCos = dot(lightDir, up);
        float3 transmittanceToLight = AtmTransmittance(A, pHeight, lightZenithCos);

        float3 phaseTimesScattering = miePhase
            ? medium.scatteringMie * miePhaseValue + medium.scatteringRay * rayleighPhaseValue
            : medium.scattering * uniformPhase;

        // The earth's shadow
        float tEarth = RaySphereIntersectNearest(P, lightDir, earthO + PLANET_RADIUS_OFFSET * up, A.BottomRadius);
        float earthShadow = tEarth >= 0.0 ? 0.0 : 1.0;

        float3 multiScattered = multiScat ? AtmMultipleScattering(A, P, lightZenithCos) : 0;

        // Under the cloud layer (±100 m round its plane): its beam with the phases, and what it diffuses (CloudDiffusedLight).
        // Its plane is flat, as the sky dome draws it: over the curved ground the air past where the ray meets the plane
        // still dimmed the dome's cloud without counting as the light in front of it, a dark band under a deck.
        float below = 1.0 - smoothstep(deck.x - 0.1, deck.x + 0.1, P.y - A.BottomRadius);
        float3 lit = phaseTimesScattering * lerp(1.0, deck.y, below)
                   + medium.scattering * CloudDiffusedLight(deck.z * below, lightZenithCos);

        float3 S = earthShadow * transmittanceToLight * lit + multiScattered * (medium.scattering * lerp(1.0, deck.w, below));

        // Energy scattered as if the light over the sphere were 1 (power series of f_ms, analytic per segment)
        float3 ms = medium.scattering;
        result.MultiScatAs1 += throughput * (ms - ms * sampleTransmittance) / medium.extinction;

        // Integrate along the segment (Frostbite 2015, slide 28)
        float3 Sint = (S - S * sampleTransmittance) / medium.extinction;
        L += throughput * Sint;
        result.LBelow += throughput * Sint * below;
        throughput *= sampleTransmittance;
    }

    if (ground && tMax == tBottom && tBottom > 0.0)
    {
        // Light bounced off the ground
        float3 P = posKm + tBottom * dir;
        float pHeight = length(P);
        float3 up = P / pHeight;
        float lightZenithCos = dot(lightDir, up);
        float3 transmittanceToLight = AtmTransmittance(A, pHeight, lightZenithCos);
        float NdotL = saturate(dot(up, lightDir));
        L += transmittanceToLight * throughput * NdotL * A.GroundAlbedo / ATM_PI;
    }

    result.L = L;
    result.OpticalDepth = opticalDepth;
    result.Transmittance = throughput;
    return result;
}

// ── Transmittance LUT ──

[numthreads(8, 8, 1)]
void CSTransmittance(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= TRANSMITTANCE_LUT_W || id.y >= TRANSMITTANCE_LUT_H)
        return;
    AtmosphereParams A = LoadAtmosphere();

    float2 uv = (float2(id.xy) + 0.5) / float2(TRANSMITTANCE_LUT_W, TRANSMITTANCE_LUT_H);
    float viewHeight, viewZenithCos;
    TransmittanceLutParams(A, uv, viewHeight, viewZenithCos);

    float3 pos = float3(0, viewHeight, 0);
    float3 dir = float3(sqrt(saturate(1.0 - viewZenithCos * viewZenithCos)), viewZenithCos, 0);

    // Optical depth to the top of the atmosphere (these rays never meet the ground)
    float tMax = RaySphereIntersectNearest(pos, dir, float3(0, 0, 0), A.TopRadius);
    const int steps = 40;
    float dt = max(tMax, 0.0) / steps;
    float3 opticalDepth = 0;
    for (int i = 0; i < steps; i++)
        opticalDepth += SampleAtmMedium(pos + ((i + 0.5) * dt) * dir, A).extinction * dt;

    RWTexture2D<float4> output = ResourceDescriptorHeap[TransmittanceUAVIdx];
    output[id.xy] = float4(exp(-opticalDepth), 1.0);
}

// ── Multiple scattering LUT: one group per texel, 64 directions over the sphere ──

groupshared float3 MultiScatAs1Shared[64];
groupshared float3 LShared[64];

[numthreads(1, 1, 64)]
void CSMultiScattering(uint3 id : SV_DispatchThreadID)
{
    AtmosphereParams A = LoadAtmosphere();

    float2 uv = (float2(id.xy) + 0.5) / MULTISCAT_LUT_RES;
    uv = (uv * MULTISCAT_LUT_RES - 0.5) / (MULTISCAT_LUT_RES - 1);   // texel centres to unit

    float sunZenithCos = uv.x * 2.0 - 1.0;
    float3 lightDir = float3(sqrt(saturate(1.0 - sunZenithCos * sunZenithCos)), sunZenithCos, 0);
    float viewHeight = A.BottomRadius + saturate(uv.y + PLANET_RADIUS_OFFSET) * (A.TopRadius - A.BottomRadius - PLANET_RADIUS_OFFSET);
    float3 pos = float3(0, viewHeight, 0);

    // One of 8 × 8 directions, uniform over the sphere
    const float sqrtSamples = 8.0;
    float i = 0.5 + float(id.z / 8);
    float j = 0.5 + float(id.z - (id.z / 8) * 8);
    float theta = 2.0 * ATM_PI * (i / sqrtSamples);
    float phi = acos(1.0 - 2.0 * (j / sqrtSamples));
    float3 dir = float3(cos(theta) * sin(phi), cos(phi), sin(theta) * sin(phi));

    ScatteringResult r = IntegrateScatteredLuminance(A, pos, dir, lightDir, true, 20.0, false, false, 9000000.0,
                                                     float4(0, 1, 0, 1));
    const float sphereSolidAngle = 4.0 * ATM_PI;
    MultiScatAs1Shared[id.z] = r.MultiScatAs1 * sphereSolidAngle / (sqrtSamples * sqrtSamples);
    LShared[id.z] = r.L * sphereSolidAngle / (sqrtSamples * sqrtSamples);
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 32; stride > 0; stride >>= 1)
    {
        if (id.z < stride)
        {
            MultiScatAs1Shared[id.z] += MultiScatAs1Shared[id.z + stride];
            LShared[id.z] += LShared[id.z + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (id.z > 0)
        return;

    const float isotropicPhase = 1.0 / sphereSolidAngle;
    float3 multiScatAs1 = MultiScatAs1Shared[0] * isotropicPhase;     // f_ms
    float3 inScattered = LShared[0] * isotropicPhase;                  // L_2nd order
    float3 L = inScattered / max(1.0 - multiScatAs1, 1e-4);           // all orders: 1 / (1 - f_ms)

    RWTexture2D<float4> output = ResourceDescriptorHeap[MultiScatUAVIdx];
    output[id.xy] = float4(L, 1.0);
}

// ── Sky-view LUT, one per light (z = 0 the sun, 1 the brightest moon) ──

[numthreads(8, 8, 1)]
void CSSkyView(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= SKYVIEW_LUT_W || id.y >= SKYVIEW_LUT_H)
        return;
    AtmosphereParams A = LoadAtmosphere();

    float viewHeight = AtmosphereViewHeight(A, CamPos.y);
    float2 uv = (float2(id.xy) + 0.5) / float2(SKYVIEW_LUT_W, SKYVIEW_LUT_H);
    float viewZenithCos, lightViewCos;
    SkyViewLutParams(A, uv, viewHeight, viewZenithCos, lightViewCos);

    // The light in the local x-y plane at its zenith angle; the view at its zenith angle and its horizontal angle to
    // the light (the sky is the same all round, apart from where the light is)
    float3 light = id.z == 0 ? A.SunDirection : A.MoonDirection;
    float lightZenithCos = light.y;
    float3 lightDir = normalize(float3(sqrt(saturate(1.0 - lightZenithCos * lightZenithCos)), lightZenithCos, 0));
    float viewZenithSin = sqrt(saturate(1.0 - viewZenithCos * viewZenithCos));
    float3 dir = float3(viewZenithSin * lightViewCos, viewZenithCos, viewZenithSin * sqrt(saturate(1.0 - lightViewCos * lightViewCos)));

    // The air below the cloud layer (the sky dome's plane, CloudAltitude over the camera) lies in its shade. The LUT knows
    // a direction only (and the light's angle from it, not its side), so the shade is the layer's mean round the camera
    // (CloudDeck): under a deck the haze toward the sun no longer glowed as if the sun shone through the clouds.
    float deckDensity = id.z == 0 ? CloudDeck().x : CloudDeck().y;
    float4 deck = float4(viewHeight - A.BottomRadius + CloudAltitude * 0.001, 1.0 - deckDensity,
                         (1.0 - CloudShadowStrength) * deckDensity, 1.0 - CloudSkyDimming(deckDensity));

    float3 pos = float3(0, viewHeight, 0);
    ScatteringResult r = IntegrateScatteredLuminance(A, pos, dir, lightDir, false, 32.0, true, true, 9000000.0, deck);

    // a: the share of the light scattered below the layer, the air in front of a cloud (sky_common.fx GetCloudLight).
    // Taken as the share of the optical depth there, the lit air above a deck (behind the cloud) showed through a
    // shaded deck as a bright band over a dark horizon.
    float total = dot(r.L, float3(0.2126, 0.7152, 0.0722));
    float belowShare = total > 1e-30 ? saturate(dot(r.LBelow, float3(0.2126, 0.7152, 0.0722)) / total) : 1.0;
    RWTexture2D<float4> output = ResourceDescriptorHeap[id.z == 0 ? SkyViewSunUAVIdx : SkyViewMoonUAVIdx];
    output[id.xy] = float4(r.L, belowShare);
}

// ── Aerial perspective volume: froxels over the screen, slices by view depth ──
// One thread per column: a single march front to back through every slice (all the slices of a column share the
// direction through its centre), writing each slice's light and transmittance as the march passes the slice's depth,
// AP_STEPS samples per slice. The volume is sized to the view (Graphics/AerialPerspectiveVolume.cs: a cell per ~20
// pixels): at 32×32 the terrain's shadow in the haze showed 40-110 px steps along a ridge, and a march per froxel from
// the camera (up to 32 samples each, ~784 a column) would have cost ~6× as much at that size.
// The terrain's shadow is not sampled at a point per step: a step a few hundred metres long that a shadow edge crosses
// was lit or dark as a whole, all the columns together, and the haze showed contour bands along the shafts. Instead the
// step's lit share comes from how far the ray lies above the shadow's top at its start, middle and end, linear between
// (APLitShare): it changes smoothly as the ray turns past an edge. Four times the samples cost 0.1 ms more and did
// less; a 4×4 dither of the sample places over the columns, smoothed by the lookup's B-spline, left a checkerboard.

#define AP_STEPS 3

// ∫ saturate(x) dx from 0: the soft edge of the terrain's shadow as a ramp
float APRampIntegral(float x)
{
    return x <= 0.0 ? 0.0 : (x >= 1.0 ? x - 0.5 : 0.5 * x * x);
}

// The lit share of a stretch of the ray along which its margin above the shadow's top (AtmTerrainLightMargin) goes
// linearly from m0 to m1: the mean of the ±4 m ramp over it
float APLitShare(float m0, float m1)
{
    float a = (m0 + 4.0) / 8.0, b = (m1 + 4.0) / 8.0;
    if (abs(b - a) < 1e-3)
        return saturate(0.5 * (a + b));
    return (APRampIntegral(b) - APRampIntegral(a)) / (b - a);
}

// One body's light scattered toward the camera at a sample, per unit illuminance and km: single scattering through the
// air above (0 in the earth's shadow) and the cloud layer (CloudAirLight: x its beam, with the given phase-weighted
// scattering and × terrainLit, the step's share out of the terrain's shadow; y what it diffuses, scattered evenly and
// coming from the whole cloud base, so past the terrain's shadow toward the light too), plus Ψ_ms × cloud.z
float3 APSampleLight(AtmosphereParams A, AtmMedium medium, float3 P, float3 up, float pHeight, float terrainLit,
                     float3 cloud, float3 lightDir, float miePhase, float rayleighPhase)
{
    float lightZenithCos = dot(lightDir, up);
    float3 toLight = AtmTransmittance(A, pHeight, lightZenithCos);
    float tEarth = RaySphereIntersectNearest(P, lightDir, PLANET_RADIUS_OFFSET * up, A.BottomRadius);
    float3 single = (medium.scatteringMie * miePhase + medium.scatteringRay * rayleighPhase) * (terrainLit * cloud.x)
                  + medium.scattering * CloudDiffusedLight(cloud.y, lightZenithCos);
    return (tEarth >= 0.0 ? 0.0 : 1.0) * toLight * single
         + AtmMultipleScattering(A, P, lightZenithCos) * (medium.scattering * cloud.z);
}

// A point of the ray in the world (the atmosphere's frame: km round the camera, the ground flat as the world's)
float3 APWorldPos(AtmosphereParams A, float3 P)
{
    return float3(CamPos.x + P.x * 1000.0, (P.y - A.BottomRadius) * 1000.0, CamPos.z + P.z * 1000.0);
}

[numthreads(8, 8, 1)]
void CSAerialPerspective(uint3 id : SV_DispatchThreadID)
{
    RWTexture3D<float4> output = ResourceDescriptorHeap[APUAVIdx];
    uint width, height, slices;
    output.GetDimensions(width, height, slices);
    // id.z > 0: a dispatch over every froxel (builds before the column march) does each column once
    if (id.x >= width || id.y >= height || id.z != 0)
        return;
    AtmosphereParams A = LoadAtmosphere();

    // Camera-relative view direction through the column's centre (reverse-Z: the far plane is z = 0)
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 farClip = mul(float4(ndc, 0, 1), CameraInverse);
    float3 dir = normalize(farClip.xyz / farClip.w);
    float3 forward = normalize(float3(View._13, View._23, View._33));
    float kmPerMetre = 0.001 / max(dot(dir, forward), 0.05);      // km along the ray per metre of view depth

    // The march ends where the ray meets the ground (the slices past it keep the last value) or leaves the air
    float3 pos = float3(0, AtmosphereViewHeight(A, CamPos.y), 0);
    float tGround = RaySphereIntersectNearest(pos, dir, float3(0, 0, 0), A.BottomRadius);
    float tTop = RaySphereIntersectNearest(pos, dir, float3(0, 0, 0), A.TopRadius);
    float tEnd = tGround >= 0.0 ? tGround : max(tTop, 0.0);

    // The phases depend on the column's direction only
    float sunCos = dot(dir, A.SunDirection), moonCos = dot(dir, A.MoonDirection);
    float sunMie = AtmMiePhase(A.MieG, sunCos), sunRayleigh = AtmRayleighPhase(sunCos);
    float moonMie = AtmMiePhase(A.MieG, moonCos), moonRayleigh = AtmRayleighPhase(moonCos);
    bool moonLit = any(A.MoonIlluminance > 0);
    float4 deck = CloudDeck();      // the layer's mean density (x, y) and the clouds in front of the sun and the moon (z, w)
    float sunSkyDimming = CloudSkyDimming(deck.x), moonSkyDimming = CloudSkyDimming(deck.y);

    float3 L = 0;               // × the illuminances
    float3 throughput = 1.0;
    float t = 0.0;
    // The ray's margin above the terrain's shadow toward the sun and the moon where the step starts
    float sunMargin = AtmTerrainLightMargin(A, APWorldPos(A, pos), false);
    float moonMargin = moonLit ? AtmTerrainLightMargin(A, APWorldPos(A, pos), true) : 1e4;
    for (uint s = 0; s < slices; s++)
    {
        float w = (s + 0.5) / slices;
        float tSlice = min(w * w * A.APMaxDepth * kmPerMetre, tEnd);     // slice s: view depth w² × APMaxDepth
        float dt = max(tSlice - t, 0.0) / AP_STEPS;
        if (dt > 0.0)
        {
            for (uint k = 0; k < AP_STEPS; k++)
            {
                float3 P = pos + (t + (k + 0.5) * dt) * dir;
                float3 worldMid = APWorldPos(A, P);
                float3 worldEnd = APWorldPos(A, pos + (t + (k + 1.0) * dt) * dir);
                float pHeight = length(P);
                float3 up = P / pHeight;
                AtmMedium medium = SampleAtmMedium(P, A);

                float midMargin = AtmTerrainLightMargin(A, worldMid, false);
                float endMargin = AtmTerrainLightMargin(A, worldEnd, false);
                float sunLit = 0.5 * (APLitShare(sunMargin, midMargin) + APLitShare(midMargin, endMargin));
                sunMargin = endMargin;
                // The cloud layer at the step's middle (the near fog volume's own CloudAirLight, so the two meet at
                // its range without a seam)
                float3 relMid = worldMid - CamPos;
                float3 sunCloud = CloudAirLight(CloudWrapSampler, relMid, A.SunDirection, deck.z, sunSkyDimming);
                float3 S = A.SunIlluminance * APSampleLight(A, medium, P, up, pHeight, sunLit, sunCloud, A.SunDirection,
                                                            sunMie, sunRayleigh);
                if (moonLit)
                {
                    midMargin = AtmTerrainLightMargin(A, worldMid, true);
                    endMargin = AtmTerrainLightMargin(A, worldEnd, true);
                    float moonShare = 0.5 * (APLitShare(moonMargin, midMargin) + APLitShare(midMargin, endMargin));
                    moonMargin = endMargin;
                    float3 moonCloud = CloudAirLight(CloudWrapSampler, relMid, A.MoonDirection, deck.w, moonSkyDimming);
                    S += A.MoonIlluminance * APSampleLight(A, medium, P, up, pHeight, moonShare, moonCloud,
                                                           A.MoonDirection, moonMie, moonRayleigh);
                }

                // ∫ S e^(−σx) dx over the step (Frostbite 2015), written so a short step of clear air keeps its light
                float3 x = medium.extinction * dt;
                float3 through = exp(-x);
                float3 share = lerp(1.0 - 0.5 * x, (1.0 - through) / max(x, 1e-12), step(1e-4, x));
                L += throughput * S * dt * share;
                throughput *= through;
            }
            t = tSlice;
        }
        // Stored as the pixel stores light (× Exposure): sun and moon together, and small enough for half floats
        output[uint3(id.xy, s)] = float4(L * Exposure, dot(throughput, 1.0 / 3.0));
    }
}

// ── Ambient: the sky's light on an up-facing surface, and the ground's bounce light on a down-facing one ──
// Ambient[0].rgb: sky (E_up / π × Exposure: a multiple of albedo, what the pixel stores)
// Ambient[1].rgb: ground (GroundAlbedo × (direct light on the ground + E_up) / π × Exposure)
// Ambient[2].rgb: sky in the engine's older units (Ambient[0] / SunLightScale), read back for Graphics.Daylight
// Ambient[3].rgb: transmittance toward the sun from the camera (LUT), read back for the CPU/GPU consistency check

groupshared float3 SkyIrradianceShared[64];

[numthreads(64, 1, 1)]
void CSAmbient(uint3 id : SV_DispatchThreadID)
{
    AtmosphereParams A = LoadAtmosphere();
    float viewHeight = AtmosphereViewHeight(A, CamPos.y);
    float3 pos = float3(0, viewHeight, 0);

    Texture2D<float4> sunLut = ResourceDescriptorHeap[A.SkyViewSunLut];
    Texture2D<float4> moonLut = ResourceDescriptorHeap[A.SkyViewMoonLut];

    // 16 × 16 directions over the upper hemisphere, uniform in (cos zenith, azimuth): dω = dμ dφ
    float3 E = 0;
    for (uint k = 0; k < 4; k++)
    {
        uint n = id.x * 4 + k;
        float mu = (float(n / 16) + 0.5) / 16.0;
        float phi = (float(n % 16) + 0.5) / 16.0 * 2.0 * ATM_PI;
        float sinZ = sqrt(saturate(1.0 - mu * mu));
        float3 dir = float3(sinZ * cos(phi), mu, sinZ * sin(phi));
        float3 L = sunLut.SampleLevel(AtmosphereLinearClamp,
            SkyViewLutUv(A, false, mu, LightViewCos(dir, A.SunDirection), viewHeight), 0).rgb * A.SunIlluminance;
        L += moonLut.SampleLevel(AtmosphereLinearClamp,
            SkyViewLutUv(A, false, mu, LightViewCos(dir, A.MoonDirection), viewHeight), 0).rgb * A.MoonIlluminance;
        L += AtmAirglow(A, dir, AtmTransmittance(A, viewHeight, mu));
        E += L * mu * (1.0 / 16.0) * (2.0 * ATM_PI / 16.0);
    }
    SkyIrradianceShared[id.x] = E;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 32; stride > 0; stride >>= 1)
    {
        if (id.x < stride)
            SkyIrradianceShared[id.x] += SkyIrradianceShared[id.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (id.x > 0)
        return;

    float3 skyUp = SkyIrradianceShared[0];   // lux on an up-facing surface from the sky

    // Direct light on flat ground: the sun and the moon through the air (0 below the horizon), the clouds' average shade
    float3 direct = A.SunIlluminance * AtmTransmittanceToLight(A, pos, A.SunDirection) * saturate(A.SunDirection.y)
                  + A.MoonIlluminance * AtmTransmittanceToLight(A, pos, A.MoonDirection) * saturate(A.MoonDirection.y);
    direct *= A.CloudShade;

    RWStructuredBuffer<float4> output = ResourceDescriptorHeap[AmbientUAVIdx];
    float3 sky = skyUp / ATM_PI * Exposure;
    output[0] = float4(sky, 0);
    output[1] = float4(A.GroundAlbedo * (direct + skyUp) / ATM_PI * Exposure, 0);
    output[2] = float4(sky / max(SunLightScale, 1e-20), 0);
    // The LUT's transmittance toward the sun from the camera: Atmosphere.CheckTransmittance holds it against the CPU's
    output[3] = float4(AtmTransmittanceToLight(A, pos, A.SunDirection), 0);
}

// ── The cloud layer round the camera (sky_common.fx CloudDeck), into the Ambient buffer's element 4 ──
// x, y: the layer's mean shadow density toward the sun / the brightest moon over CLOUD_DECK_RADIUS round the camera, 256
// places on a sunflower spiral (the sky-view LUT lights its air below the layer alike in every direction); z, w: the
// sky's clouds in front of the sun / the moon, as the sky dome hides their discs (CloudSunDiscCover; the moon's
// 1 − exp(−15 × density)). Below 17° the
// light's x, y fade into its z, w, as CloudAirLight fades: the stretched shadow pattern is no use there, the clouds in
// front of the light tell whether a deck hides it (a deck runs on to the horizon, a broken layer leaves it clear).
// Dispatched first in Atmosphere.Render (the sky-view and the AP read it); 0 without clouds.

#define CLOUD_DECK_RADIUS 16000.0       // metres; the shadow pattern's largest octave repeats every ~11 km

groupshared float2 CloudDeckShared[64];

float CloudCoverToward(float3 lightDir, bool sun)
{
    float3 dir = normalize(float3(lightDir.x, max(lightDir.y, 0.002), lightDir.z));     // at or under the horizon: its clouds
    float density = GetCloudsAt(CloudWrapSampler, dir, CloudAltitude, 0.0);
    if (CloudAltitudeBlend < 0.999)
        density = lerp(GetCloudsAt(CloudWrapSampler, dir, CloudAltitudeFrom, 0.0), density,
                       smoothstep(0.0, 1.0, saturate(CloudAltitudeBlend)));
    return sun ? CloudSunDiscCover(density) : 1.0 - exp(-15.0 * density);
}

[numthreads(64, 1, 1)]
void CSCloudDeck(uint3 id : SV_DispatchThreadID)
{
    AtmosphereParams A = LoadAtmosphere();
    bool clouds = CloudShadowStrength > 0.0 && CloudNoiseLUTIdx != 0;
    bool moonLit = any(A.MoonIlluminance > 0);

    float2 sum = 0;
    if (clouds)
    {
        for (uint k = 0; k < 4; k++)
        {
            uint n = id.x * 4 + k;
            float r = sqrt((n + 0.5) / 256.0) * CLOUD_DECK_RADIUS;
            float phi = n * 2.3999632;      // the golden angle
            float3 relPos = float3(r * cos(phi), 0.0, r * sin(phi));
            if (A.SunDirection.y > 0.05)
                sum.x += CloudShadowDensity(CloudWrapSampler, relPos, A.SunDirection);
            if (moonLit && A.MoonDirection.y > 0.05)
                sum.y += CloudShadowDensity(CloudWrapSampler, relPos, A.MoonDirection);
        }
    }
    CloudDeckShared[id.x] = sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 32; stride > 0; stride >>= 1)
    {
        if (id.x < stride)
            CloudDeckShared[id.x] += CloudDeckShared[id.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (id.x > 0)
        return;

    RWStructuredBuffer<float4> output = ResourceDescriptorHeap[AmbientUAVIdx];
    if (!clouds)
    {
        output[4] = 0;
        return;
    }
    float2 mean = CloudDeckShared[0] / 256.0;
    float sunCover = CloudCoverToward(A.SunDirection, true);
    float moonCover = moonLit ? CloudCoverToward(A.MoonDirection, false) : 0.0;
    output[4] = float4(lerp(sunCover, mean.x, smoothstep(0.05, 0.3, A.SunDirection.y)),
                       lerp(moonCover, mean.y, smoothstep(0.05, 0.3, A.MoonDirection.y)), sunCover, moonCover);
}
