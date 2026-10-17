// The atmosphere's LUTs (Hillaire 2020), made each frame by Graphics/Atmosphere.cs:
//   CSTransmittance      256×64   transmittance from a height toward a zenith angle, to the top of the atmosphere
//   CSMultiScattering    32×32    Ψ_ms: light scattered twice or more, isotropic, per unit illuminance
//   CSSkyView            192×108  sky luminance from the camera, per unit illuminance; z = 0 sun, 1 the brightest moon
//   CSAerialPerspective  32³      light scattered in between camera and each froxel (rgb, sun + moon, × Exposure), transmittance (a)
//   CSAmbient            1 group  sky light on an up-facing surface and the ground's bounce light (Ambient buffer)
// The first two depend on the medium only (made when it changes); the rest on the camera and the lights.

#pragma kernel CSTransmittance
#pragma kernel CSMultiScattering
#pragma kernel CSSkyView
#pragma kernel CSAerialPerspective
#pragma kernel CSAmbient

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

struct ScatteringResult
{
    float3 L;               // scattered light reaching the start of the ray, per unit illuminance
    float3 OpticalDepth;
    float3 Transmittance;
    float3 MultiScatAs1;    // f_ms: light scattered as if the scattered light over the sphere integrated to 1
};

// Hillaire's IntegrateScatteredLuminance: ray-march from posKm along dir (both km / unit), light from lightDir.
//   ground:      add the light the ground (GroundAlbedo, Lambert) bounces back where the ray meets it
//   miePhase:    true = Rayleigh + Mie phases, false = isotropic (for the multiple scattering LUT)
//   multiScat:   add Ψ_ms (the multiple scattering LUT must exist)
//   tMaxMax:     stop the march here (km)
ScatteringResult IntegrateScatteredLuminance(AtmosphereParams A, float3 posKm, float3 dir, float3 lightDir,
                                             bool ground, float sampleCount, bool miePhase, bool multiScat,
                                             float tMaxMax)
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

        float3 S = earthShadow * transmittanceToLight * phaseTimesScattering + multiScattered * medium.scattering;

        // Energy scattered as if the light over the sphere were 1 (power series of f_ms, analytic per segment)
        float3 ms = medium.scattering;
        result.MultiScatAs1 += throughput * (ms - ms * sampleTransmittance) / medium.extinction;

        // Integrate along the segment (Frostbite 2015, slide 28)
        float3 Sint = (S - S * sampleTransmittance) / medium.extinction;
        L += throughput * Sint;
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

    ScatteringResult r = IntegrateScatteredLuminance(A, pos, dir, lightDir, true, 20.0, false, false, 9000000.0);
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

    float3 pos = float3(0, viewHeight, 0);
    ScatteringResult r = IntegrateScatteredLuminance(A, pos, dir, lightDir, false, 32.0, true, true, 9000000.0);

    RWTexture2D<float4> output = ResourceDescriptorHeap[id.z == 0 ? SkyViewSunUAVIdx : SkyViewMoonUAVIdx];
    output[id.xy] = float4(r.L, 1.0);
}

// ── Aerial perspective volume: froxels over the screen, slices by view depth ──

[numthreads(8, 8, 1)]
void CSAerialPerspective(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= AP_RES || id.y >= AP_RES || id.z >= AP_SLICES)
        return;
    AtmosphereParams A = LoadAtmosphere();

    // Camera-relative view direction through the froxel's centre (reverse-Z: the far plane is z = 0)
    float2 uv = (float2(id.xy) + 0.5) / AP_RES;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 farClip = mul(float4(ndc, 0, 1), CameraInverse);
    float3 dir = normalize(farClip.xyz / farClip.w);
    float3 forward = normalize(float3(View._13, View._23, View._33));

    float slice = (id.z + 0.5) / AP_SLICES;
    float viewDepth = slice * slice * A.APMaxDepth;                 // metres along the view axis
    float tMaxKm = viewDepth / max(dot(dir, forward), 0.05) * 0.001;

    float viewHeight = AtmosphereViewHeight(A, CamPos.y);
    float3 pos = float3(0, viewHeight, 0);

    // A froxel under the ground: stop at the ground
    float3 end = pos + dir * tMaxKm;
    if (length(end) <= A.BottomRadius)
    {
        end = normalize(end) * (A.BottomRadius + PLANET_RADIUS_OFFSET + 0.001);
        tMaxKm = length(end - pos);
        dir = (end - pos) / max(tMaxKm, 1e-6);
    }

    float sampleCount = clamp(2.0 * (id.z + 1), 2.0, 32.0);
    ScatteringResult r = IntegrateScatteredLuminance(A, pos, dir, A.SunDirection, false, sampleCount, true, true, tMaxKm);
    float3 L = r.L * A.SunIlluminance;
    if (any(A.MoonIlluminance > 0))
        L += IntegrateScatteredLuminance(A, pos, dir, A.MoonDirection, false, sampleCount, true, true, tMaxKm).L * A.MoonIlluminance;

    // Stored as the pixel stores light (× Exposure): sun and moon together, and small enough for half floats
    RWTexture3D<float4> output = ResourceDescriptorHeap[APUAVIdx];
    output[id] = float4(L * Exposure, dot(r.Transmittance, 1.0 / 3.0));
}

// ── Ambient: the sky's light on an up-facing surface, and the ground's bounce light on a down-facing one ──
// Ambient[0].rgb: sky (E_up / π × Exposure: a multiple of albedo, what the pixel stores)
// Ambient[1].rgb: ground (GroundAlbedo × (direct light on the ground + E_up) / π × Exposure)
// Ambient[2].rgb: sky in the engine's older units (Ambient[0] / LegacyLightScale), read back for Graphics.Daylight
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
    output[2] = float4(sky / max(LegacyLightScale, 1e-20), 0);
    // The LUT's transmittance toward the sun from the camera: Atmosphere.CheckTransmittance holds it against the CPU's
    output[3] = float4(AtmTransmittanceToLight(A, pos, A.SunDirection), 0);
}
