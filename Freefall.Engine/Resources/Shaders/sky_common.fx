// ────────────────────────────────────────────────
// Shared procedural sky color: Rayleigh + Mie scattering with artist overrides
// Used by mesh_skybox.fx, composition_cs.hlsl, ocean.fx, gbuffer_transparent.fx
//
// Requires: SceneConstants (b0) from common.fx for atmosphere parameters
// ────────────────────────────────────────────────

// Rayleigh phase function: (3/16π)(1 + cos²θ)
float RayleighPhase(float cosTheta)
{
    return (3.0 / (16.0 * 3.14159265)) * (1.0 + cosTheta * cosTheta);
}

// Henyey-Greenstein phase function for Mie scattering
float HGPhase(float cosTheta, float g)
{
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(abs(denom), 1.5));
}

// Day / sunset / night weights for a sun elevation (sunDir.y). Normalized, sum to 1.
// Must match SkyboxRenderer.ComputeDayNightFactors() on the CPU (drives light intensity/color).
void GetDayNightFactors(float sunElevation, out float dayFactor, out float sunsetFactor, out float nightFactor)
{
    dayFactor = saturate((sunElevation - 0.0) / 0.8);
    dayFactor = pow(dayFactor, 0.7);

    sunsetFactor = 0.0;
    if (sunElevation < 0.15 && sunElevation > -0.2)
    {
        sunsetFactor = 1.0 - abs((sunElevation - (-0.025)) / 0.175);
        sunsetFactor = max(0.0, sunsetFactor);
    }

    nightFactor = saturate((-sunElevation - 0.15) / 0.3);

    float total = dayFactor + sunsetFactor + nightFactor;
    if (total > 0.0)
    {
        dayFactor /= total;
        sunsetFactor /= total;
        nightFactor /= total;
    }
}

float3 GetSkyColor(float3 viewDir, float3 sunDir)
{
    float cosTheta = dot(viewDir, sunDir);
    float viewY = viewDir.y;
    float horizon = abs(viewY);

    // ── Sun elevation blend factors (same structure as before for lighting sync) ──
    float dayFactor, sunsetFactor, nightFactor;
    GetDayNightFactors(sunDir.y, dayFactor, sunsetFactor, nightFactor);

    // ── Day sky: scattering-shaped gradient ──
    // Zenith = deep tinted sky, horizon = brighter/whiter from longer scatter path
    // Rayleigh phase modulates hue based on sun angle
    float rayleighMod = RayleighPhase(cosTheta) * 2.0; // ~0.12 at 90°, ~0.36 at 0°
    float3 zenithColor = SkyTintColor * (0.55 + rayleighMod * 0.3) * AtmosphereDensity;
    float3 horizonWhite = lerp(SkyTintColor, float3(0.85, 0.88, 0.95), 0.55) * AtmosphereDensity;

    float gradientPow = pow(horizon, 0.45);
    float3 dayColor = lerp(horizonWhite, zenithColor, gradientPow);

    // ── Mie scattering: sun halo / forward scatter glow ──
    float mie = HGPhase(cosTheta, MieAnisotropy) * MieScattering;
    dayColor += float3(1.0, 0.95, 0.85) * mie * 3.0;

    // ── Horizon haze: colored atmospheric haze at low angles ──
    float horizonMask = 1.0 - smoothstep(0.0, HazeHeight, horizon);
    horizonMask *= horizonMask; // softer falloff
    float3 haze = HazeColor * horizonMask * HazeIntensity;
    dayColor += haze;

    // ── Sunset/sunrise tint injection ──
    // Warm colors near horizon during sunset, modulated by sun proximity
    float sunProximity = pow(saturate(cosTheta * 0.5 + 0.5), 2.0);
    float horizonWarm = pow(1.0 - saturate(abs(viewY) / 0.3), 2.0);
    float3 sunsetTint = SunsetTintColor * SunsetTintIntensity * sunProximity * horizonWarm;

    // ── Night sky ──
    float nightAltitude = saturate(abs(viewY));
    float3 nightColor = lerp(NightHorizonColor, NightSkyColor, pow(nightAltitude, 0.5));

    // ── Blend day/sunset/night using the same factors as SkyboxRenderer.UpdateSunLight() ──
    float3 skyColor = dayColor * dayFactor
                    + (dayColor + sunsetTint) * sunsetFactor
                    + nightColor * nightFactor;

    // ── Sun-scattered warm glow (shared across all phases except deep night) ──
    float horizonScatter = pow(1.0 - saturate(abs(viewY)), 3.0);
    float sunScatter = pow(saturate(cosTheta), 5.0);
    float scatterWeight = (horizonScatter + sunScatter * 0.5) * saturate(dayFactor + sunsetFactor * 0.5);
    skyColor += float3(1.0, 0.8, 0.6) * scatterWeight * 0.15;

    return max(skyColor, 0.0);
}

// ────────────────────────────────────────────────
// Cloud layer — shared by the sky dome (mesh_skybox.fx) and the cloud shadows on lit surfaces.
// The layer is a flat plane CloudAltitude above the camera, anchored to the world in XZ so the
// shadows it casts stay put on the ground when the camera moves.
// ────────────────────────────────────────────────

// Noise UV of a point on the cloud plane, given by its camera-relative XZ.
float2 CloudLayerUV(float2 relXZ)
{
    float2 uv = (CamPos.xz + relXZ) * 0.00035;

    // CloudTime is already the integral of CloudSpeed over time (SkyboxRenderer.Update), so it must
    // NOT be scaled by CloudSpeed again here: that rescales the whole history whenever the speed
    // changes and makes the clouds scrub forward/backward while a preset lerps.
    return uv + float2(CloudTime * 0.01, CloudTime * 0.005);
}

// Cloud density before detail erosion: multi-scale Perlin-Worley (LUT alpha) against the coverage threshold.
// Can be negative (below the threshold).
float CloudBaseDensity(Texture3D<float4> noiseLUT, SamplerState wrapSampler, float2 uv, float mip)
{
    float timeZ = CloudTime * 0.005;

    float baseShape = 0;
    baseShape += noiseLUT.SampleLevel(wrapSampler, float3(uv * 0.25,        timeZ        ), mip    ).a * 0.625;
    baseShape += noiseLUT.SampleLevel(wrapSampler, float3(uv * 0.5 + 0.37,  timeZ * 0.7  ), mip    ).a * 0.25;
    baseShape += noiseLUT.SampleLevel(wrapSampler, float3(uv * 1.0 + 0.71,  timeZ * 1.3  ), mip * 0.5).a * 0.125;

    float coverageNoise = noiseLUT.SampleLevel(wrapSampler, float3(uv * 0.06, timeZ * 0.15), 0).r;
    float coverage = saturate(CloudCoverage + (coverageNoise - 0.5) * 0.3);

    // Remap: only noise above the threshold survives as clouds, density = how far above it.
    // The octave-averaged Perlin-Worley shape (cloud_noise_gen: (perlin - 0.4*worley)/(1 - 0.4*worley)) only spans
    // ~0.15..0.62 (median ~0.37), so the original threshold (1 - coverage) left coverage 0..~0.5 dead (no clouds).
    // Above 0.62 the original mapping is kept exactly (presets are tuned there: Clear Day 0.62, Overcast 0.97);
    // below it the threshold walks linearly from 0.5 (coverage 0: the noise practically never exceeds it) to 0.38, so
    // low coverages give gradually more clouds instead of a dead zone.
    // Keep the (base - T) / (1 - T) density scale: normalising density to 0..1 pushed cores to CloudShadowColor
    // (near black in Clear Day) and turned fair-weather clouds into dark grey slabs.
    float threshold = min(1.0 - coverage, 0.5 - 0.1935 * coverage);
    return remap(baseShape, threshold, 1.0, 0.0, 1.0);
}

float CloudShadowDensityAt(Texture3D<float4> noiseLUT, SamplerState wrapSampler, float3 relPos, float3 toLight, float altitude)
{
    // Follow the light ray from the surface up to the cloud plane. For shadows the plane sits at an absolute
    // world height (the sky dome keeps it above the camera): anything camera-relative here makes the shadows
    // slide over the ground when the camera changes height.
    float t = max(altitude - (CamPos.y + relPos.y), 0.0) / toLight.y;

    // Real cloud shadows are km-sized: one covers a whole town and reads as "the light changed", not as a shadow.
    // CloudShadowScale shrinks the pattern at the cost of no longer lining up exactly with the clouds in the sky.
    // Only the position is scaled, not the drift: the pattern then passes at the same rate as the clouds overhead
    // (ground speed / scale). Scaling the drift too made the small shadows race across the ground.
    float scale = max(CloudShadowScale, 1.0);
    float2 drift = CloudLayerUV(-CamPos.xz);   // UV of the world origin = wind offset only
    float2 uv = (CloudLayerUV(relPos.xz + toLight.xz * t) - drift) * scale + drift;

    // No detail erosion, and a steeper ramp than the sky dome so the shadow has a readable edge
    return smoothstep(0.0, 0.2, CloudBaseDensity(noiseLUT, wrapSampler, uv, 0.0));
}

// Fraction of the sun's light (0..1) that reaches a surface point through the cloud layer.
// relPos: camera-relative position, toLight: normalized direction toward the sun.
float GetCloudShadow(SamplerState wrapSampler, float3 relPos, float3 toLight)
{
    if (CloudShadowStrength <= 0.0 || CloudNoiseLUTIdx == 0 || toLight.y <= 0.05)
        return 1.0;

    Texture3D<float4> noiseLUT = ResourceDescriptorHeap[CloudNoiseLUTIdx];

    float density = CloudShadowDensityAt(noiseLUT, wrapSampler, relPos, toLight, CloudAltitude);
    if (CloudAltitudeBlend < 0.999)
    {
        float densityFrom = CloudShadowDensityAt(noiseLUT, wrapSampler, relPos, toLight, CloudAltitudeFrom);
        density = lerp(densityFrom, density, smoothstep(0.0, 1.0, saturate(CloudAltitudeBlend)));
    }

    // A low sun stretches the projection toward infinity — fade the shadows out before that
    float lowSunFade = smoothstep(0.05, 0.3, toLight.y);

    return 1.0 - density * CloudShadowStrength * lowSunFade;
}

// ────────────────────────────────────────────────
// Unified aerial perspective: distance fog with per-pixel sky-derived color.
// Replaces both FOG() and ocean's hardcoded haze.
//
// Uses exponential-squared extinction with FogDensity from SceneConstants,
// and GetSkyColor() for the inscatter color along the view ray.
// Result: looking toward the sun → warm/golden fog, away → cool/blue.
// ────────────────────────────────────────────────
float3 ApplyAerialPerspective(float3 color, float3 worldPos, float3 camPos, float3 sunDir)
{
    float3 toSurface = worldPos - camPos;
    float dist = length(toSurface);
    float3 viewDir = toSurface / max(dist, 0.001);

    // Exponential-squared extinction (matches old FOG() math)
    float fogFactor = 1.0 - exp(-pow(dist * FogDensity, 2.0));
    fogFactor = saturate(fogFactor);

    // Inscatter: sky color along the view ray (clamp y to avoid underground sampling)
    float3 inscatter = GetSkyColor(float3(viewDir.x, max(viewDir.y, 0.01), viewDir.z), sunDir);

    return lerp(color, inscatter, fogFactor);
}
