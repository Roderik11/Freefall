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

// Relative air mass along a direction of elevation sin = mu: 1 at the zenith, ~9 at 6°, ~40 at the horizon
// (a curved-atmosphere fit; 1/mu would go to infinity at the horizon).
float AirMass(float mu)
{
    mu = max(mu, 0.0);
    return 1.0 / (mu + 0.025 * exp(-11.0 * mu));
}

// Optical depth of clear air straight up, for red / green / blue (Rayleigh, ∝ λ⁻⁴ at ~612 / 549 / 465 nm).
static const float3 SkyRayleighDepth = float3(0.065, 0.10, 0.196);

// The preset's haze (MieScattering) as an optical depth, grey: Clear Day 0.02 → 0.03, Desert Haze 0.09 → 0.135.
float SkyMieDepth()
{
    return MieScattering * 1.5;
}

// Air mass of the haze. It lies low (a thin layer against ~8 km for the air), so its slant path grows much
// more in the last few degrees above the horizon (~330 at the horizon, like the air's above ~5°). That is
// why the horizon of a sunset is warm: there the haze, which scatters every colour alike and so shows the
// reddened sunlight as it is, outweighs the air, which would scatter the orange back toward yellow. Thin
// straight up, it leaves the noon zenith blue (a thicker haze turned it grey).
float HazeAirMass(float mu)
{
    mu = max(mu, 0.0);
    return 1.0 / (mu + 0.003 * exp(-30.0 * mu));
}

// Ozone absorbs (it does not scatter) around 600 nm, between our red and green, blue hardly at all. It
// sits high (~25 km), so the sun's light crosses it on its way to every point of the sky, along a slant that
// is long when the sun is low. That keeps the upper sky of a sunset blue and its glow orange rather than
// yellow; without it the fit against the reference stayed grey-green up high and lemon on the horizon.
static const float3 SkyOzoneDepth = float3(0.075, 0.065, 0.006);

float OzoneAirMass(float mu)
{
    mu = max(mu, 0.0);
    return rsqrt(mu * mu + 0.0079);   // a thin shell 25 km up: ~11 at the horizon
}

// Optical depth of the sun's slant path toward elevation sin = mu: air plus haze plus ozone, per channel
float3 SkySlantDepth(float mu)
{
    return SkyRayleighDepth * AirMass(mu) + SkyMieDepth() * HazeAirMass(mu) + SkyOzoneDepth * OzoneAirMass(mu);
}

// Colour of direct sunlight after its path through the air, 1 with the sun at the zenith: about white above
// 45°, golden around 10°, orange at the horizon.
float3 SunlightThroughAir(float sunY)
{
    return exp(-(SkySlantDepth(sunY) - SkySlantDepth(1.0)));
}

// Sky light in a direction, before the preset's grade: single scattering in a flat exponential atmosphere, in
// closed form, with the slant paths from AirMass so both stay finite at the horizon. Per channel, a point at
// air fraction u lights the eye with  tau·exp(-tauSun·u/muS) · exp(-tauView·(1-u)/muV);  integrated over u:
//     xv · (exp(-xs) - exp(-xv)) / (xv - xs),   xs = tauSun / muS,  xv = tauView / muV.
// That is what makes a sunset: high up all channels scatter little and blue most (blue); lower down blue has
// been dimmed along the long path while green still grows (green-grey), then green goes too and the reddened
// sunlight is left (yellow, then orange at the horizon). Blending blue into orange instead skips the green and
// turns purple. The view path uses a thinner depth than the sun's path (SkyViewDepthScale), which keeps the
// warm band narrow on the horizon while the low sun's light still turns orange.
static const float SkyViewDepthScale = 0.75;

float3 SkySingleScatter(float viewY, float sunY, float cosTheta)
{
    float3 xs = SkySlantDepth(sunY);
    float3 viewR = SkyRayleighDepth * AirMass(viewY);
    float viewM = SkyMieDepth() * HazeAirMass(viewY);
    float3 xv = (viewR + viewM) * SkyViewDepthScale;
    float3 diff = xv - xs;
    diff = sign(diff) * max(abs(diff), 1e-4) + (diff == 0.0) * 1e-4;
    float3 path = xv * (exp(-xs) - exp(-xv)) / diff;

    // How much of the scattered light goes toward the eye, by the share of each along this view: the air
    // (broad) and the haze (forward: the glow round the sun)
    float3 phase = (viewR * RayleighPhase(cosTheta) + viewM * HGPhase(cosTheta, MieAnisotropy)) / (viewR + viewM);
    return path * phase;
}

float3 GetSkyColor(float3 viewDir, float3 sunDir)
{
    float cosTheta = dot(viewDir, sunDir);
    float viewY = viewDir.y;
    float horizon = abs(viewY);

    // ── Sun elevation blend factors (same structure as before for lighting sync) ──
    float dayFactor, sunsetFactor, nightFactor;
    GetDayNightFactors(sunDir.y, dayFactor, sunsetFactor, nightFactor);

    float3 sunLight = SunlightThroughAir(sunDir.y);

    // ── Day sky: single scattering (blue → green-grey → yellow → orange toward the horizon at sunset) ──
    // Fitted against a sunset reference (vertical colour profiles, 2026-10-17) and the old noon sky.
    float3 luma = float3(0.2126, 0.7152, 0.0722);
    float3 single = SkySingleScatter(viewY, sunDir.y, cosTheta);
    float3 zenith = SkySingleScatter(1.0, sunDir.y, sunDir.y);
    float3 noonZenith = SkySingleScatter(1.0, 0.84, 0.84);   // sun 57° up
    float lumSingle = max(dot(single, luma), 1e-8);
    float lumZenith = max(dot(zenith, luma), 1e-8);
    // Single scattering alone makes the horizon ~14× the zenith (real skies: 2–4×, multiple scattering lights
    // the zenith too). Keep each direction's hue, compress its brightness relative to the zenith at this sun,
    // and add a fill in the zenith's colour for the light scattered more than once.
    float3 dayColor = single * pow(lumSingle / lumZenith, 0.55 - 1.0) + zenith * 0.6;
    // Partly undo the sky's darkening with a low sun, as an eye (or a camera) adapts: physically ~1/5 of noon
    // at sunset, which with the fixed exposure made the sunset scene dim.
    dayColor *= pow(max(dot(noonZenith, luma), 1e-8) / lumZenith, 0.3) * 16.0;

    // Glow round a low sun: the haze along the long low path scatters its light forward in a lobe much
    // tighter than single scattering's phase gives (the reference's bright cream within ~15° of the sun).
    // Lit by the transmitted sunlight. Only with the sun under ~20°: at noon it greyed the zenith. Fitted with
    // the rest against the sunset reference (whitening it a little fitted the numbers but hid the sun's core).
    float lowSun = 1.0 - saturate(sunDir.y / 0.35);
    dayColor += HGPhase(cosTheta, 0.9) * MieScattering * 57.0 * lowSun * lowSun * sunLight;

    // Sun below the horizon: the air above still catches its light for a while (twilight), then the night
    // colours take over (nightFactor below).
    dayColor *= saturate(1.0 + sunDir.y / 0.12);

    // Preset grade. A saturated SkyTintColor (Clear Day) keeps the full colours, a grey one (Overcast, Storm)
    // greys the sky out and gives it that tint's cast; AtmosphereDensity is the brightness. Saturation runs from
    // vivid with a high sun (the old noon sky's blue) to muted with a low one (the reference sunset).
    float tintMax = max(SkyTintColor.r, max(SkyTintColor.g, SkyTintColor.b));
    float tintMin = min(SkyTintColor.r, min(SkyTintColor.g, SkyTintColor.b));
    float tintSat = (tintMax - tintMin) / max(tintMax, 1e-3);
    float saturation = lerp(0.2, lerp(2.3, 0.85, lowSun), tintSat);
    float dayLum = dot(dayColor, luma);
    float3 cast = lerp(SkyTintColor / max(dot(SkyTintColor, luma), 1e-3), 1.0, tintSat);
    dayColor = max(dayLum + (dayColor - dayLum) * saturation, 0.0) * cast * AtmosphereDensity;

    // ── Horizon haze: the preset's coloured haze at low angles, lit by the sun's light ──
    // An artist extra for hazy presets (Desert Haze, Overcast); at a third of its old weight, since the
    // scattering already brightens the horizon. Clear Day's 0.3 at full weight made its sunset horizon yellow.
    float horizonMask = 1.0 - smoothstep(0.0, HazeHeight, horizon);
    horizonMask *= horizonMask; // softer falloff
    dayColor += HazeColor * horizonMask * HazeIntensity * 0.33 * sunLight * saturate(1.0 + sunDir.y / 0.12);

    // ── Sunset/sunrise tint: an artist flavour on top, low on the horizon toward the sun ──
    // The scattering above already makes the sunset; this is only a hint (at full strength it turned the
    // sky pink, at a quarter the fit against the reference still got worse).
    float sunProximity = pow(saturate(cosTheta * 0.5 + 0.5), 2.0);
    float horizonWarm = pow(1.0 - saturate(abs(viewY) / 0.15), 2.0);
    float3 sunsetTint = SunsetTintColor * SunsetTintIntensity * sunProximity * horizonWarm * 0.08;

    // ── Night sky ──
    float nightAltitude = saturate(abs(viewY));
    float3 nightColor = lerp(NightHorizonColor, NightSkyColor, pow(nightAltitude, 0.5));

    // ── Blend day/sunset/night using the same factors as SkyboxRenderer.UpdateSunLight() ──
    float3 skyColor = dayColor * dayFactor
                    + (dayColor + sunsetTint) * sunsetFactor
                    + nightColor * nightFactor;

    return max(skyColor, 0.0);
}

// Hemisphere ambient for a surface normal, as a multiple of the surface's albedo: the sky above (its zenith colour,
// which already darkens at night: never scale it by a day/night factor again), the ground's bounce light below.
// One model for every lit pass (composition, glass); water takes the sky half.
float3 SkyAmbientLight()
{
    return GetSkyColor(float3(0, 1, 0), FogSunDirection) * 0.45;
}

float3 HemisphereAmbient(float3 N)
{
    return lerp(GroundBounce, SkyAmbientLight(), N.y * 0.5 + 0.5);
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
