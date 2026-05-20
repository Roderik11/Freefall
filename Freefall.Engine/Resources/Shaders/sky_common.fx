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

float3 GetSkyColor(float3 viewDir, float3 sunDir)
{
    float cosTheta = dot(viewDir, sunDir);
    float viewY = viewDir.y;
    float horizon = abs(viewY);

    // ── Sun elevation blend factors (same structure as before for lighting sync) ──
    float sunElevation = sunDir.y;

    float dayFactor = saturate((sunElevation - 0.0) / 0.8);
    dayFactor = pow(dayFactor, 0.7);

    float sunsetFactor = 0.0;
    if (sunElevation < 0.15 && sunElevation > -0.2)
    {
        sunsetFactor = 1.0 - abs((sunElevation - (-0.025)) / 0.175);
        sunsetFactor = max(0.0, sunsetFactor);
    }

    float nightFactor = saturate((-sunElevation - 0.15) / 0.3);

    float total = dayFactor + sunsetFactor + nightFactor;
    if (total > 0.0)
    {
        dayFactor /= total;
        sunsetFactor /= total;
        nightFactor /= total;
    }

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
