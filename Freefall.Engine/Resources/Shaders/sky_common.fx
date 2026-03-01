// ────────────────────────────────────────────────
// Shared procedural sky color functions
// Used by both mesh_skybox.fx and ocean.fx
// ────────────────────────────────────────────────

float3 GetSkyColor(float3 viewDir, float3 sunDir)
{
    float horizon = abs(viewDir.y);

    float3 dayZenith = float3(0.2, 0.5, 1.0);
    float3 dayHorizon = float3(0.6, 0.8, 1.0);
    float3 sunsetZenith = float3(0.4, 0.3, 0.6);
    float3 sunsetHorizon = float3(1.0, 0.5, 0.3);
    float3 nightZenith = float3(0.01, 0.01, 0.05);
    float3 nightHorizon = float3(0.05, 0.05, 0.1);

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

    float3 zenithColor = dayZenith * dayFactor + sunsetZenith * sunsetFactor + nightZenith * nightFactor;
    float3 horizonColor = dayHorizon * dayFactor + sunsetHorizon * sunsetFactor + nightHorizon * nightFactor;

    float3 skyColor = lerp(horizonColor, zenithColor, pow(horizon, 0.5));

    float sunAngle = dot(viewDir, sunDir);
    float horizonScatter = pow(1.0 - horizon, 3.0);
    float sunScatter = pow(saturate(sunAngle), 3.0);
    float scatter = (horizonScatter + sunScatter * 0.5) * saturate(dayFactor + sunsetFactor * 0.5);
    skyColor += float3(1.0, 0.8, 0.6) * scatter * 0.3;

    return skyColor;
}
