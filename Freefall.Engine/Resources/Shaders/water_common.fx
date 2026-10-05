// ────────────────────────────────────────────────
// Shading shared by every water surface: ocean.fx (FFT sea) and water.fx (lakes, rivers).
// Keeping it in one place is what makes all water agree under day / night / sky / clouds —
// do not copy these into a shader, call them.
//
// Requires: common.fx (SceneConstants) and sky_common.fx included first.
// ────────────────────────────────────────────────

// Fresnel (Schlick, precomputed for F0 = 0.02, roughness = 0.075)
float WaterFresnel(float NdotV)
{
    return saturate(0.02 + 0.668 * pow(1.0 - NdotV, 4.08));
}

// 0 near the camera, 1 from ~330 m: how far the surface detail has collapsed below a pixel.
// Drives the reflection flattening and the specular widening.
float WaterDistanceSmooth(float dist)
{
    return saturate(dist * 0.003);
}

// Sky (and cloud layer) mirrored in the water.
// Far away the per-pixel normal is sub-pixel noise: flatten it so the reflection converges on the sky
// just above the horizon in the mirrored view direction. Uses the same GetSkyColor as the dome and
// the fog, with the TRUE sun direction (FogSunDirection), so it follows time of day and the
// environment preset. Never precompute this on the CPU or feed it the directional light's
// direction: that one is clamped above the horizon at night (the ocean glowed silver from it).
//   relPos:     surface position relative to the camera
//   cloudColor: SkyboxRenderer.CurrentCloudColor
float3 WaterSkyReflection(SamplerState wrapSampler, float3 V, float3 N, float3 relPos, float dist, float3 cloudColor)
{
    float reflSmooth = WaterDistanceSmooth(dist);
    float3 reflN = normalize(lerp(N, float3(0, 1, 0), reflSmooth * 0.9));
    float3 reflectDir = reflect(-V, reflN);
    reflectDir.y = max(abs(reflectDir.y), 0.02);
    reflectDir = normalize(reflectDir);
    float3 skyRefl = GetSkyColor(reflectDir, FogSunDirection);

    // Cloud layer: same plane / UVs / density as the sky dome, flat-shaded with the CPU-blended
    // day / sunset / night cloud color.
    if (CloudNoiseLUTIdx != 0)
    {
        Texture3D<float4> cloudLUT = ResourceDescriptorHeap[CloudNoiseLUTIdx];
        float2 cloudUV = CloudLayerUV(relPos.xz + reflectDir.xz * (CloudAltitude / reflectDir.y));
        float cloudMip = saturate(1.0 - reflectDir.y * 5.0) * 3.0;
        float cloudDensity = smoothstep(0.0, 0.45, CloudBaseDensity(cloudLUT, wrapSampler, cloudUV, cloudMip));
        cloudDensity *= smoothstep(0.0, 0.22, reflectDir.y);
        skyRefl = lerp(skyRefl, cloudColor, 1.0 - exp(-cloudDensity * 4.5));
    }

    // Near water keeps the darker art-directed reflection; toward the horizon it approaches the full
    // sky so distant water meets the sky without a hard dark line (most visible at night).
    return skyRefl * lerp(0.5, 0.85, reflSmooth);
}

// GGX sun glitter (widened with distance to keep it from sparkling). F = WaterFresnel(NdotV).
float3 WaterSunSpecular(float3 N, float3 V, float3 L, float3 sunRadiance, float F, float dist)
{
    float NdotL = saturate(dot(N, L));
    float rough = lerp(0.075, 0.6, WaterDistanceSmooth(dist));
    float3 halfDir = normalize(L + V);
    float NdotH = max(0.0001, dot(N, halfDir));
    float a = rough * rough;
    float a2 = a * a;
    float dGGX = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    float D = a2 / max(3.14159 * dGGX * dGGX, 1e-4);
    float3 specular = sunRadiance * F * D * NdotL;
    specular /= max(0.001, 4.0 * max(0.001, NdotL));
    return specular * NdotL;
}

// What is seen through the surface, from the depth buffer.
struct WaterColumn
{
    float3 body;        // color coming up through the surface (bed seen through the water, or 'scatter' when deep)
    float pathLen;      // distance the view ray travels under water before it hits the scene (1000 = nothing there)
    float bufDepth;     // vertical water depth at the point the ray hits (view dependent: prefer a heightmap where there is one)
    float edgeFade;     // 0 at the waterline, 1 from 30 cm of water: fades reflection and specular in
};

// The scene point behind a water pixel lies on the same view ray, so its distance follows from the
// ratio of the two linear depths.
//   pixelDepth: linear depth of the water pixel (clip w)
//   camHeight:  |camera y - surface y|
//   scatter:    color of deep water
//   visibility: meters of water at which the bed is tinted by shallowColor and mostly gone
//   bedFade:    0..1 extra fade of the bed (the ocean uses it where the terrain ends)
WaterColumn GetWaterColumn(SamplerState s, uint depthSRV, uint compositeSRV, float2 screenUV, float pixelDepth,
                           float dist, float camHeight, float3 N, float refraction,
                           float3 scatter, float3 shallowColor, float visibility, float bedFade)
{
    WaterColumn col;
    col.body = scatter;
    col.pathLen = 1000.0;
    col.bufDepth = 1000.0;
    col.edgeFade = 1.0;

    if (depthSRV == 0)
        return col;

    Texture2D<float> depthGB = ResourceDescriptorHeap[depthSRV];
    float sceneZ = depthGB.SampleLevel(s, screenUV, 0);

    if (sceneZ > 0)
    {
        float zRatio = max(sceneZ / max(pixelDepth, 0.001), 1.0) - 1.0;
        col.pathLen = dist * zRatio;
        col.bufDepth = camHeight * zRatio;
    }

    // Soft waterline
    col.edgeFade = saturate(col.pathLen / 0.3);

    if (compositeSRV != 0)
    {
        Texture2D<float4> compositeBuffer = ResourceDescriptorHeap[compositeSRV];

        // Refraction, rejected when the offset sample lands on something in front of the water
        float2 refrUV = saturate(screenUV + N.xz * refraction * saturate(col.pathLen * 0.5));
        float refrZ = depthGB.SampleLevel(s, refrUV, 0);
        if (refrZ > 0 && refrZ < pixelDepth)
            refrUV = screenUV;
        float3 sceneColor = compositeBuffer.SampleLevel(s, refrUV, 0).rgb;

        // Beer-Lambert
        float x = col.pathLen / max(0.01, visibility);
        float3 transmittance = pow(max(shallowColor, 0.02), x) * exp(-1.5 * x);
        col.body = lerp(scatter, sceneColor, transmittance * bedFade);
    }

    return col;
}

// Foam lighting shared by all water (sun + sky ambient; skyAmbient = GetSkyColor(up) * 0.45 * AmbientScale)
float3 WaterFoamLit(float3 foamAlbedo, float NdotL, float3 sunRadiance, float3 skyAmbient)
{
    return foamAlbedo * ((0.27 + 0.5 * NdotL) * sunRadiance + skyAmbient * 1.15);
}
