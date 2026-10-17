// ────────────────────────────────────────────────
// Shading shared by every water surface: ocean.fx (FFT sea) and water.fx (lakes, rivers).
// Keeping it in one place is what makes all water agree under day / night / sky / clouds —
// do not copy these into a shader, call them.
//
// Requires: common.fx (SceneConstants) and sky_common.fx included first.
// ────────────────────────────────────────────────

// ── Sun shadow (cascaded shadow map) ──
// The directional light broadcasts these to every effect each frame (DirectionalLight.BroadcastLightParams),
// and the forward pass puts the shadow map's SRV into push-constant slot 0 of every forward batch
// (CommandBuffer.Pass.Execute). A water shader therefore only has to name slot 0 'ShadowMapIdx' in its
// PushConstants and pass it in. Same layout as gbuffer_transparent.fx.
cbuffer ObjectConstants : register(b1)
{
    float3 LightColor;
    float LightIntensity;
    float3 LightDirection;
    float _lightPad0;

    row_major float4x4 LightSpaces[8];   // camera-relative world → light clip space, per cascade
    float4 Cascades[8];                  // y = far view depth of the cascade

    int CascadeCount;
    int _lightDebugMode;
    float2 _lightPad1;
};

SamplerComparisonState WaterShadowSampler : register(s3);

// Fraction of direct sunlight (0..1) reaching a point on the water: terrain, buildings, trees, piers.
// Apply it to the sun's radiance only — sky reflection and ambient are not shadowed by the sun.
//   relPos: surface position relative to the camera,  pixel: SV_Position.xy
float WaterSunShadow(uint shadowMapIdx, float3 relPos, float2 pixel)
{
    if (shadowMapIdx == 0 || CascadeCount <= 0)
        return 1.0;

    float viewDepth = dot(relPos, float3(View._13, View._23, View._33));
    if (viewDepth > Cascades[CascadeCount - 1].y)
        return 1.0;

    int cascadeIndex = CascadeCount - 1;
    for (int ci = 0; ci < CascadeCount; ci++)
    {
        if (viewDepth < Cascades[ci].y)
        {
            cascadeIndex = ci;
            break;
        }
    }

    float4 lsPos = mul(float4(relPos, 1.0), LightSpaces[cascadeIndex]);
    lsPos /= lsPos.w;

    float2 shadowUV = lsPos.xy * 0.5 + 0.5;
    shadowUV.y = 1.0 - shadowUV.y;
    if (any(shadowUV < 0.0) || any(shadowUV > 1.0))
        return 1.0;

    // The flat up vector, not the rippled normal: the bias must not flicker with the waves
    Texture2DArray shadowMap = ResourceDescriptorHeap[shadowMapIdx];
    float zScale = abs(LightSpaces[cascadeIndex]._33);
    return GetShadowFactor(shadowMap, WaterShadowSampler, shadowUV, lsPos.z,
        cascadeIndex, float3(0, 1, 0), LightDirection, zScale, pixel);
}

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
// just above the horizon in the mirrored view direction. The sky is the atmosphere's (SkyRadiance, the same
// as the dome and the fog) and the low-sun mix below uses the true sun (AtmSunDirection), not the directional
// light, which is the moon at night.
//   relPos:     surface position relative to the camera
float3 WaterSkyReflection(SamplerState wrapSampler, float3 V, float3 N, float3 relPos, float dist)
{
    float reflSmooth = WaterDistanceSmooth(dist);
    float3 reflN = normalize(lerp(N, float3(0, 1, 0), reflSmooth * 0.9));
    float3 reflectDir = reflect(-V, reflN);
    // How much of the cloud layer shows: the angle the surface really mirrors, going over to the flat mirror's
    // (the view's own elevation) with distance, where what is left of the wave normal is sub-pixel noise.
    float mirroredY = lerp(abs(reflectDir.y), abs(V.y), reflSmooth);
    // Seen at a grazing angle, the wave facets turned toward the eye are the ones that show, and they mirror
    // sky well above the horizon (~11° from ~330 m on; the reference's distant sea matched its sky at 10–14°):
    // distant sea is bluer than the band of sky right above it. With the flattened normal alone it mirrored
    // that band exactly (warm tan at sunset: a brown sea).
    // Reached by 200 m: in between, pixels mirroring the fast gradient just above the horizon flipped between
    // tan and blue from one ripple to the next (a grainy, speckled sea).
    float3 lowDir = normalize(float3(reflectDir.x, max(abs(reflectDir.y), 0.02), reflectDir.z));
    reflectDir.y = max(abs(reflectDir.y), lerp(0.03, 0.2, saturate(dist / 200.0)));
    reflectDir = normalize(reflectDir);
    float3 skyRefl = SkyRadiance(reflectDir);

    // The facets mirror a spread of angles. Toward a low sun the low ones see the bright glow round it and
    // outshine the rest: the broad golden sheen on the sea under a setting sun. So mix in the low mirror
    // direction, by how much the reflection points toward the sun.
    float3 sunDir = AtmSunDirection();
    float towardSun = saturate(dot(normalize(reflectDir.xz + 1e-5), normalize(sunDir.xz + 1e-5)));
    float lowWeight = lerp(0.15, 0.75, towardSun * towardSun) * saturate(1.0 - sunDir.y * 2.0);
    skyRefl = lerp(skyRefl, SkyRadiance(lowDir), lowWeight);

    // Cloud layer: same plane / UVs / density as the sky dome, flat-shaded with the light on it
    // (sky_common.fx GetCloudLight: the sun and moon through the air, the sky's ambient, the air in front)
    if (CloudNoiseLUTIdx != 0)
    {
        Texture3D<float4> cloudLUT = ResourceDescriptorHeap[CloudNoiseLUTIdx];
        float2 cloudUV = CloudLayerUV(relPos.xz + reflectDir.xz * (CloudAltitude / reflectDir.y));
        float cloudMip = saturate(1.0 - reflectDir.y * 5.0) * 3.0;
        float cloudDensity = smoothstep(0.0, 0.45, CloudBaseDensity(cloudLUT, wrapSampler, cloudUV, cloudMip));
        // Faded by the angle the surface really mirrors: with the floor's angle, every distant pixel picked up
        // the cloud layer 11° up and the far sea broke into big pale blotches.
        cloudDensity *= smoothstep(0.0, 0.22, mirroredY);
        if (cloudDensity > 0.001)
            skyRefl = lerp(skyRefl, CloudFlatRadiance(GetCloudLight(reflectDir)), 1.0 - exp(-cloudDensity * 4.5));
    }

    // The Fresnel term outside already sets how much of the sky shows. Near water used to get only half of it
    // (an art-directed darkening from the white sky of before 2026-10-17); with the scattering sky that left
    // the sunset sea far darker than the reference.
    return skyRefl;
}

// Light from the whole sky falling on a level surface. skyAmbient = SkyAmbientLight() = E / π (pre-exposed), so
// this is E. At sunset this, not the low sun, is what lights water from inside: with the sun term alone the
// shallow shelf and the deep colour went black. (Before the physical sky: zenith × 0.45 × 7, a guess at the same.)
float3 WaterSkyIrradiance(float3 skyAmbient)
{
    return skyAmbient * 3.14159;
}

// Screen-space reflections of the scene were tried here (2026-10-05) and removed: marching the depth
// buffer gives speckle on foliage (pixel-sized gaps make neighbouring rays hit or miss at random),
// barcode-like stripes from trunks against bright sky, and halos once blurred enough to hide both.
// Scene reflections are to come from ray tracing instead. See the ocean knowledge doc.
// GGX sun glitter (widened with distance to keep it from sparkling). F = WaterFresnel(NdotV).
// Cook-Torrance: F·D·G / (4·N·L·N·V) × N·L, with the height-correlated Smith term as G / (4·N·L·N·V).
// Until 2026-10-17 this returned F·D·N·L / 4: a factor N·L·N·V too small, harmless at noon but ~0.01 with a
// low sun and a grazing view, so a sunset had no glitter path at all.
// One GGX lobe (D × visibility, without F) for a perceptual roughness
float WaterGGXLobe(float NdotH, float NdotL, float NdotV, float rough)
{
    float a = rough * rough;
    float a2 = a * a;
    float dGGX = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    float D = a2 / max(3.14159 * dGGX * dGGX, 1e-4);
    float vis = 0.5 / max(NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2)
                        + NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2), 1e-5);
    return D * vis;
}

float3 WaterSunSpecular(float3 N, float3 V, float3 L, float3 sunRadiance, float F, float dist)
{
    float NdotL = saturate(dot(N, L));
    float NdotV = max(dot(N, V), 1e-3);
    float NdotH = max(0.0001, dot(N, normalize(L + V)));
    // Ripples far below a pixel spread the sun's reflection: 0.075 near the camera kept the sunset path a
    // thin line of sparkles where the reference shows a broad golden band.
    float sharp = WaterGGXLobe(NdotH, NdotL, NdotV, lerp(0.2, 0.55, saturate(dist * 0.004)));
    // Glitter is two things: sparkles off the resolved waves and a broad, dimmer sheen off the countless
    // capillary ripples the FFT does not resolve (their slopes reach ~±15°, Cox-Munk). The sheen lights the
    // whole sun side of the sea golden; with the sharp lobe alone that side stayed dark and the path's peaks
    // blew out to white.
    float broad = WaterGGXLobe(NdotH, NdotL, NdotV, lerp(0.5, 0.7, saturate(dist * 0.002)));
    return sunRadiance * F * (sharp * 0.75 + broad * 0.25) * NdotL;
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

// Foam lighting shared by all water (sun + sky ambient; skyAmbient = SkyAmbientLight())
float3 WaterFoamLit(float3 foamAlbedo, float NdotL, float3 sunRadiance, float3 skyAmbient)
{
    return foamAlbedo * ((0.27 + 0.5 * NdotL) * sunRadiance + skyAmbient * 1.15);
}
