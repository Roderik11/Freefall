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

// ── Roughness from slope moments ──
// ocean_fft.hlsl writes per band (sx, sz, sx², sz²) with a mip chain that averages all four, so a filtered sample
// gives the mean slope of the waves it covers and their variance. The mean makes the normal; the variance is the
// roughness of the surface as this pixel sees it: small near the camera (the waves are resolved, the normal carries
// them), large far away (whole wave fields under one pixel). Nothing here depends on the distance. Bands are
// independent, so their variances add; the variance is σx² + σz², which does not change with a band's rotation.
struct WaterSlopes
{
    float2 mean;        // world xz slope
    float variance;     // σx² + σz² of the slopes under the sample
};

//   moments:   a band's sample; meanWorld: its mean slope (moments.xy) turned into world xz by the caller
//   scale:     what the caller multiplies that band's slopes by
void AddWaterBand(inout WaterSlopes s, float4 moments, float2 meanWorld, float scale)
{
    s.mean += meanWorld * scale;
    s.variance += max(moments.z + moments.w - dot(moments.xy, moments.xy), 0.0) * scale * scale;
}

// A band's slope variance over its whole tile (the last mip: one texel averages the tile)
float WaterBandTotalVariance(Texture2DArray<float4> slopeTex, SamplerState s, float slice, float scale)
{
    float w, h, elements, levels;
    slopeTex.GetDimensions(0, w, h, elements, levels);
    float4 m = slopeTex.SampleLevel(s, float3(0.5, 0.5, slice), levels - 1.0);
    return max(m.z + m.w - dot(m.xy, m.xy), 0.0) * scale * scale;
}

// Mean square slope of a wind-roughened clean sea, both axes (Cox & Munk 1954): 0.003 + 5.12e-3 × wind (m/s)
float CoxMunkVariance(float windSpeed)
{
    return 0.003 + 0.00512 * max(windSpeed, 0.0);
}

// The roughness the FFT does not carry: the slopes below the finest band's resolution (capillary waves, millimetres
// to centimetres). Cox-Munk's total at the current wind (SceneConstants.WindStrength: the preset's) minus what the
// bands resolve over their whole tiles, never below Cox-Munk's calm-water floor.
float WaterUnresolvedVariance(float resolvedTotal)
{
    return max(CoxMunkVariance(WindStrength) - resolvedTotal, 0.003);
}

// GGX roughness of the surface: α² = σx² + σz² (Beckmann m² = 2σ² per axis, GGX α ≈ m). Isotropic: each band's
// slopes are rotated (BandAngles), which loses the cross term an anisotropic α would need; the sea's up-wind to
// cross-wind ratio is only ~1.6 (Cox-Munk).
float WaterAlpha(WaterSlopes s, float unresolved)
{
    return sqrt(max(s.variance + unresolved, 1e-4));
}

static const float WaterF0 = 0.02;     // water, n = 1.33

// How much of the light round the mirror direction a rough water surface sends to the eye (Fresnel and the waves'
// masking together): Karis' fit of the split-sum environment BRDF, with the mean normal. What is not reflected
// enters the water (1 − this lights the body).
float WaterSpecularReflectance(float NdotV, float alpha)
{
    float roughness = sqrt(alpha);     // perceptual
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * saturate(NdotV))) * r.x + r.y;
    float2 AB = float2(-1.04, 1.04) * a004 + r.zw;
    return saturate(WaterF0 * AB.x + AB.y);
}

// The sky in one direction of a reflection lobe. A direction under the horizon meets the next wave, which mirrors the
// sky as far above the horizon, as much of it as its Fresnel reflects at that grazing angle (the rest is the dark water
// body). Clamping such directions to the horizon instead painted every wave back with the horizon's glow: an orange
// band across the sea at sunset where the backs face the camera. The blur never reaches further under the horizon than
// the direction stands above it.
float3 WaterSkyTap(float3 dir, float width)
{
    float below = max(-dir.y, 0.0);
    float nextWave = WaterF0 + (1.0 - WaterF0) * pow(1.0 - below, 5.0);   // 1 at the horizon
    dir.y = abs(dir.y);
    return SkyReflectionCone(dir, width, 2.0 * asin(dir.y) + 0.05) * nextWave;
}

// The sky (and the cloud layer) a rough water surface mirrors, from the sky probe (sky_common.fx SkyReflectionCone).
// The lobe is centred where it peaks: from the mirror direction of the mean normal toward the normal, the more the
// rougher (Frostbite's dominant direction), which is why a rough sea far away mirrors sky well above the horizon. A GGX
// lobe spreads a mirrored direction over ~2.6 α radians (full width at half maximum) up and down, but sideways only
// N·V of that: a wave tilted sideways turns a grazing reflection little (why a sun path is a tall narrow streak). One
// probe lookup blurred as wide as the tall side mixed in sky far to the side of the sun and greyed the sunset sea, so
// five lookups lie along the tall side, each blurred as wide as the narrow side (at least their spacing).
// Until stage 6 (2026-10-18) one sharp sky lookup and a ray of its own into the cloud layer, flat-shaded.
float3 WaterSkyReflection(float3 V, float3 N, float alpha)
{
    float3 R = reflect(-V, N);
    float smoothness = saturate(1.0 - alpha);
    float3 dir = normalize(lerp(N, R, smoothness * (sqrt(smoothness) + alpha)));

    float tall = 2.6 * alpha;
    float spacing = tall * 0.2;
    float width = max(tall * saturate(dot(N, V)), spacing);
    float3 up = float3(0, 1, 0) - dir * dir.y;                     // toward the zenith, across dir
    up = dot(up, up) > 1e-6 ? normalize(up) : float3(1, 0, 0);     // looking straight down: any way is up

    float3 sky = WaterSkyTap(dir, width) * 0.4;
    [unroll] for (int k = 1; k <= 2; k++)
    {
        float s, c;
        sincos(spacing * k, s, c);
        float w = k == 1 ? 0.2 : 0.1;
        sky += (WaterSkyTap(dir * c + up * s, width) + WaterSkyTap(dir * c - up * s, width)) * w;
    }
    return sky;
}

// Light from the whole sky falling on a level surface. skyAmbient = SkyAmbient(up) = E / π (pre-exposed), so
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

// The sun's glint off the waves a pixel covers: Cook-Torrance GGX with the height-correlated Smith visibility,
// F·D·Vis·N·L. α is the slope variance under the pixel (WaterAlpha), so the one lobe gives the sparkles near the
// camera and the broad path far away. Replaced 2026-10-17: two hand-tuned lobes whose roughness grew with distance
// (tuned while the water's sun lacked its π, so with it the far sea at noon went milky white).
//   sunIrradiance: E of the directional light (× π, pre-exposed, shadowed)
float3 WaterSunSpecular(float3 N, float3 V, float3 L, float3 sunIrradiance, float alpha)
{
    float NdotL = saturate(dot(N, L));
    if (NdotL <= 0.0)
        return 0;
    float NdotV = max(dot(N, V), 1e-3);
    float3 H = normalize(L + V);
    float NdotH = saturate(dot(N, H));
    float a2 = max(alpha * alpha, 1e-6);
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (3.14159 * d * d);
    float vis = 0.5 / max(NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2)
                        + NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2), 1e-5);
    float F = WaterF0 + (1.0 - WaterF0) * pow(1.0 - saturate(dot(V, H)), 5.0);
    return sunIrradiance * (F * D * vis * NdotL);
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

// Foam lighting shared by all water (sun + sky ambient; skyAmbient = SkyAmbient(N), the sky's light on the foam's face)
float3 WaterFoamLit(float3 foamAlbedo, float NdotL, float3 sunRadiance, float3 skyAmbient)
{
    return foamAlbedo * ((0.27 + 0.5 * NdotL) * sunRadiance + skyAmbient * 1.15);
}
