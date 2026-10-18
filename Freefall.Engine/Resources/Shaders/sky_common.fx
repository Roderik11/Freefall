// ────────────────────────────────────────────────
// The sky as every lit pass sees it: its light in a direction, the ambient it gives, the ground's bounce light, the
// aerial perspective (fog) and the light on the cloud layer. All of it comes from the atmosphere
// (atmosphere_common.hlsli, made each frame by Graphics/Atmosphere.cs): there is no second sky model to keep in step.
// Light here is as the HDR targets store it: physical (cd/m²) × Exposure.
// Used by mesh_skybox.fx, composition_cs.hlsl, light_directional_cs.hlsl, ocean.fx, water.fx, gbuffer_transparent.fx.
//
// Requires: SceneConstants (b0) from common.fx.
// ────────────────────────────────────────────────
#ifndef SKY_COMMON
#define SKY_COMMON

#include "atmosphere_common.hlsli"
#include "sky_visibility.hlsli"
#include "sky_capture.hlsli"

// Direction toward the sun (the true sun, also below the horizon), for passes that need its place in the sky
float3 AtmSunDirection()
{
    if (AtmosphereIdx == 0)
        return float3(0, 1, 0);
    return LoadAtmosphere().SunDirection;
}

// The sky's light from a direction: sun, moon and airglow scattered by the air, no discs, stars or clouds
float3 SkyRadiance(float3 dir)
{
    if (AtmosphereIdx == 0)
        return 0;
    return AtmosphereSkyLuminance(LoadAtmosphere(), dir) * Exposure;
}

// Ambient light as a multiple of a surface's albedo (E / π): from the whole sky on an up-facing surface, and from
// the ground (its albedo × the sun, moon and sky on it) on a down-facing one. Made once a frame by CSAmbient; it
// already darkens at night and goes blue at dusk, so it is never scaled by a day/night factor.
float3 SkyAmbientLight()
{
    if (AtmosphereIdx == 0)
        return 0;
    StructuredBuffer<float4> ambient = ResourceDescriptorHeap[LoadAtmosphere().Ambient];
    return ambient[0].rgb;
}

float3 GroundBounceLight()
{
    if (AtmosphereIdx == 0)
        return 0;
    StructuredBuffer<float4> ambient = ResourceDescriptorHeap[LoadAtmosphere().Ambient];
    return ambient[1].rgb;
}

// The sky's ambient light on a surface facing N, as a multiple of its albedo (E / π × Exposure): from the sky probe's
// SH9 (sky_probe_cs.hlsl), so it knows where in the sky the light is (the bright side at sunset, the blue zenith, the
// clouds, the lit ground below). Without the probe (Engine.Settings.SkyProbe off): sky above, ground below, lerped by
// N.y (stage 1-3). One model for every lit pass (composition, glass).
float3 SkyAmbient(float3 N)
{
    if (AtmosphereIdx == 0)
        return 0;
    AtmosphereParams A = LoadAtmosphere();
    if (A.SkyProbeSH == 0)
        return lerp(GroundBounceLight(), SkyAmbientLight(), N.y * 0.5 + 0.5);
    StructuredBuffer<float4> sh = ResourceDescriptorHeap[A.SkyProbeSH];
    float3 e = sh[0].rgb * 0.282095
             + sh[1].rgb * (0.488603 * N.y) + sh[2].rgb * (0.488603 * N.z) + sh[3].rgb * (0.488603 * N.x)
             + sh[4].rgb * (1.092548 * N.x * N.y) + sh[5].rgb * (1.092548 * N.y * N.z)
             + sh[6].rgb * (0.315392 * (3.0 * N.z * N.z - 1.0)) + sh[7].rgb * (1.092548 * N.x * N.z)
             + sh[8].rgb * (0.546274 * (N.x * N.x - N.y * N.y));
    return max(e, 0.0);
}

// The sky's light in direction R blurred for a surface of this roughness (0 sharp .. 1 the whole hemisphere), from the
// sky probe's mips; without the probe the sky above the horizon and the ground light below it
float3 SkyReflection(float3 R, float roughness)
{
    if (AtmosphereIdx == 0)
        return 0;
    AtmosphereParams A = LoadAtmosphere();
    if (A.SkyProbeCube == 0)
        return lerp(GroundBounceLight(), SkyRadiance(normalize(float3(R.x, max(R.y, 0.02), R.z))), smoothstep(-0.15, 0.1, R.y));
    TextureCube<float4> probe = ResourceDescriptorHeap[A.SkyProbeCube];
    return probe.SampleLevel(AtmosphereLinearClamp, R, saturate(roughness) * (A.SkyProbeMips - 1.0)).rgb;
}

// The terrain under the atmosphere's parameters, for sky_visibility.hlsli (Heightmap 0 = none, or SkyVisibility off)
TerrainCover LoadTerrainCover()
{
    TerrainCover t = (TerrainCover)0;
    if (AtmosphereIdx == 0)
        return t;
    AtmosphereParams A = LoadAtmosphere();
    t.Heightmap = A.TerrainHeightmap;
    t.MaxHeight = A.TerrainMaxHeight;
    t.Origin = A.TerrainOrigin;
    t.Size = float2(A.TerrainSizeX, A.TerrainSizeZ);
    return t;
}

// 1 in the open, 0 four metres or more under the terrain (a cave): no sky and no sun there (sky_visibility.hlsli)
float TerrainOpen(float3 worldPos, float3 N)
{
    return TerrainOpenAt(LoadTerrainCover(), AtmosphereLinearClamp, worldPos, N, TerrainTolerance(distance(worldPos, CamPos)));
}

// How much of the sky's light reaches a surface past the meshes around the camera (walls, roofs, tree crowns, the cave
// kit), 0..1 (sky_capture.hlsli; 1 without captures). worldPos is absolute.
float MeshSkyVisibility(float3 worldPos, float3 N)
{
    if (AtmosphereIdx == 0)
        return 1;
    AtmosphereParams A = LoadAtmosphere();
    return MeshSkyVisibilityAt(A.SkyCaptureViews, A.SkyCaptureDepth, A.SkyCaptureCount, AtmosphereLinearClamp, worldPos, N);
}

// How much of the sky's light reaches a surface past the terrain and the meshes, 0..1, given the meshes' part. In the
// open the meshes take their share of what the terrain's horizons leave, but never less than SkyCaptureBounce: a stand-in
// for daylight bouncing in through doors and windows, which the 16 directions mostly miss (a room without a lamp was
// black at noon, user 2026-10-18). Under the terrain both judge the same rock (the portal light through a cave mouth,
// the cave kit in the captures): the smaller wins, and a cave stays black.
float SkyVisibilityWith(float3 worldPos, float3 N, float meshes)
{
    float open, portal;
    float sky = SkyVisibilityParts(LoadTerrainCover(), AtmosphereLinearClamp, worldPos, N,
        TerrainTolerance(distance(worldPos, CamPos)), open, portal);
    float bounce = AtmosphereIdx != 0 ? LoadAtmosphere().SkyCaptureBounce : 0.0;
    return lerp(min(portal, meshes), sky * max(meshes, bounce), open);
}

// The same, the meshes' part looked up here. worldPos is absolute. The opaque surfaces have theirs from
// sky_visibility_cs.hlsl (the light buffer's alpha), cheaper per pixel.
float SkyVisibility(float3 worldPos, float3 N)
{
    return SkyVisibilityWith(worldPos, N, MeshSkyVisibility(worldPos, N));
}

// Light on a lit particle or ribbon (smoke, rain, snow, a blade's trail) at the camera's height, as a multiple of its
// albedo (E / π, pre-exposed): half the sun's and the moon's light through the air and the clouds' average shade (a
// puff is lit from one side) and the sky's and the ground's ambient averaged, as for the clouds. No shadows: grey by
// day, dark at night. Callers multiply by the SkyVisibility at the particle: smoke in a cave stays dark, since neither
// the sky nor the sun reaches it there (they also scale the haze in front of it by that).
float3 ParticleLight()
{
    if (AtmosphereIdx == 0)
        return 0;
    AtmosphereParams A = LoadAtmosphere();
    float3 P = float3(0, AtmosphereViewHeight(A, CamPos.y), 0);
    float3 direct = A.SunIlluminance * AtmTransmittanceToLight(A, P, A.SunDirection)
                  + A.MoonIlluminance * AtmTransmittanceToLight(A, P, A.MoonDirection);
    StructuredBuffer<float4> ambient = ResourceDescriptorHeap[A.Ambient];
    return direct * (A.CloudShade * 0.5 * Exposure / ATM_PI) + (ambient[0].rgb + ambient[1].rgb) * 0.5;
}

// Aerial perspective: the air between the camera and a surface dims the surface's light and adds its own (the haze
// and the ground fog of the preset, lit by the sun and the moon, in the earth's shadow after sunset).
//   pixel:     SV_Position.xy (or the pixel a compute thread shades)
//   viewDepth: linear view-space depth in metres (the G-buffer's depth, clip w)
// Sky pixels (depth 0) must not come here: the sky's light already holds all of the air.
float3 ApplyAerialPerspective(float3 color, float2 pixel, float viewDepth)
{
    if (FogEnabled <= 0.0 || AtmosphereIdx == 0)
        return color;
    float4 ap = AtmosphereAerialPerspective(LoadAtmosphere(), pixel / ViewportSize, viewDepth);
    return color * ap.a + ap.rgb;
}

// How much of the air's own light to keep in front of a surface that sees skyVisible of the sky. The volume lights every
// slice of air with the sun and sky as if nothing shaded it, so in a cave (a ray ending under the hill crossed dark air)
// it glowed over the black walls. Only the stretch of the ray inside the cave is dark: at most the last 30 m when the
// camera is in the open, all of it when the camera is in the cave too. Scaling the whole ray left a cave mouth seen from
// 400 m without the haze that covers everything round it: a black blob in rain (user, 2026-10-18). A stand-in until
// the volume itself is shadowed (ATMOSPHERE_PLAN, open issues).
float AerialSkyVisible(float skyVisible, float viewDepth)
{
    float cameraOpen = TerrainOpen(CamPos, float3(0, 1, 0));
    float darkStretch = lerp(viewDepth, min(viewDepth, 30.0), cameraOpen);
    return 1.0 - (1.0 - skyVisible) * darkStretch / max(viewDepth, 1e-3);
}

// The same with the air's own light scaled by AerialSkyVisible
float3 ApplyAerialPerspective(float3 color, float2 pixel, float viewDepth, float skyVisible)
{
    if (FogEnabled <= 0.0 || AtmosphereIdx == 0)
        return color;
    float4 ap = AtmosphereAerialPerspective(LoadAtmosphere(), pixel / ViewportSize, viewDepth);
    return color * ap.a + ap.rgb * AerialSkyVisible(skyVisible, viewDepth);
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

// ── Light on the cloud layer ──
// A cloud in direction dir (the flat layer CloudAltitude above the camera) is lit by the sun and the moon through the
// air above it (nothing once the planet hides them: clouds stay lit a little after sunset, then go grey) and by the
// sky and the ground. The air between the camera and the cloud dims it and lays its own light over it. Stage 1 of
// ATMOSPHERE_PLAN.md; stage 5 adds a phase function and the sky probe's ambient.
struct CloudLight
{
    float3 sun;          // E / π × Exposure of the sun at the cloud: what a white surface facing it sends back
    float3 moon;         // the same for the brightest moon
    float3 ambient;      // sky above and ground below, as a multiple of albedo
    float3 viewT;        // transmittance from the camera to the cloud
    float3 inFront;      // the light the air in front of the cloud adds (× Exposure)
};

CloudLight GetCloudLight(float3 dir)
{
    CloudLight c = (CloudLight)0;
    c.viewT = 1;
    if (AtmosphereIdx == 0)
        return c;
    AtmosphereParams A = LoadAtmosphere();

    float viewHeight = AtmosphereViewHeight(A, CamPos.y);
    float mu = max(dir.y, 0.01);
    float tKm = min(CloudAltitude / mu, 300000.0) * 0.001;
    float3 P = float3(0, viewHeight, 0) + float3(dir.x, mu, dir.z) * tKm;
    float r = length(P);

    c.sun = A.SunIlluminance * AtmTransmittanceToLight(A, P, A.SunDirection) * (Exposure / ATM_PI);
    c.moon = A.MoonIlluminance * AtmTransmittanceToLight(A, P, A.MoonDirection) * (Exposure / ATM_PI);

    StructuredBuffer<float4> ambient = ResourceDescriptorHeap[A.Ambient];
    c.ambient = (ambient[0].rgb + ambient[1].rgb) * 0.5;

    // Transmittance camera → cloud from two LUT reads (to the top of the atmosphere from either end), and the share of
    // the sky's in-scattered light that lies in front of the cloud, taken as the share of the optical depth there
    float3 toTop = AtmTransmittance(A, viewHeight, mu);
    float3 cloudToTop = AtmTransmittance(A, r, dot(P / r, normalize(float3(dir.x, mu, dir.z))));
    c.viewT = saturate(toTop / max(cloudToTop, 1e-6));
    float3 sky = AtmosphereSkyLuminance(A, normalize(float3(dir.x, mu, dir.z))) * Exposure;
    c.inFront = sky * saturate((1.0 - c.viewT) / max(1.0 - toTop, 1e-4));
    return c;
}

// The cloud layer as one flat colour (no self-shadowing, no lining), as it appears from the camera: for reflections.
float3 CloudFlatRadiance(CloudLight c)
{
    return CloudAlbedo * ((c.sun + c.moon) * 0.25 + c.ambient) * c.viewT + c.inFront;
}

// ── The cloud layer as the sky dome draws it (mesh_skybox.fx), and the sky probe (sky_probe_cs.hlsl) with it ──
// LUT-based Nubis-style clouds. LUT channels (cloud_noise_gen.hlsl): R Perlin FBM (smooth connected shapes), G Worley
// FBM (low-freq billowy erosion), B Worley FBM (high-freq detail erosion), A Perlin-Worley (pre-combined cloud shape).
// mipBias blurs the noise further (the probe's coarse texels).
float GetCloudsAt(SamplerState wrapSampler, float3 viewDir, float altitude, float mipBias)
{
    if (viewDir.y <= 0.001 || CloudNoiseLUTIdx == 0)
        return 0.0;

    // Shared with the cloud shadows: same plane, same UVs, same base density
    float t = altitude / viewDir.y;
    float2 uv = CloudLayerUV(viewDir.xz * t);

    Texture3D<float4> noiseLUT = ResourceDescriptorHeap[CloudNoiseLUTIdx];
    float timeZ = CloudTime * 0.005;

    // Mip level based on viewing angle — blur out detail near the horizon
    // to hide tiling and create natural atmospheric softening
    float mip = saturate(1.0 - viewDir.y * 5.0) * 3.0 + mipBias;  // 0 at zenith, up to 3 at horizon

    // ── Base shape against the coverage threshold ──
    float cloudDensity = CloudBaseDensity(noiseLUT, wrapSampler, uv, mip);

    // ── Detail erosion: Worley carves billowy edges ──
    // Use higher mip near horizon to blur out repetition
    float4 detailSample = noiseLUT.SampleLevel(wrapSampler, float3(uv * 2.0 + 1.13, timeZ * 1.5), mip);
    float detailFBM = detailSample.g * 0.625 + detailSample.b * 0.375;

    // Key fix: erosion strength scales with (1 - density)
    // Thick cloud cores resist erosion; only thin edges get carved
    float erodeStrength = detailFBM * 0.2 * (1.0 - cloudDensity * 0.7);
    float eroded = remap(cloudDensity, erodeStrength, 1.0, 0.0, 1.0);

    // Softer smoothstep for fuller cloud bodies
    eroded = smoothstep(0.0, 0.45, eroded);

    // Horizon fade
    eroded *= smoothstep(0.0, 0.22, viewDir.y);

    return saturate(eroded);
}

// How much of a light (sun or moon) a cloud sends toward the eye, per unit of what a white surface facing that light
// would: the density's self-shadowing (thin edges bright, thick cores dark), brighter on the side toward the light,
// a silver lining on thin edges and a powder glow at medium density when looking toward the light. 0.6: a sunlit
// edge is a little over half as bright as a white wall in the sun (as the old palettes had it).
float CloudDirectShade(float3 viewDir, float3 lightDir, float density, float transmittance)
{
    float lightDot = saturate(dot(viewDir, lightDir));
    float gradient = lightDot * 0.5 + 0.5;              // 0.5 (shade side) to 1.0 (facing the light)
    float shade = lerp(transmittance * 0.7, transmittance, gradient);

    float edgeMask = smoothstep(0.0, 0.2, density) * smoothstep(0.5, 0.15, density);
    shade += edgeMask * lightDot * lightDot * 0.4;                                      // silver lining
    shade += (1.0 - transmittance) * transmittance * 2.0 * lightDot * lightDot * lightDot * 0.3;   // powder
    return shade * 0.6;
}

// The sky's light in direction viewDir (skyColor, × Exposure) with the cloud layer over it, lit by the sun and the
// brightest moon through the air above the cloud and by the sky and the ground; the air between the camera and the
// cloud dims it and adds its own light in front. density: the cloud's, for what lies behind it (sun, moons, stars).
float3 SkyThroughClouds(SamplerState wrapSampler, float3 viewDir, float3 skyColor, float mipBias, out float density)
{
    // Altitude scales the projected noise, so lerping it zooms the whole pattern. While a preset
    // transition runs we instead cross-fade the density of the old and new cloud layers.
    density = GetCloudsAt(wrapSampler, viewDir, CloudAltitude, mipBias);
    if (CloudAltitudeBlend < 0.999)
    {
        float densityFrom = GetCloudsAt(wrapSampler, viewDir, CloudAltitudeFrom, mipBias);
        density = lerp(densityFrom, density, smoothstep(0.0, 1.0, saturate(CloudAltitudeBlend)));
    }
    if (density <= 0.001 || AtmosphereIdx == 0)
        return skyColor;

    AtmosphereParams A = LoadAtmosphere();
    float absorption = 1.0 - exp(-density * 4.5);           // opacity, Beer-Lambert
    float transmittance = exp(-density * 3.0);              // self-shadowing: 1 at thin edges, ~0.05 in dense cores
    CloudLight cl = GetCloudLight(viewDir);
    float3 lit = cl.sun * CloudDirectShade(viewDir, A.SunDirection, density, transmittance)
               + cl.moon * CloudDirectShade(viewDir, A.MoonDirection, density, transmittance)
               + cl.ambient;
    float3 cloudColor = CloudAlbedo * lit * cl.viewT + cl.inFront;
    return lerp(skyColor, cloudColor, absorption);
}

#endif // SKY_COMMON
