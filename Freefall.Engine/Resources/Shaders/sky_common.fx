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
#include "cloud_common.hlsli"
#include "light_clusters.hlsli"

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

// The sky's light round direction R averaged over a cone `width` radians across, from the sky probe's mip whose texels
// are that wide (mip 0's texels span 90° / its size; each mip doubles them). maxWidth caps the blur (water: the lobe
// never reaches under the horizon). Without the probe as SkyReflection.
float3 SkyReflectionCone(float3 R, float width, float maxWidth)
{
    if (AtmosphereIdx == 0)
        return 0;
    AtmosphereParams A = LoadAtmosphere();
    if (A.SkyProbeCube == 0)
        return SkyReflection(R, 0.0);
    TextureCube<float4> probe = ResourceDescriptorHeap[A.SkyProbeCube];
    float size, height, mips;
    probe.GetDimensions(0, size, height, mips);
    float texel = 0.5 * ATM_PI / size;
    float mip = clamp(log2(max(min(width, maxWidth), 1e-4) / texel), 0.0, mips - 1.0);
    return probe.SampleLevel(AtmosphereLinearClamp, R, mip).rgb;
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

// The terrain's shadow on a lit surface, 0..1, for the directional light toward toLight: the shadow height map of the
// body it points at (the sun's or the brightest moon's, terrain_light_shadow_cs.hlsl; 1 for any other direction). The
// cascades hold casters only ~300 m toward the sun and end at MaxShadowDistance: without this a mountain the sun stood
// behind left the meadow and the woods in front of it sunlit while the air over them lay in its shadow (Lake Mere).
// Only the terrain from TERRAIN_SHADOW_SKIP metres toward the light on counts, M(p + s0·d) − s0·tan(e): the map's
// first texels (8 m wide, the highest of four samples each, the first step one texel toward the light) put the top a
// metre or two over sunlit slopes at a low sun, and the cascades shade those near casters anyway. The surface is
// tested on the heightmap where the coarser rendered terrain sits under it (a ridge from afar), lifted by 2 m plus the
// most the four samples gain on its slope, soft over ±3 m. worldPos is absolute.
#define TERRAIN_SHADOW_SKIP 32.0
float TerrainLightShadow(float3 worldPos, float3 toLight)
{
    if (AtmosphereIdx == 0)
        return 1.0;
    AtmosphereParams A = LoadAtmosphere();
    uint map = 0;
    if (dot(toLight, A.SunDirection) > 0.999)
        map = A.TerrainSunShadow;
    else if (dot(toLight, A.MoonDirection) > 0.999)
        map = A.TerrainMoonShadow;
    float across = length(toLight.xz);
    if (map == 0 || across < 1e-4)          // no map, or the light overhead: no shadow across the ground
        return 1.0;
    float2 d = toLight.xz / across;
    float tanE = toLight.y / across;
    float2 uv = (worldPos.xz + d * TERRAIN_SHADOW_SKIP - A.TerrainShadowOrigin) / A.TerrainShadowSize;
    if (any(uv < 0.0) || any(uv > 1.0))
        return 1.0;
    Texture2D<float> shadowTop = ResourceDescriptorHeap[map];
    float top = shadowTop.SampleLevel(AtmosphereLinearClamp, uv, 0) - TERRAIN_SHADOW_SKIP * tanE;

    float y = worldPos.y;
    float bias = 2.0;
    if (y + bias >= top + 3.0)              // lit already: the heightmap can only lift it further (most of the screen)
        return 1.0;
    TerrainCover t = LoadTerrainCover();
    if (t.Heightmap != 0)
    {
        float h = TerrainHeightAt(t, AtmosphereLinearClamp, worldPos.xz);
        if (h - y < TerrainTolerance(distance(worldPos, CamPos)))
            y = max(y, h);
        // The map's four samples per texel sit a quarter texel either way: on a slope the highest gains that much
        float2 slope = abs(float2(TerrainHeightAt(t, AtmosphereLinearClamp, worldPos.xz + float2(4, 0)),
                                  TerrainHeightAt(t, AtmosphereLinearClamp, worldPos.xz + float2(0, 4))) - h) / 4.0;
        float texelsX, texelsY;
        shadowTop.GetDimensions(texelsX, texelsY);
        bias += (slope.x + slope.y) * A.TerrainShadowSize / texelsX * 0.5;
    }
    return smoothstep(top - 3.0, top + 3.0, y + bias);
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

// Under the terrain the meshes' part loses its first 5 % inside the cave: there the captures' noise (a facet that happens
// to face the one low direction that reaches in through the mouth, slope bias) read 0.3-4 % open on scattered facets, and
// at a cave's exposure, ~13 EV under the daylight, that much sky turned them pure white: streaks over the walls and roof
// like snow (user report, Whispering Hollow, 2026-10-18). At the mouth the floor fades out with the portal light (~0.7 %
// where the streaks were, 16 m in; ~1.6 % on the floor under the mouth's overhang, which a fixed 5 % turned into dark
// islands in the daylight).
#define CAVE_MESH_FLOOR 0.05
float CaveMeshShare(float meshes, float portal)
{
    float floorShare = CAVE_MESH_FLOOR * (1.0 - smoothstep(0.008, 0.02, portal));
    return saturate((meshes - floorShare) / (1.0 - floorShare));
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
    return lerp(min(portal, CaveMeshShare(meshes, portal)), sky * max(meshes, bounce), open);
}

// The same, the meshes' part looked up here. worldPos is absolute. The opaque surfaces have theirs from
// sky_visibility_cs.hlsl (the light buffer's alpha), cheaper per pixel.
float SkyVisibility(float3 worldPos, float3 N)
{
    return SkyVisibilityWith(worldPos, N, MeshSkyVisibility(worldPos, N));
}

// How much of the air's own light to keep in front of a surface that sees skyVisible of the sky. The aerial perspective
// volume lights every slice of air with the sun and sky as if nothing shaded it, so in a cave (a ray ending under the
// hill crossed dark air) it glowed over the black walls. Only the stretch of the ray inside the cave is dark: at most the
// last 30 m when the camera is in the open, all of it when the camera is in the cave too. Scaling the whole ray left a
// cave mouth seen from 400 m without the haze that covers everything round it: a black blob in rain (user, 2026-10-18).
// With the near fog volume it applies to the far part of the ray only (AtmosphereFog): the near volume knows per froxel
// where the sky is seen.
float AerialSkyVisible(float skyVisible, float viewDepth)
{
    float cameraOpen = TerrainOpen(CamPos, float3(0, 1, 0));
    float darkStretch = lerp(viewDepth, min(viewDepth, 30.0), cameraOpen);
    return 1.0 - (1.0 - skyVisible) * darkStretch / max(viewDepth, 1e-3);
}

// ── Near fog volume (Graphics.VolumetricFog, volumetric_fog_cs.hlsl) ──
#define FOG_TILE 16     // pixels a froxel spans each way (VolumetricFog.TileSize)

// The air between the camera and viewDepth (at most FogVolumeRange) from the near fog volume: rgb = its light
// (× Exposure), a = transmittance. Slice s holds the air up to view depth ((s + 1) / slices)² × FogVolumeRange. Read
// with the B-spline across the screen (atmosphere_common.hlsli SampleScreenBSpline): linearly, the shafts through the
// trees showed the 16 px tiles as steps.
float4 NearFog(float2 pixel, float viewDepth)
{
    Texture3D<float4> fog = ResourceDescriptorHeap[FogVolumeIdx];
    float width, height, slices;
    fog.GetDimensions(width, height, slices);
    float2 uv = pixel / (FOG_TILE * float2(width, height));
    float s = sqrt(saturate(viewDepth / FogVolumeRange)) * slices;   // the far side of slice k at s = k + 1
    if (s < 1.0)
    {
        // In front of the first slice's far side: toward nothing at the camera
        float4 first = SampleScreenBSpline(fog, uv, 0.5 / slices);
        return float4(first.rgb * s, 1.0 - s * (1.0 - first.a));
    }
    return SampleScreenBSpline(fog, uv, (s - 0.5) / slices);
}

// The fog between the camera and a surface at viewDepth: rgb = the light the air adds (× Exposure), a = the share of the
// surface's own light that gets through; color × a + rgb. With the near fog volume (Engine.Settings.VolumetricFog) its
// first FogVolumeRange metres come from it, lit with the sun's and the moon's shadows (cascades, clouds, the terrain: god
// rays) and with the sky's light only where the sky is seen; the aerial perspective volume goes on from there,
//   far = (AP(d) − AP(R)) / T(R), T_far = T(d) / T(R),   fog = (color × T_far + far) × T_near + near.
// skyVisible (the surface's SkyVisibility, 1 for one in the open) dims the far part's light (AerialSkyVisible).
//   pixel:     SV_Position.xy (or the pixel a compute thread shades)
//   viewDepth: linear view-space depth in metres (the G-buffer's depth, clip w)
// Sky pixels must not come here (AtmosphereFogOverSky): the sky's light already holds all of the air.
// The air alone: AtmosphereFog (below) adds the volumetric clouds in front of the surface.
float4 AtmosphereFogAir(float2 pixel, float viewDepth, float skyVisible)
{
    if (FogEnabled <= 0.0 || AtmosphereIdx == 0)
        return float4(0, 0, 0, 1);
    AtmosphereParams A = LoadAtmosphere();
    float2 uv = pixel / ViewportSize;
    float4 ap = AtmosphereAerialPerspective(A, uv, viewDepth);
    if (FogVolumeIdx == 0)
        return float4(ap.rgb * AerialSkyVisible(skyVisible, viewDepth), ap.a);

    float range = FogVolumeRange;
    float4 nearFog = NearFog(pixel, min(viewDepth, range));
    if (viewDepth <= range)
        return nearFog;
    float4 apNear = AtmosphereAerialPerspective(A, uv, range);
    float throughNear = max(apNear.a, 1e-4);
    float3 farLight = max(ap.rgb - apNear.rgb, 0.0) / throughNear * AerialSkyVisible(skyVisible, viewDepth - range);
    float farTransmittance = saturate(ap.a / throughNear);
    return float4(nearFog.rgb + nearFog.a * farLight, nearFog.a * farTransmittance);
}

// A sky pixel (its light holds all of the air along the ray, the sky-view LUT) with the first FogVolumeRange metres of
// that air from the near fog volume instead: the shafts of light and the shade in the air reach over the sky too
float3 AtmosphereFogOverSky(float3 sky, float2 pixel)
{
    if (FogEnabled <= 0.0 || AtmosphereIdx == 0 || FogVolumeIdx == 0)
        return sky;
    float4 apNear = AtmosphereAerialPerspective(LoadAtmosphere(), pixel / ViewportSize, FogVolumeRange);
    float4 nearFog = NearFog(pixel, FogVolumeRange);
    return max(sky - apNear.rgb, 0.0) / max(apNear.a, 1e-4) * nearFog.a + nearFog.rgb;
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
    float2 uv = CloudLayerUV(relPos.xz + toLight.xz * t);

    // No detail erosion, and a steeper ramp than the sky dome so the shadow has a readable edge
    return smoothstep(0.0, 0.2, CloudBaseDensity(noiseLUT, wrapSampler, uv, 0.0));
}

// ── What the clouds let through: to the ground, the water, the air, the meter, particles ──
// A slab of clouds of optical depth τ toward a light lets through two parts, which the code keeps apart
// (VOLUMETRIC_CLOUDS_PLAN §9, stage 4):
//   - the beam, e^−τ: the light as a direction, with its shadows, its glints on water and the glow round it in the air;
//   - what the clouds diffuse: the slab passes 1 / (1 + 0.75 (1 − g) τ) in all (two-stream, g 0.85 for droplets; τ 20 →
//     31 %, τ 60 → 13 %), the beam among it. That light comes from the whole cloud base: surfaces get it through the sky
//     probe's ambient (it sees the deck's lit base), the air as an even scatterer (CloudDiffusedLight).
// It replaces the 2D layer's fixed shadow strength (20 % through any cloud) and its fade below 17°: the light volumes
// are built along the light, so a low sun's shadows hold to the horizon.

#define CLOUD_DIFFUSION 0.1125      // 0.75 (1 − g), g 0.85
#define CLOUD_LAYER_TAU 36.0        // the 2D layer's optical depth at density 1 (VolumetricClouds off): a deck passes 20 %
#define CLOUD_DECK_EDGE 100.0       // metres round the cloud base over which a place goes from under the clouds to among them

// What a slab of optical depth tau lets through in all: the beam and what it diffuses
float CloudThrough(float tau)
{
    return 1.0 / (1.0 + CLOUD_DIFFUSION * tau);
}

// The clouds round the camera, made each frame by atmosphere_cs.hlsl CSCloudDeck (the Ambient buffer's element 4),
// toward the light the clouds are lit by (CloudParams.LightDir: the sun, at night the brightest moon; for the 2D layer
// the sun while it is up):
//   x  the mean beam over CLOUD_DECK_RADIUS round the camera (the sky-view LUT's air below the clouds, the same in every
//      direction, and the light a moon gets by day);
//   y  the mean of all they let through, CloudThrough (the ground's light round the camera: CSAmbient, Daylight);
//   z  the beam at the camera;
//   w  1 when that light is the moon.
// (1, 1, 1, 0) without clouds.
float4 CloudDeck()
{
    if (AtmosphereIdx == 0)
        return float4(1, 1, 1, 0);
    StructuredBuffer<float4> ambient = ResourceDescriptorHeap[LoadAtmosphere().Ambient];
    return ambient[4];
}

// How far a place (camera-relative) lies under the clouds: 1 below their base, 0 above it, over CLOUD_DECK_EDGE
float CloudBelowBase(float3 relPos)
{
    if (CloudsOn != 0)
    {
        CloudParams C = LoadClouds();
        return 1.0 - smoothstep(C.Base - CLOUD_DECK_EDGE, C.Base + CLOUD_DECK_EDGE, CloudHeight(relPos));
    }
    if (CloudNoiseLUTIdx == 0 || CloudCoverage <= 0.0)
        return 0.0;
    return 1.0 - smoothstep(CloudAltitude - CLOUD_DECK_EDGE, CloudAltitude + CLOUD_DECK_EDGE, CamPos.y + relPos.y);
}

// The 2D layer's optical depth toward a light (VolumetricClouds off, until the layer goes in stage 10): its density ×
// CLOUD_LAYER_TAU, the old and the new layer cross-faded while a preset changes the altitude, faded out below 17° where
// the projection stretches toward infinity (a low sun shines under a 2D deck)
float CloudLayerDepth(SamplerState wrapSampler, float3 relPos, float3 toLight)
{
    if (CloudNoiseLUTIdx == 0 || CloudCoverage <= 0.0 || toLight.y <= 0.05)
        return 0.0;
    Texture3D<float4> noiseLUT = ResourceDescriptorHeap[CloudNoiseLUTIdx];
    float density = CloudShadowDensityAt(noiseLUT, wrapSampler, relPos, toLight, CloudAltitude);
    if (CloudAltitudeBlend < 0.999)
    {
        float densityFrom = CloudShadowDensityAt(noiseLUT, wrapSampler, relPos, toLight, CloudAltitudeFrom);
        density = lerp(densityFrom, density, smoothstep(0.0, 1.0, saturate(CloudAltitudeBlend)));
    }
    return CLOUD_LAYER_TAU * density * smoothstep(0.05, 0.3, toLight.y);
}

// What the clouds let through from a light to a place (camera-relative), as shares of its illuminance: x the beam, y all
// of it (CloudThrough). The volumetric clouds' τ is their light volume where the light's path crosses the slab's base
// (cloud_common.hlsli CloudShadowDepth); for a light the volumes are not built for (a moon by day, the sun once the moon
// lights the clouds) the means round the camera stand in (CloudDeck), below the base. wrapSampler: linear wrap.
float2 CloudTransmission(SamplerState wrapSampler, float3 relPos, float3 toLight)
{
    float tau;
    if (CloudsOn != 0)
    {
        CloudParams C = LoadClouds();
        if (dot(toLight, C.LightDir) < 0.99)
            return lerp(1.0, CloudDeck().xy, CloudBelowBase(relPos));
        tau = CloudShadowDepth(C, relPos, toLight, wrapSampler);
    }
    else
        tau = CloudLayerDepth(wrapSampler, relPos, toLight);
    return float2(exp(-tau), CloudThrough(tau));
}

// The share of a light's beam that reaches a place through the clouds (surfaces, water, glass, the meter, particles)
float CloudBeam(SamplerState wrapSampler, float3 relPos, float3 toLight)
{
    return CloudTransmission(wrapSampler, relPos, toLight).x;
}

// ── The clouds in the air (the aerial perspective, the sky-view LUT, the near fog volume) ──

// How much the multiple scattering (Ψ_ms) of the air below the clouds is dimmed, 0..1, from the deck round the camera
// (CloudDeck). Ψ_ms is the light of a cloudless sky and the lit ground round a place; under the clouds the share of it
// they hold back is 1 − y, and a broken layer's gaps still show the blue sky that makes most of it, hence × the share
// under cloud, 1 − x: Clear Day (x ~0.8, y ~0.85) loses ~3 %, a deck (x ~0, y 0.13-0.3) 70-87 %. Without the dimming
// the haze under a deck outshone the deck (a dark band under it on the Overcast horizon).
// Tried first: the sky probe's mean radiance against Ψ_ms × the illuminance; without a cloud it read 0.89-1.03 with the
// time of day, so it dimmed a cloudless noon by 10 %.
float CloudSkyDimming(float4 deck)
{
    return saturate((1.0 - deck.y) * (1.0 - deck.x));
}

// What the clouds let through to a place in the air toward a light, as shares of its illuminance:
//   x  the beam, which the air scatters with its phases (the Mie forward lobe is the glow round the sun);
//   y  the light the clouds have diffused, CloudThrough − the beam, which the air scatters evenly: under a deck the sun
//      still lights the air, but from the whole cloud base, not from its place in the sky (CloudDiffusedLight);
//   z  the share of the multiple scattering (Ψ_ms) left: the volumetric clouds' sky light at the place, what their
//      column over it lets through diffused (the sky volume's depth straight up); the 2D layer's 1 − skyDimming
//      (CloudSkyDimming) below its base. The deck's mean round the camera there left a storm's air lit by most of the
//      blue sky: the deck toward the sun read blue (stage 7).
// relPos: camera-relative.
// diffusedTint: what .y's light carries over the beam's colour at the place: under the volumetric clouds it entered them
// above, through less air than a beam down to here takes (clouds_cs.hlsl CloudMarchClouds), so × the air's transmittance
// there over here's (the air under a storm came out sepia); 1 for the 2D layer.
float3 CloudAirLight(SamplerState wrapSampler, float3 relPos, float3 toLight, float skyDimming, out float3 diffusedTint)
{
    float2 t = CloudTransmission(wrapSampler, relPos, toLight);
    diffusedTint = 1.0;
    float sky;
    if (CloudsOn != 0)
    {
        CloudParams C = LoadClouds();
        // The sky light comes from a wide cone of the sky, not a line straight up: the clouds over an area about as wide
        // as the place lies under the base (a gap's light spreads ~45° each way). The column straight up alone drew
        // every gap of a broken deck as a vertical shaft of brighter air, standing straight up whatever the sun's
        // direction (the real shafts, the beam's, point at the sun). One read of a coarser level of the sky volume, its
        // texels a quarter of that depth wide, filtered (four columns round the place cost the aerial perspective
        // 0.24 ms).
        float h = CloudHeight(relPos);
        float blend;
        uint near = CloudLevelAt(C, max(abs(relPos.x), abs(relPos.z)), blend);
        uint wide = (uint)clamp(floor(log2(max(C.Base - h, 250.0) / (4.0 * C.Levels[0].VoxelSize))), (float)near,
                                (float)(CLOUD_LEVELS - 1));
        float tauUp = CloudSkyDepthAt(C, wide, C.CameraCloud + relPos.xz, h, wrapSampler).x;
        sky = CloudThrough(tauUp);
        if (AtmosphereIdx != 0 && t.y > t.x + 1e-3)
        {
            AtmosphereParams A = LoadAtmosphere();
            float entry = min(max(h, C.Base) + 2.0 * tauUp / max(C.Density, 1e-4), C.SlabTop) - h;
            float3 here = AtmTransmittanceToLight(A, CloudAtmospherePos(relPos), toLight);
            float3 there = AtmTransmittanceToLight(A, CloudAtmospherePos(relPos + float3(0.0, max(entry, 0.0), 0.0)), toLight);
            diffusedTint = all(here > 1e-3) ? min(there / here, 3.0) : 1.0;
        }
    }
    else
        sky = 1.0 - skyDimming * CloudBelowBase(relPos);
    // Through a thin cloud (a cumulus' edge) droplets (g 0.85) throw most of what they scatter straight on: it lights the
    // air like the beam, with the air's phase. Taken as diffused (scattered evenly) at every depth it outshone the open
    // sun's backscatter: a bright ring round every cloud's shadow in the haze seen from above. Only through a thick
    // cloud has it spread evenly.
    float diffused = max(t.y - t.x, 0.0);
    // e^(−τ/20), τ from the share passing through, CloudThrough = 1 / (1 + CLOUD_DIFFUSION τ) (the beam underflows past
    // τ ~14)
    float tau = (1.0 / max(t.y, 1e-4) - 1.0) / CLOUD_DIFFUSION;
    float forward = diffused * exp(-tau / 20.0);
    return float3(t.x + forward, diffused - forward, sky);
}

// Light on a lit particle or ribbon (smoke, rain, snow, a blade's trail), as a multiple of its albedo (E / π,
// pre-exposed): half the sun's and the moon's light through the air (at the camera's height), the clouds' beam and the
// shadows at the particle (a puff is lit from one side), and the sky probe's ambient from above and below averaged, which
// holds the light the clouds diffuse (their lit base) as it does for surfaces: grey by day, dark at night. Callers
// multiply by the SkyVisibility at the particle: smoke in a cave stays dark, since neither the sky nor the sun reaches it
// there (they also scale the haze in front of it by that). The point lights come on top (ParticlePointLight), not scaled
// by it: the fire in the cave lights its own smoke.

// How much more of a light a particle shows seen at angle cosTheta from the light's way on (1 = looking into the light
// through it) than an even scatterer would: Henyey-Greenstein with anisotropy g, 1 everywhere for g = 0. Droplets throw
// most of the light on, so spray glows with the sun behind it and dims with the sun behind the viewer; the share `even`
// stays even, the light scattered several times inside a dense puff (ParticleEmitter.ForwardScatter, 2026-10-19;
// MultipleScattering, 2026-10-20: 0.4 for smoke and spray, 0 for rain drops, each of which scatters the light once)
float ParticlePhase(float cosTheta, float g, float even)
{
    float d = 1.0 + g * g - 2.0 * g * cosTheta;
    return even + (1.0 - even) * (1.0 - g * g) / (d * sqrt(d));
}

float ParticlePhase(float cosTheta, float g) { return ParticlePhase(cosTheta, g, 0.4); }

//   relPos: the particle, camera-relative; toEye: from it to the camera; g: its forward scattering (0 = the same from
//   every side); even: the share of it scattered evenly (ParticlePhase); shadows: x the sun's, y the moon's (1 = lit,
//   ParticleBodyShadows); wrapSampler: linear wrap (the clouds' volumes)
float3 ParticleLight(SamplerState wrapSampler, float3 relPos, float3 toEye, float g, float even, float2 shadows)
{
    if (AtmosphereIdx == 0)
        return 0;
    AtmosphereParams A = LoadAtmosphere();
    float3 P = float3(0, AtmosphereViewHeight(A, CamPos.y), 0);
    float3 direct = A.SunIlluminance * AtmTransmittanceToLight(A, P, A.SunDirection)
                  * (ParticlePhase(dot(toEye, -A.SunDirection), g, even) * CloudBeam(wrapSampler, relPos, A.SunDirection) * shadows.x)
                  + A.MoonIlluminance * AtmTransmittanceToLight(A, P, A.MoonDirection)
                  * (ParticlePhase(dot(toEye, -A.MoonDirection), g, even) * CloudBeam(wrapSampler, relPos, A.MoonDirection) * shadows.y);
    return direct * (0.5 * Exposure / ATM_PI) + (SkyAmbient(float3(0, 1, 0)) + SkyAmbient(float3(0, -1, 0))) * 0.5;
}

float3 ParticleLight(SamplerState wrapSampler, float3 relPos, float3 toEye, float g, float2 shadows)
{
    return ParticleLight(wrapSampler, relPos, toEye, g, 0.4, shadows);
}

float3 ParticleLight(SamplerState wrapSampler, float3 relPos, float3 toEye, float g)
{
    return ParticleLight(wrapSampler, relPos, toEye, g, 0.4, float2(1.0, 1.0));
}

// At the camera, lit evenly from every side (ribbons without a place of their own)
float3 ParticleLight(SamplerState wrapSampler) { return ParticleLight(wrapSampler, float3(0, 0, 0), float3(0, 1, 0), 0.0); }

// The sun's and the moon's shadow (x, y; 1 = lit) at a place off the surfaces (a particle, a ribbon's point): the
// terrain's past 32 m (TerrainLightShadow), and for the body the cascades were drawn for this frame the first cascade that
// holds the place, one comparison 0.3 m toward the light (as the near fog volume takes them). The cascades come from the
// light clusters' header (built after they are drawn). relPos: camera-relative; cmp: comparison, less-equal (s3).
float2 ParticleBodyShadows(float3 relPos, SamplerComparisonState cmp)
{
    if (AtmosphereIdx == 0)
        return 1.0;
    AtmosphereParams A = LoadAtmosphere();
    float3 absPos = CamPos + relPos;
    float2 shadows = float2(TerrainLightShadow(absPos, A.SunDirection), TerrainLightShadow(absPos, A.MoonDirection));
    if (LightClustersIdx == 0)
        return shadows;
    ByteAddressBuffer clusters = ResourceDescriptorHeap[LightClustersIdx];
    uint mapIdx = clusters.Load(LC_SUN_SHADOW_MAP * 4);
    if (mapIdx == 0)
        return shadows;
    float3 shadowDir = asfloat(clusters.Load3(LC_SUN_DIRECTION * 4));
    bool sun = dot(shadowDir, A.SunDirection) > 0.999;
    if (!sun && dot(shadowDir, A.MoonDirection) <= 0.999)
        return shadows;

    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[clusters.Load(LC_SUN_CASCADES * 4)];
    Texture2DArray<float> shadowMap = ResourceDescriptorHeap[mapIdx];
    uint count = clusters.Load(LC_SUN_CASCADE_COUNT * 4);
    float viewDepth = mul(float4(relPos, 1.0), CameraRelativeVP).w;
    float lit = 1.0;
    for (uint c = 0; c < count; c++)
    {
        CascadeData cd = cascades[c];
        if (viewDepth > cd.SplitDistances.y)
            continue;
        float4 p = mul(float4(absPos, 1.0), cd.VP);
        float2 uv = p.xy * float2(0.5, -0.5) + 0.5;
        if (any(uv < 0.0) || any(uv > 1.0))
            continue;
        float zPerMetre = length(float3(cd.VP._13, cd.VP._23, cd.VP._33));
        lit = shadowMap.SampleCmpLevelZero(cmp, float3(uv, c), p.z - 0.3 * zPerMetre);
        break;
    }
    if (sun)
        shadows.x = min(shadows.x, lit);
    else
        shadows.y = min(shadows.y, lit);
    return shadows;
}

// The point lights' light on a lit particle or ribbon, as ParticleLight gives the sun's (half of E / π, pre-exposed, with
// its forward scattering g): the lights that reach it (light_clusters.hlsli), each through its shadow where it has one
// this frame. Taken no closer to a light than the particle's radius: its light is the mean over the puff, not 1 / d² at
// its centre (smoke born inside the brazier's flame). relPos: camera-relative; cmp: comparison, less-equal (s3).
float3 ParticlePointLight(float3 relPos, float3 toEye, float g, float even, float radius, SamplerComparisonState cmp)
{
    if (LightClustersIdx == 0 || ClusterLightsIdx == 0)
        return 0;
    float4 clip = mul(float4(relPos, 1.0), CameraRelativeVP);
    if (clip.w <= 1e-3)
        return 0;
    ByteAddressBuffer clusters = ResourceDescriptorHeap[LightClustersIdx];
    StructuredBuffer<PointLightData> lights = ResourceDescriptorHeap[ClusterLightsIdx];
    LightClusterWalk walk;
    if (!LightClusterBegin(clusters, clip.xy / clip.w * float2(0.5, -0.5) + 0.5, clip.w, walk))
        return 0;

    float minD2 = max(radius * radius, 0.01);
    float3 sum = 0;
    uint s;
    [loop] while (LightClusterNext(clusters, walk, s))
    {
        PointLightData light = lights[s];
        float3 toLight = light.Position - relPos;
        float3 E = PointLightIlluminance(light, toLight, minD2);
        if (all(E <= 0.0))
            continue;
        float shadow = PointLightShadowAt(clusters, light, relPos, cmp);
        if (shadow <= 0.0)
            continue;
        float cosTheta = -dot(toEye, toLight) * rsqrt(max(dot(toLight, toLight), 1e-6));
        sum += E * (shadow * ParticlePhase(cosTheta, g, even));
    }
    return sum * (0.5 * Exposure / ATM_PI);
}

float3 ParticlePointLight(float3 relPos, float3 toEye, float g, float radius, SamplerComparisonState cmp)
{
    return ParticlePointLight(relPos, toEye, g, 0.4, radius, cmp);
}

// The light the cloud layer diffuses (CloudAirLight .y, a share of the light's illuminance E) as the air below it scatters
// it toward the eye, per unit illuminance and of the air's scattering coefficient: the base sends share × E × cos(the
// light's zenith angle) down over the lower hemisphere (radiance share × E cos / π), and the air scatters it evenly,
// σs × share × E cos / (2π). Like the ground's light under the deck, it falls with the sun.
float CloudDiffusedLight(float share, float lightZenithCos)
{
    return share * saturate(lightZenithCos) / (2.0 * ATM_PI);
}

// The sky's light from a direction scattered below the cloud layer's plane (cd/m², sun + moon, no airglow: that is
// behind the clouds), from the sky-view LUTs' alpha (atmosphere_cs.hlsl CSSkyView): the air in front of a cloud there.
// It was the sky's light × the share of the optical depth in front of the cloud; once the air under a deck lay in its
// shade, the lit air above the deck (behind the cloud) came through that share as a bright band over a dark horizon.
float3 AtmosphereSkyBelowClouds(AtmosphereParams A, float3 dir)
{
    float viewHeight = AtmosphereViewHeight(A, CamPos.y);
    Texture2D<float4> sunLut = ResourceDescriptorHeap[A.SkyViewSunLut];
    float4 sun = sunLut.SampleLevel(AtmosphereLinearClamp,
        SkyViewLutUv(A, false, dir.y, LightViewCos(dir, A.SunDirection), viewHeight), 0);
    float3 L = sun.rgb * sun.a * A.SunIlluminance;
    if (any(A.MoonIlluminance > 0))
    {
        Texture2D<float4> moonLut = ResourceDescriptorHeap[A.SkyViewMoonLut];
        float4 moon = moonLut.SampleLevel(AtmosphereLinearClamp,
            SkyViewLutUv(A, false, dir.y, LightViewCos(dir, A.MoonDirection), viewHeight), 0);
        L += moon.rgb * moon.a * A.MoonIlluminance;
    }
    return L;
}

// The air's transmittance from the camera to a point s metres along a ray (unit dir), from two transmittance LUT reads
// (to the top of the atmosphere from either end); the planet's curvature as the clouds have it. A ray going down (a
// deck seen from above) is read the other way round, upward from the point through the camera: the LUT holds rays to
// the top of the atmosphere, and one that meets the ground has none (the clouds under the camera went black).
// Within ~3° of the horizon both reads are tiny (the whole atmosphere edge-on) and their ratio falls apart channel by
// channel (a yellow line along the horizon from inside a deck): there the segment itself is integrated, 8 samples.
float3 AtmTransmittanceAlong(AtmosphereParams A, float3 dir, float s)
{
    float3 P = CloudAtmospherePos(dir * s);
    float r = length(P);
    float viewHeight = AtmosphereViewHeight(A, CamPos.y);
    float3 lut = dir.y >= 0.0
        ? saturate(AtmTransmittance(A, viewHeight, dir.y) / max(AtmTransmittance(A, r, dot(P / r, dir)), 1e-6))
        : saturate(AtmTransmittance(A, r, dot(P / r, -dir)) / max(AtmTransmittance(A, viewHeight, -dir.y), 1e-6));
    float horizon = 1.0 - smoothstep(0.02, 0.05, abs(dir.y));
    if (horizon <= 0.0)
        return lut;
    float3 depth = 0;
    float dt = s * 0.001 / 8.0;     // km
    [unroll] for (uint k = 0; k < 8; k++)
        depth += SampleAtmMedium(CloudAtmospherePos(dir * (s * (k + 0.5) / 8.0)), A).extinction * dt;
    return lerp(lut, exp(-depth), horizon);
}

// The air between the camera and clouds `depth` metres along dir without the aerial perspective volume: its
// transmittance, and the sky's light scattered below the clouds' base in that direction (AtmosphereSkyBelowClouds,
// × Exposure). The sky probe's clouds, and the sky dome with the fog off; clouds L, T: L × viewT + inFront × (1 − T).
void CloudAirInFront(AtmosphereParams A, float3 dir, float depth, out float3 viewT, out float3 inFront)
{
    viewT = AtmTransmittanceAlong(A, dir, depth);
    inFront = AtmosphereSkyBelowClouds(A, normalize(float3(dir.x, max(dir.y, 0.0), dir.z))) * Exposure;
}

// ── A sky pixel: the sky, the volumetric clouds and the air in front of them (VOLUMETRIC_CLOUDS_PLAN §10, stage 5) ──
// The air in front comes from the aerial perspective volume, out to its end (AtmosphereAPEnd, ~31 km along the ray):
// with the clouds' and the terrain's shadows in it, so the shafts through a broken layer and a mountain's shadow at
// sunset reach over the sky; the same haze in front of a cloud as in front of a hill at its distance. What lies past
// the volume comes from sky-view LUTs that start there (AtmosphereSkyPast). Nothing is subtracted: taking the volume's
// air off the whole ray's LUT went negative near a low sun (the LUT's 32 samples over hundreds of kilometres hold half
// the near haze's glow the volume's slices do) and turned the clouds and the sky there blue. With L, T the clouds' own
// light and transmittance, d their distance along the ray, past the LUTs' light:
//   d ≤ end:  AP(d) + T(0→d) L + T (AP(end) − AP(d) + past)
//   d > end:  AP(end) + T(0→d) L + (f + T (1 − f)) past,   f = (1 − T(end→d)) / (1 − T(end→∞)),
// T(a→b) the air's transmittance by colour (LUT reads; the volume's own is grey, and a far cloud must redden). Past the
// volume the air in front of the cloud is taken as a share of what lies past it by optical depth, which meets the first
// line at d = end; with T = 1 both are AP(end) + past. The volume is read at view depth (distance × the cosine to the
// view's axis). Without the volume or the LUTs (the fog off) as the march composed it until stage 5, from the whole
// ray's LUT (sky): sky T + L viewT + inFront (1 − T).
//   pixel: SV_Position.xy; dir: unit, camera-relative; depth: metres along the ray to the clouds' light
float3 AtmosphereSkyWithClouds(float2 pixel, float3 dir, float3 sky, float3 cloudLight, float cloudT, float depth)
{
    if (AtmosphereIdx == 0)
        return sky * cloudT + cloudLight;
    AtmosphereParams A = LoadAtmosphere();
    if (FogEnabled <= 0.0 || A.SkyViewPastSunLut == 0)
    {
        if (cloudT >= 0.999)
            return sky;
        float3 viewT, inFront;
        CloudAirInFront(A, dir, depth, viewT, inFront);
        return sky * cloudT + cloudLight * viewT + inFront * (1.0 - cloudT);
    }

    float2 uv = pixel / ViewportSize;
    float3 forward = normalize(float3(View._13, View._23, View._33));
    float cosForward = max(dot(dir, forward), 1e-3);
    float end = AtmosphereAPEnd(A);
    float4 apEnd = AtmosphereAerialPerspective(A, uv, end * cosForward);
    float3 past = AtmosphereSkyPast(A, dir);
    if (cloudT >= 0.999)
        return apEnd.rgb + past;

    // The clouds' light through the air by colour (the transmittance LUT; the volume's is grey): a far cloud reddens
    float3 toCloud = AtmTransmittanceAlong(A, dir, depth);
    if (depth <= end)
    {
        float3 inFront = AtmosphereAerialPerspective(A, uv, depth * cosForward).rgb;
        return inFront + toCloud * cloudLight + cloudT * (max(apEnd.rgb - inFront, 0.0) + past);
    }
    float3 toEnd = AtmTransmittanceAlong(A, dir, end);
    float3 endToCloud = saturate(toCloud / max(toEnd, 1e-6));
    float viewHeight = AtmosphereViewHeight(A, CamPos.y);
    bool hitsGround = RaySphereIntersectNearest(float3(0, viewHeight, 0), dir, float3(0, 0, 0), A.BottomRadius) >= 0.0;
    float3 endToSpace = hitsGround ? 0.0 : saturate(AtmTransmittance(A, viewHeight, dir.y) / max(toEnd, 1e-6));
    float3 f = saturate((1.0 - endToCloud) / max(1.0 - endToSpace, 1e-4));
    return apEnd.rgb + toCloud * cloudLight + (f + cloudT * (1.0 - f)) * past;
}

// ── The volumetric clouds at a pixel (stage 6: over the sky, and in front of a summit or anything else) ──
// From the view's half-size targets (CloudTargetIdx: their own light, transmittance; CloudDepthIdx: km to their light;
// CloudSceneDepthIdx: the scene depth each texel was marched to, a checkerboard of each 2×2's nearest and farthest
// surface): the four texels round the pixel, bilinearly, each weighed by how near its surface is to the pixel's, so a
// summit's silhouette takes the clouds marched to it and the sky beside it those marched past it. viewDepth: the pixel's
// (m, 0 = the sky). cloud: (light × Exposure, transmittance; (0, 0, 0, 1) = none); depth: metres along the ray.
void CloudsAt(float2 pixel, float viewDepth, out float4 cloud, out float depth)
{
    cloud = float4(0, 0, 0, 1);
    depth = 0.0;
    if (CloudTargetIdx == 0)
        return;
    Texture2D<float4> clouds = ResourceDescriptorHeap[CloudTargetIdx];
    uint width, height;
    clouds.GetDimensions(width, height);
    float2 coord = pixel * 0.5 - 0.5;
    int2 base = int2(floor(coord));
    float2 f = coord - base;
    int2 hiTexel = int2(width, height) - 1;
    int2 texels[4] = { clamp(base, 0, hiTexel), clamp(base + int2(1, 0), 0, hiTexel),
                       clamp(base + int2(0, 1), 0, hiTexel), clamp(base + int2(1, 1), 0, hiTexel) };
    float4 c[4];
    bool cloudy = false;
    [unroll] for (int k = 0; k < 4; k++)
    {
        c[k] = clouds.Load(int3(texels[k], 0));
        cloudy = cloudy || c[k].a < 0.999;
    }
    if (!cloudy)
        return;                     // most of the ground: no cloud in front of it

    Texture2D<float> scene = ResourceDescriptorHeap[CloudSceneDepthIdx];
    Texture2D<float> depths = ResourceDescriptorHeap[CloudDepthIdx];
    // A texel's clouds behind this pixel's surface count as none, each on its own: a mean of the four depths let the
    // sky's clouds beside a leaf come out in front of it
    float viewCos = 1.0;
    if (viewDepth > 0.0)
    {
        float2 uv = pixel / ViewportSize;
        float4 farClip = mul(float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0, 1), CameraInverse);
        viewCos = max(dot(normalize(farClip.xyz / farClip.w), normalize(float3(View._13, View._23, View._33))), 1e-3);
    }
    // The sky takes only texels marched as sky where it has any (CSCloudSceneDepth sees to that): those marched to a
    // surface end there, and a few per cent of their "no cloud" let the sun's disc through a storm deck
    float s[4];
    bool anySky = false;
    [unroll] for (int k = 0; k < 4; k++)
    {
        s[k] = CloudSceneDepthIdx != 0 ? scene.Load(int3(texels[k], 0)) : 0.0;
        anySky = anySky || s[k] <= 0.0;
    }
    float4 sum = 0;
    float depthSum = 0.0, depthWeight = 0.0, weight = 0.0;
    [unroll] for (int k = 0; k < 4; k++)
    {
        float wb = ((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k >> 1) != 0 ? f.y : 1.0 - f.y);
        float apart = (s[k] <= 0.0 || viewDepth <= 0.0) ? ((s[k] <= 0.0 && viewDepth <= 0.0) ? 0.0 : 10.0)
                    : abs(s[k] - viewDepth) / max(s[k], viewDepth);
        float w = (viewDepth <= 0.0 && anySky && s[k] > 0.0) ? 0.0 : (wb + 1e-3) / (0.05 + apart);
        float d = CloudDepthIdx != 0 ? depths.Load(int3(texels[k], 0)) * 1000.0 : 0.0;
        if (viewDepth > 0.0 && d * viewCos >= viewDepth)
            c[k] = float4(0, 0, 0, 1);
        sum += c[k] * w;
        float wd = w * (1.0 - c[k].a);
        depthSum += d * wd;
        depthWeight += wd;
        weight += w;
    }
    cloud = sum / weight;
    depth = depthSum / max(depthWeight, 1e-6);

    // Lightning: the flash's light on the cloud where its light comes from, per pixel and every frame (the march's
    // targets update a quarter of their texels a frame and keep a history: a 25 ms stroke came out as a few dithered
    // texels, or nothing). Spread through the cloud round the flash as clouds_cs.hlsl's probe march has it.
    if (CloudsOn != 0)
    {
        CloudParams C = LoadClouds();
        if (C.FlashIntensity > 0.0)
        {
            float2 uv = pixel / ViewportSize;
            float4 farClip = mul(float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0, 1), CameraInverse);
            float3 toFlash = normalize(farClip.xyz / farClip.w) * depth - C.FlashPos;
            float d2 = max(dot(toFlash, toFlash), 40000.0);
            float lux = C.FlashIntensity / d2 * CloudThrough(C.FlashSpread * sqrt(d2));
            cloud.rgb += (1.0 - cloud.a) * (C.Albedo * lux / ATM_PI * Exposure) * CLOUD_FLASH_COLOR;
        }
    }
}

// The fog between the camera and a surface (AtmosphereFogAir) with the volumetric clouds in front of it: a summit in a
// deck, a hill seen from above the clouds, water, glass, particles and trails all through this. With the clouds' light
// L, transmittance T and the air's light up to them (the fog at their depth), color × a + rgb becomes
//   T × (surface through the air) + (1 − T) × air up to the clouds + T(0→d) L
// (VOLUMETRIC_CLOUDS_PLAN §14). A cloud the upsampling brings from behind the surface is left out.
float4 AtmosphereFog(float2 pixel, float viewDepth, float skyVisible)
{
    float4 fog = AtmosphereFogAir(pixel, viewDepth, skyVisible);
    if (CloudTargetIdx == 0 || AtmosphereIdx == 0)
        return fog;
    float4 cloud;
    float cloudDepth;
    CloudsAt(pixel, viewDepth, cloud, cloudDepth);
    if (cloud.a >= 0.999)
        return fog;
    float2 uv = pixel / ViewportSize;
    float4 farClip = mul(float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0, 1), CameraInverse);
    float3 dir = normalize(farClip.xyz / farClip.w);
    float cloudView = cloudDepth * max(dot(dir, normalize(float3(View._13, View._23, View._33))), 1e-3);
    if (cloudView >= viewDepth)
        return fog;
    float3 inFront = AtmosphereFogAir(pixel, cloudView, 1.0).rgb;
    float3 toCloud = AtmTransmittanceAlong(LoadAtmosphere(), dir, cloudDepth);
    return float4(cloud.a * fog.rgb + (1.0 - cloud.a) * inFront + toCloud * cloud.rgb, cloud.a * fog.a);
}

// Aerial perspective: the air between the camera and a surface dims the surface's light and adds its own (the haze
// and the ground fog of the preset, lit by the sun and the moon, in the earth's shadow after sunset), and the clouds in
// front of it. AtmosphereFog.
float3 ApplyAerialPerspective(float3 color, float2 pixel, float viewDepth)
{
    float4 fog = AtmosphereFog(pixel, viewDepth, 1.0);
    return color * fog.a + fog.rgb;
}

// The same for a surface that sees skyVisible of the sky
float3 ApplyAerialPerspective(float3 color, float2 pixel, float viewDepth, float skyVisible)
{
    float4 fog = AtmosphereFog(pixel, viewDepth, skyVisible);
    return color * fog.a + fog.rgb;
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

    // Transmittance camera → cloud from two LUT reads (to the top of the atmosphere from either end), and the sky's
    // light scattered in front of the cloud (AtmosphereSkyBelowClouds)
    float3 toTop = AtmTransmittance(A, viewHeight, mu);
    float3 cloudToTop = AtmTransmittance(A, r, dot(P / r, normalize(float3(dir.x, mu, dir.z))));
    c.viewT = saturate(toTop / max(cloudToTop, 1e-6));
    c.inFront = AtmosphereSkyBelowClouds(A, normalize(float3(dir.x, mu, dir.z))) * Exposure;
    return c;
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

    // Toward the horizon the noise fades (it would show its tiling) into what a layer looks like at grazing angles: the
    // clouds behind one another close up. A broken layer leaves the horizon clear; a deck (coverage from ~0.8) runs on
    // to it and sinks into the haze there. Fading a deck to nothing showed the shaded air under it as a dark band
    // between the horizon and the deck's lower edge (stage 5).
    float horizonDensity = smoothstep(0.75, 0.95, CloudCoverage);
    eroded = lerp(horizonDensity, eroded, smoothstep(0.0, 0.22, viewDir.y));

    return saturate(eroded);
}

// How much the sky's clouds hide the sun's disc at a cloud density (GetCloudsAt, SkyThroughClouds' density out):
// 1 − exp(−30 d), so the disc (up to 60000 in the frame) is down to about the cloud's own light where the cloud reads ~3/4
// solid. The sky dome's 2D layer (mesh_skybox.fx).
float CloudSunDiscCover(float density)
{
    return 1.0 - exp(-30.0 * density);
}

// Henyey-Greenstein × 4π: 1 for every angle when g = 0 (an isotropic scatterer)
float HenyeyGreenstein4Pi(float cosTheta, float g)
{
    float g2 = g * g;
    return (1.0 - g2) / pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5);
}

// The two-lobe phase of cloud droplets (× 4π): a strong forward lobe (light passing on toward the eye when looking
// toward the light: the bright rim round the sun) and a weak backward one (ATMOSPHERE_PLAN stage 5)
#define CLOUD_PHASE_FORWARD   0.6
#define CLOUD_PHASE_BACKWARD -0.25
#define CLOUD_PHASE_MIX       0.6     // share of the forward lobe
float CloudPhase4Pi(float cosTheta)
{
    return lerp(HenyeyGreenstein4Pi(cosTheta, CLOUD_PHASE_BACKWARD), HenyeyGreenstein4Pi(cosTheta, CLOUD_PHASE_FORWARD),
        CLOUD_PHASE_MIX);
}

// How much of a light (sun or moon) a cloud sends toward the eye, per unit of what a white surface facing that light
// would (E / π). Two parts:
//   - the light that gets into the cloud past its own density (transmittance: thin edges bright, thick cores dark),
//     scattered by the two-lobe phase where the cloud is thin (single scattering) and ever more evenly where it is
//     thick (light scattered many times forgets where it came from). 0.55: a thin-to-medium cloud 90° from the sun as
//     bright as the hand-tuned shading before stage 5 (its gradient, silver lining and powder terms are gone);
//   - the light the cloud's body diffuses through to its base, CLOUD_DIFFUSE of what the core holds back: a deck seen
//     from below is lit grey-white by the sun above it, not left to the blue sky's ambient (an overcast deck was navy).
#define CLOUD_DIFFUSE 0.15
float CloudDirectShade(float3 viewDir, float3 lightDir, float density, float transmittance)
{
    float single = exp(-density * 6.0);                 // share of the light scattered once: the thin edges
    float phase = lerp(1.0, CloudPhase4Pi(dot(viewDir, lightDir)), single);
    return transmittance * phase * 0.55 + (1.0 - transmittance) * CLOUD_DIFFUSE;
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
