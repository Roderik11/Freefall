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

// Light on a lit particle or ribbon (smoke, rain, snow, a blade's trail) at the camera's height, as a multiple of its
// albedo (E / π, pre-exposed): half the sun's and the moon's light through the air and the clouds' average shade (a
// puff is lit from one side) and the sky's and the ground's ambient averaged, as for the clouds. No shadows: grey by
// day, dark at night. Callers multiply by the SkyVisibility at the particle: smoke in a cave stays dark, since neither
// the sky nor the sun reaches it there (they also scale the haze in front of it by that).

// How much more of a light a particle shows seen at angle cosTheta from the light's way on (1 = looking into the light
// through it) than an even scatterer would: Henyey-Greenstein with anisotropy g, 1 everywhere for g = 0. Droplets throw
// most of the light on, so spray glows with the sun behind it and dims with the sun behind the viewer; 40 % stays even,
// the light scattered several times inside a dense cloud (ParticleEmitter.ForwardScatter, 2026-10-19)
float ParticlePhase(float cosTheta, float g)
{
    float d = 1.0 + g * g - 2.0 * g * cosTheta;
    return 0.4 + 0.6 * (1.0 - g * g) / (d * sqrt(d));
}

//   toEye: from the particle to the camera; g: its forward scattering (0 = the same from every side)
float3 ParticleLight(float3 toEye, float g)
{
    if (AtmosphereIdx == 0)
        return 0;
    AtmosphereParams A = LoadAtmosphere();
    float3 P = float3(0, AtmosphereViewHeight(A, CamPos.y), 0);
    float3 direct = A.SunIlluminance * AtmTransmittanceToLight(A, P, A.SunDirection) * ParticlePhase(dot(toEye, -A.SunDirection), g)
                  + A.MoonIlluminance * AtmTransmittanceToLight(A, P, A.MoonDirection) * ParticlePhase(dot(toEye, -A.MoonDirection), g);
    StructuredBuffer<float4> ambient = ResourceDescriptorHeap[A.Ambient];
    return direct * (A.CloudShade * 0.5 * Exposure / ATM_PI) + (ambient[0].rgb + ambient[1].rgb) * 0.5;
}

float3 ParticleLight() { return ParticleLight(float3(0, 1, 0), 0.0); }

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
float4 AtmosphereFog(float2 pixel, float viewDepth, float skyVisible)
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

// Aerial perspective: the air between the camera and a surface dims the surface's light and adds its own (the haze
// and the ground fog of the preset, lit by the sun and the moon, in the earth's shadow after sunset). AtmosphereFog.
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

// The cloud layer's density (0..1) on the way from a place to a light, as its shadow: before CloudShadowStrength and
// the low-sun fade, the old and the new layer cross-faded while a preset changes the altitude. relPos: camera-relative.
float CloudShadowDensity(SamplerState wrapSampler, float3 relPos, float3 toLight)
{
    Texture3D<float4> noiseLUT = ResourceDescriptorHeap[CloudNoiseLUTIdx];
    float density = CloudShadowDensityAt(noiseLUT, wrapSampler, relPos, toLight, CloudAltitude);
    if (CloudAltitudeBlend < 0.999)
    {
        float densityFrom = CloudShadowDensityAt(noiseLUT, wrapSampler, relPos, toLight, CloudAltitudeFrom);
        density = lerp(densityFrom, density, smoothstep(0.0, 1.0, saturate(CloudAltitudeBlend)));
    }
    return density;
}

// Fraction of the sun's light (0..1) that reaches a surface point through the cloud layer.
// relPos: camera-relative position, toLight: normalized direction toward the sun.
float GetCloudShadow(SamplerState wrapSampler, float3 relPos, float3 toLight)
{
    if (CloudShadowStrength <= 0.0 || CloudNoiseLUTIdx == 0 || toLight.y <= 0.05)
        return 1.0;

    float density = CloudShadowDensity(wrapSampler, relPos, toLight);

    // A low sun stretches the projection toward infinity — fade the shadows out before that
    float lowSunFade = smoothstep(0.05, 0.3, toLight.y);

    return 1.0 - density * CloudShadowStrength * lowSunFade;
}

// ── The clouds in the air (the aerial perspective, the sky-view LUT, the near fog volume) ──

// The cloud layer round the camera, made each frame by atmosphere_cs.hlsl CSCloudDeck (the Ambient buffer's element 4):
//   x, y  the layer's mean density between the air and the sun / the brightest moon over CLOUD_DECK_RADIUS round the
//         camera (the sky-view LUT's air below the layer, which is the same in every direction), at a low light faded
//         into z, w as CloudAirLight fades;
//   z, w  how much the sky's clouds hide the sun / the moon seen from the camera: 1 − exp(−15 × density), as the sky
//         dome hides their discs (mesh_skybox.fx).
// 0 (no layer) where the buffer has no element 4 (a build before 2026-10-19 reading these shaders).
float4 CloudDeck()
{
    if (AtmosphereIdx == 0)
        return 0;
    StructuredBuffer<float4> ambient = ResourceDescriptorHeap[LoadAtmosphere().Ambient];
    return ambient[4];
}

// How much the multiple scattering (Ψ_ms) of the air below the cloud layer is dimmed, 0..1, given the layer's mean density
// toward the light (CloudDeck().x / .y). Ψ_ms is the light of a cloudless sky and the lit ground round a place. Under a
// deck all of it has come through the deck, which lets 1 − CloudShadowStrength through (as to the ground), and the light
// the deck gives itself is CloudDiffusedLight; without the dimming the haze under a deck outshone the deck (a dark band
// under it on the Overcast horizon). A broken layer's gaps still show the blue sky that makes most of Ψ_ms, hence the
// square: Clear Day (mean 0.15-0.25) loses 2-5 %, a deck (1) 80 %.
// Tried first: the sky probe's mean radiance against Ψ_ms × the illuminance; without a cloud it read 0.89-1.03 with the
// time of day, so it dimmed a cloudless noon by 10 %.
float CloudSkyDimming(float meanDensity)
{
    return CloudShadowStrength > 0.0 ? CloudShadowStrength * meanDensity * meanDensity : 0.0;
}

// What the cloud layer lets through to a place in the air toward a light, as shares of its illuminance:
//   x  the beam, which the air scatters with its phases (the Mie forward lobe is the glow round the sun): 1 − density,
//      what passes where the layer is open or thin;
//   y  the light the layer has diffused, (1 − CloudShadowStrength) × density, which the air scatters evenly: under a deck
//      the sun still lights the air, but from the whole cloud base, not from its place in the sky (CloudDiffusedLight);
//   z  the share of the multiple scattering (Ψ_ms) left below the layer: 1 − skyDimming (CloudSkyDimming).
// A surface takes x + y as one shadow (GetCloudShadow). In the air the split matters: the 20 % a deck lets through,
// scattered with the forward lobe, still drew the sun's glow through a thunderstorm deck.
// A low light stretches the shadow's pattern toward infinity (GetCloudShadow fades it out below 17°); here it fades into
// deckToLight instead (CloudDeck().z / .w: the clouds in front of that light as the sky draws them), so the air under a
// deck stays shaded at sunset while a broken layer, whose horizon is clear, lets a low sun through. Above the layer's
// plane (CloudAltitude, ±100 m) nothing is in the way. relPos: camera-relative.
float3 CloudAirLight(SamplerState wrapSampler, float3 relPos, float3 toLight, float deckToLight, float skyDimming)
{
    if (CloudShadowStrength <= 0.0 || CloudNoiseLUTIdx == 0)
        return float3(1.0, 0.0, 1.0);
    float below = 1.0 - smoothstep(CloudAltitude - 100.0, CloudAltitude + 100.0, CamPos.y + relPos.y);
    if (below <= 0.0)
        return float3(1.0, 0.0, 1.0);
    float fade = smoothstep(0.05, 0.3, toLight.y);
    float density = deckToLight;
    if (fade > 0.0)
        density = lerp(deckToLight, CloudShadowDensity(wrapSampler, relPos, toLight), fade);
    density *= below;
    return float3(1.0 - density, (1.0 - CloudShadowStrength) * density, 1.0 - skyDimming * below);
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
