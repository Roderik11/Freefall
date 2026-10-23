// The near fog volume (god rays, ATMOSPHERE_PLAN): the atmosphere's own air (the haze and the preset's ground fog,
// atmosphere_common.hlsli SampleAtmMedium) over the first FogRange metres of view depth, lit per froxel with what the
// aerial perspective volume cannot know: the sun's and the moon's shadows (the cascades, the clouds, the terrain) and the
// sky's light only where the sky is seen (no glowing air in a room or a cave). Made every frame by Graphics/VolumetricFog.cs
// after the shadow maps and the sky captures, before anything draws fog:
//   CSFogInject     one thread per particle pool slot: Volumetric particles (smoke) as density in the froxels they cover
//   CSFogScatter    one thread per froxel: the light the air there scatters toward the camera and its extinction, per km,
//                   at a place in the froxel jittered each frame and blended with the froxel's reprojected history
//                   (rgb × Exposure, a = σe); the sun, the moon and the point lights that reach it (light_clusters.hlsli).
//                   Beside it the froxel lamps, for the far rain and snow (precipitation_layers_cs.hlsl): the point
//                   lights' share of that light (rgb) and how much of the open sky's, the sun's and the moon's light gets
//                   to the place past walls, roofs, crowns, shadows and the terrain (a)
//   CSFogIntegrate  one thread per column, front to back: the light the air adds between the camera and the far side of
//                   each slice (× Exposure, rgb) and the transmittance (a)
// Read by sky_common.fx AtmosphereFog through SceneConstants.FogVolumeIdx / FogVolumeRange.
// Grid: FOG_TILE pixels square (sky_common.fx), FOG_SLICES slices; slice s ends at view depth ((s + 1) / FOG_SLICES)² × range.
// Docs: .agent/knowledge/atmospheric_effects/physical_atmosphere.md §13.

#pragma kernel CSFogInject
#pragma kernel CSFogScatter
#pragma kernel CSFogIntegrate

cbuffer PushConstants : register(b3)
{
    uint ScatterUAVIdx;         // this frame's lit froxels (written)
    uint HistoryTexIdx;         // last frame's lit froxels; 0 = none
    uint ScatterTexIdx;         // this frame's lit froxels (read by the integration)
    uint IntegratedUAVIdx;      // the volume the passes read
    uint ShadowMapIdx;          // the directional light's cascades (Texture2DArray); 0 = none this frame
    uint CascadeBufferIdx;      // StructuredBuffer<CascadeData>: absolute world → light clip, split distances
    uint DensityUAVIdx;         // RWByteAddressBuffer, 3 uints a froxel: Volumetric particles' σe, σs, σs·g (per km × 16); 0 = none
    uint ParticleCoreIdx;       // the particle pool as the last simulation left it (ParticleSystem.VolumeSources)
    uint ParticleVisualIdx;
    uint ParticleEmittersIdx;
    uint ParticleChunkMapIdx;
    uint ParticleSlots;         // slots to walk
    uint LampsUAVIdx;           // this frame's froxel lamps (written; 0 = none): rgb = the point lights' light per km ×
                                // Exposure, a = the share of the open sky's, the sun's and the moon's light that gets here
    uint LampsHistoryIdx;       // last frame's froxel lamps; 0 = none
};

cbuffer Params : register(b4)
{
    row_major float4x4 PrevCameraRelativeVP;    // last frame's view-projection without translation
    float3 CameraMove;                          // this frame's camera position minus last frame's
    float HistoryWeight;                        // share of the reprojected history kept (0 = none)
    float3 Jitter;                              // this frame's place in the froxel, 0..1 each way
    float FogRange;                             // metres of view depth the grid covers
    float3 ShadowLightDir;                      // toward the body the cascades were drawn for
    uint CascadeCount;
    uint3 GridSize;                             // froxels across, down, slices
    float ExposureRatio;                        // this frame's exposure / last frame's: the history is pre-exposed
    float2 TileUV;                              // a froxel's width and height as a share of the screen
    uint SkyViewPhase;                          // which 4 of the 16 sky captures this frame reads (0..3)
    float _fogPad;
};

#include "common.fx"
#include "sky_common.fx"
#include "light_clusters.hlsli"
#include "particle_common.hlsli"

#define FOG_SLICES 64           // VolumetricFog.Slices
// The air closer to a light than this (m²) takes the light it gets there: a froxel's sample can sit in the flame itself,
// where 1 / d² would make single samples flare (the flame or the lamp glass hides that air anyway)
#define FOG_POINT_LIGHT_MIN_D2 0.25

SamplerState FogWrapSampler : register(s0);                   // linear wrap (the cloud noise)
SamplerComparisonState FogShadowSampler : register(s3);       // comparison, less-equal (standard Z shadow maps)

// Camera-relative position of a place in the grid (cell = froxel index + the offset in it) and its view depth
float3 FroxelPosition(float3 cell, out float viewDepth)
{
    float2 uv = cell.xy * TileUV;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 nearClip = mul(float4(ndc, 1.0, 1.0), CameraInverse);     // reverse-Z: the near plane is z = 1
    float3 dir = normalize(nearClip.xyz / nearClip.w);
    float3 forward = normalize(float3(View._13, View._23, View._33));
    float w = cell.z / FOG_SLICES;
    viewDepth = w * w * FogRange;
    return dir * (viewDepth / max(dot(dir, forward), 0.05));
}

// ── Volumetric particles (ParticleEmitter.Volumetric): their density, injected per froxel ──
#define FOG_DENSITY_SCALE 16.0      // the density buffer holds coefficients per km × this, as uints (atomic adds)
#define FOG_INJECT_MAX_CELLS 16     // froxels a puff may cover each way (a puff right by the camera is cut down)

uint FroxelIndex(uint3 id)
{
    return (id.z * GridSize.y + id.y) * GridSize.x + id.x;
}

// A froxel's volume (m³) at its view depth: its tile's width and height there times its slice's depth
float FroxelVolume(uint slice, float viewDepth, float2 projScale)
{
    float2 size = TileUV * 2.0 * viewDepth / projScale;
    return size.x * size.y * FogRange * (2.0 * slice + 1.0) / (FOG_SLICES * FOG_SLICES);
}

// One thread per particle pool slot. A Volumetric particle is a puff of radius r (half its size) with density
// σ0 (1 − (d/r)²)², whose optical depth through the centre is the −ln(1 − alpha) its sprite hides: σ0 = τ 15 / (16 r),
// σ0 r³ 32π/105 in all. That total is shared out over the froxels the puff covers by the kernel at their centres and
// their volumes (so it holds however coarse the froxels are; a puff between froxel centres goes into the one it is in),
// × FogVolumeShare: past the volume's range the sprites draw it. σs = σe × its albedo (grey), with its ForwardScatter.
// The pool is the last simulation's (a frame old: it runs in the transparent pass).
[numthreads(64, 1, 1)]
void CSFogInject(uint3 dtid : SV_DispatchThreadID)
{
    uint slot = dtid.x;
    if (slot >= ParticleSlots || DensityUAVIdx == 0)
        return;
    StructuredBuffer<uint> chunkMap = ResourceDescriptorHeap[ParticleChunkMapIdx];
    uint row = chunkMap[slot >> PARTICLE_CHUNK_SHIFT];
    if (row == PARTICLE_NO_EMITTER)
        return;
    StructuredBuffer<ParticleEmitter> emitters = ResourceDescriptorHeap[ParticleEmittersIdx];
    ParticleEmitter e = emitters[row];
    if ((e.Flags & PARTICLE_FLAG_VOLUME) == 0 || (e.Flags & PARTICLE_FLAG_RESET) != 0)
        return;
    StructuredBuffer<ParticleCore> cores = ResourceDescriptorHeap[ParticleCoreIdx];
    ParticleCore core = cores[slot];
    if (core.Lifetime <= 0.0 || core.Age >= core.Lifetime)
        return;
    StructuredBuffer<ParticleVisual> visuals = ResourceDescriptorHeap[ParticleVisualIdx];
    ParticleVisual vis = visuals[slot];

    // Colour and size as the sprite would have them (particle_draw.fx)
    float t = saturate(core.Age / max(core.Lifetime, 0.001));
    float4 colour = lerp(e.ColorStart, e.ColorEnd, t);
    colour.rgb *= e.Intensity;
    if (e.FadeInRate > 0.0)
        colour *= saturate(t * e.FadeInRate);
    float alpha = saturate(colour.a);
    if (alpha <= 1e-3)
        return;
    float r = max(0.5 * lerp(e.SizeStartEnd.x, e.SizeStartEnd.y, t) * vis.SizeScale, 0.01);

    float3 rel = core.Position - CamPos;
    float4 clip = mul(float4(rel, 1.0), CameraRelativeVP);
    float z = clip.w;
    float share = FogVolumeShare(z);
    if (share <= 0.0 || z <= 0.05)
        return;

    float tau = -log(max(1.0 - alpha, 1e-3));
    float total = share * tau * r * r * (32.0 * ATM_PI * 15.0 / (105.0 * 16.0));    // m² (σ0 r³ 32π/105)
    float albedo = saturate(dot(colour.rgb, float3(0.2126, 0.7152, 0.0722)) / alpha);

    // Where its density goes: the kernel never narrower than 1.5 froxels along any of the view's axes (a froxel's width
    // and height there on screen, its slice's depth along the view), so a puff smaller than that glides from froxel to
    // froxel as it drifts instead of jumping (small puffs and every puff along the view, where slices are ~1.4 m deep at
    // 15 m: the plume read as blocky in motion). The total stays its own.
    float2 projScale = float2(length(float3(CameraRelativeVP._11, CameraRelativeVP._21, CameraRelativeVP._31)),
                              length(float3(CameraRelativeVP._12, CameraRelativeVP._22, CameraRelativeVP._32)));
    float3 right = float3(CameraRelativeVP._11, CameraRelativeVP._21, CameraRelativeVP._31) / projScale.x;
    float3 up = float3(CameraRelativeVP._12, CameraRelativeVP._22, CameraRelativeVP._32) / projScale.y;
    float3 forward = normalize(float3(CameraRelativeVP._14, CameraRelativeVP._24, CameraRelativeVP._34));
    float2 cellSize = TileUV * 2.0 * z / projScale;
    float cellDepth = 2.0 * sqrt(z * FogRange) / FOG_SLICES;
    // ... and never wider than the froxels a puff may cover (FOG_INJECT_MAX_CELLS): a big old puff by the camera is
    // drawn smaller and denser (its total stays), round. Cut off at the cap's box instead, it showed as a square block
    // (user, 2026-10-20, a bonfire's 3 m puffs a few metres away). Along the view it reaches at most half way to the
    // camera; across, it is held to the cap in froxels as wide as they are at its nearest depth, where it covers the most
    // of the screen.
    const float maxCells = FOG_INJECT_MAX_CELLS / 2 - 1;
    float3 radii;
    radii.z = min(min(max(r, 1.5 * cellDepth), maxCells * cellDepth), 0.5 * z);
    float nearShare = (z - radii.z) / z;
    radii.xy = min(max(float2(r, r), 1.5 * cellSize), maxCells * cellSize * nearShare);
    float3 invRadii = 1.0 / radii;

    // The froxels round it: on screen its half extent at its nearest depth, in depth its slices. Not clamped to the
    // grid: a puff at or past the screen's edge is weighed over all of its footprint (cells off the screen have
    // positions too) and leaves only its on-screen share. Clamped, a puff off the screen fell back whole into the edge
    // column, a bright band that panning then smeared across the view.
    float2 uv = clip.xy / z * float2(0.5, -0.5) + 0.5;
    float2 halfUV = 0.5 * radii.xy * projScale / max(z - radii.z, 0.1);
    int3 hi = int3(GridSize) - 1;
    int3 centre = int3(int2(floor(uv / TileUV)), (int)floor(sqrt(saturate(z / FogRange)) * FOG_SLICES));
    int3 lo3 = int3(int2(floor((uv - halfUV) / TileUV)),
                    (int)floor(sqrt(saturate(max(z - radii.z, 0.0) / FogRange)) * FOG_SLICES));
    int3 hi3 = int3(int2(floor((uv + halfUV) / TileUV)), (int)floor(sqrt((z + radii.z) / FogRange) * FOG_SLICES));
    lo3 = max(lo3, centre - FOG_INJECT_MAX_CELLS / 2);
    hi3 = min(hi3, centre + FOG_INJECT_MAX_CELLS / 2 - 1);
    // On the grid: what it may write
    int3 lo3In = max(lo3, 0), hi3In = min(hi3, hi);
    if (any(hi3In < lo3In))
        return;

    float weighted = 0.0;   // Σ kernel × volume, over the whole footprint
    for (int s = lo3.z; s <= hi3.z; s++)
        for (int y = lo3.y; y <= hi3.y; y++)
            for (int x = lo3.x; x <= hi3.x; x++)
            {
                float depthThere;
                float3 d = FroxelPosition(float3(x, y, s) + 0.5, depthThere) - rel;
                float3 q = float3(dot(d, right), dot(d, up), dot(d, forward)) * invRadii;
                float u2 = dot(q, q);
                if (u2 < 1.0)
                    weighted += (1.0 - u2) * (1.0 - u2) * FroxelVolume((uint)s, depthThere, projScale);
            }

    RWByteAddressBuffer density = ResourceDescriptorHeap[DensityUAVIdx];
    // per km × FOG_DENSITY_SCALE
    float toUnits = 1000.0 * FOG_DENSITY_SCALE;
    if (weighted <= 0.0)
    {
        // Between froxel centres (only when it sits on the grid; off it, nothing of it shows)
        if (any(centre < 0) || any(centre > hi))
            return;
        float depthThere;
        FroxelPosition(float3(centre) + 0.5, depthThere);
        float sigma = total / FroxelVolume((uint)centre.z, depthThere, projScale) * toUnits;
        uint at = FroxelIndex(uint3(centre)) * 12;
        density.InterlockedAdd(at, (uint)(sigma + 0.5));
        density.InterlockedAdd(at + 4, (uint)(sigma * albedo + 0.5));
        density.InterlockedAdd(at + 8, (uint)(sigma * albedo * e.ForwardScatter + 0.5));
        return;
    }
    float perWeight = total / weighted * toUnits;
    for (int s2 = lo3In.z; s2 <= hi3In.z; s2++)
        for (int y2 = lo3In.y; y2 <= hi3In.y; y2++)
            for (int x2 = lo3In.x; x2 <= hi3In.x; x2++)
            {
                float depthThere;
                float3 d = FroxelPosition(float3(x2, y2, s2) + 0.5, depthThere) - rel;
                float3 q = float3(dot(d, right), dot(d, up), dot(d, forward)) * invRadii;
                float u2 = dot(q, q);
                if (u2 >= 1.0)
                    continue;
                float sigma = (1.0 - u2) * (1.0 - u2) * perWeight;
                uint at = FroxelIndex(uint3(x2, y2, s2)) * 12;
                density.InterlockedAdd(at, (uint)(sigma + 0.5));
                density.InterlockedAdd(at + 4, (uint)(sigma * albedo + 0.5));
                density.InterlockedAdd(at + 8, (uint)(sigma * albedo * e.ForwardScatter + 0.5));
            }
}
// The directional light's shadow at a place in the air: the first cascade that holds it, one bilinear comparison
// (the jitter and the history do the filtering). 1 outside every cascade and without shadow maps.
float CascadeShadow(float3 worldPos, float viewDepth)
{
    if (ShadowMapIdx == 0 || CascadeBufferIdx == 0)
        return 1.0;
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferIdx];
    Texture2DArray<float> shadowMap = ResourceDescriptorHeap[ShadowMapIdx];
    for (uint c = 0; c < CascadeCount; c++)
    {
        CascadeData cd = cascades[c];
        if (viewDepth > cd.SplitDistances.y)
            continue;
        float4 p = mul(float4(worldPos, 1.0), cd.VP);
        float2 uv = p.xy * float2(0.5, -0.5) + 0.5;
        if (any(uv < 0.0) || any(uv > 1.0))
            continue;
        // 0.3 m toward the light: the air just over a lit surface stays lit
        float zPerMetre = length(float3(cd.VP._13, cd.VP._23, cd.VP._33));
        return shadowMap.SampleCmpLevelZero(FogShadowSampler, float3(uv, c), p.z - 0.3 * zPerMetre);
    }
    return 1.0;
}

// How much of the sky's light reaches a place in the air past the meshes: 4 of the 16 sky captures, a different 4 each
// frame (the history averages them), seen from the place as from an up-facing surface. 1 without captures.
float FogMeshSkyVisibility(AtmosphereParams A, float3 worldPos)
{
    if (A.SkyCaptureViews == 0 || A.SkyCaptureDepth == 0 || A.SkyCaptureCount != SKY_CAPTURE_VIEWS)
        return 1.0;
    StructuredBuffer<SkyCaptureView> views = ResourceDescriptorHeap[A.SkyCaptureViews];
    Texture2DArray<float> depths = ResourceDescriptorHeap[A.SkyCaptureDepth];
    float width, height, slices;
    depths.GetDimensions(width, height, slices);
    float lit = 0, total = 0;
    [unroll] for (uint k = 0; k < 4; k++)
    {
        uint i = k * 4 + SkyViewPhase;
        lit += SkyCaptureViewLight(views[i], i, depths, float2(width, height), AtmosphereLinearClamp, worldPos,
                                   float3(0, 1, 0), total);
    }
    return total > 0.0 ? lit / total : 1.0;
}

// One body's light scattered toward the camera per km of air (× its illuminance): single scattering through the air
// above (0 in the earth's shadow) and the cloud layer (sky_common.fx CloudAirLight, as the aerial perspective takes it):
// its beam where the shadows let it through, with the Rayleigh and Mie phases, and what it diffuses, scattered evenly
// from the whole cloud base where the sky is seen; and the multiple scattering (the sky's light, in effect, dimmed under
// a deck to the light the cloud base gives: CloudAirLight .z) there too
// Volumetric particles in a froxel (CSFogInject): their scattering per km and its Henyey-Greenstein g. medium already
// holds their scattering and extinction (CSFogScatter adds them); their single scattering takes their own phase.
struct FroxelSmoke
{
    float Scattering;
    float G;
    float ShadowOffset;     // where this froxel's smoke shadow samples start, 0..1 of a step (changes every frame)
};

float SmokePhase(float g, float cosTheta)
{
    float d = 1.0 + g * g - 2.0 * g * cosTheta;
    return (1.0 - g * g) / (4.0 * ATM_PI * d * sqrt(d));
}

// The smoke's own shadow: the light reaching smoke has come through the smoke between it and the light. A fire lights
// the underside of its column; the column above stands in the shadow of the smoke below it, and by day the side away from
// the sun is darker. Transmittance along ray (from a camera-relative place toward a light, as far as the ray is long):
// SMOKE_SHADOW_STEPS point samples of the injected density, from an offset that changes per froxel and per frame (the
// history averages them). Only smoke in the grid shadows: a column's part off the screen casts none.
#define SMOKE_SHADOW_STEPS 8
#define SMOKE_SHADOW_RANGE 8.0      // metres looked through toward a light: a tall column's height

float SmokeShadow(FroxelSmoke smoke, float3 relPos, float3 ray)
{
    if (smoke.Scattering <= 0.0)
        return 1.0;
    RWByteAddressBuffer density = ResourceDescriptorHeap[DensityUAVIdx];
    float3 stepRay = ray * (1.0 / SMOKE_SHADOW_STEPS);
    float sigmaSum = 0;
    [loop] for (uint i = 0; i < SMOKE_SHADOW_STEPS; i++)
    {
        float4 clip = mul(float4(relPos + stepRay * (i + smoke.ShadowOffset), 1.0), CameraRelativeVP);
        if (clip.w <= 1e-3)
            continue;
        float3 cell = float3((clip.xy / clip.w * float2(0.5, -0.5) + 0.5) / TileUV, sqrt(clip.w / FogRange) * FOG_SLICES);
        if (any(cell < 0.0) || any(cell >= float3(GridSize)))
            continue;
        sigmaSum += density.Load(FroxelIndex(uint3(cell)) * 12);
    }
    // per km × FOG_DENSITY_SCALE, over steps of length(stepRay) metres
    return exp(-sigmaSum * (length(stepRay) * 0.001 / FOG_DENSITY_SCALE));
}

// `openLight`: the same light where nothing but the clouds stands between the place and the body and the whole sky is seen
// (no walls, roofs, crowns, shadows or terrain): what the place's light is a share of
float3 FroxelBodyLight(AtmosphereParams A, AtmMedium medium, FroxelSmoke smoke, bool moon, float3 posKm, float3 worldPos,
                       float3 relPos, float3 V, float viewDepth, float3 lightDir, float open, float sky, float skyDimming,
                       out float3 openLight)
{
    float3 toLight = AtmTransmittanceToLight(A, posKm, lightDir);
    float3 diffusedTint;
    float3 cloud = CloudAirLight(FogWrapSampler, relPos, lightDir, skyDimming, diffusedTint);
    float visible = open * cloud.x;
    if (visible > 0.0 && any(toLight > 0.0))
    {
        if (dot(lightDir, ShadowLightDir) > 0.999)
            visible *= CascadeShadow(worldPos, viewDepth);
        visible *= AtmTerrainLightVisibility(A, worldPos, moon);
        if (visible > 0.0)
            visible *= SmokeShadow(smoke, relPos, lightDir * SMOKE_SHADOW_RANGE);
    }
    float cosTheta = dot(V, lightDir);
    float lightZenithCos = dot(normalize(posKm), lightDir);
    float3 single = medium.scatteringMie * AtmMiePhase(A.MieG, cosTheta) + medium.scatteringRay * AtmRayleighPhase(cosTheta)
                  + smoke.Scattering * SmokePhase(smoke.G, cosTheta);
    float3 diffusedOpen = medium.scattering * diffusedTint * CloudDiffusedLight(cloud.y, lightZenithCos);
    float3 multiple = AtmMultipleScattering(A, posKm, lightZenithCos) * (medium.scattering * cloud.z);
    openLight = toLight * (cloud.x * single + diffusedOpen) + multiple;
    return toLight * (visible * single + diffusedOpen * sky) + multiple * sky;
}

// The point lights' light scattered toward the camera per km of air (lux × σs × phase, as a body's single scattering):
// the lights that reach the place (light_clusters.hlsli), each through its shadow where it has one this frame. A light
// without a shadow lights the air through walls; the halo of a lamp in a room shows outside only as far as its range.
float3 FroxelPointLights(AtmosphereParams A, AtmMedium medium, FroxelSmoke smoke, float3 relPos, float3 V, float2 uv,
                         float viewDepth)
{
    if (LightClustersIdx == 0 || ClusterLightsIdx == 0)
        return 0;
    ByteAddressBuffer clusters = ResourceDescriptorHeap[LightClustersIdx];
    StructuredBuffer<PointLightData> lights = ResourceDescriptorHeap[ClusterLightsIdx];
    LightClusterWalk walk;
    if (!LightClusterBegin(clusters, uv, viewDepth, walk))
        return 0;

    float3 sum = 0;
    uint s;
    [loop] while (LightClusterNext(clusters, walk, s))
    {
        PointLightData light = lights[s];
        float3 toLight = light.Position - relPos;
        float3 E = PointLightIlluminance(light, toLight, FOG_POINT_LIGHT_MIN_D2);
        if (all(E <= 0.0))
            continue;
        float shadow = PointLightShadowAt(clusters, light, relPos, FogShadowSampler);
        if (shadow <= 0.0)
            continue;
        float d2 = dot(toLight, toLight);
        shadow *= SmokeShadow(smoke, relPos, toLight * min(1.0, SMOKE_SHADOW_RANGE * rsqrt(max(d2, 1e-6))));
        float cosTheta = dot(V, toLight) * rsqrt(max(d2, 1e-6));
        float3 phase = medium.scatteringMie * AtmMiePhase(A.MieG, cosTheta) + medium.scatteringRay * AtmRayleighPhase(cosTheta)
                     + smoke.Scattering * SmokePhase(smoke.G, cosTheta);
        sum += E * (shadow * phase);
    }
    return sum;
}

[numthreads(8, 8, 1)]
void CSFogScatter(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= GridSize))
        return;
    RWTexture3D<float4> output = ResourceDescriptorHeap[ScatterUAVIdx];
    if (AtmosphereIdx == 0)
    {
        output[id] = 0;
        if (LampsUAVIdx != 0)
        {
            RWTexture3D<float4> lampsOut = ResourceDescriptorHeap[LampsUAVIdx];
            lampsOut[id] = float4(0, 0, 0, 1);
        }
        return;
    }
    AtmosphereParams A = LoadAtmosphere();

    float viewDepth;
    float3 relPos = FroxelPosition(float3(id) + Jitter, viewDepth);
    float3 worldPos = CamPos + relPos;
    float3 V = normalize(relPos);
    // In the atmosphere's frame (km, the camera over the planet's centre, the ground flat as the world's); kept off the
    // ground, where the air is still air
    float3 posKm = float3(0, AtmosphereViewHeight(A, CamPos.y), 0) + relPos * 0.001;
    posKm.y = max(posKm.y, A.BottomRadius + 0.001);
    AtmMedium medium = SampleAtmMedium(posKm, A);

    // Volumetric particles (smoke) injected into this froxel: grey, with their own phase
    FroxelSmoke smoke = (FroxelSmoke)0;
    if (DensityUAVIdx != 0)
    {
        RWByteAddressBuffer density = ResourceDescriptorHeap[DensityUAVIdx];
        uint3 d = density.Load3(FroxelIndex(id) * 12);
        float sigmaE = d.x * (1.0 / FOG_DENSITY_SCALE);
        smoke.Scattering = d.y * (1.0 / FOG_DENSITY_SCALE);
        smoke.G = smoke.Scattering > 0.0 ? d.z * (1.0 / FOG_DENSITY_SCALE) / smoke.Scattering : 0.0;
        smoke.ShadowOffset = frac(InterleavedGradientNoise(float2(id.xy) + float2(5.588238 * id.z, 0)) + Jitter.z);
        medium.scattering += smoke.Scattering;
        medium.extinction += sigmaE;
    }

    // Under the terrain (in a cave, inside a hill) neither the sun nor the sky; past the meshes (walls, roofs, crowns)
    // only the share of the sky that they leave, never less than the light bouncing in (SkyCaptureBounce)
    TerrainCover terrain = LoadTerrainCover();
    float open = TerrainOpenAt(terrain, AtmosphereLinearClamp, worldPos, float3(0, 1, 0), TerrainTolerance(length(relPos)));
    float sky = open > 0.0 ? open * max(FogMeshSkyVisibility(A, worldPos), A.SkyCaptureBounce) : 0.0;

    float skyDimming = CloudSkyDimming(CloudDeck());      // Ψ_ms under the clouds round the camera
    float3 sunOpen, moonOpen = 0;
    float3 L = A.SunIlluminance * FroxelBodyLight(A, medium, smoke, false, posKm, worldPos, relPos, V, viewDepth,
                                                  A.SunDirection, open, sky, skyDimming, sunOpen);
    float3 openL = A.SunIlluminance * sunOpen;
    if (any(A.MoonIlluminance > 0))
    {
        L += A.MoonIlluminance * FroxelBodyLight(A, medium, smoke, true, posKm, worldPos, relPos, V, viewDepth,
                                                 A.MoonDirection, open, sky, skyDimming, moonOpen);
        openL += A.MoonIlluminance * moonOpen;
    }
    float3 lamps = FroxelPointLights(A, medium, smoke, relPos, V, (float2(id.xy) + Jitter.xy) * TileUV, viewDepth);
    // The far rain and snow's light (precipitation_layers_cs.hlsl): the lamps', and the share of the open sky's, the
    // sun's and the moon's light that gets here
    float openLum = dot(openL, float3(0.2126, 0.7152, 0.0722));
    float share = openLum > 0.0 ? saturate(dot(L, float3(0.2126, 0.7152, 0.0722)) / openLum) : 1.0;
    float4 currentLamps = float4(min(lamps * Exposure, 6e4), share);
    L += lamps;
    // Within half floats: dense fog right by a big fire would pass 65504 and turn the history to inf
    float4 current = float4(min(L * Exposure, 6e4), dot(medium.extinction, 1.0 / 3.0));

    // Last frame's light at the froxel's centre: the grid moves with the camera, so the centre is found in last frame's
    // grid; outside it (or in the first frame) this frame's sample stands alone
    if (HistoryTexIdx != 0 && HistoryWeight > 0.0)
    {
        float centreDepth;
        float3 centre = FroxelPosition(float3(id) + 0.5, centreDepth);
        float4 clip = mul(float4(centre + CameraMove, 1.0), PrevCameraRelativeVP);
        if (clip.w > 1e-3)
        {
            float2 uv = clip.xy / clip.w * float2(0.5, -0.5) + 0.5;
            float3 uvw = float3(uv / (TileUV * float2(GridSize.xy)), sqrt(clip.w / FogRange));
            if (all(uvw >= 0.0) && all(uvw <= 1.0))
            {
                Texture3D<float4> history = ResourceDescriptorHeap[HistoryTexIdx];
                float4 h = history.SampleLevel(AtmosphereLinearClamp, uvw, 0);
                h.rgb *= ExposureRatio;
                if (all(isfinite(h)))
                    current = lerp(current, h, HistoryWeight);
                if (LampsHistoryIdx != 0)
                {
                    Texture3D<float4> lampsHistory = ResourceDescriptorHeap[LampsHistoryIdx];
                    float4 hl = lampsHistory.SampleLevel(AtmosphereLinearClamp, uvw, 0);
                    hl.rgb *= ExposureRatio;
                    if (all(isfinite(hl)))
                        currentLamps = lerp(currentLamps, hl, HistoryWeight);
                }
            }
        }
    }
    output[id] = current;
    if (LampsUAVIdx != 0)
    {
        RWTexture3D<float4> lampsOut = ResourceDescriptorHeap[LampsUAVIdx];
        lampsOut[id] = currentLamps;
    }
}

[numthreads(8, 8, 1)]
void CSFogIntegrate(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= GridSize.xy))
        return;
    Texture3D<float4> scatter = ResourceDescriptorHeap[ScatterTexIdx];
    RWTexture3D<float4> output = ResourceDescriptorHeap[IntegratedUAVIdx];

    // km of the ray through the column's centre per metre of view depth
    float unused;
    float3 dir = normalize(FroxelPosition(float3(id.xy + 0.5, FOG_SLICES), unused));
    float kmPerMetre = 0.001 / max(dot(dir, normalize(float3(View._13, View._23, View._33))), 0.05);

    float3 light = 0;
    float transmittance = 1.0;
    float depth = 0.0;
    for (uint s = 0; s < GridSize.z; s++)
    {
        float w = (s + 1.0) / FOG_SLICES;
        float nextDepth = w * w * FogRange;
        float lengthKm = (nextDepth - depth) * kmPerMetre;
        depth = nextDepth;

        // Energy-conserving over the slice (Hillaire 2015): ∫ S e^(−σt) dt = S (1 − e^(−σℓ)) / σ, written so a thin
        // slice of clear air keeps its light in float precision
        float4 f = scatter.Load(int4(id.xy, s, 0));
        float x = f.a * lengthKm;
        float through = exp(-x);
        float share = x > 1e-4 ? (1.0 - through) / x : 1.0 - 0.5 * x;
        light += transmittance * f.rgb * lengthKm * share;
        transmittance *= through;
        output[uint3(id.xy, s)] = float4(light, transmittance);
    }
}
