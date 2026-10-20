// The near fog volume (god rays, ATMOSPHERE_PLAN): the atmosphere's own air (the haze and the preset's ground fog,
// atmosphere_common.hlsli SampleAtmMedium) over the first FogRange metres of view depth, lit per froxel with what the
// aerial perspective volume cannot know: the sun's and the moon's shadows (the cascades, the clouds, the terrain) and the
// sky's light only where the sky is seen (no glowing air in a room or a cave). Made every frame by Graphics/VolumetricFog.cs
// after the shadow maps and the sky captures, before anything draws fog:
//   CSFogScatter    one thread per froxel: the light the air there scatters toward the camera and its extinction, per km,
//                   at a place in the froxel jittered each frame and blended with the froxel's reprojected history
//                   (rgb × Exposure, a = σe); the sun, the moon and the point lights that reach it (light_clusters.hlsli)
//   CSFogIntegrate  one thread per column, front to back: the light the air adds between the camera and the far side of
//                   each slice (× Exposure, rgb) and the transmittance (a)
// Read by sky_common.fx AtmosphereFog through SceneConstants.FogVolumeIdx / FogVolumeRange.
// Grid: FOG_TILE pixels square (sky_common.fx), FOG_SLICES slices; slice s ends at view depth ((s + 1) / FOG_SLICES)² × range.
// Docs: .agent/knowledge/atmospheric_effects/physical_atmosphere.md §13.

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
float3 FroxelBodyLight(AtmosphereParams A, AtmMedium medium, bool moon, float3 posKm, float3 worldPos,
                       float3 relPos, float3 V, float viewDepth, float3 lightDir, float open, float sky, float skyDimming)
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
    }
    float cosTheta = dot(V, lightDir);
    float lightZenithCos = dot(normalize(posKm), lightDir);
    float3 single = medium.scatteringMie * AtmMiePhase(A.MieG, cosTheta) + medium.scatteringRay * AtmRayleighPhase(cosTheta);
    float3 diffused = medium.scattering * diffusedTint * (CloudDiffusedLight(cloud.y, lightZenithCos) * sky);
    float3 multiple = AtmMultipleScattering(A, posKm, lightZenithCos) * (medium.scattering * cloud.z);
    return toLight * (visible * single + diffused) + multiple * sky;
}

// The point lights' light scattered toward the camera per km of air (lux × σs × phase, as a body's single scattering):
// the lights that reach the place (light_clusters.hlsli), each through its shadow where it has one this frame. A light
// without a shadow lights the air through walls; the halo of a lamp in a room shows outside only as far as its range.
float3 FroxelPointLights(AtmosphereParams A, AtmMedium medium, float3 relPos, float3 V, float2 uv, float viewDepth)
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
        float cosTheta = dot(V, toLight) * rsqrt(max(dot(toLight, toLight), 1e-6));
        float3 phase = medium.scatteringMie * AtmMiePhase(A.MieG, cosTheta) + medium.scatteringRay * AtmRayleighPhase(cosTheta);
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

    // Under the terrain (in a cave, inside a hill) neither the sun nor the sky; past the meshes (walls, roofs, crowns)
    // only the share of the sky that they leave, never less than the light bouncing in (SkyCaptureBounce)
    TerrainCover terrain = LoadTerrainCover();
    float open = TerrainOpenAt(terrain, AtmosphereLinearClamp, worldPos, float3(0, 1, 0), TerrainTolerance(length(relPos)));
    float sky = open > 0.0 ? open * max(FogMeshSkyVisibility(A, worldPos), A.SkyCaptureBounce) : 0.0;

    float skyDimming = CloudSkyDimming(CloudDeck());      // Ψ_ms under the clouds round the camera
    float3 L = A.SunIlluminance * FroxelBodyLight(A, medium, false, posKm, worldPos, relPos, V, viewDepth,
                                                  A.SunDirection, open, sky, skyDimming);
    if (any(A.MoonIlluminance > 0))
        L += A.MoonIlluminance * FroxelBodyLight(A, medium, true, posKm, worldPos, relPos, V, viewDepth,
                                                 A.MoonDirection, open, sky, skyDimming);
    L += FroxelPointLights(A, medium, relPos, V, (float2(id.xy) + Jitter.xy) * TileUV, viewDepth);
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
            }
        }
    }
    output[id] = current;
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
