// Volumetric clouds (VOLUMETRIC_CLOUDS_PLAN.md), made by Graphics/Clouds.cs and Graphics/CloudView.cs:
//   CSCloudBake   one group per 8³ brick of a level's region: the cloud field's density and profile at each texel (the
//                 weather, the shapes, the towers of the storm cells; no detail) and the brick's most density
//   CSCloudMarch  one thread per texel of a view's half-size cloud target: the clouds' light in front of the sky
//                 (rgb × Exposure, the air between included) and what they let through (a)
// Docs: .agent/knowledge/atmospheric_effects/volumetric_clouds.md.

#pragma kernel CSCloudBake
#pragma kernel CSCloudMarch

// What changes from dispatch to dispatch is pushed: a ComputeShader's Params buffer holds one copy per frame
cbuffer PushConstants : register(b3)
{
    uint DensityUAVIdx;         // bake: the level's density (RWTexture3D<float2>)
    uint BrickUAVIdx;           // bake: the level's bricks (RWTexture3D<float>)
    uint BakeLevel;             // bake: which level
    uint RegionX0Bits;          // bake: asint, the region's first texel in x and z, in the level's cloud-space texels
    uint RegionZ0Bits;
    uint TargetUAVIdx;          // march: the view's cloud target (RWTexture2D<float4>)
    uint TargetWidth;           // march: its size
    uint TargetHeight;
    uint FrameIndex;            // march: the jitter's frame
};

#include "common.fx"
#include "sky_common.fx"
#include "cloud_common.hlsli"

SamplerState CloudWrap : register(s4);         // trilinear wrap: the clipmap is toroidal, the noise tiles

// ── The cloud field ──

// The ordinary layer's density over its depth (hf 0 at the base, 1 at the top) by type: stratus a slab, stratocumulus
// thinning through its upper half, cumulus a flat base and a long taper (the noise rounds the tops)
float CloudLayerProfile(float hf, float type)
{
    float stratus = saturate(hf / 0.15) * saturate((1.0 - hf) / 0.25);
    float strato = saturate(hf / 0.2) * saturate((1.0 - hf) / 0.5);
    float cumulus = saturate(hf / 0.08) * saturate((1.0 - hf) / 0.7);
    return type < 0.5 ? lerp(stratus, strato, type * 2.0) : lerp(strato, cumulus, type * 2.0 - 1.0);
}

// The coverage at a place: the preset's, varied by the large-scale pattern (~50 km)
float CloudCoverageAt(CloudParams C, Texture3D<float4> lut, float2 xz)
{
    float n = lut.SampleLevel(CloudWrap, float3(xz / CLOUD_COVER_TILE, C.Evolution / 4000.0), 0).r;
    return saturate(C.Coverage + (n - 0.5) * 0.4);
}

// How deep a place lies in one tower's envelope (0 outside, 1 on its axis): a column from the base to its top, leaning
// downwind with height; a dome over the top of a growing tower; a mature one spreads an anvil over its top fifth, pushed
// downwind, with a dome over its updraft above it
// wobble (−0.5..0.5, a large-scale noise at the place) bulges and dents the column and the anvil's edge, so a tower is
// a stack of lumps rather than a cylinder
float CloudTowerEnvelope(CloudParams C, CloudCell k, float2 center, float2 xz, float h, float wobble)
{
    float top = k.Top;
    float span = max(top - C.Base, 1.0);
    float hf = (h - C.Base) / span;
    if (hf < 0.0 || hf > 1.12)
        return 0.0;
    float2 axis = center + C.WindDir * ((h - C.Base) * 0.12);          // the wind shears the tower
    float d = length(xz - axis) * (1.0 - 0.7 * wobble);

    // The column: widest a third of the way up, narrowing above, rounding off over the top (less under an anvil)
    float dome = lerp(0.3, 0.12, k.Anvil);
    float bulge = 0.8 + 0.3 * sin(saturate(hf * 1.3) * 3.14159);
    float r = k.Radius * bulge * sqrt(saturate((1.0 - hf) / dome));
    float column = r > 0.0 ? saturate((r - d) / (0.35 * r)) : 0.0;
    column *= saturate((h - C.Base) / 60.0);                             // the flat base

    // The anvil: a sheet over the top, wider and pushed downwind as the cell matures, thinning toward its rim
    float anvil = 0.0;
    if (k.Anvil > 0.0)
    {
        float ra = k.Radius * (1.0 + 2.6 * k.Anvil);
        float2 ac = axis + C.WindDir * (k.Radius * 1.8 * k.Anvil);
        float da = length(xz - ac) * (1.0 - 0.6 * wobble) / ra;          // 0 at its middle, 1 at its rim
        float under = lerp(0.8, 0.95, da * da);                           // its underside rises toward the rim
        float sheet = saturate((hf - under) / 0.04) * saturate((1.0 - hf) / 0.05);
        anvil = saturate((1.0 - da) / 0.4) * sheet * k.Anvil;
        // the overshooting top over the updraft
        float rt = k.Radius * 0.5 * k.Anvil;
        float dome2 = rt * sqrt(saturate((1.1 - hf) / 0.12));
        anvil = max(anvil, hf > 0.9 && dome2 > 0.0 ? saturate((dome2 - d) / (0.5 * dome2)) : 0.0);
    }
    return max(column, anvil);
}

float CloudTowers(CloudParams C, float2 xz, float h, float wobble)
{
    if (C.Cells == 0 || C.TowerShare <= 0.0)
        return 0.0;
    StructuredBuffer<CloudCell> cells = ResourceDescriptorHeap[C.Cells];
    int n = (int)C.CellsPerSide;
    int2 home = (int2)floor(xz / C.CellSize);
    float env = 0.0;
    // 5 × 5: an anvil reaches up to ~1.5 cells from its tower
    [loop] for (int j = -2; j <= 2; j++)
    {
        [loop] for (int i = -2; i <= 2; i++)
        {
            int2 c = home + int2(i, j);
            int2 w = ((c % n) + n) % n;
            CloudCell k = cells[w.y * n + w.x];
            if (k.Radius <= 0.0)
                continue;
            env = max(env, CloudTowerEnvelope(C, k, float2(c) * C.CellSize + k.Center, xz, h, wobble));
        }
    }
    return env;
}

// The cloud field at a cloud-space place: density 0..1 (before CloudDensity) and the profile, after Nubis: the base
// shape (Perlin-Worley eroded by Worley FBM) against the envelope (the layer's coverage × its height profile, or a
// tower's), the detail left to the march
float2 CloudFieldAt(CloudParams C, float2 xz, float h)
{
    Texture3D<float4> lut = ResourceDescriptorHeap[C.NoiseLut];
    float3 slide = float3(0.7, 0.3, 0.4) * (C.Evolution * 3.0);         // the noise slides through the shapes
    float4 n = lut.SampleLevel(CloudWrap, (float3(xz.x, h, xz.y) + slide) / CLOUD_BASE_TILE, 0);

    float cover = CloudCoverageAt(C, lut, xz);
    float layer = cover * CloudLayerProfile((h - C.Base) / max(C.Thickness, 1.0), C.Type);
    float env = max(layer, CloudTowers(C, xz, h, n.r - 0.5));
    if (env <= 0.002)
        return 0;

    float fbm = n.g * 0.75 + n.b * 0.25;
    float shape = saturate(CloudRemap(n.a, fbm - 1.0, 1.0, 0.0, 1.0));
    // The LUT's shape sits in a narrow band (~0.45-0.8): spread over 0..1, so coverage thins the clouds gradually
    shape = saturate(CloudRemap(shape, 0.42, 0.82, 0.0, 1.0));
    // A steep ramp past the threshold: a cumulus' edge is sharp, its body dense right inside it
    float density = saturate(CloudRemap(shape, 1.0 - env, 1.0 - env + 0.3, 0.0, 1.0));
    return float2(density, env);
}

groupshared uint BrickMaxShared;

[numthreads(8, 8, 8)]
void CSCloudBake(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    if (gi == 0)
        BrickMaxShared = 0;
    GroupMemoryBarrierWithGroupSync();

    CloudParams C = LoadClouds();
    CloudLevel L = C.Levels[BakeLevel];
    int2 texel = int2(asint(RegionX0Bits), asint(RegionZ0Bits)) + int2(id.xy);
    float2 xz = (float2(texel) + 0.5) * L.VoxelSize;
    float h = C.SlabBottom + (id.z + 0.5) / L.Layers * (C.SlabTop - C.SlabBottom);
    float2 field = CloudFieldAt(C, xz, h);

    int3 wrapped = int3(((texel % CLOUD_TEXELS) + CLOUD_TEXELS) % CLOUD_TEXELS, id.z);
    RWTexture3D<float2> density = ResourceDescriptorHeap[DensityUAVIdx];
    density[wrapped] = field;
    InterlockedMax(BrickMaxShared, (uint)(field.x * 255.0 + 0.999));    // any density at all marks the brick
    GroupMemoryBarrierWithGroupSync();

    if (gi == 0)
    {
        RWTexture3D<float> bricks = ResourceDescriptorHeap[BrickUAVIdx];
        bricks[wrapped / CLOUD_BRICK] = BrickMaxShared / 255.0;
    }
}

// ── The march ──

// Henyey-Greenstein, normalised (1/sr)
float CloudHG(float cosTheta, float g)
{
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * ATM_PI * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5));
}

// The droplets' phase: a strong forward lobe (the silver lining, the glare round a sun behind a thin edge) and a weak
// backward one; scale < 1 widens both (the multiple scattering octaves)
float CloudPhase(float cosTheta, float scale)
{
    return lerp(CloudHG(cosTheta, -0.3 * scale), CloudHG(cosTheta, 0.8 * scale), 0.7);
}

// Interleaved gradient noise (Jimenez 2014), a different pattern every frame
float CloudJitter(uint2 pixel, uint frame)
{
    float2 p = float2(pixel) + 5.588238 * float(frame & 63);
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

// The density with the billows carved into its edges: the LUT's Worley FBM at the detail tile, wispy toward the base and
// billowy above (Nubis), fading with distance, where a billow becomes smaller than a texel of the target
float CloudErode(CloudParams C, Texture3D<float4> lut, float2 field, float2 xz, float h, float dist)
{
    if (field.x <= 0.0)
        return 0.0;
    float strength = 0.6 * saturate(1.0 - dist / 60000.0);
    if (strength <= 0.0)
        return field.x;
    float3 slide = float3(-0.4, 0.8, 0.2) * (C.Evolution * 2.0);
    float4 n = lut.SampleLevel(CloudWrap, (float3(xz.x, h, xz.y) + slide) / CLOUD_DETAIL_TILE, 0);
    float detail = n.g * 0.5 + n.b * 0.5;
    float hf = saturate((h - C.Base) / max(C.TowerTop - C.Base, C.Thickness));
    detail = lerp(detail, 1.0 - detail, saturate(hf * 5.0));
    // Carves the thin outer part only: the body (density near 1) keeps its bulk
    return saturate(CloudRemap(field.x, detail * strength, 1.0, 0.0, 1.0));
}

// Optical depth toward the light from a place: 6 steps growing ×2 from 60 m (3.8 km in all), each in the level its own
// distance picks, with the detail
float CloudLightDepth(CloudParams C, Texture3D<float4> lut, float3 rel, float3 lightDir, float jitter)
{
    float tau = 0.0;
    float s = 0.0, step = 60.0;
    [loop] for (int i = 0; i < 6; i++)
    {
        float t = s + step * (0.5 + 0.5 * (jitter - 0.5));
        float3 p = rel + lightDir * t;
        float h = CloudHeight(p);
        float2 xz = C.CameraCloud + p.xz;
        float dist = max(abs(p.x), abs(p.z));
        float2 field = CloudSample(C, xz, h, dist, CloudWrap);
        tau += CloudErode(C, lut, field, xz, h, dist) * step;
        s += step;
        step *= 2.0;
    }
    return tau * C.Density;
}

// Optical depth straight up from a place, roughly (3 steps to ~1 km, no detail): how much of the sky above a cloud's
// part sees past the cloud over it. The base of a deck sees none of it, only the ground and the light come through.
float CloudSkyDepth(CloudParams C, float3 rel)
{
    float tau = 0.0;
    const float ups[3] = { 60.0, 250.0, 700.0 };
    const float lens[3] = { 120.0, 260.0, 640.0 };
    [unroll] for (int i = 0; i < 3; i++)
    {
        float3 p = rel + float3(0, ups[i], 0);
        float h = CloudHeight(p);
        tau += CloudSample(C, C.CameraCloud + p.xz, h, max(abs(p.x), abs(p.z)), CloudWrap).x * lens[i];
    }
    return tau * C.Density;
}

[numthreads(8, 8, 1)]
void CSCloudMarch(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= TargetWidth || id.y >= TargetHeight)
        return;
    RWTexture2D<float4> target = ResourceDescriptorHeap[TargetUAVIdx];
    if (CloudParamsIdx == 0 || AtmosphereIdx == 0)
    {
        target[id.xy] = float4(0, 0, 0, 1);
        return;
    }
    CloudParams C = LoadClouds();
    AtmosphereParams A = LoadAtmosphere();
    Texture3D<float4> lut = ResourceDescriptorHeap[C.NoiseLut];

    // Camera-relative view direction through the texel's centre (reverse-Z: the far plane is z = 0)
    float2 uv = (float2(id.xy) + 0.5) / float2(TargetWidth, TargetHeight);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 farClip = mul(float4(ndc, 0, 1), CameraInverse);
    float3 dir = normalize(farClip.xyz / farClip.w);

    float tStart, tEnd;
    if (!CloudSlabSegment(C, dir, tStart, tEnd))
    {
        target[id.xy] = float4(0, 0, 0, 1);
        return;
    }

    // The lights: the sun, and the brightest moon where it gives the light (night)
    float3 sunE = A.SunIlluminance, moonE = A.MoonIlluminance;
    bool moonLights = dot(moonE, 1.0) > dot(sunE, 1.0) * 0.01 * saturate(A.SunDirection.y * 10.0 + 1.0);
    float3 lightDir = moonLights && A.SunDirection.y < 0.0 ? A.MoonDirection : A.SunDirection;
    float3 lightE = moonLights && A.SunDirection.y < 0.0 ? moonE : sunE;
    float cosTheta = dot(dir, lightDir);
    float phase[3];
    [unroll] for (int o = 0; o < 3; o++)
        phase[o] = CloudPhase(cosTheta, pow(0.5, o));

    StructuredBuffer<float4> ambientBuffer = ResourceDescriptorHeap[A.Ambient];
    float3 skyAmbient = ambientBuffer[0].rgb, groundAmbient = ambientBuffer[1].rgb;     // E / π × Exposure
    float viewHeightKm = AtmosphereViewHeight(A, CamPos.y);

    float jitter = CloudJitter(id.xy, FrameIndex);
    float3 L = 0;
    float T = 1.0;
    float depthSum = 0.0, depthWeight = 0.0;
    float t = tStart;
    float slabSpan = max(C.SlabTop - C.SlabBottom, 1.0);
    [loop] for (int s = 0; s < 512 && t < tEnd; s++)
    {
        float3 rel = dir * t;
        float dist = max(abs(rel.x), abs(rel.z));
        float blend;
        uint level = CloudLevelAt(C, dist, blend);
        float voxel = C.Levels[level].VoxelSize;
        float h = CloudHeight(rel);
        float2 xz = C.CameraCloud + rel.xz;

        // An empty brick: leap half a brick
        if (CloudBrickMax(C, level, xz, h) <= 0.0 && (blend <= 0.0 || CloudBrickMax(C, level + 1, xz, h) <= 0.0))
        {
            // Out of the brick (a slab: wide across, thin in height) and just into the next: a brick's layers can be a few
            // metres thick, so any more skips the base of the cloud in it
            t += CloudBrickExit(C, level, xz, h, dir) + 0.5;
            continue;
        }

        float step = voxel * 0.6 * (0.75 + 0.5 * jitter);
        float3 p = dir * (t + step * 0.5);
        float hp = CloudHeight(p);
        float2 xzp = C.CameraCloud + p.xz;
        float2 field = CloudSample(C, xzp, hp, dist, CloudWrap);
        float density = CloudErode(C, lut, field, xzp, hp, dist);
        if (density > 0.0)
        {
            float sigmaT = density * C.Density;
            float sigmaS = sigmaT * C.Albedo;

            // The light reaching this place through the air above (0 in the earth's shadow) and through the cloud
            float3 toLight = AtmTransmittanceToLight(A, CloudAtmospherePos(p), lightDir);
            float tau = any(toLight > 0.0) ? CloudLightDepth(C, lut, p, lightDir, jitter) : 0.0;
            float scattered = 0.0;
            [unroll] for (int o = 0; o < 3; o++)
                scattered += pow(0.5, o) * phase[o] * exp(-pow(0.5, o) * tau);
            // Deep in a thick cloud the light has diffused: it falls as 1 / (1 + 0.75 (1 − g) τ) (two-stream, g 0.85), not
            // exponentially, which is why an overcast deck's base is grey and not black
            scattered += 1.0 / (4.0 * ATM_PI) / (1.0 + 0.1125 * tau);

            // The sky above, as much as the cloud over this place leaves (diffuse light gets deeper than a beam), and
            // the lit ground below, by the height in the slab
            float hs = saturate((hp - C.SlabBottom) / slabSpan);
            float skySeen = exp(-0.3 * CloudSkyDepth(C, p));
            float3 ambient = skyAmbient * skySeen + groundAmbient * (1.0 - hs) * 0.6;

            float3 S = sigmaS * (lightE * toLight * scattered * Exposure + ambient);
            float x = sigmaT * step;
            float through = exp(-x);
            L += T * S * (x > 1e-4 ? (1.0 - through) / sigmaT : step);
            depthSum += (t + step * 0.5) * (T - T * through);
            depthWeight += T - T * through;
            T *= through;
            if (T < 0.005)
                break;
        }
        t += step;
    }

    if (T >= 0.999)
    {
        target[id.xy] = float4(0, 0, 0, 1);
        return;
    }

    // The air between the camera and the clouds (two transmittance LUT reads, as the 2D layer had it) and the sky's own
    // light scattered in front of them
    float d = depthSum / max(depthWeight, 1e-6);
    float3 cloudKm = CloudAtmospherePos(dir * d);
    float r = length(cloudKm);
    float3 viewT = saturate(AtmTransmittance(A, viewHeightKm, dir.y) / max(AtmTransmittance(A, r, dot(cloudKm / r, dir)), 1e-6));
    float3 inFront = AtmosphereSkyBelowClouds(A, normalize(float3(dir.x, max(dir.y, 0.0), dir.z))) * Exposure;
    target[id.xy] = float4(L * viewT + inFront * (1.0 - T), T);
}
