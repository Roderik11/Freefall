// Volumetric clouds (VOLUMETRIC_CLOUDS_PLAN.md), made by Graphics/Clouds.cs and Graphics/CloudView.cs:
//   CSCloudBake   one group per 8³ brick of a level's region: the cloud field's density and profile at each texel (the
//                 weather, the shapes, the towers of the storm cells; no detail) and the brick's most density
//   CSCloudLightInit / CSCloudLightStep / CSCloudLightChain   a level's light volume (half the density's texels each
//                 way): each texel's optical depth toward the light out of the slab, by doubling (as the terrain's
//                 shadow map is made): τ₀ = σΔ, τ_{k+1}(p) = τ_k(p) + τ_k(p + 2^k Δ d), 8 passes; where the path leaves
//                 the level's box the coarser level's depth is added (the chain)
//   CSCloudSkyScan  a level's sky volume, a thread per column: the optical depth straight up and straight down
//   CSCloudMarch  one thread per texel of a view's half-size cloud target: the clouds' light in front of the sky
//                 (rgb × Exposure, the air between included) and what they let through (a)
// Docs: .agent/knowledge/atmospheric_effects/volumetric_clouds.md.

#pragma kernel CSCloudBake
#pragma kernel CSCloudLightInit
#pragma kernel CSCloudLightStep
#pragma kernel CSCloudLightChain
#pragma kernel CSCloudSkyScan
#pragma kernel CSCloudMarch

// What changes from dispatch to dispatch is pushed: a ComputeShader's Params buffer holds one copy per frame
cbuffer PushConstants : register(b3)
{
    uint DensityUAVIdx;         // bake: the level's density (RWTexture3D<float2>)
    uint BrickUAVIdx;           // bake: the level's bricks (RWTexture3D<float>)
    uint BakeLevel;             // bake, light, sky: which level
    uint RegionX0Bits;          // bake: asint, the region's first texel in x and z, in the level's cloud-space texels
    uint RegionZ0Bits;
    uint TargetUAVIdx;          // march: the view's cloud target (RWTexture2D<float4>); light, sky: the texture written
    uint TargetWidth;           // march: its size
    uint TargetHeight;
    uint FrameIndex;            // march: the jitter's frame
    uint SourceTexIdx;          // light step: the last pass (Texture3D<float>)
    uint LightStepBits;         // light step: asfloat, this pass's reach along the light (m)
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
// turret (−0.5..0.5, a ~2 km noise at the place) bulges and dents the column and the anvil's rim, so a tower is a stack
// of turrets rather than a cylinder
float CloudTowerEnvelope(CloudParams C, CloudCell k, float2 center, float2 xz, float h, float turret)
{
    float top = k.Top;
    float span = max(top - C.Base, 1.0);
    float hf = (h - C.Base) / span;
    if (hf < 0.0 || hf > 1.12)
        return 0.0;
    float2 axis = center + C.WindDir * ((h - C.Base) * 0.12);          // the wind shears the tower
    float2 rel = xz - axis;
    float d = length(rel) * (1.0 - 0.9 * turret);

    // The column: widest a third of the way up, narrowing above, rounding off over the top (flatter under an anvil)
    float dome = lerp(0.3, 0.1, k.Anvil);
    float bulge = 0.85 + 0.3 * sin(saturate(hf * 1.3) * 3.14159);
    float r = k.Radius * bulge * sqrt(saturate((1.0 - hf) / dome));
    float column = r > 0.0 ? saturate((r - d) / (0.35 * r)) : 0.0;
    column *= saturate((h - C.Base) / 60.0);                             // the flat base

    // The anvil: the top spreading downwind as a wedge, a short overhang upwind, flat on top, its underside rising and
    // the sheet thinning toward its far rim; an overshooting dome over the updraft
    float anvil = 0.0;
    if (k.Anvil > 0.0)
    {
        float2 wind = C.WindDir, across = float2(-wind.y, wind.x);
        float u = dot(rel, wind), w = dot(rel, across);
        float reachDown = k.Radius * (1.0 + 3.5 * k.Anvil), reachUp = k.Radius * 0.9;
        float uu = u / (u >= 0.0 ? reachDown : reachUp);
        float ww = w / (k.Radius * (1.0 + 1.2 * k.Anvil * saturate(uu)));      // widening downwind
        float e = sqrt(uu * uu + ww * ww) * (1.0 - 0.7 * turret);             // 0 at the tower, 1 at the rim
        float under = lerp(0.84, 0.97, saturate(e));
        float sheet = saturate((hf - under) / 0.03) * saturate((1.0 - hf) / 0.03);
        anvil = saturate((1.0 - e) / 0.35) * sheet * k.Anvil;
        float rt = k.Radius * 0.55 * k.Anvil;
        float dome2 = rt * sqrt(saturate((1.09 - hf) / 0.1));
        anvil = max(anvil, hf > 0.9 && dome2 > 0.0 ? saturate((dome2 - d) / (0.5 * dome2)) : 0.0);
    }
    return max(column, anvil);
}

float CloudTowers(CloudParams C, float2 xz, float h, float turret)
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
            env = max(env, CloudTowerEnvelope(C, k, float2(c) * C.CellSize + k.Center, xz, h, turret));
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
    // Turrets: the Perlin-Worley at a 2.1 km tile (CLOUD_PERIOD / 120), for the towers' bulges
    float turret = lut.SampleLevel(CloudWrap, (float3(xz.x, h, xz.y) + slide) / (CLOUD_PERIOD / 120.0), 0).a - 0.4;
    float env = max(layer, CloudTowers(C, xz, h, turret));
    if (env <= 0.002)
        return 0;

    float fbm = n.g * 0.75 + n.b * 0.25;
    float shape = saturate(CloudRemap(n.a, fbm - 1.0, 1.0, 0.0, 1.0));
    // The LUT's shape sits in a narrow band (~0.45-0.8): spread over 0..1, so coverage thins the clouds gradually
    shape = saturate(CloudRemap(shape, 0.42, 0.82, 0.0, 1.0));
    // Baked smooth: how far past the threshold (cloud_common.hlsli CloudFieldDensity sharpens it in the march)
    return float2(saturate(0.5 + shape - (1.0 - env)), env);
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
    // A brick holds cloud where the field comes within the billows' reach of the surface
    InterlockedMax(BrickMaxShared, (uint)(saturate(field.x - (0.5 - CLOUD_BILLOW)) * 255.0 + 0.999));
    GroupMemoryBarrierWithGroupSync();

    if (gi == 0)
    {
        RWTexture3D<float> bricks = ResourceDescriptorHeap[BrickUAVIdx];
        bricks[wrapped / CLOUD_BRICK] = BrickMaxShared / 255.0;
    }
}

// ── The light volumes ──

// A light texel's place: its cloud-space xz and height. Physical texels are toroidal like the density's: the logical
// texel is the one in the level's current box (Origin, 128 light texels of twice the voxel each way).
float3 CloudLightTexelPos(CloudParams C, CloudLevel L, uint3 id)
{
    float size = L.VoxelSize * 2.0;
    int2 origin = (int2)round(L.Origin / size);
    int2 logical = origin + (((int2(id.xy) - origin) % 128) + 128) % 128;
    float2 xz = (float2(logical) + 0.5) * size;
    float h = C.SlabBottom + (id.z + 0.5) / (L.Layers / 2) * (C.SlabTop - C.SlabBottom);
    return float3(xz.x, h, xz.y);
}

// The step along the light that moves at most one light texel each way
float CloudLightStepLength(CloudParams C, CloudLevel L)
{
    float across = L.VoxelSize * 2.0;
    float up = (C.SlabTop - C.SlabBottom) / (L.Layers / 2);
    float3 d = abs(C.LightDir);
    return min(min(across / max(d.x, 1e-4), across / max(d.z, 1e-4)), up / max(d.y, 1e-4));
}

// A place s metres on toward the light from p (x, height, z): the height rises as the planet curves away below
float3 CloudAlongLight(CloudParams C, float3 p, float s)
{
    float3 q = p + C.LightDir * s;
    q.y += s * s * (1.0 - C.LightDir.y * C.LightDir.y) / (2.0 * CLOUD_EARTH_RADIUS);
    return q;
}

bool CloudInLevelBox(CloudParams C, uint level, CloudLevel L, float3 q)
{
    if (q.y < C.SlabBottom || q.y > C.SlabTop)
        return false;
    if (level == CLOUD_LEVELS - 1)
        return true;            // the coarsest level holds the whole period: its box wraps round the world
    float margin = L.VoxelSize;
    float2 lo = L.Origin + margin, hi = L.Origin + L.VoxelSize * CLOUD_TEXELS - margin;
    return all(q.xz >= lo) && all(q.xz <= hi);
}

[numthreads(8, 8, 8)]
void CSCloudLightInit(uint3 id : SV_DispatchThreadID)
{
    CloudParams C = LoadClouds();
    CloudLevel L = C.Levels[BakeLevel];
    if (id.z >= L.Layers / 2)
        return;
    float3 p = CloudLightTexelPos(C, L, id);
    // The density's 2×2×2 texels under this one, averaged by the trilinear sample at their middle
    float density = CloudFieldDensity(CloudSampleLevel(C, BakeLevel, p.xz, p.y, CloudWrap).x);
    RWTexture3D<float> target = ResourceDescriptorHeap[TargetUAVIdx];
    target[id] = density * C.Density * CloudLightStepLength(C, L);
}

[numthreads(8, 8, 8)]
void CSCloudLightStep(uint3 id : SV_DispatchThreadID)
{
    CloudParams C = LoadClouds();
    CloudLevel L = C.Levels[BakeLevel];
    if (id.z >= L.Layers / 2)
        return;
    Texture3D<float> source = ResourceDescriptorHeap[SourceTexIdx];
    float tau = source.Load(int4(id, 0));
    float3 q = CloudAlongLight(C, CloudLightTexelPos(C, L, id), asfloat(LightStepBits));
    if (CloudInLevelBox(C, BakeLevel, L, q))
        tau += source.SampleLevel(CloudWrap, CloudLevelUvw(C, L, q.xz, q.y), 0);
    RWTexture3D<float> target = ResourceDescriptorHeap[TargetUAVIdx];
    target[id] = tau;
}

// Where the light's path leaves this level's box sideways (before it leaves the slab), the coarser level's depth from
// there on: a storm far toward a low sun shades the near clouds
[numthreads(8, 8, 8)]
void CSCloudLightChain(uint3 id : SV_DispatchThreadID)
{
    CloudParams C = LoadClouds();
    CloudLevel L = C.Levels[BakeLevel];
    if (id.z >= L.Layers / 2 || BakeLevel >= CLOUD_LEVELS - 1)
        return;
    float3 p = CloudLightTexelPos(C, L, id);
    float3 d = C.LightDir;
    float2 lo = L.Origin + L.VoxelSize, hi = L.Origin + L.VoxelSize * (CLOUD_TEXELS - 1);
    float2 exitXZ = select(d.xz >= 0.0, (hi - p.xz) / max(d.xz, 1e-5), (p.xz - lo) / max(-d.xz, 1e-5));
    float s = min(exitXZ.x, exitXZ.y);
    float3 q = CloudAlongLight(C, p, s);
    if (q.y < C.SlabBottom || q.y > C.SlabTop)
        return;                 // out of the slab first: nothing more in the way
    RWTexture3D<float> target = ResourceDescriptorHeap[TargetUAVIdx];
    target[id] = target[id] + CloudLightDepthAt(C, BakeLevel + 1, q.xz, q.y, CloudWrap);
}

#define CLOUD_MAX_SKY_LAYERS 64

[numthreads(8, 8, 1)]
void CSCloudSkyScan(uint3 id : SV_DispatchThreadID)
{
    CloudParams C = LoadClouds();
    CloudLevel L = C.Levels[BakeLevel];
    uint layers = min(L.Layers / 2, CLOUD_MAX_SKY_LAYERS);
    float layerH = (C.SlabTop - C.SlabBottom) / layers;
    float sigma[CLOUD_MAX_SKY_LAYERS];
    float total = 0.0;
    [loop] for (uint k = 0; k < layers; k++)
    {
        float3 p = CloudLightTexelPos(C, L, uint3(id.xy, k));
        sigma[k] = CloudFieldDensity(CloudSampleLevel(C, BakeLevel, p.xz, p.y, CloudWrap).x) * C.Density * layerH;
        total += sigma[k];
    }
    RWTexture3D<float2> target = ResourceDescriptorHeap[TargetUAVIdx];
    float below = 0.0;
    [loop] for (uint j = 0; j < layers; j++)
    {
        float above = total - below - sigma[j];
        target[uint3(id.xy, j)] = float2(above + 0.5 * sigma[j], below + 0.5 * sigma[j]);
        below += sigma[j];
    }
}

// ── The march ──

// Henyey-Greenstein, normalised (1/sr)
float CloudHG(float cosTheta, float g)
{
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * ATM_PI * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5));
}

// Draine's phase (normalised), the backward rise HG lacks
float CloudDraine(float cosTheta, float g, float alpha)
{
    float g2 = g * g;
    return (1.0 - g2) * (1.0 + alpha * cosTheta * cosTheta)
         / (4.0 * ATM_PI * (1.0 + alpha * (1.0 + 2.0 * g2) / 3.0) * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5));
}

// The droplets' phase: Jendersie & d'Eon's fit of Mie scattering by water droplets (SIGGRAPH 2023), HG + Draine for a
// droplet diameter of CLOUD_DROPLET_UM: a narrow forward spike (the glare round a sun behind a thin edge), a broad
// forward lobe (the silver lining) and a backward rise. scale < 1 widens the lobes (the multiple scattering octaves).
#define CLOUD_DROPLET_UM 15.0
#define CLOUD_PHASE_DRAINE 1      // 0: the two-lobe HG (0.8 / −0.3, 70 % forward) the 2D layer used
float CloudPhase(float cosTheta, float scale)
{
#if !CLOUD_PHASE_DRAINE
    return lerp(CloudHG(cosTheta, -0.3 * scale), CloudHG(cosTheta, 0.8 * scale), 0.7);
#endif
    const float d = CLOUD_DROPLET_UM;
    const float gHG = exp(-0.0990567 / (d - 1.67154));
    const float gD = exp(-2.20679 / (d + 3.91029) - 0.428934);
    const float alpha = exp(3.62489 - 8.29288 / (d + 5.52825));
    const float wD = exp(-0.599085 / (d - 0.641583) - 0.665888);
    return lerp(CloudHG(cosTheta, gHG * scale), CloudDraine(cosTheta, gD * scale, alpha), wD);
}

// The light that reaches a place from a body above the atmosphere: the transmittance LUT, faded over the body's disc as
// the planet's edge passes it (a cloud at sunset goes grey from below, not at a line); posKm in the atmosphere's frame
float3 CloudLightThroughAir(AtmosphereParams A, float3 posKm, float3 lightDir)
{
    float r = length(posKm);
    float mu = dot(posKm / r, lightDir);
    float muHorizon = -sqrt(max(1.0 - (A.BottomRadius / r) * (A.BottomRadius / r), 0.0));
    float seen = saturate((mu - muHorizon) / 0.0093 + 0.5);    // the sun's disc is ~0.0093 rad across
    if (seen <= 0.0)
        return 0;
    return AtmTransmittance(A, r, max(mu, muHorizon + 1e-4)) * seen;
}

// Interleaved gradient noise (Jimenez 2014), a different pattern every frame
float CloudJitter(uint2 pixel, uint frame)
{
    float2 p = float2(pixel) + 5.588238 * float(frame & 63);
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

// The density at a place from the baked field with the billows: the surface moved in and out by the LUT's Worley FBM at
// the detail tile (and a second octave near the camera), wispy toward the base and billowy above (Nubis: the inverted
// Worley leaves round puffs), fading with distance where a billow becomes smaller than a texel of the target; then the
// ramp sharpens it (CloudFieldDensity)
float CloudErode(CloudParams C, Texture3D<float4> lut, float2 field, float2 xz, float h, float dist)
{
    if (field.x <= 0.5 - CLOUD_BILLOW)
        return 0.0;
    float strength = saturate(1.0 - dist / 60000.0);
    if (strength <= 0.0)
        return CloudFieldDensity(field.x);
    float3 q = float3(xz.x, h, xz.y) + float3(-0.4, 0.8, 0.2) * (C.Evolution * 2.0);
    float4 n = lut.SampleLevel(CloudWrap, q / CLOUD_DETAIL_TILE, 0);
    float detail = n.g * 0.6 + n.b * 0.4;
    float near = saturate(1.0 - dist / 15000.0);
    if (near > 0.0)
        detail = lerp(detail, detail * 0.65 + 0.35 * lut.SampleLevel(CloudWrap, q / (CLOUD_DETAIL_TILE * 0.37), 0).b, near);
    detail = saturate((detail - 0.3) / 0.45);           // the Worley FBM sits in a narrow band: spread it
    float hf = saturate((h - C.Base) / max(C.TowerTop - C.Base, C.Thickness));
    float erode = lerp(detail, 1.0 - detail, saturate(hf * 5.0));
    return CloudFieldDensity(field.x - (erode - 0.35) * CLOUD_BILLOW * strength);
}

// Optical depth toward the light from a place: three short steps through the detail (the billows, ~400 m, shade each
// other) out to CLOUD_SHORT_LIGHT metres, then the light volume from there on (the whole way out of the slab: a tower
// shades the cloud behind it, a storm far toward a low sun the near deck)
#define CLOUD_SHORT_LIGHT 560.0
float CloudLightDepth(CloudParams C, Texture3D<float4> lut, float3 rel, float3 lightDir, float jitter)
{
    float tau = 0.0;
    const float at[3] = { 40.0, 160.0, 400.0 };
    const float len[3] = { 80.0, 160.0, 320.0 };
    [unroll] for (int i = 0; i < 3; i++)
    {
        float3 p = rel + lightDir * (at[i] * (0.8 + 0.4 * jitter));
        float h = CloudHeight(p);
        float2 xz = C.CameraCloud + p.xz;
        float dist = max(abs(p.x), abs(p.z));
        tau += CloudErode(C, lut, CloudSample(C, xz, h, dist, CloudWrap), xz, h, dist) * len[i];
    }
    tau *= C.Density;
    float3 far = rel + lightDir * CLOUD_SHORT_LIGHT;
    float hf = CloudHeight(far);
    float distFar = max(abs(far.x), abs(far.z));
    float blend;
    uint level = CloudLevelAt(C, distFar, blend);
    float2 xzFar = C.CameraCloud + far.xz;
    float volume = CloudLightDepthAt(C, level, xzFar, hf, CloudWrap);
    if (blend > 0.0)
        volume = lerp(volume, CloudLightDepthAt(C, level + 1, xzFar, hf, CloudWrap), blend);
    return tau + volume;
}

// Optical depth straight up (x) and down (y) from a place, from the sky volume of the level its distance picks: how much
// of the sky above (and of the lit ground below) a part of a cloud sees past the cloud round it
float2 CloudSkyDepth(CloudParams C, float3 rel)
{
    float h = CloudHeight(rel);
    float dist = max(abs(rel.x), abs(rel.z));
    float blend;
    uint level = CloudLevelAt(C, dist, blend);
    float2 xz = C.CameraCloud + rel.xz;
    float2 tau = CloudSkyDepthAt(C, level, xz, h, CloudWrap);
    if (blend > 0.0)
        tau = lerp(tau, CloudSkyDepthAt(C, level + 1, xz, h, CloudWrap), blend);
    return tau;
}

// Schneider's in-scatter probability (Nubis 2017): light gets scattered toward the eye only where it has had some cloud
// to scatter in, so thin edges and the side of a billow away from where the light comes in are darker than a plain
// march makes them (the dark-edged, cauliflower look); stronger higher up, where the billows are. Only with the light
// behind the viewer: looking toward it, the thin edges are what glows (the silver lining). Applied to the beam's
// octaves only (the light volume already darkens a cloud's base). CLOUD_INSCATTER 0 turns it off.
#define CLOUD_INSCATTER 0.6
float CloudInScatter(CloudParams C, float density, float h, float cosTheta)
{
    float depth = 0.05 + pow(saturate(density), lerp(0.5, 2.0, saturate((h - C.Base) / 2500.0)));
    return lerp(1.0, saturate(depth), CLOUD_INSCATTER * saturate(0.5 - 0.5 * cosTheta));
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

    // The light the volumes were made for: the sun, or after it has set at the clouds too the brightest moon
    float3 lightDir = C.LightDir;
    float3 lightE = C.LightIsMoon != 0 ? A.MoonIlluminance : A.SunIlluminance;
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

            // The light reaching this place through the air above (faded out in the earth's shadow) and through the cloud
            float3 toLight = CloudLightThroughAir(A, CloudAtmospherePos(p), lightDir);
            float tau = any(toLight > 0.0) ? CloudLightDepth(C, lut, p, lightDir, jitter) : 0.0;
            float beam = 0.0;
            [unroll] for (int o = 0; o < 3; o++)
                beam += pow(0.5, o) * phase[o] * exp(-pow(0.5, o) * tau);
            // Deep in a thick cloud the light has diffused: it falls as 1 / (1 + 0.75 (1 − g) τ) (two-stream, g 0.85), not
            // exponentially, which is why an overcast deck's base is grey and not black
            float scattered = beam * CloudInScatter(C, density, hp, cosTheta) + 1.0 / (4.0 * ATM_PI) / (1.0 + 0.1125 * tau);

            // The sky above and the lit ground below, as much of each as the cloud over and under this place leaves
            // (diffuse light gets deeper than a beam): bright tops, dark cores, the ground's warm light on the bases.
            // Near the face we see, also what that face sees out toward the camera: a tower's flank half the sky (its
            // blue shade), a deck's base none of it, only the ground
            float2 skyTau = CloudSkyDepth(C, p);
            float faceOpen = pow(T, 0.3);
            float skySeen = max(exp(-0.3 * skyTau.x), saturate(0.5 - 0.5 * dir.y) * faceOpen);
            float groundSeen = max(exp(-0.3 * skyTau.y), saturate(0.5 + 0.5 * dir.y) * faceOpen);
            float3 ambient = skyAmbient * skySeen + groundAmbient * groundSeen;

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
