// Volumetric clouds (VOLUMETRIC_CLOUDS_PLAN.md), made by Graphics/Clouds.cs and Graphics/CloudView.cs:
//   CSCloudBake   one group per 8³ brick of a level's region: the cloud field's density and profile at each texel (the
//                 weather, the shapes, the towers of the storm cells; no detail) and the brick's most density
//   CSCloudLightInit / CSCloudLightStep / CSCloudLightChain   a level's light volume (half the density's texels each
//                 way): each texel's optical depth toward the light out of the slab, by doubling (as the terrain's
//                 shadow map is made): τ₀ = σΔ, τ_{k+1}(p) = τ_k(p) + τ_k(p + 2^k Δ d), 8 passes; where the path leaves
//                 the level's box the coarser level's depth is added (the chain)
//   CSCloudSkyScan  a level's sky volume, a thread per column: the optical depth straight up and straight down
//   CSCloudMarch  a view's clouds, at half its size: every texel (MarchMode 0) or one of each 2×2 block (1, a quarter-
//                 size sparse target): the clouds' light in front of the sky (rgb × Exposure, the air between included),
//                 what they let through (a) and how far they are (depth)
//   CSCloudReconstruct  the half-size target from the sparse texels and last frame's, reprojected and clamped
//   CSCloudProbeFace  one face of the sky probe's clouds (64², coarser)
//   CSCloudShadowMap  rows of the ground shadow map: the optical depth toward the light from where a ray crosses the
//                 slab's bottom, marched through the field (the shadows on the ground, the water, the air)
//   CSPrecipitation / CSPrecipitationProbe  what reaches the ground round the camera (Graphics/Weather.cs): the map, and
//                 the camera's own for the CPU
// Docs: .agent/knowledge/atmospheric_effects/volumetric_clouds.md, weather.md.

#pragma kernel CSCloudBake
#pragma kernel CSCloudLightInit
#pragma kernel CSCloudLightStep
#pragma kernel CSCloudLightChain
#pragma kernel CSCloudSkyScan
#pragma kernel CSCloudMarch
#pragma kernel CSCloudReconstruct
#pragma kernel CSCloudProbeFace
#pragma kernel CSCloudSceneDepth
#pragma kernel CSCloudShadowMap
#pragma kernel CSPrecipitation
#pragma kernel CSPrecipitationProbe

// What changes from dispatch to dispatch is pushed: a ComputeShader's Params buffer holds one copy per frame
cbuffer PushConstants : register(b3)
{
    uint DensityUAVIdx;         // bake: the level's density (RWTexture3D<float2>)
    uint BrickUAVIdx;           // bake: the level's bricks (RWTexture3D<float>)
    uint BakeLevel;             // bake, light, sky: which level
    uint RegionX0Bits;          // bake: asint, the region's first texel in x and z, in the level's cloud-space texels
    uint RegionZ0Bits;
    uint TargetUAVIdx;          // march, reconstruct: the colour written; light, sky, probe: the texture written
    uint TargetWidth;           // march, reconstruct: the view's half-size target; probe: the face's size
    uint TargetHeight;
    uint FrameIndex;            // march, reconstruct: the jitter's frame and which texel of each 2×2 is marched
    uint SourceTexIdx;          // light step: the last pass (Texture3D<float>)
    uint LightStepBits;         // light step: asfloat, this pass's reach along the light (m)
    uint DepthUAVIdx;           // march, reconstruct: the depth written (RWTexture2D<float>, km)
    uint MarchMode;             // march: 0 every texel, 1 one of each 2×2 block (sparse)
    uint SparseTexIdx;          // reconstruct: this frame's sparse colour and depth
    uint SparseDepthTexIdx;
    uint HistoryTexIdx;         // reconstruct: last frame's colour (0 = none)
    uint ProbeFace;             // probe: the face marched this frame
    uint SceneDepthTexIdx;      // march, reconstruct: this frame's half-size scene depth (CSCloudSceneDepth; 0 = none);
                                // scene depth: the G-buffer's linear depth (SourceTexIdx's role there)
    uint PrevSceneDepthTexIdx;  // reconstruct: last frame's (0 = none)
    uint HistoryDepthTexIdx;    // reconstruct: last frame's clouds' depth (km; 0 = none)
    uint PrecipProbeUAVIdx;     // precipitation probe: RWStructuredBuffer<float4>, the camera's (Graphics/Weather.cs)
    uint PrecipOriginXBits;     // precipitation: asfloat, the map's corner relative to the camera (m), and its texel
    uint PrecipOriginZBits;
    uint PrecipTexelBits;
};

// The view's reprojection (CloudView: its own ComputeShader, so two views in a frame do not share these)
cbuffer Params : register(b4)
{
    row_major float4x4 PrevCameraRelativeVP;    // last frame's view-projection without translation
    float3 CameraMove;                          // this frame's camera position minus last frame's
    float ExposureRatio;                        // this frame's exposure / last frame's: the history is pre-exposed
    float2 CloudStep;                           // how far the wind carried the clouds since last frame (world xz, m)
    float2 _cloudParamsPad;
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

// How rough a deck's base is, 0..1: a dense heaped deck (a storm's) hangs lower in places and rolls; fair-weather cumulus
// and a stratus sheet keep a flat base. CLOUD_BASE_DIP: how far below the base it hangs at the most (Clouds.cs BaseDip).
#define CLOUD_BASE_DIP 450.0
float CloudBaseRough(CloudParams C)
{
    return saturate((C.Coverage - 0.6) / 0.25) * saturate((C.Type - 0.4) / 0.3);
}

// How much the layer is a deck, 0..1: coverage over ~0.9, or a storm's (CloudBaseRough); under ~0.55 fair weather
float CloudDeckness(CloudParams C)
{
    return max(smoothstep(0.55, 0.9, C.Coverage), CloudBaseRough(C));
}

// The coverage at a place.
// A deck: the preset's, varied by the large-scale pattern (~50 km) and a second octave (~13 km); a storm's swings
// harder, from broken to solid (dark cores and brighter places between; a uniform deck read flat).
// Fair weather: fields of cloud with clear sky between them, and inside a field the clouds the noise makes at a natural
// density (~0.5). The preset's coverage decides how much of the sky lies in fields (a presence pattern of ~50, 13 and
// 7 km), so less of it gives fewer clouds, not thinner ones: lowered evenly everywhere, it left only the shape noise's
// peaks, many small puffs that flickered in the history. (Single clouds placed in 3.2 km cells instead looked drawn:
// each the same flat-topped dome on a lens base, evenly spaced, too steep for the billows to carve; user, "studio
// ghibli".)
float CloudCoverageAt(CloudParams C, Texture3D<float4> lut, float2 xz)
{
    float n = lut.SampleLevel(CloudWrap, float3(xz / CLOUD_COVER_TILE, C.Evolution / 4000.0), 0).r;
    float n2 = lut.SampleLevel(CloudWrap, float3(xz / (CLOUD_PERIOD / 19.0), 0.47 + C.Evolution / 3000.0), 0).r;
    float deckCover = saturate(C.Coverage + (n - 0.5) * (0.4 + 0.6 * CloudBaseRough(C)) + (n2 - 0.5) * 0.35);
    float deck = CloudDeckness(C);
    if (deck >= 1.0)
        return deckCover;
    float n3 = lut.SampleLevel(CloudWrap, float3(xz / (CLOUD_PERIOD / 37.0), 0.83 + C.Evolution / 2000.0), 0).r;
    // The octaves' sum spread to ~0.5 ± 0.15; the share of the sky in fields is coverage × 1.3 (inside a field the
    // clouds cover about three quarters of it), its threshold the normal quantile, linearised. Inside, ~0.6 (± 0.15 from
    // field to field): at 0.5 the noise's clouds broke into many small cloudlets.
    float presence = 0.5 + (0.25 * n + 0.45 * n2 + 0.3 * n3 - 0.5) * 2.5;
    float threshold = 0.5 + (0.5 - saturate(C.Coverage * 1.3)) * 0.4;
    float inField = smoothstep(threshold - 0.02, threshold + 0.06, presence);
    float fair = inField * (max(0.6, C.Coverage) + (n3 - 0.5) * 0.3);
    return lerp(fair, deckCover, deck);
}

// How deep a place lies in one tower's envelope (0 outside, 1 on its axis): a column from the base to its top, leaning
// downwind with height; a dome over the top of a growing tower; a mature one spreads an anvil over its top fifth, pushed
// downwind, with a dome over its updraft above it
// turret (−0.5..0.5, a ~2 km noise at the place) bulges and dents the column and the anvil's rim, so a tower is a stack
// of turrets rather than a cylinder; lobe (−0.5..0.5, the ~6 km base shape) swells and pinches it in broad masses
// (with turrets alone a tower read as a chimney); pouch (0..1, round blobs ~1.4 km apart) hangs mammatus under a mature
// anvil. x the column, y the anvil (CloudFieldAt keeps an anvil's sheet whole).
float2 CloudTowerEnvelope(CloudParams C, CloudCell k, float2 center, float2 xz, float h, float turret, float lobe, float pouch)
{
    float top = k.Top;
    float span = max(top - C.Base, 1.0);
    float hf = (h - C.Base) / span;
    if (hf < -0.05 || hf > 1.12)
        return 0.0;
    float2 axis = center + C.WindDir * ((h - C.Base) * 0.12);          // the wind shears the tower
    float2 rel = xz - axis;
    float d = length(rel) * (1.0 - 0.9 * turret - 0.6 * lobe);

    // The column: broad through its middle, narrowing above, rounding off over the top (flatter under an anvil)
    float dome = lerp(0.3, 0.1, k.Anvil);
    float bulge = 0.9 + 0.4 * sin(saturate(hf * 1.2) * 3.14159);
    float r = k.Radius * bulge * sqrt(saturate((1.0 - hf) / dome));
    float column = r > 0.0 ? saturate((r - d) / (0.4 * r)) : 0.0;
    // The base: flat on the whole, ragged where the turrets hang lower (a flat smooth base read as a dark lens from afar)
    column *= saturate((h - C.Base + 250.0 * saturate(turret + 0.2)) / 80.0);

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
        // Mammatus: pouches hanging from the underside downwind of the tower, inside the rim, up to ~8 % of the tower
        // (~700 m under a 9 km storm); at 6 % × pouch² they were too shallow to break the flat underside, and a √ of
        // the blob's rise flattened their tops into facets with creases between
        under -= 0.08 * smoothstep(0.2, 0.95, pouch) * saturate(uu * 2.0) * saturate((1.0 - e) / 0.4) * k.Anvil;
        float sheet = saturate((hf - under) / 0.03) * saturate((1.0 - hf) / 0.03);
        // Full out to near its rim: at (1 − e) / 0.35 most of the sheet sat under 1 and the noise ate it to a shelf
        anvil = saturate((1.0 - e) / 0.12) * sheet * k.Anvil;
        float rt = k.Radius * 0.55 * k.Anvil;
        float dome2 = rt * sqrt(saturate((1.09 - hf) / 0.1));
        column = max(column, hf > 0.9 && dome2 > 0.0 ? saturate((dome2 - d) / (0.5 * dome2)) : 0.0);
    }
    return float2(column, anvil);
}

float2 CloudTowers(CloudParams C, float2 xz, float h, float turret, float lobe, float pouch)
{
    if (C.Cells == 0 || C.TowerShare <= 0.0)
        return 0.0;
    StructuredBuffer<CloudCell> cells = ResourceDescriptorHeap[C.Cells];
    int n = (int)C.CellsPerSide;
    int2 home = (int2)floor(xz / C.CellSize);
    float2 env = 0.0;
    // 5 × 5: an anvil reaches up to ~1.5 cells from its tower
    [loop] for (int j = -2; j <= 2; j++)
    {
        [loop] for (int i = -2; i <= 2; i++)
        {
            int2 c = home + int2(i, j);
            int2 w = ((c % n) + n) % n;
            CloudCell k = cells[w.y * n + w.x];
            if (k.Radius <= 0.0 || h < C.Base - 250.0 || h > C.Base + (k.Top - C.Base) * 1.12)
                continue;
            // Out of the tower's reach (its widest column with the turrets, the shear, the anvil's wedge): skip it
            float2 center = float2(c) * C.CellSize + k.Center;
            float reach = k.Radius * (2.6 + 3.5 * k.Anvil) + 0.12 * (k.Top - C.Base);
            float2 off = xz - center;
            if (dot(off, off) > reach * reach)
                continue;
            env = max(env, CloudTowerEnvelope(C, k, center, xz, h, turret, lobe, pouch));
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
    // Fair weather: a finer octave (P/72, ~3.6 km) over the base shape, its place pushed about by a smooth noise of a
    // tile that shares no factor with either (P/11, ~23 km). The 6.4 km shape alone stood the same pair of clouds in
    // every 6.4 km square (a lattice across the sky) and made slabs several km wide. A storm's deck and the towers keep
    // the 6.4 km shape (rough decks, the towers' lobes).
    float rough = CloudBaseRough(C);
    float4 shapeNoise = n;
    if (rough < 1.0)
    {
        float2 warp = float2(lut.SampleLevel(CloudWrap, float3(xz / (CLOUD_PERIOD / 11.0), 0.61), 0).r,
                             lut.SampleLevel(CloudWrap, float3(xz / (CLOUD_PERIOD / 11.0), 0.13), 0).r) - 0.5;
        float2 fineXZ = xz + warp * 3000.0;
        float4 fine = lut.SampleLevel(CloudWrap, (float3(fineXZ.x, h, fineXZ.y) + slide) / (CLOUD_PERIOD / 72.0), 0);
        shapeNoise = lerp(0.6 * fine + 0.4 * n, n, rough);
    }

    float cover = CloudCoverageAt(C, lut, xz);
    // A rough base dips by up to CLOUD_BASE_DIP in broad swells (~5 km), the Perlin FBM sliding with the clouds
    float dip = CLOUD_BASE_DIP * CloudBaseRough(C)
              * saturate((lut.SampleLevel(CloudWrap, float3(xz.x + slide.x, 0.37, xz.y + slide.z) / (CLOUD_PERIOD / 50.0), 0).r - 0.35) / 0.3);
    // Turrets: the Perlin-Worley at a 2.1 km tile (CLOUD_PERIOD / 120), for the towers' bulges
    float turret = lut.SampleLevel(CloudWrap, (float3(xz.x, h, xz.y) + slide) / (CLOUD_PERIOD / 120.0), 0).a - 0.4;
    // Fair weather: the base swells ±150 m over a few km (neighbouring clouds at their own heights), and fades in over
    // ~350 m so the shape noise lowers each underside in broad lumps (every base at one height, dead flat, read as cut
    // off; user)
    float fair = 1.0 - CloudDeckness(C);
    float swell = fair > 0.0
        ? (lut.SampleLevel(CloudWrap, float3(xz / (CLOUD_PERIOD / 60.0), 0.71), 0).r - 0.5) * 300.0 * fair : 0.0;
    swell = max(swell, C.SlabBottom + 20.0 - C.Base);       // the slab has room below the base only for towers' bases
    float bottom = C.Base + swell - dip;
    float layer = cover * CloudLayerProfile((h - bottom) / max(C.Thickness + dip - swell, 1.0), C.Type);
    layer *= lerp(1.0, smoothstep(0.0, 350.0, h - bottom), fair);
    // Mammatus pouches: the Worley FBM's blobs (high at the cells' centres) at a ~1.4 km tile, flat in height
    // (a mip down: the Worley cells' creases blurred, so each pouch is a smooth dome)
    float pouch = saturate((lut.SampleLevel(CloudWrap, float3(xz.x + slide.x, 0.0, xz.y + slide.z) / (CLOUD_PERIOD / 180.0), 1.5).g - 0.5) / 0.35);
    float2 towers = CloudTowers(C, xz, h, turret, n.a - 0.45, pouch);
    float env = max(layer, max(towers.x, towers.y));
    if (env <= 0.002)
        return 0;

    float fbm = shapeNoise.g * 0.75 + shapeNoise.b * 0.25;
    float shape = saturate(CloudRemap(shapeNoise.a, fbm - 1.0, 1.0, 0.0, 1.0));
    // The LUT's shape sits in a narrow band (~0.45-0.8): spread over 0..1, so coverage thins the clouds gradually
    shape = saturate(CloudRemap(shape, 0.42, 0.82, 0.0, 1.0));
    // An anvil is a sheet, smooth underneath (where mammatus hang): the shape may not hole it (it came out like lace)
    shape = max(shape, 0.6 * saturate(towers.y * 1.5));
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
    // The coarsest level wraps the whole period: a path no longer than half of it (128 km), or a low sun's path along
    // the slab went round the torus and met the same storms again toward the light (tower tops dark at sunset)
    bool inReach = BakeLevel < CLOUD_LEVELS - 1 || asfloat(LightStepBits) <= CLOUD_PERIOD * 0.25;
    if (inReach && CloudInLevelBox(C, BakeLevel, L, q))
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
    // Billows (the inverted Worley: round puffs) on heaped clouds all the way down, a storm's base too (rolls and lumps
    // hanging down, ref 2); wisps only along the base of a stratus-like layer (~300 m). On cumulus the wisps' band near the
    // base carved where the billows above filled out: a waist ~100 m up, every cloud two stacked pancakes (user).
    float billowy = max(CloudBaseRough(C), saturate(C.Type * 2.0 - 1.0));
    float erode = lerp(lerp(detail, 1.0 - detail, billowy), 1.0 - detail, saturate((h - C.Base) / 300.0));
    // Fair-weather cumulus sharper than the volumes' ramp (CLOUD_RAMP, ~150 m): their edges are where the billows are.
    // A storm's deck keeps the soft ramp (sharp, its base read as popcorn)
    return saturate((field.x - (erode - 0.35) * CLOUD_BILLOW * strength - 0.5) / (CLOUD_RAMP * lerp(0.6, 1.0, CloudBaseRough(C))));
}

// Optical depth toward the light from a place: three short steps through the detail (the billows, ~400 m, shade each
// other) out to CLOUD_SHORT_LIGHT metres, then the light volume from there on (the whole way out of the slab: a tower
// shades the cloud behind it, a storm far toward a low sun the near deck)
#define CLOUD_SHORT_LIGHT 560.0
float CloudLightDepth(CloudParams C, Texture3D<float4> lut, float3 rel, float3 lightDir, float jitter, bool shortSteps)
{
    float tau = 0.0;
    float3 far = rel;
    if (shortSteps)
    {
        // Out past a texel and a half of the light volume where it is coarse (far away its texels are 1-2 km): read
        // within the cloud's own body, its layers (~400 m there) striped the faces a low sun lights
        float blendHere;
        uint levelHere = CloudLevelAt(C, max(abs(rel.x), abs(rel.z)), blendHere);
        float reach = max(CLOUD_SHORT_LIGHT, 3.0 * C.Levels[levelHere].VoxelSize) / CLOUD_SHORT_LIGHT;
        const float at[3] = { 40.0, 160.0, 400.0 };
        const float len[3] = { 80.0, 160.0, 320.0 };
        [unroll] for (int i = 0; i < 3; i++)
        {
            float3 p = rel + lightDir * (at[i] * reach * (0.8 + 0.4 * jitter));
            float h = CloudHeight(p);
            float2 xz = C.CameraCloud + p.xz;
            float dist = max(abs(p.x), abs(p.z));
            tau += CloudErode(C, lut, CloudSample(C, xz, h, dist, CloudWrap), xz, h, dist) * len[i] * reach;
        }
        tau *= C.Density;
        far = rel + lightDir * (CLOUD_SHORT_LIGHT * reach);
    }
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

// ── The ground shadow map (VOLUMETRIC_CLOUDS_PLAN §8) ──
// Where a ray toward the light crosses the slab's bottom, the optical depth along it out of the slab: every place below
// whose ray crosses there takes the same shadow (cloud_common.hlsli CloudShadowDepth). A toroidal window of
// CLOUD_SHADOW_TEXELS² in cloud space round ShadowCentre (Clouds.cs: the camera's own crossing), TargetHeight of its rows
// from row RegionZ0Bits a dispatch. Marched through the baked field with the billows (CloudErode: the shapes the view
// sees), empty bricks leapt; once the beam is gone (τ > 12) the rest of the way from the light volume there (its optical
// depth out of the slab), so a storm keeps its τ for the diffused share without a march through all of it.
#define CLOUD_SHADOW_STEPS 96
[numthreads(8, 8, 1)]
void CSCloudShadowMap(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= TargetWidth || id.y >= TargetHeight)
        return;
    CloudParams C = LoadClouds();
    Texture3D<float4> lut = ResourceDescriptorHeap[C.NoiseLut];
    int size = (int)TargetWidth;
    int2 phys = int2((int)id.x, (int)((asuint(RegionZ0Bits) + id.y) % TargetWidth));
    // The physical texel's place in the window round the centre (the window wraps as it follows the camera)
    int2 origin = int2(floor(C.ShadowCentre / C.ShadowTexel)) - size / 2;
    int2 logical = origin + (((phys - origin) % size) + size) % size;
    float2 rel0 = CloudWrapDelta((float2(logical) + 0.5) * C.ShadowTexel - C.CameraCloud);

    // The path rises with the planet's curvature: h = bottom + s·L.y + a·s², a = (1 − L.y²) / 2R; to the slab's top
    // (capped at 40 km; a light at the horizon would run on)
    float3 L = C.LightDir;
    float a = (1.0 - L.y * L.y) / (2.0 * CLOUD_EARTH_RADIUS);
    float rise = C.SlabTop - C.SlabBottom;
    float ly = max(L.y, 0.0);
    float sMax = min(2.0 * rise / (ly + sqrt(ly * ly + 4.0 * a * rise)), 40000.0);

    float tau = 0.0, s = 0.0;
    [loop] for (int k = 0; k < CLOUD_SHADOW_STEPS && s < sMax; k++)
    {
        float2 relXZ = rel0 + L.xz * s;
        float h = C.SlabBottom + L.y * s + a * s * s;
        if (h < C.SlabBottom)
        {
            s += 500.0;             // a light under the horizon: its path dips under the slab first
            continue;
        }
        float dist = max(abs(relXZ.x), abs(relXZ.y));
        float blend;
        uint level = CloudLevelAt(C, dist, blend);
        float2 xz = C.CameraCloud + relXZ;
        if (CloudBrickMax(C, level, xz, h) <= 0.0 && (blend <= 0.0 || CloudBrickMax(C, level + 1, xz, h) <= 0.0))
        {
            s += CloudBrickExit(C, level, xz, h, L) + 0.5;
            continue;
        }
        // A sample at the next point of the level's step grid along this path (s = 0 at the texel): the steps double from
        // level to level, so a coarser grid's points are every other point of the finer one's. Through a level's blend
        // zone the points the coarser grid lacks fade out (weight 1 − blend) and the others take their share (1 + blend),
        // so at the reach the march is the coarser level's already. The samples sit where they sat whatever level the
        // camera's distance picks or wherever a leap left s. Stepped on from s at the level's step instead (375-750 m
        // far out, through clouds with ~150 m edges), a sample hit a cloud or missed it by where the march came from: a
        // level boundary moved by a 10 m camera step or by the clouds drifting through it, or a brick going empty as they
        // changed shape, moved the samples of every texel behind it, and the map's τ flipped by tens in blocks (the
        // beam's edges popped; the clouds' forward light showed the rest as blotches of shade; user, 2026-10-22)
        float step = C.Levels[level].VoxelSize * 0.75;
        float g = floor(s / step + 1e-3) + 1.0;
        float sk = g * step;
        float2 relK = rel0 + L.xz * sk;
        float distK = max(abs(relK.x), abs(relK.y));
        float blendK;
        uint levelK = CloudLevelAt(C, distK, blendK);
        bool odd = frac(g * 0.5) > 0.25;
        float weight = 1.0 + (odd ? -blendK : blendK);
        if (levelK > level)
            weight = odd ? 0.0 : 2.0;   // past the reach: the coarser grid's points alone, as the next step takes them
        else if (levelK < level)
            weight = 1.0;               // back into a finer level (a path toward the camera): this coarse step's own
        float2 xzk = C.CameraCloud + relK;
        float hk = C.SlabBottom + L.y * sk + a * sk * sk;
        tau += CloudErode(C, lut, CloudSample(C, xzk, hk, distK, CloudWrap), xzk, hk, distK) * C.Density * step * weight;
        s = sk;
        if (tau > CLOUD_SHADOW_MARCH_TAU)
            break;
    }
    if (s < sMax)
    {
        float2 relXZ = rel0 + L.xz * s;
        float h = max(C.SlabBottom + L.y * s + a * s * s, C.SlabBottom);
        float blend;
        uint level = CloudLevelAt(C, max(abs(relXZ.x), abs(relXZ.y)), blend);
        float2 xz = C.CameraCloud + relXZ;
        float rest = CloudLightDepthAt(C, level, xz, h, CloudWrap);
        if (blend > 0.0)
            rest = lerp(rest, CloudLightDepthAt(C, level + 1, xz, h, CloudWrap), blend);
        tau += rest;
    }
    RWTexture2D<float> target = ResourceDescriptorHeap[TargetUAVIdx];
    target[phys] = tau;
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

struct CloudMarchResult
{
    float4 color;           // rgb the clouds' light with the air in front of them (× Exposure), a what they let through
    float depth;            // metres to where their light comes from (transmittance-weighted); CLOUD_FAR_DEPTH if none
};

#define CLOUD_FAR_DEPTH 20000.0

// The lit ground's light under a place in the clouds (groundAmbient: E / π × albedo × Exposure, lit through the deck's
// mean round the camera, deckThrough): as much as gets through the cloud over that ground, taken as the place's optical
// depth toward the light (tau), never more than the mean gives. Under a storm cell the mean (~0.7 with sunny gaps round
// it) lit the ground several times too bright and its warm bounce made the base tan.
float3 CloudGroundLight(float3 groundAmbient, float deckThrough, float tau)
{
    return groundAmbient * saturate(CloudThrough(tau) / deckThrough);
}

// How open the sky is over the ground a face sees far out, from a place p (camera-relative): the mean of exp(−0.3 τ up)
// over columns at the place's height, read from a coarse level (≥ 500 m texels: a column's edge fades over a texel, not a
// voxel of level 0). A deck or a storm round the place covers them all; broken cumulus leave a share open.
// A face seen low faces the eye and sees the ground out toward it (towardEye: level, unit): nine columns 4-12 km out and
// ±4 km across. A face seen from below (a base overhead) faces down and sees the ground all round it: eight columns 6 km
// round the place. eyeShare (0..1, from the view's horizontal share) weighs the two. Toward the eye alone, overhead the
// direction turned with the view's azimuth round the zenith, and every base overhead took a fan of sharp wedges from the
// columns it swept (the "zenith prism", user 2026-10-22; first taken for the march's brick walls).
// One column 8 km out decided it before, open or covered: every point of a base took its own point out there, and the
// other clouds' footprints landed on the bases as hard navy patches and stripes that slid as the camera moved (user,
// 2026-10-20)
#define CLOUD_FAR_OPEN_LEVEL 3
float CloudFarOpenAt(CloudParams C, float3 p, float2 o)
{
    float3 q = p + float3(o.x, 0.0, o.y);
    float blend;
    uint level = max(CloudLevelAt(C, max(abs(q.x), abs(q.z)), blend), (uint)CLOUD_FAR_OPEN_LEVEL);
    return exp(-0.3 * CloudSkyDepthAt(C, level, C.CameraCloud + q.xz, CloudHeight(q), CloudWrap).x);
}

float CloudFarOpen(CloudParams C, float3 p, float2 towardEye, float eyeShare)
{
    float ring = 0.0, toward = 0.0;
    if (eyeShare < 1.0)
    {
        [unroll] for (int k = 0; k < 8; k++)
        {
            float a = (k + 0.5) * (ATM_PI / 4.0);
            ring += CloudFarOpenAt(C, p, float2(cos(a), sin(a)) * 6000.0);
        }
    }
    if (eyeShare > 0.0)
    {
        float2 across = float2(-towardEye.y, towardEye.x);
        [unroll] for (int i = 0; i < 3; i++)
        {
            [unroll] for (int j = 0; j < 3; j++)
                toward += CloudFarOpenAt(C, p, towardEye * (4000.0 * (i + 1)) + across * (4000.0 * (j - 1)));
        }
    }
    return lerp(ring / 8.0, toward / 9.0, eyeShare);
}

// One view ray through the clouds (not the rain: CloudMarchRay). stepScale > 1 marches coarser (the probe's faces);
// jitter (0..1) offsets the steps; tLimit (m along the ray) ends it at the scene's surface: a summit in a deck, a hill in
// front of the clouds.
CloudMarchResult CloudMarchClouds(CloudParams C, AtmosphereParams A, Texture3D<float4> lut, float3 dir, float jitter,
                                  float stepScale, float tLimit)
{
    CloudMarchResult result;
    result.color = float4(0, 0, 0, 1);
    result.depth = CLOUD_FAR_DEPTH;
    float tStart, tEnd;
    if (!CloudSlabSegment(C, dir, tStart, tEnd))
        return result;
    tEnd = min(tEnd, tLimit);
    if (tEnd <= tStart)
        return result;

    // The light the volumes were made for: the sun, or after it has set at the clouds too the brightest moon
    float3 lightDir = C.LightDir;
    float3 lightE = C.LightIsMoon != 0 ? A.MoonIlluminance : A.SunIlluminance;
    float cosTheta = dot(dir, lightDir);
    float phase[3];
    [unroll] for (int o = 0; o < 3; o++)
        phase[o] = CloudPhase(cosTheta, pow(0.5, o));

    StructuredBuffer<float4> ambientBuffer = ResourceDescriptorHeap[A.Ambient];
    float3 skyAmbient = ambientBuffer[0].rgb, groundAmbient = ambientBuffer[1].rgb;     // E / π × Exposure
    // The ground's light is lit through the clouds' mean round the camera (CSAmbient): under a cloud it is what gets
    // through there instead (CloudGroundLight)
    float deckThrough = max(ambientBuffer[4].y, 0.05);
    // The terrain's shadow toward the light (a peak shades the cloud behind it): only below the terrain's top
    float terrainTop = A.TerrainHeightmap != 0 ? A.TerrainOrigin.y + A.TerrainMaxHeight : -1e9;
    bool moonLight = C.LightIsMoon != 0;

    float3 L = 0;
    float T = 1.0;
    float depthSum = 0.0, depthWeight = 0.0;
    // The far ground's openness (CloudFarOpen): a wide mean, worked out once a ray (at the first face it meets, and again
    // every 2 km along it), not per sample
    float2 towardEye = -dir.xz * rsqrt(max(dot(dir.xz, dir.xz), 1e-6));
    // How much the face seen faces the eye rather than down (CloudFarOpen): none above ~65°, all below ~25°
    float eyeShare = smoothstep(0.3, 0.9, length(dir.xz));
    float farOpen = -1.0, farAt = 0.0;
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

        if (CloudBrickMax(C, level, xz, h) <= 0.0 && (blend <= 0.0 || CloudBrickMax(C, level + 1, xz, h) <= 0.0))
        {
            // Out of the empty brick (a slab: wide across, thin in height) and just into the next: a brick's layers can
            // be a few metres thick, so any more skips the base of the cloud in it
            t += CloudBrickExit(C, level, xz, h, dir) + 0.5;
            continue;
        }

        // Near the camera (in a cloud) short steps, so a whiteout fades in instead of popping at the first long step;
        // never past the scene's surface
        float step = voxel * 0.6 * stepScale * (0.75 + 0.5 * jitter);
        step = min(min(step, max(4.0, 0.25 * t)), tEnd - t);
        // Anywhere in the step, by the jitter: steps restart at the wall where an empty brick is left, so at mid-step every
        // ray read the cloud's base at a depth set by which wall it came through. Seen from below, the bricks' vertical
        // walls run out from the zenith: a fan of sharp wedges over every cloud overhead, standing still as the cloud
        // drifted through the camera's bricks (user, 2026-10-21). The jitter changes every frame; the history averages it.
        float3 p = dir * (t + step * jitter);
        float hp = CloudHeight(p);
        float2 xzp = C.CameraCloud + p.xz;
        float2 field = CloudSample(C, xzp, hp, dist, CloudWrap);
        float density = CloudErode(C, lut, field, xzp, hp, dist);
        if (density > 0.0)
        {
            float sigmaT = density * C.Density;
            float sigmaS = sigmaT * C.Albedo;

            // The light reaching this place through the air above (faded out in the earth's shadow) and through the cloud
            float terrainLit = hp < terrainTop ? AtmTerrainLightVisibility(A, CamPos + p, moonLight) : 1.0;
            float3 toLight = CloudLightThroughAir(A, CloudAtmospherePos(p), lightDir) * terrainLit;
            float tau = any(toLight > 0.0) ? CloudLightDepth(C, lut, p, lightDir, jitter, stepScale < 1.5) : 0.0;
            float2 skyTau = CloudSkyDepth(C, p, CloudWrap);
            // The light diffused down to here entered the cloud above, through less air than a beam down to this height
            // takes: its colour is the air's transmittance up there (a storm's base lit with the sun as reddened by the
            // low air came out sepia). That height from the optical depth over the place: at least τ / CloudDensity of
            // cloud, taken as half full.
            float entry = min(hp + 2.0 * skyTau.x / max(C.Density, 1e-4), C.SlabTop) - hp;
            float3 toLightDiffused = CloudLightThroughAir(A, CloudAtmospherePos(p + float3(0.0, entry, 0.0)), lightDir) * terrainLit;
            float beam = 0.0;
            [unroll] for (int o = 0; o < 3; o++)
                beam += pow(0.5, o) * phase[o] * exp(-pow(0.5, o) * tau);
            // Deep in a thick cloud the light has diffused: it falls as 1 / (1 + 0.75 (1 − g) τ) (two-stream, g 0.85,
            // sky_common.fx CloudThrough), not exponentially, which is why an overcast deck's base is grey and not black.
            // Of the flux the light brings in through a deck's top, E × sin(its elevation), that share leaves the base as
            // diffuse light: radiance CloudThrough × E sin / π. The ground, the air and the meter take the same share
            // (CloudTransmission), so the base the sky probe sees lights the surfaces with what the ground's light says
            // comes through. (Until stage 4: 1 / (4π), a base ~3× too dark at noon against that.)
            // A deck is a slab: the diffused light fills it and passes CloudThrough down through its base (a storm's deck
            // without it inside went the sky's blue), of the flux through its top, E sin(elevation). Broken cumulus
            // (coverage under ~0.55; decks over ~0.9, between them a blend) are lit by it where it leaves them, as the eye
            // looks along the light: toward it (a base seen from below at noon, a cloud against the low sun) the light
            // passing through, out of the underside (near it, with little cloud below; 40 % of the slab's share, a lone
            // cumulus loses much of it through its sides: grey bases under bright edges); away from it (tops from above,
            // the sunlit side of a cloud at sunset) what the cloud reflects, 1 − CloudThrough of the column, most where
            // the light gets in (creases keep their relief), of the full flux into the faces toward the light (a deck's
            // sin(elevation) left the sides a low sun lights grey, not gold).
            // Everywhere at full strength it lay over the beam's relief as an even white: the sunlit faces took a
            // Lambertian white on top of the beam, the shaded flanks most of it, and fair-weather cumulus came out flat.
            // The light passing through takes the air's colour where it entered above (toLightDiffused); the reflected
            // light entered at the face, as reddened as the sample's own (from higher up, a low sun's gold came out
            // white).
            float seeThrough = saturate(0.5 + 1.5 * cosTheta);
            float deck = smoothstep(0.55, 0.9, C.Coverage);
            float passing = lerp(seeThrough * 0.4 * CloudThrough(tau) * exp(-0.2 * skyTau.y), CloudThrough(tau), deck)
                          * saturate(lightDir.y);
            float reflected = (1.0 - deck) * (1.0 - seeThrough) * (1.0 - CloudThrough(tau + skyTau.y)) * exp(-0.1 * tau);
            float3 scattered = toLight * (beam * CloudInScatter(C, density, hp, cosTheta) + reflected / ATM_PI)
                             + toLightDiffused * (passing / ATM_PI);

            // The sky above and the lit ground below, as much of each as the cloud over and under this place leaves
            // (diffuse light gets deeper than a beam): bright tops, dark cores, the ground's warm light on the bases.
            // Near the face we see, also what that face sees out toward the camera: a tower's flank half the sky (its
            // blue shade), a deck's base none of it, only the ground
            float faceOpen = pow(T, 0.3);
            // The face sees the sky only where the air in front of it is open to the sky: a deck's base seen low sees
            // more base, not the blue (half the blue sky on every grazing base turned a storm deck blue). The sky
            // volume's column over a place ~a kilometre toward the eye says which.
            // The column ~1 km toward the eye, and how open the sky is 4-12 km out (CloudFarOpen): under an anvil the near
            // one is covered while the face still sees the bright low sky past the anvil's edge (faces under anvils came
            // out black); a deck covers both
            // A base faces down whatever way it is seen: near the base no sky (it lit fair-weather cumulus bases seen
            // 20° up with a third of the sky, and they came out as white as their sunlit sides)
            float faceSky = saturate(0.5 - 0.5 * dir.y) * faceOpen * saturate((hp - C.Base) / 400.0);
            if (faceOpen > 0.01)
            {
                float nearSky = exp(-0.3 * CloudSkyDepth(C, p - dir * min(1000.0, 0.5 * t), CloudWrap).x);
                if (farOpen < 0.0 || t - farAt > 2000.0)
                {
                    farOpen = CloudFarOpen(C, p, towardEye, eyeShare);
                    farAt = t;
                }
                faceSky *= max(nearSky, farOpen);
            }
            // The sky's light also diffuses down through the cloud over this place, as the sun's does (CloudThrough of the
            // optical depth above): under a deck it is part of the base's light, and without it everything under a storm
            // was the sun's colour alone (a tan haze, a sepia base)
            float skySeen = max(max(exp(-0.3 * skyTau.x), CloudThrough(skyTau.x)), faceSky);
            float groundSeen = max(exp(-0.3 * skyTau.y), saturate(0.5 + 0.5 * dir.y) * faceOpen);
            // What lies below: low down the ground; high up (an anvil, a tower's upper flanks) the lit air and the sunlit
            // cloud tops under it, much brighter than the ground (lit only by the ground, anvil undersides went black)
            float3 belowAir = 0.35 * skyAmbient + 0.15 * lightE * toLight * (saturate(lightDir.y) / ATM_PI * Exposure);
            // (the ground: as lit under this place, or as the open ground far out the face sees, whichever is brighter)
            float3 ground = max(CloudGroundLight(groundAmbient, deckThrough, tau), groundAmbient * (max(farOpen, 0.0) * faceOpen));
            float3 below = lerp(ground, belowAir, saturate((hp - C.Base - 0.5 * C.Thickness) / 4000.0));
            float3 ambient = skyAmbient * skySeen + below * groundSeen;

            float3 S = sigmaS * (lightE * scattered * Exposure + ambient);
            // Lightning in the sky probe's clouds (so the land brightens through the ambient): the flash's light spread
            // through the cloud round it (its illuminance over the distance², as it diffuses through the cloud's mean
            // extinction, scattered evenly). A view's clouds take it per pixel instead (sky_common.fx CloudsAt): their
            // targets keep a history a stroke of 25 ms would not show in.
            if (C.FlashIntensity > 0.0 && stepScale > 1.5)
            {
                float3 toFlash = p - C.FlashPos;
                float d2 = max(dot(toFlash, toFlash), 40000.0);
                S += sigmaS * (C.FlashIntensity / d2 * CloudThrough(C.FlashSpread * sqrt(d2)) / (4.0 * ATM_PI) * Exposure)
                   * CLOUD_FLASH_COLOR;
            }
            float x = sigmaT * step;
            float through = exp(-x);
            L += T * S * (x > 1e-4 ? (1.0 - through) / sigmaT : step);
            depthSum += (t + step * 0.5) * (T - T * through);
            depthWeight += T - T * through;
            T *= through;
            if (T < 0.005)
            {
                // Opaque from here: what is left of the light behind (the sun's disc, ~10⁶ pre-exposed, 0.5 % of it a
                // white disc through the thickest cumulus) is gone. The cloud's own light past this point is < 0.5 %.
                T = 0.0;
                break;
            }
        }
        t += step;
    }

    if (T >= 0.999)
    {
        result.depth = 0.5 * (tStart + tEnd);
        return result;
    }

    // The clouds' own light: the air in front of them is the caller's (the sky dome takes it from the aerial perspective
    // volume, with the shadows in it; the probe from CloudAirInFront)
    result.color = float4(L, T);
    result.depth = depthSum / max(depthWeight, 1e-6);
    return result;
}

// ── Rain shafts (VOLUMETRIC_CLOUDS_PLAN §15) ──
// Under the raining cells nearest the camera (Clouds.cs PlaceRainShafts: CloudParams.RainShafts): a thin medium from the
// clouds' base down to the ground or, for virga, ending in the air, in curtains, leaning downwind as it falls. Not baked
// (the volumes start at the base: stretching them to the ground would spend their layers on rain): each shaft is a
// column the ray is intersected with and marched through in RAIN_STEPS. Drops throw the light on (HG g RAIN_PHASE_G),
// so a curtain glows toward the sun and stays grey away from it; the sun reaches it as the clouds' beam there (dark
// under the storm, bright where a low sun comes in under the anvil), the clouds' diffused light and the sky's and the
// ground's light as for the clouds.

#define RAIN_SIGMA 0.001        // extinction per metre in the heaviest rain (visibility ~3 km): heavier made a shaft a
                                // wall a few km across, black under a tower and a dark flat lens seen from afar
#define RAIN_PHASE_G 0.75
#define RAIN_STEPS 8
#define RAIN_LEAN 0.25          // metres downwind per metre fallen

struct CloudRainResult
{
    float3 L;                   // the rain's light (× Exposure)
    float T;                    // what it lets through
    float depthSum, depthWeight;    // its transmittance-weighted distance
};

CloudRainResult CloudRain(CloudParams C, AtmosphereParams A, Texture3D<float4> lut, float3 dir, float tLimit, float jitter)
{
    CloudRainResult r;
    r.L = 0;
    r.T = 1.0;
    r.depthSum = 0.0;
    r.depthWeight = 0.0;
    if (C.RainShafts == 0 || C.RainCount == 0)
        return r;
    StructuredBuffer<CloudRainShaft> shafts = ResourceDescriptorHeap[C.RainShafts];
    float3 lightDir = C.LightDir;
    float3 lightE = C.LightIsMoon != 0 ? A.MoonIlluminance : A.SunIlluminance;
    float phase = HenyeyGreenstein4Pi(dot(dir, lightDir), RAIN_PHASE_G) / (4.0 * ATM_PI);
    StructuredBuffer<float4> ambientBuffer = ResourceDescriptorHeap[A.Ambient];
    float3 skyAmbient = ambientBuffer[0].rgb, groundAmbient = ambientBuffer[1].rgb;
    float deckThrough = max(ambientBuffer[4].y, 0.05);
    float2 d2 = dir.xz;
    float a = dot(d2, d2);
    float2 d2n = a > 1e-8 ? d2 * rsqrt(a) : float2(1.0, 0.0);

    for (uint k = 0; k < C.RainCount; k++)
    {
        CloudRainShaft s = shafts[k];
        // The column round its lean (wider by the lean over the fall), between its bottom and the base
        float fall = C.Base - max(s.Bottom, -500.0);
        float2 center = s.Center + C.WindDir * (0.5 * fall * RAIN_LEAN);
        float radius = s.Radius + 0.5 * fall * RAIN_LEAN;
        float t0 = 0.0, t1 = tLimit;
        if (a > 1e-8)
        {
            float b = dot(d2, center), c = dot(center, center) - radius * radius;
            float disc = b * b - a * c;
            if (disc <= 0.0)
                continue;
            float sq = sqrt(disc);
            t0 = (b - sq) / a;
            t1 = min(t1, (b + sq) / a);
        }
        else if (dot(center, center) > radius * radius)
            continue;
        if (abs(dir.y) > 1e-5)
        {
            float tBase = (C.Base - CamPos.y) / dir.y, tBottom = (s.Bottom - CamPos.y) / dir.y;
            t0 = max(t0, min(tBase, tBottom));
            t1 = min(t1, max(tBase, tBottom));
        }
        else if (CamPos.y > C.Base || CamPos.y < s.Bottom)
            continue;
        t0 = max(t0, 0.0);
        if (t1 <= t0)
            continue;

        // How open the sky is round the shaft: the column over four places two radii out (once per shaft). Under a lone
        // tower the sky round it lights the curtains from the sides; under a deck nothing does (a shaft lit only through
        // the tower over it came out black against the white towers)
        float open = 0.0;
        [unroll] for (uint q = 0; q < 4; q++)
        {
            float2 o = float2(q == 0 ? 1.0 : (q == 1 ? -1.0 : 0.0), q == 2 ? 1.0 : (q == 3 ? -1.0 : 0.0)) * (2.0 * s.Radius);
            float3 around = float3(s.Center.x + o.x, C.Base - CamPos.y, s.Center.y + o.y);
            open += 0.25 * exp(-0.3 * CloudSkyDepth(C, around, CloudWrap).x);
        }

        float dt = (t1 - t0) / RAIN_STEPS;
        for (uint i = 0; i < RAIN_STEPS; i++)
        {
            float t = t0 + (i + jitter) * dt;
            float3 p = dir * t;
            float h = CloudHeight(p);
            // The shaft's edge where the rain at this height has drifted to, its curtains (the Worley FBM's cells
            // stretched up and down, moving with the cloud), heavier near the base, fading out at a virga's bottom
            float2 axis = s.Center + C.WindDir * ((C.Base - h) * RAIN_LEAN);
            // Curtains across the shaft as the camera sees it: the axis's offset from the ray (m) is the same all along
            // the ray at a height, so they are streaks that hold still in the view, not noise marched through (a 3D
            // pattern sampled 8 times over kilometres came out as a dithered smudge). Two octaves of the Worley FBM's
            // cells, stretched up and down, the finer ~250 m across, falling slowly
            float across = d2n.x * axis.y - d2n.y * axis.x;
            float fallen = h + C.Evolution * 6.0;
            float4 coarse = lut.SampleLevel(CloudWrap, float3(across / 1100.0 + s.Seed * 13.0, fallen / 3000.0, s.Seed), 0);
            float4 fine = lut.SampleLevel(CloudWrap, float3(across / 260.0 + s.Seed * 29.0, fallen / 1500.0, s.Seed + 0.5), 0);
            float streak = coarse.g * 0.75 + fine.b * 0.25;
            // Never quite clear between them (a hard barcode otherwise)
            float curtain = 0.3 + 0.7 * smoothstep(0.3, 0.7, streak);
            // A frayed edge (the curtains reach out past the mean radius and leave gaps inside it)
            float edge = saturate((s.Radius - length(p.xz - axis)) / (0.5 * s.Radius) + (streak - 0.5) * 0.8);
            // In from the base over ~400 m (no flat lid: the rain grows out of the cloud's shade), out at a virga's bottom
            float fade = (s.Fade > 0.0 ? saturate((h - s.Bottom) / s.Fade) : 1.0) * saturate((C.Base - h) / 400.0);
            float sigma = RAIN_SIGMA * s.Rain * edge * curtain * fade;
            if (sigma <= 1e-7)
                continue;

            // Its light: the clouds' beam and their diffused light at this place, the sky through them and the ground
            float tau = CloudShadowDepth(C, p, lightDir, CloudWrap);
            float beam = exp(-tau), through = CloudThrough(tau);
            float3 toLight = CloudLightThroughAir(A, CloudAtmospherePos(p), lightDir);
            // The diffused light comes down from the whole cloud base, a bright hemisphere overhead (irradiance E·through·
            // sin): drops scatter half of it on toward an eye looking across (an even scatterer: 0.5 / π of the
            // irradiance), more looking up into it (forward), less looking down. Taken as a point overhead (the phase at
            // the angle to straight down) the curtains came out ~3× too dark, sooty against the haze.
            float diffusedPhase = (0.5 + 0.35 * dir.y) / ATM_PI;
            // The sky's light through the column overhead, the ground's as lit under the cloud, half each (above, below);
            // the diffused light coloured by the air where it entered the cloud (as in CloudMarchClouds)
            float tauUp = CloudSkyDepth(C, p, CloudWrap).x;
            // From above through the column, from the sides as open as the sky round the shaft (half the sky's light)
            float skyThrough = max(CloudThrough(tauUp), 0.5 * open);
            float entry = min(C.Base + 2.0 * tauUp / max(C.Density, 1e-4), C.SlabTop) - h;
            float3 toLightDiffused = CloudLightThroughAir(A, CloudAtmospherePos(p + float3(0.0, entry, 0.0)), lightDir);
            float3 S = sigma * (lightE * (toLight * beam * phase
                                          + toLightDiffused * (max(through - beam, 0.0) * saturate(lightDir.y) * diffusedPhase)) * Exposure
                              + 0.5 * (skyAmbient * skyThrough
                                       + max(CloudGroundLight(groundAmbient, deckThrough, tau), 0.5 * open * groundAmbient)));
            float x = sigma * dt;
            float pass = exp(-x);
            r.L += r.T * S * (x > 1e-4 ? (1.0 - pass) / sigma : dt);
            r.depthSum += t * (r.T - r.T * pass);
            r.depthWeight += r.T - r.T * pass;
            r.T *= pass;
        }
    }
    return r;
}

// One view ray through the clouds and the rain under them, in the order the ray meets them (the rain is under the base:
// in front of the clouds seen from below, behind them seen from above). Arguments as CloudMarchClouds.
CloudMarchResult CloudMarchRay(CloudParams C, AtmosphereParams A, Texture3D<float4> lut, float3 dir, float jitter,
                               float stepScale, float tLimit)
{
    CloudMarchResult clouds = CloudMarchClouds(C, A, lut, dir, jitter, stepScale, tLimit);
    CloudRainResult rain = CloudRain(C, A, lut, dir, tLimit, jitter);
    if (rain.T >= 0.999)
        return clouds;
    float rainDepth = rain.depthSum / max(rain.depthWeight, 1e-6);
    float rainWeight = 1.0 - rain.T, cloudWeight = 1.0 - clouds.color.a;
    CloudMarchResult result;
    if (rainDepth <= clouds.depth || clouds.color.a >= 0.999)
    {
        result.color = float4(rain.L + rain.T * clouds.color.rgb, rain.T * clouds.color.a);
        result.depth = (rainDepth * rainWeight + clouds.depth * cloudWeight * rain.T) / max(rainWeight + cloudWeight * rain.T, 1e-6);
    }
    else
    {
        result.color = float4(clouds.color.rgb + clouds.color.a * rain.L, rain.T * clouds.color.a);
        result.depth = (clouds.depth * cloudWeight + rainDepth * rainWeight * clouds.color.a) / max(cloudWeight + rainWeight * clouds.color.a, 1e-6);
    }
    return result;
}

// The camera-relative view direction through a place in the view (uv 0..1; reverse-Z: the far plane is z = 0)
float3 CloudViewDir(float2 uv)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 farClip = mul(float4(ndc, 0, 1), CameraInverse);
    return normalize(farClip.xyz / farClip.w);
}

// The texel of a 2×2 block marched in frame f: Bayer order, so each frame's sparse set is spread evenly
uint2 CloudMarchOffset(uint f)
{
    const uint2 offsets[4] = { uint2(0, 0), uint2(1, 1), uint2(1, 0), uint2(0, 1) };
    return offsets[f & 3];
}

// MarchMode 0: every texel of the half-size target (TargetUAV, depth DepthUAV; the history is not used: the first frame,
// a resize, a jump, VolumetricCloudsFullRate). 1: one texel of each 2×2 block, into the quarter-size sparse targets
// (TargetUAV, DepthUAV, sized TargetWidth/2 × TargetHeight/2), for CSCloudReconstruct.
[numthreads(8, 8, 1)]
void CSCloudMarch(uint3 id : SV_DispatchThreadID)
{
    uint2 size = MarchMode == 0 ? uint2(TargetWidth, TargetHeight) : uint2((TargetWidth + 1) / 2, (TargetHeight + 1) / 2);
    if (any(id.xy >= size))
        return;
    RWTexture2D<float4> target = ResourceDescriptorHeap[TargetUAVIdx];
    RWTexture2D<float> depthTarget = ResourceDescriptorHeap[DepthUAVIdx];
    if (CloudsOn == 0 || AtmosphereIdx == 0)
    {
        target[id.xy] = float4(0, 0, 0, 1);
        depthTarget[id.xy] = CLOUD_FAR_DEPTH * 0.001;
        return;
    }
    CloudParams C = LoadClouds();
    AtmosphereParams A = LoadAtmosphere();
    Texture3D<float4> lut = ResourceDescriptorHeap[C.NoiseLut];

    uint2 pixel = MarchMode == 0 ? id.xy : min(id.xy * 2 + CloudMarchOffset(FrameIndex), uint2(TargetWidth, TargetHeight) - 1);
    float2 uv = (float2(pixel) + 0.5) / float2(TargetWidth, TargetHeight);
    float3 dir = CloudViewDir(uv);
    // Up to the scene's surface (CSCloudSceneDepth), its view depth along the ray
    float tLimit = 1e9;
    if (SceneDepthTexIdx != 0)
    {
        Texture2D<float> sceneDepth = ResourceDescriptorHeap[SceneDepthTexIdx];
        float viewDepth = sceneDepth.Load(int3(pixel, 0));
        if (viewDepth > 0.0)
            tLimit = viewDepth / max(dot(dir, normalize(float3(View._13, View._23, View._33))), 1e-3);
    }
    CloudMarchResult r = CloudMarchRay(C, A, lut, dir, CloudJitter(pixel, FrameIndex), 1.0, tLimit);
    target[id.xy] = r.color;
    depthTarget[id.xy] = r.depth * 0.001;      // km: half floats hold it
}

// The scene's view depth a half-size texel is marched to (TargetUAV, R32F, m; 0 = the sky) from the G-buffer's linear
// depth (SourceTexIdx): its 2×2 pixels' nearest surface in one texel, their farthest (the sky if any is sky) in the next,
// a checkerboard, so along a summit's silhouette there are clouds marched on both sides of it to choose from when a
// pixel is upsampled (Frostbite, UE5). The farthest looks a pixel past its block for the sky: a sky pixel reads its own
// texel and one neighbour across and one down (CloudsAt), and the neighbours of a "nearest" texel are "farthest" ones, so
// every sky pixel has the sky's clouds among its four. Without that the sky through a gap in a crown, whose four texels
// were all marched to the leaves, came out cloudless: bright specks in the trees, the sun's disc through them.
[numthreads(8, 8, 1)]
void CSCloudSceneDepth(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= TargetWidth || id.y >= TargetHeight)
        return;
    Texture2D<float> depth = ResourceDescriptorHeap[SourceTexIdx];
    uint width, height;
    depth.GetDimensions(width, height);
    bool takeFar = ((id.x + id.y) & 1) != 0;
    float nearest = 1e30, farthest = 0.0;
    bool sky = false;
    [unroll] for (int j = -1; j <= 2; j++)
    {
        [unroll] for (int i = -1; i <= 2; i++)
        {
            bool inBlock = i >= 0 && i <= 1 && j >= 0 && j <= 1;
            if (!inBlock && !takeFar)
                continue;
            int2 p = clamp(int2(id.xy) * 2 + int2(i, j), int2(0, 0), int2(width, height) - 1);
            float d = depth.Load(int3(p, 0));
            if (d <= 0.0)
                sky = true;
            else if (inBlock)
            {
                nearest = min(nearest, d);
                farthest = max(farthest, d);
            }
        }
    }
    RWTexture2D<float> target = ResourceDescriptorHeap[TargetUAVIdx];
    target[id.xy] = takeFar ? (sky ? 0.0 : farthest) : (nearest < 1e30 ? nearest : 0.0);
}

// Whether two scene depths (m, 0 = the sky) are the same surface for the clouds: within 10 %
bool CloudSameSurface(float a, float b)
{
    if (a <= 0.0 || b <= 0.0)
        return a <= 0.0 && b <= 0.0;
    return abs(a - b) <= 0.1 * max(a, b);
}

// A view's clouds this frame from the texels marched (one of each 2×2 block) and last frame's, reprojected: the place a
// texel's clouds were at last frame (the camera's move, the wind's), clamped to what this frame's marched neighbours on
// the same surface show (no ghost outlives a change), a marched texel blended with its own history to smooth the
// jitter. Off screen, without a history, or where the scene's surface there has changed (a summit's silhouette moved
// past): this frame's sparse samples on the same surface, interpolated. Writes the half-size target.
[numthreads(8, 8, 1)]
void CSCloudReconstruct(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= TargetWidth || id.y >= TargetHeight)
        return;
    Texture2D<float4> sparse = ResourceDescriptorHeap[SparseTexIdx];
    Texture2D<float> sparseDepth = ResourceDescriptorHeap[SparseDepthTexIdx];
    RWTexture2D<float4> target = ResourceDescriptorHeap[TargetUAVIdx];
    RWTexture2D<float> depthTarget = ResourceDescriptorHeap[DepthUAVIdx];
    uint2 sparseSize = uint2((TargetWidth + 1) / 2, (TargetHeight + 1) / 2);
    int2 q = int2(id.xy / 2);
    uint2 offset = CloudMarchOffset(FrameIndex);
    bool fresh = all((id.xy & 1) == offset);

    // The scene's surface at this texel and at each marched one (the half-size scene depth; all sky without it)
    Texture2D<float> sceneDepth = ResourceDescriptorHeap[SceneDepthTexIdx];
    bool hasScene = SceneDepthTexIdx != 0;
    uint2 maxTexel = uint2(TargetWidth, TargetHeight) - 1;
    float surface = hasScene ? sceneDepth.Load(int3(id.xy, 0)) : 0.0;

    // This frame's marched neighbours on the same surface: the clamp's box, their mean and their clouds' depth. None in
    // the 3×3 (a texel of sky in a tree's crown, its neighbours all marched to the leaves): the ring round it. Only where
    // that has none either all of them: samples marched to a leaf hold "no cloud", and a sky texel clamped to them let
    // its history go clear (the sky and the sun's disc through the gaps in the leaves, under a storm deck).
    float4 lo = 1e9, hi = -1e9, loAll = 1e9, hiAll = -1e9;
    float4 sameSum = 0;
    float sameDepth = 0.0, sameCount = 0.0;
    [unroll] for (int j = -1; j <= 1; j++)
    {
        [unroll] for (int i = -1; i <= 1; i++)
        {
            int2 qn = clamp(q + int2(i, j), int2(0, 0), int2(sparseSize) - 1);
            float4 s = sparse.Load(int3(qn, 0));
            loAll = min(loAll, s);
            hiAll = max(hiAll, s);
            float sd = hasScene ? sceneDepth.Load(int3(min(uint2(qn) * 2 + offset, maxTexel), 0)) : 0.0;
            if (CloudSameSurface(sd, surface))
            {
                lo = min(lo, s);
                hi = max(hi, s);
                sameSum += s;
                sameDepth += sparseDepth.Load(int3(qn, 0));
                sameCount += 1.0;
            }
        }
    }
    if (sameCount == 0.0)
    {
        for (int j = -2; j <= 2; j++)
        {
            for (int i = -2; i <= 2; i++)
            {
                if (abs(i) < 2 && abs(j) < 2)
                    continue;
                int2 qn = clamp(q + int2(i, j), int2(0, 0), int2(sparseSize) - 1);
                float sd = hasScene ? sceneDepth.Load(int3(min(uint2(qn) * 2 + offset, maxTexel), 0)) : 0.0;
                if (CloudSameSurface(sd, surface))
                {
                    float4 s = sparse.Load(int3(qn, 0));
                    lo = min(lo, s);
                    hi = max(hi, s);
                    sameSum += s;
                    sameDepth += sparseDepth.Load(int3(qn, 0));
                    sameCount += 1.0;
                }
            }
        }
    }
    if (sameCount == 0.0)
    {
        lo = loAll;
        hi = hiAll;
    }
    float4 own = sparse.Load(int3(q, 0));
    // The depth goes with the clouds this texel takes: its block's marched sample is on another surface where a leaf or
    // a summit's edge splits the block (a sky texel beside a leaf marched to it had the leaf's rain at 15 m for a depth,
    // and the sky's clouds then lay in front of the leaf: white specks in the crowns)
    float depthKm = sparseDepth.Load(int3(q, 0));
    float ownSurface = hasScene ? sceneDepth.Load(int3(min(uint2(q) * 2 + offset, maxTexel), 0)) : 0.0;
    if (!fresh && !CloudSameSurface(ownSurface, surface) && sameCount > 0.0)
        depthKm = sameDepth / sameCount;
    // Nothing on this surface marched near it this frame (a texel of sky deep in a crown, the blocks round it marched to
    // the leaves): last frame's clouds on this surface round where it was, texel by texel (a bilinear history mixes in
    // the leaves'), reprojected from the far sky where it is sky
    bool lonely = hasScene && sameCount == 0.0;
    if (lonely && surface <= 0.0)
        depthKm = CLOUD_FAR_DEPTH * 0.001;

    // Where this texel's clouds were last frame: the camera moved (CameraMove) and the wind carried them (CloudStep)
    float2 uv = (float2(id.xy) + 0.5) / float2(TargetWidth, TargetHeight);
    float3 rel = CloudViewDir(uv) * (depthKm * 1000.0);
    float3 prevRel = rel + CameraMove - float3(CloudStep.x, 0.0, CloudStep.y);
    float4 prevClip = mul(float4(prevRel, 1.0), PrevCameraRelativeVP);
    float2 prevUV = float2(prevClip.x, -prevClip.y) / max(prevClip.w, 1e-6) * 0.5 + 0.5;
    bool onScreen = prevClip.w > 0.0 && all(prevUV >= 0.0) && all(prevUV <= 1.0);

    // Last frame's surface where the history comes from must be this one (a silhouette that moved leaves none)
    bool sameSurface = true;
    if (hasScene && PrevSceneDepthTexIdx != 0 && onScreen)
    {
        Texture2D<float> prevScene = ResourceDescriptorHeap[PrevSceneDepthTexIdx];
        sameSurface = CloudSameSurface(prevScene.Load(int3(min(uint2(prevUV * float2(TargetWidth, TargetHeight)), maxTexel), 0)),
                                       surface);
    }

    float4 result = 0;
    bool kept = false;
    if (lonely && HistoryTexIdx != 0 && PrevSceneDepthTexIdx != 0 && onScreen)
    {
        Texture2D<float4> history = ResourceDescriptorHeap[HistoryTexIdx];
        Texture2D<float> prevScene = ResourceDescriptorHeap[PrevSceneDepthTexIdx];
        // Never descriptor 0 (a load behind a test of the index still runs: root_signature_bindless.md "Index 0")
        Texture2D<float> historyDepth = ResourceDescriptorHeap[HistoryDepthTexIdx != 0 ? HistoryDepthTexIdx : SparseDepthTexIdx];
        int2 pt = int2(prevUV * float2(TargetWidth, TargetHeight));
        float4 hs = 0;
        float hd = 0.0, hn = 0.0;
        [unroll] for (int j = -1; j <= 1; j++)
        {
            [unroll] for (int i = -1; i <= 1; i++)
            {
                int3 p = int3(clamp(pt + int2(i, j), int2(0, 0), int2(maxTexel)), 0);
                if (CloudSameSurface(prevScene.Load(p), surface))
                {
                    hs += history.Load(p);
                    hd += HistoryDepthTexIdx != 0 ? historyDepth.Load(p) : depthKm;
                    hn += 1.0;
                }
            }
        }
        if (hn > 0.0)
        {
            result = hs / hn;
            result.rgb *= ExposureRatio;
            depthKm = hd / hn;
            kept = true;
        }
    }
    if (kept)
    {
        // last frame's, above
    }
    else if (HistoryTexIdx != 0 && onScreen && sameSurface)
    {
        Texture2D<float4> history = ResourceDescriptorHeap[HistoryTexIdx];
        float4 h = history.SampleLevel(AtmosphereLinearClamp, prevUV, 0);
        h.rgb *= ExposureRatio;
        h = clamp(h, lo, hi);
        // A fresh texel keeps 80 % of its history: the march's jitter (its steps, the light's short steps) showed as a
        // hammered grain on sharp-edged cumulus at 50 %. The clamp to this frame's neighbours keeps it from ghosting.
        result = fresh ? lerp(own, h, 0.8) : h;
    }
    else if (fresh)
        result = own;
    else
    {
        // No history here: this frame's samples round it, bilinearly (the sparse grid's texel centres sit at 2q + offset),
        // those on another surface left out (the mean of those found further out where none of the four is on this one;
        // all of them where none is)
        float2 sq = (float2(id.xy) - float2(offset)) * 0.5;
        int2 b = int2(floor(sq));
        float2 f = sq - b;
        int2 hiq = int2(sparseSize) - 1;
        float4 sum = 0, sumAll = 0;
        float weight = 0, weightAll = 0;
        [unroll] for (int k = 0; k < 4; k++)
        {
            int2 qk = clamp(b + int2(k & 1, k >> 1), 0, hiq);
            float wk = ((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k >> 1) != 0 ? f.y : 1.0 - f.y) + 1e-4;
            float4 s = sparse.Load(int3(qk, 0));
            sumAll += s * wk;
            weightAll += wk;
            float sd = hasScene ? sceneDepth.Load(int3(min(uint2(qk) * 2 + offset, maxTexel), 0)) : 0.0;
            if (CloudSameSurface(sd, surface))
            {
                sum += s * wk;
                weight += wk;
            }
        }
        result = weight > 0.0 ? sum / weight : (sameCount > 0.0 ? sameSum / sameCount : sumAll / weightAll);
    }
    if (any(isnan(result)))
        result = own;
    target[id.xy] = result;
    depthTarget[id.xy] = depthKm;
}

// Direction of a texel of cube face `face` (D3D order +X −X +Y −Y +Z −Z), uv in −1..1 (u right, v down): as the sky
// probe's (sky_probe_cs.hlsl)
float3 CloudCubeDirection(uint face, float2 uv)
{
    float3 d;
    switch (face)
    {
        case 0:  d = float3( 1.0, -uv.y, -uv.x); break;
        case 1:  d = float3(-1.0, -uv.y,  uv.x); break;
        case 2:  d = float3( uv.x,  1.0,  uv.y); break;
        case 3:  d = float3( uv.x, -1.0, -uv.y); break;
        case 4:  d = float3( uv.x, -uv.y,  1.0); break;
        default: d = float3(-uv.x, -uv.y, -1.0); break;
    }
    return normalize(d);
}

// One face of the sky probe's clouds (Clouds.ProbeClouds, 64² × 6, the face ProbeFace this frame), marched coarser than
// the view (steps × 2): the probe composes them over its sky every frame (sky_probe_cs.hlsl)
[numthreads(8, 8, 1)]
void CSCloudProbeFace(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= TargetWidth || id.y >= TargetWidth)
        return;
    RWTexture2DArray<float4> target = ResourceDescriptorHeap[TargetUAVIdx];
    uint3 texel = uint3(id.xy, ProbeFace);
    if (CloudsOn == 0 || AtmosphereIdx == 0)
    {
        target[texel] = float4(0, 0, 0, 1);
        return;
    }
    CloudParams C = LoadClouds();
    AtmosphereParams A = LoadAtmosphere();
    Texture3D<float4> lut = ResourceDescriptorHeap[C.NoiseLut];
    float2 uv = (float2(id.xy) + 0.5) / TargetWidth * 2.0 - 1.0;
    float3 dir = CloudCubeDirection(ProbeFace, uv);
    CloudMarchResult r = CloudMarchRay(C, A, lut, dir, 0.5, 3.0, 1e9);
    if (r.color.a < 0.999)
    {
        // With the air in front (the probe has no aerial perspective volume)
        float3 viewT, inFront;
        CloudAirInFront(A, dir, r.depth, viewT, inFront);
        r.color.rgb = r.color.rgb * viewT + inFront * (1.0 - r.color.a);
    }
    target[texel] = r.color;
}

// ── Precipitation (WEATHER_PLAN.md stage 1; Graphics/Weather.cs) ──
// What reaches the ground at a place: the ordinary layer by the column over it, the storm cells by their shafts. A column
// precipitates when the layer's top reaches the ice (over the freezing level by PRECIP_ICE_LIFT: colder than about −6 °C
// ice crystals grow at the droplets' cost and fall, the Bergeron process) and its cloud here is deep: a deep deck rains or
// snows, a break in it does not, fair-weather cumulus and a thin overcast stay dry. The towers are left to their cells
// (the layer's top decides, not theirs): they rain through the middle of their lives under the shafts the march draws
// (CloudRain), with hail in the cores of the heaviest.

#define PRECIP_STEPS 32
#define PRECIP_ICE_LIFT 1000.0      // m over the freezing level the layer's top must reach (Weather.IceLift)
#define PRECIP_ICE_RANGE 1500.0     // ... and how much further up it gives its most (IceRange)
#define PRECIP_DEPTH_MIN 500.0      // m of cloud overhead before anything reaches the ground (DepthMin)
#define PRECIP_DEPTH_FULL 2000.0    // ... and with how much the deck gives its most (DepthFull)

// At a camera-relative place (xz, m): the rate (1 = heavy), the hail, the column's cloud depth (m) and its top (m above
// sea level)
float4 PrecipitationAt(CloudParams C, Texture3D<float4> lut, float2 rel, float groundY)
{
    // The column from level 0 (16 km across: the map's 4 km lie well inside)
    float2 xz = C.CameraCloud + rel;
    float dz = (C.SlabTop - C.SlabBottom) / PRECIP_STEPS;
    float depth = 0.0, top = C.SlabBottom;
    [loop] for (uint k = 0; k < PRECIP_STEPS; k++)
    {
        float h = C.SlabBottom + (k + 0.5) * dz;
        float d = CloudFieldDensity(CloudSampleLevel(C, 0, xz, h, CloudWrap).x);
        depth += d * dz;
        if (d > 0.25)
            top = h + 0.5 * dz;
    }
    // Whether the deck reaches the ice is the weather's (its top, Base + Thickness), how much falls the column's here: the
    // top measured per column swung with the coverage, and a storm's deck stayed dry over half the town
    float rate = saturate((C.Base + C.Thickness - C.FreezingLevel - PRECIP_ICE_LIFT) / PRECIP_ICE_RANGE)
               * saturate((depth - PRECIP_DEPTH_MIN) / (PRECIP_DEPTH_FULL - PRECIP_DEPTH_MIN));

    // The shafts at the ground, leaned downwind as they fall (CloudRain's axis), their edges frayed by the curtains'
    // noise; virga that ends in the air gives nothing
    float hail = 0.0;
    if (C.RainShafts != 0)
    {
        StructuredBuffer<CloudRainShaft> shafts = ResourceDescriptorHeap[C.RainShafts];
        [loop] for (uint i = 0; i < C.RainCount; i++)
        {
            CloudRainShaft s = shafts[i];
            if (s.Bottom > groundY + 50.0)
                continue;
            float2 axis = s.Center + C.WindDir * ((C.Base - groundY) * RAIN_LEAN);
            float dist = length(rel - axis);
            if (dist > 1.5 * s.Radius)
                continue;
            float streak = lut.SampleLevel(CloudWrap, float3((rel - axis) / 900.0 + s.Seed * 13.0, s.Seed), 0).g;
            float edge = saturate((s.Radius - dist) / (0.5 * s.Radius) + (streak - 0.5) * 0.8);
            rate = max(rate, s.Rain * edge);
            hail = max(hail, saturate((s.Rain - 0.7) / 0.3) * saturate((0.5 * s.Radius - dist) / (0.3 * s.Radius)));
        }
    }
    return float4(rate, hail, depth, top);
}

// The map round the camera (TargetWidth² texels of PrecipTexel m from the camera-relative corner PrecipOrigin): RG16F,
// the rate and the hail
[numthreads(8, 8, 1)]
void CSPrecipitation(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= TargetWidth || id.y >= TargetWidth)
        return;
    RWTexture2D<float2> target = ResourceDescriptorHeap[TargetUAVIdx];
    if (CloudsOn == 0)
    {
        target[id.xy] = 0;
        return;
    }
    CloudParams C = LoadClouds();
    Texture3D<float4> lut = ResourceDescriptorHeap[C.NoiseLut];
    float texel = asfloat(PrecipTexelBits);
    float2 rel = float2(asfloat(PrecipOriginXBits), asfloat(PrecipOriginZBits)) + (float2(id.xy) + 0.5) * texel;
    target[id.xy] = PrecipitationAt(C, lut, rel, CamPos.y).xy;
}

// The camera's own, for the CPU (Weather reads it back a few frames later)
[numthreads(1, 1, 1)]
void CSPrecipitationProbe(uint3 id : SV_DispatchThreadID)
{
    RWStructuredBuffer<float4> probe = ResourceDescriptorHeap[PrecipProbeUAVIdx];
    if (CloudsOn == 0)
    {
        probe[0] = 0;
        return;
    }
    CloudParams C = LoadClouds();
    Texture3D<float4> lut = ResourceDescriptorHeap[C.NoiseLut];
    probe[0] = PrecipitationAt(C, lut, 0.0, CamPos.y);
}
