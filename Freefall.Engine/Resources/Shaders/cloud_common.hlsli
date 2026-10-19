// Volumetric clouds (VOLUMETRIC_CLOUDS_PLAN.md): what every cloud pass shares. Graphics/Clouds.cs fills CloudParams each
// frame (SceneConstants.CloudParamsIdx; 0 = off) and bakes the cloud field into a clipmap of 3D textures that live in
// cloud space, the world's xz carried by the wind: the clouds are modelled once and drift with it for free.
//   Levels   5 × 256² texels horizontally, 16 / 32 / 64 / 128 / 256 km wide (62.5 m … 1 km voxels), centred on the
//            camera and addressed modulo 256 (toroidal: a move bakes only the slabs that enter); height over the slab,
//            128 layers in level 0, 64 above. RG8: density (no detail) and the profile (0 at a cloud's edge, 1 deep in).
//   Bricks   8³ texels each: the most density in them (the march leaps over empty ones).
//   Cells    the storm cells round the period (towers, anvils, rain), made on the CPU from a seed and the game time.
// Heights are metres above sea level on the curved planet (the camera at the top of a sphere of CLOUD_EARTH_RADIUS);
// cloud-space xz is CameraCloud + a camera-relative position's xz.
// Needs common.fx (SceneConstants) first.

#ifndef CLOUD_COMMON
#define CLOUD_COMMON

#define CLOUD_LEVELS 5
#define CLOUD_TEXELS 256            // texels each way per level (Clouds.TexelsPerLevel)
#define CLOUD_BRICK 8               // texels a brick spans each way
#define CLOUD_PERIOD 256000.0       // metres: the weather repeats in cloud space (noise tiles, cells); Clouds.Period
#define CLOUD_EARTH_RADIUS 6360000.0
#define CLOUD_BASE_TILE (CLOUD_PERIOD / 40.0)      // 6.4 km: the base shape's noise tile
#define CLOUD_DETAIL_TILE (CLOUD_PERIOD / 160.0)   // 1.6 km: the billows' noise tile (Worley cells of 400 m and down)
#define CLOUD_COVER_TILE (CLOUD_PERIOD / 5.0)      // 51.2 km: the coverage pattern's tile

struct CloudLevel
{
    float2 Origin;          // cloud-space xz of the level's corner (m)
    float VoxelSize;        // metres a texel spans across
    uint Layers;            // texels in height
    uint DensityTex;        // Texture3D<float2> (RG8): density, profile
    uint BrickTex;          // Texture3D<float> (R8): the most density in each brick
    uint LightTex;          // Texture3D<float> (R16F), half the density's texels each way: the optical depth toward the
                            // light (LightDir) from there out of the slab, the coarser levels' added where it leaves this one
    uint SkyTex;            // Texture3D<float2> (RG16F), the same texels: optical depth straight up (x) and down (y)
    float Reach;            // metres from the camera (each way) this level is used to
    float _levelPad0, _levelPad1, _levelPad2;
};

struct CloudParams
{
    CloudLevel Levels[CLOUD_LEVELS];
    float2 CameraCloud;     // the camera's cloud-space xz (wrapped by CLOUD_PERIOD)
    float SlabBottom;       // metres above sea level the volumes span
    float SlabTop;
    float Base;             // the ordinary layer's base and depth (m)
    float Thickness;
    float Coverage;         // 0..1
    float Type;             // 0 stratus … 0.5 stratocumulus … 1 cumulus
    float TowerShare;       // share of the cells that tower
    float TowerTop;         // the tallest tower's top (m above sea level)
    float Density;          // extinction per metre at full density
    float Albedo;           // single-scattering albedo
    float2 WindDir;         // where the wind blows (xz, unit): the towers lean and the anvils spread that way
    float Evolution;        // seconds of the clouds' own change (× CloudSpeed): the noise slides through them
    uint NoiseLut;          // Texture3D<float4> 128³ (cloud_noise_gen.hlsl)
    uint Cells;             // StructuredBuffer<CloudCell>, CellsPerSide² of them over the period
    float CellSize;         // metres a cell spans
    uint CellsPerSide;
    float Reach;            // metres along a view ray the march goes
    float3 LightDir;        // toward the light the clouds are lit by (the sun, after ~3.4° below the horizon the
                            // brightest moon): the light volumes are made along it
    uint LightIsMoon;
    uint ProbeClouds;       // Texture2DArray<float4> 64² × 6: the sky probe's clouds (rgb light, a transmittance), a face
                            // a frame (clouds_cs.hlsl CSCloudProbeFace); 0 = none yet
    float3 _paramsPad;
};

struct CloudCell
{
    float2 Center;          // in the cell, from its corner (m)
    float Radius;           // the column's radius (m); 0 = no tower here
    float Top;              // the tower's top (m above sea level) as it has grown so far
    float Anvil;            // 0..1: how far its anvil has spread
    float Rain;             // 0..1: how hard it rains under it
    float Growth;           // 0..1 of its life: growing, mature, decaying
    float _cellPad;
};

CloudParams LoadClouds()
{
    StructuredBuffer<CloudParams> buffer = ResourceDescriptorHeap[CloudParamsIdx];
    return buffer[0];
}

float CloudRemap(float v, float lo, float hi, float newLo, float newHi)
{
    return newLo + (v - lo) / max(hi - lo, 1e-5) * (newHi - newLo);
}

// The baked field is smooth: 0.5 + how far the shape stands past the coverage's threshold (0.5 = a cloud's surface,
// higher inside). The march turns it into density with the billows (CloudErode): the surface moves by the detail
// before this ramp sharpens it. Baking the sharp density instead left no room for the billows and interpolated badly.
#define CLOUD_RAMP 0.1              // field units over which density rises from 0 to 1 (~150 m): a cumulus' sharp edge
#define CLOUD_BILLOW 0.3            // how far (field units, ~450 m) the detail moves the surface in or out
float CloudFieldDensity(float field)
{
    return saturate((field - 0.5) / CLOUD_RAMP);
}

// Height above sea level (m) of a camera-relative position on the round planet, written so the 6360 km radius cancels
// without losing the metres: |P| − Rc = (2 Rc y + |rel|²) / (|P| + Rc), Rc the camera's distance from the centre
float CloudHeight(float3 rel)
{
    float rc = CLOUD_EARTH_RADIUS + CamPos.y;
    float lenP = sqrt(rc * rc + 2.0 * rc * rel.y + dot(rel, rel));
    return CamPos.y + (2.0 * rc * rel.y + dot(rel, rel)) / (lenP + rc);
}

// A camera-relative position (m) in the atmosphere's frame (km, the planet's centre at the origin, the camera above it):
// its length is CLOUD_EARTH_RADIUS + CloudHeight, as the transmittance LUT wants
float3 CloudAtmospherePos(float3 rel)
{
    return float3(rel.x, CLOUD_EARTH_RADIUS + CamPos.y + rel.y, rel.z) * 0.001;
}

// Where a ray from the camera (unit dir) meets the sphere of height H (m above sea level): the roots t0 ≤ t1 of
// t² + 2 Rc dy t + c = 0, c = (camY − H)(Rc + R + H); false when it never does. Written without cancellation.
bool CloudSphereRoots(float3 dir, float H, out float t0, out float t1)
{
    float rc = CLOUD_EARTH_RADIUS + CamPos.y;
    float b = rc * dir.y;
    float c = (CamPos.y - H) * (rc + CLOUD_EARTH_RADIUS + H);
    float disc = b * b - c;
    t0 = 0; t1 = 0;
    if (disc < 0.0)
        return false;
    float q = -(b + (b >= 0.0 ? 1.0 : -1.0) * sqrt(disc));
    float r0 = q, r1 = abs(q) > 1e-6 ? c / q : q;
    t0 = min(r0, r1);
    t1 = max(r0, r1);
    return true;
}

// The stretch of a view ray inside the slab [SlabBottom, SlabTop], before the ground and within the reach: the first one
// only (a ray from above that passes through and out again ends at the bottom)
bool CloudSlabSegment(CloudParams C, float3 dir, out float tStart, out float tEnd)
{
    tStart = 0; tEnd = 0;
    float a0, a1;
    if (!CloudSphereRoots(dir, C.SlabTop, a0, a1) || a1 <= 0.0)
        return false;                                       // never inside the top sphere ahead
    tStart = max(a0, 0.0);
    tEnd = a1;
    float b0, b1;
    if (CloudSphereRoots(dir, C.SlabBottom, b0, b1))
    {
        if (b0 <= 0.0 && b1 > 0.0)
            tStart = max(tStart, b1);                       // below the base: in from the base
        else if (b0 > tStart)
            tEnd = min(tEnd, b0);                           // from above: out through the base
    }
    float g0, g1;
    if (CloudSphereRoots(dir, 0.0, g0, g1) && g0 > 0.0)
        tEnd = min(tEnd, g0);                               // the sea's surface
    tEnd = min(tEnd, C.Reach);
    return tEnd > tStart;
}

// Which level holds a place this far from the camera (metres, the larger of |Δx|, |Δz|): the finest whose reach covers
// it, and how far into the next one it has blended (0..1, the last quarter of a level's reach)
uint CloudLevelAt(CloudParams C, float dist, out float blend)
{
    blend = 0;
    [unroll] for (uint i = 0; i < CLOUD_LEVELS; i++)
    {
        float reach = C.Levels[i].Reach;
        if (dist < reach || i == CLOUD_LEVELS - 1)
        {
            blend = i < CLOUD_LEVELS - 1 ? saturate((dist - reach * 0.75) / (reach * 0.25)) : 0.0;
            return i;
        }
    }
    return CLOUD_LEVELS - 1;
}

float3 CloudLevelUvw(CloudParams C, CloudLevel L, float2 xz, float h)
{
    float w = (h - C.SlabBottom) / max(C.SlabTop - C.SlabBottom, 1.0);
    w = clamp(w, 0.5 / L.Layers, 1.0 - 0.5 / L.Layers);    // no blending across the wrap in height
    return float3(xz / (L.VoxelSize * CLOUD_TEXELS), w);
}

// Density and profile at a cloud-space place from one level (0 outside the slab)
float2 CloudSampleLevel(CloudParams C, uint level, float2 xz, float h, SamplerState wrap)
{
    if (h < C.SlabBottom || h > C.SlabTop)
        return 0;
    CloudLevel L = C.Levels[level];
    Texture3D<float2> density = ResourceDescriptorHeap[L.DensityTex];
    return density.SampleLevel(wrap, CloudLevelUvw(C, L, xz, h), 0);
}

// The same, from the level the distance picks, blended into the next toward its edge
float2 CloudSample(CloudParams C, float2 xz, float h, float dist, SamplerState wrap)
{
    float blend;
    uint level = CloudLevelAt(C, dist, blend);
    float2 d = CloudSampleLevel(C, level, xz, h, wrap);
    if (blend > 0.0)
        d = lerp(d, CloudSampleLevel(C, level + 1, xz, h, wrap), blend);
    return d;
}

// The most density in the brick round a place (0 = the brick is empty: nothing to sample in it)
float CloudBrickMax(CloudParams C, uint level, float2 xz, float h)
{
    if (h < C.SlabBottom || h > C.SlabTop)
        return 0;
    CloudLevel L = C.Levels[level];
    Texture3D<float> bricks = ResourceDescriptorHeap[L.BrickTex];
    float3 uvw = CloudLevelUvw(C, L, xz, h);
    int3 brick = int3(floor(uvw.xy * (CLOUD_TEXELS / CLOUD_BRICK)), min((int)(uvw.z * L.Layers) / CLOUD_BRICK, (int)(L.Layers / CLOUD_BRICK) - 1));
    brick.xy = ((brick.xy % (CLOUD_TEXELS / CLOUD_BRICK)) + (CLOUD_TEXELS / CLOUD_BRICK)) % (CLOUD_TEXELS / CLOUD_BRICK);
    return bricks.Load(int4(brick, 0));
}

// The optical depth toward the light from the light volume of one level (0 above the slab)
float CloudLightDepthAt(CloudParams C, uint level, float2 xz, float h, SamplerState wrap)
{
    if (h > C.SlabTop)
        return 0.0;
    CloudLevel L = C.Levels[level];
    Texture3D<float> light = ResourceDescriptorHeap[L.LightTex];
    return light.SampleLevel(wrap, CloudLevelUvw(C, L, xz, clamp(h, C.SlabBottom, C.SlabTop)), 0);
}

// The optical depth above (x) and below (y) a place, straight up and down, from the sky volume of one level
float2 CloudSkyDepthAt(CloudParams C, uint level, float2 xz, float h, SamplerState wrap)
{
    CloudLevel L = C.Levels[level];
    Texture3D<float2> sky = ResourceDescriptorHeap[L.SkyTex];
    return sky.SampleLevel(wrap, CloudLevelUvw(C, L, xz, clamp(h, C.SlabBottom, C.SlabTop)), 0);
}

// The clouds' optical depth toward the light from a place (camera-relative): the light volume where the light's path
// crosses the slab's base, or at the place itself inside the slab. The path's height rises with the planet's curvature:
// h + s·L.y + s²/2R reaches the base at s = 2|c| / (L.y + sqrt(L.y² + 2|c|/R)), c = h − base, written without
// cancellation. A crossing past the coarsest level (a light a degree or two over the horizon: 138 km for a 1.5 km base
// at 0°) is drawn back to its edge, where the toroidal volumes would wrap onto other clouds: the clouds there toward the
// light stand for the rest of the way (a deck that runs on keeps the light out, and at that angle every place round the
// camera looks through nearly the same clouds). 0 for any other light than the one the volumes are made for
// (sky_common.fx CloudTransmission uses the deck's means for that one).
float CloudShadowDepth(CloudParams C, float3 rel, float3 toLight, SamplerState wrap)
{
    if (dot(toLight, C.LightDir) < 0.99)
        return 0.0;
    float h = CloudHeight(rel);
    float3 q = rel;
    if (h < C.SlabBottom)
    {
        float c = C.SlabBottom - h;
        float s = 2.0 * c / (toLight.y + sqrt(toLight.y * toLight.y + 2.0 * c / CLOUD_EARTH_RADIUS));
        q = rel + toLight * s;
        h = C.SlabBottom + 1.0;
    }
    float dist = max(abs(q.x), abs(q.z));
    float edge = C.Levels[CLOUD_LEVELS - 1].Reach * 0.95;
    if (dist > edge)
    {
        q.xz *= edge / dist;
        dist = edge;
    }
    float blend;
    uint level = CloudLevelAt(C, dist, blend);
    float2 xz = C.CameraCloud + q.xz;
    float tau = CloudLightDepthAt(C, level, xz, h, wrap);
    if (blend > 0.0)
        tau = lerp(tau, CloudLightDepthAt(C, level + 1, xz, h, wrap), blend);
    return tau;
}

// How far a ray (unit dir) goes from a place before it leaves that place's brick: across, the brick's xz walls; in
// height its layers' bounds, the ray's height taken as straight over so short a stretch
float CloudBrickExit(CloudParams C, uint level, float2 xz, float h, float3 dir)
{
    CloudLevel L = C.Levels[level];
    float brick = L.VoxelSize * CLOUD_BRICK;
    float2 lo = floor(xz / brick) * brick;
    float2 exitXZ = select(dir.xz >= 0.0, lo + brick - xz, xz - lo) / max(abs(dir.xz), 1e-5);
    float layerH = (C.SlabTop - C.SlabBottom) / L.Layers * CLOUD_BRICK;
    float hb = floor((h - C.SlabBottom) / layerH) * layerH + C.SlabBottom;
    float exitH = (dir.y >= 0.0 ? hb + layerH - h : h - hb) / max(abs(dir.y), 1e-5);
    return max(min(min(exitXZ.x, exitXZ.y), exitH), 0.0);
}

#endif // CLOUD_COMMON
