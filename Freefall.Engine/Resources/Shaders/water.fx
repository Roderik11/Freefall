// Lakes and rivers (WaterBody component): a flat or gently sloping mesh generated from a spline.
// No FFT displacement — ripples are normals only, borrowed from the ocean simulation's finest bands.
// All lighting comes from water_common.fx, shared with ocean.fx.

cbuffer PushConstants : register(b3)
{
    // Slots 0-1: set by the forward pass for every forward batch
    uint ShadowMapIdx;              // 0: Shadow cascade array SRV
    uint _reserved1;                // 1: CompositeSnapshot SRV (taken from WaterData instead)
    // Slots 2-3: PER-DRAW (command signature writes these)
    uint MeshPartId;                // 2: Index into MeshRegistry
    uint InstanceBaseOffset;        // 3: Base offset for instance ID (per-command)
    // Slots 4-8: PER-BATCH (set before ExecuteIndirect)
    uint DescriptorBufIdx;          // 4: StructuredBuffer<InstanceDescriptor>
    uint SortedIndicesIdx;          // 5: StructuredBuffer<uint> - sorted draw order indices
    uint MeshRegistryIdx;           // 6: StructuredBuffer<MeshPartEntry>
    uint MaterialsIdx;              // 7: Index to materials buffer
    uint GlobalTransformBufferIdx;  // 8: Index to global TransformBuffer
    // Slots 9-15: Custom / Reserved
    uint WaterDataIdx;              // 9: Per-instance WaterData buffer
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint _reserved15;
    // Slot 16: Debug
    uint DebugMode;                 // 16: Debug visualization mode
    uint IndexedDraw;               // 17: 1 = an indexed draw, the vertex index is SV_VertexID (common.fx MeshVertexIndex)
    uint _reserved18;
    uint _reserved19;
    // Slots 20-21: Shadow pass
    uint ExpansionBufferIdx;        // 20: SRV: expansion buffer
    uint CascadeBufferSRVIdx;       // 21: SRV: StructuredBuffer<CascadeData>
};

#include "common.fx"
#include "sky_common.fx"
#include "water_common.fx"
#include "underwater.hlsli"
#include "water_ripples.hlsli"
// @RenderState(RenderTargets=1, CullMode=None)

// Must match WaterBody.WaterData (C#)
struct WaterData
{
    float Time;
    float FlowSpeed;            // m/s along the spline (rivers)
    float RippleStrength;
    float RippleSize;
    float3 WaterColor;          // deep water
    float Visibility;           // meters of water at which the bed is mostly gone
    float3 ShallowColor;
    float RefractionStrength;
    float3 SunDirection;
    float SunIntensity;
    float3 SunColor;
    float EdgeFoam;
    float RapidsFoam;
    uint SlopeSRV;              // ocean FFT slope maps (Texture2DArray), 0 = flat water
    uint NoiseSRV;              // ocean noise texture (Perlin + Worley)
    uint DepthGBufferSRV;
    uint CompositeSRV;
    float2 InvViewportSize;
    float TileScaleA;           // 1 / tile size (m) of the two ripple bands
    float TileScaleB;
    uint BandA;
    uint BandB;
    float Flowing;              // 1 = river (open spline), 0 = lake
    float Ice;                  // WaterBody.Ice: how frozen this water is, 0..1
    float SnowCover;            // Graphics.Weather.SnowCover: the snow lies on the ice too
    uint ShoreField;            // Texture2D<float>: the share of the way from the shore to the middle (0 = none yet)
    float2 ShoreOrigin;         // world xz of its corner
    float2 ShoreSize;           // metres it spans
    float2 SnowAlong;           // Graphics.Weather.SnowLaidAlong: the way the snow's drifts on the ice run (xz, unit)
    uint NearestDepthSRV;       // Graphics.WaterReflections: the nearest 1 / view depth, half size and down (0 = no screen-space reflections)
    uint NearestDepthMips;
    uint SceneColourSRV;        // Graphics.WaterReflections: the composite snapshot averaged, half size and down
    uint SceneColourMips;
    uint RipplesSRV;            // Graphics.WaterRipples: the ripples round the camera (water_ripples.hlsli); 0 = still
    float2 RipplesOrigin;       // the ripple grid's corner (world xz)
    float RipplesSpan;          // metres it spans
};

SamplerState WaterSampler : register(s0);

// How thick the ice is (0..1; 0 = open water) at a place shore of the way from the shore to the middle of its water
// (WaterBody's shore field) whose current runs at speed m/s, at the ice amount: one sheet growing out from the shore, the
// middle of a lake frozen over at 0.75; a current holds it off, slowing to nothing toward the banks, so a river freezes
// from its banks in and a fast one keeps an open channel; the sheet's edge is thin over 0.1 of the ice.
// Graphics.Weather.IceThickness has the same numbers (the ice collider)
float WaterIceThickness(float ice, float shore, float speed)
{
    float needed = 0.15 + 0.6 * shore + 0.4 * saturate(speed * shore / 2.5);
    return saturate((ice - needed) / 0.1);
}

#define ICE_ALBEDO  float3(0.28, 0.32, 0.35)    // white ice, grey with the air frozen into it (the dark palette's ice;
                                                // 0.36..0.45 read as near white in the noon sun)
#define SNOW_ON_ICE float3(0.70, 0.72, 0.76)    // decals_cs.hlsl SNOW_ALBEDO: the same snow as on the ground

// The plates of a sheet of ice: cells round jittered points (Voronoi) of a grid of 1 (scale p to the plates' size),
// the distance to the nearest border between two of them (F2 − F1: about the distance to the crack) and the cell's id
float IceCells(float2 p, out float2 id)
{
    float2 i = floor(p), f = frac(p);
    float d1 = 8.0, d2 = 8.0;
    id = i;
    [unroll] for (int y = -1; y <= 1; y++)
    [unroll] for (int x = -1; x <= 1; x++)
    {
        float2 g = float2(x, y);
        float2 r = g + 0.1 + 0.8 * float2(MacroHash(i + g), MacroHash(i + g + 17.3)) - f;
        float d = dot(r, r);
        if (d < d1) { d2 = d1; d1 = d; id = i + g; }
        else if (d < d2) d2 = d;
    }
    return sqrt(d2) - sqrt(d1);
}

// Value noise interpolated linearly, -0.5..0.5: kinks at its lattice, not curves (the cracks' zigzag)
float IceJag(float2 p)
{
    float2 i = floor(p), f = frac(p);
    float a = MacroHash(i), b = MacroHash(i + float2(1, 0)), c = MacroHash(i + float2(0, 1)), d = MacroHash(i + 1.0);
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y) - 0.5;
}

// A crack along the plates' borders, width metres wide in a sheet of plates cell metres across, as a share of the pixel
// it covers: no thinner than the pixel, fainter instead, so it never shimmers
float IceCrack(float edge, float width, float cell)
{
    float w = width / cell;
    float pixel = max(fwidth(edge), 1e-5);
    return (1.0 - smoothstep(0.0, max(w, pixel), edge)) * saturate(w / pixel);
}

// Water the sky does not reach (a flooded hall, a sea cave) has no sky to mirror, and it was a flat black sheet nobody could
// see (user, 2026-10-31). What stands round it is mirrored instead: one ray over the screen from the surface, walked as
// ssr_cs.hlsl walks them, steps growing with the distance, each compared with the linear depth there, a hit refined by
// halving; the scene's own picture at the hit is what the water shows. A ray that leaves the screen or meets nothing stays dark.
#define WATER_RAY_STEPS 36
#define WATER_RAY_FIRST 0.12    // m, growing by WATER_RAY_GROWTH a step (36 steps reach ~60 m)
#define WATER_RAY_GROWTH 1.12
#define WATER_RAY_REFINE 4

// The screen place (0..1) and view depth of a camera-relative place; false behind the camera
bool WaterRayProject(float3 rel, out float2 uv, out float viewZ)
{
    float4 clip = mul(float4(rel, 1.0), CameraRelativeVP);
    viewZ = clip.w;
    uv = 0;
    if (clip.w <= 0.05)
        return false;
    uv = clip.xy / clip.w * float2(0.5, -0.5) + 0.5;
    return true;
}

float3 WaterSceneReflection(WaterData water, float3 rel, float3 V, float3 N, float2 pixel)
{
    float3 r = reflect(-V, N);
    if (water.DepthGBufferSRV == 0 || water.CompositeSRV == 0 || r.y <= 0.0)
        return 0;
    Texture2D<float> depth = ResourceDescriptorHeap[water.DepthGBufferSRV];
    float2 size = 1.0 / water.InvViewportSize;

    // The steps' start dithered per pixel, so their bands do not line up
    float dither = frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
    float stepLen = WATER_RAY_FIRST * (0.5 + dither);
    float t = stepLen, lastT = 0.0;
    [loop]
    for (uint i = 0; i < WATER_RAY_STEPS; i++)
    {
        float2 uv;
        float rayZ;
        if (!WaterRayProject(rel + r * t, uv, rayZ) || any(uv < 0.0) || any(uv >= 1.0))
            return 0;
        float sceneZ = depth.Load(int3(int2(uv * size), 0));
        // In front of the surface by less than a step's thickness: a hit (else the ray passed behind something)
        if (sceneZ > 0.0 && rayZ > sceneZ && rayZ - sceneZ < max(0.2, stepLen * 2.0))
        {
            float lo = lastT, hi = t;
            [unroll]
            for (uint k = 0; k < WATER_RAY_REFINE; k++)
            {
                float mid = 0.5 * (lo + hi);
                float2 mp;
                float midZ;
                WaterRayProject(rel + r * mid, mp, midZ);
                float midScene = depth.Load(int3(clamp(int2(mp * size), 0, int2(size) - 1), 0));
                if (midScene > 0.0 && midZ > midScene) hi = mid; else lo = mid;
            }
            float hz;
            WaterRayProject(rel + r * hi, uv, hz);
            Texture2D<float4> composite = ResourceDescriptorHeap[water.CompositeSRV];
            float edge = saturate(min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)) / 0.08);
            return composite.SampleLevel(AtmosphereLinearClamp, uv, 0).rgb * edge;
        }
        lastT = t;
        stepLen *= WATER_RAY_GROWTH;
        t += stepLen;
    }
    return 0;
}

// ── The scene in the water (Graphics/WaterReflections.cs, water_reflections_cs.hlsl) ──
// What water mirrors besides the sky: the banks, trees and houses round it, found on the screen. Tried once and taken out
// (2026-10-05, ocean_system/overview.md) for noise: rays along each pixel's own rippled normal, walked in steps with a
// per-pixel dither, hit a leaf in one pixel and fell through a gap in the next. Here every pixel traces a level surface's
// mirror ray (the scene mirrored about the water), cell by cell through the nearest-depth pyramid, which finds the first
// pixel the ray goes behind exactly, and the same for neighbours: no dither, no step that skips a post. The ripples only
// bend where along that ray's length the scene is looked up, by the ripple normal low-passed over a few pixels (so
// neighbours look up nearly the same place), and the slopes finer than that blur the look-up into a vertical streak, as
// WaterSkyReflection spreads the sky's, from the scene's colour pyramid.

#define WATER_TRACE_STEPS 96        // cells visited at most; a ray over a screen of open ground takes ~30-60
#define WATER_BEHIND_PIXELS 12      // pixels a ray walks on behind something nearer, looking for what it hides
#define WATER_LOOKUP_BIAS 2.0       // mips coarser than the shading's: the look-up normal is the mean over ~4 × 4 pixels
#define WATER_STILL_VARIANCE 0.0002 // slope variance under the ripple bands for the look-up's blur (α 0.014: ~0.2° up and down)

// The nearest 1 / view depth in a cell of the trace: level 0 is the full-size linear depth, level k ≥ 1 the pyramid's mip
// k − 1 (cells 2^k pixels wide; a mip's last texel also covers what is left over at the screen's right and bottom)
float WaterNearest(WaterData water, int2 cell, int level, int2 fullSize)
{
    if (level == 0)
    {
        Texture2D<float> depth = ResourceDescriptorHeap[water.DepthGBufferSRV];
        float w = depth.Load(int3(cell, 0));
        return w > 0.0 ? 1.0 / w : 0.0;
    }
    Texture2D<float> pyramid = ResourceDescriptorHeap[water.NearestDepthSRV];
    int2 dims = max((fullSize / 2) >> (level - 1), 1);
    return pyramid.Load(int3(min(cell, dims - 1), level - 1));
}

// WATER_TRACE_DEBUG 1 draws the water as what its trace did (saving the file reloads it in a running editor): green met
// something (olive: past something nearer), red stood in for what something nearer hid, blue reached the sky on the
// screen, grey left the screen, yellow ran out of steps, magenta met the sky
#define WATER_TRACE_DEBUG 0
// RIPPLE_DEBUG 1 draws the ripples round the camera (WaterRipples): red up, blue down (±1 cm full), green their slope
#define RIPPLE_DEBUG 0
static float3 WaterTraceDebug = 0;

// A clip-space place as full-size pixels and 1 / view depth (affine on the screen)
float3 WaterScreen(float4 clip, float2 size)
{
    float inv = 1.0 / clip.w;
    return float3((clip.xy * inv * float2(0.5, -0.5) + 0.5) * size, inv);
}

// The first place the ray from rel along r (camera-relative, unit) goes behind the depth: its pixel and metres along the
// ray, and how sure that is (0 where the ray left the screen, reached the sky or ran out of steps). A ray that went in
// behind something nearer than what it would meet (a post in front of the far bank) walks on past it for what it hides,
// and if that is too wide, the thing in front stands in. open: nothing met all the way to the ray's vanishing point,
// which is on the screen (hitPx): the sky there is what it mirrors
float WaterTrace(WaterData water, float3 rel, float3 r, out float2 hitPx, out float hitT, out bool open)
{
    hitPx = 0;
    hitT = 0;
    open = false;
    float2 size = 1.0 / water.InvViewportSize;
    int2 fullSize = int2(size + 0.5);
    float4 c0 = mul(float4(rel, 1.0), CameraRelativeVP);
    float4 cr = mul(float4(r, 0.0), CameraRelativeVP);     // the clip change per metre along the ray
    float3 o = WaterScreen(c0, size);
    // The far end on the screen: the ray's vanishing point when it goes away from the camera (1 / depth 0: infinitely
    // far), else where it comes to 0.2 m in front of the camera ...
    float3 e;
    if (cr.w > 1e-4)
        e = float3((cr.xy / cr.w * float2(0.5, -0.5) + 0.5) * size, 0.0);
    else
        e = WaterScreen(c0 + cr * min(max(c0.w - 0.2, 0.0) / max(-cr.w, 1e-6), 1e4), size);
    float3 d = e - o;
    // ... and no further than the screen's edge (affine on the screen, so 1 / depth is cut right too)
    float toX = d.x > 0.0 ? (size.x - o.x) / d.x : (d.x < 0.0 ? -o.x / d.x : 1e30);
    float toY = d.y > 0.0 ? (size.y - o.y) / d.y : (d.y < 0.0 ? -o.y / d.y : 1e30);
    bool vanishes = cr.w > 1e-4 && min(toX, toY) >= 1.0;
    d *= saturate(min(toX, toY));
    if (abs(d.x) < 1e-5) d.x = 1e-5;     // no zero to divide by; a tiny step counts as going right / down
    if (abs(d.y) < 1e-5) d.y = 1e-5;

    float3 inv = float3(1.0 / d.x, 1.0 / d.y, d.z < 0.0 ? 1.0 / d.z : 0.0);
    float2 away = float2(d.x > 0.0 ? 1.0 : 0.0, d.y > 0.0 ? 1.0 : 0.0);       // a cell's far side along the ray
    float2 nudge = float2(d.x > 0.0 ? 0.01 : -0.01, d.y > 0.0 ? 0.01 : -0.01);  // just over it, into the next cell

    // Out of the water's own pixel first (the depth there is the bed under it), then cell by cell: a cell the ray passes
    // in front of entirely is skipped and the next looked at a level coarser; one it meets is gone into a level finer
    float s = min((floor(o.x) + away.x + nudge.x - o.x) * inv.x, (floor(o.y) + away.y + nudge.y - o.y) * inv.y);
    int level = 0;
    int top = (int)water.NearestDepthMips;
    uint behind = 0;                            // pixels walked behind something nearer
    Texture2D<float> depthTex = ResourceDescriptorHeap[water.DepthGBufferSRV];
    [loop]
    for (uint i = 0; i < WATER_TRACE_STEPS; i++)
    {
        if (s >= 1.0)
        {
            open = vanishes && behind == 0;     // infinitely far with nothing met (else the screen's edge)
            hitPx = e.xy;
            WaterTraceDebug = open ? float3(0, 0, 1) : float3(0.2, 0.2, 0.2);
            return 0.0;
        }
        float3 p = o + d * s;
        float cellSize = exp2((float)level);
        float2 cell = floor(p.xy / cellSize);
        float nearest = WaterNearest(water, int2(cell), level, fullSize);
        float2 tb = ((cell + away) * cellSize + nudge - o.xy) * inv.xy;
        float tEdge = min(tb.x, tb.y);
        // Where the ray's 1 / depth falls to the cell's nearest; a cell of only sky (0) has nothing to meet, not even at the
        // ray's vanishing point (1 / depth 0 too)
        float tMeet = d.z < 0.0 && nearest > 0.0 ? (nearest - o.z) * inv.z : 1e30;
        if (p.z > nearest)
        {
            behind = 0;                         // out past whatever it was behind, in front again
            if (tMeet < tEdge) { s = tMeet; level--; }
            else { s = tEdge; level = min(level + 1, top); }
        }
        else
            level--;
        if (level >= 0)
            continue;

        // Down to a pixel the ray is not in front of: met it there, or came in from the side behind it
        p = o + d * s;
        float sceneW = depthTex.Load(int3(clamp(int2(p.xy), 0, fullSize - 1), 0));
        if (sceneW <= 0.0 || p.z <= 0.0)
        {
            WaterTraceDebug = float3(1, 0, 1);
            return 0.0;
        }
        float rayW = 1.0 / p.z;
        float thickness = 0.25 + 0.03 * sceneW;
        if (rayW - sceneW < 1.5 * thickness)
        {
            hitPx = p.xy;
            float3 at = posFromDepth(p.xy / size, Projection._33 + Projection._43 / rayW, CameraInverse).xyz;
            hitT = length(at - rel);
            float sure = 1.0 - smoothstep(thickness, 1.5 * thickness, rayW - sceneW);
            WaterTraceDebug = float3(behind > 0 ? 0.5 : 0.0, sure, 0.0);
            return sure;
        }
        // Behind something nearer (a post, a branch, a reed in front of the far bank): what it hides is not on the screen,
        // but where the ray comes out past it the bank it was going to meet usually goes on. Walk on pixel by pixel, a
        // few at most, for a pixel it meets or comes in front of again
        if (behind++ == 0)
        {
            hitPx = p.xy;                       // if nothing is found past it, the thing in front stands in for it
            hitT = length(posFromDepth(p.xy / size, Projection._33 + Projection._43 / sceneW, CameraInverse).xyz - rel);
        }
        if (behind > WATER_BEHIND_PIXELS)
            break;
        s = tEdge;
        level = 0;
    }
    // Behind something too wide to see past: its own colour, which is of the place, where the sky probe surely is not
    // (a speck of bright sky in a dark bank). Out of steps: nothing
    WaterTraceDebug = behind > 0 ? float3(1, 0, 0) : float3(1, 1, 0);
    return behind > 0 ? 1.0 : 0.0;
}

// The scene's light at a screen place, blurred over about footprint full-size pixels: the snapshot itself under one, the
// colour pyramid over (its mip 0 is half size). No brighter than WATER_SCENE_MAX, as the pyramid is made
// (water_reflections_cs.hlsl SCENE_MAX: the sky's sun disc; the sun's glint is its reflection)
#define WATER_SCENE_MAX 32.0
float3 WaterSceneTap(WaterData water, float2 uv, float footprint)
{
    float lod = log2(max(footprint, 1.0));
    Texture2D<float4> pyramid = ResourceDescriptorHeap[water.SceneColourSRV];
    float3 coarse = pyramid.SampleLevel(AtmosphereLinearClamp, uv, clamp(lod - 1.0, 0.0, water.SceneColourMips - 1.0)).rgb;
    if (lod >= 1.0)
        return coarse;
    Texture2D<float4> composite = ResourceDescriptorHeap[water.CompositeSRV];
    float3 fine = composite.SampleLevel(AtmosphereLinearClamp, uv, 0).rgb;
    fine *= min(1.0, WATER_SCENE_MAX / max(max(fine.r, max(fine.g, fine.b)), 1e-6));
    return lerp(fine, coarse, lod);
}

// What water at rel mirrors of the scene: rgb, and in a how much of it takes the sky's place (0 where the trace found
// nothing). look: the ripple normal low-passed for the look-up; lookAlpha: the roughness of the slopes finer than it
float4 WaterScreenReflection(WaterData water, float3 rel, float3 V, float3 look, float lookAlpha)
{
    float3 r = float3(-V.x, V.y, -V.z);            // a level surface's mirror ray
    if (r.y <= 0.0)
        return 0;                                   // seen from under the surface
    float2 hitPx;
    float t;
    bool open;
    float sure = WaterTrace(water, rel, r, hitPx, t, open);
    // A ray that met nothing up to its vanishing point on the screen mirrors the sky there, as the snapshot has it (the
    // clouds, the haze): looked up as directions (infinitely far), so a far silhouette blurs into the sky beside it as
    // into the sky over it. Where the sky it mirrors is off the screen the sky probe stays
    bool far = sure <= 0.0 && open;
    if (far)
    {
        sure = 1.0;
        t = 1.0;
    }
    if (sure <= 0.0)
        return 0;

    float2 size = 1.0 / water.InvViewportSize;
    // Faded toward the screen's edges, where the ray would have found more past them. Over a quarter of the screen at the
    // top, where nearly every reflection leaves it: over 8 % the sky probe took over from the screen's sky along a line
    // across the lake, the clouds stopping at it
    float2 huv = hitPx / size;
    float edge = saturate(min(huv.x, 1.0 - huv.x) / 0.08) * saturate((1.0 - huv.y) / 0.08) * smoothstep(0.0, 0.25, huv.y);

    // The rippled mirror ray carried as far as the level one went, spread up and down a lobe shaped as WaterSkyReflection
    // spreads the sky's but half as tall (1.3 α, not 2.6: the full height still smeared the rocks of a bank into streaks),
    // in nine taps 0.1 × that apart, each blurred as wide as the lobe is across: seen at a grazing angle a slope tilts the
    // mirror ray up and down far more than to the side, so a reflection streaks downward and stays narrow
    float3 R = reflect(-V, look);
    float tall = 1.3 * lookAlpha;
    float spacing = tall * 0.2;
    float width = max(tall * saturate(dot(look, V)), spacing * 0.5);
    float3 up = float3(0, 1, 0) - R * R.y;
    up = dot(up, up) > 1e-6 ? normalize(up) : float3(1, 0, 0);
    float focal = Projection._22 * 0.5 * size.y;   // pixels per metre at a view depth of 1 m

    Texture2D<float> depthTex = ResourceDescriptorHeap[water.DepthGBufferSRV];
    float3 sum = 0;
    float weight = 0;
    [unroll]
    for (int k = -4; k <= 4; k++)
    {
        float w = exp(-0.103 * k * k);              // σ 1.1 × spacing: the five taps' 0.1 0.2 0.4 0.2 0.1
        float sn, cs;
        sincos(spacing * 0.5 * k, sn, cs);
        float3 dir = R * cs + up * sn;
        float height = dir.y * t;                   // over the water: a tap under it would meet the water again
        float4 qc = far ? mul(float4(dir, 0.0), CameraRelativeVP) : mul(float4(rel + dir * t, 1.0), CameraRelativeVP);
        if (height <= 0.0 || qc.w <= (far ? 1e-3 : 0.05))
            continue;
        float2 uv = qc.xy / qc.w * float2(0.5, -0.5) + 0.5;
        if (any(uv <= 0.0) || any(uv >= 1.0))
            continue;
        // What the snapshot shows there must stand above the water: under it is the bed, the water not drawn yet
        float sceneW = depthTex.Load(int3(int2(uv * size), 0));
        if (sceneW > 0.0 && posFromDepth(uv, Projection._33 + Projection._43 / sceneW, CameraInverse).y < rel.y - 0.05)
            continue;
        // The blur no wider than the tap stands over the water, so it does not reach down into the bed either
        float footprint = min(width * t, 2.0 * height) * focal / qc.w;
        sum += WaterSceneTap(water, uv, footprint) * w;
        weight += w;
    }
    if (weight <= 0.0)
        return 0;
    return float4(sum / weight, sure * edge * saturate(weight / 2.0));    // of 5.3 in all: the middle three carry 2.8
}

// The point lights' glints, as the sun's on open water: the lights that reach the place (light_clusters.hlsli). No shadows:
// a glint a pillar should hide costs nothing, a lookup per light per pixel of water would. rel: camera-relative.
float3 WaterPointSpecular(float3 rel, float3 N, float3 V, float alpha, float2 uv, float viewDepth)
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
        float3 toLight = light.Position - rel;
        float3 E = PointLightIlluminance(light, toLight, 0.25);
        if (any(E > 0.0))
            sum += WaterSunSpecular(N, V, normalize(toLight), E, alpha);
    }
    return sum * Exposure;
}

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 WorldPos : TEXCOORD0;
    float2 UV : TEXCOORD1;             // rivers: x = meters along the spline, y = meters across
    float Depth : TEXCOORD2;
    nointerpolation uint InstanceIdx : TEXCOORD3;
    float4 SkyAmbient : TEXCOORD4;     // rgb: the sky's light on a level surface (SkyAmbient(up), the sky probe), constant per frame
    float3 Flow : TEXCOORD5;           // rivers: xy = world xz direction of the current, z = drop per meter
};

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];

    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];

    uint vertexID = MeshVertexIndex(part, primitiveVertexID, primitiveVertexID, IndexedDraw);
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];

    uint dataPos = InstanceBaseOffset + instanceID;
    uint idx = sortedIndices[dataPos] & 0x7FFFFFFFu;

    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    float3 worldPos = mul(float4(positions[vertexID], 1.0f), World).xyz;

    // WaterBody stores the current in the normal stream of a river's mesh: xz = direction, y = steepness
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[part.NormBufferIdx];
    float3 flow = normals[vertexID];
    output.Flow = float3(mul(float4(flow.x, 0.0f, flow.z, 0.0f), World).xz, flow.y);

    output.WorldPos = worldPos;
    output.UV = uvs[vertexID];
    output.Position = mul(mul(float4(worldPos, 1.0), View), Projection);
    output.Depth = output.Position.w;
    output.InstanceIdx = idx;
    output.SkyAmbient = float4(SkyAmbient(float3(0, 1, 0)), 1.0);
    return output;
}

struct PSOutput
{
    float4 Color : SV_Target0;
};

PSOutput PS(VSOutput input, bool front : SV_IsFrontFace)
{
    PSOutput output;

    StructuredBuffer<WaterData> waterData = ResourceDescriptorHeap[WaterDataIdx];
    WaterData water = waterData[input.InstanceIdx];

    float3 camPos = ViewInverse[3].xyz;
    float3 worldPos = input.WorldPos;
    // How much sky this pixel of water sees past the terrain (per pixel: a lake is a few big triangles whose rim lies
    // under the banks, and per vertex those dark corners painted wedges across it)
    float skyVisible = SkyVisibility(worldPos, float3(0, 1, 0));
    float3 V = normalize(camPos - worldPos);
    float dist = distance(worldPos, camPos);
    float t = water.Time;
    bool flowing = water.Flowing > 0.5;

    // ── Flow frame (rivers) ──
    // The current's direction and how steeply this stretch falls (rapids) come per vertex from the
    // spline. Screen-space derivatives of the surface would be constant per triangle, and in a bend
    // the two triangles of a quad differ: that showed as stripes of white water.
    float2 flowDir = flowing && dot(input.Flow.xy, input.Flow.xy) > 1e-8 ? normalize(input.Flow.xy) : float2(1, 0);
    float2 acrossDir = float2(-flowDir.y, flowDir.x);
    float steep = flowing ? saturate(input.Flow.z) : 0.0;

    // ── Ice: how thick here, by how far out from the shore (WaterBody's shore field) and the current; the sheet's edge
    // jagged by noises interpolated linearly (kinks, not blobs: two smooth octaves made islands of ice all over a lake
    // whose middle sat at the threshold). The collider over it (WaterBody) follows the same rule without the edge ──
    float ice = 0.0, shore = 1.0;
    if (water.Ice > 0.0 && water.ShoreField != 0)
    {
        Texture2D<float> shoreField = ResourceDescriptorHeap[water.ShoreField];
        shore = shoreField.SampleLevel(AtmosphereLinearClamp, (worldPos.xz - water.ShoreOrigin) / water.ShoreSize, 0);
        float speed = flowing ? water.FlowSpeed * (1.0 + 4.0 * steep) : 0.0;
        float jag = IceJag(worldPos.xz / 23.0) * 0.55 + IceJag(worldPos.xz / 7.7 + 7.3) * 0.3 + IceJag(worldPos.xz / 2.3 - 2.9) * 0.15;
        ice = WaterIceThickness(water.Ice + jag * 0.09, shore, speed);
    }

    // Pattern space: lakes sample in world space; rivers in strip space (u along the flow, v across),
    // scrolled downstream. Strip space bends with the river, so the scroll never shears.
    float2 patternPos = flowing ? float2(input.UV.x - water.FlowSpeed * t, input.UV.y) : worldPos.xz;

    // ── Ripple normal and roughness: two ocean bands' slope moments (water_common.fx WaterSlopes) ──
    WaterSlopes slopes = (WaterSlopes)0;
    WaterSlopes calm = (WaterSlopes)0;      // the same over a footprint ~4× wider: the scene reflection's look-up
    float resolvedTotal = 0;
    if (water.SlopeSRV != 0)
    {
        Texture2DArray<float4> slopeTex = ResourceDescriptorHeap[water.SlopeSRV];
        float2 p = patternPos / max(0.05, water.RippleSize);
        float scale = water.RippleStrength * (1.0 + steep * 4.0);
        float3 coordA = float3(p * water.TileScaleA, water.BandA);
        float3 coordB = float3(p * water.TileScaleB + 0.37, water.BandB);
        float4 a = slopeTex.Sample(WaterSampler, coordA);
        float4 b = slopeTex.Sample(WaterSampler, coordB);
        float4 a2 = slopeTex.SampleBias(WaterSampler, coordA, WATER_LOOKUP_BIAS);
        float4 b2 = slopeTex.SampleBias(WaterSampler, coordB, WATER_LOOKUP_BIAS);

        // Strip space → world (a rotation: the variance stays)
        float2 meanA = flowing ? flowDir * a.x + acrossDir * a.y : a.xy;
        float2 meanB = flowing ? flowDir * b.x + acrossDir * b.y : b.xy;
        AddWaterBand(slopes, a, meanA, scale);
        AddWaterBand(slopes, b, meanB, scale);
        AddWaterBand(calm, a2, flowing ? flowDir * a2.x + acrossDir * a2.y : a2.xy, scale);
        AddWaterBand(calm, b2, flowing ? flowDir * b2.x + acrossDir * b2.y : b2.xy, scale);
        resolvedTotal = WaterBandTotalVariance(slopeTex, WaterSampler, water.BandA, scale)
                      + WaterBandTotalVariance(slopeTex, WaterSampler, water.BandB, scale);
    }
    // What moves in the water round the camera ripples it (WaterRipples): rings, wakes, the steps of a wader
    float rippleHeight;
    float2 ripple = RippleSlope(water.RipplesSRV, water.RipplesOrigin, water.RipplesSpan, worldPos.xz, rippleHeight);
    slopes.mean += ripple;
    calm.mean += ripple;
    float3 N = normalize(float3(-slopes.mean.x, 1.0, -slopes.mean.y));
    float unresolved = WaterUnresolvedVariance(resolvedTotal);
    float alpha = WaterAlpha(slopes, unresolved);
    float NdotV = saturate(dot(N, V));

    // ── Sun ──
    float3 L = normalize(-water.SunDirection);
    float NdotL = saturate(dot(N, L));
    // Pre-exposed E / π of the directional light, as the deferred light has it; the specular takes E (× π)
    float3 sunRadiance = water.SunColor * water.SunIntensity * SunLightScale;
    sunRadiance *= CloudBeam(WaterSampler, worldPos - camPos, L);     // the clouds' beam: no glint under a deck
    // The cascades, and the terrain's shadow past them (a mountain the sun stands behind), as the deferred light has them
    sunRadiance *= min(WaterSunShadow(ShadowMapIdx, worldPos - camPos, input.Position.xy), TerrainLightShadow(worldPos, L));
    sunRadiance *= TerrainOpen(worldPos, float3(0, 1, 0));   // a sea cave under a hill: no sun, whatever the shadow map holds
    float3 skyAmbient = input.SkyAmbient.rgb * skyVisible;

    // ── Seen from under the surface (the eye in the water; the mesh faces up): Snell's window onto the air above, bent by
    // the ripples, and past it the water below mirrored (underwater.hlsli). The underwater pass fades it over the way
    // through the water to it, as everything else down there. Under ice, a dim milky ceiling where it holds ──
    if (!front)
    {
        float3 above = 0;
        if (water.CompositeSRV != 0)
        {
            Texture2D<float4> snapshot = ResourceDescriptorHeap[water.CompositeSRV];
            float2 bent = saturate(input.Position.xy * water.InvViewportSize - N.xz * 0.06);
            above = snapshot.SampleLevel(AtmosphereLinearClamp, bent, 0).rgb;
        }
        float3 waterLight = WaterInscatter(water.WaterColor, sunRadiance, saturate(L.y), skyAmbient, 0.0, -V,
                                           water.ShallowColor, water.Visibility);
        float3 underside = WaterUnderside(N, V, above, waterLight);
        float3 iceCeiling = ICE_ALBEDO * (sunRadiance * saturate(L.y) * 0.25 + skyAmbient * 0.5) * 0.4;
        underside = lerp(underside, iceCeiling, smoothstep(0.0, 0.3, ice));
        output.Color = float4(max(underside, 0.0), 1.0);
        return output;
    }

    // ── Deep water color (same model as the ocean, without the wave-height term) ──
    float3 scatter = (0.1 * NdotV * NdotV + 0.08 * NdotL) * water.WaterColor * sunRadiance
                   + water.WaterColor * skyAmbient * 0.15;

    // ── Shared water shading ──
    float F = WaterSpecularReflectance(NdotV, alpha);
    float3 reflectColor = WaterSkyReflection(V, N, alpha) * skyVisible;   // no sky to mirror under a hill
    float3 specular = WaterSunSpecular(N, V, L, sunRadiance * 3.14159, alpha);

    // What stands round the water, where it is on the screen, over the sky's reflection (WaterScreenReflection). Without
    // the pyramids (EngineSettings.ScreenSpaceReflections off) only water the sky does not reach walks its one ray, as it
    // did before: there it is all there is to see. Where the sky is mostly hidden the point lights' glints are added
    float3 rel = worldPos - camPos;
    float covered = saturate(1.0 - skyVisible * 2.0);
    if (water.NearestDepthSRV != 0 && water.SceneColourSRV != 0)
    {
        // Its own roughness: only the ripple bands' slopes under the look-up's footprint (the wind shows through them: the
        // ocean simulation follows it). The shading's unresolved slopes (Cox-Munk's sea, water_common.fx: 0.003 with no
        // wind, 0.00512 more a m/s) blurred every reflected shape ~3-10° up and down until no shape was left (user,
        // 2026-11-13: "overblurred WAY too much"); the sky's lobe and the glints keep them, against the smooth sky they
        // cannot be seen
        float3 look = normalize(float3(-calm.mean.x, 1.0, -calm.mean.y));
        float4 scene = WaterScreenReflection(water, rel, V, look, WaterAlpha(calm, WATER_STILL_VARIANCE));
        reflectColor = lerp(reflectColor, scene.rgb, scene.a);
    }
    else if (covered > 0.0)
        reflectColor += WaterSceneReflection(water, rel, V, N, input.Position.xy) * covered;
    if (covered > 0.0)
        specular += WaterPointSpecular(rel, N, V, alpha, input.Position.xy * water.InvViewportSize, input.Depth) * covered;

    WaterColumn column = GetWaterColumn(WaterSampler, water.DepthGBufferSRV, water.CompositeSRV,
        input.Position.xy * water.InvViewportSize, input.Depth, dist, abs(camPos.y - worldPos.y), N,
        water.RefractionStrength, scatter, water.ShallowColor, water.Visibility, 1.0);

    // Light scattered back from a bright shallow bed
    float shallow = exp(-max(column.bufDepth, 0.0) / max(0.01, water.Visibility)) * saturate(column.pathLen);
    float3 body = column.body
                + water.ShallowColor * (sunRadiance * saturate(L.y) + skyAmbient) * 0.06 * shallow;

    float3 color = (1.0 - F) * body + (specular + F * reflectColor) * column.edgeFade;
    color = max(0.0, color);

    // ── Foam: coverage eroded by a cellular texture (as on the ocean) ──
    float coverage = 0.0;

    // Along the banks and around anything standing in the water
    coverage += (1.0 - smoothstep(0.0, 0.12, column.bufDepth)) * water.EdgeFoam;

    // White water where the surface runs steeply downhill
    // (kept below full coverage so the cell texture always breaks it up into streaks of dark water)
    coverage += smoothstep(0.04, 0.3, steep) * 0.68 * water.RapidsFoam;

    // Churned where the ripples are steep: a swimmer's strokes, a splash; not the rings of one treading water (their
    // slopes reach ~0.25 at the body: from 0.25 they made a white rim round it)
    coverage += smoothstep(0.45, 0.9, length(ripple)) * 0.6;
    coverage = saturate(coverage);

    float foamTex = 0.5;
    if (water.NoiseSRV != 0 && coverage > 0.001)
    {
        Texture2D<float4> noiseTex = ResourceDescriptorHeap[water.NoiseSRV];

        // Foam rides the current, faster where it is steep (bounded extra speed keeps it from shearing)
        float2 foamPos = flowing
            ? float2(input.UV.x - water.FlowSpeed * 1.6 * t, input.UV.y)
            : worldPos.xz + t * float2(0.01, 0.006);
        float cells1 = noiseTex.SampleLevel(WaterSampler, foamPos * 0.23, 0).g;
        float cells2 = noiseTex.SampleLevel(WaterSampler, foamPos * 0.83 + 0.5, 0).g;
        foamTex = saturate((cells1 * 0.6 + cells2 * 0.4 - 0.15) / 0.6);
        foamTex = lerp(foamTex, 0.5, saturate(dist / 150.0));   // no mips on the noise texture
    }

    float foam = saturate((coverage * 1.25 - (1.0 - foamTex)) * 3.0);
    foam *= smoothstep(0.0, 0.025, column.pathLen);     // never end on the hard edge at the bank
    if (foam > 0.001)
    {
        float3 foamLit = WaterFoamLit(float3(0.80, 0.80, 0.78) * lerp(0.7, 1.0, foamTex), NdotL, sunRadiance,
            SkyAmbient(N) * skyVisible);
        color = lerp(color, foamLit, foam * 0.9);
    }

    // ── The ice over the water, and the snow on it ──
    if (ice > 0.0)
    {
        // A sheet of plates (~7 m, cracked again at ~2 m) with straight borders between their corners, jagged by a
        // small zigzag (a smooth warp bent them into curves: "cracks mustnt be smooth curves", user); each plate tilted
        // a little, so they catch the light each its own way. Two kinds of ice in broad patches: clear (black) ice that
        // shows the dark water under it and mirrors the sky, and white ice, grey with the air frozen into it. A crack
        // shows light in clear ice, dark in white ice
        float2 p = worldPos.xz;
        float2 bent = p + float2(IceJag(p * 3.0), IceJag(p * 3.0 + 9.1)) * 0.12 + float2(IceJag(p * 11.0 + 4.2), IceJag(p * 11.0 - 2.7)) * 0.04;
        float2 plate, shard;
        float plateEdge = IceCells(bent / 7.0, plate);
        float shardEdge = IceCells(bent / 2.0 + 31.7, shard);
        float cracks = max(IceCrack(plateEdge, 0.035, 7.0), 0.5 * IceCrack(shardEdge, 0.015, 2.0)) * smoothstep(0.3, 0.8, ice);
        float2 tilt = (float2(MacroHash(plate), MacroHash(plate + 5.3)) - 0.5) * 0.015;
        float3 Ni = normalize(float3(tilt.x + (MacroNoise(p / 9.0) - 0.5) * 0.03, 1.0, tilt.y + (MacroNoise(p / 3.1 + 7.7) - 0.5) * 0.03));
        float NdotLi = saturate(dot(Ni, L));
        // The oldest ice, along the shore where the sheet began, has frozen whiter; out from there it is clear black ice
        // to the newest at the sheet's edge. Mottled by frost and frozen-in air a few metres across (jagged noise). Big
        // soft patches of white ice over the whole sheet read as smears painted on it (user, 2026-10-23: "looks roughed
        // in"); the white of a frozen lake is its snow, laid in streaks
        float frost = IceJag(p / 3.0) + 0.5 * IceJag(p / 1.1 + 6.1);
        float white = smoothstep(0.4, 0.05, shore) * saturate(0.7 + frost) * smoothstep(0.5, 1.0, ice);

        // Thin ice is a skin over the dark water; the water under it keeps what it showed but its mirror
        float opacity = lerp(0.35, 0.9, ice) * lerp(0.2, 1.0, white) * (1.0 + 0.3 * frost);
        float roughness = lerp(0.04, 0.15, white);
        float3 albedo = ICE_ALBEDO * (1.0 + 0.3 * frost) * lerp(1.0, 0.6, cracks * white);
        opacity = lerp(opacity, 0.8, cracks * (1.0 - white));                  // a white crack in clear ice
        float3 lit = albedo * (sunRadiance * NdotLi + SkyAmbient(Ni) * skyVisible);
        float3 iceBody = lerp((1.0 - F) * body, lit, opacity);
        float Fi = WaterSpecularReflectance(saturate(dot(Ni, V)), roughness);
        float3 iceColor = (1.0 - Fi) * iceBody + (WaterSunSpecular(Ni, V, L, sunRadiance * 3.14159, roughness)
                        + Fi * WaterSkyReflection(V, Ni, roughness) * skyVisible) * column.edgeFade;

        // Snow lies on ice that holds it, as the cover grows (on the ground: decals_cs.hlsl), in streaks the wind laid
        // down along itself over the open sheet (round blots of noise read as foam); none on the thin ice at the edge. The
        // way it was laid, held while it lies: the gusty wind of the moment slid the drifts to and fro across the lake
        float2 windAlong = dot(water.SnowAlong, water.SnowAlong) > 1e-4 ? normalize(water.SnowAlong) : float2(1, 0);
        float2 windFrame = float2(dot(p, windAlong), dot(p, float2(-windAlong.y, windAlong.x)));
        float drift = MacroNoise(windFrame / float2(14.0, 3.5)) * 0.65 + MacroNoise(windFrame / float2(5.0, 1.4) + 21.0) * 0.35;
        float bare = 1.0 - saturate(water.SnowCover);
        float snow = smoothstep(bare - 0.08, bare + 0.08, (0.35 + 0.65 * drift) * smoothstep(0.5, 1.0, ice));
        float3 snowLit = SNOW_ON_ICE * (sunRadiance * saturate(L.y) + SkyAmbient(float3(0, 1, 0)) * skyVisible);
        iceColor = lerp(iceColor, snowLit, snow);

        // The sheet's edge a line, not a blur: the rippled open water stops at the glossy ice, and along it a thin rim of
        // grey slush (frazil the wind and the current pile against the edge), a metre or two wide
        float rim = smoothstep(0.0, 0.02, ice) * (1.0 - smoothstep(0.04, 0.12, ice)) * saturate(0.2 + 2.0 * frost);
        float3 slush = ICE_ALBEDO * (sunRadiance * saturate(L.y) + SkyAmbient(float3(0, 1, 0)) * skyVisible);
        iceColor = lerp(iceColor, slush, rim * 0.8);
        color = lerp(color, iceColor, smoothstep(0.0, 0.02, ice));
    }

    // ── Aerial perspective ──
    color = ApplyAerialPerspective(color, input.Position.xy, input.Depth, skyVisible);

    // HDR output — the finalize pass tonemaps the whole Composite
    output.Color = float4(color, 1.0);
#if WATER_TRACE_DEBUG
    output.Color = float4(WaterTraceDebug * 0.6, 1.0);
#endif
#if RIPPLE_DEBUG
    if (water.RipplesSRV != 0)
        output.Color = float4(saturate(rippleHeight * 50.0 + 0.5), saturate(length(ripple) * 2.0), saturate(-rippleHeight * 50.0 + 0.5), 1.0);
#endif
    return output;
}

technique11 GBuffer
{
    pass Forward
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
