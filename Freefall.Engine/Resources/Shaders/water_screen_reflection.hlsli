// ── The scene in the water: screen-space reflections for lakes and rivers (water.fx) and the sea (ocean.fx) ──
// (Graphics/WaterReflections.cs, water_reflections_cs.hlsl; ocean_system/overview.md "Screen-space reflections on lakes
// and rivers".) What water mirrors besides the sky: the banks, trees, houses and ships round it, found on the screen.
// Tried once and taken out (2026-10-05) for noise: rays along each pixel's own rippled normal, walked in steps with a
// per-pixel dither, hit a leaf in one pixel and fell through a gap in the next. Here every pixel traces the mirror ray of
// its ripple (or wave) normal low-passed over ~4 × 4 pixels, so neighbours trace nearly the same ray, cell by cell
// through the nearest-depth pyramid, which finds the first pixel the ray goes behind exactly: no dither, no step that
// skips a post. The slopes finer than that blur what it finds into a vertical streak, as WaterSkyReflection spreads the
// sky's, from the scene's colour pyramid.
// Needs common.fx (camera, posFromDepth) and sky_common.fx (AtmosphereLinearClamp) before it.

#ifndef WATER_SCREEN_REFLECTION_HLSLI
#define WATER_SCREEN_REFLECTION_HLSLI

#define WATER_TRACE_STEPS 96        // cells visited at most; a ray over a screen of open ground takes ~30-60
#define WATER_BEHIND_PIXELS 12      // pixels a ray walks on behind something nearer, looking for what it hides
#define WATER_LOOKUP_BIAS 2.0       // mips coarser than the shading's: the look-up normal is the mean over ~4 × 4 pixels
#define WATER_STILL_VARIANCE 0.0002 // slope variance under the ripple bands for the look-up's blur (α 0.014: ~0.2° up and down)

// WATER_TRACE_DEBUG 1 draws the water as what its trace did (saving the file reloads it in a running editor): green met
// something (olive: past something nearer), red stood in for what something nearer hid, blue reached the sky on the
// screen, grey left the screen, yellow ran out of steps, magenta met the sky. The lakes' and the sea's PS show it
#define WATER_TRACE_DEBUG 0
static float3 WaterTraceDebug = 0;

// What the trace reads: from WaterData (water.fx) or OceanData (ocean.fx)
struct WaterScreenInputs
{
    uint DepthSRV;              // the linear view depth (0 = sky)
    uint CompositeSRV;          // the composite snapshot (the scene before the water)
    uint NearestDepthSRV;       // WaterReflections: the nearest 1 / view depth, half size and down; 0 = none
    uint NearestDepthMips;
    uint SceneColourSRV;        // WaterReflections: the snapshot averaged, half size and down
    uint SceneColourMips;
    float2 InvViewportSize;
};

// The nearest 1 / view depth in a cell of the trace: level 0 is the full-size linear depth, level k ≥ 1 the pyramid's mip
// k − 1 (cells 2^k pixels wide; a mip's last texel also covers what is left over at the screen's right and bottom)
float WaterNearest(WaterScreenInputs s, int2 cell, int level, int2 fullSize)
{
    if (level == 0)
    {
        Texture2D<float> depth = ResourceDescriptorHeap[s.DepthSRV];
        float w = depth.Load(int3(cell, 0));
        return w > 0.0 ? 1.0 / w : 0.0;
    }
    Texture2D<float> pyramid = ResourceDescriptorHeap[s.NearestDepthSRV];
    int2 dims = max((fullSize / 2) >> (level - 1), 1);
    return pyramid.Load(int3(min(cell, dims - 1), level - 1));
}

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
float WaterTrace(WaterScreenInputs inputs, float3 rel, float3 r, out float2 hitPx, out float hitT, out bool open)
{
    hitPx = 0;
    hitT = 0;
    open = false;
    float2 size = 1.0 / inputs.InvViewportSize;
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
    int top = (int)inputs.NearestDepthMips;
    uint behind = 0;                            // pixels walked behind something nearer
    Texture2D<float> depthTex = ResourceDescriptorHeap[inputs.DepthSRV];
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
        float nearest = WaterNearest(inputs, int2(cell), level, fullSize);
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
float3 WaterSceneTap(WaterScreenInputs inputs, float2 uv, float footprint)
{
    float lod = log2(max(footprint, 1.0));
    Texture2D<float4> pyramid = ResourceDescriptorHeap[inputs.SceneColourSRV];
    float3 coarse = pyramid.SampleLevel(AtmosphereLinearClamp, uv, clamp(lod - 1.0, 0.0, inputs.SceneColourMips - 1.0)).rgb;
    if (lod >= 1.0)
        return coarse;
    Texture2D<float4> composite = ResourceDescriptorHeap[inputs.CompositeSRV];
    float3 fine = composite.SampleLevel(AtmosphereLinearClamp, uv, 0).rgb;
    fine *= min(1.0, WATER_SCENE_MAX / max(max(fine.r, max(fine.g, fine.b)), 1e-6));
    return lerp(fine, coarse, lod);
}

// What water at rel mirrors of the scene: rgb, and in a how much of it takes the sky's place (0 where the trace found
// nothing). levelY: the camera-relative height of the still water (a lake's surface, the sea's level under its waves):
// what the snapshot shows below it is the bed, the water not drawn yet. look: the ripple normal low-passed for the
// look-up; lookAlpha: the roughness of the slopes finer than it
float4 WaterScreenReflection(WaterScreenInputs inputs, float3 rel, float levelY, float3 V, float3 look, float lookAlpha)
{
    if (inputs.NearestDepthSRV == 0 || inputs.SceneColourSRV == 0 || inputs.DepthSRV == 0 || inputs.CompositeSRV == 0)
        return 0;
    if (V.y <= 0.0)
        return 0;                                   // seen from under the surface
    // The mirror ray of the ripple normal low-passed over ~4 × 4 pixels, traced itself: neighbours still trace nearly the
    // same ray (the 2026-10-05 noise came from each pixel's own full-resolution normal), and what the water shows follows
    // the waves. A level surface's ray (traced until 2026-11-25, the waves only bending where along its length the scene
    // was looked up) kept every reflection's outline the scene's exact mirror image: on a choppy sea an island's flat top
    // and its trees stood in the water as a still cutout (user: "the reflection on the water is like a static cutout").
    // Tilted toward the eye a wave sends the ray up past what stands on the shore, tilted away into the next wave
    float3 R = reflect(-V, look);
    if (R.y <= 0.0)
        return 0;                                   // into the next wave: the sky probe's lobe has it (WaterSkyTap)
    float2 hitPx;
    float t;
    bool open;
    float sure = WaterTrace(inputs, rel, R, hitPx, t, open);
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

    float2 size = 1.0 / inputs.InvViewportSize;
    // Faded toward the screen's edges, where the ray would have found more past them. Over a quarter of the screen at the
    // top, where nearly every reflection leaves it: over 8 % the sky probe took over from the screen's sky along a line
    // across the lake, the clouds stopping at it
    float2 huv = hitPx / size;
    float edge = saturate(min(huv.x, 1.0 - huv.x) / 0.08) * saturate((1.0 - huv.y) / 0.08) * smoothstep(0.0, 0.25, huv.y);

    // Spread up and down a lobe round that ray, as far as it went, shaped as WaterSkyReflection spreads the sky's but half
    // as tall (1.3 α, not 2.6: the full height still smeared the rocks of a bank into streaks), in nine taps 0.1 × that
    // apart, each blurred as wide as the lobe is across: seen at a grazing angle a slope tilts the mirror ray up and down
    // far more than to the side, so a reflection streaks downward and stays narrow
    float tall = 1.3 * lookAlpha;
    float spacing = tall * 0.2;
    float width = max(tall * saturate(dot(look, V)), spacing * 0.5);
    float3 up = float3(0, 1, 0) - R * R.y;
    up = dot(up, up) > 1e-6 ? normalize(up) : float3(1, 0, 0);
    float focal = Projection._22 * 0.5 * size.y;   // pixels per metre at a view depth of 1 m

    Texture2D<float> depthTex = ResourceDescriptorHeap[inputs.DepthSRV];
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
        if (sceneW > 0.0 && posFromDepth(uv, Projection._33 + Projection._43 / sceneW, CameraInverse).y < levelY - 0.05)
            continue;
        // The blur no wider than the tap stands over the water, so it does not reach down into the bed either
        float footprint = min(width * t, 2.0 * height) * focal / qc.w;
        sum += WaterSceneTap(inputs, uv, footprint) * w;
        weight += w;
    }
    if (weight <= 0.0)
        return 0;
    return float4(sum / weight, sure * edge * saturate(weight / 2.0));    // of 5.3 in all: the middle three carry 2.8
}

#endif
