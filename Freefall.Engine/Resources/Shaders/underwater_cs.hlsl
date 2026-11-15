// The camera in the water (Graphics/Underwater.cs), on the HDR composite after the forward pass has drawn the water and
// before the glass and particles. A pixel is under water when its point on the near plane is under the surface near the
// camera (a level lake, a river's slope, the sea's waves measured at the camera by CSProbe): there
//   - what it sees fades into the water's own light over the way through the water (underwater.hlsli: the same water
//     the surfaces show from above), to the scene or to the surface drawn from below, or to where the ray leaves the
//     water where nothing was drawn;
//   - what lies under the surface gets the daylight the water above it lets down, and the sun's play over it (caustics);
//   - where the near plane crosses the surface (the eye half in) a thin dark meniscus marks the waterline.
// CSProbe: the sea's height and slope at a few places (the camera's first), from the ocean's own displacement
// (ocean_surf.hlsli), for this pass and, read back, for swimmers (Components/Water.cs).
// Docs: .agent/knowledge/ocean_system/underwater.md

#pragma kernel CSUnderwater
#pragma kernel CSProbe

cbuffer PushConstants : register(b3)
{
    uint FrameIdx;          // StructuredBuffer<UnderwaterFrame>: [0]
    uint PlacesIdx;         // StructuredBuffer<float4>: the probe's places (xz)
    uint ProbeUAVIdx;       // RWStructuredBuffer<float4>: per place the sea's height, its slope over x and z, 1
    uint CompositeUAVIdx;   // RWTexture2D<float4>: the HDR composite, pre-exposed
};

#include "common.fx"
#include "sky_common.fx"
#include "ocean_surf.hlsli"
#include "underwater.hlsli"

// Underwater.Frame (112 bytes): UnderwaterFrame in underwater.hlsli

// ── The sea at a place ──

// Where the sea's surface stands over a place: the ocean displaces each point of its grid sideways too, so the point that
// ends up over xz is found by going back by its sideways displacement (three steps settle it)
float SeaHeight(OceanData ocean, float2 xz)
{
    float2 p = xz;
    float3 d = 0;
    [unroll]
    for (int k = 0; k < 3; k++)
    {
        OceanShore shore;
        SurfWave surf;
        d = OceanSurfaceDisplacement(ocean, p, 0.0, shore, surf);
        p = xz - d.xz;
    }
    return ocean.OceanPlaneY + d.y;
}

[numthreads(8, 1, 1)]
void CSProbe(uint3 id : SV_DispatchThreadID)
{
    StructuredBuffer<UnderwaterFrame> frames = ResourceDescriptorHeap[FrameIdx];
    UnderwaterFrame f = frames[0];
    if (id.x >= f.ProbeCount || f.OceanData == 0)
        return;
    StructuredBuffer<OceanData> oceans = ResourceDescriptorHeap[f.OceanData];
    OceanData ocean = oceans[0];
    StructuredBuffer<float4> places = ResourceDescriptorHeap[PlacesIdx];
    float2 xz = places[id.x].xy;

    // The slope over 0.35 m: a body's breadth, not the ripples'
    const float e = 0.35;
    float h = SeaHeight(ocean, xz);
    float hx = SeaHeight(ocean, xz + float2(e, 0.0));
    float hz = SeaHeight(ocean, xz + float2(0.0, e));
    RWStructuredBuffer<float4> probe = ResourceDescriptorHeap[ProbeUAVIdx];
    probe[id.x] = float4(h, (hx - h) / e, (hz - h) / e, 1.0);
}

// ── Caustics: the sunlight focused by the waves into a moving net of bright lines on what lies below ──

float CausticHash(float2 p)
{
    return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

// Cells round points that sway on their own circles: the distance to the nearest border between two cells (F2 − F1), 0
// on the borders, where the net's lines are
float CausticCells(float2 p, float t)
{
    float2 i = floor(p), fp = frac(p);
    float d1 = 8.0, d2 = 8.0;
    [unroll] for (int y = -1; y <= 1; y++)
    [unroll] for (int x = -1; x <= 1; x++)
    {
        float2 g = float2(x, y);
        float h = CausticHash(i + g);
        float a = 6.2831853 * h + t * (0.6 + 0.5 * CausticHash(i + g + 7.1));
        float2 r = g + 0.5 + 0.35 * float2(cos(a), sin(a)) - fp;
        float d = dot(r, r);
        if (d < d1) { d2 = d1; d1 = d; }
        else if (d < d2) d2 = d;
    }
    return sqrt(d2) - sqrt(d1);
}

// The cells' space bent by slow waves, so their borders curve as the lines of light a rippled surface throws do (straight
// borders read as paving)
float2 CausticWarp(float2 p, float t)
{
    return p + 0.28 * float2(sin(p.y * 1.7 + t * 0.6) + 0.5 * sin(p.y * 3.1 - t * 0.9),
                             sin(p.x * 1.9 - t * 0.5) + 0.5 * sin(p.x * 2.7 + t * 0.8));
}

// Two nets of thin lines a little over a metre across, drifting apart, brightest where they cross; about CAUSTICS_MEAN on
// average. p: where the sunlight that reaches the place came in through the surface (world xz)
float Caustics(float2 p, float t)
{
    float2 pa = CausticWarp(p / 1.5, t);
    float2 pb = CausticWarp(p / 0.95 + 5.3, t * 1.2);
    float a = exp(-22.0 * CausticCells(pa, t * 0.6));
    float b = exp(-22.0 * CausticCells(pb, t * 0.8));
    return 0.8 * (a + b) + 1.5 * a * b;
}
#define CAUSTICS_MEAN 0.2

// ── Light shafts: the same focusing seen in the water itself. Down through the water the net blurs (the rays from each
// focus spread), so the shafts are the caustics' cells wider and softer, sampled at points along the view ray where the
// sunlight that reaches them came in, each lit as the daylight there is and seen through the water up to it ──
#define SHAFT_STEPS 6           // 10 cost 0.48 ms at 1618 × 1249 (the pass without them 0.085)
#define SHAFT_MEAN 0.27
#define SHAFT_GAIN 0.08
float ShaftField(float2 p, float t)
{
    return exp(-6.0 * CausticCells(CausticWarp(p / 2.6, t * 0.5), t * 0.4));   // narrower cells average out along a ray
}

[numthreads(8, 8, 1)]
void CSUnderwater(uint3 id : SV_DispatchThreadID)
{
    RWTexture2D<float4> composite = ResourceDescriptorHeap[CompositeUAVIdx];
    uint width, height;
    composite.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
        return;

    StructuredBuffer<UnderwaterFrame> frames = ResourceDescriptorHeap[FrameIdx];
    UnderwaterFrame f = frames[0];

    // The surface near the camera, camera-relative: its height over a place xz is rise + slope · xz
    float surfaceY = f.SurfaceY;
    float2 slope = f.Slope;
    if (f.Sea != 0)
    {
        RWStructuredBuffer<float4> probe = ResourceDescriptorHeap[ProbeUAVIdx];
        float4 sea = probe[0];
        if (sea.w > 0.0)
        {
            surfaceY = sea.x;
            slope = sea.yz;
        }
    }
    float rise = surfaceY - CamPos.y;

    // This pixel's point on the near plane (reverse Z: depth 1), and the view ray through it
    float2 size = float2(width, height);
    float2 uv = (float2(id.xy) + 0.5) / size;
    float3 nearRel = posFromDepth(uv, 1.0, CameraInverse).xyz;
    float3 dir = normalize(nearRel);
    float under = rise + dot(slope, nearRel.xz) - nearRel.y;     // how far the near point lies under the surface

    // The waterline: a meniscus a few pixels wide where the near plane crosses the surface (the near plane's own pixel
    // size, as the line is drawn on it)
    float3 nextRel = posFromDepth(uv + float2(0.0, 1.0 / size.y), 1.0, CameraInverse).xyz;
    float pixel = max(length(nextRel - nearRel), 1e-6);
    float meniscus = 1.0 - smoothstep(0.0, 3.0 * pixel, abs(under));
    float4 here = composite[id.xy];
    if (under <= 0.0)
    {
        if (meniscus > 0.0)
            composite[id.xy] = float4(here.rgb * (1.0 - 0.5 * meniscus), here.a);
        return;
    }

    // What the ray meets: the scene, the surface drawn from below, or nothing (the sky's depth 0)
    Texture2D<float> depthTex = ResourceDescriptorHeap[f.DepthSrv];
    float d = depthTex.Load(int3(id.xy, 0));
    float3 sceneRel = d > 0.0 ? posFromDepth(uv, d, CameraInverse).xyz : dir * 1e5;
    float dist = length(sceneRel);
    float sceneSurface = rise + dot(slope, sceneRel.xz);
    bool sceneUnder = sceneRel.y < sceneSurface;

    // What the ray meets is the water itself (its surface drawn from below) where the full depth is nearer than the opaque
    // scene's: the plane measured at the camera is no guide far off, where a wave's trough lies under it
    Texture2D<float> opaqueDepth = ResourceDescriptorHeap[f.LinearDepthSrv];
    float opaqueW = opaqueDepth.Load(int3(id.xy, 0));
    float viewW = d > 0.0 ? Projection._43 / max(d - Projection._33, 1e-9) : 1e9;
    bool surface = d > 0.0 && (opaqueW <= 0.0 || viewW < opaqueW - 0.05);

    // Past the surface where nothing was drawn there (the far side of a lake's outline, the sky through a gap): the
    // water ends where the ray goes up through its plane
    if (!sceneUnder && !surface)
    {
        float leaves = dir.y - dot(slope, dir.xz);
        if (leaves > 1e-4)
            dist = min(dist, length(nearRel) + under / leaves);
    }

    float3 color = here.rgb;
    float3 L = -normalize(f.SunDirection);
    float3 sunRadiance = f.SunColor * f.SunIntensity * SunLightScale;
    float3 sky = SkyAmbient(float3(0.0, 1.0, 0.0));

    // The daylight that reaches what lies down there, and the sun's play over its tops
    if (d > 0.0 && sceneUnder && !surface)
    {
        float depthThere = sceneSurface - sceneRel.y;
        float3 daylight = WaterDaylightAt(depthThere, f.ShallowColor, f.Visibility);
        float play = 0.0;
        if (f.Caustics > 0.0 && f.SunUp > 0.02)
        {
            Texture2D<float4> normals = ResourceDescriptorHeap[f.NormalsSrv];
            float3 n = normals.Load(int3(id.xy, 0)).xyz;
            float facing = saturate(dot(n, L)) * smoothstep(0.02, 0.15, f.SunUp);
            float fade = smoothstep(0.05, 0.4, depthThere) * (1.0 - smoothstep(18.0, 35.0, dist));
            if (facing * fade > 0.0)
            {
                float3 world = sceneRel + CamPos;
                float2 entered = world.xz + depthThere * L.xz / max(L.y, 0.25);
                play = f.Caustics * facing * fade * (Caustics(entered, Time) - CAUSTICS_MEAN) * 2.0;
            }
        }
        color *= daylight * max(0.0, 1.0 + play);
    }

    // The way through the water fades into its own light, as at the depth half way along what can be seen of it
    float tMid = 0.5 * min(dist, 3.0 * f.Visibility);
    float3 mid = dir * tMid;
    float midDepth = max(0.0, rise + dot(slope, mid.xz) - mid.y);
    float3 T = WaterTransmittance(dist, f.ShallowColor, f.Visibility);
    float3 fog = WaterInscatter(f.WaterColor, sunRadiance, f.SunUp, sky, midDepth, dir, f.ShallowColor, f.Visibility);
    color = color * T + fog * (1.0 - T);

    // The shafts: brighter and darker than the water's mean glow along the way, more so looking toward the sun (light
    // in water scatters mostly forward)
    if (f.Shafts > 0.0 && f.SunUp > 0.02)
    {
        float reach = min(dist, 2.0 * f.Visibility);
        float jitter = frac(52.9829189 * frac(dot(float2(id.xy), float2(0.06711056, 0.00583715))));
        float3 shaft = 0;
        [loop]
        for (uint k = 0; k < SHAFT_STEPS; k++)
        {
            float t = (k + jitter) / SHAFT_STEPS * reach;
            float3 p = dir * t;
            float depthHere = max(0.0, rise + dot(slope, p.xz) - p.y);
            float2 entered = (p + CamPos).xz + depthHere * L.xz / max(L.y, 0.25);
            float3 seen = WaterTransmittance(t, f.ShallowColor, f.Visibility) * WaterDaylightAt(depthHere, f.ShallowColor, f.Visibility);
            shaft += seen * (ShaftField(entered, Time) - SHAFT_MEAN);
        }
        float toward = saturate(dot(dir, L));
        float phase = 0.4 + 2.0 * toward * toward * toward;
        float3 hue = sqrt(f.WaterColor / max(max(f.WaterColor.r, max(f.WaterColor.g, f.WaterColor.b)), 1e-4));
        color += hue * sunRadiance * shaft * (reach / SHAFT_STEPS) * (f.Shafts * SHAFT_GAIN * phase) * smoothstep(0.02, 0.15, f.SunUp);
    }

    color *= 1.0 - 0.5 * meniscus;
    composite[id.xy] = float4(max(color, 0.0), here.a);
}
