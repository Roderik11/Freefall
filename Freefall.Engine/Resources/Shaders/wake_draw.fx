// Wakes on the water (Graphics/WakeTrails.cs): ribbons lying on the surface along the way a ship went, drawn after the
// water (the OIT passes; at the Forward pass's end with sorted transparency). Vertex-pull quads as trail_draw.fx: one
// list of every wake's points, instance i the piece from point i to i + 1. Each vertex is put on the sea's displaced
// surface over its place (ocean_surf.hlsli, as the underwater probe finds it), or on a lake's or river's still surface.
// The foam is drawn in streaks along the wake: noise stretched along it and cut at a threshold that rises as the wake
// ages, so the white water behind a stern breaks into streaks and thins out; under it the churned water is paler. A
// bow wave's arm (WAKE_CREST) is a breaking crest: foam along its middle that breaks up fast.
// Colours are premultiplied. Docs: .agent/knowledge/engine_internals/ships.md.
//
// @RenderState(RenderTargets=1, DepthWrite=false, DepthFunc=Always, Blend=AlphaBlend, CullMode=None, RenderTargetFormat=R16G16B16A16_Float)

#include "common.fx"
#include "sky_common.fx"
#include "ocean_surf.hlsli"
#include "water_common.fx"
#include "ocean_shade.hlsli"
#include "water_hulls.hlsli"
#include "oit.hlsli"
// OceanSampler (s0, linear wrap) comes from ocean_surf.hlsli

SamplerComparisonState WakeShadowSampler : register(s3);    // less-equal: the sun's cascades, point light shadows

cbuffer PushConstants : register(b3)
{
    uint PointsIdx;     // DWORD 0  SRV: WakePoint list
    uint OceanIdx;      // DWORD 1  SRV: the ocean's OceanData (OceanRenderer.SurfDataSrv); 0 = none
    uint HullsIdx;      // DWORD 2  WaterHulls.Srv: no wake inside a hull
    uint HullCount;     // DWORD 3
};

struct WakePoint        // 48 bytes = WakeTrails.GpuPoint in C#
{
    float3 Position; float Width;
    float2 Right; float Along; float Age;
    float Strength; uint Flags; float Life; float _pad;
};

#define WAKE_FIRST 1
#define WAKE_LAST  2
#define WAKE_SEA   4    // rides the sea's waves
#define WAKE_CREST 8    // a bow wave's arm

static const float2 Corners[6] =
{
    float2(0, -1), float2(1, -1), float2(0, 1),
    float2(0, 1), float2(1, -1), float2(1, 1),
};

struct VSOutput
{
    float4 Position : SV_Position;
    float3 Light    : COLOR0;       // what lights a white surface here: the sky, the sun through its shadows, lamps
    float  Across   : TEXCOORD0;    // -1..1 from edge to edge
    float  Along    : TEXCOORD1;    // metres along the wake
    float  Width    : TEXCOORD2;
    float  Age      : TEXCOORD3;    // share of its life gone
    float  Strength : TEXCOORD4;
    nointerpolation float Crest : TEXCOORD5;
    float  Depth    : TEXCOORD6;
    float  SkyVisible : TEXCOORD7;
    float3 World    : TEXCOORD8;
    float2 Right    : TEXCOORD9;    // across the way it was laid (unit xz)
};

// The sea's surface over a place: the point of the ocean's grid that is displaced to it, found by going back by its
// sideways displacement (two steps settle it), as the underwater probe does
float SeaHeight(OceanData ocean, float2 xz, float camDist)
{
    float2 p = xz;
    float3 d = 0;
    [unroll]
    for (int k = 0; k < 2; k++)
    {
        OceanShore shore;
        SurfWave surf;
        d = OceanSurfaceDisplacement(ocean, p, camDist, shore, surf);
        p = xz - d.xz;
    }
    return ocean.OceanPlaneY + d.y;
}

VSOutput VS(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output = (VSOutput)0;
    StructuredBuffer<WakePoint> points = ResourceDescriptorHeap[PointsIdx];
    if ((points[instanceID].Flags & WAKE_LAST) != 0)
    {
        output.Position = float4(2.0, 2.0, 2.0, 1.0);
        return output;
    }
    float2 corner = Corners[vertexID % 6];
    WakePoint p = points[instanceID + (uint)corner.x];

    float3 world = p.Position + float3(p.Right.x, 0.0, p.Right.y) * (corner.y * 0.5 * p.Width);
    if ((p.Flags & WAKE_SEA) != 0 && OceanIdx != 0)
    {
        StructuredBuffer<OceanData> oceans = ResourceDescriptorHeap[OceanIdx];
        world.y = SeaHeight(oceans[0], world.xz, distance(world, CamPos));
    }
    world.y += 0.04;

    float3 relPos = world - CamPos;
    float3 toEye = normalize(-relPos);
    float skyVisible = SkyVisibility(world, float3(0, 1, 0));
    output.Light = ParticleLight(OceanSampler, relPos, toEye, 0.0, ParticleBodyShadows(relPos, WakeShadowSampler)) * skyVisible
                 + ParticlePointLight(relPos, toEye, 0.0, 1.0, WakeShadowSampler);
    output.SkyVisible = skyVisible;

    float4 clip = mul(float4(world, 1.0), ViewProjection);
    output.Position = clip;
    output.Across = corner.y;
    output.Along = p.Along;
    output.Width = p.Width;
    output.Age = p.Age;
    output.Strength = p.Strength;
    output.Crest = (p.Flags & WAKE_CREST) != 0 ? 1.0 : 0.0;
    output.Depth = clip.w;
    output.World = world;
    output.Right = p.Right;
    return output;
}

// The wake's own noise: 2D simplex (Gustavson / Ashima), -1..1, no grid in it. The ocean's noise texture sat so close
// round its middle that no cut let anything through; value noise cut at a threshold showed its cells as blocks
float3 WakeMod289(float3 x) { return x - floor(x / 289.0) * 289.0; }
float2 WakeMod289(float2 x) { return x - floor(x / 289.0) * 289.0; }
float3 WakePermute(float3 x) { return WakeMod289((x * 34.0 + 1.0) * x); }

float WakeSimplex(float2 v)
{
    const float4 C = float4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    float2 i = floor(v + dot(v, C.yy));
    float2 x0 = v - i + dot(i, C.xx);
    float2 i1 = x0.x > x0.y ? float2(1.0, 0.0) : float2(0.0, 1.0);
    float4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = WakeMod289(i);
    float3 p = WakePermute(WakePermute(i.y + float3(0.0, i1.y, 1.0)) + i.x + float3(0.0, i1.x, 1.0));
    float3 m = max(0.5 - float3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;
    float3 x = 2.0 * frac(p * C.www) - 1.0;
    float3 h = abs(x) - 0.5;
    float3 ox = floor(x + 0.5);
    float3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    float3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

// 0..1 noise of the wake's at a place in its own metres (across, along)
float WakeNoise(float2 p)
{
    return WakeSimplex(p) * 0.5 + 0.5;
}

float4 WakeColor(VSOutput input, bool fogged)
{
    if (InsideHull(HullsIdx, HullCount, input.World))
        discard;

    float a = input.Across;
    bool crest = input.Crest > 0.5;
    // Metres across and along: streaks drawn out along the way it went, fine across it
    float2 q = float2(a * input.Width * 0.5, input.Along);
    float streak = 0.5;
    if (OceanIdx != 0)
    {
        StructuredBuffer<OceanData> oceans = ResourceDescriptorHeap[OceanIdx];
        // Drawn without the depth test (DepthFunc=Always): a ribbon a few vertices across lies flat between them, and
        // the waves shorter than its width rose through it in jagged steps (2026-11-20). What is opaque in front of it
        // (a hull, the shore) hides it here, by the scene's linear depth (0 = sky)
        if (oceans[0].DepthGBufferSRV != 0)
        {
            Texture2D<float> sceneDepth = ResourceDescriptorHeap[oceans[0].DepthGBufferSRV];
            float sceneZ = sceneDepth.Load(int3(input.Position.xy, 0));
            if (sceneZ > 0.0 && input.Depth > sceneZ * 1.002 + 0.05)
                discard;
        }
    }
    {
        // Where along the wake the white water is thick and thin: broad streaks drawn out along the way it went, gently
        // curled (a strong warp smeared them into flames)
        float2 warp = float2(WakeSimplex(q * float2(0.12, 0.03) + 5.2), WakeSimplex(q * float2(0.12, 0.03) + 9.7));
        float2 w = q + warp * float2(0.7, 3.0);
        float n1 = WakeNoise(w * float2(0.3, 0.05) + 13.1);
        float n2 = WakeNoise(w * float2(0.9, 0.15) + 41.7);
        streak = n1 * 0.65 + n2 * 0.35;
        streak = lerp(streak, 0.5, saturate(input.Depth / 400.0 - 0.3));
    }

    // How much white water there is: most right behind the stern, thinning over its life; a bow wave's crest along its
    // middle, whole where it leaves the bow and breaking up as it runs out
    float age = saturate(input.Age);
    float edge = crest ? exp(-a * a * 4.0) : smoothstep(1.0, 0.3, abs(a));
    float life = crest ? pow(1.0 - age, 1.8) : pow(1.0 - age, 1.3);
    float spread = saturate((streak - 0.5) * 2.4 + 0.5);
    float amount = saturate(input.Strength * life * edge) * (crest ? lerp(0.95, 0.6, age) : 0.85) * lerp(0.35, 1.15, spread);

    // The white water itself is the surf's (ocean_shade.hlsli SurfFoamLook): lumps a few decimetres across that catch
    // the light on their tops, eroded at their edges into bubbles, pulled out along the trail. Flat white shapes cut from
    // a noise read as paint, not foam (user, 2026-11-20)
    float foam = saturate(amount * 1.6 - 0.2);
    float shade = 1.0;
    if (OceanIdx != 0)
    {
        StructuredBuffer<OceanData> oceans = ResourceDescriptorHeap[OceanIdx];
        if (oceans[0].NoiseSRV != 0)
        {
            Texture2D<float4> noise = ResourceDescriptorHeap[oceans[0].NoiseSRV];
            float2 heading = float2(-input.Right.y, input.Right.x);
            float footprint = input.Depth * 2.0 / (Projection._22 * max(ViewportSize.y, 1.0));
            SurfFoamMass m = SurfFoamLook(noise, input.World.xz, Time, amount, 1.0, heading, input.Depth, footprint);
            foam = m.cover;
            shade = SurfFoamShade(m, amount);
        }
    }
    // The churned water under the stern's wake: paler where it is full of bubbles, mottled by the broad streaks, lasting
    // longer than the foam (an even band of it read as a beam of light on the water)
    float churned = crest ? 0.0 : 0.14 * input.Strength * pow(1.0 - age, 0.7) * smoothstep(1.0, 0.15, abs(a))
                                  * saturate(0.35 + 1.3 * spread);

    float3 foamLight = input.Light * 0.85 * shade;
    float3 churnLight = input.Light * float3(0.32, 0.55, 0.52) * 0.35;
    float foamA = foam * 0.92;
    float4 color;
    color.a = 1.0 - (1.0 - foamA) * (1.0 - churned);
    color.rgb = foamLight * foamA + churnLight * churned * (1.0 - foamA);
    if (color.a < 0.002)
        discard;

    if (fogged && FogEnabled > 0.0 && AtmosphereIdx != 0)
    {
        float4 fog = AtmosphereFog(input.Position.xy, input.Depth, input.SkyVisible);
        color.rgb = color.rgb * fog.a + fog.rgb * color.a;
    }
    return color;
}

float4 PS(VSOutput input) : SV_Target0
{
    return WakeColor(input, true);
}

OITMomentTargets PS_OITMoments(VSOutput input)
{
    return OITMoments(WakeColor(input, false).a, input.Depth);
}

OITTargets PS_OIT(VSOutput input)
{
    return OITAccumulate(WakeColor(input, true), input.Depth, input.Position.xy);
}

technique11 Wakes
{
    pass Forward
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
    pass OITMoments
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS_OITMoments()));
        SetRenderState(RenderTargets=2, RenderTargetFormat=OITMoments, Blend=OIT, DepthWrite=false, DepthFunc=Always);
    }
    pass OIT
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS_OIT()));
        SetRenderState(RenderTargets=2, RenderTargetFormat=OIT, Blend=OIT, DepthWrite=false, DepthFunc=Always);
    }
}
