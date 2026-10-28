// Trail ribbons (TrailRenderer / TrailSystem)
// Vertex-pull quads via SV_VertexID + SV_InstanceID, no VB/IB. One draw covers every trail: the CPU
// writes the points of all trails into one list, each trail's points in a row from its far end to
// the entity, already carrying the width and colour of their age. Instance i is the piece of ribbon
// from point i to point i + 1; the last point of a trail starts none.
//
// Colours are premultiplied, as in particle_draw.fx: "AlphaBlend" is One / InvSrcAlpha.
//
// @RenderState(RenderTargets=1, DepthWrite=false, Blend=AlphaBlend, CullMode=None, RenderTargetFormat=R16G16B16A16_Float)

#include "common.fx"
#include "sky_common.fx"
#include "oit.hlsli"

SamplerState TrailWrapSampler : register(s0);      // linear wrap (the clouds' volumes, ParticleLight)
SamplerComparisonState TrailShadowSampler : register(s3);   // less-equal: the sun's cascades, point light shadows

cbuffer PushConstants : register(b3)
{
    uint  PointsIdx;     // DWORD 0   SRV: TrailPoint list
    float ViewHeight;    // DWORD 1   height of the target in pixels
};

struct TrailPoint          // 48 bytes = TrailSystem.TrailPoint in C#
{
    float3 Position;   float Width;
    float4 Color;
    uint   Flags;      float3 Edge;    // a blade's other edge (TRAIL_BLADE)
};

#define TRAIL_FIRST 1      // far end of a trail: nothing before it
#define TRAIL_LAST  2      // the end at the entity: nothing after it
#define TRAIL_BLADE 4      // the sheet between Position and Edge, not a strip that faces the camera
#define TRAIL_LIT   8      // the colour is an albedo lit by the sky and the sun; else a luminance in nits

// A ribbon narrower than this on screen is drawn this wide and fainter by as much: a thin trail far
// away stays a steady line instead of a row of pixels that come and go
static const float MinPixels = 1.5;

// x: which end of the piece, y: which edge of the ribbon
static const float2 Corners[6] =
{
    float2(0, -1),
    float2(1, -1),
    float2(0,  1),
    float2(0,  1),
    float2(1, -1),
    float2(1,  1),
};

struct VSOutput
{
    float4 Position : SV_Position;
    float4 Color    : COLOR0;
    float  Across   : TEXCOORD0;   // -1..1 from edge to edge
    float  Blade    : TEXCOORD1;   // 1 on a blade's sheet
    float  Depth    : TEXCOORD2;   // view depth (clip w), for the aerial perspective
    float  SkyVisible : TEXCOORD3; // how much sky the point sees past the terrain (scales the haze in front)
};

VSOutput VS(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output = (VSOutput)0;

    StructuredBuffer<TrailPoint> Points = ResourceDescriptorHeap[PointsIdx];

    // Between the end of one trail and the start of the next there is no ribbon
    if ((Points[instanceID].Flags & TRAIL_LAST) != 0)
    {
        output.Position = float4(2.0, 2.0, 2.0, 1.0);
        return output;
    }

    float2 corner = Corners[vertexID % 6];
    uint index = instanceID + (uint)corner.x;
    TrailPoint p = Points[index];
    // Premultiplied: only the light scales. Lit: an albedo under the sky and the sun (through their shadows) and the
    // point lights that reach it; else nits, pre-exposed as a light source
    float skyVisible = SkyVisibility(p.Position, float3(0, 1, 0));
    if ((p.Flags & TRAIL_LIT) != 0)
    {
        float3 relPos = p.Position - CamPos;
        float3 toEye = normalize(-relPos);
        p.Color.rgb *= ParticleLight(TrailWrapSampler, relPos, toEye, 0.0, ParticleBodyShadows(relPos, TrailShadowSampler)) * skyVisible
                     + ParticlePointLight(relPos, toEye, 0.0, 0.5 * p.Width, TrailShadowSampler);
    }
    else
        p.Color.rgb *= EmissiveExposure;
    output.SkyVisible = skyVisible;

    // A blade: the sheet is given, from the point (the tip's edge) to its other edge
    if ((p.Flags & TRAIL_BLADE) != 0)
    {
        output.Position = mul(float4(corner.y > 0.0 ? p.Position : p.Edge, 1.0), ViewProjection);
        output.Color = p.Color;
        output.Across = corner.y;
        output.Blade = 1.0;
        output.Depth = output.Position.w;
        return output;
    }

    // The ribbon runs the way its neighbours lie, so two pieces share their edge where they meet
    uint before = (p.Flags & TRAIL_FIRST) != 0 ? index : index - 1;
    uint after  = (p.Flags & TRAIL_LAST)  != 0 ? index : index + 1;
    float3 along = Points[after].Position - Points[before].Position;

    float3 side = cross(along, CamPos - p.Position);
    float sideLength = length(side);
    // Running straight at the camera: any direction across will do
    side = sideLength > 1e-6 ? side / sideLength : float3(View._11, View._21, View._31);

    float4 clip = mul(float4(p.Position, 1.0), ViewProjection);

    // What one pixel spans in the world at this depth (clip.w is the view-space depth)
    float pixel = max(clip.w, 0.0) * 2.0 / (Projection._22 * max(ViewHeight, 1.0));
    float halfWidth = 0.5 * p.Width;
    float least = 0.5 * MinPixels * pixel;
    float share = 1.0;
    if (halfWidth < least)
    {
        share = halfWidth / max(least, 1e-6);
        halfWidth = least;
    }

    output.Position = mul(float4(p.Position + side * (corner.y * halfWidth), 1.0), ViewProjection);
    output.Color = p.Color * share;
    output.Across = corner.y;
    output.Depth = output.Position.w;
    return output;
}

float4 RibbonColor(VSOutput input, bool fogged)
{
    // Full along the middle, nothing at the edges. A blade's sheet: full at the tip's edge, fading
    // out towards the other, faster at first
    float tipward = saturate(input.Across * 0.5 + 0.5);
    float across = input.Blade > 0.5 ? tipward * tipward : saturate(1.0 - input.Across * input.Across);
    float4 color = input.Color * across;

    if (max(color.a, max(color.r, max(color.g, color.b))) < 0.002) discard;

    // The air in front (premultiplied, as in particle_draw); the light was scaled in VS
    if (fogged && FogEnabled > 0.0 && AtmosphereIdx != 0)
    {
        float4 fog = AtmosphereFog(input.Position.xy, input.Depth, input.SkyVisible);   // no lit haze in a cave
        color.rgb = color.rgb * fog.a + fog.rgb * color.a;
    }
    return color;
}

float4 PS(VSOutput input) : SV_Target0
{
    return RibbonColor(input, true);
}

// Order-independent transparency (EngineSettings.Transparency, oit.hlsli)
OITMomentTargets PS_OITMoments(VSOutput input)
{
    return OITMoments(RibbonColor(input, false).a, input.Depth);
}

OITTargets PS_OIT(VSOutput input)
{
    return OITAccumulate(RibbonColor(input, true), input.Depth, input.Position.xy);
}

technique11 Trails
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
        SetRenderState(RenderTargets=2, RenderTargetFormat=OITMoments, Blend=OIT, DepthWrite=false);
    }
    pass OIT
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS_OIT()));
        SetRenderState(RenderTargets=2, RenderTargetFormat=OIT, Blend=OIT, DepthWrite=false);
    }
}
