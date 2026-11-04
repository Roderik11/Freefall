// Tube trails (TubeTrail / TubeTrailSystem): the vfx layers look (vfx_layers.hlsli) drawn on a tube along the way an
// entity went, as a TrailRenderer's ribbon is drawn along it: an arrow's or a bolt's fire, a ball of flame's, an
// afterburner's. One draw covers every trail. The CPU writes the rings of all trails into one list, each trail's from its
// head (the entity, its nose) to its far end, every ring with its place, a direction across it, its age as a share of the
// trail's life and the trail's look; instance i is the piece of tube from ring i to ring i + 1 (the last ring of a trail
// starts none), TUBE_SIDES quads round. Its heat, erosion, smoke and flare go by the age, so a trail is shaded as the vfx
// layers tube is along its length; its fade at the end by where along it a ring is, its nose by the age behind the head.
// Its textures' V is in metres of the way flown (TubeTrail Texture Span a V), counted from where the trail began, so its
// flames stay where they were made and run back along the tube at the entity's own speed. A trail whose entity is gone
// burns out where it is: its head ages too.
//
// Premultiplied, as vfx_layers.fx: both faces, no depth written.

cbuffer PushConstants : register(b3)
{
    uint RingsIdx;          // 0: StructuredBuffer<TubeRing>
    uint MaterialsIdx;      // 1: StructuredBuffer<MaterialData> (every material's settings buffer index)
    uint SceneDepthIdx;     // 2: the scene's linear view depth (R32, 0 = sky); 0 = none
};

#include "common.fx"
#include "sky_common.fx"
#include "oit.hlsli"
#include "vfx_layers.hlsli"

// @RenderState(RenderTargets=1, DepthWrite=false, Blend=Premultiplied, CullMode=None, RenderTargetFormat=R16G16B16A16_Float)

struct TubeRing             // 96 bytes = TubeTrailSystem.TubeRing in C#
{
    float3 Position;    float Age;          // its middle; its age as a share of the trail's life (0 new, 1 gone)
    float3 Normal;      float Along;        // a direction across it (where its U starts); its place along the trail, 0 at the head, 1 at the far end
    float3 Color;       float Intensity;    // the trail's look (TubeTrail Color, Intensity)
    uint MaterialId;    float Radius;       float TexV;     uint Flags;    // Radius: metres before Flare; TexV: the textures' V there, by the way flown
    float Erode;        float Heat;         float2 Seed;
    float Nose;         float Pace;         float2 _pad;    // Nose: its age behind the head; Pace: metres a whole life of the trail spans at its pace
};

#define TUBE_FIRST 1        // the head: nothing before it
#define TUBE_LAST  2        // the far end: nothing after it
#define TUBE_SIDES 16       // quads round the tube (= TubeTrailSystem.Sides)

// x: which ring of the piece, y: which edge of the quad round it
static const uint2 Corners[6] =
{
    uint2(0, 0),
    uint2(1, 0),
    uint2(0, 1),
    uint2(0, 1),
    uint2(1, 0),
    uint2(1, 1),
};

VSOutput VS(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output = (VSOutput)0;
    StructuredBuffer<TubeRing> rings = ResourceDescriptorHeap[RingsIdx];

    // Between the far end of one trail and the head of the next there is no tube
    if ((rings[instanceID].Flags & TUBE_LAST) != 0)
    {
        output.Position = float4(2.0, 2.0, 2.0, 1.0);
        return output;
    }

    uint2 corner = Corners[vertexID % 6];
    uint index = instanceID + corner.x;
    float u = (float)(vertexID / 6 + corner.y) / TUBE_SIDES;
    TubeRing r = rings[index];

    S = LoadSettings(r.MaterialId);
    I.Color = r.Color;
    I.Intensity = r.Intensity;
    I.Erode = r.Erode;
    I.Heat = r.Heat;
    WriteLook(output, I);
    output.MaterialId = r.MaterialId;
    output.Seed = r.Seed;

    // Its axis the way its neighbours lie, from the head towards the far end (the vfx tube's local Y), so two pieces share
    // the ring where they meet; across it the ring's own direction, squared to the axis, and the third
    uint before = (r.Flags & TUBE_FIRST) != 0 ? index : index - 1;
    uint after = (r.Flags & TUBE_LAST) != 0 ? index : index + 1;
    float3 axis = rings[after].Position - rings[before].Position;
    axis = dot(axis, axis) > 1e-12 ? normalize(axis) : float3(0.0, 0.0, 1.0);
    float3 X = r.Normal - axis * dot(r.Normal, axis);
    X = dot(X, X) > 1e-8 ? normalize(X) : normalize(abs(axis.y) < 0.99 ? cross(axis, float3(0.0, 1.0, 0.0)) : float3(1.0, 0.0, 0.0));
    float3 Z = cross(X, axis);

    // As the vfx tube: its radius from Flare x to y by its age, closed at the head into a rounded nose over Cap of its
    // life behind it (the cap's length is Cap of the way a whole life spans: the nose keeps its size while the trail grows
    // out of the hand), its normals tilting forward with the slope
    float age = saturate(r.Age);
    float grow = lerp(S.Flare.x, S.Flare.y, pow(age, max(S.FlareCurve, 0.01)));
    float theta = u * 6.28318530718;
    float2 across = float2(cos(theta), sin(theta));
    float3 out_ = X * across.x + Z * across.y;
    float slope = 0.0;
    if (S.Cap > 0.0)
    {
        float c = saturate(r.Nose / S.Cap);
        float dome = sqrt(c * (2.0 - c));
        slope = (1.0 - c) / max(dome, 1e-3) * (r.Radius * grow) / max(S.Cap * r.Pace, 1e-4);
        grow *= dome;
        across *= dome;
    }
    float radius = r.Radius * grow;
    float3 worldPos = r.Position + out_ * radius;
    float3 N = normalize(out_ - axis * slope);

    // Its outline perturbed by its own fire, as much as it is hot (vfx_layers.fx's VS)
    if (S.Wave != 0.0)
    {
        float hot = lerp(S.HeatAlong.x, S.HeatAlong.y, pow(age, max(S.HeatCurve, 0.01))) / max(S.HeatAlong.x, 1e-3);
        worldPos += N * (S.Wave * radius * hot * (VertexDetail(float2(u, r.Along), r.Age, r.TexV - AlongV(r.Age), across, r.Seed) * 2.0 - 1.0));
    }

    output.Position = mul(mul(float4(worldPos, 1.0), View), Projection);
    output.Normal = N;
    output.TexCoord = float2(u, r.Along);
    output.ToEye = CamPos - worldPos;
    output.Depth = output.Position.w;
    output.Axis = axis;
    output.Across = across;
    output.Ball = 0.0;
    output.BallRadius = 0.0;
    // The textures' V by the way flown: what the pixel shader adds to its age's V (it takes the age's per pixel)
    output.Life = float3(r.Age, r.Nose, r.TexV - AlongV(r.Age));
    return output;
}

technique11 VfxTrail
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
