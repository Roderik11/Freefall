// GPU Particle Billboard Renderer
// Vertex-pull quads via SV_VertexID + SV_InstanceID — no VB/IB needed.
// One draw covers every emitter of a render mode: the instance picks a pool slot from the draw
// list, the slot's chunk names the emitter row, and the row carries texture and look.
//
// Colours are premultiplied: "AlphaBlend" is One / InvSrcAlpha, so RGB is added as it is and alpha
// only says how much of the background the particle hides. Alpha 0 = additive, and everything that
// fades a particle has to scale RGB along with alpha.
//
// Order-independent transparency (EngineSettings.Transparency, oit.hlsli): both render modes are drawn in the OITMoments
// and OIT passes after the water, Transparent-mode particles lit here like Forward-mode ones (there is no composition
// after them to light an albedo).
//
// @RenderState(RenderTargets=1, DepthWrite=false, Blend=AlphaBlend, CullMode=None)

#include "common.fx"
#include "sky_common.fx"
#include "particle_common.hlsli"
#include "oit.hlsli"
#include "underwater.hlsli"

// ────────────── Push Constants ──────────────

cbuffer PushConstants : register(b3)
{
    uint ParticleCoreIdx;      // DWORD 0   SRV: ParticleCore pool
    uint ParticleVisualIdx;    // DWORD 1   SRV: ParticleVisual pool
    uint DrawListIdx;          // DWORD 2   SRV: slots to draw
    uint DrawListOffset;       // DWORD 3   first entry of this render mode's list
    uint EmittersIdx;          // DWORD 4   SRV: ParticleEmitter rows
    uint ChunkMapIdx;          // DWORD 5   SRV: chunk -> emitter row
    uint DepthGBufIdx;         // DWORD 6   SRV: depth buffer for soft particles
    float LightScale;          // DWORD 7   Forward mode: EmissiveExposure (a light source's nits); Transparent (an albedo): 1
    uint ApplyFog;             // DWORD 8   1 in Forward mode: aerial perspective here (Transparent: the composition does it)
    uint UnderwaterFrameIdx;   // DWORD 9   the eye under water, the OIT draws (after the underwater pass): StructuredBuffer<UnderwaterFrame>; else 0
};

#define BILLBOARD_CAMERA   0
#define BILLBOARD_VELOCITY 1
#define BILLBOARD_UPRIGHT  2

#define PIXEL_FLOOR 1.5     // pixels: the narrowest a quad is drawn (thinner ones are widened and dimmed to match)

// ────────────── Samplers ──────────────

SamplerState SamplerLinearWrap : register(s0);
SamplerState SamplerPointClamp : register(s1);
SamplerComparisonState ParticleShadowSampler : register(s3);    // less-equal: the sun's cascades, point light shadows

// ────────────── Quad Geometry ──────────────

static const float2 QuadCorners[6] =
{
    float2(-0.5, -0.5), // bottom-left
    float2( 0.5, -0.5), // bottom-right
    float2(-0.5,  0.5), // top-left
    float2(-0.5,  0.5), // top-left
    float2( 0.5, -0.5), // bottom-right
    float2( 0.5,  0.5), // top-right
};

static const float2 QuadUVs[6] =
{
    float2(0, 1),
    float2(1, 1),
    float2(0, 0),
    float2(0, 0),
    float2(1, 1),
    float2(1, 0),
};

// ────────────── VS / PS ──────────────

struct VSOutput
{
    float4 Position : SV_Position;
    float2 TexCoord : TEXCOORD0;
    float4 Color    : COLOR0;
    float  Depth    : TEXCOORD1;  // linear view-space depth for soft particles
    nointerpolation uint  TextureIdx : TEXCOORD2;
    nointerpolation float SoftRange  : TEXCOORD3;  // <= 0: soft particles off
    float2 NextTexCoord : TEXCOORD4;               // the following flipbook frame
    nointerpolation float FrameBlend : TEXCOORD5;  // 0..1 towards it; 0 = no cross-fade
    nointerpolation float SkyVisible : TEXCOORD6;  // how much sky the particle sees past the terrain (scales the haze in front)
    nointerpolation float2 Pyro : TEXCOORD7;       // a pyro bake: the fire's light at its brightest (pre-exposed) and its kelvin; x 0 = not pyro
    nointerpolation uint MotionIdx : TEXCOORD8;    // the flipbook's motion atlas, 0 = none
    nointerpolation float3 Motion : TEXCOORD9;     // its full-scale motion in atlas UV (x across, y down: a frame's size times the strength), z unused
};

// The colour of a body at t kelvin, linear sRGB, its largest channel 1 (Planckian locus, Kim et al. 2002, 1000-15000 K)
float3 PyroBlackbody(float t)
{
    t = clamp(t, 1000.0, 15000.0);
    float t2 = t * t;
    float u = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t2) / (1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t2);
    float v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t2) / (1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t2);
    float d = 2.0 * u - 8.0 * v + 4.0;
    float x = 3.0 * u / d, y = 2.0 * v / d;
    float3 XYZ = float3(x / y, 1.0, (1.0 - x - y) / y);
    float3 rgb = max(float3( 3.2404542 * XYZ.x - 1.5371385 * XYZ.y - 0.4985314 * XYZ.z,
                            -0.9692660 * XYZ.x + 1.8760108 * XYZ.y + 0.0415560 * XYZ.z,
                             0.0556434 * XYZ.x - 0.2040259 * XYZ.y + 1.0572252 * XYZ.z), 0.0);
    return rgb / max(max(rgb.r, rgb.g), max(rgb.b, 1e-4));
}

// A particle seen by an eye under water (UnderwaterFrameIdx): what the underwater pass does to what is opaque, which it
// runs before the OIT draws. A lit one gets the daylight the water lets down to its depth; the way to it fades into the
// water's own light, over the part of it under the surface (a particle in the air above is seen through the surface).
// Before, bubbles and what floats in the water glowed through murky water at any distance
float4 UnderwaterParticle(float3 position, float4 color, bool lit)
{
    StructuredBuffer<UnderwaterFrame> frames = ResourceDescriptorHeap[UnderwaterFrameIdx];
    UnderwaterFrame f = frames[0];
    float3 rel = position - CamPos;
    float depth = f.SurfaceY + dot(f.Slope, rel.xz) - position.y;      // under the surface over it (< 0: above)
    float eyeDepth = max(f.SurfaceY - CamPos.y, 0.0);
    float dist = length(rel);
    float path = depth >= 0.0 ? dist : dist * eyeDepth / max(eyeDepth - depth, 1e-3);
    float3 dir = rel / max(dist, 1e-4);
    if (lit && depth > 0.0)
        color.rgb *= WaterDaylightAt(depth, f.ShallowColor, f.Visibility);
    float3 sunRadiance = f.SunColor * f.SunIntensity * SunLightScale;
    float midDepth = 0.5 * (eyeDepth + max(depth, 0.0));
    float3 T = WaterTransmittance(path, f.ShallowColor, f.Visibility);
    float3 fog = WaterInscatter(f.WaterColor, sunRadiance, f.SunUp, SkyAmbient(float3(0.0, 1.0, 0.0)), midDepth, dir,
                                f.ShallowColor, f.Visibility);
    color.rgb = color.rgb * T + fog * (1.0 - T) * color.a;
    // Right at the eye a speck would be out of focus, a faint blur: drawn sharp, a mote a hand's width away was a blot
    return color * smoothstep(0.25, 0.7, dist);
}

#define DRAW_SORTED   0     // the Transparent and Forward passes
#define DRAW_COVERAGE 1     // OITMoments: only what the particle hides, no light
#define DRAW_OIT      2     // OIT: both modes lit as Forward-mode particles

VSOutput DrawVS(uint vertexID, uint instanceID, uint draw)
{
    VSOutput output = (VSOutput)0;

    StructuredBuffer<uint> DrawList = ResourceDescriptorHeap[DrawListIdx];
    uint slot = DrawList[DrawListOffset + instanceID];

    StructuredBuffer<uint> ChunkMap = ResourceDescriptorHeap[ChunkMapIdx];
    StructuredBuffer<ParticleEmitter> Emitters = ResourceDescriptorHeap[EmittersIdx];
    ParticleEmitter e = Emitters[ChunkMap[slot >> PARTICLE_CHUNK_SHIFT]];

    StructuredBuffer<ParticleCore> Cores = ResourceDescriptorHeap[ParticleCoreIdx];
    StructuredBuffer<ParticleVisual> Visuals = ResourceDescriptorHeap[ParticleVisualIdx];

    ParticleCore core = Cores[slot];
    ParticleVisual vis = Visuals[slot];

    // Age ratio [0..1]
    float t = saturate(core.Age / max(core.Lifetime, 0.001));

    float size = lerp(e.SizeStartEnd.x, e.SizeStartEnd.y, t) * vis.SizeScale;
    output.Color = lerp(e.ColorStart, e.ColorEnd, t);
    output.Color.rgb *= e.Intensity;
    // Premultiplied: only the light scales, not how much of the background it hides. Forward mode: a lit particle's
    // colour is an albedo under the sky, the sun (through their shadows) and the point lights that reach it, else a
    // luminance (nits) pre-exposed as a light source. Transparent mode writes an albedo the composition lights
    // (LightScale 1).
    bool forward = e.RenderMode == PARTICLE_MODE_FORWARD || draw != DRAW_SORTED;
    output.SkyVisible = forward ? SkyVisibility(core.Position, float3(0, 1, 0)) : 1.0;
    bool lit = (e.Flags & PARTICLE_FLAG_LIT) != 0 || e.RenderMode != PARTICLE_MODE_FORWARD;
    // A pyro bake's fire is light of its own, faded with the particle as its smoke is (Color.a)
    // (the coverage pass draws no light, but its smoke still hides by the bake's coverage)
    if ((e.Flags & PARTICLE_FLAG_PYRO) != 0)
        output.Pyro = float2(draw == DRAW_COVERAGE ? 0.0 : e.PyroFire * EmissiveExposure * output.Color.a * (e.FadeInRate > 0.0 ? saturate(t * e.FadeInRate) : 1.0),
                             max(e.PyroTemperature, 1000.0));
    if (draw != DRAW_COVERAGE && forward && lit)
    {
        float3 relPos = core.Position - CamPos;
        float3 toEye = normalize(-relPos);
        float3 light = ParticleLight(SamplerLinearWrap, relPos, toEye, e.ForwardScatter, e.MultipleScattering,
                                     ParticleBodyShadows(relPos, ParticleShadowSampler));
        // A water drop shows the sky above and behind it through itself: the sky's light from above in full, not the
        // average with the ground's that ParticleLight gives a speck lit from every side
        if ((e.Flags & PARTICLE_FLAG_WATER) != 0)
            light += (SkyAmbient(float3(0, 1, 0)) - SkyAmbient(float3(0, -1, 0))) * 0.5;
        output.Color.rgb *= light * output.SkyVisible
                          + ParticlePointLight(relPos, toEye, e.ForwardScatter, e.MultipleScattering, 0.5 * size,
                                               ParticleShadowSampler);
    }
    else
        output.Color.rgb *= LightScale;
    // A water drop is a lens over a small part of its streak: it shows the sky's light through just as much of the
    // pixel as it hides, its alpha (premultiplied by it), each drop its own share (size, how the light catches it). With
    // the light added in full over a 30 % alpha every streak was a white bar brighter than the sky (user, 2026-10-23)
    if ((e.Flags & PARTICLE_FLAG_WATER) != 0)
    {
        output.Color.rgb *= output.Color.a;
        output.Color *= lerp(0.3, 1.0, vis.Random);
    }
    // The eye under water: the water's light and fog on the way to it (the OIT draws come after the underwater pass)
    if (UnderwaterFrameIdx != 0 && draw != DRAW_SORTED)
        output.Color = UnderwaterParticle(core.Position, output.Color, lit && forward);
    if (e.FadeInRate > 0.0)
        output.Color *= saturate(t * e.FadeInRate);
    // Volumetric: inside its range the near fog volume carries the particle (volumetric_fog_cs.hlsl CSFogInject), the
    // sprite only past it; nothing to draw while the volume takes all of it
    float spriteShare = 1.0;
    if ((e.Flags & PARTICLE_FLAG_VOLUME) != 0 && e.RenderMode == PARTICLE_MODE_FORWARD)
    {
        spriteShare = 1.0 - FogVolumeShare(mul(float4(core.Position - CamPos, 1.0), CameraRelativeVP).w);
        output.Color *= spriteShare;
    }
    output.TextureIdx = e.TextureIdx;
    output.SoftRange = e.SoftEnabled != 0 ? max(e.SoftRange, 0.01) : 0.0;

    // Never narrower than PIXEL_FLOOR pixels: a thinner quad is drawn that wide and as much fainter, the same light over
    // more pixels. Rain streaks a few millimetres wide and distant flakes are sub-pixel; drawn as they were, they
    // flickered on and off as they crossed the pixel grid.
    float drawSize = size, drawWidth = size;
    {
        float viewDepth = mul(float4(core.Position, 1.0), View).z;
        // Rain and snow right at the eye would be out of focus, a blur too faint to see: drawn, a drop a hand's width away
        // was a white bar across the screen, a flake a dark blot on the sky. Falling drops only (streaks): water on the
        // ground (the rain's splashes) is seen at one's feet
        // Weather round the camera only (it wraps in its box and stops under cover): a splash's drops at a third-person
        // camera's 3-4 m were faded to nothing; they fade only right at the lens
        if ((e.Flags & PARTICLE_FLAG_WATER) != 0 && e.BillboardMode == BILLBOARD_VELOCITY)
            output.Color *= (e.Flags & (PARTICLE_FLAG_WRAP | PARTICLE_FLAG_COVER)) != 0 ? smoothstep(1.0, 3.5, viewDepth)
                                                                                     : smoothstep(0.2, 0.6, viewDepth);
        else if ((e.Flags & PARTICLE_FLAG_COVER) != 0)
            output.Color *= smoothstep(0.5, 2.0, viewDepth);
        float minWidth = viewDepth > 0.0 ? PIXEL_FLOOR * 2.0 * viewDepth / (Projection._22 * ViewportSize.y) : 0.0;
        if (e.BillboardMode == BILLBOARD_VELOCITY)
        {
            if (size < minWidth)
            {
                output.Color *= size / minWidth;          // only the width grows: the streak's length is its own
                drawWidth = minWidth;
            }
        }
        else
        {
            float narrow = size * min(e.Aspect, 1.0);
            if (narrow > 0.0 && narrow < minWidth)
            {
                float grow = minWidth / narrow;
                output.Color /= grow * grow;              // both sides grow
                drawSize = size * grow;
            }
        }
    }

    // Quad corner in local space
    float2 corner = QuadCorners[vertexID % 6];
    float2 uv = QuadUVs[vertexID % 6];

    // Flipbook UV adjustment
    uint flipCols = max(e.FlipbookCols, 1u);
    uint flipRows = max(e.FlipbookRows, 1u);
    uint totalFrames = e.FlipbookFrameCount > 0 ? e.FlipbookFrameCount : flipCols * flipRows;

    float2 nextUV = uv;
    if (totalFrames > 1)
    {
        float position = core.Age * e.FlipbookAnimSpeed;
        if ((e.Flags & PARTICLE_FLAG_FLIPBOOK_RANDOM) != 0)
            position += vis.Random * (float)totalFrames;

        uint frame = (uint)position % totalFrames;
        uint next = (frame + 1) % totalFrames;

        float2 uvSize = float2(1.0 / (float)flipCols, 1.0 / (float)flipRows);
        nextUV = float2((float)(next % flipCols), (float)(next / flipCols)) * uvSize + uv * uvSize;
        uv = float2((float)(frame % flipCols), (float)(frame / flipCols)) * uvSize + uv * uvSize;

        if ((e.Flags & PARTICLE_FLAG_FLIPBOOK_BLEND) != 0)
        {
            output.FrameBlend = frac(position);
            output.MotionIdx = e.MotionTextureIdx;
            output.Motion = float3(uvSize * e.MotionStrength, 0.0);
        }
    }
    output.TexCoord = uv;
    output.NextTexCoord = nextUV;

    float3 worldPos;

    [branch]
    if (e.BillboardMode == BILLBOARD_VELOCITY)
    {
        // Velocity-stretched: quad up axis follows the velocity, right axis is
        // perpendicular to both velocity and the view direction so the streak
        // always presents its face to the camera.
        float3 toCam = normalize(CamPos - core.Position);

        float speed = length(core.Velocity);
        float3 vdir = speed > 1e-4 ? core.Velocity / speed : float3(View._12, View._22, View._32);

        float3 right = cross(vdir, toCam);
        float rl = length(right);
        // Velocity pointing straight at the camera: fall back to the camera right vector
        right = rl > 1e-4 ? right / rl : float3(View._11, View._21, View._31);

        float width  = drawWidth;
        float height = size * e.Aspect + speed * e.StretchFactor;

        // No rotation in this mode — orientation is fully defined by velocity
        worldPos = core.Position + right * (corner.x * width) + vdir * (corner.y * height);
    }
    else
    {
        // Rotation is a function of age, so the simulation never has to write it back
        float rotation = vis.Rotation + vis.RotationSpeed * core.Age;
        float cosR = cos(rotation);
        float sinR = sin(rotation);
        // Pivot: where the quad sits on its particle; (0, 0.5) stands it on its foot
        float2 scaled = float2(corner.x + e.PivotX, (corner.y + e.PivotY) * e.Aspect);
        float2 rotated = float2(
            scaled.x * cosR - scaled.y * sinR,
            scaled.x * sinR + scaled.y * cosR
        );

        // Billboard: extract camera right and up from View matrix
        float3 right = float3(View._11, View._21, View._31);
        float3 up    = float3(View._12, View._22, View._32);

        [branch]
        if (e.BillboardMode == BILLBOARD_UPRIGHT)
        {
            // Stands along world up: the foot of the quad stays where it is from every angle. Rotation is
            // about that axis. Without any (RotationRange 0) the quad turns to face the camera; with it,
            // every card keeps its own angle, so seen from above they cross and the fire has a body.
            up = float3(0.0, 1.0, 0.0);
            if (e.RotationRange > 0.0)
            {
                right = float3(cosR, 0.0, sinR);
            }
            else
            {
                float3 side = cross(CamPos - core.Position, up);
                float sl = length(side);
                right = sl > 1e-4 ? side / sl : right;   // looking straight down the axis: any direction will do
            }
            rotated = scaled;
        }

        worldPos = core.Position + (rotated.x * right + rotated.y * up) * drawSize;

        // Lean with the wind: the bottom edge stays, the top moves downwind (a flame in its basket).
        // Each sprite wavers at its own phase, or all of them would stand at one fixed slant. A flame
        // that follows its emitter also leans back from the way the emitter goes (LeanWind), its
        // older sprites further than the fresh ones (0.4 to 1.6 times over a life): the flame
        // stretches back from the head instead of tilting as one card.
        if (e.WindLean > 0.0)
        {
            float waver = 1.0 + 0.4 * sin(core.Age * 5.0 + vis.Random * 6.283185);
            float aged = lerp(1.0, 0.4 + 1.2 * saturate(core.Age / max(core.Lifetime, 0.001)), e.Follow);
            worldPos += e.LeanWind * (e.WindLean * waver * aged * (corner.y + 0.5));
        }
    }

    // Transform to clip space (off the screen when the near fog volume draws it)
    output.Position = spriteShare > 0.0 ? mul(float4(worldPos, 1.0), ViewProjection) : float4(2.0, 2.0, 2.0, 1.0);

    // Linear depth for soft particles (view-space Z)
    float3 viewPos = mul(float4(worldPos, 1.0), View).xyz;
    output.Depth = viewPos.z;

    return output;
}

VSOutput VS(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) { return DrawVS(vertexID, instanceID, DRAW_SORTED); }
VSOutput VS_Coverage(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) { return DrawVS(vertexID, instanceID, DRAW_COVERAGE); }
VSOutput VS_OIT(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) { return DrawVS(vertexID, instanceID, DRAW_OIT); }

float4 ParticleColor(VSOutput input, bool fogged)
{
    // Emitters of one draw use different textures
    Texture2D ParticleTex = ResourceDescriptorHeap[NonUniformResourceIndex(input.TextureIdx)];

    // Both frames are the same size on screen: one set of gradients, taken outside the branch
    float2 dx = ddx(input.TexCoord);
    float2 dy = ddy(input.TexCoord);
    float2 here = input.TexCoord, next = input.NextTexCoord;
    if (input.MotionIdx != 0u && input.FrameBlend > 0.0)
    {
        // Each frame pushed towards the other along the way its pixels go, by the share of the step still to come or
        // already gone: the motion flows instead of one frame fading into the next
        Texture2D MotionTex = ResourceDescriptorHeap[NonUniformResourceIndex(input.MotionIdx)];
        float2 way = (MotionTex.SampleGrad(SamplerLinearWrap, here, dx, dy).rg * 2.0 - 1.0) * input.Motion.xy;
        here -= way * input.FrameBlend;
        next += way * (1.0 - input.FrameBlend);
    }
    float4 texColor = ParticleTex.SampleGrad(SamplerLinearWrap, here, dx, dy);
    if (input.FrameBlend > 0.0)
        texColor = lerp(texColor, ParticleTex.SampleGrad(SamplerLinearWrap, next, dx, dy), input.FrameBlend);

    float4 finalColor = texColor * input.Color;

    // A pyro bake: smoke lit as the particle is (Color, its key-light shadows darkening it), the fire's light coloured as a
    // body that hot (its faint parts cooler and far dimmer), and the fire's glow in its own smoke
    if (input.Pyro.x > 0.0 || input.Pyro.y > 0.0)
    {
        float fire = texColor.r, cover = texColor.g, key = texColor.b, glow = texColor.a;
        float3 smoke = input.Color.rgb * cover * lerp(0.3, 1.0, key);
        float heat = input.Pyro.y * lerp(0.42, 1.0, fire);
        float3 flame = PyroBlackbody(heat) * (fire * fire * sqrt(fire)) * input.Pyro.x;
        float3 lit = PyroBlackbody(input.Pyro.y * 0.62) * (glow * cover * 0.12) * input.Pyro.x;
        finalColor = float4(smoke + flame + lit, input.Color.a * cover);
    }

    // Soft particles: fade near opaque surfaces
    // DepthGBuffer = R32_Float linear view-space depth, 0 = sky/empty
    if (input.SoftRange > 0.0 && DepthGBufIdx > 0)
    {
        Texture2D<float> DepthBuf = ResourceDescriptorHeap[DepthGBufIdx];

        float sceneDepth = DepthBuf.Load(int3(int2(input.Position.xy), 0));

        // sceneDepth > 0 means geometry exists (0 = sky/cleared)
        // Both sceneDepth and input.Depth are linear view-space Z (positive into screen)
        if (sceneDepth > 0.001)
        {
            float depthDiff = sceneDepth - input.Depth;
            finalColor *= saturate(depthDiff / input.SoftRange);
        }
    }

    // Discard fragments that neither add nor hide anything (an additive particle has alpha 0)
    if (max(finalColor.a, max(finalColor.r, max(finalColor.g, finalColor.b))) < 0.004) discard;

    // The air in front dims the particle's own light and adds its haze over the part of the background it hides
    if (fogged && FogEnabled > 0.0 && AtmosphereIdx != 0)
    {
        float4 fog = AtmosphereFog(input.Position.xy, input.Depth, input.SkyVisible);   // no lit haze in a cave
        finalColor.rgb = finalColor.rgb * fog.a + fog.rgb * finalColor.a;
    }
    return finalColor;
}

float4 PS(VSOutput input) : SV_Target0
{
    return ParticleColor(input, ApplyFog != 0);
}

OITMomentTargets PS_OITMoments(VSOutput input)
{
    return OITMoments(ParticleColor(input, false).a, input.Depth);
}

OITTargets PS_OIT(VSOutput input)
{
    return OITAccumulate(ParticleColor(input, true), input.Depth, input.Position.xy);
}

// ────────────── Technique ──────────────

technique11 Particles
{
    pass Transparent
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
    pass OITMoments
    {
        SetVertexShader(CompileShader(vs_6_6, VS_Coverage()));
        SetPixelShader(CompileShader(ps_6_6, PS_OITMoments()));
        SetRenderState(RenderTargets=2, RenderTargetFormat=OITMoments, Blend=OIT, DepthWrite=false);
    }
    pass OIT
    {
        SetVertexShader(CompileShader(vs_6_6, VS_OIT()));
        SetPixelShader(CompileShader(ps_6_6, PS_OIT()));
        SetRenderState(RenderTargets=2, RenderTargetFormat=OIT, Blend=OIT, DepthWrite=false);
    }
}
