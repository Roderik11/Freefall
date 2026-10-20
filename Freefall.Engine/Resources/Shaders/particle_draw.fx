// GPU Particle Billboard Renderer
// Vertex-pull quads via SV_VertexID + SV_InstanceID — no VB/IB needed.
// One draw covers every emitter of a render mode: the instance picks a pool slot from the draw
// list, the slot's chunk names the emitter row, and the row carries texture and look.
//
// Colours are premultiplied: "AlphaBlend" is One / InvSrcAlpha, so RGB is added as it is and alpha
// only says how much of the background the particle hides. Alpha 0 = additive, and everything that
// fades a particle has to scale RGB along with alpha.
//
// @RenderState(RenderTargets=1, DepthWrite=false, Blend=AlphaBlend, CullMode=None)

#include "common.fx"
#include "sky_common.fx"
#include "particle_common.hlsli"

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
};

#define BILLBOARD_CAMERA   0
#define BILLBOARD_VELOCITY 1
#define BILLBOARD_UPRIGHT  2

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
};

VSOutput VS(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
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
    output.SkyVisible = e.RenderMode == PARTICLE_MODE_FORWARD ? SkyVisibility(core.Position, float3(0, 1, 0)) : 1.0;
    if (e.RenderMode == PARTICLE_MODE_FORWARD && (e.Flags & PARTICLE_FLAG_LIT) != 0)
    {
        float3 relPos = core.Position - CamPos;
        float3 toEye = normalize(-relPos);
        output.Color.rgb *= ParticleLight(SamplerLinearWrap, relPos, toEye, e.ForwardScatter, e.MultipleScattering,
                                          ParticleBodyShadows(relPos, ParticleShadowSampler)) * output.SkyVisible
                          + ParticlePointLight(relPos, toEye, e.ForwardScatter, e.MultipleScattering, 0.5 * size,
                                               ParticleShadowSampler);
    }
    else
        output.Color.rgb *= LightScale;
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
            output.FrameBlend = frac(position);
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

        float width  = size;
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

        worldPos = core.Position + (rotated.x * right + rotated.y * up) * size;

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

float4 PS(VSOutput input) : SV_Target0
{
    // Emitters of one draw use different textures
    Texture2D ParticleTex = ResourceDescriptorHeap[NonUniformResourceIndex(input.TextureIdx)];

    // Both frames are the same size on screen: one set of gradients, taken outside the branch
    float2 dx = ddx(input.TexCoord);
    float2 dy = ddy(input.TexCoord);
    float4 texColor = ParticleTex.SampleGrad(SamplerLinearWrap, input.TexCoord, dx, dy);
    if (input.FrameBlend > 0.0)
        texColor = lerp(texColor, ParticleTex.SampleGrad(SamplerLinearWrap, input.NextTexCoord, dx, dy), input.FrameBlend);

    float4 finalColor = texColor * input.Color;

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
    if (ApplyFog != 0 && FogEnabled > 0.0 && AtmosphereIdx != 0)
    {
        float4 fog = AtmosphereFog(input.Position.xy, input.Depth, input.SkyVisible);   // no lit haze in a cave
        finalColor.rgb = finalColor.rgb * fog.a + fog.rgb * finalColor.a;
    }
    return finalColor;
}

// ────────────── Technique ──────────────

technique11 Particles
{
    pass Transparent
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
