// GPU Particle Billboard Renderer
// Vertex-pull quads via SV_VertexID + SV_InstanceID — no VB/IB needed.
// One draw covers every emitter of a render mode: the instance picks a pool slot from the draw
// list, the slot's chunk names the emitter row, and the row carries texture and look.
//
// @RenderState(RenderTargets=1, DepthWrite=false, Blend=AlphaBlend, CullMode=None)

#include "common.fx"
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
};

#define BILLBOARD_CAMERA   0
#define BILLBOARD_VELOCITY 1

// ────────────── Samplers ──────────────

SamplerState SamplerLinearWrap : register(s0);
SamplerState SamplerPointClamp : register(s1);

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
    output.TextureIdx = e.TextureIdx;
    output.SoftRange = e.SoftEnabled != 0 ? max(e.SoftRange, 0.01) : 0.0;

    // Quad corner in local space
    float2 corner = QuadCorners[vertexID % 6];
    float2 uv = QuadUVs[vertexID % 6];

    // Flipbook UV adjustment
    uint flipCols = max(e.FlipbookCols, 1u);
    uint flipRows = max(e.FlipbookRows, 1u);
    uint totalFrames = e.FlipbookFrameCount > 0 ? e.FlipbookFrameCount : flipCols * flipRows;

    if (totalFrames > 1)
    {
        uint frame = (uint)(core.Age * e.FlipbookAnimSpeed) % totalFrames;

        uint col = frame % flipCols;
        uint row = frame / flipCols;

        float2 uvSize = float2(1.0 / (float)flipCols, 1.0 / (float)flipRows);
        uv = float2((float)col, (float)row) * uvSize + uv * uvSize;
    }
    output.TexCoord = uv;

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
        float2 scaled = float2(corner.x, corner.y * e.Aspect);
        float2 rotated = float2(
            scaled.x * cosR - scaled.y * sinR,
            scaled.x * sinR + scaled.y * cosR
        );

        // Billboard: extract camera right and up from View matrix
        float3 right = float3(View._11, View._21, View._31);
        float3 up    = float3(View._12, View._22, View._32);

        worldPos = core.Position + (rotated.x * right + rotated.y * up) * size;
    }

    // Transform to clip space
    output.Position = mul(float4(worldPos, 1.0), ViewProjection);

    // Linear depth for soft particles (view-space Z)
    float3 viewPos = mul(float4(worldPos, 1.0), View).xyz;
    output.Depth = viewPos.z;

    return output;
}

float4 PS(VSOutput input) : SV_Target0
{
    // Emitters of one draw use different textures
    Texture2D ParticleTex = ResourceDescriptorHeap[NonUniformResourceIndex(input.TextureIdx)];
    float4 texColor = ParticleTex.Sample(SamplerLinearWrap, input.TexCoord);

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
            finalColor.a *= saturate(depthDiff / input.SoftRange);
        }
    }

    // Discard fully transparent fragments (optimization + avoids depth artifacts)
    if (finalColor.a < 0.004) discard;

    // Output with standard alpha — BlendState is SrcAlpha/InvSrcAlpha
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
