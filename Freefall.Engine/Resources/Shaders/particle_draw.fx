// GPU Particle Billboard Renderer
// Vertex-pull quads via SV_VertexID + SV_InstanceID — no VB/IB needed.
// Reads particle data from bindless structured buffers.
//
// @RenderState(RenderTargets=1, DepthWrite=false, Blend=AlphaBlend, CullMode=None)

#include "common.fx"

// ────────────── Data Structures ──────────────

struct ParticleCore
{
    float3 Position;
    float  Age;
    float3 Velocity;
    float  Lifetime;
};

struct ParticleVisual
{
    float2 SizeStartEnd;
    float4 ColorStart;
    float  Rotation;
    float  RotationSpeed;
    uint   FlipbookFrame;
    uint   FlipbookCount;
    float  AnimSpeed;
    float  _pad0;
};

// ────────────── Push Constants ──────────────

cbuffer PushConstants : register(b3)
{
    uint ParticleCoreIdx;      // DWORD 0   SRV: ParticleCore buffer
    uint ParticleVisualIdx;    // DWORD 1   SRV: ParticleVisual buffer
    uint AliveListIdx;         // DWORD 2   SRV: alive indices
    uint TextureIdx;           // DWORD 3   SRV: particle texture
    uint DepthGBufIdx;         // DWORD 4   SRV: depth buffer for soft particles
    uint SoftEnabledIdx;       // DWORD 5   0 or 1
    float SoftRangeVal;        // DWORD 6   depth fade distance
    // NOTE: float4 would add padding here due to 16-byte cbuffer alignment.
    // Use 4 separate floats so DWORD layout matches C# root constant writes.
    float ColorEndR;           // DWORD 7
    float ColorEndG;           // DWORD 8
    float ColorEndB;           // DWORD 9
    float ColorEndA;           // DWORD 10
    uint FlipbookColsVal;      // DWORD 11  columns in flipbook atlas
    uint FlipbookRowsVal;      // DWORD 12  rows in flipbook atlas
};

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
};

VSOutput VS(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output = (VSOutput)0;

    // Look up which particle slot this instance maps to
    StructuredBuffer<uint> AliveList = ResourceDescriptorHeap[AliveListIdx];
    uint slot = AliveList[instanceID];

    StructuredBuffer<ParticleCore> Cores = ResourceDescriptorHeap[ParticleCoreIdx];
    StructuredBuffer<ParticleVisual> Visuals = ResourceDescriptorHeap[ParticleVisualIdx];

    ParticleCore core = Cores[slot];
    ParticleVisual vis = Visuals[slot];

    // Age ratio [0..1]
    float t = saturate(core.Age / max(core.Lifetime, 0.001));

    // Size interpolation
    float size = lerp(vis.SizeStartEnd.x, vis.SizeStartEnd.y, t);

    // Color interpolation
    float4 colorEnd = float4(ColorEndR, ColorEndG, ColorEndB, ColorEndA);
    output.Color = lerp(vis.ColorStart, colorEnd, t);

    // Quad corner in local space
    float2 corner = QuadCorners[vertexID % 6];
    float2 uv = QuadUVs[vertexID % 6];

    // Flipbook UV adjustment
    uint flipCols = max(FlipbookColsVal, 1u);
    uint flipRows = max(FlipbookRowsVal, 1u);
    uint totalFrames = vis.FlipbookCount > 0 ? vis.FlipbookCount : flipCols * flipRows;

    if (totalFrames > 1)
    {
        uint frame = vis.FlipbookFrame + (uint)(core.Age * vis.AnimSpeed);
        frame = frame % totalFrames;

        uint col = frame % flipCols;
        uint row = frame / flipCols;

        float2 uvSize = float2(1.0 / (float)flipCols, 1.0 / (float)flipRows);
        uv = float2((float)col, (float)row) * uvSize + uv * uvSize;
    }
    output.TexCoord = uv;

    // Apply rotation
    float cosR = cos(vis.Rotation);
    float sinR = sin(vis.Rotation);
    float2 rotated = float2(
        corner.x * cosR - corner.y * sinR,
        corner.x * sinR + corner.y * cosR
    );

    // Billboard: extract camera right and up from View matrix
    float3 right = float3(View._11, View._21, View._31);
    float3 up    = float3(View._12, View._22, View._32);

    float3 worldPos = core.Position + (rotated.x * right + rotated.y * up) * size;

    // Transform to clip space
    output.Position = mul(float4(worldPos, 1.0), ViewProjection);

    // Linear depth for soft particles (view-space Z)
    float3 viewPos = mul(float4(worldPos, 1.0), View).xyz;
    output.Depth = viewPos.z;

    return output;
}

float4 PS(VSOutput input) : SV_Target0
{
    // Sample particle texture
    Texture2D ParticleTex = ResourceDescriptorHeap[TextureIdx];
    float4 texColor = ParticleTex.Sample(SamplerLinearWrap, input.TexCoord);

    float4 finalColor = texColor * input.Color;

    // Soft particles: fade near opaque surfaces
    // DepthGBuffer = R32_Float linear view-space depth, 0 = sky/empty
    if (SoftEnabledIdx > 0 && DepthGBufIdx > 0)
    {
        Texture2D<float> DepthBuf = ResourceDescriptorHeap[DepthGBufIdx];

        float sceneDepth = DepthBuf.Load(int3(int2(input.Position.xy), 0));

        // sceneDepth > 0 means geometry exists (0 = sky/cleared)
        // Both sceneDepth and input.Depth are linear view-space Z (positive into screen)
        if (sceneDepth > 0.001)
        {
            float depthDiff = sceneDepth - input.Depth;
            float softFade = saturate(depthDiff / max(SoftRangeVal, 0.01));
            finalColor.a *= softFade;
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
