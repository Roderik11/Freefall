#include "common.fx"
// @RenderState(DepthTest=false, DepthWrite=false, Blend=Additive)

// Per-instance light data — packed into a StructuredBuffer, same pattern as terrain's TerrainPatchData
struct PointLightData
{
    float3 Color;
    float Intensity;
    float3 Position; // camera-relative
    float Range;
};

// Standard InstanceBatch push constant layout (slots 2-15 set by command signature)
#define DescriptorBufIdx GET_INDEX(2)
#define Reserved0Idx GET_INDEX(3)
#define SortedIndicesIdx GET_INDEX(4)
#define IndexBufferIdx GET_INDEX(7)
#define BaseIndex GET_INDEX(8)
#define PosBufferIdx GET_INDEX(9)
#define InstanceBaseOffset GET_INDEX(13)

// Per-instance buffer: PointLightData (slot 1, same as terrain's TerrainDataIdx)
#define LightDataIdx GET_INDEX(1)

// G-Buffer textures (slots 17-18, set by Material.Apply, above command signature range)
#define NormalTexIdx GET_INDEX(17)
#define DepthTexIdx GET_INDEX(18)

// Per-instance descriptor (matches C# InstanceDescriptor: 12 bytes)
struct InstanceDescriptor
{
    uint TransformSlot;
    uint MaterialId;
    uint CustomDataIdx;
};

SamplerState Sampler : register(s0);

struct VSOutput
{
    float4 Position : SV_POSITION;
    float4 ScreenPos : TEXCOORD0;
    nointerpolation uint InstanceIdx : TEXCOORD1;
};

// ── Vertex Shader ──────────────────────────────────────────────────────────
// Same pattern as terrain: full GPU-driven pipeline with bindless buffers
VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    // Instance data buffers
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    
    // Bindless index buffer
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];
    
    // Mesh position buffer
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    
    // Instance lookup
    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;
    
    InstanceDescriptor desc = descriptors[idx];
    uint slot = desc.TransformSlot;
    
    row_major matrix World = globalTransforms[slot];
    
    float3 pos = positions[vertexID];
    
    // Transform through World × View × Projection
    float4 worldPos = mul(float4(pos, 1.0f), World);
    output.Position = mul(mul(worldPos, View), Projection);
    output.ScreenPos = output.Position;
    output.InstanceIdx = idx;
    
    return output;
}

// ── Pixel Shader ───────────────────────────────────────────────────────────
// Reconstruct world pos from depth, compute point light contribution
float4 PS(VSOutput input) : SV_Target
{
    // Get screen UV from rasterized position
    float2 screenPos = input.Position.xy;
    
    // Get render target dimensions
    Texture2D NormalTex = ResourceDescriptorHeap[NormalTexIdx];
    Texture2D DepthTex = ResourceDescriptorHeap[DepthTexIdx];
    
    float w, h;
    NormalTex.GetDimensions(w, h);
    float2 uv = screenPos / float2(w, h);
    
    // Sample G-Buffer
    float3 normal = NormalTex.Sample(Sampler, uv).xyz;
    float depth = DepthTex.Sample(Sampler, uv).r;
    
    // Skip sky pixels
    if (depth >= 1.0f)
        discard;
    
    // Reconstruct world position from depth (camera-relative, zero-translation)
    float4 worldPos = posFromDepth(uv, depth, CameraInverse);
    
    // Read per-instance light data from structured buffer
    StructuredBuffer<PointLightData> lightData = ResourceDescriptorHeap[LightDataIdx];
    PointLightData light = lightData[input.InstanceIdx];
    
    // Direction from surface to light
    float3 toLight = light.Position - worldPos.xyz;
    float dist = length(toLight);
    
    // Early out if beyond range
    if (dist > light.Range)
        discard;
    
    float3 L = toLight / dist;
    
    // N·L
    float NdotL = max(dot(normal, L), 0.0);
    
    if (NdotL <= 0.0)
        discard;
    
    // Attenuation: smooth falloff at range boundary
    float attenuation = saturate(1.0 - (dist / light.Range));
    attenuation *= attenuation; // Quadratic falloff
    
    // Final lighting
    float3 lighting = light.Color * light.Intensity * NdotL * attenuation;
    
    return float4(lighting, 1.0);
}

technique11 PointLight
{
    pass Light
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
