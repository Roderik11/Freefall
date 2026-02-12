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

// G-Buffer textures (slots 17-20, set by Material.Apply, above command signature range)
#define NormalTexIdx GET_INDEX(17)
#define DepthTexIdx GET_INDEX(18)
#define AlbedoTexIdx GET_INDEX(19)
#define DataTexIdx GET_INDEX(20)

// Per-instance descriptor (matches C# InstanceDescriptor: 12 bytes)
struct InstanceDescriptor
{
    uint TransformSlot;
    uint MaterialId;
    uint CustomDataIdx;
};

SamplerState Sampler : register(s0);

// PBR helpers
float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

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
    
    // Skip sky pixels — reverse depth: far=0
    if (depth <= 0.0f)
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
    // PBR lighting
    Texture2D AlbedoTex = ResourceDescriptorHeap[AlbedoTexIdx];
    Texture2D DataTex = ResourceDescriptorHeap[DataTexIdx];
    float3 albedo = AlbedoTex.Sample(Sampler, uv).rgb;
    float3 data = DataTex.Sample(Sampler, uv).rgb;
    
    float rough = max(data.r, 0.04);
    float metal = saturate(data.g);
    float ao    = saturate(data.b);
    
    // View and half vectors (worldPos is camera-relative, camera at origin)
    float3 V = normalize(-worldPos.xyz);
    float3 H = normalize(L + V);
    float NdotV = max(dot(normal, V), 0.001);
    float NdotH = max(dot(normal, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    
    float a  = rough * rough;
    float a2 = a * a;
    
    // GGX Normal Distribution Function
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    float D = a2 / max(3.14159 * denom * denom, 1e-4);
    
    // Smith GGX Geometry (Schlick-GGX)
    float k = (rough + 1.0);
    k = (k * k) / 8.0;
    float Gv = NdotV / max(NdotV * (1.0 - k) + k, 1e-4);
    float Gl = NdotL / max(NdotL * (1.0 - k) + k, 1e-4);
    float G = Gv * Gl;
    
    // Fresnel (Schlick)
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metal);
    float3 F = FresnelSchlick(VdotH, F0);
    
    // Specular BRDF
    float3 spec = (D * G) * F / max(4.0 * NdotL * NdotV, 1e-4);
    
    // Diffuse BRDF (energy-conserving Lambert)
    float3 kd = (1.0 - F) * (1.0 - metal);
    float3 diffuse = kd * (albedo / 3.14159);
    
    // Final lighting with attenuation
    float3 radiance = light.Color * light.Intensity * NdotL * attenuation;
    float3 lighting = (diffuse + spec) * radiance * ao;
    
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
