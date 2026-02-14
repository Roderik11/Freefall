#include "common.fx"
// @RenderState(RenderTargets=4)

// Unified Push Constant Layout (Slots 2-11)
// Used by all batched geometry shaders (gbuffer, gbuffer_skinned, mesh_skybox)
// Slots 0-1: Reserved for light/composition passes (texture indices)
#define DescriptorBufIdx GET_INDEX(2)   // StructuredBuffer<InstanceDescriptor> - per-instance descriptor
#define Reserved0Idx GET_INDEX(3)        // Reserved (was MaterialIDsIdx)
#define SortedIndicesIdx GET_INDEX(4)   // StructuredBuffer<uint> - sorted draw order indices
#define BoneWeightsIdx GET_INDEX(5)     // Unused for static meshes (0)
#define BonesIdx GET_INDEX(6)           // Unused for static meshes (0)
#define IndexBufferIdx GET_INDEX(7)     // StructuredBuffer<uint> - mesh index buffer (BINDLESS!)
#define BaseIndex GET_INDEX(8)          // Base index offset into index buffer for mesh parts
#define PosBufferIdx GET_INDEX(9)       // StructuredBuffer<float3> - vertex positions
#define NormBufferIdx GET_INDEX(10)     // StructuredBuffer<float3> - vertex normals
#define UVBufferIdx GET_INDEX(11)       // StructuredBuffer<float2> - vertex UVs
#define NumBones GET_INDEX(12)           // Number of bones (for skinned shaders)
#define InstanceBaseOffset GET_INDEX(13) // Base offset for instance ID (per-command)
#define DebugMode GET_INDEX(16)          // Debug visualization mode (set per-frame, not per-draw)

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 WorldPos : TEXCOORD1;
    nointerpolation uint MaterialID : TEXCOORD2;
    float Depth : TEXCOORD3;
};

// Per-instance descriptor (matches C# InstanceDescriptor: 12 bytes)
struct InstanceDescriptor
{
    uint TransformSlot;
    uint MaterialId;
    uint CustomDataIdx;
};

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    // Instance data buffers - descriptor contains TransformSlot + MaterialId
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    
    // Bindless index buffer - primitiveVertexID is 0 to N-1, add BaseIndex to offset into correct mesh part
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];
    
    // Mesh data buffers - use resolved vertexID
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[UVBufferIdx];
    
    // Get instance data position using InstanceBaseOffset + local instance ID
    uint dataPos = InstanceBaseOffset + instanceID;
    
    // sortedIndices contains compacted original instanceIdx
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;
    
    // Double-indirect: use original instance index to look up per-instance data from descriptor
    InstanceDescriptor desc = descriptors[idx];
    uint slot = desc.TransformSlot;
    uint materialID = desc.MaterialId;
    
    row_major matrix World = globalTransforms[slot];
    
    float3 pos = positions[vertexID];
    float3 norm = normals[vertexID];
    float2 uv = uvs[vertexID];
    
    // Apply world transform (no skinning for static meshes)
    float4 worldPos = mul(float4(pos, 1.0f), World);
    output.WorldPos = worldPos;
    output.Position = mul(mul(worldPos, View), Projection);
    
    output.Normal = mul(norm, (float3x3)World);
    output.TexCoord = uv;
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = materialID;
    output.Depth = output.Position.w; // View-space Z (linear)
    return output;
}

struct PSOutput
{
    float4 Albedo : SV_Target0;
    float4 Normal : SV_Target1;
    float4 Data : SV_Target2;
    float  Depth : SV_Target3;
};

// Shadow pass - minimal output structure
struct ShadowVSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    nointerpolation uint MaterialID : TEXCOORD1;
};

SamplerState Sampler : register(s0);

// Shadow vertex shader - minimal output for depth-only rendering
ShadowVSOutput VS_Shadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    ShadowVSOutput output;

    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];
    
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[UVBufferIdx];
    
    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;
    
    // Double-indirect: use original instance index to look up per-instance data
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    InstanceDescriptor desc = descriptors[idx];
    uint slot = desc.TransformSlot;
    row_major matrix World = globalTransforms[slot];
    uint materialID = desc.MaterialId;
    
    float3 pos = positions[vertexID];
    float2 uv = uvs[vertexID];
    
    float4 worldPos = mul(float4(pos, 1.0f), World);
    output.Position = mul(mul(worldPos, View), Projection);
    output.TexCoord = uv;
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = materialID;
    return output;
}

// Shadow pixel shader - alpha test only, depth written by hardware
void PS_Shadow(ShadowVSOutput input)
{
    MaterialData mat = GET_MATERIAL(input.MaterialID);
    Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
    float alpha = albedoTex.Sample(Sampler, input.TexCoord).a;
    clip(alpha - 0.25f);
}

PSOutput PS(VSOutput input)
{
    PSOutput output;
    
    uint materialID = input.MaterialID;
    
    // Material lookup via MaterialID indirection
    MaterialData mat = GET_MATERIAL(materialID);
    Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
    
    float4 color = albedoTex.Sample(Sampler, input.TexCoord);
    clip(color.a - 0.25f);
    
    // PBR material properties — defaults for meshes without PBR textures
    float roughness = 0.65;
    float metal = 0.0;
    float ao = 1.0;
    
    // Sample PBR textures if bound (index 0 = not bound)
    // RoughnessIdx holds a specular map — invert to get roughness
    if (mat.RoughnessIdx != 0) { Texture2D rTex = ResourceDescriptorHeap[mat.RoughnessIdx]; roughness = 1.0 - rTex.Sample(Sampler, input.TexCoord).r; }
    if (mat.MetallicIdx  != 0) { Texture2D mTex = ResourceDescriptorHeap[mat.MetallicIdx];  metal     = mTex.Sample(Sampler, input.TexCoord).r; }
    if (mat.AOIdx        != 0) { Texture2D aTex = ResourceDescriptorHeap[mat.AOIdx];        ao        = aTex.Sample(Sampler, input.TexCoord).r; }
    
    // Normal mapping via cotangent frame (no tangent buffer needed)
    float3 N = normalize(input.Normal);
    if (mat.NormalIdx != 0)
    {
        Texture2D normalTex = ResourceDescriptorHeap[mat.NormalIdx];
        float3 texNormal = normalTex.Sample(Sampler, input.TexCoord).rgb * 2.0 - 1.0;
        //texNormal.xy *= 1.5; // Slightly boosted for visible surface detail
        texNormal = normalize(texNormal);
        
        // Cotangent frame from screen-space derivatives
        float3 dp1 = ddx(input.WorldPos.xyz);
        float3 dp2 = ddy(input.WorldPos.xyz);
        float2 duv1 = ddx(input.TexCoord);
        float2 duv2 = ddy(input.TexCoord);
        
        float3 dp2perp = cross(dp2, N);
        float3 dp1perp = cross(N, dp1);
        float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
        float3 B = dp2perp * duv1.y + dp1perp * duv2.y;
        float invmax = rsqrt(max(dot(T, T), dot(B, B)));
        float3x3 TBN = float3x3(T * invmax, B * invmax, N);
        
        N = normalize(mul(texNormal, TBN));
    }
    
    output.Albedo = color;
    output.Normal = float4(N, 1.0f);
    output.Data = float4(saturate(roughness), saturate(metal), saturate(ao), 1.0);
    output.Depth = input.Depth;
    return output;
}

technique11 GBuffer
{
    pass Opaque
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
    
    pass Shadow
    {
        SetVertexShader(CompileShader(vs_6_6, VS_Shadow()));
        SetPixelShader(CompileShader(ps_6_6, PS_Shadow()));
    }
}

