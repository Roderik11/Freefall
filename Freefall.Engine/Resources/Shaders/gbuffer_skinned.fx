cbuffer PushConstants : register(b3)
{
    uint _reserved0;
    uint _reserved1;
    uint DescriptorBufIdx;      // 2
    uint _reserved3;            // 3
    uint SortedIndicesIdx;      // 4
    uint BoneWeightsIdx;        // 5: StructuredBuffer<BoneWeight> - per-mesh bone weights
    uint BonesIdx;              // 6: StructuredBuffer<matrix> - batched bone matrices
    uint IndexBufferIdx;        // 7
    uint BaseIndex;             // 8
    uint PosBufferIdx;          // 9
    uint NormBufferIdx;         // 10
    uint UVBufferIdx;           // 11
    uint NumBones;              // 12: Number of bones per skeleton
    uint InstanceBaseOffset;    // 13
    uint MaterialsIdx;          // 14
    uint GlobalTransformBufferIdx; // 15
    uint DebugMode;             // 16
    uint _reserved17;
    uint _reserved18;
    uint _reserved19;
    uint ExpansionBufferIdx;    // 20
    uint CascadeBufferSRVIdx;   // 21
};

#include "common.fx"
// @RenderState(RenderTargets=5)

inline MaterialData GetMaterial(uint id)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    return materials[id];
}
#define GET_MATERIAL(id) GetMaterial(id)

// Per-vertex bone weights (matches BoneWeight struct in Mesh.cs)
struct BoneWeight
{
    float4 BoneIDs;
    float4 Weights;
};

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 WorldPos : TEXCOORD1;
    nointerpolation uint MaterialID : TEXCOORD2;
    float Depth : TEXCOORD3;
    nointerpolation uint TransformSlot : TEXCOORD4;
    nointerpolation uint MeshPartIdx : TEXCOORD5;
};



VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    // Instance data buffers - descriptor contains TransformSlot + MaterialId
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    StructuredBuffer<BoneWeight> boneWeights = ResourceDescriptorHeap[BoneWeightsIdx];
    StructuredBuffer<row_major matrix> bones = ResourceDescriptorHeap[BonesIdx];
    
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
    BoneWeight bw = boneWeights[vertexID];
    
    // Bone matrices for this instance — indexed by arrival-order instance index
    // (bone data is uploaded densely per-instance, not by TransformSlot)
    uint boneOffset = idx * NumBones;
    
    matrix bone0 = bones[boneOffset + (uint)bw.BoneIDs.x];
    matrix bone1 = bones[boneOffset + (uint)bw.BoneIDs.y];
    matrix bone2 = bones[boneOffset + (uint)bw.BoneIDs.z];
    matrix bone3 = bones[boneOffset + (uint)bw.BoneIDs.w];
    
    // Skinning transformation
    float4 skinned = float4(0, 0, 0, 0);
    float3 skinnedNormal = float3(0, 0, 0);
    
    skinned += mul(float4(pos, 1), bone0) * bw.Weights.x;
    skinned += mul(float4(pos, 1), bone1) * bw.Weights.y;
    skinned += mul(float4(pos, 1), bone2) * bw.Weights.z;
    skinned += mul(float4(pos, 1), bone3) * bw.Weights.w;
    
    skinnedNormal += mul(norm, (float3x3)bone0) * bw.Weights.x;
    skinnedNormal += mul(norm, (float3x3)bone1) * bw.Weights.y;
    skinnedNormal += mul(norm, (float3x3)bone2) * bw.Weights.z;
    skinnedNormal += mul(norm, (float3x3)bone3) * bw.Weights.w;
    
    pos = skinned.xyz;
    norm = normalize(skinnedNormal);
    
    // Apply world transform
    float4 worldPos = mul(float4(pos, 1.0f), World);
    
    output.WorldPos = worldPos;
    output.Position = mul(mul(worldPos, View), Projection);
    
    output.Normal = mul(norm, (float3x3)World);
    output.TexCoord = uv;
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = materialID;
    output.Depth = output.Position.w; // View-space Z (linear)
    output.TransformSlot = slot;
    output.MeshPartIdx = desc.MeshPartIdx;
    return output;
}

struct PSOutput
{
    float4 Albedo : SV_Target0;
    float4 Normal : SV_Target1;
    float4 Data : SV_Target2;
    float  Depth : SV_Target3;
    uint   EntityId : SV_Target4;
};


// Shadow pass
struct ShadowVSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    nointerpolation uint MaterialID : TEXCOORD1;
    nointerpolation uint RTIndex : SV_RenderTargetArrayIndex;
};

SamplerState Sampler : register(s0);

// Shadow vertex shader - skinned, single-pass multi-cascade via expansion buffer
ShadowVSOutput VS_Shadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    ShadowVSOutput output;
    output.Position = float4(0, 0, 0, 1);
    output.TexCoord = float2(0, 0);
    output.MaterialID = 0;
    output.RTIndex = 0;

    // Read expansion entry: bits 30-31 = cascadeIdx, bits 0-29 = instance index
    StructuredBuffer<uint> expansion = ResourceDescriptorHeap[ExpansionBufferIdx];
    uint entry = expansion[InstanceBaseOffset + instanceID];
    uint cascadeIdx = entry >> 30;
    uint idx = entry & 0x3FFFFFFFu;
    output.RTIndex = cascadeIdx;

    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<BoneWeight> boneWeights = ResourceDescriptorHeap[BoneWeightsIdx];
    StructuredBuffer<row_major matrix> bones = ResourceDescriptorHeap[BonesIdx];

    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[UVBufferIdx];

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    float3 pos = positions[vertexID];
    BoneWeight bw = boneWeights[vertexID];

    uint boneOffset = idx * NumBones;
    matrix bone0 = bones[boneOffset + (uint)bw.BoneIDs.x];
    matrix bone1 = bones[boneOffset + (uint)bw.BoneIDs.y];
    matrix bone2 = bones[boneOffset + (uint)bw.BoneIDs.z];
    matrix bone3 = bones[boneOffset + (uint)bw.BoneIDs.w];

    float4 skinned = float4(0, 0, 0, 0);
    skinned += mul(float4(pos, 1), bone0) * bw.Weights.x;
    skinned += mul(float4(pos, 1), bone1) * bw.Weights.y;
    skinned += mul(float4(pos, 1), bone2) * bw.Weights.z;
    skinned += mul(float4(pos, 1), bone3) * bw.Weights.w;
    pos = skinned.xyz;

    float4 worldPos = mul(float4(pos, 1.0f), World);
    StructuredBuffer<CascadeData> cascadeData = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    output.Position = mul(worldPos, cascadeData[cascadeIdx].VP);
    output.TexCoord = uvs[vertexID];
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = desc.MaterialId;
    return output;
}

// Shadow pixel shader - alpha test only, depth written by hardware
void PS_Shadow(ShadowVSOutput input)
{
    MaterialData mat = GET_MATERIAL(input.MaterialID);
    Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
    float alpha = albedoTex.Sample(Sampler, input.TexCoord).a;
    //clip(alpha - 0.25f);
}

PSOutput PS(VSOutput input)
{
    PSOutput output;
    
    uint materialID = input.MaterialID;
    
    // Material lookup via MaterialID indirection
    MaterialData mat = GET_MATERIAL(materialID);
    Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
    
    float4 color = albedoTex.Sample(Sampler, input.TexCoord);
    //clip(color.a - 0.25f);
    
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
    output.EntityId = (input.TransformSlot << 8u) | (input.MeshPartIdx & 0xFFu);
    return output;
}

technique11 GBufferSkinned
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

