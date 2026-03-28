// Ghost placement preview — Forward pass, semi-transparent, depth-tested.
// Does NOT write to GBuffer or EntityId, so picks pass through to underlying geometry.

cbuffer PushConstants : register(b3)
{
    uint _reserved0;
    uint _reserved1;
    uint DescriptorBufIdx;      // 2
    uint _reserved3;            // 3
    uint SortedIndicesIdx;      // 4
    uint BoneWeightsIdx;        // 5
    uint BonesIdx;              // 6
    uint IndexBufferIdx;        // 7
    uint BaseIndex;             // 8
    uint PosBufferIdx;          // 9
    uint NormBufferIdx;         // 10
    uint UVBufferIdx;           // 11
    uint NumBones;              // 12
    uint InstanceBaseOffset;    // 13
    uint MaterialsIdx;          // 14
    uint GlobalTransformBufferIdx; // 15
    uint DebugMode;             // 16
};

#include "common.fx"

SamplerState Sampler : register(s0);

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 WorldPos : TEXCOORD1;
    nointerpolation uint MaterialID : TEXCOORD2;
};

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[UVBufferIdx];

    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;

    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    float3 pos = positions[vertexID];
    float3 norm = normals[vertexID];
    float2 uv = uvs[vertexID];

    float4 worldPos = mul(float4(pos, 1.0f), World);
    output.WorldPos = worldPos;
    output.Position = mul(mul(worldPos, View), Projection);
    output.Normal = mul(norm, (float3x3)World);
    output.TexCoord = uv;
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = desc.MaterialId;
    return output;
}

struct PSOutput
{
    float4 Color : SV_Target0;      // Composite buffer (R16G16B16A16_Float)
    uint   EntityId : SV_Target1;   // EntityIdBuffer (R32_UInt) — 0 = not pickable
};

PSOutput PS(VSOutput input)
{
    PSOutput output;

    // Simple rim lighting for hologram look
    float3 N = normalize(input.Normal);
    float3 V = normalize(CamPos - input.WorldPos.xyz);
    float rim = 1.0 - saturate(dot(N, V));
    rim = pow(rim, 2.0);

    // Flat hologram: blue-tinted with rim glow
    float3 ghostColor = float3(0.15, 0.25, 0.4) + float3(0.3, 0.6, 1.0) * rim;
    output.Color = float4(ghostColor, 0.4);
    output.EntityId = 0;
    return output;
}

technique11 GBuffer
{
    pass Forward
    {
        SetRenderState(RenderTargets=2, DepthWrite=false, Blend=AlphaBlend);
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
