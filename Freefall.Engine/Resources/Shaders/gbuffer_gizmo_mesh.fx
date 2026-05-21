// Gizmo mesh shader — Forward pass with depth test and alpha blending.
// Used for solid mesh gizmos (navmesh overlay, etc.)
// Reads depth buffer so gizmos are properly occluded by scene geometry.
// Alpha blending for semi-transparent overlay.
// Outputs entity ID to RT1 for mouse picking.

cbuffer PushConstants : register(b3)
{
    // Slots 0-1: Reserved for light/composition passes
    uint _reserved0;
    uint _reserved1;
    // Slots 2-3: PER-DRAW (command signature writes these)
    uint MeshPartId;                // 2: Index into MeshRegistry
    uint InstanceBaseOffset;        // 3: Base offset for instance ID (per-command)
    // Slots 4-8: PER-BATCH (set before ExecuteIndirect)
    uint DescriptorBufIdx;          // 4: StructuredBuffer<InstanceDescriptor>
    uint SortedIndicesIdx;          // 5: StructuredBuffer<uint> - sorted draw order indices
    uint MeshRegistryIdx;           // 6: StructuredBuffer<MeshPartEntry>
    uint MaterialsIdx;              // 7: Index to materials buffer
    uint GlobalTransformBufferIdx;  // 8: Index to global TransformBuffer
    // Slots 9-15: Reserved
    uint _reserved9;
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint _reserved15;
    // Slot 16: Debug
    uint DebugMode;             // 16: Debug visualization mode
};

#include "common.fx"

SamplerState Sampler : register(s0);

struct VSOutput
{
    float4 Position : SV_POSITION;
    nointerpolation uint TransformSlot : TEXCOORD0;
    nointerpolation uint MaterialID : TEXCOORD1;
    nointerpolation uint MeshPartId : TEXCOORD2;
};

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];

    // Look up mesh buffer indices from MeshRegistry
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];

    StructuredBuffer<uint> indices = ResourceDescriptorHeap[part.IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + part.BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];

    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;

    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    float3 pos = positions[vertexID];
    float4 worldPos = mul(float4(pos, 1.0f), World);
    output.Position = mul(mul(worldPos, View), Projection);
    output.TransformSlot = desc.TransformSlot;
    output.MaterialID = desc.MaterialId;
    output.MeshPartId = desc.MeshPartIdx;
    return output;
}

struct PSOutput
{
    float4 Color : SV_Target0;      // Composite buffer
    uint   EntityId : SV_Target1;   // EntityIdBuffer
};

PSOutput PS(VSOutput input)
{
    PSOutput output;

    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    MaterialData mat = materials[input.MaterialID];
    Texture2D albedoTex = ResourceDescriptorHeap[mat.AlbedoIdx];
    float4 color = albedoTex.Sample(Sampler, float2(0.5, 0.5));

    output.Color = color; // alpha from texture/material
    output.EntityId = (input.TransformSlot << 8u) | (input.MeshPartId & 0xFFu);
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
