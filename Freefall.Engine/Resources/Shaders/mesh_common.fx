// What every effect that draws batched meshes (MeshRenderer, SkinnedMeshRenderer's static parts, PCG output) starts
// with, so a project's own .fx is its settings, a vertex and a pixel shader and a technique. Include it instead of
// common.fx (it includes common.fx, after the push constants):
//
//     #include "mesh_common.fx"
//
// It gives
//   - the push constants of a batched mesh draw (per draw: MeshPartId, InstanceBaseOffset; per batch: the
//     descriptor, sorted-index, mesh-registry, materials and transform buffers; ShadowMapIdx and CompositeSRVIdx
//     in the Forward pass);
//   - FetchMeshVertex(SV_VertexID, SV_InstanceID): the vertex in world space, its normal, UV and transform, and
//     whose material settings and instance data the draw has;
//   - LoadMaterialSettings<T>(materialId) for the effect's Settings struct (its "//=" members: material settings,
//     saved with the .mat) and LoadInstanceData<T>(slot) for its Instance struct (values per object, saved on the
//     renderer);
//   - SamplerWrap (anisotropic, wrap) and SamplerClamp (linear, clamp), static samplers of the root signature;
//   - from common.fx: SceneConstants (Time, CamPos, View, Projection, ViewProjection, fog), MaterialData,
//     InstanceDescriptor, MeshPartEntry, TextureRef.
//
// Shadow passes draw their instances from the cascade expansion buffer instead (see gbuffer.fx VS_Shadow).

#ifndef MESH_COMMON_FX
#define MESH_COMMON_FX

cbuffer PushConstants : register(b3)
{
    uint ShadowMapIdx;              // 0: shadow cascades (Forward pass only)
    uint CompositeSRVIdx;           // 1: snapshot of the lit scene before the Forward pass (Forward pass only)
    uint MeshPartId;                // 2: per draw (command signature)
    uint InstanceBaseOffset;        // 3: per draw (command signature)
    uint DescriptorBufIdx;          // 4: StructuredBuffer<InstanceDescriptor>
    uint SortedIndicesIdx;          // 5: StructuredBuffer<uint>, sorted draw order
    uint MeshRegistryIdx;           // 6: StructuredBuffer<MeshPartEntry>
    uint MaterialsIdx;              // 7: StructuredBuffer<MaterialData>
    uint GlobalTransformBufferIdx;  // 8: StructuredBuffer<row_major matrix>
    uint _reserved9;
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint _reserved15;
    uint DebugMode;                 // 16: Engine.Settings.DebugVisualizationMode
    uint IndexedDraw;               // 17: 1 = an indexed draw, the vertex index is SV_VertexID (common.fx MeshVertexIndex)
};

#include "common.fx"

SamplerState SamplerWrap : register(s0);    // anisotropic x16, wrap
SamplerState SamplerClamp : register(s2);   // linear, clamp

struct MeshVertex
{
    float3 Position;        // world space
    float3 Normal;          // world space, unit length; by the inverse transpose, so a stretched mesh keeps it square to its surface
    float2 UV;              // as in the mesh (the engine's surface shaders sample textures at (u, 1 - v))
    float3 LocalPosition;   // the mesh's own space
    float3 LocalNormal;
    row_major float4x4 World;
    uint MaterialId;        // LoadMaterialSettings<Settings>(MaterialId)
    uint InstanceSlot;      // LoadInstanceData<Instance>(InstanceSlot): the object's slot, or the effect's defaults
    uint TransformSlot;
};

MeshVertex FetchMeshVertex(uint vertexId, uint instanceId)
{
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> transforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];

    // The high bit of a sorted index marks a LOD the culler chose; the rest is the instance
    InstanceDescriptor desc = descriptors[sortedIndices[InstanceBaseOffset + instanceId] & 0x7FFFFFFFu];
    row_major float4x4 world = transforms[desc.TransformSlot];
    float3x3 w3 = (float3x3)world;

    // A mirrored transform turns the triangles round: swap two corners so they keep facing out
    uint fetch = vertexId;
    if (determinant(w3) < 0.0)
    {
        uint corner = vertexId % 3u;
        fetch = vertexId - corner + (corner == 1u ? 2u : (corner == 2u ? 1u : 0u));
    }

    // An indexed draw hands the vertex over (the camera draws a part with a mirrored instance in view pulled)
    StructuredBuffer<MeshPartEntry> registry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = registry[MeshPartId];
    uint index = MeshVertexIndex(part, vertexId, fetch, IndexedDraw);

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[part.NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];

    MeshVertex v;
    v.LocalPosition = positions[index];
    v.LocalNormal = normals[index];
    v.UV = uvs[index];
    v.World = world;
    v.Position = mul(float4(v.LocalPosition, 1.0), world).xyz;
    float3 n = v.LocalNormal;
    v.Normal = normalize(n.x * cross(w3[1], w3[2]) + n.y * cross(w3[2], w3[0]) + n.z * cross(w3[0], w3[1]));
    v.MaterialId = desc.MaterialId;
    v.InstanceSlot = desc.CustomDataIdx;
    v.TransformSlot = desc.TransformSlot;
    return v;
}

// The clip-space position of a world-space point
float4 WorldToClip(float3 position)
{
    return mul(mul(float4(position, 1.0), View), Projection);
}

// A material's settings: the members of the effect's Settings struct ("//=" lines), Material.SettingsStride (256)
// bytes at the material's ID, written by the material inspector, the .mat and Material.SetProperty
template<typename T>
T LoadMaterialSettings(uint materialId)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    ByteAddressBuffer all = ResourceDescriptorHeap[NonUniformResourceIndex(materials[materialId].SettingsBufferIdx)];
    return all.Load<T>(materialId * 256u);
}

// An object's values for the effect's Instance struct ("//=" lines): InstanceData.Stride (128) bytes at its slot,
// written by its renderer (inspector, PersistentRenderer.SetInstance)
template<typename T>
T LoadInstanceData(uint slot)
{
    ByteAddressBuffer all = ResourceDescriptorHeap[InstanceDataIdx];
    return all.Load<T>(slot * 128u);
}

#endif
