using System.Numerics;
using System.Runtime.InteropServices;

namespace Freefall.Graphics
{
    /// <summary>
    /// Compact per-instance descriptor: packs TransformSlot + MaterialId + CustomDataIdx + BoneBufferIdx
    /// into a single GPU struct. Replaces parallel TransformSlots[] and MaterialIds[] arrays.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct InstanceDescriptor
    {
        public uint TransformSlot;   // index into GlobalTransformBuffer
        public uint MaterialId;      // index into MaterialsBuffer
        public uint CustomDataIdx;   // the object's slot of the instance data buffer (InstanceData; its effect's defaults slot if it has none)
        public uint MeshPartIdx;     // meshpart index within the mesh (for GPU picking)
        public uint BoneBufferIdx;   // per-Animator bone buffer SRV (0 = static mesh)
    }


    /// <summary>
    /// Indirect draw command for ExecuteIndirect with MeshRegistry indirection.
    /// Contains root constants (slots 2-3) + D3D12_DRAW_INSTANCED_ARGUMENTS.
    /// VS looks up mesh buffer indices from MeshRegistry using MeshPartId.
    /// Must match command signature layout exactly.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct IndirectDrawCommand
    {
        // Root constants matching slots 2-3 in push constant buffer
        public uint MeshPartId;          // Slot 2: Index into MeshRegistry
        public uint InstanceBaseOffset;  // Slot 3: Base offset for instance ID

        // D3D12_DRAW_INSTANCED_ARGUMENTS (16 bytes) - NOT DrawIndexed because indices are bindless!
        public uint VertexCountPerInstance; // This is the INDEX count (triangles * 3)
        public uint InstanceCount;
        public uint StartVertexLocation;
        public uint StartInstanceLocation;
    }

    /// <summary>
    /// IndirectDrawCommand as an indexed draw (GraphicsDevice.IndexedCommandSignature): the mesh's index buffer is set
    /// per command, so the input assembler reads the indices and the GPU reuses a vertex shared by neighbouring triangles
    /// instead of shading it once per corner. The vertex shader gets the vertex index itself in SV_VertexID. Written by
    /// cull_instances.hlsl CSCompactCommands (IndexedDrawCommand there); 48 bytes, so each index buffer view stays
    /// 8-byte aligned.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct IndexedDrawCommand
    {
        public uint MeshPartId;          // Slot 2
        public uint InstanceBaseOffset;  // Slot 3

        // D3D12_INDEX_BUFFER_VIEW (16 bytes)
        public ulong IndexBufferAddress;
        public uint IndexBufferBytes;
        public uint IndexFormat;         // DXGI_FORMAT_R32_UINT

        // D3D12_DRAW_INDEXED_ARGUMENTS (20 bytes)
        public uint IndexCountPerInstance;
        public uint InstanceCount;
        public uint StartIndexLocation;
        public int BaseVertexLocation;
        public uint StartInstanceLocation;
        public uint Padding;
    }

    /// <summary>
    /// Size constants for buffer allocation
    /// </summary>
    public static class IndirectDrawSizes
    {
        public const int DrawInstanceSize = 36;  // 9 uints = 36 bytes (removed Vector4 BoundingSphere)
        public const int IndirectCommandSize = 24; // 2 root constants (8) + draw args (16) = 24 bytes
        public const int IndexedCommandSize = 48;  // 2 root constants (8) + index buffer view (16) + indexed draw args (20) + 4 padding
    }
}

