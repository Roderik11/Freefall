using System.Numerics;
using System.Runtime.InteropServices;

namespace Freefall.Graphics
{
    /// <summary>
    /// Per-instance data for GPU-driven rendering.
    /// Stored in a GPU buffer and used by culling compute shader.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DrawInstance
    {
        /// <summary>Bindless index to position buffer (GET_INDEX(0))</summary>
        public uint PosBufferIdx;
        
        /// <summary>Bindless index to instance/transform data (GET_INDEX(1))</summary>
        public uint InstanceDataIdx;
        
        /// <summary>Bindless index to material data (GET_INDEX(2))</summary>
        public uint MaterialIdx;
        
        /// <summary>Bindless index to bone matrices - 0 for static meshes (GET_INDEX(3))</summary>
        public uint BoneBufferIdx;
        
        
        /// <summary>Mesh draw arguments - copied to indirect buffer when visible</summary>
        public uint IndexCount;
        public uint StartIndexLocation;
        public int BaseVertexLocation;
        
        /// <summary>PSO batch ID - used to route to correct indirect buffer</summary>
        public uint BatchId;
    }

    /// <summary>
    /// Indirect draw command for ExecuteIndirect with BINDLESS INDEX BUFFERS.
    /// Contains root constants (slots 2-15) + D3D12_DRAW_INSTANCED_ARGUMENTS.
    /// Must match command signature layout exactly.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct IndirectDrawCommand
    {
        // Root constants matching slots 2-15 in push constant buffer
        public uint TransformSlotsIdx;       // Slot 2: Index to transform slot buffer (uint indices into global buffer)
        public uint MaterialIdBufferIdx;    // Slot 3: Index to material ID buffer 
        public uint SortedIndicesBufferIdx; // Slot 4: Index to sorted indices buffer
        public uint BoneWeightsBufferIdx;   // Slot 5: Index to bone weights buffer (0 for static)
        public uint BonesBufferIdx;         // Slot 6: Index to bones buffer (0 for static)
        public uint IndexBufferIdx;         // Slot 7: Index to BINDLESS index buffer
        public uint BaseIndex;              // Slot 8: Base index offset into index buffer for mesh parts
        public uint PosBufferIdx;           // Slot 9: Index to position buffer
        public uint NormBufferIdx;          // Slot 10: Index to normals buffer
        public uint UVBufferIdx;            // Slot 11: Index to UV buffer
        public uint NumBones;               // Slot 12: Number of bones (for skinned shaders)
        public uint InstanceBaseOffset;     // Slot 13: Base offset for instance ID (=StartInstanceLocation)
        public uint MaterialsIdx;           // Slot 14: Index to materials buffer
        public uint GlobalTransformBufferIdx; // Slot 15: Index to global TransformBuffer
        
        // D3D12_DRAW_INSTANCED_ARGUMENTS (16 bytes) - NOT DrawIndexed because indices are bindless!
        public uint VertexCountPerInstance; // This is the INDEX count (triangles * 3)
        public uint InstanceCount;
        public uint StartVertexLocation;    // Start index in the bindless index buffer
        public uint StartInstanceLocation;
    }
    
    /// <summary>
    /// Size constants for buffer allocation
    /// </summary>
    public static class IndirectDrawSizes
    {
        public const int DrawInstanceSize = 36;  // 9 uints = 36 bytes (removed Vector4 BoundingSphere)
        public const int IndirectCommandSize = 72; // 14 root constants (56) + draw args (16) = 72 bytes
    }
}

