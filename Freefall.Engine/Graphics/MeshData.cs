using System.Collections.Generic;
using System.Numerics;
using Vortice.Mathematics;
using Freefall.Animation;

namespace Freefall.Graphics
{
    /// <summary>
    /// CPU-only parsed mesh data from Assimp. No GPU resources.
    /// Created on background threads, consumed on main thread to create Mesh.
    /// </summary>
    public class MeshData
    {
        public Vector3[] Positions;
        public Vector3[] Normals;
        public Vector4[] Tangents;  // XYZ = tangent direction, W = bitangent sign (±1)
        public Vector2[] UVs;
        public uint[] Indices;
        public List<MeshPart> Parts = new List<MeshPart>();
        public BoundingBox BoundingBox;

        // Skeleton
        public Bone[] Bones;
        public BoneWeight[] BoneWeights;

        // LOD grouping (populated by ModelImporter when sub-meshes have LOD naming)
        public List<MeshLOD> LODs = new List<MeshLOD>();

        /// <summary>
        /// Material name per MaterialSlot. Populated from FBX material assignments.
        /// Used at asset creation time to resolve DefaultMaterials for MeshRenderer auto-init.
        /// Index = MaterialSlot, Value = material name from FBX.
        /// </summary>
        public string[] MaterialNames;
    }
}
