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
        public Vector2[] UVs;
        public uint[] Indices;
        public List<MeshPart> Parts = new List<MeshPart>();
        public BoundingBox BoundingBox;

        // Skeleton
        public Bone[] Bones;
        public BoneWeight[] BoneWeights;
    }
}
