using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json.Serialization;
using System.Threading;
using Freefall.Base;
using Freefall.Graphics;
using PhysX;

namespace Freefall.Assets
{
    public interface IStaticMesh
    {
        Mesh Mesh { get; }
        List<MeshElement> MeshParts { get; }
    }

    public class StaticMesh : Asset, IStaticMesh
    {
        private static volatile int _instanceCount;
        private readonly int _instanceId = Interlocked.Increment(ref _instanceCount);
        public int GetInstanceId() => _instanceId;

        public Mesh Mesh { get; set; }
        public List<MeshElement> MeshParts { get; set; } = new List<MeshElement>();

        public LODGroup LODGroup = LODGroups.LargeProps;
        public List<StaticMeshLOD> LODs = new List<StaticMeshLOD>();

        /// <summary>
        /// Pre-cooked PhysX triangle mesh. Populated on background thread during asset loading
        /// so that RigidBody.Awake() doesn't need to cook on the main thread.
        /// </summary>
        [JsonIgnore]
        public TriangleMesh CookedTriMesh { get; set; }

        public StaticMesh()
        {
        }

        /// <summary>
        /// Cook the physics triangle mesh on the calling thread (background) and cache it.
        /// Uses lowest LOD for fewer triangles. Thread-safe: each call creates its own Cooking instance.
        /// </summary>
        public void CookPhysicsMesh()
        {
            if (Mesh == null || Mesh.Positions == null || Mesh.CpuIndices == null)
                return;

            int[] triangles;

            // Pick lowest LOD mesh parts for collision (fewer triangles)
            if (LODs?.Count > 0)
            {
                var lowestLod = LODs[^1];
                var lodIndices = new List<int>();
                foreach (var part in lowestLod.MeshParts)
                {
                    var meshPart = lowestLod.Mesh.MeshParts[part.MeshPartIndex];
                    for (int i = 0; i < meshPart.NumIndices; i++)
                        lodIndices.Add((int)lowestLod.Mesh.CpuIndices[meshPart.BaseIndex + i]);
                }
                triangles = lodIndices.ToArray();
            }
            else
            {
                triangles = Array.ConvertAll(Mesh.CpuIndices, i => (int)i);
            }

            var cooking = PhysicsWorld.Physics.CreateCooking();
            var desc = new TriangleMeshDesc()
            {
                Flags = (MeshFlag)0,
                Triangles = triangles,
                Points = Mesh.Positions
            };

            var stream = new MemoryStream();
            cooking.CookTriangleMesh(desc, stream);

            stream.Position = 0;
            CookedTriMesh = PhysicsWorld.Physics.CreateTriangleMesh(stream);
        }
    }

    [Serializable]
    public class StaticMeshLOD : IStaticMesh
    {
        public Mesh Mesh { get; set; }

        public List<MeshElement> MeshParts { get; set; } = new List<MeshElement>();

        public StaticMeshLOD()
        {
        }
    }

    [Serializable]
    public class MeshElement
    {
        public Mesh Mesh;
        public Graphics.Material Material; // Removed default assignment for now to avoid accidental nulls if DefaultOpaque isn't ready
        public int MeshPartIndex;
    }
}
