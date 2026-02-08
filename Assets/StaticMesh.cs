using System;
using System.Collections.Generic;
using System.Threading;
using Freefall.Graphics;

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

        public LODGroup LODGroup = LODGroups.Trees;
        public List<StaticMeshLOD> LODs = new List<StaticMeshLOD>();

        public StaticMesh()
        {
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
        public Material Material; // Removed default assignment for now to avoid accidental nulls if DefaultOpaque isn't ready
        public int MeshPartIndex;
    }
}
