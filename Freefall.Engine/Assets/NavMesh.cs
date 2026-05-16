using System.Text.Json.Serialization;
using System.ComponentModel;
using Freefall.Reflection;

namespace Freefall.Assets
{
    /// <summary>
    /// Baked navigation mesh asset. Created by NavMeshSurface.Bake(),
    /// loaded by NavMeshWorld on scene start.
    /// Binary navmesh bytes are stored as a hidden subasset (NavMeshData).
    /// </summary>
    [CreateAsset("NavMesh")]
    public class NavMesh : Asset
    {
        // ── Bake Parameters (persisted in YAML, editable in inspector) ──

        [ValueRange(0.1f, 2f)]
        public float AgentRadius = 0.35f;

        [ValueRange(0.5f, 5f)]
        public float AgentHeight = 2.0f;

        /// <summary>Max step height the agent can climb.</summary>
        [ValueRange(0.1f, 2f)]
        public float MaxClimb = 0.4f;

        /// <summary>Max walkable slope in degrees.</summary>
        [ValueRange(0f, 85f)]
        public float MaxSlope = 45f;

        /// <summary>Voxel cell size (XZ). Smaller = more precise, slower bake.</summary>
        [ValueRange(0.1f, 1f)]
        public float CellSize = 0.3f;

        /// <summary>Voxel cell height (Y). Smaller = more precise, slower bake.</summary>
        [ValueRange(0.05f, 0.5f)]
        public float CellHeight = 0.2f;

        /// <summary>Terrain sampling step in meters. Controls terrain triangle density.</summary>
        [ValueRange(0.5f, 10f)]
        public float TerrainSampleStep = 2.0f;

        // ── Runtime Data (loaded from hidden subasset, not serialized) ──

        [DontSerialize]
        [JsonIgnore]
        [Browsable(false)]
        internal byte[]? MeshData;

        // ── Metadata ──

        [ReadOnly(true)]
        [JsonIgnore]
        public int PolyCount { get; internal set; }

        [ReadOnly(true)]
        [JsonIgnore]
        public int VertexCount { get; internal set; }
    }

    /// <summary>
    /// Container for baked DotRecast navmesh bytes.
    /// Stored as a hidden subasset, matching the CollisionMeshData pattern.
    /// </summary>
    public class NavMeshData
    {
        public byte[]? BakedBytes { get; set; }
    }
}
