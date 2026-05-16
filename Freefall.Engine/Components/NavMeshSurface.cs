using System;
using System.Numerics;
using System.Text.Json.Serialization;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Graphics;
using Freefall.Navigation;
using Freefall.Reflection;
using Vortice.DXGI;
using Vortice.Mathematics;

using Component = Freefall.Base.Component;

namespace Freefall.Components
{
    /// <summary>
    /// Collects scene geometry and bakes a NavMesh asset.
    /// Attach to any entity in the scene.
    /// </summary>
    [Icon("icon_collider.png")]
    public class NavMeshSurface : Component, ISceneGizmo
    {
        /// <summary>Reference to the baked NavMesh asset.</summary>
        public Assets.NavMesh? NavMesh;

        // ── Debug Visualization (GPU mesh, built once on bake) ──

        [DontSerialize] [JsonIgnore]
        private Mesh? _gizmoMesh;

        [DontSerialize] [JsonIgnore]
        private Material? _gizmoMaterial;

        // ── Baking ──

        /// <summary>
        /// Bake the navmesh from current scene geometry.
        /// Binary data is persisted through the NavMeshLoader when the asset is saved.
        /// </summary>
        public Assets.NavMesh? Bake()
        {
            NavMesh ??= new Assets.NavMesh { Name = "NavMesh" };

            var dtNavMesh = NavMeshBuilder.Build(NavMesh, out int polyCount, out int vertCount);
            if (dtNavMesh == null)
            {
                Debug.Log("[NavMeshSurface] Bake failed — no navmesh produced.");
                return null;
            }

            NavMesh.PolyCount = polyCount;
            NavMesh.VertexCount = vertCount;

            // Serialize to bytes — NavMeshLoader.Save() will persist to cache
            NavMesh.MeshData = NavMeshSerializer.Serialize(dtNavMesh);
            NavMesh.MarkDirty();

            // Initialize the runtime world immediately
            NavMeshWorld.Initialize(dtNavMesh);

            // Build GPU debug mesh for gizmo rendering
            BuildGizmoMesh(dtNavMesh);

            Debug.Log($"[NavMeshSurface] Bake complete: {polyCount} polys, {vertCount} verts, {NavMesh.MeshData.Length} bytes");
            return NavMesh;
        }

        protected override void Awake()
        {
            // Initialize runtime if we have baked data (loaded by NavMeshLoader)
            if (NavMesh?.MeshData != null && NavMesh.MeshData.Length > 0)
            {
                NavMeshWorld.Initialize(NavMesh);
                var dtNav = NavMeshSerializer.Deserialize(NavMesh.MeshData);
                if (dtNav != null)
                    BuildGizmoMesh(dtNav);
            }
        }

        public override void Destroy()
        {
            NavMeshWorld.Shutdown();
            _gizmoMesh?.Dispose();
            _gizmoMesh = null;
        }

        // ── Debug Visualization ──

        public void DrawGizmos(GizmoContext ctx)
        {
            if (_gizmoMesh == null) return;

            // Lazy-create material (needs ctx.MeshEffect)
            if (_gizmoMaterial == null)
            {
                var color = new Color4(0.2f, 0.7f, 0.9f, 0.4f);
                byte r = (byte)(color.R * 255f);
                byte g = (byte)(color.G * 255f);
                byte b = (byte)(color.B * 255f);
                byte a = (byte)(color.A * 255f);

                byte[] texData = new byte[4 * 4 * 4];
                for (int i = 0; i < texData.Length; i += 4)
                {
                    texData[i] = r; texData[i + 1] = g; texData[i + 2] = b; texData[i + 3] = a;
                }

                var tex = Texture.CreateFromData(Engine.Device, 4, 4, texData, Format.R8G8B8A8_UNorm);
                _gizmoMaterial = new Material(ctx.MeshEffect);
                _gizmoMaterial.SetTexture("AlbedoTex", tex);
            }

            ctx.EnqueueMesh(_gizmoMesh, 0, _gizmoMaterial);
        }

        /// <summary>
        /// Build a GPU Mesh from the DotRecast navmesh for gizmo rendering.
        /// Called once on bake or load — zero per-frame CPU work.
        /// </summary>
        private void BuildGizmoMesh(DotRecast.Detour.DtNavMesh navMesh)
        {
            _gizmoMesh?.Dispose();
            _gizmoMesh = null;
            _gizmoMaterial = null; // force rebuild with potentially new effect

            var verts = new System.Collections.Generic.List<Vector3>();
            var indices = new System.Collections.Generic.List<uint>();

            for (int tileIdx = 0; tileIdx < navMesh.GetMaxTiles(); tileIdx++)
            {
                var tile = navMesh.GetTile(tileIdx);
                if (tile?.data == null) continue;

                uint baseVert = (uint)verts.Count;

                for (int v = 0; v < tile.data.header.vertCount; v++)
                {
                    verts.Add(new Vector3(
                        tile.data.verts[v * 3],
                        tile.data.verts[v * 3 + 1],
                        tile.data.verts[v * 3 + 2]));
                }

                for (int p = 0; p < tile.data.header.polyCount; p++)
                {
                    var poly = tile.data.polys[p];
                    if (poly.GetPolyType() == DotRecast.Detour.DtPolyTypes.DT_POLYTYPE_OFFMESH_CONNECTION)
                        continue;

                    for (int j = 2; j < poly.vertCount; j++)
                    {
                        indices.Add(baseVert + (uint)poly.verts[0]);
                        indices.Add(baseVert + (uint)poly.verts[j - 1]);
                        indices.Add(baseVert + (uint)poly.verts[j]);
                    }
                }
            }

            if (verts.Count == 0 || indices.Count == 0) return;

            // Build normals + UVs (flat up, zero UVs — unlit gizmo)
            var normals = new Vector3[verts.Count];
            var uvs = new Vector2[verts.Count];
            Array.Fill(normals, Vector3.UnitY);

            var mesh = new Mesh(Engine.Device,
                verts.ToArray(), normals, uvs, indices.ToArray());

            // Compute bounds
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            foreach (var v in verts)
            {
                min = Vector3.Min(min, v);
                max = Vector3.Max(max, v);
            }
            mesh.BoundingBox = new BoundingBox(min, max);
            mesh.Guid = Guid.NewGuid().ToString("N");
            mesh.Name = "NavMeshGizmo";
            mesh.IsDynamic = true;

            mesh.MeshParts.Add(new MeshPart
            {
                Name = "NavMesh",
                NumIndices = indices.Count,
                BoundingBox = mesh.BoundingBox,
                BoundingSphere = mesh.LocalBoundingSphere
            });

            _gizmoMesh = mesh;
        }
    }
}
