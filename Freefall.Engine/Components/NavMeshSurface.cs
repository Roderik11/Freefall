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
    [Icon("icon_nav_mesh.png")]
    public class NavMeshSurface : Component, ISceneGizmo
    {
        /// <summary>Reference to the baked NavMesh asset.</summary>
        public Assets.NavMesh? NavMesh;

        // ── Debug Visualization (GPU mesh, built once on bake) ──

        [DontSerialize] [JsonIgnore]
        private Mesh? _gizmoMesh;

        [DontSerialize] [JsonIgnore]
        private Material? _gizmoMaterial;

        [DontSerialize] [JsonIgnore]
        private NavMeshBake? _bake;

        // From Bake() until the result has been applied on the main thread: a little longer than
        // the bake itself runs, so "not baking" always means the asset shows the result.
        [DontSerialize] [JsonIgnore]
        private bool _bakePending;

        // ── Baking ──

        /// <summary>
        /// The bake started last, running or finished. Null before the first bake.
        /// A method, so reflection-driven code (inspector, entity dumps) does not walk into it.
        /// </summary>
        public NavMeshBake? GetBake() => _bake;

        [System.ComponentModel.Browsable(false)] [JsonIgnore]
        public bool IsBaking => _bakePending;

        /// <summary>0..1 while baking, 1 once a bake has finished.</summary>
        [System.ComponentModel.Browsable(false)] [JsonIgnore]
        public float BakeProgress => _bake == null ? 0f : _bake.IsCompleted ? 1f : _bake.Progress;

        /// <summary>
        /// Start baking the navmesh from the current scene geometry. Main thread; returns once the
        /// scene has been captured, the bake itself runs on worker threads and is applied when done.
        /// Tiles whose surroundings did not change since the last bake are kept, unless
        /// <paramref name="rebuildAll"/> is set. The baked data is persisted through the
        /// NavMeshLoader when the asset is saved.
        /// </summary>
        public NavMeshBake Bake(bool rebuildAll = false)
        {
            if (IsBaking) return _bake!;

            NavMesh ??= new Assets.NavMesh { Name = "NavMesh" };

            var asset = NavMesh;
            var bake = NavMeshBuilder.Start(asset, rebuildAll ? null : asset.Tiles);
            _bake = bake;
            _bakePending = true;

            bake.Completion.ContinueWith(_ =>
            {
                // Still off the main thread: the debug mesh of a large navmesh is millions of vertices
                Vector3[]? vertices = null;
                uint[]? indices = null;
                try
                {
                    if (bake.Result?.NavMesh != null)
                        NavMeshBuilder.BuildDebugGeometry(bake.Result.NavMesh, out vertices, out indices);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("NavMeshSurface", $"Could not build the navmesh debug mesh: {ex.Message}");
                    vertices = null;
                    indices = null;
                }

                Engine.RunOnMainThreadAsync(() => ApplyBake(bake, asset, vertices, indices));
            });

            return bake;
        }

        public void CancelBake() => _bake?.Cancel();

        private void ApplyBake(NavMeshBake bake, Assets.NavMesh asset, Vector3[]? vertices, uint[]? indices)
        {
            if (_bake == bake) _bakePending = false;

            var result = bake.Result;
            if (result == null)
            {
                if (bake.Stage == NavMeshBakeStage.Cancelled)
                    Debug.Log("[NavMeshSurface] Bake cancelled.");
                return;
            }

            // A bake that found every tile unchanged leaves the asset as it was: not dirty
            bool changed = asset.Tiles == null || result.RebuiltCells > 0;

            // The asset keeps its data even if the surface went away meanwhile
            asset.Tiles = result.Tiles;
            asset.PolyCount = result.Tiles.PolyCount;
            asset.VertexCount = result.Tiles.VertexCount;
            if (changed) asset.MarkDirty();

            Debug.Log($"[NavMeshSurface] Bake complete in {result.Seconds:0.0} s: {asset.PolyCount} polys in {result.Tiles.TileCount} tiles, " +
                      $"{result.RebuiltCells} rebuilt, {result.ReusedCells} unchanged, {result.Tiles.ByteSize / 1024} KB ({result.Profile})");

            if (IsDestroyed || NavMesh != asset || _bake != bake) return;

            // Nothing new and the runtime already has it: swapping would only drop the crowd's agents
            if (!changed && NavMeshWorld.IsReady && _gizmoMesh != null) return;

            if (result.NavMesh == null)
            {
                Debug.LogWarning("NavMeshSurface", "Bake produced no walkable surface.");
                NavMeshWorld.Shutdown();
                SetGizmoMesh(null, null);
                return;
            }

            NavMeshWorld.Initialize(result.NavMesh);
            SetGizmoMesh(vertices, indices);
        }

        private static readonly ProfilerMarker _worldMarker = ProfilerMarker.Get("NavMesh world");
        private static readonly ProfilerMarker _gizmoMarker = ProfilerMarker.Get("NavMesh gizmo mesh");

        // Bumped by every gizmo change (a bake's result included): a navmesh or debug mesh still being built on a worker
        // for an older state is dropped
        [DontSerialize] [JsonIgnore]
        private int _gizmoGeneration;

        private const float GizmoLift = 0.05f;

        protected override void Awake()
        {
            // Initialize runtime if we have baked data (loaded by NavMeshLoader)
            var tiles = NavMesh?.Tiles;
            if (tiles == null) return;

            // Assembling the tiles (~0.5 s on Eldermoor) and the editor's debug mesh run on a worker; the runtime takes
            // the navmesh on the main thread. A scene load waits for it (agents only move once the world runs again);
            // outside a load the navmesh arrives a moment after the component wakes. A bake started meanwhile wins.
            bool gizmo = Engine.HostedByEditor;
            int generation = ++_gizmoGeneration;
            var load = System.Threading.Tasks.Task.Run(() =>
            {
                var navMesh = tiles.CreateNavMesh();
                Vector3[]? vertices = null;
                uint[]? indices = null;
                if (navMesh != null && gizmo)
                {
                    try
                    {
                        NavMeshBuilder.BuildDebugGeometry(navMesh, out vertices, out indices);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning("NavMeshSurface", $"Could not build the navmesh debug mesh: {ex.Message}");
                        vertices = null;
                        indices = null;
                    }
                }

                return Engine.RunOnMainThreadAsync(() =>
                {
                    if (IsDestroyed || generation != _gizmoGeneration || navMesh == null) return;
                    using (_worldMarker.Auto())
                        NavMeshWorld.Initialize(navMesh);
                    if (gizmo) SetGizmoMesh(vertices, indices);
                });
            });

            SceneManager.Loading?.WaitFor(load, "Navmesh");
        }

        public override void Destroy()
        {
            _bake?.Cancel();
            NavMeshWorld.Shutdown();

            // Deferred: gizmo draws of the frames in flight still reference the mesh
            Engine.Device.DeferDispose(_gizmoMesh);
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
        /// Replace the GPU mesh for gizmo rendering with the navmesh polygons from
        /// NavMeshBuilder.BuildDebugGeometry. Called once on bake or load — zero per-frame CPU work.
        /// </summary>
        private void SetGizmoMesh(Vector3[]? vertices, uint[]? indices)
        {
            using var _ = _gizmoMarker.Auto();
            _gizmoGeneration++;

            // Deferred: on a rebake, gizmo draws of the frames in flight still reference the old mesh
            Engine.Device.DeferDispose(_gizmoMesh);
            _gizmoMesh = null;
            _gizmoMaterial = null; // force rebuild with potentially new effect

            if (vertices == null || indices == null || vertices.Length == 0 || indices.Length == 0) return;

            // Drawn a little over the polygons: they lie exactly on floors and paving at round heights (the bake
            // keeps a surface on a cell boundary in its cell), where the overlay would z-fight with the ground
            for (int i = 0; i < vertices.Length; i++)
                vertices[i].Y += GizmoLift;

            // Build normals + UVs (flat up, zero UVs — unlit gizmo)
            var normals = new Vector3[vertices.Length];
            var uvs = new Vector2[vertices.Length];
            Array.Fill(normals, Vector3.UnitY);

            var mesh = new Mesh(Engine.Device, vertices, normals, uvs, indices);

            // Compute bounds
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            foreach (var v in vertices)
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
                NumIndices = indices.Length,
                BoundingBox = mesh.BoundingBox,
                BoundingSphere = mesh.LocalBoundingSphere
            });

            _gizmoMesh = mesh;
        }
    }
}
