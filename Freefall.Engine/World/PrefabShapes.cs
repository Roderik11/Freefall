using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;

namespace Freefall.World
{
    /// <summary>
    /// What a prefab draws, read once per prefab from one copy of it: its renderers' LOD chains (for the world cells'
    /// instances) and its coarsest LOD as triangles (for the far forms). Main thread; the parts are read-only after.
    /// </summary>
    internal static class PrefabShapes
    {
        /// <summary>How a far form takes a part: a closed shell in a world-space material (merged, hidden faces dropped,
        /// simplified), a tree's cards (a far forest, or merged as they are), or not at all.</summary>
        internal enum Kind { Shell, Card, Skip }

        internal sealed class Part
        {
            public Material? Material;
            public Kind Kind;
            public Vector3[] Pos = Array.Empty<Vector3>();
            public int[] Tri = Array.Empty<int>();
            public Vector2[] UV = Array.Empty<Vector2>();   // cards: per vertex
            public Vector3 Centre;                           // cards: the part's bounds centre (what the effect lays them flat about)
            public Mesh? Mesh;                               // where the part comes from: a far forest draws it as it is
            public int PartIndex;
            public Matrix4x4 ToRoot;                         // its renderer under the prefab's root
        }

        /// <summary>The far form a part's material gives it: CliffRock shells merge, gbuffer_crossmesh cards make trees.</summary>
        internal static Kind KindOf(Material? m) => m?.Effect?.Name switch
        {
            "CliffRock" => Kind.Shell,
            "gbuffer_crossmesh" => Kind.Card,
            _ => Kind.Skip,
        };

        /// <summary>One MeshRenderer of a prefab: its mesh, where it sits under the root, and its LOD chain heads with their
        /// materials, as MeshRenderer.AddDraws registers them.</summary>
        internal sealed class Renderer
        {
            public Mesh Mesh = null!;
            public Matrix4x4 ToRoot;
            public (int Part, Material Material)[] Chains = Array.Empty<(int, Material)>();

            /// <summary>The local matrices from the renderer's entity up to the root's child (the root's own is the
            /// placement's): a copy's world matrix is made from them level by level, as Transform.Matrix makes it, so a
            /// record's renderer has exactly the matrix its entity would have (<see cref="WorldOf"/>). Empty: on the root.</summary>
            public Matrix4x4[] Chain = Array.Empty<Matrix4x4>();

            /// <summary>Drawn: the renderer is enabled.</summary>
            public bool Enabled;

            /// <summary>Static for the navmesh bake: no RigidBody on it or above it in the prefab, or a static one.</summary>
            public bool Static;

            /// <summary>Under a NavMeshBlocker in the prefab: the bake clears the navmesh over and round it.</summary>
            public bool Blocks;

            /// <summary>Where it reaches its coarsest LOD: its radius × LODBias × its scale in the prefab over the coarsest
            /// LOD's screen size; × the placement's scale × LODScale × √FoVFactor = the distance (as cull_instances.hlsl
            /// switches). 0 = no LOD chain: it looks the same at every distance.</summary>
            public float CoarseTerm;

            /// <summary>What its coarsest LOD (with the parts outside the chain) is made of: tree cards (a far forest or
            /// merged cards can stand for them), CliffRock shells (a merged far mesh), anything else (no far form).</summary>
            public bool CoarseCards, CoarseShells, CoarseOther;

            /// <summary>Its world matrix in a copy whose root stands at <paramref name="rootWorld"/>, computed as the
            /// engine's transforms compute it (local × parent's world, level by level), to the bit.</summary>
            public Matrix4x4 WorldOf(in Matrix4x4 rootWorld)
            {
                var m = rootWorld;
                for (int i = Chain.Length - 1; i >= 0; i--) m = Chain[i] * m;
                return m;
            }
        }

        internal static float MaxAxisScale(in Matrix4x4 m) => MathF.Sqrt(MathF.Max(
            new Vector3(m.M11, m.M12, m.M13).LengthSquared(),
            MathF.Max(new Vector3(m.M21, m.M22, m.M23).LengthSquared(), new Vector3(m.M31, m.M32, m.M33).LengthSquared())));

        // A transform's local matrix, as Transform.Matrix makes it
        private static Matrix4x4 LocalMatrix(Transform t) =>
            Matrix4x4.CreateScale(t.Scale) * Matrix4x4.CreateFromQuaternion(t.Rotation) * Matrix4x4.CreateTranslation(t.Position);

        private static readonly Dictionary<Prefab, Part[]> s_parts = new();
        private static readonly Dictionary<Prefab, Renderer[]> s_renderers = new();

        public static Renderer[] RenderersOf(Prefab prefab)
        {
            if (!s_renderers.TryGetValue(prefab, out var renderers))
            {
                Of(prefab);
                renderers = s_renderers.GetValueOrDefault(prefab) ?? Array.Empty<Renderer>();
            }
            return renderers;
        }

        /// <summary>Forget what was read of a prefab (it was reimported).</summary>
        public static void Forget(Prefab prefab)
        {
            s_parts.Remove(prefab);
            s_renderers.Remove(prefab);
            s_live.Remove(prefab);
        }

        // Live prefabs, with the first component that makes each one live; record prefabs map to null
        private static readonly Dictionary<Prefab, string?> s_live = new();

        /// <summary>
        /// Why a prefab must be an entity wherever it is placed (the type of the first component that is neither drawn,
        /// solid, nor marked <see cref="CellRecordAttribute"/>), or null: it is kept as a record.
        /// </summary>
        public static string? LiveBecause(Prefab prefab)
        {
            if (!s_live.ContainsKey(prefab)) Of(prefab);
            return s_live.GetValueOrDefault(prefab);
        }

        /// <summary>A component a record stands for: drawn, solid or blocking the navmesh, or marked as needing no copy far away.</summary>
        internal static bool IsRecordComponent(Component c) =>
            c is Transform or MeshRenderer or Collider or NavMeshBlocker
            || c is RigidBody { IsStatic: true }
            || c.GetType().IsDefined(typeof(CellRecordAttribute), true);

        public static Part[] Of(Prefab prefab)
        {
            if (s_parts.TryGetValue(prefab, out var parts)) return parts;
            var list = new List<Part>();
            var renderers = new List<Renderer>();
            string? live = null;
            var root = prefab.Instantiate();
            if (root != null)
            {
                // Live: anything a record does not stand for, anywhere in the prefab
                var stack = new Stack<Transform>();
                stack.Push(root.Transform);
                while (stack.Count > 0 && live == null)
                {
                    var t = stack.Pop();
                    foreach (var c in t.Entity.Components)
                        if (!IsRecordComponent(c)) { live = c.GetType().Name; break; }
                    for (int i = 0; i < t.Count; i++) stack.Push(t.GetChild(i));
                }

                foreach (var r in root.GetComponentsInChildren<MeshRenderer>())
                {
                    var mesh = r.Mesh;
                    if (mesh == null || r.Entity == null) continue;
                    Material? MaterialOf(int slot) => r.Materials.FirstOrDefault(m => m.MaterialSlot == slot)?.Material ?? r.Material;

                    // Its place under the root, level by level; a body or blocker on its way up
                    var chain = new List<Matrix4x4>();
                    bool isStatic = true, bodyFound = false, blocks = false;
                    for (var t = r.Transform; t != null; t = t.Parent)
                    {
                        var e = t.Entity;
                        // The nearest body up decides, as the bake's GetComponentInParents does; above the root there is
                        // only the cell's or the output's static body
                        if (!bodyFound && e?.GetComponent<RigidBody>() is { } body) { bodyFound = true; isStatic = body.IsStatic; }
                        if (e?.GetComponent<NavMeshBlocker>() is { Enabled: true }) blocks = true;
                        if (t == root.Transform) break;
                        chain.Add(LocalMatrix(t));
                    }
                    var renderer = new Renderer { Mesh = mesh, Chain = chain.ToArray(), Enabled = r.Enabled, Static = isStatic, Blocks = blocks };
                    var toRoot = renderer.WorldOf(Matrix4x4.Identity);

                    var chains = new List<(int, Material)>();
                    foreach (int head in mesh.DrawPartIndices)
                    {
                        if (head >= mesh.MeshParts.Count) continue;
                        var material = MaterialOf(mesh.MeshParts[head].MaterialSlot);
                        if (material != null) chains.Add((head, material));
                    }
                    renderer.ToRoot = toRoot;
                    renderer.Chains = chains.ToArray();
                    renderers.Add(renderer);

                    // Its coarsest LOD: where it starts, and what it is made of (the parts outside the chain draw at every
                    // distance, so they count as well)
                    var meshBox = mesh.BoundingBox;
                    float radius = Vector3.Distance(meshBox.Max, (meshBox.Min + meshBox.Max) * 0.5f) * MaxAxisScale(toRoot);
                    IEnumerable<int> coarseParts;
                    if (mesh.LODs.Count > 1)
                    {
                        var coarsest = mesh.LODs[^1];
                        if (coarsest.ScreenSize > 0f && coarsest.ScreenSize < float.MaxValue)
                            renderer.CoarseTerm = radius * mesh.LODBias / coarsest.ScreenSize;
                        coarseParts = coarsest.MeshPartIndices.Concat(mesh.NonLodPartIndices ?? Array.Empty<int>());
                    }
                    else coarseParts = Enumerable.Range(0, mesh.MeshParts.Count);
                    foreach (int pi in coarseParts)
                    {
                        if (pi < 0 || pi >= mesh.MeshParts.Count || !mesh.MeshParts[pi].Enabled) continue;
                        var kind = KindOf(MaterialOf(mesh.MeshParts[pi].MaterialSlot));
                        if (kind == Kind.Card && mesh.CoarseUVs == null) kind = Kind.Skip;
                        if (kind == Kind.Card) renderer.CoarseCards = true;
                        else if (kind == Kind.Shell) renderer.CoarseShells = true;
                        else renderer.CoarseOther = true;
                    }

                    if (mesh.Positions == null || mesh.CpuIndices == null) continue;
                    var last = mesh.LODs.Count > 0 ? mesh.LODs[^1].MeshPartIndices : Enumerable.Range(0, mesh.MeshParts.Count).ToArray();
                    foreach (int pi in last)
                    {
                        var part = mesh.MeshParts[pi];
                        var material = MaterialOf(part.MaterialSlot);
                        var kind = KindOf(material);
                        if (kind == Kind.Card && mesh.CoarseUVs == null) kind = Kind.Skip;   // no UVs to merge them with
                        if (kind == Kind.Skip) continue;
                        var map = new Dictionary<int, int>();
                        var pos = new List<Vector3>();
                        var uv = new List<Vector2>();
                        var tri = new List<int>();
                        int end = Math.Min(part.BaseIndex + part.NumIndices, mesh.CpuIndices.Length);
                        for (int i = part.BaseIndex; i < end; i++)
                        {
                            int v = (int)mesh.CpuIndices[i] + part.BaseVertex;
                            if (!map.TryGetValue(v, out int local))
                            {
                                local = pos.Count;
                                map[v] = local;
                                pos.Add(Vector3.Transform(mesh.Positions[v], toRoot));
                                if (kind == Kind.Card)
                                {
                                    int u = v - mesh.CoarseUVBase;
                                    uv.Add(u >= 0 && u < mesh.CoarseUVs!.Length ? mesh.CoarseUVs[u] : Vector2.Zero);
                                }
                            }
                            tri.Add(local);
                        }
                        var sphere = part.BoundingSphere;
                        list.Add(new Part
                        {
                            Material = material, Kind = kind, Pos = pos.ToArray(), Tri = tri.ToArray(), UV = uv.ToArray(),
                            Centre = Vector3.Transform(new Vector3(sphere.X, sphere.Y, sphere.Z), toRoot),
                            Mesh = mesh, PartIndex = pi, ToRoot = toRoot,
                        });
                    }
                }
                root.Destroy();
            }
            parts = list.ToArray();
            s_parts[prefab] = parts;
            s_renderers[prefab] = renderers.ToArray();
            s_live[prefab] = root == null ? "nothing (the prefab made no entity)" : live;
            return parts;
        }
    }
}
