using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Freefall.Assets;
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
        }

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
        }

        public static Part[] Of(Prefab prefab)
        {
            if (s_parts.TryGetValue(prefab, out var parts)) return parts;
            var list = new List<Part>();
            var renderers = new List<Renderer>();
            var root = prefab.Instantiate();
            if (root != null)
            {
                foreach (var r in root.GetComponentsInChildren<MeshRenderer>())
                {
                    var mesh = r.Mesh;
                    if (mesh == null) continue;
                    var toRoot = r.Transform?.WorldMatrix ?? Matrix4x4.Identity;   // the copy's root is at the origin
                    Material? MaterialOf(int slot) => r.Materials.FirstOrDefault(m => m.MaterialSlot == slot)?.Material ?? r.Material;

                    var chains = new List<(int, Material)>();
                    foreach (int head in mesh.DrawPartIndices)
                    {
                        if (head >= mesh.MeshParts.Count) continue;
                        var material = MaterialOf(mesh.MeshParts[head].MaterialSlot);
                        if (material != null) chains.Add((head, material));
                    }
                    if (chains.Count > 0) renderers.Add(new Renderer { Mesh = mesh, ToRoot = toRoot, Chains = chains.ToArray() });

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
            return parts;
        }
    }
}
