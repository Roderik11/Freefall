using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Numerics;
using Freefall.Graph;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;

namespace Freefall.PCG
{
    /// <summary>
    /// Drops each point straight down onto the scene surface under it: the highest mesh triangle (any MeshRenderer,
    /// RuntimeMesh streets and islands included) or the terrain within [point − MaxDrop, point + RayStart].
    /// Use it where TerrainProjection would bury or float props: quay decks, piers, bridges, raised pavements.
    /// Points with no surface in range are dropped; with a footprint, so are points on uneven ground (stairs, edges).
    /// </summary>
    [Category("Terrain")]
    public class MeshProjection : Node, IWorldSpaceNode
    {
        [Input]
        public SamplePointSet Input;

        /// <summary>Rays start this far above each point (m); higher surfaces (roofs, canopies) are ignored.</summary>
        [ValueRange(0f, 50f)]
        public float RayStart = 1.5f;

        /// <summary>Search this far below each point (m).</summary>
        [ValueRange(0f, 100f)]
        public float MaxDrop = 3f;

        /// <summary>The terrain counts as a surface too; the higher hit wins.</summary>
        public bool IncludeTerrain = true;

        /// <summary>Probe four more rays this far from the point (m) and require an even surface. 0 = off; ~0.3 keeps props off stairs and edges.</summary>
        [ValueRange(0f, 5f)]
        public float FootprintRadius = 0f;

        /// <summary>Largest height difference within the footprint that still counts as even (m).</summary>
        [ValueRange(0f, 2f)]
        public float MaxUnevenness = 0.05f;

        /// <summary>Skip generated PCG output (DontSave entities): land on authored geometry, not on scattered props.</summary>
        public bool IgnoreGenerated = true;

        /// <summary>Local-to-world matrix of the PCG entity. Injected by PCGComponent.</summary>
        [Browsable(false)] [Freefall.Reflection.DontSerialize]
        public Matrix4x4 WorldMatrix { get; set; } = Matrix4x4.Identity;

        /// <summary>Spawned output of the executing PCGComponent. Injected before execution.</summary>
        [Browsable(false)]
        public Entity IgnoreRoot;

        /// <summary>The executing PCGComponent: decides which generated output counts. Injected before execution.</summary>
        [Browsable(false)] [Freefall.Reflection.DontSerialize]
        public PCGComponent Owner;

        [Output]
        public SamplePointSet Output;

        private const float CellSize = 2f;

        private struct Tri
        {
            public Vector3 A, B, C;
            public float MinX, MinZ, MaxX, MaxZ;
        }

        public override void Process()
        {
            var points = GetInputValue<SamplePointSet>("Input");
            if (points == null || points.Count == 0)
            {
                SetOutput("Output", SamplePointSet.Empty());
                return;
            }

            var m = WorldMatrix;
            Matrix4x4.Invert(m, out var inv);
            var world = new Vector3[points.Count];
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            for (int i = 0; i < points.Count; i++)
            {
                world[i] = Vector3.Transform(points.position[i], m);
                min = Vector3.Min(min, world[i]);
                max = Vector3.Max(max, world[i]);
            }
            float pad = FootprintRadius + 0.01f;
            min -= new Vector3(pad, MaxDrop, pad);
            max += new Vector3(pad, RayStart, pad);

            var grid = BuildGrid(CollectTriangles(min, max), min, max, out int cols, out int rows);
            var terrain = IncludeTerrain && ComponentCache<TerrainRenderer>.All.Count > 0
                ? ComponentCache<TerrainRenderer>.All[0] as IHeightProvider
                : null;

            bool Probe(float x, float z, float top, float bottom, out float y, out Vector3 n)
            {
                y = float.MinValue;
                n = Vector3.UnitY;
                int cx = (int)((x - min.X) / CellSize), cz = (int)((z - min.Z) / CellSize);
                if (cx >= 0 && cz >= 0 && cx < cols && cz < rows && grid[cz * cols + cx] is { } cell)
                {
                    foreach (var t in cell)
                    {
                        if (x < t.MinX || x > t.MaxX || z < t.MinZ || z > t.MaxZ) continue;
                        if (!HeightAt(t, x, z, out float h) || h > top || h < bottom || h <= y) continue;
                        y = h;
                        n = Vector3.Normalize(Vector3.Cross(t.B - t.A, t.C - t.A));
                        if (n.Y < 0) n = -n;
                    }
                }
                if (terrain != null)
                {
                    float h = terrain.GetHeight(new Vector3(x, 0, z));
                    if (h <= top && h >= bottom && h > y) { y = h; n = Vector3.UnitY; }
                }
                return y > float.MinValue;
            }

            var result = points.Clone();
            if (result.normal == null || result.normal.Length != result.Count)
                result.normal = new Vector3[result.Count];
            var keep = new bool[result.Count];
            var offsets = new[] { new Vector2(1, 0), new Vector2(-1, 0), new Vector2(0, 1), new Vector2(0, -1) };

            for (int i = 0; i < result.Count; i++)
            {
                var p = world[i];
                float top = p.Y + RayStart, bottom = p.Y - MaxDrop;
                if (!Probe(p.X, p.Z, top, bottom, out float y, out var normal)) continue;

                if (FootprintRadius > 0f)
                {
                    bool even = true;
                    foreach (var o in offsets)
                    {
                        if (!Probe(p.X + o.X * FootprintRadius, p.Z + o.Y * FootprintRadius, top, bottom, out float yo, out _)
                            || MathF.Abs(yo - y) > MaxUnevenness)
                        {
                            even = false;
                            break;
                        }
                    }
                    if (!even) continue;
                }

                result.position[i] = Vector3.Transform(new Vector3(p.X, y, p.Z), inv);
                result.normal[i] = Vector3.Normalize(Vector3.TransformNormal(normal, inv));
                keep[i] = true;
            }

            SetOutput("Output", result.Filter(i => keep[i]));
        }

        /// <summary>Vertical ray vs triangle: the triangle's height at (x, z), if (x, z) lies inside its XZ projection.</summary>
        private static bool HeightAt(in Tri t, float x, float z, out float y)
        {
            y = 0;
            float d = (t.B.Z - t.C.Z) * (t.A.X - t.C.X) + (t.C.X - t.B.X) * (t.A.Z - t.C.Z);
            if (MathF.Abs(d) < 1e-9f) return false; // vertical triangle
            float w1 = ((t.B.Z - t.C.Z) * (x - t.C.X) + (t.C.X - t.B.X) * (z - t.C.Z)) / d;
            float w2 = ((t.C.Z - t.A.Z) * (x - t.C.X) + (t.A.X - t.C.X) * (z - t.C.Z)) / d;
            float w3 = 1f - w1 - w2;
            const float eps = -1e-5f;
            if (w1 < eps || w2 < eps || w3 < eps) return false;
            y = w1 * t.A.Y + w2 * t.B.Y + w3 * t.C.Y;
            return true;
        }

        /// <summary>World-space triangles of every static MeshRenderer overlapping the query box (LOD0 parts only).</summary>
        private List<Tri> CollectTriangles(Vector3 min, Vector3 max)
        {
            var tris = new List<Tri>();
            foreach (var component in ComponentCache<MeshRenderer>.All)
            {
                if (component is not MeshRenderer { Enabled: true, Mesh: { Positions: { } pos, CpuIndices: { } idx } mesh } mr) continue;
                var entity = mr.Entity;
                if (entity == null || IsUnder(entity, IgnoreRoot) || (IgnoreGenerated && entity.DontSave)) continue;

                var worldM = entity.Transform.WorldMatrix;
                if (!Overlaps(mesh.BoundingBox.Min, mesh.BoundingBox.Max, worldM, min, max)) continue;

                // Output of PCG components that run after this one does not count, even while it is in the scene
                if (Owner != null && !Owner.Sees(entity)) continue;

                var wp = new Vector3[pos.Length];
                for (int v = 0; v < pos.Length; v++) wp[v] = Vector3.Transform(pos[v], worldM);

                void AddRange(int baseVertex, int start, int count)
                {
                    int end = Math.Min(start + count, idx.Length) - 2;
                    for (int i = start; i < end; i += 3)
                    {
                        long ia = baseVertex + idx[i], ib = baseVertex + idx[i + 1], ic = baseVertex + idx[i + 2];
                        if (ia >= wp.Length || ib >= wp.Length || ic >= wp.Length) continue;
                        var t = new Tri { A = wp[ia], B = wp[ib], C = wp[ic] };
                        t.MinX = MathF.Min(t.A.X, MathF.Min(t.B.X, t.C.X));
                        t.MaxX = MathF.Max(t.A.X, MathF.Max(t.B.X, t.C.X));
                        t.MinZ = MathF.Min(t.A.Z, MathF.Min(t.B.Z, t.C.Z));
                        t.MaxZ = MathF.Max(t.A.Z, MathF.Max(t.B.Z, t.C.Z));
                        float tMinY = MathF.Min(t.A.Y, MathF.Min(t.B.Y, t.C.Y));
                        float tMaxY = MathF.Max(t.A.Y, MathF.Max(t.B.Y, t.C.Y));
                        if (t.MaxX < min.X || t.MinX > max.X || t.MaxZ < min.Z || t.MinZ > max.Z) continue;
                        if (tMaxY < min.Y || tMinY > max.Y) continue;
                        tris.Add(t);
                    }
                }

                var parts = mesh.MeshParts;
                if (parts.Count == 0)
                {
                    AddRange(0, 0, idx.Length);
                    continue;
                }
                // Only the highest-detail geometry: LOD0 plus parts outside the LOD chain.
                IEnumerable<int> partIndices = mesh.LODs.Count > 0
                    ? Concat(mesh.LODs[0].MeshPartIndices, mesh.NonLodPartIndices)
                    : Range(parts.Count);
                foreach (int pi in partIndices)
                {
                    if (pi < 0 || pi >= parts.Count || !parts[pi].Enabled) continue;
                    AddRange(parts[pi].BaseVertex, parts[pi].BaseIndex, parts[pi].NumIndices);
                }
            }
            return tris;
        }

        private static List<Tri>?[] BuildGrid(List<Tri> tris, Vector3 min, Vector3 max, out int cols, out int rows)
        {
            cols = Math.Max(1, (int)MathF.Ceiling((max.X - min.X) / CellSize));
            rows = Math.Max(1, (int)MathF.Ceiling((max.Z - min.Z) / CellSize));
            var grid = new List<Tri>?[cols * rows];
            foreach (var t in tris)
            {
                int x0 = Math.Clamp((int)((t.MinX - min.X) / CellSize), 0, cols - 1);
                int x1 = Math.Clamp((int)((t.MaxX - min.X) / CellSize), 0, cols - 1);
                int z0 = Math.Clamp((int)((t.MinZ - min.Z) / CellSize), 0, rows - 1);
                int z1 = Math.Clamp((int)((t.MaxZ - min.Z) / CellSize), 0, rows - 1);
                for (int z = z0; z <= z1; z++)
                    for (int x = x0; x <= x1; x++)
                        (grid[z * cols + x] ??= new List<Tri>()).Add(t);
            }
            return grid;
        }

        private static bool Overlaps(Vector3 bbMin, Vector3 bbMax, Matrix4x4 world, Vector3 min, Vector3 max)
        {
            var wMin = new Vector3(float.MaxValue);
            var wMax = new Vector3(float.MinValue);
            for (int c = 0; c < 8; c++)
            {
                var corner = new Vector3((c & 1) != 0 ? bbMax.X : bbMin.X,
                                         (c & 2) != 0 ? bbMax.Y : bbMin.Y,
                                         (c & 4) != 0 ? bbMax.Z : bbMin.Z);
                var p = Vector3.Transform(corner, world);
                wMin = Vector3.Min(wMin, p);
                wMax = Vector3.Max(wMax, p);
            }
            return wMax.X >= min.X && wMin.X <= max.X && wMax.Y >= min.Y && wMin.Y <= max.Y && wMax.Z >= min.Z && wMin.Z <= max.Z;
        }

        private static IEnumerable<int> Concat(int[]? a, int[]? b)
        {
            if (a != null) foreach (var i in a) yield return i;
            if (b != null) foreach (var i in b) yield return i;
        }

        private static IEnumerable<int> Range(int count)
        {
            for (int i = 0; i < count; i++) yield return i;
        }

        private static bool IsUnder(Entity entity, Entity root)
        {
            if (root == null) return false;
            for (var t = entity?.Transform; t != null; t = t.Parent)
                if (t.Entity == root) return true;
            return false;
        }
    }
}
