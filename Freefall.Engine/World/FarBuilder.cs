using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.World
{
    /// <summary>
    /// Builds an owner's far mesh on workers: for each of its cells its pieces' coarsest LODs in world space, the triangles
    /// whose three corners are under the ground or inside another piece dropped (each piece's shell rasterized into vertical
    /// columns once, so "inside" is a lookup), welded, simplified and normals split at creases within the cell, so no
    /// triangle spans two cells; then the cells in one mesh, a part per chunk and material. Each vertex names its cell: a
    /// card by its tree's origin (the normal stream), a shell's vertex by its UV, the cell's centre (Mesh.CellFade).
    /// </summary>
    internal static class FarBuilder
    {
        /// <summary>Simplification error of the rock shells, in metres.</summary>
        internal const float Error = 0.25f;

        /// <summary>Metres a side of a chunk: a part per chunk and material, culled on its own.</summary>
        internal const float ChunkSize = 512f;

        private const float ColumnStep = 0.25f;
        private static readonly float CreaseCos = MathF.Cos(50f * MathF.PI / 180f);

        internal sealed class Result
        {
            public WorldCells.Owner Owner = null!;
            public int Generation;
            public Vector3[] Pos = Array.Empty<Vector3>(), Nrm = Array.Empty<Vector3>();
            public Vector2[] UV = Array.Empty<Vector2>();
            public Vector4[] Tan = Array.Empty<Vector4>();
            public uint[] Idx = Array.Empty<uint>();
            public bool Cards;            // has merged cards: the normal stream holds tree origins (Mesh.CardOrigins)
            public long CardTriangles;
            public List<Material> Materials = new();
            public List<(int Material, int BaseIndex, int Count, BoundingBox Box)> Parts = new();
            public double TotalMs;
        }

        // One record's shells in world space
        private sealed class Piece
        {
            public Vector3[] Pos = Array.Empty<Vector3>();
            public int[] Tri = Array.Empty<int>();
            public Material?[] TriMaterial = Array.Empty<Material?>();   // per triangle
            public Vector3 Min, Max;
            public Columns? Columns;
        }

        // One cell's output for one material
        private sealed class Output
        {
            public Material Material = null!;
            public Vector3[] Pos = Array.Empty<Vector3>(), Nrm = Array.Empty<Vector3>();
            public Vector2[] UV = Array.Empty<Vector2>();
            public Vector4[] Tan = Array.Empty<Vector4>();
            public uint[] Idx = Array.Empty<uint>();
        }

        // What the worker needs of a cell, copied on the main thread
        private sealed class CellIn
        {
            public (int X, int Z) Key;
            public Vector2 Centre;
            public int[] Records = Array.Empty<int>();
        }

        // A shell's crossings with vertical lines on a grid: column c's heights are Ys[Start[c]..Start[c + 1]), sorted
        private sealed class Columns
        {
            public float MinX, MinZ;
            public int NX, NZ;
            public int[] Start = Array.Empty<int>();
            public float[] Ys = Array.Empty<float>();

            public static Columns Of(Piece p)
            {
                var c = new Columns { MinX = p.Min.X, MinZ = p.Min.Z };
                c.NX = Math.Max(1, (int)MathF.Ceiling((p.Max.X - p.Min.X) / ColumnStep));
                c.NZ = Math.Max(1, (int)MathF.Ceiling((p.Max.Z - p.Min.Z) / ColumnStep));
                var hits = new List<(int Col, float Y)>();
                for (int t = 0; t < p.Tri.Length; t += 3)
                {
                    var a = p.Pos[p.Tri[t]]; var b = p.Pos[p.Tri[t + 1]]; var d = p.Pos[p.Tri[t + 2]];
                    float area = (b.X - a.X) * (d.Z - a.Z) - (d.X - a.X) * (b.Z - a.Z);
                    if (MathF.Abs(area) < 1e-9f) continue;   // seen edge-on from above: no vertical line crosses it
                    int i0 = Math.Max(0, (int)MathF.Floor((MathF.Min(a.X, MathF.Min(b.X, d.X)) - c.MinX) / ColumnStep));
                    int i1 = Math.Min(c.NX - 1, (int)MathF.Floor((MathF.Max(a.X, MathF.Max(b.X, d.X)) - c.MinX) / ColumnStep));
                    int k0 = Math.Max(0, (int)MathF.Floor((MathF.Min(a.Z, MathF.Min(b.Z, d.Z)) - c.MinZ) / ColumnStep));
                    int k1 = Math.Min(c.NZ - 1, (int)MathF.Floor((MathF.Max(a.Z, MathF.Max(b.Z, d.Z)) - c.MinZ) / ColumnStep));
                    for (int k = k0; k <= k1; k++)
                    for (int i = i0; i <= i1; i++)
                    {
                        float x = c.MinX + (i + 0.5f) * ColumnStep + 1.3e-4f, z = c.MinZ + (k + 0.5f) * ColumnStep + 0.7e-4f;
                        float w0 = (b.X - x) * (d.Z - z) - (d.X - x) * (b.Z - z);
                        float w1 = (d.X - x) * (a.Z - z) - (a.X - x) * (d.Z - z);
                        float w2 = (a.X - x) * (b.Z - z) - (b.X - x) * (a.Z - z);
                        if (!((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0))) continue;
                        float y = (w0 * a.Y + w1 * b.Y + w2 * d.Y) / (w0 + w1 + w2);
                        hits.Add((k * c.NX + i, y));
                    }
                }
                hits.Sort((u, v) => u.Col != v.Col ? u.Col.CompareTo(v.Col) : u.Y.CompareTo(v.Y));
                c.Start = new int[c.NX * c.NZ + 1];
                c.Ys = new float[hits.Count];
                for (int h = 0; h < hits.Count; h++) { c.Start[hits[h].Col + 1]++; c.Ys[h] = hits[h].Y; }
                for (int s = 0; s < c.NX * c.NZ; s++) c.Start[s + 1] += c.Start[s];
                return c;
            }

            // Inside when an odd number of the shell's crossings lie above the point
            public bool Inside(Vector3 p)
            {
                int i = (int)MathF.Floor((p.X - MinX) / ColumnStep), k = (int)MathF.Floor((p.Z - MinZ) / ColumnStep);
                if (i < 0 || k < 0 || i >= NX || k >= NZ) return false;
                int col = k * NX + i, above = 0;
                for (int h = Start[col]; h < Start[col + 1]; h++) if (Ys[h] > p.Y + 0.02f) above++;
                return (above & 1) == 1;
            }
        }

        /// <summary>Main thread: copies what the build needs and starts it on a worker; the result is made into the scene by <see cref="WorldCellSystem"/>.</summary>
        public static void Start(WorldCells.Owner o, PrefabShapes.Part[][] shapes, bool mergeCards)
        {
            var records = o.Records.ToArray();
            var cells = o.Slices.Select(s => new CellIn { Key = s.Cell.Key, Centre = s.Cell.Centre, Records = s.Records.ToArray() }).ToArray();
            var toWorld = o.ToWorld;
            Matrix4x4.Invert(toWorld, out var toLocal);
            var terrain = ComponentCache<TerrainRenderer>.All.FirstOrDefault();
            int generation = o.Generation;
            string name = o.Name;
            Task.Run(() =>
            {
                try
                {
                    var result = Build(o, generation, records, cells, shapes, toWorld, toLocal, terrain, mergeCards);
                    WorldCellSystem.Built.Enqueue(result);
                }
                catch (Exception ex) { Debug.LogWarning("WorldCells", $"'{name}': far mesh build failed: {ex}"); }
            });
        }

        private static Result Build(WorldCells.Owner owner, int generation, WorldCells.Record[] records, CellIn[] cells,
            PrefabShapes.Part[][] shapes, Matrix4x4 toWorld, Matrix4x4 toLocal, TerrainRenderer? terrain, bool mergeCards)
        {
            var total = Stopwatch.StartNew();
            // Every piece's shells in world space, made once before the cells look at them
            var pieces = new Piece[records.Length];
            Parallel.For(0, records.Length, i => pieces[i] = MakePiece(i));
            Piece MakePiece(int i)
            {
                var r = records[i];
                var m = WorldCells.LocalOf(r) * toWorld;
                var pos = new List<Vector3>(); var tri = new List<int>(); var mats = new List<Material?>();
                foreach (var part in shapes[r.Prefab])
                {
                    if (part.Kind != PrefabShapes.Kind.Shell) continue;   // cards are merged as they are
                    int baseV = pos.Count;
                    foreach (var v in part.Pos) pos.Add(Vector3.Transform(v, m));
                    for (int t = 0; t < part.Tri.Length; t += 3)
                    {
                        tri.Add(part.Tri[t] + baseV); tri.Add(part.Tri[t + 1] + baseV); tri.Add(part.Tri[t + 2] + baseV);
                        mats.Add(part.Material);
                    }
                }
                var p = new Piece { Pos = pos.ToArray(), Tri = tri.ToArray(), TriMaterial = mats.ToArray() };
                p.Min = new Vector3(float.MaxValue); p.Max = new Vector3(float.MinValue);
                foreach (var v in p.Pos) { p.Min = Vector3.Min(p.Min, v); p.Max = Vector3.Max(p.Max, v); }
                return p;
            }

            var byKey = cells.ToDictionary(c => c.Key);
            var perCell = new List<Output>[cells.Length];

            Parallel.For(0, cells.Length, ci =>
            {
                var cell = cells[ci];
                var outputs = new List<Output>();
                var cellUV = cell.Centre;   // what each shell vertex of this cell carries (world XZ)
                var own = cell.Records.Select(i => pieces[i]).ToArray();

                // Cards, merged as they are, in the owner's frame: their UVs, each vertex naming its tree's origin (the
                // normal stream) and its card's centre (the tangent stream), as the tree's own cards take them (Mesh.CardOrigins)
                var cards = new Dictionary<Material, (List<Vector3> P, List<Vector3> O, List<Vector2> U, List<Vector4> C, List<uint> I)>();
                foreach (int ri in cell.Records)
                {
                    var r = records[ri];
                    Matrix4x4? local = null;
                    foreach (var part in shapes[r.Prefab])
                    {
                        if (!mergeCards || part.Kind != PrefabShapes.Kind.Card || part.Material == null) continue;
                        var lm = local ??= WorldCells.LocalOf(r);
                        if (!cards.TryGetValue(part.Material, out var c)) cards[part.Material] = c = (new(), new(), new(), new(), new());
                        uint baseV = (uint)c.P.Count;
                        var centre = Vector3.Transform(part.Centre, lm);
                        for (int v = 0; v < part.Pos.Length; v++)
                        {
                            c.P.Add(Vector3.Transform(part.Pos[v], lm));
                            c.O.Add(r.Position);
                            c.U.Add(part.UV[v]);
                            c.C.Add(new Vector4(centre, 1f));
                        }
                        foreach (int t in part.Tri) c.I.Add(baseV + (uint)t);
                    }
                }
                foreach (var (material, c) in cards)
                    outputs.Add(new Output { Material = material, Pos = c.P.ToArray(), Nrm = c.O.ToArray(), UV = c.U.ToArray(), Tan = c.C.ToArray(), Idx = c.I.ToArray() });

                if (own.Any(p => p.Tri.Length > 0))
                {
                    var neighbours = new List<int>(cell.Records);
                    for (int dz = -1; dz <= 1; dz++)
                    for (int dx = -1; dx <= 1; dx++)
                        if ((dx != 0 || dz != 0) && byKey.TryGetValue((cell.Key.X + dx, cell.Key.Z + dz), out var n)) neighbours.AddRange(n.Records);
                    var others = neighbours.Select(i => (Index: i, Piece: pieces[i])).Where(o => o.Piece.Tri.Length > 0).ToArray();

                    // The neighbours by a 2 m grid over the ground: a corner looks only at the pieces over its square
                    const float G = 2f;
                    var nearBy = new Dictionary<(int, int), List<int>>();
                    for (int n = 0; n < others.Length; n++)
                    {
                        var q = others[n].Piece;
                        for (int gz = (int)MathF.Floor(q.Min.Z / G); gz <= (int)MathF.Floor(q.Max.Z / G); gz++)
                        for (int gx = (int)MathF.Floor(q.Min.X / G); gx <= (int)MathF.Floor(q.Max.X / G); gx++)
                        {
                            if (!nearBy.TryGetValue((gx, gz), out var list)) nearBy[(gx, gz)] = list = new List<int>(4);
                            list.Add(n);
                        }
                    }

                    // Hidden corners
                    var hidden = new bool[own.Length][];
                    for (int o = 0; o < own.Length; o++)
                    {
                        var p = own[o];
                        int self = cell.Records[o];
                        var h = hidden[o] = new bool[p.Pos.Length];
                        for (int v = 0; v < p.Pos.Length; v++)
                        {
                            var x = p.Pos[v];
                            if (terrain != null && x.Y < terrain.GetHeight(x) - 0.05f) { h[v] = true; continue; }
                            if (!nearBy.TryGetValue(((int)MathF.Floor(x.X / G), (int)MathF.Floor(x.Z / G)), out var candidates)) continue;
                            foreach (int n in candidates)
                            {
                                var (index, q) = others[n];
                                if (index == self) continue;
                                if (x.X < q.Min.X || x.Y < q.Min.Y || x.Z < q.Min.Z || x.X > q.Max.X || x.Y > q.Max.Y || x.Z > q.Max.Z) continue;
                                Columns cols;
                                lock (q) cols = q.Columns ??= Columns.Of(q);
                                if (cols.Inside(x)) { h[v] = true; break; }
                            }
                        }
                    }

                    // Per material: the visible triangles welded, simplified, normals split at creases, all within the cell;
                    // each vertex's UV the cell's centre
                    foreach (var material in own.SelectMany(p => p.TriMaterial).Distinct())
                    {
                        if (material == null) continue;
                        var weld = new Dictionary<(int, int, int), uint>();
                        var pos = new List<float>();
                        var idx = new List<uint>();
                        for (int o = 0; o < own.Length; o++)
                        {
                            var p = own[o]; var h = hidden[o];
                            for (int t = 0; t < p.Tri.Length; t += 3)
                            {
                                if (p.TriMaterial[t / 3] != material) continue;
                                int a = p.Tri[t], b = p.Tri[t + 1], c = p.Tri[t + 2];
                                if (h[a] && h[b] && h[c]) continue;
                                foreach (int k in new[] { a, b, c })
                                {
                                    var x = Vector3.Transform(p.Pos[k], toLocal);
                                    var key = ((int)MathF.Round(x.X * 1000f), (int)MathF.Round(x.Y * 1000f), (int)MathF.Round(x.Z * 1000f));
                                    if (!weld.TryGetValue(key, out uint id)) { id = (uint)(pos.Count / 3); weld[key] = id; pos.Add(x.X); pos.Add(x.Y); pos.Add(x.Z); }
                                    idx.Add(id);
                                }
                            }
                        }
                        if (idx.Count == 0) continue;
                        var src = idx.ToArray();
                        var dst = new uint[src.Length];
                        var positions = pos.ToArray();
                        int count = Simplify(dst, src, positions);
                        if (count == 0) continue;
                        var (vs, ns, ins) = CreaseNormals(positions, dst, count);
                        var uvs = new Vector2[vs.Length];
                        Array.Fill(uvs, cellUV);
                        outputs.Add(new Output { Material = material, Pos = vs, Nrm = ns, UV = uvs, Tan = new Vector4[vs.Length], Idx = ins });
                    }
                }
                perCell[ci] = outputs;
            });

            // All cells in one mesh, a part per chunk and material
            var result = new Result { Owner = owner, Generation = generation };
            var materialIndex = new Dictionary<Material, int>();
            var byChunk = new Dictionary<(int, int, int), List<Output>>();
            for (int ci = 0; ci < cells.Length; ci++)
            {
                var centre = cells[ci].Centre;
                var ck = ((int)MathF.Floor(centre.X / ChunkSize), (int)MathF.Floor(centre.Y / ChunkSize));
                foreach (var o in perCell[ci])
                {
                    if (!materialIndex.TryGetValue(o.Material, out int mi)) { mi = result.Materials.Count; materialIndex[o.Material] = mi; result.Materials.Add(o.Material); }
                    var key = (ck.Item1, ck.Item2, mi);
                    if (!byChunk.TryGetValue(key, out var list)) byChunk[key] = list = new List<Output>();
                    list.Add(o);
                }
            }
            var allPos = new List<Vector3>(); var allNrm = new List<Vector3>(); var allIdx = new List<uint>();
            var allUV = new List<Vector2>(); var allTan = new List<Vector4>();
            foreach (var ((_, _, mi), outputs) in byChunk)
            {
                int baseI = allIdx.Count;
                var min = new Vector3(float.MaxValue); var max = new Vector3(float.MinValue);
                foreach (var o in outputs)
                {
                    uint baseV = (uint)allPos.Count;
                    foreach (var v in o.Pos) { min = Vector3.Min(min, v); max = Vector3.Max(max, v); }
                    allPos.AddRange(o.Pos); allNrm.AddRange(o.Nrm); allUV.AddRange(o.UV); allTan.AddRange(o.Tan);
                    foreach (var i in o.Idx) allIdx.Add(i + baseV);
                }
                result.Parts.Add((mi, baseI, allIdx.Count - baseI, new BoundingBox(min, max)));
                if (PrefabShapes.KindOf(result.Materials[mi]) == PrefabShapes.Kind.Card) { result.Cards = true; result.CardTriangles += (allIdx.Count - baseI) / 3; }
            }
            result.Pos = allPos.ToArray(); result.Nrm = allNrm.ToArray(); result.Idx = allIdx.ToArray();
            result.UV = allUV.ToArray(); result.Tan = allTan.ToArray();
            result.TotalMs = total.Elapsed.TotalMilliseconds;
            return result;
        }

        private static unsafe int Simplify(uint[] dst, uint[] src, float[] positions)
        {
            fixed (uint* d = dst)
            fixed (uint* s = src)
            fixed (float* p = positions)
                return (int)MeshOptimizer.meshopt_simplify(d, s, (nuint)src.Length, p, (nuint)(positions.Length / 3), 12, 0, Error,
                    MeshOptimizer.SimplifyErrorAbsolute | MeshOptimizer.SimplifyPrune, null);
        }

        // Vertices split where the faces round them turn by more than the crease angle, so hard edges stay hard
        private static (Vector3[] Pos, Vector3[] Nrm, uint[] Idx) CreaseNormals(float[] positions, uint[] idx, int count)
        {
            int tris = count / 3;
            var faceN = new Vector3[tris]; var faceU = new Vector3[tris];
            Vector3 P(uint v) => new(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
            for (int t = 0; t < tris; t++)
            {
                var n = Vector3.Cross(P(idx[t * 3 + 1]) - P(idx[t * 3]), P(idx[t * 3 + 2]) - P(idx[t * 3]));
                faceN[t] = n;
                faceU[t] = n.LengthSquared() > 0 ? Vector3.Normalize(n) : Vector3.UnitY;
            }
            var byVertex = new Dictionary<uint, List<int>>();
            for (int c = 0; c < count; c++)
            {
                if (!byVertex.TryGetValue(idx[c], out var list)) byVertex[idx[c]] = list = new List<int>(6);
                list.Add(c);
            }
            var pos = new List<Vector3>(); var nrm = new List<Vector3>();
            var outIdx = new uint[count];
            var seeds = new List<(Vector3 Seed, Vector3 Sum, List<int> Corners)>();
            foreach (var (v, corners) in byVertex)
            {
                seeds.Clear();
                foreach (int c in corners)
                {
                    var u = faceU[c / 3];
                    int found = -1;
                    for (int s = 0; s < seeds.Count; s++) if (Vector3.Dot(u, seeds[s].Seed) >= CreaseCos) { found = s; break; }
                    if (found < 0) { seeds.Add((u, faceN[c / 3], new List<int> { c })); continue; }
                    var e = seeds[found];
                    e.Sum += faceN[c / 3];
                    e.Corners.Add(c);
                    seeds[found] = e;
                }
                foreach (var (_, sum, list) in seeds)
                {
                    uint id = (uint)pos.Count;
                    pos.Add(P(v));
                    nrm.Add(sum.LengthSquared() > 0 ? Vector3.Normalize(sum) : Vector3.UnitY);
                    foreach (int c in list) outIdx[c] = id;
                }
            }
            return (pos.ToArray(), nrm.ToArray(), outIdx);
        }

        /// <summary>Main thread: the built mesh into the scene under the owner's output; the shaders show it past FarTier.From.</summary>
        public static void Apply(Result r)
        {
            var o = r.Owner;
            if (o.Generation != r.Generation || !o.Alive || r.Idx.Length == 0) return;
            WorldCells.DropFar(o);
            var min = new Vector3(float.MaxValue); var max = new Vector3(float.MinValue);
            foreach (var v in r.Pos) { min = Vector3.Min(min, v); max = Vector3.Max(max, v); }
            var data = new MeshData
            {
                Positions = r.Pos, Normals = r.Nrm, UVs = r.UV, Indices = r.Idx,
                Tangents = r.Cards ? r.Tan : null!,       // cards: each card's centre (Mesh.CardOrigins)
                PushNormals = Array.Empty<Vector3>(),     // nothing displaces it
                BoundingBox = new BoundingBox(min, max),
            };
            var mesh = new Mesh(Engine.Device, data) { CardOrigins = r.Cards, CellFade = true };
            mesh.DropCpuCopies();
            mesh.Guid = Guid.NewGuid().ToString("N");
            mesh.Name = $"{o.Name} far";
            for (int p = 0; p < r.Parts.Count; p++)
            {
                var (material, baseIndex, count, box) = r.Parts[p];
                var center = (box.Min + box.Max) * 0.5f;
                mesh.MeshParts.Add(new MeshPart
                {
                    Name = $"chunk {p}", BaseIndex = baseIndex, NumIndices = count, MaterialSlot = material,
                    BoundingBox = box, BoundingSphere = new Vector4(center, (box.Max - center).Length()),
                });
            }
            mesh.ComputeNonLodPartIndices();   // the parts came after the constructor worked out which to draw
            var entity = new Entity($"{o.Name} far mesh");
            entity.Transform.Parent = o.Container!.Transform;
            var renderer = entity.AddComponent<MeshRenderer>();
            renderer.Mesh = mesh;
            renderer.Materials = r.Materials.Select((m, i) => new MaterialOverride { MaterialSlot = i, Material = m }).ToList();
            o.FarEntity = entity;
            o.FarMesh = mesh;
            o.FarVertices = r.Pos.Length;
            o.FarTriangles = r.Idx.Length / 3;
            o.FarParts = r.Parts.Count;
            o.FarCardTriangles = r.CardTriangles;
            long[] buffers = { r.Pos.Length * 12L, r.Pos.Length * 12L, r.Pos.Length * 8L, r.Cards ? r.Pos.Length * 16L : 0, r.Idx.Length * 4L };
            o.FarVram = buffers.Sum(b => (b + 65535) / 65536 * 65536);
            o.FarBuildMs = r.TotalMs;
        }
    }
}
