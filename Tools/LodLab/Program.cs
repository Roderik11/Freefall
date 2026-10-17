using System.Globalization;
using System.Numerics;
using System.Text;
using Freefall.Assets.Importers;
using Freefall.Graphics;

namespace LodLab
{
    /// <summary>
    /// Offline check of the LODs the model importer generates (LODGenerator): imports model files through the
    /// real ModelImporter, as the editor does, and prints per mesh the triangles of every LOD, the screen size at
    /// which it takes over (Mesh.ComputeLODScreenSizes) and the distance that is at a 60° vertical field of view.
    ///
    ///   LodLab model... [--ratios 0.5,0.25,0.125] [--min 64] [--error px] [--off] [--mesh text]
    ///   LodLab model... --sweep        the chains for error limits 1 … 8 px and none side by side
    ///   LodLab model... --obj dir      also write every LOD of every mesh as OBJ (for a Blender render)
    ///   LodLab model... --parts        list every mesh part (name, material slot, triangles)
    /// Build Debug: the generator's error report is a Debug.Log.
    /// </summary>
    static class Program
    {
        const float VerticalFovDegrees = 60f;
        static bool ShowParts;

        static int Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

            var files = new List<string>();
            List<float> ratios = null;
            int? minTriangles = null;
            float? maxError = null;
            bool off = false, sweep = false;
            string meshFilter = null, objDir = null;

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--ratios": ratios = args[++i].Split(',').Select(float.Parse).ToList(); break;
                    case "--min": minTriangles = int.Parse(args[++i]); break;
                    case "--error": maxError = float.Parse(args[++i]); break;
                    case "--off": off = true; break;
                    case "--sweep": sweep = true; break;
                    case "--mesh": meshFilter = args[++i]; break;
                    case "--obj": objDir = args[++i]; break;
                    case "--parts": ShowParts = true; break;
                    default: files.Add(args[i]); break;
                }
            }

            if (files.Count == 0)
            {
                Console.WriteLine("LodLab model... [--ratios a,b,c] [--min n] [--error px] [--off] [--mesh text] [--parts] [--sweep] [--obj dir]");
                return 1;
            }

            ModelImporter NewImporter(float? error = null)
            {
                var importer = new ModelImporter { GenerateLODs = !off };
                if (ratios != null) importer.LODRatios = ratios;
                if (minTriangles != null) importer.LODMinTriangles = minTriangles.Value;
                var limit = error ?? maxError;
                if (limit != null) importer.LODPixelError = limit.Value;
                return importer;
            }

            foreach (var file in files)
            {
                Console.WriteLine();
                Console.WriteLine($"=== {Path.GetFileName(file)}");

                if (sweep)
                {
                    Sweep(file, NewImporter, meshFilter);
                    continue;
                }

                var importer = NewImporter();
                var (meshes, errors) = Import(importer, file);
                Console.WriteLine($"ratios {string.Join(", ", importer.LODRatios)}, min {importer.LODMinTriangles} tris, " +
                                  $"max error {importer.LODPixelError:0.#} px{(importer.GenerateLODs ? "" : ", generation off")}");

                foreach (var (name, data) in meshes)
                {
                    if (meshFilter != null && !name.Contains(meshFilter, StringComparison.OrdinalIgnoreCase)) continue;
                    Report(name, data, errors.GetValueOrDefault(name));
                    if (objDir != null) WriteObj(objDir, name, data);
                }
            }
            return 0;
        }

        /// <summary>Import with the importer's own log captured: the generator reports its error there.</summary>
        static (List<(string name, MeshData data)> meshes, Dictionary<string, string> errors) Import(ModelImporter importer, string file)
        {
            Freefall.Debug.ClearLog();
            var console = Console.Out;
            Console.SetOut(TextWriter.Null);
            List<(string, MeshData)> meshes;
            try { meshes = importer.ParseMeshes(file); }
            finally { Console.SetOut(console); }

            var errors = new Dictionary<string, string>();
            const string prefix = "ModelImporter: Generated LODs for '";
            foreach (var line in Freefall.Debug.Lines)
            {
                var message = line.Message ?? string.Empty;
                if (!message.StartsWith(prefix)) continue;
                int end = message.IndexOf("':", prefix.Length, StringComparison.Ordinal);
                int open = message.IndexOf("(error ", StringComparison.Ordinal);
                if (end < 0 || open < 0) continue;
                errors[message.Substring(prefix.Length, end - prefix.Length)] = message.Substring(open + 1).TrimEnd(')');
            }
            return (meshes, errors);
        }

        static void Report(string name, MeshData data, string error)
        {
            var mesh = new Mesh();
            mesh.MeshParts.AddRange(data.Parts);
            mesh.BoundingBox = data.BoundingBox;
            mesh.LODs.AddRange(data.LODs);
            mesh.ComputeNonLodPartIndices();

            var box = data.BoundingBox;
            var size = box.Max - box.Min;
            float radius = Vector3.Distance(box.Max, (box.Min + box.Max) * 0.5f);
            float tanHalfFov = MathF.Tan(VerticalFovDegrees * MathF.PI / 360f);

            int drawn = data.Parts.Count(p => p.Enabled);
            Console.WriteLine($"{name}: {size.X:0.#} x {size.Y:0.#} x {size.Z:0.#} m, {data.Positions.Length} verts, " +
                              $"{data.Parts.Count} parts ({drawn} drawn), {(data.BoneWeights != null ? "skinned, " : "")}" +
                              $"{(data.LODs.Count > 0 ? $"{data.LODs.Count} LODs" : "no LODs")}{(error != null ? $", {error}" : "")}");

            if (ShowParts)
                foreach (var p in data.Parts)
                    Console.WriteLine($"    part '{p.Name}' slot {p.MaterialSlot} {p.NumIndices / 3} tris{(p.Enabled ? "" : " (not drawn)")}");

            if (data.LODs.Count == 0)
            {
                Console.WriteLine($"  {data.Indices.Length / 3} tris");
                return;
            }

            long tris0 = Triangles(mesh, mesh.LODs[0]);
            for (int l = 0; l < mesh.LODs.Count; l++)
            {
                long tris = Triangles(mesh, mesh.LODs[l]);
                string from = l == 0 ? "" : $"   screen size {mesh.LODs[l].ScreenSize:0.000}, from {radius / (tanHalfFov * mesh.LODs[l].ScreenSize):0} m";
                Console.WriteLine($"  LOD{l}  {tris,6} tris  {100.0 * tris / tris0,5:0.0} %{from}");
            }
            Console.WriteLine($"  culled from {radius / (tanHalfFov * Mesh.LODCullScreenSize):0} m");
        }

        static long Triangles(Mesh mesh, MeshLOD lod)
        {
            long indices = 0;
            foreach (var p in lod.MeshPartIndices) indices += mesh.MeshParts[p].NumIndices;
            return indices / 3;
        }

        static void Sweep(string file, Func<float?, ModelImporter> newImporter, string meshFilter)
        {
            float[] limits = { 1f, 2f, 4f, 8f, 0f };
            var rows = new Dictionary<string, List<string>>();
            var order = new List<string>();
            foreach (var limit in limits)
            {
                var (meshes, _) = Import(newImporter(limit), file);
                foreach (var (name, data) in meshes)
                {
                    if (meshFilter != null && !name.Contains(meshFilter, StringComparison.OrdinalIgnoreCase)) continue;
                    if (!rows.TryGetValue(name, out var row)) { rows[name] = row = new List<string>(); order.Add(name); }
                    row.Add(Chain(data));
                }
            }

            int width = Math.Max(4, order.Count == 0 ? 4 : order.Max(n => n.Length));
            int cell = Math.Max(22, rows.Values.SelectMany(r => r).DefaultIfEmpty("").Max(c => c.Length) + 2);
            Console.WriteLine("error limit".PadRight(width + 2) + string.Concat(limits.Select(l => (l > 0 ? $"{l:0.#} px" : "none").PadRight(cell))));
            foreach (var name in order)
                Console.WriteLine(name.PadRight(width + 2) + string.Concat(rows[name].Select(c => c.PadRight(cell))));
        }

        static string Chain(MeshData data)
        {
            if (data.LODs.Count == 0) return $"{data.Indices.Length / 3} (none)";
            return string.Join(" > ", data.LODs.Select(l => l.MeshPartIndices.Sum(p => (long)data.Parts[p].NumIndices) / 3));
        }

        /// <summary>
        /// One OBJ per LOD with only the vertices it uses. The engine's mesh space is left-handed (Y up, Z forward,
        /// clockwise fronts); OBJ is right-handed, so Z is negated and the winding turned to keep the real shape.
        /// </summary>
        static void WriteObj(string dir, string name, MeshData data)
        {
            Directory.CreateDirectory(dir);
            var lods = data.LODs.Count > 0
                ? data.LODs.Select(l => l.MeshPartIndices).ToList()
                : new List<int[]> { Enumerable.Range(0, data.Parts.Count).Where(p => data.Parts[p].Enabled).ToArray() };

            for (int l = 0; l < lods.Count; l++)
            {
                var remap = new Dictionary<uint, int>();
                var faces = new StringBuilder();
                var verts = new StringBuilder();
                foreach (var p in lods[l])
                {
                    var part = data.Parts[p];
                    for (int i = part.BaseIndex; i + 2 < part.BaseIndex + part.NumIndices; i += 3)
                    {
                        faces.Append('f');
                        foreach (int k in new[] { 0, 2, 1 })
                        {
                            uint v = (uint)(data.Indices[i + k] + part.BaseVertex);
                            if (!remap.TryGetValue(v, out int n))
                            {
                                n = remap.Count + 1;
                                remap[v] = n;
                                var pos = data.Positions[v];
                                var nrm = data.Normals[v];
                                var uv = data.UVs[v];
                                verts.Append($"v {pos.X:0.#####} {pos.Y:0.#####} {-pos.Z:0.#####}\n");
                                verts.Append($"vn {nrm.X:0.####} {nrm.Y:0.####} {-nrm.Z:0.####}\n");
                                verts.Append($"vt {uv.X:0.#####} {uv.Y:0.#####}\n");
                            }
                            faces.Append($" {n}/{n}/{n}");
                        }
                        faces.Append('\n');
                    }
                }
                File.WriteAllText(Path.Combine(dir, $"{name}_LOD{l}.obj"), verts.ToString() + faces);
            }
        }
    }
}
