using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;
using Freefall.Assets;
using Freefall.Assets.Packers;
using Freefall.Graphics;

namespace Freefall.Editor.Publishing
{
    /// <summary>One texture of a build: its size, and how sharp it shows where it is used.</summary>
    public sealed class BudgetTexture
    {
        public BuildArtifact Artifact = null!;
        public int Width, Height, Mips;
        public string Format = "";
        /// <summary>The size and the bytes when capped at the budget (the same when it is within it).</summary>
        public int CappedWidth, CappedHeight;
        public long CappedBytes;
        /// <summary>Pixels per metre where it shows coarsest, now and capped at the budget; NaN when no use was found.</summary>
        public double Coarsest = double.NaN, CoarsestAtBudget = double.NaN;
        /// <summary>That use: "Rock_Chunk_1A1 › Rock_Chunk_1A1 part 0 (Albedo, scale 2.5)".</summary>
        public string CoarsestUse = "";
        public int Uses;

        public bool OverBudget(int budget) => budget > 0 && Math.Max(Width, Height) > budget;
    }

    /// <summary>What <see cref="TextureBudgetScan.Measure"/> found.</summary>
    public sealed class TextureBudgetResult
    {
        public int Budget;
        public float Target;
        public List<BudgetTexture> Textures = new();
        public TimeSpan Elapsed;

        public List<BudgetTexture> Over => Textures.Where(t => t.OverBudget(Budget)).OrderByDescending(t => t.Artifact.Size).ToList();
        public bool BelowTarget(BudgetTexture t) => !double.IsNaN(t.CoarsestAtBudget) && t.CoarsestAtBudget < Target;

        public void WriteTo(StringBuilder sb)
        {
            var over = Over;
            long now = over.Sum(t => t.Artifact.Size), capped = over.Sum(t => t.CappedBytes);
            int below = over.Count(BelowTarget), noUse = over.Count(t => double.IsNaN(t.Coarsest));

            sb.AppendLine($"Texture budget: {Budget} px, sharpness target {Target:0} px/m (the profile's TextureBudget, TexelDensityTarget; a guideline, nothing is capped):");
            sb.AppendLine($"  {over.Count:N0} of {Textures.Count:N0} textures are over it: {BuildScan.Size(now)}, about {BuildScan.Size(capped)} capped at {Budget} (saves {BuildScan.Size(now - capped)}).");
            sb.AppendLine($"  {below:N0} of those would show below {Target:0} px/m where they are coarsest: candidates for kit pieces, tiling materials or trim sheets.");
            sb.AppendLine($"  {noUse:N0} are on no mesh or terrain layer the scan could read (UI, particles, stamps, code): no sharpness for them.");
            sb.AppendLine("  Sharpness: texture pixels per metre of surface, where the texture is coarsest (the mesh part with the most surface per UV area, at the largest scale it is");
            sb.AppendLine("  placed at: in a prefab, a scene, or a PCG graph's ScaleMax). Now and capped. A ! marks those below the target.");
            sb.AppendLine($"  (measured in {Elapsed.TotalSeconds:0.0} s)");
            sb.AppendLine();
            foreach (var t in over)
            {
                string density = double.IsNaN(t.Coarsest) ? "      no use found" : $"{t.Coarsest,6:0} → {t.CoarsestAtBudget,5:0} px/m";
                string flag = BelowTarget(t) ? "!" : " ";
                sb.AppendLine($"  {BuildScan.Size(t.Artifact.Size),10} {t.Width,5}x{t.Height,-5} {t.Format,-6} → {BuildScan.Size(t.CappedBytes),9}  {density} {flag} " +
                              $"{t.Artifact.SourcePath}{(t.CoarsestUse.Length > 0 ? "  [" + t.CoarsestUse + "]" : "")}");
            }
            sb.AppendLine();
        }
    }

    /// <summary>
    /// How large a build's textures are against a size budget, and how sharp each would show capped at it. Reads the
    /// artifacts the scan reached, nothing on the GPU:
    /// <list type="bullet">
    /// <item>textures: the DDS header (size, mips, format);</item>
    /// <item>uses: every YAML block with a <c>Mesh:</c> and its materials (MeshRenderer overrides by MaterialSlot,
    /// SkinnedMeshRenderer by list order, a CharacterPiece's Looks on slot 0 and MoreMaterials after), in prefabs, scenes
    /// and assets; terrain layers by their Tiling;</item>
    /// <item>scale: the transform chain in the prefab or scene, times the largest scale the prefab is placed at (a scene's
    /// instances, PCG graphs' ScaleMax for the prefabs a graph names, directly or through a kit);</item>
    /// <item>meshes: per LOD0 part, the surface in m² and its UV area: pixels per metre = √(texture pixels × UV area / surface) ÷ scale.</item>
    /// </list>
    /// A texture's sharpness is its coarsest use. Detail slots (Detail*) tile by the material's DetailTiling.
    /// </summary>
    public static class TextureBudgetScan
    {
        static readonly Regex KeyGuid = new(@"^(\s*)(-\s+)?([A-Za-z_][\w.]*):\s*""?([0-9a-fA-F]{32})""?\s*$", RegexOptions.Compiled);
        static readonly Regex ItemGuid = new(@"^(\s*)-\s+""?([0-9a-fA-F]{32})""?\s*$", RegexOptions.Compiled);
        static readonly Regex KeyOnly = new(@"^(\s*)(-\s+)?([A-Za-z_][\w.]*):\s*$", RegexOptions.Compiled);
        static readonly Regex KeyValue = new(@"^(\s*)(-\s+)?([A-Za-z_][\w.]*):\s*(.+?)\s*$", RegexOptions.Compiled);
        static readonly Regex Guid32 = new(@"\b[0-9a-fA-F]{32}\b", RegexOptions.Compiled);

        sealed class Use
        {
            public string Mesh = "";
            public List<(int Slot, string Material)> Materials = new();
            public BuildArtifact Container = null!;
            public string? Transform;   // UID in the container's YAML
            public string Label = "";
        }

        sealed class TransformInfo { public float Scale = 1; public string? Parent; }

        sealed class MaterialInfo
        {
            public List<(string Slot, string Texture)> Textures = new();
            public float DetailTiling = 1;
        }

        /// <summary>Worker-safe: reads Cache files and the database's lookups (concurrent).</summary>
        public static TextureBudgetResult Measure(BuildScan scan, int budget, float target)
        {
            var watch = Stopwatch.StartNew();
            var result = new TextureBudgetResult { Budget = budget, Target = target };
            var byGuid = scan.Artifacts.GroupBy(a => a.Guid, StringComparer.OrdinalIgnoreCase).ToDictionary(g => g.Key, g => g.First(), StringComparer.OrdinalIgnoreCase);

            // ── Textures: the DDS headers ──
            var textures = new Dictionary<string, BudgetTexture>(StringComparer.OrdinalIgnoreCase);
            foreach (var a in scan.Artifacts.Where(a => a.DataType == nameof(DdsTextureData)))
            {
                var t = ReadHeader(a, budget);
                if (t == null) continue;
                textures[a.Guid] = t;
                result.Textures.Add(t);
            }

            // ── YAML: uses, materials, terrain layers, prefab scales ──
            var uses = new List<Use>();
            var materials = new Dictionary<string, MaterialInfo>(StringComparer.OrdinalIgnoreCase);
            var layers = new List<(string Texture, float Tiling, BuildArtifact Layer, string Slot)>();
            var prefabScale = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
            var transforms = new Dictionary<BuildArtifact, Dictionary<string, TransformInfo>>();
            var graphs = new List<(float Scale, List<string> Refs)>();
            var definitions = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);   // asset GUID → GUIDs it names

            foreach (var a in scan.Artifacts)
            {
                string? yaml = ReadYaml(a, out var typeName);
                if (yaml == null) continue;

                if (a.DataType == nameof(AssetDefinitionData))
                {
                    definitions[a.Guid] = Guid32.Matches(yaml).Select(m => m.Value).ToList();
                    if (typeName == "Material") { materials[a.Guid] = ReadMaterial(yaml, textures); continue; }
                    if (typeName == "TerrainLayer") { ReadLayer(yaml, a, textures, layers); continue; }
                    if (yaml.Contains("ScaleMax:"))
                        graphs.Add((MaxScaleMax(yaml), definitions[a.Guid]));
                }

                var local = new Dictionary<string, TransformInfo>();
                transforms[a] = local;
                ReadUses(yaml, a, uses, local, prefabScale);
            }

            // A graph places what it names (and what a kit it names lists) up to its largest ScaleMax
            foreach (var (scale, refs) in graphs)
            {
                foreach (var g in refs.Distinct(StringComparer.OrdinalIgnoreCase))
                {
                    var named = definitions.TryGetValue(g, out var kit) ? kit.Append(g) : new[] { g };
                    foreach (var n in named)
                        if (byGuid.TryGetValue(n, out var hit) && hit.DataType == nameof(PrefabData))
                            prefabScale[n] = Math.Max(prefabScale.GetValueOrDefault(n, 1f), scale);
                }
            }

            // ── Meshes: surface and UV area per LOD0 part ──
            var meshPacker = new MeshPacker();
            var areas = new Dictionary<string, (int Slot, double Uv, double Surface, string Name)[]>(StringComparer.OrdinalIgnoreCase);
            foreach (var mesh in uses.Select(u => u.Mesh).Distinct(StringComparer.OrdinalIgnoreCase))
            {
                if (!byGuid.TryGetValue(mesh, out var art) || art.DataType != nameof(MeshData) || !File.Exists(art.CachePath)) continue;
                try
                {
                    using var stream = File.OpenRead(art.CachePath);
                    areas[mesh] = PartAreas(meshPacker.Read(stream));
                }
                catch (Exception) { }   // unreadable: its uses get no sharpness
            }

            // ── Densities ──
            void Consider(string textureGuid, double pixelsPerMetreAtOne, string label)
            {
                if (!textures.TryGetValue(textureGuid, out var t) || double.IsNaN(pixelsPerMetreAtOne) || double.IsInfinity(pixelsPerMetreAtOne)) return;
                t.Uses++;
                double now = Math.Sqrt((double)t.Width * t.Height) * pixelsPerMetreAtOne;
                if (!double.IsNaN(t.Coarsest) && now >= t.Coarsest) return;
                t.Coarsest = now;
                t.CoarsestAtBudget = Math.Sqrt((double)t.CappedWidth * t.CappedHeight) * pixelsPerMetreAtOne;
                t.CoarsestUse = label;
            }

            foreach (var use in uses)
            {
                if (!areas.TryGetValue(use.Mesh, out var parts)) continue;
                float scale = WorldScale(transforms[use.Container], use.Transform);
                if (use.Container.DataType == nameof(PrefabData))
                    scale *= prefabScale.GetValueOrDefault(use.Container.Guid, 1f);

                foreach (var (slot, material) in use.Materials)
                {
                    if (!materials.TryGetValue(material, out var info)) continue;
                    foreach (var part in parts.Where(p => p.Slot == slot && p.Surface > 1e-8 && p.Uv > 1e-12))
                    {
                        // Pixels per metre per pixel of texture side: √(UV area / surface), smaller by the scale
                        double perSide = Math.Sqrt(part.Uv / part.Surface) / scale;
                        foreach (var (slotName, texture) in info.Textures)
                        {
                            double tiling = slotName.StartsWith("Detail", StringComparison.Ordinal) ? info.DetailTiling : 1;
                            Consider(texture, perSide * tiling, $"{use.Label} › {part.Name} ({slotName}, scale {scale:0.##})");
                        }
                    }
                }
            }
            foreach (var (texture, tiling, layer, slot) in layers)
                Consider(texture, 1.0 / tiling, $"{layer.Name} ({slot}, {tiling:0.##} m a repeat)");

            result.Elapsed = watch.Elapsed;
            return result;
        }

        // ── Textures ──

        static BudgetTexture? ReadHeader(BuildArtifact a, int budget)
        {
            if (!File.Exists(a.CachePath)) return null;
            var head = new byte[256];
            int n;
            using (var fs = File.OpenRead(a.CachePath))
                n = fs.Read(head, 0, head.Length);

            int i = 0;
            while (i + 148 <= n && !(head[i] == 'D' && head[i + 1] == 'D' && head[i + 2] == 'S' && head[i + 3] == ' ')) i++;
            if (i + 148 > n) return null;

            int height = BitConverter.ToInt32(head, i + 12), width = BitConverter.ToInt32(head, i + 16);
            int mips = Math.Max(1, BitConverter.ToInt32(head, i + 28));
            string fourCC = Encoding.ASCII.GetString(head, i + 84, 4);
            int rgbBits = BitConverter.ToInt32(head, i + 88);
            var (format, bytesPerPixel) = fourCC == "DX10" ? Dxgi(BitConverter.ToInt32(head, i + 128)) : FourCC(fourCC, rgbBits);

            var t = new BudgetTexture { Artifact = a, Width = width, Height = height, Mips = mips, Format = format };
            (t.CappedWidth, t.CappedHeight) = (width, height);
            t.CappedBytes = a.Size;
            Cap(t, budget, bytesPerPixel);
            return t;
        }

        /// <summary>Caps the textures over <paramref name="budget"/> (as the importer's MaxSize would) and estimates their bytes.</summary>
        static void Cap(BudgetTexture t, int budget, double bytesPerPixel)
        {
            if (budget <= 0 || Math.Max(t.Width, t.Height) <= budget) return;
            (t.CappedWidth, t.CappedHeight) = Freefall.Assets.Importers.TextureImporter.ScaledSize(t.Width, t.Height, budget);
            double mips = t.Mips > 1 ? 4.0 / 3.0 : 1.0;
            t.CappedBytes = (long)(t.CappedWidth * (double)t.CappedHeight * bytesPerPixel * mips) + 148;
        }

        static (string Name, double BytesPerPixel) Dxgi(int format) => format switch
        {
            >= 70 and <= 72 => ("BC1", 0.5),
            >= 73 and <= 75 => ("BC2", 1),
            >= 76 and <= 78 => ("BC3", 1),
            >= 79 and <= 81 => ("BC4", 0.5),
            >= 82 and <= 84 => ("BC5", 1),
            >= 94 and <= 96 => ("BC6H", 1),
            >= 97 and <= 99 => ("BC7", 1),
            27 or 28 or 29 or 87 or 91 => ("RGBA8", 4),
            10 => ("RGBA16F", 8),
            2 => ("RGBA32F", 16),
            41 => ("R32F", 4),
            54 or 56 => ("R16", 2),
            49 => ("RG8", 2),
            61 => ("R8", 1),
            _ => ($"dxgi {format}", 1),
        };

        static (string Name, double BytesPerPixel) FourCC(string fourCC, int rgbBits) => fourCC switch
        {
            "DXT1" => ("BC1", 0.5),
            "DXT3" or "DXT2" => ("BC2", 1),
            "DXT5" or "DXT4" => ("BC3", 1),
            "ATI1" or "BC4U" or "BC4S" => ("BC4", 0.5),
            "ATI2" or "BC5U" or "BC5S" => ("BC5", 1),
            _ when rgbBits > 0 => ($"{rgbBits}bit", rgbBits / 8.0),
            _ => (fourCC.Trim('\0', ' '), 1),
        };

        // ── YAML ──

        static string? ReadYaml(BuildArtifact a, out string typeName)
        {
            typeName = "";
            if (!File.Exists(a.CachePath)) return null;
            try
            {
                using var stream = File.OpenRead(a.CachePath);
                switch (a.DataType)
                {
                    case nameof(PrefabData): return Encoding.UTF8.GetString(new PrefabPacker().Read(stream).Yaml);
                    case nameof(SceneData): return Encoding.UTF8.GetString(new ScenePacker().Read(stream).Yaml);
                    case nameof(AssetDefinitionData):
                        var data = new AssetDefinitionPacker().Read(stream);
                        typeName = data.TypeName ?? "";
                        return data.YamlBytes != null ? Encoding.UTF8.GetString(data.YamlBytes) : null;
                    default: return null;
                }
            }
            catch (Exception) { return null; }   // an artifact of an older packer version: left out, never a failed scan
        }

        static MaterialInfo ReadMaterial(string yaml, Dictionary<string, BudgetTexture> textures)
        {
            var info = new MaterialInfo();
            foreach (var line in yaml.Split('\n'))
            {
                var m = KeyGuid.Match(line.TrimEnd('\r'));
                if (m.Success && textures.ContainsKey(m.Groups[4].Value))
                    info.Textures.Add((m.Groups[3].Value, m.Groups[4].Value));
                else if (KeyValue.Match(line.TrimEnd('\r')) is { Success: true } kv && kv.Groups[3].Value == "DetailTiling")
                    info.DetailTiling = Math.Max(1e-3f, MaxComponent(kv.Groups[4].Value, 1f));
            }
            return info;
        }

        static void ReadLayer(string yaml, BuildArtifact layer, Dictionary<string, BudgetTexture> textures, List<(string, float, BuildArtifact, string)> layers)
        {
            float tiling = 1;
            var found = new List<(string Slot, string Texture)>();
            foreach (var raw in yaml.Split('\n'))
            {
                var line = raw.TrimEnd('\r');
                var m = KeyGuid.Match(line);
                if (m.Success && textures.ContainsKey(m.Groups[4].Value)) found.Add((m.Groups[3].Value, m.Groups[4].Value));
                else if (KeyValue.Match(line) is { Success: true } kv && kv.Groups[3].Value == "Tiling")
                    tiling = Math.Max(1e-3f, MaxComponent(kv.Groups[4].Value, 1f));
            }
            foreach (var (slot, texture) in found) layers.Add((texture, tiling, layer, slot));
        }

        /// <summary>The largest component of any ScaleMax in a graph's YAML.</summary>
        static float MaxScaleMax(string yaml)
        {
            float max = 1;
            foreach (var raw in yaml.Split('\n'))
                if (KeyValue.Match(raw.TrimEnd('\r')) is { Success: true } kv && kv.Groups[3].Value == "ScaleMax")
                    max = Math.Max(max, MaxComponent(kv.Groups[4].Value, 1f));
            return max;
        }

        /// <summary>
        /// The mesh uses in a prefab, scene or asset (a <c>Mesh:</c> and the material lists beside it), the transforms
        /// (UID, Parent, Scale) and the scale every prefab instance has.
        /// </summary>
        static void ReadUses(string yaml, BuildArtifact container, List<Use> uses, Dictionary<string, TransformInfo> transforms,
                             Dictionary<string, float> prefabScale)
        {
            string docType = "";
            string? entityUid = null, currentTransform = null, pendingPrefab = null;
            Use? use = null;
            int useIndent = -1;
            string list = "";       // Materials, Looks, MoreMaterials
            int listIndex = 0;
            int pendingSlot = -1;   // a MaterialOverride's MaterialSlot until its Material line
            TransformInfo? transform = null;
            string? transformUid = null;
            var instances = new List<(string Prefab, string Transform)>();
            var entityTransform = new Dictionary<string, string>();

            void Flush()
            {
                if (use != null && use.Materials.Count > 0) uses.Add(use);
                use = null;
                useIndent = -1;
                list = "";
            }

            foreach (var raw in yaml.Split('\n'))
            {
                var line = raw.TrimEnd('\r').TrimStart('\uFEFF');
                if (line.StartsWith("---", StringComparison.Ordinal))
                {
                    Flush();
                    docType = "";
                    continue;
                }
                if (line.Trim().Length == 0) continue;

                int indent = line.Length - line.TrimStart().Length;
                if (indent == 0 && docType.Length == 0)
                {
                    docType = line.Trim().TrimEnd(':', ' ');
                    if (docType == "Entity") { entityUid = null; currentTransform = null; pendingPrefab = null; }
                    if (docType == "Transform") { transform = new TransformInfo(); transformUid = null; }
                    continue;
                }

                // Entities and transforms (prefabs, scenes)
                if (docType == "Entity" || docType == "Transform")
                {
                    var kv = KeyValue.Match(line);
                    if (!kv.Success) continue;
                    string key = kv.Groups[3].Value, value = kv.Groups[4].Value.Trim('"');
                    if (docType == "Entity")
                    {
                        if (key == "UID") entityUid = value;
                        else if (key == "Prefab" && Guid32.IsMatch(value)) pendingPrefab = value;
                    }
                    else
                    {
                        if (key == "UID")
                        {
                            transformUid = value;
                            transforms[value] = transform!;
                            currentTransform = value;
                            if (entityUid != null) entityTransform[entityUid] = value;
                            if (pendingPrefab != null) { instances.Add((pendingPrefab, value)); pendingPrefab = null; }
                        }
                        else if (key == "Parent") transform!.Parent = value;
                        else if (key == "Scale") transform!.Scale = MaxComponent(value, 1f);
                    }
                    continue;
                }

                // A use ends at a line left of its Mesh key
                if (use != null && indent < useIndent) Flush();

                var keyGuid = KeyGuid.Match(line);
                if (keyGuid.Success && keyGuid.Groups[3].Value == "Mesh")
                {
                    Flush();
                    use = new Use
                    {
                        Mesh = keyGuid.Groups[4].Value, Container = container, Transform = currentTransform,
                        Label = container.Name,
                    };
                    useIndent = keyGuid.Groups[1].Value.Length + keyGuid.Groups[2].Value.Length;
                    continue;
                }
                if (use == null) continue;

                var keyOnly = KeyOnly.Match(line);
                if (keyOnly.Success && indent == useIndent && keyOnly.Groups[2].Value.Length == 0)
                {
                    list = keyOnly.Groups[3].Value is "Materials" or "Looks" or "MoreMaterials" ? keyOnly.Groups[3].Value : "";
                    listIndex = 0;
                    continue;
                }
                if (list.Length == 0) continue;

                int SlotFor() => list switch { "Looks" => 0, "MoreMaterials" => 1 + listIndex, _ => listIndex };

                var item = ItemGuid.Match(line);
                if (item.Success && indent == useIndent)
                {
                    use.Materials.Add((SlotFor(), item.Groups[2].Value));
                    listIndex++;
                    continue;
                }
                if (keyOnly.Success && keyOnly.Groups[2].Value.Length > 0 && indent == useIndent)
                {
                    // "- Freefall.Components.MaterialOverride:": its MaterialSlot and Material follow, further in
                    pendingSlot = SlotFor();
                    listIndex++;
                    continue;
                }
                var kv2 = KeyValue.Match(line);
                if (kv2.Success && indent > useIndent)
                {
                    if (kv2.Groups[3].Value == "MaterialSlot" && int.TryParse(kv2.Groups[4].Value, out int slot)) pendingSlot = slot;
                    else if (kv2.Groups[3].Value == "Material" && Guid32.IsMatch(kv2.Groups[4].Value))
                        use.Materials.Add((Math.Max(0, pendingSlot), kv2.Groups[4].Value.Trim('"')));
                }
            }
            Flush();

            // A transform's Parent names the parent's transform or its entity
            foreach (var t in transforms.Values)
                if (t.Parent != null && entityTransform.TryGetValue(t.Parent, out var parentTransform))
                    t.Parent = parentTransform;

            foreach (var (prefab, transformUidOf) in instances)
                prefabScale[prefab] = Math.Max(prefabScale.GetValueOrDefault(prefab, 0f), WorldScale(transforms, transformUidOf));
        }

        static float WorldScale(Dictionary<string, TransformInfo> transforms, string? uid)
        {
            float scale = 1;
            for (int depth = 0; uid != null && depth < 32 && transforms.TryGetValue(uid, out var t); depth++)
            {
                scale *= t.Scale;
                uid = t.Parent;
            }
            return scale;
        }

        /// <summary>The largest absolute number in "[x, y, z]" or "x".</summary>
        static float MaxComponent(string value, float fallback)
        {
            float max = float.NaN;
            foreach (var part in value.Trim('[', ']', ' ').Split(','))
                if (float.TryParse(part.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                    max = float.IsNaN(max) ? Math.Abs(v) : Math.Max(max, Math.Abs(v));
            return float.IsNaN(max) || max <= 0 ? fallback : max;
        }

        // ── Meshes ──

        static (int Slot, double Uv, double Surface, string Name)[] PartAreas(MeshData mesh)
        {
            var indices = mesh.LODs.Count > 0 && mesh.LODs[0].MeshPartIndices is { Length: > 0 } lod0
                ? lod0
                : Enumerable.Range(0, mesh.Parts.Count).ToArray();
            var result = new List<(int, double, double, string)>();
            foreach (int pi in indices)
            {
                if (pi < 0 || pi >= mesh.Parts.Count) continue;
                var part = mesh.Parts[pi];
                double uv = 0, surface = 0;
                if (mesh.UVs != null && mesh.Positions != null && mesh.Indices != null)
                {
                    for (int k = part.BaseIndex; k + 2 < part.BaseIndex + part.NumIndices && k + 2 < mesh.Indices.Length; k += 3)
                    {
                        long i0 = part.BaseVertex + mesh.Indices[k], i1 = part.BaseVertex + mesh.Indices[k + 1], i2 = part.BaseVertex + mesh.Indices[k + 2];
                        if (i0 >= mesh.Positions.Length || i1 >= mesh.Positions.Length || i2 >= mesh.Positions.Length) continue;
                        if (i0 >= mesh.UVs.Length || i1 >= mesh.UVs.Length || i2 >= mesh.UVs.Length) continue;
                        var p0 = mesh.Positions[i0];
                        surface += Vector3.Cross(mesh.Positions[i1] - p0, mesh.Positions[i2] - p0).Length() * 0.5;
                        var u0 = mesh.UVs[i0];
                        Vector2 e1 = mesh.UVs[i1] - u0, e2 = mesh.UVs[i2] - u0;
                        uv += Math.Abs(e1.X * e2.Y - e1.Y * e2.X) * 0.5;
                    }
                }
                result.Add((part.MaterialSlot, uv, surface, string.IsNullOrEmpty(part.Name) ? $"part {pi}" : part.Name));
            }
            return result.ToArray();
        }
    }
}
