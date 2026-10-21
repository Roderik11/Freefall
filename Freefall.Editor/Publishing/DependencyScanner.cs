using System.Diagnostics;
using System.Text;
using Freefall.Assets;

namespace Freefall.Editor.Publishing
{
    /// <summary>One artifact a build packs (one Cache file), and how the scan reached it.</summary>
    public sealed class BuildArtifact
    {
        /// <summary>The artifact's own GUID: its Cache file's name (a compound source's sub-asset, not the source).</summary>
        public string Guid = "";
        /// <summary>The packer's type name (SceneData, PrefabData, MeshData, DdsTextureData, ...).</summary>
        public string DataType = "";
        public string CachePath = "";
        public long Size;
        public string Name = "";
        /// <summary>The source file, relative to Assets; empty for a cache file no meta lists (a terrain's baked maps).</summary>
        public string SourcePath = "";
        /// <summary>The artifact that reached this one first; null for a root.</summary>
        public BuildArtifact? From;
        /// <summary>Why a root is one ("scene 1", "AlwaysPublish"); null for the others.</summary>
        public string? Root;

        /// <summary>The root this one was reached from, through <see cref="From"/>.</summary>
        public BuildArtifact Origin
        {
            get
            {
                var a = this;
                while (a.From != null) a = a.From;
                return a;
            }
        }

        /// <summary>Names from the root down to this artifact: "Eldermoor › BeechForest › Beech_A".</summary>
        public string Chain
        {
            get
            {
                var names = new List<string>();
                for (var a = this; a != null; a = a.From) names.Add(a.Name);
                names.Reverse();
                return string.Join(" › ", names);
            }
        }
    }

    /// <summary>What a scan found: the artifacts a build would pack, in the order they were reached, and the warnings.</summary>
    public sealed class BuildScan
    {
        public BuildProfile Profile = null!;
        public List<BuildArtifact> Artifacts = new();
        public List<string> Warnings = new();
        /// <summary>Set by the compareLoaded check (DependencyScanner.LoadedNotReached): written into the report.</summary>
        public List<(string Guid, string Name, string Type, string SourcePath)>? LoadedNotReached;
        public TimeSpan Elapsed;
        public DateTime Time = DateTime.Now;

        public long TotalBytes => Artifacts.Sum(a => a.Size);

        /// <summary>The roots that are scenes, in the profile's order, then the scenes reached by reference.</summary>
        public IEnumerable<BuildArtifact> Scenes => Artifacts.Where(a => a.DataType == nameof(SceneData));

        public static string Size(long bytes) =>
            bytes >= 1L << 30 ? $"{bytes / (double)(1L << 30):0.00} GB"
            : bytes >= 1L << 20 ? $"{bytes / (double)(1L << 20):0.0} MB"
            : $"{bytes / 1024.0:0.0} KB";

        /// <summary>Totals per data type, largest first.</summary>
        public List<(string Type, int Count, long Bytes)> ByType() => Artifacts
            .GroupBy(a => a.DataType)
            .Select(g => (g.Key, g.Count(), g.Sum(a => a.Size)))
            .OrderByDescending(t => t.Item3).ToList();

        /// <summary>Totals per folder (the first two levels under Assets), largest first.</summary>
        public List<(string Folder, int Count, long Bytes)> ByFolder() => Artifacts
            .GroupBy(a => FolderOf(a.SourcePath))
            .Select(g => (g.Key, g.Count(), g.Sum(a => a.Size)))
            .OrderByDescending(t => t.Item3).ToList();

        /// <summary>
        /// Totals per root, in root order: what each scene adds that an earlier one did not reach, and what the
        /// AlwaysPublish folders add, together.
        /// </summary>
        public List<(string Root, int Count, long Bytes)> ByRoot() => Artifacts
            .GroupBy(a => a.Origin.Root == DependencyScanner.AlwaysPublishFolder
                ? DependencyScanner.AlwaysPublishFolder + " folders" : $"{a.Origin.Name} ({a.Origin.Root})")
            .Select(g => (g.Key, g.Count(), g.Sum(a => a.Size))).ToList();

        public static string FolderOf(string sourcePath)
        {
            if (string.IsNullOrEmpty(sourcePath)) return "(Cache only)";
            var parts = sourcePath.Split('\\', '/');
            return parts.Length <= 2 ? string.Join("\\", parts.Take(parts.Length - 1).DefaultIfEmpty("(Assets)"))
                : parts[0] + "\\" + parts[1];
        }

        /// <summary>Write the build report: totals first, then every artifact with the chain that reached it.</summary>
        public void WriteReport(string path)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Build report: {Profile.ResolvedProductName} (profile {Profile.Name}), {Time:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine("Scan only: nothing was published.");
            sb.AppendLine();

            sb.AppendLine("Scenes (the game starts in the first):");
            int n = 1;
            foreach (var scene in Scenes)
                sb.AppendLine(scene.Root != null
                    ? $"  {n++}. {scene.Name}  ({scene.SourcePath}, {scene.Guid})"
                    : $"  +  {scene.Name}  ({scene.SourcePath}): not in the list, reached by {scene.Chain}");
            sb.AppendLine();

            sb.AppendLine($"Total: {Artifacts.Count:N0} assets, {Size(TotalBytes)} (scan {Elapsed.TotalSeconds:0.0} s)");
            sb.AppendLine();

            sb.AppendLine("By root (what each adds that an earlier one did not reach):");
            foreach (var (root, count, bytes) in ByRoot())
                sb.AppendLine($"  {Size(bytes),10}  {count,7:N0}  {root}");
            sb.AppendLine();

            sb.AppendLine("By type:");
            foreach (var (type, count, bytes) in ByType())
                sb.AppendLine($"  {Size(bytes),10}  {count,7:N0}  {type}");
            sb.AppendLine();

            sb.AppendLine("Largest folders:");
            foreach (var (folder, count, bytes) in ByFolder().Take(25))
                sb.AppendLine($"  {Size(bytes),10}  {count,7:N0}  {folder}");
            sb.AppendLine();

            sb.AppendLine($"Warnings ({Warnings.Count}):");
            foreach (var w in Warnings)
                sb.AppendLine("  " + w);
            if (Warnings.Count == 0) sb.AppendLine("  none");
            sb.AppendLine();

            if (LoadedNotReached != null)
            {
                sb.AppendLine($"Loaded in this editor but not reached by the scan ({LoadedNotReached.Count}): code that loads by name or by a GUID written in it, or the editor's own use:");
                foreach (var (folder, count) in LoadedNotReached.GroupBy(m => FolderOf(m.SourcePath)).Select(g => (g.Key, g.Count())))
                    sb.AppendLine($"  {count,7:N0}  {folder}");
                foreach (var m in LoadedNotReached)
                    sb.AppendLine($"  {m.Type,-14}  {m.Guid}  {m.SourcePath}  ({m.Name})");
                sb.AppendLine();
            }

            sb.AppendLine("Largest assets:");
            foreach (var a in Artifacts.OrderByDescending(a => a.Size).Take(40))
                sb.AppendLine($"  {Size(a.Size),10}  {a.DataType,-20}  {a.Chain}");
            sb.AppendLine();

            sb.AppendLine("All assets, in the order the scan reached them (size, type, GUID, source, chain from the root):");
            foreach (var a in Artifacts)
                sb.AppendLine($"  {Size(a.Size),10}  {a.DataType,-20}  {a.Guid}  {(a.SourcePath.Length > 0 ? a.SourcePath : a.CachePath)}  ({a.Chain})");

            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, sb.ToString(), new UTF8Encoding(false));
        }
    }

    /// <summary>
    /// Finds what a build packs: everything the profile's scenes reach by reference, all the way down, plus every
    /// asset under a folder named <see cref="AlwaysPublishFolder"/>. The engine loads assets only through
    /// references, so what this does not reach is not needed. The unit is the artifact (a Cache file): one clip of a
    /// model does not bring the model's mesh. Edges:
    /// <list type="bullet">
    /// <item>An artifact whose data names assets (scene, prefab, .asset: AssetPackerAttribute.HoldsReferences) is
    /// read as text; every 32-digit hex token the asset database knows is a reference. Its Cache artifact is read,
    /// not the source: a model's prefabs and materials exist only there. A text scan, not the deserializer: that
    /// would create entities, needs the scripts loaded, and drops references held in abstract fields.</item>
    /// <item>What the artifact's loader reads besides it (IAssetLoader.GetImplicitDependencies): a skeleton, a
    /// navmesh's tiles, a terrain's baked maps.</item>
    /// </list>
    /// </summary>
    public static class DependencyScanner
    {
        /// <summary>Assets under a folder of this name (any depth, any case) are always packed.</summary>
        public const string AlwaysPublishFolder = "AlwaysPublish";

        const int MaxWarningsPerKind = 100;

        public static BuildScan Scan(BuildProfile profile)
        {
            var watch = Stopwatch.StartNew();
            var scan = new BuildScan { Profile = profile };
            var reached = new Dictionary<string, BuildArtifact>(StringComparer.OrdinalIgnoreCase);
            var queue = new Queue<BuildArtifact>();
            var unknown = new Dictionary<string, (BuildArtifact First, int Count)>(StringComparer.OrdinalIgnoreCase);
            var engineAssets = new Dictionary<string, bool>(StringComparer.OrdinalIgnoreCase);   // FindByGuid walks every loaded asset
            int notImported = 0;

            void Reach(string guid, BuildArtifact? from, string? root, string? cachePath = null, string? dataType = null)
            {
                BuildArtifact artifact;
                if (cachePath != null)
                {
                    // A cache file no meta lists (a loader named it with its path)
                    if (reached.ContainsKey(guid)) return;
                    artifact = new BuildArtifact { Guid = guid, CachePath = cachePath, DataType = dataType ?? "", Name = from?.Name + " " + Path.GetFileName(cachePath) };
                }
                else
                {
                    if (AssetDatabase.ResolveArtifact(guid) is not { } resolved) return;
                    if (reached.ContainsKey(resolved.Guid)) return;
                    artifact = new BuildArtifact
                    {
                        Guid = resolved.Guid,
                        DataType = resolved.DataType,
                        CachePath = resolved.CachePath,
                        Name = AssetDatabase.ResolveFriendlyName(resolved.Guid),
                        SourcePath = AssetDatabase.GetMeta(resolved.Guid)?.SourcePath ?? "",
                    };
                }

                artifact.From = from;
                artifact.Root = root;
                var file = new FileInfo(artifact.CachePath);
                if (file.Exists)
                    artifact.Size = file.Length;
                else if (notImported++ < MaxWarningsPerKind)
                    scan.Warnings.Add($"Not imported (no Cache file): {artifact.Name} ({artifact.SourcePath}), reached by {artifact.Chain}");

                reached.Add(artifact.Guid, artifact);
                scan.Artifacts.Add(artifact);
                queue.Enqueue(artifact);
            }

            // Roots: the scenes in order, then what lies under an AlwaysPublish folder
            for (int i = 0; i < profile.Scenes.Count; i++)
            {
                var scene = profile.Scenes[i];
                if (scene == null || string.IsNullOrEmpty(scene.Guid) || AssetDatabase.ResolveArtifact(scene.Guid) == null)
                    scan.Warnings.Add($"Scene {i + 1} of the profile is empty or not an asset");
                else
                    Reach(scene.Guid, null, $"scene {i + 1}");
            }
            if (profile.Scenes.Count == 0)
                scan.Warnings.Add("The profile has no scenes");

            foreach (var meta in AssetDatabase.GetAllMeta().Where(m => IsAlwaysPublished(m.SourcePath)).OrderBy(m => m.SourcePath, StringComparer.OrdinalIgnoreCase))
            {
                if (meta.MainAssetType != null)
                    Reach(meta.Guid, null, AlwaysPublishFolder);
                else
                    foreach (var sub in meta.SubAssets.Where(s => !s.Hidden))
                        Reach(sub.Guid, null, AlwaysPublishFolder);
            }

            // Walk: references in the data, then what the loader reads besides it
            while (queue.Count > 0)
            {
                var artifact = queue.Dequeue();
                var claimed = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

                var assetType = artifact.SourcePath.Length > 0 ? AssetDatabase.GetAssetType(artifact.Guid) : null;
                var loader = assetType != null ? AssetManager.FindLoader(assetType) : null;
                if (loader != null)
                {
                    foreach (var dep in loader.GetImplicitDependencies(artifact.Guid))
                    {
                        claimed.Add(dep.Guid);
                        Reach(dep.Guid, artifact, null, dep.CachePath, dep.DataType);
                    }
                }

                if (!AssetDatabase.HoldsReferences(artifact.DataType) || artifact.Size == 0) continue;

                var sourceGuid = AssetDatabase.GetMeta(artifact.Guid)?.Guid;
                foreach (var token in GuidTokens(File.ReadAllBytes(artifact.CachePath)))
                {
                    if (token.Equals(artifact.Guid, StringComparison.OrdinalIgnoreCase) ||
                        token.Equals(sourceGuid, StringComparison.OrdinalIgnoreCase) || claimed.Contains(token))
                        continue;

                    if (AssetDatabase.GetMeta(token) != null)
                        Reach(token, artifact, null);
                    else if (!IsEngineAsset(token))
                        unknown[token] = unknown.TryGetValue(token, out var u) ? (u.First, u.Count + 1) : (artifact, 1);
                }
            }

            if (notImported > MaxWarningsPerKind)
                scan.Warnings.Add($"... and {notImported - MaxWarningsPerKind} more not imported");

            foreach (var (token, (first, count)) in unknown.Take(MaxWarningsPerKind))
                scan.Warnings.Add($"Unknown GUID {token} (a deleted asset?) named by {first.Chain} ({first.SourcePath})" +
                                  (count > 1 ? $" and {count - 1} more" : ""));
            if (unknown.Count > MaxWarningsPerKind)
                scan.Warnings.Add($"... and {unknown.Count - MaxWarningsPerKind} more unknown GUIDs");

            scan.Elapsed = watch.Elapsed;
            return scan;

            // An engine asset (InternalAssets: built-in effects, materials, meshes) is registered with the manager, not the database
            bool IsEngineAsset(string token)
            {
                if (!engineAssets.TryGetValue(token, out var known))
                    engineAssets[token] = known = Engine.Assets?.FindByGuid(token) != null;
                return known;
            }
        }

        /// <summary>
        /// Assets the editor has loaded that the scan did not reach (<see cref="AssetManager.LoadedGuids"/>): with the
        /// profile's scenes open and played, what code loads by name or by a GUID written in it. The editor loads
        /// some things for itself too (a selected asset, an open graph), so read the list, don't fail on it.
        /// </summary>
        public static List<(string Guid, string Name, string Type, string SourcePath)> LoadedNotReached(BuildScan scan)
        {
            var reached = new HashSet<string>(scan.Artifacts.Select(a => a.Guid), StringComparer.OrdinalIgnoreCase);
            var missed = new List<(string, string, string, string)>();
            foreach (var guid in Engine.Assets.LoadedGuids().Distinct(StringComparer.OrdinalIgnoreCase))
            {
                if (AssetDatabase.ResolveArtifact(guid) is not { } artifact || reached.Contains(artifact.Guid)) continue;
                var type = AssetDatabase.GetAssetType(guid);
                if (type == typeof(BuildProfile)) continue;
                missed.Add((artifact.Guid, AssetDatabase.ResolveFriendlyName(artifact.Guid), type?.Name ?? artifact.DataType,
                            AssetDatabase.GetMeta(artifact.Guid)?.SourcePath ?? ""));
            }
            missed.Sort((a, b) => string.Compare(a.Item4, b.Item4, StringComparison.OrdinalIgnoreCase));
            return missed;
        }

        /// <summary>Whether a source path (relative to Assets) lies under a folder named AlwaysPublish.</summary>
        public static bool IsAlwaysPublished(string sourcePath)
        {
            if (string.IsNullOrEmpty(sourcePath)) return false;
            var parts = sourcePath.Split('\\', '/');
            for (int i = 0; i < parts.Length - 1; i++)
                if (parts[i].Equals(AlwaysPublishFolder, StringComparison.OrdinalIgnoreCase))
                    return true;
            return false;
        }

        /// <summary>Every run of exactly 32 hex digits in the bytes (quoted or not), lower-cased, each once.</summary>
        static IEnumerable<string> GuidTokens(byte[] bytes)
        {
            var seen = new HashSet<string>();
            int start = -1;
            for (int i = 0; i <= bytes.Length; i++)
            {
                bool hex = i < bytes.Length && IsHex(bytes[i]);
                if (hex)
                {
                    if (start < 0) start = i;
                    continue;
                }
                if (start >= 0 && i - start == 32)
                {
                    var token = Encoding.ASCII.GetString(bytes, start, 32).ToLowerInvariant();
                    if (seen.Add(token)) yield return token;
                }
                start = -1;
            }
        }

        static bool IsHex(byte b) => (b >= '0' && b <= '9') || (b >= 'a' && b <= 'f') || (b >= 'A' && b <= 'F');
    }
}
