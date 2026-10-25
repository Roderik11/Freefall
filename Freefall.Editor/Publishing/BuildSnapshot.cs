using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Freefall.Assets;
using Freefall.Assets.Packs;

namespace Freefall.Editor.Publishing
{
    /// <summary>
    /// What a build pushed to itch.io held (<see cref="ItchUpload"/>), kept per profile and channel in Library/Itch:
    /// every packed artifact with its name, type, source file and content hash, and the project's scripts with theirs.
    /// The next push compares its own against it (<see cref="Compare"/>), so the change list says what testers get since
    /// they last updated, the project's scripts included, without the game's project under version control. The list
    /// is raw material for the patch notes, which are written for players in the project's Patch Notes asset.
    /// </summary>
    public sealed class BuildSnapshot
    {
        public sealed class Item
        {
            public string Name { get; set; } = "";
            public string Type { get; set; } = "";
            public string Source { get; set; } = "";
            public string Hash { get; set; } = "";
        }

        public string Version { get; set; } = "";
        public DateTime Taken { get; set; }
        /// <summary>GUID → the artifact.</summary>
        public Dictionary<string, Item> Assets { get; set; } = new(StringComparer.OrdinalIgnoreCase);
        /// <summary>Script path relative to Assets → hash.</summary>
        public Dictionary<string, string> Scripts { get; set; } = new(StringComparer.OrdinalIgnoreCase);

        static readonly JsonSerializerOptions Json = new() { WriteIndented = true };

        /// <summary>Where the snapshot of the last push of <paramref name="profile"/> to its channel is kept.</summary>
        public static string PathFor(BuildProfile profile, string libraryFolder)
        {
            var channel = string.IsNullOrWhiteSpace(profile.ItchChannel) ? "windows" : profile.ItchChannel.Trim();
            var name = string.Concat($"{profile.Name}.{channel}".Split(Path.GetInvalidFileNameChars()));
            return Path.Combine(libraryFolder, "Itch", name + ".json");
        }

        public static BuildSnapshot? Load(string path)
        {
            if (!File.Exists(path)) return null;
            try { return JsonSerializer.Deserialize<BuildSnapshot>(File.ReadAllText(path), Json); }
            catch (JsonException) { return null; }
        }

        public void Save(string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, JsonSerializer.Serialize(this, Json));
        }

        /// <summary>Worker: what the scan's build will hold, its artifacts hashed from their Cache files.</summary>
        public static BuildSnapshot FromScan(BuildScan scan, string assetsFolder, string version, Action<string, float>? report = null)
        {
            var snapshot = new BuildSnapshot { Version = version, Taken = DateTime.UtcNow };
            int done = 0, count = scan.Artifacts.Count;
            foreach (var a in scan.Artifacts)
            {
                if ((++done & 63) == 0) report?.Invoke($"Comparing with the last push: {done:N0} of {count:N0}", done / (float)Math.Max(1, count));
                if (!File.Exists(a.CachePath)) continue;
                using var file = File.OpenRead(a.CachePath);
                snapshot.Assets[a.Guid] = new Item
                {
                    Name = a.Name,
                    Type = AssetDatabase.GetAssetType(a.Guid)?.Name ?? a.DataType,
                    Source = a.SourcePath,
                    Hash = Hash(file),
                };
            }
            snapshot.HashScripts(assetsFolder);
            return snapshot;
        }

        /// <summary>
        /// Worker: what a build already written to <paramref name="output"/> holds, from its manifest and packs (they store
        /// the artifacts byte for byte, so the hashes equal those of the Cache files). For the first push after this was
        /// built: the build pushed before had no snapshot. The scripts are hashed from the project as they are now.
        /// </summary>
        public static BuildSnapshot? FromBuild(string output, string assetsFolder, string fallbackVersion)
        {
            var data = Path.Combine(output, PackBuilder.DataFolderName);
            var manifestPath = Path.Combine(data, PackManifest.FileName);
            if (!File.Exists(manifestPath)) return null;
            var manifest = PackManifest.Load(manifestPath);

            // Names and types from the metas: a source's own artifact under its GUID, its sub-assets under theirs
            var names = new Dictionary<string, Item>(StringComparer.OrdinalIgnoreCase);
            foreach (var meta in manifest.Metas)
            {
                names[meta.Guid] = new Item
                {
                    Name = Path.GetFileNameWithoutExtension(meta.SourcePath ?? ""),
                    Type = meta.MainSemanticType ?? meta.MainAssetType ?? "",
                    Source = meta.SourcePath ?? "",
                };
                foreach (var sub in meta.SubAssets)
                    names[sub.Guid] = new Item { Name = sub.Name ?? "", Type = sub.AssetType ?? sub.Type ?? "", Source = meta.SourcePath ?? "" };
            }

            var snapshot = new BuildSnapshot
            {
                Version = string.IsNullOrWhiteSpace(manifest.BuildVersion) ? fallbackVersion : manifest.BuildVersion,
                Taken = manifest.Created,
            };
            foreach (var pack in manifest.Packs)
            {
                using var reader = PackReader.Open(Path.Combine(data, pack));
                foreach (var entry in reader.Entries)
                {
                    using var stream = reader.OpenEntry(entry);
                    var item = names.TryGetValue(entry.Guid, out var known)
                        ? new Item { Name = known.Name, Type = known.Type, Source = known.Source }
                        : new Item { Type = entry.DataType };
                    item.Hash = Hash(stream);
                    snapshot.Assets[entry.Guid] = item;
                }
            }
            snapshot.HashScripts(assetsFolder);
            return snapshot;
        }

        void HashScripts(string assetsFolder)
        {
            foreach (var file in Directory.EnumerateFiles(assetsFolder, "*.cs", SearchOption.AllDirectories))
            {
                using var stream = File.OpenRead(file);
                Scripts[Path.GetRelativePath(assetsFolder, file)] = Hash(stream);
            }
        }

        static string Hash(Stream stream) => Convert.ToHexString(SHA256.HashData(stream), 0, 16);

        /// <summary>What changed from <paramref name="previous"/> (null: everything is new) to <paramref name="next"/>.</summary>
        public static BuildChanges Compare(BuildSnapshot? previous, BuildSnapshot next)
        {
            var changes = new BuildChanges { From = previous?.Version ?? "", To = next.Version };
            var before = previous?.Assets ?? new Dictionary<string, Item>(StringComparer.OrdinalIgnoreCase);
            foreach (var (guid, item) in next.Assets)
            {
                if (!before.TryGetValue(guid, out var old)) changes.Added.Add(item);
                else if (old.Hash != item.Hash) changes.Changed.Add(item);
            }
            foreach (var (guid, item) in before)
                if (!next.Assets.ContainsKey(guid)) changes.Removed.Add(item);

            var scripts = previous?.Scripts ?? new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var (path, hash) in next.Scripts)
            {
                if (!scripts.TryGetValue(path, out var old)) changes.ScriptsAdded.Add(path);
                else if (old != hash) changes.ScriptsChanged.Add(path);
            }
            foreach (var path in scripts.Keys)
                if (!next.Scripts.ContainsKey(path)) changes.ScriptsRemoved.Add(path);
            changes.First = previous == null;
            return changes;
        }
    }

    /// <summary>The difference between two builds (<see cref="BuildSnapshot.Compare"/>).</summary>
    public sealed class BuildChanges
    {
        public string From = "", To = "";
        /// <summary>No earlier push to compare with: everything is new.</summary>
        public bool First;
        public List<BuildSnapshot.Item> Added = new(), Changed = new(), Removed = new();
        public List<string> ScriptsAdded = new(), ScriptsChanged = new(), ScriptsRemoved = new();

        public int Count => Added.Count + Changed.Count + Removed.Count + ScriptsAdded.Count + ScriptsChanged.Count + ScriptsRemoved.Count;

        /// <summary>One line: the counts.</summary>
        public string Short => First
            ? $"first push of {To}: {Added.Count:N0} assets, {ScriptsAdded.Count} scripts"
            : $"{From} → {To}: {Added.Count} assets added, {Changed.Count} changed, {Removed.Count} removed; " +
              $"scripts {ScriptsAdded.Count} added, {ScriptsChanged.Count} changed, {ScriptsRemoved.Count} removed";

        /// <summary>The whole list, grouped by asset type, names sorted.</summary>
        public string Text()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Changes since the last push to itch.io: {Short}");
            void Group(string title, List<BuildSnapshot.Item> items)
            {
                if (items.Count == 0) return;
                sb.AppendLine();
                sb.AppendLine($"{title} ({items.Count}):");
                foreach (var g in items.GroupBy(i => i.Type).OrderByDescending(g => g.Count()))
                    sb.AppendLine($"  {g.Key} ({g.Count()}): " + string.Join(", ", g.Select(i => i.Name).Distinct().OrderBy(n => n)));
            }
            void Scripts(string title, List<string> paths)
            {
                if (paths.Count == 0) return;
                sb.AppendLine();
                sb.AppendLine($"{title} ({paths.Count}):");
                foreach (var p in paths.OrderBy(p => p)) sb.AppendLine("  " + p);
            }
            Scripts("Scripts added", ScriptsAdded);
            Scripts("Scripts changed", ScriptsChanged);
            Scripts("Scripts removed", ScriptsRemoved);
            Group("Assets added", Added);
            Group("Assets changed", Changed);
            Group("Assets removed", Removed);
            return sb.ToString();
        }
    }
}
