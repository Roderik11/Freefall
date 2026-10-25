using System.Diagnostics;
using System.Text.Json;
using Freefall.Assets;
using Freefall.Assets.Packs;

namespace Freefall.Editor.Publishing
{
    /// <summary>What writing a profile's packs gave.</summary>
    public sealed class PackResult
    {
        public string DataFolder = "";
        public List<(string File, int Entries, long Bytes)> Packs = new();
        public int Entries;
        public long Bytes;
        public TimeSpan WriteTime, VerifyTime;
        /// <summary>Entries whose bytes in the pack differ from their Cache file (verify only).</summary>
        public int Mismatches;
        public List<string> Problems = new();
        public bool Verified;
    }

    /// <summary>What the packs will hold: the manifest and the artifacts in the order the scan reached them.</summary>
    public sealed class PackJob
    {
        public PackManifest Manifest = new();
        public List<(string Guid, string DataType, string CachePath, long Size)> Items = new();
        public long Limit;
    }

    /// <summary>
    /// A profile's packs: the scan's artifacts byte for byte into Data/data0.ffpack, data1.ffpack… (a new one at the
    /// profile's size limit, in the order the scan reached them) and Data/game.manifest (the scenes, the packs, the
    /// packed assets' metas cut down to their packed artifacts). <see cref="Prepare"/> takes the metas on the main
    /// thread; writing and verifying run on a worker (gigabytes). Publisher runs them as steps of a publish.
    /// </summary>
    public static class PackBuilder
    {
        public const string DataFolderName = "Data";
        /// <summary>The project's scripts, compiled for the game (Release), next to Data/: the player loads them from there.</summary>
        public const string ScriptsFile = "FreefallScripts.dll";

        /// <summary>Main thread: the manifest and the artifacts to pack, from a scan.</summary>
        public static PackJob Prepare(BuildProfile profile, BuildScan scan)
        {
            return new PackJob
            {
                Manifest = new PackManifest
                {
                    Product = profile.ResolvedProductName,
                    BuildVersion = string.IsNullOrWhiteSpace(profile.Version) ? null : profile.Version.Trim(),
                    Created = DateTime.UtcNow,
                    Scenes = scan.Artifacts.Where(a => a.Root != null && a.DataType == nameof(SceneData)).Select(a => a.Guid).ToList(),
                    EnginePack = EngineFiles.PackFile,
                    Metas = PackedMetas(scan),
                },
                Items = scan.Artifacts.Select(a => (a.Guid, a.DataType, a.CachePath, a.Size)).ToList(),
                Limit = Math.Max(1, profile.PackSizeLimitMB) * (1L << 20),
            };
        }

        /// <summary>The project's scripts compiled for the game (Release) into the output folder, with their plugin DLLs.</summary>
        internal static void WriteScripts(ScriptCompiler.BuildResult build, string output)
        {
            File.WriteAllBytes(Path.Combine(output, ScriptsFile), build.Pe!);
            if (build.Pdb != null) File.WriteAllBytes(Path.Combine(output, Path.ChangeExtension(ScriptsFile, ".pdb")), build.Pdb);
            foreach (var plugin in build.PluginPaths)
                File.Copy(plugin, Path.Combine(output, Path.GetFileName(plugin)), overwrite: true);
        }

        /// <summary>The metas of the packed assets, each cut down to its packed sub-assets (copies: the database's stay as they are).</summary>
        static List<MetaFile> PackedMetas(BuildScan scan)
        {
            var packed = new HashSet<string>(scan.Artifacts.Select(a => a.Guid), StringComparer.OrdinalIgnoreCase);
            var metas = new List<MetaFile>();
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var a in scan.Artifacts)
            {
                var meta = a.SourcePath.Length > 0 ? AssetDatabase.GetMeta(a.Guid) : null;
                if (meta == null || !seen.Add(meta.Guid)) continue;

                MetaFile copy;
                lock (meta)
                    copy = JsonSerializer.Deserialize<MetaFile>(JsonSerializer.Serialize(meta))!;
                copy.SubAssets = copy.SubAssets.Where(s => packed.Contains(s.Guid)).ToList();
                copy.LastImported = default;
                copy.FileSize = 0;
                metas.Add(copy);
            }
            return metas;
        }

        /// <summary>Worker: the packs and the manifest into <paramref name="dataFolder"/>. <paramref name="report"/> gets each artifact and the share written.</summary>
        public static PackResult Write(string dataFolder, PackJob job, Action<string, float>? report = null)
        {
            var result = new PackResult { DataFolder = dataFolder };
            Directory.CreateDirectory(dataFolder);
            foreach (var old in Directory.GetFiles(dataFolder, "*" + PackFormat.Extension).Append(Path.Combine(dataFolder, PackManifest.FileName)))
                if (File.Exists(old)) File.Delete(old);

            var manifest = job.Manifest;
            manifest.Packs.Clear();
            long total = Math.Max(1, job.Items.Sum(i => i.Size));
            var watch = Stopwatch.StartNew();
            PackWriter? writer = null;
            void Close()
            {
                if (writer == null) return;
                writer.Dispose();
                result.Packs.Add((Path.GetFileName(writer.Path), writer.Entries.Count, new FileInfo(writer.Path).Length));
                manifest.Packs.Add(Path.GetFileName(writer.Path));
                writer = null;
            }

            foreach (var (guid, dataType, cachePath, size) in job.Items)
            {
                if (!File.Exists(cachePath))
                {
                    result.Problems.Add($"Not in the Cache, left out: {cachePath}");
                    continue;
                }
                if (writer != null && writer.Entries.Count > 0 && !writer.Fits(size, dataType, job.Limit))
                    Close();
                writer ??= new PackWriter(Path.Combine(dataFolder, $"data{result.Packs.Count}{PackFormat.Extension}"));
                report?.Invoke($"{Path.GetFileName(writer.Path)}: {Path.GetFileName(cachePath)} ({dataType})", result.Bytes / (float)total);
                using (var source = File.OpenRead(cachePath))
                    writer.Add(guid, dataType, source);
                result.Entries++;
                result.Bytes += size;
            }
            Close();
            manifest.Save(Path.Combine(dataFolder, PackManifest.FileName));
            result.WriteTime = watch.Elapsed;
            return result;
        }

        /// <summary>
        /// Worker: the engine's own files (every file under <paramref name="resources"/> but <paramref name="leaveOut"/>
        /// folders) into Data/engine.ffpack, each under the key of its path relative to Resources (EngineFiles.KeyOf),
        /// so the game folder has no Resources folder. Returns the files and bytes packed.
        /// </summary>
        public static (int Files, long Bytes) WriteEngineFiles(string resources, string dataFolder, IEnumerable<string> leaveOut,
                                                              Action<string, float>? report = null)
        {
            var skip = leaveOut.Select(f => f.TrimEnd('\\', '/') + Path.DirectorySeparatorChar).ToList();
            var files = Directory.GetFiles(resources, "*", SearchOption.AllDirectories)
                .Select(f => Path.GetRelativePath(resources, f))
                .Where(r => !skip.Any(s => r.StartsWith(s, StringComparison.OrdinalIgnoreCase)))
                .OrderBy(r => r, StringComparer.OrdinalIgnoreCase)
                .ToList();

            long bytes = 0;
            using (var writer = new PackWriter(Path.Combine(dataFolder, EngineFiles.PackFile)))
            {
                for (int i = 0; i < files.Count; i++)
                {
                    report?.Invoke($"{EngineFiles.PackFile}: {files[i]}", i / (float)Math.Max(1, files.Count));
                    using var source = File.OpenRead(Path.Combine(resources, files[i]));
                    writer.Add(EngineFiles.KeyOf(files[i]), EngineFiles.DataType, source);
                    bytes += source.Length;
                }
            }
            return (files.Count, bytes);
        }

        /// <summary>Worker: every artifact read back from the packs and compared byte for byte with its Cache file; the manifest read back too.</summary>
        public static void Verify(PackJob job, PackResult result, Action<string, float>? report = null)
        {
            var watch = Stopwatch.StartNew();
            var dataFolder = result.DataFolder;
            var manifest = job.Manifest;
            var reread = PackManifest.Load(Path.Combine(dataFolder, PackManifest.FileName));
            if (reread.Metas.Count != manifest.Metas.Count || !reread.Packs.SequenceEqual(manifest.Packs) || !reread.Scenes.SequenceEqual(manifest.Scenes))
                result.Problems.Add("The manifest does not read back as it was written");

            using var packs = PackSet.Open(manifest.Packs.Select(p => Path.Combine(dataFolder, p)));
            var a = new byte[1 << 20];
            var b = new byte[1 << 20];
            long total = Math.Max(1, result.Bytes), done = 0;
            foreach (var (guid, dataType, cachePath, size) in job.Items)
            {
                if (!File.Exists(cachePath)) continue;
                report?.Invoke(Path.GetFileName(cachePath), done / (float)total);
                done += size;
                using var packed = packs.Open(guid);
                if (packed == null)
                {
                    result.Mismatches++;
                    result.Problems.Add($"Not in the packs: {guid} ({cachePath})");
                    continue;
                }
                packs.TryGet(guid, out var entry);
                using var cached = File.OpenRead(cachePath);
                if (packed.Length != cached.Length || entry.DataType != dataType || !SameBytes(packed, cached, a, b))
                {
                    result.Mismatches++;
                    if (result.Problems.Count < 50) result.Problems.Add($"Differs from its Cache file: {guid} ({cachePath})");
                }
            }
            result.VerifyTime = watch.Elapsed;
            result.Verified = true;
        }

        static bool SameBytes(Stream x, Stream y, byte[] a, byte[] b)
        {
            while (true)
            {
                int n = x.ReadAtLeast(a, a.Length, throwOnEndOfStream: false);
                int m = y.ReadAtLeast(b, b.Length, throwOnEndOfStream: false);
                if (n != m || !a.AsSpan(0, n).SequenceEqual(b.AsSpan(0, m))) return false;
                if (n == 0) return true;
            }
        }
    }
}
