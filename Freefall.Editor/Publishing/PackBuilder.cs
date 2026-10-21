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
        public TimeSpan ScanTime, WriteTime, VerifyTime;
        /// <summary>Entries whose bytes in the pack differ from their Cache file (verify only).</summary>
        public int Mismatches;
        public List<string> Problems = new();
        public bool Verified;
    }

    /// <summary>
    /// Writes a profile's packs: the scan's artifacts byte for byte into Data/data0.ffpack, data1.ffpack… (a new one at
    /// the profile's size limit, in the order the scan reached them) and Data/game.manifest (the scenes, the packs, the
    /// packed assets' metas cut down to their packed artifacts), inside the output folder. The scan and the metas are
    /// taken on the main thread; writing and verifying run on a worker (gigabytes: a main-thread call would freeze the
    /// editor and time out). The output folder is marked with <see cref="MarkerFile"/>; a folder that is not empty and
    /// has no marker is refused, so Publish never clears a folder it did not make.
    /// </summary>
    public static class PackBuilder
    {
        public const string MarkerFile = ".freefall-build";
        public const string DataFolderName = "Data";

        /// <summary>The write in progress (or the last one).</summary>
        public static Task<PackResult>? Current { get; private set; }

        /// <summary>Main thread: scan and start writing. Throws if a write is running or the output folder is not Publish's.</summary>
        public static Task<PackResult> Start(BuildProfile profile, bool verify)
        {
            if (Current is { IsCompleted: false })
                throw new InvalidOperationException("Packs are being written already");

            var watch = Stopwatch.StartNew();
            var scan = DependencyScanner.Scan(profile);
            if (scan.Artifacts.Count == 0)
                throw new InvalidOperationException("The scan found nothing to pack: " + string.Join("; ", scan.Warnings));

            var output = profile.ResolvedOutputFolder;
            PrepareOutputFolder(output);
            scan.WriteReport(profile.ReportPath);

            var manifest = new PackManifest
            {
                Product = profile.ResolvedProductName,
                Created = DateTime.UtcNow,
                Scenes = scan.Artifacts.Where(a => a.Root != null && a.DataType == nameof(SceneData)).Select(a => a.Guid).ToList(),
                Metas = PackedMetas(scan),
            };
            var items = scan.Artifacts.Select(a => (a.Guid, a.DataType, a.CachePath, a.Size)).ToList();
            long limit = Math.Max(1, profile.PackSizeLimitMB) * (1L << 20);
            var scanTime = watch.Elapsed;

            Current = Task.Run(() => Write(Path.Combine(output, DataFolderName), items, manifest, limit, verify, scanTime));
            return Current;
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

        static void PrepareOutputFolder(string output)
        {
            if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any() && !File.Exists(Path.Combine(output, MarkerFile)))
                throw new InvalidOperationException($"{output} is not empty and was not made by Publish (no {MarkerFile}): choose another output folder");
            Directory.CreateDirectory(output);
            File.WriteAllText(Path.Combine(output, MarkerFile), "Written by the Freefall editor's Publish. Its contents are replaced on every publish.\n");
        }

        static PackResult Write(string dataFolder, List<(string Guid, string DataType, string CachePath, long Size)> items,
                                PackManifest manifest, long limit, bool verify, TimeSpan scanTime)
        {
            var result = new PackResult { DataFolder = dataFolder, ScanTime = scanTime };
            Directory.CreateDirectory(dataFolder);
            foreach (var old in Directory.GetFiles(dataFolder, "*" + PackFormat.Extension).Append(Path.Combine(dataFolder, PackManifest.FileName)))
                if (File.Exists(old)) File.Delete(old);

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

            foreach (var (guid, dataType, cachePath, size) in items)
            {
                if (!File.Exists(cachePath))
                {
                    result.Problems.Add($"Not in the Cache, left out: {cachePath}");
                    continue;
                }
                if (writer != null && writer.Entries.Count > 0 && !writer.Fits(size, dataType, limit))
                    Close();
                writer ??= new PackWriter(Path.Combine(dataFolder, $"data{result.Packs.Count}{PackFormat.Extension}"));
                using (var source = File.OpenRead(cachePath))
                    writer.Add(guid, dataType, source);
                result.Entries++;
                result.Bytes += size;
            }
            Close();
            manifest.Save(Path.Combine(dataFolder, PackManifest.FileName));
            result.WriteTime = watch.Elapsed;

            if (verify)
            {
                watch.Restart();
                Verify(dataFolder, manifest, items, result);
                result.VerifyTime = watch.Elapsed;
                result.Verified = true;
            }
            return result;
        }

        /// <summary>Every artifact read back from the packs and compared byte for byte with its Cache file; the manifest read back too.</summary>
        static void Verify(string dataFolder, PackManifest manifest, List<(string Guid, string DataType, string CachePath, long Size)> items, PackResult result)
        {
            var reread = PackManifest.Load(Path.Combine(dataFolder, PackManifest.FileName));
            if (reread.Metas.Count != manifest.Metas.Count || !reread.Packs.SequenceEqual(manifest.Packs) || !reread.Scenes.SequenceEqual(manifest.Scenes))
                result.Problems.Add("The manifest does not read back as it was written");

            using var packs = PackSet.Open(manifest.Packs.Select(p => Path.Combine(dataFolder, p)));
            var a = new byte[1 << 20];
            var b = new byte[1 << 20];
            foreach (var (guid, dataType, cachePath, size) in items)
            {
                if (!File.Exists(cachePath)) continue;
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
