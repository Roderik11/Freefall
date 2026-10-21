using System.Collections.Concurrent;
using Freefall.Assets;
using Freefall.Assets.Importers;

namespace Freefall.Editor.Publishing
{
    /// <summary>
    /// A build profile's Scan with the progress popup (ProgressJob), from its button or MCP publish_scan: the references,
    /// the textures against the budget (TextureBudgetScan, ~3 s cold), the report. All on workers. Every finished scan
    /// (also Publish's and a cap's) is kept as the profile's latest: the profile inspector's texture list shows it.
    /// </summary>
    public static class BuildScanJob
    {
        const string Following = "Following the references", Measuring = "Measuring the textures", Writing = "Writing the report";

        static readonly ConcurrentDictionary<string, BuildScan> _latest = new(StringComparer.OrdinalIgnoreCase);
        static int _version;

        /// <summary>Goes up with every scan kept: the inspector compares it to know when to show a new one.</summary>
        public static int Version => Volatile.Read(ref _version);

        /// <summary>The profile's last scan in this editor, or null.</summary>
        public static BuildScan? Latest(BuildProfile profile) =>
            profile.Guid != null && _latest.TryGetValue(profile.Guid, out var scan) ? scan : null;

        /// <summary>Keep a finished scan as the profile's latest (any thread).</summary>
        public static void Store(BuildProfile profile, BuildScan scan)
        {
            if (profile.Guid == null) return;
            _latest[profile.Guid] = scan;
            Interlocked.Increment(ref _version);
        }

        /// <summary>The steps a scan adds to a job, with their weights (seconds measured on the Game profile).</summary>
        internal static (string, double)[] Steps => new (string, double)[] { (Following, 0.5), (Measuring, 3), (Writing, 0.2) };

        /// <summary>Main thread: start a Scan. Throws InvalidOperationException while another job runs.</summary>
        public static Task<BuildScan> Start(BuildProfile profile, bool compareLoaded = false)
        {
            var job = ProgressJob.Start($"Scanning {profile.Name}", "Scan", Steps);
            return Task.Run(async () =>
            {
                try
                {
                    var scan = await Run(job, profile, compareLoaded);
                    var over = scan.Textures?.Over;
                    Debug.LogAlways($"[Build] {profile.Name}: {scan.Artifacts.Count:N0} assets, {BuildScan.Size(scan.TotalBytes)}, " +
                                    $"{scan.Warnings.Count} warnings, {job.Elapsed.TotalSeconds:0.0} s" +
                                    (over != null ? $"; {over.Count} textures over {profile.TextureBudget} px ({BuildScan.Size(over.Sum(t => t.Artifact.Size))})" : "") +
                                    $". Report: {profile.ReportPath}");
                    await job.Finish($"{scan.Artifacts.Count:N0} assets, {BuildScan.Size(scan.TotalBytes)}: report written");
                    return scan;
                }
                catch (Exception ex)
                {
                    var error = ProgressJob.Unwrap(ex);
                    await job.Fail($"Could not scan {profile.Name}", error);
                    throw error;
                }
            });
        }

        /// <summary>Worker: a scan's steps inside a running job, the report written and the scan kept.</summary>
        internal static async Task<BuildScan> Run(ProgressJob job, BuildProfile profile, bool compareLoaded = false)
        {
            BuildScan scan = null!;
            await job.Step(Following, () => Task.Run(() =>
            {
                job.Report("From the scenes down", 0f);
                scan = DependencyScanner.Scan(profile);
                if (compareLoaded) scan.LoadedNotReached = DependencyScanner.LoadedNotReached(scan);
            }));
            await job.Step(Measuring, () => Task.Run(() => scan.Textures = profile.MeasureTextures(scan, job.Report)));
            await job.Step(Writing, () => Task.Run(() => scan.WriteReport(profile.ReportPath)));
            Store(profile, scan);
            return scan;
        }
    }

    /// <summary>
    /// Capping textures at a size (their importer's Max Size) or taking the cap off (0), with the progress popup: the
    /// settings saved, the textures reimported side by side on workers (AssetDatabase.ImportAssetsAsync; loaded ones
    /// update in place), then the profile scanned again so its list and report show the new sizes. Undo is the other
    /// way round: the source files are never touched. Settings saved keep what the import guessed (GuessSettings),
    /// so a normal map stays BC5.
    /// </summary>
    public static class TextureCaps
    {
        const string Saving = "Saving the import settings", Reimporting = "Reimporting the textures";

        /// <summary>Main thread: start capping <paramref name="textures"/> (GUIDs) at <paramref name="maxSize"/> (0 = uncap).</summary>
        public static Task<int> Start(BuildProfile profile, IReadOnlyList<string> textures, int maxSize)
        {
            string title = maxSize > 0 ? $"Capping {textures.Count} textures at {maxSize}" : $"Taking the cap off {textures.Count} textures";
            // ~0.7 s a texture, several side by side
            var steps = new List<(string, double)> { (Saving, 0.3), (Reimporting, Math.Max(1, textures.Count * 0.7 / Math.Max(1, Environment.ProcessorCount / 2))) };
            steps.AddRange(BuildScanJob.Steps);
            var job = ProgressJob.Start(title, "Textures", steps.ToArray());

            return Task.Run(async () =>
            {
                try
                {
                    var changed = new List<string>();
                    await job.Step(Saving, () => Task.Run(() =>
                    {
                        for (int i = 0; i < textures.Count; i++)
                        {
                            if (AssetDatabase.GetImporter(textures[i]) is not TextureImporter importer || importer.MaxSize == maxSize) continue;
                            importer.MaxSize = maxSize;
                            importer.UserConfigured = true;
                            if (AssetDatabase.SaveImporter(textures[i], importer) != null) changed.Add(textures[i]);
                            if (i % 25 == 0) job.Report(AssetDatabase.GuidToPath(textures[i]) ?? textures[i], i / (float)textures.Count);
                        }
                    }));

                    int failed = 0;
                    await job.Step(Reimporting, async () => failed = await AssetDatabase.ImportAssetsAsync(changed, job.Report));
                    await BuildScanJob.Run(job, profile);

                    string done = maxSize > 0 ? $"capped at {maxSize}" : "back at their source size";
                    Debug.LogAlways($"[Textures] {changed.Count} textures {done}, {failed} failed, {job.Elapsed.TotalSeconds:0} s");
                    await job.Finish($"{changed.Count} textures {done}" + (failed > 0 ? $", {failed} failed (see the console)" : ""));
                    return changed.Count;
                }
                catch (Exception ex)
                {
                    var error = ProgressJob.Unwrap(ex);
                    await job.Fail(maxSize > 0 ? "Could not cap the textures" : "Could not take the caps off", error);
                    throw error;
                }
            });
        }
    }
}
