using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using Freefall.Assets;
using Freefall.Graphics;

namespace Freefall.Editor
{
    /// <summary>
    /// Opening a project: everything between "open project" (landing page, MCP project_open) and the editor desktop.
    /// Work that needs no main thread (scanning the asset database, importing, compiling scripts and the culler's
    /// shaders) runs on workers; the rest (thumbnails, loading the scripts, building the editor UI, GPU buffers) runs
    /// in short slices at the end of a frame (Engine.RunOnMainThreadAsync), so the window, the landing page's progress
    /// popup and the MCP server stay responsive the whole time. It all used to run in one main-thread call: 10.4 s on
    /// Freefall-Solid with nothing to import (2026-10-17), long enough for MCP calls to time out with a 504.
    /// See .agent/knowledge/editor_debugging/project_open.md.
    /// </summary>
    internal static class ProjectOpening
    {
        /// <summary>Where an open stands; a new instance per change, so readers on any thread see a consistent one.</summary>
        public sealed record Status(
            string Path,
            string Name,
            int StepIndex,
            int StepCount,
            string Step,
            string Detail,
            float Fraction,       // the whole open, 0..1
            double Seconds);

        private enum Runs { Worker, Main }

        /// <summary>
        /// The steps in order, with their cost in seconds measured on Freefall-Solid (13k assets, nothing to import,
        /// 2026-10-17) as weights for the overall fraction.
        /// </summary>
        private static readonly (string Name, double Weight)[] Steps =
        {
            ("Scanning assets", 0.5),
            ("Importing changed assets", 0.2),       // + per asset to import, set once the count is known
            ("Checking for new files", 0.45),
            ("Generating thumbnails", 0.05),         // + per thumbnail
            ("Compiling scripts and shaders", 2.3),
            ("Loading scripts", 0.25),
            ("Building the editor", 0.7),
            ("Opening the scene view", 0.05),
        };

        private const double SecondsPerImport = 0.05;
        private const double SecondsPerThumbnail = 0.2;

        // Main-thread time per frame for thumbnails; one thumbnail always runs, however long it takes
        private const double ThumbnailSliceMs = 12;

        private static readonly double[] _weights = new double[Steps.Length];
        private static volatile Status _status;
        private static Task<string> _completion = Task.FromResult<string>(null);
        private static Stopwatch _clock;
        private static int _stepIndex;
        private static double _longestSliceMs;
        private static string _longestSlice;

        /// <summary>True from Start until the editor desktop is up (or the open failed).</summary>
        public static bool IsOpening => _status != null;

        /// <summary>The open in progress, or null.</summary>
        public static Status Current => _status;

        /// <summary>
        /// The latest open: completes once the editor is fully up with null, or with the error message when the open
        /// failed. Already completed when no open was ever started.
        /// </summary>
        public static Task<string> Completion => _completion;

        /// <summary>
        /// Start opening a project. Main thread. False (and nothing starts) while an open is already running or a
        /// project is open: the editor cannot switch projects without a restart.
        /// </summary>
        public static bool Start(string path, LandingDesktop landing)
        {
            if (IsOpening || Program.IsProjectOpen || landing == null) return false;

            var name = Path.GetFileNameWithoutExtension(path.TrimEnd('\\', '/'));
            _clock = Stopwatch.StartNew();
            _longestSliceMs = 0;
            _longestSlice = null;
            for (int i = 0; i < Steps.Length; i++)
                _weights[i] = Steps[i].Weight;
            _stepIndex = 0;
            _status = new Status(path, name, 0, Steps.Length, Steps[0].Name, "", 0f, 0);

            // The popup shows from the first frame; it reads Current every frame (LandingDesktop.OnUpdate)
            landing.ShowImportDialog(name);

            _completion = Task.Run(() => RunAsync(path, landing));
            return true;
        }

        private static async Task<string> RunAsync(string path, LandingDesktop landing)
        {
            try
            {
                // ── Workers: the asset database ──
                await Step(0, Runs.Worker, () => Engine.OpenProject(path, Report));
                Rename(Engine.Project.Name);
                await OnMain("Recent projects", () => RecentProjects.Add(Engine.Project.Name, path));

                await StepAsync(1, () => AssetDatabase.ImportAllAsync((detail, fraction) =>
                {
                    // The report after the dirty check names the count ("Importing 12 assets..."): weigh the step by it
                    var words = detail.Split(' ', 3);
                    if (fraction == 0f && words.Length > 1 && words[0] == "Importing" && int.TryParse(words[1], out int count))
                        SetWeight(1, Steps[1].Weight + count * SecondsPerImport);
                    Report(detail, fraction);
                }));

                // Importers may write new source files (and the import may have left some dirty): pick them up,
                // with the same per-file progress as the first pass
                await StepAsync(2, async () =>
                {
                    Report("Scanning for files the import wrote", 0f);
                    AssetDatabase.Refresh();
                    await AssetDatabase.ImportAllAsync((detail, fraction) => Report(detail, 0.6f + 0.4f * fraction));
                });

                // ── Main thread: thumbnails for a few milliseconds per frame (at least one) ──
                AssetDatabase.ThumbnailBatch thumbnails = null;
                await Step(3, Runs.Main, () =>
                    thumbnails = AssetDatabase.BeginMissingThumbnails(() => new ThumbnailRenderer()));
                try
                {
                    if (thumbnails.Total > 0)
                    {
                        SetWeight(3, Steps[3].Weight + thumbnails.Total * SecondsPerThumbnail);
                        while (!thumbnails.IsFinished)
                        {
                            Report($"Thumbnail {thumbnails.Done + 1}/{thumbnails.Total}: {thumbnails.NextTypeName}",
                                   thumbnails.Done / (float)thumbnails.Total);
                            await OnMain("Thumbnails", () =>
                            {
                                var slice = Stopwatch.StartNew();
                                while (thumbnails.Step() && slice.Elapsed.TotalMilliseconds < ThumbnailSliceMs) { }
                            });
                        }
                    }
                }
                finally
                {
                    if (thumbnails != null)
                        await OnMain("Thumbnail renderer", thumbnails.Dispose);
                }

                // ── Workers: Roslyn and DXC, side by side ──
                var assetsDirectory = Engine.Project.AssetsDirectory;
                ScriptCompiler.BuildResult scripts = null;
                await Step(4, Runs.Worker, () =>
                {
                    Report("C# scripts, GPU culling shaders", 0f);
                    var compileScripts = Task.Run(() => scripts = ScriptCompiler.BuildForInitialize(assetsDirectory));
                    var compileCuller = Task.Run(() => CommandBuffer.PrepareCuller(Engine.Device));
                    Task.WaitAll(compileScripts, compileCuller);
                });

                // ── Main thread: slices between frames ──
                await Step(5, Runs.Main, () => ScriptCompiler.Initialize(assetsDirectory, scripts));

                // The asset browser's type map scans every loaded assembly (0.4 s): after the scripts, off the main thread
                await Step(6, Runs.Worker, AssetDragData.WarmTypeMap);

                EditorUI editorUI = null;
                Report("Editor panels", 0.2f);
                await OnMain("Editor panels", () => editorUI = new EditorUI(Program.Form));
                Report("Asset browser", 0.6f);
                await OnMain("Asset browser", () =>
                {
                    EditorPreferences.RegisterStyles(editorUI);
                    editorUI.AssetBrowser.Initialize();
                });
                Report("Asset types", 0.9f);
                await OnMain("Asset types", EditorPreferences.DiscoverAllTypes);

                await Step(7, Runs.Main, () =>
                {
                    CommandBuffer.InitializeCuller(Engine.Device);   // buffers only: the shaders compiled above
                    Program.ShowEditor(editorUI);
                    _status = null;   // in the same slice: IsProjectOpen and IsOpening are never both true
                });

                var total = _clock.Elapsed.TotalSeconds;
                var shaders = Shader.CacheStats;
                Debug.LogAlways($"[OpenProject] {Engine.Project.Name} open in {total:F1} s; " +
                                $"longest main-thread slice {_longestSliceMs:F0} ms ({_longestSlice}); " +
                                $"shaders since start: {shaders.Misses} compiled ({shaders.CompileMs / 1000:F1} s), " +
                                $"{shaders.DiskHits} from the disk cache, {shaders.Hits} reused in memory " +
                                $"({shaders.Bytes / (1024.0 * 1024.0):F1} MB kept)");
                return null;
            }
            catch (Exception ex)
            {
                var error = ex is AggregateException agg ? agg.Flatten().InnerException ?? ex : ex;
                Debug.LogWarning("Editor", $"Failed to open project: {error}");
                var name = _status?.Name ?? Path.GetFileName(path);
                _status = null;

                // A half-open project cannot be undone in place: the editor closes, a restart tries again
                await Engine.RunOnMainThreadAsync(() =>
                {
                    landing.ShowErrorDialog($"Could not open {name}", error.Message, "Close editor");
                    landing.OnErrorDismissed += () => Program.Form.Close();
                });
                return error.Message;
            }
        }

        // ── Steps ──

        private static Task Step(int index, Runs runs, Action work)
        {
            BeginStep(index);
            return runs == Runs.Worker ? TimedWorker(Steps[index].Name, () => { work(); return Task.CompletedTask; })
                                           : OnMain(Steps[index].Name, work);
        }

        private static Task StepAsync(int index, Func<Task> work)
        {
            BeginStep(index);
            return TimedWorker(Steps[index].Name, work);
        }

        private static void BeginStep(int index)
        {
            _stepIndex = index;
            Publish(Steps[index].Name, "", 0f);
        }

        private static async Task TimedWorker(string name, Func<Task> work)
        {
            var t = Stopwatch.StartNew();
            await Task.Run(work);
            Debug.LogAlways($"[OpenProject] {name}: {t.Elapsed.TotalMilliseconds:F0} ms (worker)");
        }

        /// <summary>Run a slice on the main thread at the end of a frame and wait for it; logs its time, keeps the longest.</summary>
        private static Task OnMain(string name, Action work)
        {
            return Engine.RunOnMainThreadAsync(() =>
            {
                var t = Stopwatch.StartNew();
                work();
                double ms = t.Elapsed.TotalMilliseconds;
                if (ms > _longestSliceMs)
                {
                    _longestSliceMs = ms;
                    _longestSlice = name;
                }
                if (ms >= 20)
                    Debug.LogAlways($"[OpenProject] {name}: {ms:F0} ms (main thread)");
            });
        }

        // ── Progress ──

        /// <summary>Progress within the current step (any thread): what it is doing, fraction of the step 0..1.</summary>
        private static void Report(string detail, float fraction)
        {
            var status = _status;
            if (status == null) return;
            Publish(status.Step, detail, fraction);
        }

        private static void SetWeight(int index, double weight) => _weights[index] = weight;

        private static void Rename(string name)
        {
            var status = _status;
            if (status != null) _status = status with { Name = name };
        }

        private static void Publish(string step, string detail, float stepFraction)
        {
            var status = _status;
            if (status == null) return;

            double total = 0, done = 0;
            for (int i = 0; i < _weights.Length; i++)
            {
                total += _weights[i];
                if (i < _stepIndex) done += _weights[i];
            }
            done += _weights[_stepIndex] * Math.Clamp(stepFraction, 0f, 1f);

            _status = status with
            {
                StepIndex = _stepIndex,
                Step = step,
                Detail = detail ?? "",
                Fraction = total > 0 ? (float)(done / total) : 0f,
                Seconds = _clock.Elapsed.TotalSeconds,
            };
        }
    }
}
