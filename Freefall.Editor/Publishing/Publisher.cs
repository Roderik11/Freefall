using System.Diagnostics;
using System.Text;
using Freefall.Assets;
using Squid;

namespace Freefall.Editor.Publishing
{
    /// <summary>How to publish: all optional.</summary>
    public sealed class PublishOptions
    {
        /// <summary>Write here instead of the profile's output folder (absolute).</summary>
        public string? Output;
        /// <summary>Read every packed artifact back and compare it with its Cache file.</summary>
        public bool Verify;
        /// <summary>Save the changed assets the build reaches first; without it, such assets stop the publish.</summary>
        public bool SaveChanged;
        /// <summary>Start the game once it is written.</summary>
        public bool Run;
        /// <summary>Make the installer (true) or not (false) whatever the profile says; null = the profile's MakeInstaller.</summary>
        public bool? Installer;
    }

    /// <summary>What a publish gave.</summary>
    public sealed class PublishResult
    {
        public string Product = "", Output = "", Exe = "", Report = "";
        public PackResult Packs = new();
        public int Scripts;
        public int PlayerFiles;
        public long PlayerBytes;
        /// <summary>The version of the Visual C++ runtime copied beside the exe (<see cref="VCRuntime"/>).</summary>
        public string Runtime = "";
        /// <summary>The engine's own files in Data/engine.ffpack.</summary>
        public int EngineFiles;
        public long EngineBytes;
        /// <summary>The scan's warnings and the packs' problems: published anyway.</summary>
        public List<string> Warnings = new();
        public List<(string Step, double Seconds)> Steps = new();
        public TimeSpan Elapsed;
        /// <summary>The installer's setup exe, when one was made; its .bin parts lie beside it when the game is over ~2 GB.</summary>
        public string Installer = "";
        public long InstallerBytes;
    }

    /// <summary>
    /// Publish: a BuildProfile's game, ready to run, in its output folder (or one given). The steps:
    /// <list type="number">
    /// <item>Checking and scanning: the scan on a worker (cold, it reads every scene, prefab and material artifact: ~5 s);
    /// on the main thread, changed assets the build reaches are saved first if asked, otherwise they stop it (the packs
    /// hold what is saved); an artifact reached but not in the Cache stops it.</item>
    /// <item>Compiling scripts (worker): the output folder cleared (only one Publish made: <see cref="MarkerFile"/>), then
    /// FreefallScripts.dll, Release without DEBUG. Compile errors stop it.</item>
    /// <item>Building the player (worker): <see cref="PlayerBuild"/>, dotnet publish into the project's Library.</item>
    /// <item>Copying the player (worker): into the output folder, the exe named after the product, and the Visual C++
    /// runtime PhysX needs beside it (<see cref="VCRuntime"/>).</item>
    /// <item>Writing packs (worker): the assets, then the engine's own files (Data/engine.ffpack: no Resources folder in the
    /// game's), then Verifying packs if asked.</item>
    /// <item>Building the installer (worker), when the profile asks for one (MakeInstaller): <see cref="InstallerBuild"/>.</item>
    /// </list>
    /// The editor shows a modal progress popup (ModalDialogs, the landing page's) whoever started it, the profile's
    /// button or MCP game_publish. A failure shows a message popup. The summary goes to the console and the end of the
    /// build report. See .agent/knowledge/engine_internals/publishing.md.
    /// </summary>
    public static class Publisher
    {
        public const string MarkerFile = ".freefall-build";


        const string Checking = "Checking and scanning", Scripts = "Compiling scripts", Player = "Building the player",
                     Copying = "Copying the player", Packs = "Writing packs", Verifying = "Verifying packs",
                     Installer = "Building the installer";

        // Seconds measured on Freefall-Solid's Game profile (4.7 GB), as weights for the overall fraction; the packs'
        // weights are set from their size once the scan is done
        static readonly Dictionary<string, double> Weights = new()
        {
            [Checking] = 0.6, [Scripts] = 3, [Player] = 13, [Copying] = 2, [Packs] = 24, [Verifying] = 13, [Installer] = 20,
        };
        const double WriteBytesPerSecond = 200e6, VerifyBytesPerSecond = 370e6, InstallerBytesPerSecond = 150e6;

        static Task<PublishResult> _completion = Task.FromResult(new PublishResult());

        /// <summary>True from Start until the publish is written (or failed).</summary>
        public static bool IsPublishing => !_completion.IsCompleted;

        /// <summary>The latest publish (faulted with the reason when it failed). Already completed when none was started.</summary>
        public static Task<PublishResult> Completion => _completion;

        /// <summary>
        /// The profile's buttons: when changed assets wait for Save Changed Assets, ask first (save and publish, or
        /// cancel); a refusal to start shows as a message popup. Main thread.
        /// </summary>
        public static void StartFromEditor(BuildProfile profile, bool run)
        {
            var desktop = Program.EditorUI;
            void Go(bool save)
            {
                try { Start(profile, new PublishOptions { Run = run, SaveChanged = save }); }
                catch (InvalidOperationException ex) { ModalDialogs.ShowMessage(desktop, $"Could not publish {profile.ResolvedProductName}", ex.Message, ("OK", null)); }
            }

            var changed = AssetCreator.ChangedAssets;
            if (changed.Count == 0)
            {
                Go(save: false);
                return;
            }
            ModalDialogs.ShowMessage(desktop, "Save changed assets first?",
                $"A game holds the assets as they are saved. These have changes that are not: {Names(changed)}. " +
                "Save them (as Save Changed Assets does) and publish?",
                ("Save and publish", () => Go(save: true)), ("Cancel", null));
        }

        /// <summary>
        /// Start a publish. Main thread. Throws InvalidOperationException (nothing starts) while one runs, without a
        /// project, scenes or the player's source; problems found later end <see cref="Completion"/> with the reason.
        /// </summary>
        public static Task<PublishResult> Start(BuildProfile profile, PublishOptions options)
        {
            if (ProgressJob.Running is { } running) throw new InvalidOperationException($"Wait for it to finish: {running.Title} is running");
            if (!Program.IsProjectOpen || Engine.Project == null) throw new InvalidOperationException("No project is open");
            if (profile.Scenes.All(s => s == null)) throw new InvalidOperationException($"The profile {profile.Name} has no scenes");
            var playerProject = PlayerBuild.FindProject() ?? throw new InvalidOperationException(
                "Publishing builds the player from the engine's source (Freefall.Player), and no folder above the editor's holds it");
            var runtime = VCRuntime.Find() ?? throw new InvalidOperationException(
                $"The game carries the Visual C++ runtime PhysX needs ({string.Join(", ", VCRuntime.Files)}), copied from Visual Studio's " +
                "C++ redist, and no Visual Studio here has it: add \"Desktop development with C++\" (or install the Build Tools), " +
                "or set FREEFALL_VCREDIST to a folder that holds those files");

            var output = Path.GetFullPath(string.IsNullOrWhiteSpace(options.Output) ? profile.ResolvedOutputFolder : options.Output.Trim());
            var product = profile.ResolvedProductName;
            CheckOutputFolder(output, Engine.Project.RootDirectory);
            string? iscc = null;
            if (options.Installer ?? profile.MakeInstaller)
                iscc = InstallerBuild.FindCompiler() ?? throw new InvalidOperationException(
                    "The profile makes an installer, and Inno Setup 6 is not installed: install it (jrsoftware.org, or " +
                    "winget install JRSoftware.InnoSetup), set FREEFALL_ISCC to its ISCC.exe, or turn MakeInstaller off");
            var steps = new List<string> { Checking, Scripts, Player, Copying, Packs };
            if (options.Verify) steps.Add(Verifying);
            if (iscc != null) steps.Add(Installer);
            var progress = ProgressJob.Start($"Publishing {product}", "Publish", steps.Select(s => (s, Weights[s])).ToArray());
            string scripts = Path.Combine(Engine.Project.LibraryDirectory, "Installer");
            _completion = Task.Run(() => RunAsync(profile, options, output, playerProject, runtime, progress, iscc, scripts));
            return _completion;
        }

        static async Task<PublishResult> RunAsync(BuildProfile profile, PublishOptions options, string output, string playerProject,
                                                  (string Folder, string Version) runtime, ProgressJob progress, string? iscc, string installerScripts)
        {
            var result = new PublishResult { Product = profile.ResolvedProductName, Output = output, Report = BuildProfile.ReportPathFor(output) };
            void Report(string detail, float fraction) => progress.Report(detail, fraction);
            void SetWeight(string step, double weight) => progress.SetWeight(step, weight);
            Task Step(string name, Func<Task> work) => progress.Step(name, work);
            try
            {
                // ── The scan (worker; the database's lookups are concurrent), what is unsaved (main thread), the metas ──
                PackJob job = null!;
                string assets = "", staging = "";
                await Step(Checking, async () =>
                {
                    Report("Following the references from the scenes", 0f);
                    var scan = await Task.Run(() => DependencyScanner.Scan(profile));

                    bool saved = false;
                    await Engine.RunOnMainThreadAsync(() =>
                    {
                        var reached = new HashSet<string>(scan.Artifacts.Select(a => a.Guid), StringComparer.OrdinalIgnoreCase);
                        var changed = AssetCreator.ChangedAssets.Where(a => !string.IsNullOrEmpty(a.Guid) && reached.Contains(a.Guid)).ToList();
                        if (changed.Count > 0)
                        {
                            if (!options.SaveChanged)
                                throw new InvalidOperationException($"The game would not hold the changes of {Names(changed)}: they are not saved. " +
                                                                    "Save them (Save Changed Assets), or publish with saving");
                            AssetCreator.SaveChangedAssets();
                            saved = true;
                        }
                        assets = Engine.Project.AssetsDirectory;
                        staging = PlayerBuild.StagingFolder(Engine.Project.LibraryDirectory);
                    });
                    if (saved)
                    {
                        Report("Following the references again: assets were saved", 0.4f);
                        scan = await Task.Run(() => DependencyScanner.Scan(profile));   // what was saved: the baked maps' sizes may differ
                    }

                    Report("The build report and the packed metas", 0.8f);
                    await Task.Run(() =>
                    {
                        var missing = scan.Warnings.Where(w => w.StartsWith("Not imported", StringComparison.Ordinal)).ToList();
                        if (missing.Count > 0)
                            throw new InvalidOperationException($"{missing.Count} assets the game reaches have nothing in the Cache to pack " +
                                                                $"(not imported, or a NavMesh never baked): {string.Join("; ", missing.Take(3))}");
                        if (!scan.Scenes.Any(s => s.Root != null))
                            throw new InvalidOperationException("None of the profile's scenes is a saved scene asset");

                        scan.Textures = profile.MeasureTextures(scan);
                        scan.WriteReport(result.Report, published: true);
                        BuildScanJob.Store(profile, scan);
                        result.Warnings.AddRange(scan.Warnings);
                        job = PackBuilder.Prepare(profile, scan);

                        long bytes = job.Items.Sum(i => i.Size);
                        SetWeight(Packs, bytes / WriteBytesPerSecond);
                        SetWeight(Verifying, bytes / VerifyBytesPerSecond);
                        SetWeight(Installer, bytes / InstallerBytesPerSecond);
                    });
                });

                // ── Workers ──
                await Step(Scripts, () => Task.Run(() =>
                {
                    Report($"Clearing {output}", 0f);
                    ClearOutputFolder(output);
                    Report("C# scripts, Release", 0.2f);
                    var build = ScriptCompiler.BuildForPublish(assets)
                        ?? throw new InvalidOperationException("The scripts do not compile: the errors are in the console");
                    if (build.Pe != null)
                    {
                        PackBuilder.WriteScripts(build, output);
                        result.Scripts = build.ScriptCount;
                    }
                }));

                string publish = "";
                await Step(Player, () => Task.Run(() => publish = PlayerBuild.Build(playerProject, staging, Report)));

                await Step(Copying, () => Task.Run(() =>
                {
                    result.Exe = Path.Combine(output, ExeName(result.Product));
                    (result.PlayerFiles, result.PlayerBytes) = PlayerBuild.Copy(publish, output, Path.GetFileName(result.Exe), Report);
                    Report($"Visual C++ runtime {runtime.Version}", 1f);
                    result.PlayerBytes += VCRuntime.Copy(runtime.Folder, output);
                    result.PlayerFiles += VCRuntime.Files.Length;
                    result.Runtime = runtime.Version;
                }));

                await Step(Packs, () => Task.Run(() =>
                {
                    var data = Path.Combine(output, PackBuilder.DataFolderName);
                    result.Packs = PackBuilder.Write(data, job, (detail, f) => Report(detail, 0.98f * f));
                    result.Warnings.AddRange(result.Packs.Problems);
                    (result.EngineFiles, result.EngineBytes) = PackBuilder.WriteEngineFiles(
                        PlayerBuild.EngineResources(playerProject), data, PlayerBuild.UnusedResources, (detail, f) => Report(detail, 0.98f + 0.02f * f));
                }));

                if (options.Verify)
                    await Step(Verifying, () => Task.Run(() =>
                    {
                        int before = result.Packs.Problems.Count;
                        PackBuilder.Verify(job, result.Packs, Report);
                        result.Warnings.AddRange(result.Packs.Problems.Skip(before));
                    }));

                if (iscc != null)
                    await Step(Installer, () => Task.Run(() =>
                    {
                        result.Installer = InstallerBuild.Build(iscc, profile, result.Product, output, Path.GetFileName(result.Exe), installerScripts,
                                                                PlayerBuild.AppIcon(playerProject), Report);
                        var folder = Path.GetDirectoryName(result.Installer)!;
                        result.InstallerBytes = Directory.GetFiles(folder, InstallerBuild.SetupName(result.Product) + "*").Sum(f => new FileInfo(f).Length);
                    }));

                result.Elapsed = progress.Elapsed;
                result.Steps = progress.StepTimes;
                File.AppendAllText(result.Report, Summary(result));
                Debug.LogAlways($"[Publish] {result.Product}: {BuildScan.Size(result.Packs.Bytes)} in {result.Packs.Packs.Count} packs, " +
                                $"player {BuildScan.Size(result.PlayerBytes)}, {result.Warnings.Count} warnings, " +
                                $"{result.Elapsed.TotalSeconds:0} s ({string.Join(", ", result.Steps.Select(s => $"{s.Step} {s.Seconds:0.0} s"))}): {result.Exe}");

                await progress.Finish($"Published {result.Product}");
                if (options.Run)
                    Process.Start(new ProcessStartInfo(result.Exe) { WorkingDirectory = output, UseShellExecute = true });
                return result;
            }
            catch (Exception ex)
            {
                var error = ProgressJob.Unwrap(ex);
                await progress.Fail($"Could not publish {result.Product}", error);
                throw error;
            }
        }

        /// <summary>
        /// Refused as an output folder: a folder that is not empty and has no marker (Publish clears only what it made),
        /// a drive, the project's folders and any folder holding the project or the editor. Main thread, before the popup.
        /// </summary>
        static void CheckOutputFolder(string output, string projectRoot)
        {
            var full = Path.GetFullPath(output).TrimEnd('\\', '/');
            static bool SameOrInside(string path, string folder)
            {
                var f = Path.GetFullPath(folder).TrimEnd('\\', '/');
                return path.Equals(f, StringComparison.OrdinalIgnoreCase) || path.StartsWith(f + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
            }

            if (string.Equals(Path.GetPathRoot(full)?.TrimEnd('\\', '/'), full, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException($"{output} is a drive: choose a folder");
            foreach (var guarded in new[] { "Assets", "Library", "Cache" })
                if (SameOrInside(full, Path.Combine(projectRoot, guarded)))
                    throw new InvalidOperationException($"{output} is inside the project's {guarded} folder: choose another output folder");
            if (SameOrInside(Path.GetFullPath(projectRoot).TrimEnd('\\', '/'), full) || SameOrInside(AppContext.BaseDirectory.TrimEnd('\\', '/'), full))
                throw new InvalidOperationException($"{output} holds the project or the editor: choose another output folder");
            if (Directory.Exists(full) && Directory.EnumerateFileSystemEntries(full).Any() && !File.Exists(Path.Combine(full, MarkerFile)))
                throw new InvalidOperationException($"{output} is not empty and was not made by Publish (no {MarkerFile}): choose another output folder");
        }

        /// <summary>The output folder (checked by <see cref="CheckOutputFolder"/>) made empty for the game, but for its marker. Worker.</summary>
        static void ClearOutputFolder(string output)
        {
            var marker = Path.Combine(output, MarkerFile);
            Directory.CreateDirectory(output);
            try
            {
                // Everything of the last publish goes (an old exe name, DLLs the player no longer has)
                foreach (var dir in Directory.GetDirectories(output))
                    Directory.Delete(dir, recursive: true);
                foreach (var file in Directory.GetFiles(output))
                    if (!string.Equals(file, marker, StringComparison.OrdinalIgnoreCase))
                        File.Delete(file);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                throw new InvalidOperationException($"Could not clear {output}: {ex.Message} Is the game running from there?");
            }
            File.WriteAllText(marker, "Written by the Freefall editor's Publish. Its contents are replaced on every publish.\n");
        }

        /// <summary>The game's exe: the product's name without characters a file name cannot have.</summary>
        static string ExeName(string product)
        {
            var name = string.Concat(product.Split(Path.GetInvalidFileNameChars())).Trim().TrimEnd('.');
            return (name.Length > 0 ? name : "Game") + ".exe";
        }

        static string Names(List<Asset> assets) =>
            string.Join(", ", assets.Take(6).Select(a => $"{a.Name} ({a.GetType().Name})")) + (assets.Count > 6 ? $" and {assets.Count - 6} more" : "");

        static string Summary(PublishResult r)
        {
            var sb = new StringBuilder();
            sb.AppendLine();
            sb.AppendLine($"Publish ({DateTime.Now:yyyy-MM-dd HH:mm:ss}): {r.Elapsed.TotalSeconds:0.0} s");
            sb.AppendLine($"  Output: {r.Output}");
            sb.AppendLine($"  Exe: {Path.GetFileName(r.Exe)}");
            sb.AppendLine($"  Player: {r.PlayerFiles:N0} files, {BuildScan.Size(r.PlayerBytes)}");
            sb.AppendLine($"  Visual C++ runtime: {r.Runtime}, beside the exe ({string.Join(", ", VCRuntime.Files)})");
            sb.AppendLine($"  Scripts: {r.Scripts} files in {PackBuilder.ScriptsFile}");
            sb.AppendLine($"  Engine files: {r.EngineFiles} in {Freefall.Assets.EngineFiles.PackFile}, {BuildScan.Size(r.EngineBytes)}");
            sb.AppendLine($"  Packs: {r.Packs.Entries:N0} artifacts, {BuildScan.Size(r.Packs.Bytes)}" +
                          (r.Packs.Verified ? $", verified: {r.Packs.Mismatches} mismatches" : ", not verified"));
            foreach (var (file, entries, bytes) in r.Packs.Packs)
                sb.AppendLine($"    {file}: {entries:N0} artifacts, {BuildScan.Size(bytes)}");
            if (r.Installer.Length > 0)
                sb.AppendLine($"  Installer: {r.Installer}, {BuildScan.Size(r.InstallerBytes)}");
            sb.AppendLine("  Steps:");
            foreach (var (step, seconds) in r.Steps)
                sb.AppendLine($"    {seconds,6:0.0} s  {step}");
            sb.AppendLine($"  Problems while writing ({r.Packs.Problems.Count}):");
            foreach (var p in r.Packs.Problems)
                sb.AppendLine("    " + p);
            if (r.Packs.Problems.Count == 0) sb.AppendLine("    none");
            return sb.ToString();
        }
    }
}
