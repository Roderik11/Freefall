using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Loader;
using System.Threading;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Emit;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Reflection;
using Freefall.Serialization;

namespace Freefall.Editor
{
    /// <summary>
    /// Compiles loose .cs files and loads pre-compiled .dll plugins from
    /// the project Assets/ folder using Roslyn + collectible AssemblyLoadContext.
    /// Supports full hot-reload: unload old → recompile → load new.
    /// </summary>
    public static class ScriptCompiler
    {
        private static ScriptLoadContext? _context;
        private static Assembly? _compiledAssembly;
        private static readonly List<Assembly> _pluginAssemblies = new();
        private static FileSystemWatcher? _watcher;
        private static Timer? _debounceTimer;
        private static string? _devenvPath;
        private static string? _assetsDirectory;

        /// <summary>
        /// Open a file in Visual Studio with the FreefallScripts project.
        /// Uses vswhere to locate the latest VS install.
        /// </summary>
        public static void OpenInVisualStudio(string filePath)
        {
            _devenvPath ??= FindDevenv();
            if (_devenvPath == null)
            {
                // Fallback: shell execute
                try { System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(filePath) { UseShellExecute = true }); }
                catch { }
                return;
            }

            var csproj = _assetsDirectory != null
                ? Path.Combine(Path.GetDirectoryName(_assetsDirectory)!, "ScriptProject", "FreefallScripts.csproj")
                : null;

            // If project exists, open project + file. Otherwise just the file.
            var arguments = csproj != null && File.Exists(csproj)
                ? $"\"{csproj}\" \"{filePath}\""
                : $"\"{filePath}\"";

            try
            {
                System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
                {
                    FileName = _devenvPath,
                    Arguments = arguments,
                    UseShellExecute = false
                });
            }
            catch (Exception ex)
            {
                Debug.Log($"[ScriptCompiler] Failed to open VS: {ex.Message}");
            }
        }

        private static string? FindDevenv()
        {
            var vswhere = @"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe";
            if (!File.Exists(vswhere)) return null;

            try
            {
                var psi = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = vswhere,
                    Arguments = "-latest -property installationPath",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                var proc = System.Diagnostics.Process.Start(psi);
                var output = proc?.StandardOutput.ReadToEnd().Trim();
                proc?.WaitForExit();

                if (!string.IsNullOrEmpty(output))
                {
                    var devenv = Path.Combine(output, "Common7", "IDE", "devenv.exe");
                    if (File.Exists(devenv))
                        return devenv;
                }
            }
            catch { }

            return null;
        }


        /// <summary>All script/plugin types currently loaded.</summary>
        public static IReadOnlyList<Assembly> LoadedAssemblies
        {
            get
            {
                var list = new List<Assembly>(_pluginAssemblies);
                if (_compiledAssembly != null) list.Add(_compiledAssembly);
                return list;
            }
        }

        /// <summary>
        /// True for a script/plugin assembly from an earlier build: unloaded but possibly not collected yet, so it
        /// still shows up in AppDomain.GetAssemblies(). Type scans must skip it (and caches must not keep it).
        /// </summary>
        public static bool IsStale(Assembly assembly)
            => assembly.IsCollectible && !LoadedAssemblies.Contains(assembly);

        /// <summary>
        /// Scan Assets/ for .cs and .dll files, compile/load them, register with Reflector.
        /// Call once after project is opened.
        /// </summary>
        public static void Initialize(string assetsDirectory)
        {
            _assetsDirectory = assetsDirectory;
            GenerateScriptProject(assetsDirectory);
            CompileAndLoad(assetsDirectory);
            StartWatching(assetsDirectory);
        }

        /// <summary>
        /// Generate a FreefallScripts.csproj in the Assets/ folder for IDE IntelliSense.
        /// References the engine assembly and all its dependencies.
        /// </summary>
        private static void GenerateScriptProject(string assetsDirectory)
        {
            var engineDll = typeof(Component).Assembly.Location;

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("<Project Sdk=\"Microsoft.NET.Sdk\">");
            sb.AppendLine("  <PropertyGroup>");
            sb.AppendLine("    <TargetFramework>net10.0-windows</TargetFramework>");
            sb.AppendLine("    <ImplicitUsings>enable</ImplicitUsings>");
            sb.AppendLine("    <Nullable>enable</Nullable>");
            sb.AppendLine("    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>");
            sb.AppendLine("    <EnableDefaultItems>false</EnableDefaultItems>");
            sb.AppendLine("  </PropertyGroup>");
            sb.AppendLine("  <ItemGroup>");
            sb.AppendLine("    <Compile Include=\"..\\Assets\\**\\*.cs\" />");
            sb.AppendLine("  </ItemGroup>");
            sb.AppendLine("  <ItemGroup>");
            sb.AppendLine($"    <Reference Include=\"Freefall.Engine\" HintPath=\"{engineDll}\" />");
            sb.AppendLine("  </ItemGroup>");
            sb.AppendLine("</Project>");

            // Place in ScriptProject/ alongside Assets/Library/Cache
            var projectRoot = Path.GetDirectoryName(assetsDirectory)!;
            var scriptProjectDir = Path.Combine(projectRoot, "ScriptProject");
            Directory.CreateDirectory(scriptProjectDir);
            var projectPath = Path.Combine(scriptProjectDir, "FreefallScripts.csproj");
            var content = sb.ToString();

            // Only write if changed to avoid triggering FileSystemWatcher
            if (!File.Exists(projectPath) || File.ReadAllText(projectPath) != content)
            {
                File.WriteAllText(projectPath, content);
                Debug.Log("[ScriptCompiler] Generated FreefallScripts.csproj for IntelliSense");
            }
        }

        /// <summary>
        /// Full hot-reload: recompile off the main thread; if that succeeds, swap assemblies on the main thread and
        /// migrate the live script components onto the new types. A failed compile keeps the current scripts.
        /// </summary>
        public static void Reload()
        {
            var assetsDir = Engine.Project?.AssetsDirectory;
            if (assetsDir == null) return;

            Debug.Log("[ScriptCompiler] Reloading scripts...");

            BuildResult? build;
            int generation;
            lock (_buildLock)
            {
                generation = ++_buildGeneration;
                build = Build(assetsDir);
            }

            if (build == null)
            {
                Debug.LogWarning("ScriptCompiler", "Scripts not reloaded — the scene keeps running the previous build until the errors are fixed");
                return;
            }

            Engine.RunOnMainThreadAsync(() =>
            {
                // A newer change already compiled (or is compiling): only the latest build gets swapped in
                if (generation != Volatile.Read(ref _buildGeneration)) return;
                Swap(build);
            });
        }

        /// <summary>
        /// Unload all script assemblies and their ALC, without migrating components (shutdown).
        /// </summary>
        public static void Unload()
        {
            foreach (var asm in LoadedAssemblies)
                Reflector.UnregisterAssembly(asm);

            _compiledAssembly = null;
            _pluginAssemblies.Clear();

            if (_context != null)
            {
                _context.Unload();
                _context = null;
            }
        }

        public static void Shutdown()
        {
            _watcher?.Dispose();
            _watcher = null;
            _debounceTimer?.Dispose();
            _debounceTimer = null;
            Unload();
        }

        // ── Core ─────────────────────────────────────────────

        private static readonly object _buildLock = new();
        private static int _buildGeneration;

        /// <summary>Output of a compile pass, ready to load into a fresh context.</summary>
        private sealed class BuildResult
        {
            public string[] PluginPaths = [];
            public byte[]? Pe;
            public byte[]? Pdb;
            public int ScriptCount;
        }

        private static void CompileAndLoad(string assetsDirectory)
        {
            // A failed first compile still loads the plugins
            var build = Build(assetsDirectory) ?? new BuildResult { PluginPaths = FindPlugins(assetsDirectory) };
            Swap(build);
        }

        private static string[] FindPlugins(string assetsDirectory)
            => Directory.GetFiles(assetsDirectory, "*.dll", SearchOption.AllDirectories).Select(Path.GetFullPath).ToArray();

        /// <summary>
        /// Compile loose .cs files to memory. Thread-agnostic: touches no live state. Null on compile errors.
        /// </summary>
        private static BuildResult? Build(string assetsDirectory)
        {
            var build = new BuildResult { PluginPaths = FindPlugins(assetsDirectory) };

            var csFiles = Directory.GetFiles(assetsDirectory, "*.cs", SearchOption.AllDirectories);
            build.ScriptCount = csFiles.Length;
            if (csFiles.Length == 0)
            {
                Debug.Log("[ScriptCompiler] No .cs files found in Assets/");
                return build;
            }

            if (!Compile(csFiles, build))
                return null;
            return build;
        }

        /// <summary>
        /// Main thread: replace the loaded script assemblies with a build and move every live component of a script
        /// type onto the same-named type of the new build (see ScriptComponentMigrator).
        /// </summary>
        private static void Swap(BuildResult build)
        {
            var oldAssemblies = LoadedAssemblies.ToList();

            // 1. Snapshot + remove the old instances while their code is still loaded (Destroy() runs old code)
            var migrator = ScriptComponentMigrator.Capture(oldAssemblies);
            migrator.Detach();

            // The editor's own references to assets of a script-defined class: the dirty list, and the
            // selection with the inspector controls built on it
            var unsavedAssets = AssetCreator.DetachScriptAssets(oldAssemblies);
            (string Guid, string TypeName)? shownAsset = null;
            if (Selector.SelectedObject is Asset shown && Reflector.ReferencesAssembly(shown.GetType(), oldAssemblies))
            {
                shownAsset = (shown.Guid, shown.GetType().FullName!);
                Selector.SelectedObject = null;
            }

            // 2. Forget everything keyed by the old types (cached assets of script classes included), then
            //    start unloading their context
            foreach (var asm in oldAssemblies)
                Reflector.UnregisterAssembly(asm);
            ScriptComponentMigrator.ForgetTypes(oldAssemblies);

            _compiledAssembly = null;
            _pluginAssemblies.Clear();
            WeakReference? oldContext = null;
            if (_context != null)
            {
                oldContext = new WeakReference(_context);
                _context.Unload();
                _context = null;
            }

            // 3. Load the new build
            _context = new ScriptLoadContext();

            foreach (var dll in build.PluginPaths)
            {
                try
                {
                    var asm = _context.LoadFromAssemblyPath(dll);
                    _pluginAssemblies.Add(asm);
                    Reflector.RegisterAssemblies(asm);
                    Debug.Log($"[ScriptCompiler] Loaded plugin: {Path.GetFileName(dll)}");
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("ScriptCompiler", $"Failed to load {Path.GetFileName(dll)}: {ex.Message}");
                }
            }

            if (build.Pe != null)
            {
                using var pe = new MemoryStream(build.Pe);
                using var pdb = build.Pdb != null ? new MemoryStream(build.Pdb) : null;
                var assembly = _context.LoadFromStream(pe, pdb);
                _compiledAssembly = assembly;
                Reflector.RegisterAssemblies(assembly);

                try
                {
                    var types = assembly.GetTypes();
                    Debug.Log($"[ScriptCompiler] Compiled {build.ScriptCount} script(s) → {types.Length} type(s)");
                }
                catch (ReflectionTypeLoadException ex)
                {
                    Debug.Log($"[ScriptCompiler] Compiled with partial type loading: {ex.Types.Count(t => t != null)} of {ex.Types.Length} types");
                }
            }

            // 4. Recreate the live components from the new types
            // Editor-side type caches: rebuilt so they neither miss new script types nor pin the old assembly
            GUIInspector.RebuildTypeMaps();
            PropertyFrame.RebuildTypeMaps();
            AssetDragData.ResetTypeMap();
            Commands.CreateAssetCommand.ResetTypeCache();
            Commands.AuthoringHelpers.ResetTypeCache();
            if (migrator.Count > 0)
            {
                int restored = migrator.Restore();
                Debug.LogAlways($"[ScriptCompiler] Scripts reloaded: migrated {restored} of {migrator.Count} live script component(s)");
            }

            MessageDispatcher.Send(Msg.ScriptsReloaded);
            MessageDispatcher.Send(Msg.RefreshInspector);

            // Script-class assets again, as instances of the new build
            AssetCreator.RestoreScriptAssets(unsavedAssets);
            if (shownAsset is (var guid, var typeName) && Reflector.GetType(typeName) is { } assetType
                && Engine.Assets.LoadByGuid(guid, assetType) is { } reloaded)
                Selector.SelectedObject = reloaded;

            if (oldContext != null)
                WatchUnload(oldContext);
        }

        /// <summary>
        /// Report whether the previous script context actually got collected. If it stays alive, something still
        /// references an old script type or instance (a static cache, an event handler a script never removed in
        /// Destroy(), a captured lambda...) and every reload leaks one assembly.
        /// </summary>
        private static void WatchUnload(WeakReference oldContext)
        {
            lock (_unloading) _unloading.Add(oldContext);

            System.Threading.Tasks.Task.Run(async () =>
            {
                await System.Threading.Tasks.Task.Delay(1000);
                for (int i = 0; i < 10 && oldContext.IsAlive; i++)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    if (oldContext.IsAlive)
                        await System.Threading.Tasks.Task.Delay(200);
                }

                int stillLoaded;
                lock (_unloading)
                {
                    _unloading.RemoveAll(w => !w.IsAlive);
                    stillLoaded = _unloading.Count;
                }

                if (oldContext.IsAlive)
                    Debug.LogWarning("ScriptCompiler", $"Previous script assembly is still referenced after reload and could not be unloaded ({stillLoaded} old script assemblies still loaded)");
                else if (stillLoaded > 0)
                    Debug.LogWarning("ScriptCompiler", $"Previous script assembly unloaded, but {stillLoaded} older one(s) are still loaded");
                else
                    Debug.LogAlways("[ScriptCompiler] Previous script assembly unloaded");
            });
        }

        // Contexts handed to Unload() that may not have been collected yet
        private static readonly List<WeakReference> _unloading = new();

        /// <summary>Read a source file, retrying briefly while an editor still holds it open mid-save.</summary>
        private static string? ReadSource(string file)
        {
            for (int attempt = 0; ; attempt++)
            {
                try
                {
                    return File.ReadAllText(file);
                }
                catch (FileNotFoundException) { return null; }
                catch (IOException) when (attempt < 5)
                {
                    Thread.Sleep(100);
                }
                catch (Exception)
                {
                    return null;
                }
            }
        }

        private static bool Compile(string[] csFiles, BuildResult build)
        {
            // Parse all source files
            var syntaxTrees = new List<SyntaxTree>(csFiles.Length);
            foreach (var file in csFiles)
            {
                var code = ReadSource(file);
                if (code == null)
                {
                    // Compiling without this file would make its types vanish and orphan their live components
                    Debug.LogWarning("ScriptCompiler", $"Could not read {Path.GetFileName(file)} (still being written?) — build skipped");
                    return false;
                }

                var parseOptions = CSharpParseOptions.Default.WithPreprocessorSymbols("DEBUG");
                var tree = CSharpSyntaxTree.ParseText(code, parseOptions, path: file, encoding: System.Text.Encoding.UTF8);
                syntaxTrees.Add(tree);
            }

            if (syntaxTrees.Count == 0) return false;

            // Build references — engine + all its dependencies + BCL
            var references = BuildReferences(build.PluginPaths);

            var compilation = CSharpCompilation.Create(
                assemblyName: "FreefallScripts",
                syntaxTrees: syntaxTrees,
                references: references,
                options: new CSharpCompilationOptions(
                    OutputKind.DynamicallyLinkedLibrary,
                    optimizationLevel: OptimizationLevel.Debug,
                    allowUnsafe: true
                )
            );

            // Emit to memory
            using var peStream = new MemoryStream();
            using var pdbStream = new MemoryStream();

            EmitResult result = compilation.Emit(peStream, pdbStream);

            if (!result.Success)
            {
                var errors = result.Diagnostics
                    .Where(d => d.Severity == DiagnosticSeverity.Error)
                    .ToList();

                Debug.LogWarning("ScriptCompiler", $"Compilation failed: {errors.Count} error(s)");
                foreach (var error in errors)
                {
                    var location = error.Location.GetMappedLineSpan();
                    var file = Path.GetFileName(location.Path);
                    var line = location.StartLinePosition.Line + 1;
                    Debug.LogWarning("ScriptCompiler", $"{file}({line}): {error.GetMessage()}");
                }
                return false;
            }

            // Warn about non-error diagnostics
            var warnings = result.Diagnostics
                .Where(d => d.Severity == DiagnosticSeverity.Warning)
                .ToList();

            if (warnings.Count > 0)
                Debug.Log($"[ScriptCompiler] {warnings.Count} warning(s)");

            build.Pe = peStream.ToArray();
            build.Pdb = pdbStream.ToArray();
            return true;
        }

        /// <summary>
        /// Build MetadataReferences from the engine assembly and all its loaded dependencies.
        /// This gives scripts access to all engine types, System.Numerics, etc.
        /// </summary>
        private static List<MetadataReference> BuildReferences(string[] pluginPaths)
        {
            var refs = new List<MetadataReference>();
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            // Seed with the engine assembly — scripts always reference this
            var engineAssembly = typeof(Component).Assembly;
            AddAssemblyAndDependencies(engineAssembly, refs, seen);

            // Also reference the plugin DLLs on disk (the ones the build will load) so scripts can use them
            foreach (var path in pluginPaths)
            {
                if (seen.Add(path))
                    refs.Add(MetadataReference.CreateFromFile(path));
            }

            return refs;
        }

        private static void AddAssemblyAndDependencies(Assembly root, List<MetadataReference> refs, HashSet<string> seen)
        {
            var queue = new Queue<Assembly>();
            queue.Enqueue(root);

            while (queue.Count > 0)
            {
                var asm = queue.Dequeue();
                if (string.IsNullOrEmpty(asm.Location)) continue;
                if (!seen.Add(asm.Location)) continue;

                refs.Add(MetadataReference.CreateFromFile(asm.Location));

                foreach (var dep in asm.GetReferencedAssemblies())
                {
                    try
                    {
                        var loaded = Assembly.Load(dep);
                        if (loaded != null)
                            queue.Enqueue(loaded);
                    }
                    catch { }
                }
            }
        }

        // ── File Watching ────────────────────────────────────

        private static void StartWatching(string assetsDirectory)
        {
            _watcher?.Dispose();
            _watcher = new FileSystemWatcher(assetsDirectory)
            {
                IncludeSubdirectories = true,
                NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.FileName | NotifyFilters.CreationTime,
                EnableRaisingEvents = true
            };

            _watcher.Changed += OnFileChanged;
            _watcher.Created += OnFileChanged;
            _watcher.Deleted += OnFileChanged;
            _watcher.Renamed += (s, e) => OnFileChanged(s, e);
        }

        private static void OnFileChanged(object sender, FileSystemEventArgs e)
        {
            var ext = Path.GetExtension(e.FullPath);
            if (!ext.Equals(".cs", StringComparison.OrdinalIgnoreCase) &&
                !ext.Equals(".dll", StringComparison.OrdinalIgnoreCase))
                return;

            // Debounce — wait 500ms after last change before reloading.
            // Multiple files often change in bursts (save all, git checkout, etc.)
            _debounceTimer?.Dispose();
            _debounceTimer = new Timer(_ =>
            {
                Debug.Log($"[ScriptCompiler] Detected change: {Path.GetFileName(e.FullPath)}");
                Reload();
            }, null, 500, Timeout.Infinite);
        }

        // ── Collectible ALC ──────────────────────────────────

        /// <summary>
        /// Collectible AssemblyLoadContext for script isolation and unloadability.
        /// </summary>
        private class ScriptLoadContext : AssemblyLoadContext
        {
            public ScriptLoadContext() : base(isCollectible: true) { }

            protected override Assembly? Load(AssemblyName assemblyName)
            {
                // Fall back to the default context for engine/BCL assemblies
                return null;
            }
        }
    }
}
