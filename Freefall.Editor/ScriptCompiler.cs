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
using Freefall.Base;
using Freefall.Reflection;

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
        /// Full hot-reload: unload everything, rescan, recompile, reload.
        /// </summary>
        public static void Reload()
        {
            var assetsDir = Engine.Project?.AssetsDirectory;
            if (assetsDir == null) return;

            Debug.Log("[ScriptCompiler] Reloading scripts...");
            Unload();
            CompileAndLoad(assetsDir);
        }

        /// <summary>
        /// Unload all script assemblies and their ALC.
        /// </summary>
        public static void Unload()
        {
            // Unregister from Reflector before unloading
            if (_compiledAssembly != null)
                Reflector.UnregisterAssembly(_compiledAssembly);

            foreach (var asm in _pluginAssemblies)
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

        private static void CompileAndLoad(string assetsDirectory)
        {
            _context = new ScriptLoadContext();

            // 1. Load pre-compiled DLLs
            var dlls = Directory.GetFiles(assetsDirectory, "*.dll", SearchOption.AllDirectories);
            foreach (var dll in dlls)
            {
                try
                {
                    var asm = _context.LoadFromAssemblyPath(Path.GetFullPath(dll));
                    _pluginAssemblies.Add(asm);
                    Reflector.RegisterAssemblies(asm);
                    Debug.Log($"[ScriptCompiler] Loaded plugin: {Path.GetFileName(dll)}");
                }
                catch (Exception ex)
                {
                    Debug.Log($"[ScriptCompiler] Failed to load {Path.GetFileName(dll)}: {ex.Message}");
                }
            }

            // 2. Compile loose .cs files
            var csFiles = Directory.GetFiles(assetsDirectory, "*.cs", SearchOption.AllDirectories);
            if (csFiles.Length == 0)
            {
                Debug.Log("[ScriptCompiler] No .cs files found in Assets/");
                return;
            }

            var assembly = Compile(csFiles);
            if (assembly != null)
            {
                _compiledAssembly = assembly;
                Reflector.RegisterAssemblies(assembly);

                try
                {
                    var types = assembly.GetTypes();
                    Debug.Log($"[ScriptCompiler] Compiled {csFiles.Length} script(s) → {types.Length} type(s)");
                }
                catch (ReflectionTypeLoadException ex)
                {
                    Debug.Log($"[ScriptCompiler] Compiled with partial type loading: {ex.Types.Count(t => t != null)} of {ex.Types.Length} types");
                }
            }

            MessageDispatcher.Send(Msg.ScriptsReloaded);
        }

        private static Assembly? Compile(string[] csFiles)
        {
            // Parse all source files
            var syntaxTrees = new List<SyntaxTree>(csFiles.Length);
            foreach (var file in csFiles)
            {
                try
                {
                    var code = File.ReadAllText(file);
                    var parseOptions = CSharpParseOptions.Default.WithPreprocessorSymbols("DEBUG");
                    var tree = CSharpSyntaxTree.ParseText(code, parseOptions, path: file, encoding: System.Text.Encoding.UTF8);
                    syntaxTrees.Add(tree);
                }
                catch (Exception ex)
                {
                    Debug.Log($"[ScriptCompiler] Failed to read {Path.GetFileName(file)}: {ex.Message}");
                }
            }

            if (syntaxTrees.Count == 0) return null;

            // Build references — engine + all its dependencies + BCL
            var references = BuildReferences();

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

                Debug.Log($"[ScriptCompiler] Compilation failed: {errors.Count} error(s)");
                foreach (var error in errors)
                {
                    var location = error.Location.GetMappedLineSpan();
                    var file = Path.GetFileName(location.Path);
                    var line = location.StartLinePosition.Line + 1;
                    Debug.Log($"  {file}({line}): {error.GetMessage()}");
                }
                return null;
            }

            // Warn about non-error diagnostics
            var warnings = result.Diagnostics
                .Where(d => d.Severity == DiagnosticSeverity.Warning)
                .ToList();

            if (warnings.Count > 0)
                Debug.Log($"[ScriptCompiler] {warnings.Count} warning(s)");

            peStream.Seek(0, SeekOrigin.Begin);
            pdbStream.Seek(0, SeekOrigin.Begin);

            return _context!.LoadFromStream(peStream, pdbStream);
        }

        /// <summary>
        /// Build MetadataReferences from the engine assembly and all its loaded dependencies.
        /// This gives scripts access to all engine types, System.Numerics, etc.
        /// </summary>
        private static List<MetadataReference> BuildReferences()
        {
            var refs = new List<MetadataReference>();
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            // Seed with the engine assembly — scripts always reference this
            var engineAssembly = typeof(Component).Assembly;
            AddAssemblyAndDependencies(engineAssembly, refs, seen);

            // Also add loaded plugin DLLs so scripts can reference each other
            foreach (var plugin in _pluginAssemblies)
                AddAssemblyAndDependencies(plugin, refs, seen);

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
