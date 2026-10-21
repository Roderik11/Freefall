using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;

namespace Freefall.Graphics
{
    /// <summary>
    /// Resolves where engine shaders are loaded from and recompiles them when a file changes.
    ///
    /// In a development checkout the shaders are read straight from the repository
    /// (Freefall.Engine/Resources/Shaders) instead of the copy in the build output, so an edit shows up
    /// without rebuilding. A FileSystemWatcher queues changed files; <see cref="Update"/> (main thread,
    /// between frames) recompiles every Effect / ComputeShader that includes one of them and rebuilds
    /// the pipeline states of the materials using it. A shader that fails to compile keeps running
    /// with its previous version and the error goes to the console.
    /// </summary>
    public static class ShaderHotReload
    {
        // Wait for the editor to finish writing (save = truncate + write, or temp file + rename)
        private const long DebounceMs = 150;

        private static readonly object _lock = new();
        private static readonly List<WeakReference<ComputeShader>> _computeShaders = new();
        private static readonly ConcurrentDictionary<string, long> _pending = new(StringComparer.OrdinalIgnoreCase);
        private static FileSystemWatcher? _watcher;
        private static string? _directory;
        private static bool _isSourceDirectory;

        // Numbers the reload lines: the console drops a line equal to the one before it, so a second
        // "Reloaded x.hlsl" in a row would not show and the reload would look ignored
        private static int _reloads;

        /// <summary>
        /// Off in a published game (Engine.Initialize with editor: false), before any shader loads: engine shaders come
        /// from &lt;exe&gt;/Resources/Shaders only (no search for a source checkout above the exe, which a build inside the
        /// repository would find), and no folder is watched.
        /// </summary>
        public static bool Enabled { get; set; } = true;

        /// <summary>Directory engine shaders are loaded from.</summary>
        public static string Directory
        {
            get
            {
                if (_directory == null) ResolveDirectory();
                return _directory!;
            }
        }

        /// <summary>True when shaders come from the repository checkout rather than the build output.</summary>
        public static bool IsSourceDirectory
        {
            get
            {
                if (_directory == null) ResolveDirectory();
                return _isSourceDirectory;
            }
        }

        private static void ResolveDirectory()
        {
            lock (_lock)
            {
                if (_directory != null) return;

                // Development layout: <repo>/Freefall.Editor/bin/<config>/<tfm>/ → walk up to the repo
                var dir = Enabled ? new DirectoryInfo(AppContext.BaseDirectory) : null;
                for (int i = 0; i < 8 && dir != null; i++, dir = dir.Parent)
                {
                    string candidate = Path.Combine(dir.FullName, "Freefall.Engine", "Resources", "Shaders");
                    if (File.Exists(Path.Combine(candidate, "common.fx")))
                    {
                        _isSourceDirectory = true;
                        _directory = candidate;
                        return;
                    }
                }

                _directory = Path.Combine(AppContext.BaseDirectory, "Resources", "Shaders");
            }
        }

        /// <summary>
        /// Full path of a shader file (with extension), or null if it does not exist.
        /// </summary>
        public static string? ResolvePath(string fileName)
        {
            string path = Path.Combine(Directory, fileName);
            if (File.Exists(path)) return path;

            // Build output, then next to the executable (dev convenience, as before)
            path = Path.Combine(AppContext.BaseDirectory, "Resources", "Shaders", fileName);
            if (File.Exists(path)) return path;

            path = Path.Combine(AppContext.BaseDirectory, fileName);
            return File.Exists(path) ? path : null;
        }

        /// <summary>
        /// The file itself plus everything it #includes, recursively (full paths).
        /// Used to know which shaders to recompile when a shared include changes.
        /// </summary>
        public static HashSet<string> CollectDependencies(string path)
        {
            var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            Collect(Path.GetFullPath(path), set);
            return set;
        }

        // At the start of a line, as Effect.Flatten reads them (an "#include" quoted in a comment is none)
        private static readonly Regex IncludeRegex = new(@"^[ \t]*#include\s+""([^""]+)""", RegexOptions.Compiled | RegexOptions.Multiline);

        /// <summary>
        /// Where an effect's <c>#include "file"</c> comes from: next to the including file first, then the engine's
        /// shader directory, so a project shader in Assets can include "common.fx" (full path; null if neither has it).
        /// </summary>
        public static string? ResolveInclude(string includingDirectory, string file)
        {
            string local = Path.GetFullPath(Path.Combine(includingDirectory, file));
            if (File.Exists(local)) return local;
            string engine = Path.GetFullPath(Path.Combine(Directory, file));
            return File.Exists(engine) ? engine : null;
        }

        private static void Collect(string path, HashSet<string> set)
        {
            if (!set.Add(path) || !File.Exists(path)) return;

            string text;
            try { text = File.ReadAllText(path); }
            catch (IOException) { return; }

            string dir = Path.GetDirectoryName(path) ?? "";
            foreach (Match m in IncludeRegex.Matches(text))
            {
                // One not found yet stays in the set where it would be made next to the file, so making it reloads
                var file = m.Groups[1].Value;
                Collect(ResolveInclude(dir, file) ?? Path.GetFullPath(Path.Combine(dir, file)), set);
            }
        }

        internal static void Register(ComputeShader shader)
        {
            lock (_lock)
                _computeShaders.Add(new WeakReference<ComputeShader>(shader));
            EnsureWatching();
        }

        // A project's Assets: its .fx assets (Effect.FromAsset) and the files they include
        private static FileSystemWatcher? _projectWatcher;
        private static string? _projectDirectory;

        /// <summary>Raised on the main thread after a new .fx in the project's Assets was imported (the editor shows it).</summary>
        public static event Action? ProjectEffectAdded;

        /// <summary>
        /// Watch a project's Assets folder for shader files too: saving a project .fx (or a file it includes)
        /// recompiles it like an engine shader, and a new .fx is imported. AssetDatabase.Initialize calls it;
        /// another project replaces the folder.
        /// </summary>
        public static void WatchProject(string assetsDirectory)
        {
            if (!Enabled) return;
            lock (_lock)
            {
                string full = Path.GetFullPath(assetsDirectory);
                if (string.Equals(full, _projectDirectory, StringComparison.OrdinalIgnoreCase)) return;

                _projectWatcher?.Dispose();
                _projectWatcher = null;
                _projectDirectory = full;
                if (!System.IO.Directory.Exists(full)) return;

                try
                {
                    var watcher = new FileSystemWatcher(full)
                    {
                        IncludeSubdirectories = true,
                        NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.FileName | NotifyFilters.Size,
                    };
                    watcher.Changed += (_, e) => Queue(e.FullPath);
                    watcher.Created += (_, e) => Queue(e.FullPath);
                    watcher.Renamed += (_, e) => Queue(e.FullPath);
                    watcher.EnableRaisingEvents = true;
                    _projectWatcher = watcher;
                    Debug.Log("ShaderHotReload", $"Watching project shaders in {full}");
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("ShaderHotReload", $"Could not watch {full}: {ex.Message}");
                }
            }
        }

        private static bool IsInProject(string path) =>
            _projectDirectory != null && path.StartsWith(_projectDirectory + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);

        internal static void EnsureWatching()
        {
            if (_watcher != null || !Enabled) return;
            lock (_lock)
            {
                if (_watcher != null) return;
                try
                {
                    var watcher = new FileSystemWatcher(Directory)
                    {
                        IncludeSubdirectories = true,
                        NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.FileName | NotifyFilters.Size,
                    };
                    watcher.Changed += (_, e) => Queue(e.FullPath);
                    watcher.Created += (_, e) => Queue(e.FullPath);
                    watcher.Renamed += (_, e) => Queue(e.FullPath);
                    watcher.EnableRaisingEvents = true;
                    _watcher = watcher;
                    Debug.Log("ShaderHotReload", $"Watching {Directory}{(_isSourceDirectory ? " (source checkout)" : "")}");
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("ShaderHotReload", $"Could not watch {Directory}: {ex.Message}");
                }
            }
        }

        private static void Queue(string path)
        {
            string ext = Path.GetExtension(path);
            if (!ext.Equals(".fx", StringComparison.OrdinalIgnoreCase) &&
                !ext.Equals(".fxh", StringComparison.OrdinalIgnoreCase) &&
                !ext.Equals(".hlsl", StringComparison.OrdinalIgnoreCase) &&
                !ext.Equals(".hlsli", StringComparison.OrdinalIgnoreCase))
                return;

            _pending[Path.GetFullPath(path)] = Environment.TickCount64;
        }

        /// <summary>
        /// Recompile shaders whose files changed. Main thread only, between frames
        /// (pipeline states and constant buffers of live materials are replaced).
        /// </summary>
        public static void Update()
        {
            if (_pending.IsEmpty) return;

            long now = Environment.TickCount64;
            List<string>? changed = null;
            foreach (var (path, stamp) in _pending)
            {
                if (now - stamp < DebounceMs) continue;
                if (_pending.TryRemove(path, out _))
                    (changed ??= new List<string>()).Add(path);
            }
            if (changed == null) return;

            // A new .fx in the project's Assets becomes an asset (a GUID, an Effect card); it compiles when a
            // material first uses it
            ImportNewProjectEffects(changed);

            // Effects: every instance of a name shares its compiled state with the master
            foreach (var effect in new List<Effect>(Effect.MasterEffects.Values))
            {
                if (!DependsOn(effect.Dependencies, changed)) continue;

                try
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    effect.Reload();
                    int materials = Material.RebuildForEffect(effect);
                    Debug.LogAlways($"[ShaderHotReload] Reloaded {effect.Name}.fx ({materials} material(s), {sw.ElapsedMilliseconds} ms, reload {++_reloads})");

                    // A project effect's own file changed: import it, so its cached source (what the next start
                    // loads) is the one now running. EffectLoader.Reload then finds it compiled and does nothing.
                    if (effect.SourcePath != null && IsInProject(effect.SourcePath) && changed.Exists(p => string.Equals(p, effect.SourcePath, StringComparison.OrdinalIgnoreCase)))
                        Freefall.Assets.AssetDatabase.ImportSavedFile(effect.SourcePath);
                }
                catch (IOException)
                {
                    Requeue(changed);   // file still being written — try again next tick
                }
                catch (Exception ex)
                {
                    Debug.LogError("ShaderHotReload", $"{effect.Name}.fx not reloaded, keeping the previous version: {ex.Message}");
                }
            }

            List<ComputeShader> computeShaders = new();
            lock (_lock)
            {
                for (int i = _computeShaders.Count - 1; i >= 0; i--)
                {
                    if (_computeShaders[i].TryGetTarget(out var cs)) computeShaders.Add(cs);
                    else _computeShaders.RemoveAt(i);
                }
            }

            foreach (var cs in computeShaders)
            {
                if (!DependsOn(cs.Dependencies, changed)) continue;

                try
                {
                    if (cs.Reload())
                        Debug.LogAlways($"[ShaderHotReload] Reloaded {cs.FileName} (reload {++_reloads})");
                }
                catch (IOException)
                {
                    Requeue(changed);
                }
                catch (Exception ex)
                {
                    Debug.LogError("ShaderHotReload", $"{cs.FileName} not reloaded, keeping the previous version: {ex.Message}");
                }
            }
        }

        private static void ImportNewProjectEffects(List<string> changed)
        {
            if (Freefall.Assets.AssetDatabase.Project == null) return;

            List<string>? added = null;
            foreach (var path in changed)
            {
                if (!IsInProject(path) || !Path.GetExtension(path).Equals(".fx", StringComparison.OrdinalIgnoreCase) || !File.Exists(path))
                    continue;
                var relative = Path.GetRelativePath(Freefall.Assets.AssetDatabase.Project.AssetsDirectory, path);
                if (Freefall.Assets.AssetDatabase.PathToGuid(relative) == null)
                    (added ??= new List<string>()).Add(path);
            }
            if (added == null) return;

            try
            {
                Freefall.Assets.AssetDatabase.Refresh();
                foreach (var path in added)
                {
                    Freefall.Assets.AssetDatabase.ImportSavedFile(path);
                    Debug.Log("ShaderHotReload", $"Imported new effect {Path.GetFileName(path)}");
                }
                ProjectEffectAdded?.Invoke();
            }
            catch (Exception ex)
            {
                Debug.LogWarning("ShaderHotReload", $"Could not import new effect(s): {ex.Message}");
            }
        }

        private static bool DependsOn(HashSet<string>? dependencies, List<string> changed)
        {
            if (dependencies == null) return false;
            foreach (var path in changed)
                if (dependencies.Contains(path)) return true;
            return false;
        }

        private static void Requeue(List<string> paths)
        {
            long now = Environment.TickCount64;
            foreach (var path in paths)
                _pending[path] = now;
        }
    }
}
