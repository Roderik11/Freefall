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
                var dir = new DirectoryInfo(AppContext.BaseDirectory);
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

        private static readonly Regex IncludeRegex = new(@"#include\s+""([^""]+)""", RegexOptions.Compiled);

        private static void Collect(string path, HashSet<string> set)
        {
            if (!set.Add(path) || !File.Exists(path)) return;

            string text;
            try { text = File.ReadAllText(path); }
            catch (IOException) { return; }

            string dir = Path.GetDirectoryName(path) ?? "";
            foreach (Match m in IncludeRegex.Matches(text))
                Collect(Path.GetFullPath(Path.Combine(dir, m.Groups[1].Value)), set);
        }

        internal static void Register(ComputeShader shader)
        {
            lock (_lock)
                _computeShaders.Add(new WeakReference<ComputeShader>(shader));
            EnsureWatching();
        }

        internal static void EnsureWatching()
        {
            if (_watcher != null) return;
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

            // Effects: every instance of a name shares its compiled state with the master
            foreach (var effect in new List<Effect>(Effect.MasterEffects.Values))
            {
                if (!DependsOn(effect.Dependencies, changed)) continue;

                try
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    effect.Reload();
                    int materials = Material.RebuildForEffect(effect.Name);
                    Debug.LogAlways($"[ShaderHotReload] Reloaded {effect.Name}.fx ({materials} material(s), {sw.ElapsedMilliseconds} ms)");
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
                        Debug.LogAlways($"[ShaderHotReload] Reloaded {cs.FileName}");
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
