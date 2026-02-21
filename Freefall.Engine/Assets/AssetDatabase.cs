using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Freefall.Assets
{
    /// <summary>
    /// Central asset registry. Tracks all known assets by GUID, manages meta files,
    /// and triggers imports. Editor-facing — the game loads from Cache by name.
    /// 
    /// NOTE: Currently uses JSON for .meta files. Will switch to YAML once
    /// the YAML serializer is ported. Serialization is isolated to
    /// ReadMetaFile() / WriteMetaFile() for easy swap.
    /// </summary>
    public static class AssetDatabase
    {
        public static FreefallProject Project { get; private set; }

        // Source GUID → relative source path
        private static readonly Dictionary<string, string> _guidToPath = new(StringComparer.OrdinalIgnoreCase);
        // Relative source path → source GUID
        private static readonly Dictionary<string, string> _pathToGuid = new(StringComparer.OrdinalIgnoreCase);
        // Source GUID → full meta data
        private static readonly Dictionary<string, MetaFile> _guidToMeta = new(StringComparer.OrdinalIgnoreCase);
        // Subasset GUID → source GUID (for reverse lookup)
        private static readonly Dictionary<string, string> _subAssetToSource = new(StringComparer.OrdinalIgnoreCase);
        // Name → subasset GUID (for user-facing Load("name"))
        private static readonly Dictionary<string, string> _nameToSubAssetGuid = new(StringComparer.OrdinalIgnoreCase);
        // Source GUID → cache type name (for simple assets where source GUID = cache key)
        private static readonly Dictionary<string, string> _sourceGuidCacheType = new(StringComparer.OrdinalIgnoreCase);

        // Known importable extensions (discovered from AssetImporter attributes)
        private static readonly HashSet<string> _importableExtensions = new(StringComparer.OrdinalIgnoreCase);

        // Extension → IImporter type (discovered at init)
        private static readonly Dictionary<string, Type> _importersByExtension = new(StringComparer.OrdinalIgnoreCase);

        // Artifact type name → packer instance (discovered at init)
        private static readonly Dictionary<string, object> _packers = new(StringComparer.OrdinalIgnoreCase);

        // Artifact type name → cache file extension (discovered from AssetPackerAttribute)
        private static readonly Dictionary<string, string> _cacheExtensions = new(StringComparer.OrdinalIgnoreCase);

        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        /// <summary>
        /// Initialize the asset database for a project.
        /// Scans Assets/ for source files, loads/creates meta files in Library/.
        /// </summary>
        public static void Initialize(FreefallProject project)
        {
            Project = project;
            Clear();
            DiscoverImporters();
            DiscoverPackers();
            ScanAndSync();
            Debug.Log($"[AssetDatabase] Initialized: {_guidToMeta.Count} assets tracked, " +
                      $"{_importersByExtension.Count} importers, {_packers.Count} packers");
        }

        /// <summary>
        /// Re-scan for changes (new/deleted/modified source files).
        /// </summary>
        public static void Refresh()
        {
            ScanAndSync();
        }

        /// <summary>
        /// Import all dirty assets (source newer than cache).
        /// </summary>
        public static void ImportAll()
        {
            int imported = 0;
            foreach (var meta in _guidToMeta.Values.ToList())
            {
                if (NeedsReimport(meta))
                {
                    ImportAsset(meta);
                    imported++;
                }
            }
            if (imported > 0)
                Debug.Log($"[AssetDatabase] Imported {imported} assets");
        }

        /// <summary>
        /// Import all dirty assets on a background thread with progress reporting.
        /// Uses Action&lt;string&gt; instead of IProgress&lt;T&gt; to avoid SynchronizationContext
        /// marshaling issues with RenderLoop-based message pumps.
        /// </summary>
        public static async System.Threading.Tasks.Task ImportAllAsync(
            Action<string> progress = null,
            System.Threading.CancellationToken ct = default)
        {
            var dirtyMetas = _guidToMeta.Values.Where(m => NeedsReimport(m)).ToList();
            if (dirtyMetas.Count == 0)
            {
                return;
            }

            progress?.Invoke($"Importing {dirtyMetas.Count} assets...");

            await System.Threading.Tasks.Task.Run(() =>
            {
                int done = 0;
                int failed = 0;
                long lastReportTicks = 0;
                var options = new System.Threading.Tasks.ParallelOptions
                {
                    MaxDegreeOfParallelism = Environment.ProcessorCount,
                    CancellationToken = ct
                };
                System.Threading.Tasks.Parallel.ForEach(dirtyMetas, options, meta =>
                {
                    var current = System.Threading.Interlocked.Increment(ref done);
                    // Throttle progress to avoid flooding the UI thread
                    long now = Environment.TickCount64;
                    long last = System.Threading.Interlocked.Read(ref lastReportTicks);
                    if (now - last > 250 && System.Threading.Interlocked.CompareExchange(ref lastReportTicks, now, last) == last)
                    {
                        progress?.Invoke($"[{current}/{dirtyMetas.Count}] {meta.SourcePath}");
                    }
                    try
                    {
                        ImportAsset(meta);
                    }
                    catch (OperationCanceledException) { throw; }
                    catch (Exception ex)
                    {
                        Debug.LogWarning("AssetDatabase", $"Unhandled import error: {meta.SourcePath} — {ex.Message}");
                        // Stamp so we don't retry next launch
                        meta.LastImported = DateTime.UtcNow;
                        WriteMetaFile(meta);
                        System.Threading.Interlocked.Increment(ref failed);
                    }
                });
                progress?.Invoke($"Done — {done - failed} imported, {failed} failed.");
            }, ct);
        }

        /// <summary>
        /// Import a single asset by its relative source path.
        /// </summary>
        public static void ImportAssetByPath(string relativePath)
        {
            var normalized = NormalizePath(relativePath);
            if (!_pathToGuid.TryGetValue(normalized, out var guid))
                throw new FileNotFoundException($"Asset not tracked: {relativePath}");

            ImportAsset(_guidToMeta[guid]);
        }

        // ── Lookups ──

        public static string PathToGuid(string relativePath)
        {
            var normalized = NormalizePath(relativePath);
            return _pathToGuid.TryGetValue(normalized, out var guid) ? guid : null;
        }

        public static string GuidToPath(string guid)
        {
            return _guidToPath.TryGetValue(guid, out var path) ? path : null;
        }

        public static MetaFile GetMeta(string guid)
        {
            return _guidToMeta.TryGetValue(guid, out var meta) ? meta : null;
        }

        /// <summary>
        /// Resolve an asset name to its cache file path.
        /// Handles both simple assets (source GUID = cache key) and compound subassets.
        /// </summary>
        internal static string ResolveCachePath(string name)
        {
            // Try subasset lookup first (compound assets)
            if (_nameToSubAssetGuid.TryGetValue(name, out var subGuid))
            {
                if (_subAssetToSource.TryGetValue(subGuid, out var sourceGuid))
                {
                    var meta = _guidToMeta[sourceGuid];
                    var sub = meta.SubAssets.FirstOrDefault(s => s.Guid == subGuid);
                    if (sub != null)
                        return GetCachePath(sub.Guid, sub.Type);
                }
            }

            // Try simple asset lookup (source GUID = cache key)
            // Name maps to a source path, which maps to a GUID
            foreach (var kvp in _pathToGuid)
            {
                var fileName = Path.GetFileNameWithoutExtension(kvp.Key);
                if (fileName.Equals(name, StringComparison.OrdinalIgnoreCase))
                {
                    var guid = kvp.Value;
                    if (_sourceGuidCacheType.TryGetValue(guid, out var typeName))
                        return GetCachePath(guid, typeName);
                }
            }

            return null;
        }

        /// <summary>
        /// Resolve a friendly name to its source GUID (for simple assets) or subasset GUID.
        /// </summary>
        public static string ResolveGuidByName(string name)
        {
            // Try subasset lookup first (compound assets)
            if (_nameToSubAssetGuid.TryGetValue(name, out var subGuid))
                return subGuid;

            // Try simple asset lookup — prefer .asset files over raw model files (.fbx, .dae, .obj)
            // because .asset files are StaticMesh definitions while raw model files produce .mesh cache
            string assetFileGuid = null;
            string fallback = null;
            foreach (var kvp in _pathToGuid)
            {
                var fileName = Path.GetFileNameWithoutExtension(kvp.Key);
                if (fileName.Equals(name, StringComparison.OrdinalIgnoreCase))
                {
                    // Strongly prefer .asset source files
                    if (kvp.Key.EndsWith(".asset", StringComparison.OrdinalIgnoreCase))
                    {
                        assetFileGuid = kvp.Value;
                    }
                    fallback ??= kvp.Value;
                }
            }

            return assetFileGuid ?? fallback;
        }

        /// <summary>
        /// Resolve a GUID directly to its cache file path.
        /// </summary>
        internal static string ResolveCachePathByGuid(string guid)
        {
            // Try as source GUID (simple asset)
            if (_sourceGuidCacheType.TryGetValue(guid, out var typeName))
                return GetCachePath(guid, typeName);

            // Try as subasset GUID
            if (_subAssetToSource.TryGetValue(guid, out var sourceGuid))
            {
                var meta = _guidToMeta[sourceGuid];
                var sub = meta.SubAssets.FirstOrDefault(s => s.Guid == guid);
                if (sub != null)
                    return GetCachePath(sub.Guid, sub.Type);
            }

            return null;
        }

        /// <summary>
        /// Compute the cache file path for a subasset: Cache/{guid[..2]}/{guid}{ext}
        /// </summary>
        private static string GetCachePath(string guid, string typeName)
        {
            var ext = _cacheExtensions.TryGetValue(typeName, out var e) ? e : ".bin";
            var bucket = guid[..2];
            return Path.Combine(Project.CacheDirectory, bucket, $"{guid}{ext}");
        }

        /// <summary>
        /// Get all tracked source paths.
        /// </summary>
        public static IEnumerable<string> GetAllPaths() => _guidToPath.Values;

        /// <summary>
        /// Get all meta files.
        /// </summary>
        public static IEnumerable<MetaFile> GetAllMeta() => _guidToMeta.Values;

        /// <summary>
        /// Check if a file extension has a registered importer.
        /// Used by editor UI to filter displayable assets.
        /// </summary>
        public static bool IsImportableExtension(string extension)
        {
            var ext = extension.StartsWith(".") ? extension : "." + extension;
            return _importableExtensions.Contains(ext);
        }

        // ── Core Logic ──

        private static void Clear()
        {
            _guidToPath.Clear();
            _pathToGuid.Clear();
            _guidToMeta.Clear();
            _subAssetToSource.Clear();
            _nameToSubAssetGuid.Clear();
            _sourceGuidCacheType.Clear();
            _importableExtensions.Clear();
            _importersByExtension.Clear();
            _packers.Clear();
        }

        /// <summary>
        /// Discover all IImporter types and their supported extensions.
        /// </summary>
        private static void DiscoverImporters()
        {
            var assemblies = new[] { Assembly.GetExecutingAssembly(), Assembly.GetEntryAssembly() };

            foreach (var assembly in assemblies)
            {
                if (assembly == null) continue;
                try
                {
                    foreach (var type in assembly.GetTypes())
                    {
                        var attr = type.GetCustomAttribute<AssetImporterAttribute>();
                        if (attr == null) continue;
                        if (!typeof(IImporter).IsAssignableFrom(type)) continue;

                        foreach (var ext in attr.Extensions)
                        {
                            var normalized = ext.StartsWith(".") ? ext : "." + ext;
                            _importableExtensions.Add(normalized);

                            // IImporter implementations take priority over AssetImporter<T>
                            if (!_importersByExtension.ContainsKey(normalized) ||
                                !typeof(IImporter).IsAssignableFrom(_importersByExtension[normalized]))
                            {
                                _importersByExtension[normalized] = type;
                            }
                        }
                    }
                }
                catch (ReflectionTypeLoadException) { }
            }
        }

        /// <summary>
        /// Discover all AssetPacker<T> implementations.
        /// </summary>
        private static void DiscoverPackers()
        {
            var assemblies = new[] { Assembly.GetExecutingAssembly(), Assembly.GetEntryAssembly() };

            foreach (var assembly in assemblies)
            {
                if (assembly == null) continue;
                try
                {
                    foreach (var type in assembly.GetTypes())
                    {
                        if (type.IsAbstract) continue;

                        var attr = type.GetCustomAttribute<AssetPackerAttribute>();
                        if (attr == null) continue;

                        var baseType = type.BaseType;
                        if (baseType == null || !baseType.IsGenericType) continue;
                        if (baseType.GetGenericTypeDefinition() != typeof(AssetPacker<>)) continue;

                        var packedType = baseType.GetGenericArguments()[0];
                        var instance = Activator.CreateInstance(type);
                        _packers[packedType.Name] = instance;
                        _cacheExtensions[packedType.Name] = attr.CacheExtension;
                    }
                }
                catch (ReflectionTypeLoadException) { }
            }
        }

        /// <summary>
        /// Scan Assets/ for source files, load existing .meta files from Library/,
        /// create new .meta for untracked assets, remove orphaned .meta files.
        /// </summary>
        public static void ScanAndSync()
        {
            var assetsDir = Project.AssetsDirectory;
            var libraryDir = Project.LibraryDirectory;

            // 1. Get all source files with importable extensions
            var sourceFiles = new Dictionary<string, FileInfo>(StringComparer.OrdinalIgnoreCase);
            if (Directory.Exists(assetsDir))
            {
                foreach (var file in new DirectoryInfo(assetsDir).EnumerateFiles("*", SearchOption.AllDirectories))
                {
                    if (_importableExtensions.Contains(file.Extension))
                    {
                        var relativePath = GetRelativePath(assetsDir, file.FullName);
                        sourceFiles[NormalizePath(relativePath)] = file;
                    }
                }
            }

            // 2. Load existing .meta files
            var existingMetas = new Dictionary<string, MetaFile>(StringComparer.OrdinalIgnoreCase);
            if (Directory.Exists(libraryDir))
            {
                foreach (var metaFile in Directory.EnumerateFiles(libraryDir, "*.meta", SearchOption.AllDirectories))
                {
                    var meta = ReadMetaFile(metaFile);
                    if (meta != null)
                        existingMetas[meta.Guid] = meta;
                }
            }

            // 3. Match existing metas to source files, register or clean up
            var matchedPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var meta in existingMetas.Values)
            {
                var normalized = NormalizePath(meta.SourcePath);
                if (sourceFiles.ContainsKey(normalized))
                {
                    // Source exists — register
                    RegisterMeta(meta);
                    matchedPaths.Add(normalized);
                }
                else
                {
                    // Source deleted — remove orphaned .meta
                    var bucket = meta.Guid[..2];
                    var metaPath = Path.Combine(libraryDir, bucket, $"{meta.Guid}.meta");
                    if (File.Exists(metaPath))
                    {
                        File.Delete(metaPath);
                        Debug.Log($"[AssetDatabase] Removed orphan: {meta.SourcePath}");
                    }
                }
            }

            // 4. Create .meta for new (untracked) source files
            foreach (var (path, fileInfo) in sourceFiles)
            {
                if (matchedPaths.Contains(path))
                    continue;

                var guid = System.Guid.NewGuid().ToString("N");
                var meta = new MetaFile
                {
                    Guid = guid,
                    SourcePath = path,
                    ImporterType = FindImporterTypeName(fileInfo.Extension),
                    SubAssets = new List<SubAssetEntry>(),
                    LastImported = DateTime.MinValue
                };

                WriteMetaFile(meta);
                RegisterMeta(meta);

                Debug.Log($"[AssetDatabase] Tracking new asset: {path}");
            }
        }

        private static void RegisterMeta(MetaFile meta)
        {
            var normalized = NormalizePath(meta.SourcePath);
            _guidToPath[meta.Guid] = meta.SourcePath;
            _pathToGuid[normalized] = meta.Guid;
            _guidToMeta[meta.Guid] = meta;

            if (meta.SubAssets.Count > 0)
            {
                // Compound asset: register subasset GUIDs
                foreach (var sub in meta.SubAssets)
                {
                    _subAssetToSource[sub.Guid] = meta.Guid;
                    _nameToSubAssetGuid[sub.Name] = sub.Guid;
                }
            }
            else if (!string.IsNullOrEmpty(meta.MainAssetType))
            {
                // Simple asset: source GUID IS the cache key
                _sourceGuidCacheType[meta.Guid] = meta.MainAssetType;
            }
        }

        private static bool NeedsReimport(MetaFile meta)
        {
            // Never imported
            if (meta.LastImported == DateTime.MinValue)
                return true;

            // Cache file missing — reimport even if timestamps say up-to-date
            if (meta.MainAssetType != null)
            {
                var cachePath = GetCachePath(meta.Guid, meta.MainAssetType);
                if (!File.Exists(cachePath))
                    return true;
            }

            // Check if source is newer than last import (with 2s tolerance for filesystem timestamp precision)
            var sourcePath = Path.Combine(Project.AssetsDirectory, meta.SourcePath);
            if (!File.Exists(sourcePath))
                return false;

            var sourceTime = File.GetLastWriteTimeUtc(sourcePath);
            var sourceSize = new FileInfo(sourcePath).Length;

            // Reimport if timestamp is newer OR file size changed (handles copied files with old timestamps)
            if (sourceTime > meta.LastImported.AddSeconds(2))
                return true;
            if (meta.FileSize > 0 && sourceSize != meta.FileSize)
                return true;

            return false;
        }

        private static void ImportAsset(MetaFile meta)
        {
            var sourcePath = Path.Combine(Project.AssetsDirectory, meta.SourcePath);
            if (!File.Exists(sourcePath))
            {
                Debug.LogWarning("AssetDatabase", $"Source not found: {meta.SourcePath}");
                return;
            }

            // Find importer
            var ext = Path.GetExtension(sourcePath);
            if (!_importersByExtension.TryGetValue(ext, out var importerType))
            {
                Debug.LogWarning("AssetDatabase", $"No importer for extension '{ext}'");
                meta.LastImported = DateTime.UtcNow;
                WriteMetaFile(meta);
                return;
            }

            var importer = (IImporter)Activator.CreateInstance(importerType);

            Debug.Log($"[AssetDatabase] Importing: {meta.SourcePath}");

            // Run import
            ImportResult result;
            try
            {
                result = importer.Import(sourcePath);
            }
            catch (Exception ex)
            {
                Debug.LogWarning("AssetDatabase", $"Import failed: {meta.SourcePath} — {ex.Message}");
                meta.LastImported = DateTime.UtcNow;
                WriteMetaFile(meta);
                return;
            }

            // Pack artifacts to Cache/
            // Save existing subasset GUIDs so we can reuse them on reimport (preserves scene references)
            var oldSubAssets = new List<SubAssetEntry>(meta.SubAssets);
            meta.SubAssets.Clear();
            meta.MainAssetType = null;

            if (result.Artifacts.Count == 1)
            {
                // Simple asset: source GUID = cache key, no subassets
                var artifact = result.Artifacts[0];
                var dataTypeName = artifact.Data.GetType().Name;
                var cachePath = GetCachePath(meta.Guid, dataTypeName);
                Directory.CreateDirectory(Path.GetDirectoryName(cachePath));
                PackArtifact(artifact, cachePath);
                meta.MainAssetType = dataTypeName;
            }
            else
            {
                // Compound asset: each artifact gets its own GUID
                // Reuse existing GUIDs from previous import to keep scene references stable
                foreach (var artifact in result.Artifacts)
                {
                    var dataTypeName = artifact.Data.GetType().Name;
                    var existing = oldSubAssets.Find(s => s.Name == artifact.Name && s.Type == dataTypeName);
                    var subGuid = existing?.Guid ?? System.Guid.NewGuid().ToString("N");
                    var cachePath = GetCachePath(subGuid, dataTypeName);
                    Directory.CreateDirectory(Path.GetDirectoryName(cachePath));

                    if (PackArtifact(artifact, cachePath))
                    {
                        meta.SubAssets.Add(new SubAssetEntry
                        {
                            Guid = subGuid,
                            Name = artifact.Name,
                            Type = dataTypeName,
                        });
                    }
                }
            }

            meta.ImporterType = importerType.FullName;
            meta.LastImported = DateTime.UtcNow;
            meta.FileSize = new FileInfo(sourcePath).Length;
            WriteMetaFile(meta);

            // Re-register subassets for lookup
            RegisterMeta(meta);

            Debug.Log($"[AssetDatabase] Imported: {meta.SourcePath} → {meta.SubAssets.Count} artifacts");
        }

        /// <summary>
        /// Pack a single artifact to a cache file using the appropriate packer.
        /// </summary>
        private static bool PackArtifact(ImportArtifact artifact, string cachePath)
        {
            var dataTypeName = artifact.Data.GetType().Name;
            if (!_packers.TryGetValue(dataTypeName, out var packer))
            {
                Debug.LogWarning("AssetDatabase", $"No packer for type '{dataTypeName}' (artifact type: '{artifact.Type}')");
                return false;
            }

            try
            {
                // Use reflection to call AssetPacker<T>.Write(Stream, T)
                var packerType = packer.GetType();
                var writeMethod = packerType.GetMethod("Write");
                using var stream = File.Create(cachePath);
                writeMethod.Invoke(packer, new object[] { stream, artifact.Data });
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("AssetDatabase", $"Pack failed for '{artifact.Name}': {ex.Message}");
                return false;
            }
        }

        // ── Serialization (isolated for JSON→YAML swap) ──

        private static MetaFile ReadMetaFile(string path)
        {
            try
            {
                var json = File.ReadAllText(path);
                return JsonSerializer.Deserialize<MetaFile>(json, _jsonOptions);
            }
            catch (Exception ex)
            {
                Debug.LogWarning("AssetDatabase", $"Failed to read meta: {path} — {ex.Message}");
                return null;
            }
        }

        private static void WriteMetaFile(MetaFile meta)
        {
            var bucket = meta.Guid[..2];
            var bucketDir = Path.Combine(Project.LibraryDirectory, bucket);
            Directory.CreateDirectory(bucketDir);
            var path = Path.Combine(bucketDir, $"{meta.Guid}.meta");
            var json = JsonSerializer.Serialize(meta, _jsonOptions);
            File.WriteAllText(path, json);
        }

        // ── Helpers ──

        private static string FindImporterTypeName(string extension)
        {
            if (_importersByExtension.TryGetValue(extension, out var type))
                return type.FullName;

            var normalized = extension.StartsWith(".") ? extension : "." + extension;
            if (_importersByExtension.TryGetValue(normalized, out type))
                return type.FullName;

            return null;
        }

        private static string NormalizePath(string path)
        {
            return path.Replace('/', '\\').TrimStart('\\');
        }

        private static string GetRelativePath(string basePath, string fullPath)
        {
            if (!basePath.EndsWith(Path.DirectorySeparatorChar.ToString()))
                basePath += Path.DirectorySeparatorChar;
            return fullPath.Substring(basePath.Length);
        }
    }
}
