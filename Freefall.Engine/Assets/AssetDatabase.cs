using System;
using System.Collections.Concurrent;
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
        private static readonly ConcurrentDictionary<string, string> _guidToPath = new(StringComparer.OrdinalIgnoreCase);
        // Relative source path → source GUID
        private static readonly ConcurrentDictionary<string, string> _pathToGuid = new(StringComparer.OrdinalIgnoreCase);
        // Source GUID → full meta data
        private static readonly ConcurrentDictionary<string, MetaFile> _guidToMeta = new(StringComparer.OrdinalIgnoreCase);
        // Subasset GUID → source GUID (for reverse lookup)
        private static readonly ConcurrentDictionary<string, string> _subAssetToSource = new(StringComparer.OrdinalIgnoreCase);
        // Name → subasset entries (for user-facing Load("name"), supports multiple types per name)
        private static readonly ConcurrentDictionary<string, List<SubAssetEntry>> _nameToSubAssets = new(StringComparer.OrdinalIgnoreCase);
        // Source GUID → cache type name (for simple assets where source GUID = cache key)
        private static readonly ConcurrentDictionary<string, string> _sourceGuidCacheType = new(StringComparer.OrdinalIgnoreCase);

        // Known importable extensions (discovered from AssetImporter attributes)
        private static readonly HashSet<string> _importableExtensions = new(StringComparer.OrdinalIgnoreCase);

        // Extension → IImporter type (discovered at init)
        private static readonly Dictionary<string, Type> _importersByExtension = new(StringComparer.OrdinalIgnoreCase);

        // Extension → import priority (from AssetImporterAttribute.ImportPriority)
        private static readonly Dictionary<string, int> _importerPriority = new(StringComparer.OrdinalIgnoreCase);

        // Artifact type name → packer instance (discovered at init)
        private static readonly Dictionary<string, object> _packers = new(StringComparer.OrdinalIgnoreCase);

        // Artifact type name → cache file extension (discovered from AssetPackerAttribute)
        private static readonly Dictionary<string, string> _cacheExtensions = new(StringComparer.OrdinalIgnoreCase);

        // ── Thumbnail tracking ──
        // GUID → thumbnail file path (absolute). Populated by ScanThumbnails().
        private static readonly ConcurrentDictionary<string, string> _guidToThumb = new(StringComparer.OrdinalIgnoreCase);
        // GUID → loaded Texture for Squid display. Populated lazily by GetThumbnail().
        private static readonly ConcurrentDictionary<string, Graphics.Texture> _thumbTextures = new(StringComparer.OrdinalIgnoreCase);
        // Asset type → thumbnail generator instance (discovered from ThumbnailGeneratorAttribute)
        private static readonly Dictionary<Type, IThumbnailGenerator> _thumbGenerators = new();
        // Data type name → runtime Type (reverse of AssetTypeAliasAttribute, built during discovery)
        private static readonly Dictionary<string, Type> _typeAliases = new(StringComparer.OrdinalIgnoreCase);

        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        private static readonly JsonSerializerOptions _importerJsonOptions = new()
        {
            WriteIndented = false,
            IncludeFields = true,
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
            ScanThumbnails();
            CleanOrphanedThumbnails();
            Debug.Log($"[AssetDatabase] Initialized: {_guidToMeta.Count} assets tracked, " +
                      $"{_importersByExtension.Count} importers, {_packers.Count} packers, {_guidToThumb.Count} thumbnails");
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
            var dirtyMetas = _guidToMeta.Values
                .Where(m => NeedsReimport(m))
                .OrderBy(m => GetImportPriority(m))
                .ToList();

            foreach (var meta in dirtyMetas)
                ImportAsset(meta);

            if (dirtyMetas.Count > 0)
                Debug.Log($"[AssetDatabase] Imported {dirtyMetas.Count} assets");
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
            var dirtyMetas = _guidToMeta.Values
                .Where(m => NeedsReimport(m))
                .OrderBy(m => GetImportPriority(m))
                .ToList();
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

                // Process each priority group sequentially, assets within a group in parallel
                var groups = dirtyMetas.GroupBy(m => GetImportPriority(m)).OrderBy(g => g.Key);
                foreach (var group in groups)
                {
                    System.Threading.Tasks.Parallel.ForEach(group, options, meta =>
                    {
                        var current = System.Threading.Interlocked.Increment(ref done);
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
                            meta.LastImported = DateTime.UtcNow;
                            WriteMetaFile(meta);
                            System.Threading.Interlocked.Increment(ref failed);
                        }
                    });
                }
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
            // Try as source GUID first
            if (_guidToMeta.TryGetValue(guid, out var meta))
                return meta;

            // Try resolving subasset GUID → source GUID
            if (_subAssetToSource.TryGetValue(guid, out var sourceGuid))
                return _guidToMeta.TryGetValue(sourceGuid, out meta) ? meta : null;

            return null;
        }

        /// <summary>
        /// Find a sibling sub-asset of a given type from the same source file.
        /// Used by loaders to resolve related assets (e.g., Mesh → Skeleton from the same model).
        /// </summary>
        public static SubAssetEntry FindSiblingSubAsset(string subAssetGuid, string siblingType)
        {
            // Resolve to source GUID
            string sourceGuid = subAssetGuid;
            if (_subAssetToSource.TryGetValue(subAssetGuid, out var resolved))
                sourceGuid = resolved;

            if (!_guidToMeta.TryGetValue(sourceGuid, out var meta))
                return null;

            lock (meta)
            {
                foreach (var sub in meta.SubAssets)
                {
                    if (sub.Type == siblingType || sub.AssetType == siblingType)
                        return sub;
                }
            }

            return null;
        }

        /// <summary>
        /// Resolve an asset name or relative path to its cache file path.
        /// Tries: 1) exact relative path match, 2) subasset name lookup, 3) filename-only path scan.
        /// </summary>
        internal static string ResolveCachePath(string nameOrPath, string dataType = null)
        {
            var normalized = NormalizePath(nameOrPath);
            var name = Path.GetFileNameWithoutExtension(nameOrPath);


            // 1) Exact relative path match (unambiguous)
            if (_pathToGuid.TryGetValue(normalized, out var pathGuid))
            {
                // Compound asset: find the subasset with matching data type
                if (_guidToMeta.TryGetValue(pathGuid, out var meta))
                {
                    lock (meta)
                    {
                        if (meta.SubAssets.Count > 0)
                        {
                            var sub = dataType != null
                                ? meta.SubAssets.FirstOrDefault(s => s.Type.Equals(dataType, StringComparison.OrdinalIgnoreCase))
                                : meta.SubAssets.FirstOrDefault(s => !s.Hidden);
                            if (sub != null)
                                return GetCachePath(sub.Guid, sub.Type);
                        }
                    }
                }

                // Simple asset: source GUID = cache key
                if (_sourceGuidCacheType.TryGetValue(pathGuid, out var typeName))
                    return GetCachePath(pathGuid, typeName);
            }

            // 2) Subasset name lookup (for Load("assetName") pattern)
            if (_nameToSubAssets.TryGetValue(name, out var entries))
            {
                lock (entries)
                {
                    var sub = dataType != null
                        ? entries.FirstOrDefault(s => s.Type.Equals(dataType, StringComparison.OrdinalIgnoreCase))
                        : entries.Count > 0 ? entries[0] : null;
                    if (sub != null)
                        return GetCachePath(sub.Guid, sub.Type);
                }
            }

            // 3) Filename-only scan for simple assets
            foreach (var kvp in _pathToGuid)
            {
                var fileName = Path.GetFileNameWithoutExtension(kvp.Key);
                if (fileName.Equals(name, StringComparison.OrdinalIgnoreCase))
                {
                    var guid = kvp.Value;
                    if (_sourceGuidCacheType.TryGetValue(guid, out var tn))
                        return GetCachePath(guid, tn);
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
            if (_nameToSubAssets.TryGetValue(name, out var entries))
            {
                lock (entries)
                {
                    if (entries.Count > 0)
                        return entries[0].Guid;
                }
            }

            // Try simple asset lookup — prefer simple assets (source GUID = cache key, e.g. .staticmesh)
            // over compound source files (.fbx, .dae) that produce multiple subassets
            string preferredGuid = null;
            string fallback = null;
            foreach (var kvp in _pathToGuid)
            {
                var fileName = Path.GetFileNameWithoutExtension(kvp.Key);
                if (fileName.Equals(name, StringComparison.OrdinalIgnoreCase))
                {
                    // Prefer simple assets — they are the final definition, not a raw source
                    if (_sourceGuidCacheType.ContainsKey(kvp.Value))
                    {
                        preferredGuid = kvp.Value;
                    }
                    fallback ??= kvp.Value;
                }
            }

            return preferredGuid ?? fallback;
        }

        /// <summary>
        /// Resolve a friendly name to its source GUID (for simple assets) or subasset GUID.
        /// </summary>
        public static string FindGuidByName(string name, string type)
        {
            // Try simple asset lookup — prefer simple assets (source GUID = cache key, e.g. .asset)
            // over compound source files (.fbx, .dae) that produce multiple subassets
            string preferredGuid = null;
            string fallback = null;
            foreach (var kvp in _pathToGuid)
            {
                var fileName = Path.GetFileNameWithoutExtension(kvp.Key);
                if (fileName.Equals(name, StringComparison.OrdinalIgnoreCase))
                {
                    // Prefer simple assets — they are the final definition, not a raw source
                    if (_sourceGuidCacheType.ContainsKey(kvp.Value))
                    {
                        preferredGuid = kvp.Value;
                    }
                    fallback ??= kvp.Value;
                }
            }

            return preferredGuid ?? fallback;
        }

        /// <summary>
        /// Resolve a friendly name to a subasset GUID, filtered by type.
        /// </summary>
        public static string ResolveGuidByName(string name, string type)
        {
            if (_nameToSubAssets.TryGetValue(name, out var entries))
            {
                lock (entries)
                {
                    var match = entries.FirstOrDefault(e => e.Type.Equals(type, StringComparison.OrdinalIgnoreCase));
                    if (match != null) return match.Guid;
                }
            }
            return null;
        }

        /// <summary>
        /// Resolve a GUID to a human-readable name.
        /// Checks: 1) source path (file name), 2) subasset Name, 3) falls back to GUID.
        /// </summary>
        public static string ResolveFriendlyName(string guid)
        {
            // Source GUID → source file name
            if (_guidToPath.TryGetValue(guid, out var path))
                return Path.GetFileNameWithoutExtension(path);

            // Subasset GUID → subasset Name from meta
            if (_subAssetToSource.TryGetValue(guid, out var sourceGuid) &&
                _guidToMeta.TryGetValue(sourceGuid, out var meta))
            {
                lock (meta)
                {
                    var sub = meta.SubAssets.FirstOrDefault(s => s.Guid == guid);
                    if (sub != null)
                        return sub.Name;
                }
            }

            return guid;
        }

        /// <summary>
        /// Resolve a GUID directly to its cache file path.
        /// </summary>
        public static string ResolveCachePathByGuid(string guid, Type dataType = null)
        {
            var dataTypeName = dataType?.Name;
            // Try as source GUID (simple asset)
            if (_sourceGuidCacheType.TryGetValue(guid, out var typeName))
            {
                if (dataTypeName == null || typeName.Equals(dataTypeName, StringComparison.OrdinalIgnoreCase))
                    return GetCachePath(guid, typeName);
            }

            // Try as subasset GUID
            if (_subAssetToSource.TryGetValue(guid, out var sourceGuid))
            {
                if (_guidToMeta.TryGetValue(sourceGuid, out var meta))
                {
                    lock (meta)
                    {
                        var sub = meta.SubAssets.FirstOrDefault(s => s.Guid == guid);
                        if (sub != null && (dataTypeName == null || sub.Type.Equals(dataTypeName, StringComparison.OrdinalIgnoreCase)))
                            return GetCachePath(sub.Guid, sub.Type);
                    }
                }
            }

            // Try as source GUID for a compound asset → find matching subasset
            if (_guidToMeta.TryGetValue(guid, out var compoundMeta))
            {
                lock (compoundMeta)
                {
                    if (compoundMeta.SubAssets.Count > 0)
                    {
                        SubAssetEntry primary;
                        if (dataTypeName != null)
                            primary = compoundMeta.SubAssets.FirstOrDefault(s => s.Type.Equals(dataTypeName, StringComparison.OrdinalIgnoreCase));
                        else
                            primary = compoundMeta.SubAssets.FirstOrDefault(s => !s.Hidden)
                                   ?? compoundMeta.SubAssets[0];

                        if (primary != null)
                            return GetCachePath(primary.Guid, primary.Type);
                    }
                }
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
        /// Adds or updates a subasset entry on a source asset's meta, writes the meta file,
        /// and registers the subasset for runtime lookup.
        /// Returns the subasset GUID (reused if already exists, new if created).
        /// </summary>
        public static string AddOrUpdateSubAsset(string sourceGuid, string type, string name, bool hidden = false)
        {
            if (!_guidToMeta.TryGetValue(sourceGuid, out var meta))
                return null;

            string subGuid;
            lock (meta)
            {
                var existing = meta.SubAssets.FirstOrDefault(
                    s => s.Type == type && s.Name == name);

                if (existing != null)
                    return existing.Guid;

                subGuid = System.Guid.NewGuid().ToString("N");
                var entry = new SubAssetEntry
                {
                    Guid = subGuid,
                    Name = name,
                    Type = type,
                    Hidden = hidden
                };
                meta.SubAssets.Add(entry);
            }
            WriteMetaFile(meta);

            // Register for runtime lookup
            _subAssetToSource[subGuid] = sourceGuid;

            return subGuid;
        }

        /// <summary>
        /// Check if a file extension has a registered importer.
        /// Used by editor UI to filter displayable assets.
        /// </summary>
        public static bool IsImportableExtension(string extension)
        {
            var ext = extension.StartsWith(".") ? extension : "." + extension;
            return _importableExtensions.Contains(ext);
        }

        /// <summary>
        /// Get all importable extensions and their importer type names.
        /// </summary>
        public static IEnumerable<(string extension, string importerType)> GetImportableExtensions()
        {
            foreach (var kvp in _importersByExtension)
                yield return (kvp.Key, kvp.Value.Name);
        }

        /// <summary>
        /// Instantiate the importer for a given GUID.
        /// Accepts both source GUIDs and sub-asset GUIDs (resolves to source automatically).
        /// Returns null if no importer is registered for the file's extension.
        /// </summary>
        public static IImporter GetImporter(string guid)
        {
            // Try as source GUID first, then resolve sub-asset → source
            if (!_guidToPath.TryGetValue(guid, out var sourcePath))
            {
                if (_subAssetToSource.TryGetValue(guid, out var sourceGuid))
                    _guidToPath.TryGetValue(sourceGuid, out sourcePath);

                if (sourcePath == null) return null;
                guid = sourceGuid;
            }
            var sourceGuidResolved = guid;

            var ext = Path.GetExtension(sourcePath);
            if (string.IsNullOrEmpty(ext)) return null;

            if (!_importersByExtension.TryGetValue(ext, out var importerType))
                return null;

            var importer = (IImporter)Activator.CreateInstance(importerType);

            // Restore saved settings if available
            var meta = GetMeta(sourceGuidResolved);
            if (meta != null && !string.IsNullOrEmpty(meta.ImporterSettings))
            {
                try
                {
                    var restored = System.Text.Json.JsonSerializer.Deserialize(meta.ImporterSettings, importerType, _importerJsonOptions) as IImporter;
                    if (restored != null) importer = restored;
                }
                catch { }
            }

            return importer;
        }

        /// <summary>
        /// Save modified importer settings to the meta file and reimport the asset.
        /// Used by the inspector "Apply & Reimport" button.
        /// </summary>
        public static void SaveImporterAndReimport(string guid, IImporter importer)
        {
            // Resolve to source GUID (in case we got a sub-asset GUID)
            if (_subAssetToSource.TryGetValue(guid, out var sourceGuid))
                guid = sourceGuid;

            if (!_guidToMeta.TryGetValue(guid, out var meta))
            {
                Debug.LogWarning("AssetDatabase", $"Cannot save importer settings: unknown GUID {guid}");
                return;
            }

            // Serialize current importer state to the meta file
            var importerType = importer.GetType();
            try { meta.ImporterSettings = System.Text.Json.JsonSerializer.Serialize(importer, importerType, _importerJsonOptions); }
            catch (Exception ex)
            {
                Debug.LogWarning("AssetDatabase", $"Failed to serialize importer settings: {ex.Message}");
                return;
            }

            WriteMetaFile(meta);
            Debug.Log($"[AssetDatabase] Saved importer settings for {meta.SourcePath}");

            // Reimport with the new settings
            ImportAsset(meta);
        }

        // ── Core Logic ──

        private static void Clear()
        {
            _guidToPath.Clear();
            _pathToGuid.Clear();
            _guidToMeta.Clear();
            _subAssetToSource.Clear();
            _nameToSubAssets.Clear();
            _sourceGuidCacheType.Clear();
            _importableExtensions.Clear();
            _importersByExtension.Clear();
            _importerPriority.Clear();
            _packers.Clear();
            _guidToThumb.Clear();
            _thumbTextures.Clear();

            _thumbGenerators.Clear();
            _typeAliases.Clear();
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
                                _importerPriority[normalized] = attr.ImportPriority;
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
        /// create new .meta for untracked assets, detect renamed files, remove orphaned .meta files.
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

            // 3. Match existing metas to source files, register or collect orphans
            var matchedPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var orphanedMetas = new List<MetaFile>();

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
                    orphanedMetas.Add(meta);
                }
            }

            // 4. Collect untracked source files (potential rename targets)
            var untrackedFiles = new Dictionary<string, FileInfo>(StringComparer.OrdinalIgnoreCase);
            foreach (var (path, fileInfo) in sourceFiles)
            {
                if (!matchedPaths.Contains(path))
                    untrackedFiles[path] = fileInfo;
            }

            // 5. Try to match orphaned metas to untracked files (external rename detection)
            //    Match by: same parent folder + same extension + same file size
            foreach (var meta in orphanedMetas.ToList())
            {
                var oldDir = Path.GetDirectoryName(NormalizePath(meta.SourcePath)) ?? "";
                var oldExt = Path.GetExtension(meta.SourcePath);

                MetaFile bestMatch = null;
                string bestPath = null;
                int bestDistance = int.MaxValue;

                foreach (var (candidatePath, candidateInfo) in untrackedFiles)
                {
                    var candidateDir = Path.GetDirectoryName(candidatePath) ?? "";
                    var candidateExt = candidateInfo.Extension;

                    // Must be same folder and same extension
                    if (!oldDir.Equals(candidateDir, StringComparison.OrdinalIgnoreCase))
                        continue;
                    if (!oldExt.Equals(candidateExt, StringComparison.OrdinalIgnoreCase))
                        continue;

                    // Prefer same file size (strong signal for rename vs replace)
                    if (meta.FileSize > 0 && candidateInfo.Length != meta.FileSize)
                        continue;

                    // Pick closest name (simple character distance for tie-breaking)
                    var oldName = Path.GetFileNameWithoutExtension(meta.SourcePath);
                    var newName = Path.GetFileNameWithoutExtension(candidatePath);
                    int dist = Math.Abs(oldName.Length - newName.Length);
                    if (dist < bestDistance)
                    {
                        bestDistance = dist;
                        bestPath = candidatePath;
                    }
                }

                if (bestPath != null)
                {
                    // Adopt: update meta's SourcePath to the renamed file
                    var oldPath = meta.SourcePath;
                    meta.SourcePath = bestPath;
                    meta.LastImported = DateTime.MinValue; // Force reimport
                    WriteMetaFile(meta);
                    RegisterMeta(meta);
                    matchedPaths.Add(bestPath);
                    untrackedFiles.Remove(bestPath);
                    orphanedMetas.Remove(meta);

                    Debug.Log($"[AssetDatabase] Detected rename: {oldPath} → {bestPath} (GUID preserved)");
                }
            }

            // 6. Delete truly orphaned metas (source was deleted, not renamed)
            foreach (var meta in orphanedMetas)
            {
                var bucket = meta.Guid[..2];
                var metaPath = Path.Combine(libraryDir, bucket, $"{meta.Guid}.meta");
                if (File.Exists(metaPath))
                {
                    File.Delete(metaPath);
                    Debug.Log($"[AssetDatabase] Removed orphan: {meta.SourcePath}");
                }
            }

            // 7. Create .meta for remaining untracked source files
            foreach (var (path, fileInfo) in untrackedFiles)
            {
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

        /// <summary>
        /// Rename an asset's source file on disk and update all internal lookups.
        /// The GUID is preserved, so all scene references remain valid.
        /// Returns true on success.
        /// </summary>
        public static bool RenameAsset(string guid, string newName)
        {
            if (string.IsNullOrEmpty(guid) || string.IsNullOrEmpty(newName))
                return false;

            if (!_guidToMeta.TryGetValue(guid, out var meta))
                return false;

            var assetsDir = Project.AssetsDirectory;
            var oldRelative = meta.SourcePath;
            var oldFull = Path.Combine(assetsDir, oldRelative);

            if (!File.Exists(oldFull))
            {
                Debug.LogWarning("AssetDatabase", $"Cannot rename: source file not found: {oldRelative}");
                return false;
            }

            var ext = Path.GetExtension(oldRelative);
            var dir = Path.GetDirectoryName(oldRelative) ?? "";
            var newRelative = Path.Combine(dir, newName + ext);
            var newFull = Path.Combine(assetsDir, newRelative);

            // Don't overwrite existing files
            if (File.Exists(newFull))
            {
                Debug.LogWarning("AssetDatabase", $"Cannot rename: file already exists: {newRelative}");
                return false;
            }

            try
            {
                File.Move(oldFull, newFull);
            }
            catch (Exception ex)
            {
                Debug.LogWarning("AssetDatabase", $"Rename failed: {ex.Message}");
                return false;
            }

            // Update internal lookups
            var oldNormalized = NormalizePath(oldRelative);
            var newNormalized = NormalizePath(newRelative);

            _pathToGuid.TryRemove(oldNormalized, out _);
            _pathToGuid[newNormalized] = guid;
            _guidToPath[guid] = newRelative;

            // Update meta file
            meta.SourcePath = newRelative;
            WriteMetaFile(meta);

            // Update name on any loaded asset instance
            var loadedAsset = Engine.Assets?.FindByGuid(guid);
            if (loadedAsset != null)
                loadedAsset.Name = newName;

            Debug.Log($"[AssetDatabase] Renamed: {oldRelative} → {newRelative}");
            return true;
        }

        private static void RegisterMeta(MetaFile meta)
        {
            var normalized = NormalizePath(meta.SourcePath);
            _guidToPath[meta.Guid] = meta.SourcePath;
            _pathToGuid[normalized] = meta.Guid;
            _guidToMeta[meta.Guid] = meta;

            lock (meta)
            {
                if (meta.SubAssets.Count > 0)
                {
                    foreach (var sub in meta.SubAssets)
                    {
                        _subAssetToSource[sub.Guid] = meta.Guid;
                        var list = _nameToSubAssets.GetOrAdd(sub.Name, _ => new List<SubAssetEntry>());
                        lock (list) { list.Add(sub); }
                    }
                }
                else if (!string.IsNullOrEmpty(meta.MainAssetType))
                {
                    _sourceGuidCacheType[meta.Guid] = meta.MainAssetType;
                }
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
            else if (meta.SubAssets.Count > 0)
            {
                // Compound asset: check if any subasset cache file is missing
                foreach (var sub in meta.SubAssets)
                {
                    var cachePath = GetCachePath(sub.Guid, sub.Type);
                    if (!File.Exists(cachePath))
                        return true;
                }
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

        /// <summary>
        /// Import priority from AssetImporterAttribute. Lower = earlier.
        /// Falls back to int.MaxValue for unknown extensions.
        /// </summary>
        private static int GetImportPriority(MetaFile meta)
        {
            var ext = Path.GetExtension(meta.SourcePath);
            return _importerPriority.TryGetValue(ext, out var priority) ? priority : int.MaxValue;
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

            // Restore saved importer settings (preserves user overrides across reimports)
            if (!string.IsNullOrEmpty(meta.ImporterSettings))
            {
                try
                {
                    var restored = System.Text.Json.JsonSerializer.Deserialize(meta.ImporterSettings, importerType, _importerJsonOptions) as IImporter;
                    if (restored != null) importer = restored;
                }
                catch { /* ignore deserialization errors — use fresh defaults */ }
            }

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
            // Save existing state so we can preserve GUIDs on reimport
            List<SubAssetEntry> oldSubAssets;
            string oldMainAssetType;
            lock (meta)
            {
                oldSubAssets = new List<SubAssetEntry>(meta.SubAssets);
                oldMainAssetType = meta.MainAssetType;
                meta.SubAssets.Clear();
                meta.MainAssetType = null;
            }

            if (result.Artifacts.Count == 1 && !result.Compound)
            {
                // Simple asset: source GUID = cache key, no subassets
                var artifact = result.Artifacts[0];
                var dataTypeName = artifact.Data.GetType().Name;
                var cachePath = GetCachePath(meta.Guid, dataTypeName);
                Directory.CreateDirectory(Path.GetDirectoryName(cachePath));
                PackArtifact(artifact, cachePath);
                meta.MainAssetType = dataTypeName;

                // Store semantic type when it differs from packer type
                // (e.g. artifact.Type = "PCGGraph" but dataTypeName = "AssetDefinitionData")
                if (!string.IsNullOrEmpty(artifact.Type) && artifact.Type != dataTypeName)
                    meta.MainSemanticType = artifact.Type;
                else
                    meta.MainSemanticType = null;
            }
            else
            {
                // Compound asset: each artifact gets its own GUID
                // Reuse existing GUIDs from previous import to keep scene references stable
                foreach (var artifact in result.Artifacts)
                {
                    var dataTypeName = artifact.Data.GetType().Name;
                    var existing = oldSubAssets.Find(s => s.Name == artifact.Name && s.Type == dataTypeName);

                    string subGuid;
                    if (existing != null)
                        subGuid = existing.Guid;                        // Compound→compound: reuse subasset GUID
                    else if (oldMainAssetType == dataTypeName)
                        subGuid = meta.Guid;                            // Simple→compound: reuse source GUID
                    else
                        subGuid = System.Guid.NewGuid().ToString("N");  // New artifact

                    var cachePath = GetCachePath(subGuid, dataTypeName);
                    Directory.CreateDirectory(Path.GetDirectoryName(cachePath));

                    if (PackArtifact(artifact, cachePath))
                    {
                        lock (meta)
                        {
                            meta.SubAssets.Add(new SubAssetEntry
                            {
                                Guid = subGuid,
                                Name = artifact.Name,
                                Type = dataTypeName,
                                AssetType = artifact.Type != dataTypeName ? artifact.Type : null,
                                Hidden = artifact.Hidden,
                            });
                        }
                    }
                }
            }

            // ── Post-import: emit additional artifacts that reference first-pass GUIDs ──
            if (importer is IPostImporter postImporter)
            {
                List<ImportArtifact> extras = null;
                List<SubAssetEntry> subAssetSnapshot;
                lock (meta) { subAssetSnapshot = new List<SubAssetEntry>(meta.SubAssets); }

                try
                {
                    extras = postImporter.PostImport(sourcePath, subAssetSnapshot);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("AssetDatabase", $"PostImport failed: {meta.SourcePath} — {ex.Message}");
                }

                if (extras != null)
                {
                    foreach (var artifact in extras)
                    {
                        var dataTypeName = artifact.Data.GetType().Name;
                        var existing = oldSubAssets.Find(s => s.Name == artifact.Name && s.Type == dataTypeName);
                        var subGuid = existing?.Guid ?? System.Guid.NewGuid().ToString("N");

                        var cachePath = GetCachePath(subGuid, dataTypeName);
                        Directory.CreateDirectory(Path.GetDirectoryName(cachePath));

                        if (PackArtifact(artifact, cachePath))
                        {
                            lock (meta)
                            {
                                meta.SubAssets.Add(new SubAssetEntry
                                {
                                    Guid = subGuid,
                                    Name = artifact.Name,
                                    Type = dataTypeName,
                                    AssetType = artifact.Type != dataTypeName ? artifact.Type : null,
                                    Hidden = artifact.Hidden,
                                });
                            }
                        }
                    }
                }
            }

            meta.ImporterType = importerType.FullName;
            meta.LastImported = DateTime.UtcNow;
            meta.FileSize = new FileInfo(sourcePath).Length;

            // Serialize importer state (includes auto-populated Parts list, user overrides, etc.)
            try { meta.ImporterSettings = System.Text.Json.JsonSerializer.Serialize(importer, importerType, _importerJsonOptions); }
            catch (Exception ex) { Debug.LogWarning("AssetDatabase", $"Failed to serialize importer settings for {meta.SourcePath}: {ex.Message}"); meta.ImporterSettings = null; }

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

        /// <summary>
        /// Callback invoked when a thumbnail texture is loaded for the first time.
        /// The Editor hooks this to inject the texture into SquidRenderer.
        /// </summary>
        public static Action<string, Graphics.Texture> OnThumbnailLoaded;

        /// <summary>
        /// Get a thumbnail texture name for an asset by GUID.
        /// Lazy-loads the thumbnail PNG from disk on first access.
        /// Returns InternalAssets.White name if no thumbnail exists.
        /// </summary>
        public static string GetThumbnail(string guid)
        {
            var fallback = InternalAssets.Gray?.Name ?? "";

            if (string.IsNullOrEmpty(guid))
                return fallback;

            // Already loaded?
            if (_thumbTextures.TryGetValue(guid, out var tex))
                return tex?.Name ?? fallback;
            // Thumbnail file exists?
            if (!_guidToThumb.TryGetValue(guid, out var thumbPath) || !File.Exists(thumbPath))
                return fallback;

            // Lazy load the PNG into a GPU texture
            try
            {
                var texture = Graphics.Texture.LoadFromFile(Engine.Device, thumbPath);
                if (texture != null)
                {
                    texture.Name = $"thumb_{guid}";
                    _thumbTextures[guid] = texture;
                    OnThumbnailLoaded?.Invoke(texture.Name, texture);
                    return texture.Name;
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("AssetDatabase", $"Failed to load thumbnail: {thumbPath} — {ex.Message}");
            }

            return fallback;
        }

        /// <summary>
        /// Get a thumbnail texture name for an asset.
        /// Convenience overload that extracts the GUID from the asset.
        /// </summary>
        public static string GetThumbnail(Asset asset)
        {
            return GetThumbnail(asset?.Guid);
        }

        // ── Thumbnail Management ──

        /// <summary>
        /// Compute the thumbnail file path for a GUID: Thumbnails/{guid[..2]}/{guid}.png
        /// </summary>
        public static string GetThumbnailPath(string guid)
        {
            if (Project == null || string.IsNullOrEmpty(guid)) return null;
            var bucket = guid[..2];
            return Path.Combine(Project.ThumbnailsDirectory, bucket, $"{guid}.png");
        }

        /// <summary>
        /// Check if a thumbnail exists for the given GUID.
        /// </summary>
        public static bool HasThumbnail(string guid)
        {
            return _guidToThumb.ContainsKey(guid);
        }

        /// <summary>
        /// Register a thumbnail after generation.
        /// </summary>
        public static void RegisterThumbnail(string guid, string path)
        {
            _guidToThumb[guid] = path;
        }

        /// <summary>
        /// Mark a thumbnail as permanently failed. Writes a .fail marker file to disk
        /// so it survives restarts and is never retried.
        /// </summary>
        public static void MarkThumbnailFailed(string guid)
        {
            _guidToThumb[guid] = "FAILED";
            if (Project != null)
            {
                var thumbPath = GetThumbnailPath(guid);
                if (thumbPath != null)
                {
                    var failPath = Path.ChangeExtension(thumbPath, ".fail");
                    try
                    {
                        Directory.CreateDirectory(Path.GetDirectoryName(failPath)!);
                        File.WriteAllBytes(failPath, Array.Empty<byte>());
                    }
                    catch { /* best effort */ }
                }
            }
        }

        /// <summary>
        /// Scan the Thumbnails/ directory and populate _guidToThumb.
        /// </summary>
        private static void ScanThumbnails()
        {
            if (Project == null) return;
            var thumbDir = Project.ThumbnailsDirectory;
            if (!Directory.Exists(thumbDir)) return;

            foreach (var file in Directory.EnumerateFiles(thumbDir, "*.png", SearchOption.AllDirectories))
            {
                var name = Path.GetFileNameWithoutExtension(file);
                _guidToThumb[name] = file;
            }

            // Also scan .fail markers — assets that will never produce a thumbnail
            foreach (var file in Directory.EnumerateFiles(thumbDir, "*.fail", SearchOption.AllDirectories))
            {
                var name = Path.GetFileNameWithoutExtension(file);
                _guidToThumb.TryAdd(name, "FAILED");
            }
        }

        /// <summary>
        /// Delete thumbnails whose GUID no longer maps to a tracked asset.
        /// </summary>
        private static void CleanOrphanedThumbnails()
        {
            if (Project == null) return;
            var thumbDir = Project.ThumbnailsDirectory;
            if (!Directory.Exists(thumbDir)) return;

            int removed = 0;
            foreach (var file in Directory.EnumerateFiles(thumbDir, "*.png", SearchOption.AllDirectories))
            {
                var guid = Path.GetFileNameWithoutExtension(file);
                // Check if this GUID is a known source or subasset
                if (!_guidToMeta.ContainsKey(guid) && !_subAssetToSource.ContainsKey(guid))
                {
                    try
                    {
                        File.Delete(file);
                        _guidToThumb.TryRemove(guid, out _);
                        removed++;
                    }
                    catch { }
                }
            }

            if (removed > 0)
                Debug.Log($"[AssetDatabase] Cleaned {removed} orphaned thumbnails");
        }

        /// <summary>
        /// Resolve a data type name (from meta files) to a runtime Type.
        /// Uses the reverse alias map built during DiscoverThumbnailGenerators.
        /// </summary>
        private static Type ResolveAssetType(string dataTypeName)
        {
            if (string.IsNullOrEmpty(dataTypeName)) return null;
            return _typeAliases.TryGetValue(dataTypeName, out var type) ? type : null;
        }

        /// <summary>
        /// Get all tracked GUIDs that are missing thumbnails and have a registered generator.
        /// Returns (guid, resolvedType) pairs.
        /// </summary>
        public static List<(string guid, Type assetType)> GetMissingThumbnails()
        {
            var missing = new List<(string guid, Type assetType)>();

            foreach (var meta in _guidToMeta.Values)
            {
                // Simple asset: source GUID is the cache key
                if (meta.MainAssetType != null)
                {
                    if (!_guidToThumb.ContainsKey(meta.Guid))
                    {
                        var type = ResolveAssetType(meta.MainAssetType);
                        if (type != null)
                            missing.Add((meta.Guid, type));
                    }
                    continue;
                }

                // Compound asset: check each subasset
                lock (meta)
                {
                    foreach (var sub in meta.SubAssets)
                    {
                        if (sub.Hidden) continue;
                        if (!_guidToThumb.ContainsKey(sub.Guid))
                        {
                            var type = ResolveAssetType(sub.AssetType ?? sub.Type);
                            if (type != null)
                                missing.Add((sub.Guid, type));
                        }
                    }
                }
            }

            return missing;
        }

        /// <summary>
        /// Discover IThumbnailGenerator implementations marked with [ThumbnailGenerator].
        /// Also builds the reverse type alias map from AssetTypeAliasAttribute.
        /// Must be called after both Engine and Editor assemblies are loaded.
        /// </summary>
        public static void DiscoverThumbnailGenerators()
        {
            _thumbGenerators.Clear();
            _typeAliases.Clear();

            var assemblies = new[] { Assembly.GetExecutingAssembly(), Assembly.GetEntryAssembly() };

            // Build reverse alias map: data type name string → runtime Type
            // e.g. "MeshData" → typeof(Mesh), "PrefabData" → typeof(Prefab)
            foreach (var assembly in assemblies)
            {
                if (assembly == null) continue;
                try
                {
                    foreach (var type in assembly.GetTypes())
                    {
                        if (!typeof(Asset).IsAssignableFrom(type)) continue;
                        _typeAliases[type.Name] = type;

                        var aliases = type.GetCustomAttributes<AssetTypeAliasAttribute>();
                        foreach (var alias in aliases)
                            _typeAliases[alias.Alias] = type;
                    }
                }
                catch (System.Reflection.ReflectionTypeLoadException) { }
            }

            // Discover thumbnail generator implementations.
            // Cache instances so one class with multiple [ThumbnailGenerator] attributes
            // only gets instantiated once.
            var instances = new Dictionary<Type, IThumbnailGenerator>();
            foreach (var assembly in assemblies)
            {
                if (assembly == null) continue;
                try
                {
                    foreach (var type in assembly.GetTypes())
                    {
                        if (type.IsAbstract || type.IsInterface) continue;
                        if (!typeof(IThumbnailGenerator).IsAssignableFrom(type)) continue;

                        var attrs = type.GetCustomAttributes<ThumbnailGeneratorAttribute>();
                        foreach (var attr in attrs)
                        {
                            if (_thumbGenerators.ContainsKey(attr.AssetType)) continue;

                            if (!instances.TryGetValue(type, out var instance))
                            {
                                instance = (IThumbnailGenerator)Activator.CreateInstance(type);
                                instances[type] = instance;
                            }
                            _thumbGenerators[attr.AssetType] = instance;
                        }
                    }
                }
                catch (System.Reflection.ReflectionTypeLoadException) { }
            }

            Debug.Log($"[AssetDatabase] Discovered {_thumbGenerators.Count} thumbnail generators, {_typeAliases.Count} type aliases");
        }

        /// <summary>
        /// Generate thumbnails for all tracked assets that don't have one yet.
        /// Creates a shared GPU rendering context for the batch.
        /// Must be called on the main thread (uses GPU).
        /// </summary>
        public static void GenerateMissingThumbnails(Action<string> progress = null,
            Func<IDisposable> rendererFactory = null)
        {
            if (_thumbGenerators.Count == 0)
                DiscoverThumbnailGenerators();

            var missing = GetMissingThumbnails();
            if (missing.Count == 0) return;

            // Flush streaming uploads so meshes have valid GPU buffers
            Graphics.StreamingManager.Instance?.Flush();

            progress?.Invoke($"Generating {missing.Count} thumbnails...");

            // Snapshot cache: preserve lightweight/shared assets (Effects, Materials).
            // Meshes and Textures must be evicted after each thumb to stay within
            // MeshRegistry capacity (4096 slots). We use EvictForThumbnails which
            // frees GPU buffers and MeshRegistry slots but does NOT release bindless
            // descriptor indices — recycling those mid-batch causes DEVICE_REMOVED.
            var cacheSnapshot = new HashSet<string>();
            foreach (var key in Engine.Assets.GetCachedKeys())
            {
                var asset = Engine.Assets.TryGet(key);
                if (asset is Graphics.Mesh || asset is Graphics.Texture) continue;
                cacheSnapshot.Add(key);
            }

            // Create the shared rendering context for this batch
            using var renderer = rendererFactory?.Invoke();

            int generated = 0;
            int skipped = 0;
            int index = 0;

            foreach (var (guid, assetType) in missing)
            {
                index++;

                if (!_thumbGenerators.TryGetValue(assetType, out var generator))
                {
                    skipped++;
                    continue;
                }

                // Abort if GPU is dead (DEVICE_REMOVED from a prior thumbnail)
                if (Engine.Device.IsDeviceLost)
                {
                    Debug.LogWarning("AssetDatabase", $"GPU device lost before thumbnail #{index} — aborting");
                    break;
                }

                try
                {
                    progress?.Invoke($"Thumbnail {generated + skipped + 1}/{missing.Count}: {assetType.Name}");
                    if (generator.Generate(guid, Engine.Assets, renderer))
                    {
                        generated++;
                        Debug.Log($"[Thumb] OK #{index}: {guid}");
                    }
                    else
                    {
                        Debug.Log($"[Thumb] SKIP #{index} (no result): {guid}");
                        MarkThumbnailFailed(guid);
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("AssetDatabase", $"Thumbnail #{index} failed for {guid} ({assetType.Name}): {ex.Message}");
                    MarkThumbnailFailed(guid);
                }

                if (Engine.Device.IsDeviceLost)
                {
                    Debug.LogWarning("AssetDatabase", $"GPU device lost AFTER Generate #{index} ({guid})");
                    break;
                }

                // Evict after each thumbnail to stay within MeshRegistry capacity.
                // Flush copy queue first so no pending uploads target buffers we're about to free.
                Graphics.StreamingManager.Instance?.Flush();

                if (Engine.Device.IsDeviceLost)
                {
                    Debug.LogWarning("AssetDatabase", $"GPU device lost AFTER Flush #{index} ({guid})");
                    break;
                }

                Engine.Assets.EvictAllExcept(cacheSnapshot, leakBindlessIndices: true);
            }

            if (generated > 0)
                Debug.Log($"[AssetDatabase] Generated {generated} thumbnails ({skipped} skipped, no generator)");
        }
    }
}
