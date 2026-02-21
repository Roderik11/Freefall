using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Freefall.Graphics;

namespace Freefall.Assets
{
    /// <summary>
    /// Manages loading and caching of assets with generic Load&lt;T&gt; support.
    /// 
    /// Runtime path: discovers IAssetLoader implementations via [AssetLoader] attribute,
    /// resolves cache paths through AssetDatabase, and delegates loading to the appropriate loader.
    /// 
    /// Fallback path: if no loader is found, falls back to legacy AssetImporter&lt;T&gt; for
    /// direct source file loading (will be removed once all types have loaders).
    /// </summary>
    public class AssetManager : IDisposable
    {
        private readonly GraphicsDevice _device;
        private readonly ConcurrentDictionary<string, Asset> _assets = new();
        private readonly ConcurrentDictionary<string, Lock> _loadLocks = new();

        // Asset type → loader instance (discovered from [AssetLoader] attribute)
        private static readonly Dictionary<Type, IAssetLoader> Loaders = [];

        // Legacy: Asset type → (extension → importer type) for fallback
        private static readonly Dictionary<Type, Dictionary<string, Type>> Importers = [];

        public string BaseDirectory { get; set; }
        public Texture White { get; private set; }

        static AssetManager()
        {
            DiscoverLoaders();
            DiscoverImporters();
        }

        public AssetManager(GraphicsDevice device)
        {
            _device = device;
            
            // Create default white texture
            byte[] data = new byte[4 * 4 * 4];
            for (int i = 0; i < data.Length; i++) data[i] = 255;
            White = Texture.CreateFromData(device, 4, 4, data, Vortice.DXGI.Format.R8G8B8A8_UNorm);
        }

        /// <summary>
        /// Register a pre-built asset (e.g. InternalAssets) with a stable GUID.
        /// Makes it discoverable via LoadByGuid&lt;T&gt;().
        /// </summary>
        public void RegisterAsset<T>(string guid, T asset) where T : Asset
        {
            asset.Guid = guid;
            string cacheKey = $"{typeof(T).Name}:guid:{guid}";
            _assets[cacheKey] = asset;
        }

        /// <summary>
        /// Discover all IAssetLoader implementations marked with [AssetLoader].
        /// </summary>
        private static void DiscoverLoaders()
        {
            Loaders.Clear();
            var assemblies = new[] { Assembly.GetExecutingAssembly(), Assembly.GetEntryAssembly() };

            foreach (var assembly in assemblies)
            {
                if (assembly == null) continue;

                Type[] types;
                try { types = assembly.GetTypes(); }
                catch (ReflectionTypeLoadException ex)
                {
                    types = ex.Types.Where(t => t != null).ToArray()!;
                }

                foreach (var type in types)
                {
                    var attr = type.GetCustomAttribute<AssetLoaderAttribute>();
                    if (attr == null) continue;

                    if (!typeof(IAssetLoader).IsAssignableFrom(type)) continue;

                    if (!Loaders.ContainsKey(attr.AssetType))
                    {
                        Loaders[attr.AssetType] = (IAssetLoader)Activator.CreateInstance(type);
                    }
                }
            }
            Debug.Log($"[AssetManager] Discovered {Loaders.Count} asset loaders");
        }

        /// <summary>
        /// Legacy: Discover AssetImporter&lt;T&gt; implementations for fallback loading.
        /// </summary>
        private static void DiscoverImporters()
        {
            Importers.Clear();
            var assemblies = new[] { Assembly.GetExecutingAssembly(), Assembly.GetEntryAssembly() };

            foreach (var assembly in assemblies)
            {
                if (assembly == null) continue;

                Type[] types;
                try { types = assembly.GetTypes(); }
                catch (ReflectionTypeLoadException ex)
                {
                    types = ex.Types.Where(t => t != null).ToArray()!;
                    Debug.LogWarning("AssetManager", $"Partial type load for {assembly.GetName().Name}: {ex.LoaderExceptions.Length} failures");
                }

                foreach (var type in types)
                {
                    var attr = type.GetCustomAttribute<AssetImporterAttribute>();
                    if (attr == null) continue;

                    var baseType = type.BaseType;
                    if (baseType == null || !baseType.IsGenericType) continue;
                    
                    var assetType = baseType.GetGenericArguments()[0];

                    if (!Importers.TryGetValue(assetType, out var extensionMap))
                    {
                        extensionMap = new Dictionary<string, Type>(StringComparer.OrdinalIgnoreCase);
                        Importers[assetType] = extensionMap;
                    }

                    foreach (var ext in attr.Extensions)
                    {
                        var normalizedExt = ext.StartsWith(".") ? ext : "." + ext;
                        if (!extensionMap.ContainsKey(normalizedExt))
                            extensionMap[normalizedExt] = type;
                    }
                }
            }
            Debug.Log($"[AssetManager] Discovered {Importers.Count} legacy asset importers");
        }

        /// <summary>
        /// Load an asset by name or path. Tries cache-based loaders first,
        /// falls back to legacy source-file importers.
        /// </summary>
        public T Load<T>(string path) where T : Asset
        {
            string fullPath = path;
            if (!Path.IsPathRooted(path) && !string.IsNullOrEmpty(BaseDirectory))
            {
                fullPath = Path.Combine(BaseDirectory, path);
            }
            string cacheKey = $"{typeof(T).Name}:{fullPath}";

            // Fast path: already cached
            if (_assets.TryGetValue(cacheKey, out var cached))
                return (T)cached;

            var assetType = typeof(T);

            // Per-key lock prevents duplicate loads of the same asset
            var loadLock = _loadLocks.GetOrAdd(cacheKey, _ => new());
            lock (loadLock)
            {
                // Double-check after acquiring lock
                if (_assets.TryGetValue(cacheKey, out var cached2))
                    return (T)cached2;

                T asset = null;

                // ── Primary path: cache-based loader ──
                if (Loaders.TryGetValue(assetType, out var loader))
                {
                    var name = Path.GetFileNameWithoutExtension(path);
                    try
                    {
                        asset = (T)loader.Load(name, this);
                    }
                    catch (FileNotFoundException)
                    {
                        // Cache miss — fall through to legacy importer
                        Debug.Log("AssetManager", $"Cache miss for {assetType.Name} '{name}', trying legacy importer");
                    }
                }

                // ── Fallback: legacy source-file importer ──
                if (asset == null)
                {
                    asset = LoadLegacy<T>(fullPath, assetType);
                }

                if (asset != null)
                {
                    asset.Name ??= Path.GetFileNameWithoutExtension(path);
                    asset.AssetPath = path;
                    _assets[cacheKey] = asset;
                }

                return asset;
            }
        }

        /// <summary>
        /// Load an asset by GUID. Used by loaders to resolve cross-asset references.
        /// </summary>
        public T LoadByGuid<T>(string guid) where T : Asset
        {
            if (string.IsNullOrEmpty(guid))
                return null;

            string cacheKey = $"{typeof(T).Name}:guid:{guid}";

            if (_assets.TryGetValue(cacheKey, out var cached))
                return (T)cached;

            // Resolve GUID → cache file path
            var cachePath = AssetDatabase.ResolveCachePathByGuid(guid);
            if (cachePath == null || !File.Exists(cachePath))
            {
                Debug.LogWarning("AssetManager", $"Cannot resolve GUID '{guid}' for type {typeof(T).Name}");
                return null;
            }

            var assetType = typeof(T);
            T asset = null;

            // Load via cache-based loader if available
            if (Loaders.TryGetValue(assetType, out var loader))
            {
                try
                {
                    // Use the friendly name from AssetDatabase for display
                    var sourcePath = AssetDatabase.GuidToPath(guid);
                    var name = sourcePath != null
                        ? Path.GetFileNameWithoutExtension(sourcePath)
                        : guid;

                    asset = (T)loader.LoadFromCache(cachePath, name, this);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("AssetManager", $"Failed to load GUID '{guid}': {ex.Message}");
                }
            }

            if (asset != null)
            {
                asset.Guid = guid;
                _assets[cacheKey] = asset;
            }

            return asset;
        }

        /// <summary>
        /// Legacy loading path: creates importer instance and calls Load(filepath).
        /// Will be removed once all asset types have cache-based loaders.
        /// </summary>
        private T LoadLegacy<T>(string fullPath, Type assetType) where T : Asset
        {
            var extension = Path.GetExtension(fullPath);

            if (!Importers.TryGetValue(assetType, out var extensionMap))
                throw new InvalidOperationException($"No loader or importer registered for asset type {assetType.Name}");

            if (!extensionMap.TryGetValue(extension, out var importerType))
                throw new InvalidOperationException($"No importer for extension {extension} for asset type {assetType.Name}");

            var importer = Activator.CreateInstance(importerType);
            var importMethod = importerType.GetMethod("Load");
            return (T)importMethod.Invoke(importer, [fullPath]);
        }

        public Task<T> LoadAsync<T>(string path) where T : Asset
        {
             // For now, wrap synchronous load.
             // Ideally we'd use an async loader interface.
             return Task.Run(() => Load<T>(path));
        }

        /// <summary>
        /// Legacy method - use Load&lt;Texture&gt; instead.
        /// </summary>
        public Texture LoadTexture(string path) => Load<Texture>(path);

        /// <summary>
        /// Legacy method - use Load&lt;Mesh&gt; instead.
        /// </summary>
        public Mesh LoadMesh(string path) => Load<Mesh>(path);

        public void Dispose()
        {
            foreach (var asset in _assets.Values)
            {
                if (asset is IDisposable disposable)
                    disposable.Dispose();
            }
            _assets.Clear();
            Debug.Log("[AssetManager] Disposed all assets.");
        }
    }
}
