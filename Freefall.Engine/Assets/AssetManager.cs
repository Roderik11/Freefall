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
    /// Manages loading and caching of assets with generic Load<T> support.
    /// Like Apex's AssetManager.
    /// </summary>
    public class AssetManager : IDisposable
    {
        private readonly GraphicsDevice _device;
        private readonly ConcurrentDictionary<string, Asset> _assets = new ConcurrentDictionary<string, Asset>();
        private readonly ConcurrentDictionary<string, object> _importLocks = new ConcurrentDictionary<string, object>();
        private static readonly Dictionary<Type, Dictionary<string, Type>> Importers = new Dictionary<Type, Dictionary<string, Type>>();

        public string BaseDirectory { get; set; }
        public Texture White { get; private set; }

        static AssetManager()
        {
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

        private static void DiscoverImporters()
        {
            Importers.Clear();
            var assemblies = new[] { Assembly.GetExecutingAssembly(), Assembly.GetEntryAssembly() };

            foreach (var assembly in assemblies)
            {
                if (assembly == null) continue;

                Type[] types;
                try
                {
                    types = assembly.GetTypes();
                }
                catch (ReflectionTypeLoadException ex)
                {
                    // Use the types that loaded successfully (skip nulls from failed loads)
                    types = ex.Types.Where(t => t != null).ToArray()!;
                    Debug.LogWarning("AssetManager", $"Partial type load for {assembly.GetName().Name}: {ex.LoaderExceptions.Length} failures");
                }

                foreach (var type in types)
                {
                    var attr = type.GetCustomAttribute<AssetReaderAttribute>();
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
            Debug.Log($"[AssetManager] Discovered {Importers.Count} asset types with importers");
        }

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
            var extension = Path.GetExtension(fullPath);

            if (!Importers.TryGetValue(assetType, out var extensionMap))
                throw new InvalidOperationException($"No importer registered for asset type {assetType.Name}");

            if (!extensionMap.TryGetValue(extension, out var importerType))
                throw new InvalidOperationException($"No importer for extension {extension} for asset type {assetType.Name}");

            // Per-key lock prevents duplicate imports of the same asset
            var importLock = _importLocks.GetOrAdd(cacheKey, _ => new object());
            lock (importLock)
            {
                // Double-check after acquiring lock
                if (_assets.TryGetValue(cacheKey, out var cached2))
                    return (T)cached2;

                var importer = Activator.CreateInstance(importerType);
                var importMethod = importerType.GetMethod("Import");
                var asset = (T)importMethod.Invoke(importer, new object[] { fullPath });

                if (asset != null)
                {
                    asset.Name = Path.GetFileNameWithoutExtension(path);
                    asset.AssetPath = path;
                    _assets[cacheKey] = asset;
                }

                return asset;
            }
        }
        
        public Task<T> LoadAsync<T>(string path) where T : Asset
        {
             // For now, wrap synchronous load.
             // Ideally we'd use an async importer interface.
             return Task.Run(() => Load<T>(path));
        }

        /// <summary>
        /// Legacy method - use Load<Texture> instead.
        /// </summary>
        public Texture LoadTexture(string path) => Load<Texture>(path);

        /// <summary>
        /// Legacy method - use Load<Mesh> instead.
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
