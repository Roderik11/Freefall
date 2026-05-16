using System;
using System.IO;
using System.Linq;
using System.Text;
using Freefall.Assets.Packers;
using Freefall.Serialization;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads and saves NavMesh assets.
    /// Load: unpacks AssetDefinitionData (YAML), deserializes NavMesh,
    ///       loads baked NavMeshData from hidden subasset.
    /// Save: writes YAML + saves baked navmesh bytes to cache.
    /// Mirrors the TerrainLoader pattern.
    /// </summary>
    [AssetLoader(typeof(NavMesh), ".navmesh")]
    public class NavMeshLoader : IAssetLoader
    {
        private readonly AssetDefinitionPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, "AssetDefinitionData");
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for navmesh '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            if (string.IsNullOrEmpty(sourceGuid))
                sourceGuid = AssetDatabase.ResolveGuidByName(name);

            try
            {
                AssetDefinitionData defData;
                using (var stream = File.OpenRead(cachePath))
                    defData = _packer.Read(stream);

                var yaml = Encoding.UTF8.GetString(defData.YamlBytes);
                var navMesh = NativeImporter.LoadFromString(yaml, manager) as NavMesh;

                if (navMesh == null)
                    throw new InvalidDataException($"Failed to deserialize NavMesh from cache: {name}");

                navMesh.Name = name;

                // Load baked navmesh binary data from hidden subasset
                LoadBakedData(navMesh, sourceGuid);

                navMesh.MarkReady();
                Debug.Log($"[NavMeshLoader] '{name}' loaded: {navMesh.PolyCount} polys, {navMesh.VertexCount} verts, " +
                          $"MeshData={(navMesh.MeshData?.Length ?? 0)} bytes");

                return navMesh;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("NavMeshLoader", $"FAILED to load '{name}': {ex}");
                return null;
            }
        }

        /// <summary>
        /// Find the NavMeshData hidden subasset and load the baked bytes.
        /// </summary>
        private static void LoadBakedData(NavMesh navMesh, string guid)
        {
            if (string.IsNullOrEmpty(guid)) return;

            var meta = AssetDatabase.GetMeta(guid);
            if (meta == null) return;

            var navSub = meta.SubAssets.FirstOrDefault(
                s => s.Type == nameof(NavMeshData));
            if (navSub == null) return;

            var cachePath = AssetDatabase.ResolveCachePathByGuid(navSub.Guid);
            if (cachePath == null || !File.Exists(cachePath)) return;

            try
            {
                var packer = new NavMeshDataPacker();
                using var stream = File.OpenRead(cachePath);
                var data = packer.Read(stream);

                if (data?.BakedBytes != null && data.BakedBytes.Length > 0)
                {
                    navMesh.MeshData = data.BakedBytes;
                    Debug.Log($"[NavMeshLoader] Loaded baked data: {data.BakedBytes.Length} bytes");
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("NavMeshLoader", $"Failed to load baked navmesh data: {ex.Message}");
            }
        }

        // ── Save ──

        /// <summary>
        /// Save NavMesh YAML + baked binary data to cache.
        /// </summary>
        public void Save(Asset asset, string savePath)
        {
            if (asset is not NavMesh navMesh) return;

            try
            {
                // 1. Save baked data to cache (must happen before YAML so GUID is set)
                SaveBakedData(navMesh);

                // 2. Save YAML definition
                NativeImporter.Save(savePath, navMesh);
                Debug.Log($"[NavMeshLoader] YAML saved: {savePath}");
            }
            catch (Exception ex)
            {
                Debug.LogWarning("NavMeshLoader", $"Failed to save navmesh: {ex.Message}");
            }
        }

        /// <summary>
        /// Save the baked navmesh bytes as a hidden subasset in cache.
        /// </summary>
        private static void SaveBakedData(NavMesh navMesh)
        {
            if (navMesh.MeshData == null || navMesh.MeshData.Length == 0)
                return;

            if (string.IsNullOrEmpty(navMesh.Guid)) return;

            // Find existing or create new NavMeshData subasset
            var meta = AssetDatabase.GetMeta(navMesh.Guid);
            if (meta == null) return;

            var navSub = meta.SubAssets.FirstOrDefault(
                s => s.Type == nameof(NavMeshData));

            string subGuid;
            if (navSub != null)
            {
                subGuid = navSub.Guid;
            }
            else
            {
                subGuid = AssetDatabase.AddOrUpdateSubAsset(
                    navMesh.Guid, nameof(NavMeshData), navMesh.Name, hidden: true);
                if (subGuid == null) return;
            }

            // Write baked bytes to cache
            var cachePath = AssetDatabase.ResolveCachePathByGuid(subGuid);
            if (cachePath == null)
            {
                var cacheDir = AssetDatabase.Project.CacheDirectory;
                var bucket = subGuid[..2];
                cachePath = Path.Combine(cacheDir, bucket, $"{subGuid}.navmesh");
                Directory.CreateDirectory(Path.GetDirectoryName(cachePath)!);
            }

            var packer = new NavMeshDataPacker();
            using var stream = File.Create(cachePath);
            packer.Write(stream, new NavMeshData { BakedBytes = navMesh.MeshData });

            Debug.Log($"[NavMeshLoader] Baked data saved: {navMesh.MeshData.Length} bytes → {cachePath}");
        }
    }
}
