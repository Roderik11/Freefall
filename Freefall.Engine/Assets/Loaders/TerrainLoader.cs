using System;
using System.IO;
using System.Linq;
using System.Text;
using Freefall.Assets.Packers;
using Freefall.Base;
using Freefall.Graphics;
using Freefall.Serialization;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads and saves Terrain assets.
    /// Load: unpacks AssetDefinitionData (YAML) from cache, deserializes Terrain,
    ///       resolves GUID references, loads PhysX HeightField and ControlMap subassets.
    /// Save: writes YAML + reads back all ControlMap GPU data to cache.
    /// </summary>
    [AssetLoader(typeof(Terrain))]
    public class TerrainLoader : IAssetLoader
    {
        private readonly AssetDefinitionPacker _packer = new();
        private readonly DdsTexturePacker _ddsPacker = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, "AssetDefinitionData");
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for terrain '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            // If no explicit GUID, resolve it from name
            if (string.IsNullOrEmpty(sourceGuid))
                sourceGuid = AssetDatabase.ResolveGuidByName(name);

            try
            {
                Debug.Log($"[TerrainLoader] Loading '{name}' from {cachePath}");

                AssetDefinitionData defData;
                using (var stream = File.OpenRead(cachePath))
                    defData = _packer.Read(stream);

                // Deserialize YAML and auto-resolve all asset GUID references
                var yaml = Encoding.UTF8.GetString(defData.YamlBytes);
                Terrain terrain;
                try
                {
                    terrain = NativeImporter.LoadFromString(yaml, manager) as Terrain;
                }
                catch (Exception ex)
                {
                    throw new InvalidDataException(
                        $"Failed to deserialize Terrain from cache: {name} - {ex.GetType().Name}: {ex.Message}", ex);
                }

                if (terrain == null)
                    throw new InvalidDataException($"Failed to deserialize Terrain from cache: {name}");

                terrain.Name = name;

                // Build CPU-side height field from the resolved Heightmap texture
                if (terrain.Heightmap != null && !string.IsNullOrEmpty(terrain.Heightmap.Guid))
                {
                    var sourcePath = AssetDatabase.GuidToPath(terrain.Heightmap.Guid);
                    if (!string.IsNullOrEmpty(sourcePath))
                    {
                        var fullPath = Path.Combine(AssetDatabase.Project.AssetsDirectory, sourcePath);
                        if (File.Exists(fullPath))
                            terrain.BuildHeightField(fullPath);
                    }
                }

                // Load pre-cooked PhysX HeightField
                LoadCookedHeightField(terrain, sourceGuid);

                // Load persisted ControlMap data (painted height)
                LoadControlMaps(terrain);

                Debug.Log($"[TerrainLoader] '{name}' loaded: {terrain.Layers?.Count ?? 0} layers, " +
                          $"{terrain.Decorations?.Count ?? 0} decorations, " +
                          $"HeightField={terrain.HeightField != null}, " +
                          $"CookedHeightField={terrain.CookedHeightField != null}");

                return terrain;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("TerrainLoader", $"FAILED to load '{name}': {ex}");
                return null;
            }
        }

        /// <summary>
        /// Find a CollisionMeshData subasset by type in the meta, load it, and create
        /// a PhysX HeightField from the cooked bytes.
        /// </summary>
        private static void LoadCookedHeightField(Terrain terrain, string guid)
        {
            try
            {
                if (string.IsNullOrEmpty(guid)) return;

                var meta = AssetDatabase.GetMeta(guid);
                if (meta == null) return;

                var collisionSub = meta.SubAssets.FirstOrDefault(
                    s => s.Type == nameof(CollisionMeshData));
                if (collisionSub == null) return;

                var physxPath = AssetDatabase.ResolveCachePathByGuid(collisionSub.Guid);
                if (physxPath == null || !File.Exists(physxPath)) return;

                var packer = new CollisionMeshPacker();
                using var stream = File.OpenRead(physxPath);
                var cooked = packer.Read(stream);

                var hf = PhysicsWorld.Physics.CreateHeightField(new MemoryStream(cooked.CookedBytes));
                terrain.SetCookedHeightField(hf);

                Debug.Log($"[TerrainLoader] Pre-cooked HeightField loaded: {guid}");
            }
            catch (Exception ex)
            {
                Debug.LogWarning("TerrainLoader", $"Failed to load cooked HeightField '{guid}': {ex.Message}");
            }
        }

        /// <summary>
        /// Loads persisted ControlMap data for PaintHeightLayers.
        /// The ControlMap texture is resolved via normal GUID-based loading (it's a DDS subasset).
        /// This method handles the PendingControlMapBytes staging for GPU upload.
        /// </summary>
        private void LoadControlMaps(Terrain terrain)
        {
            // PaintHeightLayer ControlMaps need GPU upload staging
            // The Texture itself is already resolved by NativeImporter via GUID deferred refs,
            // but we need raw bytes for the GPU upload path
            foreach (var layer in terrain.HeightLayers)
            {
                if (layer is PaintHeightLayer paint && paint.ControlMap != null)
                {
                    // Load the DDS bytes from the subasset cache file
                    var bytes = LoadDdsBytes(paint.ControlMap.Guid);
                    if (bytes != null)
                    {
                        paint.PendingControlMapBytes = bytes;
                        Debug.Log($"[TerrainLoader] PaintHeightLayer ControlMap loaded: {bytes.Length} bytes");
                    }
                }
            }
        }

        /// <summary>
        /// Reads raw DDS bytes from a subasset cache file by GUID.
        /// </summary>
        private byte[] LoadDdsBytes(string guid)
        {
            if (string.IsNullOrEmpty(guid)) return null;

            var cachePath = AssetDatabase.ResolveCachePathByGuid(guid);
            if (cachePath == null || !File.Exists(cachePath)) return null;

            try
            {
                using var stream = File.OpenRead(cachePath);
                var data = _ddsPacker.Read(stream);
                return data?.Bytes;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("TerrainLoader", $"Failed to load DDS subasset '{guid}': {ex.Message}");
                return null;
            }
        }

        // ── Save ──

        /// <summary>
        /// Save terrain YAML + all ControlMap GPU textures to cache.
        /// </summary>
        public void Save(Asset asset, string savePath)
        {
            if (asset is not Terrain terrain) return;

            try
            {
                // 1. Save YAML definition
                NativeImporter.Save(savePath, terrain);
                Debug.Log($"[TerrainLoader] YAML saved: {savePath}");

                // 2. Save all ControlMap data (GPU readback → cache)
                SaveControlMaps(terrain);
            }
            catch (Exception ex)
            {
                Debug.LogWarning("TerrainLoader", $"Failed to save terrain: {ex.Message}");
            }
        }

        /// <summary>
        /// Finds the active TerrainBaker, reads back all ControlMaps from GPU, packs to cache.
        /// Handles: PaintHeightLayer ControlMaps, TextureLayer ControlMaps, Decoration ControlMaps.
        /// </summary>
        private void SaveControlMaps(Terrain terrain)
        {
            if (string.IsNullOrEmpty(terrain.Guid)) return;

            // Save PaintHeightLayer ControlMaps
            foreach (var layer in terrain.HeightLayers)
            {
                if (layer is PaintHeightLayer paint && paint.ControlMap != null)
                {
                    var pixels = TerrainBaker.Instance.ReadbackControlMap(TerrainBaker.ControlMapTarget.Height, 0);
                    if (pixels != null)
                    {
                        SaveDdsSubasset(paint.ControlMap.Guid, pixels);
                        Debug.Log($"[TerrainLoader] PaintHeightLayer ControlMap saved ({pixels.Length} bytes)");
                    }
                }
            }

            // TODO: Save TextureLayer ControlMaps (splatmaps)
            // TODO: Save Decoration ControlMaps (density maps)
        }

        /// <summary>
        /// Writes raw pixel bytes as a DDS subasset to the cache.
        /// </summary>
        private void SaveDdsSubasset(string guid, byte[] pixels)
        {
            if (string.IsNullOrEmpty(guid) || pixels == null) return;

            var cachePath = AssetDatabase.ResolveCachePathByGuid(guid);
            if (cachePath == null)
            {
                // Create a cache path for this subasset
                var cacheDir = AssetDatabase.Project.CacheDirectory;
                cachePath = Path.Combine(cacheDir, $"{guid}.dds");
            }

            using var stream = File.Create(cachePath);
            _ddsPacker.Write(stream, new DdsTextureData(pixels));
        }
    }
}
