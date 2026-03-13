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
    ///       resolves GUID references, loads PhysX HeightField and DeltaMap subassets.
    /// Save: writes YAML + reads back DeltaMap GPU data to cache.
    /// </summary>
    [AssetLoader(typeof(Terrain))]
    public class TerrainLoader : IAssetLoader
    {
        private readonly AssetDefinitionPacker _packer = new();

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

                // Load persisted DeltaMap data (painted terrain)
                LoadDeltaMaps(terrain, sourceGuid);

                Debug.Log($"[TerrainLoader] '{name}' loaded: {terrain.Layers?.Count ?? 0} layers, " +
                          $"{terrain.ControlMaps?.Count ?? 0} controlmaps, " +
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
        /// Loads persisted DeltaMap data from the cache and populates PendingDeltaMapData
        /// on the terrain's PaintHeightLayer. GPU upload happens later in TerrainRenderer.
        /// </summary>
        private static void LoadDeltaMaps(Terrain terrain, string guid)
        {
            try
            {
                if (string.IsNullOrEmpty(guid)) return;

                var cacheDir = AssetDatabase.Project?.CacheDirectory;
                if (string.IsNullOrEmpty(cacheDir)) return;

                var deltaPath = Path.Combine(cacheDir, $"{guid}.deltamap");
                if (!File.Exists(deltaPath)) return;

                var packer = new DeltaMapPacker();
                DeltaMapData data;
                using (var stream = File.OpenRead(deltaPath))
                    data = packer.Read(stream);

                // Assign to the first PaintHeightLayer found
                foreach (var layer in terrain.HeightLayers)
                {
                    if (layer is PaintHeightLayer paint)
                    {
                        paint.PendingDeltaMapData = data;
                        Debug.Log($"[TerrainLoader] DeltaMap loaded: {data.Width}x{data.Height} ({data.Pixels.Length} bytes)");
                        return;
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("TerrainLoader", $"Failed to load DeltaMap '{guid}': {ex.Message}");
            }
        }

        // ── Save ──

        /// <summary>
        /// Save terrain YAML + DeltaMap GPU data to cache.
        /// </summary>
        public void Save(Asset asset, string savePath)
        {
            if (asset is not Terrain terrain) return;

            try
            {
                // 1. Save YAML definition
                NativeImporter.Save(savePath, terrain);
                Debug.Log($"[TerrainLoader] YAML saved: {savePath}");

                // 2. Save DeltaMap data (GPU readback → cache)
                SaveDeltaMaps(terrain);
            }
            catch (Exception ex)
            {
                Debug.LogWarning("TerrainLoader", $"Failed to save terrain: {ex.Message}");
            }
        }

        /// <summary>
        /// Finds the active TerrainBaker, reads back DeltaMap from GPU, packs to cache.
        /// </summary>
        private static void SaveDeltaMaps(Terrain terrain)
        {
            if (string.IsNullOrEmpty(terrain.Guid)) return;

            // Find the active TerrainRenderer that owns this terrain
            var renderer = EntityManager.Entities
                .Select(e => e.GetComponent<Freefall.Components.TerrainRenderer>())
                .FirstOrDefault(tr => tr?.Terrain == terrain);

            if (renderer?.Baker == null) return;

            var deltaData = renderer.Baker.ReadbackDeltaMap();
            if (deltaData == null) return;

            var cacheDir = AssetDatabase.Project.CacheDirectory;
            var deltaPath = Path.Combine(cacheDir, $"{terrain.Guid}.deltamap");

            var packer = new DeltaMapPacker();
            using (var stream = File.Create(deltaPath))
                packer.Write(stream, deltaData);

            Debug.Log($"[TerrainLoader] DeltaMap saved: {deltaPath} ({deltaData.Width}x{deltaData.Height})");
        }
    }
}
