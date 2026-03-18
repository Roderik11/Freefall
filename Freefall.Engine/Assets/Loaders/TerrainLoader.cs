using System;
using System.IO;
using System.Linq;
using System.Text;
using Freefall.Assets.Packers;
using Freefall.Base;
using Freefall.Graphics;
using Freefall.Serialization;
using PhysX;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads and saves Terrain assets.
    /// Load: unpacks AssetDefinitionData (YAML) from cache, deserializes Terrain,
    ///       resolves GUID references, loads PhysX HeightField and ControlMap subassets.
    /// Save: writes YAML + reads back all ControlMap GPU data to cache.
    /// </summary>
    [AssetLoader(typeof(Terrain), ".terrain")]
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

                // Load persisted baked heightmap (R16_Float DDS)
                LoadBakedHeightmap(terrain);

                // Load persisted ControlMap data (painted height, splatmaps, density)
                LoadControlMaps(terrain);

                Debug.Log($"[TerrainLoader] '{name}' loaded: {terrain.Layers?.Count ?? 0} layers, " +
                          $"{terrain.Decorations?.Count ?? 0} decorations, " +
                          $"HeightField={terrain.HeightField != null}, " +
                          $"CookedHeightField={terrain.CookedHeightField != null}");

                MessageDispatcher.Send("TerrainLoaded", terrain);

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
        /// Loads the saved baked heightmap DDS into PendingBakedHeightmapBytes for GPU upload.
        /// Also builds the CPU-side HeightField immediately so GetHeight() works before first render.
        /// </summary>
        private void LoadBakedHeightmap(Terrain terrain)
        {
            if (terrain.BakedHeightmapRef == null || string.IsNullOrEmpty(terrain.BakedHeightmapRef.Guid))
                return;

            var bytes = LoadDdsBytes(terrain.BakedHeightmapRef.Guid);
            if (bytes == null) return;

            terrain.PendingBakedHeightmapBytes = bytes;

            // Build CPU HeightField from R16_Float bytes so GetHeight() works immediately
            int pixelDataLen = bytes.Length;
            int resolution = (int)Math.Sqrt(pixelDataLen / 2);
            if (resolution * resolution * 2 == pixelDataLen)
            {
                var heights = new float[resolution, resolution];
                for (int y = 0; y < resolution; y++)
                    for (int x = 0; x < resolution; x++)
                    {
                        int idx = (y * resolution + x) * 2;
                        Half h = BitConverter.ToHalf(bytes, idx);
                        heights[x, y] = (float)h;
                    }
                terrain.SetHeightField(heights);
                Debug.Log($"[TerrainLoader] Baked heightmap loaded + CPU HeightField built: {resolution}x{resolution}");
            }
            else
            {
                Debug.Log($"[TerrainLoader] Baked heightmap loaded: {bytes.Length} bytes (CPU HeightField deferred)");
            }
        }

        /// <summary>
        /// Loads persisted ControlMap data for PaintHeightLayers.
        /// The ControlMap texture is resolved via normal GUID-based loading (it's a DDS subasset).
        /// This method handles the PendingControlMapBytes staging for GPU upload.
        /// </summary>
        private void LoadControlMaps(Terrain terrain)
        {
            // PaintHeightLayer ControlMaps
            foreach (var layer in terrain.HeightLayers)
            {
                if (layer is PaintHeightLayer paint && paint.ControlMap != null)
                {
                    var bytes = LoadDdsBytes(paint.ControlMap.Guid);
                    if (bytes != null)
                    {
                        paint.PendingControlMapBytes = bytes;
                        Debug.Log($"[TerrainLoader] PaintHeightLayer ControlMap loaded: {bytes.Length} bytes");
                    }
                }
            }

            // TextureLayer ControlMaps (splatmaps)
            if (terrain.Layers != null)
            {
                for (int i = 0; i < terrain.Layers.Count; i++)
                {
                    var layer = terrain.Layers[i];
                    if (layer.ControlMap == null || string.IsNullOrEmpty(layer.ControlMap.Guid)) continue;

                    var bytes = LoadDdsBytes(layer.ControlMap.Guid);
                    if (bytes != null)
                    {
                        layer.PendingControlMapBytes = bytes;
                        Debug.Log($"[TerrainLoader] TextureLayer[{i}] ControlMap loaded: {bytes.Length} bytes");
                    }
                }
            }

            // Decoration ControlMaps (density maps)
            if (terrain.Decorations != null)
            {
                for (int i = 0; i < terrain.Decorations.Count; i++)
                {
                    var deco = terrain.Decorations[i];
                    if (deco.ControlMap == null || string.IsNullOrEmpty(deco.ControlMap.Guid)) continue;

                    var bytes = LoadDdsBytes(deco.ControlMap.Guid);
                    if (bytes != null)
                    {
                        deco.PendingControlMapBytes = bytes;
                        Debug.Log($"[TerrainLoader] Decoration[{i}] ControlMap loaded: {bytes.Length} bytes");
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
            if (cachePath == null || !File.Exists(cachePath))
            {
                // Fallback: match SaveDdsSubasset's fallback path
                var cacheDir = AssetDatabase.Project.CacheDirectory;
                cachePath = Path.Combine(cacheDir, $"{guid}.dds");
            }
            if (!File.Exists(cachePath)) return null;

            try
            {
                using var stream = File.OpenRead(cachePath);
                var data = _ddsPacker.Read(stream);
                if (data?.Bytes == null || data.Bytes.Length == 0) return null;

                var bytes = data.Bytes;

                // Check if bytes start with DDS magic ("DDS " = 0x20534444).
                // If so, strip the DDS header to get raw pixel data.
                if (bytes.Length > 128 && BitConverter.ToInt32(bytes, 0) == 0x20534444)
                {
                    int headerSize = 128;
                    // Check for DX10 extended header
                    if (bytes.Length > 148 && BitConverter.ToInt32(bytes, 84) == 0x30315844)
                        headerSize = 148;

                    int pixelDataLen = bytes.Length - headerSize;
                    Debug.Log($"[TerrainLoader] LoadDdsBytes '{guid}': stripped {headerSize}-byte DDS header, {pixelDataLen} pixel bytes");
                    var pixels = new byte[pixelDataLen];
                    Array.Copy(bytes, headerSize, pixels, 0, pixelDataLen);
                    return pixels;
                }

                Debug.Log($"[TerrainLoader] LoadDdsBytes '{guid}': {bytes.Length} raw bytes (no DDS header)");
                return bytes;
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
                // 1. Save baked heightmap (GPU readback → cache)
                SaveBakedHeightmap(terrain);

                // 2. Save all ControlMap data (GPU readback → cache)
                //    This also assigns GUIDs to new GPU-only ControlMaps.
                //    Must happen BEFORE YAML save so the GUIDs are serialized.
                SaveControlMaps(terrain);

                // 3. Save YAML definition (now includes ControlMap + BakedHeightmapRef GUIDs)
                NativeImporter.Save(savePath, terrain);
                Debug.Log($"[TerrainLoader] YAML saved: {savePath}");
            }
            catch (Exception ex)
            {
                Debug.LogWarning("TerrainLoader", $"Failed to save terrain: {ex.Message}");
            }
        }

        /// <summary>
        /// Reads back baked heightmap from GPU, saves as DDS subasset,
        /// and cooks + saves the PhysX HeightField as a CollisionMeshData subasset.
        /// </summary>
        private void SaveBakedHeightmap(Terrain terrain)
        {
            var baker = TerrainBaker.Instance;
            if (baker == null) return;

            var bytes = baker.ReadbackBakedHeightmapBytes();
            if (bytes == null || bytes.Length == 0) return;

            int resolution = baker.BakedResolution;

            // Ensure the BakedHeightmapRef has a GUID
            if (terrain.BakedHeightmapRef == null)
                terrain.BakedHeightmapRef = new Texture();
            if (string.IsNullOrEmpty(terrain.BakedHeightmapRef.Guid))
                terrain.BakedHeightmapRef.Guid = System.Guid.NewGuid().ToString("N");

            // Save baked heightmap DDS
            SaveDdsSubasset(terrain.BakedHeightmapRef.Guid, bytes);
            Debug.Log($"[TerrainLoader] Baked heightmap saved: {bytes.Length} bytes, res={resolution}");

            // Cook PhysX HeightField from the R16_Float bytes
            try
            {
                var heights = new float[resolution, resolution];
                for (int y = 0; y < resolution; y++)
                    for (int x = 0; x < resolution; x++)
                    {
                        int idx = (y * resolution + x) * 2;
                        Half h = BitConverter.ToHalf(bytes, idx);
                        heights[x, y] = (float)h;
                    }

                var samples = heights.ToSamples();
                var hfDesc = new HeightFieldDesc
                {
                    NumberOfRows = resolution,
                    NumberOfColumns = resolution,
                    Samples = samples,
                };
                var cooking = PhysicsWorld.Physics.CreateCooking();
                var cookedStream = new MemoryStream();
                cooking.CookHeightField(hfDesc, cookedStream);
                var cookedBytes = cookedStream.ToArray();

                // Save as CollisionMeshData subasset
                SaveCollisionSubasset(terrain, cookedBytes);

                Debug.Log($"[TerrainLoader] PhysX HeightField cooked + saved: {resolution}x{resolution}, {cookedBytes.Length} bytes");
            }
            catch (Exception ex)
            {
                Debug.LogWarning("TerrainLoader", $"Failed to cook/save PhysX HeightField: {ex.Message}");
            }
        }

        /// <summary>
        /// Saves cooked PhysX bytes to the collision subasset in cache.
        /// Uses AssetDatabase.AddOrUpdateSubAsset to find or create the entry.
        /// </summary>
        private void SaveCollisionSubasset(Terrain terrain, byte[] cookedBytes)
        {
            if (string.IsNullOrEmpty(terrain.Guid)) return;

            // Find existing or create new CollisionMeshData subasset
            var meta = AssetDatabase.GetMeta(terrain.Guid);
            if (meta == null) return;

            var collisionSub = meta.SubAssets.FirstOrDefault(
                s => s.Type == nameof(CollisionMeshData));

            string subGuid;
            if (collisionSub != null)
            {
                subGuid = collisionSub.Guid;
            }
            else
            {
                subGuid = AssetDatabase.AddOrUpdateSubAsset(
                    terrain.Guid, nameof(CollisionMeshData), terrain.Name, hidden: true);
                if (subGuid == null) return;
            }

            // Write cooked bytes to cache
            var cachePath = AssetDatabase.ResolveCachePathByGuid(subGuid);
            if (cachePath == null)
            {
                var cacheDir = AssetDatabase.Project.CacheDirectory;
                var bucket = subGuid[..2];
                cachePath = Path.Combine(cacheDir, bucket, $"{subGuid}.physx");
                Directory.CreateDirectory(Path.GetDirectoryName(cachePath));
            }

            var packer = new CollisionMeshPacker();
            using var stream = File.Create(cachePath);
            packer.Write(stream, new CollisionMeshData { CookedBytes = cookedBytes });
        }

        /// <summary>
        /// Finds the active TerrainBaker, reads back all ControlMaps from GPU, packs to cache.
        /// Handles: PaintHeightLayer ControlMaps, TextureLayer ControlMaps, Decoration ControlMaps.
        /// </summary>
        private void SaveControlMaps(Terrain terrain)
        {
            if (string.IsNullOrEmpty(terrain.Guid)) return;
            var baker = TerrainBaker.Instance;

            // Save PaintHeightLayer ControlMaps
            foreach (var layer in terrain.HeightLayers)
            {
                if (layer is PaintHeightLayer paint && paint.ControlMap != null)
                {
                    // Ensure the ControlMap has a GUID (may be a new GPU-only texture)
                    if (string.IsNullOrEmpty(paint.ControlMap.Guid))
                        paint.ControlMap.Guid = System.Guid.NewGuid().ToString("N");

                    var pixels = baker.ReadbackControlMap(TerrainBaker.ControlMapTarget.Height, 0);
                    if (pixels != null)
                    {
                        SaveDdsSubasset(paint.ControlMap.Guid, pixels);
                        Debug.Log($"[TerrainLoader] PaintHeightLayer ControlMap saved ({pixels.Length} bytes)");
                    }
                }
            }

            // Save TextureLayer ControlMaps (splatmaps)
            if (terrain.Layers != null)
            {
                for (int i = 0; i < terrain.Layers.Count; i++)
                {
                    var layer = terrain.Layers[i];
                    if (layer.ControlMap == null) continue;

                    var pixels = baker.ReadbackControlMap(TerrainBaker.ControlMapTarget.Splatmap, i);
                    if (pixels != null)
                    {
                        // Ensure the ControlMap has a GUID (may be a new GPU-only texture)
                        if (string.IsNullOrEmpty(layer.ControlMap.Guid))
                        layer.ControlMap.Guid = System.Guid.NewGuid().ToString("N");

                        SaveDdsSubasset(layer.ControlMap.Guid, pixels);
                        Debug.Log($"[TerrainLoader] TextureLayer[{i}] ControlMap saved ({pixels.Length} bytes)");
                    }
                }
            }

            // Save Decoration ControlMaps (density maps)
            if (terrain.Decorations != null)
            {
                for (int i = 0; i < terrain.Decorations.Count; i++)
                {
                    var deco = terrain.Decorations[i];
                    if (deco.ControlMap == null) continue;

                    var pixels = baker.ReadbackControlMap(TerrainBaker.ControlMapTarget.Density, i);
                    if (pixels != null)
                    {
                        if (string.IsNullOrEmpty(deco.ControlMap.Guid))
                            deco.ControlMap.Guid = System.Guid.NewGuid().ToString("N");

                        SaveDdsSubasset(deco.ControlMap.Guid, pixels);
                        Debug.Log($"[TerrainLoader] Decoration[{i}] ControlMap saved ({pixels.Length} bytes)");
                    }
                }
            }
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
