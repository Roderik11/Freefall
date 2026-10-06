using System;
using System.IO;
using System.Linq;
using System.Text;
using Freefall.Assets.Packers;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Freefall.Serialization;
using PhysX;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads and saves Terrain assets.
    /// Load: unpacks AssetDefinitionData (YAML) from cache, deserializes Terrain,
    ///       resolves GUID references, loads the baked heightmap subasset.
    /// Save: reads the baked heightmap back from the GPU into the cache, then writes the YAML.
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

                // Migrate old power-of-2 resolutions to power-of-2+1
                terrain.MigrateResolution();

                // Pre-cooked PhysX cache skipped — always cook from CPU HeightField at play time
                // LoadCookedHeightField(terrain, sourceGuid);

                // Load persisted baked heightmap (R16_UNorm DDS)
                LoadBakedHeightmap(terrain);

                Debug.Log($"[TerrainLoader] '{name}' loaded: HeightField={terrain.HeightField != null}, " +
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

                Debug.Log($"[TerrainLoader] Cooked bytes loaded: {cooked.CookedBytes.Length} bytes from {physxPath}");

                var hf = PhysicsWorld.Physics.CreateHeightField(new MemoryStream(cooked.CookedBytes));
                terrain.SetCookedHeightField(hf);

                Debug.Log($"[TerrainLoader] Pre-cooked HeightField loaded: {guid}, {cooked.CookedBytes.Length} cooked bytes");
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

            // Build CPU HeightField from R16_UNorm bytes so GetHeight() works immediately
            int pixelDataLen = bytes.Length;
            int resolution = (int)Math.Sqrt(pixelDataLen / 2);
            if (resolution * resolution * 2 == pixelDataLen)
            {
                var heights = new float[resolution, resolution];
                for (int y = 0; y < resolution; y++)
                    for (int x = 0; x < resolution; x++)
                    {
                        int idx = (y * resolution + x) * 2;
                        ushort raw = BitConverter.ToUInt16(bytes, idx);
                        heights[x, y] = raw / 65535.0f;
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
        /// Save the baked heightmap to cache, then the terrain YAML.
        /// </summary>
        public void Save(Asset asset, string savePath)
        {
            if (asset is not Terrain terrain) return;

            try
            {
                // 1. Save baked heightmap (GPU readback → cache)
                SaveBakedHeightmap(terrain);

                // 2. Save YAML definition (includes the BakedHeightmapRef GUID)
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
            var baker = ComponentCache<TerrainRenderer>.All
                .FirstOrDefault(r => r.Terrain == terrain)?.Baker;
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

            // Cook PhysX HeightField from the CPU HeightField (always in sync after readback)
            try
            {
                var heightMap = terrain.HeightField;
                if (heightMap == null)
                {
                    Debug.LogWarning("TerrainLoader", "No CPU HeightField available for PhysX cooking");
                    return;
                }

                int rows = heightMap.GetLength(0);
                int cols = heightMap.GetLength(1);

                // Diagnostic: check height range
                float hMin = float.MaxValue, hMax = float.MinValue;
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                    {
                        float h = heightMap[i, j];
                        if (h < hMin) hMin = h;
                        if (h > hMax) hMax = h;
                    }
                Debug.Log($"[TerrainLoader] HeightField at save: {rows}x{cols}, range=[{hMin:F6}..{hMax:F6}]");

                var samples = heightMap.ToSamples();
                var hfDesc = new HeightFieldDesc
                {
                    NumberOfRows = rows,
                    NumberOfColumns = cols,
                    Samples = samples,
                };
                var cooking = PhysicsWorld.Physics.CreateCooking();
                var cookedStream = new MemoryStream();
                cooking.CookHeightField(hfDesc, cookedStream);
                var cookedBytes = cookedStream.ToArray();

                // Update in-memory CookedHeightField so collider stays in sync
                cookedStream.Position = 0;
                var hf = PhysicsWorld.Physics.CreateHeightField(cookedStream);
                terrain.SetCookedHeightField(hf);

                // Save as CollisionMeshData subasset
                SaveCollisionSubasset(terrain, cookedBytes);

                Debug.Log($"[TerrainLoader] PhysX HeightField cooked + saved: {rows}x{cols}, {cookedBytes.Length} bytes");
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
