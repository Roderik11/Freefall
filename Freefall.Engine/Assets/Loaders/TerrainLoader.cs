using System;
using System.IO;
using System.Text;
using Freefall.Assets.Packers;
using Freefall.Graphics;
using Freefall.Serialization;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads Terrain assets from cache (.asset files).
    /// Unpacks AssetDefinitionData (YAML), deserializes the Terrain definition,
    /// then resolves all GUID references (Heightmap, Layer textures, ControlMaps)
    /// via AssetManager. Finally builds the CPU-side HeightField.
    /// </summary>
    [AssetLoader(typeof(Terrain))]
    public class TerrainLoader : IAssetLoader
    {
        private readonly AssetDefinitionPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name);
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for terrain '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager)
        {
            try
            {
                Debug.Log($"[TerrainLoader] Loading '{name}' from {cachePath}");

                AssetDefinitionData defData;
                using (var stream = File.OpenRead(cachePath))
                    defData = _packer.Read(stream);

                // Deserialize YAML → Terrain
                var yaml = Encoding.UTF8.GetString(defData.YamlBytes);
                Terrain terrain;
                try
                {
                    terrain = NativeImporter.LoadFromString(yaml) as Terrain;
                }
                catch (Exception ex)
                {
                    throw new InvalidDataException(
                        $"Failed to deserialize Terrain from cache: {name} — {ex.GetType().Name}: {ex.Message}", ex);
                }

                if (terrain == null)
                    throw new InvalidDataException($"Failed to deserialize Terrain from cache: {name}");

                terrain.Name = name;

                // ── Resolve GUID references ──

                // Resolve Heightmap
                if (terrain.Heightmap != null && !string.IsNullOrEmpty(terrain.Heightmap.Guid))
                {
                    var heightmapGuid = terrain.Heightmap.Guid;
                    terrain.Heightmap = manager.LoadByGuid<Texture>(heightmapGuid);
                    Debug.Log($"[TerrainLoader] Heightmap resolved: {(terrain.Heightmap != null ? "OK" : "NULL")}");

                    // Build CPU-side heightfield from source DDS
                    if (terrain.Heightmap != null)
                    {
                        var sourcePath = AssetDatabase.GuidToPath(heightmapGuid);
                        if (!string.IsNullOrEmpty(sourcePath))
                        {
                            var fullPath = Path.Combine(AssetDatabase.Project.AssetsDirectory, sourcePath);
                            if (File.Exists(fullPath))
                                terrain.BuildHeightField(fullPath);
                        }
                    }
                }

                // Resolve Layer textures
                if (terrain.Layers != null)
                {
                    foreach (var layer in terrain.Layers)
                    {
                        if (layer.Diffuse != null && !string.IsNullOrEmpty(layer.Diffuse.Guid))
                            layer.Diffuse = manager.LoadByGuid<Texture>(layer.Diffuse.Guid);
                        if (layer.Normals != null && !string.IsNullOrEmpty(layer.Normals.Guid))
                            layer.Normals = manager.LoadByGuid<Texture>(layer.Normals.Guid);
                    }
                }

                // Resolve ControlMap textures
                if (terrain.ControlMaps != null)
                {
                    for (int i = 0; i < terrain.ControlMaps.Count; i++)
                    {
                        var cm = terrain.ControlMaps[i];
                        if (cm != null && !string.IsNullOrEmpty(cm.Guid))
                            terrain.ControlMaps[i] = manager.LoadByGuid<Texture>(cm.Guid);
                    }
                }

                Debug.Log($"[TerrainLoader] '{name}' loaded: {terrain.Layers?.Count ?? 0} layers, " +
                          $"{terrain.ControlMaps?.Count ?? 0} controlmaps, " +
                          $"HeightField={terrain.HeightField != null}");

                return terrain;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("TerrainLoader", $"FAILED to load '{name}': {ex}");
                return null;
            }
        }
    }
}
