using System;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using Freefall.Assets.Packers;
using Freefall.Base;
using Freefall.Graphics;
using PhysX;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports .terrain files (YAML-serialized Terrain definitions).
    /// Produces two artifacts:
    /// 1. The terrain definition (AssetDefinitionData) — same as AssetFileImporter
    /// 2. A hidden CollisionMeshData subasset containing a pre-cooked PhysX HeightField
    /// </summary>
    [AssetImporter(".terrain")]
    public class TerrainImporter : IImporter
    {
        public Type AssetType => typeof(Terrain);
        public ImportResult Import(string filepath)
        {
            var result = new ImportResult();
            var name = Path.GetFileNameWithoutExtension(filepath);
            var yaml = File.ReadAllText(filepath, Encoding.UTF8);

            // ── 1. Main artifact: terrain definition (YAML) ──
            result.Artifacts.Add(new ImportArtifact
            {
                Name = name,
                Type = "Terrain",
                Data = new AssetDefinitionData
                {
                    TypeName = "Terrain",
                    YamlBytes = Encoding.UTF8.GetBytes(yaml)
                }
            });

            // ── 2. Cook PhysX HeightField ──
            try
            {
                var cookedBytes = CookHeightField(filepath, yaml);
                if (cookedBytes != null)
                {
                    result.Artifacts.Add(new ImportArtifact
                    {
                        Name = name,
                        Type = nameof(CollisionMeshData),
                        Data = new CollisionMeshData { CookedBytes = cookedBytes },
                        Hidden = true
                    });
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("TerrainImporter", $"Failed to cook HeightField for '{name}': {ex.Message}");
            }

            return result;
        }

        /// <summary>
        /// Reads the heightmap from the source texture, cooks a PhysX HeightField,
        /// and returns the cooked bytes. Returns null if heightmap not found.
        /// </summary>
        private static byte[] CookHeightField(string terrainFilePath, string yaml)
        {
            // Find the heightmap source file path.
            // The YAML contains a Heightmap GUID reference. Resolve it to the source file.
            var heightmapGuid = ExtractHeightmapGuid(yaml);
            if (string.IsNullOrEmpty(heightmapGuid))
            {
                Debug.LogWarning("TerrainImporter", "No Heightmap GUID found in terrain YAML");
                return null;
            }

            var sourcePath = AssetDatabase.GuidToPath(heightmapGuid);
            if (string.IsNullOrEmpty(sourcePath))
            {
                Debug.LogWarning("TerrainImporter", $"Cannot resolve Heightmap GUID '{heightmapGuid}' to source path");
                return null;
            }

            var fullPath = Path.Combine(AssetDatabase.Project.AssetsDirectory, sourcePath);
            if (!File.Exists(fullPath))
            {
                Debug.LogWarning("TerrainImporter", $"Heightmap source file not found: {fullPath}");
                return null;
            }

            // Read height field from the source texture (DDS)
            var heightField = Texture.ReadHeightField(fullPath);
            if (heightField == null)
            {
                Debug.LogWarning("TerrainImporter", "Failed to read height field from heightmap texture");
                return null;
            }

            int rows = heightField.GetLength(0);
            int cols = heightField.GetLength(1);

            var samples = heightField.ToSamples();

            var heightFieldDesc = new HeightFieldDesc()
            {
                NumberOfRows = rows,
                NumberOfColumns = cols,
                Samples = samples,
            };

            var cooking = PhysicsWorld.Physics.CreateCooking();
            var stream = new MemoryStream();
            cooking.CookHeightField(heightFieldDesc, stream);

            Debug.Log($"[TerrainImporter] Cooked HeightField: {rows}x{cols}, {stream.Length} bytes");

            return stream.ToArray();
        }

        /// <summary>
        /// Extract the heightmap source GUID from the terrain YAML.
        /// Looks for ImportHeightLayer Source first, then falls back to legacy "Heightmap:" field.
        /// </summary>
        private static string ExtractHeightmapGuid(string yaml)
        {
            // Primary: Source inside ImportHeightLayer
            var match = Regex.Match(yaml, @"ImportHeightLayer:.*?Source:\s*([0-9a-fA-F]{32})", RegexOptions.Singleline);
            if (match.Success) return match.Groups[1].Value;

            // Fallback: legacy Heightmap field
            match = Regex.Match(yaml, @"Heightmap:\s*([0-9a-fA-F]{32})", RegexOptions.Multiline);
            return match.Success ? match.Groups[1].Value : null;
        }
    }
}
