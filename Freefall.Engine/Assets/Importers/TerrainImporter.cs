using System;
using System.IO;
using System.Linq;
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
    ///    cooked from the saved baked heightmap DDS.
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

            // ── 2. Cook PhysX HeightField from saved baked heightmap ──
            try
            {
                var cookedBytes = CookHeightField(yaml);
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
        /// Reads the saved baked heightmap DDS from cache, converts R16_Float to float[,],
        /// and cooks a PhysX HeightField. Returns null if no baked heightmap found.
        /// </summary>
        private static byte[] CookHeightField(string yaml)
        {
            // Extract the BakedHeightmapRef GUID from the terrain YAML
            var bakedGuid = ExtractBakedHeightmapGuid(yaml);
            if (string.IsNullOrEmpty(bakedGuid))
            {
                Debug.Log("[TerrainImporter] No BakedHeightmapRef found — skipping HeightField cook (terrain needs editor save first)");
                return null;
            }

            // Load the baked heightmap DDS bytes from cache
            var cachePath = AssetDatabase.ResolveCachePathByGuid(bakedGuid);
            if (cachePath == null || !File.Exists(cachePath))
            {
                // Fallback path
                var cacheDir = AssetDatabase.Project?.CacheDirectory;
                if (cacheDir != null)
                    cachePath = Path.Combine(cacheDir, $"{bakedGuid}.dds");
            }

            if (cachePath == null || !File.Exists(cachePath))
            {
                Debug.LogWarning("TerrainImporter", $"Baked heightmap cache not found for GUID '{bakedGuid}'");
                return null;
            }

            // Read the DDS file
            byte[] rawBytes;
            var packer = new DdsTexturePacker();
            using (var stream = File.OpenRead(cachePath))
            {
                var dds = packer.Read(stream);
                rawBytes = dds?.Bytes;
            }

            if (rawBytes == null || rawBytes.Length == 0)
            {
                Debug.LogWarning("TerrainImporter", "Baked heightmap DDS is empty");
                return null;
            }

            // Strip DDS header if present
            int offset = 0;
            if (rawBytes.Length > 128 && BitConverter.ToInt32(rawBytes, 0) == 0x20534444)
            {
                offset = 128;
                if (rawBytes.Length > 148 && BitConverter.ToInt32(rawBytes, 84) == 0x30315844)
                    offset = 148;
            }

            int pixelDataLen = rawBytes.Length - offset;

            // Parse the R16_Float data into float[,]
            // R16_Float = 2 bytes per pixel
            int resolution = (int)Math.Sqrt(pixelDataLen / 2);
            if (resolution * resolution * 2 != pixelDataLen)
            {
                Debug.LogWarning("TerrainImporter", $"Baked heightmap pixel data size {pixelDataLen} doesn't match R16 square texture");
                return null;
            }

            var heightField = new float[resolution, resolution];
            for (int y = 0; y < resolution; y++)
            {
                for (int x = 0; x < resolution; x++)
                {
                    int idx = offset + (y * resolution + x) * 2;
                    Half h = BitConverter.ToHalf(rawBytes, idx);
                    heightField[x, y] = (float)h;
                }
            }

            // Cook PhysX HeightField
            var samples = heightField.ToSamples();
            var heightFieldDesc = new HeightFieldDesc()
            {
                NumberOfRows = resolution,
                NumberOfColumns = resolution,
                Samples = samples,
            };

            var cooking = PhysicsWorld.Physics.CreateCooking();
            var stream2 = new MemoryStream();
            cooking.CookHeightField(heightFieldDesc, stream2);

            Debug.Log($"[TerrainImporter] Cooked HeightField from baked heightmap: {resolution}x{resolution}, {stream2.Length} bytes");
            return stream2.ToArray();
        }

        /// <summary>
        /// Extract the BakedHeightmapRef GUID from the terrain YAML.
        /// </summary>
        private static string ExtractBakedHeightmapGuid(string yaml)
        {
            var match = Regex.Match(yaml, @"BakedHeightmapRef:\s*([0-9a-fA-F]{32})", RegexOptions.Multiline);
            return match.Success ? match.Groups[1].Value : null;
        }
    }
}
