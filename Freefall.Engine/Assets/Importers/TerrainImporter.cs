using System;
using System.IO;
using System.Text;
using Freefall.Assets.Packers;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports .terrain files (YAML-serialized Terrain definitions) into one artifact, the terrain definition
    /// (AssetDefinitionData), as AssetFileImporter does. The heights and holes live in the terrain's pages file, which a
    /// save writes (TerrainPagesFile). Until 4c (2026-11-07) every import also cooked a PhysX height field of the whole
    /// terrain from the saved flat heightmap into a hidden CollisionMeshData subasset, which nothing read at run time
    /// (1.07 GB at 16 km and 1 m per vertex, on every import).
    /// </summary>
    [AssetImporter(".terrain", ImportPriority = 4)]
    public class TerrainImporter : IImporter
    {
        [System.Text.Json.Serialization.JsonIgnore]
        public Type AssetType => typeof(Terrain);

        public ImportResult Import(string filepath)
        {
            var result = new ImportResult();
            var name = Path.GetFileNameWithoutExtension(filepath);
            var yaml = File.ReadAllText(filepath, Encoding.UTF8);

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

            return result;
        }
    }
}
