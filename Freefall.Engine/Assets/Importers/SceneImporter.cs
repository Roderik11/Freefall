using System.IO;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports .scene source files into the cache pipeline.
    /// Reads the YAML bytes and produces a SceneData artifact (packed by ScenePacker).
    /// Nothing in a scene is resolved at import, so scenes import last.
    /// </summary>
    [AssetImporter(".scene", ImportPriority = 20)]
    public class SceneImporter : IImporter
    {
        [System.Text.Json.Serialization.JsonIgnore]
        public Type AssetType => typeof(SceneAsset);

        public ImportResult Import(string filepath)
        {
            var result = new ImportResult();

            result.Artifacts.Add(new ImportArtifact
            {
                Name = Path.GetFileNameWithoutExtension(filepath),
                Type = nameof(SceneData),
                Data = new SceneData { Yaml = File.ReadAllBytes(filepath) }
            });

            return result;
        }
    }
}
