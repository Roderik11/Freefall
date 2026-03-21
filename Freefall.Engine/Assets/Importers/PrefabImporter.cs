using System.IO;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports .prefab source files into the cache pipeline.
    /// Reads the YAML bytes and produces a PrefabData artifact
    /// that gets packed into Library/ via PrefabPacker.
    /// </summary>
    [AssetImporter(".prefab")]
    public class PrefabImporter : IImporter
    {
        public Type AssetType => typeof(Prefab);
        
        public ImportResult Import(string filepath)
        {
            var result = new ImportResult();
            var name = Path.GetFileNameWithoutExtension(filepath);
            var yaml = File.ReadAllBytes(filepath);

            result.Artifacts.Add(new ImportArtifact
            {
                Name = name,
                Type = nameof(PrefabData),
                Data = new PrefabData { Yaml = yaml }
            });

            return result;
        }
    }
}
