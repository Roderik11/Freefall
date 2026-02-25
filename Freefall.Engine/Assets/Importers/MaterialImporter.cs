using System.IO;
using System.Text;
using Freefall.Assets.Packers;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports .mat files (YAML-serialized Material definitions).
    /// Simple pass-through: reads the YAML and stores it as AssetDefinitionData.
    /// </summary>
    [AssetImporter(".mat")]
    public class MaterialImporter : IImporter
    {
        public ImportResult Import(string filepath)
        {
            var result = new ImportResult();
            var name = Path.GetFileNameWithoutExtension(filepath);
            var yaml = File.ReadAllText(filepath, Encoding.UTF8);

            result.Artifacts.Add(new ImportArtifact
            {
                Name = name,
                Type = "Material",
                Data = new AssetDefinitionData
                {
                    TypeName = "Material",
                    YamlBytes = Encoding.UTF8.GetBytes(yaml)
                }
            });

            return result;
        }
    }
}
