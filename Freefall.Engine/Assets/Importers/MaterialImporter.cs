using System.IO;
using System.Text;
using Freefall.Assets.Packers;
using Freefall.Graphics;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports .mat files (YAML-serialized Material definitions).
    /// Simple pass-through: reads the YAML and stores it as AssetDefinitionData.
    /// </summary>
    [AssetImporter(".mat")]
    public class MaterialImporter : IImporter
    {
        public Type AssetType => typeof(Graphics.Material);
        /// <summary>
        /// .mat files ARE the asset definition — inspect the loaded Material, not the importer.
        /// </summary>
        public object GetInspectionTarget(MetaFile meta)
        {
            return Engine.Assets.LoadByGuid<Material>(meta.Guid);
        }
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
