using System;
using System.IO;
using Freefall.Assets.Packers;

namespace Freefall.Assets.Importers
{
    /// <summary>Imports font files (.ttf, .otf) as <see cref="Typeface"/> assets, bytes unchanged.</summary>
    [AssetImporter(".ttf", ".otf")]
    public class TypefaceImporter : IImporter
    {
        // Without it the browser card has no asset type and no Typeface field accepts the drop
        [System.Text.Json.Serialization.JsonIgnore]
        public Type AssetType => typeof(Typeface);

        // A typeface has nothing to edit: selecting the file shows the importer
        public object GetInspectionTarget(MetaFile meta) => this;

        public ImportResult Import(string filepath)
        {
            var result = new ImportResult();
            result.Artifacts.Add(new ImportArtifact
            {
                Name = Path.GetFileNameWithoutExtension(filepath),
                Type = nameof(TypefaceData),
                Data = new TypefaceData(File.ReadAllBytes(filepath))
            });
            return result;
        }
    }
}
