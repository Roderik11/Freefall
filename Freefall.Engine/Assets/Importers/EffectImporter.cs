using System;
using System.IO;
using System.Text;
using Freefall.Assets.Packers;
using Freefall.Graphics;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports a project's .fx files as <see cref="Effect"/> assets: a GUID, an Effect card that drops on a material's
    /// Effect field. The cache holds the text; it compiles when a material first uses it (EffectLoader), its includes
    /// looked up next to the file, then among the engine's shaders. Saving the file recompiles it (ShaderHotReload
    /// watches the project's Assets). Files it includes (.fxh, .hlsli) are no assets.
    /// </summary>
    [AssetImporter(".fx", ImportPriority = 0)]
    public class EffectImporter : IImporter
    {
        // The browser card's type and what an Effect field accepts on a drop
        [System.Text.Json.Serialization.JsonIgnore]
        public Type AssetType => typeof(Effect);

        // An effect has nothing to edit in the inspector: its settings are its materials'
        public object GetInspectionTarget(MetaFile meta) => this;

        public ImportResult Import(string filepath)
        {
            var result = new ImportResult();
            result.Artifacts.Add(new ImportArtifact
            {
                Name = Path.GetFileNameWithoutExtension(filepath),
                Type = nameof(Effect),
                Data = new EffectSourceData(File.ReadAllText(filepath, Encoding.UTF8))
            });
            return result;
        }
    }
}
