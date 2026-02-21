using System.IO;
using Freefall.Assets.Packers;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports audio files (.wav, .ogg) for the editor pipeline.
    /// Reads raw audio bytes and produces an AudioClipData artifact for caching.
    /// </summary>
    [AssetImporter(".wav", ".ogg")]
    public class AudioClipImporter : IImporter
    {
        public ImportResult Import(string filepath)
        {
            var result = new ImportResult();
            var name = Path.GetFileNameWithoutExtension(filepath);
            var bytes = File.ReadAllBytes(filepath);

            result.Artifacts.Add(new ImportArtifact
            {
                Name = name,
                Type = nameof(AudioClipData),
                Data = new AudioClipData(bytes, Path.GetExtension(filepath).ToLowerInvariant())
            });

            return result;
        }
    }
}
