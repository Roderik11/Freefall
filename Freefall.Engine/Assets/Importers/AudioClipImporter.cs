using System;
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
        // What the source file is to the editor: without it the browser card has no asset type,
        // drags as a plain Asset, and no AudioClip field accepts the drop.
        [System.Text.Json.Serialization.JsonIgnore]
        public Type AssetType => typeof(AudioClip);

        // A clip has nothing to edit (buffer, wave format, packet table): selecting the file shows the importer.
        public object GetInspectionTarget(MetaFile meta) => this;

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
