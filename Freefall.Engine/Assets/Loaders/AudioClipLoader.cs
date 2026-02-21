using System.IO;
using Freefall.Assets;
using Freefall.Assets.Packers;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads AudioClip assets from cache (.audio files).
    /// Unpacks raw audio bytes + extension via AudioClipPacker,
    /// then delegates to AudioClipReader for XAudio2 buffer creation.
    /// </summary>
    [AssetLoader(typeof(AudioClip))]
    public class AudioClipLoader : IAssetLoader
    {
        private readonly AudioClipPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name);
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for audio clip '{name}'");

            AudioClipData clipData;
            using (var stream = File.OpenRead(cachePath))
                clipData = _packer.Read(stream);

            // Write to temp file with correct extension for SoundStream parsing
            var tempPath = Path.Combine(Path.GetTempPath(),
                $"freefall_audio_{System.Guid.NewGuid():N}{clipData.Extension}");
            try
            {
                File.WriteAllBytes(tempPath, clipData.Bytes);
                var reader = new AudioClipReader();
                var clip = reader.Load(tempPath);
                clip.Name = name;
                return clip;
            }
            finally
            {
                try { File.Delete(tempPath); } catch { }
            }
        }
    }
}
