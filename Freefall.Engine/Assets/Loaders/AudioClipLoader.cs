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
            var cachePath = AssetDatabase.ResolveCachePath(name, "AudioClipData");
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for audio clip '{name}'");

            AudioClipData clipData;
            using (var stream = File.OpenRead(cachePath))
                clipData = _packer.Read(stream);

            var reader = new AudioClipReader();
            using (var ms = new MemoryStream(clipData.Bytes))
            {
                var clip = reader.Load(ms);
                clip.Name = name;
                return clip;
            }
        }
    }
}
