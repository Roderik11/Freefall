using Freefall.Assets;
using Freefall.Assets.Packers;
using System.IO;
using System.Runtime.InteropServices;
using Vortice.Multimedia;
using Vortice.XAudio2;

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

        public object GetInspectionTarget(MetaFile meta) => this;

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, "AudioClipData");
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for audio clip '{name}'");

            AudioClipData clipData;
            using (var stream = File.OpenRead(cachePath))
                clipData = _packer.Read(stream);

            using (var ms = new MemoryStream(clipData.Bytes))
            {
                var clip = Load(ms);
                clip.Name = name;
                return clip;
            }
        }

        private AudioClip Load(Stream inputStream)
        {
            var audioClip = new AudioClip();
            var stream = new SoundStream(inputStream);

            var format = stream.Format;
            audioClip.DecodedPacketsInfo = stream.DecodedPacketsInfo;

            // XAudio2 doesn't support 24-bit PCM — convert to 16-bit
            if (format.BitsPerSample == 24)
            {
                var ms = new MemoryStream();
                stream.CopyTo(ms);
                var srcBytes = ms.ToArray();

                int sampleCount = srcBytes.Length / 3;
                var dstBytes = new byte[sampleCount * 2];
                for (int i = 0; i < sampleCount; i++)
                {
                    dstBytes[i * 2] = srcBytes[i * 3 + 1];
                    dstBytes[i * 2 + 1] = srcBytes[i * 3 + 2];
                }

                format = new WaveFormat(format.SampleRate, 16, format.Channels);

                var pinnedData = GCHandle.Alloc(dstBytes, GCHandleType.Pinned);
                audioClip.WaveFormat = format;
                audioClip.Buffer = new AudioBuffer(pinnedData.AddrOfPinnedObject(), (uint)dstBytes.Length);
                audioClip._pinnedHandle = pinnedData;
            }
            else
            {
                // Stream hasn't been consumed — use it directly
                audioClip.WaveFormat = format;
                audioClip.Buffer = new AudioBuffer(stream);
            }

            return audioClip;
        }
    }
}
