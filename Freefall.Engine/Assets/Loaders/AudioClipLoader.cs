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

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, "AudioClipData");
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for audio clip '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        /// <summary>
        /// By GUID: the cache file that was resolved for it. The name-based Load above takes whichever
        /// clip of that name it finds first, and names are not unique: a .wav next to an .ogg of the
        /// same name, a take number repeated in another folder.
        /// </summary>
        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            AudioClipData clipData;
            using (var stream = File.OpenRead(cachePath))
                clipData = _packer.Read(stream);

            // Kept compressed: only the length is read here (from the headers and the last page)
            if (clipData.Streamed)
            {
                using var reader = new NVorbis.VorbisReader(new MemoryStream(clipData.Bytes, writable: false), closeOnDispose: true);
                return new AudioClip
                {
                    Name = name,
                    StreamData = clipData.Bytes,
                    Length = (float)reader.TotalTime.TotalSeconds,
                };
            }

            // The importer decodes an .ogg to a wav since 2026-10-14; a cache from before holds the
            // ogg itself, which SoundStream reports as "Invalid RIFF file format"
            if (clipData.Extension == ".ogg")
                throw new InvalidDataException($"Audio clip '{name}' was imported before .ogg files were decoded at import: reimport it");

            using (var ms = new MemoryStream(clipData.Bytes))
            {
                var clip = Load(ms);
                clip.Name = name;
                var format = clip.WaveFormat;
                if (format != null && format.AverageBytesPerSecond > 0)
                    clip.Length = (float)clip.Buffer.AudioBytes / format.AverageBytesPerSecond;
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
