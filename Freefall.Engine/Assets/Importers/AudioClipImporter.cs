using System;
using System.IO;
using Freefall.Assets.Packers;
using NVorbis;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports audio files (.wav, .ogg) for the editor pipeline.
    /// A .wav is cached as it is; an .ogg is decoded here, once, into a mono 16-bit wav, because the
    /// loader reads clips through SoundStream, which knows RIFF only. An .ogg with Stream on is cached
    /// as it is, to be decoded while it plays.
    /// </summary>
    [AssetImporter(".wav", ".ogg")]
    public class AudioClipImporter : IImporter
    {
        // What the source file is to the editor: without it the browser card has no asset type,
        // drags as a plain Asset, and no AudioClip field accepts the drop.
        [System.Text.Json.Serialization.JsonIgnore]
        public Type AssetType => typeof(AudioClip);

        /// <summary>
        /// For music and other long sounds (.ogg only): kept compressed, with all its channels, and
        /// decoded a little at a time while an AudioStream plays it, flat, not placed in the world. A
        /// three-minute stereo track is 5 MB so, and 33 MB decoded. An AudioSource cannot play it.
        /// </summary>
        public bool Stream;

        // A clip has nothing to edit (buffer, wave format, packet table): selecting the file shows the importer.
        public object GetInspectionTarget(MetaFile meta) => this;

        public ImportResult Import(string filepath)
        {
            var result = new ImportResult();
            var name = Path.GetFileNameWithoutExtension(filepath);
            var extension = Path.GetExtension(filepath).ToLowerInvariant();

            if (Stream && extension != ".ogg")
                Debug.LogWarning("[AudioClipImporter]", $"{Path.GetFileName(filepath)}: only an .ogg can be streamed; held decoded");
            bool streamed = Stream && extension == ".ogg";

            var bytes = extension == ".ogg" && !streamed ? DecodeOgg(filepath) : File.ReadAllBytes(filepath);
            if (extension == ".ogg" && !streamed) extension = ".wav";

            result.Artifacts.Add(new ImportArtifact
            {
                Name = name,
                Type = nameof(AudioClipData),
                Data = new AudioClipData(bytes, extension, streamed)
            });

            return result;
        }

        /// <summary>
        /// Ogg Vorbis to a wav file in memory: 16-bit PCM at the file's own sample rate, one channel.
        /// Mono because every AudioSource places its sound in the world, and X3DAudio places one
        /// channel; there is nothing yet that plays a sound flat, where stereo would be kept.
        /// </summary>
        private static byte[] DecodeOgg(string filepath)
        {
            using var reader = new VorbisReader(filepath);
            int channels = reader.Channels;
            int sampleRate = reader.SampleRate;

            // TotalSamples is per channel, read from the last page; only a hint for the size
            var pcm = new MemoryStream((int)Math.Min(Math.Max(reader.TotalSamples, 0) * 2 + 44, int.MaxValue));
            pcm.Position = 44;
            using (var writer = new BinaryWriter(pcm, System.Text.Encoding.ASCII, leaveOpen: true))
            {
                var buffer = new float[4096 * channels];
                int read;
                while ((read = reader.ReadSamples(buffer, 0, buffer.Length)) > 0)
                {
                    for (int i = 0; i + channels <= read; i += channels)
                    {
                        float sum = 0;
                        for (int c = 0; c < channels; c++)
                            sum += buffer[i + c];
                        writer.Write((short)Math.Clamp(MathF.Round(sum / channels * 32767f), -32768f, 32767f));
                    }
                }
            }

            int dataBytes = (int)pcm.Length - 44;
            pcm.Position = 0;
            using (var writer = new BinaryWriter(pcm, System.Text.Encoding.ASCII, leaveOpen: true))
            {
                writer.Write("RIFF"u8);
                writer.Write(36 + dataBytes);
                writer.Write("WAVE"u8);
                writer.Write("fmt "u8);
                writer.Write(16);                 // fmt chunk size
                writer.Write((ushort)1);          // PCM
                writer.Write((ushort)1);          // channels
                writer.Write(sampleRate);
                writer.Write(sampleRate * 2);     // bytes per second
                writer.Write((ushort)2);          // bytes per frame
                writer.Write((ushort)16);         // bits per sample
                writer.Write("data"u8);
                writer.Write(dataBytes);
            }

            return pcm.ToArray();
        }
    }
}
