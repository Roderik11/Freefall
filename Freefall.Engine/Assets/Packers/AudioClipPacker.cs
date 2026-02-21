using System.IO;

namespace Freefall.Assets.Packers
{
    /// <summary>
    /// Wrapper for raw audio data so the packer discovery
    /// registers by type name ("AudioClipData").
    /// </summary>
    public class AudioClipData
    {
        public byte[] Bytes { get; set; }

        /// <summary>
        /// Original file extension (e.g. ".wav", ".ogg") preserved
        /// so the runtime loader knows the source format.
        /// </summary>
        public string Extension { get; set; }

        public AudioClipData() { }
        public AudioClipData(byte[] bytes, string extension)
        {
            Bytes = bytes;
            Extension = extension;
        }
    }

    /// <summary>
    /// Packs audio data to cache. Stores the original extension
    /// and raw bytes for runtime loading via XAudio2 SoundStream.
    /// </summary>
    [AssetPacker(".audio")]
    public class AudioClipPacker : AssetPacker<AudioClipData>
    {
        public override int Version => 1;

        public override void Pack(BinaryWriter writer, AudioClipData asset)
        {
            writer.Write(asset.Extension ?? ".wav");
            writer.Write(asset.Bytes.Length);
            writer.Write(asset.Bytes);
        }

        public override AudioClipData Unpack(BinaryReader reader, int version)
        {
            var ext = reader.ReadString();
            int length = reader.ReadInt32();
            return new AudioClipData(reader.ReadBytes(length), ext);
        }
    }
}
