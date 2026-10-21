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
        /// What Bytes are: ".wav" (an .ogg is decoded to a wav at import), or ".ogg" for a streamed clip,
        /// which keeps the file as it is. ".ogg" without Streamed only in caches written before the
        /// importer decoded .ogg files, which have to be reimported.
        /// </summary>
        public string Extension { get; set; }

        /// <summary>Imported with Stream on: Bytes are the compressed file, decoded while it plays.</summary>
        public bool Streamed { get; set; }

        public AudioClipData() { }
        public AudioClipData(byte[] bytes, string extension, bool streamed = false)
        {
            Bytes = bytes;
            Extension = extension;
            Streamed = streamed;
        }
    }

    /// <summary>
    /// Packs audio data to cache. Stores the original extension
    /// and raw bytes for runtime loading via XAudio2 SoundStream.
    /// Version 2 adds whether the clip is streamed.
    /// </summary>
    [AssetPacker(".audio")]
    public class AudioClipPacker : AssetPacker<AudioClipData>
    {
        public override int Version => 2;

        public override void Pack(BinaryWriter writer, AudioClipData asset)
        {
            writer.Write(asset.Extension ?? ".wav");
            writer.Write(asset.Bytes.Length);
            writer.Write(asset.Bytes);
            writer.Write(asset.Streamed);
        }

        public override AudioClipData Unpack(BinaryReader reader, int version)
        {
            var ext = reader.ReadString();
            int length = reader.ReadInt32();
            var bytes = reader.ReadBytes(length);
            bool streamed = version >= 2 && reader.ReadBoolean();
            return new AudioClipData(bytes, ext, streamed);
        }
    }
}
