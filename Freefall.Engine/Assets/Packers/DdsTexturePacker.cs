using System.IO;

namespace Freefall.Assets.Packers
{
    /// <summary>
    /// Wrapper for raw DDS texture bytes so the packer discovery system
    /// can register this packer by type name ("DdsTextureData").
    /// </summary>
    public class DdsTextureData
    {
        public byte[] Bytes { get; set; }

        public DdsTextureData() { }
        public DdsTextureData(byte[] bytes) { Bytes = bytes; }
    }

    /// <summary>
    /// Packs DDS texture data to cache. Since DDS is already the optimal
    /// GPU-ready format, this packer simply writes the raw DDS bytes
    /// with a version header for future format evolution.
    /// </summary>
    [AssetPacker(".dds")]
    public class DdsTexturePacker : AssetPacker<DdsTextureData>
    {
        public override int Version => 1;

        public override void Pack(BinaryWriter writer, DdsTextureData asset)
        {
            writer.Write(asset.Bytes.Length);
            writer.Write(asset.Bytes);
        }

        public override DdsTextureData Unpack(BinaryReader reader, int version)
        {
            int length = reader.ReadInt32();
            return new DdsTextureData(reader.ReadBytes(length));
        }
    }
}
