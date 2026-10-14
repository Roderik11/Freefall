using System.IO;

namespace Freefall.Assets.Packers
{
    /// <summary>A font file's bytes, registered by type name ("TypefaceData").</summary>
    public class TypefaceData
    {
        public byte[] Bytes { get; set; }

        public TypefaceData() { }
        public TypefaceData(byte[] bytes) => Bytes = bytes;
    }

    /// <summary>Packs a font file to cache as it is.</summary>
    [AssetPacker(".font")]
    public class TypefacePacker : AssetPacker<TypefaceData>
    {
        public override int Version => 1;

        public override void Pack(BinaryWriter writer, TypefaceData asset)
        {
            writer.Write(asset.Bytes.Length);
            writer.Write(asset.Bytes);
        }

        public override TypefaceData Unpack(BinaryReader reader, int version)
        {
            int length = reader.ReadInt32();
            return new TypefaceData(reader.ReadBytes(length));
        }
    }
}
