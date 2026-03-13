using System.IO;

namespace Freefall.Assets.Packers
{
    /// <summary>
    /// Packs DeltaMap pixel data (R32_Float) to/from binary cache files (.deltamap).
    /// Used as a hidden subasset of .terrain imports.
    /// </summary>
    [AssetPacker(".deltamap")]
    public class DeltaMapPacker : AssetPacker<DeltaMapData>
    {
        public override int Version => 1;

        public override void Pack(BinaryWriter writer, DeltaMapData data)
        {
            writer.Write(data.Width);
            writer.Write(data.Height);
            writer.Write(data.Pixels.Length);
            writer.Write(data.Pixels);
        }

        public override DeltaMapData Unpack(BinaryReader reader, int version)
        {
            int w = reader.ReadInt32();
            int h = reader.ReadInt32();
            int len = reader.ReadInt32();
            return new DeltaMapData
            {
                Width = w,
                Height = h,
                Pixels = reader.ReadBytes(len)
            };
        }
    }
}
