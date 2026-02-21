using System.IO;
using System.Text;
using Freefall.Serialization;

namespace Freefall.Assets.Packers
{
    /// <summary>
    /// Wrapper for Asset data that flows through the packer pipeline.
    /// Stores the serialized YAML bytes of the asset definition.
    /// </summary>
    public class AssetDefinitionData
    {
        public byte[] YamlBytes { get; set; }
        public string TypeName { get; set; }

        public AssetDefinitionData() { }

        public AssetDefinitionData(Asset asset)
        {
            TypeName = asset.GetType().Name;
            var yaml = NativeImporter.SaveToString(asset);
            YamlBytes = Encoding.UTF8.GetBytes(yaml);
        }
    }

    /// <summary>
    /// Caches .asset definitions as UTF-8 YAML bytes with a version header.
    /// </summary>
    [AssetPacker(".asset")]
    public class AssetDefinitionPacker : AssetPacker<AssetDefinitionData>
    {
        public override int Version => 2; // Bumped: JSON → YAML

        public override void Pack(BinaryWriter writer, AssetDefinitionData asset)
        {
            writer.Write(asset.TypeName);
            writer.Write(asset.YamlBytes.Length);
            writer.Write(asset.YamlBytes);
        }

        public override AssetDefinitionData Unpack(BinaryReader reader, int version)
        {
            var typeName = reader.ReadString();
            int length = reader.ReadInt32();
            var bytes = reader.ReadBytes(length);
            return new AssetDefinitionData
            {
                TypeName = typeName,
                YamlBytes = bytes
            };
        }
    }
}
