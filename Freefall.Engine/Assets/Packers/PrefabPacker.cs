using System.IO;
using Freefall.Assets;

namespace Freefall.Assets.Packers
{
    /// <summary>
    /// Binary packer for PrefabData.
    /// Stores YAML bytes as a length-prefixed blob. Simple but consistent
    /// with the cache pipeline — game builds get the prefab in Library/.
    /// </summary>
    [AssetPacker(".prefab")]
    public class PrefabPacker : AssetPacker<PrefabData>
    {
        public override int Version => 1;

        public override void Pack(BinaryWriter w, PrefabData data)
        {
            var yaml = data.Yaml ?? System.Array.Empty<byte>();
            w.Write(yaml.Length);
            w.Write(yaml);
        }

        public override PrefabData Unpack(BinaryReader r, int version)
        {
            int length = r.ReadInt32();
            return new PrefabData
            {
                Yaml = r.ReadBytes(length)
            };
        }
    }
}
