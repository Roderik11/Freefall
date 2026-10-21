using System.IO;
using Freefall.Assets;

namespace Freefall.Assets.Packers
{
    /// <summary>
    /// Binary packer for SceneData: the YAML bytes as a length-prefixed blob.
    /// </summary>
    [AssetPacker(".scene")]
    public class ScenePacker : AssetPacker<SceneData>
    {
        public override int Version => 1;

        public override void Pack(BinaryWriter w, SceneData data)
        {
            var yaml = data.Yaml ?? System.Array.Empty<byte>();
            w.Write(yaml.Length);
            w.Write(yaml);
        }

        public override SceneData Unpack(BinaryReader r, int version)
        {
            int length = r.ReadInt32();
            return new SceneData
            {
                Yaml = r.ReadBytes(length)
            };
        }
    }
}
