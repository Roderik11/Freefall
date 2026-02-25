using System.IO;

namespace Freefall.Assets.Packers
{
    /// <summary>
    /// Packs pre-cooked PhysX collision mesh data to/from binary cache files (.physx).
    /// </summary>
    [AssetPacker(".physx")]
    public class CollisionMeshPacker : AssetPacker<CollisionMeshData>
    {
        public override int Version => 1;

        public override void Pack(BinaryWriter writer, CollisionMeshData data)
        {
            writer.Write(data.CookedBytes.Length);
            writer.Write(data.CookedBytes);
        }

        public override CollisionMeshData Unpack(BinaryReader reader, int version)
        {
            int len = reader.ReadInt32();
            return new CollisionMeshData { CookedBytes = reader.ReadBytes(len) };
        }
    }
}
