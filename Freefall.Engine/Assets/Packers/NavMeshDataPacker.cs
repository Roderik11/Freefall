using System.IO;

namespace Freefall.Assets.Packers
{
    /// <summary>
    /// Packs baked DotRecast navmesh bytes to/from binary cache files (.navmesh).
    /// Mirrors the CollisionMeshPacker pattern.
    /// </summary>
    [AssetPacker(".navmesh")]
    public class NavMeshDataPacker : AssetPacker<NavMeshData>
    {
        public override int Version => 1;

        public override void Pack(BinaryWriter writer, NavMeshData data)
        {
            writer.Write(data.BakedBytes?.Length ?? 0);
            if (data.BakedBytes != null)
                writer.Write(data.BakedBytes);
        }

        public override NavMeshData Unpack(BinaryReader reader, int version)
        {
            int len = reader.ReadInt32();
            return new NavMeshData { BakedBytes = reader.ReadBytes(len) };
        }
    }
}
