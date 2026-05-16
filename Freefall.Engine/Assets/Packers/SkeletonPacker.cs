using System.IO;
using Freefall.Assets;

namespace Freefall.Animation
{
    /// <summary>
    /// Binary packer for Skeleton assets.
    /// Stores complete bone hierarchy including bind poses, offset matrices,
    /// and parent indices. Used for skinning and animation retargeting.
    /// V2: delegates to Bone.Write/Read for format parity with MeshPacker.
    /// </summary>
    [AssetPacker(".skel")]
    public class SkeletonPacker : AssetPacker<Skeleton>
    {
        public override int Version => 2;

        public override void Pack(BinaryWriter w, Skeleton skeleton)
        {
            w.Write(skeleton.Bones.Length);

            foreach (var bone in skeleton.Bones)
                bone.Write(w);
        }

        public override Skeleton Unpack(BinaryReader r, int version)
        {
            int count = r.ReadInt32();
            var bones = new Bone[count];
            var names = new string[count];

            for (int i = 0; i < count; i++)
            {
                var bone = new Bone();
                bone.Read(r);
                bones[i] = bone;
                names[i] = bone.Name;
            }

            return new Skeleton { Bones = bones, BoneNames = names };
        }
    }
}
