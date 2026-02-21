using System.IO;
using Freefall.Assets;

namespace Freefall.Animation
{
    /// <summary>
    /// Binary packer for Skeleton assets.
    /// Stores complete bone hierarchy including bind poses, offset matrices,
    /// and parent indices. Used for skinning and animation retargeting.
    /// </summary>
    [AssetPacker(".skel")]
    public class SkeletonPacker : AssetPacker<Skeleton>
    {
        public override int Version => 1;

        public override void Pack(BinaryWriter w, Skeleton skeleton)
        {
            w.Write(skeleton.Bones.Length);

            foreach (var bone in skeleton.Bones)
            {
                w.Write(bone.Name);
                w.Write(bone.Parent);
                w.Write(bone.BindPose.Position);
                w.Write(bone.BindPose.Rotation);
                w.Write(bone.BindPose.Scale);
                w.Write(bone.BindPoseMatrix);
                w.Write(bone.OffsetMatrix);
            }
        }

        public override Skeleton Unpack(BinaryReader r, int version)
        {
            int count = r.ReadInt32();
            var bones = new Bone[count];
            var names = new string[count];

            for (int i = 0; i < count; i++)
            {
                var bone = new Bone();
                bone.Name = r.ReadString();
                bone.Parent = r.ReadInt32();
                bone.BindPose.Position = r.ReadVector3();
                bone.BindPose.Rotation = r.ReadQuaternion();
                bone.BindPose.Scale = r.ReadVector3();
                bone.BindPoseMatrix = r.ReadMatrix4x4();
                bone.OffsetMatrix = r.ReadMatrix4x4();

                bones[i] = bone;
                names[i] = bone.Name;
            }

            return new Skeleton { Bones = bones, BoneNames = names };
        }
    }
}
