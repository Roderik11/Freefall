using System;
using System.Numerics;

namespace Freefall.Animation
{
    /// <summary>
    /// One clip of a state as a pose reads it: which channel moves each bone and the time to
    /// read it at. Worked out once per pose, so that a bone only reads keys.
    /// </summary>
    internal struct ClipSample
    {
        /// <summary>Per bone of the skeleton being posed, the clip's channel for it (null: the clip leaves it in its bind pose).</summary>
        public AnimationChannel[] Channels;

        /// <summary>Time of the keys to read.</summary>
        public float Ticks;

        /// <summary>Share of this clip in the blend of the ones before it and itself (the first one stands as it is).</summary>
        public float Share;

        /// <summary>
        /// While the Animator's animation set is being swapped: the clip the old set had here, at
        /// the same point of its cycle, and how far the fade to the new one has come. Null otherwise.
        /// </summary>
        public AnimationChannel[] PreviousChannels;
        public float PreviousTicks;
        public float SetFade;

        /// <summary>Set for a clip played left for right: each bone reads its counterpart and turns the pose over. Null otherwise.</summary>
        public MirrorMap Mirror;

        /// <summary>
        /// Mirrored, only the hips and legs change sides: the upper body plays as it is and keeps
        /// facing the way it does in the clip (see <see cref="AnimationSet.MirrorLegsOnly"/>).
        /// </summary>
        public bool MirrorLegsOnly;

        /// <summary>
        /// A bone whose position is moved by a fixed amount, in its parent's space, and the
        /// amount (see <see cref="AnimationState.Offset"/>). -1: none.
        /// </summary>
        public int ShiftBone;
        public Vector3 Shift;
    }

    /// <summary>
    /// The clips a state plays for one pose: one for a plain state, the weighted children of a
    /// blend tree, none for a state that plays nothing. Filled by <see cref="AnimationState.Collect"/>.
    /// </summary>
    internal sealed class StateSamples
    {
        private ClipSample[] _items = new ClipSample[1];
        private int _count;

        /// <summary>
        /// The hips of the skeleton being posed, or -1 when it has none. Clips are collected
        /// without their travel over the ground on that bone (see <see cref="RootMotionTrack"/>).
        /// </summary>
        public int RootBone = -1;

        /// <summary>The skeleton being posed: a mirrored clip looks up its bone pairs by it.</summary>
        public Skeleton Skeleton;

        public int Count => _count;

        public void Clear() => _count = 0;

        public ref ClipSample Add()
        {
            if (_count == _items.Length)
                Array.Resize(ref _items, _count * 2);
            return ref _items[_count++];
        }

        /// <summary>
        /// The pose these clips give one bone, starting from its bind pose. False when there are
        /// none, which lets the layers below show through.
        /// </summary>
        public bool Sample(Bone bone, int boneIndex, ref BonePose pose)
        {
            if (_count == 0) return false;

            var items = _items;
            Read(ref items[0], bone, boneIndex, ref pose);

            for (int i = 1; i < _count; i++)
            {
                BonePose other = default;
                Read(ref items[i], bone, boneIndex, ref other);

                float share = items[i].Share;
                pose.Position = Vector3.Lerp(pose.Position, other.Position, share);
                pose.Scale = Vector3.Lerp(pose.Scale, other.Scale, share);
                pose.Rotation = Quaternion.Lerp(pose.Rotation, other.Rotation, share);
            }

            return true;
        }

        private static void Read(ref ClipSample sample, Bone bone, int boneIndex, ref BonePose pose)
        {
            ReadClip(ref sample, bone, boneIndex, ref pose);
            if (boneIndex == sample.ShiftBone) pose.Position += sample.Shift;
        }

        private static void ReadClip(ref ClipSample sample, Bone bone, int boneIndex, ref BonePose pose)
        {
            var mirror = sample.Mirror;
            if (mirror == null)
            {
                ReadPlain(ref sample, bone, boneIndex, ref pose);
                return;
            }

            // The stride alone changes sides: the upper body plays as it is
            if (sample.MirrorLegsOnly && mirror.Hips >= 0 && !mirror.Lower[boneIndex])
            {
                ReadPlain(ref sample, bone, boneIndex, ref pose);

                // What sits on the hips keeps facing the way it does in the clip, although the
                // hips under it have turned over
                if (mirror.OnHips[boneIndex])
                {
                    BonePose hips = default;
                    ReadPlain(ref sample, mirror.Bones[mirror.Hips], mirror.Hips, ref hips);

                    BonePose turned = hips;
                    mirror.Apply(mirror.Hips, ref turned);

                    pose.Rotation = Quaternion.Normalize(Quaternion.Inverse(turned.Rotation) * hips.Rotation * pose.Rotation);
                }
                return;
            }

            // Left for right: the pose of the bone on the other side, turned over
            int partner = mirror.Partner[boneIndex];
            if (partner < 0)
            {
                pose = bone.BindPose; // no counterpart
                return;
            }

            ReadPlain(ref sample, mirror.Bones[partner], partner, ref pose);
            mirror.Apply(boneIndex, ref pose);
        }

        private static void ReadPlain(ref ClipSample sample, Bone bone, int boneIndex, ref BonePose pose)
        {
            pose = bone.BindPose;
            sample.Channels[boneIndex]?.GetBoneTransform(bone, sample.Ticks, ref pose);

            if (sample.PreviousChannels != null)
            {
                BonePose old = bone.BindPose;
                sample.PreviousChannels[boneIndex]?.GetBoneTransform(bone, sample.PreviousTicks, ref old);

                pose.Position = Vector3.Lerp(old.Position, pose.Position, sample.SetFade);
                pose.Scale = Vector3.Lerp(old.Scale, pose.Scale, sample.SetFade);
                pose.Rotation = Quaternion.Lerp(old.Rotation, pose.Rotation, sample.SetFade);
            }
        }
    }
}
