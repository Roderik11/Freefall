using System.Collections.Generic;
using System.IO;
using System.Numerics;
using Freefall.Assets;

namespace Freefall.Animation
{
    /// <summary>
    /// Binary packer for AnimationClip assets.
    /// Stores all channels with their position, rotation, and scale keyframes.
    /// </summary>
    [AssetPacker(".anim")]
    public class AnimationClipPacker : AssetPacker<AnimationClip>
    {
        public override int Version => 1;

        public override void Pack(BinaryWriter w, AnimationClip clip)
        {
            w.Write(clip.Name ?? string.Empty);
            w.Write(clip.Duration);
            w.Write(clip.TicksPerSecond);

            w.Write(clip.Channels.Count);
            foreach (var channel in clip.Channels)
            {
                w.Write(channel.Target ?? string.Empty);
                WriteVectorKeys(w, channel.Position);
                WriteQuaternionKeys(w, channel.Rotation);
                WriteVectorKeys(w, channel.Scale);
            }
        }

        public override AnimationClip Unpack(BinaryReader r, int version)
        {
            var clip = new AnimationClip();
            clip.Name = r.ReadString();
            clip.Duration = r.ReadSingle();
            clip.TicksPerSecond = r.ReadSingle();

            int channelCount = r.ReadInt32();
            for (int i = 0; i < channelCount; i++)
            {
                var channel = new AnimationChannel();
                channel.Target = r.ReadString();
                channel.Position = ReadVectorKeys(r);
                channel.Rotation = ReadQuaternionKeys(r);
                channel.Scale = ReadVectorKeys(r);
                clip.AddChannel(channel);
            }

            return clip;
        }

        // ── Keyframe helpers (AnimationClip-specific) ──

        private static void WriteVectorKeys(BinaryWriter w, VectorKeys keys)
        {
            if (keys == null) { w.Write(0); return; }
            w.Write(keys.Count);
            for (int i = 0; i < keys.Count; i++)
            {
                ref var key = ref keys[i];
                w.Write(key.Time);
                w.Write(key.Value);
            }
        }

        private static VectorKeys ReadVectorKeys(BinaryReader r)
        {
            int count = r.ReadInt32();
            if (count == 0) return null;
            var list = new List<VectorKey>(count);
            for (int i = 0; i < count; i++)
                list.Add(new VectorKey { Time = r.ReadSingle(), Value = r.ReadVector3() });
            return new VectorKeys(list);
        }

        private static void WriteQuaternionKeys(BinaryWriter w, QuaternionKeys keys)
        {
            if (keys == null) { w.Write(0); return; }
            w.Write(keys.Count);
            for (int i = 0; i < keys.Count; i++)
            {
                ref var key = ref keys[i];
                w.Write(key.Time);
                w.Write(key.Value);
            }
        }

        private static QuaternionKeys ReadQuaternionKeys(BinaryReader r)
        {
            int count = r.ReadInt32();
            if (count == 0) return null;
            var list = new List<QuaternionKey>(count);
            for (int i = 0; i < count; i++)
                list.Add(new QuaternionKey { Time = r.ReadSingle(), Value = r.ReadQuaternion() });
            return new QuaternionKeys(list);
        }
    }
}
