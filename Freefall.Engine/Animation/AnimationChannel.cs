using System;
using System.Collections.Generic;
using System.Numerics;

namespace Freefall.Animation
{
    /// <summary>
    /// Keyframe for Vector3 values (position, scale).
    /// </summary>
    public struct VectorKey
    {
        public float Time;
        public Vector3 Value;
    }

    /// <summary>
    /// Keyframe for Quaternion values (rotation).
    /// </summary>
    public struct QuaternionKey
    {
        public float Time;
        public Quaternion Value;
    }

    /// <summary>
    /// Collection of Vector3 keyframes with interpolation.
    /// </summary>
    public class VectorKeys
    {
        private readonly VectorKey[] array;
        public int Count { get; private set; }

        public VectorKeys(List<VectorKey> list)
        {
            array = list.ToArray();
            Count = array.Length;
        }

        public Vector3 GetValue(float time)
        {
            if (Count < 1) return Vector3.One;
            if (Count == 1) return this[0].Value;

            int index = GetKeyframe(time);
            int next = index + 1;
            if (next > Count - 1) next = 0;
            if (next == index) return this[index].Value;

            float delta = this[next].Time - this[index].Time;
            float factor = (time - this[index].Time) / delta;

            return Vector3.Lerp(this[index].Value, this[next].Value, factor);
        }

        public void GetValue(float time, ref Vector3 result)
        {
            if (Count < 1) return;
            if (Count == 1)
            {
                result = this[0].Value;
                return;
            }

            int index = GetKeyframe(time);
            int next = index + 1;
            if (next > Count - 1) next = 0;
            if (next == index)
            {
                result = this[index].Value;
                return;
            }

            ref VectorKey thisFrame = ref this[index];
            ref VectorKey nextFrame = ref this[next];

            float delta = nextFrame.Time - thisFrame.Time;
            float factor = (time - thisFrame.Time) / delta;

            result = Vector3.Lerp(thisFrame.Value, nextFrame.Value, factor);
        }

        public VectorKey Last() => array[^1];
        public ref VectorKey this[int i] => ref array[i];

        private int GetKeyframe(float time)
        {
            for (int i = 0; i < Count; i++)
            {
                ref var frameTime = ref this[i].Time;
                if (frameTime < time) continue;
                if (frameTime == time) return i;
                if (frameTime > time) return i < 2 ? 0 : i - 1;
            }
            return 0;
        }
    }

    /// <summary>
    /// Collection of Quaternion keyframes with interpolation.
    /// </summary>
    public class QuaternionKeys
    {
        private readonly QuaternionKey[] array;
        public int Count { get; private set; }

        public QuaternionKeys(List<QuaternionKey> list)
        {
            array = list.ToArray();
            Count = array.Length;
        }

        public Quaternion GetValue(float time)
        {
            if (Count < 1) return Quaternion.Identity;
            if (Count == 1) return this[0].Value;

            int index = GetKeyframe(time);
            int next = index + 1;
            if (next > Count - 1) next = 0;
            if (next == index) return this[index].Value;

            float delta = this[next].Time - this[index].Time;
            float factor = (time - this[index].Time) / delta;

            return Quaternion.Slerp(this[index].Value, this[next].Value, factor);
        }

        public void GetValue(float time, ref Quaternion result)
        {
            if (Count < 1) return;
            if (Count == 1)
            {
                result = this[0].Value;
                return;
            }

            int index = GetKeyframe(time);
            int next = index + 1;
            if (next > Count - 1) next = 0;
            if (next == index)
            {
                result = this[index].Value;
                return;
            }

            ref QuaternionKey thisFrame = ref this[index];
            ref QuaternionKey nextFrame = ref this[next];

            float delta = nextFrame.Time - thisFrame.Time;
            float factor = (time - thisFrame.Time) / delta;

            result = Quaternion.Slerp(thisFrame.Value, nextFrame.Value, factor);
        }

        public QuaternionKey Last() => array[^1];
        public ref QuaternionKey this[int i] => ref array[i];

        private int GetKeyframe(float time)
        {
            for (int i = 0; i < Count; i++)
            {
                ref var frameTime = ref this[i].Time;
                if (frameTime < time) continue;
                if (frameTime == time) return i;
                if (frameTime > time) return i < 2 ? 0 : i - 1;
            }
            return 0;
        }
    }

    /// <summary>
    /// Animation data for a single bone.
    /// Contains position, rotation, and scale keyframes.
    /// </summary>
    public class AnimationChannel
    {
        private string _target = string.Empty;
        public int Hash { get; private set; }

        public string Target
        {
            get => _target;
            set
            {
                _target = value;
                Hash = string.IsNullOrEmpty(value) ? 0 : _target.GetHashCode();
            }
        }

        public VectorKeys Position = null!;
        public QuaternionKeys Rotation = null!;
        public VectorKeys Scale = null!;

        public void GetBoneTransform(Bone bone, float time, ref BonePose result)
        {
            Scale?.GetValue(time, ref result.Scale);
            Position?.GetValue(time, ref result.Position);
            Rotation?.GetValue(time, ref result.Rotation);
        }

        public BonePose GetBoneTransform(Bone bone, float time)
        {
            return new BonePose
            {
                Scale = Scale?.GetValue(time) ?? Vector3.One,
                Position = Position?.GetValue(time) ?? Vector3.Zero,
                Rotation = Rotation?.GetValue(time) ?? Quaternion.Identity
            };
        }
    }
}
