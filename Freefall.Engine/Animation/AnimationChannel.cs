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
    /// How the keys of one track are laid out in time, worked out once so sampling does not search.
    /// Baked clips have a key per frame: the key before a time is then found by index. Any other
    /// track is searched by bisection.
    /// </summary>
    internal readonly struct KeyTiming
    {
        private readonly bool _uniform;
        private readonly float _start, _inverseStep;
        private readonly int _last;

        public KeyTiming(int count, Func<int, float> timeOf)
        {
            _last = count - 1;
            _uniform = false;
            _start = 0;
            _inverseStep = 0;
            if (count < 2) return;

            _start = timeOf(0);
            float step = (timeOf(_last) - _start) / _last;
            if (step <= 0) return;

            float tolerance = step * 1e-3f;
            for (int i = 1; i < _last; i++)
                if (MathF.Abs(timeOf(i) - (_start + i * step)) > tolerance) return;

            _uniform = true;
            _inverseStep = 1f / step;
        }

        /// <summary>The keys are evenly spaced: key i is at Start + i / InverseStep.</summary>
        public bool IsUniform => _uniform;
        public float Start => _start;
        public float InverseStep => _inverseStep;

        /// <summary>Index of the last key.</summary>
        public int Last => _last;
    }

    /// <summary>
    /// Collection of Vector3 keyframes with interpolation.
    /// </summary>
    public class VectorKeys
    {
        private readonly VectorKey[] array;
        public int Count { get; private set; }

        private readonly KeyTiming _timing;

        // Every key holds the same value, as on most position and scale tracks
        private readonly bool _constant;

        public VectorKeys(List<VectorKey> list)
        {
            array = list.ToArray();
            Count = array.Length;

            var keys = array;
            _timing = new KeyTiming(Count, i => keys[i].Time);

            _constant = true;
            for (int i = 1; i < Count && _constant; i++)
                _constant = array[i].Value == array[0].Value;
        }

        public Vector3 GetValue(float time)
        {
            if (Count < 1) return Vector3.Zero;
            if (_constant) return array[0].Value;

            Locate(time, out int index, out float fraction);
            if (fraction <= 0f) return array[index].Value;

            return Vector3.Lerp(array[index].Value, array[index + 1].Value, fraction);
        }

        public void GetValue(float time, ref Vector3 result)
        {
            if (Count < 1) return;
            result = GetValue(time);
        }

        public VectorKey Last() => array[^1];
        public ref VectorKey this[int i] => ref array[i];

        private void Locate(float time, out int index, out float fraction)
        {
            int last = _timing.Last;

            if (_timing.IsUniform)
            {
                float position = (time - _timing.Start) * _timing.InverseStep;
                if (position <= 0f) { index = 0; fraction = 0f; return; }
                if (position >= last) { index = last; fraction = 0f; return; }

                index = (int)position;
                fraction = position - index;
                return;
            }

            if (time <= array[0].Time) { index = 0; fraction = 0f; return; }
            if (time >= array[last].Time) { index = last; fraction = 0f; return; }

            // array[low].Time <= time < array[high].Time
            int low = 0, high = last;
            while (high - low > 1)
            {
                int middle = (low + high) >> 1;
                if (array[middle].Time <= time) low = middle;
                else high = middle;
            }

            index = low;
            float span = array[low + 1].Time - array[low].Time;
            fraction = span > 0f ? (time - array[low].Time) / span : 0f;
        }
    }

    /// <summary>
    /// Collection of Quaternion keyframes with interpolation.
    /// </summary>
    public class QuaternionKeys
    {
        private readonly QuaternionKey[] array;
        public int Count { get; private set; }

        private readonly KeyTiming _timing;

        // Every key holds the same value: a bone the clip does not move
        private readonly bool _constant;

        public QuaternionKeys(List<QuaternionKey> list)
        {
            array = list.ToArray();
            Count = array.Length;

            var keys = array;
            _timing = new KeyTiming(Count, i => keys[i].Time);

            _constant = true;
            for (int i = 1; i < Count && _constant; i++)
                _constant = array[i].Value == array[0].Value;
        }

        public Quaternion GetValue(float time)
        {
            if (Count < 1) return Quaternion.Identity;
            if (_constant) return array[0].Value;

            Locate(time, out int index, out float fraction);
            if (fraction <= 0f) return array[index].Value;

            // Neighbouring keys are a frame apart: a normalised lerp is indistinguishable from a
            // slerp over so small an angle, without its trigonometry
            return Quaternion.Lerp(array[index].Value, array[index + 1].Value, fraction);
        }

        public void GetValue(float time, ref Quaternion result)
        {
            if (Count < 1) return;
            result = GetValue(time);
        }

        public QuaternionKey Last() => array[^1];
        public ref QuaternionKey this[int i] => ref array[i];

        private void Locate(float time, out int index, out float fraction)
        {
            int last = _timing.Last;

            if (_timing.IsUniform)
            {
                float position = (time - _timing.Start) * _timing.InverseStep;
                if (position <= 0f) { index = 0; fraction = 0f; return; }
                if (position >= last) { index = last; fraction = 0f; return; }

                index = (int)position;
                fraction = position - index;
                return;
            }

            if (time <= array[0].Time) { index = 0; fraction = 0f; return; }
            if (time >= array[last].Time) { index = last; fraction = 0f; return; }

            // array[low].Time <= time < array[high].Time
            int low = 0, high = last;
            while (high - low > 1)
            {
                int middle = (low + high) >> 1;
                if (array[middle].Time <= time) low = middle;
                else high = middle;
            }

            index = low;
            float span = array[low + 1].Time - array[low].Time;
            fraction = span > 0f ? (time - array[low].Time) / span : 0f;
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
