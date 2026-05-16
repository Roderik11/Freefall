using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Freefall.Assets;

namespace Freefall.Animation
{
    /// <summary>
    /// Contains animation data for a skeleton.
    /// Loaded from FBX/DAE files.
    /// </summary>
    public class AnimationClip : Asset
    {
        private readonly List<AnimationChannel> _channels = new List<AnimationChannel>();
        public readonly ReadOnlyCollection<AnimationChannel> Channels;

        /// <summary>Animation events triggered at specific times.</summary>
        public List<AnimationEvent> Events = new List<AnimationEvent>();

        /// <summary>The skeleton this clip was imported from (for retargeting).</summary>
        public Skeleton Skeleton { get; set; }

        /// <summary>Duration in ticks.</summary>
        public float Duration;

        /// <summary>Ticks per second (usually 24 or 30).</summary>
        public float TicksPerSecond = 30;

        /// <summary>Duration in seconds.</summary>
        public float DurationSeconds => Duration / TicksPerSecond;

        public float DurationSecondsInverse => 1f / DurationSeconds;

        // Lazy-built lookup: bone name hash → channel index
        private Dictionary<int, int> _channelLookup;

        public AnimationClip()
        {
            Channels = new ReadOnlyCollection<AnimationChannel>(_channels);
        }

        public void AddChannel(AnimationChannel channel)
        {
            _channels.Add(channel);
            _channelLookup = null; // invalidate cache
        }

        public void RemoveChannel(AnimationChannel channel)
        {
            _channels.Remove(channel);
            _channelLookup = null;
        }

        private void EnsureChannelLookup()
        {
            if (_channelLookup != null) return;

            _channelLookup = new Dictionary<int, int>(_channels.Count);
            for (int i = 0; i < _channels.Count; i++)
                _channelLookup[_channels[i].Hash] = i;
        }

        /// <summary>
        /// Gets the bone pose at the specified time.
        /// </summary>
        public void GetBonePose(Bone bone, float time, ref BonePose result)
        {
            if (time < DurationSeconds)
                time = (time * TicksPerSecond) % Duration;
            else
                time = Duration;

            EnsureChannelLookup();

            if (_channelLookup.TryGetValue(bone.Hash, out int channelIndex))
                _channels[channelIndex].GetBoneTransform(bone, time, ref result);
        }
    }

    /// <summary>
    /// An event that fires at a specific normalized time during animation playback.
    /// </summary>
    public class AnimationEvent
    {
        public bool fired;
        public float Time;
        public string Name;

        public void Reset() => fired = false;

        public bool Fire(float time)
        {
            if (fired) return false;
            if (time < Time) return false;
            fired = true;
            return true;
        }
    }
}
