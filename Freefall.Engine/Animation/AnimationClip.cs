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

        /// <summary>Duration in ticks.</summary>
        public float Duration;

        /// <summary>Ticks per second (usually 24 or 30).</summary>
        public float TicksPerSecond = 30;

        /// <summary>Duration in seconds.</summary>
        public float DurationSeconds => Duration / TicksPerSecond;

        public float DurationSecondsInverse => 1f / DurationSeconds;

        public AnimationClip()
        {
            Channels = new ReadOnlyCollection<AnimationChannel>(_channels);
        }

        public void AddChannel(AnimationChannel channel)
        {
            _channels.Add(channel);
        }

        public void RemoveChannel(AnimationChannel channel)
        {
            _channels.Remove(channel);
        }

        /// <summary>
        /// Gets the bone pose at the specified time.
        /// </summary>
        public void GetBonePose(Bone bone, float time, ref BonePose result)
        {
            if (time < DurationSeconds)
                time = (time * TicksPerSecond) % Duration;
            else
                time = DurationSeconds;

            int channelCount = Channels.Count;
            for (int i = 0; i < channelCount; i++)
            {
                if (Channels[i].Hash == bone.Hash)
                {
                    Channels[i].GetBoneTransform(bone, time, ref result);
                    return;
                }
            }
            // DEBUG: Bone didn't match any channel - this will cause issues
            //Debug.LogWarning("AnimationClip", $"Bone '{bone.Name}' (hash {bone.Hash}) has no matching channel!");
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
