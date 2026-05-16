using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Numerics;
using Freefall.Components;

namespace Freefall.Animation
{
    /// <summary>
    /// Represents a single animation state with a clip and playback settings.
    /// All mutable playback state (time, weight) lives in AnimationPlayback.
    /// </summary>
    [Serializable]
    public class AnimationState
    {
        [Browsable(false)]
        public int ID;

        [Browsable(false)]
        public Vector2 Position;

        public string Name;
        public AnimationClip Clip;
        public float Speed = 1f;
        public bool Loop;

        public List<AnimationCurve> Curves = new List<AnimationCurve>();

        public virtual AnimationClip GetDominantClip() => Clip;

        // --- Playback accessors (read from blackboard) ---

        public float GetTimeElapsed(AnimationPlayback pb) => pb.Get(PK.State(ID, PK.Time));

        public void SetTimeElapsed(AnimationPlayback pb, float value) => pb.Set(PK.State(ID, PK.Time), value);

        public float GetWeight(AnimationPlayback pb) => pb.Get(PK.State(ID, PK.Weight));

        public void SetWeight(AnimationPlayback pb, float value) => pb.Set(PK.State(ID, PK.Weight), value);

        public float GetTimeNormalized(AnimationPlayback pb)
        {
            if (Clip == null) return 0;
            return GetTimeElapsed(pb) * Clip.DurationSecondsInverse;
        }

        public void SetTimeNormalized(AnimationPlayback pb, float value)
        {
            if (Clip == null) return;
            SetTimeElapsed(pb, Clip.DurationSeconds * Math.Clamp(value, 0, 1));
        }

        public virtual int GetStateCount() => 0;
        public virtual AnimationState GetState(int index) => this;

        public virtual void Update(Animator animator, AnimationPlayback pb, bool fireEvents = true)
        {
            if (Clip == null) return;

            float timeElapsed = GetTimeElapsed(pb);

            if (Loop)
            {
                timeElapsed += Base.Time.SmoothDelta * Speed;

                if (timeElapsed > Clip.DurationSeconds)
                {
                    timeElapsed %= Clip.DurationSeconds;
                    ResetEvents(pb);
                }
            }
            else
            {
                if (timeElapsed < Clip.DurationSeconds)
                {
                    timeElapsed += Base.Time.SmoothDelta * Speed;

                    if (timeElapsed > Clip.DurationSeconds)
                    {
                        timeElapsed = Clip.DurationSeconds;
                        ResetEvents(pb);
                    }
                }
            }

            SetTimeElapsed(pb, timeElapsed);

            if (fireEvents)
                FireEvents(animator, pb);

            foreach (var curve in Curves)
            {
                float value = curve.Evaluate(GetTimeNormalized(pb));
                animator.SetParam(curve.TargetParameter, value);
            }
        }

        private void FireEvents(Animator animator, AnimationPlayback pb)
        {
            if (Clip?.Events == null) return;
            float normalizedTime = GetTimeNormalized(pb);

            for (int i = 0; i < Clip.Events.Count; i++)
            {
                var evt = Clip.Events[i];
                long key = PK.Event(ID, i);

                if (!pb.GetBool(key) && normalizedTime >= evt.Time)
                {
                    pb.Set(key, 1);
                    animator.FireAnimationEvent(evt.Name);
                }
            }
        }

        private void ResetEvents(AnimationPlayback pb)
        {
            if (Clip?.Events == null) return;
            for (int i = 0; i < Clip.Events.Count; i++)
                pb.Set(PK.Event(ID, i), 0);
        }

        internal virtual void BlendBone(Bone bone, ref BonePose blendPose, ref BonePose temp, AnimationPlayback pb)
        {
            float weight = GetWeight(pb);
            if (weight <= 0) return;

            if (Clip != null)
            {
                Clip.GetBonePose(bone, GetTimeElapsed(pb), ref temp);
                blendPose.Position = Vector3.Lerp(blendPose.Position, temp.Position, weight);
                blendPose.Scale = Vector3.Lerp(blendPose.Scale, temp.Scale, weight);
                blendPose.Rotation = Quaternion.Lerp(blendPose.Rotation, temp.Rotation, weight);
                return;
            }

            int stateCount = GetStateCount();
            for (int i = 0; i < stateCount; i++)
                GetState(i).BlendBone(bone, ref blendPose, ref temp, pb);
        }
    }
}
