using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Components;

namespace Freefall.Animation
{
    /// <summary>
    /// Represents a single animation state with a clip and playback settings.
    /// Can be used directly or as part of a blend tree.
    /// </summary>
    [Serializable]
    public class AnimationState
    {
        public string Name;
        public AnimationClip Clip;
        public float Speed = 1f;
        public bool Loop;

        public List<AnimationCurve> Curves = new List<AnimationCurve>();

        public float Weight = 1;

        protected float timeElapsed;

        public float TimeElapsed => timeElapsed;

        public float TimeNormalized
        {
            get
            {
                if (Clip == null) return 0;
                return timeElapsed * Clip.DurationSecondsInverse;
            }
        }

        public void SetTimeElapsed(float value)
        {
            timeElapsed = value;
        }

        public void SetTimeNormalized(float value)
        {
            if (Clip == null) return;
            timeElapsed = Clip.DurationSeconds * Math.Clamp(value, 0, 1);
        }

        public virtual int GetStateCount() => 0;
        public virtual AnimationState GetState(int index) => this;

        public virtual void Update(Animator animator, bool fireEvents = true)
        {
            if (Clip == null) return;

            if (Loop)
            {
                timeElapsed += Base.Time.SmoothDelta * Speed;

                if (timeElapsed > Clip.DurationSeconds)
                {
                    timeElapsed %= Clip.DurationSeconds;
                    ResetEvents();
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
                        ResetEvents();
                    }
                }
            }

            if (fireEvents && Weight > .9f)
                FireEvents(animator);

            foreach (var curve in Curves)
            {
                float value = curve.Evaluate(TimeNormalized);
                animator.SetParam(curve.TargetParameter, value);
            }
        }

        private void FireEvents(Animator animator)
        {
            if (Clip?.Events == null) return;
            foreach (var evt in Clip.Events)
            {
                if (evt.Fire(TimeNormalized))
                    animator.FireAnimationEvent(evt.Name);
            }
        }

        private void ResetEvents()
        {
            if (Clip?.Events == null) return;
            foreach (var evt in Clip.Events)
                evt.Reset();
        }

        internal void BlendBone(Bone bone, ref BonePose blendPose, ref BonePose temp)
        {
            if (Weight <= 0) return;

            if (Clip != null)
            {
                Clip.GetBonePose(bone, TimeElapsed, ref temp);
                blendPose.Position = Vector3.Lerp(blendPose.Position, temp.Position, Weight);
                blendPose.Scale = Vector3.Lerp(blendPose.Scale, temp.Scale, Weight);
                blendPose.Rotation = Quaternion.Lerp(blendPose.Rotation, temp.Rotation, Weight);
                return;
            }

            int stateCount = GetStateCount();
            for (int i = 0; i < stateCount; i++)
                GetState(i).BlendBone(bone, ref blendPose, ref temp);
        }
    }
}
