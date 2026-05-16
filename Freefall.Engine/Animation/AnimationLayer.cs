using System;
using System.Collections.Generic;
using Freefall.Components;

namespace Freefall.Animation
{
    /// <summary>
    /// Manages a group of animation states and transitions between them.
    /// All mutable playback state lives in AnimationPlayback.
    /// </summary>
    [Serializable]
    public class AnimationLayer
    {
        public string Name;
        public AnimationMask Mask;
        public float Weight;
        public List<AnimationState> States = new List<AnimationState>();
        public List<AnimationTransition> Transitions = new List<AnimationTransition>();

        /// <summary>Index in the parent Animation's Layers list. Set during RebuildAfterLoad.</summary>
        public int Index;

        // --- Playback accessors ---

        private AnimationState GetCurrentState(AnimationPlayback pb)
        {
            int id = (int)pb.Get(PK.Layer(Index, PK.Current), -1);
            if (id < 0 && States.Count > 0)
            {
                // Initialize to first state
                SetCurrentState(pb, States[0]);
                return States[0];
            }
            return FindStateById(id);
        }

        private void SetCurrentState(AnimationPlayback pb, AnimationState state)
            => pb.Set(PK.Layer(Index, PK.Current), state?.ID ?? -1);

        private bool GetInTransition(AnimationPlayback pb)
            => pb.GetBool(PK.Layer(Index, PK.InTransition));

        private void SetInTransition(AnimationPlayback pb, bool value)
            => pb.Set(PK.Layer(Index, PK.InTransition), value ? 1 : 0);

        private float GetTransitionTime(AnimationPlayback pb)
            => pb.Get(PK.Layer(Index, PK.TransTime));

        private void SetTransitionTime(AnimationPlayback pb, float value)
            => pb.Set(PK.Layer(Index, PK.TransTime), value);

        private float GetTransitionDuration(AnimationPlayback pb)
            => pb.Get(PK.Layer(Index, PK.TransDuration), 0.2f);

        private void SetTransitionDuration(AnimationPlayback pb, float value)
            => pb.Set(PK.Layer(Index, PK.TransDuration), value);

        private AnimationState GetTransSource(AnimationPlayback pb)
            => FindStateById((int)pb.Get(PK.Layer(Index, PK.TransSource), -1));

        private void SetTransSource(AnimationPlayback pb, AnimationState state)
            => pb.Set(PK.Layer(Index, PK.TransSource), state?.ID ?? -1);

        private AnimationState GetTransTarget(AnimationPlayback pb)
            => FindStateById((int)pb.Get(PK.Layer(Index, PK.TransTarget), -1));

        private void SetTransTarget(AnimationPlayback pb, AnimationState state)
            => pb.Set(PK.Layer(Index, PK.TransTarget), state?.ID ?? -1);

        private AnimationState GetReturnToState(AnimationPlayback pb)
            => FindStateById((int)pb.Get(PK.Layer(Index, PK.ReturnTo), -1));

        private void SetReturnToState(AnimationPlayback pb, AnimationState state)
            => pb.Set(PK.Layer(Index, PK.ReturnTo), state?.ID ?? -1);

        private AnimationState FindStateById(int id)
        {
            if (id < 0) return null;
            foreach (var s in States)
                if (s.ID == id) return s;
            return null;
        }

        // --- Public API ---

        public int GetStateCount(AnimationPlayback pb) => GetInTransition(pb) ? 2 : 1;

        public AnimationState GetState(int index, AnimationPlayback pb)
        {
            if (GetInTransition(pb))
            {
                if (index == 0)
                    return GetTransSource(pb);
                return GetTransTarget(pb);
            }

            return GetCurrentState(pb);
        }

        public void Update(Animator animator, AnimationPlayback pb)
        {
            var currentState = GetCurrentState(pb);

            if (GetInTransition(pb))
            {
                float transitionTime = GetTransitionTime(pb) + Base.Time.SmoothDelta;
                float duration = GetTransitionDuration(pb);

                if (transitionTime > duration)
                {
                    if (currentState != null)
                    {
                        currentState.SetWeight(pb, 1);
                        currentState.Update(animator, pb, false);
                    }

                    SetInTransition(pb, false);
                    return;
                }

                SetTransitionTime(pb, transitionTime);

                var source = GetTransSource(pb);
                var target = GetTransTarget(pb);

                if (target != null)
                    target.SetWeight(pb, Math.Clamp(transitionTime / duration, 0, 1));
                if (source != null)
                    source.SetWeight(pb, 1);

                source?.Update(animator, pb, false);
                target?.Update(animator, pb, true);
            }
            else
            {
                foreach (AnimationTransition transition in Transitions)
                {
                    if (transition.Source != currentState)
                        continue;

                    bool allConditionsMet = true;

                    if (transition.Conditions.Count == 0)
                        allConditionsMet = currentState.GetTimeNormalized(pb) >= .8f;

                    foreach (AnimationCondition condition in transition.Conditions)
                    {
                        if (condition.AutoReturn)
                        {
                            allConditionsMet = false;

                            if (currentState.GetTimeElapsed(pb) > currentState.Clip.DurationSeconds * condition.AutoThreshold)
                            {
                                SetInTransition(pb, true);
                                SetTransitionTime(pb, 0);
                                SetTransitionDuration(pb, transition.Duration);

                                var returnTo = GetReturnToState(pb);
                                SetTransSource(pb, currentState);
                                SetTransTarget(pb, returnTo);

                                SetCurrentState(pb, returnTo);
                                SetReturnToState(pb, null);
                                Update(animator, pb);
                                return;
                            }
                        }
                        else if (!condition.IsMet(animator))
                        {
                            allConditionsMet = false;
                            break;
                        }
                    }

                    if (allConditionsMet)
                    {
                        SetInTransition(pb, true);
                        SetTransitionTime(pb, 0);
                        SetTransitionDuration(pb, transition.Duration);
                        SetTransSource(pb, currentState);
                        SetTransTarget(pb, transition.Target);
                        SetReturnToState(pb, currentState);

                        // Consume trigger parameters
                        foreach (var cond in transition.Conditions)
                        {
                            if (!string.IsNullOrEmpty(cond.Parameter))
                                animator.ConsumeTrigger(cond.Parameter);
                        }

                        SetCurrentState(pb, transition.Target);
                        transition.Target.SetTimeElapsed(pb, 0);
                        Update(animator, pb);
                        return;
                    }
                }

                if (currentState != null)
                {
                    currentState.SetWeight(pb, 1);
                    currentState.Update(animator, pb);
                }
            }
        }
    }
}
