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

        /// <summary>
        /// How much of this layer goes on top of the ones before it, 0..1. At 0 the layer is not
        /// evaluated. The first layer is the base and always plays in full.
        /// </summary>
        public float Weight = 1f;

        /// <summary>
        /// Optional animation parameter that scales <see cref="Weight"/> at runtime: set it from
        /// gameplay, or drive it with a curve on a state so a layer fades in and out with its clip.
        /// </summary>
        public string WeightParameter;

        /// <summary>
        /// How much the masked bones follow the rotation of the body below the mask.
        /// 1 = fully: the layer blends in local space, so an upper-body layer leans and sways
        /// with the hips of the layer underneath. 0 = the layer keeps its own model-space
        /// orientation, e.g. an attack stays as authored over a running lower body.
        /// Only used with a <see cref="Mask"/>.
        /// </summary>
        public float ParentInfluence = 1f;

        // States and transitions are edited on the animation editor's canvas, not as lists
        [System.ComponentModel.Browsable(false)]
        public List<AnimationState> States = new List<AnimationState>();
        [System.ComponentModel.Browsable(false)]
        public List<AnimationTransition> Transitions = new List<AnimationTransition>();

        /// <summary>Index in the parent Animation's Layers list, kept by <see cref="Animation.ReindexLayers"/>.</summary>
        [System.ComponentModel.Browsable(false)]
        public int Index;

        // --- Playback accessors ---

        private AnimationState GetCurrentState(AnimationPlayback pb)
        {
            int id = (int)pb.Get(PK.Layer(Index, PK.Current), -1);
            var state = FindStateById(id);
            if (state == null && States.Count > 0)
            {
                // Initialize to first state. Also when the recorded one is gone: the state was deleted,
                // or the layers were reordered while this animator was playing.
                SetCurrentState(pb, States[0]);
                return States[0];
            }
            return state;
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

        /// <summary>The layer's weight on one Animator this frame.</summary>
        public float GetWeight(Animator animator)
        {
            if (Index == 0) return 1f;

            float weight = Weight;
            if (!string.IsNullOrEmpty(WeightParameter))
                weight *= animator.GetParam(WeightParameter);

            return Math.Clamp(weight, 0f, 1f);
        }

        /// <summary>The state the layer plays on one Animator. During a transition that is already its target.</summary>
        public AnimationState GetCurrent(AnimationPlayback pb) => GetCurrentState(pb);

        /// <summary>
        /// How far the running transition has come, 0..1 with ease in and out; 1 when none is running.
        /// The pose carries what is left of the difference to the pose shown when it started.
        /// </summary>
        public float GetFade(AnimationPlayback pb)
        {
            if (!GetInTransition(pb)) return 1f;

            float duration = GetTransitionDuration(pb);
            if (duration <= 0f) return 1f;

            float x = Math.Clamp(GetTransitionTime(pb) / duration, 0f, 1f);
            return x * x * (3f - 2f * x);
        }

        /// <summary>Counts the transitions started on one Animator, so a pose can tell a new one has begun.</summary>
        public int GetTransitionSerial(AnimationPlayback pb) => (int)pb.Get(PK.Layer(Index, PK.Serial));

        /// <summary>The state the last transition left, advanced to the frame it started in.</summary>
        public AnimationState GetPrevious(AnimationPlayback pb)
            => FindStateById((int)pb.Get(PK.Layer(Index, PK.TransSource), -1));

        /// <summary>How far the transition before the last one had come when it was interrupted (1 = it had finished).</summary>
        public float GetPreviousFade(AnimationPlayback pb) => pb.Get(PK.Layer(Index, PK.PreviousFade), 1f);

        public int GetStateCount(AnimationPlayback pb) => 1;

        public AnimationState GetState(int index, AnimationPlayback pb) => GetCurrentState(pb);

        /// <summary>
        /// Make <paramref name="target"/> the current state. Only it is played from here on: the
        /// pose blends out of whatever the layer was showing (Animator.BlendLayers), so a transition
        /// can start at any moment, also in the middle of another one.
        /// </summary>
        private void StartTransition(Animator animator, AnimationPlayback pb, AnimationState current, AnimationState target, float duration)
        {
            // The pose blends out of what the layer would have shown this frame: the state it leaves,
            // moved on by this frame, with what was left of a transition it interrupts.
            pb.Set(PK.Layer(Index, PK.PreviousFade), GetFade(pb));
            pb.Set(PK.Layer(Index, PK.TransSource), current.ID);
            current.Update(animator, pb, false);

            SetInTransition(pb, duration > 0f);
            SetTransitionTime(pb, 0);
            SetTransitionDuration(pb, duration);
            pb.Set(PK.Layer(Index, PK.Serial), pb.Get(PK.Layer(Index, PK.Serial)) + 1);

            current.SetWeight(pb, 0);
            SetCurrentState(pb, target);
        }

        public void Update(Animator animator, AnimationPlayback pb)
        {
            var currentState = GetCurrentState(pb);

            if (GetInTransition(pb))
            {
                float transitionTime = GetTransitionTime(pb) + Base.Time.SmoothDelta;
                SetTransitionTime(pb, transitionTime);

                if (transitionTime >= GetTransitionDuration(pb))
                    SetInTransition(pb, false);
            }

            if (currentState == null) return;

            // Checked every frame, also while a transition is still fading: input is never held back
            foreach (AnimationTransition transition in Transitions)
            {
                if (transition.Source != currentState)
                    continue;

                AnimationState target = null;
                bool allConditionsMet = true;

                // By exit time. A state that plays nothing has nothing to wait for and leaves at
                // once: its clip can go from under it when the animation set changes (a bow's
                // draw has no clip with a sword), and it would never get to its exit time.
                if (transition.Conditions.Count == 0)
                    allConditionsMet = currentState.GetTimeNormalized(pb) >= transition.ExitTime || !currentState.Plays(pb);

                foreach (AnimationCondition condition in transition.Conditions)
                {
                    if (condition.AutoReturn)
                    {
                        allConditionsMet = false;

                        var clip = currentState.GetClip(pb);
                        if (clip != null && currentState.GetTimeElapsed(pb) > clip.DurationSeconds * condition.AutoThreshold)
                        {
                            // Back to where the layer came from
                            target = GetReturnToState(pb) ?? States[0];
                            SetReturnToState(pb, null);
                            break;
                        }
                    }
                    else if (!condition.IsMet(animator))
                    {
                        allConditionsMet = false;
                        break;
                    }
                }

                if (target == null && allConditionsMet)
                {
                    target = transition.Target;
                    SetReturnToState(pb, currentState);

                    // Consume trigger parameters
                    foreach (var cond in transition.Conditions)
                    {
                        if (!string.IsNullOrEmpty(cond.Parameter))
                            animator.ConsumeTrigger(cond.Parameter);
                    }

                    target.Enter(animator, pb, currentState.GetPhase(pb));
                }

                if (target != null)
                {
                    StartTransition(animator, pb, currentState, target, transition.Duration);
                    currentState = target;
                    break;
                }
            }

            currentState.SetWeight(pb, 1);

            animator.BeginLayerRootMotion();
            currentState.Update(animator, pb);
            animator.EndLayerRootMotion(this);
        }
    }
}
