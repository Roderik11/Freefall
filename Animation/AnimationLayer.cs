using System;
using System.Collections.Generic;
using Freefall.Components;

namespace Freefall.Animation
{
    /// <summary>
    /// Manages a group of animation states and transitions between them.
    /// Supports masking for partial body animation.
    /// </summary>
    [Serializable]
    public class AnimationLayer
    {
        public string Name;
        public AnimationMask Mask;
        public float Weight;
        public List<AnimationState> States = new List<AnimationState>();
        public List<AnimationTransition> Transitions = new List<AnimationTransition>();

        public bool InTransition { get; private set; }
        public AnimationState CurrentState { get; private set; }
        public AnimationTransition CurrentTransition { get; private set; }

        private AnimationState returnToState;
        private float transitionTime;

        public int GetStateCount() => InTransition ? 2 : 1;

        public AnimationState GetState(int index)
        {
            if (InTransition)
            {
                if (index == 0)
                    return CurrentTransition.Source;
                return CurrentTransition.Target;
            }

            if (CurrentState == null && States.Count > 0)
                CurrentState = States[0];

            return CurrentState;
        }

        public void Update(Animator animator)
        {
            if (CurrentState == null && States.Count > 0)
                CurrentState = States[0];

            if (InTransition)
            {
                transitionTime += Base.Time.SmoothDelta;

                if (transitionTime > CurrentTransition.Duration)
                {
                    if (CurrentState != null)
                    {
                        CurrentState.Weight = 1;
                        CurrentState.Update(animator, false);
                    }

                    InTransition = false;
                    CurrentTransition = null;
                    return;
                }

                CurrentTransition.Target.Weight = Math.Clamp(transitionTime / CurrentTransition.Duration, 0, 1);
                CurrentTransition.Source.Weight = 1;

                CurrentTransition.Source.Update(animator, false);
                CurrentTransition.Target.Update(animator, true);
            }
            else
            {
                foreach (AnimationTransition transition in Transitions)
                {
                    if (transition.Source != CurrentState)
                        continue;

                    bool allConditionsMet = true;

                    if (transition.Conditions.Count == 0)
                        allConditionsMet = CurrentState.TimeNormalized >= .8f;

                    foreach (AnimationCondition condition in transition.Conditions)
                    {
                        if (condition.AutoReturn)
                        {
                            allConditionsMet = false;

                            if (CurrentState.TimeElapsed > CurrentState.Clip.DurationSeconds * condition.AutoThreshold)
                            {
                                InTransition = true;
                                transitionTime = 0;

                                CurrentTransition = transition;
                                CurrentTransition.Target = returnToState;

                                CurrentState = returnToState;
                                returnToState = null;
                                Update(animator);
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
                        InTransition = true;
                        transitionTime = 0;

                        CurrentTransition = transition;
                        returnToState = CurrentState;

                        CurrentState = transition.Target;
                        CurrentState.SetTimeElapsed(0);
                        Update(animator);
                        return;
                    }
                }

                if (CurrentState != null)
                {
                    CurrentState.Weight = 1;
                    CurrentState.Update(animator);
                }
            }
        }
    }
}
