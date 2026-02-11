using System;
using System.Collections.Generic;
using Freefall.Components;

namespace Freefall.Animation
{
    public enum ComparisonType
    {
        Equals, Greater, Smaller
    }

    /// <summary>
    /// Defines a transition between two animation states with conditions.
    /// </summary>
    public class AnimationTransition
    {
        public List<AnimationCondition> Conditions = new List<AnimationCondition>();
        public float Duration = .2f;
        public AnimationState Source;
        public AnimationState Target;
    }

    /// <summary>
    /// A condition that must be met for a transition to occur.
    /// </summary>
    public class AnimationCondition
    {
        public bool AutoReturn;
        public float AutoThreshold = 0.89f;

        public ComparisonType Comparison;
        public string Parameter;
        public float Value;

        public bool IsMet(Animator animator)
        {
            if (Comparison == ComparisonType.Smaller)
                return animator.GetParam(Parameter) < Value;
            else if (Comparison == ComparisonType.Greater)
                return animator.GetParam(Parameter) > Value;
            else
                return animator.GetParam(Parameter) == Value;
        }
    }
}
