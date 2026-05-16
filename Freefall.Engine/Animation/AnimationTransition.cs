using System;
using System.Collections.Generic;
using System.ComponentModel;
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
    [Serializable]
    public class AnimationTransition
    {
        public float Duration = .2f;
        public List<AnimationCondition> Conditions = new List<AnimationCondition>();

        /// <summary>Serialized state IDs — resolved to Source/Target by RebuildAfterLoad.</summary>
        [Browsable(false)]
        public int SourceID;
        [Browsable(false)]
        public int TargetID;

        /// <summary>Runtime references — not serialized.</summary>
        [Browsable(false)]
        [NonSerialized] public AnimationState Source;
        [Browsable(false)]
        [NonSerialized] public AnimationState Target;
    }

    /// <summary>
    /// A condition that must be met for a transition to occur.
    /// </summary>
    [Serializable]
    public class AnimationCondition
    {
        public bool AutoReturn;
        public float AutoThreshold = 0.89f;

        public ComparisonType Comparison;
        public string Parameter;
        public float Value;

        public bool IsMet(Animator animator)
        {
            float param = animator.GetParam(Parameter);

            if (Comparison == ComparisonType.Smaller)
                return param < Value;
            else if (Comparison == ComparisonType.Greater)
                return param > Value;
            else
                return MathF.Abs(param - Value) < 0.001f;
        }
    }
}
