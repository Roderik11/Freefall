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
    /// How a transition carries the pose the layer showed when it started over into the new state.
    /// Either way only the new state is played: the layer keeps the difference per bone and lets
    /// it go over the transition's duration.
    /// </summary>
    public enum TransitionBlend
    {
        /// <summary>
        /// The difference dies away with the motion it had (inertialization): a bone that was
        /// moving at the switch keeps moving at the same speed and comes to rest on the new state
        /// by the end, without going past it. A stride into an attack or a swing into a hit keeps
        /// its momentum through the switch.
        /// </summary>
        Inertial,

        /// <summary>
        /// The difference fades out on an eased curve that starts still: at the switch every bone
        /// takes up the pace of the new state at once.
        /// </summary>
        Fade,
    }

    /// <summary>
    /// Defines a transition between two animation states with conditions.
    /// </summary>
    [Serializable]
    public class AnimationTransition
    {
        public float Duration = .2f;

        /// <summary>How the pose shown at the start goes over into the new state (<see cref="TransitionBlend"/>).</summary>
        public TransitionBlend Blend;

        /// <summary>
        /// For a transition without conditions: the point of the source clip (0..1) at which it starts.
        /// </summary>
        public float ExitTime = .8f;

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
