using System;
using System.Numerics;

namespace Freefall.Animation
{
    public enum RootMotionKind
    {
        /// <summary>The clip stays where it is: nothing is taken out of it.</summary>
        None,

        /// <summary>
        /// The clip travels at an even pace (walk, run): that pace is the root motion. What the
        /// hips do around it, the surge of each step, stays in the pose, so feet stay planted on
        /// a character that moves at the pace.
        /// </summary>
        Steady,

        /// <summary>
        /// The clip moves in bursts (a lunge, a knock-back, a dodge): the whole path of the hips
        /// over the ground is the root motion, and the pose keeps them where the clip starts.
        /// </summary>
        Path,
    }

    /// <summary>
    /// The travel over the ground that a clip carries on its hips, worked out for one skeleton.
    /// It is taken out of the pose, so every clip plays in place, and handed to whoever moves
    /// the character through <c>Animator.RootMotionDelta</c> for the states that ask for it
    /// (<see cref="AnimationState.RootMotion"/>). Height is never part of it: a jump or a
    /// crouch stays in the pose.
    /// </summary>
    public sealed class RootMotionTrack
    {
        internal static readonly RootMotionTrack Nothing = new RootMotionTrack();

        public RootMotionKind Kind { get; internal set; }

        /// <summary>Travel over the whole clip, in the model space of the skeleton (Y is 0).</summary>
        public Vector3 Net { get; internal set; }

        /// <summary>Length of the clip in seconds.</summary>
        public float Duration { get; internal set; }

        /// <summary>Average pace over the ground, in model units per second.</summary>
        public float Speed => Duration > 0f ? Net.Length() / Duration : 0f;

        /// <summary>
        /// How far the clip turns the character about the vertical from its first key to its last,
        /// in radians (positive as a positive yaw); 0 for a clip that ends facing the way it
        /// starts. Only states with <see cref="AnimationState.RootTurn"/> take it out of the
        /// pose and report it; for them the character turns as the hips do, moment by moment,
        /// so a state can be left at any point of the turn without the body swinging back.
        /// </summary>
        public float Turn { get; internal set; }

        // Key times the travel runs between
        internal float StartTicks, EndTicks;

        // Path: travel since the first key, at each key of the hips
        internal VectorKeys Path;

        // The hips' heading since the first key, at each of their rotation keys (in X); null
        // for a clip that does not turn
        internal VectorKeys Heading;

        /// <summary>How far a turning state has turned the character by a key time.</summary>
        internal float TurnAt(float ticks) => Heading != null ? Heading.GetValue(ticks).X : 0f;

        /// <summary>
        /// Travel and turn between two key times of a clip that started over <paramref name="wraps"/>
        /// times in between. With <paramref name="turning"/> the travel is in the heading the
        /// character has at the first of the two times, which the turn so far has moved on.
        /// </summary>
        internal void Step(float fromTicks, float toTicks, int wraps, bool turning, out Vector3 travel, out float turned)
        {
            if (!turning || Turn == 0f)
            {
                travel = Between(fromTicks, toTicks, wraps);
                turned = 0f;
                return;
            }

            float from = TurnAt(fromTicks), to = TurnAt(toTicks);
            if (wraps <= 0)
            {
                travel = Turned(At(toTicks) - At(fromTicks), -from);
                turned = to - from;
                return;
            }

            // On to the end, then from the start again in the heading the clip ended with
            travel = Turned(At(EndTicks) - At(fromTicks), -from)
                   + Turned(At(toTicks) - At(StartTicks), Turn - from);
            turned = Turn - from + to + Turn * (wraps - 1);
        }

        internal static Vector3 Turned(Vector3 vector, float radians)
            => radians == 0f ? vector : Vector3.Transform(vector, Quaternion.CreateFromAxisAngle(Vector3.UnitY, radians));

        /// <summary>Travel from the start of the clip up to a key time.</summary>
        internal Vector3 At(float ticks)
        {
            switch (Kind)
            {
                case RootMotionKind.Path:
                    return Path.GetValue(ticks);

                case RootMotionKind.Steady:
                    float span = EndTicks - StartTicks;
                    return span > 0f ? Net * Math.Clamp((ticks - StartTicks) / span, 0f, 1f) : Vector3.Zero;

                default:
                    return Vector3.Zero;
            }
        }

        /// <summary>Travel between two key times of a clip that started over <paramref name="wraps"/> times in between.</summary>
        internal Vector3 Between(float fromTicks, float toTicks, int wraps)
        {
            if (Kind == RootMotionKind.None) return Vector3.Zero;
            return At(toTicks) - At(fromTicks) + Net * wraps;
        }
    }
}
