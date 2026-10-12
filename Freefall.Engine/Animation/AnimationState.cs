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

        /// <summary>
        /// Where the clip is taken up when the state is entered, as a fraction of its length.
        /// A one-shot skips what comes before (a stop clip that begins with a stretch of plain
        /// walking). A loop begins its cycle there, which lines up clips whose first frames are
        /// on different feet: the clips of a blend tree share their cycle counted from here.
        /// An animation set brings its own value with each clip it replaces.
        /// </summary>
        public float Start;

        /// <summary>
        /// Play the clip left for right: every bone takes the pose of its counterpart on the other
        /// side, turned over, and the travel and turn go the other way. A stop made for one foot
        /// then serves the other. On a blend tree this covers its children, and the tree reads
        /// its sideways parameter the other way round, so that the clip for going left, mirrored,
        /// is the one that plays for going right.
        /// </summary>
        public bool Mirror;

        /// <summary>
        /// For a clip that takes over from a loop (a stop): the point of that loop's cycle, counted
        /// from the loop's own <see cref="Start"/>, at which this clip continues it, when played
        /// as it is and when played mirrored. Negative: not known. A blend tree with
        /// <see cref="AnimationBlendTree.MirrorByPhase"/> picks by these whichever way round is
        /// nearer to where the character is in its stride.
        /// </summary>
        public float Phase = -1f;
        public float MirrorPhase = -1f;

        /// <summary>
        /// The clip's travel over the ground moves the character: it is reported through
        /// <c>Animator.RootMotionDelta</c> for whoever moves it to apply. Off, the state plays in
        /// place and the travel is dropped. Either way the travel is not in the pose.
        /// On a blend tree this covers its children.
        /// </summary>
        public bool RootMotion;

        /// <summary>
        /// The clip turns the character: a clip that ends facing another way (a turn on the spot,
        /// a start that swings round) has the turning of its hips taken out of the pose and
        /// reported through <c>Animator.RootTurnDelta</c> for whoever moves the character to
        /// apply. Off, the body turns inside the character and comes back when the clip is over.
        /// On a blend tree this covers its children.
        /// </summary>
        public bool RootTurn;

        /// <summary>
        /// The clip leaves the ground (a jump, a fall, a landing): how far its lowest point is
        /// above the ground is taken out of the pose, which then keeps that point where the
        /// character stands, and whoever moves the character decides how high it is. Off, a
        /// jump clip rises inside the character on top of the character's own jump. Nothing is
        /// reported for it. On a blend tree this covers its children.
        /// </summary>
        public bool RootLift;

        public List<AnimationCurve> Curves = new List<AnimationCurve>();

        public virtual AnimationClip GetDominantClip() => Clip;

        /// <summary>The clip this state plays on one Animator: its animation set's override, or <see cref="Clip"/>.</summary>
        public AnimationClip GetClip(AnimationPlayback pb) => pb.ClipOf(this);

        // --- Playback accessors (read from blackboard) ---

        public float GetTimeElapsed(AnimationPlayback pb) => pb.Get(PK.State(ID, PK.Time));

        /// <summary>Where the state's clip stands, in seconds.</summary>
        public void SetTimeElapsed(AnimationPlayback pb, float value) => pb.Set(PK.State(ID, PK.Time), value);

        /// <summary>
        /// Start the state over: its clip is taken up at <see cref="Start"/>. A blend tree starts
        /// every child over, so a tree of one-shots can play again.
        /// </summary>
        /// <param name="fromPhase">Where the state that is left stands in its cycle (<see cref="GetPhase"/>), negative when it has none.</param>
        internal virtual void Enter(Animator animator, AnimationPlayback pb, float fromPhase)
        {
            var clip = pb.ClipOf(this);
            SetTimeElapsed(pb, clip != null ? clip.DurationSeconds * Math.Clamp(pb.StartOf(this), 0f, 1f) : 0f);
            pb.Set(PK.State(ID, PK.Entered), 1);
        }

        /// <summary>Whether the state plays left for right on one Animator: its own flag, turned round by a choice made when it was entered.</summary>
        internal bool IsMirrored(AnimationPlayback pb) => Mirror != pb.GetBool(PK.State(ID, PK.Mirrored));

        /// <summary>
        /// Where the state stands in its cycle, 0..1 counted from where its clip is taken up;
        /// negative when it plays nothing. A blend tree answers with its heaviest child.
        /// </summary>
        internal virtual float GetPhase(AnimationPlayback pb)
        {
            var clip = pb.ClipOf(this);
            if (clip == null) return -1f;

            float phase = GetTimeElapsed(pb) * clip.DurationSecondsInverse - pb.StartOf(this);
            return phase - MathF.Floor(phase);
        }

        /// <summary>
        /// The point of a clip (0..1) that goes with a point of another, each counted from its own
        /// start: loops are at the same point of the cycle they share, one-shots as far through
        /// what they have left to play.
        /// </summary>
        internal static float Aligned(float time, float start, float otherStart, bool loop)
        {
            if (start == otherStart) return time;

            if (loop)
            {
                float shifted = time - start + otherStart;
                return shifted - MathF.Floor(shifted);
            }

            float left = 1f - start;
            float done = left > 1e-5f ? Math.Clamp((time - start) / left, 0f, 1f) : 1f;
            return otherStart + done * (1f - otherStart);
        }

        public float GetWeight(AnimationPlayback pb) => pb.Get(PK.State(ID, PK.Weight));

        public void SetWeight(AnimationPlayback pb, float value) => pb.Set(PK.State(ID, PK.Weight), value);

        /// <summary>
        /// How far the state is through its clip, 0 to 1. Transitions without conditions leave at
        /// their exit time by this. A blend tree answers with the child that carries most weight.
        /// </summary>
        public virtual float GetTimeNormalized(AnimationPlayback pb)
        {
            var clip = pb.ClipOf(this);
            if (clip == null) return 0;
            return GetTimeElapsed(pb) * clip.DurationSecondsInverse;
        }

        public void SetTimeNormalized(AnimationPlayback pb, float value)
        {
            var clip = pb.ClipOf(this);
            if (clip == null) return;
            SetTimeElapsed(pb, clip.DurationSeconds * Math.Clamp(value, 0, 1));
        }

        /// <summary>Whether the state has a clip to play on one Animator (a blend tree: any of its children).</summary>
        internal virtual bool Plays(AnimationPlayback pb) => pb.ClipOf(this) != null;

        public virtual int GetStateCount() => 0;
        public virtual AnimationState GetState(int index) => this;

        public virtual void Update(Animator animator, AnimationPlayback pb, bool fireEvents = true)
        {
            float before = GetTimeElapsed(pb);
            int wraps = Advance(animator, pb, fireEvents, 1f, IsMirrored(pb));

            // The frame a transition leaves this state it is moved on once more, without effects
            if (wraps >= 0 && fireEvents)
            {
                Vector3 travel = Vector3.Zero;
                float turned = 0f;
                if (RootMotion || RootTurn)
                {
                    RootStep(animator, pb, before, GetTimeElapsed(pb), wraps, RootTurn, out travel, out turned);
                    if (IsMirrored(pb))
                    {
                        travel.X = -travel.X;
                        turned = -turned;
                    }
                }

                animator.AddRootMotion(RootMotion ? travel : Vector3.Zero, turned);
            }
        }

        /// <summary>
        /// The travel over the ground and the turn of this state's clip between two of its times,
        /// in the model space of the skeleton the Animator takes root motion from. With
        /// <paramref name="turning"/> the travel is in the heading the character has by then.
        /// </summary>
        internal void RootStep(Animator animator, AnimationPlayback pb, float fromSeconds, float toSeconds, int wraps,
            bool turning, out Vector3 travel, out float turned)
        {
            travel = Vector3.Zero;
            turned = 0f;

            var clip = pb.ClipOf(this);
            if (clip == null || !animator.TryGetRootSource(out var bones, out int rootBone)) return;

            clip.GetRootMotion(bones, rootBone)
                .Step(clip.TicksAt(fromSeconds), clip.TicksAt(toSeconds), wraps, turning, out travel, out turned);
        }

        /// <summary>
        /// Advance this state's clip. <paramref name="rate"/> scales its own speed: a blend tree
        /// plays its children at the blended pace instead of each at its own. Returns how often
        /// the clip started over, or -1 when the state plays no clip.
        /// </summary>
        /// <param name="mirrored">Whether the clip is played left for right: the events it passes are reported that way round.</param>
        internal int Advance(Animator animator, AnimationPlayback pb, bool fireEvents, float rate, bool mirrored)
        {
            var clip = pb.ClipOf(this);
            if (clip == null) return -1;

            float before = GetTimeElapsed(pb);
            float timeElapsed = before;
            float step = Base.Time.SmoothDelta * Speed * rate;
            int wraps = 0;

            if (Loop)
            {
                timeElapsed += step;

                if (timeElapsed > clip.DurationSeconds)
                {
                    wraps = (int)(timeElapsed * clip.DurationSecondsInverse);
                    timeElapsed %= clip.DurationSeconds;
                }
            }
            else if (timeElapsed < clip.DurationSeconds)
            {
                timeElapsed = MathF.Min(timeElapsed + step, clip.DurationSeconds);
            }

            SetTimeElapsed(pb, timeElapsed);

            if (fireEvents)
            {
                long entered = PK.State(ID, PK.Entered);
                bool first = pb.GetBool(entered);
                if (first) pb.Set(entered, 0);

                PassEvents(clip, animator, before * clip.DurationSecondsInverse, timeElapsed * clip.DurationSecondsInverse,
                    wraps > 0, first, mirrored);
            }

            foreach (var curve in Curves)
            {
                float value = curve.Evaluate(GetTimeNormalized(pb));
                animator.SetParam(curve.TargetParameter, value);
            }

            return wraps;
        }

        /// <summary>
        /// Report what lies on the stretch of the clip that this update played: the events marked
        /// on it and, for an Animator that wants them, the feet coming down. Going by the stretch
        /// and not by a mark per event means that each is reported once however the state was
        /// come by: taken up part-way, played to its end, left and entered again, or made the
        /// heaviest child of a blend tree in the middle of its cycle.
        /// </summary>
        private void PassEvents(AnimationClip clip, Animator animator, float from, float to, bool wrapped, bool first, bool mirrored)
        {
            var events = clip.Events;
            if (events != null)
            {
                for (int i = 0; i < events.Count; i++)
                {
                    var evt = events[i];
                    if (evt == null || !Passed(evt.Time, from, to, wrapped, first)) continue;

                    var bone = evt.Bone;
                    if (mirrored && bone.HasValue) bone = HumanoidSides.Other(bone.Value);
                    animator.ReportEvent(this, clip, evt.Name, bone, 1f);
                }
            }

            if (animator.FootfallEvents && animator.TryGetClipSkeleton(out var skeleton))
            {
                var footfalls = clip.GetFootfalls(skeleton);
                for (int i = 0; i < footfalls.Length; i++)
                {
                    ref readonly var fall = ref footfalls[i];
                    if (!Passed(fall.Time, from, to, wrapped, first)) continue;

                    bool left = fall.Left != mirrored;
                    animator.ReportEvent(this, clip, AnimationEvent.Footfall, left ? HumanoidBone.LeftFoot : HumanoidBone.RightFoot, fall.Strength);
                }
            }
        }

        // Whether a point of the clip lies on the stretch played: after where it stood, up to where
        // it stands now. The point a state is taken up at counts on its first update; a loop that
        // started over has played to its end and on from its beginning.
        private static bool Passed(float time, float from, float to, bool wrapped, bool first)
        {
            if (wrapped) return time > from || time <= to;
            return time <= to && (time > from || (first && time >= from));
        }

        /// <summary>
        /// Adds this state's clip to the samples of a pose on <paramref name="bones"/>, at the time
        /// the state has reached. While the Animator's animation set is being swapped, the sample
        /// also carries the clip the old set had here, taken at the same point of its cycle.
        /// False when the state plays no clip.
        /// </summary>
        internal bool CollectClip(Bone[] bones, AnimationPlayback pb, StateSamples samples, float share, bool turning, bool lifting, bool mirrored)
        {
            var clip = pb.ClipOf(this);
            if (clip == null) return false;

            float time = GetTimeElapsed(pb);

            ref var sample = ref samples.Add();
            sample.Mirror = mirrored && samples.Skeleton != null ? MirrorMap.Of(samples.Skeleton) : null;
            sample.MirrorLegsOnly = mirrored && pb.AnimationSet != null && pb.AnimationSet.MirrorLegsOnly;
            sample.Channels = clip.ChannelsFor(bones, samples.RootBone, turning, lifting);
            sample.Ticks = clip.TicksAt(time);
            sample.Share = share;
            sample.PreviousChannels = null;

            if (pb.SetFade < 1f)
            {
                var previous = pb.PreviousClipOf(this);
                if (previous != null && previous != clip)
                {
                    float point = Aligned(time * clip.DurationSecondsInverse, pb.StartOf(this), pb.PreviousStartOf(this), Loop);
                    sample.PreviousChannels = previous.ChannelsFor(bones, samples.RootBone, turning, lifting);
                    sample.PreviousTicks = previous.TicksAt(point * previous.DurationSeconds);
                    sample.SetFade = pb.SetFade;
                }
            }

            return true;
        }

        /// <summary>
        /// Adds what this state plays to the samples of a pose: its clip, or for a blend tree its
        /// weighted children. Adds nothing when it plays nothing. Called once per pose; the bones
        /// then read <see cref="StateSamples.Sample"/>.
        /// </summary>
        internal virtual void Collect(Bone[] bones, AnimationPlayback pb, StateSamples samples)
            => CollectClip(bones, pb, samples, 1f, RootTurn, RootLift, IsMirrored(pb));
    }
}
