using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Components;

namespace Freefall.Animation
{
    /// <summary>
    /// Blend Tree using 2D Cartesian Gradient Blending (Freeform Directional).
    /// Blends multiple animations based on two parameters (e.g., movement direction).
    /// All mutable state (weights) lives in AnimationPlayback.
    /// </summary>
    [Serializable]
    public class AnimationBlendTree : AnimationState
    {
        public List<BlendLayer> Layers = new List<BlendLayer>();
        public string ParameterA;
        public string ParameterB;

        /// <summary>
        /// For a tree of clips that take over from a loop (stops in eight directions): when the
        /// tree is entered it plays as it is or left for right (<see cref="AnimationState.Mirror"/>),
        /// whichever continues the stride the character is in. Each way round it looks at the
        /// child that would play and at where that clip takes over in the cycle
        /// (<see cref="AnimationState.Phase"/>, <see cref="AnimationState.MirrorPhase"/>), and
        /// takes the nearer to where the state it comes from stands.
        /// </summary>
        public bool MirrorByPhase;

        public override AnimationClip GetDominantClip() => _cachedDominantClip;
        private AnimationClip _cachedDominantClip;

        /// <summary>
        /// The children share one point of their cycles: the tree is as far along as the child
        /// that carries most weight. This is what lets a tree of one-shots (a roll or a hit
        /// reaction in eight directions) end by exit time like a plain state.
        /// </summary>
        public override float GetTimeNormalized(AnimationPlayback pb)
        {
            float most = 0f;
            AnimationState dominant = null;
            for (int i = 0; i < Layers.Count; i++)
            {
                float w = pb.Get(PK.BlendWeight(ID, i));
                if (w > most)
                {
                    most = w;
                    dominant = Layers[i].Animation;
                }
            }
            return dominant != null ? dominant.GetTimeNormalized(pb) : 0f;
        }

        /// <summary>Entering the tree starts every child over, so a one-shot can play again.</summary>
        internal override void Enter(Animator animator, AnimationPlayback pb, float fromPhase)
        {
            SetTimeElapsed(pb, 0);

            bool turnedRound = false;
            if (MirrorByPhase && fromPhase >= 0f)
            {
                var input = new Vector2(animator.GetParam(ParameterA), animator.GetParam(ParameterB));
                int plain = Heaviest(input);
                int mirrored = Heaviest(new Vector2(-input.X, input.Y));

                float plainAt = plain >= 0 ? pb.PhaseOf(Layers[plain].Animation, false) : -1f;
                float mirroredAt = mirrored >= 0 ? pb.PhaseOf(Layers[mirrored].Animation, true) : -1f;

                // Mirror is the tree's own setting; the choice turns it round
                bool wantMirrored = mirroredAt >= 0f && (plainAt < 0f || Apart(fromPhase, mirroredAt) < Apart(fromPhase, plainAt));
                turnedRound = wantMirrored != Mirror;
            }
            pb.Set(PK.State(ID, PK.Mirrored), turnedRound ? 1 : 0);

            for (int i = 0; i < Layers.Count; i++)
                Layers[i].Animation.Enter(animator, pb, fromPhase);
        }

        internal override bool Plays(AnimationPlayback pb)
        {
            for (int i = 0; i < Layers.Count; i++)
                if (Layers[i].Animation.Plays(pb)) return true;
            return false;
        }

        internal override float GetPhase(AnimationPlayback pb)
        {
            float most = 0f;
            AnimationState dominant = null;
            for (int i = 0; i < Layers.Count; i++)
            {
                float w = pb.Get(PK.BlendWeight(ID, i));
                if (w > most)
                {
                    most = w;
                    dominant = Layers[i].Animation;
                }
            }
            return dominant != null ? dominant.GetPhase(pb) : -1f;
        }

        // How far two points of a cycle are apart, the short way round
        private static float Apart(float a, float b)
        {
            float d = MathF.Abs(a - b);
            d -= MathF.Floor(d);
            return MathF.Min(d, 1f - d);
        }

        // The child that carries most weight for an input
        private int Heaviest(Vector2 input)
        {
            int count = Layers.Count;
            Span<float> weights = count <= 32 ? stackalloc float[count] : new float[count];
            Weigh(input, weights);

            int best = -1;
            float most = 0f;
            for (int i = 0; i < count; i++)
                if (weights[i] > most)
                {
                    most = weights[i];
                    best = i;
                }
            return best;
        }

        public override void Update(Animator animator, AnimationPlayback pb, bool fireEvents = true)
        {
            UpdateWeights(animator, pb);

            float maxWeight = 0;
            BlendLayer maxLayer = null;

            // The cycle runs at the blend of the children's own paces (cycles per second). Taking
            // the dominant clip's pace made it jump where one clip takes over from another.
            float pace = 0;
            float maxPace = 0;

            // Find anim with highest weight
            for (int i = 0; i < Layers.Count; i++)
            {
                float w = pb.Get(PK.BlendWeight(ID, i));
                if (w <= 0) continue;

                var clip = Layers[i].Animation.GetClip(pb);
                if (clip == null) continue;

                // A one-shot taken up part-way has only the rest of its clip to play
                var child = Layers[i].Animation;
                float childPace = child.Speed * clip.DurationSecondsInverse;
                if (!child.Loop) childPace /= MathF.Max(1f - pb.StartOf(child), 0.01f);
                pace += w * childPace;

                if (w > maxWeight)
                {
                    maxWeight = w;
                    maxLayer = Layers[i];
                    maxPace = childPace;
                    _cachedDominantClip = clip;
                }
            }

            if (maxLayer == null) return;

            // Update that anim
            float fromNorm = maxLayer.Animation.GetTimeNormalized(pb);
            // Its events stand for the tree's: the children are at the same point of the cycle
            int wraps = maxLayer.Animation.Advance(animator, pb, fireEvents, maxPace > 0 ? pace / maxPace : 1f, IsMirrored(pb));

            // Synchronize all active anims: the same point of the cycle they share, each counted
            // from where its own clip is taken up
            float maxTimeNorm = maxLayer.Animation.GetTimeNormalized(pb);
            float maxStart = pb.StartOf(maxLayer.Animation);
            bool loops = maxLayer.Animation.Loop;
            foreach (var layer in Layers)
            {
                float w = layer.Animation.GetWeight(pb);
                if (w <= 0) continue;
                if (layer == maxLayer) continue;

                layer.Animation.SetTimeNormalized(pb, Aligned(maxTimeNorm, maxStart, pb.StartOf(layer.Animation), loops));
            }

            if (!fireEvents || wraps < 0) return;

            // Every child covers the same stretch of its own cycle: the tree travels and turns by
            // the weighted sum of what each does over it
            Vector3 travel = Vector3.Zero;
            float turned = 0f;
            if (RootMotion || RootTurn)
            {
                for (int i = 0; i < Layers.Count; i++)
                {
                    float w = pb.Get(PK.BlendWeight(ID, i));
                    if (w <= 0) continue;

                    var child = Layers[i].Animation;
                    var clip = child.GetClip(pb);
                    if (clip == null) continue;

                    // A loop whose cycle is counted from another point starts over at another moment
                    float childStart = pb.StartOf(child);
                    float from = Aligned(fromNorm, maxStart, childStart, loops);
                    float to = Aligned(maxTimeNorm, maxStart, childStart, loops);
                    int childWraps = childStart == maxStart ? wraps
                                   : loops ? (to < from ? 1 : 0) + Math.Max(wraps - 1, 0) : 0;

                    child.RootStep(animator, pb, from * clip.DurationSeconds, to * clip.DurationSeconds, childWraps,
                        RootTurn, out var childTravel, out float childTurned);
                    travel += w * childTravel;
                    turned += w * childTurned;
                }

                if (IsMirrored(pb))
                {
                    travel.X = -travel.X;
                    turned = -turned;
                }
            }
            animator.AddRootMotion(RootMotion ? travel : Vector3.Zero, turned);
        }

        void UpdateWeights(Animator animator, AnimationPlayback pb)
        {
            float valueA = animator.GetParam(ParameterA);
            float valueB = animator.GetParam(ParameterB);

            // Left for right, the clip for going left is the one that plays for going right
            Vector2 input = new Vector2(IsMirrored(pb) ? -valueA : valueA, valueB);

            int count = Layers.Count;

            // Use stackalloc for small counts, fallback for large
            Span<float> normalizedWeights = count <= 32
                ? stackalloc float[count]
                : new float[count];

            Weigh(input, normalizedWeights);

            for (int i = 0; i < count; ++i)
            {
                pb.Set(PK.BlendWeight(ID, i), normalizedWeights[i]);
                Layers[i].Animation.SetWeight(pb, normalizedWeights[i]);
            }
        }

        /// <summary>The share of each child for an input, adding up to 1 (gradient band interpolation).</summary>
        void Weigh(Vector2 input, Span<float> normalizedWeights)
        {
            float total_weight = 0.0f;
            int count = Layers.Count;

            for (int i = 0; i < count; ++i)
            {
                Vector2 point_i = Layers[i].Values;
                Vector2 vec_is = input - point_i;

                float weight = 1.0f;

                for (int j = 0; j < count; ++j)
                {
                    if (j == i)
                        continue;

                    Vector2 point_j = Layers[j].Values;
                    Vector2 vec_ij = point_j - point_i;

                    float lensq_ij = Vector2.Dot(vec_ij, vec_ij);
                    float new_weight = Vector2.Dot(vec_is, vec_ij) / lensq_ij;
                    if (lensq_ij == 0)
                        new_weight = 1;

                    new_weight = Math.Clamp(1.0f - new_weight, 0.0f, 1.0f);
                    weight = Math.Min(weight, new_weight);
                }

                normalizedWeights[i] = weight;
                total_weight += weight;
            }

            float total = total_weight > 0 ? 1f / total_weight : 0;
            for (int i = 0; i < count; ++i)
                normalizedWeights[i] *= total;
        }

        /// <summary>
        /// The children that carry weight, each with its share of the running weighted average:
        /// child i blends over the ones before it by w / (weight so far + w).
        /// </summary>
        internal override void Collect(Bone[] bones, AnimationPlayback pb, StateSamples samples)
        {
            int count = Layers.Count;
            float cumWeight = 0;
            bool mirrored = IsMirrored(pb);

            for (int i = 0; i < count; i++)
            {
                float w = pb.Get(PK.BlendWeight(ID, i));
                if (w <= 0) continue;

                if (!Layers[i].Animation.CollectClip(bones, pb, samples, cumWeight == 0 ? 1f : w / (cumWeight + w), RootTurn, RootLift, mirrored)) continue;

                cumWeight += w;
            }
        }

        public override int GetStateCount()
        {
            return Layers.Count;
        }

        public override AnimationState GetState(int index)
        {
            return Layers[index].Animation;
        }

        [Serializable]
        public class BlendLayer
        {
            public AnimationState Animation = new AnimationState();
            public Vector2 Values;
        }
    }
}
