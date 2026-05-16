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

        public override AnimationClip GetDominantClip() => _cachedDominantClip;
        private AnimationClip _cachedDominantClip;

        public override void Update(Animator animator, AnimationPlayback pb, bool fireEvents = true)
        {
            UpdateWeights(animator, pb);

            float maxWeight = 0;
            BlendLayer maxLayer = null;

            // Find anim with highest weight
            for (int i = 0; i < Layers.Count; i++)
            {
                float w = pb.Get(PK.BlendWeight(ID, i));
                if (w > maxWeight)
                {
                    maxWeight = w;
                    maxLayer = Layers[i];
                    _cachedDominantClip = maxLayer.Animation.Clip;
                }
            }

            if (maxLayer == null) return;

            // Update that anim
            maxLayer.Animation.Update(animator, pb, fireEvents);

            // Synchronize all active anims
            float maxTimeNorm = maxLayer.Animation.GetTimeNormalized(pb);
            foreach (var layer in Layers)
            {
                float w = layer.Animation.GetWeight(pb);
                if (w <= 0) continue;
                if (layer == maxLayer) continue;

                layer.Animation.SetTimeNormalized(pb, maxTimeNorm);
            }
        }

        void UpdateWeights(Animator animator, AnimationPlayback pb)
        {
            float valueA = animator.GetParam(ParameterA);
            float valueB = animator.GetParam(ParameterB);
            Vector2 input = new Vector2(valueA, valueB);

            float total_weight = 0.0f;
            int count = Layers.Count;

            // Use stackalloc for small counts, fallback for large
            Span<float> normalizedWeights = count <= 32
                ? stackalloc float[count]
                : new float[count];

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

            // Normalize and store in playback
            float total = total_weight > 0 ? 1f / total_weight : 0;

            for (int i = 0; i < count; ++i)
            {
                float w = normalizedWeights[i] * total;
                pb.Set(PK.BlendWeight(ID, i), w);
                Layers[i].Animation.SetWeight(pb, w);
            }
        }

        /// <summary>
        /// Weighted average blend. Reads per-child weights from playback.
        /// </summary>
        internal override void BlendBone(Bone bone, ref BonePose blendPose, ref BonePose temp, AnimationPlayback pb)
        {
            float treeWeight = GetWeight(pb);
            if (treeWeight <= 0) return;

            int count = Layers.Count;
            BonePose treePose = default;
            float cumWeight = 0;

            for (int i = 0; i < count; i++)
            {
                float w = pb.Get(PK.BlendWeight(ID, i));
                if (w <= 0) continue;

                var child = Layers[i].Animation;
                if (child.Clip == null) continue;

                child.Clip.GetBonePose(bone, child.GetTimeElapsed(pb), ref temp);

                if (cumWeight == 0)
                {
                    treePose = temp;
                }
                else
                {
                    float factor = w / (cumWeight + w);
                    treePose.Position = Vector3.Lerp(treePose.Position, temp.Position, factor);
                    treePose.Scale = Vector3.Lerp(treePose.Scale, temp.Scale, factor);
                    treePose.Rotation = Quaternion.Slerp(treePose.Rotation, temp.Rotation, factor);
                }

                cumWeight += w;
            }

            if (cumWeight <= 0) return;

            blendPose.Position = Vector3.Lerp(blendPose.Position, treePose.Position, treeWeight);
            blendPose.Scale = Vector3.Lerp(blendPose.Scale, treePose.Scale, treeWeight);
            blendPose.Rotation = Quaternion.Slerp(blendPose.Rotation, treePose.Rotation, treeWeight);
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
