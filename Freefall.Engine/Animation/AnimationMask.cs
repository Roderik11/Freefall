using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace Freefall.Animation
{
    /// <summary>
    /// Defines which bones an animation layer affects, and how strongly.
    /// Used for partial body animations (e.g., an upper-body attack over a running lower body).
    /// Body parts are named by humanoid bone, so one mask works on every humanoid rig.
    /// </summary>
    public class AnimationMask : Assets.Asset, Assets.IRebuildAfterLoad
    {
        /// <summary>
        /// Body parts: a humanoid bone with everything below it. A later branch overrides an earlier
        /// one where they overlap, so a branch with Weight 0 cuts a part back out (upper body without
        /// the head: Spine, then Neck at 0).
        /// </summary>
        public List<Branch> Branches = new List<Branch>();

        /// <summary>
        /// Extra bones by name, exactly these at full weight. For bones outside the humanoid set and
        /// for rigs that are not humanoid; the names are those of the rig the clips play on.
        /// </summary>
        public List<string> Bones = new List<string>();

        [Serializable]
        public class Branch
        {
            public HumanoidBone Bone = HumanoidBone.Spine;

            public float Weight = 1f;

            /// <summary>
            /// Humanoid bones the weight ramps up over, from the branch bone down, so the layer
            /// eases in along the chain instead of starting at one joint. 0 = full weight at once.
            /// 2 on Spine: spine 1/3, chest 2/3, upper chest and everything beyond 1.
            /// </summary>
            public int BlendDepth;
        }

        /// <summary>A mask resolved for one skeleton, indexed by bone.</summary>
        public sealed class BoneWeights
        {
            /// <summary>How much of the layer each bone takes, 0..1.</summary>
            public float[] Weight;

            /// <summary>Masked bones and their ancestors: what a layer needs to know its own model-space pose.</summary>
            public bool[] Chain;

            internal int Signature;
        }

        // Animators update in parallel and share the mask
        private readonly ConcurrentDictionary<Skeleton, BoneWeights> _resolved = new ConcurrentDictionary<Skeleton, BoneWeights>();

        /// <summary>The mask's weights for the bones of one skeleton, cached until the mask changes.</summary>
        public BoneWeights GetWeights(Skeleton skeleton)
        {
            int signature = GetSignature();
            if (_resolved.TryGetValue(skeleton, out var weights) && weights.Signature == signature)
                return weights;

            weights = Resolve(skeleton);
            weights.Signature = signature;
            _resolved[skeleton] = weights;
            return weights;
        }

        /// <summary>Drops the resolved weights. Edits to branches and to the number of bones are noticed without it.</summary>
        public void InvalidateCache() => _resolved.Clear();

        public void RebuildAfterLoad() => _resolved.Clear();

        // Cheap enough to check on every use, so edits in the inspector show at once
        private int GetSignature()
        {
            var hash = new HashCode();
            hash.Add(Branches.Count);
            foreach (var branch in Branches)
            {
                if (branch == null) continue;
                hash.Add((int)branch.Bone);
                hash.Add(branch.Weight);
                hash.Add(branch.BlendDepth);
            }

            hash.Add(Bones.Count);
            foreach (var name in Bones)
                hash.Add(name?.Length ?? 0);

            return hash.ToHashCode();
        }

        private BoneWeights Resolve(Skeleton skeleton)
        {
            var bones = skeleton.Bones;
            int count = bones.Length;
            var weight = new float[count];

            if (Bones.Count > 0)
            {
                var names = new HashSet<string>(Bones);
                for (int i = 0; i < count; i++)
                    if (names.Contains(bones[i].Name))
                        weight[i] = 1f;
            }

            var humanoid = Branches.Count > 0 ? HumanoidTemplate.Detect(skeleton) : null;
            if (humanoid != null)
            {
                var isHumanoid = new bool[count];
                foreach (var index in humanoid.Values)
                    isHumanoid[index] = true;

                // Humanoid bones from the branch bone down to each bone; 0 = outside the branch.
                // Bones in between (twist bones, importer pivot nodes) count with the one above them.
                var depth = new int[count];

                foreach (var branch in Branches)
                {
                    if (branch == null || !humanoid.TryGetValue(branch.Bone, out int root)) continue;

                    Array.Clear(depth);
                    depth[root] = 1;
                    for (int i = root + 1; i < count; i++) // parents come before their children
                    {
                        int parent = bones[i].Parent;
                        if (parent >= 0 && depth[parent] > 0)
                            depth[i] = depth[parent] + (isHumanoid[i] ? 1 : 0);
                    }

                    float full = Math.Max(0, branch.BlendDepth) + 1;
                    for (int i = root; i < count; i++)
                        if (depth[i] > 0)
                            weight[i] = Math.Clamp(branch.Weight, 0f, 1f) * Math.Min(1f, depth[i] / full);
                }
            }

            var chain = new bool[count];
            for (int i = count - 1; i >= 0; i--)
            {
                if (weight[i] > 0f) chain[i] = true;
                if (chain[i] && bones[i].Parent >= 0) chain[bones[i].Parent] = true;
            }

            return new BoneWeights { Weight = weight, Chain = chain };
        }
    }
}
