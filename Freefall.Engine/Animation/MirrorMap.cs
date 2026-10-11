using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Freefall.Animation
{
    /// <summary>
    /// What it takes to play a clip left for right on one skeleton: which bone is each bone's
    /// counterpart on the other side, and the fixed rotations that carry a pose over between the
    /// two sides' own axes (rigs do not mirror their bone axes the same way).
    ///
    /// A mirrored pose is the pose seen in a mirror standing on the character's middle plane
    /// (x = 0 in model space). In model space that is M'(b) = K(b) · S·M(partner)·S, with S the
    /// reflection and K(b) = Bind(b) · (S·Bind(partner)·S)⁻¹, so that the bind pose mirrors to
    /// itself. Per bone, in its parent's space, it comes to
    ///   rotation' = K(parent)⁻¹ · reflected(rotation of the partner) · K(b)
    ///   position' = reflected(position of the partner), turned by K(parent)⁻¹
    /// which needs nothing but the partner's own local pose.
    /// </summary>
    internal sealed class MirrorMap
    {
        public Bone[] Bones;

        /// <summary>
        /// Per bone, the bone to read: the one on the other side, the bone itself in the middle,
        /// -1 for a bone that has no counterpart (it keeps its bind pose).
        /// </summary>
        public int[] Partner;

        /// <summary>
        /// For mirroring the stride alone, under an upper body that holds something and must not
        /// change hands: the hips (-1 when the skeleton has no humanoid hips), the bones that
        /// change sides (hips and legs), and the bones that sit on the hips and carry the rest
        /// (the first of the spine).
        /// </summary>
        public int Hips = -1;
        public bool[] Lower;
        public bool[] OnHips;

        private Quaternion[] _fix;              // K(b)
        private Quaternion[] _parentFixInverse; // K(parent of b)⁻¹

        private static readonly ConditionalWeakTable<Skeleton, MirrorMap> s_maps = new();

        public static MirrorMap Of(Skeleton skeleton) => s_maps.GetValue(skeleton, static s => new MirrorMap(s));

        /// <summary>Turn the local pose read from a bone's partner into the bone's own mirrored pose.</summary>
        public void Apply(int bone, ref BonePose pose)
        {
            var q = pose.Rotation;
            q = new Quaternion(q.X, -q.Y, -q.Z, q.W);
            pose.Rotation = Quaternion.Normalize(_parentFixInverse[bone] * q * _fix[bone]);

            var t = pose.Position;
            t.X = -t.X;
            pose.Position = Vector3.Transform(t, _parentFixInverse[bone]);
        }

        private MirrorMap(Skeleton skeleton)
        {
            var bones = Bones = skeleton.Bones;
            int count = bones.Length;

            Partner = FindPartners(skeleton);

            // Model rotation of each bone in the bind pose (parents come before their children)
            var bind = new Quaternion[count];
            for (int i = 0; i < count; i++)
            {
                var local = bones[i].BindPose.Rotation;
                bind[i] = bones[i].Parent >= 0 ? bind[bones[i].Parent] * local : local;
            }

            static Quaternion Reflected(Quaternion q) => new Quaternion(q.X, -q.Y, -q.Z, q.W);

            _fix = new Quaternion[count];
            _parentFixInverse = new Quaternion[count];
            for (int i = 0; i < count; i++)
            {
                int partner = Partner[i] >= 0 ? Partner[i] : i;
                _fix[i] = Quaternion.Normalize(Quaternion.Inverse(Reflected(bind[partner])) * bind[i]);
            }
            for (int i = 0; i < count; i++)
                _parentFixInverse[i] = bones[i].Parent >= 0 ? Quaternion.Inverse(_fix[bones[i].Parent]) : Quaternion.Identity;

            // Hips and legs
            Lower = new bool[count];
            OnHips = new bool[count];
            var humanoid = HumanoidTemplate.Detect(skeleton);
            if (humanoid != null && humanoid.TryGetValue(HumanoidBone.Hips, out int hips)
                && humanoid.TryGetValue(HumanoidBone.LeftUpperLeg, out int leftLeg)
                && humanoid.TryGetValue(HumanoidBone.RightUpperLeg, out int rightLeg))
            {
                Hips = hips;
                Lower[hips] = true;
                for (int i = 0; i < count; i++)
                {
                    int parent = bones[i].Parent;
                    if (i == leftLeg || i == rightLeg || (parent >= 0 && parent != hips && Lower[parent]))
                        Lower[i] = true;
                }
                for (int i = 0; i < count; i++)
                    OnHips[i] = bones[i].Parent == hips && !Lower[i];
            }
        }

        private static int[] FindPartners(Skeleton skeleton)
        {
            var bones = skeleton.Bones;
            int count = bones.Length;
            var partner = new int[count];
            Array.Fill(partner, -2); // not looked at yet

            // The humanoid bones by what they are
            var humanoid = HumanoidTemplate.Detect(skeleton);
            if (humanoid != null)
            {
                foreach (var entry in humanoid)
                {
                    string name = entry.Key.ToString();
                    string other = name.StartsWith("Left", StringComparison.Ordinal) ? "Right" + name.Substring(4)
                                 : name.StartsWith("Right", StringComparison.Ordinal) ? "Left" + name.Substring(5)
                                 : null;

                    if (other == null) partner[entry.Value] = entry.Value;
                    else if (Enum.TryParse<HumanoidBone>(other, out var otherBone) && humanoid.TryGetValue(otherBone, out int index))
                        partner[entry.Value] = index;
                }
            }

            // The rest (twist bones, clothing, props) by their names
            var byName = new Dictionary<string, int>(count, StringComparer.Ordinal);
            for (int i = 0; i < count; i++)
                if (bones[i].Name != null) byName.TryAdd(bones[i].Name, i);

            for (int i = 0; i < count; i++)
            {
                if (partner[i] != -2) continue;

                partner[i] = i;
                foreach (var other in OtherSide(bones[i].Name))
                {
                    if (!byName.TryGetValue(other, out int index) || index == i) continue;
                    partner[i] = index;
                    break;
                }
            }

            // Only pairs that agree with each other and with the hierarchy can be mirrored bone by
            // bone: a bone whose partner hangs somewhere else keeps its bind pose
            bool changed = true;
            while (changed)
            {
                changed = false;
                for (int i = 0; i < count; i++)
                {
                    int p = partner[i];
                    if (p < 0) continue;

                    int parent = bones[i].Parent, partnerParent = bones[p].Parent;
                    bool fits = partner[p] == i
                             && (parent < 0) == (partnerParent < 0)
                             && (parent < 0 || partner[parent] == partnerParent);
                    if (fits) continue;

                    partner[i] = -1;
                    changed = true;
                }
            }

            return partner;
        }

        // Side markers as rigs write them, each with its counterpart. Tried at the start, at the end
        // and between separators.
        private static readonly (string, string)[] Sides =
        {
            ("Left", "Right"), ("left", "right"), ("LEFT", "RIGHT"),
            ("L", "R"), ("l", "r"),
        };

        private static IEnumerable<string> OtherSide(string name)
        {
            if (string.IsNullOrEmpty(name)) yield break;

            foreach (var (a, b) in Sides)
            {
                foreach (var (from, to) in new[] { (a, b), (b, a) })
                {
                    for (int at = name.IndexOf(from, StringComparison.Ordinal); at >= 0; at = name.IndexOf(from, at + 1, StringComparison.Ordinal))
                    {
                        // A whole word like "Left" may stand anywhere; a single letter only as a
                        // token of its own (".L", "_l_", "L_", " R")
                        if (from.Length == 1)
                        {
                            bool startsToken = at == 0 || !char.IsLetterOrDigit(name[at - 1]);
                            bool endsToken = at + 1 == name.Length || !char.IsLetterOrDigit(name[at + 1]);
                            if (!startsToken || !endsToken) continue;
                        }

                        yield return name.Substring(0, at) + to + name.Substring(at + from.Length);
                    }
                }
            }
        }
    }
}
