using System;
using System.Collections.Generic;

namespace Freefall.Animation
{
    /// <summary>
    /// Canonical humanoid bone definitions. Used only as an intermediary
    /// during auto-mapping — the output is always a generic BoneMap.
    /// </summary>
    public enum HumanoidBone
    {
        // Spine
        Hips, Spine, Chest, UpperChest, Neck, Head,

        // Left Arm
        LeftShoulder, LeftUpperArm, LeftLowerArm, LeftHand,

        // Right Arm
        RightShoulder, RightUpperArm, RightLowerArm, RightHand,

        // Left Leg
        LeftUpperLeg, LeftLowerLeg, LeftFoot, LeftToes,

        // Right Leg
        RightUpperLeg, RightLowerLeg, RightFoot, RightToes,

        // Left Hand Fingers
        LeftThumbProximal, LeftThumbIntermediate, LeftThumbDistal,
        LeftIndexProximal, LeftIndexIntermediate, LeftIndexDistal,
        LeftMiddleProximal, LeftMiddleIntermediate, LeftMiddleDistal,
        LeftRingProximal, LeftRingIntermediate, LeftRingDistal,
        LeftLittleProximal, LeftLittleIntermediate, LeftLittleDistal,

        // Right Hand Fingers
        RightThumbProximal, RightThumbIntermediate, RightThumbDistal,
        RightIndexProximal, RightIndexIntermediate, RightIndexDistal,
        RightMiddleProximal, RightMiddleIntermediate, RightMiddleDistal,
        RightRingProximal, RightRingIntermediate, RightRingDistal,
        RightLittleProximal, RightLittleIntermediate, RightLittleDistal,

        // Eyes/Jaw
        LeftEye, RightEye, Jaw,

        Count
    }

    /// <summary>
    /// Produces a BoneMap between two humanoid skeletons using canonical bone
    /// definitions and pattern-based name matching.
    /// </summary>
    public static class HumanoidTemplate
    {
        /// <summary>Minimum required bone matches to consider a skeleton humanoid.</summary>
        private const int MinRequiredBones = 15;

        private enum BoneSide { Center, Left, Right }

        /// <summary>
        /// Try to create a BoneMap between two skeletons using humanoid auto-detection.
        /// Returns null if either skeleton doesn't have enough humanoid bone matches.
        /// </summary>
        public static BoneMap TryMap(Skeleton source, Skeleton target)
        {
            var srcMapping = Detect(source);
            var dstMapping = Detect(target);

            if (srcMapping == null || dstMapping == null)
                return null;

            // Build pairs for bones mapped in both skeletons
            var pairs = new List<(int src, int dst)>();
            int srcRoot = 0, dstRoot = 0;

            foreach (var (bone, srcIdx) in srcMapping)
            {
                if (dstMapping.TryGetValue(bone, out int dstIdx))
                {
                    pairs.Add((srcIdx, dstIdx));

                    if (bone == HumanoidBone.Hips)
                    {
                        srcRoot = srcIdx;
                        dstRoot = dstIdx;
                    }
                }
            }

            if (pairs.Count < MinRequiredBones)
                return null;

            // Twist hints: hands and fingers roll about axes that are nearly
            // collinear with their parent bones, so the generic parent-direction
            // twist constraint is noise there. Use the palm axis (index knuckle →
            // pinky knuckle) as the anatomical roll reference instead.
            var twistHints = new List<BoneMap.TwistHint>();
            AddHandTwistHints(twistHints, srcMapping, dstMapping, left: true);
            AddHandTwistHints(twistHints, srcMapping, dstMapping, left: false);

            // Log the mapping
            var log = new System.Text.StringBuilder();
            log.Append($"[HumanoidTemplate] Mapping {source.Name} → {target.Name}: {pairs.Count} paired bones");
            foreach (var (bone, srcIdx) in srcMapping)
            {
                if (dstMapping.TryGetValue(bone, out int dstIdx))
                    log.Append($"\n  {bone}: [{srcIdx}] {source.BoneNames[srcIdx]} → [{dstIdx}] {target.BoneNames[dstIdx]}");
                else
                    log.Append($"\n  {bone}: [{srcIdx}] {source.BoneNames[srcIdx]} → (no target)");
            }
            foreach (var (bone, dstIdx) in dstMapping)
            {
                if (!srcMapping.ContainsKey(bone))
                    log.Append($"\n  {bone}: (no source) → [{dstIdx}] {target.BoneNames[dstIdx]}");
            }
            Freefall.Debug.Log(log.ToString());

            return BoneMap.Create(source, target, pairs.ToArray(), srcRoot, dstRoot,
                twistHints.ToArray());
        }

        /// <summary>Hand and finger bones per side, in HumanoidBone order.</summary>
        private static readonly HumanoidBone[] LeftHandBones =
        {
            HumanoidBone.LeftHand,
            HumanoidBone.LeftThumbProximal, HumanoidBone.LeftThumbIntermediate, HumanoidBone.LeftThumbDistal,
            HumanoidBone.LeftIndexProximal, HumanoidBone.LeftIndexIntermediate, HumanoidBone.LeftIndexDistal,
            HumanoidBone.LeftMiddleProximal, HumanoidBone.LeftMiddleIntermediate, HumanoidBone.LeftMiddleDistal,
            HumanoidBone.LeftRingProximal, HumanoidBone.LeftRingIntermediate, HumanoidBone.LeftRingDistal,
            HumanoidBone.LeftLittleProximal, HumanoidBone.LeftLittleIntermediate, HumanoidBone.LeftLittleDistal,
        };

        private static readonly HumanoidBone[] RightHandBones =
        {
            HumanoidBone.RightHand,
            HumanoidBone.RightThumbProximal, HumanoidBone.RightThumbIntermediate, HumanoidBone.RightThumbDistal,
            HumanoidBone.RightIndexProximal, HumanoidBone.RightIndexIntermediate, HumanoidBone.RightIndexDistal,
            HumanoidBone.RightMiddleProximal, HumanoidBone.RightMiddleIntermediate, HumanoidBone.RightMiddleDistal,
            HumanoidBone.RightRingProximal, HumanoidBone.RightRingIntermediate, HumanoidBone.RightRingDistal,
            HumanoidBone.RightLittleProximal, HumanoidBone.RightLittleIntermediate, HumanoidBone.RightLittleDistal,
        };

        /// <summary>
        /// Add palm-axis twist hints (index knuckle → pinky knuckle) for the hand
        /// and all finger bones of one side, when both rigs have the knuckles mapped.
        /// </summary>
        private static void AddHandTwistHints(List<BoneMap.TwistHint> hints,
            Dictionary<HumanoidBone, int> srcMapping, Dictionary<HumanoidBone, int> dstMapping, bool left)
        {
            var indexBone = left ? HumanoidBone.LeftIndexProximal : HumanoidBone.RightIndexProximal;
            var littleBone = left ? HumanoidBone.LeftLittleProximal : HumanoidBone.RightLittleProximal;

            if (!srcMapping.TryGetValue(indexBone, out int srcFrom)) return;
            if (!srcMapping.TryGetValue(littleBone, out int srcTo)) return;
            if (!dstMapping.TryGetValue(indexBone, out int dstFrom)) return;
            if (!dstMapping.TryGetValue(littleBone, out int dstTo)) return;

            foreach (var bone in left ? LeftHandBones : RightHandBones)
            {
                if (srcMapping.ContainsKey(bone) && dstMapping.TryGetValue(bone, out int targetIdx))
                    hints.Add(new BoneMap.TwistHint
                    {
                        TargetBone = targetIdx,
                        SrcFrom = srcFrom, SrcTo = srcTo,
                        DstFrom = dstFrom, DstTo = dstTo,
                    });
            }
        }

        /// <summary>
        /// Detect which bones in a skeleton match humanoid naming conventions.
        /// Returns a mapping of HumanoidBone → skeleton bone index, or null if
        /// not enough bones match.
        /// </summary>
        public static Dictionary<HumanoidBone, int> Detect(Skeleton skeleton)
        {
            var result = new Dictionary<HumanoidBone, int>();

            for (int i = 0; i < skeleton.BoneNames.Length; i++)
            {
                string name = skeleton.BoneNames[i];
                if (string.IsNullOrEmpty(name)) continue;

                var (coreName, side) = NormalizeBoneName(name);
                if (string.IsNullOrEmpty(coreName)) continue;

                // Try center bones (hips, spine, etc.)
                foreach (var (humanBone, aliases) in CenterBoneAliases)
                {
                    if (result.ContainsKey(humanBone)) continue;
                    foreach (string alias in aliases)
                    {
                        if (string.Equals(coreName, alias, StringComparison.OrdinalIgnoreCase))
                        {
                            result[humanBone] = i;
                            goto nextBone;
                        }
                    }
                }

                // Try sided bones (arms, legs, fingers, etc.)
                if (side != BoneSide.Center)
                {
                    foreach (var (leftBone, rightBone, aliases) in SidedBoneAliases)
                    {
                        var humanBone = side == BoneSide.Left ? leftBone : rightBone;
                        if (result.ContainsKey(humanBone)) continue;
                        foreach (string alias in aliases)
                        {
                            if (string.Equals(coreName, alias, StringComparison.OrdinalIgnoreCase))
                            {
                                result[humanBone] = i;
                                goto nextBone;
                            }
                        }
                    }
                }

                nextBone:;
            }

            if (result.Count < MinRequiredBones)
            {
                var unmatched = new System.Text.StringBuilder();
                for (int i = 0; i < skeleton.BoneNames.Length; i++)
                {
                    string name = skeleton.BoneNames[i];
                    if (string.IsNullOrEmpty(name)) continue;
                    bool matched = false;
                    foreach (var kv in result)
                    {
                        if (kv.Value == i) { matched = true; break; }
                    }
                    if (!matched)
                    {
                        var (norm, side) = NormalizeBoneName(name);
                        unmatched.Append($"\n  [{i}] \"{name}\" → \"{norm}\" ({side})");
                    }
                }
                Freefall.Debug.Log($"[HumanoidTemplate] Skeleton '{skeleton.Name}': only {result.Count}/{MinRequiredBones} humanoid bones matched. Unmatched:{unmatched}");
            }

            return result.Count >= MinRequiredBones ? result : null;
        }

        /// <summary>
        /// Normalize a bone name by stripping rig prefixes and detecting left/right side.
        /// Uses pattern matching instead of hardcoded prefix lists.
        /// </summary>
        private static (string name, BoneSide side) NormalizeBoneName(string raw)
        {
            string name = raw;

            // Strip prefixes delimited by ':' or '|' (handles mixamorig:, Armature|, etc.)
            int sepIdx = name.LastIndexOfAny(PrefixSeparators);
            if (sepIdx >= 0 && sepIdx < name.Length - 1)
                name = name.Substring(sepIdx + 1);

            // Strip short prefix before '-' (handles B-hips, CC-spine, etc.)
            int dashIdx = name.IndexOf('-');
            if (dashIdx > 0 && dashIdx <= 3 && dashIdx < name.Length - 1)
                name = name.Substring(dashIdx + 1);

            // Strip Bip0X_ prefixes (3ds Max Biped)
            if (name.StartsWith("Bip", StringComparison.OrdinalIgnoreCase))
            {
                int bipEnd = 3;
                while (bipEnd < name.Length && char.IsDigit(name[bipEnd])) bipEnd++;
                if (bipEnd < name.Length && (name[bipEnd] == '_' || name[bipEnd] == ' '))
                    name = name.Substring(bipEnd + 1);
            }

            // Detect and strip side indicators
            BoneSide side = BoneSide.Center;

            // Suffix patterns: .L, .R, _L, _R (case-insensitive)
            if (name.Length > 2 && name[^2] is '.' or '_')
            {
                char last = char.ToUpperInvariant(name[^1]);
                if (last == 'L') { side = BoneSide.Left; name = name[..^2]; }
                else if (last == 'R') { side = BoneSide.Right; name = name[..^2]; }
            }

            // Suffix patterns: _Left, _Right
            if (side == BoneSide.Center)
            {
                if (name.EndsWith("_Left", StringComparison.OrdinalIgnoreCase))
                { side = BoneSide.Left; name = name[..^5]; }
                else if (name.EndsWith("_Right", StringComparison.OrdinalIgnoreCase))
                { side = BoneSide.Right; name = name[..^6]; }
            }

            // Prefix patterns: Left, Right (e.g. LeftArm, RightUpLeg)
            if (side == BoneSide.Center)
            {
                if (name.StartsWith("Left", StringComparison.OrdinalIgnoreCase) && name.Length > 4)
                { side = BoneSide.Left; name = name[4..]; }
                else if (name.StartsWith("Right", StringComparison.OrdinalIgnoreCase) && name.Length > 5)
                { side = BoneSide.Right; name = name[5..]; }
            }

            // Prefix patterns: L_, R_ (e.g. L_Clavicle, R_Thigh)
            if (side == BoneSide.Center && name.Length > 2)
            {
                char first = char.ToUpperInvariant(name[0]);
                if (name[1] is '_' or ' ')
                {
                    if (first == 'L') { side = BoneSide.Left; name = name[2..]; }
                    else if (first == 'R') { side = BoneSide.Right; name = name[2..]; }
                }
            }

            name = name.Trim('_', '-', ' ');

            return (name, side);
        }

        private static readonly char[] PrefixSeparators = { ':', '|' };

        // ─────────────────────────────────────────────────────
        // Alias tables
        // Core names are matched case-insensitively after normalization.
        // ─────────────────────────────────────────────────────

        /// <summary>Center bones (no left/right variant).</summary>
        private static readonly (HumanoidBone bone, string[] aliases)[] CenterBoneAliases =
        {
            (HumanoidBone.Hips, new[] { "Hips", "pelvis", "hip" }),
            (HumanoidBone.Spine, new[] { "Spine", "Spine1", "spine_01" }),
            (HumanoidBone.Chest, new[] { "Spine1", "Spine2", "Chest", "spine_02" }),
            (HumanoidBone.UpperChest, new[] { "Spine2", "Spine3", "UpperChest", "spine_03", "upperChest" }),
            (HumanoidBone.Neck, new[] { "Neck", "neck_01" }),
            (HumanoidBone.Head, new[] { "Head" }),
            (HumanoidBone.Jaw, new[] { "Jaw" }),
        };

        /// <summary>
        /// Sided bones — each entry defines a left/right pair and shared core-name aliases.
        /// The detected BoneSide selects which HumanoidBone to assign.
        /// </summary>
        private static readonly (HumanoidBone left, HumanoidBone right, string[] aliases)[] SidedBoneAliases =
        {
            // Arms
            (HumanoidBone.LeftShoulder, HumanoidBone.RightShoulder,
                new[] { "Shoulder", "Clavicle" }),
            (HumanoidBone.LeftUpperArm, HumanoidBone.RightUpperArm,
                new[] { "Arm", "UpperArm", "upper_arm", "upperarm" }),
            (HumanoidBone.LeftLowerArm, HumanoidBone.RightLowerArm,
                new[] { "ForeArm", "LowerArm", "lower_arm", "lowerarm", "forearm" }),
            (HumanoidBone.LeftHand, HumanoidBone.RightHand,
                new[] { "Hand" }),

            // Legs
            (HumanoidBone.LeftUpperLeg, HumanoidBone.RightUpperLeg,
                new[] { "UpLeg", "UpperLeg", "upper_leg", "upperleg", "Thigh", "thigh" }),
            (HumanoidBone.LeftLowerLeg, HumanoidBone.RightLowerLeg,
                new[] { "Leg", "LowerLeg", "lower_leg", "lowerleg", "Calf", "calf", "shin" }),
            (HumanoidBone.LeftFoot, HumanoidBone.RightFoot,
                new[] { "Foot" }),
            (HumanoidBone.LeftToes, HumanoidBone.RightToes,
                new[] { "ToeBase", "Toe", "Toes", "ball", "toe" }),

            // Fingers — Thumb
            (HumanoidBone.LeftThumbProximal, HumanoidBone.RightThumbProximal,
                new[] { "HandThumb1", "thumb_01", "thumb1", "f_thumb_01", "ThumbProximal" }),
            (HumanoidBone.LeftThumbIntermediate, HumanoidBone.RightThumbIntermediate,
                new[] { "HandThumb2", "thumb_02", "thumb2", "f_thumb_02", "ThumbIntermediate" }),
            (HumanoidBone.LeftThumbDistal, HumanoidBone.RightThumbDistal,
                new[] { "HandThumb3", "thumb_03", "thumb3", "f_thumb_03", "ThumbDistal" }),

            // Fingers — Index
            (HumanoidBone.LeftIndexProximal, HumanoidBone.RightIndexProximal,
                new[] { "HandIndex1", "index_01", "index1", "f_index_01", "IndexProximal" }),
            (HumanoidBone.LeftIndexIntermediate, HumanoidBone.RightIndexIntermediate,
                new[] { "HandIndex2", "index_02", "index2", "f_index_02", "IndexIntermediate" }),
            (HumanoidBone.LeftIndexDistal, HumanoidBone.RightIndexDistal,
                new[] { "HandIndex3", "index_03", "index3", "f_index_03", "IndexDistal" }),

            // Fingers — Middle
            (HumanoidBone.LeftMiddleProximal, HumanoidBone.RightMiddleProximal,
                new[] { "HandMiddle1", "middle_01", "middle1", "f_middle_01", "MiddleProximal" }),
            (HumanoidBone.LeftMiddleIntermediate, HumanoidBone.RightMiddleIntermediate,
                new[] { "HandMiddle2", "middle_02", "middle2", "f_middle_02", "MiddleIntermediate" }),
            (HumanoidBone.LeftMiddleDistal, HumanoidBone.RightMiddleDistal,
                new[] { "HandMiddle3", "middle_03", "middle3", "f_middle_03", "MiddleDistal" }),

            // Fingers — Ring
            (HumanoidBone.LeftRingProximal, HumanoidBone.RightRingProximal,
                new[] { "HandRing1", "ring_01", "ring1", "f_ring_01", "RingProximal" }),
            (HumanoidBone.LeftRingIntermediate, HumanoidBone.RightRingIntermediate,
                new[] { "HandRing2", "ring_02", "ring2", "f_ring_02", "RingIntermediate" }),
            (HumanoidBone.LeftRingDistal, HumanoidBone.RightRingDistal,
                new[] { "HandRing3", "ring_03", "ring3", "f_ring_03", "RingDistal" }),

            // Fingers — Little/Pinky
            (HumanoidBone.LeftLittleProximal, HumanoidBone.RightLittleProximal,
                new[] { "HandPinky1", "pinky_01", "pinky1", "f_pinky_01", "LittleProximal" }),
            (HumanoidBone.LeftLittleIntermediate, HumanoidBone.RightLittleIntermediate,
                new[] { "HandPinky2", "pinky_02", "pinky2", "f_pinky_02", "LittleIntermediate" }),
            (HumanoidBone.LeftLittleDistal, HumanoidBone.RightLittleDistal,
                new[] { "HandPinky3", "pinky_03", "pinky3", "f_pinky_03", "LittleDistal" }),

            // Eyes
            (HumanoidBone.LeftEye, HumanoidBone.RightEye,
                new[] { "Eye", "eye" }),
        };
    }
}
