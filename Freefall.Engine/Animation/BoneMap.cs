using System;
using System.Numerics;

namespace Freefall.Animation
{
    /// <summary>
    /// Maps bones between a source and target skeleton for animation retargeting.
    /// Uses bone-direction alignment from joint positions (rig-agnostic).
    /// </summary>
    public class BoneMap
    {
        public Skeleton Source;
        public Skeleton Target;

        /// <summary>Per source bone index: the corresponding target bone index. -1 = unmapped.</summary>
        public int[] SourceToTarget;

        /// <summary>Per target bone index: the corresponding source bone index. -1 = unmapped.</summary>
        public int[] TargetToSource;

        /// <summary>
        /// Per target bone: full model-space bind rotation (including structural nodes).
        /// Computed by hierarchy walk of bind locals.
        /// </summary>
        public Quaternion[] DstBindModelRot;

        /// <summary>
        /// Per source bone: full model-space bind rotation.
        /// Used to compute source animation delta: srcAnim * Inv(SrcBindModelRot).
        /// </summary>
        public Quaternion[] SrcBindModelRot;

        /// <summary>
        /// Per source bone: alignment rotation that maps the source bone's direction
        /// frame to the target bone's direction frame. Computed from model-space
        /// positions (rig-convention agnostic).
        ///
        /// At runtime: dstAnimModel = DstBindModel * Correction * srcAnimModel * Inv(Correction)
        /// </summary>
        public Quaternion[] Correction;

        /// <summary>Position scale factor (target height / source height) for root motion.</summary>
        public float PositionScale;

        /// <summary>Root bone index in source skeleton (for position retargeting).</summary>
        public int SourceRoot;

        /// <summary>Root bone index in target skeleton.</summary>
        public int TargetRoot;

        /// <summary>
        /// Per target bone: local rotation that would place the bone in T-pose
        /// (matching source bind directions). Computed by correcting each bone's
        /// model-space direction to match source, then deriving locals from hierarchy.
        /// </summary>
        public Quaternion[] DstTPoseLocal;

        /// <summary>Per target bone: model-space rotation when target is posed into T-pose
        /// (matching source bind directions). Used as auxiliary pose for retargeting.</summary>
        public Quaternion[] DstTPoseModelRot;

        /// <summary>Per target bone: source bone direction at bind (model-space). Zero if unmapped.</summary>
        public Vector3[] SrcBoneDir;

        /// <summary>Per target bone: target bone direction at bind (model-space). Zero if unmapped.</summary>
        public Vector3[] DstBoneDir;

        /// <summary>Per target bone: bind direction difference angle in degrees. 0 if unmapped.</summary>
        public float[] CorrectionAngle;

        /// <summary>
        /// Create a BoneMap from explicit bone pairs.
        /// Each pair is (sourceBoneIndex, targetBoneIndex).
        /// </summary>
        public static BoneMap Create(Skeleton source, Skeleton target,
            ReadOnlySpan<(int src, int dst)> pairs,
            int sourceRoot = 0, int targetRoot = 0)
        {
            var map = new BoneMap
            {
                Source = source,
                Target = target,
                SourceToTarget = new int[source.Bones.Length],
                TargetToSource = new int[target.Bones.Length],
                Correction = new Quaternion[source.Bones.Length],
                SourceRoot = sourceRoot,
                TargetRoot = targetRoot,
            };

            Array.Fill(map.SourceToTarget, -1);
            Array.Fill(map.TargetToSource, -1);

            for (int i = 0; i < source.Bones.Length; i++)
                map.Correction[i] = Quaternion.Identity;

            // Build index mappings
            foreach (var (src, dst) in pairs)
            {
                if (src < 0 || src >= source.Bones.Length) continue;
                if (dst < 0 || dst >= target.Bones.Length) continue;
                map.SourceToTarget[src] = dst;
                map.TargetToSource[dst] = src;
            }

            // Precompute full model-space bind rotations for ALL target bones.
            map.DstBindModelRot = new Quaternion[target.Bones.Length];
            for (int i = 0; i < target.Bones.Length; i++)
            {
                int parent = target.Bones[i].Parent;
                map.DstBindModelRot[i] = parent >= 0
                    ? map.DstBindModelRot[parent] * target.Bones[i].BindPose.Rotation
                    : target.Bones[i].BindPose.Rotation;
            }

            // Precompute source model-space bind rotations
            map.SrcBindModelRot = new Quaternion[source.Bones.Length];
            var srcModelRot = map.SrcBindModelRot;
            for (int i = 0; i < source.Bones.Length; i++)
            {
                int parent = source.Bones[i].Parent;
                srcModelRot[i] = parent >= 0
                    ? srcModelRot[parent] * source.Bones[i].BindPose.Rotation
                    : source.Bones[i].BindPose.Rotation;
            }

            // Structural rotation = model-space rotation ABOVE the root bone.
            // DstBindModelRot already accounts for this, so directions must NOT include it.
            Quaternion srcStructural = source.Bones[sourceRoot].Parent >= 0
                ? srcModelRot[source.Bones[sourceRoot].Parent]
                : Quaternion.Identity;
            Quaternion dstStructural = target.Bones[targetRoot].Parent >= 0
                ? map.DstBindModelRot[target.Bones[targetRoot].Parent]
                : Quaternion.Identity;
            var invSrcStructural = Quaternion.Inverse(srcStructural);
            var invDstStructural = Quaternion.Inverse(dstStructural);

            // Compute T-pose normalization and precompute bone directions.
            var tposeModel = new Quaternion[target.Bones.Length];
            map.DstTPoseLocal = new Quaternion[target.Bones.Length];
            map.SrcBoneDir = new Vector3[target.Bones.Length];
            map.DstBoneDir = new Vector3[target.Bones.Length];
            map.CorrectionAngle = new float[target.Bones.Length];

            var dirLog = new System.Text.StringBuilder();
            dirLog.AppendLine($"[TPose] {source.Name} → {target.Name}");

            for (int i = 0; i < target.Bones.Length; i++)
            {
                int parentIdx = target.Bones[i].Parent;
                var parentTPose = parentIdx >= 0 ? tposeModel[parentIdx] : Quaternion.Identity;

                int srcIdx = map.TargetToSource[i];
                if (srcIdx >= 0)
                {
                    // Mapped bone: store directions and compute T-pose local
                    var srcDir = GetBoneDirection(source, srcIdx);
                    var dstDir = GetBoneDirection(target, i);

                    // If bone→child directions diverge (different children selected),
                    // fall back to parent→bone direction which is always consistent.
                    if (Vector3.Dot(srcDir, dstDir) < 0f)
                    {
                        srcDir = GetParentDirection(source, srcIdx);
                        dstDir = GetParentDirection(target, i);
                    }

                    map.SrcBoneDir[i] = srcDir;
                    map.DstBoneDir[i] = dstDir;

                    float angle = MathF.Acos(Math.Clamp(Vector3.Dot(srcDir, dstDir), -1, 1)) * 180f / MathF.PI;
                    map.CorrectionAngle[i] = angle;

                    var correction = RotationFromTo(dstDir, srcDir);
                    tposeModel[i] = Quaternion.Normalize(correction * map.DstBindModelRot[i]);

                    if (angle > 2f)
                        dirLog.AppendLine($"  [{i}] {target.BoneNames[i],-22} " +
                            $"bindDir=({dstDir.X,6:F2},{dstDir.Y,6:F2},{dstDir.Z,6:F2}) " +
                            $"tposeDir=({srcDir.X,6:F2},{srcDir.Y,6:F2},{srcDir.Z,6:F2}) " +
                            $"correction={angle:F1}°");
                }
                else
                {
                    // Unmapped: cascade from parent's T-pose + own bind local
                    tposeModel[i] = Quaternion.Normalize(
                        parentTPose * target.Bones[i].BindPose.Rotation);
                }

                map.DstTPoseLocal[i] = Quaternion.Normalize(
                    Quaternion.Inverse(parentTPose) * tposeModel[i]);
            }
            map.DstTPoseModelRot = tposeModel;
            Freefall.Debug.Log(dirLog.ToString());

            // Compute position scale from root bone positions
            map.PositionScale = ComputePositionScale(source, target, sourceRoot, targetRoot);

            return map;
        }

        /// <summary>
        /// Create a BoneMap by matching bone names directly.
        /// Fallback when no template applies — uses name-hash matching like the old system
        /// but adds rotation corrections.
        /// </summary>
        public static BoneMap CreateFromNames(Skeleton source, Skeleton target)
        {
            // Build pairs from matching bone names
            int pairCount = 0;
            Span<(int src, int dst)> pairs = stackalloc (int, int)[Math.Min(source.Bones.Length, target.Bones.Length)];

            for (int s = 0; s < source.Bones.Length; s++)
            {
                int d = target.FindBone(source.BoneNames[s]);
                if (d >= 0 && pairCount < pairs.Length)
                    pairs[pairCount++] = (s, d);
            }

            return Create(source, target, pairs[..pairCount]);
        }

        /// <summary>
        /// Get the model-space position of a bone from its OffsetMatrix.
        /// OffsetMatrix = Inv(ModelSpaceBindTransform), so we invert to get position.
        /// </summary>
        private static Vector3 GetModelPosition(Bone bone)
        {
            if (Matrix4x4.Invert(bone.OffsetMatrix, out var modelBind))
                return modelBind.Translation;
            return bone.BindPose.Position; // fallback for structural nodes
        }

        /// <summary>
        /// Get the direction from the bone's parent to the bone (incoming direction).
        /// Used as secondary axis for twist alignment.
        /// </summary>
        private static Vector3 GetParentDirection(Skeleton skel, int boneIdx)
        {
            int parentIdx = skel.Bones[boneIdx].Parent;
            if (parentIdx >= 0)
            {
                var pos = GetModelPosition(skel.Bones[boneIdx]);
                var parentPos = GetModelPosition(skel.Bones[parentIdx]);
                var d = pos - parentPos;
                if (d.LengthSquared() > 1e-8f)
                    return Vector3.Normalize(d);
            }
            return Vector3.UnitY;
        }

        /// <summary>
        /// Get the bone's direction vector in model space.
        /// For bones with multiple children, picks the child whose direction
        /// best continues the parent-to-bone chain (avoids branch mismatches
        /// between skeletons, e.g. spine → neck vs spine → clavicle).
        /// For leaf bones, uses direction from parent to bone.
        /// </summary>
        public static Vector3 GetBoneDirectionPublic(Skeleton skel, int boneIdx)
            => GetBoneDirection(skel, boneIdx);

        private static Vector3 GetBoneDirection(Skeleton skel, int boneIdx)
        {
            var pos = GetModelPosition(skel.Bones[boneIdx]);

            // Get the "incoming" direction (parent → this bone) to detect chain continuation
            int parentIdx = skel.Bones[boneIdx].Parent;
            Vector3 incomingDir = Vector3.UnitY; // default up
            if (parentIdx >= 0)
            {
                var parentPos = GetModelPosition(skel.Bones[parentIdx]);
                var d = pos - parentPos;
                if (d.LengthSquared() > 1e-8f)
                    incomingDir = Vector3.Normalize(d);
            }

            // Find the child whose direction best continues the chain
            Vector3 bestDir = Vector3.Zero;
            float bestDot = -2f;

            for (int i = boneIdx + 1; i < skel.Bones.Length; i++)
            {
                if (skel.Bones[i].Parent == boneIdx)
                {
                    var childPos = GetModelPosition(skel.Bones[i]);
                    var dir = childPos - pos;
                    if (dir.LengthSquared() > 1e-8f)
                    {
                        var dirNorm = Vector3.Normalize(dir);
                        float dot = Vector3.Dot(dirNorm, incomingDir);
                        if (dot > bestDot)
                        {
                            bestDot = dot;
                            bestDir = dirNorm;
                        }
                    }
                }
            }

            if (bestDir.LengthSquared() > 0.5f)
                return bestDir;

            // No child — use incoming direction
            return incomingDir;
        }

        /// <summary>
        /// Compute the shortest rotation quaternion that maps direction 'from' to direction 'to'.
        /// Both must be unit vectors.
        /// </summary>
        public static Quaternion RotationFromTo(Vector3 from, Vector3 to)
        {
            float dot = Vector3.Dot(from, to);

            if (dot > 0.9999f)
                return Quaternion.Identity;

            if (dot < -0.9999f)
            {
                // Anti-parallel: 180° around any perpendicular axis
                var perp = MathF.Abs(from.X) < 0.9f
                    ? Vector3.Cross(Vector3.UnitX, from)
                    : Vector3.Cross(Vector3.UnitY, from);
                return Quaternion.CreateFromAxisAngle(Vector3.Normalize(perp), MathF.PI);
            }

            // Half-angle formula: q = (cross, 1 + dot), normalized
            var w = Vector3.Cross(from, to);
            return Quaternion.Normalize(new Quaternion(w.X, w.Y, w.Z, 1f + dot));
        }

        /// <summary>
        /// Compute a twist-free correction rotation from dstDir→srcDir using two axes.
        /// Step 1: RotationFromTo aligns the primary direction (bone→child).
        /// Step 2: Compute twist correction around the aligned axis using parent direction
        /// as secondary constraint (prevents foot roll, hand curl, etc.)
        /// Falls back to RotationFromTo when parent directions are degenerate.
        /// </summary>
        private static Quaternion ComputeFrameCorrection(
            Vector3 srcDir, Vector3 dstDir,
            Vector3 srcParentDir, Vector3 dstParentDir)
        {
            // Step 1: Align primary direction
            var correction = RotationFromTo(dstDir, srcDir);

            // Step 2: After primary alignment, check where dst parent direction ends up
            var correctedDstParDir = Vector3.Transform(dstParentDir, correction);

            // Project both parent directions onto plane perpendicular to srcDir
            var projCorrected = correctedDstParDir - Vector3.Dot(correctedDstParDir, srcDir) * srcDir;
            var projSource = srcParentDir - Vector3.Dot(srcParentDir, srcDir) * srcDir;

            // Only apply twist if both projections are valid (not degenerate)
            if (projCorrected.LengthSquared() > 0.01f && projSource.LengthSquared() > 0.01f)
            {
                projCorrected = Vector3.Normalize(projCorrected);
                projSource = Vector3.Normalize(projSource);
                var twistFix = RotationFromTo(projCorrected, projSource);
                correction = Quaternion.Normalize(twistFix * correction);
            }

            return correction;
        }

        /// <summary>
        /// Compute model-space rotation relative to a reference bone (e.g. Hips/pelvis).
        /// Walks the parent chain but stops at the reference bone, excluding its rotation
        /// and everything above it. This strips coordinate-system structural nodes
        /// that differ between FBX files from different tools.
        /// Returns Identity when boneIndex == refBone.
        /// </summary>
        private static Quaternion GetModelSpaceRotationRelativeTo(Skeleton skeleton, int boneIndex, int refBone)
        {
            if (boneIndex == refBone)
                return Quaternion.Identity;

            var result = skeleton.Bones[boneIndex].BindPose.Rotation;
            int parent = skeleton.Bones[boneIndex].Parent;
            while (parent >= 0 && parent != refBone)
            {
                result = skeleton.Bones[parent].BindPose.Rotation * result;
                parent = skeleton.Bones[parent].Parent;
            }
            return Quaternion.Normalize(result);
        }

        /// <summary>
        /// Compute model-space rotation by walking the parent chain.
        /// This is essential for retargeting because structural nodes (axis-conversion,
        /// armature nodes) carry rotations that affect the parent frame but have no
        /// offset matrices — so GetModelRotationFromOffset would miss them.
        /// </summary>
        private static Quaternion GetModelSpaceRotation(Skeleton skeleton, int boneIndex)
        {
            var result = skeleton.Bones[boneIndex].BindPose.Rotation;
            int parent = skeleton.Bones[boneIndex].Parent;
            while (parent >= 0)
            {
                result = skeleton.Bones[parent].BindPose.Rotation * result;
                parent = skeleton.Bones[parent].Parent;
            }
            return Quaternion.Normalize(result);
        }

        /// <summary>
        /// Compute a position scale factor by comparing skeleton heights.
        /// Measures model-space distance from root (hips) to head in bind pose.
        /// Falls back to root translation length if head bone isn't mapped.
        /// </summary>
        private static float ComputePositionScale(Skeleton source, Skeleton target,
            int sourceRoot, int targetRoot)
        {
            float srcHeight = ComputeBindHeight(source, sourceRoot);
            float dstHeight = ComputeBindHeight(target, targetRoot);

            float scale = (srcHeight > 1e-6f) ? dstHeight / srcHeight : 1f;
            Freefall.Debug.Log($"[BoneMap] PositionScale: srcHeight={srcHeight:F2} dstHeight={dstHeight:F2} scale={scale:F4}");
            return scale;
        }

        /// <summary>
        /// Compute the bind-pose height of a skeleton by accumulating translations
        /// from the root up through the spine chain to the head (longest chain).
        /// </summary>
        private static float ComputeBindHeight(Skeleton skeleton, int rootIndex)
        {
            // Accumulate model-space positions by walking the hierarchy
            // and find the bone furthest from root (vertically)
            float maxDist = 0;
            for (int i = 0; i < skeleton.Bones.Length; i++)
            {
                // Walk up to root and accumulate local translations
                var pos = Vector3.Zero;
                int current = i;
                while (current >= 0 && current != rootIndex)
                {
                    pos += skeleton.Bones[current].BindPose.Position;
                    current = skeleton.Bones[current].Parent;
                }
                if (current == rootIndex)
                {
                    float dist = pos.Length();
                    if (dist > maxDist) maxDist = dist;
                }
            }
            // Include the root's own position as fallback
            if (maxDist < 1e-6f)
                maxDist = skeleton.Bones[rootIndex].BindPoseMatrix.Translation.Length();
            return maxDist;
        }
    }
}
