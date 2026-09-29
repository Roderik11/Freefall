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
        /// <summary>Bind-direction differences below this are kept (target anatomy).</summary>
        public const float CorrectionDeadbandDeg = 10f;

        /// <summary>Fade range above the deadband over which the correction blends in fully.</summary>
        public const float CorrectionFadeDeg = 10f;

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
        /// Per source bone: model-space bind position from full hierarchy walk.
        /// Valid for structural nodes too (unlike inverting OffsetMatrix, which is
        /// identity for Assimp $AssimpFbx$ pivot nodes).
        /// </summary>
        public Vector3[] SrcBindModelPos;

        /// <summary>Per target bone: model-space bind position from full hierarchy walk.</summary>
        public Vector3[] DstBindModelPos;

        /// <summary>Position scale factor (target height / source height) for root motion.</summary>
        public float PositionScale = 1f;

        /// <summary>
        /// Transforms the source root bone's local position delta into the target
        /// root's local space: parent-chain rotation/scale of the source bind,
        /// uniform height scale, then inverse parent-chain of the target bind.
        /// Handles unit and axis differences between rigs (cm vs m, Z-up vs Y-up).
        /// </summary>
        public Matrix4x4 RootDeltaTransform = Matrix4x4.Identity;

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
        /// Secondary-axis override for twist alignment. The reference direction is
        /// measured between two bones' bind positions in each rig (e.g. the palm
        /// axis from index to pinky knuckle) — an anatomical reference that is far
        /// more reliable than the incoming parent direction for hands and fingers,
        /// where the parent direction is nearly collinear with the bone itself.
        /// </summary>
        public struct TwistHint
        {
            public int TargetBone;
            public int SrcFrom, SrcTo;
            public int DstFrom, DstTo;
        }

        /// <summary>Per target bone: twist reference direction in the source rig. Zero = use parent direction.</summary>
        public Vector3[] SrcTwistRef;

        /// <summary>Per target bone: twist reference direction in the target rig. Zero = use parent direction.</summary>
        public Vector3[] DstTwistRef;

        /// <summary>
        /// Create a BoneMap from explicit bone pairs.
        /// Each pair is (sourceBoneIndex, targetBoneIndex).
        /// </summary>
        public static BoneMap Create(Skeleton source, Skeleton target,
            ReadOnlySpan<(int src, int dst)> pairs,
            int sourceRoot = 0, int targetRoot = 0,
            ReadOnlySpan<TwistHint> twistHints = default)
        {
            var map = new BoneMap
            {
                Source = source,
                Target = target,
                SourceToTarget = new int[source.Bones.Length],
                TargetToSource = new int[target.Bones.Length],
                SourceRoot = sourceRoot,
                TargetRoot = targetRoot,
            };

            Array.Fill(map.SourceToTarget, -1);
            Array.Fill(map.TargetToSource, -1);

            // Build index mappings
            foreach (var (src, dst) in pairs)
            {
                if (src < 0 || src >= source.Bones.Length) continue;
                if (dst < 0 || dst >= target.Bones.Length) continue;
                map.SourceToTarget[src] = dst;
                map.TargetToSource[dst] = src;
            }

            // Bind world transforms by full hierarchy walk. Structural nodes
            // (axis conversion, $AssimpFbx$ pivots) carry transforms but have no
            // OffsetMatrix, so positions must come from the walk — inverting
            // OffsetMatrix reads them as sitting at the world origin.
            var srcWorld = ComputeBindWorld(source);
            var dstWorld = ComputeBindWorld(target);

            map.SrcBindModelPos = ExtractPositions(srcWorld);
            map.DstBindModelPos = ExtractPositions(dstWorld);

            // Resolve twist hints into reference directions
            map.SrcTwistRef = new Vector3[target.Bones.Length];
            map.DstTwistRef = new Vector3[target.Bones.Length];
            foreach (var hint in twistHints)
            {
                if (hint.TargetBone < 0 || hint.TargetBone >= target.Bones.Length) continue;
                var s = map.SrcBindModelPos[hint.SrcTo] - map.SrcBindModelPos[hint.SrcFrom];
                var d = map.DstBindModelPos[hint.DstTo] - map.DstBindModelPos[hint.DstFrom];
                if (s.LengthSquared() < 1e-8f || d.LengthSquared() < 1e-8f) continue;
                map.SrcTwistRef[hint.TargetBone] = Vector3.Normalize(s);
                map.DstTwistRef[hint.TargetBone] = Vector3.Normalize(d);
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
                    var srcDir = GetBoneDirection(source, map.SrcBindModelPos, srcIdx);
                    var dstDir = GetBoneDirection(target, map.DstBindModelPos, i);
                    var srcParentDir = GetIncomingDirection(source, map.SrcBindModelPos, srcIdx);
                    var dstParentDir = GetIncomingDirection(target, map.DstBindModelPos, i);

                    Quaternion correction;
                    if (Vector3.Dot(srcDir, dstDir) < 0f)
                    {
                        // Bone→child directions diverge (different children selected).
                        // Fall back to parent→bone direction which is always consistent;
                        // no twist constraint since the child directions are unreliable.
                        srcDir = srcParentDir;
                        dstDir = dstParentDir;
                        correction = RotationFromTo(dstDir, srcDir);
                    }
                    else
                    {
                        // Two-axis alignment: bone direction plus twist about it.
                        // Anatomical twist hints (palm axis for hands/fingers) are
                        // trusted at any angle; the parent→bone fallback is only a
                        // weld approximation, so large twists are rejected there.
                        var srcSecondary = srcParentDir;
                        var dstSecondary = dstParentDir;
                        bool limitTwist = true;
                        if (map.SrcTwistRef[i] != Vector3.Zero)
                        {
                            srcSecondary = map.SrcTwistRef[i];
                            dstSecondary = map.DstTwistRef[i];
                            limitTwist = false;
                        }
                        correction = ComputeFrameCorrection(srcDir, dstDir, srcSecondary, dstSecondary, limitTwist);
                    }

                    map.SrcBoneDir[i] = srcDir;
                    map.DstBoneDir[i] = dstDir;

                    float angle = MathF.Acos(Math.Clamp(Vector3.Dot(srcDir, dstDir), -1, 1)) * 180f / MathF.PI;
                    map.CorrectionAngle[i] = angle;

                    // Small bind-direction differences are anatomy (shoulder slope,
                    // foot pitch, posture), not pose differences — fully correcting
                    // them makes the target mimic the source's build. Fade the
                    // correction in so only genuine pose mismatches (A-pose vs
                    // T-pose arms) are corrected and the target keeps its own
                    // proportions below the deadband.
                    float fade = Math.Clamp((angle - CorrectionDeadbandDeg) / CorrectionFadeDeg, 0f, 1f);
                    if (fade < 1f)
                        correction = Quaternion.Slerp(Quaternion.Identity, correction, fade);

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

            // Position scale + root motion transform from model-space bind data
            ComputeRootMotionTransform(map, source, target, srcWorld, dstWorld, sourceRoot, targetRoot);

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
        /// Compute retargeted local + model-space rotations for the target skeleton
        /// from source animated model-space rotations (world-space offset method).
        ///
        /// For each mapped target bone i (source bone j):
        ///   srcLocal = Inv(srcAnimModel[srcParent]) * srcAnimModel[j]
        ///   trgLocal = Inv(tposeTrgWorldParent) * bindSrcWorldParent * srcLocal * Inv(bindSrcWorld) * tposeTrgWorld
        ///
        /// This works by:
        ///   1. srcWorldRot = bindSrcWorldParent * srcLocal  (hybrid: bind parents + anim bone)
        ///   2. offsetWorld = srcWorldRot * Inv(bindSrcWorld)  (world-space offset from bind)
        ///   3. trgWorldRot = offsetWorld * tposeTrgWorld  (apply offset to corrected target T-pose)
        ///   4. trgLocal = Inv(tposeTrgWorldParent) * trgWorldRot  (back to local)
        ///
        /// Unmapped target bones hold their bind pose.
        /// </summary>
        public void Retarget(Quaternion[] srcAnimModelRot, Quaternion[] localRot, Quaternion[] modelRot)
        {
            var bones = Target.Bones;
            var srcBones = Source.Bones;

            for (int i = 0; i < bones.Length; i++)
            {
                int dstParentIdx = bones[i].Parent;
                var parentModel = dstParentIdx >= 0
                    ? modelRot[dstParentIdx]
                    : Quaternion.Identity;

                int srcIdx = TargetToSource[i];
                if (srcIdx >= 0)
                {
                    // Extract source animated local rotation
                    int srcParentIdx = srcBones[srcIdx].Parent;
                    var srcParentModel = srcParentIdx >= 0
                        ? srcAnimModelRot[srcParentIdx]
                        : Quaternion.Identity;
                    var srcLocal = Quaternion.Inverse(srcParentModel) * srcAnimModelRot[srcIdx];

                    // Bind world rotations (source uses actual bind = T-pose,
                    // target uses T-pose auxiliary = DstTPoseModelRot)
                    var bindSrcWorldParent = srcParentIdx >= 0
                        ? SrcBindModelRot[srcParentIdx]
                        : Quaternion.Identity;
                    var invBindSrcWorld = Quaternion.Inverse(SrcBindModelRot[srcIdx]);
                    var bindTrgWorld = DstTPoseModelRot[i];
                    var invBindTrgWorldParent = dstParentIdx >= 0
                        ? Quaternion.Inverse(DstTPoseModelRot[dstParentIdx])
                        : Quaternion.Identity;

                    localRot[i] = Quaternion.Normalize(
                        invBindTrgWorldParent * bindSrcWorldParent * srcLocal * invBindSrcWorld * bindTrgWorld);
                    modelRot[i] = Quaternion.Normalize(parentModel * localRot[i]);
                }
                else
                {
                    localRot[i] = bones[i].BindPose.Rotation;
                    modelRot[i] = parentModel * bones[i].BindPose.Rotation;
                }
            }
        }

        /// <summary>
        /// Bind-pose world matrices by hierarchy walk (row-vector convention:
        /// world = local * parentWorld — same as Animator's hierarchy multiply).
        /// </summary>
        private static Matrix4x4[] ComputeBindWorld(Skeleton skel)
        {
            var world = new Matrix4x4[skel.Bones.Length];
            for (int i = 0; i < skel.Bones.Length; i++)
            {
                world[i] = skel.Bones[i].BindPoseMatrix;
                int parent = skel.Bones[i].Parent;
                if (parent >= 0)
                    world[i] = world[i] * world[parent];
            }
            return world;
        }

        private static Vector3[] ExtractPositions(Matrix4x4[] world)
        {
            var positions = new Vector3[world.Length];
            for (int i = 0; i < world.Length; i++)
                positions[i] = world[i].Translation;
            return positions;
        }

        /// <summary>
        /// Get the direction from the bone's parent to the bone (incoming direction).
        /// Walks up the parent chain past coincident ancestors — Assimp pivot nodes
        /// can sit exactly on the bone and would yield a degenerate direction.
        /// </summary>
        private static Vector3 GetIncomingDirection(Skeleton skel, Vector3[] positions, int boneIdx)
        {
            var pos = positions[boneIdx];
            int parent = skel.Bones[boneIdx].Parent;
            while (parent >= 0)
            {
                var d = pos - positions[parent];
                if (d.LengthSquared() > 1e-8f)
                    return Vector3.Normalize(d);
                parent = skel.Bones[parent].Parent;
            }
            return Vector3.UnitY;
        }

        /// <summary>
        /// Get the bone's direction vector in model space.
        /// For bones with multiple children, picks the child whose direction
        /// best continues the parent-to-bone chain (avoids branch mismatches
        /// between skeletons, e.g. spine → neck vs spine → clavicle).
        /// Children that coincide with the bone (pivot nodes, twist bones) are
        /// looked through to their own children. For leaf bones, uses the
        /// incoming direction.
        /// </summary>
        private static Vector3 GetBoneDirection(Skeleton skel, Vector3[] positions, int boneIdx)
        {
            var incomingDir = GetIncomingDirection(skel, positions, boneIdx);

            Vector3 bestDir = Vector3.Zero;
            float bestDot = -2f;
            FindBestChildDirection(skel, positions, boneIdx, positions[boneIdx], incomingDir, 0, ref bestDir, ref bestDot);

            if (bestDir.LengthSquared() > 0.5f)
                return bestDir;

            // No child — use incoming direction
            return incomingDir;
        }

        private static void FindBestChildDirection(Skeleton skel, Vector3[] positions,
            int boneIdx, Vector3 bonePos, Vector3 incomingDir, int depth,
            ref Vector3 bestDir, ref float bestDot)
        {
            if (depth > 8) return;

            for (int i = boneIdx + 1; i < skel.Bones.Length; i++)
            {
                if (skel.Bones[i].Parent != boneIdx) continue;

                var dir = positions[i] - bonePos;
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
                else
                {
                    // Child sits exactly on this bone (pivot/twist node) —
                    // look through it to its own children.
                    FindBestChildDirection(skel, positions, i, bonePos, incomingDir, depth + 1, ref bestDir, ref bestDot);
                }
            }
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
            Vector3 srcSecondary, Vector3 dstSecondary,
            bool limitTwist)
        {
            // Step 1: Align primary direction
            var correction = RotationFromTo(dstDir, srcDir);

            // Step 2: After primary alignment, check where the dst secondary axis ends up
            var correctedDstSec = Vector3.Transform(dstSecondary, correction);

            // Project both secondary axes onto plane perpendicular to srcDir
            var projCorrected = correctedDstSec - Vector3.Dot(correctedDstSec, srcDir) * srcDir;
            var projSource = srcSecondary - Vector3.Dot(srcSecondary, srcDir) * srcDir;

            // Only apply twist if both projections are valid (not degenerate)
            if (projCorrected.LengthSquared() > 0.01f && projSource.LengthSquared() > 0.01f)
            {
                projCorrected = Vector3.Normalize(projCorrected);
                projSource = Vector3.Normalize(projSource);

                // Signed twist angle about the bone axis. Must rotate about srcDir —
                // RotationFromTo would pick an arbitrary axis for anti-parallel
                // projections and break the primary alignment.
                float cos = Math.Clamp(Vector3.Dot(projCorrected, projSource), -1f, 1f);
                float sin = Vector3.Dot(Vector3.Cross(projCorrected, projSource), srcDir);
                float angle = MathF.Atan2(sin, cos);

                // With the parent-direction fallback, a near-opposite secondary
                // means the weld assumption broke (e.g. a large A→T swing where
                // the parent stays put) — the constraint is unreliable there,
                // keep swing only. Anatomical hints are trusted at any angle.
                if (!limitTwist || MathF.Abs(angle) < 120f * MathF.PI / 180f)
                {
                    var twistFix = Quaternion.CreateFromAxisAngle(srcDir, angle);
                    correction = Quaternion.Normalize(twistFix * correction);
                }
            }

            return correction;
        }

        /// <summary>
        /// Compute the position scale (target height / source height, in model units)
        /// and the root motion delta transform: source root local delta → source model
        /// space → height scale → target root local space. Model-space heights come
        /// from the bind world walk, so mixed units (cm rigs vs m rigs) and axis
        /// conventions are handled by the parent-chain matrices.
        /// </summary>
        private static void ComputeRootMotionTransform(BoneMap map,
            Skeleton source, Skeleton target,
            Matrix4x4[] srcWorld, Matrix4x4[] dstWorld,
            int sourceRoot, int targetRoot)
        {
            float srcHeight = ComputeBindHeight(source, map.SrcBindModelPos, sourceRoot);
            float dstHeight = ComputeBindHeight(target, map.DstBindModelPos, targetRoot);
            map.PositionScale = srcHeight > 1e-6f ? dstHeight / srcHeight : 1f;

            int srcParent = source.Bones[sourceRoot].Parent;
            int dstParent = target.Bones[targetRoot].Parent;
            var srcParentWorld = srcParent >= 0 ? srcWorld[srcParent] : Matrix4x4.Identity;
            var dstParentWorld = dstParent >= 0 ? dstWorld[dstParent] : Matrix4x4.Identity;
            srcParentWorld.Translation = Vector3.Zero;
            dstParentWorld.Translation = Vector3.Zero;

            if (!Matrix4x4.Invert(dstParentWorld, out var invDstParentWorld))
                invDstParentWorld = Matrix4x4.Identity;

            map.RootDeltaTransform = srcParentWorld
                * Matrix4x4.CreateScale(map.PositionScale)
                * invDstParentWorld;

            Freefall.Debug.Log($"[BoneMap] PositionScale: srcHeight={srcHeight:F2} dstHeight={dstHeight:F2} scale={map.PositionScale:F4}");
        }

        /// <summary>
        /// Bind-pose "height": model-space distance from the root bone to the
        /// furthest bone beneath it (typically hips → head).
        /// </summary>
        private static float ComputeBindHeight(Skeleton skeleton, Vector3[] positions, int rootIndex)
        {
            float maxDist = 0;
            for (int i = 0; i < skeleton.Bones.Length; i++)
            {
                // Only consider bones under the root
                int current = i;
                while (current >= 0 && current != rootIndex)
                    current = skeleton.Bones[current].Parent;
                if (current != rootIndex) continue;

                float dist = (positions[i] - positions[rootIndex]).Length();
                if (dist > maxDist) maxDist = dist;
            }
            return maxDist;
        }
    }
}
