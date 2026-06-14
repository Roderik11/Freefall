using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Animation;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Components
{
    public delegate void BoneTransformHandler(Bone bone, ref Matrix4x4 matrix);
    public delegate void PostHierarchyHandler(Bone[] skeleton, Matrix4x4[] bones);

    /// <summary>
    /// Controls animation playback for skinned meshes using an animation state machine.
    /// Owns per-Skeleton bone buffers (StreamingBuffer) that are shared across all
    /// SkinnedMeshRenderers on this entity. Each unique Skeleton gets posed once per frame.
    /// </summary>
    [Icon("icon_animator.png")]
    public class Animator : Component, IUpdate, IParallel
    {
        /// <summary>The animation state machine (shared definition).</summary>
        public Animation.Animation Animation;

        /// <summary>Per-instance mutable state (time, weights, parameters).</summary>
        public readonly AnimationPlayback Playback = new();

        private Skeleton _retargetSource;

        /// <summary>
        /// The source skeleton that the animation clips were made for.
        /// Set this when the clips come from a different character than the mesh.
        /// </summary>
        public Skeleton RetargetSource
        {
            get => _retargetSource;
            set
            {
                _retargetSource = value;
                _retargetInitialized = false;
            }
        }

        /// <summary>Event fired when a bone transform is calculated.</summary>
        public event BoneTransformHandler OnBoneTransform;

        /// <summary>Event fired after hierarchy multiplication — bones are in model space.</summary>
        public event PostHierarchyHandler OnPostHierarchy;

        /// <summary>Event fired when an animation event occurs.</summary>
        public Action<string> OnAnimationEvent;

        // Retargeting
        private BoneMap _boneMap;
        private bool _retargetInitialized;
        private bool _debugLogged;
        private Quaternion[] _srcAnimModelRot;
        private Quaternion[] _retargetedLocalRot;
        private Quaternion[] _retargetedModelRot;

        // Parameter name → index lookup (built once when Animation is set)
        private Dictionary<string, AnimationParameter> _paramMap;

        // Per-Skeleton bone buffers: each unique Skeleton gets its own GPU buffer.
        // Multiple SMRs sharing the same Skeleton share the same buffer (no redundant posing).
        private readonly Dictionary<Skeleton, BoneBufferEntry> _boneBuffers = new();
        
        private class BoneBufferEntry
        {
            public StreamingBuffer<Matrix4x4> Buffer;
            public Matrix4x4[] StagingMatrices;
        }

        protected override void Awake()
        {
        }

        private void EnsureParamMap()
        {
            if (_paramMap != null || Animation == null) return;

            _paramMap = new Dictionary<string, AnimationParameter>(Animation.Parameters.Count);
            foreach (var p in Animation.Parameters)
                _paramMap[p.Name] = p;
        }

        // --- Parameter API ---

        public float GetParam(string name)
        {
            EnsureParamMap();
            if (_paramMap != null && _paramMap.TryGetValue(name, out var param))
                return Playback.Get(PK.Param(param.Index), param.DefaultValue);
            return 0f;
        }

        public void SetParam(string name, float value)
        {
            EnsureParamMap();
            if (_paramMap != null && _paramMap.TryGetValue(name, out var param))
                Playback.Set(PK.Param(param.Index), value);
        }

        internal void ConsumeTrigger(string name)
        {
            EnsureParamMap();
            if (_paramMap != null && _paramMap.TryGetValue(name, out var param) && param.IsTrigger)
                Playback.Set(PK.Param(param.Index), 0);
        }

        internal void FireAnimationEvent(string name) => OnAnimationEvent?.Invoke(name);

        // --- Update ---

        public void Update()
        {
            if (Entity == null) return;
            if (Animation == null) return;

            if (!_retargetInitialized)
                InitRetargeting();

            foreach (AnimationLayer layer in Animation.Layers)
                layer.Update(this, Playback);

            WalkChildren(Entity.Transform);
        }

        // Walk child SMRs: pose each unique Skeleton once, set BoneBufferIdx
        void WalkChildren(Transform parent)
        {
            var smr = parent?.Entity?.GetComponent<SkinnedMeshRenderer>();
            if (smr?.Mesh?.Skeleton != null)
            {
                var skeleton = smr.Mesh.Skeleton;
                var bones = skeleton.Bones;

                var entry = EnsureBoneBuffer(skeleton);

                // Pose + upload (only once per unique Skeleton per frame)
                if (entry.Buffer.LastWriteFrame != Engine.FrameIndex)
                {
                    GetPose(bones, entry.StagingMatrices, skeleton.FlipXZ);
                    entry.Buffer.BulkWrite(entry.StagingMatrices);
                    entry.Buffer.LastWriteFrame = Engine.FrameIndex;
                }

                // Set BoneBufferIdx on SMR so it passes it through to Enqueue
                smr.BoneBufferIdx = entry.Buffer.SrvIndex;
            }

            foreach (Transform child in parent)
                WalkChildren(child);
        }

        /// <summary>
        /// Get or create a bone buffer for a Skeleton. Lazily allocated.
        /// </summary>
        private BoneBufferEntry EnsureBoneBuffer(Skeleton skeleton)
        {
            if (_boneBuffers.TryGetValue(skeleton, out var existing))
                return existing;

            int boneCount = skeleton.Bones.Length;
            var buffer = new StreamingBuffer<Matrix4x4>(Engine.Device, boneCount);
            var entry = new BoneBufferEntry
            {
                Buffer = buffer,
                StagingMatrices = new Matrix4x4[boneCount]
            };
            _boneBuffers[skeleton] = entry;
            return entry;
        }

        public override void Destroy()
        {
            foreach (var entry in _boneBuffers.Values)
                entry.Buffer.Dispose();
            _boneBuffers.Clear();
        }

        // --- Retargeting ---

        private void InitRetargeting()
        {
            _retargetInitialized = true;
            _debugLogged = false;
            _boneMap = null;

            if (RetargetSource == null) return;

            var renderer = Entity?.GetComponentInChildren<SkinnedMeshRenderer>();
            var meshSkeleton = renderer?.Mesh?.Skeleton;
            if (meshSkeleton == null) return;
            if (RetargetSource == meshSkeleton) return;

            // Try humanoid template first, fall back to name matching
            _boneMap = HumanoidTemplate.TryMap(RetargetSource, meshSkeleton)
                    ?? BoneMap.CreateFromNames(RetargetSource, meshSkeleton);

            Debug.Log($"[Animator] Retarget: {RetargetSource.Name} → {meshSkeleton.Name}, BoneMap created");

            // One-time: compare model-space positions of key bones
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"[BonePos] {RetargetSource.Name} → {meshSkeleton.Name}");
            for (int i = 0; i < meshSkeleton.Bones.Length; i++)
            {
                int srcIdx = _boneMap.TargetToSource[i];
                if (srcIdx < 0) continue;

                Vector3 srcPos = Vector3.Zero, dstPos = Vector3.Zero;
                if (Matrix4x4.Invert(RetargetSource.Bones[srcIdx].OffsetMatrix, out var srcInv))
                    srcPos = srcInv.Translation;
                if (Matrix4x4.Invert(meshSkeleton.Bones[i].OffsetMatrix, out var dstInv))
                    dstPos = dstInv.Translation;

                sb.AppendLine($"  dst[{i}] {meshSkeleton.BoneNames[i],-22} src=({srcPos.X,7:F2},{srcPos.Y,7:F2},{srcPos.Z,7:F2})  dst=({dstPos.X,7:F2},{dstPos.Y,7:F2},{dstPos.Z,7:F2})");
            }
            Debug.Log(sb.ToString());
        }

        /// <summary>
        /// Evaluate the source skeleton's animation to get full model-space rotations.
        /// These are accumulated from the root through all bones (including structural nodes).
        /// </summary>
        private void EvaluateSourceModelRotations()
        {
            var source = _boneMap.Source;
            int count = source.Bones.Length;

            if (_srcAnimModelRot == null || _srcAnimModelRot.Length != count)
                _srcAnimModelRot = new Quaternion[count];

            BonePose temp = new BonePose { Scale = Vector3.One, Rotation = Quaternion.Identity };

            for (int j = 0; j < count; j++)
            {
                var srcBone = source.Bones[j];
                BonePose blendPose = srcBone.BindPose;

                if (Animation != null)
                {
                    int layerCount = Animation.Layers.Count;
                    for (int i = 0; i < layerCount; i++)
                    {
                        var layer = Animation.Layers[i];
                        if (layer.Mask != null && !layer.Mask.Contains(srcBone.Name))
                            continue;

                        int stateCount = layer.GetStateCount(Playback);
                        for (int st = 0; st < stateCount; st++)
                        {
                            var state = layer.GetState(st, Playback);
                            state?.BlendBone(srcBone, ref blendPose, ref temp, Playback);
                        }
                    }
                }

                int parent = srcBone.Parent;
                _srcAnimModelRot[j] = parent >= 0
                    ? _srcAnimModelRot[parent] * blendPose.Rotation
                    : blendPose.Rotation;
            }
        }

        /// <summary>
        /// Compute retargeted rotations using the world-space offset method.
        /// (upf-gti / sketchpunk algorithm)
        ///
        /// For each mapped target bone i (source bone j):
        ///   srcLocal = Inv(srcAnimModel[srcParent]) * srcAnimModel[j]
        ///   trgLocal = Inv(bindTrgWorldParent) * bindSrcWorldParent * srcLocal * Inv(bindSrcWorld) * bindTrgWorld
        ///
        /// This works by:
        ///   1. srcWorldRot = bindSrcWorldParent * srcLocal  (hybrid: bind parents + anim bone)
        ///   2. offsetWorld = srcWorldRot * Inv(bindSrcWorld)  (world-space offset from bind)
        ///   3. trgWorldRot = offsetWorld * bindTrgWorld  (apply offset to target bind)
        ///   4. trgLocal = Inv(bindTrgWorldParent) * trgWorldRot  (back to local)
        /// </summary>
        private void ComputeRetargetedRotations(Bone[] skeleton)
        {
            int count = skeleton.Length;

            if (_retargetedLocalRot == null || _retargetedLocalRot.Length != count)
            {
                _retargetedLocalRot = new Quaternion[count];
                _retargetedModelRot = new Quaternion[count];
            }

            var srcBones = _boneMap.Source.Bones;

            for (int i = 0; i < count; i++)
            {
                int dstParentIdx = skeleton[i].Parent;
                var parentModel = dstParentIdx >= 0
                    ? _retargetedModelRot[dstParentIdx]
                    : Quaternion.Identity;

                int srcIdx = _boneMap.TargetToSource[i];
                if (srcIdx >= 0)
                {
                    // Extract source animated local rotation
                    int srcParentIdx = srcBones[srcIdx].Parent;
                    var srcParentModel = srcParentIdx >= 0
                        ? _srcAnimModelRot[srcParentIdx]
                        : Quaternion.Identity;
                    var srcLocal = Quaternion.Inverse(srcParentModel) * _srcAnimModelRot[srcIdx];

                    // Bind world rotations (source uses actual bind = T-pose,
                    // target uses T-pose auxiliary = DstTPoseModelRot)
                    var bindSrcWorldParent = srcParentIdx >= 0
                        ? _boneMap.SrcBindModelRot[srcParentIdx]
                        : Quaternion.Identity;
                    var invBindSrcWorld = Quaternion.Inverse(_boneMap.SrcBindModelRot[srcIdx]);
                    var bindTrgWorld = _boneMap.DstTPoseModelRot[i];
                    var invBindTrgWorldParent = dstParentIdx >= 0
                        ? Quaternion.Inverse(_boneMap.DstTPoseModelRot[dstParentIdx])
                        : Quaternion.Identity;

                    // World-space offset retargeting
                    _retargetedLocalRot[i] = Quaternion.Normalize(
                        invBindTrgWorldParent * bindSrcWorldParent * srcLocal * invBindSrcWorld * bindTrgWorld);
                    _retargetedModelRot[i] = Quaternion.Normalize(
                        parentModel * _retargetedLocalRot[i]);
                }
                else
                {
                    _retargetedLocalRot[i] = skeleton[i].BindPose.Rotation;
                    _retargetedModelRot[i] = parentModel * skeleton[i].BindPose.Rotation;
                }
            }

            // One-time debug: trace arm bone directions
            if (!_debugLogged)
            {
                _debugLogged = true;
                var builder = new System.Text.StringBuilder();
                builder.AppendLine($"[RetargetDiag] {_boneMap.Target.Name}");
                for (int i = 0; i < count; i++)
                {
                    int srcIdx2 = _boneMap.TargetToSource[i];
                    if (srcIdx2 < 0) continue;

                    var srcBoneDir = BoneMap.GetBoneDirectionPublic(_boneMap.Source, srcIdx2);
                    var dstBoneDir = BoneMap.GetBoneDirectionPublic(_boneMap.Target, i);

                    // Where does the source bone point during animation?
                    var srcDelta2 = _srcAnimModelRot[srcIdx2] *
                                    Quaternion.Inverse(_boneMap.SrcBindModelRot[srcIdx2]);
                    var srcAnimDir = Vector3.Transform(srcBoneDir, srcDelta2);

                    // Where does our retargeted bone point?
                    var retargetedModel2 = _retargetedModelRot[i];
                    var retDelta = retargetedModel2 * Quaternion.Inverse(_boneMap.DstBindModelRot[i]);
                    var retDir = Vector3.Transform(dstBoneDir, retDelta);

                    // How far off is the retargeted direction from the source's direction?
                    float dirError = MathF.Acos(Math.Clamp(Vector3.Dot(
                        Vector3.Normalize(srcAnimDir), Vector3.Normalize(retDir)), -1, 1)) * 180f / MathF.PI;

                    if (dirError > 5f)
                        builder.AppendLine($"  [{i}] {skeleton[i].Name,-22} " +
                            $"srcAnimDir=({srcAnimDir.X,6:F2},{srcAnimDir.Y,6:F2},{srcAnimDir.Z,6:F2}) " +
                            $"retDir=({retDir.X,6:F2},{retDir.Y,6:F2},{retDir.Z,6:F2}) " +
                            $"error={dirError:F1}°");
                }
                Debug.Log(builder.ToString());
            }
        }

        public void GetPose(Bone[] skeleton, Matrix4x4[] bones, bool flipXZ = false)
        {
            int count = skeleton.Length;

            // For retargeted characters: evaluate source skeleton and compute
            // retargeted rotations via model-space approach with direction correction.
            if (_boneMap != null)
            {
                EvaluateSourceModelRotations();
                ComputeRetargetedRotations(skeleton);
            }

            BonePose tempPose = new BonePose { Scale = Vector3.One, Rotation = Quaternion.Identity };
            for (int i = 0; i < count; i++)
                BlendBone(i, skeleton[i], ref tempPose, out bones[i]);

            if (OnBoneTransform != null)
            {
                for (int i = 0; i < count; i++)
                    OnBoneTransform(skeleton[i], ref bones[i]);
            }

            for (int i = 0; i < count; i++)
            {
                if (skeleton[i].Parent > -1)
                    bones[i] = bones[i] * bones[skeleton[i].Parent];
            }

            OnPostHierarchy?.Invoke(skeleton, bones);

            for (int i = 0; i < count; i++)
            {
                var m = skeleton[i].OffsetMatrix * bones[i];
                if (flipXZ) m = ConjugateFlipXZ(m);
                bones[i] = Matrix4x4.Transpose(m);
            }
        }

        /// <summary>
        /// Conjugate a matrix by F = diag(-1,1,-1,1): F * M * F.
        /// Converts bone matrices from Assimp LH space to match -X/-Z flipped mesh vertices.
        /// </summary>
        private static Matrix4x4 ConjugateFlipXZ(Matrix4x4 m)
        {
            return new Matrix4x4(
                 m.M11, -m.M12,  m.M13, -m.M14,
                -m.M21,  m.M22, -m.M23,  m.M24,
                 m.M31, -m.M32,  m.M33, -m.M34,
                -m.M41,  m.M42, -m.M43,  m.M44
            );
        }

        private void BlendBone(int boneIndex, Bone bone, ref BonePose temp, out Matrix4x4 matrix)
        {
            // When retargeting, find the source bone to sample from the animation clip.
            // The clip references source bone names, not target bone names.
            int sourceBoneIndex = -1;
            Bone sourceBone = bone;

            if (_boneMap != null)
            {
                sourceBoneIndex = _boneMap.TargetToSource[boneIndex];
                if (sourceBoneIndex < 0)
                {
                    // Unmapped bone — hold bind pose
                    var bp = bone.BindPose;
                    matrix = Matrix4x4.CreateScale(bp.Scale)
                           * Matrix4x4.CreateFromQuaternion(bp.Rotation)
                           * Matrix4x4.CreateTranslation(bp.Position);
                    return;
                }
                sourceBone = _boneMap.Source.Bones[sourceBoneIndex];
            }

            BonePose blendPose = sourceBone.BindPose;

            if (Animation != null)
            {
                int layerCount = Animation.Layers.Count;
                for (int i = 0; i < layerCount; i++)
                {
                    var layer = Animation.Layers[i];

                    if (layer.Mask != null && !layer.Mask.Contains(sourceBone.Name))
                        continue;

                    int stateCount = layer.GetStateCount(Playback);

                    for (int st = 0; st < stateCount; st++)
                    {
                        var state = layer.GetState(st, Playback);
                        state?.BlendBone(sourceBone, ref blendPose, ref temp, Playback);
                    }
                }
            }

            // Apply retarget correction
            if (_boneMap != null && sourceBoneIndex >= 0)
            {
                // Use the precomputed model-space retarget rotation
                blendPose.Rotation = _retargetedLocalRot[boneIndex];

                // Position: transfer root delta, keep bind for all others
                if (sourceBoneIndex == _boneMap.SourceRoot)
                {
                    var srcDelta = blendPose.Position - sourceBone.BindPose.Position;
                    blendPose.Position = bone.BindPose.Position + srcDelta * _boneMap.PositionScale;
                }
                else
                {
                    blendPose.Position = bone.BindPose.Position;
                }
            }

            var scale = Matrix4x4.CreateScale(blendPose.Scale);
            var rotation = Matrix4x4.CreateFromQuaternion(blendPose.Rotation);
            var translation = Matrix4x4.CreateTranslation(blendPose.Position);
            matrix = scale * rotation * translation;
        }
    }
}
