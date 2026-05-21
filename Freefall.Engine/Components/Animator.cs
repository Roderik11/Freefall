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
        private float[] _retargetFactors;
        private bool _retargetInitialized;

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
                    GetPose(bones, entry.StagingMatrices);
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

            if (RetargetSource == null) return;

            var renderer = Entity?.GetComponentInChildren<SkinnedMeshRenderer>();
            var meshSkeleton = renderer?.Mesh?.Skeleton;
            if (meshSkeleton == null) return;
            if (RetargetSource == meshSkeleton) return;

            int count = Math.Min(RetargetSource.Bones.Length, meshSkeleton.Bones.Length);
            _retargetFactors = new float[count];

            for (int i = 0; i < count; i++)
            {
                float srcLen = RetargetSource.Bones[i].BindPoseMatrix.Translation.Length();
                float dstLen = meshSkeleton.Bones[i].BindPoseMatrix.Translation.Length();
                _retargetFactors[i] = srcLen > 0 ? dstLen / srcLen : 1f;
            }

            Debug.Log($"[Animator] Retarget: {RetargetSource.Name} → {meshSkeleton.Name}, {count} bones");
        }

        // --- Pose computation ---

        public void GetPose(Bone[] skeleton, Matrix4x4[] bones)
        {
            int count = skeleton.Length;

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
                bones[i] = Matrix4x4.Transpose(skeleton[i].OffsetMatrix * bones[i]);
        }

        private void BlendBone(int boneIndex, Bone bone, ref BonePose temp, out Matrix4x4 matrix)
        {
            BonePose blendPose = bone.BindPose;

            if (Animation != null)
            {
                int layerCount = Animation.Layers.Count;
                for (int i = 0; i < layerCount; i++)
                {
                    var layer = Animation.Layers[i];

                    if (layer.Mask != null && !layer.Mask.Contains(bone.Name))
                        continue;

                    int stateCount = layer.GetStateCount(Playback);

                    for (int st = 0; st < stateCount; st++)
                    {
                        var state = layer.GetState(st, Playback);
                        state?.BlendBone(bone, ref blendPose, ref temp, Playback);
                    }
                }
            }

            float sf = (_retargetFactors != null && boneIndex < _retargetFactors.Length)
                ? _retargetFactors[boneIndex]
                : 1f;

            var p = blendPose.Position * sf;

            var scale = Matrix4x4.CreateScale(blendPose.Scale);
            var rotation = Matrix4x4.CreateFromQuaternion(blendPose.Rotation);
            var translation = Matrix4x4.CreateTranslation(p);
            matrix = scale * rotation * translation;
        }
    }
}
