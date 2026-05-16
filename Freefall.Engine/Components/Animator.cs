using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Animation;
using Freefall.Base;

namespace Freefall.Components
{
    public delegate void BoneTransformHandler(Bone bone, ref Matrix4x4 matrix);
    public delegate void PostHierarchyHandler(Bone[] skeleton, Matrix4x4[] bones);

    /// <summary>
    /// Controls animation playback for skinned meshes using an animation state machine.
    /// Owns a per-instance AnimationPlayback blackboard for all runtime state.
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
            if (Animation == null) return;

            if (!_retargetInitialized)
                InitRetargeting();

            foreach (AnimationLayer layer in Animation.Layers)
                layer.Update(this, Playback);
        }

        // --- Retargeting ---

        private void InitRetargeting()
        {
            _retargetInitialized = true;

            if (RetargetSource == null) return;

            var renderer = Entity?.GetComponent<SkinnedMeshRenderer>();
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
