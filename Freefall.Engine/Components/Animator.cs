using System;
using System.Numerics;
using Freefall.Animation;
using Freefall.Base;

namespace Freefall.Components
{
    public delegate void BoneTransformHandler(Bone bone, ref Matrix4x4 matrix);
    public delegate void PostHierarchyHandler(Bone[] skeleton, Matrix4x4[] bones);

    /// <summary>
    /// Controls animation playback for skinned meshes using an animation state machine.
    /// Calculates bone matrices that are uploaded to the GPU for vertex skinning.
    /// </summary>
    public class Animator : Component, IUpdate, IParallel
    {
        /// <summary>The animation state machine containing layers, states, and transitions.</summary>
        public Animation.Animation Animation;

        /// <summary>Event fired when a bone transform is calculated (for custom modifications).</summary>
        public event BoneTransformHandler OnBoneTransform;

        /// <summary>Event fired after hierarchy multiplication — bones are in model space.</summary>
        public event PostHierarchyHandler OnPostHierarchy;

        /// <summary>Event fired when an animation event occurs (e.g., footstep, jump).</summary>
        public Action<string> OnAnimationEvent;
        
        private bool _debugPrinted = false;

        public void Update()
        {
            if (Animation == null) return;

            foreach (AnimationLayer layer in Animation.Layers)
                layer.Update(this);
        }

        public float GetParam(string name) => Animation?.GetParam(name) ?? 0f;

        public void SetParam(string name, float value) => Animation?.SetParam(name, value);

        internal void FireAnimationEvent(string name) => OnAnimationEvent?.Invoke(name);

        /// <summary>
        /// Calculates final bone matrices for GPU skinning.
        /// </summary>
        /// <param name="skeleton">The mesh's skeleton bones.</param>
        /// <param name="boneMatrices">Output array of bone matrices (must match skeleton length).</param>
        public void GetPose(Bone[] skeleton, Matrix4x4[] bones)
        {
            int count = skeleton.Length;

            // Blend poses from all active animation states
            BonePose tempPose = new BonePose { Scale = Vector3.One, Rotation = Quaternion.Identity };
            for (int i = 0; i < count; i++)
                BlendBone(skeleton[i], ref tempPose, out bones[i]);
            
            // DEBUG: Print comparison once
            if (!_debugPrinted && Animation?.Layers.Count > 0)
            {
                _debugPrinted = true;
                Debug.Log("[Animator] Final bone matrix debug (bone 0 Hips):");
                // After all processing, bones[0] should contain the final matrix
                // Let's print what we compute for bone 0 now
                Debug.Log($"  BindPose: Pos={skeleton[0].BindPose.Position}, Rot={skeleton[0].BindPose.Rotation}");
                Debug.Log($"  OffsetMatrix row0: {skeleton[0].OffsetMatrix.M11}, {skeleton[0].OffsetMatrix.M12}, {skeleton[0].OffsetMatrix.M13}, {skeleton[0].OffsetMatrix.M14}");
                Debug.Log($"  OffsetMatrix row3 (trans): {skeleton[0].OffsetMatrix.M41}, {skeleton[0].OffsetMatrix.M42}, {skeleton[0].OffsetMatrix.M43}, {skeleton[0].OffsetMatrix.M44}");
            }

            // Apply custom bone transforms
            if (OnBoneTransform != null)
            {
                for (int i = 0; i < count; i++)
                    OnBoneTransform(skeleton[i], ref bones[i]);
            }

            // Multiply hierarchy (child * parent)
            for (int i = 0; i < count; i++)
            {
                if (skeleton[i].Parent > -1)
                    bones[i] = bones[i] * bones[skeleton[i].Parent];
            }

            // Post-hierarchy callback — bones are now in model space
            OnPostHierarchy?.Invoke(skeleton, bones);

            // apply bone offset matrix (match Apex exactly)
            for (int i = 0; i < count; i++)
                bones[i] = Matrix4x4.Transpose(skeleton[i].OffsetMatrix * bones[i]);
        }

        private void BlendBone(Bone bone, ref BonePose temp, out Matrix4x4 matrix)
        {
            BonePose blendPose = bone.BindPose;

            if (Animation != null)
            {
                int layerCount = Animation.Layers.Count;
                for (int i = 0; i < layerCount; i++)
                {
                    var layer = Animation.Layers[i];

                    // Masking - skip bones not in mask
                    if (layer.Mask != null && !layer.Mask.Contains(bone.Name))
                        continue;

                    int stateCount = layer.GetStateCount();

                    for (int st = 0; st < stateCount; st++)
                    {
                        var state = layer.GetState(st);
                        state?.BlendBone(bone, ref blendPose, ref temp);
                    }
                }
            }

            var p = blendPose.Position * bone.ScaleFactor;

            var scale = Matrix4x4.CreateScale(blendPose.Scale);
            var rotation = Matrix4x4.CreateFromQuaternion(blendPose.Rotation);
            var translation = Matrix4x4.CreateTranslation(p);
            matrix = scale * rotation * translation;
        }
    }
}
