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
        private bool _diagDone;

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
            
            // DIAG: capture after blend, before hierarchy
            Matrix4x4 diagAfterBlend = count > 1 ? bones[1] : default;

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

            // DIAG: capture after hierarchy, before offset
            Matrix4x4 diagAfterHier = count > 1 ? bones[1] : default;

            // Post-hierarchy callback — bones are now in model space
            OnPostHierarchy?.Invoke(skeleton, bones);

            // apply bone offset matrix (match Apex exactly)
            for (int i = 0; i < count; i++)
                bones[i] = Matrix4x4.Transpose(skeleton[i].OffsetMatrix * bones[i]);

            // DIAG: Dump bone data at each stage
            if (!_diagDone && count > 1 && Animation?.Layers.Count > 0)
            {
                _diagDone = true;
                var lines = new System.Collections.Generic.List<string>();
                lines.Add($"=== HIPS (bone[1] '{skeleton[1].Name}') ===");
                lines.Add($"After BlendBone (S*R*T matrix):");
                lines.Add($"  [{diagAfterBlend.M11:F4},{diagAfterBlend.M12:F4},{diagAfterBlend.M13:F4},{diagAfterBlend.M14:F4}]");
                lines.Add($"  [{diagAfterBlend.M21:F4},{diagAfterBlend.M22:F4},{diagAfterBlend.M23:F4},{diagAfterBlend.M24:F4}]");
                lines.Add($"  [{diagAfterBlend.M31:F4},{diagAfterBlend.M32:F4},{diagAfterBlend.M33:F4},{diagAfterBlend.M34:F4}]");
                lines.Add($"  [{diagAfterBlend.M41:F4},{diagAfterBlend.M42:F4},{diagAfterBlend.M43:F4},{diagAfterBlend.M44:F4}]");
                lines.Add($"After Hierarchy (bones[1] * bones[0]):");
                lines.Add($"  [{diagAfterHier.M11:F4},{diagAfterHier.M12:F4},{diagAfterHier.M13:F4},{diagAfterHier.M14:F4}]");
                lines.Add($"  [{diagAfterHier.M21:F4},{diagAfterHier.M22:F4},{diagAfterHier.M23:F4},{diagAfterHier.M24:F4}]");
                lines.Add($"  [{diagAfterHier.M31:F4},{diagAfterHier.M32:F4},{diagAfterHier.M33:F4},{diagAfterHier.M34:F4}]");
                lines.Add($"  [{diagAfterHier.M41:F4},{diagAfterHier.M42:F4},{diagAfterHier.M43:F4},{diagAfterHier.M44:F4}]");
                lines.Add($"After Offset+Transpose (FINAL):");
                var fm = bones[1];
                lines.Add($"  [{fm.M11:F4},{fm.M12:F4},{fm.M13:F4},{fm.M14:F4}]");
                lines.Add($"  [{fm.M21:F4},{fm.M22:F4},{fm.M23:F4},{fm.M24:F4}]");
                lines.Add($"  [{fm.M31:F4},{fm.M32:F4},{fm.M33:F4},{fm.M34:F4}]");
                lines.Add($"  [{fm.M41:F4},{fm.M42:F4},{fm.M43:F4},{fm.M44:F4}]");
                var bp = skeleton[1].BindPose;
                lines.Add($"BindPose: P=({bp.Position.X:F6},{bp.Position.Y:F6},{bp.Position.Z:F6})");
                lines.Add($"ScaleFactor={skeleton[1].ScaleFactor:F6}");
                System.IO.File.WriteAllLines(@"d:\Projects\2026\Freefall\.tmp\hips.txt", lines);
            }
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
