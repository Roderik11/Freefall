using System;
using System.Numerics;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Graphics;
using Freefall.Animation;
using Vortice.Direct3D12;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// Renders skinned/animated meshes with bone transforms.
    /// Uses bindless instancing with a shared bone matrix buffer.
    /// </summary>
    public class SkinnedMeshRenderer : Component, IDraw, IParallel
    {
        public Mesh? Mesh;
        public List<Material> Materials = new List<Material>();
        public MaterialBlock Params = new MaterialBlock();

        private Animator? animator;
        private Matrix4x4[]? boneMatrices = new Matrix4x4[4];
        private int _sceneSlot = -1;
        
        protected override void Awake()
        {
            animator = Entity?.GetComponent<Animator>();

            // Allocate a SceneBuffers slot for future GPU-driven path
            _sceneSlot = SceneBuffers.AllocateSlot();

            // Subscribe to event-driven transform updates
            if (Entity?.Transform != null)
            {
                Entity.Transform.OnChanged += OnTransformChanged;
                OnTransformChanged();
            }
        }

        public override void Destroy()
        {
            if (Entity?.Transform != null)
                Entity.Transform.OnChanged -= OnTransformChanged;

            if (_sceneSlot >= 0)
            {
                SceneBuffers.ReleaseSlot(_sceneSlot);
                _sceneSlot = -1;
            }
        }

        private void OnTransformChanged()
        {
            if (Mesh == null || Entity?.Transform == null) return;

            var world = Entity.Transform.WorldMatrix;
            if (!Mesh.RootRotation.IsIdentity)
                world = Mesh.RootRotation * world;

            // Write to legacy TransformBuffer (current pipeline reads from this)
            var slot = Entity.Transform.TransformSlot;
            if (slot >= 0)
                TransformBuffer.Instance.SetTransform(slot, world);

            // Write to SceneBuffers (new pipeline)
            if (_sceneSlot >= 0)
                SceneBuffers.Transforms.Set(_sceneSlot, Matrix4x4.Transpose(world));
        }

        // Draw still needed for per-frame bone computation + CommandBuffer.Enqueue
        public void Draw()
        {
            if(!Enabled) return;
            if (Mesh?.Bones == null || Entity?.Transform == null) return;
            if (Materials == null || Materials.Count == 0) return;

            // Calculate bone poses (animated data — changes every frame)
            if (animator != null)
            {
                if (boneMatrices.Length != Mesh.Bones.Length)
                    Array.Resize(ref boneMatrices, Mesh.Bones.Length);

                animator.GetPose(Mesh.Bones, boneMatrices);
                
                // Store in MaterialBlock for batched upload during rendering
                Params.SetParameterArray("Bones", boneMatrices);
            }

            var slot = Entity.Transform.TransformSlot;
            
            for (int i = 0; i < Mesh.MeshParts.Count; i++)
            {
                if (!Mesh.MeshParts[i].Enabled) continue;
                var material = i < Materials.Count ? Materials[i] : Materials[0];
                CommandBuffer.Enqueue(Mesh, i, material, Params, slot);
            }
        }
    }
}
