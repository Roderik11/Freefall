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
    public class SkinnedMeshRenderer : Component, IUpdate, IDraw, IParallel
    {
        public Mesh? Mesh;
        // Materials list for multi-material support
        public List<Material> Materials { get; set; } = new List<Material>();

        private Animator? animator;
        private Matrix4x4[]? boneMatrices;
        private MaterialBlock _materialBlock = new MaterialBlock();
        
        // GPU persistent transform slot
        private int _transformSlot = -1;
        
        private bool _initialized = false;
        
        private void EnsureInitialized()
        {
            if (_initialized) return;
            if (Mesh?.Bones == null) return;
            
            _initialized = true;
            animator = Entity?.GetComponent<Animator>();
            boneMatrices = new Matrix4x4[Mesh.Bones.Length];
        }

        public void Update()
        {
            if(!Enabled) return;
            
            if (Mesh == null || Entity?.Transform == null) return;
            
            EnsureInitialized();

            // Calculate bone poses in Update (logic phase)
            if (Mesh.Bones != null && animator != null)
            {
                if (boneMatrices == null || boneMatrices.Length != Mesh.Bones.Length)
                    boneMatrices = new Matrix4x4[Mesh.Bones.Length];

                animator.GetPose(Mesh.Bones, boneMatrices);
                
                // Store in MaterialBlock for batched upload during rendering
                _materialBlock.SetParameterArray("Bones", boneMatrices);
            }
            else if (Mesh.Bones != null)
            {
                // Fallback: use identity if no animator
                if (boneMatrices == null || boneMatrices.Length != Mesh.Bones.Length)
                    boneMatrices = new Matrix4x4[Mesh.Bones.Length];
                    
                for (int b = 0; b < boneMatrices.Length; b++)
                    boneMatrices[b] = Matrix4x4.Identity;
                    
                _materialBlock.SetParameterArray("Bones", boneMatrices);
            }
            
            // Store world matrix (apply root rotation if set)
            var world = Entity.Transform.WorldMatrix;
            if (!Mesh.RootRotation.IsIdentity)
            {
                world = Mesh.RootRotation * world;
            }
            _materialBlock.SetParameter("World", world);
        }

        public void Draw()
        {
            if(!Enabled) return;

            if (Mesh == null || Entity?.Transform == null) return;
            if (Materials == null || Materials.Count == 0) return;

            // Allocate transform slot on first use
            if (_transformSlot < 0 && TransformBuffer.Instance != null)
            {
                _transformSlot = TransformBuffer.Instance.AllocateSlot();
            }
            
            // Update TransformBuffer with world matrix (apply root rotation if set)
            if (_transformSlot >= 0)
            {
                var world = Entity.Transform.WorldMatrix;
                if (!Mesh.RootRotation.IsIdentity)
                    world = Mesh.RootRotation * world;
                TransformBuffer.Instance.SetTransform(_transformSlot, world);
            }

            // Enqueue skinned draw command with MaterialBlock
            for (int i = 0; i < Mesh.MeshParts.Count; i++)
            {
                if (!Mesh.MeshParts[i].Enabled) continue;
                
                // Get material for this part
                Material? material = null;
                if (i < Materials.Count) material = Materials[i];
                else if (Materials.Count > 0) material = Materials[0];

                if (material != null)
                {
                    CommandBuffer.Enqueue(
                        RenderPass.Opaque, 
                        Mesh, 
                        i, 
                        material, 
                        _materialBlock,
                        _transformSlot
                    );
                }
            }
        }
    }
}
