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
        }

        public void Draw()
        {
            if(!Enabled) return;

            if (Mesh == null || Entity?.Transform == null) return;
            if (Materials == null || Materials.Count == 0) return;

            var slot = Entity.Transform.TransformSlot;
            
            // Update TransformBuffer with world matrix (apply root rotation if set)
            if (slot >= 0)
            {
                var world = Entity.Transform.WorldMatrix;
                if (!Mesh.RootRotation.IsIdentity)
                    world = Mesh.RootRotation * world;
                TransformBuffer.Instance.SetTransform(slot, world);
            }

            // Enqueue skinned draw command with MaterialBlock
            if (Engine.FrameIndex % 60 == 0)
            {
                var boneCount = boneMatrices?.Length ?? 0;
                var effectName = Materials.Count > 0 ? Materials[0]?.Effect?.Name ?? "null" : "no-mat";
                var passCount = Materials.Count > 0 ? Materials[0]?.GetPasses()?.Count ?? 0 : 0;
                Debug.Log($"[SkinnedMesh] Entity={Entity?.Name} Parts={Mesh.MeshParts.Count} Bones={boneCount} Effect={effectName} Passes={passCount} Slot={slot}");
            }
            for (int i = 0; i < Mesh.MeshParts.Count; i++)
            {
                if (!Mesh.MeshParts[i].Enabled) continue;
                
                // Get material for this part
                Material? material = null;
                if (i < Materials.Count) material = Materials[i];
                else if (Materials.Count > 0) material = Materials[0];

                if (material != null)
                {
                    // Use pass-inferring Enqueue: Effect defines which passes (Opaque, Shadow, etc.)
                    CommandBuffer.Enqueue(
                        Mesh, 
                        i, 
                        material, 
                        _materialBlock,
                        slot
                    );
                }
            }
        }
    }
}
