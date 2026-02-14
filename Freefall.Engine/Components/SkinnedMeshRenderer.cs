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
        
        protected override void Awake()
        {
            animator = Entity?.GetComponent<Animator>();
        }

        public void Draw()
        {
            if(!Enabled) return;
            if (Mesh?.Bones == null || Entity?.Transform == null) return;
            if (Materials == null || Materials.Count == 0) return;

            // Calculate bone poses in Update (logic phase)
            if (animator != null)
            {
                if (boneMatrices.Length != Mesh.Bones.Length)
                    Array.Resize(ref boneMatrices, Mesh.Bones.Length);

                animator.GetPose(Mesh.Bones, boneMatrices);
                
                // Store in MaterialBlock for batched upload during rendering
                Params.SetParameterArray("Bones", boneMatrices);
            }

            var slot = Entity.Transform.TransformSlot;
            
            // Update TransformBuffer with world matrix (apply root rotation if set)
            if (slot >= 0)
            {
                var world = Entity.Transform.WorldMatrix;
                if (!Mesh.RootRotation.IsIdentity)
                    world = Mesh.RootRotation * world;
                TransformBuffer.Instance.SetTransform(slot, world);
            }

            for (int i = 0; i < Mesh.MeshParts.Count; i++)
            {
                if (!Mesh.MeshParts[i].Enabled) continue;
                var material = i < Materials.Count ? Materials[i] : Materials[0];
                CommandBuffer.Enqueue(Mesh, i, material, Params, slot);
            }
        }
    }
}
