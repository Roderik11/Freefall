using System;
using System.Numerics;
using System.Collections.Generic;
using Freefall.Graphics;
using Freefall.Base;
using Vortice.Direct3D12;

namespace Freefall.Components
{
    public class MeshRenderer : Component, IDraw, IParallel
    {
        public Mesh? Mesh { get; set; }
        public Material? Material { get; set; }
        public List<Material> Materials { get; set; } = new List<Material>();
        protected MaterialBlock Params = new MaterialBlock();



        public void Draw()
        {
            if (Mesh == null) return;
            if (Transform == null) return;

            var slot = Entity.Transform.TransformSlot;

            // Update TransformBuffer directly (GPU path)
            if (slot >= 0)
                TransformBuffer.Instance.SetTransform(slot, Entity.Transform.WorldMatrix);


            // Use per-part materials if available
            if (Materials.Count > 0)
            {
                for (int i = 0; i < Mesh.MeshParts.Count; i++)
                {
                    var mat = i < Materials.Count ? Materials[i] : Materials[Materials.Count - 1];
                    if (mat != null)
                    {
                        CommandBuffer.Enqueue(Mesh, i, mat, Params, slot);
                    }
                }
            }
            else if (Material != null)
            {
                // Single material for all parts
                for (int i = 0; i < Mesh.MeshParts.Count; i++)
                {
                    CommandBuffer.Enqueue(Mesh, i, Material, Params, slot);
                }
            }
        }

        public override void Destroy()
        {
             // Resource cleanup logic if component owns resources
        }
    }
}
