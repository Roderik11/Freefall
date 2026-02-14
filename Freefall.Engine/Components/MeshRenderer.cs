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
        public MaterialBlock Params = new MaterialBlock();
        private int _sceneSlot = -1;

        protected override void Awake()
        {
            _sceneSlot = SceneBuffers.AllocateSlot();
            Entity.Transform.OnChanged += OnTransformChanged;
            OnTransformChanged();
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
            var world = Entity.Transform.WorldMatrix;

            var slot = Entity.Transform.TransformSlot;
            if (slot >= 0)
                TransformBuffer.Instance.SetTransform(slot, world);

            if (_sceneSlot >= 0)
                SceneBuffers.Transforms.Set(_sceneSlot, Matrix4x4.Transpose(world));
        }

        public void Draw()
        {
            if (Mesh == null) return;
            if (Transform == null) return;

            var slot = Entity.Transform.TransformSlot;

            // Use per-part materials if available
            if (Materials.Count > 0)
            {
                for (int i = 0; i < Mesh.MeshParts.Count; i++)
                {
                    var mat = i < Materials.Count ? Materials[i] : Materials[Materials.Count - 1];
                    if (mat != null)
                        CommandBuffer.Enqueue(Mesh, i, mat, Params, slot);
                }
            }
            else if (Material != null)
            {
                // Single material for all parts
                for (int i = 0; i < Mesh.MeshParts.Count; i++)
                    CommandBuffer.Enqueue(Mesh, i, Material, Params, slot);
            }
        }
    }
}
