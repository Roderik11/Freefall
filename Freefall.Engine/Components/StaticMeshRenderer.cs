using System;
using System.Numerics;
using System.Collections.Generic;
using Freefall.Assets;
using static Freefall.Assets.InternalAssets;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.Components
{
    public class StaticMeshRenderer : Component, IDraw, IParallel
    {
        public StaticMesh StaticMesh;
        public MaterialBlock Params;
        public BoundingSphere BoundingSphere;

        private Vector3[] points = new Vector3[8];

        const float MinDrawSizeSq = 0.01f * 0.01f;

        public StaticMeshRenderer()
        {
            Params = new MaterialBlock();
        }

        protected override void Awake()
        {
            OnTransformChanged();
            Transform.OnChanged += OnTransformChanged;
        }

        public override void Destroy()
        {
            Transform.OnChanged -= OnTransformChanged;
        }

        void OnTransformChanged()
        {
            var mesh = StaticMesh.Mesh;
            if (mesh == null) return;

            mesh.BoundingBox.GetCorners(points, mesh.RootRotation * Transform.WorldMatrix);
            BoundingSphere = BoundingSphere.CreateFromPoints(points);
        }

        private IStaticMesh GetMesh()
        {
            if (StaticMesh == null) return null;
            if (StaticMesh.LODs.Count == 0) return StaticMesh;
            
            var cam = Camera.Main;
            if (cam == null) return StaticMesh;
            if (StaticMesh.LODGroup == null) return StaticMesh;

            var lodgroup = StaticMesh.LODGroup;
            float diameter = BoundingSphere.Radius;
            float distanceSq = Vector3.DistanceSquared(Transform.Position, cam.Position);
            float sizeSq = (diameter * diameter / MathF.Max(distanceSq, 0.001f)) * cam.FoVFactor;
            int min = Math.Min(lodgroup.Ranges.Count, StaticMesh.LODs.Count);

            for (int i = 0; i < min; i++)
            {
                float t = lodgroup.Ranges[i];
                if (sizeSq > t * t)
                    return (i == 0) ? StaticMesh : StaticMesh.LODs[i - 1];
            }

            // no LODs, cull it early (note the squared threshold)
            if (sizeSq < MinDrawSizeSq) return null;

            return (min > 0) ? StaticMesh.LODs[min - 1] : StaticMesh;
        }

        // Draw method called by Renderer or System
        public void Draw()
        {
             IStaticMesh lod = GetMesh();
             if (lod == null || lod.Mesh == null) return;

             var mesh = lod.Mesh;
             var elements = lod.MeshParts;
             var slot = Entity.Transform.TransformSlot;
             
             // iterate through MeshParts and use element.Material
             if (elements != null && elements.Count > 0)
             {
                 foreach (var element in elements)
                 {
                     // Enqueue into all passes defined by the Effect
                     if (element.Material != null)
                         CommandBuffer.Enqueue(mesh, element.MeshPartIndex, element.Material, Params, slot);
                 }
             }
        }
    }
}
