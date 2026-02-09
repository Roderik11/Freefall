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

        public BoundingBox _boundingBox;
        public BoundingSphere _boundingSphere;
        
        public BoundingBox BoundingBox => _boundingBox;
        public BoundingSphere BoundingSphere => _boundingSphere;

        private Vector3[] points = new Vector3[8];

        public StaticMeshRenderer()
        {
            Params = new MaterialBlock();
        }

        private IStaticMesh GetMesh()
        {
            if (StaticMesh == null) return null;
            if (StaticMesh.LODs.Count == 0) return StaticMesh;
            
            var cam = Camera.Main;
            if (cam == null) return StaticMesh;
            if (StaticMesh.LODGroup == null) return StaticMesh;

            var distSq = Vector3.DistanceSquared(Entity.Transform.Position, cam.Position);
            var ranges = StaticMesh.LODGroup.Ranges;

            // ranges[0] = base mesh → LODs[0] transition
            // ranges[1] = LODs[0] → LODs[1] transition, etc.
            int lodLevel = -1; // -1 = base mesh (highest detail)
            for (int i = 0; i < ranges.Count; i++)
            {
                float r = ranges[i];
                if (distSq >= r * r)
                    lodLevel = i;
                else
                    break;
            }

            if (lodLevel < 0)
                return StaticMesh; // Closest: use base mesh

            // Clamp to available LODs
            if (lodLevel >= StaticMesh.LODs.Count)
                lodLevel = StaticMesh.LODs.Count - 1;

            return StaticMesh.LODs[lodLevel];
        }

        // Draw method called by Renderer or System
        public void Draw()
        {
             var lod = GetMesh();
             if (lod == null || lod.Mesh == null) return;

             var mesh = lod.Mesh;
             var elements = lod.MeshParts;
             var slot = Entity.Transform.TransformSlot;
             
             // Update TransformBuffer directly (GPU path)
             if (slot >= 0)
                 TransformBuffer.Instance.SetTransform(slot, Entity.Transform.WorldMatrix);
             
             // Like Apex: iterate through MeshParts and use element.Material
             if (elements != null && elements.Count > 0)
             {
                 foreach (var element in elements)
                 {
                     // Enqueue into all passes defined by the Effect (Spark pattern)
                     if (element.Material != null)
                         CommandBuffer.Enqueue(mesh, element.MeshPartIndex, element.Material, Params, slot);
                 }
             }
        }
    }
}
