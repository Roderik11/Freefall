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
    public class StaticMeshRenderer : Component, IUpdate, IDraw, IParallel  // Re-enabled for parallel test
    {
        const float MinDrawSizeSq = 0.01f * 0.01f;
        public StaticMesh StaticMesh;

        public MaterialBlock Params;

        private Vector3 worldPosition;
        private Vector3[] points = new Vector3[8];
        public BoundingBox _boundingBox;
        public BoundingSphere _boundingSphere;
        
        // GPU persistent transform slot
        private int _transformSlot = -1;

        public BoundingBox BoundingBox => _boundingBox;
        public BoundingSphere BoundingSphere => _boundingSphere;

        // private static readonly int worldMatrixHash = "World".GetHashCode(); // Not using hash yet, passing string

        public StaticMeshRenderer()
        {
            Params = new MaterialBlock();
        }

        public Texture AlbedoTexture { get; set; }

        private IStaticMesh GetMesh()
        {
            if (StaticMesh == null) return null;
            if (StaticMesh.LODs.Count == 0) return StaticMesh;
            
            var cam = Camera.Main;
            if (cam == null) return StaticMesh;
            if (StaticMesh.LODGroup == null) return StaticMesh;

            var distSq = Vector3.DistanceSquared(worldPosition, cam.Position);
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

        public void Update()
        {
             // Update world position for LOD calculation
             if (Entity?.Transform != null)
             {
                 worldPosition = Entity.Transform.Position;
             }
        }

        public bool UseIndirectDrawing { get; set; } = false;

        // Draw method called by Renderer or System
        public void Draw()
        {
             var lod = GetMesh();
             if (lod == null || lod.Mesh == null) return;

             var world = Entity.Transform.WorldMatrix;
             var mesh = lod.Mesh;
             var elements = lod.MeshParts;
             
             // Allocate transform slot on first use
             if (_transformSlot < 0 && TransformBuffer.Instance != null)
             {
                 _transformSlot = TransformBuffer.Instance.AllocateSlot();
             }
             
             // Update TransformBuffer directly (GPU path)
             if (_transformSlot >= 0)
             {
                 TransformBuffer.Instance.SetTransform(_transformSlot, world);
             }
             

             // Like Apex: iterate through MeshParts and use element.Material
             if (elements != null && elements.Count > 0)
             {
                 foreach (var element in elements)
                 {
                     // Enqueue into all passes defined by the Effect (Spark pattern)
                     if (element.Material != null)
                     {
                         CommandBuffer.Enqueue(mesh, element.MeshPartIndex, element.Material, Params, _transformSlot);
                     }
                 }
             }
             else
             {
                 // Fallback: draw all parts using AlbedoTexture if set
                 // Cache the fallback material to avoid creating new ones each frame
                 if (_cachedFallbackMaterial == null)
                 {
                     if (AlbedoTexture != null)
                     {
                         _cachedFallbackMaterial = new Material(InternalAssets.DefaultEffect);
                         _cachedFallbackMaterial.SetTexture("AlbedoTex", AlbedoTexture);
                         _cachedFallbackMaterial.SetTexture("NormalTex", InternalAssets.FlatNormal);
                     }
                     else
                     {
                         _cachedFallbackMaterial = InternalAssets.DefaultMaterial;
                     }
                 }
                 for (int i = 0; i < mesh.MeshParts.Count; i++)
                 {
                     CommandBuffer.Enqueue(mesh, i, _cachedFallbackMaterial, Params, _transformSlot);
                 }
             }
        }
        
        private Material _cachedFallbackMaterial;
    }
}
