using System;
using System.Numerics;
using System.Collections.Generic;
using Freefall.Assets;
using Freefall.Graphics;
using Freefall.Base;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// Per-MeshPart material override. Sparse — only the exceptions.
    /// </summary>
    [Serializable]
    public class MaterialOverride
    {
        //[Reflection.ValueSelect(typeof(Reflection.MeshPartProvider))]
        public int MeshPartIndex;
        public Material Material;
    }

    public class MeshRenderer : Component, IDraw, IParallel
    {
        public Mesh? Mesh { get; set; }

        /// <summary>
        /// Default material applied to all MeshParts.
        /// </summary>
        public Material? Material { get; set; }

        /// <summary>
        /// Sparse per-MeshPart material overrides. Only exceptions to the default.
        /// </summary>
        public List<MaterialOverride> MaterialOverrides { get; set; } = new List<MaterialOverride>();

        public MaterialBlock Params = new MaterialBlock();
        public BoundingSphere BoundingSphere;

        private bool _boundsDirty = true;

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
            if (Mesh == null) { _boundsDirty = true; return; }

            var corners = new Vector3[8];
            Mesh.BoundingBox.GetCorners(corners, Mesh.RootRotation * Transform.WorldMatrix);
            BoundingSphere = BoundingSphere.CreateFromPoints(corners);
            _boundsDirty = false;
        }

        /// <summary>
        /// Select active LOD index based on screen-relative size.
        /// Uses LODGroup threshold ranges, matching StaticMeshRenderer behavior.
        /// Returns -1 if no LOD chain (use all MeshParts).
        /// </summary>
        private int GetActiveLOD()
        {
            if (Mesh == null || Mesh.LODs.Count == 0) return -1;

            var cam = Camera.Main;
            if (cam == null) return 0;

            float distanceSq = Vector3.DistanceSquared(Transform.Position, cam.Position);
            if (distanceSq < 0.001f) return 0;

            float diameter = BoundingSphere.Radius;
            float sizeSq = (diameter * diameter / MathF.Max(distanceSq, 0.001f)) * cam.FoVFactor;
            sizeSq *= Engine.Settings.LODScale * Mesh.LODBias;

            // Use SmallProps LODGroup ranges as thresholds
            var ranges = LODGroups.SmallProps.Ranges;
            int lodCount = Mesh.LODs.Count;

            for (int i = 0; i < Math.Min(ranges.Count, lodCount); i++)
            {
                float t = ranges[i];
                if (sizeSq > t * t)
                    return i;
            }

            return lodCount - 1;
        }

        /// <summary>
        /// Resolve material for a given MeshPart index.
        /// Checks sparse overrides first, falls back to default Material.
        /// </summary>
        private Material? GetMaterial(int meshPartIndex)
        {
            if (MaterialOverrides != null && MaterialOverrides.Count > 0)
            {
                for (int i = 0; i < MaterialOverrides.Count; i++)
                {
                    if (MaterialOverrides[i].MeshPartIndex == meshPartIndex)
                        return MaterialOverrides[i].Material;
                }
            }
            return Material;
        }

        public void Draw()
        {
            if (Mesh == null) return;
            if (_boundsDirty) OnTransformChanged();

            var slot = Transform.TransformSlot;
            int lod = GetActiveLOD();

            if (lod >= 0 && Mesh.LODs[lod].MeshPartIndices != null)
            {
                // LOD-selected parts
                // Material overrides reference LOD0 part indices; map to equivalent slot in current LOD
                var indices = Mesh.LODs[lod].MeshPartIndices;
                var lod0Indices = Mesh.LODs[0].MeshPartIndices;
                for (int i = 0; i < indices.Length; i++)
                {
                    int partIdx = indices[i];
                    // Map: override targets LOD0 part → find same slot position in current LOD
                    int lod0PartIdx = (lod0Indices != null && i < lod0Indices.Length) ? lod0Indices[i] : partIdx;
                    var mat = GetMaterial(lod0PartIdx);
                    if (mat != null && partIdx < Mesh.MeshParts.Count)
                        CommandBuffer.Enqueue(Mesh, partIdx, mat, Params, slot);
                }
            }
            else
            {
                // No LODs — render all parts
                for (int i = 0; i < Mesh.MeshParts.Count; i++)
                {
                    var mat = GetMaterial(i);
                    if (mat != null)
                        CommandBuffer.Enqueue(Mesh, i, mat, Params, slot);
                }
            }
        }
    }
}
