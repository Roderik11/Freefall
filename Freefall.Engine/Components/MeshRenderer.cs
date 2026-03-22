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
    /// Sparse per-slot material override. Only specified slots get a material;
    /// unspecified slots fall back to the default Material, or null (invisible).
    /// </summary>
    [Serializable]
    public class MaterialOverride
    {
        public int MaterialSlot;
        public Material Material;
    }

    public class MeshRenderer : Component, IDraw, IParallel
    {
        public Mesh? Mesh;

        /// <summary>
        /// Default material applied to all MeshParts (unless overridden).
        /// </summary>
        public Material? Material;

        /// <summary>
        /// Sparse per-slot material overrides. Only the slots that differ
        /// from the default need entries. MeshParts whose slot has no override
        /// and no default Material are invisible.
        /// </summary>
        public List<MaterialOverride> Materials = [];

        public MaterialBlock Params = new();
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
        /// Resolve material for a MeshPart by its MaterialSlot.
        /// Checks sparse overrides first, falls back to default Material.
        /// </summary>
        private Material? GetMaterial(int materialSlot)
        {
            if (Materials != null)
            {
                for (int i = 0; i < Materials.Count; i++)
                {
                    if (Materials[i].MaterialSlot == materialSlot)
                        return Materials[i].Material;
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
                var indices = Mesh.LODs[lod].MeshPartIndices;

                for (int i = 0; i < indices.Length; i++)
                {
                    int partIdx = indices[i];
                    if (partIdx >= Mesh.MeshParts.Count) continue;
                    var mat = GetMaterial(Mesh.MeshParts[partIdx].MaterialSlot);
                    if (mat != null)
                        CommandBuffer.Enqueue(Mesh, partIdx, mat, Params, slot);
                }

                // Also render non-LOD parts that have material overrides
                // (e.g., tiles in a mixed FBX). Skip parts in ANY LOD group.
                if (Materials != null && Materials.Count > 0)
                {
                    var allLodParts = new HashSet<int>();
                    foreach (var lodLevel in Mesh.LODs)
                        if (lodLevel.MeshPartIndices != null)
                            foreach (var idx in lodLevel.MeshPartIndices)
                                allLodParts.Add(idx);

                    for (int i = 0; i < Mesh.MeshParts.Count; i++)
                    {
                        if (allLodParts.Contains(i)) continue;
                        var mat = GetMaterial(Mesh.MeshParts[i].MaterialSlot);
                        if (mat != null)
                            CommandBuffer.Enqueue(Mesh, i, mat, Params, slot);
                    }
                }
            }
            else
            {
                // No LODs — render all parts
                for (int i = 0; i < Mesh.MeshParts.Count; i++)
                {
                    var mat = GetMaterial(Mesh.MeshParts[i].MaterialSlot);
                    if (mat != null)
                        CommandBuffer.Enqueue(Mesh, i, mat, Params, slot);
                }
            }
        }
    }
}
