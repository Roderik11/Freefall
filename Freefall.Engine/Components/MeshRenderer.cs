using System;
using System.Numerics;
using System.Collections.Generic;
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


    [Icon("icon_mesh.png")]
    public class MeshRenderer : Component, IDraw, IParallel
    {
        public Mesh? Mesh;

        /// <summary>
        /// Default material applied to all MeshParts (unless overridden).
        /// When null, only explicit MaterialOverrides render (mixed-mesh mode).
        /// </summary>
        public Material? Material;

        /// <summary>
        /// Sparse per-slot material overrides. Only the slots that differ
        /// from the default need entries. MeshParts whose slot has no override
        /// and no default Material are invisible.
        /// </summary>
        public List<MaterialOverride> Materials = [];

        /// <summary>
        /// When set, overrides ALL material routing (Material + Materials).
        /// Used for placement ghost mode. Not serialized.
        /// </summary>
        [NonSerialized]
        public Material? ReplacementMaterial;

        public MaterialBlock Params = new();
        [NonSerialized]
        public BoundingSphere BoundingSphere;

        private bool _boundsDirty = true;
        private Mesh? _boundsMesh; // tracks which mesh instance bounds were computed from
        private int _boundsVersion; // Mesh.GeometryVersion the bounds were computed from (hot reload)
        private Vector3[] _boundsCorners = new Vector3[8];

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

            Mesh.BoundingBox.GetCorners(_boundsCorners, Mesh.RootRotation * Transform.WorldMatrix);
            BoundingSphere = BoundingSphere.CreateFromPoints(_boundsCorners);
            _boundsMesh = Mesh;
            _boundsVersion = Mesh.GeometryVersion;
            _boundsDirty = false;
        }

        /// <summary>
        /// Resolve material for a MeshPart by its MaterialSlot.
        /// Checks sparse overrides first, falls back to default Material.
        /// When Material is null, unmatched slots are invisible (mixed-mesh mode).
        /// </summary>
        private Material? GetMaterial(int materialSlot)
        {
            if (Materials != null)
            {
                for (int i = 0; i < Materials.Count; i++)
                {
                    if (Materials[i].MaterialSlot == materialSlot)
                        return ReplacementMaterial ?? Materials[i].Material;
                }
            }

            return Material != null ? ReplacementMaterial ?? Material : null;
        }

        public void Draw()
        {
            if (!Enabled) return;
            if (Mesh == null) return;

            // Re-dirty bounds when the mesh reference changes (stub → loaded)
            if (Mesh != _boundsMesh || Mesh.GeometryVersion != _boundsVersion) _boundsDirty = true;

            if (_boundsDirty) OnTransformChanged();

            if (Mesh.IsBelowCullSize(BoundingSphere)) return;

            var slot = Transform.TransformSlot;

            // LOD chain heads + non-LOD parts. The GPU culler picks the LOD and culls by screen size.
            var parts = Mesh.DrawPartIndices;
            var meshParts = Mesh.MeshParts;
            for (int i = 0; i < parts.Length; i++)
            {
                int partIdx = parts[i];
                if (partIdx >= meshParts.Count) continue;
                var mat = GetMaterial(meshParts[partIdx].MaterialSlot);
                if (mat != null)
                    CommandBuffer.Enqueue(Mesh, partIdx, mat, Params, slot, lodManaged: true);
            }
        }
    }
}
