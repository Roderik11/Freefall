using System;
using System.Collections.Generic;
using Freefall.Graphics;
using Freefall.Base;
using Freefall.Reflection;

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


    /// <summary>
    /// Draws a static mesh. The draws are GPU-resident (see PersistentRenderer): nothing runs per frame.
    ///
    /// Assigning Mesh, Material, Materials, ReplacementMaterial, Params or Enabled re-registers
    /// automatically. Changing the contents of Materials in place (adding an override, editing one)
    /// cannot be seen: call OnMemberChanged() afterwards. The inspector and the editor's
    /// set-property commands already do.
    /// </summary>
    [Icon("icon_mesh.png")]
    public class MeshRenderer : PersistentRenderer
    {
        public Mesh? Mesh
        {
            get => _mesh;
            set
            {
                if (_mesh == value) return;
                _mesh = value;
                Invalidate();
            }
        }
        private Mesh? _mesh;

        /// <summary>
        /// Default material applied to all MeshParts (unless overridden).
        /// When null, only explicit MaterialOverrides render (mixed-mesh mode).
        /// </summary>
        public Material? Material
        {
            get => _material;
            set
            {
                if (_material == value) return;
                _material = value;
                Invalidate();
            }
        }
        private Material? _material;

        /// <summary>
        /// Sparse per-slot material overrides. Only the slots that differ
        /// from the default need entries. MeshParts whose slot has no override
        /// and no default Material are invisible.
        /// After changing the list's contents in place, call OnMemberChanged().
        /// </summary>
        public List<MaterialOverride> Materials
        {
            get => _materials;
            set
            {
                _materials = value;
                Invalidate();
            }
        }
        private List<MaterialOverride> _materials = [];

        /// <summary>
        /// When set, overrides ALL material routing (Material + Materials).
        /// Used for placement ghost mode. Not serialized.
        /// </summary>
        [DontSerialize]
        public Material? ReplacementMaterial
        {
            get => _replacementMaterial;
            set
            {
                if (_replacementMaterial == value) return;
                _replacementMaterial = value;
                Invalidate();
            }
        }
        private Material? _replacementMaterial;

        /// <summary>
        /// Per-instance shader parameters. Their values are copied when the draws are registered;
        /// the Set* methods re-register, mutating a value in place does not.
        /// </summary>
        public MaterialBlock Params
        {
            get => _params!;
            set => SetParams(ref _params, value);
        }
        private MaterialBlock? _params;

        public MeshRenderer()
        {
            Params = new MaterialBlock();
        }

        protected override Mesh? RenderMesh => _mesh;

        public override void Destroy()
        {
            base.Destroy();
            if (_params != null) _params.Changed -= Invalidate;
        }

        /// <summary>
        /// Resolve material for a MeshPart by its MaterialSlot.
        /// Checks sparse overrides first, falls back to default Material.
        /// When Material is null, unmatched slots are invisible (mixed-mesh mode).
        /// </summary>
        private Material? GetMaterial(int materialSlot)
        {
            if (_materials != null)
            {
                for (int i = 0; i < _materials.Count; i++)
                {
                    if (_materials[i].MaterialSlot == materialSlot)
                        return _replacementMaterial ?? _materials[i].Material;
                }
            }

            return _material != null ? _replacementMaterial ?? _material : null;
        }

        protected override void AddDraws(DrawGroup group, Mesh mesh, int transformSlot)
        {
            // LOD chain heads + non-LOD parts; the GPU culler resolves the LOD for each
            var parts = mesh.DrawPartIndices;
            var meshParts = mesh.MeshParts;

            for (int i = 0; i < parts.Length; i++)
            {
                int partIdx = parts[i];
                if (partIdx >= meshParts.Count) continue;
                var mat = GetMaterial(meshParts[partIdx].MaterialSlot);
                if (mat != null)
                    CommandBuffer.AddPersistent(group, mesh, partIdx, mat, Params, transformSlot, lodManaged: true);
            }
        }
    }
}
