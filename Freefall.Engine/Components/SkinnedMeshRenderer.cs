using System.Collections.Generic;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Components
{
    /// <summary>
    /// Renders skinned/animated meshes with bone transforms. The draws are GPU-resident
    /// (see PersistentRenderer): nothing of this component runs per frame.
    ///
    /// The bone buffer is owned by the parent Animator. It is triple-buffered, so its SRV index
    /// changes every frame; the Animator sets BoneBufferIdx after posing, which patches the index
    /// into the registered draws in place instead of re-registering them.
    ///
    /// After changing the contents of Materials in place, call OnMemberChanged().
    /// </summary>
    [Icon("icon_skinnedmesh.png")]
    public class SkinnedMeshRenderer : PersistentRenderer
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

        /// <summary>Material per MaterialSlot. After changing the list in place, call OnMemberChanged().</summary>
        public List<Material> Materials
        {
            get => _materials;
            set
            {
                _materials = value;
                Invalidate();
            }
        }
        private List<Material> _materials = new List<Material>();

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

        /// <summary>
        /// Per-Animator bone buffer SRV index for the current frame. Set by Animator.Update() each frame.
        /// 0 = no bones (fallback to bind pose).
        /// </summary>
        internal uint BoneBufferIdx
        {
            get => _boneBufferIdx;
            set
            {
                if (_boneBufferIdx == value) return;
                _boneBufferIdx = value;

                // Patch the registered draws; a pending registration picks the value up in AddDraws
                var draws = Draws;
                if (draws != null)
                    CommandBuffer.SetBoneBuffer(draws, value);
            }
        }
        private uint _boneBufferIdx;

        public SkinnedMeshRenderer()
        {
            Params = new MaterialBlock();
        }

        protected override Mesh? RenderMesh => _mesh;

        public override void Destroy()
        {
            base.Destroy();
            if (_params != null) _params.Changed -= Invalidate;
        }

        private Material? GetMaterial(int materialSlot)
        {
            if (_materials != null && _materials.Count > materialSlot)
                return _materials[materialSlot];

            return null;
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
                    CommandBuffer.AddPersistent(group, mesh, partIdx, mat, Params, transformSlot, _boneBufferIdx, lodManaged: true);
            }
        }
    }
}
