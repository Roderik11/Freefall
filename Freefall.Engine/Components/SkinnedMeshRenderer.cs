using Freefall.Base;
using Freefall.Graphics;
using System.Numerics;
using Vortice.Mathematics;
using static System.Net.WebRequestMethods;

namespace Freefall.Components
{
    /// <summary>
    /// Renders skinned/animated meshes with bone transforms.
    /// Bone buffer is set by the parent Animator — no per-SMR bone staging.
    /// </summary>
    [Icon("icon_skinnedmesh.png")]
    public class SkinnedMeshRenderer : Component, IDraw
    {
        public Mesh? Mesh;
        public List<Material> Materials = new List<Material>();
        public MaterialBlock Params = new MaterialBlock();

        /// <summary>
        /// Per-Animator bone buffer SRV index. Set by Animator.Update() each frame.
        /// 0 = no bones (fallback to bind pose).
        /// </summary>
        internal uint BoneBufferIdx;

        [NonSerialized]
        public BoundingSphere BoundingSphere;

        private bool _boundsDirty = true;
        private Mesh? _boundsMesh; // tracks which mesh instance bounds were computed from
        private int _boundsVersion; // Mesh.GeometryVersion the bounds were computed from (hot reload)
        private Vector3[] _boundsCorners = new Vector3[8];

        protected override void Awake()
        {
            OnTransformChanged();
            Transform?.OnChanged += OnTransformChanged;
        }

        public override void Destroy()
        {
            Transform?.OnChanged -= OnTransformChanged;
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

        public void Draw()
        {
            if (!Enabled) return;
            if (Mesh == null) return;
            if (Materials == null || Materials.Count == 0) return;

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
                    CommandBuffer.Enqueue(Mesh, partIdx, mat, Params, slot, BoneBufferIdx, lodManaged: true);
            }
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
                if (Materials.Count > materialSlot)
                    return Materials[materialSlot];
            }

            return null;
        }
    }
}
