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
            _boundsDirty = false;
        }

        public void Draw()
        {
            if (!Enabled) return;
            if (Mesh == null) return;
            if (Materials == null || Materials.Count == 0) return;

            // Re-dirty bounds when the mesh reference changes (stub → loaded)
            if (Mesh != _boundsMesh) _boundsDirty = true;

            if (_boundsDirty) OnTransformChanged();

            var slot = Transform.TransformSlot;
            int lod = GetActiveLOD(out bool tooSmall);

            if (tooSmall)
                return;

            if (lod >= 0 && Mesh.LODs[lod].MeshPartIndices != null)
            {
                // Draw active LOD parts
                var indices = Mesh.LODs[lod].MeshPartIndices;
                for (int i = 0; i < indices.Length; i++)
                {
                    int partIdx = indices[i];
                    if (partIdx >= Mesh.MeshParts.Count) continue;
                    var mat = GetMaterial(Mesh.MeshParts[partIdx].MaterialSlot);
                    if (mat != null)
                        CommandBuffer.Enqueue(Mesh, partIdx, mat, Params, slot, BoneBufferIdx);
                }

                // Draw truly non-LOD parts (precomputed, zero alloc)
                if (Mesh.NonLodPartIndices != null)
                {
                    for (int i = 0; i < Mesh.NonLodPartIndices.Length; i++)
                    {
                        int partIdx = Mesh.NonLodPartIndices[i];
                        var mat = GetMaterial(Mesh.MeshParts[partIdx].MaterialSlot);
                        if (mat != null)
                            CommandBuffer.Enqueue(Mesh, partIdx, mat, Params, slot, BoneBufferIdx);
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
                        CommandBuffer.Enqueue(Mesh, i, mat, Params, slot, BoneBufferIdx);
                }
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

        /// <summary>
        /// Select active LOD index based on screen-relative size.
        /// Returns -1 if no LOD chain (use all MeshParts).
        /// </summary>
        private int GetActiveLOD(out bool tooSmall)
        {
            tooSmall = true;

            var cam = Camera.Main;
            if (cam == null) return 0;

            float distanceSq = Vector3.DistanceSquared(BoundingSphere.Center, cam.Position);
            if (distanceSq < 0.001f) return 0;

            float diameter = BoundingSphere.Radius;
            float sizeSq = (diameter * diameter / MathF.Max(distanceSq, 0.001f)) * cam.FoVFactor;
            sizeSq *= Engine.Settings.LODScale * Mesh.LODBias;

            tooSmall = sizeSq < 0.00001f;

            int lodCount = Mesh.LODs.Count;

            if (lodCount == 0) return -1;

            // Geometric progression: each LOD transition at half the screen size of the previous.
            for (int i = 0; i < lodCount - 1; i++)
            {
                float t = MathF.Pow(0.5f, i + 1);
                if (sizeSq > t * t)
                    return i;
            }

            return lodCount - 1;
        }
    }
}
