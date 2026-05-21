using Freefall.Base;
using Freefall.Graphics;

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

        public void Draw()
        {
            if(!Enabled) return;
            if (Mesh== null) return;
            if (Materials == null || Materials.Count == 0) return;

            var slot = Transform.TransformSlot;

            for (int i = 0; i < Mesh.MeshParts.Count; i++)
            {
                if (!Mesh.MeshParts[i].Enabled) continue;
                var material = i < Materials.Count ? Materials[i] : Materials[0];
                if(material != null)
                    CommandBuffer.Enqueue(Mesh, i, material, Params, slot, BoneBufferIdx);
            }
        }
    }
}
