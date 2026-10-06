using System.Collections.Generic;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;

namespace Freefall.Editor
{
    /// <summary>
    /// Bridges ISceneGizmo components to the editor's rendering pipeline.
    /// Iterates selected entities, finds components implementing ISceneGizmo,
    /// and invokes DrawGizmos with a shared GizmoContext.
    /// Routes GPU pick results back to handles.
    /// </summary>
    public class GizmoManager
    {
        private readonly GizmoContext _ctx = new GizmoContext();
        private int _gizmoSlot;
        private bool _initialized;
        private System.Func<bool>? _isDragging;

        /// <summary>
        /// The shared GizmoContext. Components draw into this during DrawGizmos.
        /// </summary>
        public GizmoContext Context => _ctx;

        /// <summary>
        /// The TransformBuffer slot used for gizmo geometry.
        /// Used by EditorTools to register in the gizmo slot map.
        /// </summary>
        public int GizmoSlot => _gizmoSlot;

        private void EnsureInitialized()
        {
            if (_initialized) return;
            _initialized = true;

            _gizmoSlot = TransformBuffer.Instance.AllocateSlot();
        }

        /// <summary>
        /// Called each frame from EditorTools.Draw().
        /// Iterates the current selection and invokes DrawGizmos on any ISceneGizmo components.
        /// </summary>
        public void DrawGizmos(IReadOnlyList<Entity> selection)
        {
            EnsureInitialized();

            // Clear accumulated geometry
            _ctx.Clear();

            // Set camera for billboard orientation and handle projection
            var cam = Camera.Main;
            if (cam == null) return;
            _ctx.SetCamera(cam.Position);

            // Feed mouse/camera state to the handle system
            _ctx.Camera = cam;
            _ctx.MouseDown = Input.IsMouseDown(0);

            // Begin handle tracking for this frame
            _ctx.BeginHandles();

            // Release hot control on mouse up
            if (!_ctx.MouseDown)
                _ctx.EndHandles();

            // PCG output regenerates when the drag ends, not on every frame of it
            if (_ctx.HotControl >= 0)
                Freefall.PCG.PCGScheduler.HoldWhile(_isDragging ??= () => _ctx.HotControl >= 0 && Input.IsMouseDown(0));

            // Iterate all selected entities
            bool hasGizmos = false;
            foreach (var entity in selection)
            {
                foreach (var component in entity.Components)
                {
                    if (component is ISceneGizmo gizmo)
                    {
                        // Strip scale from world transform — components like PointLight
                        // bake properties (e.g. Range) into Transform.Scale, so gizmos
                        // need an unscaled matrix to avoid double-scaling.
                        var m = entity.Transform.Matrix;
                        System.Numerics.Matrix4x4.Decompose(m, out _, out var rot, out var pos);
                        _ctx.Matrix = System.Numerics.Matrix4x4.CreateFromQuaternion(rot)
                                    * System.Numerics.Matrix4x4.CreateTranslation(pos);
                        gizmo.DrawGizmos(_ctx);
                        hasGizmos = true;
                    }
                }
            }

            // Submit all accumulated geometry (always flush so stale
            // upload-heap data from previous frames gets invalidated)
            _ctx.Flush(_gizmoSlot);
        }

        /// <summary>
        /// Called by EditorTools when a GPU pick hits the gizmo slot.
        /// Routes the MeshPart ID to the handle system.
        /// </summary>
        public void OnGizmoPicked(uint meshPartId, bool isClick)
        {
            if (isClick)
            {
                // Start drag on this handle
                _ctx.HotControl = (int)meshPartId;
            }
            else
            {
                // Hover highlight
                _ctx.HoverControl = (int)meshPartId;
            }
        }

        /// <summary>
        /// Clear hover highlight (called when hover pick misses gizmo).
        /// </summary>
        public void ClearHover()
        {
            _ctx.HoverControl = -1;
        }
    }
}
