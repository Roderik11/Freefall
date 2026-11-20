using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// Draws the gizmos of the systems that are a debug view in play (<see cref="IPlayGizmo"/>) while the game plays: in
    /// the editor's play mode, where the editor's gizmos are gone with its camera, and in a published game. Lines through
    /// the same <see cref="GizmoContext"/> the editor uses (always on top), in world space, for the main camera's view.
    /// Nothing is made while no such system exists, and the geometry is cleared every frame, so a view switched off is
    /// gone the next. Each system decides itself whether it has anything to show (a Dev tab switch of its own).
    /// </summary>
    [UpdateInGroup(typeof(RenderGroup))]
    [UpdateAfter(typeof(ScriptDrawSystem))]
    public sealed class PlayGizmoSystem : EntitySystem
    {
        private GizmoContext? _ctx;
        private int _slot = -1;
        // Weak: a system of a script assembly must not keep it loaded after a reload
        private readonly System.Runtime.CompilerServices.ConditionalWeakTable<IPlayGizmo, object> _failed = new();

        public override void Update()
        {
            var gizmos = Systems.Gizmos;
            bool any = false;
            for (int i = 0; i < gizmos.Count && !any; i++)
                any = gizmos[i] is IPlayGizmo;
            if (!any && _ctx == null) return;

            var camera = Camera.Main;
            if (camera == null) return;
            _ctx ??= new GizmoContext();
            if (_slot < 0) _slot = TransformBuffer.Instance.AllocateSlot();

            _ctx.Clear();
            _ctx.SetCamera(camera.Position);
            _ctx.Camera = camera;
            for (int i = 0; i < gizmos.Count; i++)
            {
                if (gizmos[i] is not IPlayGizmo gizmo) continue;
                _ctx.Matrix = Matrix4x4.Identity;
                try { gizmo.DrawGizmos(_ctx); }
                catch (Exception ex)
                {
                    if (_failed.TryAdd(gizmo, string.Empty))
                        Debug.LogWarning("Gizmos", $"{gizmo.GetType().Name}.DrawGizmos threw in play: {ex.Message}");
                }
            }
            // Always flushed: what was drawn last frame and not this one is cleared
            _ctx.Flush(_slot);
        }
    }
}
