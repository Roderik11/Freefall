using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// Base for renderers whose draws are GPU-resident: there is no per-frame Draw(). The renderer
    /// registers its instance records once (CommandBuffer.AddPersistent) and replaces them only when
    /// something they depend on changes. Culling and LOD selection happen on the GPU.
    ///
    /// Subclasses call Invalidate() from the setters of everything their draws depend on, and
    /// implement AddDraws. Lifecycle, enable/disable, moving between entities, bounds and the
    /// reaction to shared mesh/material changes are handled here.
    ///
    /// See .agent/knowledge/rendering_engine/implementation/gpu_resident_draws.md.
    /// </summary>
    public abstract class PersistentRenderer : Component, IPersistentDrawSource
    {
        [NonSerialized]
        public BoundingSphere BoundingSphere;

        /// <summary>The mesh the draws are built from (null = nothing to draw).</summary>
        protected abstract Mesh? RenderMesh { get; }

        /// <summary>
        /// Add this renderer's draws to <paramref name="group"/> with CommandBuffer.AddPersistent.
        /// Called on the main thread with a non-null mesh and a valid transform slot.
        /// </summary>
        protected abstract void AddDraws(DrawGroup group, Mesh mesh, int transformSlot);

        /// <summary>The persistent draws currently registered (null = none).</summary>
        protected DrawGroup? Draws => _drawGroup;

        private DrawGroup? _drawGroup;
        private bool _live;                   // between Awake and Destroy: only then may draws be registered
        private int _refreshQueued;           // 1 while a RefreshDraws is pending (Interlocked)
        private Transform? _trackedTransform; // the transform UpdateBounds is subscribed to
        private Vector3[] _boundsCorners = new Vector3[8];

        // Live renderers, for the rare pushes from shared assets below
        private static readonly HashSet<PersistentRenderer> _liveRenderers = new();
        private static readonly Lock _liveLock = new();

        static PersistentRenderer()
        {
            // Nothing polls the mesh or material per frame, so shared assets push their changes.
            // Both are rare (hot reload, effect swap in the editor), so a sweep over all renderers is fine.
            Mesh.DrawPartsChanged += mesh =>
            {
                lock (_liveLock)
                    foreach (var renderer in _liveRenderers)
                        if (renderer.RenderMesh == mesh) renderer.Invalidate();
            };

            Material.EffectChanged += material =>
            {
                lock (_liveLock)
                    foreach (var renderer in _liveRenderers)
                        renderer.Invalidate();
            };
        }

        protected override void Awake()
        {
            _live = true;
            lock (_liveLock) _liveRenderers.Add(this);
            TrackTransform(Transform);
            UpdateBounds();
            Invalidate();
        }

        // An awake renderer moved to another entity (prefab hydration): its draws reference the old
        // entity's transform slot, so follow the new transform and register again.
        protected internal override void OnAttached()
        {
            if (!_live) return;
            TrackTransform(Transform);
            Invalidate();
        }

        public override void Destroy()
        {
            _live = false;
            lock (_liveLock) _liveRenderers.Remove(this);
            TrackTransform(null);
            Unregister();
        }

        protected override void OnEnabledChanged() => Invalidate();

        /// <summary>
        /// Called by the inspector and the editor's set-property commands after any member edit,
        /// including in-place edits of a material list. Code that changes a list's contents on a
        /// live renderer must call it too: there is no per-frame check that would notice.
        /// </summary>
        public override void OnMemberChanged() => Invalidate();

        private void TrackTransform(Transform? transform)
        {
            if (_trackedTransform == transform) return;
            if (_trackedTransform != null) _trackedTransform.OnChanged -= UpdateBounds;
            _trackedTransform = transform;
            if (_trackedTransform != null) _trackedTransform.OnChanged += UpdateBounds;
        }

        private void UpdateBounds()
        {
            var mesh = RenderMesh;
            if (mesh == null || Transform == null) return;

            mesh.BoundingBox.GetCorners(_boundsCorners, mesh.RootRotation * Transform.WorldMatrix);
            BoundingSphere = BoundingSphere.CreateFromPoints(_boundsCorners);
        }

        /// <summary>
        /// Queue a RefreshDraws before the next frame is rendered. Thread-safe and cheap to call
        /// repeatedly: only the first call until the refresh runs queues anything.
        /// </summary>
        public void Invalidate()
        {
            if (Interlocked.Exchange(ref _refreshQueued, 1) == 0)
                CommandBuffer.Invalidate(this);
        }

        /// <summary>
        /// Subscribe to a MaterialBlock's changes (and unsubscribe from the previous one), then invalidate.
        /// For the Params setter of a subclass: the block's values are copied into the draws at registration.
        /// </summary>
        protected void SetParams(ref MaterialBlock? field, MaterialBlock? value)
        {
            if (field == value) return;
            if (field != null) field.Changed -= Invalidate;
            field = value;
            if (field != null) field.Changed += Invalidate;
            Invalidate();
        }

        /// <summary>
        /// Replace the registered draws with ones matching the current state. Main thread,
        /// called by CommandBuffer.RefreshDrawSources.
        /// </summary>
        void IPersistentDrawSource.RefreshDraws()
        {
            Volatile.Write(ref _refreshQueued, 0);

            Unregister();

            var mesh = RenderMesh;
            if (!_live || !Enabled || mesh == null || Transform == null) return;

            int slot = Transform.TransformSlot;
            if (slot < 0)
            {
                // TransformBuffer not ready yet: try again next frame
                Invalidate();
                return;
            }

            UpdateBounds();

            var group = new DrawGroup();
            AddDraws(group, mesh, slot);
            _drawGroup = group;
        }

        private void Unregister()
        {
            if (_drawGroup == null) return;
            CommandBuffer.RemovePersistent(_drawGroup);
            _drawGroup = null;
        }
    }
}
