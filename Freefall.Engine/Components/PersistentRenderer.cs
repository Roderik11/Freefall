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

        /// <summary>
        /// This object's values for the per-object data its effects declare (their struct Instance, "//=" members):
        /// only the ones set here; every other member keeps the effect's default. Saved with the entity, shown in
        /// the inspector under "Instance". Change them with <see cref="SetInstance(string, Vector4)"/> (cheap
        /// enough for every frame: it rewrites this object's bytes of the instance data buffer and registers
        /// nothing again); assigning the list rewrites them all.
        /// </summary>
        [System.ComponentModel.Browsable(false)]
        public List<InstanceValue> Instance
        {
            get => _instance;
            set
            {
                _instance = value ?? new List<InstanceValue>();
                foreach (var (effect, slot) in _instanceSlots)
                    WriteInstanceSlot(effect, slot);
            }
        }
        private List<InstanceValue> _instance = new();

        // A slot of the instance data buffer per effect this renderer draws with (by effect name: Effect objects of
        // one name share their layout), and the ones the current draws use
        private readonly List<(Effect effect, int slot)> _instanceSlots = new();
        private readonly HashSet<string> _instanceSlotsUsed = new();

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
            foreach (var (_, slot) in _instanceSlots) InstanceData.Release(slot);
            _instanceSlots.Clear();
        }

        // ── Per-object data (struct Instance of the effects) ──

        /// <summary>
        /// Sets one of this object's instance values (a member of its effects' struct Instance, by name) and writes
        /// it where its draws read it. Every frame is fine: nothing is registered again. A float uses X, a float2
        /// XY, a float3 or colour XYZ.
        /// </summary>
        public void SetInstance(string name, Vector4 value)
        {
            int index = _instance.FindIndex(v => v.Name == name);
            if (index >= 0) _instance[index].Value = value;
            else _instance.Add(new InstanceValue { Name = name, Value = value });

            foreach (var (effect, slot) in _instanceSlots)
            {
                var member = effect.InstanceParams.Find(p => p.Name == name);
                if (member != null) InstanceData.Write(slot, member, value);
            }
        }

        public void SetInstance(string name, float value) => SetInstance(name, new Vector4(value, 0f, 0f, 0f));
        public void SetInstance(string name, Vector2 value) => SetInstance(name, new Vector4(value, 0f, 0f));
        public void SetInstance(string name, Vector3 value) => SetInstance(name, new Vector4(value, 0f));
        public void SetInstance(string name, Color3 value) => SetInstance(name, new Vector4(value.R, value.G, value.B, 0f));

        /// <summary>
        /// This object's value of an instance member: its own if set, else the default of the first of its
        /// effects that declares one. False if none of them does.
        /// </summary>
        public bool TryGetInstance(string name, out Vector4 value)
        {
            int index = _instance.FindIndex(v => v.Name == name);
            if (index >= 0) { value = _instance[index].Value; return true; }
            foreach (var (effect, _) in _instanceSlots)
            {
                var member = effect.InstanceParams.Find(p => p.Name == name);
                if (member != null) { value = InstanceData.DefaultOf(member); return true; }
            }
            value = default;
            return false;
        }

        /// <summary>Drops this object's own value of a member: the effect's default applies again.</summary>
        public void ResetInstance(string name)
        {
            if (_instance.RemoveAll(v => v.Name == name) == 0) return;
            foreach (var (effect, slot) in _instanceSlots) WriteInstanceSlot(effect, slot);
        }

        /// <summary>
        /// The effects whose instance data this renderer's current draws use (for the inspector), each once.
        /// A method, not a property: reflection (entity_get, the serializer) lists public properties.
        /// </summary>
        public List<Effect> GetInstanceEffects()
        {
            var effects = new List<Effect>(_instanceSlots.Count);
            foreach (var (effect, _) in _instanceSlots) effects.Add(effect);
            return effects;
        }

        /// <summary>
        /// The instance slot for a draw with <paramref name="material"/>, for CommandBuffer.AddPersistent: this
        /// renderer's own for the material's effect (made and filled the first time), 0 if the effect declares no
        /// struct Instance. Call from AddDraws.
        /// </summary>
        protected int InstanceSlotFor(Material material)
        {
            var effect = material?.Effect;
            if (effect == null || effect.InstanceParams.Count == 0) return 0;

            _instanceSlotsUsed.Add(effect.Name);
            foreach (var (known, slot) in _instanceSlots)
                if (known.Name == effect.Name) return slot;

            int made = InstanceData.Allocate();
            _instanceSlots.Add((effect, made));
            WriteInstanceSlot(effect, made);
            return made;
        }

        // Defaults first, then this object's own values, in the effect's layout
        private void WriteInstanceSlot(Effect effect, int slot)
        {
            InstanceData.WriteDefaults(slot, effect.InstanceParams);
            foreach (var own in _instance)
            {
                var member = effect.InstanceParams.Find(p => p.Name == own.Name);
                if (member != null) InstanceData.Write(slot, member, own.Value);
            }
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

        /// <summary>
        /// The world bounding sphere of the mesh's box. Runs on every transform change of this renderer or
        /// any parent (a character moving re-runs it for each body part and held item), so it must not
        /// allocate: Vortice's BoundingSphere.CreateFromPoints copies the points into a new array (120 B per
        /// call, ~77 KB per frame with Eldermoor's townsfolk in play mode). The transformed box is centrally
        /// symmetric, so the smallest sphere around its corners is centred on the box centre and reaches
        /// the farthest corner.
        /// </summary>
        private void UpdateBounds()
        {
            var mesh = RenderMesh;
            if (mesh == null || Transform == null) return;

            var world = mesh.RootRotation * Transform.WorldMatrix;
            var box = mesh.BoundingBox;
            var center = Vector3.Transform((box.Min + box.Max) * 0.5f, world);

            float radiusSquared = 0f;
            for (int i = 0; i < 8; i++)
            {
                var corner = new Vector3((i & 1) != 0 ? box.Max.X : box.Min.X,
                                         (i & 2) != 0 ? box.Max.Y : box.Min.Y,
                                         (i & 4) != 0 ? box.Max.Z : box.Min.Z);
                radiusSquared = MathF.Max(radiusSquared, Vector3.DistanceSquared(Vector3.Transform(corner, world), center));
            }

            BoundingSphere = new BoundingSphere(center, MathF.Sqrt(radiusSquared));
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
            _instanceSlotsUsed.Clear();
            AddDraws(group, mesh, slot);
            _drawGroup = group;

            // Slots of effects the draws no longer use go back; the ones in use are rewritten (an effect's layout
            // may have changed: a recompile invalidates every renderer through Material.EffectChanged)
            for (int i = _instanceSlots.Count - 1; i >= 0; i--)
            {
                var (effect, instanceSlot) = _instanceSlots[i];
                if (_instanceSlotsUsed.Contains(effect.Name))
                    WriteInstanceSlot(effect, instanceSlot);
                else
                {
                    InstanceData.Release(instanceSlot);
                    _instanceSlots.RemoveAt(i);
                }
            }
        }

        private void Unregister()
        {
            if (_drawGroup == null) return;
            CommandBuffer.RemovePersistent(_drawGroup);
            _drawGroup = null;
        }
    }

    /// <summary>
    /// One of a renderer's own values for its effects' struct Instance (PersistentRenderer.Instance): the member's
    /// name and up to four floats (a float uses X, a float2 XY, a float3 or colour XYZ).
    /// </summary>
    [Serializable]
    public class InstanceValue
    {
        public string Name = "";
        public Vector4 Value;
    }
}
