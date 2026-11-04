using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Vortice.Direct3D;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// Records and draws every <see cref="TubeTrail"/>, as <see cref="TrailSystem"/> does every TrailRenderer.
    ///
    /// Per trail the CPU keeps the places its entity has been, each with its age. Per frame this system (UpdateGroup,
    /// after scripts) ages them, adds where the entity is now, drops what has run out, and lays rings along the way: from
    /// the head (the entity, or where the trail's newest place is once it has stopped) to the far end, closer together at
    /// the head where the nose closes, through a curve between the places, each with its age and a direction across it
    /// carried along from the head (parallel transport: the tube does not twist). <see cref="TubeTrailRenderSystem"/>
    /// (RenderGroup) uploads the rings and enqueues one draw in the Forward pass: vfx_trail.fx builds the tube between
    /// neighbouring rings and shades it with each trail's vfx layers material (vfx_layers.hlsli).
    ///
    /// A trail is a few dozen places and a hundred-odd rings, so there is no GPU simulation and no culling here.
    /// </summary>
    [UpdateInEditor]
    public sealed class TubeTrailSystem : EntitySystem
    {
        private const int MaxPoints = 1024;        // per trail; beyond it the oldest go
        private const uint FlagFirst = 1;          // = TUBE_FIRST
        private const uint FlagLast = 2;           // = TUBE_LAST
        internal const int Sides = 16;             // = TUBE_SIDES
        private const float SamePlaceSq = 1e-8f;   // closer than a tenth of a millimetre: no piece of tube in between

        // Rings: one every RingSpacing metres along the trail, at least MinRings and at most MaxRings, closer at the head
        // (by the power RingBunch) where the nose rounds and the fire is detailed; Wave moves the outline by the fire's
        // detail, so it needs rings a couple of centimetres apart
        private const float RingSpacing = 0.02f;
        private const int MinRings = 8;
        private const int MaxRings = 160;
        private const float RingBunch = 1.5f;

        /// <summary>One ring of the list. Must match struct TubeRing in vfx_trail.fx.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct TubeRing
        {
            public Vector3 Position;   public float Age;
            public Vector3 Normal;     public float Along;
            public Vector3 Color;      public float Intensity;
            public uint MaterialId;    public float Radius;   public float TexV;     public uint Flags;
            public float Erode;        public float Heat;     public Vector2 Seed;
            public float Nose;         public float Pace;     public Vector2 Pad;
        }

        /// <summary>Everything the system keeps per trail. The component itself only holds authored data.</summary>
        private sealed class Entry
        {
            public TubeTrail? Trail;   // null once its component is gone: the tube burns out where it is

            // Ring of the places the entity has been, oldest first. The capacity is a power of two.
            public Vector3[] Positions = new Vector3[32];
            public float[] Ages = new float[32];
            public int Start;
            public int Count;
            public bool WasEmitting;

            // The look, kept for when the component is gone
            public Material? Material;
            public float Lifetime = 0.1f;
            public float Radius, TextureSpan, Intensity, Erode, Heat;
            public Vector3 Color;
            public Vector2 Seed;

            // Metres the head has gone since the trail began (wrapped): the textures' V is the way back from the head less
            // this, so a place keeps its V while the head flies on
            public float Flown;
            public Vector3 LastHead;
            public bool HasHead;

            public int Slot(int i) => (Start + i) & (Positions.Length - 1);

            public void Push(Vector3 position)
            {
                if (Count == Positions.Length)
                {
                    if (Count >= MaxPoints)
                    {
                        Start = Slot(1);
                        Count--;
                    }
                    else
                    {
                        var positions = new Vector3[Count * 2];
                        var ages = new float[Count * 2];
                        for (int i = 0; i < Count; i++)
                        {
                            positions[i] = Positions[Slot(i)];
                            ages[i] = Ages[Slot(i)];
                        }
                        Positions = positions;
                        Ages = ages;
                        Start = 0;
                    }
                }

                int slot = Slot(Count++);
                Positions[slot] = position;
                Ages[slot] = 0f;
            }

            public void PopOldest()
            {
                Start = Slot(1);
                Count--;
            }

            public Entry Copy() => new()
            {
                Positions = Positions, Ages = Ages, Start = Start, Count = Count, Material = Material, Lifetime = Lifetime,
                Radius = Radius, TextureSpan = TextureSpan, Intensity = Intensity, Erode = Erode, Heat = Heat, Color = Color, Seed = Seed,
                Flown = Flown, LastHead = LastHead, HasHead = HasHead,
            };
        }

        private readonly List<Entry> _entries = [];
        private TubeRing[] _rings = new TubeRing[512];
        private int _ringCount;

        // Scratch for one trail's way, head first
        private Vector3[] _way = new Vector3[64];
        private float[] _wayAge = new float[64];

        // GPU
        private StreamingBuffer<TubeRing>? _buffer;
        private Effect? _effect;
        private Material? _material;
        private ID3D12DescriptorHeap[]? _heaps;
        private bool _failed;
        private bool _warnedMaterial;

        private int _preparedTick = -1;
        private bool _prepared;
        private int _uploadedRings;

        /// <summary>Trails currently recorded, including those burning out after their component went.</summary>
        public int TrailCount => _entries.Count;

        /// <summary>Rings written for the last frame.</summary>
        public int RingCount => _ringCount;

        // ────────────── Registration ──────────────

        protected internal override void Initialize()
        {
            if (Unsafe.SizeOf<TubeRing>() != 96)
                throw new InvalidOperationException("TubeRing must match struct TubeRing in vfx_trail.fx (96 bytes)");

            ComponentCache<TubeTrail>.Added += OnAdded;
            ComponentCache<TubeTrail>.Removed += OnRemoved;

            foreach (var trail in ComponentCache<TubeTrail>.All)
                OnAdded(trail);
        }

        protected internal override void Destroy()
        {
            ComponentCache<TubeTrail>.Added -= OnAdded;
            ComponentCache<TubeTrail>.Removed -= OnRemoved;
        }

        private void OnAdded(TubeTrail trail)
        {
            if (trail.SystemIndex >= 0) return;
            trail.SystemIndex = _entries.Count;
            _entries.Add(new Entry { Trail = trail, Seed = SeedOf(trail) });
        }

        private void OnRemoved(TubeTrail trail)
        {
            int index = trail.SystemIndex;
            if (index >= 0 && index < _entries.Count && _entries[index].Trail == trail)
                Orphan(_entries[index]);
        }

        // Its own offset in its textures, so two trails of one material never move in step
        private static Vector2 SeedOf(TubeTrail trail)
        {
            uint h = (uint)(trail.Entity?.Id ?? 0) * 2654435761u;
            return new Vector2(h & 0xFFFFu, h >> 16) * (1f / 65535f);
        }

        /// <summary>The component is gone: what it left stays until it has burnt out.</summary>
        private static void Orphan(Entry entry)
        {
            if (entry.Trail != null) entry.Trail.SystemIndex = -1;
            entry.Trail = null;
        }

        private void RemoveEntry(int index)
        {
            Orphan(_entries[index]);

            int last = _entries.Count - 1;
            if (index != last)
            {
                var moved = _entries[last];
                _entries[index] = moved;
                if (moved.Trail != null) moved.Trail.SystemIndex = index;
            }
            _entries.RemoveAt(last);
        }

        // ────────────── Gather (UpdateGroup, after scripts) ──────────────

        public override void Update()
        {
            float dt = Time.Delta;
            _ringCount = 0;

            for (int i = _entries.Count - 1; i >= 0; i--)
            {
                var entry = _entries[i];

                // Destroyed or detached this frame (Removed arrives at the next wake-up, or never if it was destroyed
                // before it was announced)
                var trail = entry.Trail;
                if (trail != null && (trail.IsDestroyed || trail.Entity == null))
                {
                    Orphan(entry);
                    trail = null;
                }

                Record(entry, trail, dt);

                if (trail == null && entry.Count == 0)
                    RemoveEntry(i);
            }
        }

        private void Record(Entry entry, TubeTrail? trail, float dt)
        {
            for (int i = 0; i < entry.Count; i++)
                entry.Ages[entry.Slot(i)] += dt;

            bool emitting = false;
            var here = Vector3.Zero;

            // While no time passes (paused) nothing is recorded: a place where the entity was moved to would be as old as
            // the one before it, and that whole stretch would be the tip of the nose (a needle, as its shape goes by age)
            bool still = trail != null && dt <= 0f;
            if (still)
            {
                emitting = entry.WasEmitting;
                trail = null;
            }

            if (trail != null)
            {
                entry.Material = trail.Material;
                entry.Lifetime = MathF.Max(0.01f, trail.Lifetime);
                entry.Radius = MathF.Max(0f, trail.Radius);
                entry.TextureSpan = MathF.Max(0.01f, trail.TextureSpan);
                entry.Intensity = MathF.Max(0f, trail.Intensity);
                entry.Erode = trail.Erode;
                entry.Heat = MathF.Max(0f, trail.Heat);
                entry.Color = new Vector3(trail.Color.R, trail.Color.G, trail.Color.B);

                emitting = trail.Enabled && trail.Emitting;
                here = trail.Transform!.WorldPosition;

                // The way the head went, while it is the head (and as far as it got when it stopped); wrapped after a
                // thousand of its spans, so its V stays precise
                if (entry.HasHead && (emitting || entry.WasEmitting))
                    entry.Flown = (entry.Flown + Vector3.Distance(here, entry.LastHead)) % (entry.TextureSpan * 1024f);
                entry.LastHead = here;
                entry.HasHead = true;

                if (emitting)
                {
                    // Switched on again while the last stretch is still burning out: that stretch goes on as a trail of its
                    // own, or the new one would be joined to where the old one ended
                    if (!entry.WasEmitting && entry.Count > 0)
                    {
                        var rest = entry.Copy();
                        _entries.Add(rest);
                        Record(rest, null, 0f);   // this frame's entries have been gone through already

                        entry.Positions = new Vector3[32];
                        entry.Ages = new float[32];
                        entry.Start = 0;
                        entry.Count = 0;
                    }

                    float spacing = MathF.Max(0.01f, trail.PointSpacing);
                    if (entry.Count == 0 || Vector3.DistanceSquared(here, entry.Positions[entry.Slot(entry.Count - 1)]) >= spacing * spacing)
                        entry.Push(here);
                }
                else if (entry.WasEmitting && entry.Count > 0
                    && Vector3.DistanceSquared(here, entry.Positions[entry.Slot(entry.Count - 1)]) > SamePlaceSq)
                {
                    // It stopped: the tube reaches as far as the entity got
                    entry.Push(here);
                }
            }
            entry.WasEmitting = emitting;

            // While it is being left, the tube runs on from its newest place to where the entity is now
            bool head = emitting && !still && entry.Count > 0 && Vector3.DistanceSquared(here, entry.Positions[entry.Slot(entry.Count - 1)]) > SamePlaceSq;

            // The oldest place goes once the one after it has run out as well. Until then the tube ends between the two,
            // where it is exactly as old as it gets.
            float lifetime = entry.Lifetime;
            while (entry.Count > 1 && entry.Ages[entry.Slot(1)] >= lifetime)
                entry.PopOldest();
            if (entry.Count == 1 && !head && entry.Ages[entry.Start] >= lifetime)
                entry.PopOldest();

            int count = entry.Count + (head ? 1 : 0);
            if (count < 2 || entry.Material == null) return;
            if (entry.Material.Effect != InternalAssets.VfxLayersEffect)
            {
                if (!_warnedMaterial)
                {
                    _warnedMaterial = true;
                    Debug.LogWarning("TubeTrailSystem", $"A tube trail's material '{entry.Material.Name}' is not a vfx layers material (vfx_layers.fx): it is not drawn");
                }
                return;
            }

            // Its way, head first: the entity, then its places from the newest, the oldest cut where it runs out
            if (_way.Length < count)
            {
                Array.Resize(ref _way, Math.Max(count, _way.Length * 2));
                Array.Resize(ref _wayAge, _way.Length);
            }
            int n = 0;
            if (head)
            {
                _way[n] = here;
                _wayAge[n++] = 0f;
            }
            for (int i = entry.Count - 1; i >= 0; i--)
            {
                int slot = entry.Slot(i);
                var position = entry.Positions[slot];
                float age = entry.Ages[slot];
                if (i == 0 && age > lifetime && n > 0)
                {
                    // Run out, with a younger place before it: the end lies in between
                    float younger = _wayAge[n - 1];
                    if (age - younger > 1e-6f)
                        position = Vector3.Lerp(_way[n - 1], position, (lifetime - younger) / (age - younger));
                    age = lifetime;
                }
                _way[n] = position;
                _wayAge[n++] = MathF.Min(age, lifetime);
            }

            float length = 0f;
            for (int i = 1; i < n; i++) length += Vector3.Distance(_way[i - 1], _way[i]);
            if (length < 1e-4f) return;

            Lay(entry, n, length);
        }

        /// <summary>
        /// The rings along a trail's way (the first <paramref name="n"/> of _way, head first): spaced by age from the head's
        /// to the far end's, closer at the head, through a curve between the places (Catmull-Rom), each turned the way the
        /// one before it was, squared to the tube's axis there.
        /// </summary>
        private void Lay(Entry entry, int n, float length)
        {
            float lifetime = entry.Lifetime;
            float headAge = _wayAge[0] / lifetime;
            float endAge = _wayAge[n - 1] / lifetime;
            float span = MathF.Max(endAge - headAge, 1e-5f);
            float pace = length / span;

            int rings = Math.Clamp((int)MathF.Ceiling(length / RingSpacing), MinRings, MaxRings);
            if (_rings.Length < _ringCount + rings)
                Array.Resize(ref _rings, Math.Max(_ringCount + rings, _rings.Length * 2));

            int first = _ringCount;
            int segment = 0;
            float back = 0f;            // metres from the head along the rings laid so far
            for (int k = 0; k < rings; k++)
            {
                float along = MathF.Pow((float)k / (rings - 1), RingBunch);
                float age = headAge + span * along;
                float seconds = age * lifetime;
                while (segment < n - 2 && _wayAge[segment + 1] < seconds) segment++;
                float from = _wayAge[segment], to = _wayAge[segment + 1];
                float t = to - from > 1e-7f ? Math.Clamp((seconds - from) / (to - from), 0f, 1f) : 0f;
                var position = CatmullRom(_way[Math.Max(segment - 1, 0)], _way[segment], _way[segment + 1], _way[Math.Min(segment + 2, n - 1)], t);
                if (k > 0) back += Vector3.Distance(position, _rings[_ringCount - 1].Position);

                _rings[_ringCount++] = new TubeRing
                {
                    Position = position,
                    Age = age,
                    Along = along,
                    Color = entry.Color,
                    Intensity = entry.Intensity,
                    MaterialId = (uint)entry.Material!.MaterialID,
                    Radius = entry.Radius,
                    TexV = (back - entry.Flown) / entry.TextureSpan,
                    Erode = entry.Erode,
                    Heat = entry.Heat,
                    Seed = entry.Seed,
                    Nose = age - headAge,
                    Pace = pace,
                };
            }

            // Across each ring: from the head's (square to its axis and as level as can be, so a straight trail's never
            // turns), carried along the tube
            var normal = Vector3.Zero;
            for (int k = first; k < _ringCount; k++)
            {
                var axis = _rings[Math.Min(k + 1, _ringCount - 1)].Position - _rings[Math.Max(k - 1, first)].Position;
                axis = axis.LengthSquared() > 1e-12f ? Vector3.Normalize(axis) : Vector3.UnitZ;
                if (k == first)
                    normal = MathF.Abs(axis.Y) < 0.99f ? Vector3.Cross(axis, Vector3.UnitY) : Vector3.Cross(axis, Vector3.UnitX);
                normal -= axis * Vector3.Dot(normal, axis);
                normal = normal.LengthSquared() > 1e-12f ? Vector3.Normalize(normal) : Vector3.Normalize(Vector3.Cross(axis, Vector3.UnitX));
                _rings[k].Normal = normal;
            }

            _rings[first].Flags |= FlagFirst;
            _rings[_ringCount - 1].Flags |= FlagLast;
        }

        private static Vector3 CatmullRom(Vector3 a, Vector3 b, Vector3 c, Vector3 d, float t)
        {
            float t2 = t * t, t3 = t2 * t;
            return 0.5f * (2f * b + (c - a) * t + (2f * a - 5f * b + 4f * c - d) * t2 + (3f * b - a - 3f * c + d) * t3);
        }

        // ────────────── Render side (called by TubeTrailRenderSystem) ──────────────

        /// <summary>Once per tick, before the first view enqueues: upload this frame's rings. False when there is nothing to draw.</summary>
        internal bool PrepareFrame()
        {
            if (_preparedTick == Engine.TickCount) return _prepared;
            _preparedTick = Engine.TickCount;
            _prepared = false;

            if (_failed || _ringCount < 2) return false;

            try
            {
                if (_material == null)
                {
                    _effect = new Effect("vfx_trail");
                    _material = new Material(_effect);
                    _buffer = new StreamingBuffer<TubeRing>(Engine.Device, 1024);
                    _heaps = [Engine.Device.SrvHeap];
                }
            }
            catch (Exception ex)
            {
                _failed = true;
                Debug.LogError("TubeTrailSystem", $"Init failed, tube trails are disabled: {ex.Message}");
                return false;
            }

            _buffer!.BulkWrite(_rings.AsSpan(0, _ringCount));
            _uploadedRings = _ringCount;
            return _prepared = true;
        }

        /// <summary>Forward pass: every tube trail of this frame in one draw (kept to its side of a split compare while the
        /// other takes them through the OIT pass, EngineSettings.Transparency).</summary>
        internal void Render(ID3D12GraphicsCommandList cmd)
        {
            if (!OrderIndependentTransparency.Active)
            {
                Draw(cmd, RenderPass.Forward);
                return;
            }
            if (!OrderIndependentTransparency.Sorted) return;
            cmd.RSSetScissorRect(OrderIndependentTransparency.SortedScissor);
            Draw(cmd, RenderPass.Forward);
            cmd.RSSetScissorRect(OrderIndependentTransparency.FullScissor);
        }

        /// <summary>OITMoments pass: what the trails' smoke hides.</summary>
        internal void RenderOITMoments(ID3D12GraphicsCommandList cmd) => Draw(cmd, RenderPass.OITMoments);

        /// <summary>OIT pass, after the Forward pass's water.</summary>
        internal void RenderOIT(ID3D12GraphicsCommandList cmd) => Draw(cmd, RenderPass.OIT);

        private void Draw(ID3D12GraphicsCommandList cmd, RenderPass pass)
        {
            if (_uploadedRings < 2) return;

            var device = Engine.Device;
            cmd.SetDescriptorHeaps(1, _heaps!);
            _material!.SetPass(pass);
            _material.Apply(cmd, device);
            _material.SetPass(RenderPass.Forward);

            cmd.SetGraphicsRoot32BitConstant(0, _buffer!.SrvIndex, 0);                                      // RingsIdx
            cmd.SetGraphicsRoot32BitConstant(0, Material.MaterialsBufferIndex, 1);                           // MaterialsIdx
            cmd.SetGraphicsRoot32BitConstant(0, DeferredRenderer.Current?.DepthGBuffer?.BindlessIndex ?? 0u, 2);   // SceneDepthIdx

            // One instance per ring but the last: the piece of tube from it to the next, Sides quads round
            cmd.IASetPrimitiveTopology(PrimitiveTopology.TriangleList);
            cmd.DrawInstanced((uint)(6 * Sides), (uint)(_uploadedRings - 1), 0, 0);
        }
    }

    /// <summary>
    /// The render half of <see cref="TubeTrailSystem"/>: runs once per rendered view while draws are being enqueued. The
    /// draw is a custom action of the Forward pass, after those of the IDraw components, so trails end up on top of what
    /// those draw (the ocean).
    /// </summary>
    [UpdateInEditor]
    [UpdateInGroup(typeof(RenderGroup))]
    [UpdateAfter(typeof(ScriptDrawSystem))]
    public sealed class TubeTrailRenderSystem : EntitySystem
    {
        private TubeTrailSystem? _trails;
        private Action<ID3D12GraphicsCommandList>? _render;
        private Action<ID3D12GraphicsCommandList>? _oitMoments;
        private Action<ID3D12GraphicsCommandList>? _oit;

        protected internal override void Initialize()
        {
            _trails = Systems.Get<TubeTrailSystem>();
            if (_trails == null) return;
            _render = _trails.Render;
            _oitMoments = _trails.RenderOITMoments;
            _oit = _trails.RenderOIT;
        }

        public override void Update()
        {
            if (_trails == null || !_trails.PrepareFrame()) return;

            // After the pass's batches, so the ocean (a batch) does not cover the tubes in front of it
            CommandBuffer.EnqueueLate(RenderPass.Forward, _render!);
            // Without order (EngineSettings.Transparency): this view's split is known before the draw systems run
            if (OrderIndependentTransparency.MomentsDue && OrderIndependentTransparency.Active)
                CommandBuffer.Enqueue(RenderPass.OITMoments, _oitMoments!);
            if (OrderIndependentTransparency.Active)
                CommandBuffer.Enqueue(RenderPass.OIT, _oit!);
        }
    }
}
