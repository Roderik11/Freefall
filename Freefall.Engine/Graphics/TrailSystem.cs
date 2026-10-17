using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Freefall.Base;
using Freefall.Components;
using Vortice.Direct3D;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// Records and draws every <see cref="TrailRenderer"/>.
    ///
    /// Per trail the CPU keeps the places its entity has been, each with its age. Per frame this system
    /// (UpdateGroup, after scripts) ages them, adds where the entity is now, drops what has run out and
    /// writes the points of all trails into one list, each already carrying the width and colour of its
    /// age. <see cref="TrailRenderSystem"/> (RenderGroup) uploads the list and enqueues one draw in the
    /// Forward pass: trail_draw.fx builds a camera-facing quad per pair of neighbouring points.
    ///
    /// A trail is a few dozen points, so there is no GPU simulation and no culling here.
    /// </summary>
    [UpdateInEditor]
    public sealed class TrailSystem : EntitySystem
    {
        private const int MaxPoints = 1024;        // per trail; beyond it the oldest go
        private const uint FlagFirst = 1;          // = TRAIL_FIRST
        private const uint FlagLast = 2;           // = TRAIL_LAST
        private const uint FlagBlade = 4;          // = TRAIL_BLADE
        private const uint FlagLit = 8;            // = TRAIL_LIT
        private const float SamePlaceSq = 1e-8f;   // closer than a tenth of a millimetre: no piece of ribbon in between

        // A blade's sheet is rounded between the places it was recorded at: a swing moves a tip a
        // hand's width or more per frame, and straight pieces between those would show as corners
        private const int BladeSteps = 4;

        /// <summary>One point of the list. Must match struct TrailPoint in trail_draw.fx.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct TrailPoint
        {
            public Vector3 Position;   public float Width;
            public Vector4 Color;
            public uint Flags;         public Vector3 Edge;   // a blade's other edge (TRAIL_BLADE)
        }

        /// <summary>Everything the system keeps per trail. The component itself only holds authored data.</summary>
        private sealed class Entry
        {
            public TrailRenderer? Trail;   // null once its component is gone: the ribbon fades out where it is

            // Ring of the places the entity has been, oldest first. The capacity is a power of two.
            public Vector3[] Positions = new Vector3[32];
            public Vector3[] Edges = new Vector3[32];   // a blade's other edge at that place
            public float[] Ages = new float[32];
            public int Start;
            public int Count;
            public bool WasEmitting;
            public bool Blade;
            public bool Lit;

            // The look, kept for when the component is gone
            public float Lifetime = 0.5f;
            public float StartWidth, EndWidth;
            public Vector4 ColorStart, ColorEnd;

            public int Slot(int i) => (Start + i) & (Positions.Length - 1);

            public void Push(Vector3 position, Vector3 edge)
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
                        var edges = new Vector3[Count * 2];
                        var ages = new float[Count * 2];
                        for (int i = 0; i < Count; i++)
                        {
                            positions[i] = Positions[Slot(i)];
                            edges[i] = Edges[Slot(i)];
                            ages[i] = Ages[Slot(i)];
                        }
                        Positions = positions;
                        Edges = edges;
                        Ages = ages;
                        Start = 0;
                    }
                }

                int slot = Slot(Count++);
                Positions[slot] = position;
                Edges[slot] = edge;
                Ages[slot] = 0f;
            }

            public void PopOldest()
            {
                Start = Slot(1);
                Count--;
            }
        }

        private readonly List<Entry> _entries = [];
        private TrailPoint[] _points = new TrailPoint[256];
        private TrailPoint[] _round = new TrailPoint[64];   // scratch for rounding a blade's points
        private int _pointCount;

        // GPU
        private StreamingBuffer<TrailPoint>? _buffer;
        private Effect? _effect;
        private Material? _material;
        private ID3D12DescriptorHeap[]? _heaps;
        private bool _failed;

        private int _preparedTick = -1;
        private bool _prepared;
        private int _uploadedPoints;

        /// <summary>Trails currently recorded, including those fading out after their component went.</summary>
        public int TrailCount => _entries.Count;

        /// <summary>Points written for the last frame.</summary>
        public int PointCount => _pointCount;

        // ────────────── Registration ──────────────

        protected internal override void Initialize()
        {
            if (Unsafe.SizeOf<TrailPoint>() != 48)
                throw new InvalidOperationException("TrailPoint must match struct TrailPoint in trail_draw.fx (48 bytes)");

            ComponentCache<TrailRenderer>.Added += OnAdded;
            ComponentCache<TrailRenderer>.Removed += OnRemoved;

            foreach (var trail in ComponentCache<TrailRenderer>.All)
                OnAdded(trail);
        }

        protected internal override void Destroy()
        {
            ComponentCache<TrailRenderer>.Added -= OnAdded;
            ComponentCache<TrailRenderer>.Removed -= OnRemoved;
        }

        private void OnAdded(TrailRenderer trail)
        {
            if (trail.SystemIndex >= 0) return;
            trail.SystemIndex = _entries.Count;
            _entries.Add(new Entry { Trail = trail });
        }

        private void OnRemoved(TrailRenderer trail)
        {
            int index = trail.SystemIndex;
            if (index >= 0 && index < _entries.Count && _entries[index].Trail == trail)
                Orphan(_entries[index]);
        }

        /// <summary>The component is gone: what it left stays until it has faded.</summary>
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
            _pointCount = 0;

            for (int i = _entries.Count - 1; i >= 0; i--)
            {
                var entry = _entries[i];

                // Destroyed or detached this frame (Removed arrives at the next wake-up, or never if it
                // was destroyed before it was announced)
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

        private void Record(Entry entry, TrailRenderer? trail, float dt)
        {
            for (int i = 0; i < entry.Count; i++)
                entry.Ages[entry.Slot(i)] += dt;

            bool emitting = false;
            var here = Vector3.Zero;
            var hereEdge = Vector3.Zero;
            if (trail != null)
            {
                entry.Blade = trail.BladeOffset != Vector3.Zero;
                entry.Lit = trail.Lit;
                entry.Lifetime = MathF.Max(0.01f, trail.Lifetime);
                entry.StartWidth = MathF.Max(0f, trail.StartWidth);
                entry.EndWidth = MathF.Max(0f, trail.EndWidth);
                float intensity = MathF.Max(0f, trail.Intensity);
                entry.ColorStart = trail.ColorStart * new Vector4(intensity, intensity, intensity, 1f);
                entry.ColorEnd = trail.ColorEnd * new Vector4(intensity, intensity, intensity, 1f);

                emitting = trail.Enabled && trail.Emitting;
                here = trail.Transform!.WorldPosition;
                // Without a blade the other edge is the place itself: a trail that is made a blade's
                // while it has points gets a sheet of no width there, not one to the world's origin
                hereEdge = entry.Blade ? Vector3.Transform(trail.BladeOffset, trail.Transform.WorldMatrix) : here;

                if (emitting)
                {
                    // Switched on again while the last stretch is still fading (the next swing of a
                    // combo): that stretch goes on fading as a ribbon of its own, or the new one would
                    // be joined to where the old one ended
                    if (!entry.WasEmitting && entry.Count > 0)
                    {
                        var rest = new Entry
                        {
                            Positions = entry.Positions, Edges = entry.Edges, Ages = entry.Ages,
                            Start = entry.Start, Count = entry.Count, Blade = entry.Blade, Lit = entry.Lit,
                            Lifetime = entry.Lifetime, StartWidth = entry.StartWidth, EndWidth = entry.EndWidth,
                            ColorStart = entry.ColorStart, ColorEnd = entry.ColorEnd,
                        };
                        _entries.Add(rest);
                        Record(rest, null, 0f);   // this frame's entries have been gone through already

                        entry.Positions = new Vector3[32];
                        entry.Edges = new Vector3[32];
                        entry.Ages = new float[32];
                        entry.Start = 0;
                        entry.Count = 0;
                    }

                    float spacing = MathF.Max(0.01f, trail.PointSpacing);
                    if (entry.Count == 0 || Vector3.DistanceSquared(here, entry.Positions[entry.Slot(entry.Count - 1)]) >= spacing * spacing)
                        entry.Push(here, hereEdge);
                }
                else if (entry.WasEmitting && entry.Count > 0
                    && Vector3.DistanceSquared(here, entry.Positions[entry.Slot(entry.Count - 1)]) > SamePlaceSq)
                {
                    // It stopped: the ribbon reaches as far as the entity got
                    entry.Push(here, hereEdge);
                }
            }
            entry.WasEmitting = emitting;

            // While it is being left, the ribbon runs on from its newest point to where the entity is now
            bool head = emitting && Vector3.DistanceSquared(here, entry.Positions[entry.Slot(entry.Count - 1)]) > SamePlaceSq;

            // The oldest point goes once the one after it has run out as well. Until then the ribbon
            // ends between the two, where it is exactly as old as it gets.
            float lifetime = entry.Lifetime;
            while (entry.Count > 1 && entry.Ages[entry.Slot(1)] >= lifetime)
                entry.PopOldest();
            if (entry.Count == 1 && !head && entry.Ages[entry.Start] >= lifetime)
                entry.PopOldest();

            int count = entry.Count + (head ? 1 : 0);
            if (count < 2) return;

            // A blade's points are rounded: every piece becomes BladeSteps pieces
            int room = entry.Blade ? (count - 1) * BladeSteps + 1 : count;
            if (_points.Length < _pointCount + room)
                Array.Resize(ref _points, Math.Max(_pointCount + room, _points.Length * 2));

            // A blade's sheet also fades over its length, to nothing where it started: by how old a
            // place is against the oldest there is, so the far end is soft from the first frame on
            // and not only once it is as old as it gets
            float oldest = entry.Count > 0 ? MathF.Max(MathF.Min(entry.Ages[entry.Start], lifetime), 1e-5f) : 1e-5f;

            int first = _pointCount;
            for (int i = 0; i < entry.Count; i++)
            {
                int slot = entry.Slot(i);
                var position = entry.Positions[slot];
                var edge = entry.Edges[slot];
                float age = entry.Ages[slot];

                if (i == 0 && age > lifetime)
                {
                    // Run out, with a younger point after it: the end lies in between
                    bool last = i + 1 >= entry.Count;
                    float nextAge = last ? 0f : entry.Ages[entry.Slot(1)];
                    var next = last ? here : entry.Positions[entry.Slot(1)];
                    var nextEdge = last ? hereEdge : entry.Edges[entry.Slot(1)];
                    if (age - nextAge > 1e-6f)
                    {
                        float along = (age - lifetime) / (age - nextAge);
                        position = Vector3.Lerp(position, next, along);
                        edge = Vector3.Lerp(edge, nextEdge, along);
                    }
                    age = lifetime;
                }

                Write(entry, position, edge, age / lifetime, entry.Blade ? MathF.Max(0f, 1f - age / oldest) : 1f);
            }
            if (head) Write(entry, here, hereEdge, 0f, 1f);

            if (entry.Blade) Round(first);

            _points[first].Flags |= FlagFirst;
            _points[_pointCount - 1].Flags |= FlagLast;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Write(Entry entry, Vector3 position, Vector3 edge, float t, float fade)
        {
            _points[_pointCount++] = new TrailPoint
            {
                Position = position,
                Width = entry.StartWidth + (entry.EndWidth - entry.StartWidth) * t,
                Color = Vector4.Lerp(entry.ColorStart, entry.ColorEnd, t) * fade,
                Flags = (entry.Blade ? FlagBlade : 0) | (entry.Lit ? FlagLit : 0),
                Edge = edge,
            };
        }

        /// <summary>
        /// Round a blade's points (from 'first' to the end of the list) in place: a curve through
        /// them (Catmull-Rom, both edges), BladeSteps pieces for every piece there was. The room for
        /// it has been made by the caller.
        /// </summary>
        private void Round(int first)
        {
            int count = _pointCount - first;
            if (count < 3) return;

            if (_round.Length < count) Array.Resize(ref _round, Math.Max(count, _round.Length * 2));
            Array.Copy(_points, first, _round, 0, count);

            int at = first;
            for (int i = 0; i < count - 1; i++)
            {
                ref readonly var before = ref _round[Math.Max(i - 1, 0)];
                ref readonly var from = ref _round[i];
                ref readonly var to = ref _round[i + 1];
                ref readonly var after = ref _round[Math.Min(i + 2, count - 1)];

                _points[at++] = from;
                for (int step = 1; step < BladeSteps; step++)
                {
                    float t = (float)step / BladeSteps;
                    _points[at++] = new TrailPoint
                    {
                        Position = CatmullRom(before.Position, from.Position, to.Position, after.Position, t),
                        Edge = CatmullRom(before.Edge, from.Edge, to.Edge, after.Edge, t),
                        Width = from.Width + (to.Width - from.Width) * t,
                        Color = Vector4.Lerp(from.Color, to.Color, t),
                        Flags = FlagBlade | (from.Flags & FlagLit),
                    };
                }
            }
            _points[at++] = _round[count - 1];
            _pointCount = at;
        }

        private static Vector3 CatmullRom(Vector3 a, Vector3 b, Vector3 c, Vector3 d, float t)
        {
            float t2 = t * t, t3 = t2 * t;
            return 0.5f * (2f * b + (c - a) * t + (2f * a - 5f * b + 4f * c - d) * t2 + (3f * b - a - 3f * c + d) * t3);
        }

        // ────────────── Render side (called by TrailRenderSystem) ──────────────

        /// <summary>Once per tick, before the first view enqueues: upload this frame's points. False when there is nothing to draw.</summary>
        internal bool PrepareFrame()
        {
            if (_preparedTick == Engine.TickCount) return _prepared;
            _preparedTick = Engine.TickCount;
            _prepared = false;

            if (_failed || _pointCount < 2) return false;

            try
            {
                if (_material == null)
                {
                    _effect = new Effect("trail_draw");
                    _material = new Material(_effect);
                    _buffer = new StreamingBuffer<TrailPoint>(Engine.Device, 1024);
                    _heaps = [Engine.Device.SrvHeap];
                }
            }
            catch (Exception ex)
            {
                _failed = true;
                Debug.LogError("TrailSystem", $"Init failed, trails are disabled: {ex.Message}");
                return false;
            }

            _buffer!.BulkWrite(_points.AsSpan(0, _pointCount));
            _uploadedPoints = _pointCount;
            return _prepared = true;
        }

        /// <summary>Forward pass: every trail of this frame in one draw.</summary>
        internal void Render(ID3D12GraphicsCommandList cmd)
        {
            if (_uploadedPoints < 2) return;

            var device = Engine.Device;
            float viewHeight = DeferredRenderer.Current?.Composite?.Native.Description.Height ?? 1080f;

            cmd.SetDescriptorHeaps(1, _heaps!);
            _material!.Apply(cmd, device);

            cmd.SetGraphicsRoot32BitConstant(0, _buffer!.SrvIndex, 0);                              // PointsIdx
            cmd.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(viewHeight), 1);    // ViewHeight

            // One instance per point but the last: the piece of ribbon from it to the next
            cmd.IASetPrimitiveTopology(PrimitiveTopology.TriangleList);
            cmd.DrawInstanced(6, (uint)(_uploadedPoints - 1), 0, 0);
        }
    }

    /// <summary>
    /// The render half of <see cref="TrailSystem"/>: runs once per rendered view while draws are being
    /// enqueued. The draw is a custom action of the Forward pass, after those of the IDraw components,
    /// so trails end up on top of what those draw (the ocean).
    /// </summary>
    [UpdateInEditor]
    [UpdateInGroup(typeof(RenderGroup))]
    [UpdateAfter(typeof(ScriptDrawSystem))]
    public sealed class TrailRenderSystem : EntitySystem
    {
        private TrailSystem? _trails;
        private Action<ID3D12GraphicsCommandList>? _render;

        protected internal override void Initialize()
        {
            _trails = Systems.Get<TrailSystem>();
            if (_trails == null) return;
            _render = _trails.Render;
        }

        public override void Update()
        {
            if (_trails == null || !_trails.PrepareFrame()) return;

            CommandBuffer.Enqueue(RenderPass.Forward, _render!);
        }
    }
}
