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
        private const float SamePlaceSq = 1e-8f;   // closer than a tenth of a millimetre: no piece of ribbon in between

        /// <summary>One point of the list. Must match struct TrailPoint in trail_draw.fx.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct TrailPoint
        {
            public Vector3 Position;   public float Width;
            public Vector4 Color;
            public uint Flags;         public Vector3 Pad;
        }

        /// <summary>Everything the system keeps per trail. The component itself only holds authored data.</summary>
        private sealed class Entry
        {
            public TrailRenderer? Trail;   // null once its component is gone: the ribbon fades out where it is

            // Ring of the places the entity has been, oldest first. The capacity is a power of two.
            public Vector3[] Positions = new Vector3[32];
            public float[] Ages = new float[32];
            public int Start;
            public int Count;
            public bool WasEmitting;

            // The look, kept for when the component is gone
            public float Lifetime = 0.5f;
            public float StartWidth, EndWidth;
            public Vector4 ColorStart, ColorEnd;

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
        }

        private readonly List<Entry> _entries = [];
        private TrailPoint[] _points = new TrailPoint[256];
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
            if (trail != null)
            {
                entry.Lifetime = MathF.Max(0.01f, trail.Lifetime);
                entry.StartWidth = MathF.Max(0f, trail.StartWidth);
                entry.EndWidth = MathF.Max(0f, trail.EndWidth);
                float intensity = MathF.Max(0f, trail.Intensity);
                entry.ColorStart = trail.ColorStart * new Vector4(intensity, intensity, intensity, 1f);
                entry.ColorEnd = trail.ColorEnd * new Vector4(intensity, intensity, intensity, 1f);

                emitting = trail.Enabled && trail.Emitting;
                here = trail.Transform!.WorldPosition;

                if (emitting)
                {
                    float spacing = MathF.Max(0.01f, trail.PointSpacing);
                    if (entry.Count == 0 || Vector3.DistanceSquared(here, entry.Positions[entry.Slot(entry.Count - 1)]) >= spacing * spacing)
                        entry.Push(here);
                }
                else if (entry.WasEmitting && entry.Count > 0
                    && Vector3.DistanceSquared(here, entry.Positions[entry.Slot(entry.Count - 1)]) > SamePlaceSq)
                {
                    // It stopped: the ribbon reaches as far as the entity got
                    entry.Push(here);
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

            if (_points.Length < _pointCount + count)
                Array.Resize(ref _points, Math.Max(_pointCount + count, _points.Length * 2));

            int first = _pointCount;
            for (int i = 0; i < entry.Count; i++)
            {
                int slot = entry.Slot(i);
                var position = entry.Positions[slot];
                float age = entry.Ages[slot];

                if (i == 0 && age > lifetime)
                {
                    // Run out, with a younger point after it: the end lies in between
                    bool last = i + 1 >= entry.Count;
                    float nextAge = last ? 0f : entry.Ages[entry.Slot(1)];
                    var next = last ? here : entry.Positions[entry.Slot(1)];
                    if (age - nextAge > 1e-6f)
                        position = Vector3.Lerp(position, next, (age - lifetime) / (age - nextAge));
                    age = lifetime;
                }

                Write(entry, position, age / lifetime);
            }
            if (head) Write(entry, here, 0f);

            _points[first].Flags |= FlagFirst;
            _points[_pointCount - 1].Flags |= FlagLast;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Write(Entry entry, Vector3 position, float t)
        {
            _points[_pointCount++] = new TrailPoint
            {
                Position = position,
                Width = entry.StartWidth + (entry.EndWidth - entry.StartWidth) * t,
                Color = Vector4.Lerp(entry.ColorStart, entry.ColorEnd, t),
            };
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
