using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Base;
using Freefall.Components;
using Vortice.Direct3D;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// A wake left on the water: a ribbon that lies on the surface along the way something went (a ship's stern, the
    /// shoulders of its bow), riding the sea's waves, widening and thinning out as it ages, its foam drawn in streaks along
    /// it (wake_draw.fx). Game code makes one (<see cref="WakeTrails.Create"/>), lays it every frame while it goes
    /// (<see cref="Lay"/>) and lets it go (<see cref="WakeTrails.Release"/>): what was laid fades out where it is.
    /// A bow wave's arm is a trail whose points drift sideways as they age (<see cref="Lay"/>'s drift): two of them open
    /// into the V behind a ship. Docs: .agent/knowledge/engine_internals/ships.md.
    /// </summary>
    public sealed class WakeTrail
    {
        /// <summary>Seconds a piece of the wake lasts.</summary>
        public float Lifetime = 30f;

        /// <summary>Its width where it is laid and at the end of its life (m): a wake spreads behind its ship.</summary>
        public float StartWidth = 4f, EndWidth = 12f;

        /// <summary>Metres between its points.</summary>
        public float Spacing = 2f;

        /// <summary>A bow wave's arm: a breaking crest, foam along its middle that breaks up fast. Else the churned water
        /// behind a stern: pale, aerated water with foam in streaks that thin out over its life.</summary>
        public bool Crest;

        internal struct Point
        {
            public Vector3 Position;   // y: the surface's height at rest (a lake's, a river's; the sea's level)
            public Vector2 Right;      // across the way it was laid (unit xz)
            public Vector2 Drift;      // m/s it moves sideways as it ages
            public float Along;        // metres along the trail where it was laid
            public float Age;
            public float Strength;     // 0..1 how much wake there was when it was laid
            public bool Sea;           // rides the sea's waves
        }

        internal Point[] Points = new Point[64];
        internal int Start, Count;
        internal Point Head;           // where it is laid now: the trail's live end
        internal bool HasHead, Emitting;
        internal float Along;
        internal bool Released;

        /// <summary>
        /// Where its end is this frame: <paramref name="at"/> on the water (y: the surface's height at rest),
        /// <paramref name="forward"/> the way it goes, <paramref name="strength"/> 0..1 how much wake it leaves (a ship's
        /// speed), <paramref name="drift"/> the sideways speed its points keep as they age (m/s, world xz; a bow wave's
        /// arm opens outward), <paramref name="sea"/> whether it rides the sea's waves. Main thread.
        /// </summary>
        public void Lay(Vector3 at, Vector3 forward, float strength, Vector2 drift, bool sea)
        {
            var dir = new Vector2(forward.X, forward.Z);
            float l = dir.Length();
            dir = l > 1e-4f ? dir / l : new Vector2(1f, 0f);
            var right = new Vector2(dir.Y, -dir.X);
            if (HasHead)
                Along += Vector2.Distance(new Vector2(Head.Position.X, Head.Position.Z), new Vector2(at.X, at.Z));
            Head = new Point
            {
                Position = at, Right = right, Drift = drift, Along = Along, Age = 0f,
                Strength = Math.Clamp(strength, 0f, 1f), Sea = sea,
            };
            HasHead = true;
            Emitting = true;

            // A point is kept each Spacing metres of the way
            if (Count == 0 || Vector2.Distance(new Vector2(Last.Position.X, Last.Position.Z), new Vector2(at.X, at.Z)) >= Spacing)
                Push(Head);
        }

        /// <summary>Stops laying it: the end stays where it was and the wake fades out behind.</summary>
        public void Stop()
        {
            if (Emitting && HasHead) Push(Head);
            Emitting = false;
            HasHead = false;
        }

        internal ref Point Last => ref Points[(Start + Count - 1) & (Points.Length - 1)];

        private void Push(Point p)
        {
            if (Count == Points.Length)
            {
                var grown = new Point[Points.Length * 2];
                for (int i = 0; i < Count; i++) grown[i] = Points[(Start + i) & (Points.Length - 1)];
                Points = grown;
                Start = 0;
            }
            Points[(Start + Count) & (Points.Length - 1)] = p;
            Count++;
        }

        // Ages and drifts its points, drops those that ran out
        internal void Step(float dt)
        {
            int mask = Points.Length - 1;
            for (int i = 0; i < Count; i++)
            {
                ref var p = ref Points[(Start + i) & mask];
                p.Age += dt;
                p.Position.X += p.Drift.X * dt;
                p.Position.Z += p.Drift.Y * dt;
            }
            while (Count > 0 && Points[Start].Age >= Lifetime)
            {
                Start = (Start + 1) & mask;
                Count--;
            }
        }
    }

    /// <summary>
    /// Keeps and draws every <see cref="WakeTrail"/>: one list of points a frame, one draw after the water (the OIT
    /// passes; the Forward pass's end with sorted transparency), as TrailSystem does for ribbons in the air.
    /// </summary>
    public static class WakeTrails
    {
        private static readonly List<WakeTrail> _trails = new();

        /// <summary>A new wake, laid from the next <see cref="WakeTrail.Lay"/> on. Main thread.</summary>
        public static WakeTrail Create()
        {
            var trail = new WakeTrail();
            _trails.Add(trail);
            return trail;
        }

        /// <summary>Lets a wake go: what was laid fades out where it is, then it is gone. Main thread.</summary>
        public static void Release(WakeTrail? trail)
        {
            if (trail == null || trail.Released) return;
            trail.Stop();
            trail.Released = true;
        }

        /// <summary>Every wake is gone at once (a scene load).</summary>
        public static void Clear() => _trails.Clear();

        private const uint FlagFirst = 1, FlagLast = 2, FlagSea = 4, FlagCrest = 8;

        /// <summary>One point of the list: wake_draw.fx WakePoint, 48 bytes.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct GpuPoint
        {
            public Vector3 Position; public float Width;
            public Vector2 Right; public float Along; public float Age;   // Age: share of its life gone, 0..1
            public float Strength; public uint Flags; public float Life; public float _pad;
        }

        private static GpuPoint[] _list = new GpuPoint[1024];
        private static int _listCount;
        private static int _steppedTick = -1, _preparedTick = -1;
        private static bool _prepared, _failed;
        private static Effect? _effect;
        private static Material? _material;
        private static StreamingBuffer<GpuPoint>? _buffer;
        private static ID3D12DescriptorHeap[]? _heaps;
        private static int _uploaded;

        // Once a tick: age the trails, drop the gone, write the list
        internal static void Step()
        {
            if (_steppedTick == Engine.TickCount) return;
            _steppedTick = Engine.TickCount;
            float dt = MathF.Min(Base.Time.SmoothDelta, 0.1f);
            _listCount = 0;
            for (int t = _trails.Count - 1; t >= 0; t--)
            {
                var trail = _trails[t];
                trail.Step(dt);
                if (trail.Released && trail.Count == 0)
                {
                    _trails.RemoveAt(t);
                    continue;
                }
                Write(trail);
            }
        }

        private static void Write(WakeTrail trail)
        {
            int n = trail.Count + (trail.Emitting && trail.HasHead ? 1 : 0);
            if (n < 2) return;
            if (_listCount + n > _list.Length) Array.Resize(ref _list, Math.Max(_list.Length * 2, _listCount + n));
            int mask = trail.Points.Length - 1;
            float life = MathF.Max(0.1f, trail.Lifetime);
            for (int i = 0; i < n; i++)
            {
                var p = i < trail.Count ? trail.Points[(trail.Start + i) & mask] : trail.Head;
                float age = Math.Clamp(p.Age / life, 0f, 1f);
                uint flags = (p.Sea ? FlagSea : 0) | (trail.Crest ? FlagCrest : 0);
                if (i == 0) flags |= FlagFirst;
                if (i == n - 1) flags |= FlagLast;
                _list[_listCount++] = new GpuPoint
                {
                    Position = p.Position,
                    Width = trail.StartWidth + (trail.EndWidth - trail.StartWidth) * MathF.Sqrt(age),
                    Right = p.Right, Along = p.Along, Age = age, Strength = p.Strength, Flags = flags, Life = life,
                };
            }
        }

        internal static bool PrepareFrame()
        {
            Step();
            if (_preparedTick == Engine.TickCount) return _prepared;
            _preparedTick = Engine.TickCount;
            _prepared = false;
            if (_failed || _listCount < 2) return false;
            try
            {
                if (_material == null)
                {
                    _effect = new Effect("wake_draw");
                    _material = new Material(_effect);
                    _buffer = new StreamingBuffer<GpuPoint>(Engine.Device, 1024);
                    _heaps = [Engine.Device.SrvHeap];
                }
            }
            catch (Exception ex)
            {
                _failed = true;
                Debug.LogError("WakeTrails", $"Init failed, wakes are not drawn: {ex.Message}");
                return false;
            }
            _buffer!.BulkWrite(_list.AsSpan(0, _listCount));
            _uploaded = _listCount;
            return _prepared = true;
        }

        internal static void Render(ID3D12GraphicsCommandList cmd)
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

        internal static void RenderOITMoments(ID3D12GraphicsCommandList cmd) => Draw(cmd, RenderPass.OITMoments);

        internal static void RenderOIT(ID3D12GraphicsCommandList cmd) => Draw(cmd, RenderPass.OIT);

        private static void Draw(ID3D12GraphicsCommandList cmd, RenderPass pass)
        {
            if (_uploaded < 2) return;
            var device = Engine.Device;
            cmd.SetDescriptorHeaps(1, _heaps!);
            _material!.SetPass(pass);
            _material.Apply(cmd, device);
            _material.SetPass(RenderPass.Forward);

            uint ocean = 0;
            var oceans = ComponentCache<OceanRenderer>.All;
            for (int i = 0; i < oceans.Count; i++)
                if (oceans[i] is { Enabled: true } o && o.SurfDataSrv != 0) { ocean = o.SurfDataSrv; break; }
            cmd.SetGraphicsRoot32BitConstant(0, _buffer!.SrvIndex, 0);   // PointsIdx
            cmd.SetGraphicsRoot32BitConstant(0, ocean, 1);               // OceanIdx
            cmd.SetGraphicsRoot32BitConstant(0, WaterHulls.Srv, 2);      // HullsIdx
            cmd.SetGraphicsRoot32BitConstant(0, WaterHulls.Count, 3);    // HullCount

            cmd.IASetPrimitiveTopology(PrimitiveTopology.TriangleList);
            cmd.DrawInstanced(6, (uint)(_uploaded - 1), 0, 0);
        }
    }

    /// <summary>
    /// The render half of <see cref="WakeTrails"/>: once per rendered view while draws are enqueued, after the Forward
    /// pass's batches (the water), and in the OIT passes.
    /// </summary>
    [UpdateInEditor]
    [UpdateInGroup(typeof(RenderGroup))]
    [UpdateAfter(typeof(ScriptDrawSystem))]
    public sealed class WakeTrailRenderSystem : EntitySystem
    {
        private static readonly Action<ID3D12GraphicsCommandList> _render = WakeTrails.Render;
        private static readonly Action<ID3D12GraphicsCommandList> _oitMoments = WakeTrails.RenderOITMoments;
        private static readonly Action<ID3D12GraphicsCommandList> _oit = WakeTrails.RenderOIT;

        public override void Update()
        {
            if (!WakeTrails.PrepareFrame()) return;
            CommandBuffer.EnqueueLate(RenderPass.Forward, _render);
            if (OrderIndependentTransparency.MomentsDue && OrderIndependentTransparency.Active)
                CommandBuffer.Enqueue(RenderPass.OITMoments, _oitMoments);
            if (OrderIndependentTransparency.Active)
                CommandBuffer.Enqueue(RenderPass.OIT, _oit);
        }
    }
}
