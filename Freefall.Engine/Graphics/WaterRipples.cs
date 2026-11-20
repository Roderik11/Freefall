using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Ripples on the water round the camera: a height field <see cref="Size"/>² over <see cref="Span"/> metres, kept
    /// centred on the camera (moved by whole cells, its waves going along), stepped at 60 Hz by the wave equation
    /// (water_ripples_cs.hlsl) and stirred by whatever moves in the water (<see cref="Disturb"/>: a swimmer's strokes and
    /// wake, a wader's steps, something falling in). Lakes, rivers and the sea read its slopes as ripples on their own
    /// (water.fx, ocean.fx: water_ripples.hlsli), and its strongest crests as a little white water. It runs only while
    /// something stirred it in the last <see cref="LivesFor"/> seconds; otherwise nothing is done and the shaders skip it.
    /// Docs: .agent/knowledge/ocean_system/underwater.md "Ripples".
    /// </summary>
    public static class WaterRipples
    {
        /// <summary>Cells a side.</summary>
        public const int Size = 512;

        /// <summary>Metres a side: ~9 cm cells, rings out to 24 m from the camera.</summary>
        public const float Span = 48f;

        public const float Cell = Span / Size;

        /// <summary>How fast a ripple runs (m/s). The step is stable while it is under Cell / dt / √2 (2.5 m/s here).</summary>
        public const float WaveSpeed = 1.1f;

        /// <summary>
        /// What is left of a ripple's height after a second (the step's damping, 60 steps a second). 0.15 left nothing a metre
        /// from a swimmer: the rings were gone before they spread (2026-11-15).
        /// </summary>
        public const float KeptPerSecond = 0.7f;

        /// <summary>
        /// What is left of the foam a splash or a wader leaves on the surface after a second (it also spreads a little):
        /// a splash's patch shows for 3-5 s.
        /// </summary>
        public const float FoamKeptPerSecond = 0.6f;

        /// <summary>Seconds the water goes on moving after the last stir.</summary>
        public const float LivesFor = 10f;

        private const float StepSeconds = 1f / 60f;
        private const int MaxStepsPerFrame = 3;
        private const int MaxDrops = 64;
        private const int FrameCount = 3;

        /// <summary>One stir of the water (water_ripples_cs.hlsl RippleDrop, 32 bytes).</summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct Drop
        {
            public Vector2 XZ;
            public float Radius;
            public float Strength;      // m of height pushed
            public float Foam;          // foam left (0..1.5 coverage)
            public float Pad0, Pad1, Pad2;
        }

        private static readonly object _lock = new();
        private static readonly List<Drop> _pending = new();
        private static readonly Drop[] _taken = new Drop[MaxDrops];
        private static float _alive;
        private static double _owed;

        private static ComputeShader? _shader;
        private static int _kStep, _kClear, _kShift;
        private static RenderTexture2D[]? _heights;        // h at t-1, t, and the one written next
        private static ResourceStates[] _states = Array.Empty<ResourceStates>();
        private static int _prev, _cur;
        private static readonly GraphicsBuffer?[] _drops = new GraphicsBuffer?[FrameCount];
        private static Vector2 _origin;                    // world xz of the grid's corner (cell 0, 0)
        private static bool _cleared;
        private static int _steppedTick = -1;

        /// <summary>The ripples' heights now (Texture2D&lt;float&gt;, m), 0 while the water is still.</summary>
        public static uint Srv { get; private set; }

        /// <summary>World xz of the ripple grid's corner, for the shaders (the grid spans <see cref="Span"/>).</summary>
        public static Vector2 Origin => _origin;

        /// <summary>
        /// Stir the water once at a place: a smooth push <paramref name="radius"/> metres round <paramref name="at"/>, by
        /// <paramref name="strength"/> metres of height (negative pushes down: a body going in, an arm coming down). For a
        /// splash or a stroke; something that goes on moving in the water uses <see cref="Churn"/>. Any thread; places off
        /// the grid, or more than 64 stirs before the next step, are left out.
        /// </summary>
        public static void Disturb(Vector3 at, float radius, float strength)
        {
            if (strength == 0f) return;
            Add(at, radius, strength, 0f);
        }

        /// <summary>
        /// Leave foam on the water at a place: <paramref name="amount"/> of coverage (1 = white water, up to 1.5) over
        /// <paramref name="radius"/> metres round <paramref name="at"/>, fading by <see cref="FoamKeptPerSecond"/> and
        /// spreading a little. Once for a splash; every frame for something churning the water, amount × the frame's time.
        /// The water's own foam pattern breaks it up (water.fx, ocean.fx). Any thread.
        /// </summary>
        public static void Foam(Vector3 at, float radius, float amount)
        {
            if (amount <= 0f) return;
            Add(at, radius, 0f, amount);
        }

        private static void Add(Vector3 at, float radius, float strength, float foam)
        {
            if (!Engine.Settings.WaterRipples || radius <= 0f) return;
            lock (_lock)
            {
                if (_pending.Count < MaxDrops)
                    _pending.Add(new Drop { XZ = new Vector2(at.X, at.Z), Radius = radius, Strength = strength, Foam = foam });
                _alive = LivesFor;
            }
        }

        /// <summary>
        /// Something that goes on moving in the water, every frame (a swimmer treading water or swimming, a wader's legs): a
        /// push that rises and falls <paramref name="hz"/> times a second, so that it rings out. A steady push only sank a dip
        /// round the body that never moved, its rim as white water (2026-11-15). <paramref name="rate"/>: metres of height a
        /// second at the push's peak (0.08 over 0.6 m at 1.4 Hz rings out ~4 m round a swimmer); <paramref name="phase"/>:
        /// 0..1, its own, so that two side by side do not ring in step. Scaled by the frame's time, so the frame rate does
        /// not change it. Any thread.
        /// </summary>
        public static void Churn(Vector3 at, float radius, float rate, float hz, float phase)
        {
            if (!Engine.Settings.WaterRipples || rate == 0f) return;
            float cycle = Base.Time.TotalTime * hz + phase;
            Disturb(at, radius, rate * MathF.Sin((cycle - MathF.Floor(cycle)) * MathF.Tau) * Base.Time.SmoothDelta);
        }

        /// <summary>
        /// The frame's steps, before the water is drawn (DeferredRenderer, ahead of the forward pass): the grid follows the
        /// camera by whole cells, the stirs of the frame go into its first step, and as many steps as the time owes (at
        /// most <see cref="MaxStepsPerFrame"/>). Leaves the texture readable by the pixel shaders.
        /// </summary>
        internal static unsafe void Step(ID3D12GraphicsCommandList list, Vector3 camera)
        {
            // Once a tick, by the first view drawn (the editor's other viewports read the same water)
            if (_steppedTick == Engine.TickCount) return;
            _steppedTick = Engine.TickCount;
            // The hulls nearest the camera keep the water out (WaterHulls): measured from here
            WaterHulls.Eye = camera;
            float dt = MathF.Min(Base.Time.SmoothDelta, 0.1f);
            int count = 0;
            bool live = Engine.Settings.WaterRipples;
            lock (_lock)
            {
                _alive -= dt;
                live &= _alive > 0f;
                if (!live)
                {
                    _alive = MathF.Max(_alive, 0f);
                    _pending.Clear();
                }
                else
                {
                    // The stirs wait for a frame with a step: at 140 fps more than half the frames have none, and their
                    // stirs were taken and dropped (a swimmer's push came through at less than half its strength)
                    _owed = Math.Min(_owed + dt, MaxStepsPerFrame * StepSeconds);
                    if (_owed >= StepSeconds)
                    {
                        count = Math.Min(_pending.Count, MaxDrops);
                        for (int i = 0; i < count; i++) _taken[i] = _pending[i];
                        _pending.Clear();
                    }
                }
            }
            if (!live)
            {
                Srv = 0;
                _owed = 0;
                _cleared = false;
                return;
            }
            // No step owed this frame: the water reads the last one's heights
            if (_owed < StepSeconds && _cleared) return;

            EnsureResources();
            var device = Engine.Device;
            list.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            list.SetComputeRootSignature(device.GlobalRootSignature);
            PixMarker.Begin(list, "Water ripples");

            // Woken: the grid starts still where the camera is
            var origin = new Vector2(MathF.Floor(camera.X / Cell) * Cell - Span * 0.5f, MathF.Floor(camera.Z / Cell) * Cell - Span * 0.5f);
            if (!_cleared)
            {
                _origin = origin;
                for (int i = 0; i < _heights!.Length; i++)
                {
                    ToState(list, i, ResourceStates.UnorderedAccess);
                    _shader!.SetPushConstant(_kClear, "OutUAV", _heights[i].UavIndex);
                    _shader.Dispatch(_kClear, list, Size / 8, Size / 8);
                }
                list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
                _cleared = true;
            }
            int shiftX = (int)MathF.Round((origin.X - _origin.X) / Cell);
            int shiftZ = (int)MathF.Round((origin.Y - _origin.Y) / Cell);

            // The grid follows the camera: h(t) into the free texture and h(t−1) into h(t)'s, both moved, before any step.
            // Moving only what a step read left h(t−1) of the next step in the old grid, and every move read as motion
            if (shiftX != 0 || shiftZ != 0)
            {
                int free = 3 - _prev - _cur;
                Shift(list, _cur, free, shiftX, shiftZ);
                Shift(list, _prev, _cur, shiftX, shiftZ);
                (_prev, _cur) = (_cur, free);
                _origin = origin;
            }

            int slot = Engine.FrameIndex % FrameCount;
            var drops = _drops[slot] ??= GraphicsBuffer.CreateUpload<Drop>(MaxDrops, mapped: true);
            var write = drops.WritePtr<Drop>();
            for (int i = 0; i < count; i++) write[i] = _taken[i];

            float courant = WaveSpeed * StepSeconds / Cell;
            float damping = MathF.Pow(KeptPerSecond, StepSeconds);
            float foamKept = MathF.Pow(FoamKeptPerSecond, StepSeconds);
            bool first = true;
            while (_owed >= StepSeconds)
            {
                _owed -= StepSeconds;
                int next = 3 - _prev - _cur;
                ToState(list, _prev, Read);
                ToState(list, _cur, Read);
                ToState(list, next, ResourceStates.UnorderedAccess);
                var newOrigin = _origin;
                var s = _shader!;
                s.SetPushConstant(_kStep, "PrevSrv", _heights![_prev].BindlessIndex);
                s.SetPushConstant(_kStep, "CurSrv", _heights[_cur].BindlessIndex);
                s.SetPushConstant(_kStep, "OutUAV", _heights[next].UavIndex);
                s.SetPushConstant(_kStep, "DropsSrv", drops.SrvIndex);
                s.SetPushConstant(_kStep, "DropCount", first ? (uint)count : 0u);
                s.SetPushConstant(_kStep, "OriginXBits", BitConverter.SingleToUInt32Bits(newOrigin.X));
                s.SetPushConstant(_kStep, "OriginZBits", BitConverter.SingleToUInt32Bits(newOrigin.Y));
                s.SetPushConstant(_kStep, "Courant2Bits", BitConverter.SingleToUInt32Bits(courant * courant));
                s.SetPushConstant(_kStep, "DampingBits", BitConverter.SingleToUInt32Bits(damping));
                s.SetPushConstant(_kStep, "FoamKeptBits", BitConverter.SingleToUInt32Bits(foamKept));
                s.Dispatch(_kStep, list, Size / 8, Size / 8);
                _origin = newOrigin;
                _prev = _cur;
                _cur = next;
                first = false;
            }

            // The water's pixel shaders read the latest heights, from the next frame's draws (this frame's were queued
            // before this step with the last frame's: none of the three is left to be written)
            for (int i = 0; i < 3; i++) ToState(list, i, Read);
            Srv = _heights![_cur].BindlessIndex;
            PixMarker.End(list);
        }

        private static void EnsureResources()
        {
            if (_heights != null) return;
            _shader = new ComputeShader("water_ripples_cs.hlsl");
            _kStep = _shader.FindKernel("CSStep");
            _kClear = _shader.FindKernel("CSClear");
            _kShift = _shader.FindKernel("CSShift");
            _heights = new RenderTexture2D[3];
            _states = new ResourceStates[3];
            for (int i = 0; i < 3; i++)
            {
                _heights[i] = new RenderTexture2D(Engine.Device, Size, Size, Format.R16G16_Float, randomWrite: true);   // height, foam
                _states[i] = ResourceStates.Common;
            }
            _prev = 0;
            _cur = 1;
        }

        // A height field moved by whole cells into another texture (CSShift)
        private static void Shift(ID3D12GraphicsCommandList list, int from, int to, int shiftX, int shiftZ)
        {
            ToState(list, from, Read);
            ToState(list, to, ResourceStates.UnorderedAccess);
            var s = _shader!;
            s.SetPushConstant(_kShift, "CurSrv", _heights![from].BindlessIndex);
            s.SetPushConstant(_kShift, "OutUAV", _heights[to].UavIndex);
            s.SetPushConstant(_kShift, "ShiftX", unchecked((uint)shiftX));
            s.SetPushConstant(_kShift, "ShiftZ", unchecked((uint)shiftZ));
            s.Dispatch(_kShift, list, Size / 8, Size / 8);
        }

        // Read by the step (compute) and by the water (pixel shaders)
        private const ResourceStates Read = ResourceStates.PixelShaderResource | ResourceStates.NonPixelShaderResource;

        private static void ToState(ID3D12GraphicsCommandList list, int i, ResourceStates state)
        {
            if (_states[i] == state) return;
            list.ResourceBarrierTransition(_heights![i].Native, _states[i], state);
            _states[i] = state;
        }
    }
}
