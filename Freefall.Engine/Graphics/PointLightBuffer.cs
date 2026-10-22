using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.Mathematics;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// Point lights on the GPU: a persistent slot per PointLight in a registry the GPU reads every frame. Nothing
    /// is gathered per frame: a light's settings write its slot when they change (PointLight's property setters,
    /// OnEnabledChanged), lights take and free slots as they enter and leave ComponentCache&lt;PointLight&gt;, the
    /// position comes from the light entity's world matrix in the global transform buffer, and the flicker is
    /// computed in the shader. The GPU culls the registry into each renderer's visible list
    /// (point_light_cull.hlsl), which the tiled lighting and the shadow selection read. Lamps (OnlyWhenDark) are judged
    /// once per frame by the daylight where each one hangs (<see cref="UpdateLamps"/>, point_light_lamps.hlsl).
    /// Shader side: point_lights.hlsli.
    /// </summary>
    public sealed class PointLightBuffer : IDisposable
    {
        /// <summary>Visible lights per renderer and frame (point_lights.hlsli MAX_VISIBLE_LIGHTS).</summary>
        public const int MaxVisible = 4096;

        private const uint Enabled = 1;
        private const uint OnlyWhenDark = 2;    // lit by the daylight where it hangs (UpdateLamps)

        /// <summary>One per light. Must match PointLightRecord in point_lights.hlsli (48 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Record
        {
            public Vector3 Color;
            public float Intensity;
            public uint TransformSlot;
            public float Range;
            public float Flicker;
            public float FlickerSpeed;
            public float FlickerPhase;
            public uint Flags;
            public float InnerRadius;
            private uint _pad0;
        }

        /// <summary>A light in view, written by the GPU. Must match PointLightData in point_lights.hlsli (48 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct VisibleLight
        {
            public Vector3 Color;
            public float Intensity;
            public Vector3 Position;
            public float Range;
            public uint ShadowLight;
            public uint Slot;
            public float InnerRadius;
            public float Score;
        }

        /// <summary>The camera and frame for the cull and the shadow selection. Must match PointLightFrame (224 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct FrameParams
        {
            public Matrix4x4 CameraRelativeVP;
            public Vector4 Plane0, Plane1, Plane2, Plane3, Plane4;  // camera-relative left, right, bottom, top, near
            public Vector3 CameraPosition;
            public float Time;
            public float PixelsPerMetre;
            public float FadeStep;
            public float SunLight;          // lux on a surface facing the sun (or moon), clouds' average shade included
            public uint ShadowBudget;
            public uint Frame;
            public float Ambient;           // lux from the whole sky on level ground
            public float Lamps;             // Daylight.Lamps: for a lamp UpdateLamps has not judged yet
            public uint TerrainHeightmap;   // sky_visibility.hlsli: daylight reaches a light only past the terrain
            public float TerrainMaxHeight;  // (Atmosphere.TerrainCover); 0 = no terrain
            public Vector3 TerrainOrigin;
            public Vector2 TerrainSize;
            public float SunHeight;         // Daylight.SunHeight: SunLight × it = the sun's lux on level ground
            public float LampFadeStep;      // Δt / Daylight.LampFadeSeconds
        }

        /// <summary>Per registry slot: whether a lamp (OnlyWhenDark) is lit. Must match LampState in point_lights.hlsli (16 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct LampState
        {
            public float Lamp;          // a lamp: 0 = out, 1 = lit; < 0 = judged as a plain light
            public float SurfaceLux;    // daylight on the surfaces round the light, with the bounce (shadow selection)
            public uint Seen;           // tick it was judged (UpdateLamps' TickCount + 1) | LampLit, 0 = never
            public float Lux;           // daylight where it hangs, on level ground, no bounce (a lamp switches on it)
        }

        private const uint LampLit = 0x80000000;    // LampState.Seen: the lamp's hysteresis side (point_lights.hlsli LAMP_LIT)

        public static PointLightBuffer? Instance { get; private set; }

        private const int FrameCount = 3;
        private const int LampsPerGroup = 64 / SkyCapture.ViewCount;   // point_light_lamps.hlsl: a thread per sky direction

        private readonly StreamingBuffer<Record> _records;
        private readonly Stack<int> _freeSlots = new();
        private readonly object _lock = new();
        private PointLight?[] _lights;
        private int _nextSlot;

        // Lamps: lit by the daylight where each hangs (point_light_lamps.hlsl), judged once per frame for every slot
        private ComputeShader? _lampsCS;
        private int _kLamps;
        private GraphicsBuffer? _lampStates;
        private int _lampCapacity;
        private bool _clearLamps;
        private readonly List<(GraphicsBuffer Buffer, long DisposeAfterTick)> _retired = new();
        private readonly GraphicsBuffer?[] _lampReadback = new GraphicsBuffer?[FrameCount];
        private readonly int[] _lampReadbackSlots = new int[FrameCount];   // slots each readback holds, 0 = not written yet
        private int _lampsTick = -1;

        /// <summary>Registry slots handed out so far: the cull walks this many.</summary>
        public int SlotCount => _nextSlot;

        /// <summary>Slots the GPU buffer holds (per-slot buffers elsewhere size by this).</summary>
        public int Capacity => _records.Capacity;

        public uint SrvIndex => _records.SrvIndex;

        public static void Initialize(GraphicsDevice device)
        {
            if (Instance != null) return;
            Instance = new PointLightBuffer(device);
            ComponentCache<PointLight>.Added += Instance.Register;
            ComponentCache<PointLight>.Removed += Instance.Unregister;
            foreach (var light in ComponentCache<PointLight>.All)
                Instance.Register(light);
        }

        private PointLightBuffer(GraphicsDevice device)
        {
            _records = new StreamingBuffer<Record>(device, 1024);
            _lights = new PointLight?[_records.Capacity];
        }

        /// <summary>The light at a registry slot (editor stats), or null.</summary>
        public PointLight? LightAt(int slot)
        {
            lock (_lock)
                return slot >= 0 && slot < _lights.Length ? _lights[slot] : null;
        }

        // Called again for a light that is already in (ComponentCache.Added may follow Initialize's walk)
        private void Register(PointLight light)
        {
            lock (_lock)
            {
                if (light.GpuSlot < 0)
                {
                    int slot = _freeSlots.Count > 0 ? _freeSlots.Pop() : _nextSlot++;
                    _records.EnsureCapacity(slot + 1);
                    if (_lights.Length < _records.Capacity)
                        Array.Resize(ref _lights, _records.Capacity);
                    _lights[slot] = light;
                    light.GpuSlot = slot;
                }
            }

            // A transform slot taken on first access holds identity until the transform next changes
            var transform = light.Entity?.Transform;
            if (transform != null)
                TransformBuffer.Instance.SetTransform(transform.TransformSlot, transform.Matrix);
            Changed(light);
        }

        private void Unregister(PointLight light)
        {
            lock (_lock)
            {
                int slot = light.GpuSlot;
                if (slot < 0) return;
                _records.Set(slot, default); // disabled
                _lights[slot] = null;
                _freeSlots.Push(slot);
                light.GpuSlot = -1;
            }
        }

        /// <summary>Writes a light's slot from its settings. PointLight calls it when one of them changes.</summary>
        internal void Changed(PointLight light)
        {
            int slot = light.GpuSlot;
            if (slot < 0) return;

            int transformSlot = light.Entity?.Transform?.TransformSlot ?? -1;
            // The colour is only the hue: scaled to luminance 1, so Lumens alone says how bright the light is
            var c = new Vector3(light.Color.R, light.Color.G, light.Color.B);
            float luminance = 0.2126f * c.X + 0.7152f * c.Y + 0.0722f * c.Z;
            _records.Set(slot, new Record
            {
                Color = luminance > 1e-4f ? c / luminance : Vector3.Zero,
                Intensity = light.Lumens / (4f * MathF.PI),     // luminous intensity (candela): lux at 1 m
                TransformSlot = (uint)Math.Max(transformSlot, 0),
                Range = light.Range,
                Flicker = light.Flicker,
                FlickerSpeed = light.FlickerSpeed,
                FlickerPhase = (light.Id & 1023) * 0.37f, // neighbouring lights do not waver in step
                Flags = (light.Enabled && transformSlot >= 0 ? Enabled : 0) | (light.OnlyWhenDark ? OnlyWhenDark : 0),
                InnerRadius = MathF.Max(0f, light.InnerRadius),
            });
        }

        /// <summary>Copies the slots that changed into this frame's GPU buffer. Once per frame, before the cull.</summary>
        public void Upload()
        {
            _records.Upload();
            StreamingBuffer<Record>.FlushDeferredDisposals();
        }

        /// <summary>
        /// The daylight at each light per slot (RWStructuredBuffer of point_lights.hlsli LampState), as
        /// <see cref="UpdateLamps"/> judged it last frame: how lit each light marked OnlyWhenDark is (the light cull) and
        /// what each light's glow competes with (the point light shadow selection). Call with the global root signature's
        /// descriptor heap set (a new buffer is cleared).
        /// </summary>
        public uint LampStatesUav(ID3D12GraphicsCommandList list)
        {
            EnsureLampStates(list);
            return _lampStates!.UavIndex;
        }

        /// <summary>
        /// Judges the daylight where every enabled light hangs, past the terrain and the sky captures round the camera
        /// (point_light_lamps.hlsl), and lights the lamps (OnlyWhenDark) by it, with a hysteresis and a fade per lamp. Once
        /// per frame, by the first renderer to get here, after the sky captures are drawn; the light cull and the shadow
        /// selection read it the frame after. Sets the compute root signature and descriptor heap.
        /// </summary>
        /// <param name="frameParamsSrv">This renderer's PointLightFrame (sun, sky, terrain, lamp fade step).</param>
        public void UpdateLamps(ID3D12GraphicsCommandList list, uint frameParamsSrv)
        {
            if (_lampsTick == Engine.TickCount || SlotCount == 0)
                return;
            _lampsTick = Engine.TickCount;

            if (_lampsCS == null)
            {
                _lampsCS = new ComputeShader("point_light_lamps.hlsl");
                _kLamps = _lampsCS.FindKernel("CSLamps");
            }

            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            EnsureLampStates(list);

            var captures = SkyCapture.Current;
            bool meshes = captures.Views != 0 && captures.Depth != 0 && captures.Count == SkyCapture.ViewCount;
            PixMarker.Begin(list, "Lamps");
            _lampsCS.SetPushConstant("Records", _records.SrvIndex);
            _lampsCS.SetPushConstant("RecordCount", (uint)SlotCount);
            _lampsCS.SetPushConstant("Transforms", TransformBuffer.Instance.SrvIndex);
            _lampsCS.SetPushConstant("FrameParams", frameParamsSrv);
            _lampsCS.SetPushConstant("LampsUAV", _lampStates!.UavIndex);
            _lampsCS.SetPushConstant("Tick", (uint)Engine.TickCount + 1);
            _lampsCS.SetPushConstant("SkyCaptureViews", meshes ? captures.Views : 0u);
            _lampsCS.SetPushConstant("SkyCaptureDepth", meshes ? captures.Depth : 0u);
            _lampsCS.SetPushConstant("SkyCaptureBounce", BitConverter.SingleToUInt32Bits(Math.Clamp(Engine.Settings.SkyCaptureBounce, 0f, 1f)));
            _lampsCS.Dispatch(_kLamps, list, (uint)((SlotCount + LampsPerGroup - 1) / LampsPerGroup));
            list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_lampStates.Native)));

            // For the stats (read FrameCount - 1 frames later)
            int r = Engine.FrameIndex % FrameCount;
            var readback = _lampReadback[r];
            if (readback == null || readback.ElementCount < _lampCapacity)
            {
                if (readback != null)
                    _retired.Add((readback, Engine.TickCount + FrameCount + 1));
                readback = _lampReadback[r] = GraphicsBuffer.CreateReadback<LampState>(_lampCapacity);
            }
            int slots = Math.Min(SlotCount, _lampCapacity);
            list.ResourceBarrierTransition(_lampStates.Native, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            list.CopyBufferRegion(readback.Native, 0, _lampStates.Native, 0, (ulong)(slots * Marshal.SizeOf<LampState>()));
            list.ResourceBarrierTransition(_lampStates.Native, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
            _lampReadbackSlots[r] = slots;
            PixMarker.End(list);
        }

        // The lamp states follow the registry's capacity; a new buffer starts every lamp as it should be
        private void EnsureLampStates(ID3D12GraphicsCommandList list)
        {
            for (int i = _retired.Count - 1; i >= 0; i--)
            {
                if (Engine.TickCount < _retired[i].DisposeAfterTick) continue;
                _retired[i].Buffer.Dispose();
                _retired.RemoveAt(i);
            }

            int capacity = Capacity;
            if (_lampStates == null || _lampCapacity < capacity)
            {
                // Frames still in flight may use the old one
                if (_lampStates != null)
                    _retired.Add((_lampStates, Engine.TickCount + FrameCount + 1));
                _lampCapacity = Math.Max(capacity, 1024);
                _lampStates = GraphicsBuffer.CreateStructured<LampState>(_lampCapacity, uav: true);
                _clearLamps = true;
            }
            if (_clearLamps)
            {
                _lampStates.ClearUAV(list, new Int4(0, 0, 0, 0));
                list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_lampStates.Native)));
                _clearLamps = false;
            }
        }

        /// <summary>
        /// For engine_stats: the lamps (enabled lights marked OnlyWhenDark) as judged FrameCount - 1 frames ago: how many
        /// are lit, the open ground's daylight and lamps (Daylight), the share of it that reaches the lamps (the middle and
        /// the darkest tenth), and the lamps lit while those on open ground are out (in rooms, caves).
        /// </summary>
        public unsafe object DescribeLamps()
        {
            int r = (Engine.FrameIndex + 1) % FrameCount;
            var readback = _lampReadback[r];
            int slots = readback != null ? _lampReadbackSlots[r] : 0;
            var states = slots > 0 ? readback!.ReadPtr<LampState>() : null;

            int lamps = 0, lit = 0, fading = 0;
            float open = Daylight.GroundIlluminance();
            var shares = new List<float>();
            var litByDay = new List<object>();
            for (int s = 0; s < slots; s++)
            {
                var light = LightAt(s);
                if (light == null || !light.Enabled || !light.OnlyWhenDark || states![s].Seen == 0 || states[s].Lamp < 0f)
                    continue;
                var state = states[s];
                bool isLit = (state.Seen & LampLit) != 0;
                lamps++;
                if (isLit) lit++;
                if (state.Lamp > 0f && state.Lamp < 1f) fading++;
                if (open > 0f) shares.Add(state.Lux / open);
                if (isLit && Daylight.Lamps < 1f && litByDay.Count < 32)
                    litByDay.Add(new { entity = light.Entity?.Id ?? 0, name = light.Entity?.Name, lux = MathF.Round(state.Lux, 3) });
            }
            shares.Sort();
            return new
            {
                lamps,
                lit,
                fading,
                openGroundLux = open,
                openGroundLamps = Daylight.Lamps,
                daylightShareMedian = shares.Count > 0 ? MathF.Round(shares[shares.Count / 2], 3) : (float?)null,
                daylightShareLowestTenth = shares.Count > 0 ? MathF.Round(shares[shares.Count / 10], 3) : (float?)null,
                litByDay,
            };
        }

        public void Dispose()
        {
            _records.Dispose();
            _lampsCS?.Dispose();
            _lampStates?.Dispose();
            foreach (var (buffer, _) in _retired) buffer.Dispose();
            foreach (var b in _lampReadback) b?.Dispose();
        }
    }
}
