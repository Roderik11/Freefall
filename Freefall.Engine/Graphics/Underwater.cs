using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// The camera in the water (underwater_cs.hlsl), after the forward pass has drawn the water and before the glass and
    /// particles: every pixel whose near-plane point is under the surface sees through water. What it sees fades into the
    /// water's own light with the distance (the same Beer-Lambert as the surface's view of its bed: ShallowColor over
    /// Visibility), the light that reaches things down there is dimmed by the water above them and plays over them as
    /// caustics, and where the near plane crosses the surface (the camera half in) a waterline runs across the picture.
    /// The surfaces themselves are drawn from below by water.fx and ocean.fx (Snell's window, the mirror past it).
    ///
    /// The surface near the camera: a lake's level or a river's slope from Water.SurfaceAt (the CPU); the sea's with its
    /// waves, measured on the GPU at the camera by the probe kernel from the ocean's own displacement (ocean_surf.hlsli),
    /// at the places swimmers asked for too (Water.WatchSea), read back a few frames later for them.
    /// Docs: .agent/knowledge/ocean_system/underwater.md.
    /// </summary>
    public sealed class Underwater : IDisposable
    {
        /// <summary>This frame's water at the camera. Must match UnderwaterFrame in underwater_cs.hlsl (112 bytes).</summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct Frame
        {
            public Vector2 Slope; public float SurfaceY; public uint Sea;         // the surface near the camera (Sea: take it from the probe)
            public Vector3 ShallowColor; public float Visibility;                  // the water's look (WaterBody / OceanRenderer)
            public Vector3 WaterColor; public float Caustics;
            public Vector3 SunDirection; public float SunUp;                       // the light's way (from the sun) and how high it stands
            public Vector3 SunColor; public float SunIntensity;
            public uint DepthSrv, NormalsSrv, ProbeCount, OceanData;
            public uint LinearDepthSrv;                                            // the opaque scene's view depth: what is not water
            public float Shafts;                                                   // EngineSettings.UnderwaterShafts
            public uint Pad1, Pad2;
        }

        private const int FrameCount = 3;
        private readonly ComputeShader _shader;
        private readonly int _kWater, _kProbe;
        private readonly GraphicsBuffer[] _frames = new GraphicsBuffer[FrameCount];
        private readonly GraphicsBuffer[] _places = new GraphicsBuffer[FrameCount];   // the probe's places (xz, as float4)
        private readonly GraphicsBuffer _probe;                                       // its answers: height, slope x, slope z, 1
        private readonly GraphicsBuffer[] _readback = new GraphicsBuffer[FrameCount];
        private readonly Vector2[][] _asked = new Vector2[FrameCount][];
        private readonly int[] _askedCount = new int[FrameCount];
        private DirectionalLight? _sun;

        /// <summary>
        /// The camera's eye was under the water when the last frame was drawn (its near point at the middle of the picture):
        /// the weather's falling rain and snow stop for it (EnvironmentController, the precipitation layers).
        /// </summary>
        public static bool CameraUnder { get; private set; }

        public Underwater()
        {
            _shader = new ComputeShader("underwater_cs.hlsl");
            _kWater = _shader.FindKernel("CSUnderwater");
            _kProbe = _shader.FindKernel("CSProbe");
            _probe = GraphicsBuffer.CreateStructured<Vector4>(Water.SeaProbes + 1, srv: true, uav: true);
            for (int i = 0; i < FrameCount; i++)
            {
                _frames[i] = GraphicsBuffer.CreateUpload<Frame>(1, mapped: true);
                _places[i] = GraphicsBuffer.CreateUpload<Vector4>(Water.SeaProbes + 1, mapped: true);
                _readback[i] = GraphicsBuffer.CreateReadback<Vector4>(Water.SeaProbes + 1);
                _asked[i] = new Vector2[Water.SeaProbes + 1];
            }
        }

        /// <summary>
        /// The pass over <paramref name="composite"/> (RenderTarget on entry and on return; <paramref name="depth"/> in
        /// DepthWrite on entry and on return), when the camera is in or near water. False when there was nothing to do.
        /// </summary>
        public unsafe bool Execute(ID3D12GraphicsCommandList list, Camera camera, RenderTexture2D composite, DepthTexture2D depth,
            RenderTexture2D linearDepth, RenderTexture2D normals, Action<ID3D12GraphicsCommandList> bindSceneConstants,
            ResourceStates depthRead)
        {
            int slot = Engine.FrameIndex % FrameCount;

            // What came back from the probe FrameCount frames ago (this slot's last use): the swimmers' sea, the camera's
            if (_askedCount[slot] > 0)
            {
                var results = _readback[slot].ReadPtr<Vector4>();
                int n = _askedCount[slot];
                Span<Vector4> got = stackalloc Vector4[Water.SeaProbes + 1];
                for (int i = 0; i < n; i++) got[i] = results[i];
                Water.TakeMeasured(_asked[slot].AsSpan(0, n), got.Slice(0, n));
                _askedCount[slot] = 0;
            }

            var ocean = OceanRenderer.DrawnThisTick;
            var eye = camera.Position;
            var surface = default(WaterSurface);
            bool found = Engine.Settings.Underwater && Water.SurfaceAt(eye, 6f, out surface);
            // Near the water: a lake or river within a hand's breadth over the eye's height, the sea within its waves
            bool near = found && (surface.Kind == WaterKind.Sea
                ? ocean != null && surface.Ocean == ocean && eye.Y < surface.Height + 4f
                : eye.Y < surface.Height + 0.3f);
            CameraUnder = near && eye.Y < surface.Height;

            // The world's sounds as through water while the eye is under (Engine.Muffled), eased over about a tenth of a second
            float muffled = Engine.Muffled + ((CameraUnder ? 1f : 0f) - Engine.Muffled) * MathF.Min(1f, Base.Time.SmoothDelta * 10f);
            Engine.Muffled = MathF.Abs(muffled - (CameraUnder ? 1f : 0f)) < 0.01f ? (CameraUnder ? 1f : 0f) : muffled;

            // The places to measure the sea at: the camera's (when it is at sea), then the swimmers'
            Span<Vector2> watched = stackalloc Vector2[Water.SeaProbes];
            int watchedCount = Water.TakeWatched(watched);
            bool seaEye = near && surface.Kind == WaterKind.Sea;
            int probes = 0;
            if (ocean != null && ocean.SurfDataSrv != 0 && (seaEye || watchedCount > 0))
            {
                var places = _places[slot].WritePtr<Vector4>();
                var asked = _asked[slot];
                asked[probes] = new Vector2(eye.X, eye.Z);
                places[probes++] = new Vector4(eye.X, eye.Z, 0f, 0f);
                for (int i = 0; i < watchedCount && probes < Water.SeaProbes + 1; i++)
                {
                    asked[probes] = watched[i];
                    places[probes++] = new Vector4(watched[i].X, watched[i].Y, 0f, 0f);
                }
            }
            if (!near && probes == 0) return false;

            if (_sun == null || _sun.IsDestroyed) _sun = EntityManager.FindComponent<DirectionalLight>();
            var sun = _sun;
            var sunDir = sun != null ? Vector3.Normalize(Vector3.Transform(Vector3.UnitZ, sun.Entity.Transform.Rotation)) : -Vector3.UnitY;
            Vector3 shallow, colour;
            float visibility;
            if (surface.Body is { } body)
            {
                shallow = new Vector3(body.ShallowColor.R, body.ShallowColor.G, body.ShallowColor.B);
                colour = new Vector3(body.WaterColor.R, body.WaterColor.G, body.WaterColor.B);
                visibility = body.Visibility;
            }
            else if (surface.Ocean is { } sea)
            {
                shallow = new Vector3(sea.ShallowColor.R, sea.ShallowColor.G, sea.ShallowColor.B);
                colour = new Vector3(sea.OceanColor.R, sea.OceanColor.G, sea.OceanColor.B);
                visibility = sea.ShoreFadeDepth;
            }
            else
            {
                shallow = new Vector3(0.45f, 0.7f, 0.55f);
                colour = new Vector3(0.004f, 0.02f, 0.018f);
                visibility = 3.5f;
            }

            *_frames[slot].WritePtr<Frame>() = new Frame
            {
                Slope = surface.Slope,
                SurfaceY = surface.Height,
                Sea = seaEye && probes > 0 ? 1u : 0u,
                ShallowColor = shallow,
                Visibility = MathF.Max(visibility, 0.1f),
                WaterColor = colour,
                Caustics = Engine.Settings.UnderwaterCaustics,
                SunDirection = sunDir,
                SunUp = MathF.Max(0f, -sunDir.Y),
                SunColor = sun != null ? new Vector3(sun.Color.R, sun.Color.G, sun.Color.B) : Vector3.Zero,
                SunIntensity = sun?.Intensity ?? 0f,
                DepthSrv = depth.BindlessIndex,
                NormalsSrv = normals.BindlessIndex,
                ProbeCount = (uint)probes,
                OceanData = ocean?.SurfDataSrv ?? 0u,
                LinearDepthSrv = linearDepth.BindlessIndex,
                Shafts = Engine.Settings.UnderwaterShafts,
            };

            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            bindSceneConstants(list);
            _shader.SetPushConstant("Frame", _frames[slot].SrvIndex);
            _shader.SetPushConstant("Places", _places[slot].SrvIndex);
            _shader.SetPushConstant("ProbeUAV", _probe.UavIndex);
            _shader.SetPushConstant("CompositeUAV", composite.UavIndex);

            PixMarker.Begin(list, "Underwater");
            if (probes > 0)
            {
                _probe.Transition(list, ResourceStates.UnorderedAccess);
                _shader.Dispatch(_kProbe, list, 1);
                list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_probe.Native)));
            }
            if (near)
            {
                _probe.Transition(list, ResourceStates.UnorderedAccess);
                list.ResourceBarrierTransition(depth.Native, ResourceStates.DepthWrite, depthRead);
                list.ResourceBarrierTransition(composite.Native, ResourceStates.RenderTarget, ResourceStates.UnorderedAccess);
                var desc = composite.Native.Description;
                _shader.Dispatch(_kWater, list, (uint)(desc.Width + 7) / 8, (uint)(desc.Height + 7) / 8);
                list.ResourceBarrierTransition(composite.Native, ResourceStates.UnorderedAccess, ResourceStates.RenderTarget);
                list.ResourceBarrierTransition(depth.Native, depthRead, ResourceStates.DepthWrite);
            }
            if (probes > 0)
            {
                _probe.Transition(list, ResourceStates.CopySource);
                list.CopyBufferRegion(_readback[slot].Native, 0, _probe.Native, 0, (ulong)(probes * 16));
                _askedCount[slot] = probes;
            }
            PixMarker.End(list);
            return near;
        }

        public void Dispose()
        {
            foreach (var f in _frames) f?.Dispose();
            foreach (var p in _places) p?.Dispose();
            foreach (var r in _readback) r?.Dispose();
            _probe.Dispose();
            _shader.Dispose();
        }
    }
}
