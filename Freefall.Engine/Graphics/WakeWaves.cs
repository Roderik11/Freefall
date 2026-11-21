using System;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// The waves ships raise on the sea round the camera: a grid <see cref="Size"/>² over <see cref="Span"/> metres, kept
    /// centred on the camera (moved by whole cells, its waves going along), where linear deep-water waves run by their
    /// own dispersion (wake_waves_cs.hlsl: stepped exactly in Fourier space), and every hull that keeps the water out
    /// (<see cref="WaterHulls"/>) presses the water down by its depth under the still sea. A hull moving through it leaves
    /// the Kelvin wake behind by itself, bow and stern waves, curved in a turn, crossing another ship's, staying in the
    /// water when the ship turns away or stops; a hull rolling in the swell rings out a little. The cost is the grid's, not
    /// the ships': one FFT pair a step whatever the number of hulls in it. The sea reads its height in the ocean's domain
    /// shader and its slopes in the pixel shader (ocean_surf.hlsli WakeAt), the wakes' ribbons lie on it. Runs while a hull
    /// is in the grid and for <see cref="LivesFor"/> seconds after; the sea's own swell and the ships' buoyancy do not
    /// read it. Docs: .agent/knowledge/engine_internals/ships.md.
    /// </summary>
    public static class WakeWaves
    {
        /// <summary>Cells a side.</summary>
        public const int Size = 512;

        /// <summary>Metres a side: 0.5 m cells, waves 1 m long and longer, out to 128 m from the camera.</summary>
        public const float Span = 256f;

        public const float Cell = Span / Size;

        // How hard a hull presses the water (EngineSettings.ShipWavePush: the dent it would sink under itself at rest, per
        // metre of its depth) and the damping by k² (ShipWaveViscosity: at 0.02 a 1 m wave dies within a second, a 7 m one
        // lasts ~25 s): settings, so they can be tuned while watching

        /// <summary>Damping of every wave (1/s): what is left after a second is e^−this.</summary>
        public static float Damping = 0.03f;

        /// <summary>Seconds the water goes on moving after the last hull left the grid.</summary>
        public const float LivesFor = 60f;

        // A step every frame by the frame's own time (the waves are turned exactly for any step, the push is by its time):
        // stepped at a fixed 30 Hz under a 140 fps picture, the waves and the foam moved in jumps under a hull gliding
        // smoothly ("the water is unusually jumpy in the wake", user, 2026-11-21). A long frame is split into steps of
        // at most LongestStep, at most MaxStepsPerFrame of them
        private const float LongestStep = 1f / 30f;
        private const int MaxStepsPerFrame = 2;

        private static ComputeShader? _shader;
        private static int _kFft, _kEvolve, _kForce, _kFoam, _kSurface, _kClear, _kClear4;
        private static RenderTexture2D? _state, _a, _b;
        // The foam, how near a hull's waterline each cell was a step ago, and the calm a hull left in its track, in two
        // textures in turn (they are carried by the water: read round)
        private static readonly RenderTexture2D?[] _foam = new RenderTexture2D?[2];
        private static int _foamCur;
        // Two surfaces in turn: the ocean's data for a frame is filled before that frame's step, so it reads the surface
        // and the origin of the step before as a pair; one surface written over with a moved grid under the old origin
        // put every wave half a metre off for a frame whenever the grid moved a cell. The calm the ocean reads is
        // published with them, in a pair of its own
        private static readonly RenderTexture2D?[] _surfaces = new RenderTexture2D?[2];
        private static readonly RenderTexture2D?[] _calm = new RenderTexture2D?[2];
        private static int _surfaceNext;
        private static readonly ResourceStates[] _states = new ResourceStates[9];
        private static Vector2 _origin, _published;
        private static bool _cleared;
        private static float _alive;
        private static int _steppedTick = -1;

        /// <summary>The wake's surface (Texture2D&lt;float4&gt;: height m, ∂h/∂x, ∂h/∂z, foam 0..1.5), 0 while the water is still.</summary>
        public static uint Srv { get; private set; }

        /// <summary>World xz of <see cref="Srv"/>'s corner (it spans <see cref="Span"/>).</summary>
        public static Vector2 Origin => _published;

        /// <summary>
        /// The calm the hulls left in their track over the same grid (Texture2D&lt;float&gt; 0..1, published with
        /// <see cref="Srv"/>), 0 while the water is still. The ocean turns its short waves down by it (ocean_surf.hlsli
        /// CalmShare): a hull going through the sea leaves a smoother band behind it.
        /// </summary>
        public static uint CalmSrv { get; private set; }

        /// <summary>
        /// The frame's steps, before the water is drawn (DeferredRenderer, after the ripples): the grid follows the camera by
        /// whole cells and takes as many steps as the time owes (at most <see cref="MaxStepsPerFrame"/>).
        /// </summary>
        internal static void Step(ID3D12GraphicsCommandList list, Vector3 camera)
        {
            if (_steppedTick == Engine.TickCount) return;
            _steppedTick = Engine.TickCount;
            float dt = MathF.Min(Base.Time.SmoothDelta, 0.1f);

            OceanRenderer? ocean = null;
            var oceans = ComponentCache<OceanRenderer>.All;
            for (int i = 0; i < oceans.Count; i++)
                if (oceans[i] is { Enabled: true } o) { ocean = o; break; }
            uint hulls = WaterHulls.Srv, hullCount = WaterHulls.Count, hullSea = WaterHulls.SeaSrv;
            if (ocean == null || Engine.Settings.DisableShipWaves || Engine.Device == null)
            {
                Sleep();
                return;
            }

            // Awake while a hull is in the grid, and for a while after
            if (hullCount > 0 && WaterHulls.AnyWithin(camera, Span * 0.5f)) _alive = LivesFor;
            else _alive -= dt;
            if (_alive <= 0f)
            {
                Sleep();
                return;
            }

            float frame = MathF.Min(dt, MaxStepsPerFrame * LongestStep);
            int owedSteps = frame > 0f ? Math.Min((int)MathF.Ceiling(frame / LongestStep), MaxStepsPerFrame) : 0;
            float stepSeconds = owedSteps > 0 ? frame / owedSteps : 0f;
            var origin = new Vector2(MathF.Floor(camera.X / Cell) * Cell - Span * 0.5f, MathF.Floor(camera.Z / Cell) * Cell - Span * 0.5f);
            if (owedSteps == 0 && _cleared && origin == _origin) return;

            EnsureResources();
            var device = Engine.Device;
            list.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            list.SetComputeRootSignature(device.GlobalRootSignature);
            PixMarker.Begin(list, "Wake waves");
            var s = _shader!;

            if (!_cleared)
            {
                _origin = origin;
                foreach (int i in new[] { 0, 5, 6 })
                {
                    int kernel = i == 0 ? _kClear : _kClear4;   // the foam textures have four channels
                    ToState(list, i, ResourceStates.UnorderedAccess);
                    s.SetPushConstant(kernel, "OutUAV", Texture(i).UavIndex);
                    s.Dispatch(kernel, list, Size / 8, Size / 8);
                }
                Uav(list);
                _cleared = true;
            }
            int shiftX = (int)MathF.Round((origin.X - _origin.X) / Cell);
            int shiftZ = (int)MathF.Round((origin.Y - _origin.Y) / Cell);
            _origin = origin;

            float seaLevel = ocean.Transform.WorldPosition.Y;
            int steps = 0;
            while (steps < owedSteps || (steps == 0 && (shiftX != 0 || shiftZ != 0)))
            {
                bool stepping = steps < owedSteps;
                float stepDt = stepping ? stepSeconds : 0f;

                // state → A → B: the spectrum; B → A: stepped; A → B → A: back in the grid
                Fft(list, 0, 1, false);
                Fft(list, 1, 2, false);
                Run(list, _kEvolve, 2, 1, k =>
                {
                    s.SetPushConstant(k, "DtBits", BitConverter.SingleToUInt32Bits(stepDt));
                    s.SetPushConstant(k, "DampBits", BitConverter.SingleToUInt32Bits(Damping));
                    s.SetPushConstant(k, "ViscBits", BitConverter.SingleToUInt32Bits(Engine.Settings.ShipWaveViscosity));
                });
                Fft(list, 1, 2, true);
                Fft(list, 2, 1, true);
                int sx = steps == 0 ? shiftX : 0, sz = steps == 0 ? shiftZ : 0;
                Run(list, _kForce, 1, 0, k =>
                {
                    s.SetPushConstant(k, "ShiftX", unchecked((uint)sx));
                    s.SetPushConstant(k, "ShiftZ", unchecked((uint)sz));
                    s.SetPushConstant(k, "OriginXBits", BitConverter.SingleToUInt32Bits(origin.X));
                    s.SetPushConstant(k, "OriginZBits", BitConverter.SingleToUInt32Bits(origin.Y));
                    s.SetPushConstant(k, "HullsSrv", hulls);
                    s.SetPushConstant(k, "HullCount", hullCount);
                    s.SetPushConstant(k, "SeaLevelBits", BitConverter.SingleToUInt32Bits(seaLevel));
                    s.SetPushConstant(k, "PushBits", BitConverter.SingleToUInt32Bits(9.81f * Engine.Settings.ShipWavePush * stepDt));
                    s.SetPushConstant(k, "SeaSrv", hullSea);
                });
                // The white water: carried by this step's water, made where it breaks and where the hulls churn it; and
                // the calm the hulls leave in their track, which turns the sea's short waves down and makes the white
                // water there go within a second
                int foamFrom = 5 + _foamCur, foamTo = 5 + (_foamCur ^ 1);
                ToState(list, foamFrom, Read);
                Run(list, _kFoam, 0, foamTo, k =>
                {
                    s.SetPushConstant(k, "FoamSrv", Texture(foamFrom).BindlessIndex);
                    s.SetPushConstant(k, "DtBits", BitConverter.SingleToUInt32Bits(stepDt));
                    s.SetPushConstant(k, "ShiftX", unchecked((uint)sx));
                    s.SetPushConstant(k, "ShiftZ", unchecked((uint)sz));
                    s.SetPushConstant(k, "OriginXBits", BitConverter.SingleToUInt32Bits(origin.X));
                    s.SetPushConstant(k, "OriginZBits", BitConverter.SingleToUInt32Bits(origin.Y));
                    s.SetPushConstant(k, "HullsSrv", hulls);
                    s.SetPushConstant(k, "HullCount", hullCount);
                    s.SetPushConstant(k, "SeaLevelBits", BitConverter.SingleToUInt32Bits(seaLevel));
                    s.SetPushConstant(k, "NoCalm", Engine.Settings.DisableWakeCalm ? 1u : 0u);
                    s.SetPushConstant(k, "SeaSrv", hullSea);
                });
                _foamCur ^= 1;
                steps++;
                if (!stepping) break;
            }

            // What the water reads, from the next frame's draws on, with the origin it was written for
            int surface = 3 + _surfaceNext;
            int calm = 7 + _surfaceNext;
            _surfaceNext ^= 1;
            ToState(list, 0, Read);
            ToState(list, 5 + _foamCur, Read);
            ToState(list, surface, ResourceStates.UnorderedAccess);
            ToState(list, calm, ResourceStates.UnorderedAccess);
            s.SetPushConstant(_kSurface, "InSrv", _state!.BindlessIndex);
            s.SetPushConstant(_kSurface, "FoamSrv", Texture(5 + _foamCur).BindlessIndex);
            s.SetPushConstant(_kSurface, "OutUAV", Texture(surface).UavIndex);
            s.SetPushConstant(_kSurface, "CalmUAV", Texture(calm).UavIndex);
            s.Dispatch(_kSurface, list, Size / 8, Size / 8);
            ToState(list, surface, Read);
            ToState(list, calm, Read);
            Srv = Texture(surface).BindlessIndex;
            CalmSrv = Texture(calm).BindlessIndex;
            _published = origin;
            PixMarker.End(list);
        }

        private static void Sleep()
        {
            Srv = 0;
            CalmSrv = 0;
            _cleared = false;
            _alive = 0f;
        }

        // One FFT pass over the rows of texture `from` into `to`, transposed
        private static void Fft(ID3D12GraphicsCommandList list, int from, int to, bool inverse)
        {
            var s = _shader!;
            ToState(list, from, Read);
            ToState(list, to, ResourceStates.UnorderedAccess);
            s.SetPushConstant(_kFft, "InSrv", Texture(from).BindlessIndex);
            s.SetPushConstant(_kFft, "OutUAV", Texture(to).UavIndex);
            s.SetPushConstant(_kFft, "Inverse", inverse ? 1u : 0u);
            s.Dispatch(_kFft, list, Size, 1);
        }

        private static void Run(ID3D12GraphicsCommandList list, int kernel, int from, int to, Action<int> set)
        {
            var s = _shader!;
            ToState(list, from, Read);
            ToState(list, to, ResourceStates.UnorderedAccess);
            s.SetPushConstant(kernel, "InSrv", Texture(from).BindlessIndex);
            s.SetPushConstant(kernel, "OutUAV", Texture(to).UavIndex);
            set(kernel);
            s.Dispatch(kernel, list, Size / 8, Size / 8);
        }

        private static RenderTexture2D Texture(int i) => i switch
        {
            0 => _state!, 1 => _a!, 2 => _b!, 3 or 4 => _surfaces[i - 3]!, 5 or 6 => _foam[i - 5]!, _ => _calm[i - 7]!,
        };

        private static void EnsureResources()
        {
            if (_state != null) return;
            _shader = new ComputeShader("wake_waves_cs.hlsl");
            _kFft = _shader.FindKernel("CSFft");
            _kEvolve = _shader.FindKernel("CSEvolve");
            _kForce = _shader.FindKernel("CSForce");
            _kFoam = _shader.FindKernel("CSFoam");
            _kSurface = _shader.FindKernel("CSSurface");
            _kClear = _shader.FindKernel("CSClear");
            _kClear4 = _shader.FindKernel("CSClear4");
            var device = Engine.Device;
            _state = new RenderTexture2D(device, Size, Size, Format.R32G32_Float, randomWrite: true);   // h, φ
            _a = new RenderTexture2D(device, Size, Size, Format.R32G32_Float, randomWrite: true);
            _b = new RenderTexture2D(device, Size, Size, Format.R32G32_Float, randomWrite: true);
            for (int i = 0; i < _surfaces.Length; i++)
                _surfaces[i] = new RenderTexture2D(device, Size, Size, Format.R16G16B16A16_Float, randomWrite: true);
            for (int i = 0; i < _foam.Length; i++)   // foam, nearness a step ago, calm, unused
                _foam[i] = new RenderTexture2D(device, Size, Size, Format.R16G16B16A16_Float, randomWrite: true);
            for (int i = 0; i < _calm.Length; i++)
                _calm[i] = new RenderTexture2D(device, Size, Size, Format.R16_Float, randomWrite: true);
            for (int i = 0; i < _states.Length; i++) _states[i] = ResourceStates.Common;
        }

        // Read by the steps (compute) and by the water (domain and pixel shaders)
        private const ResourceStates Read = ResourceStates.PixelShaderResource | ResourceStates.NonPixelShaderResource;

        private static void ToState(ID3D12GraphicsCommandList list, int i, ResourceStates state)
        {
            if (_states[i] == state)
            {
                if (state == ResourceStates.UnorderedAccess) Uav(list);
                return;
            }
            list.ResourceBarrierTransition(Texture(i).Native, _states[i], state);
            _states[i] = state;
        }

        private static void Uav(ID3D12GraphicsCommandList list) =>
            list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
    }
}
