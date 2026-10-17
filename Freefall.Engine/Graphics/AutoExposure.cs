using System;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// The camera's exposure (EV100), which <see cref="Atmosphere.Exposure"/> turns into the pre-exposure of every light
    /// the frame draws. With <c>Engine.Settings.AutoExposure</c> on it follows the scene like an eye: each frame
    /// exposure_histogram_cs.hlsl bins the HDR composite's luminance (cd/m², the composite divided by the exposure it was
    /// drawn with); a few frames later the CPU reads the histogram back, takes the mean of log luminance between two
    /// percentiles (so a few very dark or very bright pixels do not decide), meters it as EV100 = log2(L) + 3 (ISO 100,
    /// K = 12.5), keeps a share of the darkness of dim light (ExposureDimLightDarkening below ExposureDimLightEV),
    /// applies the compensations, clamps it to [ExposureMinEV, ExposureMaxEV] and moves the exposure toward it,
    /// fast toward darker images (a bright scene), slowly toward brighter ones (the dark). Off: the manual
    /// <c>ExposureEV</c> (the captures lock it this way). Docs: .agent/knowledge/atmospheric_effects/physical_atmosphere.md.
    /// </summary>
    public static class AutoExposure
    {
        private const int Bins = 64;
        private const float MinLog2 = -14f, MaxLog2 = 18f;      // must match exposure_histogram_cs.hlsl
        private const float LowPercentile = 0.5f, HighPercentile = 0.95f;
        private const int FrameCount = 3;

        /// <summary>The exposure (EV100) of the frame being drawn.</summary>
        public static float EV => Engine.Settings.AutoExposure ? _current : Engine.Settings.ExposureEV;

        /// <summary>Where the exposure is heading: the last metering, compensated and clamped.</summary>
        public static float TargetEV { get; private set; } = Atmosphere.ReferenceEV;

        /// <summary>The scene's metered luminance (cd/m², the mean of log luminance between the percentiles).</summary>
        public static float MeteredLuminance { get; private set; }

        /// <summary>The environment preset's exposure compensation (EV, + = brighter), set by EnvironmentController.</summary>
        public static float PresetCompensation { get; set; }

        private static float _current = Atmosphere.ReferenceEV;
        private static float _snapSeconds = SnapSeconds;     // > 0: follow the metering without a fade
        private static bool _wasAuto;
        private const float SnapSeconds = 0.5f;              // a new scene's first frames stream in: follow them, then adapt
        private static int _updatedTick = -1;

        private static ComputeShader? _cs;
        private static int _kClear, _kHistogram;
        private static GraphicsBuffer? _histogram;
        private static readonly GraphicsBuffer?[] _readback = new GraphicsBuffer?[FrameCount];
        private static readonly bool[] _written = new bool[FrameCount];

        private static void EnsureResources()
        {
            if (_cs != null) return;
            _cs = new ComputeShader("exposure_histogram_cs.hlsl");
            _kClear = _cs.FindKernel("CSClear");
            _kHistogram = _cs.FindKernel("CSHistogram");
            _histogram = GraphicsBuffer.CreateStructured<uint>(Bins, srv: false, uav: true);
            for (int i = 0; i < FrameCount; i++)
                _readback[i] = GraphicsBuffer.CreateReadback<uint>(Bins);
        }

        /// <summary>
        /// Once per tick, before the frame's scene constants are set: read the histogram of the frame that last used this
        /// slot and move the exposure toward its metering.
        /// </summary>
        public static unsafe void Update()
        {
            if (_updatedTick == Engine.TickCount) return;
            _updatedTick = Engine.TickCount;
            EnsureResources();

            var settings = Engine.Settings;
            if (!settings.AutoExposure)
            {
                _wasAuto = false;
                return;
            }
            if (!_wasAuto)
                _snapSeconds = SnapSeconds; // turned on (or first frame): start where the scene is, without a fade
            _wasAuto = true;
            float dt = Math.Clamp((float)Base.Time.Delta, 0f, 0.1f);

            int slot = Engine.FrameIndex % FrameCount;
            if (!_written[slot]) return;

            uint* bins = _readback[slot]!.ReadPtr<uint>();
            double total = 0;
            for (int i = 0; i < Bins; i++) total += bins[i];
            if (total <= 0) return;

            // Mean of the bins' log luminance between the two percentiles (partial bins by their share)
            double low = total * LowPercentile, high = total * HighPercentile;
            double below = 0, sum = 0, weight = 0;
            for (int i = 0; i < Bins; i++)
            {
                double count = bins[i];
                double from = Math.Max(below, low), to = Math.Min(below + count, high);
                if (to > from)
                {
                    double log2 = MinLog2 + (i + 0.5) * (MaxLog2 - MinLog2) / Bins;
                    sum += log2 * (to - from);
                    weight += to - from;
                }
                below += count;
            }
            if (weight <= 0) return;
            double meteredLog2 = sum / weight;
            MeteredLuminance = (float)Math.Pow(2.0, meteredLog2);

            // EV100 = log2(L · S / K) with S = 100, K = 12.5. In dim light the eye adapts only partly: a share of what the
            // scene loses below ExposureDimLightEV stays in the image. Compensation brightens (lowers the EV).
            float metered = (float)meteredLog2 + 3f;
            if (metered < settings.ExposureDimLightEV)
                metered += settings.ExposureDimLightDarkening * (settings.ExposureDimLightEV - metered);
            float target = metered - settings.ExposureCompensation - PresetCompensation;
            TargetEV = Math.Clamp(target, settings.ExposureMinEV, settings.ExposureMaxEV);

            if (_snapSeconds > 0f)
            {
                _current = TargetEV;
                _snapSeconds -= MathF.Max(dt, 1f / 60f);
                return;
            }

            // Toward a darker image (higher EV: the scene got brighter) the eye adapts fast, toward the dark slowly
            float delta = TargetEV - _current;
            float rate = delta > 0f ? settings.ExposureSpeedUp : settings.ExposureSpeedDown;
            _current += delta * (1f - MathF.Exp(-dt * rate));
        }

        /// <summary>
        /// Follow the metering without adapting for the next half second (a scene load, a teleport): the first frames of
        /// a new view are often still streaming in (EntityManager.ClearScene calls it).
        /// </summary>
        public static void Snap() => _snapSeconds = SnapSeconds;

        /// <summary>
        /// Bin the composite's luminance (it must be in a shader-readable state) and copy the histogram for the CPU.
        /// <paramref name="exposureUsed"/> is the exposure the composite was drawn with.
        /// </summary>
        public static void Measure(ID3D12GraphicsCommandList list, uint compositeSrv, int width, int height, float exposureUsed)
        {
            if (!Engine.Settings.AutoExposure) return;
            EnsureResources();
            var cs = _cs!;
            var device = Engine.Device;

            list.SetComputeRootSignature(device.GlobalRootSignature);
            list.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            PixMarker.Begin(list, "Exposure histogram");
            _histogram!.Transition(list, ResourceStates.UnorderedAccess);
            cs.SetPushConstant(_kClear, "HistogramUAV", _histogram.UavIndex);
            cs.Dispatch(_kClear, list, 1);
            _histogram.UAVBarrier(list);

            cs.SetPushConstant(_kHistogram, "InputSrv", compositeSrv);
            cs.SetPushConstant(_kHistogram, "HistogramUAV", _histogram.UavIndex);
            cs.SetPushConstant(_kHistogram, "Width", (uint)width);
            cs.SetPushConstant(_kHistogram, "Height", (uint)height);
            cs.SetPushConstant(_kHistogram, "InvExposureBits", BitConverter.SingleToUInt32Bits(1f / MathF.Max(exposureUsed, 1e-12f)));
            cs.Dispatch(_kHistogram, list, (uint)((width + 31) / 32), (uint)((height + 31) / 32));

            _histogram.Transition(list, ResourceStates.CopySource);
            int slot = Engine.FrameIndex % FrameCount;
            list.CopyBufferRegion(_readback[slot]!.Native, 0, _histogram.Native, 0, Bins * sizeof(uint));
            _written[slot] = true;
            PixMarker.End(list);
        }

        /// <summary>For engine_stats: where the exposure stands.</summary>
        public static object Describe() => new
        {
            auto = Engine.Settings.AutoExposure,
            ev = EV,
            targetEV = TargetEV,
            meteredLuminance = MeteredLuminance,
            presetCompensation = PresetCompensation,
            groundIlluminanceLux = Daylight.GroundIlluminance(),
            lamps = Daylight.Lamps,
        };
    }
}
