using System;
using System.Numerics;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// The camera's exposure (EV100), which <see cref="Atmosphere.Exposure"/> turns into the pre-exposure of every light
    /// the frame draws. With <c>Engine.Settings.AutoExposure</c> on it follows the light where the camera is, like an
    /// eye, whichever way it looks: each frame exposure_meter_cs.hlsl meters, at the camera, the light there (a sphere's:
    /// the sun and the moon past the terrain, the meshes and the clouds, the sky the place sees, the lamps near it) and the
    /// brightness round it (the sky dome where the camera sees it, the crowns and walls round it, the ground below), and
    /// exposure_histogram_cs.hlsl bins the HDR composite's luminance (cd/m², the composite divided by the exposure it was
    /// drawn with). A few frames later the CPU reads both back: the meter's two readings half and half, and the screen as
    /// the mean of log luminance between two percentiles (so a few very dark or very bright pixels do not decide), each
    /// as EV100 = log2(L) + 3 (ISO 100, K = 12.5); the screen moves the meter's EV by at most ExposureScreenRange (softly), so
    /// turning round in the same shade no longer swings the exposure by what the frame holds. Then it keeps a share of the
    /// darkness of dim light (ExposureDimLightDarkening below ExposureDimLightEV), applies the compensations, clamps it to
    /// [ExposureMinEV, ExposureMaxEV] and moves the exposure toward it, fast toward darker images (a bright scene), slowly
    /// toward brighter ones (the dark). Off: the manual <c>ExposureEV</c> (the captures lock it this way).
    /// Docs: .agent/knowledge/atmospheric_effects/physical_atmosphere.md §7, §7.1.
    /// </summary>
    public static class AutoExposure
    {
        private const int Bins = 64;
        private const float MinLog2 = -14f, MaxLog2 = 18f;      // must match exposure_histogram_cs.hlsl
        private const float LowPercentile = 0.5f, HighPercentile = 0.95f;
        private const int FrameCount = 3;

        // The meter's two readings as the screen's metering would read them (EV100 = log2(L) + 3), calibrated 2026-10-18 on
        // Eldermoor's open views (coast, town, lake, plaza; the screen's metering is the mean between the 50th and 95th
        // percentiles, bright-biased): the brightness round the camera sat 0.7 EV under it at noon, 15:00, 17:00 and 18:00
        // (0.5 at 18:24); the light at the camera, as a grey card's luminance (× this / π), 1.5 EV under it in full sun
        // (noon to 17:00) and over it once the sun is down (+1 at 18:00). The exposure takes them half and half.
        private const float FieldCalibration = 1.62f;
        private const float SphereCard = 0.35f;
        private const float FieldShare = 0.5f;
        private const int MeterRows = 4;                         // exposure_meter_cs.hlsl METER_ROWS

        /// <summary>The exposure (EV100) of the frame being drawn.</summary>
        public static float EV => Engine.Settings.AutoExposure ? _current : Engine.Settings.ExposureEV;

        /// <summary>Where the exposure is heading: the last metering, compensated and clamped.</summary>
        public static float TargetEV { get; private set; } = Atmosphere.ReferenceEV;

        /// <summary>The scene's metered luminance (cd/m², the mean of log luminance between the percentiles).</summary>
        public static float MeteredLuminance { get; private set; }

        /// <summary>The meter at the camera: the brightness round it (cd/m²); 0 until it has been read.</summary>
        public static float MeterLuminance { get; private set; }

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

        private static ComputeShader? _meterCs;
        private static int _kMeter;
        private static GraphicsBuffer? _meter;
        private static readonly GraphicsBuffer?[] _meterReadback = new GraphicsBuffer?[FrameCount];
        private static readonly bool[] _meterWritten = new bool[FrameCount];
        private static readonly Vector4[] _meterRows = new Vector4[MeterRows];
        private static float _screenEV, _incidentEV;

        private static void EnsureResources()
        {
            if (_cs != null) return;
            _cs = new ComputeShader("exposure_histogram_cs.hlsl");
            _kClear = _cs.FindKernel("CSClear");
            _kHistogram = _cs.FindKernel("CSHistogram");
            _histogram = GraphicsBuffer.CreateStructured<uint>(Bins, srv: false, uav: true);
            for (int i = 0; i < FrameCount; i++)
                _readback[i] = GraphicsBuffer.CreateReadback<uint>(Bins);

            _meterCs = new ComputeShader("exposure_meter_cs.hlsl");
            _kMeter = _meterCs.FindKernel("CSMeter");
            _meter = GraphicsBuffer.CreateStructured<Vector4>(MeterRows, srv: false, uav: true);
            for (int i = 0; i < FrameCount; i++)
                _meterReadback[i] = GraphicsBuffer.CreateReadback<Vector4>(MeterRows);
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

            // EV100 = log2(L · S / K) with S = 100, K = 12.5
            float metered = (float)meteredLog2 + 3f;
            _screenEV = metered;

            // The meter at the camera decides; the screen moves it by up to ExposureScreenRange EV, softly
            // (tanh: one for one near the meter, never past the range). 8 or more: the screen alone.
            if (_meterWritten[slot])
            {
                Vector4* rows = _meterReadback[slot]!.ReadPtr<Vector4>();
                for (int i = 0; i < MeterRows; i++)
                    _meterRows[i] = rows[i];
                MeterLuminance = _meterRows[0].X;
                float fieldEV = MathF.Log2(MathF.Max(MeterLuminance, 1e-9f) * FieldCalibration) + 3f;
                float sphereEV = MathF.Log2(MathF.Max(_meterRows[0].Y, 1e-9f) * SphereCard / MathF.PI) + 3f;
                _incidentEV = sphereEV + (fieldEV - sphereEV) * FieldShare;
                float range = Math.Clamp(settings.ExposureScreenRange, 0f, 8f);
                if (range < 8f)
                    metered = range > 0f ? _incidentEV + range * MathF.Tanh((metered - _incidentEV) / range) : _incidentEV;
            }

            // In dim light the eye adapts only partly: a share of what the scene loses below ExposureDimLightEV stays in
            // the image. Compensation brightens (lowers the EV).
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

            // Lightning: an eye does not adapt to a flash of a tenth of a second, and the readback that meters it arrives
            // a few frames late: hold still until its light is out of the metering (it chased each flash by ~1 EV)
            if (Clouds.FlashInMetering)
                return;

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

            list.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            list.SetComputeRootSignature(device.GlobalRootSignature);

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

        /// <summary>
        /// The incident meter at the camera (exposure_meter_cs.hlsl), after the sky captures and the lamps of this frame;
        /// the scene constants (camera, AtmosphereIdx, Exposure) must be bound for compute by
        /// <paramref name="bindSceneConstants"/>.
        /// </summary>
        public static void MeterIncident(ID3D12GraphicsCommandList list, Vector3 cameraPos,
            Action<ID3D12GraphicsCommandList> bindSceneConstants)
        {
            if (!Engine.Settings.AutoExposure) return;
            EnsureResources();
            var cs = _meterCs!;
            var device = Engine.Device;

            list.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            list.SetComputeRootSignature(device.GlobalRootSignature);
            bindSceneConstants(list);

            PixMarker.Begin(list, "Exposure meter");
            var lights = PointLightBuffer.Instance;
            bool points = lights != null && lights.SlotCount > 0;
            cs.SetPushConstant("OutputUAV", _meter!.UavIndex);
            cs.SetPushConstant("Records", points ? lights!.SrvIndex : 0u);
            cs.SetPushConstant("RecordCount", points ? (uint)lights!.SlotCount : 0u);
            cs.SetPushConstant("Transforms", TransformBuffer.Instance.SrvIndex);
            cs.SetPushConstant("LampsUAV", points ? lights!.LampStatesUav(list) : 0u);
            cs.SetParam("MeterPosition", cameraPos);
            cs.SetParam("OpenLamps", Daylight.Lamps);

            _meter.Transition(list, ResourceStates.UnorderedAccess);
            cs.Dispatch(_kMeter, list, 1);
            _meter.Transition(list, ResourceStates.CopySource);
            int slot = Engine.FrameIndex % FrameCount;
            list.CopyBufferRegion(_meterReadback[slot]!.Native, 0, _meter.Native, 0, (ulong)(MeterRows * 16));
            _meterWritten[slot] = true;
            PixMarker.End(list);
        }

        /// <summary>For engine_stats: where the exposure stands, and what the incident meter read.</summary>
        public static object Describe() => new
        {
            auto = Engine.Settings.AutoExposure,
            ev = EV,
            targetEV = TargetEV,
            meteredLuminance = MeteredLuminance,
            screenEV = _screenEV,
            incidentEV = _incidentEV,
            meterLuminance = MeterLuminance,
            sphereLux = _meterRows[0].Y,
            skyUpperLuminance = _meterRows[3].W,
            sunLux = _meterRows[1].X,
            sunVisible = _meterRows[1].Y,
            sunTerrain = _meterRows[3].X,
            sunMeshes = _meterRows[3].Y,
            sunClouds = _meterRows[3].Z,
            tapsCounted = _meterRows[0].W,
            moonLux = _meterRows[1].Z,
            moonVisible = _meterRows[1].W,
            skyLux = _meterRows[2].X,
            skyVisible = _meterRows[2].Y,
            groundBounceLux = _meterRows[2].Z,
            pointLux = _meterRows[0].Z,
            presetCompensation = PresetCompensation,
            groundIlluminanceLux = Daylight.GroundIlluminance(),
            lamps = Daylight.Lamps,
        };
    }
}
