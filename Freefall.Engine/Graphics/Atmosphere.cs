using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>What the atmosphere is made of and what lights it. <see cref="SkyboxRenderer"/> sets it every frame (from
    /// its fields, which an EnvironmentPreset drives); without a sky the default stands: Earth's air under a sun
    /// straight overhead.</summary>
    public struct AtmosphereSettings
    {
        public Vector3 SunDirection;        // toward the sun
        public Vector3 SunIlluminance;      // lux at the top of the atmosphere, colour × lux
        public float SunAngularRadius;      // radians
        public Vector3 MoonDirection;       // toward the brightest moon
        public Vector3 MoonIlluminance;     // lux at the top of the atmosphere, its phase included
        public float AirglowLuminance;      // cd/m²: the faint glow of the upper air on a dark night

        public float RayleighScale;         // 1 = Earth's air
        public float MieScale;              // 1 = Earth's average aerosols
        public float MieAnisotropy;         // g of the aerosols and the fog
        public float OzoneScale;
        public Vector3 GroundAlbedo;        // average albedo of the ground: light it bounces into the sky and onto surfaces
        public float FogDensity;            // ground fog: extinction per metre at sea level (0.0005 = 6 km visibility)
        public float FogHeight;             // metres in which the fog thins to 1/e
        public Vector3 FogColor;            // the fog's single-scattering albedo

        public static AtmosphereSettings Default => new()
        {
            SunDirection = Vector3.UnitY,
            SunIlluminance = new Vector3(120000f),
            SunAngularRadius = 0.2665f * (MathF.PI / 180f),
            MoonDirection = -Vector3.UnitY,
            MoonIlluminance = Vector3.Zero,
            AirglowLuminance = 2e-4f,
            RayleighScale = 1f,
            MieScale = 1f,
            MieAnisotropy = 0.8f,
            OzoneScale = 1f,
            GroundAlbedo = new Vector3(0.11f, 0.10f, 0.085f),
            FogDensity = 0f,
            FogHeight = 300f,
            FogColor = new Vector3(0.9f),
        };
    }

    /// <summary>
    /// The physically based atmosphere (Hillaire 2020): the one source of the sky, the sun's and the moons' light, the
    /// sky's ambient light, the ground's bounce light and the aerial perspective (fog). Its LUTs are made every frame by
    /// atmosphere_cs.hlsl (<see cref="Render"/>, once per view, before the sky is drawn); shaders find them through
    /// SceneConstants.AtmosphereIdx (atmosphere_common.hlsli). Light is in physical units (lux, cd/m²); the camera's
    /// <see cref="Exposure"/> scales it into what the pixels store.
    ///
    /// The directional light's intensity is the sun's or moon's lux in the engine's older units (× ReferenceExposure / π,
    /// SkyboxRenderer); <see cref="SunLightScale"/> pre-exposes it. Point lights are in lumens and take
    /// <see cref="Exposure"/>; light sources seen directly (emissive materials, flames, embers, VFX glow, all in nits)
    /// take <see cref="EmissiveExposure"/>. Docs: .agent/knowledge/atmospheric_effects/physical_atmosphere.md,
    /// ATMOSPHERE_PLAN.md.
    /// </summary>
    public static class Atmosphere
    {
        /// <summary>The exposure (EV100) at which light in the older units shows as before: a noon sun through Earth's
        /// air at this exposure gives what the directional light gave at intensity 2.58.</summary>
        public const float ReferenceEV = 13.36f;

        public static float ReferenceExposure => 1f / (1.2f * MathF.Pow(2f, ReferenceEV));
        public static float Exposure => 1f / (1.2f * MathF.Pow(2f, AutoExposure.EV));
        /// <summary>
        /// The terrain the sky visibility reads this frame (sky_visibility.hlsli TerrainCover): heightmap bindless index (0 =
        /// none, or Engine.Settings.SkyVisibility off), metres at 1, the terrain's corner, the metres it spans.
        /// </summary>
        public static (uint Heightmap, float MaxHeight, Vector3 Origin, Vector2 Size) TerrainCover =>
            (_frame.TerrainHeightmap, _frame.TerrainMaxHeight, _frame.TerrainOrigin, new Vector2(_frame.TerrainSizeX, _frame.TerrainSizeZ));

        /// <summary>The directional light (physical lux in the older units' encoding) × this = pre-exposed.</summary>
        public static float SunLightScale => Exposure / ReferenceExposure;

        /// <summary>
        /// The pre-exposure of light sources seen directly: their luminance in nits × this is what the pixels store.
        /// Physically it would be <see cref="Exposure"/>, but a flame (~40000 nits) is 10⁵ times the moonlit street and
        /// would blow out white at night, as would lantern glass and lit windows. The eye adapts to what it looks at, so
        /// below <see cref="ReferenceEV"/> (a sunny noon) this follows the exposure only by 1 − EmissiveAdaptation
        /// (Engine.Settings, 0.8): Exposure^(1−w) × ReferenceExposure^w. In daylight it is the exposure itself.
        /// </summary>
        public static float EmissiveExposure
        {
            get
            {
                float exposure = Exposure, reference = ReferenceExposure;
                if (exposure <= reference) return exposure;
                float w = Math.Clamp(Engine.Settings.EmissiveAdaptation, 0f, 1f);
                return MathF.Pow(exposure, 1f - w) * MathF.Pow(reference, w);
            }
        }

        public const float BottomRadiusKm = 6360f;
        public const float TopRadiusKm = 6460f;
        public const float APMaxDepth = 32000f;     // metres of view depth the aerial perspective volume covers

        // Earth (Hillaire 2020, per km)
        private static readonly Vector3 RayleighEarth = new(0.005802f, 0.013558f, 0.033100f);
        private const float RayleighHeightKm = 8f;
        private const float MieScatteringEarth = 0.003996f;
        private const float MieAbsorptionEarth = 0.000444f;
        private const float MieHeightKm = 1.2f;
        private static readonly Vector3 OzoneEarth = new(0.000650f, 0.001881f, 0.000085f);

        /// <summary>The atmosphere this frame. Set by SkyboxRenderer; <see cref="AtmosphereSettings.Default"/> without one.</summary>
        public static AtmosphereSettings Settings = AtmosphereSettings.Default;

        /// <summary>The sky's light on an up-facing surface as a multiple of its albedo, in the older light units (as
        /// <c>Daylight</c> measures), read back from the GPU a few frames late.</summary>
        public static Vector3 SkyAmbientLegacy { get; private set; }

        /// <summary>The clouds round the camera as the ground and the air below them take them (atmosphere_cs.hlsl
        /// CSCloudDeck, sky_common.fx CloudDeck), read back a few frames late, toward the light the clouds are lit by (the
        /// sun, at night the brightest moon): X the mean beam over 16 km (e^−τ: the light as a direction), Y the mean of
        /// all they let through (the beam and what they diffuse), Z the beam at the camera, W 1 when the light is the
        /// moon. (1, 1, 1, 0) without clouds.</summary>
        public static Vector4 CloudDeck { get; private set; } = new(1f, 1f, 1f, 0f);

        /// <summary>The share of the daylight the clouds round the camera let through to the ground: <see cref="CloudDeck"/>'s Y.</summary>
        public static float CloudThrough => Math.Clamp(CloudDeck.Y, 0f, 1f);

        /// <summary>For engine_stats: <see cref="CloudDeck"/>.</summary>
        public static object DescribeCloudDeck() => new
        {
            meanBeam = CloudDeck.X,
            meanThrough = CloudDeck.Y,
            beamAtCamera = CloudDeck.Z,
            light = CloudDeck.W > 0.5f ? "moon" : "sun",
        };

        /// <summary>Largest relative difference (0..1) between the CPU's transmittance toward the sun (which sets the
        /// directional light) and the transmittance LUT's, for the same frame: the two copies of the medium must agree.
        /// Logged as a warning when it goes over 3% (the LUT's own interpolation error reaches ~1.6% with a low sun).</summary>
        public static float TransmittanceMismatch { get; private set; }
        private static bool _mismatchReported;
        private static readonly Vector3[] _cpuSunTransmittance = new Vector3[3];

        // ── CPU side: transmittance toward a light, for the directional light ──

        private static Vector3 Extinction(in AtmosphereSettings s, float heightKm)
        {
            heightKm = MathF.Max(heightKm, 0f);
            float ozone = Math.Clamp(heightKm < 25f ? heightKm / 15f - 2f / 3f : -heightKm / 15f + 8f / 3f, 0f, 1f);
            float fogKm = s.FogDensity * 1000f;
            return RayleighEarth * s.RayleighScale * MathF.Exp(-heightKm / RayleighHeightKm)
                 + new Vector3((MieScatteringEarth + MieAbsorptionEarth) * s.MieScale * MathF.Exp(-heightKm / MieHeightKm))
                 + new Vector3(fogKm * MathF.Exp(-heightKm / MathF.Max(s.FogHeight * 0.001f, 0.001f)))
                 + OzoneEarth * s.OzoneScale * ozone;
        }

        private static float RaySphereNearest(Vector3 origin, Vector3 dir, float radius)
        {
            float b = 2f * Vector3.Dot(dir, origin);
            float c = Vector3.Dot(origin, origin) - radius * radius;
            float delta = b * b - 4f * c;
            if (delta < 0f) return -1f;
            float s = MathF.Sqrt(delta);
            float t0 = (-b - s) * 0.5f, t1 = (-b + s) * 0.5f;
            if (t0 < 0f && t1 < 0f) return -1f;
            if (t0 < 0f) return MathF.Max(0f, t1);
            if (t1 < 0f) return MathF.Max(0f, t0);
            return MathF.Max(0f, MathF.Min(t0, t1));
        }

        /// <summary>The share of a light's illuminance (per channel) that reaches a point at this height above sea
        /// level, from a direction: 0 when the planet is in the way. Integrated like the transmittance LUT.</summary>
        public static Vector3 TransmittanceToLight(float heightMeters, Vector3 dir)
        {
            var s = Settings;
            dir = Vector3.Normalize(dir);
            var origin = new Vector3(0f, BottomRadiusKm + MathF.Max(heightMeters * 0.001f, 0.001f), 0f);
            if (RaySphereNearest(origin, dir, BottomRadiusKm) >= 0f)
                return Vector3.Zero;

            float tMax = RaySphereNearest(origin, dir, TopRadiusKm);
            const int steps = 40;
            float dt = MathF.Max(tMax, 0f) / steps;
            Vector3 depth = Vector3.Zero;
            for (int i = 0; i < steps; i++)
            {
                var p = origin + dir * ((i + 0.5f) * dt);
                depth += Extinction(s, p.Length() - BottomRadiusKm) * dt;
            }
            return new Vector3(MathF.Exp(-depth.X), MathF.Exp(-depth.Y), MathF.Exp(-depth.Z));
        }

        // The LUT's transmittance toward the sun (read back) against the CPU's for the same frame. Channels the air
        // has all but put out (a set sun) are left out: there both are ~0 and only their noise would differ.
        // Measured 2026-10-17: under 0.5% with the sun high, 1.1–1.6% with it 2–4° up (the LUT always a little higher:
        // its bilinear interpolation of a steep curve near the horizon). A medium that differs between the two copies
        // shows far more (no ground fog on one side: ~70% at a low sun), hence the 3% mark.
        private static void CheckTransmittance(Vector3 gpu, Vector3 cpu)
        {
            float worst = 0f;
            for (int c = 0; c < 3; c++)
            {
                float a = c == 0 ? gpu.X : c == 1 ? gpu.Y : gpu.Z;
                float b = c == 0 ? cpu.X : c == 1 ? cpu.Y : cpu.Z;
                if (b < 0.01f) continue;
                worst = MathF.Max(worst, MathF.Abs(a - b) / b);
            }
            TransmittanceMismatch = worst;

            if (worst > 0.03f && !_mismatchReported)
            {
                _mismatchReported = true;
                Debug.LogWarning("Atmosphere", $"Transmittance toward the sun: LUT {gpu} vs CPU {cpu} ({worst:P1} apart). " +
                    "The CPU copy of the medium (Atmosphere.Extinction) and atmosphere_common.hlsli SampleAtmMedium must match.");
            }
            else if (worst <= 0.015f)
                _mismatchReported = false;
        }

        // ── GPU side ──

        // Must match AtmosphereParams in atmosphere_common.hlsli, field for field
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct GpuParams
        {
            public Vector3 RayleighScattering; public float BottomRadius;
            public Vector3 MieScattering; public float TopRadius;
            public Vector3 MieExtinction; public float MieG;
            public Vector3 OzoneAbsorption; public float RayleighExpScale;
            public Vector3 FogScattering; public float MieExpScale;
            public Vector3 FogExtinction; public float FogExpScale;
            public Vector3 GroundAlbedo; public float SunAngularRadius;
            public Vector3 SunDirection; public float _pad1;
            public Vector3 SunIlluminance; public float APMaxDepth;
            public Vector3 MoonDirection; public float _pad0;
            public Vector3 MoonIlluminance; public float AirglowLuminance;
            public uint TransmittanceLut, MultiScatLut, SkyViewSunLut, SkyViewMoonLut;
            public uint AerialPerspective, Ambient, TerrainHeightmap; public float TerrainMaxHeight;
            public Vector3 TerrainOrigin; public float TerrainSizeX;
            public float TerrainSizeZ; public uint SkyCaptureViews, SkyCaptureDepth, SkyCaptureCount;
            public float SkyCaptureBounce; public uint SkyProbeCube, SkyProbeSH; public float SkyProbeMips;
            public uint TerrainSunShadow, TerrainMoonShadow; public Vector2 TerrainShadowOrigin;
            public float TerrainShadowSize;

            /// <summary>The part the transmittance and multiple scattering LUTs depend on</summary>
            public bool SameMedium(in GpuParams o) =>
                RayleighScattering == o.RayleighScattering && MieScattering == o.MieScattering &&
                MieExtinction == o.MieExtinction && OzoneAbsorption == o.OzoneAbsorption &&
                FogScattering == o.FogScattering && FogExtinction == o.FogExtinction && GroundAlbedo == o.GroundAlbedo &&
                RayleighExpScale == o.RayleighExpScale && MieExpScale == o.MieExpScale && FogExpScale == o.FogExpScale;
        }

        private const int FrameCount = 3;
        private const int AmbientElements = 5;      // atmosphere_cs.hlsl CSAmbient: sky, ground, sky (older units), sun
                                                    // transmittance; CSCloudDeck: the cloud layer
        private const int TransmittanceW = 256, TransmittanceH = 64, MultiScatRes = 32;
        private const int SkyViewW = 192, SkyViewH = 108;

        private static ComputeShader? _cs;
        private static int _kTransmittance, _kMultiScat, _kSkyView, _kAP, _kAmbient, _kCloudDeck;
        private static RenderTexture2D? _transmittance, _multiScat, _skyViewSun, _skyViewMoon;
        private static GraphicsBuffer? _ambient;
        private static readonly GraphicsBuffer?[] _params = new GraphicsBuffer?[FrameCount];
        private static readonly GraphicsBuffer?[] _readback = new GraphicsBuffer?[FrameCount];
        private static bool _texturesWritten;      // first use: textures start in Common
        private static bool _mediumValid;
        private static GpuParams _medium;
        private static GpuParams _frame;
        private static int _preparedTick = -1;

        private static void EnsureResources()
        {
            if (_cs != null) return;
            var device = Engine.Device;
            _cs = new ComputeShader("atmosphere_cs.hlsl");
            _kTransmittance = _cs.FindKernel("CSTransmittance");
            _kMultiScat = _cs.FindKernel("CSMultiScattering");
            _kSkyView = _cs.FindKernel("CSSkyView");
            _kAP = _cs.FindKernel("CSAerialPerspective");
            _kAmbient = _cs.FindKernel("CSAmbient");
            _kCloudDeck = _cs.FindKernel("CSCloudDeck");

            // fp32: these hold light per unit of the sun's illuminance, and in twilight the sky's share falls to ~1e-6 and
            // below, fp16's subnormals (steps of 6e-8): an exposure adapted to dusk showed the sky-view LUT's texels as
            // stair-stepped rings round the zenith. The AP volume (one per view, AerialPerspectiveVolume) is pre-exposed
            // and stays fp16.
            _transmittance = new RenderTexture2D(device, TransmittanceW, TransmittanceH, Format.R32G32B32A32_Float, randomWrite: true);
            _multiScat = new RenderTexture2D(device, MultiScatRes, MultiScatRes, Format.R32G32B32A32_Float, randomWrite: true);
            _skyViewSun = new RenderTexture2D(device, SkyViewW, SkyViewH, Format.R32G32B32A32_Float, randomWrite: true);
            _skyViewMoon = new RenderTexture2D(device, SkyViewW, SkyViewH, Format.R32G32B32A32_Float, randomWrite: true);
            _ambient = GraphicsBuffer.CreateStructured<Vector4>(AmbientElements, srv: true, uav: true);
            for (int i = 0; i < FrameCount; i++)
            {
                _params[i] = GraphicsBuffer.CreateUpload<GpuParams>(1, mapped: true);
                _readback[i] = GraphicsBuffer.CreateReadback<Vector4>(AmbientElements);
            }

            _cs.SetPushConstant("TransmittanceUAV", _transmittance.UavIndex);
            _cs.SetPushConstant("MultiScatUAV", _multiScat.UavIndex);
            _cs.SetPushConstant("SkyViewSunUAV", _skyViewSun.UavIndex);
            _cs.SetPushConstant("SkyViewMoonUAV", _skyViewMoon.UavIndex);
            _cs.SetPushConstant("AmbientUAV", _ambient.UavIndex);
        }

        /// <summary>
        /// Write this frame's parameters (once per frame) and return the bindless index of their buffer, for
        /// SceneConstants.AtmosphereIdx. Also takes the ambient light read back from the frame that last used this slot,
        /// and checks the LUT's transmittance toward the sun against the CPU's for that frame.
        /// </summary>
        public static uint PrepareFrame(float cameraHeight)
        {
            EnsureResources();
            int slot = Engine.FrameIndex % FrameCount;
            if (_preparedTick == Engine.TickCount)
                return _params[slot]!.SrvIndex;
            _preparedTick = Engine.TickCount;

            unsafe
            {
                var back = _readback[slot]!.ReadPtr<Vector4>();
                SkyAmbientLegacy = new Vector3(back[2].X, back[2].Y, back[2].Z);
                if (back[4] != Vector4.Zero)        // all 0: not written yet (Y, all the clouds let through, is never 0)
                    CloudDeck = back[4];
                CheckTransmittance(new Vector3(back[3].X, back[3].Y, back[3].Z), _cpuSunTransmittance[slot]);
            }
            _cpuSunTransmittance[slot] = TransmittanceToLight(cameraHeight, Settings.SunDirection);

            var s = Settings;
            float fogKm = s.FogDensity * 1000f;
            float mieScattering = MieScatteringEarth * s.MieScale;
            _frame = new GpuParams
            {
                RayleighScattering = RayleighEarth * s.RayleighScale,
                BottomRadius = BottomRadiusKm,
                MieScattering = new Vector3(mieScattering),
                TopRadius = TopRadiusKm,
                MieExtinction = new Vector3(mieScattering + MieAbsorptionEarth * s.MieScale),
                MieG = Math.Clamp(s.MieAnisotropy, -0.95f, 0.95f),
                OzoneAbsorption = OzoneEarth * s.OzoneScale,
                RayleighExpScale = -1f / RayleighHeightKm,
                FogScattering = s.FogColor * fogKm,
                MieExpScale = -1f / MieHeightKm,
                FogExtinction = new Vector3(fogKm),
                FogExpScale = -1f / MathF.Max(s.FogHeight * 0.001f, 0.001f),
                GroundAlbedo = s.GroundAlbedo,
                SunAngularRadius = s.SunAngularRadius,
                SunDirection = Vector3.Normalize(s.SunDirection),
                SunIlluminance = s.SunIlluminance,
                APMaxDepth = APMaxDepth,
                MoonDirection = Vector3.Normalize(s.MoonDirection),
                MoonIlluminance = s.MoonIlluminance,
                AirglowLuminance = s.AirglowLuminance,
                TransmittanceLut = _transmittance!.BindlessIndex,
                MultiScatLut = _multiScat!.BindlessIndex,
                SkyViewSunLut = _skyViewSun!.BindlessIndex,
                SkyViewMoonLut = _skyViewMoon!.BindlessIndex,
                AerialPerspective = 0,      // each view has its own (SceneConstants.AerialPerspectiveIdx)
                Ambient = _ambient!.SrvIndex,
            };
            // Sky visibility (sky_common.fx SkyVisibility): the terrain shades the sky's light where hills rise round a
            // place or over it (a cave under a hill)
            foreach (var terrain in Base.ComponentCache<Components.TerrainRenderer>.All)
            {
                if (!Engine.Settings.SkyVisibility) break;
                var heightmap = terrain.Enabled ? terrain.Terrain?.Heightmap : null;
                if (heightmap == null) continue;
                _frame.TerrainHeightmap = heightmap.BindlessIndex;
                _frame.TerrainMaxHeight = terrain.Terrain!.MaxHeight;
                _frame.TerrainOrigin = terrain.Transform!.WorldPosition;
                _frame.TerrainSizeX = terrain.Terrain.TerrainSize.X;
                _frame.TerrainSizeZ = terrain.Terrain.TerrainSize.Y;
                break;
            }
            // ... and the meshes round the camera: walls, roofs, tree crowns (SkyCapture, sky_capture.hlsli)
            (_frame.SkyCaptureViews, _frame.SkyCaptureDepth, _frame.SkyCaptureCount) = SkyCapture.PrepareFrame(slot);
            _frame.SkyCaptureBounce = Math.Clamp(Engine.Settings.SkyCaptureBounce, 0f, 1f);
            // The sky probe (SkyProbe, sky_probe_cs.hlsl): the ambient light and the sky's reflections
            (_frame.SkyProbeCube, _frame.SkyProbeSH, _frame.SkyProbeMips) = SkyProbe.Indices;
            // The terrain's shadow in the air toward the sun and the moon (TerrainLightShadow, made in Render when due):
            // the haze and the near fog in front of a mountain the light stands behind
            (_frame.TerrainSunShadow, _frame.TerrainMoonShadow, _frame.TerrainShadowOrigin, _frame.TerrainShadowSize) =
                TerrainLightShadow.Prepare(TerrainCover, _frame.SunDirection, _frame.MoonDirection, s.MoonIlluminance != Vector3.Zero);
            unsafe { *_params[slot]!.WritePtr<GpuParams>() = _frame; }
            return _params[slot]!.SrvIndex;
        }

        /// <summary>
        /// Make the LUTs for the view about to be drawn, and its aerial perspective volume (<paramref name="aerialPerspective"/>,
        /// prepared to the view's size): the scene constants (camera, AtmosphereIdx, Exposure) must be set and bound for
        /// compute by <paramref name="bindSceneConstants"/>. Call after <see cref="PrepareFrame"/> (via
        /// Camera.SetShaderParams), before anything samples the sky.
        /// </summary>
        public static void Render(ID3D12GraphicsCommandList list, Action<ID3D12GraphicsCommandList> bindSceneConstants,
            AerialPerspectiveVolume aerialPerspective)
        {
            EnsureResources();
            var cs = _cs!;
            var device = Engine.Device;

            list.SetComputeRootSignature(device.GlobalRootSignature);
            list.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            bindSceneConstants(list);

            var shaderRead = ResourceStates.NonPixelShaderResource | ResourceStates.PixelShaderResource;
            ResourceStates from = _texturesWritten ? shaderRead : ResourceStates.Common;

            PixMarker.Begin(list, "Atmosphere");

            // The terrain's shadow toward each light, when the light moved or the terrain changed: before the aerial
            // perspective, which reads it
            TerrainLightShadow.Render(list);

            // The volumetric clouds' field round the camera: the brick columns that entered or are due (once per frame)
            Clouds.Render(list);

            // Transmittance and multiple scattering: only when the medium changed
            if (!_mediumValid || !_frame.SameMedium(_medium))
            {
                list.ResourceBarrierTransition(_transmittance!.Native, from, ResourceStates.UnorderedAccess);
                cs.Dispatch(_kTransmittance, list, TransmittanceW / 8, TransmittanceH / 8);
                list.ResourceBarrierTransition(_transmittance.Native, ResourceStates.UnorderedAccess, shaderRead);

                list.ResourceBarrierTransition(_multiScat!.Native, from, ResourceStates.UnorderedAccess);
                cs.Dispatch(_kMultiScat, list, MultiScatRes, MultiScatRes);
                list.ResourceBarrierTransition(_multiScat.Native, ResourceStates.UnorderedAccess, shaderRead);

                _medium = _frame;
                _mediumValid = true;
            }

            // The cloud layer round the camera (Ambient[4]): the sky-view, the aerial perspective and the near fog volume
            // shade the air below it with it
            _ambient!.Transition(list, ResourceStates.UnorderedAccess);
            cs.Dispatch(_kCloudDeck, list, 1);
            _ambient.Transition(list, shaderRead);

            // Sky-view (sun and moon)
            list.ResourceBarrierTransition(_skyViewSun!.Native, from, ResourceStates.UnorderedAccess);
            list.ResourceBarrierTransition(_skyViewMoon!.Native, from, ResourceStates.UnorderedAccess);
            cs.Dispatch(_kSkyView, list, (SkyViewW + 7) / 8, (SkyViewH + 7) / 8, 2);
            list.ResourceBarrierTransition(_skyViewSun.Native, ResourceStates.UnorderedAccess, shaderRead);
            list.ResourceBarrierTransition(_skyViewMoon.Native, ResourceStates.UnorderedAccess, shaderRead);

            // This view's aerial perspective: one thread per column, marching through all of its slices
            var ap = aerialPerspective.Texture;
            if (ap != null)
            {
                PixMarker.Begin(list, "Aerial perspective");
                if (aerialPerspective.State != ResourceStates.UnorderedAccess)
                    list.ResourceBarrierTransition(ap.Native, aerialPerspective.State, ResourceStates.UnorderedAccess);
                cs.SetPushConstant("APUAV", ap.UavIndex);
                cs.Dispatch(_kAP, list, (uint)(aerialPerspective.Width + 7) / 8, (uint)(aerialPerspective.Height + 7) / 8);
                list.ResourceBarrierTransition(ap.Native, ResourceStates.UnorderedAccess, shaderRead);
                aerialPerspective.State = shaderRead;
                PixMarker.End(list);
            }

            // Ambient from the sky-view LUTs, then a copy for the CPU (Daylight)
            _ambient!.Transition(list, ResourceStates.UnorderedAccess);
            cs.Dispatch(_kAmbient, list, 1);
            _ambient.Transition(list, ResourceStates.CopySource);
            int slot = Engine.FrameIndex % FrameCount;
            list.CopyBufferRegion(_readback[slot]!.Native, 0, _ambient.Native, 0, AmbientElements * 16);
            _ambient.Transition(list, shaderRead);

            // The sky probe: the sky, the clouds and the lit ground round the camera (reads the ambient's ground light)
            SkyProbe.Render(list);

            _texturesWritten = true;
            PixMarker.End(list);
        }
    }
}
