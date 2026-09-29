using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Graphics;
using Freefall.Base;
using Freefall.Assets;
using Vortice.Mathematics;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Category = System.ComponentModel.CategoryAttribute;

namespace Freefall.Components
{
    // TODO: Sky shader is ~1ms on RTX 4080 — excessive for a flat-dome sky.
    // Root cause: mesh_skybox.fx GetClouds() computes all noise procedurally per pixel:
    //   - fbm(8 octaves) + fbm(4) + worley(27 iterations) + fbm(3) + fbm(3) + domain warp fbm(3)x2
    //   - Total: ~24 octaves of sin()-based noise + 27-iter Worley per sky pixel at full res
    // Fix: Replace procedural noise with precomputed 3D noise textures (128³ Perlin-Worley + 32³ detail).
    //   - Single texture fetch vs 100+ trig ops per pixel
    //   - Budget freed up could support volumetric ray-marched clouds (32-64 steps at quarter-res)
    //     with proper volumetric lighting, god rays, and camera parallax — still under 1ms.
    //
    // Look parameters (sun colors, atmosphere, clouds, stars) can be driven from an
    // EnvironmentPreset asset via EnvironmentController — see ApplyPreset(). The fields
    // stay public so the skybox still works standalone with hand-tweaked values.
    [Icon("icon_sky.png")]
    [UpdateInEditor]
    public class SkyboxRenderer : Component, IUpdate, IDraw
    {
        private Mesh Mesh;
        private Material Material = InternalAssets.SkyboxMaterial;
        private MaterialBlock Params = new MaterialBlock();

        [Category("Time of Day")]
        public bool AnimateTimeOfDay = false;

        [ValueRange(0f, 24f)]
        public float TimeOfDay = 16f;         // 0-24 hours

        [ValueRange(0.1f, 1f)]
        public float TimeOfDaySpeed = 0.1f; // Hours per second (tweak for testing)

        [ValueRange(0f, 360f)]
        public float SunAzimuthAngle = 30.0f;    // Compass heading for sunrise in degrees (0=+X, 90=+Z, 180=-X, 270=-Z)

        [ValueRange(5f, 90f)]
        public float MaxSunElevation = 60.0f;    // Highest sun angle above the horizon at noon (90 = straight overhead). Lower = longer shadows all day.

        [Category("Sun")]
        public DirectionalLight SunLight;

        [ValueRange(0f, 10f)]
        public float SunIntensity = 1.2f;       // Sun brightness multiplier

        [ValueRange(0f, 10f)]
        public float DayIntensity = 3.14159f;       // Sun intensity at noon

        [ValueRange(0f, 10f)]
        public float SunsetIntensity = 1.5f;    // Sun intensity during sunset/sunrise

        [ValueRange(0f, 10f)]
        public float NightIntensity = 0.1f;     // Ambient intensity at night

        public Color3 SunDayColor = new Color3(1.0f, 0.95f, 0.85f);
        public Color3 SunSunsetColor = new Color3(1.0f, 0.7f, 0.4f);
        public Color3 SunNightColor = new Color3(0.1f, 0.15f, 0.3f);

        public Vector3 SunDirection = new Vector3(0, 1, 0);

        [Category("Atmosphere")]
        public Color3 SkyTintColor = new Color3(0.5f, 0.7f, 1.0f);       // Overall sky color multiplier

        [ValueRange(0.5f, 3f)]
        public float AtmosphereDensity = 1.0f;          // Global atmosphere thickness

        [ValueRange(0f, 1f)]
        public float MieScattering = 0.02f;             // Mie haze/glow strength (sun halo)

        [ValueRange(0f, 0.99f)]
        public float MieAnisotropy = 0.76f;             // HG anisotropy — higher = tighter sun glow

        [Category("Haze")]
        public Color3 HazeColor = new Color3(0.8f, 0.85f, 0.9f);         // Colored haze at horizon

        [ValueRange(0f, 2f)]
        public float HazeIntensity = 0.3f;              // Horizon haze strength

        [ValueRange(0f, 1f)]
        public float HazeHeight = 0.15f;                // How high haze reaches (viewDir.y)

        [Category("Sunset")]
        public Color3 SunsetTintColor = new Color3(1.0f, 0.5f, 0.2f);   // Warm color near horizon at sunset

        [ValueRange(0f, 2f)]
        public float SunsetTintIntensity = 0.8f;        // Sunset color strength

        [Category("Night")]
        public Color3 NightSkyColor = new Color3(0.01f, 0.01f, 0.04f);   // Zenith color at night
        public Color3 NightHorizonColor = new Color3(0.03f, 0.04f, 0.08f); // Horizon glow at night

        [ValueRange(0f, 1f)]
        public float StarDensity = 0.5f;        // 0-1, controls how many stars

        [ValueRange(0f, 10f)]
        public float StarBrightness = 1.0f;     // Star intensity multiplier

        [Category("Clouds")]
        [ValueRange(0f, 1f)]
        public float CloudCoverage = 0.5f;      // 0-1

        [ValueRange(0f, 10f)]
        public float CloudSpeed = 1.0f;         // Speed multiplier

        [ValueRange(500f, 5000f)]
        public float CloudAltitude = 1800.0f;       // Cloud layer height in world units

        [ValueRange(0f, 3f)]
        public float CloudBrightness = 1.0f;        // Cloud brightness (day)

        public Color3 CloudShadowColor = new Color3(0.35f, 0.4f, 0.55f);   // Cloud shade side (day)
        public Color3 CloudSunlitColor = new Color3(1.0f, 0.98f, 0.95f);   // Sun-facing cloud tops (day)
        public Color3 CloudSunsetTintColor = new Color3(1.0f, 0.6f, 0.3f); // Warm tint mixed in at sunset/sunrise
        public Color3 CloudNightColor = new Color3(0.08f, 0.09f, 0.14f);   // Moonlit clouds at night

        [ValueRange(0f, 3f)]
        public float CloudNightBrightness = 1.0f;   // Cloud brightness (night)

        // Cross-fade state for altitude changes, driven by EnvironmentController. Altitude scales the
        // projected cloud pattern, so instead of lerping it the shader blends two cloud layers.
        [Freefall.Reflection.DontSerialize, System.ComponentModel.Browsable(false)]
        public float CloudAltitudeFrom = 1800.0f;

        [Freefall.Reflection.DontSerialize, System.ComponentModel.Browsable(false)]
        public float CloudAltitudeBlend = 1.0f;     // 1 = only CloudAltitude is shown


        // Static ambient scale accessible by composition pass
        public static float AmbientScale { get; private set; } = 1.0f;
        public static Vector3 CurrentSunDirection { get; private set; } = new Vector3(0, 1, 0);

        // Static atmosphere accessors for Camera.SetShaderParams()
        public static Vector3 CurrentSkyTintColor { get; private set; } = new Vector3(0.5f, 0.7f, 1.0f);
        public static Vector3 CurrentHazeColor { get; private set; } = new Vector3(0.8f, 0.85f, 0.9f);
        public static float CurrentHazeIntensity { get; private set; } = 0.3f;
        public static float CurrentHazeHeight { get; private set; } = 0.15f;
        public static Vector3 CurrentSunsetTintColor { get; private set; } = new Vector3(1.0f, 0.5f, 0.2f);
        public static float CurrentSunsetTintIntensity { get; private set; } = 0.8f;
        public static float CurrentAtmosphereDensity { get; private set; } = 1.0f;
        public static float CurrentMieScattering { get; private set; } = 0.02f;
        public static float CurrentMieAnisotropy { get; private set; } = 0.76f;
        public static Vector3 CurrentNightSkyColor { get; private set; } = new Vector3(0.01f, 0.01f, 0.04f);
        public static Vector3 CurrentNightHorizonColor { get; private set; } = new Vector3(0.03f, 0.04f, 0.08f);

        // Day/sunset/night weights of the current frame (normalized, sum to 1). Useful for gameplay/audio.
        public static float CurrentDayFactor { get; private set; } = 1.0f;
        public static float CurrentSunsetFactor { get; private set; } = 0.0f;
        public static float CurrentNightFactor { get; private set; } = 0.0f;

        private float CloudTime = 0.0f;

        // Cloud noise LUT — generated once at startup via compute shader
        private static RenderTexture3D? _cloudNoiseLUT;
        private static bool _noiseGenerated;


        public SkyboxRenderer()
        {
        }

        protected override void Awake()
        {
            Mesh = Mesh.CreateCube(Engine.Device, 100.0f);
            SunLight ??= EntityManager.FindComponent<DirectionalLight>();
            GenerateCloudNoiseLUT();
        }

        public override void Destroy()
        {
            Mesh?.Dispose();
        }

        private void GenerateCloudNoiseLUT()
        {
            if (_noiseGenerated) return;
            _noiseGenerated = true;

            const int size = 128;
            var device = Engine.Device;

            _cloudNoiseLUT = new RenderTexture3D(device, size, size, size, Format.R8G8B8A8_UNorm);

            var shader = new ComputeShader("cloud_noise_gen.hlsl", "CSGenNoise");
            int kernel = shader.FindKernel("CSGenNoise");

            shader.SetPushConstant(kernel, "OutputUAV", _cloudNoiseLUT.UavIndex);
            shader.SetPushConstant(kernel, "VolumeSize", (uint)size);

            // Dispatch: 128/4 = 32 groups per axis
            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            var cmd = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, allocator);
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            shader.Dispatch(kernel, cmd, (uint)(size / 4), (uint)(size / 4), (uint)(size / 4));

            cmd.Close();
            device.SubmitAndWait(cmd);
            cmd.Dispose();
            allocator.Dispose();
            shader.Dispose();

            Debug.Log("SkyboxRenderer", $"Cloud noise LUT generated: {size}x{size}x{size} RGBA8");
        }

        public void Update()
        {
            if (Camera.Main == null) return;

            // Move with camera to simulate infinite distance
            Transform.Position = Camera.Main.Position;

            // Animate time of day if enabled
            if (AnimateTimeOfDay)
            {
                TimeOfDay += (float)Time.Delta * TimeOfDaySpeed;
                if (TimeOfDay >= 24.0f) TimeOfDay -= 24.0f;
                if (TimeOfDay < 0.0f) TimeOfDay += 24.0f;

            }

            CloudTime += (float)Time.Delta * CloudSpeed;

            UpdateSunLight();
        }

        /// <summary>
        /// Sun direction for a time of day, on a circle tilted away from the zenith so the sun
        /// peaks at <paramref name="maxElevationDeg"/> at noon instead of passing straight overhead
        /// (same geometry as an equinox day at latitude 90 - maxElevation). Keeps noon shadows from
        /// collapsing to nothing. Returns the unclamped direction (y &lt; 0 at night).
        /// </summary>
        public static Vector3 ComputeSunDirection(float timeOfDay, float azimuthDeg, float maxElevationDeg, out Vector3 horizontal, out float elevation)
        {
            // 0 at sunrise (06:00), PI/2 at noon, PI at sunset
            float sunAngle = (timeOfDay / 24.0f) * MathF.PI * 2.0f - MathF.PI * 0.5f;

            float maxElevRad = Math.Clamp(maxElevationDeg, 1f, 90f) * (MathF.PI / 180.0f);
            float along = MathF.Cos(sunAngle);              // position along the sunrise → sunset axis
            float arc = MathF.Sin(sunAngle);                // height along the tilted arc
            elevation = arc * MathF.Sin(maxElevRad);        // vertical component
            float lean = arc * MathF.Cos(maxElevRad);       // horizontal lean away from the zenith

            // Local frame: X = sunrise heading, Z = direction the arc leans toward; rotate by azimuth around Y
            float azimuthRad = azimuthDeg * (MathF.PI / 180.0f);
            float cosAz = MathF.Cos(azimuthRad), sinAz = MathF.Sin(azimuthRad);
            Vector3 east = new Vector3(cosAz, 0, sinAz);
            Vector3 south = new Vector3(-sinAz, 0, cosAz);
            horizontal = east * along + south * lean;

            return Vector3.Normalize(horizontal + Vector3.UnitY * elevation);
        }

        private void UpdateSunLight()
        {
            if (SunLight == null) return;

            SunDirection = ComputeSunDirection(TimeOfDay, SunAzimuthAngle, MaxSunElevation, out Vector3 horizontal, out float elevation);

            // Clamp light direction to prevent it from shining from below ground
            float clampedElevation = MathF.Max(0.1f, elevation);
            Vector3 lightDir = Vector3.Normalize(horizontal + Vector3.UnitY * clampedElevation);

            // Set directional light rotation using CreateWorld
            // Transform.Forward is what DirectionalLight uses as LightDirection
            // The shader already negates this (L = -LightDirection), so we pass the direction light is pointing (toward ground)
            SunLight.Transform.Rotation = Quaternion.CreateFromRotationMatrix(Matrix4x4.CreateWorld(Vector3.Zero, lightDir, Vector3.UnitY));

            // ── Three-state blend matching sky_common.fx GetSkyColor ──
            ComputeDayNightFactors(elevation, out float dayFactor, out float sunsetFactor, out float nightFactor);
            CurrentDayFactor = dayFactor;
            CurrentSunsetFactor = sunsetFactor;
            CurrentNightFactor = nightFactor;

            // Blended intensity
            float intensity = dayFactor * DayIntensity
                            + sunsetFactor * SunsetIntensity
                            + nightFactor * NightIntensity;
            SunLight.Intensity = intensity * SunIntensity;

            // Ambient tracks blended sky brightness
            AmbientScale = dayFactor * 1.0f + sunsetFactor * 0.4f + nightFactor * 0.05f;
            CurrentSunDirection = SunDirection;

            // Sync atmosphere params to static accessors
            CurrentSkyTintColor = new Vector3(SkyTintColor.R, SkyTintColor.G, SkyTintColor.B);
            CurrentHazeColor = new Vector3(HazeColor.R, HazeColor.G, HazeColor.B);
            CurrentHazeIntensity = HazeIntensity;
            CurrentHazeHeight = HazeHeight;
            CurrentSunsetTintColor = new Vector3(SunsetTintColor.R, SunsetTintColor.G, SunsetTintColor.B);
            CurrentSunsetTintIntensity = SunsetTintIntensity;
            CurrentAtmosphereDensity = AtmosphereDensity;
            CurrentMieScattering = MieScattering;
            CurrentMieAnisotropy = MieAnisotropy;
            CurrentNightSkyColor = new Vector3(NightSkyColor.R, NightSkyColor.G, NightSkyColor.B);
            CurrentNightHorizonColor = new Vector3(NightHorizonColor.R, NightHorizonColor.G, NightHorizonColor.B);

            // Blended light color (day/sunset/night palettes matching shader)
            SunLight.Color = new Color3(
                dayFactor * SunDayColor.R + sunsetFactor * SunSunsetColor.R + nightFactor * SunNightColor.R,
                dayFactor * SunDayColor.G + sunsetFactor * SunSunsetColor.G + nightFactor * SunNightColor.G,
                dayFactor * SunDayColor.B + sunsetFactor * SunSunsetColor.B + nightFactor * SunNightColor.B);
        }

        /// <summary>
        /// Day / sunset / night weights (normalized, sum to 1) for a given sun elevation (sunDir.y).
        /// Must stay in sync with GetSkyColor() in sky_common.fx and the cloud shading in mesh_skybox.fx.
        /// </summary>
        public static void ComputeDayNightFactors(float elevation, out float day, out float sunset, out float night)
        {
            // Day: non-linear ramp (same pow(0.7) as shader)
            day = MathF.Pow(Math.Clamp(elevation / 0.8f, 0.0f, 1.0f), 0.7f);

            // Sunset: triangular peak centered at elevation ≈ -0.025
            sunset = 0.0f;
            if (elevation < 0.15f && elevation > -0.2f)
                sunset = MathF.Max(0.0f, 1.0f - MathF.Abs((elevation - (-0.025f)) / 0.175f));

            // Night: smooth onset below horizon
            night = Math.Clamp((-elevation - 0.15f) / 0.3f, 0.0f, 1.0f);

            float total = day + sunset + night;
            if (total > 0.0f)
            {
                day /= total;
                sunset /= total;
                night /= total;
            }
        }

        /// <summary>
        /// Overwrite every look parameter (sun, atmosphere, clouds, stars) from a preset.
        /// Time of day, azimuth and max elevation are left alone — they describe *when*,
        /// the preset describes *how it looks*. Called every frame by EnvironmentController
        /// with its blended preset.
        /// </summary>
        public void ApplyPreset(EnvironmentPreset p)
        {
            if (p == null) return;

            SunIntensity = p.SunIntensity;
            DayIntensity = p.DayIntensity;
            SunsetIntensity = p.SunsetIntensity;
            NightIntensity = p.NightIntensity;
            SunDayColor = p.SunDayColor;
            SunSunsetColor = p.SunSunsetColor;
            SunNightColor = p.SunNightColor;

            SkyTintColor = p.SkyTintColor;
            AtmosphereDensity = p.AtmosphereDensity;
            MieScattering = p.MieScattering;
            MieAnisotropy = p.MieAnisotropy;
            HazeColor = p.HazeColor;
            HazeIntensity = p.HazeIntensity;
            HazeHeight = p.HazeHeight;

            SunsetTintColor = p.SunsetTintColor;
            SunsetTintIntensity = p.SunsetTintIntensity;

            NightSkyColor = p.NightSkyColor;
            NightHorizonColor = p.NightHorizonColor;
            StarDensity = p.StarDensity;
            StarBrightness = p.StarBrightness;

            CloudCoverage = p.CloudCoverage;
            CloudSpeed = p.CloudSpeed;
            CloudAltitude = p.CloudAltitude;
            CloudBrightness = p.CloudBrightness;
            CloudSunlitColor = p.CloudSunlitColor;
            CloudShadowColor = p.CloudShadowColor;
            CloudSunsetTintColor = p.CloudSunsetTintColor;
            CloudNightColor = p.CloudNightColor;
            CloudNightBrightness = p.CloudNightBrightness;
        }

        /// <summary>Capture the current look into a preset (inverse of ApplyPreset) — "save what I tweaked".</summary>
        public void CaptureToPreset(EnvironmentPreset p)
        {
            if (p == null) return;

            p.SunIntensity = SunIntensity;
            p.DayIntensity = DayIntensity;
            p.SunsetIntensity = SunsetIntensity;
            p.NightIntensity = NightIntensity;
            p.SunDayColor = SunDayColor;
            p.SunSunsetColor = SunSunsetColor;
            p.SunNightColor = SunNightColor;

            p.SkyTintColor = SkyTintColor;
            p.AtmosphereDensity = AtmosphereDensity;
            p.MieScattering = MieScattering;
            p.MieAnisotropy = MieAnisotropy;
            p.HazeColor = HazeColor;
            p.HazeIntensity = HazeIntensity;
            p.HazeHeight = HazeHeight;

            p.SunsetTintColor = SunsetTintColor;
            p.SunsetTintIntensity = SunsetTintIntensity;

            p.NightSkyColor = NightSkyColor;
            p.NightHorizonColor = NightHorizonColor;
            p.StarDensity = StarDensity;
            p.StarBrightness = StarBrightness;

            p.CloudCoverage = CloudCoverage;
            p.CloudSpeed = CloudSpeed;
            p.CloudAltitude = CloudAltitude;
            p.CloudBrightness = CloudBrightness;
            p.CloudSunlitColor = CloudSunlitColor;
            p.CloudShadowColor = CloudShadowColor;
            p.CloudSunsetTintColor = CloudSunsetTintColor;
            p.CloudNightColor = CloudNightColor;
            p.CloudNightBrightness = CloudNightBrightness;
        }

        public void Draw()
        {
            if (Material == null || Mesh == null) return;

            var slot = Entity.Transform.TransformSlot;

            // Set sky parameters on the MaterialBlock
            Material.SetParameter("World", Entity.Transform.WorldMatrix);
            Material.SetParameter("SunDirection", SunDirection);
            Material.SetParameter("TimeOfDay", TimeOfDay);
            Material.SetParameter("CloudCoverage", CloudCoverage);
            Material.SetParameter("CloudTime", CloudTime);
            Material.SetParameter("CloudSpeed", CloudSpeed);
            Material.SetParameter("SunIntensity", SunIntensity);
            Material.SetParameter("StarDensity", StarDensity);
            Material.SetParameter("StarBrightness", StarBrightness);
            Material.SetParameter("CloudBrightness", CloudBrightness);
            Material.SetParameter("CloudShadowColor", new Vector3(CloudShadowColor.R, CloudShadowColor.G, CloudShadowColor.B));
            Material.SetParameter("CloudAltitude", CloudAltitude);
            Material.SetParameter("CloudSunlitColor", new Vector3(CloudSunlitColor.R, CloudSunlitColor.G, CloudSunlitColor.B));
            Material.SetParameter("CloudNoiseLUTIdx", _cloudNoiseLUT?.BindlessIndex ?? 0u);
            Material.SetParameter("CloudSunsetTintColor", new Vector3(CloudSunsetTintColor.R, CloudSunsetTintColor.G, CloudSunsetTintColor.B));
            Material.SetParameter("CloudNightBrightness", CloudNightBrightness);
            Material.SetParameter("CloudNightColor", new Vector3(CloudNightColor.R, CloudNightColor.G, CloudNightColor.B));
            Material.SetParameter("CloudAltitudeFrom", CloudAltitudeFrom);
            Material.SetParameter("CloudAltitudeBlend", CloudAltitudeBlend);

            CommandBuffer.Enqueue(Mesh, Material, Params, slot);
        }
    }
}
