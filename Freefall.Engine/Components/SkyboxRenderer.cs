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
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Components
{
    /// <summary>
    /// The sky: time of day, the sun and the moons, the atmosphere they shine through, the stars and the cloud layer.
    /// Every frame it hands the atmosphere its settings (<see cref="Atmosphere.Settings"/>: the one source of the sky's
    /// light, the sun's light on the ground, the ambient and the fog, made on the GPU by atmosphere_cs.hlsl) and points
    /// the one directional light at the sun, or at night at the moon that gives the most light, with the illuminance that
    /// reaches the camera through the air. Below the horizon the sun gives no direct light at all: twilight is the
    /// sky's light alone.
    ///
    /// Look parameters can be driven from an EnvironmentPreset asset via EnvironmentController — see ApplyPreset(). The
    /// fields stay public so the skybox still works standalone with hand-tweaked values.
    /// Docs: .agent/knowledge/atmospheric_effects/, ATMOSPHERE_PLAN.md.
    /// </summary>
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

        public int Day = 0;                   // days since day 0: counts on as TimeOfDay wraps, moves the moons along their orbits

        [ValueRange(0.001f, 0.1f)]
        public float TimeOfDaySpeed = 0.001f; // Hours per second (tweak for testing)

        [ValueRange(0f, 360f)]
        public float SunAzimuthAngle = 30.0f;    // Compass heading for sunrise in degrees (0=+X, 90=+Z, 180=-X, 270=-Z)

        [ValueRange(5f, 90f)]
        public float MaxSunElevation = 60.0f;    // Highest sun angle above the horizon at noon (90 = straight overhead). Lower = longer shadows all day.

        [Category("Sun")]
        public DirectionalLight SunLight;

        [Description("Sunlight at the top of the atmosphere in lux (Earth: about 120000). The air takes its share on the way down: the light, the sky and the sun's disc all follow from it")]
        [ValueRange(0f, 200000f)]
        public float SunIlluminance = 120000f;

        [Description("The sun's own tint (its brightness stays SunIlluminance); the reddening of a low sun comes from the air")]
        public Color3 SunDayColor = new Color3(1.0f, 0.95f, 0.85f);

        [Description("Size of the sun's disc as a multiple of the real sun (0.53° across). Only the disc changes: it sends " +
                     "the same light spread wider, so the sunlight, the sky's glow round the sun and the glint on the water stay as they are")]
        [ValueRange(1f, 10f)]
        public float SunDiscScale = 1f;

        public Vector3 SunDirection = new Vector3(0, 1, 0);

        // The world's moons. Their light comes through the one directional light (SunLight): at night it turns to the
        // moon that gives the most light. The sky draws at most MaxMoons of them.
        [Category("Moons")]
        public List<Moon> Moons = [];

        [Category("Atmosphere")]
        [Description("The air's Rayleigh scattering (1 = Earth): the blue of the sky, the reddening of a low sun")]
        [ValueRange(0f, 4f)]
        public float RayleighScale = 1f;

        [Description("Aerosols (1 = Earth's average): the haze over the land, the glow round the sun")]
        [ValueRange(0f, 10f)]
        public float MieScale = 1f;

        [Description("How much the aerosols and the fog scatter forward: higher = a tighter glow round the sun")]
        [ValueRange(0f, 0.95f)]
        public float MieAnisotropy = 0.8f;

        [Description("Ozone (1 = Earth): absorbs orange light high up, which keeps the twilight sky blue overhead")]
        [ValueRange(0f, 4f)]
        public float OzoneScale = 1f;

        [Description("Average albedo of the ground: the light it bounces into the sky and onto surfaces facing down")]
        public Color3 GroundAlbedo = new Color3(0.11f, 0.10f, 0.085f);

        [Category("Ground Fog")]
        [Description("Extinction of the ground fog per metre at sea level: 0.0001 = about 40 km visibility, 0.001 = 4 km, 0 = none")]
        [ValueRange(0f, 0.01f, 0.00001f)]
        public float FogDensity = 0.0001f;

        [Description("Metres of height over which the fog thins to a third")]
        [ValueRange(10f, 3000f)]
        public float FogHeight = 400f;

        [Description("The fog's single-scattering albedo: near white for mist, warm for dust, dark for smoke")]
        public Color3 FogColor = new Color3(0.9f, 0.9f, 0.9f);

        [Category("Night")]
        [Description("Glow of the upper air on a moonless night, cd/m² at the zenith (Earth: about 0.0002); the stars are as bright against it as they were against the old night colours")]
        [ValueRange(0f, 0.01f, 0.00001f)]
        public float AirglowLuminance = 2e-4f;

        [ValueRange(0f, 1f)]
        public float StarDensity = 0.5f;        // 0-1, controls how many stars

        [ValueRange(0f, 10f)]
        public float StarBrightness = 1.0f;     // Star intensity multiplier

        [ValueRange(0f, 4f)]
        public float MilkyWayBrightness = 1.0f; // Glow of the Milky Way band (it turns with the stars)

        [Category("Clouds")]
        [ValueRange(0f, 1f)]
        public float CloudCoverage = 0.5f;      // 0-1

        [ValueRange(0f, 10f)]
        public float CloudSpeed = 1.0f;         // Speed multiplier

        [ValueRange(500f, 5000f)]
        public float CloudAltitude = 1800.0f;       // Cloud layer height in world units

        [Description("Share of the light a cloud sends back: lower = darker clouds (a storm)")]
        [ValueRange(0f, 1f)]
        public float CloudAlbedo = 0.8f;

        [ValueRange(0f, 1f)]
        public float CloudShadowStrength = 0.8f;    // How much sunlight the clouds block on the ground (0 = no cloud shadows)

        [ValueRange(1f, 16f)]
        public float CloudShadowScale = 1.0f;       // Shadow pattern is this many times smaller than the visible clouds (1 = shadows match the clouds in the sky)

        // Cross-fade state for altitude changes, driven by EnvironmentController. Altitude scales the
        // projected cloud pattern, so instead of lerping it the shader blends two cloud layers.
        [Freefall.Reflection.DontSerialize, System.ComponentModel.Browsable(false)]
        public float CloudAltitudeFrom = 1800.0f;

        [Freefall.Reflection.DontSerialize, System.ComponentModel.Browsable(false)]
        public float CloudAltitudeBlend = 1.0f;     // 1 = only CloudAltitude is shown

        public static Vector3 CurrentSunDirection { get; private set; } = new Vector3(0, 1, 0);

        // Cloud layer accessors for Camera.SetShaderParams() (SceneConstants: sky dome + cloud shadows)
        public static float CurrentCloudCoverage { get; private set; } = 0.5f;
        public static float CurrentCloudTime { get; private set; } = 0.0f;
        public static float CurrentCloudAltitude { get; private set; } = 1800.0f;
        public static float CurrentCloudAltitudeFrom { get; private set; } = 1800.0f;
        public static float CurrentCloudAltitudeBlend { get; private set; } = 1.0f;
        public static float CurrentCloudShadowStrength { get; private set; } = 0.0f;   // 0 until a skybox exists
        public static float CurrentCloudShadowScale { get; private set; } = 1.0f;
        public static float CurrentCloudAlbedo { get; private set; } = 0.8f;
        public static uint CurrentCloudNoiseLUTIdx => _cloudNoiseLUT?.BindlessIndex ?? 0u;

        private float CloudTime = 0.0f;

        /// <summary>Moons the sky draws (mesh_skybox.fx MoonDirSize etc. are arrays of this size).</summary>
        public const int MaxMoons = 4;

        // The one directional light fades out, turns to its new source (sun ↔ moon) and fades in again, so its
        // shadows never jump to a new direction while they show
        private const float LightHandoverSeconds = 0.75f;

        private struct MoonLight
        {
            public Vector3 Direction;   // towards the moon
            public Vector3 Top;         // lux at the top of the atmosphere: its LightIntensity × the lit part of the disc, coloured
            public Vector3 AtCamera;    // what reaches the camera through the air (0 below the horizon)
        }

        private readonly MoonLight[] _moonLights = new MoonLight[MaxMoons];
        private readonly Vector4[] _moonDirSize = new Vector4[MaxMoons];
        private readonly Vector4[] _moonColor = new Vector4[MaxMoons];
        private Int4 _moonTextures;
        private int _moonCount;

        private int _lightSource = -1;      // -1 = the sun, else an index into _moonLights
        private float _lightFade = 1f;
        private bool _lightStarted;

        // Star sphere: world direction → star sphere (mesh_skybox.fx StarRotation), and the Milky Way on it
        private Matrix4x4 _starRotation = Matrix4x4.Identity;
        private Vector3 _galaxyPole = Vector3.UnitZ;
        private Vector3 _galaxyCenter = Vector3.UnitX;
        private const float GalaxyTilt = 35f * (MathF.PI / 180f);             // band off the meridian at midnight
        private const float GalaxyCoreFromZenith = 50f * (MathF.PI / 180f);  // where its bright core sits then

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
            // Deferred: the sky draw of the frames in flight still references the mesh
            Engine.Device.DeferDispose(Mesh);
            CurrentCloudShadowStrength = 0.0f;
            Atmosphere.Settings = AtmosphereSettings.Default;
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
                if (TimeOfDay >= 24.0f) { TimeOfDay -= 24.0f; Day++; }
                if (TimeOfDay < 0.0f) { TimeOfDay += 24.0f; Day--; }
            }

            CloudTime += (float)Time.Delta * CloudSpeed;

            CurrentCloudCoverage = CloudCoverage;
            CurrentCloudTime = CloudTime;
            CurrentCloudAltitude = CloudAltitude;
            CurrentCloudAltitudeFrom = CloudAltitudeFrom;
            CurrentCloudAltitudeBlend = CloudAltitudeBlend;
            CurrentCloudShadowStrength = Enabled ? CloudShadowStrength : 0.0f;
            CurrentCloudShadowScale = CloudShadowScale;
            CurrentCloudAlbedo = CloudAlbedo;

            UpdateSunLight(Camera.Main.Position.Y);
            UpdateStarSphere();
            Daylight.Update((float)Time.Delta);
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

        /// <summary>
        /// The atmosphere's settings for this frame, then the directional light: the sun's (or the brightest moon's)
        /// illuminance at the camera's height after its path through the air, from where it really is. Nothing below
        /// the horizon, no clamp to a minimum height: the twilight is the sky's own light.
        /// </summary>
        private void UpdateSunLight(float cameraHeight)
        {
            SunDirection = ComputeSunDirection(TimeOfDay, SunAzimuthAngle, MaxSunElevation, out _, out _);
            CurrentSunDirection = SunDirection;

            UpdateMoons();
            UpdateAtmosphere();

            // The light that reaches the camera through this frame's air (the CPU twin of the transmittance LUT)
            Vector3 sun = Atmosphere.Settings.SunIlluminance * Atmosphere.TransmittanceToLight(cameraHeight, SunDirection);
            for (int i = 0; i < _moonCount; i++)
                _moonLights[i].AtCamera = _moonLights[i].Top * Atmosphere.TransmittanceToLight(cameraHeight, _moonLights[i].Direction);

            if (SunLight == null) return;

            ChooseLightSource(Luminance(sun));
            Vector3 lightDir = SunDirection;
            Vector3 illuminance = sun;
            if (_lightSource >= 0)
            {
                lightDir = _moonLights[_lightSource].Direction;
                illuminance = _moonLights[_lightSource].AtCamera;
            }

            // Transform.Forward is what DirectionalLight uses as LightDirection; CreateWorld's forward becomes -Z, so
            // passing the direction towards the source leaves Forward pointing from it towards the ground
            Vector3 up = MathF.Abs(lightDir.Y) > 0.999f ? Vector3.UnitZ : Vector3.UnitY;
            SunLight.Transform.Rotation = Quaternion.CreateFromRotationMatrix(Matrix4x4.CreateWorld(Vector3.Zero, lightDir, up));

            // DirectionalLight.Intensity is in the engine's older units, which the shaders scale by SunLightScale:
            // lux × ReferenceExposure / π (light_directional_cs lights with Color × Intensity × π)
            float peak = MathF.Max(illuminance.X, MathF.Max(illuminance.Y, illuminance.Z));
            SunLight.Intensity = peak * Atmosphere.ReferenceExposure / MathF.PI * _lightFade;
            if (peak > 0f)
                SunLight.Color = new Color3(illuminance.X / peak, illuminance.Y / peak, illuminance.Z / peak);
        }

        private void UpdateAtmosphere()
        {
            // The brightest moon (above the atmosphere) lights the sky's second slice and the night's fog
            int brightest = -1;
            for (int i = 0; i < _moonCount; i++)
                if (brightest < 0 || Luminance(_moonLights[i].Top) > Luminance(_moonLights[brightest].Top))
                    brightest = i;

            Atmosphere.Settings = new AtmosphereSettings
            {
                SunDirection = SunDirection,
                SunIlluminance = Tint(SunDayColor) * SunIlluminance,
                SunAngularRadius = AtmosphereSettings.Default.SunAngularRadius * Math.Clamp(SunDiscScale, 1f, 10f),
                MoonDirection = brightest >= 0 ? _moonLights[brightest].Direction : -SunDirection,
                MoonIlluminance = brightest >= 0 ? _moonLights[brightest].Top : Vector3.Zero,
                AirglowLuminance = AirglowLuminance,
                CloudShade = Math.Clamp(1f - CurrentCloudShadowStrength * CloudCoverage, 0f, 1f),
                RayleighScale = RayleighScale,
                MieScale = MieScale,
                MieAnisotropy = MieAnisotropy,
                OzoneScale = OzoneScale,
                GroundAlbedo = V(GroundAlbedo),
                FogDensity = FogDensity,
                FogHeight = FogHeight,
                FogColor = V(FogColor),
            };
        }

        /// <summary>
        /// Where each moon stands, how much of it the sun lights and the light it gives; and what the sky shader draws.
        /// A moon falls behind the sun by its phase × 24 h: beside the sun when new (phase 0), opposite it when full.
        /// </summary>
        private void UpdateMoons()
        {
            _moonCount = 0;
            Span<int> textures = stackalloc int[MaxMoons];
            double days = Day + TimeOfDay / 24.0;
            foreach (var moon in Moons)
            {
                if (moon == null || _moonCount == MaxMoons) continue;

                double cycle = days / Math.Max(moon.PeriodDays, 0.01f) + moon.PhaseOffset;
                float phase = (float)(cycle - Math.Floor(cycle));
                float hour = TimeOfDay - phase * 24f;
                if (hour < 0f) hour += 24f;
                Vector3 dir = ComputeSunDirection(hour, moon.Azimuth, moon.MaxElevation, out _, out _);

                // The part of the disc that is lit, seen from here: the sun lights the half facing it
                float lit = 0.5f * (1f - Vector3.Dot(dir, SunDirection));

                int i = _moonCount++;
                _moonLights[i] = new MoonLight
                {
                    Direction = dir,
                    Top = Tint(moon.LightColor) * (moon.LightIntensity * lit),
                };

                // The disc gives the moon's light: its luminance where the sun shines straight on (a Lambert sphere's
                // disc averages 2/3 of that when full) times its solid angle is LightIntensity
                float radius = moon.Size * 0.5f * (MathF.PI / 180f);
                float fullLuminance = moon.LightIntensity / (MathF.PI * radius * radius * (2f / 3f));
                _moonDirSize[i] = new Vector4(dir, radius);
                _moonColor[i] = new Vector4(moon.Tint.R, moon.Tint.G, moon.Tint.B, 0f) * (moon.Brightness * fullLuminance);
                textures[i] = (int)(moon.Texture?.BindlessIndex ?? 0u);
            }
            _moonTextures = new Int4(textures[0], textures[1], textures[2], textures[3]);
        }

        /// <summary>
        /// The one directional light shines from the sun or from the moon that gives the most light. It changes source
        /// only when another gives 10% more, fading out first, so the shadows do not swing round while they show.
        /// </summary>
        private void ChooseLightSource(float sunStrength)
        {
            float Strength(int source) => source < 0 ? sunStrength : Luminance(_moonLights[source].AtCamera);

            int best = -1;
            for (int i = 0; i < _moonCount; i++)
                if (Strength(i) > Strength(best))
                    best = i;

            if (!_lightStarted || _lightSource >= _moonCount)
            {
                // A scene starts (or the moon list changed) with the light where it belongs, not fading
                _lightStarted = true;
                _lightSource = best;
                _lightFade = 1f;
                return;
            }

            int target = Strength(best) > Strength(_lightSource) * 1.1f ? best : _lightSource;
            // A source that gives next to no light any more (a set sun, a moon gone under) needs no fading out
            if (target != _lightSource && Strength(_lightSource) * _lightFade <= Strength(target) * 0.05f)
                _lightFade = 0f;
            if (target != _lightSource && _lightFade <= 0f)
                _lightSource = target;

            float step = Math.Clamp((float)Time.Delta, 0f, 0.1f) / LightHandoverSeconds;
            _lightFade = target == _lightSource ? MathF.Min(1f, _lightFade + step) : MathF.Max(0f, _lightFade - step);
        }

        private static float Luminance(Vector3 c) => 0.2126f * c.X + 0.7152f * c.Y + 0.0722f * c.Z;

        private static Vector3 V(Color3 c) => new Vector3(c.R, c.G, c.B);

        /// <summary>A colour as a tint: scaled to luminance 1, so it changes the hue of a light, not its brightness.</summary>
        private static Vector3 Tint(Color3 c)
        {
            float l = Luminance(V(c));
            return l > 1e-6f ? V(c) / l : Vector3.One;
        }

        /// <summary>
        /// The stars turn round the pole of the sun's path once a day, the way the sun crosses the sky; at 06:00 the
        /// star sphere lines up with the world. The Milky Way's band crosses the zenith at midnight, tilted off the
        /// meridian, its bright core GalaxyCoreFromZenith away from it.
        /// </summary>
        private void UpdateStarSphere()
        {
            Vector3 rise = ComputeSunDirection(6f, SunAzimuthAngle, MaxSunElevation, out _, out _);
            Vector3 noon = ComputeSunDirection(12f, SunAzimuthAngle, MaxSunElevation, out _, out _);
            Vector3 pole = Vector3.Normalize(Vector3.Cross(rise, noon));
            if (Vector3.Dot(Vector3.TransformNormal(rise, Matrix4x4.CreateFromAxisAngle(pole, MathF.PI / 2f)), noon) < 0f)
                pole = -pole;

            static float Turn(float hours) => (hours - 6f) / 24f * MathF.Tau;
            _starRotation = Matrix4x4.CreateFromAxisAngle(pole, -Turn(TimeOfDay));

            Vector3 zenith = Vector3.TransformNormal(Vector3.UnitY, Matrix4x4.CreateFromAxisAngle(pole, -Turn(24f)));
            Vector3 meridian = Vector3.Cross(zenith, pole);     // normal of the plane through the pole and that zenith
            meridian = meridian.LengthSquared() > 1e-6f ? Vector3.Normalize(meridian) : Vector3.UnitX;
            _galaxyPole = Vector3.Normalize(Vector3.TransformNormal(meridian, Matrix4x4.CreateFromAxisAngle(zenith, GalaxyTilt)));
            Vector3 along = Vector3.Normalize(Vector3.Cross(_galaxyPole, zenith));
            _galaxyCenter = Vector3.Normalize(zenith * MathF.Cos(GalaxyCoreFromZenith) + along * MathF.Sin(GalaxyCoreFromZenith));
        }

        /// <summary>
        /// Overwrite every look parameter (sun, atmosphere, fog, clouds, stars) from a preset.
        /// Time of day, azimuth and max elevation are left alone — they describe *when*,
        /// the preset describes *how it looks*. Called every frame by EnvironmentController
        /// with its blended preset.
        /// </summary>
        public void ApplyPreset(EnvironmentPreset p)
        {
            if (p == null) return;

            SunIlluminance = p.SunIlluminance;
            SunDayColor = p.SunDayColor;

            RayleighScale = p.RayleighScale;
            MieScale = p.MieScale;
            MieAnisotropy = p.MieAnisotropy;
            OzoneScale = p.OzoneScale;
            GroundAlbedo = p.GroundAlbedo;

            FogDensity = p.FogDensity;
            FogHeight = p.FogHeight;
            FogColor = p.FogColor;

            AirglowLuminance = p.AirglowLuminance;
            StarDensity = p.StarDensity;
            StarBrightness = p.StarBrightness;
            MilkyWayBrightness = p.MilkyWayBrightness;

            CloudCoverage = p.CloudCoverage;
            CloudSpeed = p.CloudSpeed;
            CloudAltitude = p.CloudAltitude;
            CloudAlbedo = p.CloudAlbedo;
            CloudShadowStrength = p.CloudShadowStrength;
        }

        /// <summary>Capture the current look into a preset (inverse of ApplyPreset) — "save what I tweaked".</summary>
        public void CaptureToPreset(EnvironmentPreset p)
        {
            if (p == null) return;

            p.SunIlluminance = SunIlluminance;
            p.SunDayColor = SunDayColor;

            p.RayleighScale = RayleighScale;
            p.MieScale = MieScale;
            p.MieAnisotropy = MieAnisotropy;
            p.OzoneScale = OzoneScale;
            p.GroundAlbedo = GroundAlbedo;

            p.FogDensity = FogDensity;
            p.FogHeight = FogHeight;
            p.FogColor = FogColor;

            p.AirglowLuminance = AirglowLuminance;
            p.StarDensity = StarDensity;
            p.StarBrightness = StarBrightness;
            p.MilkyWayBrightness = MilkyWayBrightness;

            p.CloudCoverage = CloudCoverage;
            p.CloudSpeed = CloudSpeed;
            p.CloudAltitude = CloudAltitude;
            p.CloudAlbedo = CloudAlbedo;
            p.CloudShadowStrength = CloudShadowStrength;
        }

        public void Draw()
        {
            if (Material == null || Mesh == null) return;

            var slot = Entity.Transform.TransformSlot;

            // The sky's light, the sun and the fog come from the atmosphere (SceneConstants.AtmosphereIdx); the cloud
            // layer's shape reaches the shader through SceneConstants too (Camera.SetShaderParams), since the sun light
            // reads the same values for cloud shadows. What is left is the sky dome's own.
            Material.SetParameter("World", Entity.Transform.WorldMatrix);
            Material.SetParameter("StarDensity", StarDensity);
            Material.SetParameter("StarBrightness", StarBrightness);
            Material.SetParameter("MilkyWayBrightness", MilkyWayBrightness);
            Material.SetParameter("MoonDirSize", _moonDirSize);
            Material.SetParameter("MoonColor", _moonColor);
            Material.SetParameter("MoonTextures", _moonTextures);
            Material.SetParameter("MoonCount", (uint)_moonCount);
            Material.SetParameter("StarRotation", _starRotation);
            Material.SetParameter("GalaxyPole", _galaxyPole);
            Material.SetParameter("GalaxyCenter", _galaxyCenter);

            CommandBuffer.Enqueue(Mesh, Material, Params, slot);
        }
    }
}
