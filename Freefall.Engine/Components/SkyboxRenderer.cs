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

        public int Day = 0;                   // days since day 0: counts on as TimeOfDay wraps, moves the moons along their orbits

        [ValueRange(0.001f, 0.1f)]
        public float TimeOfDaySpeed = 0.001f; // Hours per second (tweak for testing)

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

        public Color3 SunDayColor = new Color3(1.0f, 0.95f, 0.85f);
        public Color3 SunSunsetColor = new Color3(1.0f, 0.7f, 0.4f);

        public Vector3 SunDirection = new Vector3(0, 1, 0);

        // The world's moons. Their light comes through the one directional light (SunLight): at night it turns to the
        // moon that gives the most light. The sky draws at most MaxMoons of them.
        [Category("Moons")]
        public List<Moon> Moons = [];

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
        public Color3 NightSkyColor = new Color3(0.004f, 0.0065f, 0.018f);  // Zenith color at night
        public Color3 NightHorizonColor = new Color3(0.02f, 0.027f, 0.045f); // Horizon glow at night

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

        [ValueRange(0f, 3f)]
        public float CloudBrightness = 1.0f;        // Cloud brightness (day)

        public Color3 CloudShadowColor = new Color3(0.35f, 0.4f, 0.55f);   // Cloud shade side (day)
        public Color3 CloudSunlitColor = new Color3(1.0f, 0.98f, 0.95f);   // Sun-facing cloud tops (day)
        public Color3 CloudSunsetTintColor = new Color3(1.0f, 0.6f, 0.3f); // Warm tint mixed in at sunset/sunrise
        public Color3 CloudNightColor = new Color3(0.08f, 0.09f, 0.14f);   // Moonlit clouds at night

        [ValueRange(0f, 3f)]
        public float CloudNightBrightness = 1.0f;   // Cloud brightness (night)

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


        // Light the ground sends back up onto surfaces facing down, as a multiple of their albedo: the hemisphere
        // ambient's lower half (sky_common.fx HemisphereAmbient). The directional light on flat ground (after the
        // clouds' average shade) plus the sky's ambient, times GroundAlbedo. Without a sky: a fixed warm grey.
        private static readonly Vector3 NoSkyGroundBounce = new Vector3(0.12f, 0.11f, 0.10f);
        public static Vector3 CurrentGroundBounce { get; private set; } = NoSkyGroundBounce;

        // Average albedo of what lies around (earth, cobbles, grass), a part of it in shade
        private static readonly Vector3 GroundAlbedo = new Vector3(0.11f, 0.10f, 0.085f);

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
        public static Vector3 CurrentNightSkyColor { get; private set; } = new Vector3(0.004f, 0.0065f, 0.018f);
        public static Vector3 CurrentNightHorizonColor { get; private set; } = new Vector3(0.02f, 0.027f, 0.045f);

        // Cloud layer accessors for Camera.SetShaderParams() (SceneConstants: sky dome + cloud shadows)
        public static float CurrentCloudCoverage { get; private set; } = 0.5f;
        public static float CurrentCloudTime { get; private set; } = 0.0f;
        public static float CurrentCloudAltitude { get; private set; } = 1800.0f;
        public static float CurrentCloudAltitudeFrom { get; private set; } = 1800.0f;
        public static float CurrentCloudAltitudeBlend { get; private set; } = 1.0f;
        public static float CurrentCloudShadowStrength { get; private set; } = 0.0f;   // 0 until a skybox exists
        public static float CurrentCloudShadowScale { get; private set; } = 1.0f;
        public static uint CurrentCloudNoiseLUTIdx => _cloudNoiseLUT?.BindlessIndex ?? 0u;

        // Average cloud color for the current time of day (flat-shaded version of the mesh_skybox.fx
        // cloud palettes). Used where clouds are reflected rather than drawn, e.g. the ocean.
        public static Vector3 CurrentCloudColor { get; private set; } = new Vector3(0.7f, 0.72f, 0.78f);

        // Day/sunset/night weights of the current frame (normalized, sum to 1). Useful for gameplay/audio.
        public static float CurrentDayFactor { get; private set; } = 1.0f;
        public static float CurrentSunsetFactor { get; private set; } = 0.0f;
        public static float CurrentNightFactor { get; private set; } = 0.0f;

        private float CloudTime = 0.0f;

        /// <summary>Moons the sky draws (mesh_skybox.fx MoonDirSize etc. are arrays of this size).</summary>
        public const int MaxMoons = 4;

        // The one directional light fades out, turns to its new source (sun ↔ moon) and fades in again, so its
        // shadows never jump to a new direction while they show
        private const float LightHandoverSeconds = 0.75f;

        private struct MoonLight
        {
            public Vector3 Direction;   // towards the moon
            public float Intensity;     // its LightIntensity × the lit part of the disc × above the horizon
            public Color3 Color;
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
            CurrentGroundBounce = NoSkyGroundBounce;
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

            UpdateSunLight();
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

        private void UpdateSunLight()
        {
            if (SunLight == null) return;

            SunDirection = ComputeSunDirection(TimeOfDay, SunAzimuthAngle, MaxSunElevation, out Vector3 horizontal, out float elevation);
            CurrentSunDirection = SunDirection;

            // ── Three-state blend matching sky_common.fx GetSkyColor ──
            ComputeDayNightFactors(elevation, out float dayFactor, out float sunsetFactor, out float nightFactor);
            CurrentDayFactor = dayFactor;
            CurrentSunsetFactor = sunsetFactor;
            CurrentNightFactor = nightFactor;

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

            // The sun's light: the day and sunset palettes (matching the shader); the night belongs to the moons.
            // After sunset it keeps shining from 6° up (twilight) while it fades, never from below the ground.
            float sunWeight = dayFactor + sunsetFactor;
            float sunIntensity = (dayFactor * DayIntensity + sunsetFactor * SunsetIntensity) * SunIntensity;
            float sunsetShare = sunWeight > 0f ? sunsetFactor / sunWeight : 1f;
            var sunColor = new Color3(
                SunDayColor.R + (SunSunsetColor.R - SunDayColor.R) * sunsetShare,
                SunDayColor.G + (SunSunsetColor.G - SunDayColor.G) * sunsetShare,
                SunDayColor.B + (SunSunsetColor.B - SunDayColor.B) * sunsetShare);
            Vector3 sunLightDir = Vector3.Normalize(horizontal + Vector3.UnitY * MathF.Max(0.1f, elevation));

            UpdateMoons();
            ChooseLightSource(sunIntensity * Luminance(sunColor));

            Vector3 lightDir = sunLightDir;
            float lightIntensity = sunIntensity;
            Color3 lightColor = sunColor;
            if (_lightSource >= 0)
            {
                ref var moon = ref _moonLights[_lightSource];
                lightDir = moon.Direction;
                lightIntensity = moon.Intensity;
                lightColor = moon.Color;
            }

            // Transform.Forward is what DirectionalLight uses as LightDirection; CreateWorld's forward becomes -Z, so
            // passing the direction towards the source leaves Forward pointing from it towards the ground
            Vector3 up = MathF.Abs(lightDir.Y) > 0.999f ? Vector3.UnitZ : Vector3.UnitY;
            SunLight.Transform.Rotation = Quaternion.CreateFromRotationMatrix(Matrix4x4.CreateWorld(Vector3.Zero, lightDir, up));
            SunLight.Intensity = lightIntensity * _lightFade;
            SunLight.Color = lightColor;

            float clouds = Math.Clamp(1f - CurrentCloudShadowStrength * CurrentCloudCoverage, 0f, 1f);
            float onGround = SunLight.Enabled ? SunLight.Intensity * MathF.Max(0f, lightDir.Y) * clouds : 0f;
            CurrentGroundBounce = GroundAlbedo * (new Vector3(lightColor.R, lightColor.G, lightColor.B) * onGround + Daylight.SkyAmbientColor());

            // Flat-shaded cloud color: mid shadow/sunlit gradient of each palette in mesh_skybox.fx
            static Vector3 V(Color3 c) => new Vector3(c.R, c.G, c.B);
            float dayBrightness = CloudBrightness * 1.3f;
            Vector3 dayCloud = Vector3.Lerp(V(CloudShadowColor), V(CloudSunlitColor), 0.5f) * dayBrightness;
            Vector3 sunsetCloud = Vector3.Lerp(dayCloud, V(CloudSunsetTintColor) * dayBrightness, 0.5f);
            Vector3 nightCloud = V(CloudNightColor) * 0.75f * CloudNightBrightness;
            CurrentCloudColor = dayCloud * dayFactor + sunsetCloud * sunsetFactor + nightCloud * nightFactor;
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
                Vector3 dir = ComputeSunDirection(hour, moon.Azimuth, moon.MaxElevation, out _, out float moonElevation);

                // The part of the disc that is lit, seen from here: the sun lights the half facing it
                float lit = 0.5f * (1f - Vector3.Dot(dir, SunDirection));
                float t = Math.Clamp(moonElevation / 0.08f, 0f, 1f);
                float aboveHorizon = t * t * (3f - 2f * t);

                int i = _moonCount++;
                _moonLights[i] = new MoonLight
                {
                    Direction = dir,
                    Intensity = moon.LightIntensity * lit * aboveHorizon,
                    Color = moon.LightColor,
                };
                _moonDirSize[i] = new Vector4(dir, moon.Size * 0.5f * (MathF.PI / 180f));
                _moonColor[i] = new Vector4(moon.Tint.R, moon.Tint.G, moon.Tint.B, 0f) * moon.Brightness;
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
            float Strength(int source) => source < 0
                ? sunStrength
                : _moonLights[source].Intensity * Luminance(_moonLights[source].Color);

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

        private static float Luminance(Color3 c) => 0.2126f * c.R + 0.7152f * c.G + 0.0722f * c.B;

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
            SunDayColor = p.SunDayColor;
            SunSunsetColor = p.SunSunsetColor;

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
            MilkyWayBrightness = p.MilkyWayBrightness;

            CloudCoverage = p.CloudCoverage;
            CloudSpeed = p.CloudSpeed;
            CloudAltitude = p.CloudAltitude;
            CloudBrightness = p.CloudBrightness;
            CloudSunlitColor = p.CloudSunlitColor;
            CloudShadowColor = p.CloudShadowColor;
            CloudSunsetTintColor = p.CloudSunsetTintColor;
            CloudNightColor = p.CloudNightColor;
            CloudNightBrightness = p.CloudNightBrightness;
            CloudShadowStrength = p.CloudShadowStrength;
        }

        /// <summary>Capture the current look into a preset (inverse of ApplyPreset) — "save what I tweaked".</summary>
        public void CaptureToPreset(EnvironmentPreset p)
        {
            if (p == null) return;

            p.SunIntensity = SunIntensity;
            p.DayIntensity = DayIntensity;
            p.SunsetIntensity = SunsetIntensity;
            p.SunDayColor = SunDayColor;
            p.SunSunsetColor = SunSunsetColor;

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
            p.MilkyWayBrightness = MilkyWayBrightness;

            p.CloudCoverage = CloudCoverage;
            p.CloudSpeed = CloudSpeed;
            p.CloudAltitude = CloudAltitude;
            p.CloudBrightness = CloudBrightness;
            p.CloudSunlitColor = CloudSunlitColor;
            p.CloudShadowColor = CloudShadowColor;
            p.CloudSunsetTintColor = CloudSunsetTintColor;
            p.CloudNightColor = CloudNightColor;
            p.CloudNightBrightness = CloudNightBrightness;
            p.CloudShadowStrength = CloudShadowStrength;
        }

        public void Draw()
        {
            if (Material == null || Mesh == null) return;

            var slot = Entity.Transform.TransformSlot;

            // Set sky parameters on the MaterialBlock
            Material.SetParameter("World", Entity.Transform.WorldMatrix);
            Material.SetParameter("SunDirection", SunDirection);
            Material.SetParameter("TimeOfDay", TimeOfDay);
            // Cloud layer shape (coverage, time, altitude, noise LUT) reaches the shader through SceneConstants,
            // see Camera.SetShaderParams — the sun light reads the same values for cloud shadows.
            Material.SetParameter("SunIntensity", SunIntensity);
            Material.SetParameter("StarDensity", StarDensity);
            Material.SetParameter("StarBrightness", StarBrightness);
            Material.SetParameter("CloudBrightness", CloudBrightness);
            Material.SetParameter("CloudShadowColor", new Vector3(CloudShadowColor.R, CloudShadowColor.G, CloudShadowColor.B));
            Material.SetParameter("CloudSunlitColor", new Vector3(CloudSunlitColor.R, CloudSunlitColor.G, CloudSunlitColor.B));
            Material.SetParameter("CloudSunsetTintColor", new Vector3(CloudSunsetTintColor.R, CloudSunsetTintColor.G, CloudSunsetTintColor.B));
            Material.SetParameter("CloudNightBrightness", CloudNightBrightness);
            Material.SetParameter("CloudNightColor", new Vector3(CloudNightColor.R, CloudNightColor.G, CloudNightColor.B));
            Material.SetParameter("MoonDirSize", _moonDirSize);
            Material.SetParameter("MoonColor", _moonColor);
            Material.SetParameter("MoonTextures", _moonTextures);
            Material.SetParameter("MoonCount", (uint)_moonCount);
            Material.SetParameter("StarRotation", _starRotation);
            Material.SetParameter("GalaxyPole", _galaxyPole);
            Material.SetParameter("GalaxyCenter", _galaxyCenter);
            Material.SetParameter("MilkyWayBrightness", MilkyWayBrightness);

            CommandBuffer.Enqueue(Mesh, Material, Params, slot);
        }
    }
}
