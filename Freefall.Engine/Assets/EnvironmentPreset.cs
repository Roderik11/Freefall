using System;
using System.Numerics;
using Vortice.Mathematics;
using Category = System.ComponentModel.CategoryAttribute;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Assets
{
    /// <summary>
    /// A complete look for the sky, lighting, fog and weather particles.
    /// Every field is a plain float or color so presets can be linearly blended
    /// by <see cref="Lerp"/> — that is why precipitation is stored as one intensity
    /// per type instead of an enum (rain can fade out while snow fades in).
    ///
    /// Parameters are organized into Day / Sunset / Night groups: the sun elevation
    /// (driven by <c>SkyboxRenderer.TimeOfDay</c>) blends between those groups at
    /// runtime, so a preset describes the full 24h cycle, not a single moment.
    ///
    /// Presets are applied by <c>EnvironmentController</c>, which pushes the blended
    /// result into <c>SkyboxRenderer</c>, the fog settings and the weather emitters.
    /// </summary>
    [CreateAsset("Environment Preset")]
    public class EnvironmentPreset : Asset
    {
        // ── Sun ──

        [Category("Sun")]
        [Description("Sun brightness multiplier applied on top of the day/night intensity")]
        [ValueRange(0f, 10f)]
        public float SunIntensity = 1.2f;

        [Description("Directional light intensity when the sun is at its highest")]
        [ValueRange(0f, 10f)]
        public float DayIntensity = 3.14159f;

        [Description("Directional light intensity during sunset/sunrise")]
        [ValueRange(0f, 10f)]
        public float SunsetIntensity = 1.5f;

        [Description("Directional light (moon) intensity at night")]
        [ValueRange(0f, 10f)]
        public float NightIntensity = 0.1f;

        public Color3 SunDayColor = new Color3(1.0f, 0.95f, 0.85f);

        public Color3 SunSunsetColor = new Color3(1.0f, 0.7f, 0.4f);

        public Color3 SunNightColor = new Color3(0.1f, 0.15f, 0.3f);

        // ── Day sky ──

        [Category("Day Sky")]
        [Description("Overall daytime sky color multiplier")]
        public Color3 SkyTintColor = new Color3(0.5f, 0.7f, 1.0f);

        [Description("Global atmosphere thickness")]
        [ValueRange(0.5f, 3f)]
        public float AtmosphereDensity = 1.0f;

        [Description("Mie haze/glow strength (sun halo)")]
        [ValueRange(0f, 1f)]
        public float MieScattering = 0.02f;

        [Description("Henyey-Greenstein anisotropy — higher = tighter sun glow")]
        [ValueRange(0f, 0.99f)]
        public float MieAnisotropy = 0.76f;

        [Description("Colored haze at the horizon")]
        public Color3 HazeColor = new Color3(0.8f, 0.85f, 0.9f);

        [ValueRange(0f, 2f)]
        public float HazeIntensity = 0.3f;

        [Description("How high the haze reaches (viewDir.y)")]
        [ValueRange(0f, 1f)]
        public float HazeHeight = 0.15f;

        // ── Sunset ──

        [Category("Sunset")]
        [Description("Warm color near the horizon at sunset/sunrise")]
        public Color3 SunsetTintColor = new Color3(1.0f, 0.5f, 0.2f);

        [ValueRange(0f, 2f)]
        public float SunsetTintIntensity = 0.8f;

        // ── Night sky ──

        [Category("Night Sky")]
        [Description("Zenith color at night")]
        public Color3 NightSkyColor = new Color3(0.01f, 0.01f, 0.04f);

        [Description("Horizon glow at night")]
        public Color3 NightHorizonColor = new Color3(0.03f, 0.04f, 0.08f);

        [ValueRange(0f, 1f)]
        public float StarDensity = 0.5f;

        [ValueRange(0f, 10f)]
        public float StarBrightness = 1.0f;

        // ── Clouds ──

        [Category("Clouds")]
        [ValueRange(0f, 1f)]
        public float CloudCoverage = 0.5f;

        [ValueRange(0f, 10f)]
        public float CloudSpeed = 1.0f;

        [Description("Cloud layer height in world units")]
        [ValueRange(500f, 5000f)]
        public float CloudAltitude = 1800.0f;

        [Description("Daytime cloud brightness")]
        [ValueRange(0f, 3f)]
        public float CloudBrightness = 1.0f;

        [Description("Daytime color of sun-facing cloud tops")]
        public Color3 CloudSunlitColor = new Color3(1.0f, 0.98f, 0.95f);

        [Description("Daytime color of the cloud shade side")]
        public Color3 CloudShadowColor = new Color3(0.35f, 0.4f, 0.55f);

        [Description("Warm tint mixed into clouds at sunset/sunrise")]
        public Color3 CloudSunsetTintColor = new Color3(1.0f, 0.6f, 0.3f);

        [Description("Moonlit cloud color at night")]
        public Color3 CloudNightColor = new Color3(0.08f, 0.09f, 0.14f);

        [Description("Night-time cloud brightness")]
        [ValueRange(0f, 3f)]
        public float CloudNightBrightness = 1.0f;

        [Description("How much sunlight the clouds block on the ground (0 = no cloud shadows)")]
        [ValueRange(0f, 1f)]
        public float CloudShadowStrength = 0.8f;

        // ── Fog ──

        [Category("Fog")]
        [Description("Exponential-squared fog density (see Engine.Settings.FogDensity)")]
        [ValueRange(0f, 0.01f, 0.0001f)]
        public float FogDensity = 0.0005f;

        // ── Weather ──

        [Category("Weather")]
        [Description("Rain emitter strength, 0 = off, 1 = the emitter's configured EmitRate")]
        [ValueRange(0f, 2f)]
        public float RainIntensity = 0f;

        [ValueRange(0f, 2f)]
        public float SnowIntensity = 0f;

        [ValueRange(0f, 2f)]
        public float HailIntensity = 0f;

        [ValueRange(0f, 2f)]
        public float DustIntensity = 0f;

        [Description("Compass heading the wind blows toward, degrees (0=+X, 90=+Z)")]
        [ValueRange(0f, 360f)]
        public float WindDirection = 0f;

        [Description("Wind speed in m/s, applied to weather emitters and cloud drift")]
        [ValueRange(0f, 50f)]
        public float WindStrength = 0f;

        /// <summary>Wind as a world-space velocity vector.</summary>
        public Vector3 WindVector
        {
            get
            {
                float rad = WindDirection * (MathF.PI / 180f);
                return new Vector3(MathF.Cos(rad), 0f, MathF.Sin(rad)) * WindStrength;
            }
        }

        // ── Blending ──

        private static float L(float a, float b, float t) => a + (b - a) * t;

        private static Color3 L(Color3 a, Color3 b, float t)
            => new Color3(L(a.R, b.R, t), L(a.G, b.G, t), L(a.B, b.B, t));

        /// <summary>Copy every parameter (not the asset identity) from another preset.</summary>
        public void CopyFrom(EnvironmentPreset src) => Lerp(src, src, 0f, this);

        /// <summary>
        /// Linearly blend <paramref name="a"/> and <paramref name="b"/> into <paramref name="dest"/>.
        /// Wind direction is blended through its vector so 350° → 10° does not swing through 180°.
        /// </summary>
        public static void Lerp(EnvironmentPreset a, EnvironmentPreset b, float t, EnvironmentPreset dest)
        {
            t = Math.Clamp(t, 0f, 1f);

            dest.SunIntensity = L(a.SunIntensity, b.SunIntensity, t);
            dest.DayIntensity = L(a.DayIntensity, b.DayIntensity, t);
            dest.SunsetIntensity = L(a.SunsetIntensity, b.SunsetIntensity, t);
            dest.NightIntensity = L(a.NightIntensity, b.NightIntensity, t);
            dest.SunDayColor = L(a.SunDayColor, b.SunDayColor, t);
            dest.SunSunsetColor = L(a.SunSunsetColor, b.SunSunsetColor, t);
            dest.SunNightColor = L(a.SunNightColor, b.SunNightColor, t);

            dest.SkyTintColor = L(a.SkyTintColor, b.SkyTintColor, t);
            dest.AtmosphereDensity = L(a.AtmosphereDensity, b.AtmosphereDensity, t);
            dest.MieScattering = L(a.MieScattering, b.MieScattering, t);
            dest.MieAnisotropy = L(a.MieAnisotropy, b.MieAnisotropy, t);
            dest.HazeColor = L(a.HazeColor, b.HazeColor, t);
            dest.HazeIntensity = L(a.HazeIntensity, b.HazeIntensity, t);
            dest.HazeHeight = L(a.HazeHeight, b.HazeHeight, t);

            dest.SunsetTintColor = L(a.SunsetTintColor, b.SunsetTintColor, t);
            dest.SunsetTintIntensity = L(a.SunsetTintIntensity, b.SunsetTintIntensity, t);

            dest.NightSkyColor = L(a.NightSkyColor, b.NightSkyColor, t);
            dest.NightHorizonColor = L(a.NightHorizonColor, b.NightHorizonColor, t);
            dest.StarDensity = L(a.StarDensity, b.StarDensity, t);
            dest.StarBrightness = L(a.StarBrightness, b.StarBrightness, t);

            dest.CloudCoverage = L(a.CloudCoverage, b.CloudCoverage, t);
            dest.CloudSpeed = L(a.CloudSpeed, b.CloudSpeed, t);
            dest.CloudAltitude = L(a.CloudAltitude, b.CloudAltitude, t);
            dest.CloudBrightness = L(a.CloudBrightness, b.CloudBrightness, t);
            dest.CloudSunlitColor = L(a.CloudSunlitColor, b.CloudSunlitColor, t);
            dest.CloudShadowColor = L(a.CloudShadowColor, b.CloudShadowColor, t);
            dest.CloudSunsetTintColor = L(a.CloudSunsetTintColor, b.CloudSunsetTintColor, t);
            dest.CloudNightColor = L(a.CloudNightColor, b.CloudNightColor, t);
            dest.CloudNightBrightness = L(a.CloudNightBrightness, b.CloudNightBrightness, t);
            dest.CloudShadowStrength = L(a.CloudShadowStrength, b.CloudShadowStrength, t);

            dest.FogDensity = L(a.FogDensity, b.FogDensity, t);

            dest.RainIntensity = L(a.RainIntensity, b.RainIntensity, t);
            dest.SnowIntensity = L(a.SnowIntensity, b.SnowIntensity, t);
            dest.HailIntensity = L(a.HailIntensity, b.HailIntensity, t);
            dest.DustIntensity = L(a.DustIntensity, b.DustIntensity, t);

            var wind = Vector3.Lerp(a.WindVector, b.WindVector, t);
            dest.WindStrength = wind.Length();
            dest.WindDirection = dest.WindStrength > 1e-4f
                ? (MathF.Atan2(wind.Z, wind.X) * (180f / MathF.PI) + 360f) % 360f
                : L(a.WindDirection, b.WindDirection, t);
        }
    }
}
