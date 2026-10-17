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
    /// A preset says what the air is made of and what lights it; the time of day (SkyboxRenderer.TimeOfDay) does the
    /// rest through the physical atmosphere (Graphics.Atmosphere): the same preset gives the noon, the sunset and the
    /// night, with nothing per time of day.
    ///
    /// Presets are applied by <c>EnvironmentController</c>, which pushes the blended
    /// result into <c>SkyboxRenderer</c> and the weather emitters.
    /// </summary>
    [CreateAsset("Environment Preset")]
    public class EnvironmentPreset : Asset
    {
        // ── Sun ──

        [Category("Sun")]
        [Description("Sunlight at the top of the atmosphere in lux (Earth: about 120000); the air takes its share on the way down")]
        [ValueRange(0f, 200000f)]
        public float SunIlluminance = 120000f;

        [Description("The sun's own tint. Its reddening at sunset comes from the air")]
        public Color3 SunDayColor = new Color3(1.0f, 0.95f, 0.85f);

        // ── Atmosphere ──

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

        [Description("Ozone (1 = Earth): keeps the twilight sky blue overhead")]
        [ValueRange(0f, 4f)]
        public float OzoneScale = 1f;

        [Description("Average albedo of the ground: the light it bounces into the sky and onto surfaces facing down")]
        public Color3 GroundAlbedo = new Color3(0.11f, 0.10f, 0.085f);

        // ── Ground fog ──

        [Category("Ground Fog")]
        [Description("Extinction of the ground fog per metre at sea level: 0.0001 = about 40 km visibility, 0.001 = 4 km, 0 = none")]
        [ValueRange(0f, 0.01f, 0.00001f)]
        public float FogDensity = 0.0001f;

        [Description("Metres of height over which the fog thins to a third")]
        [ValueRange(10f, 3000f)]
        public float FogHeight = 400f;

        [Description("The fog's single-scattering albedo: near white for mist, warm for dust, dark for smoke")]
        public Color3 FogColor = new Color3(0.9f, 0.9f, 0.9f);

        // ── Night sky ──

        [Category("Night Sky")]
        [Description("Glow of the upper air on a moonless night, cd/m² at the zenith (Earth: about 0.0002)")]
        [ValueRange(0f, 0.01f, 0.00001f)]
        public float AirglowLuminance = 2e-4f;

        [ValueRange(0f, 1f)]
        public float StarDensity = 0.5f;

        [ValueRange(0f, 10f)]
        public float StarBrightness = 1.0f;

        [Description("Glow of the Milky Way band (it turns with the stars; a moon's light in the air washes it out)")]
        [ValueRange(0f, 4f)]
        public float MilkyWayBrightness = 1.0f;

        // ── Clouds ──

        [Category("Clouds")]
        [ValueRange(0f, 1f)]
        public float CloudCoverage = 0.5f;

        [ValueRange(0f, 10f)]
        public float CloudSpeed = 1.0f;

        [Description("Cloud layer height in world units")]
        [ValueRange(500f, 5000f)]
        public float CloudAltitude = 1800.0f;

        [Description("Share of the light a cloud sends back: lower = darker clouds (a storm)")]
        [ValueRange(0f, 1f)]
        public float CloudAlbedo = 0.8f;

        [Description("How much sunlight the clouds block on the ground (0 = no cloud shadows)")]
        [ValueRange(0f, 1f)]
        public float CloudShadowStrength = 0.8f;

        // ── Exposure ──

        [Category("Exposure")]
        [Description("EV added to the auto exposure's metering (+ = brighter): how bright this weather reads")]
        [ValueRange(-4f, 4f)]
        public float ExposureCompensation = 0f;

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

            dest.SunIlluminance = L(a.SunIlluminance, b.SunIlluminance, t);
            dest.SunDayColor = L(a.SunDayColor, b.SunDayColor, t);

            dest.RayleighScale = L(a.RayleighScale, b.RayleighScale, t);
            dest.MieScale = L(a.MieScale, b.MieScale, t);
            dest.MieAnisotropy = L(a.MieAnisotropy, b.MieAnisotropy, t);
            dest.OzoneScale = L(a.OzoneScale, b.OzoneScale, t);
            dest.GroundAlbedo = L(a.GroundAlbedo, b.GroundAlbedo, t);

            dest.FogDensity = L(a.FogDensity, b.FogDensity, t);
            dest.FogHeight = L(a.FogHeight, b.FogHeight, t);
            dest.FogColor = L(a.FogColor, b.FogColor, t);

            dest.AirglowLuminance = L(a.AirglowLuminance, b.AirglowLuminance, t);
            dest.StarDensity = L(a.StarDensity, b.StarDensity, t);
            dest.StarBrightness = L(a.StarBrightness, b.StarBrightness, t);
            dest.MilkyWayBrightness = L(a.MilkyWayBrightness, b.MilkyWayBrightness, t);

            dest.CloudCoverage = L(a.CloudCoverage, b.CloudCoverage, t);
            dest.CloudSpeed = L(a.CloudSpeed, b.CloudSpeed, t);
            dest.CloudAltitude = L(a.CloudAltitude, b.CloudAltitude, t);
            dest.CloudAlbedo = L(a.CloudAlbedo, b.CloudAlbedo, t);
            dest.CloudShadowStrength = L(a.CloudShadowStrength, b.CloudShadowStrength, t);

            dest.ExposureCompensation = L(a.ExposureCompensation, b.ExposureCompensation, t);

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
