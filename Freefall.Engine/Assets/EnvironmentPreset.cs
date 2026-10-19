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

        [Description("The clouds' base, metres above sea level (the 2D layer's height)")]
        [ValueRange(300f, 5000f)]
        public float CloudAltitude = 1800.0f;

        [Description("Share of the light a droplet scatters rather than absorbs (real clouds ~0.99+); lower darkens a thick " +
                     "cloud, whose light has scattered many times")]
        [ValueRange(0.5f, 1f)]
        public float CloudAlbedo = 0.99f;

        [Description("Volumetric clouds: depth of the ordinary layer above its base, metres")]
        [ValueRange(100f, 4000f)]
        public float CloudThickness = 1000f;

        [Description("Volumetric clouds: 0 stratus (a flat sheet), 0.5 stratocumulus, 1 cumulus (heaped, flat bases)")]
        [ValueRange(0f, 1f)]
        public float CloudType = 1f;

        [Description("Volumetric clouds: share of the storm cells that grow into towers (0 = none, 1 = a sky of them)")]
        [ValueRange(0f, 1f)]
        public float CloudTowers = 0f;

        [Description("Volumetric clouds: the tallest tower's top, metres above sea level; from ~7000 they spread anvils")]
        [ValueRange(1000f, 13000f)]
        public float CloudTowerTop = 6000f;

        [Description("Volumetric clouds: extinction per metre inside a cloud (cumulus 0.05-0.1, stratus 0.02-0.05)")]
        [ValueRange(0.005f, 0.2f)]
        public float CloudDensity = 0.05f;

        [Description("Volumetric clouds: lightning flashes a minute in the raining storm cells near the camera (0 = none)")]
        [ValueRange(0f, 30f)]
        public float LightningRate = 0f;

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

        [Description("Wind speed in m/s near the ground: vegetation, the weather's particles, the sea. Its direction is " +
                     "not set here: it wanders with the time of day (Graphics.Wind), the clouds drifting the same way")]
        [ValueRange(0f, 50f)]
        public float WindStrength = 0f;

        // ── Blending ──

        private static float L(float a, float b, float t) => a + (b - a) * t;

        private static Color3 L(Color3 a, Color3 b, float t)
            => new Color3(L(a.R, b.R, t), L(a.G, b.G, t), L(a.B, b.B, t));

        /// <summary>Copy every parameter (not the asset identity) from another preset.</summary>
        public void CopyFrom(EnvironmentPreset src) => Lerp(src, src, 0f, this);

        /// <summary>Linearly blend <paramref name="a"/> and <paramref name="b"/> into <paramref name="dest"/>.</summary>
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
            dest.CloudThickness = L(a.CloudThickness, b.CloudThickness, t);
            dest.CloudType = L(a.CloudType, b.CloudType, t);
            dest.CloudTowers = L(a.CloudTowers, b.CloudTowers, t);
            dest.CloudTowerTop = L(a.CloudTowerTop, b.CloudTowerTop, t);
            dest.CloudDensity = L(a.CloudDensity, b.CloudDensity, t);
            dest.LightningRate = L(a.LightningRate, b.LightningRate, t);

            dest.ExposureCompensation = L(a.ExposureCompensation, b.ExposureCompensation, t);

            dest.RainIntensity = L(a.RainIntensity, b.RainIntensity, t);
            dest.SnowIntensity = L(a.SnowIntensity, b.SnowIntensity, t);
            dest.HailIntensity = L(a.HailIntensity, b.HailIntensity, t);
            dest.DustIntensity = L(a.DustIntensity, b.DustIntensity, t);

            dest.WindStrength = L(a.WindStrength, b.WindStrength, t);
        }
    }
}
