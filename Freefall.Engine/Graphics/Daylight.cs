using System;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// How much daylight reaches the ground, and whether lamps are lit. The point light shadow selection weighs a
    /// light against the daylight (local_shadow_select.hlsl); lights marked PointLight.OnlyWhenDark and materials
    /// marked EmissiveWhenDark follow <see cref="Lamps"/>: lit when the daylight falls below
    /// <see cref="LampsOnBelow"/> (night, a thunderstorm, an overcast dusk), out again above
    /// <see cref="LampsOffAbove"/>, so a cloud passing at the mark does not blink them. Noon daylight with the
    /// presets: Sunny 2.35, Clear Day 1.55, Snowfall 0.6, Overcast 0.41, Thunderstorm 0.14; a clear night 0.06.
    /// A street lantern adds about 0.15 to the ground under it.
    /// </summary>
    public static class Daylight
    {
        public const float LampsOnBelow = 0.25f;
        public const float LampsOffAbove = 0.35f;
        public const float LampFadeSeconds = 1.5f;

        /// <summary>0 = lamps out, 1 = lit; in between while they fade. 1 until a sky updates it.</summary>
        public static float Lamps { get; private set; } = 1f;

        /// <summary>Raised when <see cref="Lamps"/> changes: once per frame while lamps fade, never otherwise.</summary>
        public static event Action? LampsChanged;

        private static bool _lit = true;
        private static bool _started;

        /// <summary>Once per frame, after the sky has set the sun (SkyboxRenderer.Update).</summary>
        public static void Update(float deltaSeconds)
        {
            float level = SunLight() + SkyAmbient();
            if (_lit ? level > LampsOffAbove : level < LampsOnBelow)
                _lit = !_lit;

            float target = _lit ? 1f : 0f;
            if (!_started)
            {
                _started = true;    // a scene loads with its lamps as they should be, not fading
                SetLamps(target);
                return;
            }
            if (Lamps != target)
            {
                float step = Math.Clamp(deltaSeconds, 0f, 0.1f) / LampFadeSeconds;
                SetLamps(target > Lamps ? MathF.Min(target, Lamps + step) : MathF.Max(target, Lamps - step));
            }
        }

        private static void SetLamps(float value)
        {
            if (Lamps == value) return;
            Lamps = value;
            LampsChanged?.Invoke();
        }

        // What the sun adds to a lit surface, as a multiple of its albedo: light_directional_cs's π × intensity × colour
        // through Lambert's 1/π, dimmed by the clouds' average shade (coverage × cloud shadow strength), so overcast
        // days count as dark. 0 without an enabled directional light.
        public static float SunLight()
        {
            foreach (var sun in ComponentCache<DirectionalLight>.All)
            {
                if (!sun.Enabled) continue;
                float clouds = 1f - SkyboxRenderer.CurrentCloudShadowStrength * SkyboxRenderer.CurrentCloudCoverage;
                return sun.Intensity * Luminance(sun.Color.R, sun.Color.G, sun.Color.B) * Math.Clamp(clouds, 0f, 1f);
            }
            return 0f;
        }

        // What the sky's ambient adds to an up-facing surface, as a multiple of its albedo: composition_cs's
        // GetSkyColor(up) × 0.45 × AmbientScale, with the zenith colour as sky_common.fx's day sky has it
        // (tint × ~0.65 × density) and the night sky colour at night. 0 without a sky.
        public static float SkyAmbient()
        {
            if (ComponentCache<SkyboxRenderer>.All.Count == 0)
                return 0f;
            var tint = SkyboxRenderer.CurrentSkyTintColor;
            var night = SkyboxRenderer.CurrentNightSkyColor;
            float day = Luminance(tint.X, tint.Y, tint.Z) * 0.65f * SkyboxRenderer.CurrentAtmosphereDensity;
            float sky = (SkyboxRenderer.CurrentDayFactor + SkyboxRenderer.CurrentSunsetFactor) * day
                      + SkyboxRenderer.CurrentNightFactor * Luminance(night.X, night.Y, night.Z);
            return sky * 0.45f * SkyboxRenderer.AmbientScale;
        }

        private static float Luminance(float r, float g, float b) => 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }
}
