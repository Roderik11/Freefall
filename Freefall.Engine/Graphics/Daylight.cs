using System;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// How much daylight reaches the ground, and whether lamps are lit. Lights marked PointLight.OnlyWhenDark and
    /// materials marked EmissiveWhenDark follow <see cref="Lamps"/>: lit when the illuminance on level ground
    /// (<see cref="GroundIlluminance"/>, lux: the sun or moon through the air and the clouds' average shade, plus the
    /// sky) falls below <see cref="LampsOnBelowLux"/>, out again above <see cref="LampsOffAboveLux"/>, so a cloud
    /// passing at the mark does not blink them. Clear Day (SunAzimuth 161.2, Day 1): noon ~50000 lux, 18:00 ~940,
    /// 18:20 ~50, 18:35 ~1.5 (lamps on), 18:45 ~0.06, 19:00 ~0.001 (no moon up); a thunderstorm noon is still thousands
    /// of lux, so storms by day do not light the lamps. The point light shadow selection weighs a light's lux against
    /// the daylight's (<see cref="SunIlluminance"/>, <see cref="SkyIlluminance"/>, local_shadow_select.hlsl).
    /// </summary>
    public static class Daylight
    {
        public const float LampsOnBelowLux = 1f;      // ~18:37 on a clear evening (stage 0 lamps: 18:40-18:50)
        public const float LampsOffAboveLux = 2f;
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
            float lux = GroundIlluminance();
            if (_lit ? lux > LampsOffAboveLux : lux < LampsOnBelowLux)
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

        /// <summary>
        /// Illuminance on level open ground in lux: <see cref="SunIlluminance"/> at the light's height plus
        /// <see cref="SkyIlluminance"/>. 0 without a sky.
        /// </summary>
        public static float GroundIlluminance()
        {
            if (ComponentCache<SkyboxRenderer>.All.Count == 0)
                return 0f;
            float height = 0f;
            foreach (var sun in ComponentCache<DirectionalLight>.All)
            {
                if (!sun.Enabled) continue;
                height = MathF.Max(-sun.Transform.Forward.Y, 0f);   // Forward points from the light to the ground
                break;
            }
            return SunIlluminance() * height + SkyIlluminance();
        }

        /// <summary>
        /// Lux on a surface facing the directional light (the sun, or at night the brightest moon) after the air,
        /// through the clouds' average shade (coverage × cloud shadow strength). 0 without an enabled directional light.
        /// </summary>
        public static float SunIlluminance()
        {
            foreach (var sun in ComponentCache<DirectionalLight>.All)
            {
                if (!sun.Enabled) continue;
                // Intensity is lux × ReferenceExposure / π (SkyboxRenderer)
                float normal = sun.Intensity * MathF.PI / Atmosphere.ReferenceExposure * Luminance(sun.Color.R, sun.Color.G, sun.Color.B);
                float clouds = Math.Clamp(1f - SkyboxRenderer.CurrentCloudShadowStrength * SkyboxRenderer.CurrentCloudCoverage, 0f, 1f);
                return normal * clouds;
            }
            return 0f;
        }

        /// <summary>
        /// Lux from the whole sky on level ground (sky_common.fx's SkyAmbientLight × π), read back from the atmosphere's
        /// ambient pass a few frames late. 0 without a sky.
        /// </summary>
        public static float SkyIlluminance()
        {
            if (ComponentCache<SkyboxRenderer>.All.Count == 0)
                return 0f;
            var sky = Atmosphere.SkyAmbientLegacy;
            return Luminance(sky.X, sky.Y, sky.Z) * MathF.PI / Atmosphere.ReferenceExposure;
        }

        private static float Luminance(float r, float g, float b) => 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }
}
