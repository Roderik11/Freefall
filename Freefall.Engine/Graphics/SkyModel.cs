using System;
using System.Numerics;

namespace Freefall.Graphics
{
    /// <summary>
    /// CPU copy of the parts of the sky model in sky_common.fx that the CPU needs: the colour of the sunlight after its
    /// path through the air (the directional light takes it, so the sun path on the water and every sunlit surface turn
    /// golden at sunset with the sky) and the zenith colour (the sky ambient, for the ground bounce and the lamps).
    /// Every constant and step must match sky_common.fx (SkyRayleighDepth, SkyOzoneDepth, SkyMieDepth, AirMass,
    /// HazeAirMass, OzoneAirMass, SkySingleScatter, GetSkyColor); change both together.
    /// </summary>
    public static class SkyModel
    {
        private static readonly Vector3 RayleighDepth = new(0.065f, 0.10f, 0.196f);
        private static readonly Vector3 OzoneDepth = new(0.075f, 0.065f, 0.006f);
        private const float ViewDepthScale = 0.75f;

        private static float AirMass(float mu)
        {
            mu = MathF.Max(mu, 0f);
            return 1f / (mu + 0.025f * MathF.Exp(-11f * mu));
        }

        private static float HazeAirMass(float mu)
        {
            mu = MathF.Max(mu, 0f);
            return 1f / (mu + 0.003f * MathF.Exp(-30f * mu));
        }

        private static float OzoneAirMass(float mu)
        {
            mu = MathF.Max(mu, 0f);
            return 1f / MathF.Sqrt(mu * mu + 0.0079f);
        }

        private static float MieDepth(float mieScattering) => mieScattering * 1.5f;

        private static Vector3 SlantDepth(float mu, float mieScattering) =>
            RayleighDepth * AirMass(mu) + new Vector3(MieDepth(mieScattering) * HazeAirMass(mu)) + OzoneDepth * OzoneAirMass(mu);

        /// <summary>Colour of direct sunlight after its path through the air, 1 with the sun at the zenith.</summary>
        public static Vector3 SunlightThroughAir(float sunY, float mieScattering)
        {
            Vector3 d = SlantDepth(sunY, mieScattering) - SlantDepth(1f, mieScattering);
            return new Vector3(MathF.Exp(-d.X), MathF.Exp(-d.Y), MathF.Exp(-d.Z));
        }

        private static float RayleighPhase(float c) => 3f / (16f * MathF.PI) * (1f + c * c);

        private static float HGPhase(float c, float g)
        {
            float g2 = g * g;
            float denom = 1f + g2 - 2f * g * c;
            return (1f - g2) / (4f * MathF.PI * MathF.Pow(MathF.Abs(denom), 1.5f));
        }

        private static Vector3 SingleScatter(float viewY, float sunY, float cosTheta, float mieScattering, float g)
        {
            Vector3 xs = SlantDepth(sunY, mieScattering);
            Vector3 viewR = RayleighDepth * AirMass(viewY);
            float viewM = MieDepth(mieScattering) * HazeAirMass(viewY);
            Vector3 xv = (viewR + new Vector3(viewM)) * ViewDepthScale;
            Vector3 phase = (viewR * RayleighPhase(cosTheta) + new Vector3(viewM * HGPhase(cosTheta, g))) / (viewR + new Vector3(viewM));
            return new Vector3(Path(xs.X, xv.X), Path(xs.Y, xv.Y), Path(xs.Z, xv.Z)) * phase;

            static float Path(float s, float v)
            {
                float diff = v - s;
                if (MathF.Abs(diff) < 1e-4f) diff = diff < 0f ? -1e-4f : 1e-4f;
                return v * (MathF.Exp(-s) - MathF.Exp(-v)) / diff;
            }
        }

        private static float Luminance(Vector3 c) => 0.2126f * c.X + 0.7152f * c.Y + 0.0722f * c.Z;

        /// <summary>
        /// GetSkyColor straight up (sky_common.fx), with the day / sunset / night weights the CPU already has.
        /// At the zenith the brightness compression is neutral and the horizon haze and sunset tint are zero.
        /// </summary>
        public static Vector3 ZenithSkyColor(float sunY, Vector3 skyTint, float atmosphereDensity, float mieScattering,
            float mieAnisotropy, float dayFactor, float sunsetFactor, float nightFactor, Vector3 nightSkyColor)
        {
            Vector3 zenith = SingleScatter(1f, sunY, sunY, mieScattering, mieAnisotropy);
            Vector3 noonZenith = SingleScatter(1f, 0.84f, 0.84f, mieScattering, mieAnisotropy);
            float lumZenith = MathF.Max(Luminance(zenith), 1e-8f);
            Vector3 day = zenith * 1.6f * (MathF.Pow(MathF.Max(Luminance(noonZenith), 1e-8f) / lumZenith, 0.3f) * 16f);

            float lowSun = 1f - Math.Clamp(sunY / 0.35f, 0f, 1f);
            day += SunlightThroughAir(sunY, mieScattering) * (HGPhase(sunY, 0.9f) * mieScattering * 57f * lowSun * lowSun);
            day *= Math.Clamp(1f + sunY / 0.12f, 0f, 1f);

            float tintMax = MathF.Max(skyTint.X, MathF.Max(skyTint.Y, skyTint.Z));
            float tintMin = MathF.Min(skyTint.X, MathF.Min(skyTint.Y, skyTint.Z));
            float tintSat = (tintMax - tintMin) / MathF.Max(tintMax, 1e-3f);
            float saturation = 0.2f + (2.3f + (0.85f - 2.3f) * lowSun - 0.2f) * tintSat;
            float dayLum = Luminance(day);
            Vector3 cast = Vector3.Lerp(skyTint / MathF.Max(Luminance(skyTint), 1e-3f), Vector3.One, tintSat);
            day = Vector3.Max(new Vector3(dayLum) + (day - new Vector3(dayLum)) * saturation, Vector3.Zero) * cast * atmosphereDensity;

            return day * (dayFactor + sunsetFactor) + nightSkyColor * nightFactor;
        }
    }
}
