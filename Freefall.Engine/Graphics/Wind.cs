using System;
using System.Numerics;

namespace Freefall.Graphics
{
    /// <summary>
    /// The wind vegetation sways in, the weather's particles drift in and the clouds are carried by. Grass cards, blade
    /// clumps, flowers and trees read it from the scene constants (WindDirection, WindStrength, WindPhase, WindOffset in
    /// common.fx). Its direction is never authored (user, 2026-10-19): it wanders with the game time
    /// (<see cref="GeneratedHeading"/>), the same for the clouds and the ground. Its speed is the weather's:
    /// EnvironmentController sets <see cref="Speed"/> from the preset's WindStrength; without one it is a light breeze.
    /// Settings › Wind can override it for tuning.
    /// </summary>
    public static class Wind
    {
        /// <summary>The breeze when no EnvironmentController sets the wind (m/s).</summary>
        public const float DefaultSpeed = 3.58f;

        /// <summary>The weather's wind speed (m/s).</summary>
        public static float Speed = DefaultSpeed;

        /// <summary>Game hours (Day × 24 + TimeOfDay) the wind's direction follows; SkyboxRenderer sets it every frame.</summary>
        public static double GameHours;

        /// <summary>Where the wind blows this frame (xz, unit), smooth: the clouds drift with it.</summary>
        public static Vector2 GeneratedDirection { get; private set; } = Vector2.Normalize(new Vector2(2f, 1f));

        // The direction the ground's wind uses: the generated one in whole degrees, so the ocean's wind sea (whose initial
        // spectrum is rebuilt when the wind changes) is rebuilt once a degree, not every frame
        private static Vector2 _steppedDirection = GeneratedDirection;

        /// <summary>World-space wind velocity (m/s); only its horizontal part is used.</summary>
        public static Vector3 Velocity => new Vector3(_steppedDirection.X, 0f, _steppedDirection.Y) * Speed;

        /// <summary>The wind in effect: Settings › Wind override when it is on, else <see cref="Velocity"/>.</summary>
        public static Vector3 Current
        {
            get
            {
                var settings = Engine.Settings;
                if (settings == null || !settings.OverrideWind) return Velocity;
                float rad = settings.OverrideWindDirection * (MathF.PI / 180f);       // 0 = +x, 90 = +z
                return new Vector3(MathF.Cos(rad), 0f, MathF.Sin(rad)) * settings.OverrideWindStrength;
            }
        }

        /// <summary>
        /// The wind's heading (radians, 0 = +x, π/2 = +z) at a game time: two smooth noises of the time, one turning it
        /// round over days, one swinging it by tens of degrees over a few hours. At hour 0 it starts near today's old
        /// fixed heading (2, 1). The same time gives the same wind everywhere (a reload, another machine).
        /// </summary>
        public static double GeneratedHeading(double gameHours)
            => 0.4636 + 2.6 * Noise(gameHours / 30.0) + 0.5 * Noise(gameHours / 5.0 + 37.0);

        // Smooth value noise, -1..1, 0 at x = 0
        private static double Noise(double x)
        {
            double i = Math.Floor(x), f = x - i;
            double u = f * f * (3.0 - 2.0 * f);
            return Lattice(i) + (Lattice(i + 1.0) - Lattice(i)) * u - Lattice(0.0);
        }

        private static double Lattice(double i)
        {
            ulong h = unchecked((ulong)(long)i * 0x9E3779B97F4A7C15UL);
            h ^= h >> 29; h = unchecked(h * 0xBF58476D1CE4E5B9UL); h ^= h >> 32;
            return (h & 0xFFFFFF) / (double)0xFFFFFF - 0.5;
        }

        /// <summary>
        /// Phase of the sway and flutter: the integral of their speed, which rises with the wind, so a change of
        /// strength never makes the vegetation jump.
        /// </summary>
        public static float Phase { get; private set; }

        /// <summary>
        /// How far the air has moved (m, horizontal), wrapped every <see cref="OffsetPeriod"/>: the gust pattern
        /// (GustPattern in common.fx) scrolls by it, so gusts roll downwind and a turning wind only turns where the
        /// pattern goes. A wave over dot(position, direction) instead raced through hundreds of radians whenever a
        /// preset transition turned the wind, ~1800 m from the origin on Eldermoor.
        /// </summary>
        public static Vector2 Offset { get; private set; }

        /// <summary>The gust noise repeats every this many metres (its lattices wrap), so the offset's wrap is invisible.</summary>
        public const float OffsetPeriod = 200f;

        public static float Strength
        {
            get
            {
                var v = Current;
                return MathF.Sqrt(v.X * v.X + v.Z * v.Z);
            }
        }

        /// <summary>Horizontal unit vector the wind blows towards; zero in a calm.</summary>
        public static Vector3 Direction
        {
            get
            {
                var v = Current;
                float s = MathF.Sqrt(v.X * v.X + v.Z * v.Z);
                return s > 1e-3f ? new Vector3(v.X / s, 0f, v.Z / s) : Vector3.Zero;
            }
        }

        /// <summary>Once per frame (Engine.Tick).</summary>
        public static void Advance(float delta)
        {
            double heading = GeneratedHeading(GameHours);
            GeneratedDirection = new Vector2((float)Math.Cos(heading), (float)Math.Sin(heading));
            double stepped = Math.Round(heading * (180.0 / Math.PI)) * (Math.PI / 180.0);
            _steppedDirection = new Vector2((float)Math.Cos(stepped), (float)Math.Sin(stepped));

            var v = Current;
            Phase += delta * (0.8f + 0.12f * Strength);
            if (Phase > 10000f) Phase -= 10000f * MathF.Floor(Phase / 10000f);   // keep float precision; the waves don't line up at the wrap, which is rare and brief

            var o = Offset + new Vector2(v.X, v.Z) * delta;
            Offset = new Vector2(o.X - OffsetPeriod * MathF.Floor(o.X / OffsetPeriod), o.Y - OffsetPeriod * MathF.Floor(o.Y / OffsetPeriod));
        }
    }
}
