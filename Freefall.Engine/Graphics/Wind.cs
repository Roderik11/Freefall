using System;
using System.Numerics;

namespace Freefall.Graphics
{
    /// <summary>
    /// The wind vegetation sways in. Grass cards, blade clumps, flowers and tree leaves read it from the scene
    /// constants (WindDirection, WindStrength, WindPhase in common.fx). EnvironmentController sets it from the
    /// environment preset while it drives the weather; without one it is a light breeze.
    /// </summary>
    public static class Wind
    {
        /// <summary>What vegetation sways in when no EnvironmentController sets the wind: a light breeze.</summary>
        public static readonly Vector3 DefaultVelocity = new(3.2f, 0f, 1.6f);

        /// <summary>World-space wind velocity (m/s); only its horizontal part is used.</summary>
        public static Vector3 Velocity = DefaultVelocity;

        /// <summary>
        /// Phase of the gusts rolling through grass and crowns: the integral of their speed, which rises with the
        /// wind, so a change of strength never makes the vegetation jump.
        /// </summary>
        public static float Phase { get; private set; }

        /// <summary>
        /// How far the air has moved (m, horizontal), wrapped every <see cref="OffsetPeriod"/>: the gust pattern
        /// (GustPattern in common.fx) scrolls by it, so gusts roll downwind and a turning wind only turns where the
        /// pattern goes. A wave over dot(position, direction) instead raced through hundreds of radians whenever a
        /// preset transition turned the wind, ~1800 m from the origin on Eldermoor.
        /// </summary>
        public static Vector2 Offset { get; private set; }

        /// <summary>The gust waves repeat every this many metres in x and z (their wave numbers are multiples of 2π / it).</summary>
        public const float OffsetPeriod = 200f;

        public static float Strength => MathF.Sqrt(Velocity.X * Velocity.X + Velocity.Z * Velocity.Z);

        /// <summary>Horizontal unit vector the wind blows towards; zero in a calm.</summary>
        public static Vector3 Direction
        {
            get
            {
                float s = Strength;
                return s > 1e-3f ? new Vector3(Velocity.X / s, 0f, Velocity.Z / s) : Vector3.Zero;
            }
        }

        /// <summary>Once per frame (Engine.Tick).</summary>
        public static void Advance(float delta)
        {
            Phase += delta * (0.8f + 0.12f * Strength);
            if (Phase > 10000f) Phase -= 10000f * MathF.Floor(Phase / 10000f);   // keep float precision; the waves don't line up at the wrap, which is rare and brief

            var o = Offset + new Vector2(Velocity.X, Velocity.Z) * delta;
            Offset = new Vector2(o.X - OffsetPeriod * MathF.Floor(o.X / OffsetPeriod), o.Y - OffsetPeriod * MathF.Floor(o.Y / OffsetPeriod));
        }
    }
}
