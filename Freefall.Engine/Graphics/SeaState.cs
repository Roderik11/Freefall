using System;
using System.Numerics;

namespace Freefall.Graphics
{
    /// <summary>
    /// The sea the weather has raised, which the ocean's wind bands, the band its whitecaps show on and the surf it
    /// carries to the shore read (OceanRenderer), in place of the wind itself (how much of the sea whitens follows the
    /// wind of the moment, <see cref="Wind.Current"/>: OceanRenderer.Whitecaps):
    ///   • the wind sea's wind: the wind as the waves have taken it up. Under a <see cref="Assets.Climate"/> it trails the
    ///     wind, building over a couple of game hours and dying down over several (<see cref="Assets.Climate.SeaRises"/>,
    ///     <see cref="Assets.Climate.SeaFalls"/>), its heading trailing the wind's; a held preset or a zone says its own
    ///     (<see cref="Assets.EnvironmentPreset.SeaWind"/>), a sea as its wind has raised it. Sails, cloth and plants keep
    ///     the wind of the moment (<see cref="Wind.Local"/>): the ship and the sea feel one wind, only the waves trail it.
    ///   • the swell from far-off weather: the height of the ocean's authored bands (the swell, when wind bands make the
    ///     local sea) as a share of what they are authored with, and a turn of their heading.
    /// EnvironmentController sets it every frame; without one the sea follows the wind at once and the swell is as
    /// authored. The Settings › Wind override raises its sea at once, for tuning.
    /// Docs: .agent/knowledge/atmospheric_effects/weather.md "The wind and the sea".
    /// </summary>
    public static class SeaState
    {
        // The wind sea's speed in steps and its heading in whole degrees, as Wind steps its own: the initial spectrum is
        // rebuilt whenever they change, and the sea's wind turns for hours
        private const float SpeedStep = 0.25f;
        private const float SwellStep = 0.01f;

        private static bool _driven;
        private static float _speed, _swell = 1f, _swellTurn;
        private static Vector2 _direction = Vector2.UnitX;

        /// <summary>
        /// Set this frame's sea: the wind that raised it (m/s), the heading its waves run along (radians, 0 = +x, π/2 =
        /// +z), the swell's height as a share of the authored one and its turn (radians) off the authored heading.
        /// </summary>
        public static void Set(float wind, double heading, float swell, double swellTurn)
        {
            _driven = true;
            _speed = MathF.Round(MathF.Max(wind, 0f) / SpeedStep) * SpeedStep;
            double stepped = Math.Round(heading * (180.0 / Math.PI)) * (Math.PI / 180.0);
            _direction = new Vector2((float)Math.Cos(stepped), (float)Math.Sin(stepped));
            _swell = MathF.Round(MathF.Max(swell, 0f) / SwellStep) * SwellStep;
            _swellTurn = (float)(Math.Round(swellTurn * (180.0 / Math.PI)) * (Math.PI / 180.0));
        }

        /// <summary>Nobody drives the sea: it follows the wind at once, the swell as authored.</summary>
        public static void Release()
        {
            _driven = false;
            _swell = 1f;
            _swellTurn = 0f;
        }

        /// <summary>The wind the wind sea was raised by, as a velocity (xz).</summary>
        public static Vector3 Velocity
        {
            get
            {
                var settings = Engine.Settings;
                if (!_driven || (settings != null && settings.OverrideWind)) return Wind.Current;
                return new Vector3(_direction.X, 0f, _direction.Y) * _speed;
            }
        }

        /// <summary>Its speed (m/s).</summary>
        public static float Strength
        {
            get
            {
                var v = Velocity;
                return MathF.Sqrt(v.X * v.X + v.Z * v.Z);
            }
        }

        /// <summary>The way its waves run (xz unit); zero in a calm.</summary>
        public static Vector3 Direction
        {
            get
            {
                var v = Velocity;
                float s = MathF.Sqrt(v.X * v.X + v.Z * v.Z);
                return s > 1e-3f ? new Vector3(v.X / s, 0f, v.Z / s) : Vector3.Zero;
            }
        }

        /// <summary>The swell's height as a share of the authored swell bands' (1 = as authored).</summary>
        public static float Swell => _swell;

        /// <summary>How far the swell's heading is turned off the authored one (radians, whole degrees).</summary>
        public static float SwellTurn => _swellTurn;

        /// <summary>For engine_stats.</summary>
        public static object Describe()
        {
            var v = Velocity;
            return new
            {
                driven = _driven,
                wind = Strength,
                heading = MathF.Atan2(v.Z, v.X) * (180f / MathF.PI),
                swell = _swell,
                swellTurn = _swellTurn * (180f / MathF.PI),
            };
        }
    }
}
