using Freefall.Graphics;
using System;

namespace Freefall.Base
{
    /// <summary>
    /// The world's time: the hours since hour 0 of day 0, which only the game moves on. <see cref="Time"/> is the process's
    /// real time; this is what the sun, the moons, the wind's heading, the storm cells, the climate's weather and people's
    /// days read, so it needs no sky to run (a dedicated server has none). It runs in <see cref="Engine.Tick"/> while the
    /// world does (not during a scene load). A scene sets it through a <see cref="Components.WorldClock"/>; a scene without
    /// one goes on at the time it finds.
    /// Docs: .agent/knowledge/atmospheric_effects/sky_and_environment.md.
    /// </summary>
    public static class GameClock
    {
        /// <summary>Hours since day 0, in one number: never wraps, the same time gives the same world.</summary>
        public static double Hours
        {
            get => _hours;
            set { _hours = value; Jumps++; }
        }

        /// <summary>The hour of the day, 0..24.</summary>
        public static float TimeOfDay
        {
            get => (float)(_hours - Math.Floor(_hours / 24.0) * 24.0);
            set => Hours = Day * 24.0 + Math.Clamp(value, 0f, 23.999f);
        }

        /// <summary>Days since day 0: counts on as the hour wraps, moves the moons along their orbits.</summary>
        public static int Day
        {
            get => (int)Math.Floor(_hours / 24.0);
            set => Hours = value * 24.0 + TimeOfDay;
        }

        /// <summary>The clock runs in play. In the editor it also takes <see cref="RunsInEditor"/>.</summary>
        public static bool Running;

        /// <summary>The clock runs in the editor too, outside play, with <see cref="Running"/>.</summary>
        public static bool RunsInEditor;

        /// <summary>How fast it runs: 1 = real time, 60 = an hour a minute, 360 = a day in 4 min.</summary>
        public static float GameSecondsPerSecond = 3.6f;

        /// <summary>
        /// How many times something set the time rather than the clock running on to it: whoever schedules by the hour
        /// compares this to start over after a jump (a scene load, a drag of the hour, a sleep).
        /// </summary>
        public static int Jumps { get; private set; }

        private static double _hours = 16.0;

        internal static void Advance(float delta)
        {
            if (!Running || (Engine.IsEditor && !RunsInEditor)) return;
            _hours += delta * GameSecondsPerSecond / 3600.0;
            Daylight.Update(delta);
        }
    }
}
