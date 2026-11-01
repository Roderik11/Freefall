using Freefall.Base;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Components
{
    /// <summary>
    /// The scene's view of <see cref="GameClock"/>: what it shows in the inspector is the clock itself, live, and what a
    /// saved scene holds is the time and pace it starts at, set on the clock as the scene loads. Holds nothing of its own,
    /// so nothing here can drift from the clock. One in a scene; on an entity of its own the hour is a click away in the
    /// Explorer, where on the sky's it comes under all the sky's fields.
    /// </summary>
    [Icon("icon_sky.png")]
    public class WorldClock : Component
    {
        [Description("The hour, 0-24. The sun, the moons, the sky and the lamps follow at once")]
        [ValueRange(0f, 24f)]
        public float TimeOfDay
        {
            get => GameClock.TimeOfDay;
            set => GameClock.TimeOfDay = value;
        }

        [Description("Days since day 0: counts on as the hour wraps, moves the moons along their orbits")]
        public int Day
        {
            get => GameClock.Day;
            set => GameClock.Day = value;
        }

        [Description("The clock runs in play. Stopped, the sun stands, and the weather, the wind and the storm cells with it")]
        public bool Running
        {
            get => GameClock.Running;
            set => GameClock.Running = value;
        }

        [Description("The clock runs in the editor too, outside play, with Running: to watch a day go by")]
        public bool RunsInEditor
        {
            get => GameClock.RunsInEditor;
            set => GameClock.RunsInEditor = value;
        }

        [Description("How fast it runs: 1 = real time, 60 = an hour a minute, 360 = a day in 4 min")]
        [ValueRange(1f, 360f)]
        public float GameSecondsPerSecond
        {
            get => GameClock.GameSecondsPerSecond;
            set => GameClock.GameSecondsPerSecond = value;
        }
    }
}
