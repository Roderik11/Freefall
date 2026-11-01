using System;
using System.Collections.Generic;
using System.Numerics;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Assets
{
    /// <summary>One weather of a <see cref="Climate"/>: how often it comes, how long it stays, what it may follow.</summary>
    [Serializable]
    public class ClimateWeather
    {
        public EnvironmentPreset Preset;

        [Description("How often it comes, against the others' (a weight, not a percentage)")]
        public float Chance = 1f;

        [Description("Game hours it stays, the shortest and the longest. The change into it counts")]
        public Vector2 Hours = new(4f, 10f);

        [Description("Game hours it takes to come in: the clouds gather, the fog rises, then the rain follows from the clouds")]
        public float ChangeHours = 1.5f;

        [Description("It begins only between these hours of the day (morning mist 3 to 8; 22 to 4 runs over midnight). Both the same = at any hour")]
        public Vector2 BeginsBetween;

        [Description("It comes only after these weathers (a thunderstorm after overcast). Empty = after any")]
        public List<EnvironmentPreset> After = [];
    }

    /// <summary>
    /// How the weather goes: the presets a place moves between and how often, as a run of spells of weather from hour 0 of
    /// the game time. The run is worked out from <see cref="Seed"/> and the asset alone, nothing about it is kept, so the
    /// same game time gives the same weather (a reload, another machine, the server), and a jump of the clock lands
    /// where the run has got to by then. Which preset comes next is drawn from those that may follow the last, so it
    /// goes from one to the next the way the asset says; a change is a blend of the two presets' numbers over game hours,
    /// and the rain then follows from the clouds as it does for any preset (Graphics.Weather).
    /// EnvironmentController plays it. Docs: .agent/knowledge/atmospheric_effects/weather.md.
    /// </summary>
    [CreateAsset("Climate")]
    public class Climate : Asset, IRebuildAfterLoad
    {
        [Description("Another number gives another run of weather from the same list")]
        public int Seed = 1;

        public List<ClimateWeather> Weathers = [];

        // Where the walk along the run stands: only a cache of the spell at the time last asked for. Going forward takes
        // the steps it takes; a step back in time walks again from hour 0 (some hundred steps a game year)
        private bool _walked;
        private long _index;
        private int _now, _before;      // into Weathers; _before -1 for the first spell
        private double _start, _end;    // game hours

        /// <summary>The game hour the weather now showing gives way to the next (after <see cref="Sample"/>).</summary>
        public double SpellEnds => _end;

        /// <summary>There is a weather to play.</summary>
        public bool HasWeather
        {
            get
            {
                foreach (var weather in Weathers)
                    if (weather?.Preset != null && weather.Chance > 0f) return true;
                return false;
            }
        }

        /// <summary>
        /// The weather at a game time: <paramref name="to"/> is the spell it falls in, and while that comes in
        /// <paramref name="blend"/> (0..1) runs from <paramref name="from"/>, the last one. False without a weather.
        /// </summary>
        public bool Sample(double hours, out EnvironmentPreset from, out EnvironmentPreset to, out float blend)
        {
            from = to = null;
            blend = 1f;
            if (!HasWeather) return false;

            hours = Math.Max(hours, 0.0);
            if (!_walked || hours < _start || _now >= Weathers.Count || _before >= Weathers.Count) Restart();
            while (hours >= _end) Step();

            var weather = Weathers[_now];
            to = weather.Preset;
            from = _before >= 0 ? Weathers[_before].Preset : to;
            float t = Math.Clamp((float)((hours - _start) / MathF.Max(weather.ChangeHours, 0.01f)), 0f, 1f);
            blend = t * t * (3f - 2f * t);
            return true;
        }

        private void Restart()
        {
            _walked = true;
            _index = 0;
            _before = -1;
            _start = 0.0;
            _now = Pick(-1, 0.0);
            _end = Length(_now);
        }

        private void Step()
        {
            _index++;
            _before = _now;
            _start = _end;
            _now = Pick(_before, _start % 24.0);
            _end = _start + Length(_now);
        }

        private double Length(int weather)
        {
            var w = Weathers[weather];
            float length = w.Hours.X + (w.Hours.Y - w.Hours.X) * Random(1);
            return MathF.Max(MathF.Max(length, w.ChangeHours), 0.1f);
        }

        // Drawn by chance from those that may come now; none may: the weather stays for another spell
        private int Pick(int before, double hour)
        {
            float total = 0f;
            for (int i = 0; i < Weathers.Count; i++)
                if (MayFollow(i, before, hour)) total += Weathers[i].Chance;
            if (total <= 0f)
                return before >= 0 ? before : FirstWeather();

            float pick = Random(0) * total;
            int last = -1;
            for (int i = 0; i < Weathers.Count; i++)
            {
                if (!MayFollow(i, before, hour)) continue;
                last = i;
                pick -= Weathers[i].Chance;
                if (pick < 0f) break;
            }
            return last;
        }

        private bool MayFollow(int weather, int before, double hour)
        {
            var w = Weathers[weather];
            if (w?.Preset == null || w.Chance <= 0f || weather == before) return false;
            if (before >= 0 && w.After.Count > 0 && !w.After.Contains(Weathers[before].Preset)) return false;
            return Within(w.BeginsBetween, hour);
        }

        private int FirstWeather()
        {
            for (int i = 0; i < Weathers.Count; i++)
                if (Weathers[i]?.Preset != null && Weathers[i].Chance > 0f) return i;
            return 0;
        }

        private static bool Within(Vector2 hours, double hour)
        {
            if (hours.X == hours.Y) return true;
            return hours.X < hours.Y ? hour >= hours.X && hour < hours.Y : hour >= hours.X || hour < hours.Y;
        }

        // 0..1 from the seed, the spell and what it is for: the same spell draws the same, whenever it is reached
        private float Random(int salt)
        {
            ulong h = unchecked(((ulong)_index * 2 + (ulong)salt + 1) * 0x9E3779B97F4A7C15UL + (ulong)(uint)Seed * 0xD6E8FEB86659FD93UL);
            h ^= h >> 29; h = unchecked(h * 0xBF58476D1CE4E5B9UL); h ^= h >> 32;
            return (h & 0xFFFFFF) / (float)0x1000000;
        }

        public override void MarkDirty()
        {
            _walked = false;
            base.MarkDirty();
        }

        public void RebuildAfterLoad() => _walked = false;
    }
}
