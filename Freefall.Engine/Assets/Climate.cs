using System;
using System.Numerics;
using Category = System.ComponentModel.CategoryAttribute;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Assets
{
    /// <summary>What a climate's weather is made of at a game time: the slow causes behind the sky, the wind and the sea.</summary>
    public struct ClimateState
    {
        public double Hours;
        public float Season;         // −1 the depth of winter … +1 the height of summer
        public float Pressure;       // −1 a deep low … +1 a strong high: the weather systems passing
        public float Overcast;       // 0 fair … 1 a deck over the sky (a low)
        public float Deep;           // 0 … 1 the deck deepened until it rains or snows
        public float Fair;           // 0 … 1 how much cloud fair weather keeps (its own noise)
        public float Cumulus;        // 0 … 1 heaped cloud of the day's heating
        public float Thunder;        // 0 … 1 towers grown into thunderstorms
        public float Mist;           // 0 … 1 mist left by a still, clear night
        public float FreezingLevel;  // m above sea level
        public float Wind;           // m/s near the ground
        public float Turn;           // radians the weather systems turn the wind's heading (Graphics.Wind.Turn)
        public float SeaWind;        // m/s: the wind as the sea has taken it up
        public double SeaHeading;    // radians the wind sea runs along (0 = +x)
        public float Swell;          // the swell's height as a share of the authored one
        public float SwellTurn;      // radians off the authored swell's heading
    }

    /// <summary>
    /// How the weather goes in a place. Nothing here is a weather to pick: the weather is made as it goes from a few slow
    /// causes, each a smooth noise of the game time drawn from <see cref="Seed"/>, apart from each other:
    ///   • weather systems: highs and lows passing every couple of days (<see cref="SystemHours"/>). A low decks the sky,
    ///     lowers its base and deepens it until it rains or snows (the clouds then decide where: Graphics.Weather); a
    ///     high is fair, with as much fair-weather cloud as its own noise says.
    ///   • the season over <see cref="YearDays"/>, and warm and cold spells: the freezing level, which says rain or snow.
    ///   • the day's heating: cumulus heaping up through warm afternoons, on unstable days towers and thunderstorms
    ///     (<see cref="Storminess"/>), gone by night; a warm low has storms in it at any hour.
    ///   • mist on still, clear nights, lying until mid-morning (<see cref="Mistiness"/>).
    ///   • the wind, a noise of its own: calm to several times <see cref="Windiness"/>, stronger as a front passes and in a
    ///     deep low, by day than by night. So a clear day can blow hard and an overcast one be still. The weather systems
    ///     turn its heading too (<see cref="ClimateState.Turn"/>): it backs as a low comes and veers as it goes.
    ///   • the sea: the wind as the waves have taken it up, built over <see cref="SeaRises"/> and run down over
    ///     <see cref="SeaFalls"/> game hours, its heading trailing the wind's (worked out by playing the wind of the last
    ///     day and a half through those rates, so it is a function of the time like the rest); the swell from far-off
    ///     weather, coming and going over days and running ahead of a low (<see cref="Swell"/>).
    /// What weather does not change (the sun, the air, the night sky) is the <see cref="Look"/>. The weather is a pure
    /// function of the seed, these numbers and the game time: a reload, a jump of the clock, another machine or the
    /// server land on the same. Presets only override it (a held preset, a WeatherZone). EnvironmentController plays it.
    /// Docs: .agent/knowledge/atmospheric_effects/weather.md §6.
    /// </summary>
    [CreateAsset("Climate")]
    public class Climate : Asset, IRebuildAfterLoad
    {
        [Description("Another number gives another run of weather from the same climate")]
        public int Seed = 1;

        [Description("How the place looks whatever the weather: the sun, the air, the fog's colour, the night sky, the " +
                     "exposure. The climate makes the rest: the clouds, the fog's thickness, the freezing level, the wind " +
                     "and the sea. Empty = a new Environment Preset's")]
        public EnvironmentPreset Look;

        [Category("Year")]
        [Description("Game days the seasons take to come round")]
        [ValueRange(4f, 400f)]
        public float YearDays = 120f;

        [Description("The day of the year at the height of summer; day 0 is the clock's first day")]
        [ValueRange(0f, 400f)]
        public float WarmestDay = 20f;

        [Description("The 0 °C level in metres above sea level at the depth of winter (x) and the height of summer (y): " +
                     "what falls is snow over it and rain under it (Graphics.Weather)")]
        public Vector2 FreezingLevel = new(-500f, 3500f);

        [Description("Metres warm and cold spells of a few days move the freezing level up or down")]
        [ValueRange(0f, 3000f)]
        public float Spells = 700f;

        [Category("Sky")]
        [Description("The share of the time under a deck of cloud (the lows) rather than in fair weather (the highs)")]
        [ValueRange(0f, 1f)]
        public float Cloudiness = 0.4f;

        [Description("Game hours from a high to the next low, about: how fast weather systems pass (in a temperate place " +
                     "a front every two to five days)")]
        [ValueRange(6f, 200f)]
        public float SystemHours = 30f;

        [Description("How often a warm afternoon builds its clouds into thunderstorms, and a warm low has storms in it " +
                     "(0 never)")]
        [ValueRange(0f, 1f)]
        public float Storminess = 0.4f;

        [Description("How often a still, clear night leaves mist lying until mid-morning (0 never)")]
        [ValueRange(0f, 1f)]
        public float Mistiness = 0.4f;

        [Category("Wind")]
        [Description("The usual wind near the ground (m/s). It rises and drops by itself, apart from the sky: from a calm " +
                     "to several times this, stronger as a front passes and in a deep low, by day than by night")]
        [ValueRange(0f, 20f)]
        public float Windiness = 4.5f;

        [Category("Sea")]
        [Description("Game hours a wind takes to raise its sea (most of the way: a time constant)")]
        [ValueRange(0.1f, 24f)]
        public float SeaRises = 2f;

        [Description("Game hours a sea takes to die down once its wind drops: the old waves run on")]
        [ValueRange(0.1f, 48f)]
        public float SeaFalls = 5f;

        [Description("The swell from far-off weather as a share of the ocean's authored swell bands' height: the least (x) " +
                     "and the most (y). It comes and goes over days, and runs ahead of a low")]
        public Vector2 Swell = new(0.2f, 1.2f);

        // The last state worked out (the clock stands still in the editor: no work then)
        private ClimateState _state;
        private bool _sampled;
        private static readonly EnvironmentPreset Defaults = new();

        /// <summary>The weather's causes as of the last <see cref="Sample"/>.</summary>
        public ClimateState State => _state;

        /// <summary>Fill <paramref name="dest"/> with the weather at a game time, and keep its causes in <see cref="State"/>.</summary>
        public void Sample(double hours, EnvironmentPreset dest)
        {
            if (!_sampled || hours != _state.Hours)
            {
                _state = Evaluate(hours);
                _sampled = true;
            }
            Fill(_state, dest);
        }

        /// <summary>The weather's causes at a game time (a pure function of the time and this asset).</summary>
        public ClimateState Evaluate(double h)
        {
            var s = new ClimateState { Hours = h };
            float hour = (float)(h - 24.0 * Math.Floor(h / 24.0));

            // The season and the spells: the freezing level
            s.Season = (float)Math.Cos(2.0 * Math.PI * (h / 24.0 - WarmestDay) / Math.Max(YearDays, 1f));
            s.FreezingLevel = Lerp(FreezingLevel.X, FreezingLevel.Y, 0.5f + 0.5f * s.Season) + Spells * (float)Noise(h / 50.0, 3);
            float warmth = SmoothStep(1000f, 3000f, s.FreezingLevel);

            // The weather systems
            s.Pressure = Pressure(h);
            s.Overcast = Overcast(s.Pressure);
            s.Deep = Deep(s.Pressure);
            s.Fair = 0.5f + 0.5f * (float)Noise(h / 14.0, 10);

            // The day's heating: heaped cloud in the afternoon; on an unstable day towers, and storms in a warm low at any hour
            float heat = Heat(hour);
            float unstable = 0.5f + 0.5f * (float)Noise(h / 20.0, 4);
            float storms = SmoothStep(1f - 0.6f * Storminess, 1.05f - 0.3f * Storminess, unstable);
            s.Cumulus = heat * SmoothStep(0.2f, 0.7f, unstable) * (0.4f + 0.6f * warmth) * (1f - s.Overcast);
            s.Thunder = Math.Clamp((heat + 0.6f * s.Deep) * warmth * storms, 0f, 1f);

            // The wind, and the turn the systems give its heading
            s.Wind = WindAt(h, out s.Turn);

            // Mist: a still, clear night's, from late evening until it burns off by mid-morning
            float sinceEvening = (hour - 21f + 24f) % 24f;
            float mistHours = SmoothStep(0f, 6f, sinceEvening) * (1f - SmoothStep(10f, 13f, sinceEvening));
            float damp = 0.5f + 0.5f * (float)Noise(h / 24.0, 5);
            s.Mist = mistHours * (1f - s.Overcast) * (1f - SmoothStep(2f, 5f, s.Wind))
                     * SmoothStep(1f - Mistiness, 1.15f - Mistiness, damp);

            // The sea
            Sea(h, out s.SeaWind, out s.SeaHeading);
            float far = 0.5f + 0.5f * (float)Noise(h / 40.0, 8);
            float ahead = Deep(Pressure(h + 8.0));              // a low's swell runs ahead of it
            float share = Math.Clamp(0.9f * SmoothStep(0.3f, 0.85f, far) + 0.5f * ahead + 0.3f * s.Deep, 0f, 1f);
            s.Swell = Lerp(Swell.X, Swell.Y, share);
            s.SwellTurn = 0.35f * (float)Noise(h / 30.0, 9);    // ±20°
            return s;
        }

        // ── The parts ──

        // −1 a deep low … +1 a strong high
        private float Pressure(double h) =>
            (float)(0.75 * Noise(h / SystemHours, 1) + 0.25 * Noise(h / (0.37 * SystemHours), 2));

        // The pressure under which a deck covers the sky: the lows take Cloudiness of the time (fitted to the pressure's
        // spread over a game year)
        private float DeckPressure => -0.55f + 1.1f * Cloudiness;

        private float Overcast(float pressure) => SmoothStep(DeckPressure + 0.12f, DeckPressure - 0.12f, pressure);

        private float Deep(float pressure) => SmoothStep(DeckPressure - 0.15f, DeckPressure - 0.45f, pressure);

        // How warm the afternoon has made the ground: from the morning to the evening
        private static float Heat(float hour) => SmoothStep(9f, 13f, hour) * (1f - SmoothStep(16f, 20f, hour));

        private static float Night(float hour) => MathF.Max(SmoothStep(19f, 22f, hour), 1f - SmoothStep(6f, 9f, hour));

        /// <summary>
        /// The wind near the ground at a game time (m/s), and the turn the weather systems give its heading (radians): a
        /// noise of its own around <see cref="Windiness"/>, spread over a calm to several times it, raised as a front
        /// passes (the pressure changing fast) and in a deep low, by the day's mixing, lowered at night.
        /// </summary>
        public float WindAt(double h, out float turn)
        {
            float hour = (float)(h - 24.0 * Math.Floor(h / 24.0));
            float p = Pressure(h);
            float change = MathF.Abs(Pressure(h + 0.5) - Pressure(h - 0.5)) * SystemHours;
            float front = SmoothStep(0.5f, 1.6f, change);
            float own = (float)(Noise(h / 9.0, 6) + 0.35 * Noise(h / 2.7, 7));
            float wind = Windiness * MathF.Exp(own) * (1f + 0.6f * front + 0.5f * Deep(p))
                         * (1f + 0.3f * Heat(hour) - 0.3f * Night(hour));
            turn = 0.6f * p;
            return Math.Clamp(wind, 0.2f, 28f);
        }

        // The sea's wind: the wind of the last 6 × the longer rate played through the rise and fall rates on a fixed grid of
        // quarter hours (so it is the same whenever it is asked for), its heading as the wind's velocity trailed at the
        // rise rate: new waves grow along the new wind while the old ones run down. The play up to the last grid point is
        // kept until the time passes the next (a game quarter hour), then only the step from it to the time is new: a whole
        // play was 35 µs a frame
        private const double SeaStep = 0.25;
        private long _seaGrid = long.MinValue;
        private float _seaSpeed;
        private Vector2 _seaTrail;

        private void Sea(double h, out float speed, out double heading)
        {
            long grid = (long)Math.Floor(h / SeaStep);
            if (grid != _seaGrid)
            {
                // From a whole number of steps back (so the start, and the result, follow from the grid point alone)
                long steps = (long)Math.Ceiling(6.0 * Math.Max(SeaRises, SeaFalls) / SeaStep);
                double t = (grid - steps) * SeaStep;
                float u = WindAt(t, out float turn);
                float s = u;
                var trail = Along(t, turn) * u;
                for (long k = grid - steps + 1; k <= grid; k++)
                    SeaStepTo(k * SeaStep, SeaStep, ref s, ref trail);
                _seaGrid = grid;
                _seaSpeed = s;
                _seaTrail = trail;
            }
            speed = _seaSpeed;
            var last = _seaTrail;
            double rest = h - grid * SeaStep;
            if (rest > 1e-9) SeaStepTo(h, rest, ref speed, ref last);
            if (last.LengthSquared() <= 1e-6f) last = Along(h, 0f);
            heading = Math.Atan2(last.Y, last.X);
        }

        // One step of the sea toward the wind at time t, over dt game hours
        private void SeaStepTo(double t, double dt, ref float speed, ref Vector2 trail)
        {
            float u = WindAt(t, out float turn);
            float rise = 1f - MathF.Exp(-(float)dt / MathF.Max(SeaRises, 0.01f));
            float fall = 1f - MathF.Exp(-(float)dt / MathF.Max(SeaFalls, 0.01f));
            speed += (u - speed) * (u > speed ? rise : fall);
            trail += (Along(t, turn) * u - trail) * rise;
        }

        // The wind's heading at a time as a unit vector (xz)
        private static Vector2 Along(double t, float turn)
        {
            double a = Graphics.Wind.GeneratedHeading(t) + turn;
            return new Vector2((float)Math.Cos(a), (float)Math.Sin(a));
        }

        // ── What the causes look like ──

        // EV a full storm (Thunder × Overcast 1) takes from the sun's and moons' light (EnvironmentPreset.LightDimming)
        private const float StormGloom = 8f;

        private void Fill(in ClimateState s, EnvironmentPreset dest)
        {
            dest.CopyFrom(Look ?? Defaults);

            // The clouds: fair weather's scattered cumulus, heaped by the afternoon; a low's deck, stratus while it is thin,
            // deep enough to rain or snow as it deepens; towers and lightning in a storm
            float fairCover = 0.15f + 0.2f * s.Fair + 0.3f * s.Cumulus;
            dest.CloudCoverage = MathF.Min(Lerp(fairCover, Lerp(0.78f, 0.9f, s.Deep), s.Overcast) + 0.1f * s.Thunder, 0.92f);
            dest.CloudType = MathF.Max(Lerp(1f, Lerp(0.1f, 0.3f, s.Deep), s.Overcast), 0.75f * s.Thunder);
            dest.CloudAltitude = Lerp(1800f + 700f * s.Fair, Lerp(1400f, 1000f, s.Deep), s.Overcast);
            // A deep deck reaches a kilometre and more over the freezing level, into the ice, so it rains in summer as it
            // snows in winter (Graphics.Weather's rule); a storm's deck is 3.5 km at least
            float deepThickness = Math.Clamp(s.FreezingLevel + 2000f - dest.CloudAltitude, 2000f, 6000f);
            float thickness = Lerp(1000f + 800f * s.Cumulus, Lerp(700f, deepThickness, s.Deep), s.Overcast);
            dest.CloudThickness = MathF.Max(thickness, Lerp(thickness, 3500f, s.Thunder * s.Overcast));
            dest.CloudTowers = 0.03f + 0.1f * s.Cumulus + 0.4f * s.Thunder;
            dest.CloudTowerTop = Lerp(Lerp(3500f, 5500f, s.Cumulus), 12000f, s.Thunder);
            dest.CloudDensity = Lerp(Lerp(0.1f, Lerp(0.03f, 0.06f, s.Deep), s.Overcast), 0.12f, s.Thunder);
            dest.LightningRate = 4f * SmoothStep(0.4f, 0.9f, s.Thunder);
            dest.CloudAlbedo = Lerp(dest.CloudAlbedo, 0.97f, s.Thunder);
            dest.CloudSpeed = 0.3f + 0.05f * s.Wind;     // the wind aloft, from the ground's

            // The air: a low's damp haze, the mist, a storm's murk
            dest.FogDensity = Lerp(dest.FogDensity, 0.0016f, s.Overcast) + 0.01f * s.Mist;
            dest.FogHeight = Lerp(dest.FogHeight, 150f, s.Mist);
            dest.MieScale *= 1f + 0.5f * s.Mist + s.Thunder;
            // A storm's gloom is light the sun and moons lose, not the camera's (EnvironmentPreset.LightDimming): lamps
            // and fires show against it as at dusk. It was ExposureCompensation −2.5 × Thunder × Overcast until 2026-11-25,
            // a dark picture over an overcast day's light (2-7k lux at noon) that no lantern could show in; −8 EV of
            // light looks about as dark, the eye keeping a share of the fall (AutoExposure's dim light)
            dest.LightDimming -= StormGloom * s.Thunder * s.Overcast;

            dest.FreezingLevel = s.FreezingLevel;
            dest.WindStrength = s.Wind;
            dest.SeaWind = s.SeaWind;
            dest.Swell = s.Swell;
        }

        // ── Noise ──

        // Smooth value noise, −1..1, from the seed and what it is for
        private double Noise(double x, int salt)
        {
            double i = Math.Floor(x), f = x - i;
            double u = f * f * (3.0 - 2.0 * f);
            double a = Lattice((long)i, salt), b = Lattice((long)i + 1, salt);
            return a + (b - a) * u;
        }

        private double Lattice(long i, int salt)
        {
            ulong h = unchecked((ulong)i * 0x9E3779B97F4A7C15UL + (ulong)(uint)salt * 0xD6E8FEB86659FD93UL
                                + (ulong)(uint)Seed * 0xA24BAED4963EE407UL);
            h ^= h >> 29; h = unchecked(h * 0xBF58476D1CE4E5B9UL); h ^= h >> 32;
            return (h & 0xFFFFFF) / (double)0xFFFFFF * 2.0 - 1.0;
        }

        private static float Lerp(float a, float b, float t) => a + (b - a) * t;

        private static float SmoothStep(float a, float b, float x)
        {
            float t = Math.Clamp((x - a) / (b - a), 0f, 1f);
            return t * t * (3f - 2f * t);
        }

        public override void MarkDirty()
        {
            Forget();
            base.MarkDirty();
        }

        public void RebuildAfterLoad() => Forget();

        // An edit or a load: what was worked out from the old numbers goes
        private void Forget()
        {
            _sampled = false;
            _seaGrid = long.MinValue;
        }
    }
}
