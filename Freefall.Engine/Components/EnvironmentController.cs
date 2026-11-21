using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Category = System.ComponentModel.CategoryAttribute;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Components
{
    /// <summary>
    /// Drives the sky, sun, fog and weather particle emitters from <see cref="EnvironmentPreset"/> assets.
    ///
    /// The weather comes in layers, each laid over the last:
    ///   • <see cref="Climate"/>: the world's weather, made as it goes from the game time (<see cref="GameClock"/>):
    ///     the sky, the wind and the sea each moving on by themselves
    ///   • <see cref="Preset"/>: a weather held in its place while one is set, for a scene that keeps one look
    ///     or to look at one (or call <see cref="TransitionTo"/> from code)
    ///   • every <see cref="WeatherZone"/> the camera is in
    /// A change of <see cref="Preset"/> cross-fades from whatever is shown over <see cref="TransitionDuration"/> seconds.
    /// The blended result is pushed every frame into:
    ///   • <see cref="SkyboxRenderer"/> via <c>ApplyPreset</c> (sun, atmosphere, ground fog, clouds, stars)
    ///   • <c>Graphics.Wind</c> (its speed, the climate's turn of its heading) and <c>Graphics.SeaState</c> (the wind the
    ///     sea was raised by, its heading, the swell): the heading and the sea's heading are always the climate's, which
    ///     presets do not set
    ///   • <c>Graphics.AutoExposure.PresetCompensation</c> (how bright the weather reads)
    ///   • the optional Rain / Snow / Hail / Dust <see cref="ParticleEmitter"/>s (emit rate scaled by what
    ///     falls at the camera, wind copied to the emitter, emitter parked above the camera)
    ///
    /// The time is <see cref="GameClock"/>'s — a preset describes how a day looks,
    /// not what time it is. Rain, snow and hail are not the preset's: the clouds make them where they are
    /// deep enough (<see cref="Graphics.Weather"/>), the preset's freezing level says which; walking out from
    /// under a shower stops it. Dust stays the preset's DustIntensity.
    /// </summary>
    [Icon("icon_sky.png")]
    [UpdateInEditor]
    public class EnvironmentController : Component, IUpdate
    {
        [Category("Weather")]
        [Description("How the weather goes with the game time. It only moves while the clock runs (WorldClock)")]
        public Climate Climate;

        [Description("A weather held in the climate's place. Empty = the climate decides. Changing it starts a cross-fade from the current look.")]
        public EnvironmentPreset Preset;

        [Description("Seconds a preset change takes to fully blend in")]
        [ValueRange(0f, 120f)]
        public float TransitionDuration = 5f;

        [Description("Skybox to drive. Found automatically when left empty.")]
        public SkyboxRenderer Skybox;

        [Category("Weather Emitters")]
        [Description("Emitter scaled by the rain the clouds let fall at the camera (Weather.Rain). Its EmitRate is the rate in heavy rain.")]
        public ParticleEmitter RainEmitter;

        [Description("Emitter scaled by the snow the clouds let fall at the camera (Weather.Snow)")]
        public ParticleEmitter SnowEmitter;

        [Description("Emitter scaled by the hail in a storm cell's core (Weather.Hail)")]
        public ParticleEmitter HailEmitter;

        [Description("The splashes where the rain lands (shape Ground), kept at the camera (not raised). Its rate follows " +
                     "RainEmitter's drops striking its square (SplashShare of them); its own EmitRate is the most it makes")]
        public ParticleEmitter SplashEmitter;

        [Description("Of the rain emitter's drops striking the ground, the share that makes a visible splash (0.036: one in " +
                     "27, Eldermoor's 30000 drops a second over 24 × 24 m against 3000 splashes over 40 × 40 m)")]
        [ValueRange(0f, 0.2f)]
        public float SplashShare = 0.036f;

        [Description("Emitter scaled by DustIntensity")]
        public ParticleEmitter DustEmitter;

        [Description("Keep the weather emitters centered above the main camera")]
        public bool FollowCamera = true;

        [Description("Height above the camera at which precipitation emitters are placed (Dust stays at camera height). Keep it below what the particles can fall within their lifetime.")]
        [ValueRange(0f, 200f)]
        public float EmitterHeight = 8f;

        [Description("Height above the terrain surface for the Dust emitter. Dust hugs the ground, so it is placed relative to the terrain under the camera, not the camera itself.")]
        [ValueRange(0f, 20f)]
        public float DustHeight = 0.5f;

        [Description("Use the weather's wind for the emitters instead of their authored Wind")]
        public bool DriveWind = true;

        [Description("Seconds of wind drift to lead by: emitters are shifted upwind by Wind * WindLead so blown particles still arrive at the camera. Not for an emitter with WrapInBox (its particles stay in the box round the camera)")]
        [ValueRange(0f, 5f)]
        public float WindLead = 1.0f;

        // ── Runtime state ──

        /// <summary>The blended preset currently being applied. Read-only snapshot, re-filled every frame.</summary>
        public EnvironmentPreset Current => _current;

        /// <summary>0 → 1 progress of the running transition (1 when idle).</summary>
        public float TransitionProgress => _duration <= 0f ? 1f : Math.Clamp(_elapsed / _duration, 0f, 1f);

        public bool IsTransitioning => TransitionProgress < 1f;

        private readonly EnvironmentPreset _from = new();     // look at the moment the last transition started
        private readonly EnvironmentPreset _world = new();    // this frame's weather: the climate or the held preset, zones over it
        private readonly EnvironmentPreset _current = new();  // blended output
        private EnvironmentPreset _held;                      // the Preset the last transition went to (null: the climate)
        private readonly EnvironmentPreset _climate = new();  // the climate's weather this frame (held presets lie over it)
        private bool _climateSampled;
        private float _elapsed;
        private float _duration;
        private bool _primed;
        private bool _driving;
        private readonly List<WeatherZone> _zones = new();
        private static readonly Comparison<WeatherZone> ByPriority = (a, b) => a.Priority.CompareTo(b.Priority);

        // Emitters we touched last frame, so we can reset their runtime overrides when unassigned
        private ParticleEmitter _lastRain, _lastSnow, _lastHail, _lastDust, _lastSplash;

        protected override void Awake()
        {
            Skybox ??= EntityManager.FindComponent<SkyboxRenderer>();
        }

        public override void OnMemberChanged()
        {
            // Inspector changed something — if it was the preset, start a cross-fade to it
            if (Preset != _held)
                TransitionTo(Preset, TransitionDuration);
        }

        /// <summary>
        /// Cross-fade to <paramref name="preset"/> over <paramref name="duration"/> seconds and hold it (null: back to the
        /// climate; negative duration = use <see cref="TransitionDuration"/>, 0 = snap). Safe to call mid-transition:
        /// the fade restarts from whatever is on screen right now, so there is no pop.
        /// </summary>
        public void TransitionTo(EnvironmentPreset preset, float duration = -1f)
        {
            Preset = preset;
            _held = preset;
            _duration = duration < 0f ? TransitionDuration : duration;
            _elapsed = 0f;

            if (_primed)
            {
                // Continue from the current blend. The 2D layer's altitude is cross-faded rather than lerped, so
                // pick whichever layer was dominating on screen as the "from" layer.
                _fromAltitude = _altitudeBlend >= 0.5f ? _toAltitude : _fromAltitude;
                _from.CopyFrom(_current);
            }
        }

        private float _fromAltitude, _toAltitude, _altitudeBlend = 1f;

        public void Update()
        {
            // Disabled or nothing to show: stop driving the sky. Without this the last target kept being
            // re-applied every frame, so an unassigned preset "lingered" and SkyboxRenderer edits were overwritten.
            if (!Enabled || (Preset == null && Climate == null))
            {
                Release();
                return;
            }

            // First frame with a weather (scene load): straight to it, nothing to fade from
            bool first = !_primed;
            if (first)
            {
                _primed = true;
                _held = Preset;
                _duration = 0f;
            }
            else if (Preset != _held)
            {
                // Field was swapped from code without going through TransitionTo
                TransitionTo(Preset, TransitionDuration);
            }

            _driving = true;
            _elapsed += (float)Time.Delta;
            MakeWorld(first);
            if (TransitionProgress < 1f)
                EnvironmentPreset.Lerp(_from, _world, TransitionProgress, _current);
            else
                _current.CopyFrom(_world);

            _toAltitude = _world.CloudAltitude;
            if (first) _fromAltitude = _toAltitude;
            ApplyCurrent();
        }

        // This frame's weather, whatever a running transition still hides: the climate or the held preset, then the zones
        // the camera is in, from the lowest priority up
        private void MakeWorld(bool first)
        {
            // The climate is sampled under a held preset too: the wind's heading and the sea's are always its
            _climateSampled = Climate != null;
            if (_climateSampled)
                Climate.Sample(GameClock.Hours, _climate);
            _world.CopyFrom(Preset ?? _climate);

            var place = Camera.Main?.Position ?? Vector3.Zero;
            _zones.Clear();
            foreach (var zone in ComponentCache<WeatherZone>.All)
                _zones.Add(zone);
            _zones.Sort(ByPriority);
            foreach (var zone in _zones)
            {
                float weight = zone.Step(place, (float)Time.Delta, first);
                if (weight > 0f)
                    EnvironmentPreset.Lerp(_world, zone.Preset, weight, _world);
            }
        }

        /// <summary>For engine_stats: what the weather is made of now.</summary>
        public object Describe()
        {
            var zones = new List<object>();
            foreach (var zone in _zones)
                if (zone.Weight > 0f) zones.Add(new { zone = zone.Entity?.Name, preset = zone.Preset?.Name, weight = zone.Weight });
            var s = Climate?.State ?? default;
            return new
            {
                climate = Climate?.Name,
                held = Preset?.Name,
                // The climate's causes (Climate.Evaluate): season −1 winter … 1 summer, pressure −1 a deep low … 1 a high,
                // what they make of the sky (0..1), the freezing level (m), the wind (m/s) and its turn (°), the sea's wind
                // (m/s) and heading (°), the swell as a share of the authored one and its turn (°)
                causes = _climateSampled ? new
                {
                    s.Season, s.Pressure, s.Overcast, s.Deep, s.Fair, s.Cumulus, s.Thunder, s.Mist, s.FreezingLevel,
                    s.Wind, turn = s.Turn * (180f / MathF.PI), s.SeaWind, seaHeading = s.SeaHeading * (180.0 / Math.PI),
                    s.Swell, swellTurn = s.SwellTurn * (180f / MathF.PI),
                } : null,
                // What is shown: the wind, the sea's wind and swell after held presets and zones
                wind = _current.WindStrength,
                seaWind = _current.SeaWind,
                swell = _current.Swell,
                sea = Graphics.SeaState.Describe(),
                zones,
            };
        }

        private void ApplyCurrent()
        {
            Skybox ??= EntityManager.FindComponent<SkyboxRenderer>();
            if (Skybox != null)
            {
                Skybox.ApplyPreset(_current);

                // The 2D layer's altitude: cross-fade two cloud layers instead of zooming the pattern. The volumetric
                // clouds take the blended height: at the new one's height from the first frame, as this gave both until
                // a change of weather took game hours, they would drop to it as the change began
                _altitudeBlend = TransitionProgress;
                bool layers = !Engine.Settings.VolumetricClouds;
                Skybox.CloudAltitude = layers ? _toAltitude : _current.CloudAltitude;
                Skybox.CloudAltitudeFrom = layers ? _fromAltitude : _current.CloudAltitude;
                Skybox.CloudAltitudeBlend = layers ? _altitudeBlend : 1f;
            }

            // How bright this weather reads: added to the auto exposure's metering
            Graphics.AutoExposure.PresetCompensation = _current.ExposureCompensation;

            // Vegetation sways in the weather's wind (grass, flowers, leaves; Graphics.Wind): its speed; the direction is
            // the generated one turned by the climate's weather systems, which the clouds drift in too; a storm's outflow and
            // the gusts on top (Weather)
            Graphics.Wind.Speed = _current.WindStrength;
            Graphics.Wind.Turn = _climateSampled ? Climate.State.Turn : 0f;
            var wind = Graphics.Wind.Local;

            // The sea: its wind (trailing the climate's, or as a held preset or a zone says), along the climate's trailing
            // heading (the wind's own without a climate), and the swell from afar
            var generated = Graphics.Wind.GeneratedDirection;
            double seaHeading = _climateSampled ? Climate.State.SeaHeading : Math.Atan2(generated.Y, generated.X);
            Graphics.SeaState.Set(_current.SeaWind, seaHeading, _current.Swell, _climateSampled ? Climate.State.SwellTurn : 0f);
            var camPos = Camera.Main?.Position ?? Vector3.Zero;

            // Every other emitter takes the share of it that its WeatherWind asks for (flames, smoke)
            Graphics.ParticleSystem.WeatherWind = DriveWind ? wind : Vector3.Zero;

            // Rain, snow and hail drift in the wind the far rain and snow lean in (Weather.FallWind: steady, capped by how
            // fast each falls), so the near drops and the far field agree
            // None round an eye under the water: the drops would fall through it (Graphics.Underwater)
            float falls = Graphics.Underwater.CameraUnder ? 0f : 1f;
            DriveEmitter(ref _lastRain, RainEmitter, Graphics.Weather.Rain * falls, FallWind(RainEmitter), camPos, EmitterHeight);
            DriveEmitter(ref _lastSnow, SnowEmitter, Graphics.Weather.Snow * falls, FallWind(SnowEmitter), camPos, EmitterHeight);
            DriveEmitter(ref _lastHail, HailEmitter, Graphics.Weather.Hail * falls, FallWind(HailEmitter), camPos, EmitterHeight);
            // ... and the far layers past them hold as many drops, falling as fast (Weather.RainDensity and the rest)
            if (RainEmitter != null)
            {
                Graphics.Weather.RainDensity = FallDensity(RainEmitter);
                Graphics.Weather.RainFallSpeed = FallSpeed(RainEmitter);
            }
            if (SnowEmitter != null)
            {
                Graphics.Weather.SnowDensity = FallDensity(SnowEmitter);
                Graphics.Weather.SnowFallSpeed = FallSpeed(SnowEmitter);
            }
            DriveEmitter(ref _lastSplash, SplashEmitter, Graphics.Weather.Rain * falls, wind, camPos, 0f);
            // The splashes as many as the drops striking their square make (the rain emitter's flux, density × fall speed,
            // × the square's area × SplashShare); at most the splash emitter's EmitRate, which sizes its particle slots
            if (SplashEmitter != null && RainEmitter != null)
            {
                float flux = Graphics.Weather.RainDensity * Graphics.Weather.RainFallSpeed;
                float area = 4f * SplashEmitter.ShapeExtents.X * SplashEmitter.ShapeExtents.Z;
                float rate = flux * area * SplashShare * MathF.Max(0f, Graphics.Weather.Rain) * falls;
                SplashEmitter.EmitRateScale = Math.Clamp(rate / MathF.Max(SplashEmitter.EmitRate, 1e-3f), 0f, 1f);
            }
            // Dust: park it on the terrain under the camera (or a body height below the camera when
            // there is no terrain) so sheets of sand skim the ground instead of floating at eye level.
            _terrain ??= EntityManager.FindComponent<TerrainRenderer>();
            float groundY = _terrain != null ? _terrain.GetHeight(camPos) : camPos.Y - 1.7f;
            var dustAnchor = new Vector3(camPos.X, groundY, camPos.Z);
            DriveEmitter(ref _lastDust, DustEmitter, _current.DustIntensity, wind, dustAnchor, DustHeight);
        }

        private TerrainRenderer _terrain;

        // A weather emitter's fall speed is its terminal one, gravity over drag (rain 9.8 / 1.2 ≈ 8 m/s, snow 2 / 2 = 1)
        private static float FallSpeed(ParticleEmitter emitter) => MathF.Abs(emitter.Gravity.Y) / MathF.Max(emitter.Drag, 0.01f);

        private static Vector3 FallWind(ParticleEmitter emitter) =>
            emitter == null ? Vector3.Zero : Graphics.Weather.FallWind(FallSpeed(emitter));

        // Drops per m³ a weather emitter makes at its full rate: what it emits a second over its box (or disc) falling
        // through at its fall speed
        private static float FallDensity(ParticleEmitter emitter)
        {
            float area = emitter.Shape == EmissionShape.Box
                ? 4f * emitter.ShapeExtents.X * emitter.ShapeExtents.Z
                : MathF.PI * emitter.ShapeRadius * emitter.ShapeRadius;
            return emitter.EmitRate / MathF.Max(area * FallSpeed(emitter), 1e-3f);
        }

        /// <summary>
        /// Drive one emitter through its runtime overrides only. The authored EmitRate / Wind are never
        /// written, so saving the scene mid-storm (or on a clear day) does not bake the weather into the emitter.
        /// </summary>
        private void DriveEmitter(ref ParticleEmitter last, ParticleEmitter emitter, float intensity, Vector3 wind, Vector3 camPos, float height)
        {
            if (last != null && last != emitter)
                ReleaseEmitter(last);
            last = emitter;

            if (emitter == null) return;

            emitter.EmitRateScale = MathF.Max(0f, intensity);
            emitter.WindOverride = DriveWind ? wind : null;

            if (FollowCamera && Camera.Main != null)
            {
                // Particles born above the camera drift downwind while they fall; spawn them upwind
                // so the shower still passes through the view instead of blowing past it.
                var effectiveWind = DriveWind ? wind : emitter.Wind;
                var lead = height > 0f && !emitter.WrapInBox ? -effectiveWind * WindLead : Vector3.Zero;
                emitter.Transform.Position = camPos + Vector3.UnitY * height + lead;
            }
        }

        /// <summary>
        /// Hand the sky back: it keeps the last applied look (now editable on SkyboxRenderer) and weather emitters
        /// return to their authored rates. _primed stays set so a later preset cross-fades from what is on screen.
        /// </summary>
        private void Release()
        {
            if (!_driving) return;
            _driving = false;
            _held = null;
            Graphics.AutoExposure.PresetCompensation = 0f;
            Graphics.ParticleSystem.WeatherWind = Vector3.Zero;
            Graphics.Wind.Speed = Graphics.Wind.DefaultSpeed;
            Graphics.Wind.Turn = 0f;
            Graphics.SeaState.Release();
            if (_lastRain != null) ReleaseEmitter(_lastRain);
            if (_lastSnow != null) ReleaseEmitter(_lastSnow);
            if (_lastHail != null) ReleaseEmitter(_lastHail);
            if (_lastDust != null) ReleaseEmitter(_lastDust);
            if (_lastSplash != null) ReleaseEmitter(_lastSplash);
            _lastRain = _lastSnow = _lastHail = _lastDust = _lastSplash = null;
        }

        private static void ReleaseEmitter(ParticleEmitter emitter)
        {
            emitter.EmitRateScale = 1f;
            emitter.WindOverride = null;
        }

        public override void Destroy()
        {
            if (_driving)
            {
                Graphics.Wind.Turn = 0f;
                Graphics.SeaState.Release();
            }
            if (_lastRain != null) ReleaseEmitter(_lastRain);
            if (_lastSnow != null) ReleaseEmitter(_lastSnow);
            if (_lastHail != null) ReleaseEmitter(_lastHail);
            if (_lastDust != null) ReleaseEmitter(_lastDust);
            if (_lastSplash != null) ReleaseEmitter(_lastSplash);
        }
    }
}
