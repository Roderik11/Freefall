using System;
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
    /// Pick a <see cref="Preset"/> in the inspector (or call <see cref="TransitionTo"/> from code) and the
    /// controller cross-fades from whatever is currently shown to the new preset over
    /// <see cref="TransitionDuration"/> seconds. The blended result is pushed every frame into:
    ///   • <see cref="SkyboxRenderer"/> via <c>ApplyPreset</c> (sun colors, atmosphere, clouds, stars)
    ///   • <c>Engine.Settings.FogDensity</c>
    ///   • the optional Rain / Snow / Hail / Dust <see cref="ParticleEmitter"/>s (emit rate scaled by
    ///     the preset's per-type intensity, wind copied to the emitter, emitter parked above the camera)
    ///
    /// Time of day stays on the <see cref="SkyboxRenderer"/> — a preset describes how a day looks,
    /// not what time it is. Precipitation is a set of per-type intensities rather than an enum so a
    /// rain → snow transition simply fades one out while the other fades in.
    /// </summary>
    [Icon("icon_sky.png")]
    [UpdateInEditor]
    public class EnvironmentController : Component, IUpdate
    {
        [Category("Preset")]
        [Description("Environment preset to show. Changing it starts a cross-fade from the current look.")]
        public EnvironmentPreset Preset;

        [Description("Seconds a preset change takes to fully blend in")]
        [ValueRange(0f, 120f)]
        public float TransitionDuration = 5f;

        [Description("Skybox to drive. Found automatically when left empty.")]
        public SkyboxRenderer Skybox;

        [Description("Push the preset's fog density into Engine.Settings.FogDensity")]
        public bool DriveFog = true;

        [Category("Weather Emitters")]
        [Description("Emitter scaled by RainIntensity. Its EmitRate is treated as the rate at intensity 1.")]
        public ParticleEmitter RainEmitter;

        [Description("Emitter scaled by SnowIntensity")]
        public ParticleEmitter SnowEmitter;

        [Description("Emitter scaled by HailIntensity")]
        public ParticleEmitter HailEmitter;

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

        [Description("Use the preset's wind for the emitters instead of their authored Wind")]
        public bool DriveWind = true;

        [Description("Seconds of wind drift to lead by: emitters are shifted upwind by Wind * WindLead so blown particles still arrive at the camera")]
        [ValueRange(0f, 5f)]
        public float WindLead = 1.0f;

        // ── Runtime state ──

        /// <summary>The blended preset currently being applied. Read-only snapshot, re-filled every frame.</summary>
        public EnvironmentPreset Current => _current;

        /// <summary>0 → 1 progress of the running transition (1 when idle).</summary>
        public float TransitionProgress => _duration <= 0f ? 1f : Math.Clamp(_elapsed / _duration, 0f, 1f);

        public bool IsTransitioning => TransitionProgress < 1f;

        private readonly EnvironmentPreset _from = new();     // look at the moment the last transition started
        private readonly EnvironmentPreset _current = new();  // blended output
        private EnvironmentPreset _target;                    // preset asset we are heading toward
        private float _elapsed;
        private float _duration;
        private bool _primed;

        // Emitters we touched last frame, so we can reset their runtime overrides when unassigned
        private ParticleEmitter _lastRain, _lastSnow, _lastHail, _lastDust;

        protected override void Awake()
        {
            Skybox ??= EntityManager.FindComponent<SkyboxRenderer>();
        }

        public override void OnMemberChanged()
        {
            // Inspector changed something — if it was the preset, start a cross-fade to it
            if (Preset != _target)
                TransitionTo(Preset, TransitionDuration);
        }

        /// <summary>
        /// Cross-fade to <paramref name="preset"/> over <paramref name="duration"/> seconds
        /// (negative = use <see cref="TransitionDuration"/>, 0 = snap). Safe to call mid-transition:
        /// the fade restarts from whatever is on screen right now, so there is no pop.
        /// </summary>
        public void TransitionTo(EnvironmentPreset preset, float duration = -1f)
        {
            if (preset == null) return;

            Preset = preset;
            _target = preset;
            _duration = duration < 0f ? TransitionDuration : duration;
            _elapsed = 0f;

            if (_primed)
            {
                // Continue from the current blend. Altitude is cross-faded rather than lerped, so pick
                // whichever layer was dominating on screen as the "from" layer.
                _fromAltitude = _altitudeBlend >= 0.5f ? _toAltitude : _fromAltitude;
                _from.CopyFrom(_current);
            }
            else
            {
                _from.CopyFrom(preset);         // first preset ever: snap, nothing to fade from
                _fromAltitude = preset.CloudAltitude;
            }

            _toAltitude = preset.CloudAltitude;
            _primed = true;
        }

        private float _fromAltitude, _toAltitude, _altitudeBlend = 1f;

        public void Update()
        {
            // Disabled or preset unassigned: stop driving the sky. Without this the last target kept being
            // re-applied every frame, so an unassigned preset "lingered" and SkyboxRenderer edits were overwritten.
            if (!Enabled || Preset == null)
            {
                Release();
                return;
            }

            if (!_primed)
            {
                if (Preset == null) return;
                // First frame with a preset (scene load) — snap straight to it
                TransitionTo(Preset, 0f);
            }
            else if (Preset != _target)
            {
                // Field was swapped from code without going through TransitionTo
                TransitionTo(Preset, TransitionDuration);
            }

            if (_target == null) return;

            _elapsed += (float)Time.Delta;
            EnvironmentPreset.Lerp(_from, _target, TransitionProgress, _current);

            ApplyCurrent();
        }

        private void ApplyCurrent()
        {
            Skybox ??= EntityManager.FindComponent<SkyboxRenderer>();
            if (Skybox != null)
            {
                Skybox.ApplyPreset(_current);

                // Altitude: cross-fade two cloud layers instead of zooming the pattern
                _altitudeBlend = TransitionProgress;
                Skybox.CloudAltitude = _toAltitude;
                Skybox.CloudAltitudeFrom = _fromAltitude;
                Skybox.CloudAltitudeBlend = _altitudeBlend;
            }

            if (DriveFog)
                Engine.Settings.FogDensity = _current.FogDensity;

            var wind = _current.WindVector;
            var camPos = Camera.Main?.Position ?? Vector3.Zero;

            DriveEmitter(ref _lastRain, RainEmitter, _current.RainIntensity, wind, camPos, EmitterHeight);
            DriveEmitter(ref _lastSnow, SnowEmitter, _current.SnowIntensity, wind, camPos, EmitterHeight);
            DriveEmitter(ref _lastHail, HailEmitter, _current.HailIntensity, wind, camPos, EmitterHeight);
            // Dust: park it on the terrain under the camera (or a body height below the camera when
            // there is no terrain) so sheets of sand skim the ground instead of floating at eye level.
            _terrain ??= EntityManager.FindComponent<TerrainRenderer>();
            float groundY = _terrain != null ? _terrain.GetHeight(camPos) : camPos.Y - 1.7f;
            var dustAnchor = new Vector3(camPos.X, groundY, camPos.Z);
            DriveEmitter(ref _lastDust, DustEmitter, _current.DustIntensity, wind, dustAnchor, DustHeight);
        }

        private TerrainRenderer _terrain;

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
                var lead = height > 0f ? -effectiveWind * WindLead : Vector3.Zero;
                emitter.Transform.Position = camPos + Vector3.UnitY * height + lead;
            }
        }

        /// <summary>
        /// Hand the sky back: it keeps the last applied look (now editable on SkyboxRenderer) and weather emitters
        /// return to their authored rates. _primed stays set so a later preset cross-fades from what is on screen.
        /// </summary>
        private void Release()
        {
            if (_target == null) return;
            _target = null;
            if (_lastRain != null) ReleaseEmitter(_lastRain);
            if (_lastSnow != null) ReleaseEmitter(_lastSnow);
            if (_lastHail != null) ReleaseEmitter(_lastHail);
            if (_lastDust != null) ReleaseEmitter(_lastDust);
            _lastRain = _lastSnow = _lastHail = _lastDust = null;
        }

        private static void ReleaseEmitter(ParticleEmitter emitter)
        {
            emitter.EmitRateScale = 1f;
            emitter.WindOverride = null;
        }

        public override void Destroy()
        {
            if (_lastRain != null) ReleaseEmitter(_lastRain);
            if (_lastSnow != null) ReleaseEmitter(_lastSnow);
            if (_lastHail != null) ReleaseEmitter(_lastHail);
            if (_lastDust != null) ReleaseEmitter(_lastDust);
        }
    }
}
