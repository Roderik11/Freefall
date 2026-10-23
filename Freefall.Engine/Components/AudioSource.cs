using System;
using Freefall.Assets;
using Freefall.Audio;
using Freefall.Base;

namespace Freefall.Components
{
    /// <summary>Where an <see cref="AudioSource"/> sounds from. The spline is the entity's own, or its parent's.</summary>
    public enum AudioShape
    {
        /// <summary>The entity's place.</summary>
        Point = 0,

        /// <summary>Along the spline, open or closed, from its point nearest the listener: a river (from its near bank), a coastline, a lake's shore.</summary>
        Along = 1,

        /// <summary>A closed spline: all around the listener inside it (a forest, a marsh), from its edge outside. An open one is heard as Along.</summary>
        Area = 2,

        /// <summary>All around the listener wherever it is, the clip's own channels as they are: the rain, the weather. No place, no distance.</summary>
        Around = 3,
    }

    /// <summary>
    /// A sound in the world: what it plays and from where. Data only: <see cref="AudioSourceSystem"/>
    /// gives the few the listener hears best a voice and plays the rest on without one, so a source
    /// costs nothing while it is silent or out of hearing, and a looped one (a fire) stays in step.
    /// </summary>
    [Icon("icon_audio.png")]
    public class AudioSource : Component
    {
        [System.ComponentModel.Category("Playback")]
        public AudioClip AudioClip;
        public float Volume = 1;

        /// <summary>Plays faster and higher (above 1) or slower and lower. Set before Play to vary a sound that repeats.</summary>
        public float Pitch = 1;

        /// <summary><see cref="Play()"/> plays the clip on and on, until <see cref="Stop"/>.</summary>
        public bool Loop;
        public bool PlayOnAwake;

        [System.ComponentModel.Category("Spatial")]
        public float Range = 10;
        public float MinDistance = 0;

        /// <summary>
        /// Within this many metres of the listener the sound is at full volume; beyond it the
        /// volume is this over the distance (half at twice as far). 1 by default. A listener on
        /// a camera that follows a character from a few metres away hears that character's own
        /// sounds at a fraction unless this reaches the camera.
        /// </summary>
        public float FalloffStart = 1;

        /// <summary>
        /// Further than this from the listener it is not heard, in metres; it fades out over the
        /// last fifth of the way. 0 = no limit: only the distance makes it quieter.
        /// </summary>
        public float HeardWithin = 0;

        /// <summary>
        /// Along or Area: sounds along the entity's Spline (or its parent's, so a day and a night sound can
        /// share one area) instead of from its place. Along is heard from the spline's point nearest the
        /// listener, a river from its near bank (<see cref="Width"/>); Area all around the listener inside
        /// it, and from the nearest point of its edge outside. Near that point (within FalloffStart) the
        /// clip's own channels come through as they are, a stereo bed around the listener, narrowing to
        /// one point further off.
        /// </summary>
        public AudioShape Shape = AudioShape.Point;

        /// <summary>
        /// Along: how wide its sound is in metres (a river's water), so it is heard from the near bank.
        /// The Spline's Widths scale it along the way.
        /// </summary>
        public float Width = 0;

        /// <summary>Plays (or plays on without a voice, out of hearing).</summary>
        [Freefall.Reflection.DontSerialize]
        public bool IsPlaying => Entry?.PlayingIndex >= 0;

        /// <summary>Has a voice: is one of the sounds heard just now.</summary>
        [Freefall.Reflection.DontSerialize]
        public bool IsHeard => Entry?.Voice != null;

        /// <summary>Where it was last heard from: its place, or its spline's point nearest the listener.</summary>
        [Freefall.Reflection.DontSerialize]
        public System.Numerics.Vector3 HeardFrom => Entry?.Position ?? Transform?.WorldPosition ?? default;

        /// <summary>What the system keeps of it while it plays.</summary>
        internal AudioSourceSystem.Entry? Entry;

        /// <summary>Tells the system it moved, while it plays (subscribed to the Transform then only).</summary>
        internal readonly Action OnMoved;

        private bool warnedStreamed;

        public AudioSource()
        {
            OnMoved = () => Systems.Get<AudioSourceSystem>()?.Moved(this);
        }

        protected override void Awake()
        {
            if (PlayOnAwake) Play();
        }

        /// <summary>Plays <see cref="AudioClip"/>, looped if <see cref="Loop"/> is on.</summary>
        public void Play()
        {
            Play(AudioClip, Loop);
        }

        /// <summary>Plays a clip once, from the start.</summary>
        public void Play(AudioClip clip)
        {
            Play(clip, false);
        }

        private void Play(AudioClip clip, bool loop)
        {
            if (clip == null) return;
            if (Engine.AudioDevice == null) return;

            // Resolve deferred stub: has GUID but no audio data loaded yet
            if (clip.WaveFormat == null && !clip.IsStreamed && !string.IsNullOrEmpty(clip.Guid) && Engine.Assets != null)
            {
                var loaded = Engine.Assets.LoadByGuid<AudioClip>(clip.Guid);
                if (loaded == null) return;
                if (clip == AudioClip) AudioClip = loaded;
                clip = loaded;
            }

            // Kept compressed for an AudioStream, which plays it flat; a source places one held whole
            if (clip.IsStreamed)
            {
                if (!warnedStreamed)
                    Debug.LogWarning("[AudioSource]", $"'{clip.Name}' is imported with Stream on: an AudioStream plays it, not an AudioSource");
                warnedStreamed = true;
                return;
            }
            if (clip.WaveFormat == null || clip.Buffer == null) return;

            Systems.Get<AudioSourceSystem>()?.Play(this, clip, loop);
        }

        public void Stop()
        {
            Systems.Get<AudioSourceSystem>()?.Stop(this);
        }

        public override void OnMemberChanged()
        {
            // A loop that plays from the start is heard as it is set up (in the inspector, from MCP): it starts,
            // or starts again with another clip, when its settings change
            if (Loop && PlayOnAwake && AudioClip != null && (Entry == null || Entry.PlayingIndex < 0 || Entry.Clip != AudioClip))
                Play();
            else
                Systems.Get<AudioSourceSystem>()?.Changed(this);
        }

        protected override void OnEnabledChanged() => Systems.Get<AudioSourceSystem>()?.EnabledChanged(this);

        public override void Destroy()
        {
            Stop();
            base.Destroy();
        }
    }
}
