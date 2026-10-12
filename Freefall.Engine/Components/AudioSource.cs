using System;
using Vortice.Multimedia;
using Vortice.XAudio2;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;

namespace Freefall.Components
{
    [Icon("icon_audio.png")]
    public class AudioSource : Component, IUpdate
    {
        [System.ComponentModel.Category("Playback")]
        public AudioClip AudioClip;
        public float Volume = 1;

        /// <summary>Plays faster and higher (above 1) or slower and lower. Set before Play to vary a sound that repeats.</summary>
        public float Pitch = 1;
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

        private Emitter emitter;
        private IXAudio2SourceVoice sourceVoice;
        private WaveFormat currentFormat;
        private AudioClip currentClip;
        private bool firstUpdate = true;

        public AudioSource() { }

        protected override void Awake()
        {
            emitter = new Emitter
            {
                InnerRadius = MinDistance,
                ChannelRadius = Range,
                ChannelCount = 1,
                CurveDistanceScaler = 1,
                Position = Transform.WorldPosition,
                OrientFront = Transform.Forward,
                OrientTop = Transform.Up,
                Velocity = Vector3.Zero,
            };
        }

        public void Play()
        {
            Play(AudioClip, Loop);
        }

        public void Play(AudioClip clip)
        {
            Play(clip, false);
        }

        private void Play(AudioClip clip, bool loop)
        {
            if (clip == null) return;
            if (Engine.AudioDevice == null) return;

            // Resolve deferred stub: has GUID but no audio data loaded yet
            if (clip.WaveFormat == null && !string.IsNullOrEmpty(clip.Guid) && Engine.Assets != null)
            {
                var loaded = Engine.Assets.LoadByGuid<AudioClip>(clip.Guid);
                if (loaded != null)
                {
                    AudioClip = loaded;
                    clip = loaded;
                }
                else return;
            }
            if (clip.WaveFormat == null) return;

            if (sourceVoice == null || currentFormat != clip.WaveFormat)
            {
                sourceVoice?.BufferEnd -= SourceVoice_BufferEnd;
                sourceVoice?.DestroyVoice();
                sourceVoice?.Dispose();
                sourceVoice = null;

                if (clip.WaveFormat == null) return;

                try
                {
                    sourceVoice = Engine.AudioDevice.CreateSourceVoice(clip.WaveFormat, true);
                    if (loop) sourceVoice.BufferEnd += SourceVoice_BufferEnd;
                }
                catch
                {
                    return;
                }

                currentFormat = clip.WaveFormat;
            }

            currentClip = clip;

            sourceVoice.FlushSourceBuffers();
            sourceVoice.SubmitSourceBuffer(clip.Buffer, clip.DecodedPacketsInfo);
            sourceVoice.SetVolume(Volume);

            // Where it is heard from, before the first sample: until the next Update the voice
            // would play at full volume in the middle, however far away it is
            if (emitter != null)
            {
                emitter.Position = Transform.WorldPosition;
                emitter.Velocity = Vector3.Zero;
                Place();
            }

            sourceVoice.Start();
        }

        // Volume per speaker and pitch from where the listener is
        private void Place()
        {
            var listener = AudioListener.Main?.Listener;
            if (listener == null || sourceVoice == null) return;

            emitter.OrientFront = Transform.Forward;
            emitter.OrientTop = Transform.Up;
            emitter.ChannelRadius = Range;
            emitter.InnerRadius = MinDistance;
            emitter.CurveDistanceScaler = MathF.Max(FalloffStart, 0.01f);

            int sourceChannels = currentFormat?.Channels ?? 1;
            var dspSettings = Engine.Audio3D.Calculate(listener, emitter, CalculateFlags.Matrix | CalculateFlags.Doppler, (uint)sourceChannels, 2);
            sourceVoice.SetOutputMatrix((uint)sourceChannels, 2, dspSettings.MatrixCoefficients);
            sourceVoice.SetFrequencyRatio(dspSettings.DopplerFactor * Pitch, 0);
            sourceVoice.SetVolume(Volume);
        }

        private void SourceVoice_BufferEnd(System.IntPtr obj)
        {
            if (!Loop) return;
            if (currentClip == null) return;

            sourceVoice?.SubmitSourceBuffer(currentClip.Buffer, currentClip.DecodedPacketsInfo);
            sourceVoice?.Start();
        }

        public void Stop()
        {
            sourceVoice?.Stop();
        }

        public override void Destroy()
        {
            if (sourceVoice != null)
                sourceVoice.BufferEnd -= SourceVoice_BufferEnd;
            sourceVoice?.DestroyVoice();
            sourceVoice?.Dispose();
        }

        public void Update()
        {
            // Not awake yet: a source added while this frame's updates are running can get its
            // first Update before its Awake, and there is nothing to place a sound with until then
            if (emitter == null) return;

            var listener = AudioListener.Main?.Listener;
            if (listener == null) return;

            if (firstUpdate && PlayOnAwake)
            {
                Play();
                firstUpdate = false;
            }

            if (sourceVoice == null) return;

            var newpos = Transform.WorldPosition;
            emitter.Velocity = Time.SmoothDelta > 0 ? (newpos - emitter.Position) / Time.SmoothDelta : Vector3.Zero;
            emitter.Position = newpos;

            if (sourceVoice.State.BuffersQueued == 0)
                return;

            Place();
        }
    }
}
