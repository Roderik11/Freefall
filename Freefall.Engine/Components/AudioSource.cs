using Vortice.Multimedia;
using Vortice.XAudio2;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;

namespace Freefall.Components
{
    public class AudioSource : Component, IUpdate
    {
        public AudioClip AudioClip;
        public float Volume = 1;
        public float Range = 10;
        public float MinDistance = 0;

        public bool Loop;
        public bool PlayOnAwake;

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

            if (sourceVoice == null || currentFormat != clip.WaveFormat)
            {
                sourceVoice?.BufferEnd -= SourceVoice_BufferEnd;
                sourceVoice?.DestroyVoice();
                sourceVoice?.Dispose();

                sourceVoice = Engine.AudioDevice.CreateSourceVoice(clip.WaveFormat, true);
                if (loop) sourceVoice.BufferEnd += SourceVoice_BufferEnd;

                currentFormat = clip.WaveFormat;
            }

            currentClip = clip;

            sourceVoice.FlushSourceBuffers();
            sourceVoice.SubmitSourceBuffer(clip.Buffer, clip.DecodedPacketsInfo);
            sourceVoice.SetVolume(Volume);
            sourceVoice.Start();
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
            sourceVoice?.DestroyVoice();
            sourceVoice?.Dispose();
        }

        public void Update()
        {
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
            emitter.OrientFront = Transform.Forward;
            emitter.OrientTop = Transform.Up;
            emitter.ChannelRadius = Range;
            emitter.InnerRadius = MinDistance;

            if (sourceVoice?.State.BuffersQueued == 0)
                return;

            int sourceChannels = currentFormat?.Channels ?? 1;
            var dspSettings = Engine.Audio3D.Calculate(listener, emitter, CalculateFlags.Matrix | CalculateFlags.Doppler, (uint)sourceChannels, 2);
            sourceVoice?.SetOutputMatrix((uint)sourceChannels, 2, dspSettings.MatrixCoefficients);
            sourceVoice?.SetFrequencyRatio(dspSettings.DopplerFactor, 0);
            sourceVoice?.SetVolume(Volume);
        }
    }
}
