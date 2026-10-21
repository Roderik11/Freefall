using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using Freefall.Assets;
using NVorbis;
using Vortice.Multimedia;
using Vortice.XAudio2;

namespace Freefall.Audio
{
    /// <summary>
    /// Plays a streamed clip (an .ogg imported with Stream on) flat, with all its channels, the way
    /// music is heard: not placed in the world. The file stays compressed in memory. A thread of the
    /// stream's own decodes it a little ahead of what is heard, into a few short buffers that the voice
    /// plays in turn, and decodes the next one as each finishes. Two seconds are decoded ahead, so a
    /// stall of the main thread (a scene load, a script reload) does not reach the sound.
    ///
    /// <see cref="Play"/> starts one; <see cref="Dispose"/> stops it and frees its voice, and has to be
    /// called by whoever started it, also after it has played to the end (<see cref="IsPlaying"/>). A
    /// stream plays its clip once: the next track is another stream.
    /// </summary>
    public sealed class AudioStream : IDisposable
    {
        const int BufferCount = 4;
        const float BufferSeconds = 0.5f;

        readonly VorbisReader _reader;
        readonly IXAudio2SourceVoice _voice;
        readonly nint[] _buffers = new nint[BufferCount];
        readonly float[] _decoded;
        readonly int _sampleRate;
        readonly int _channels;
        readonly Thread _thread;
        readonly AutoResetEvent _bufferEnded = new(false);
        volatile bool _stopping;
        volatile bool _allSubmitted;
        float _volume;
        bool _disposed;

        public AudioClip Clip { get; }

        /// <summary>How loud, 1 = as the file is. Fades are made by changing it every frame.</summary>
        public float Volume
        {
            get => _volume;
            set
            {
                _volume = value;
                if (!_disposed) _voice.SetVolume(value);
            }
        }

        /// <summary>Seconds heard so far.</summary>
        public float Position => _disposed ? 0f : (float)((double)_voice.State.SamplesPlayed / _sampleRate);

        /// <summary>Seconds it lasts.</summary>
        public float Length => Clip.Length;

        /// <summary>Until the last of it has been heard, or it was disposed.</summary>
        public bool IsPlaying => !_disposed && !(_allSubmitted && _voice.StateNoSamplesPlayed.BuffersQueued == 0);

        /// <summary>
        /// Starts playing a streamed clip at a volume. Null when it cannot be played: no audio device, or
        /// a clip that is not streamed (a warning says so).
        /// </summary>
        public static AudioStream Play(AudioClip clip, float volume = 1f)
        {
            if (clip == null || Engine.AudioDevice == null) return null;

            // A clip referenced from an asset is a stub with its GUID until it is loaded
            if (!clip.IsStreamed && clip.WaveFormat == null && !string.IsNullOrEmpty(clip.Guid) && Engine.Assets != null)
                clip = Engine.Assets.LoadByGuid<AudioClip>(clip.Guid) ?? clip;

            if (!clip.IsStreamed)
            {
                Debug.LogWarning("[AudioStream]", $"'{clip.Name}' is not streamed: turn Stream on in its import settings (an .ogg)");
                return null;
            }

            try
            {
                return new AudioStream(clip, volume);
            }
            catch (Exception e)
            {
                Debug.LogWarning("[AudioStream]", $"'{clip.Name}' cannot be played: {e.Message}");
                return null;
            }
        }

        AudioStream(AudioClip clip, float volume)
        {
            Clip = clip;
            _volume = volume;
            _reader = new VorbisReader(new MemoryStream(clip.StreamData, writable: false), closeOnDispose: true);
            _sampleRate = _reader.SampleRate;
            _channels = _reader.Channels;
            _decoded = new float[(int)(_sampleRate * BufferSeconds) * _channels];

            try
            {
                for (int i = 0; i < BufferCount; i++)
                    _buffers[i] = Marshal.AllocHGlobal(_decoded.Length * sizeof(float));

                _voice = Engine.AudioDevice.CreateSourceVoice(WaveFormat.CreateIeeeFloatWaveFormat(_sampleRate, _channels), true);
            }
            catch
            {
                FreeBuffers();
                _reader.Dispose();
                throw;
            }

            _voice.BufferEnd += OnBufferEnd;
            _voice.SetVolume(volume);

            // Started empty, it sounds from the first buffer the thread queues
            _voice.Start();
            _thread = new Thread(Feed) { IsBackground = true, Name = "AudioStream " + clip.Name, Priority = ThreadPriority.AboveNormal };
            _thread.Start();
        }

        // On XAudio2's thread: only wakes the decoding thread
        void OnBufferEnd(nint context) => _bufferEnded.Set();

        void Feed()
        {
            int next = 0;
            try
            {
                while (!_stopping && !_allSubmitted)
                {
                    // Buffers end in the order they were queued, so with fewer than all queued the next one is free
                    while (!_stopping && !_allSubmitted && _voice.StateNoSamplesPlayed.BuffersQueued < BufferCount)
                    {
                        Submit(_buffers[next]);
                        next = (next + 1) % BufferCount;
                    }
                    _bufferEnded.WaitOne(250);
                }
            }
            catch (Exception e)
            {
                // A damaged file: what was queued plays out
                Debug.LogWarning("[AudioStream]", $"'{Clip.Name}' stopped decoding: {e.Message}");
                _allSubmitted = true;
            }
        }

        void Submit(nint buffer)
        {
            int count = 0;
            while (count < _decoded.Length)
            {
                int read = _reader.ReadSamples(_decoded, count, _decoded.Length - count);
                if (read <= 0) break;
                count += read;
            }

            bool last = count < _decoded.Length;

            // The end fell on a buffer's edge: one frame of silence carries the end mark
            if (count == 0)
            {
                Array.Clear(_decoded, 0, _channels);
                count = _channels;
            }

            Marshal.Copy(_decoded, 0, buffer, count);
            _voice.SubmitSourceBuffer(new AudioBuffer(buffer, (uint)(count * sizeof(float)), last ? BufferFlags.EndOfStream : BufferFlags.None), (uint[])null);
            if (last) _allSubmitted = true;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _stopping = true;
            _bufferEnded.Set();
            _thread.Join();

            _voice.BufferEnd -= OnBufferEnd;
            _voice.Stop();

            // Returns once the voice is done with the buffers, so they can be freed after it
            _voice.DestroyVoice();
            _voice.Dispose();

            FreeBuffers();
            _reader.Dispose();
            _bufferEnded.Dispose();
        }

        void FreeBuffers()
        {
            for (int i = 0; i < BufferCount; i++)
            {
                if (_buffers[i] != 0) Marshal.FreeHGlobal(_buffers[i]);
                _buffers[i] = 0;
            }
        }
    }
}
