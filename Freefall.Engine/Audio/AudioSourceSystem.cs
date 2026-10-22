using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Vortice.Multimedia;
using Vortice.XAudio2;

namespace Freefall.Audio
{
    /// <summary>
    /// Plays every <see cref="AudioSource"/> from one small set of voices. A source is data: it says
    /// what it plays and from where; the few the listener hears best get a voice, the rest play on
    /// without one, in step, so a looped sound that comes within hearing is where it would have been.
    ///
    /// A source that plays is placed in a grid of <see cref="CellSize"/> cells by where it is (moved
    /// ones tell, through their Transform), and every <see cref="SelectEvery"/> seconds the cells around
    /// the listener are asked which sources are within hearing and how loud each is there; the loudest
    /// <see cref="MaxVoices"/> have voices. A one-shot gets its voice when it is played, from a free one
    /// or the quietest heard, or not at all. Per frame only the voices are worked on: where the listener
    /// hears each from, its volume and pitch, whether a one-shot has ended. A source that is not playing
    /// costs nothing.
    /// </summary>
    [UpdateInEditor]
    public sealed class AudioSourceSystem : EntitySystem
    {
        /// <summary>How many sounds are heard at once, at most.</summary>
        public static int MaxVoices = 32;

        /// <summary>Metres a side of a cell of the grid sources are found by.</summary>
        public const float CellSize = 32f;

        /// <summary>Seconds between two choices of what is heard.</summary>
        public const float SelectEvery = 0.2f;

        // A looped sound that gets or loses its voice fades in or out over this long
        const float FadeTime = 0.3f;

        // A voice that is heard keeps it against one up to this much louder (loudness squared: x2 = 3 dB),
        // so two fires about as far off do not swap as the listener moves between them
        const float Keep = 2f;

        // A one-shot that is already this far through is not given a voice late
        const float LateOneShot = 0.5f;

        // The last share of HeardWithin a sound fades out over (squared: from 0.8 of the way)
        const float EdgeStart = 0.64f;

        /// <summary>What the system keeps of a source that plays.</summary>
        internal sealed class Entry
        {
            public AudioSource Source = null!;
            public AudioClip Clip = null!;
            public bool Looping;
            public float Started;           // Time.TotalTime when it was played
            public Vector3 Position;
            public bool Moved;
            public long Cell = long.MinValue; // in the grid, or in the list of sources heard everywhere
            public bool Everywhere;
            public Voice? Voice;
            public float Loud;              // loudness squared at the listener, at the last choice
            public float Score;             // the same, with the edge a source that is heard keeps (Keep)
            public int PlayingIndex = -1;   // in _playing
            public int Chosen;              // the choice it was among the heard at
        }

        internal sealed class Voice
        {
            public IXAudio2SourceVoice Native = null!;
            public WaveFormat Format = null!;
            public Entry? Owner;
            public float Fade = 1f;         // 0..1
            public float FadeTo = 1f;
            public Vector3 LastPosition;
            public readonly Emitter Emitter = new() { ChannelCount = 1, CurveDistanceScaler = 1 };
            public readonly DspSettings Dsp = new(1, 2);
            public float[]? Folded;
        }

        private readonly List<Entry> _playing = [];                 // every source that plays
        private readonly Dictionary<long, List<Entry>> _cells = [];
        private readonly List<Entry> _everywhere = [];              // HeardWithin 0: no limit
        private readonly List<Entry> _moved = [];
        private readonly List<Voice> _voices = [];
        private readonly List<Entry> _candidates = [];
        private readonly List<Entry> _ended = [];                   // one-shots found played out during a choice
        private float _farthest;                                    // the longest HeardWithin of a source in the grid
        private float _sinceSelect = SelectEvery;
        private int _choice;

        private static readonly Comparison<Entry> Louder = (a, b) => b.Score.CompareTo(a.Score);

        /// <summary>Voices in use.</summary>
        public int Heard { get; private set; }

        /// <summary>Sources that play, heard or not.</summary>
        public int Playing => _playing.Count;

        public override void Update()
        {
            if (Engine.AudioDevice == null || Engine.Audio3D == null) return;
            var listener = AudioListener.Main?.Listener;

            _sinceSelect += Time.Delta;
            if (_sinceSelect >= SelectEvery && listener != null)
            {
                _sinceSelect = 0f;
                Select(listener);
            }

            int heard = 0;
            for (int i = 0; i < _voices.Count; i++)
            {
                var voice = _voices[i];
                var entry = voice.Owner;
                if (entry == null) continue;

                var source = entry.Source;

                // A one-shot that has played to its end
                if (!entry.Looping && voice.Native.StateNoSamplesPlayed.BuffersQueued == 0)
                {
                    Finish(entry);
                    continue;
                }

                // Faded out: the voice is free again, the source plays on without one
                if (voice.FadeTo < voice.Fade)
                {
                    voice.Fade = MathF.Max(voice.FadeTo, voice.Fade - Time.Delta / FadeTime);
                    if (voice.Fade <= 0f)
                    {
                        Release(voice);
                        continue;
                    }
                }
                else if (voice.FadeTo > voice.Fade)
                    voice.Fade = MathF.Min(voice.FadeTo, voice.Fade + Time.Delta / FadeTime);

                heard++;
                if (listener != null && source.Transform != null)
                    Place(voice, entry, listener, source.Transform.WorldPosition, Time.SmoothDelta);
            }
            Heard = heard;
        }

        // ---- Playing and stopping (from AudioSource) ----

        internal void Play(AudioSource source, AudioClip clip, bool loop)
        {
            if (source.Entity == null || source.Transform == null) return;
            var entry = source.Entry;
            if (entry == null)
            {
                entry = new Entry { Source = source };
                source.Entry = entry;
            }
            else if (entry.Voice != null)
                Release(entry.Voice);

            entry.Clip = clip;
            entry.Looping = loop;

            // A loop starts at a point of its clip drawn from the source: forty braziers that wake in one frame
            // with the same clip would otherwise play in step, one sound heard from forty places
            float into = loop ? (float)((source.UID * 0x9E3779B97F4A7C15UL) >> 40) / (1 << 24) * clip.Length : 0f;
            entry.Started = Time.TotalTime - into;
            entry.Position = source.Transform.WorldPosition;
            entry.Moved = false;

            if (entry.PlayingIndex < 0)
            {
                entry.PlayingIndex = _playing.Count;
                _playing.Add(entry);
                Insert(entry);
                source.Transform.OnChanged += source.OnMoved;
            }
            else
                Rebin(entry);

            // Heard from the start if it is loud enough: a free voice, or the quietest one heard
            var listener = AudioListener.Main?.Listener;
            if (listener == null || !source.Enabled) return;

            entry.Loud = entry.Score = Loudness(entry, listener.Position);
            if (entry.Loud <= 0f) return;

            var voice = FreeVoice(clip.WaveFormat);
            if (voice == null)
            {
                var quietest = Quietest();
                if (quietest?.Owner == null) return;
                if (quietest.FadeTo > 0f && quietest.Owner.Loud * Keep >= entry.Loud) return;
                Release(quietest);
                voice = FreeVoice(clip.WaveFormat);
                if (voice == null) return;
            }
            Start(voice, entry, listener, fadeIn: false);
        }

        internal void Stop(AudioSource source)
        {
            if (source.Entry != null) Finish(source.Entry);
        }

        internal void Moved(AudioSource source)
        {
            var entry = source.Entry;
            if (entry == null || entry.Moved || entry.PlayingIndex < 0) return;
            entry.Moved = true;
            _moved.Add(entry);
        }

        /// <summary>
        /// Enabled or a member changed: a source switched off goes quiet now, one switched on is heard
        /// from the next choice; a changed HeardWithin moves it in the grid.
        /// </summary>
        internal void Changed(AudioSource source)
        {
            var entry = source.Entry;
            if (entry == null) return;
            if (entry.Voice != null && !source.Enabled) Release(entry.Voice);
            if (entry.PlayingIndex >= 0) Rebin(entry);
        }

        // Stopped, ended or destroyed: out of the grid, no voice
        private void Finish(Entry entry)
        {
            if (entry.Voice != null) Release(entry.Voice);
            if (entry.PlayingIndex < 0) return;

            Remove(entry);
            int last = _playing.Count - 1;
            var moved = _playing[last];
            _playing[entry.PlayingIndex] = moved;
            moved.PlayingIndex = entry.PlayingIndex;
            _playing.RemoveAt(last);
            entry.PlayingIndex = -1;
            entry.Moved = false;
            if (entry.Source.Transform != null)
                entry.Source.Transform.OnChanged -= entry.Source.OnMoved;
        }

        // ---- What is heard ----

        private void Select(Listener listener)
        {
            // Sources that moved since the last choice: where they are now, and in which cell
            for (int i = 0; i < _moved.Count; i++)
            {
                var entry = _moved[i];
                if (!entry.Moved) continue;
                entry.Moved = false;
                if (entry.Source.Transform != null)
                {
                    entry.Position = entry.Source.Transform.WorldPosition;
                    Rebin(entry);
                }
            }
            _moved.Clear();

            var at = listener.Position;
            _candidates.Clear();
            for (int i = 0; i < _everywhere.Count; i++)
                Consider(_everywhere[i], at);

            if (_cells.Count > 0)
            {
                int reach = (int)MathF.Ceiling(_farthest / CellSize);
                int cx = (int)MathF.Floor(at.X / CellSize), cz = (int)MathF.Floor(at.Z / CellSize);
                for (int z = cz - reach; z <= cz + reach; z++)
                    for (int x = cx - reach; x <= cx + reach; x++)
                        if (_cells.TryGetValue(Key(x, z), out var cell))
                            for (int i = 0; i < cell.Count; i++)
                                Consider(cell[i], at);
            }

            // One-shots that played out unheard, now that no list is being walked
            for (int i = 0; i < _ended.Count; i++)
                Finish(_ended[i]);
            _ended.Clear();

            _candidates.Sort(Louder);
            int heard = Math.Min(_candidates.Count, MaxVoices);
            _choice++;
            for (int i = 0; i < heard; i++)
                _candidates[i].Chosen = _choice;

            // Those that lost their place in the first MaxVoices go quiet; a one-shot just stops
            for (int i = 0; i < _voices.Count; i++)
            {
                var voice = _voices[i];
                var owner = voice.Owner;
                if (owner == null || voice.FadeTo == 0f || owner.Chosen == _choice) continue;
                if (owner.Looping) voice.FadeTo = 0f;
                else Release(voice);
            }

            // Those in it that have no voice get one: a loop where it is now, fading in
            for (int i = 0; i < heard; i++)
            {
                var entry = _candidates[i];
                if (entry.Voice != null)
                {
                    entry.Voice.FadeTo = 1f;
                    continue;
                }
                if (!entry.Looping && Time.TotalTime - entry.Started > LateOneShot) continue;
                var voice = FreeVoice(entry.Clip.WaveFormat);
                if (voice == null) break;
                Start(voice, entry, listener, fadeIn: entry.Looping);
            }
        }

        private void Consider(Entry entry, Vector3 at)
        {
            var source = entry.Source;
            if (!source.Enabled) return;

            // A one-shot that has played out without being heard
            if (!entry.Looping && entry.Voice == null && Time.TotalTime - entry.Started > entry.Clip.Length / MathF.Max(source.Pitch, 0.01f))
            {
                _ended.Add(entry);
                return;
            }

            entry.Loud = entry.Score = Loudness(entry, at);
            if (entry.Loud <= 0f) return;
            if (entry.Voice != null && entry.Voice.FadeTo > 0f) entry.Score *= Keep;
            _candidates.Add(entry);
        }

        /// <summary>
        /// How loud a source is at a place, squared: its volume, full within FalloffStart, over the
        /// distance beyond (as X3DAudio's curve), fading out before HeardWithin. 0 = not heard.
        /// </summary>
        private static float Loudness(Entry entry, Vector3 at)
        {
            var source = entry.Source;
            float away = Vector3.DistanceSquared(entry.Position, at);
            float edge = Edge(source.HeardWithin, away);
            if (edge <= 0f) return 0f;

            float full = MathF.Max(source.FalloffStart, 0.01f);
            float curve = away > full * full ? full * full / away : 1f;
            return source.Volume * source.Volume * curve * edge * edge;
        }

        // 1 well within hearing, falling to 0 over the last fifth of HeardWithin (0 = no limit)
        private static float Edge(float heardWithin, float away)
        {
            if (heardWithin <= 0f) return 1f;
            float share = away / (heardWithin * heardWithin);
            if (share >= 1f) return 0f;
            return share <= EdgeStart ? 1f : (1f - share) / (1f - EdgeStart);
        }

        // ---- Voices ----

        // A voice for this format that nothing has: one of the format, another format's made anew, or a new one
        private Voice? FreeVoice(WaveFormat format)
        {
            Voice? other = null;
            for (int i = 0; i < _voices.Count; i++)
            {
                var voice = _voices[i];
                if (voice.Owner != null) continue;
                if (SameFormat(voice.Format, format)) return voice;
                other ??= voice;
            }

            if (other != null)
            {
                other.Native.DestroyVoice();
                other.Native.Dispose();
                other.Native = Engine.AudioDevice.CreateSourceVoice(format, false);
                other.Format = format;
                return other;
            }

            if (_voices.Count >= MaxVoices) return null;
            var made = new Voice { Native = Engine.AudioDevice.CreateSourceVoice(format, false), Format = format };
            _voices.Add(made);
            return made;
        }

        // The voice to take for a sound played now: one fading out first, else the one heard least
        private Voice? Quietest()
        {
            Voice? quietest = null;
            for (int i = 0; i < _voices.Count; i++)
            {
                var voice = _voices[i];
                if (voice.Owner == null || voice.FadeTo == 0f) return voice;
                if (quietest == null || voice.Owner.Loud < quietest.Owner!.Loud) quietest = voice;
            }
            return quietest;
        }

        private void Start(Voice voice, Entry entry, Listener listener, bool fadeIn)
        {
            var clip = entry.Clip;
            var buffer = clip.Buffer;
            var format = clip.WaveFormat;
            var source = entry.Source;

            // Where it is in the clip by now: a loop goes on in step, a late one-shot starts part way
            uint begin = 0;
            bool pcm = format.Encoding == WaveFormatEncoding.Pcm || format.Encoding == WaveFormatEncoding.IeeeFloat;
            if (pcm && format.BlockAlign > 0)
            {
                long samples = buffer.AudioBytes / format.BlockAlign;
                long at = (long)((Time.TotalTime - entry.Started) * MathF.Max(source.Pitch, 0.01f) * format.SampleRate);
                if (entry.Looping) at %= Math.Max(samples, 1);
                if (at > 0 && at < samples) begin = (uint)at;
            }

            var submit = new AudioBuffer(buffer.AudioDataPointer, buffer.AudioBytes, BufferFlags.EndOfStream)
            {
                PlayBegin = begin,
                LoopCount = entry.Looping ? (uint)XAudio2.LoopInfinite : 0u,
            };

            var native = voice.Native;
            native.FlushSourceBuffers();
            native.SubmitSourceBuffer(submit, clip.DecodedPacketsInfo);

            voice.Owner = entry;
            entry.Voice = voice;
            voice.Fade = fadeIn ? 0f : 1f;
            voice.FadeTo = 1f;

            // Placed before the first sample, or it would play at full volume in the middle until the next update
            var position = source.Transform!.WorldPosition;
            voice.LastPosition = position;
            Place(voice, entry, listener, position, 0f);
            native.Start();
        }

        // Stopped and free; the source plays on without it, if it plays
        private static void Release(Voice voice)
        {
            voice.Native.Stop();
            voice.Native.FlushSourceBuffers();
            if (voice.Owner != null) voice.Owner.Voice = null;
            voice.Owner = null;
        }

        // Volume per speaker and pitch from where the listener is
        private static void Place(Voice voice, Entry entry, Listener listener, Vector3 position, float delta)
        {
            var source = entry.Source;
            var emitter = voice.Emitter;
            var transform = source.Transform!;

            emitter.Position = position;
            emitter.Velocity = delta > 0f ? (position - voice.LastPosition) / delta : Vector3.Zero;
            voice.LastPosition = position;
            emitter.OrientFront = transform.Forward;
            emitter.OrientTop = transform.Up;
            emitter.ChannelRadius = source.Range;
            emitter.InnerRadius = source.MinDistance;
            emitter.CurveDistanceScaler = MathF.Max(source.FalloffStart, 0.01f);

            // The emitter is one point with one channel. A clip with more is folded into it, every channel at
            // its share (the average, as an .ogg is at import): asked for the clip's own count, X3DAudio filled
            // the first channel's levels and left the others silent
            var dsp = voice.Dsp;
            Engine.Audio3D.Calculate(listener, emitter, CalculateFlags.Matrix | CalculateFlags.Doppler, dsp);
            int channels = voice.Format.Channels;
            if (channels == 1)
                voice.Native.SetOutputMatrix(1, 2, dsp.MatrixCoefficients);
            else
            {
                if (voice.Folded?.Length != channels * 2)
                    voice.Folded = new float[channels * 2];

                // XAudio2 reads the level from channel S to speaker D at [D * channels + S]
                var levels = dsp.MatrixCoefficients;
                for (int d = 0; d < 2; d++)
                    for (int s = 0; s < channels; s++)
                        voice.Folded[d * channels + s] = levels[d] / channels;
                voice.Native.SetOutputMatrix((uint)channels, 2, voice.Folded);
            }

            float edge = Edge(source.HeardWithin, Vector3.DistanceSquared(position, listener.Position));
            voice.Native.SetFrequencyRatio(dsp.DopplerFactor * source.Pitch, 0);
            voice.Native.SetVolume(source.Volume * voice.Fade * edge);
        }

        private static bool SameFormat(WaveFormat a, WaveFormat b)
            => a.Encoding == b.Encoding && a.SampleRate == b.SampleRate && a.Channels == b.Channels && a.BitsPerSample == b.BitsPerSample;

        // ---- The grid ----

        private static long Key(int x, int z) => ((long)x << 32) | (uint)z;

        private static long CellOf(Vector3 p) => Key((int)MathF.Floor(p.X / CellSize), (int)MathF.Floor(p.Z / CellSize));

        private void Insert(Entry entry)
        {
            float heardWithin = entry.Source.HeardWithin;
            if (heardWithin <= 0f)
            {
                entry.Everywhere = true;
                _everywhere.Add(entry);
                return;
            }

            _farthest = MathF.Max(_farthest, heardWithin);
            entry.Cell = CellOf(entry.Position);
            if (!_cells.TryGetValue(entry.Cell, out var cell))
                _cells[entry.Cell] = cell = [];
            cell.Add(entry);
        }

        private void Remove(Entry entry)
        {
            if (entry.Everywhere)
                _everywhere.Remove(entry);
            else if (_cells.TryGetValue(entry.Cell, out var cell))
            {
                cell.Remove(entry);
                if (cell.Count == 0) _cells.Remove(entry.Cell);
            }
            entry.Everywhere = false;
            entry.Cell = long.MinValue;
        }

        // Where it belongs now (another cell, or HeardWithin changed between grid and everywhere)
        private void Rebin(Entry entry)
        {
            bool everywhere = entry.Source.HeardWithin <= 0f;
            if (everywhere == entry.Everywhere && (everywhere || entry.Cell == CellOf(entry.Position))) return;
            Remove(entry);
            Insert(entry);
        }

        protected internal override void Destroy()
        {
            for (int i = 0; i < _voices.Count; i++)
            {
                Release(_voices[i]);
                _voices[i].Native.DestroyVoice();
                _voices[i].Native.Dispose();
            }
            _voices.Clear();
        }
    }
}
