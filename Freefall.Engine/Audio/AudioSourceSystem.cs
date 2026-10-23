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
    ///
    /// A source <see cref="AudioShape.Along"/> a spline sounds from its point nearest the listener (a
    /// river's near bank, a coastline); one over an <see cref="AudioShape.Area"/> all around the listener
    /// inside it. Its spline (its entity's or its parent's) is kept as a line of world points, made again
    /// when the spline is edited or moved; a listener outside its bounds grown by HeardWithin is not
    /// looked at further.
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

        // One switched off or on (Enabled: a bed of the day giving way to the night's) over this long
        const float SwitchFadeTime = 4f;

        // A voice that is heard keeps it against one up to this much louder (loudness squared: x2 = 3 dB),
        // so two fires about as far off do not swap as the listener moves between them
        const float Keep = 2f;

        // A one-shot that is already this far through is not given a voice late
        const float LateOneShot = 0.5f;

        // The last share of HeardWithin a sound fades out over (squared: from 0.8 of the way)
        const float EdgeStart = 0.64f;

        // A spline's line: a point every this many metres, at most this many
        const float LineSpacing = 2f;
        const int MostLinePoints = 4096;

        // A mono clip heard all around: each speaker at the level X3DAudio gives a sound straight ahead
        const float Centre = 0.70710678f;

        /// <summary>Where a playing source is kept, to be found by the listener.</summary>
        internal enum Home { None, Cell, Everywhere, Along }

        /// <summary>What the system keeps of a source that plays.</summary>
        internal sealed class Entry
        {
            public AudioSource Source = null!;
            public AudioClip Clip = null!;
            public bool Looping;
            public float Started;           // Time.TotalTime when it was played
            public Vector3 Position;        // where it is heard from (on its line, for a spline)
            public bool Moved;
            public Home Home;
            public long Cell;
            public Line? Line;              // Shape Spline: the spline, in world points
            public Voice? Voice;
            public float Loud;              // loudness squared at the listener, at the last choice
            public float Score;             // the same, with the edge a source that is heard keeps (Keep)
            public int PlayingIndex = -1;   // in _playing
            public int Chosen;              // the choice it was among the heard at
            public bool SwitchedOn;         // Enabled again: its next voice fades in over SwitchFadeTime
        }

        /// <summary>A spline in world points, with the half width of its sound at each.</summary>
        internal sealed class Line
        {
            public Vector3[] Points = [];
            public float[] HalfWidths = [];
            public Spline Spline = null!;
            public bool Area;               // heard all around inside it
            public Vector2 Min, Max;        // XZ
        }

        internal sealed class Voice
        {
            public IXAudio2SourceVoice Native = null!;
            public WaveFormat Format = null!;
            public Entry? Owner;
            public float Fade = 1f;         // 0..1
            public float FadeTo = 1f;
            public float FadeSeconds = FadeTime;
            public Vector3 LastPosition;
            public readonly Emitter Emitter = new() { ChannelCount = 1, CurveDistanceScaler = 1 };
            public readonly DspSettings Dsp = new(1, 2);
            public float[]? Matrix;
        }

        private readonly List<Entry> _playing = [];                 // every source that plays
        private readonly Dictionary<long, List<Entry>> _cells = [];
        private readonly List<Entry> _everywhere = [];              // HeardWithin 0: no limit
        private readonly List<Entry> _along = [];                   // Shape Spline
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

        protected internal override void Initialize()
        {
            MessageDispatcher.AddListener(EngineMsg.SplineChanged, OnSplineChanged);
        }

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

                // A one-shot that has played to its end
                if (!entry.Looping && voice.Native.StateNoSamplesPlayed.BuffersQueued == 0)
                {
                    Finish(entry);
                    continue;
                }

                // Faded out: the voice is free again, the source plays on without one
                if (voice.FadeTo < voice.Fade)
                {
                    voice.Fade = MathF.Max(voice.FadeTo, voice.Fade - Time.Delta / voice.FadeSeconds);
                    if (voice.Fade <= 0f)
                    {
                        Release(voice);
                        continue;
                    }
                }
                else if (voice.FadeTo > voice.Fade)
                    voice.Fade = MathF.Min(voice.FadeTo, voice.Fade + Time.Delta / voice.FadeSeconds);

                heard++;
                if (listener != null && entry.Source.Transform != null)
                    Place(voice, entry, listener, Where(entry, listener.Position), Time.SmoothDelta);
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
            entry.Line = MakeLine(source);
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

            entry.Position = Where(entry, listener.Position);
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

        /// <summary>A member changed: a changed Shape, Width or HeardWithin is taken up at once.</summary>
        internal void Changed(AudioSource source)
        {
            var entry = source.Entry;
            if (entry == null || entry.PlayingIndex < 0) return;
            entry.Line = MakeLine(source);
            if (entry.Line == null && source.Transform != null) entry.Position = source.Transform.WorldPosition;
            Rebin(entry);
        }

        /// <summary>
        /// Switched off: a loop that is heard fades out over a few seconds and plays on unheard, a one-shot
        /// stops. Switched on: heard from the next choice, a loop fading in as slowly.
        /// </summary>
        internal void EnabledChanged(AudioSource source)
        {
            var entry = source.Entry;
            if (entry == null) return;
            if (source.Enabled)
            {
                entry.SwitchedOn = true;
                if (entry.Voice != null) entry.Voice.FadeSeconds = SwitchFadeTime;
            }
            else if (entry.Voice != null)
            {
                if (!entry.Looping) Release(entry.Voice);
                else
                {
                    entry.Voice.FadeTo = 0f;
                    entry.Voice.FadeSeconds = SwitchFadeTime;
                }
            }
        }

        // A spline edited (points, widths, closed): the sources along it take up its new line
        private void OnSplineChanged(Message message)
        {
            if (message.Data is not Spline spline) return;
            for (int i = 0; i < _along.Count; i++)
                if (_along[i].Line?.Spline == spline)
                    _along[i].Line = MakeLine(_along[i].Source);
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
            // Sources that moved since the last choice: where they are now and in which cell, or their spline's new line
            for (int i = 0; i < _moved.Count; i++)
            {
                var entry = _moved[i];
                if (!entry.Moved) continue;
                entry.Moved = false;
                if (entry.Source.Transform == null) continue;
                if (entry.Home == Home.Along)
                    entry.Line = MakeLine(entry.Source);
                else
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

            for (int i = 0; i < _along.Count; i++)
            {
                var entry = _along[i];
                var line = entry.Line!;
                float reach = entry.Source.HeardWithin;
                if (reach > 0f && (at.X < line.Min.X - reach || at.X > line.Max.X + reach || at.Z < line.Min.Y - reach || at.Z > line.Max.Y + reach))
                    continue;
                entry.Position = Nearest(line, at);
                Consider(entry, at);
            }

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

        // ---- Splines ----

        // Where a source is heard from by a listener at a place: its own place, or the nearest of its line
        private static Vector3 Where(Entry entry, Vector3 at)
            => entry.Line != null ? Nearest(entry.Line, at) : entry.Source.Transform!.WorldPosition;

        // The line of a source along its entity's Spline, or its parent's; null for a point source or without a spline to follow
        private static Line? MakeLine(AudioSource source)
        {
            if (source.Shape == AudioShape.Point) return null;
            var spline = source.Entity?.GetComponent<Spline>() ?? source.Transform?.Parent?.Entity?.GetComponent<Spline>();
            if (spline == null || spline.Points.Count < 2) return null;

            float length = spline.GetLength(128);
            int count = Math.Clamp((int)(length / LineSpacing), 8, MostLinePoints);
            var line = new Line
            {
                Points = new Vector3[count + 1],
                HalfWidths = new float[count + 1],
                Spline = spline,
                Area = source.Shape == AudioShape.Area && spline.Closed,
                Min = new Vector2(float.MaxValue),
                Max = new Vector2(float.MinValue),
            };
            float half = MathF.Max(source.Width, 0f) * 0.5f;
            for (int i = 0; i <= count; i++)
            {
                float t = (float)i / count;
                var p = spline.GetWorldPoint(t);
                line.Points[i] = p;
                line.HalfWidths[i] = half * spline.GetWidth(t);
                line.Min = Vector2.Min(line.Min, new Vector2(p.X, p.Z));
                line.Max = Vector2.Max(line.Max, new Vector2(p.X, p.Z));
            }
            return line;
        }

        /// <summary>
        /// The point of a line a listener at a place hears it from, nearest on the ground (XZ): inside a
        /// closed one the place itself; for an open one with a width, its near edge (the bank).
        /// </summary>
        private static Vector3 Nearest(Line line, Vector3 at)
        {
            var points = line.Points;
            float best = float.MaxValue;
            Vector3 nearest = points[0];
            float halfWidth = 0f;
            bool inside = false;

            for (int i = 0; i + 1 < points.Length; i++)
            {
                var a = points[i];
                var b = points[i + 1];
                float abx = b.X - a.X, abz = b.Z - a.Z;
                float length = abx * abx + abz * abz;
                float t = length > 0f ? Math.Clamp(((at.X - a.X) * abx + (at.Z - a.Z) * abz) / length, 0f, 1f) : 0f;
                float dx = at.X - (a.X + abx * t), dz = at.Z - (a.Z + abz * t);
                float away = dx * dx + dz * dz;
                if (away < best)
                {
                    best = away;
                    nearest = Vector3.Lerp(a, b, t);
                    halfWidth = line.HalfWidths[i] + (line.HalfWidths[i + 1] - line.HalfWidths[i]) * t;
                }

                // Inside an area: an odd number of its edges cross the line from the place towards +X
                if (line.Area && (a.Z > at.Z) != (b.Z > at.Z) && at.X < a.X + abx * (at.Z - a.Z) / abz)
                    inside = !inside;
            }

            if (inside) return at;

            // A line with a width: its near bank, or the place itself when that is on the water
            if (!line.Area && halfWidth > 0f && best > 0f)
            {
                float away = MathF.Sqrt(best);
                if (away <= halfWidth) return new Vector3(at.X, nearest.Y, at.Z);
                float toBank = halfWidth / away;
                nearest.X += (at.X - nearest.X) * toBank;
                nearest.Z += (at.Z - nearest.Z) * toBank;
            }
            return nearest;
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
            voice.FadeSeconds = entry.SwitchedOn ? SwitchFadeTime : FadeTime;
            entry.SwitchedOn = false;

            // Placed before the first sample, or it would play at full volume in the middle until the next update
            var position = Where(entry, listener.Position);
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
            voice.FadeSeconds = FadeTime;
        }

        // Volume per speaker and pitch from where the listener is
        private static void Place(Voice voice, Entry entry, Listener listener, Vector3 position, float delta)
        {
            var source = entry.Source;
            var transform = source.Transform!;
            int channels = voice.Format.Channels;
            if (voice.Matrix?.Length != channels * 2)
                voice.Matrix = new float[channels * 2];
            var matrix = voice.Matrix;

            // Along a spline, near it the clip's own channels come through as they are (all around the
            // listener inside an area), narrowing to one point at FalloffStart
            float away = Vector3.DistanceSquared(position, listener.Position);
            float full = MathF.Max(source.FalloffStart, 0.01f);
            float spread = entry.Line != null && away < full * full ? 1f - MathF.Sqrt(away) / full : 0f;
            float pitch = source.Pitch;

            if (spread < 1f)
            {
                var emitter = voice.Emitter;
                emitter.Position = position;
                // A spline's point slides along with the listener: no Doppler from that
                emitter.Velocity = entry.Line == null && delta > 0f ? (position - voice.LastPosition) / delta : Vector3.Zero;
                emitter.OrientFront = transform.Forward;
                emitter.OrientTop = transform.Up;
                emitter.ChannelRadius = source.Range;
                emitter.InnerRadius = source.MinDistance;
                emitter.CurveDistanceScaler = full;

                // The emitter is one point with one channel. A clip with more is folded into it, every channel at
                // its share (the average, as an .ogg is at import): asked for the clip's own count, X3DAudio filled
                // the first channel's levels and left the others silent
                var dsp = voice.Dsp;
                var flags = entry.Line == null ? CalculateFlags.Matrix | CalculateFlags.Doppler : CalculateFlags.Matrix;
                Engine.Audio3D.Calculate(listener, emitter, flags, dsp);
                var levels = dsp.MatrixCoefficients;

                // XAudio2 reads the level from channel S to speaker D at [D * channels + S]
                for (int d = 0; d < 2; d++)
                    for (int s = 0; s < channels; s++)
                        matrix[d * channels + s] = levels[d] / channels;
                if (entry.Line == null) pitch *= dsp.DopplerFactor;
            }
            voice.LastPosition = position;

            if (spread > 0f)
            {
                // The clip as it is: left to left and right to right (one channel to both, as if ahead)
                for (int d = 0; d < 2; d++)
                    for (int s = 0; s < channels; s++)
                    {
                        float flat = channels == 1 ? Centre : s % 2 == d ? 2f / channels : 0f;
                        matrix[d * channels + s] = spread >= 1f ? flat : matrix[d * channels + s] * (1f - spread) + flat * spread;
                    }
            }
            voice.Native.SetOutputMatrix((uint)channels, 2, matrix);

            float edge = Edge(source.HeardWithin, away);
            voice.Native.SetFrequencyRatio(pitch, 0);
            voice.Native.SetVolume(source.Volume * voice.Fade * edge);
        }

        private static bool SameFormat(WaveFormat a, WaveFormat b)
            => a.Encoding == b.Encoding && a.SampleRate == b.SampleRate && a.Channels == b.Channels && a.BitsPerSample == b.BitsPerSample;

        // ---- The grid ----

        private static long Key(int x, int z) => ((long)x << 32) | (uint)z;

        private static long CellOf(Vector3 p) => Key((int)MathF.Floor(p.X / CellSize), (int)MathF.Floor(p.Z / CellSize));

        // Where a source belongs: along its spline, everywhere (no limit) or in the cell it is in
        private static Home HomeOf(Entry entry)
            => entry.Line != null ? Home.Along : entry.Source.HeardWithin <= 0f ? Home.Everywhere : Home.Cell;

        private void Insert(Entry entry)
        {
            entry.Home = HomeOf(entry);
            switch (entry.Home)
            {
                case Home.Along:
                    _along.Add(entry);
                    break;
                case Home.Everywhere:
                    _everywhere.Add(entry);
                    break;
                default:
                    _farthest = MathF.Max(_farthest, entry.Source.HeardWithin);
                    entry.Cell = CellOf(entry.Position);
                    if (!_cells.TryGetValue(entry.Cell, out var cell))
                        _cells[entry.Cell] = cell = [];
                    cell.Add(entry);
                    break;
            }
        }

        private void Remove(Entry entry)
        {
            switch (entry.Home)
            {
                case Home.Along:
                    _along.Remove(entry);
                    break;
                case Home.Everywhere:
                    _everywhere.Remove(entry);
                    break;
                case Home.Cell when _cells.TryGetValue(entry.Cell, out var cell):
                    cell.Remove(entry);
                    if (cell.Count == 0) _cells.Remove(entry.Cell);
                    break;
            }
            entry.Home = Home.None;
        }

        // Where it belongs now (another cell, or a changed Shape or HeardWithin)
        private void Rebin(Entry entry)
        {
            var home = HomeOf(entry);
            if (home == entry.Home && (home != Home.Cell || entry.Cell == CellOf(entry.Position))) return;
            Remove(entry);
            Insert(entry);
        }

        protected internal override void Destroy()
        {
            MessageDispatcher.RemoveListener(EngineMsg.SplineChanged, OnSplineChanged);
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
