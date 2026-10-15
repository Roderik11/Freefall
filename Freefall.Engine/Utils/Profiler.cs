using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Freefall
{
    /// <summary>
    /// A named section of main-thread CPU work: <c>using var _ = Marker.Auto();</c>. Keep markers in static
    /// fields. They are interned by name, so a script type reloaded under the same name keeps its history,
    /// and nothing here holds a Type (no pinned script assembly after a hot reload).
    /// </summary>
    public sealed class ProfilerMarker
    {
        public string Name { get; }

        /// <summary>Time over all call sites in the last recorded frame, ms (0 when it did not run).</summary>
        public double LastMs { get; internal set; }

        internal long FrameTicks;

        internal ProfilerMarker(string name) => Name = name;

        public static ProfilerMarker Get(string name) => Profiler.GetMarker(name);

        public ProfilerScope Auto() => Profiler.Begin(this, 0);

        /// <param name="items">What the section worked on (e.g. components updated), shown next to its time.</param>
        public ProfilerScope Auto(int items) => Profiler.Begin(this, items);
    }

    /// <summary>Ends its section when disposed. Default (not on the main thread) does nothing.</summary>
    public readonly struct ProfilerScope : IDisposable
    {
        private readonly int _depth;

        internal ProfilerScope(int depth) => _depth = depth;

        public void Dispose()
        {
            if (_depth > 0) Profiler.End(_depth);
        }
    }

    /// <summary>One frame of a node.</summary>
    public readonly record struct ProfilerSample(float Ms, int AllocBytes, int Calls, int Items);

    /// <summary>A node over the last frames: averages are per frame, Max is the slowest frame.</summary>
    public readonly record struct ProfilerStats(float AvgMs, float MaxMs, float AvgAllocBytes, float AvgCalls, float AvgItems);

    /// <summary>
    /// A marker at one place in the call tree. The same marker reached from two parents is two nodes; one
    /// reached twice from the same parent in a frame (a system group run per rendered view) adds up.
    /// </summary>
    public sealed class ProfilerNode
    {
        public ProfilerMarker Marker { get; }
        public ProfilerNode? Parent { get; }
        public IReadOnlyList<ProfilerNode> Children => _children;
        public int Depth { get; }

        /// <summary>In the GPU tree (<see cref="Profiler.GpuRoot"/>): times arrive a few frames late, and frames
        /// whose timings were not taken (or not read back yet) are left out of samples and averages.</summary>
        public bool IsGpu { get; }

        internal readonly List<ProfilerNode> _children = [];

        // History, one slot per recorded frame (Profiler.Slot)
        private readonly float[] _ms = new float[Profiler.HistoryLength];
        private readonly int[] _alloc = new int[Profiler.HistoryLength];
        private readonly int[] _calls = new int[Profiler.HistoryLength];
        private readonly int[] _items = new int[Profiler.HistoryLength];

        // The frame in progress
        internal long Ticks;
        internal long Alloc;
        internal int Calls;
        internal int Items;

        private long _lastHit;
        private int _hint;

        internal ProfilerNode(ProfilerMarker marker, ProfilerNode? parent, long frame, bool gpu = false)
        {
            Marker = marker;
            Parent = parent;
            Depth = parent == null ? 0 : parent.Depth + 1;
            IsGpu = parent?.IsGpu ?? gpu;
            _lastHit = frame;
        }

        public string Name => Marker.Name;

        private bool HasFrame(long frame) => IsGpu ? Profiler.IsGpuRecorded(frame) : Profiler.IsRecorded(frame);

        /// <summary>A recorded frame (0 for frames outside the history or before the node existed).</summary>
        public ProfilerSample Sample(long frame)
        {
            if (!HasFrame(frame)) return default;
            int slot = Profiler.Slot(frame);
            return new ProfilerSample(_ms[slot], _alloc[slot], _calls[slot], _items[slot]);
        }

        /// <summary>Over the last <paramref name="frames"/> recorded frames.</summary>
        public ProfilerStats Stats(int frames)
        {
            long end = Profiler.FrameCount;
            long start = Math.Max(end - Math.Min(frames, Profiler.HistoryLength), 0);

            int n = 0;
            float sumMs = 0, maxMs = 0;
            long alloc = 0, calls = 0, items = 0;
            for (long f = start; f < end; f++)
            {
                if (IsGpu && !Profiler.IsGpuRecorded(f)) continue;
                n++;

                int slot = Profiler.Slot(f);
                float ms = _ms[slot];
                sumMs += ms;
                if (ms > maxMs) maxMs = ms;
                alloc += _alloc[slot];
                calls += _calls[slot];
                items += _items[slot];
            }

            if (n == 0) return default;
            return new ProfilerStats(sumMs / n, maxMs, (float)alloc / n, (float)calls / n, (float)items / n);
        }

        internal ProfilerNode Child(ProfilerMarker marker)
        {
            // Call sites come in the same order every frame: try the one after the last hit first
            var list = _children;
            if (_hint < list.Count && ReferenceEquals(list[_hint].Marker, marker))
                return list[_hint++];

            for (int i = 0; i < list.Count; i++)
            {
                if (!ReferenceEquals(list[i].Marker, marker)) continue;
                _hint = i + 1;
                return list[i];
            }

            var node = new ProfilerNode(marker, this, IsGpu ? Profiler.GpuFrameCount : Profiler.FrameCount);
            list.Add(node);
            _hint = list.Count;
            return node;
        }

        /// <summary>Writes the frame in progress into history slot <paramref name="slot"/> (or drops it, slot -1).</summary>
        internal void EndFrame(int slot, long frame, double msPerTick)
        {
            if (slot >= 0)
            {
                _ms[slot] = (float)(Ticks * msPerTick);
                _alloc[slot] = (int)Math.Min(Alloc, int.MaxValue);
                _calls[slot] = Calls;
                _items[slot] = Items;
            }

            if (Calls > 0) _lastHit = frame;
            Ticks = Alloc = 0;
            Calls = Items = 0;
            _hint = 0;

            for (int i = _children.Count - 1; i >= 0; i--)
            {
                var child = _children[i];
                child.EndFrame(slot, frame, msPerTick);

                // Gone for a whole history: a removed component type, a system that stopped running
                if (frame - child._lastHit > Profiler.HistoryLength)
                    _children.RemoveAt(i);
            }
        }
    }

    /// <summary>
    /// Main-thread CPU profiler. Sections are timed with <see cref="ProfilerMarker.Auto()"/> and form a
    /// call tree under <see cref="Root"/> (the whole frame, from one <see cref="EndFrame"/> to the next).
    /// Keeps the last <see cref="HistoryLength"/> frames: time, bytes allocated on the main thread, calls
    /// and items per node, plus frame-wide allocation and garbage collections.
    ///
    /// Always on: a section costs two timestamps and two allocation counter reads. Calls from other
    /// threads are ignored, so a parallel loop is timed as a whole by the scope around it.
    /// </summary>
    public static class Profiler
    {
        public const int HistoryLength = 300;
        private const int MaxDepth = 64;

        private static readonly Dictionary<string, ProfilerMarker> _markers = [];
        private static readonly List<ProfilerMarker> _markerList = [];

        // Open sections, innermost last
        private static readonly ProfilerNode[] _stack = new ProfilerNode[MaxDepth];
        private static readonly long[] _start = new long[MaxDepth];
        private static readonly long[] _allocStart = new long[MaxDepth];
        private static int _depth;

        // Set by the first EndFrame: the engine tick's thread
        private static int _mainThread = -1;
        private static long _frameStart;
        private static long _frameAllocStart;
        private static long _frameTotalAllocStart;
        private static readonly int[] _gcCounts = new int[3];

        private static readonly double MsPerTick = 1000.0 / Stopwatch.Frequency;

        private static readonly float[] _frameMs = new float[HistoryLength];
        private static readonly long[] _frameTotalAlloc = new long[HistoryLength];
        private static readonly byte[] _frameGc = new byte[HistoryLength];

        /// <summary>The whole frame on the CPU. Its children are the top-level sections of the engine tick.</summary>
        public static ProfilerNode Root { get; } = new(new ProfilerMarker("CPU"), null, 0);

        /// <summary>
        /// The primary view's command list on the GPU, from its first to its last command; children are the
        /// PixMarker regions in it (Graphics.GpuProfiler). Filled a few frames after the CPU frame that recorded it.
        /// </summary>
        public static ProfilerNode GpuRoot { get; } = new(new ProfilerMarker("GPU"), null, 0, gpu: true);

        // Per history slot: the frame whose GPU times the slot holds (-1: none)
        private static readonly long[] _gpuFrameOf = CreateGpuFrameOf();

        private static long[] CreateGpuFrameOf()
        {
            var result = new long[HistoryLength];
            Array.Fill(result, -1);
            return result;
        }

        /// <summary>One past the newest frame whose GPU times arrived.</summary>
        public static long GpuFrameCount { get; private set; }

        /// <summary>Whether a recorded frame has GPU times (they arrive a few frames late, and can be skipped).</summary>
        public static bool IsGpuRecorded(long frame) => IsRecorded(frame) && _gpuFrameOf[Slot(frame)] == frame;

        internal static bool IsMainThread => Environment.CurrentManagedThreadId == _mainThread;

        /// <summary>
        /// Writes what the GPU tree accumulated (node Ticks/Calls, in GPU timestamp ticks) as the GPU times of
        /// <paramref name="frame"/>, or drops it when that frame was not recorded (paused, or already out of history).
        /// </summary>
        internal static void RecordGpuFrame(long frame, double msPerTick)
        {
            if (!IsRecorded(frame))
            {
                GpuRoot.EndFrame(-1, Math.Max(frame, GpuFrameCount), msPerTick);
                return;
            }

            int slot = Slot(frame);
            GpuRoot.EndFrame(slot, frame, msPerTick);
            _gpuFrameOf[slot] = frame;
            GpuFrameCount = Math.Max(GpuFrameCount, frame + 1);
        }

        /// <summary>Frames recorded so far; the newest is FrameCount - 1.</summary>
        public static long FrameCount { get; private set; }

        /// <summary>While paused, frames are measured but not recorded, so the history holds still.</summary>
        public static bool Paused { get; set; }

        public static int RecordedFrames => (int)Math.Min(FrameCount, HistoryLength);

        internal static int Slot(long frame) => (int)(frame % HistoryLength);

        public static bool IsRecorded(long frame) => frame >= 0 && frame < FrameCount && frame >= FrameCount - HistoryLength;

        /// <summary>Wall time of a recorded frame, ms.</summary>
        public static float FrameMs(long frame) => IsRecorded(frame) ? _frameMs[Slot(frame)] : 0;

        /// <summary>
        /// Bytes allocated by all threads during a recorded frame. The runtime's counter is imprecise per
        /// frame (a frame can read 0 and the next one twice as much), so use it averaged over frames; a
        /// node's allocations (main thread) are exact.
        /// </summary>
        public static long FrameAllocatedBytes(long frame) => IsRecorded(frame) ? _frameTotalAlloc[Slot(frame)] : 0;

        /// <summary>Garbage collections during a recorded frame: bit 0 = gen0, 1 = gen1, 2 = gen2.</summary>
        public static int FrameCollections(long frame) => IsRecorded(frame) ? _frameGc[Slot(frame)] : 0;

        internal static ProfilerMarker GetMarker(string name)
        {
            lock (_markers)
            {
                if (!_markers.TryGetValue(name, out var marker))
                {
                    marker = new ProfilerMarker(name);
                    _markers.Add(name, marker);
                    _markerList.Add(marker);
                }
                return marker;
            }
        }

        internal static ProfilerScope Begin(ProfilerMarker marker, int items)
        {
            if (Environment.CurrentManagedThreadId != _mainThread || _depth >= MaxDepth) return default;

            var node = (_depth == 0 ? Root : _stack[_depth - 1]).Child(marker);
            node.Items += items;

            _stack[_depth] = node;
            _allocStart[_depth] = GC.GetAllocatedBytesForCurrentThread();
            _start[_depth] = Stopwatch.GetTimestamp();
            return new ProfilerScope(++_depth);
        }

        internal static void End(int depth)
        {
            if (Environment.CurrentManagedThreadId != _mainThread) return;

            // A section further in that was never closed ends here too
            while (_depth >= depth && _depth > 0)
                Close();
        }

        private static void Close()
        {
            long now = Stopwatch.GetTimestamp();
            int i = --_depth;
            var node = _stack[i];
            long ticks = now - _start[i];

            node.Ticks += ticks;
            node.Calls++;
            node.Alloc += GC.GetAllocatedBytesForCurrentThread() - _allocStart[i];
            node.Marker.FrameTicks += ticks;
            _stack[i] = null!;
        }

        /// <summary>Closes the frame. Called once per engine tick, on the tick's thread, outside any section.</summary>
        public static void EndFrame()
        {
            long now = Stopwatch.GetTimestamp();
            long alloc = GC.GetAllocatedBytesForCurrentThread();
            long totalAlloc = GC.GetTotalAllocatedBytes(false);

            if (_mainThread < 0)
            {
                _mainThread = Environment.CurrentManagedThreadId;
                _frameStart = now;
                _frameAllocStart = alloc;
                _frameTotalAllocStart = totalAlloc;
                for (int g = 0; g < 3; g++) _gcCounts[g] = GC.CollectionCount(g);
                return;
            }

            if (Environment.CurrentManagedThreadId != _mainThread) return;

            while (_depth > 0) Close();

            Root.Ticks = now - _frameStart;
            Root.Alloc = alloc - _frameAllocStart;
            Root.Calls = 1;

            int gc = 0;
            for (int g = 0; g < 3; g++)
            {
                int count = GC.CollectionCount(g);
                if (count != _gcCounts[g]) gc |= 1 << g;
                _gcCounts[g] = count;
            }

            lock (_markers)
            {
                foreach (var marker in _markerList)
                {
                    marker.LastMs = marker.FrameTicks * MsPerTick;
                    marker.FrameTicks = 0;
                }
            }

            if (Paused)
            {
                Root.EndFrame(-1, FrameCount, MsPerTick);
            }
            else
            {
                int slot = Slot(FrameCount);
                _frameMs[slot] = (float)(Root.Ticks * MsPerTick);
                _frameTotalAlloc[slot] = totalAlloc - _frameTotalAllocStart;
                _frameGc[slot] = (byte)gc;
                Root.EndFrame(slot, FrameCount, MsPerTick);
                FrameCount++;
            }

            _frameStart = now;
            _frameAllocStart = alloc;
            _frameTotalAllocStart = totalAlloc;
        }
    }
}
