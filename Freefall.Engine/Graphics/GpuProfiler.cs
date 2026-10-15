using System;
using System.Collections.Generic;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// GPU time per <see cref="PixMarker"/> region, from timestamp queries on the primary view's command list.
    /// The list gets a timestamp at its start (RenderView.Prepare), at every PixMarker Begin/End, and at its end
    /// (RenderView.Present), which also resolves them into a readback buffer. A fence signalled after the
    /// submit tells when they can be read; that happens at a later Prepare, never waiting, and the times land
    /// in <see cref="Profiler.GpuRoot"/> under the CPU frame that recorded them.
    ///
    /// Markers on any other command list are ignored. A frame whose readback slot is still in flight is not
    /// timed (and simply missing from the averages).
    /// </summary>
    internal static class GpuProfiler
    {
        private const int Slots = 4;                 // frames that can be in flight with their queries
        private const int QueriesPerFrame = 1024;    // two per region plus the list's start and end
        private const int MaxDepth = 32;

        private sealed class Frame
        {
            public int Base;                         // first query index of this slot
            public int Count;                        // queries used
            public long ProfilerFrame;               // -1: recorded while paused
            public ulong Fence;
            public bool Pending;                     // submitted, waiting for the fence
            public readonly List<(ProfilerNode Node, int Begin, int End)> Regions = [];
        }

        private static readonly Frame[] _frames = new Frame[Slots];
        private static int _next;
        private static Frame? _recording;            // between BeginFrame and EndFrame
        private static Frame? _resolved;             // between EndFrame and Submitted
        private static IntPtr _list;

        private static readonly (ProfilerNode? Node, int Begin)[] _stack = new (ProfilerNode?, int)[MaxDepth];
        private static int _depth;
        private static int _tooDeep;

        private static ID3D12QueryHeap? _heap;
        private static GraphicsBuffer? _readback;
        private static ID3D12Fence? _fence;
        private static ulong _fenceValue;
        private static double _msPerTick;
        private static bool _unavailable;

        private static bool EnsureResources()
        {
            if (_heap != null) return true;
            if (_unavailable) return false;

            try
            {
                var device = Engine.Device;
                _heap = device.NativeDevice.CreateQueryHeap<ID3D12QueryHeap>(
                    new QueryHeapDescription(QueryHeapType.Timestamp, Slots * QueriesPerFrame));
                _readback = GraphicsBuffer.CreateReadback<ulong>(Slots * QueriesPerFrame);
                _fence = device.NativeDevice.CreateFence(0, FenceFlags.None);
                device.CommandQueue.GetTimestampFrequency(out ulong frequency).CheckError();
                _msPerTick = 1000.0 / frequency;

                for (int i = 0; i < Slots; i++)
                    _frames[i] = new Frame { Base = i * QueriesPerFrame };
                return true;
            }
            catch (Exception ex)
            {
                _unavailable = true;
                Debug.LogWarning("GpuProfiler", $"GPU timestamps unavailable: {ex.Message}");
                return false;
            }
        }

        /// <summary>The primary view's command list was reset and is about to be recorded.</summary>
        public static void BeginFrame(ID3D12GraphicsCommandList list)
        {
            Collect();

            _recording = null;
            if (_resolved != null)
            {
                // Resolved but never submitted (Close threw): its queries are worthless
                _resolved.Regions.Clear();
                _resolved = null;
            }

            if (!Profiler.IsMainThread || Engine.Device.IsDeviceLost || !EnsureResources()) return;

            var frame = _frames[_next % Slots];
            if (frame.Pending) return; // GPU still behind with that slot: this frame goes untimed

            frame.Count = 0;
            frame.Regions.Clear();
            frame.ProfilerFrame = Profiler.Paused ? -1 : Profiler.FrameCount;

            _recording = frame;
            _list = list.NativePointer;
            _depth = 0;
            _tooDeep = 0;

            // The list as a whole (closed in EndFrame)
            _stack[_depth++] = (Profiler.GpuRoot, Timestamp(list));
        }

        /// <summary>From <see cref="PixMarker.Begin(ID3D12GraphicsCommandList, string)"/>.</summary>
        public static void Begin(ID3D12GraphicsCommandList list, string name)
        {
            var frame = _recording;
            if (frame == null || list.NativePointer != _list) return;

            if (_depth >= MaxDepth)
            {
                _tooDeep++; // matched by an End that must not pop
                return;
            }

            // Room for this region's two queries plus the end of every open region and the list
            if (frame.Count + 2 + _depth > QueriesPerFrame)
            {
                _stack[_depth++] = (null, -1);
                return;
            }

            var parent = _stack[_depth - 1].Node ?? Profiler.GpuRoot;
            var node = parent.Child(ProfilerMarker.Get(name));
            _stack[_depth++] = (node, Timestamp(list));
        }

        /// <summary>From <see cref="PixMarker.End"/>.</summary>
        public static void End(ID3D12GraphicsCommandList list)
        {
            var frame = _recording;
            if (frame == null || list.NativePointer != _list) return;
            if (_tooDeep > 0) { _tooDeep--; return; }
            if (_depth <= 1) return; // depth 1 = the list itself

            var (node, begin) = _stack[--_depth];
            if (node == null) return;

            frame.Regions.Add((node, begin, Timestamp(list)));
        }

        /// <summary>The primary view's list is about to be closed: time its end and resolve the queries.</summary>
        public static void EndFrame(ID3D12GraphicsCommandList list)
        {
            var frame = _recording;
            if (frame == null || list.NativePointer != _list) return;

            // Regions left open by an unbalanced Begin end here, the list itself last
            while (_depth > 0)
            {
                var (node, begin) = _stack[--_depth];
                if (node != null) frame.Regions.Add((node, begin, Timestamp(list)));
            }

            list.ResolveQueryData(_heap!, QueryType.Timestamp, (uint)frame.Base, (uint)frame.Count,
                _readback!.Native, (ulong)frame.Base * sizeof(ulong));

            _recording = null;
            _resolved = frame;
        }

        /// <summary>The list was submitted: signal when the GPU is past it.</summary>
        public static void Submitted(ID3D12CommandQueue queue)
        {
            var frame = _resolved;
            if (frame == null) return;
            _resolved = null;

            frame.Fence = ++_fenceValue;
            frame.Pending = true;
            queue.Signal(_fence!, frame.Fence);
            _next++;
        }

        private static int Timestamp(ID3D12GraphicsCommandList list)
        {
            var frame = _recording!;
            int index = frame.Base + frame.Count++;
            list.EndQuery(_heap!, QueryType.Timestamp, (uint)index);
            return index;
        }

        /// <summary>Reads every frame the GPU has finished into the profiler's GPU tree.</summary>
        private static unsafe void Collect()
        {
            if (_fence == null || Engine.Device.IsDeviceLost) return;

            ulong completed = _fence.CompletedValue;
            if (completed == ulong.MaxValue) return; // device removed

            // Oldest first, so the GPU tree's frame bookkeeping only moves forward
            for (int i = 0; i < Slots; i++)
            {
                var frame = _frames[(_next + i) % Slots];
                if (!frame.Pending || frame.Fence > completed) continue;
                frame.Pending = false;

                ulong* stamps = _readback!.ReadPtr<ulong>();
                foreach (var (node, begin, end) in frame.Regions)
                {
                    ulong t0 = stamps[begin], t1 = stamps[end];
                    node.Ticks += t1 > t0 ? (long)(t1 - t0) : 0;
                    node.Calls++;
                }
                frame.Regions.Clear();

                Profiler.RecordGpuFrame(frame.ProfilerFrame, _msPerTick);
            }
        }
    }
}
