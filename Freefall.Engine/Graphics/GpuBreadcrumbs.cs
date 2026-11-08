using System;
using System.Collections.Generic;
using System.Text;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// Our own GPU breadcrumbs (opt-in: FREEFALL_BREADCRUMBS=1), for a device removal that DRED cannot place: on the
    /// RTX 4080's driver DRED's auto-breadcrumbs come back with no nodes (TestWorld's play hang, 2026-11-08).
    /// Every <see cref="PixMarker"/> region on a direct or compute list writes a stamp into a readback buffer when the
    /// GPU reaches its start (WriteBufferImmediate, MARKER_IN: everything before it has started) and another when its
    /// end is reached with everything before it finished (MARKER_OUT). The buffer is system memory, mapped for good, so
    /// it can still be read once the device is gone: <see cref="GraphicsDevice.LogDeviceRemoved"/> prints the regions
    /// the GPU began and never finished, the hung pass among them (the GPU front end may also have begun regions after
    /// it, up to the next barrier, so the first one is the one to look at).
    ///
    /// Costs two writes and two QueryInterface calls a region, allocations included, so it stays a switch for hunting.
    /// </summary>
    internal static class GpuBreadcrumbs
    {
        private const int Slots = 1 << 15;           // regions remembered; a stamp's slot is stamp % Slots

        public static readonly bool Enabled = Environment.GetEnvironmentVariable("FREEFALL_BREADCRUMBS") == "1";

        private static readonly object _lock = new();
        private static GraphicsBuffer? _buffer;
        private static unsafe uint* _words;          // per slot: [begun stamp, finished stamp], written by the GPU
        private static ulong _address;
        private static bool _unavailable;
        private static uint _stamp;                  // the last stamp handed out

        // What the CPU knows of each slot's region
        private static string?[] _names = [];
        private static uint[] _stamps = [];
        private static int[] _depths = [];
        private static IntPtr[] _lists = [];
        private static int[] _ticks = [];

        private static readonly Dictionary<IntPtr, Stack<int>> _open = new();
        private static readonly WriteBufferImmediateParameter[] _parameter = new WriteBufferImmediateParameter[1];
        private static readonly WriteBufferImmediateMode[] _mode = new WriteBufferImmediateMode[1];

        private static unsafe bool EnsureResources()
        {
            if (_words != null) return true;
            if (_unavailable) return false;
            try
            {
                var options = new FeatureDataD3D12Options3();
                if (!Engine.Device.NativeDevice.CheckFeatureSupport(Feature.Options3, ref options)
                    || (options.WriteBufferImmediateSupportFlags & CommandListSupportFlags.Direct) == 0)
                    throw new NotSupportedException("WriteBufferImmediate is not supported on direct lists");
                _buffer = GraphicsBuffer.CreateReadback<uint>(Slots * 2);
                _address = _buffer.Native.GPUVirtualAddress;
                _names = new string?[Slots];
                _stamps = new uint[Slots];
                _depths = new int[Slots];
                _lists = new IntPtr[Slots];
                _ticks = new int[Slots];
                _words = _buffer.ReadPtr<uint>();
                Console.WriteLine("[Breadcrumbs] GPU breadcrumbs on (FREEFALL_BREADCRUMBS=1)");
                return true;
            }
            catch (Exception ex)
            {
                _unavailable = true;
                Debug.LogWarning("GpuBreadcrumbs", $"GPU breadcrumbs unavailable: {ex.Message}");
                return false;
            }
        }

        /// <summary>A PixMarker region begins on the list.</summary>
        public static void Begin(ID3D12GraphicsCommandList list, string name)
        {
            if (!Enabled) return;
            lock (_lock)
            {
                if (!EnsureResources()) return;
                var type = list.CommandListType;
                if (type != CommandListType.Direct && type != CommandListType.Compute) return;
                if (++_stamp == 0) _stamp = 1;
                int slot = (int)(_stamp % Slots);
                if (!_open.TryGetValue(list.NativePointer, out var stack))
                    _open[list.NativePointer] = stack = new Stack<int>();
                _names[slot] = name;
                _stamps[slot] = _stamp;
                _depths[slot] = stack.Count;
                _lists[slot] = list.NativePointer;
                _ticks[slot] = Engine.TickCount;
                stack.Push(slot);
                Write(list, slot * 2, _stamp, WriteBufferImmediateMode.MarkerIn);
            }
        }

        /// <summary>The list's innermost open region ends.</summary>
        public static void End(ID3D12GraphicsCommandList list)
        {
            if (!Enabled) return;
            lock (_lock)
            {
                if (_names.Length == 0 || !_open.TryGetValue(list.NativePointer, out var stack) || stack.Count == 0) return;
                int slot = stack.Pop();
                Write(list, slot * 2 + 1, _stamps[slot], WriteBufferImmediateMode.MarkerOut);
            }
        }

        private static void Write(ID3D12GraphicsCommandList list, int word, uint value, WriteBufferImmediateMode mode)
        {
            using var list2 = list.QueryInterfaceOrNull<ID3D12GraphicsCommandList2>();
            if (list2 == null) return;
            _parameter[0] = new WriteBufferImmediateParameter(_address + (ulong)word * 4, value);
            _mode[0] = mode;
            list2.WriteBufferImmediate(_parameter, _mode);
        }

        /// <summary>
        /// After a removal: the regions the GPU began and did not finish, oldest first, each with the regions round it;
        /// then the last ones it finished. Empty when the switch is off.
        /// </summary>
        public static unsafe string Report()
        {
            if (!Enabled) return "";
            lock (_lock)
            {
                if (_words == null) return "  GPU breadcrumbs: none recorded\n";
                var sb = new StringBuilder();
                uint count = Math.Min(_stamp, (uint)Slots);
                uint first = _stamp - count + 1;
                var open = new List<uint>();
                uint lastFinished = 0;
                int notReached = 0;
                for (uint i = 0; i < count; i++)
                {
                    uint stamp = first + i;
                    int slot = (int)(stamp % Slots);
                    if (_stamps[slot] != stamp) continue;
                    bool begun = _words[slot * 2] == stamp, finished = _words[slot * 2 + 1] == stamp;
                    if (begun && !finished) open.Add(stamp);
                    else if (finished) lastFinished = stamp;
                    else notReached++;
                }
                sb.AppendLine($"  GPU breadcrumbs (tick {Engine.TickCount}): {open.Count} regions begun and not finished, " +
                              $"{notReached} recorded and never reached");
                foreach (uint stamp in open.Count > 16 ? open.GetRange(0, 16) : open)
                    sb.AppendLine($"    open:     {Describe(stamp, first)}");
                if (lastFinished != 0)
                {
                    // The last few finished regions, for what the GPU did just before
                    var done = new List<uint>();
                    for (uint stamp = lastFinished; stamp >= first && done.Count < 6; stamp--)
                    {
                        int slot = (int)(stamp % Slots);
                        if (_stamps[slot] == stamp && _words[slot * 2 + 1] == stamp) done.Add(stamp);
                        if (stamp == 0) break;
                    }
                    done.Reverse();
                    foreach (uint stamp in done) sb.AppendLine($"    finished: {Describe(stamp, first)}");
                }
                return sb.ToString();
            }
        }

        // "tick T, list L: Outer > Inner > Name": the enclosing regions are the latest earlier ones of the same list one
        // level up each
        private static string Describe(uint stamp, uint first)
        {
            int slot = (int)(stamp % Slots);
            var path = new List<string> { _names[slot] ?? "?" };
            int depth = _depths[slot];
            IntPtr list = _lists[slot];
            for (uint s = stamp - 1; s >= first && depth > 0 && s != 0; s--)
            {
                int other = (int)(s % Slots);
                if (_stamps[other] != s || _lists[other] != list || _depths[other] != depth - 1) continue;
                path.Insert(0, _names[other] ?? "?");
                depth--;
            }
            return $"tick {_ticks[slot]}, list {list.ToInt64() & 0xFFFFFF:X6}: {string.Join(" > ", path)}";
        }
    }
}
