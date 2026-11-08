using System;
using System.Collections.Generic;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// Which page of a terrain map is in which slot of its pool, shared by the height pool and the paint pool
    /// (TERRAIN_PAGES_PLAN.md 5a, 5b). Pages come in levels: a level L page covers 2^L × 2^L level 0 pages. Two layouts:
    /// every level 0 page at the slot of its own index (one level, no scratch: <see cref="AllResident"/>), or a fixed
    /// number of slots given out from a free list, plus scratch slots kept aside. The page table has one entry per level 0
    /// page: the finest resident page over it, slot | level &lt;&lt; 24; the CPU keeps it and <see cref="RecordTable"/>
    /// copies it up on a list when it changed. Each slot remembers the frame whose list last wrote it, so a readback only
    /// reads slots whose writes were handed to the GPU before it (<see cref="Settled"/>).
    /// </summary>
    public sealed class PageSlots
    {
        private const int LevelShift = 24;

        public int PagesAcross { get; private set; }
        public int Levels { get; private set; }
        /// <summary>Slots, scratch included.</summary>
        public int Capacity { get; private set; }
        public bool AllResident { get; private set; }
        public int ScratchCount { get; private set; }

        // Per level, per page: its slot, or -1 when it is not resident
        private int[][] _slots = Array.Empty<int[]>();
        // Per slot: the page in it (level << 24 | index at that level), or -1
        private int[] _pageOfSlot = Array.Empty<int>();
        // Per slot: the frame whose command list last wrote it; long.MinValue = never
        private long[] _writtenInFrame = Array.Empty<long>();
        // Per slot: in the page table. A page baked over several frames is given its slot hidden and shown once whole
        private bool[] _shown = Array.Empty<bool>();
        private readonly Stack<int> _free = new();
        private int _firstScratch;
        private uint[] _entries = Array.Empty<uint>();
        private bool _entriesDirty;
        private GraphicsBuffer? _table;

        /// <summary>The page table (StructuredBuffer&lt;uint&gt;, one entry per level 0 page), read only.</summary>
        public uint TableSrv => _table?.SrvIndex ?? 0u;

        /// <summary>The table needs copying up (<see cref="RecordTable"/>).</summary>
        public bool TableDirty => _entriesDirty;

        /// <summary>Slots that can still take a page.</summary>
        public int FreeSlots => _free.Count;

        /// <summary>Levels from 0 up to the one whose single page covers the map.</summary>
        public static int LevelsFor(int pagesAcross)
        {
            int levels = 1;
            while (((pagesAcross + (1 << (levels - 1)) - 1) >> (levels - 1)) > 1) levels++;
            return levels;
        }

        /// <summary>Pages a side at level L of a map of this many level 0 pages a side.</summary>
        public static int PagesAcrossAt(int pagesAcross, int level) => Math.Max(1, (pagesAcross + (1 << level) - 1) >> level);

        public int PagesAcrossAt(int level) => PagesAcrossAt(PagesAcross, level);

        /// <summary>
        /// Start over: this many level 0 pages a side, in <paramref name="pageSlots"/> slots plus <paramref name="scratch"/>
        /// scratch ones, or (<paramref name="allResident"/>) every level 0 page at its own index. The table is made anew
        /// (the old one released behind the frames in flight) and copied up on its next <see cref="RecordTable"/>.
        /// </summary>
        public void Reset(int pagesAcross, int pageSlots, int scratch, bool allResident)
        {
            Release();
            int pages = pagesAcross * pagesAcross;
            PagesAcross = pagesAcross;
            AllResident = allResident;
            Levels = allResident ? 1 : LevelsFor(pagesAcross);
            ScratchCount = allResident ? 0 : scratch;
            Capacity = allResident ? pages : pageSlots + scratch;

            _slots = new int[Levels][];
            for (int l = 0; l < Levels; l++)
            {
                int n = PagesAcrossAt(l);
                _slots[l] = new int[n * n];
                Array.Fill(_slots[l], -1);
            }
            _pageOfSlot = new int[Capacity];
            Array.Fill(_pageOfSlot, -1);
            _writtenInFrame = new long[Capacity];
            Array.Fill(_writtenInFrame, long.MinValue);
            _shown = new bool[Capacity];
            if (allResident) Array.Fill(_shown, true);
            _free.Clear();
            _firstScratch = Capacity - ScratchCount;
            if (allResident)
            {
                for (int i = 0; i < pages; i++)
                {
                    _slots[0][i] = i;
                    _pageOfSlot[i] = i;
                }
            }
            else
            {
                for (int s = _firstScratch - 1; s >= 0; s--) _free.Push(s);
            }

            _entries = new uint[pages];
            for (int qy = 0; qy < pagesAcross; qy++)
                for (int qx = 0; qx < pagesAcross; qx++)
                    _entries[qy * pagesAcross + qx] = EntryFor(qx, qy);
            _table = GraphicsBuffer.CreateStructured<uint>(pages, srv: true);
            _entriesDirty = true;
        }

        /// <summary>The slot of page (px, py) at a level, or -1 when it is not resident.</summary>
        public int SlotOf(int level, int px, int py) =>
            level < Levels ? _slots[level][py * PagesAcrossAt(level) + px] : -1;

        /// <summary>The page in a slot: its level and place, or null when the slot is free or a scratch one.</summary>
        public (int level, int px, int py)? PageIn(int slot)
        {
            int packed = slot >= 0 && slot < _pageOfSlot.Length ? _pageOfSlot[slot] : -1;
            if (packed < 0) return null;
            int level = packed >> LevelShift, index = packed & ((1 << LevelShift) - 1);
            int n = PagesAcrossAt(level);
            return (level, index % n, index / n);
        }

        /// <summary>Give page (px, py) of a level a slot from the free list (its content is the caller's to write): the slot,
        /// or -1 when none is free. The page table follows on its next copy; with <paramref name="shown"/> false only once
        /// <see cref="Show"/> says the slot is whole (the coarser page over it shows meanwhile).</summary>
        public int Add(int level, int px, int py, bool shown = true)
        {
            if (AllResident || level >= Levels) return -1;
            int index = py * PagesAcrossAt(level) + px;
            if (_slots[level][index] >= 0) return _slots[level][index];
            if (_free.Count == 0) return -1;
            int slot = _free.Pop();
            _slots[level][index] = slot;
            _pageOfSlot[slot] = (level << LevelShift) | index;
            _writtenInFrame[slot] = long.MinValue;
            _shown[slot] = shown;
            if (shown) UpdateEntries(level, px, py);
            return slot;
        }

        /// <summary>A slot given out hidden is whole: the table takes it on its next copy.</summary>
        public void Show(int slot)
        {
            if (slot < 0 || slot >= _shown.Length || _shown[slot]) return;
            _shown[slot] = true;
            if (PageIn(slot) is { } page) UpdateEntries(page.level, page.px, page.py);
        }

        /// <summary>Whether the slot is in the page table.</summary>
        public bool Shown(int slot) => slot >= 0 && slot < _shown.Length && _shown[slot];

        /// <summary>Let page (px, py) of a level go: its slot back to the free list, the table over it to what is left.</summary>
        public void Remove(int level, int px, int py)
        {
            if (AllResident || level >= Levels) return;
            int index = py * PagesAcrossAt(level) + px;
            int slot = _slots[level][index];
            if (slot < 0) return;
            _slots[level][index] = -1;
            _pageOfSlot[slot] = -1;
            bool wasShown = _shown[slot];
            _shown[slot] = false;
            _free.Push(slot);
            if (wasShown) UpdateEntries(level, px, py);
        }

        /// <summary>Scratch slot i (0 .. <see cref="ScratchCount"/> − 1), or -1 in a layout without them.</summary>
        public int ScratchSlot(int i) => i >= 0 && i < ScratchCount ? _firstScratch + i : -1;

        /// <summary>
        /// A command list wrote the slot: <paramref name="submitted"/> when it has already run (SubmitAndWait), else the
        /// frame's own list, which runs after this frame's CPU work. Frames are counted by <c>Engine.TickCount</c>, never
        /// <c>Engine.FrameIndex</c>: that is the back buffer's index and comes round every three frames (until 2026-11-07 it
        /// was used here, and a slot written in a frame of index 2 never counted as settled).
        /// </summary>
        public void MarkWritten(int slot, bool submitted = false)
        {
            if (slot >= 0 && slot < _writtenInFrame.Length)
                _writtenInFrame[slot] = submitted ? long.MinValue + 1 : Engine.TickCount;
        }

        /// <summary>Whatever last wrote the slot has been handed to the GPU before this frame, and something has.</summary>
        public bool Settled(int slot) =>
            slot >= 0 && slot < _writtenInFrame.Length && _writtenInFrame[slot] != long.MinValue && _writtenInFrame[slot] < Engine.TickCount;

        /// <summary>Whether something has written the slot since it was given out.</summary>
        public bool Written(int slot) => slot >= 0 && slot < _writtenInFrame.Length && _writtenInFrame[slot] != long.MinValue;

        // The table entries over a page of a level: each level 0 page under it gets the finest resident page over it
        private void UpdateEntries(int level, int px, int py)
        {
            int span = 1 << level;
            int x0 = px * span, y0 = py * span;
            int x1 = Math.Min(x0 + span, PagesAcross), y1 = Math.Min(y0 + span, PagesAcross);
            for (int qy = y0; qy < y1; qy++)
                for (int qx = x0; qx < x1; qx++)
                    _entries[qy * PagesAcross + qx] = EntryFor(qx, qy);
            _entriesDirty = true;
        }

        private uint EntryFor(int qx, int qy)
        {
            for (int l = 0; l < Levels; l++)
            {
                int s = _slots[l][(qy >> l) * PagesAcrossAt(l) + (qx >> l)];
                if (s >= 0 && _shown[s]) return (uint)s | ((uint)l << LevelShift);
            }
            return 0u;   // nothing resident over it yet
        }

        /// <summary>
        /// Copy the page table up on this list when it changed. In the frame's own list before its terrain reads: the
        /// frames before it ran on the table they had, and pages written on the same list before it are there.
        /// </summary>
        public void RecordTable(ID3D12GraphicsCommandList cmd)
        {
            if (!_entriesDirty || _table == null) return;
            _entriesDirty = false;
            var upload = GraphicsBuffer.CreateUpload<uint>(_entries.Length, mapped: true);
            unsafe
            {
                var dst = upload.WritePtr<uint>();
                for (int i = 0; i < _entries.Length; i++) dst[i] = _entries[i];
            }
            _table.Transition(cmd, ResourceStates.CopyDest);
            cmd.CopyBufferRegion(_table.Native, 0, upload.Native, 0, (ulong)(_entries.Length * sizeof(uint)));
            _table.Transition(cmd, ResourceStates.AllShaderResource);
            Engine.Device.DeferDispose(upload);   // read when the list runs
        }

        /// <summary>The table, released behind the frames in flight.</summary>
        public void Release()
        {
            Engine.Device?.DeferDispose(_table);
            _table = null;
            _free.Clear();
            Capacity = 0;
        }
    }
}
