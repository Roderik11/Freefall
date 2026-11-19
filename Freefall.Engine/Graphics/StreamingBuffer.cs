using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Numerics;
using System.Threading;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// Generic triple-buffered GPU upload buffer with dirty-list tracking and persistent mapping.
    /// Each element type T gets its own SoA buffer. Used by SceneBuffers to compose per-instance data channels.
    /// </summary>
    public unsafe class StreamingBuffer<T> : IDisposable where T : unmanaged
    {
        private const int FrameCount = 3;

        // CPU staging (fast reads, source-of-truth for propagation)
        private T[] _staging;

        // GPU: triple-buffered upload heaps
        private ID3D12Resource[] _buffers = new ID3D12Resource[FrameCount];
        private T*[] _mapped = new T*[FrameCount]; // persistently mapped pointers
        private uint[] _srvIndices = new uint[FrameCount];

        // Dirty tracking, per frame buffer: a bit per slot, set by Set from any thread without a lock, and walked by Upload
        // for the frame it uploads. A lock and a HashSet per frame buffer made the characters' transform writes on workers
        // slower than one thread doing them all: every child transform took the three locks (2026-11-03). After a Grow a
        // frame buffer is wholly dirty.
        private uint[][] _dirtyBits = new uint[FrameCount][];
        private bool[] _anyDirty = new bool[FrameCount];
        private bool[] _allDirty = new bool[FrameCount];

        // Capacity
        private int _capacity;
        private readonly int _stride;
        private readonly GraphicsDevice _device;

        // Deferred disposal for old buffers during resize
        private static readonly List<(ID3D12Resource? Resource, uint BindlessIndex, long DisposeAfterTick)> _deferredDisposals = new();

        /// <summary>
        /// Bindless SRV index for the current frame's buffer — pass to shaders.
        /// </summary>
        public uint SrvIndex => _srvIndices[Engine.FrameIndex % FrameCount];

        /// <summary>
        /// Bindless SRV index of last frame's buffer, which holds every slot as last frame drew it (each frame's buffer
        /// is brought up to date at its own <see cref="Upload"/>): the previous transforms of the motion vectors. The
        /// GPU is done with the frame before last before this frame writes (RenderView.Prepare), so nothing rewrites
        /// it while this frame reads it. For two ticks after a <see cref="Grow"/> the new buffers are not all filled
        /// yet: then the current one, as if nothing had moved.
        /// </summary>
        public uint PreviousSrvIndex => Engine.TickCount - _grownTick <= 2
            ? SrvIndex
            : _srvIndices[(Engine.FrameIndex + FrameCount - 1) % FrameCount];
        private long _grownTick = long.MinValue / 2;

        /// <summary>
        /// Frame index of last BulkWrite/Upload. Used to avoid redundant writes within a frame.
        /// </summary>
        public int LastWriteFrame = -1;

        // The slot BulkWriteNext wrote last; -1 = none yet
        private int _writtenSlot = -1;

        /// <summary>
        /// Bindless SRV index of the slot <see cref="BulkWriteNext"/> wrote last: what to draw with until the next
        /// write. The current frame's slot before the first.
        /// </summary>
        public uint WrittenSrvIndex => _writtenSlot >= 0 ? _srvIndices[_writtenSlot] : SrvIndex;

        /// <summary>Whether <see cref="BulkWriteNext"/> has written a slot yet (<see cref="WrittenSrvIndex"/> is one).</summary>
        public bool Written => _writtenSlot >= 0;

        /// <summary>
        /// Current capacity in number of elements.
        /// </summary>
        public int Capacity => _capacity;

        /// <summary>
        /// The slots the next <see cref="Upload"/> writes: those set since the current frame's buffer was last written.
        /// </summary>
        public int DirtyCount
        {
            get
            {
                int frame = Engine.FrameIndex % FrameCount;
                if (_allDirty[frame]) return _capacity;
                int count = 0;
                foreach (uint word in _dirtyBits[frame]) count += BitOperations.PopCount(word);
                return count;
            }
        }

        public StreamingBuffer(GraphicsDevice device, int initialCapacity = 1024)
        {
            _device = device;
            _stride = sizeof(T);
            _capacity = initialCapacity;
            _staging = new T[initialCapacity];

            for (int f = 0; f < FrameCount; f++)
                _dirtyBits[f] = new uint[(initialCapacity + 31) / 32];

            AllocateGPUResources(initialCapacity);
        }

        /// <summary>
        /// Write a value to a slot. O(1) — appends to dirty list for all 3 frame buffers.
        /// Thread-safe for different slots. Same-slot concurrent writes are caller's responsibility.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Set(int slot, T value)
        {
            _staging[slot] = value;

            // Dirty for all 3 frame buffers: a bit already set costs a read, a new one an atomic or (two threads may set
            // two bits of one word). Upload never runs beside a Set: it is the main thread's, between parallel work.
            int word = slot >> 5;
            uint bit = 1u << (slot & 31);
            for (int f = 0; f < FrameCount; f++)
            {
                ref uint bits = ref _dirtyBits[f][word];
                if ((bits & bit) == 0) Interlocked.Or(ref bits, bit);
                _anyDirty[f] = true;
            }
        }

        /// <summary>
        /// Read a value from CPU staging (fast, normal memory — not write-combined).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T Get(int slot) => _staging[slot];

        /// <summary>
        /// Upload dirty slots to the current frame's GPU buffer. O(D) where D = dirty count.
        /// Call once per frame before GPU dispatch.
        /// </summary>
        /// <param name="uploadedBits">Optional, one bit per slot (at least Capacity bits): the bits of the slots written
        /// are set. The caller clears them.</param>
        /// <returns>The number of slots written.</returns>
        public int Upload(uint[]? uploadedBits = null)
        {
            int frame = Engine.FrameIndex % FrameCount;

            if (!_anyDirty[frame]) return 0;

            T* dst = _mapped[frame];
            var bits = _dirtyBits[frame];
            int count = 0;
            if (_allDirty[frame])
            {
                for (int slot = 0; slot < _capacity; slot++) dst[slot] = _staging[slot];
                if (uploadedBits != null)
                    for (int slot = 0; slot < _capacity; slot++) uploadedBits[slot >> 5] |= 1u << (slot & 31);
                count = _capacity;
                Array.Clear(bits);
                _allDirty[frame] = false;
            }
            else
            {
                for (int w = 0; w < bits.Length; w++)
                {
                    uint word = bits[w];
                    if (word == 0) continue;
                    bits[w] = 0;
                    if (uploadedBits != null) uploadedBits[w] |= word;
                    while (word != 0)
                    {
                        int slot = (w << 5) + BitOperations.TrailingZeroCount(word);
                        dst[slot] = _staging[slot];
                        word &= word - 1;
                        count++;
                    }
                }
            }
            _anyDirty[frame] = false;
            return count;
        }

        /// <summary>
        /// Write all elements directly to current frame's mapped buffer.
        /// Use for data that's fully rewritten every frame (e.g. bone matrices).
        /// Bypasses dirty tracking — cheaper than N × Set() calls.
        /// </summary>
        public void BulkWrite(ReadOnlySpan<T> data)
        {
            EnsureCapacity(data.Length);
            int frame = Engine.FrameIndex % FrameCount;
            T* dst = _mapped[frame];
            fixed (T* src = data)
                System.Buffer.MemoryCopy(src, dst, _capacity * _stride, data.Length * _stride);
        }

        /// <summary>
        /// Like <see cref="BulkWrite"/>, for data that is not written every frame (a character posed every few
        /// frames): into the slot after the one this wrote last, then drawn with <see cref="WrittenSrvIndex"/>.
        /// The frames still on the GPU read the slot written last or the one before, never that one. BulkWrite
        /// picks the slot by frame instead, so a write a multiple of three frames after the last would land in
        /// the slot those frames read.
        /// </summary>
        public void BulkWriteNext(ReadOnlySpan<T> data)
        {
            EnsureCapacity(data.Length);
            _writtenSlot = (_writtenSlot + 1) % FrameCount;
            T* dst = _mapped[_writtenSlot];
            fixed (T* src = data)
                System.Buffer.MemoryCopy(src, dst, _capacity * _stride, data.Length * _stride);
        }

        /// <summary>
        /// Grow the buffer to at least newCapacity elements.
        /// Old GPU resources are deferred-disposed after FrameCount+1 ticks.
        /// </summary>
        public void Grow(int newCapacity)
        {
            if (newCapacity <= _capacity) return;

            // Double until sufficient
            int target = _capacity;
            while (target < newCapacity) target *= 2;

            // Resize CPU staging
            Array.Resize(ref _staging, target);

            // Defer disposal of old GPU resources
            for (int f = 0; f < FrameCount; f++)
            {
                _mapped[f] = null; // unmap happens on dispose
                _buffers[f]?.Unmap(0);
                DeferDispose(_buffers[f]!, _srvIndices[f]);
            }

            _capacity = target;
            AllocateGPUResources(target);
            _grownTick = Engine.TickCount;

            // Re-upload all existing data to new buffers: everything dirty for all frames
            for (int f = 0; f < FrameCount; f++)
            {
                Array.Resize(ref _dirtyBits[f], (target + 31) / 32);
                _allDirty[f] = true;
                _anyDirty[f] = true;
            }
        }

        /// <summary>
        /// Ensure capacity for at least 'needed' elements, growing if necessary.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void EnsureCapacity(int needed)
        {
            if (needed > _capacity)
                Grow(needed);
        }

        /// <summary>
        /// Flush deferred disposals that have passed their safe tick.
        /// Call once per frame (from SceneBuffers or Engine).
        /// </summary>
        public static void FlushDeferredDisposals()
        {
            long currentTick = Engine.TickCount;
            for (int i = _deferredDisposals.Count - 1; i >= 0; i--)
            {
                var (resource, bindlessIndex, disposeAfterTick) = _deferredDisposals[i];
                if (currentTick >= disposeAfterTick)
                {
                    resource?.Dispose();
                    if (bindlessIndex > 0)
                        Engine.Device.ReleaseBindlessIndex(bindlessIndex);
                    _deferredDisposals.RemoveAt(i);
                }
            }
        }

        public void Dispose()
        {
            for (int f = 0; f < FrameCount; f++)
            {
                if (_mapped[f] != null)
                {
                    _buffers[f]?.Unmap(0);
                    _mapped[f] = null;
                }
                _buffers[f]?.Dispose();
                if (_srvIndices[f] > 0)
                    Engine.Device.ReleaseBindlessIndex(_srvIndices[f]);
            }
        }

        // ────────────── Private ──────────────

        private void AllocateGPUResources(int capacity)
        {
            int bufferSize = capacity * _stride;

            for (int f = 0; f < FrameCount; f++)
            {
                // Create upload heap buffer
                _buffers[f] = _device.CreateUploadBuffer(bufferSize);

                // Persistent map — one kernel transition, kept open until Dispose
                void* ptr;
                _buffers[f].Map(0, null, &ptr);
                _mapped[f] = (T*)ptr;

                // Clear mapped memory
                NativeMemory.Clear(ptr, (nuint)bufferSize);

                // Allocate bindless SRV
                _srvIndices[f] = _device.AllocateBindlessIndex();
                _device.CreateStructuredBufferSRV(_buffers[f], (uint)capacity, (uint)_stride, _srvIndices[f]);
            }
        }

        private static void DeferDispose(ID3D12Resource resource, uint bindlessIndex)
        {
            _deferredDisposals.Add((resource, bindlessIndex, Engine.TickCount + FrameCount + 1));
        }
    }
}
