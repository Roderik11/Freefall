using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// Generic triple-buffered GPU upload buffer with dirty-list tracking and persistent mapping.
    /// Each element type T gets its own SoA buffer. Used by SceneBuffers to compose per-instance data channels.
    /// </summary>
    public unsafe class GPUBuffer<T> : IDisposable where T : unmanaged
    {
        private const int FrameCount = 3;

        // CPU staging (fast reads, source-of-truth for propagation)
        private T[] _staging;

        // GPU: triple-buffered upload heaps
        private ID3D12Resource[] _buffers = new ID3D12Resource[FrameCount];
        private T*[] _mapped = new T*[FrameCount]; // persistently mapped pointers
        private uint[] _srvIndices = new uint[FrameCount];

        // Dirty tracking: per-frame lists of modified slot indices
        private HashSet<int>[] _dirtySlots = new HashSet<int>[FrameCount];
        private bool[] _anyDirty = new bool[FrameCount];

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
        /// Current capacity in number of elements.
        /// </summary>
        public int Capacity => _capacity;

        public GPUBuffer(GraphicsDevice device, int initialCapacity = 1024)
        {
            _device = device;
            _stride = sizeof(T);
            _capacity = initialCapacity;
            _staging = new T[initialCapacity];

            for (int f = 0; f < FrameCount; f++)
                _dirtySlots[f] = new HashSet<int>();

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

            // Mark dirty for all 3 frame buffers
            for (int f = 0; f < FrameCount; f++)
            {
                lock (_dirtySlots[f])
                {
                    _dirtySlots[f].Add(slot);
                    _anyDirty[f] = true;
                }
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
        public void Upload()
        {
            int frame = Engine.FrameIndex % FrameCount;

            if (!_anyDirty[frame]) return;

            HashSet<int> dirty;
            lock (_dirtySlots[frame])
            {
                dirty = _dirtySlots[frame];
            }

            T* dst = _mapped[frame];
            foreach (int slot in dirty)
            {
                dst[slot] = _staging[slot];
            }

            lock (_dirtySlots[frame])
            {
                dirty.Clear();
                _anyDirty[frame] = false;
            }
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

            // Re-upload all existing data to new buffers
            // Mark everything as dirty for all frames
            for (int f = 0; f < FrameCount; f++)
            {
                lock (_dirtySlots[f])
                {
                    for (int i = 0; i < _capacity; i++)
                        _dirtySlots[f].Add(i);
                    _anyDirty[f] = true;
                }
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
