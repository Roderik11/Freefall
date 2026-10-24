using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Numerics;

namespace Freefall.Graphics
{
    /// <summary>
    /// Global persistent GPU buffer for all entity transforms + per-instance materialIDs.
    /// Thin wrapper around StreamingBuffer&lt;T&gt; with a free-list slot allocator.
    /// </summary>
    public class TransformBuffer : IDisposable
    {
        private const int InitialCapacity = 1024;

        private StreamingBuffer<Matrix4x4> _transforms;
        private StreamingBuffer<uint> _materials;

        // The slots uploaded this tick, one bit per slot (cull_instances.hlsl ChangedSlotsIdx): the camera culling's
        // transform cache (InstanceBatch) reads an instance's transform again only when its slot's bit is set. A tick
        // that renders several views uploads once per view; the bits are the union of them. A Set reaches each of the
        // three frame buffers at its next upload, so a slot changed once is set in the bits of the next ticks too.
        private StreamingBuffer<uint> _uploadedSlots;
        private uint[] _uploadedBits = Array.Empty<uint>();
        private int _uploadedTick = -1;
        private int _uploadedCount;

        // Slot allocation. Slots are allocated lazily on first Transform.TransformSlot access, which
        // happens inside the parallel draw loop (ComponentCache.Draw) — the allocator must be thread-safe.
        private readonly object _slotLock = new();
        private readonly Stack<int> _freeSlots = new();
        private int _nextSlot = 0;
        private int _activeSlots = 0;
        
        // Reverse lookup: slot → Entity (for mouse picking)
        private readonly ConcurrentDictionary<int, Base.Entity> _slotToEntity = new();

        // Singleton instance
        public static TransformBuffer Instance { get; private set; } = null!;

        public int ActiveSlots => _activeSlots;

        /// <summary>
        /// The transforms the next <see cref="Upload"/> writes (the profiler's count on "Transform upload"). Upload's
        /// cost follows it: a moved entity rewrites the slots of everything under it.
        /// </summary>
        public int PendingUploads => _transforms.DirtyCount;

        /// <summary>
        /// Bindless SRV index for the current frame's transform buffer.
        /// </summary>
        public uint SrvIndex => _transforms.SrvIndex;

        /// <summary>
        /// Bindless SRV index for the current frame's materialID buffer.
        /// </summary>
        public uint MaterialSrvIndex => _materials.SrvIndex;

        /// <summary>
        /// Bindless SRV index of the bits of the slots uploaded this tick (StructuredBuffer&lt;uint&gt;, one bit per slot),
        /// or 0 when no slot was uploaded this tick.
        /// </summary>
        public uint ChangedSlotsSrvIndex =>
            _uploadedTick == Engine.TickCount && _uploadedCount > 0 ? _uploadedSlots.SrvIndex : 0;

        public static void Initialize(GraphicsDevice device)
        {
            Instance = new TransformBuffer(device);
            Debug.Log($"[TransformBuffer] Initialized (dynamic, starting at {InitialCapacity} slots)");
        }

        private TransformBuffer(GraphicsDevice device)
        {
            _transforms = new StreamingBuffer<Matrix4x4>(device, InitialCapacity);
            _materials = new StreamingBuffer<uint>(device, InitialCapacity);
            _uploadedSlots = new StreamingBuffer<uint>(device, InitialCapacity / 32);
        }

        /// <summary>
        /// Allocate a slot for an entity. Returns the slot index.
        /// </summary>
        public int AllocateSlot()
        {
            lock (_slotLock)
            {
                int slot;
                if (_freeSlots.Count > 0)
                {
                    slot = _freeSlots.Pop();
                }
                else
                {
                    slot = _nextSlot++;
                    _transforms.EnsureCapacity(_nextSlot);
                    _materials.EnsureCapacity(_nextSlot);
                }

                _activeSlots++;
                _transforms.Set(slot, Matrix4x4.Identity);
                return slot;
            }
        }

        /// <summary>
        /// Release a slot when entity is destroyed.
        /// </summary>
        public void ReleaseSlot(int slot)
        {
            lock (_slotLock)
            {
                if (slot < 0 || slot >= _nextSlot) return;
                _freeSlots.Push(slot);
                _activeSlots--;
                _transforms.Set(slot, Matrix4x4.Identity);
                _slotToEntity.TryRemove(slot, out _);
            }
        }

        /// <summary>
        /// Set transform for a slot. Pre-transposes for GPU consumption.
        /// </summary>
        public void SetTransform(int slot, Matrix4x4 world)
        {
            if (slot < 0 || slot >= _nextSlot) return;
            _transforms.Set(slot, Matrix4x4.Transpose(world));
        }

        /// <summary>
        /// Set materialID for a slot. Enables per-instance materials indexed by TransformSlot.
        /// </summary>
        public void SetMaterialId(int slot, uint materialId)
        {
            if (slot < 0 || slot >= _nextSlot) return;
            _materials.Set(slot, materialId);
        }

        /// <summary>
        /// Register an entity for a slot (for mouse picking reverse lookup).
        /// </summary>
        public void RegisterEntity(int slot, Base.Entity entity)
        {
            if (slot >= 0) _slotToEntity[slot] = entity;
        }

        /// <summary>
        /// Get the entity at a given TransformSlot, or null if not found.
        /// </summary>
        public Base.Entity? GetEntityBySlot(int slot)
        {
            return _slotToEntity.GetValueOrDefault(slot);
        }

        /// <summary>
        /// Get transform directly from CPU-side staging (already transposed).
        /// </summary>
        public Matrix4x4 GetTransformDirect(int slot)
        {
            if (slot < 0 || slot >= _nextSlot) return Matrix4x4.Identity;
            return _transforms.Get(slot);
        }

        /// <summary>
        /// Upload dirty transforms and materialIDs to the current frame's GPU buffers.
        /// Call once per frame before GPU dispatch.
        /// </summary>
        public void Upload()
        {
            if (_uploadedTick != Engine.TickCount)
            {
                if (_uploadedCount > 0)
                    Array.Clear(_uploadedBits);
                _uploadedCount = 0;
                _uploadedTick = Engine.TickCount;
            }
            int words = (_transforms.Capacity + 31) / 32;
            if (_uploadedBits.Length < words)
                Array.Resize(ref _uploadedBits, words);

            int uploaded = _transforms.Upload(_uploadedBits);
            if (uploaded > 0)
            {
                _uploadedCount += uploaded;
                _uploadedSlots.BulkWrite(_uploadedBits);
            }
            _materials.Upload();
            StreamingBuffer<Matrix4x4>.FlushDeferredDisposals();
        }

        public void Dispose()
        {
            _transforms?.Dispose();
            _materials?.Dispose();
            _uploadedSlots?.Dispose();
        }
    }
}
