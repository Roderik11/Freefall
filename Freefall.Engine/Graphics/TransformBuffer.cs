using System;
using System.Collections.Generic;
using System.Numerics;

namespace Freefall.Graphics
{
    /// <summary>
    /// Global persistent GPU buffer for all entity transforms + per-instance materialIDs.
    /// Thin wrapper around GPUBuffer&lt;T&gt; with a free-list slot allocator.
    /// </summary>
    public class TransformBuffer : IDisposable
    {
        private const int InitialCapacity = 1024;

        private GPUBuffer<Matrix4x4> _transforms;
        private GPUBuffer<uint> _materials;

        // Slot allocation
        private readonly Stack<int> _freeSlots = new();
        private int _nextSlot = 0;
        private int _activeSlots = 0;

        // Singleton instance
        public static TransformBuffer Instance { get; private set; } = null!;

        public int ActiveSlots => _activeSlots;

        /// <summary>
        /// Bindless SRV index for the current frame's transform buffer.
        /// </summary>
        public uint SrvIndex => _transforms.SrvIndex;

        /// <summary>
        /// Bindless SRV index for the current frame's materialID buffer.
        /// </summary>
        public uint MaterialSrvIndex => _materials.SrvIndex;

        public static void Initialize(GraphicsDevice device)
        {
            Instance = new TransformBuffer(device);
            Debug.Log($"[TransformBuffer] Initialized (dynamic, starting at {InitialCapacity} slots)");
        }

        private TransformBuffer(GraphicsDevice device)
        {
            _transforms = new GPUBuffer<Matrix4x4>(device, InitialCapacity);
            _materials = new GPUBuffer<uint>(device, InitialCapacity);
        }

        /// <summary>
        /// Allocate a slot for an entity. Returns the slot index.
        /// </summary>
        public int AllocateSlot()
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

        /// <summary>
        /// Release a slot when entity is destroyed.
        /// </summary>
        public void ReleaseSlot(int slot)
        {
            if (slot < 0 || slot >= _nextSlot) return;
            _freeSlots.Push(slot);
            _activeSlots--;
            _transforms.Set(slot, Matrix4x4.Identity);
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
            _transforms.Upload();
            _materials.Upload();
            GPUBuffer<Matrix4x4>.FlushDeferredDisposals();
        }

        public void Dispose()
        {
            _transforms?.Dispose();
            _materials?.Dispose();
        }
    }
}
