using System;
using System.Numerics;

namespace Freefall.Graphics
{
    /// <summary>
    /// Composes multiple GPUBuffer&lt;T&gt; channels with a shared RenderSlotAllocator.
    /// Each renderer gets one slot and writes to whichever channels it uses.
    /// Call UploadAll() once per frame before GPU dispatch.
    /// </summary>
    public static class SceneBuffers
    {
        // Shared slot allocator — every renderer gets exactly one slot
        public static RenderSlotAllocator Slots { get; private set; } = null!;

        // Per-instance GPU data channels
        public static GPUBuffer<Matrix4x4> Transforms { get; private set; } = null!;

        // Future channels (added as needed, no existing code changes required):
        // public static GPUBuffer<BoundingSphere> Bounds { get; private set; }
        // public static GPUBuffer<uint> MeshPartIds { get; private set; }
        // public static GPUBuffer<uint> MaterialIds { get; private set; }
        // public static GPUBuffer<uint> PSOIds { get; private set; }
        // public static GPUBuffer<uint> Visibility { get; private set; }
        // public static GPUBuffer<uint> BoneOffsets { get; private set; }

        private static bool _initialized;

        /// <summary>
        /// Initialize all buffers. Call once during engine startup.
        /// </summary>
        public static void Initialize(GraphicsDevice device, int initialCapacity = 4096)
        {
            if (_initialized) return;

            Slots = new RenderSlotAllocator();
            Transforms = new GPUBuffer<Matrix4x4>(device, initialCapacity);

            _initialized = true;
            Debug.Log($"[SceneBuffers] Initialized (capacity: {initialCapacity})");
        }

        /// <summary>
        /// Upload all dirty data across all channels for the current frame.
        /// Call once per frame before GPU culling/rendering.
        /// </summary>
        public static void UploadAll()
        {
            if (!_initialized) return;

            Transforms.Upload();
            // Future: Bounds.Upload(); MaterialIds.Upload(); etc.

            GPUBuffer<Matrix4x4>.FlushDeferredDisposals();
        }

        /// <summary>
        /// Allocate a render slot and ensure all buffers have capacity.
        /// </summary>
        public static int AllocateSlot()
        {
            int slot = Slots.Allocate();

            // Ensure all channels can hold this slot
            int needed = slot + 1;
            Transforms.EnsureCapacity(needed);

            return slot;
        }

        /// <summary>
        /// Release a render slot.
        /// </summary>
        public static void ReleaseSlot(int slot)
        {
            // Clear the data in each channel
            Transforms.Set(slot, Matrix4x4.Identity);

            Slots.Release(slot);
        }

        /// <summary>
        /// Bindless SRV index for the current frame's transform buffer.
        /// Backward-compatible with existing shaders that read from the transform buffer.
        /// </summary>
        public static uint TransformSrvIndex => Transforms.SrvIndex;

        public static void Dispose()
        {
            Transforms?.Dispose();
            _initialized = false;
        }
    }
}
