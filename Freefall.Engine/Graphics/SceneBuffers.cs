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
        public static GPUBuffer<InstanceDescriptor> Descriptors { get; private set; } = null!;
        public static GPUBuffer<Vector4> BoundingSpheres { get; private set; } = null!;
        public static GPUBuffer<uint> MeshPartIds { get; private set; } = null!;

        /// <summary>High-water mark: maximum slot index currently in use + 1.</summary>
        public static int ActiveSlotCount => Slots.HighWaterMark;

        private static bool _initialized;

        /// <summary>
        /// Initialize all buffers. Call once during engine startup.
        /// </summary>
        public static void Initialize(GraphicsDevice device, int initialCapacity = 4096)
        {
            if (_initialized) return;

            Slots = new RenderSlotAllocator();
            Transforms = new GPUBuffer<Matrix4x4>(device, initialCapacity);
            Descriptors = new GPUBuffer<InstanceDescriptor>(device, initialCapacity);
            BoundingSpheres = new GPUBuffer<Vector4>(device, initialCapacity);
            MeshPartIds = new GPUBuffer<uint>(device, initialCapacity);

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
            Descriptors.Upload();
            BoundingSpheres.Upload();
            MeshPartIds.Upload();

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
            Descriptors.EnsureCapacity(needed);
            BoundingSpheres.EnsureCapacity(needed);
            MeshPartIds.EnsureCapacity(needed);

            return slot;
        }

        /// <summary>
        /// Release a render slot.
        /// </summary>
        public static void ReleaseSlot(int slot)
        {
            // Clear the data in each channel
            Transforms.Set(slot, Matrix4x4.Identity);
            Descriptors.Set(slot, default);
            BoundingSpheres.Set(slot, Vector4.Zero);
            MeshPartIds.Set(slot, 0);

            Slots.Release(slot);
        }

        /// <summary>Bindless SRV index for the current frame's transform buffer.</summary>
        public static uint TransformSrvIndex => Transforms.SrvIndex;
        /// <summary>Bindless SRV index for the current frame's descriptor buffer.</summary>
        public static uint DescriptorSrvIndex => Descriptors.SrvIndex;
        /// <summary>Bindless SRV index for the current frame's bounding sphere buffer.</summary>
        public static uint BoundingSphereSrvIndex => BoundingSpheres.SrvIndex;
        /// <summary>Bindless SRV index for the current frame's mesh part ID buffer.</summary>
        public static uint MeshPartIdSrvIndex => MeshPartIds.SrvIndex;

        public static void Dispose()
        {
            Transforms?.Dispose();
            Descriptors?.Dispose();
            BoundingSpheres?.Dispose();
            MeshPartIds?.Dispose();
            _initialized = false;
        }
    }
}
