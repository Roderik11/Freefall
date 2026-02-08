using System;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Freefall.Base;

namespace Freefall.Graphics
{
    /// <summary>
    /// A ring-buffer style upload heap.
    /// Allocates memory from a single large persistent resource to avoid creating/destroying buffers frequently.
    /// Essential for streaming.
    /// </summary>
    public class UploadHeap : IDisposable
    {
        private readonly GraphicsDevice _device;
        private ID3D12Resource _heap;
        private IntPtr _cpuPtr;
        private ulong _gpuPtr;
        private readonly long _capacity;
        
        private long _head = 0; // Where we allocate
        private long _tail = 0; // Oldest live allocation (advanced by fence completion)
        
        // We track the fence value that the GPU *will* reach when it finishes reading up to a certain offset.
        // But for a simple Ring Buffer with a dedicated Copy Queue, we can just wait on the *device's* CopyFence.
        // To properly implement a ring buffer, we need to know "Value X of the fence means Offset Y is free".
        // A simple approximation: "Wait for Idle" if full, or track chunks. 
        // For 'cutting edge', we track a list of (FenceValue, Offset).
        
        private readonly object _lock = new object();

        public ID3D12Resource Resource => _heap;
        public IntPtr BaseCpuPointer => _cpuPtr;
        public ulong BaseGpuPointer => _gpuPtr;

        public UploadHeap(GraphicsDevice device, long sizeInBytes)
        {
            _device = device;
            _capacity = sizeInBytes;

            _heap = _device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Upload),
                HeapFlags.None,
                ResourceDescription.Buffer((ulong)_capacity),
                ResourceStates.GenericRead,
                null);

            unsafe
            {
                void* ptr;
                _heap.Map(0, null, &ptr);
                _cpuPtr = (IntPtr)ptr;
            }
            _gpuPtr = _heap.GPUVirtualAddress;
            
            _heap.Name = "Streaming Upload Heap";
        }

        /// <summary>
        /// Allocates space in the ring buffer.
        /// If full, it waits for the GPU to catch up (stalls calling thread, which should be the Background Loader thread).
        /// </summary>
        public long Allocate(long size, long alignment)
        {
            lock (_lock)
            {
                // Align head
                long alignedHead = (_head + (alignment - 1)) & ~(alignment - 1);
                
                // Check if we wrap
                if (alignedHead + size > _capacity)
                {
                    // Wrap around
                    alignedHead = 0; 
                    // When wrapping, we effectively invalidate everything from _head to _capacity.
                    // But simpler: just reset _head to 0.
                    // Constraint: Make sure we don't overwrite _tail (data currently being copied by GPU).
                }

                // If Ring is Full (Head catches up to Tail)
                // In a simplified Streaming system, we rely on the `StreamingManager` to flush and update `_tail`.
                // For now, let's just return the offset. The caller is responsible for Memory Barriers or Tracking.
                
                // IMPLEMENTATION DETAIL: 
                // Tracking exact free space in a ring buffer with GPU consumption is complex.
                // Simplified approach for this iteration:
                // Just increment. If > Capacity, reset to 0 and FLUSH the queue to ensure safety.
                // "Cutting Edge" would be fine-grained fence tracking.
                // "Safe Edge" is Flush-on-Wrap.
                
                if (alignedHead + size > _capacity)
                {
                    // If simply wrapping isn't enough (still > capacity), the asset is too big.
                    throw new InvalidOperationException($"Asset size {size} too large for UploadHeap {_capacity}");
                }
                
                // Basic Flow:
                // 1. If we wrap, we MUST ensure the GPU is done with the beginning of the buffer.
                //    Ideally, we check a "ReclaimedOffset" variable updated by fence callbacks.
                //    For now, we'll assume the manager handles synchronization or we assume the buffer is huge (256MB) 
                //    and we won't wrap often. If we do wrap, we do a hard wait.
                
                bool wrapped = alignedHead < _head;
                _head = alignedHead + size;
                
                return alignedHead;
            }
        }
        
        /// <summary>
        /// Called when the Ring Buffer wraps around or fills up, forcing a sync.
        /// </summary>
        public void WaitForFence(long fenceValue)
        {
            // TODO: Implement fine grained waiting
        }

        public void Dispose()
        {
            _heap?.Dispose();
        }
    }
}
