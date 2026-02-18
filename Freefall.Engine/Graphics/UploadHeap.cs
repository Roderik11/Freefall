using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using Vortice.Direct3D12;
using Freefall.Base;

namespace Freefall.Graphics
{
    /// <summary>
    /// A ring-buffer style upload heap with fence-based space reclamation.
    /// Tracks GPU consumption via fence values to safely wrap around.
    /// </summary>
    public class UploadHeap : IDisposable
    {
        private readonly GraphicsDevice _device;
        private ID3D12Resource _heap;
        private IntPtr _cpuPtr;
        private ulong _gpuPtr;
        private readonly long _capacity;
        
        private long _head = 0; // Where we allocate next
        private long _tail = 0; // Oldest live allocation (advanced by fence completion)
        
        private readonly ID3D12Fence _fence; // Copy queue fence to poll completion
        
        // Tracks (fenceValue, headAtSubmit) — when fenceValue completes, 
        // everything up to headAtSubmit is reclaimable.
        private readonly Queue<(long fenceValue, long headAtSubmit)> _pendingFences = new();
        
        private readonly object _lock = new object();

        public ID3D12Resource Resource => _heap;
        public IntPtr BaseCpuPointer => _cpuPtr;
        public ulong BaseGpuPointer => _gpuPtr;

        public UploadHeap(GraphicsDevice device, long sizeInBytes, ID3D12Fence fence)
        {
            _device = device;
            _capacity = sizeInBytes;
            _fence = fence;

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
        /// Called by StreamingManager after submitting a batch to the copy queue.
        /// Records the fence value and current head position for space reclamation.
        /// </summary>
        public void OnBatchSubmitted(long fenceValue)
        {
            lock (_lock)
            {
                _pendingFences.Enqueue((fenceValue, _head));
            }
        }

        /// <summary>
        /// Drain completed fences to advance _tail, freeing ring buffer space.
        /// </summary>
        private void ReclaimCompleted()
        {
            long completedValue = (long)_fence.CompletedValue;
            while (_pendingFences.Count > 0)
            {
                var (fv, head) = _pendingFences.Peek();
                if (fv <= completedValue)
                {
                    _pendingFences.Dequeue();
                    _tail = head; // Everything up to this head is now free
                }
                else
                {
                    break; // Fences are ordered — stop at first incomplete
                }
            }
        }

        /// <summary>
        /// Allocates space in the ring buffer.
        /// Reclaims GPU-completed space first. If still insufficient, waits for the GPU.
        /// </summary>
        public long Allocate(long size, long alignment)
        {
            lock (_lock)
            {
                // Try to reclaim space that the GPU has finished reading
                ReclaimCompleted();
                
                // Align head
                long alignedHead = (_head + (alignment - 1)) & ~(alignment - 1);
                
                // Check if we need to wrap
                if (alignedHead + size > _capacity)
                {
                    // Wrap to beginning of buffer
                    alignedHead = 0;
                }
                
                // Check if we'd overwrite in-flight data
                // In a ring buffer: if head < tail, we must not write past tail
                // If head >= tail, we're fine unless we wrapped
                if (WouldOverlap(alignedHead, size))
                {
                    // Wait for the oldest pending fence to complete
                    WaitForOldestFence();
                    ReclaimCompleted();
                    
                    // Re-check after reclamation
                    if (WouldOverlap(alignedHead, size))
                    {
                        // Extreme case: asset may be larger than available contiguous space.
                        // Try wrapping to 0 if we haven't already.
                        if (alignedHead != 0)
                        {
                            alignedHead = 0;
                            if (WouldOverlap(alignedHead, size))
                            {
                                // Full flush — wait for all pending fences
                                while (_pendingFences.Count > 0)
                                {
                                    WaitForOldestFence();
                                    ReclaimCompleted();
                                }
                            }
                        }
                    }
                }
                
                if (alignedHead + size > _capacity)
                {
                    throw new InvalidOperationException($"Asset size {size} exceeds UploadHeap capacity {_capacity}");
                }
                
                _head = alignedHead + size;
                return alignedHead;
            }
        }
        
        /// <summary>
        /// Checks if writing 'size' bytes at 'offset' would overlap with in-flight data.
        /// </summary>
        private bool WouldOverlap(long offset, long size)
        {
            // If no pending fences, the entire buffer is available
            if (_pendingFences.Count == 0)
                return false;
                
            long writeEnd = offset + size;
            
            // Ring buffer overlap check:
            // tail marks the end of the oldest in-flight data
            // If head >= tail: free space is [head, capacity) + [0, tail)
            // If head < tail (wrapped): free space is [head, tail)
            if (offset >= _tail)
            {
                // Writing forward from offset — only overflows if we'd go past capacity
                return writeEnd > _capacity;
            }
            else
            {
                // We've wrapped — must not write past tail
                return writeEnd > _tail;
            }
        }

        /// <summary>
        /// CPU-waits for the oldest outstanding fence to complete.
        /// </summary>
        private void WaitForOldestFence()
        {
            if (_pendingFences.Count == 0) return;
            
            var (fv, _) = _pendingFences.Peek();
            if ((long)_fence.CompletedValue >= fv) return;
            
            // Spin-wait with short sleeps (we're on the upload thread, not main thread)
            while ((long)_fence.CompletedValue < fv)
            {
                Thread.Sleep(1);
            }
        }

        public void Dispose()
        {
            _heap?.Dispose();
        }
    }
}
