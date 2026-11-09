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
        
        private readonly Lock _lock = new();

        public ID3D12Resource Resource => _heap;
        public IntPtr BaseCpuPointer => _cpuPtr;
        public ulong BaseGpuPointer => _gpuPtr;

        public UploadHeap(GraphicsDevice device, long sizeInBytes, ID3D12Fence fence)
        {
            _device = device;
            _capacity = sizeInBytes;
            _fence = fence;

            _heap = _device.CreateCommittedResource(
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
                _batchStart = -1;
            }
        }

        /// <summary>The batch being recorded is not submitted after all (nothing in it to copy): its bytes are free again.</summary>
        public void AbandonBatch()
        {
            lock (_lock)
            {
                if (_batchStart >= 0) _head = _batchStart;
                _batchStart = -1;
            }
        }

        // Where the batch being recorded began (its first allocation), -1 before it has one. Its bytes are in use too:
        // until 2026-11-09 only submitted batches counted, so one batch of more than the ring (43 terrain brushes at once,
        // 344 MiB through 256) wrapped onto its own first textures before the GPU had copied them, and those textures came
        // out as rows of other textures' bytes (spiky terrain where the brushes were read).
        private long _batchStart = -1;

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
        /// Allocates space in the ring buffer for the batch being recorded. Reclaims what the GPU has finished copying,
        /// waits for older batches while they are in the way, and throws <see cref="UploadBatchFullException"/> (before
        /// anything is written) when only this batch's own bytes are: the batch must go to the GPU first, and the upload
        /// that asked goes first in the next one (StreamingManager).
        /// </summary>
        public long Allocate(long size, long alignment)
        {
            if (size > _capacity)
                throw new InvalidOperationException($"Asset size {size} exceeds UploadHeap capacity {_capacity}");
            lock (_lock)
            {
                while (true)
                {
                    ReclaimCompleted();
                    if (TryPlace(size, alignment, out long offset))
                    {
                        if (_batchStart < 0) _batchStart = offset;
                        _head = offset + size;
                        return offset;
                    }
                    if (_pendingFences.Count > 0)
                    {
                        WaitForOldestFence();
                        if (_device.IsDeviceLost)
                            throw new InvalidOperationException("UploadHeap: the device was lost while waiting for the copy queue");
                        continue;
                    }
                    throw new UploadBatchFullException();
                }
            }
        }

        /// <summary>
        /// A place for <paramref name="size"/> bytes that touches nothing in use: the submitted batches not yet copied and
        /// the batch being recorded, the ring from the oldest of them (start) to the head. After the head, or wrapped to
        /// 0 when the end is too close. Under the lock.
        /// </summary>
        private bool TryPlace(long size, long alignment, out long offset)
        {
            long head = (_head + (alignment - 1)) & ~(alignment - 1);
            bool inUse = _pendingFences.Count > 0 || _batchStart >= 0;
            if (!inUse)
            {
                offset = head + size <= _capacity ? head : 0;
                return true;
            }

            long start = _pendingFences.Count > 0 ? _tail : _batchStart;
            if (_head > start)
            {
                // In use: [start, head). Free: [head, end) and [0, start)
                if (head + size <= _capacity) { offset = head; return true; }
                if (size <= start) { offset = 0; return true; }
            }
            else if (_head < start)
            {
                // Wrapped. In use: [start, end) and [0, head). Free: [head, start)
                if (head + size <= start) { offset = head; return true; }
            }
            // _head == start with something in use: the ring is full
            offset = 0;
            return false;
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
                if (_fence.CompletedValue == ulong.MaxValue)
                {
                    _device.IsDeviceLost = true;
                    return;
                }
                Thread.Sleep(1);
            }
        }

        public void Dispose()
        {
            _heap?.Dispose();
        }
    }

    /// <summary>
    /// The batch being recorded fills the upload ring: submit it, then record the upload that asked again (nothing of it
    /// was written). Thrown by <see cref="UploadHeap.Allocate"/> only when no older batch is left to wait for.
    /// </summary>
    public sealed class UploadBatchFullException : Exception
    {
        public UploadBatchFullException() : base("The upload batch fills the upload heap: submit it first") { }
    }
}
