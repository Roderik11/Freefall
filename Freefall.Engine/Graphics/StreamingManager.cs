using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Vortice.Direct3D12;
using Freefall.Base;
using Freefall.Assets;

namespace Freefall.Graphics
{
    public class StreamingManager : IDisposable
    {
        private readonly GraphicsDevice _device;
        private readonly UploadHeap _uploadHeap;
        private readonly ConcurrentQueue<Action<ID3D12GraphicsCommandList, UploadHeap>> _pendingUploads = new();
        private readonly Thread _uploadThread;
        private readonly CancellationTokenSource _cts = new();
        private readonly AutoResetEvent _workAvailable = new(false);
        private volatile bool _processing;

        private ID3D12Fence _fence;
        private long _nextFenceValue = 1;
        private long _lastCompletedFenceValue = 0;

        private static long _bytesUploaded;
        public static long BytesUploaded => Interlocked.Read(ref _bytesUploaded);
        public static int PendingItems => _instance?._pendingUploads.Count ?? 0;

        private static StreamingManager _instance;
        public static StreamingManager Instance => _instance;

        private List<Asset> _currentBatchAssets = new();

        public StreamingManager(GraphicsDevice device, long heapSize = 256 * 1024 * 1024)
        {
            _instance = this;
            _device = device;
            _fence = device.NativeDevice.CreateFence(0, FenceFlags.None);
            _uploadHeap = new UploadHeap(device, heapSize, _fence);

            _uploadThread = new Thread(UploadLoop)
            {
                Name = "Streaming Upload Thread",
                Priority = ThreadPriority.AboveNormal, 
                IsBackground = true
            };
            _uploadThread.Start();
        }

        public void EnqueueTextureUpload(Texture texture, CpuTextureData cpuData)
        {
            _pendingUploads.Enqueue((cmdList, heap) =>
            {
                RecordTextureUpload(cmdList, heap, texture, cpuData);
                _currentBatchAssets.Add(texture);
            });
            _workAvailable.Set();
        }

        public void EnqueueBufferUpload<T>(ID3D12Resource targetBuffer, T[] data, Asset owner = null) where T : unmanaged
        {
             // Capture data in closure (careful with large arrays, but T[] is ref type)
            _pendingUploads.Enqueue((cmdList, heap) =>
            {
                RecordBufferUpload(cmdList, heap, targetBuffer, data);
                 if(owner != null) _currentBatchAssets.Add(owner);
            });
            _workAvailable.Set();
        }

        private void RecordTextureUpload(ID3D12GraphicsCommandList cmdList, UploadHeap heap, Texture texture, CpuTextureData data)
        {
            var desc = texture.Native.Description;
            int numSubresources = (int)(desc.MipLevels * desc.DepthOrArraySize);
            
            var layouts = new PlacedSubresourceFootPrint[numSubresources];
            var numRows = new uint[numSubresources];
            var rowSizes = new ulong[numSubresources];
            ulong totalSize = 0;
            
            unsafe 
            {
                fixed (PlacedSubresourceFootPrint* pLayouts = layouts)
                fixed (uint* pNumRows = numRows)
                fixed (ulong* pRowSizes = rowSizes)
                {
                    _device.NativeDevice.GetCopyableFootprints(desc, 0, (uint)numSubresources, 0, pLayouts, pNumRows, pRowSizes, out totalSize);
                }
            }
            
            long offset = heap.Allocate((long)totalSize, 512);
            
            unsafe
            {
                byte* pBase = (byte*)heap.BaseCpuPointer + offset;
                fixed (byte* pPixelData = data.PixelData)
                {
                    byte* pSrcBase = pPixelData;
                    for (int i = 0; i < numSubresources; i++)
                    {
                        var layout = layouts[i];
                        byte* pDestSub = pBase + layout.Offset;
                        
                        long srcOffset = data.Mips[i].Offset; 
                        int rows = (int)numRows[i];
                        int rowPitch = (int)layout.Footprint.RowPitch;
                        int srcPitch = data.Mips[i].RowPitch;
                        int copySize = Math.Min(rowPitch, srcPitch);
                        
                        byte* pSrcSub = pSrcBase + srcOffset;
                        for (int r = 0; r < rows; r++)
                        {
                            Buffer.MemoryCopy(pSrcSub, pDestSub, rowPitch, copySize);
                            pSrcSub += srcPitch;
                            pDestSub += rowPitch;
                        }
                    }
                }
            }
            
            for (int i = 0; i < numSubresources; i++)
            {
               var dstLocation = new TextureCopyLocation(texture.Native, (uint)i);
               var layout = layouts[i];
               layout.Offset += (ulong)offset;
               var srcLocation = new TextureCopyLocation(heap.Resource, layout);
               cmdList.CopyTextureRegion(dstLocation, 0, 0, 0, srcLocation, null);
            }
            
            Interlocked.Add(ref _bytesUploaded, (long)totalSize);
        }

        private void RecordBufferUpload<T>(ID3D12GraphicsCommandList cmdList, UploadHeap heap, ID3D12Resource targetBuffer, T[] data) where T : unmanaged
        {
            int size = data.Length * System.Runtime.InteropServices.Marshal.SizeOf<T>();

            // Validate: data must fit in the target buffer
            var targetSize = targetBuffer.Description.Width;
            if ((ulong)size > targetSize)
            {
                Debug.LogError("StreamingManager", $"Buffer upload size mismatch: data={size} bytes, target={targetSize} bytes — skipping");
                return;
            }

            long offset = heap.Allocate(size, 256); // 256 byte alignment for buffer copies preferred

            unsafe
            {
                byte* pDest = (byte*)heap.BaseCpuPointer + offset;
                fixed (void* pSrc = data)
                {
                    Buffer.MemoryCopy(pSrc, pDest, size, size);
                }
            }

            // CopyBufferRegion
            cmdList.CopyBufferRegion(targetBuffer, 0, heap.Resource, (ulong)offset, (ulong)size);
            Interlocked.Add(ref _bytesUploaded, size);
        }

        private void UploadLoop()
        {
            // Double-buffered allocators: alternate each batch so the GPU has time
            // to finish with one allocator while we record into the other.
            // D3D12 requires GPU to be done with an allocator before Reset().
            var allocators = new ID3D12CommandAllocator[2];
            var fences = new long[2]; // fence value of last submission per allocator
            for (int i = 0; i < 2; i++)
                allocators[i] = _device.NativeDevice.CreateCommandAllocator(CommandListType.Copy);

            var cmdList = _device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Copy, allocators[0]);
            cmdList.Close();

            int currentSlot = 0;

            while (!_cts.IsCancellationRequested)
            {
                _workAvailable.WaitOne(10);
                if (_cts.IsCancellationRequested) break;

                if (_pendingUploads.IsEmpty) continue;
                if (_device.IsDeviceLost) continue;

                _processing = true;
                try
                {
                    // Wait for THIS slot's previous batch (if any) to complete
                    long prevFence = fences[currentSlot];
                    if (prevFence > 0)
                    {
                        while ((long)_fence.CompletedValue < prevFence)
                        {
                            if (_fence.CompletedValue == ulong.MaxValue) { _device.IsDeviceLost = true; break; }
                            Thread.Sleep(1);
                        }
                        if (_device.IsDeviceLost) continue;
                    }

                    allocators[currentSlot].Reset();
                    cmdList.Reset(allocators[currentSlot], null);
                    _currentBatchAssets.Clear();
                    
                    int processed = 0;
                    while (_pendingUploads.TryDequeue(out var uploadAction))
                    {
                         try 
                         {
                             uploadAction(cmdList, _uploadHeap);
                             processed++;
                             if (processed > 64) break;
                         }
                         catch (Exception ex)
                         {
                             Debug.LogError("StreamingManager", $"Upload failed: {ex.Message}");
                         }
                    }

                    if (processed > 0)
                    {
                        cmdList.Close();
                        
                        // Submit through the device so _copyFence is signaled — 
                        // this is what WaitForCopyQueue checks before rendering.
                        // Without this, the render queue starts drawing before vertex data is copied.
                        _device.CopyQueueSubmit(cmdList);
                        
                        // Also signal our private fence for UploadHeap ring buffer reclamation
                        long fenceValue = Interlocked.Increment(ref _nextFenceValue);
                        _device.CopyQueue.Signal(_fence, (ulong)fenceValue);
                        fences[currentSlot] = fenceValue;
                        
                        // Record this batch's fence so the UploadHeap can reclaim space
                        _uploadHeap.OnBatchSubmitted(fenceValue);
                        
                        foreach (var asset in _currentBatchAssets)
                        {
                            asset.SetReadyFence(fenceValue);
                        }
                    }

                    // Alternate to the other allocator
                    currentSlot = 1 - currentSlot;
                }
                finally
                {
                    _processing = false;
                }
            }
            allocators[0].Dispose();
            allocators[1].Dispose();
            cmdList.Dispose();
        }

        public long GetCompletedFence()
        {
            long val = (long)_fence.CompletedValue;
            Interlocked.Exchange(ref _lastCompletedFenceValue, val);
            return val;
        }

        /// <summary>
        /// Block until all pending uploads are processed and GPU copy queue has completed.
        /// Call during initialization to ensure textures are resident before first render.
        /// </summary>
        public void Flush()
        {
            // Wait until the upload thread has drained the queue AND finished its current batch
            int drainTimeout = 30000;
            while (!_pendingUploads.IsEmpty || _processing)
            {
                if (_device.IsDeviceLost) break;
                _workAvailable.Set(); // Wake the upload thread in case it's sleeping
                Thread.Sleep(1);
                if (--drainTimeout <= 0)
                {
                    Debug.LogWarning("StreamingManager", $"Flush drain timed out (pending={!_pendingUploads.IsEmpty}, processing={_processing})");
                    break;
                }
            }
            // Wait for the last GPU fence to complete
            long lastSubmitted = Interlocked.Read(ref _nextFenceValue) - 1;
            if (lastSubmitted > 0)
            {
                int timeout = 5000; // 5s safety timeout
                while ((long)_fence.CompletedValue < lastSubmitted)
                {
                    // DEVICE_REMOVED makes CompletedValue return UINT64_MAX
                    if (_fence.CompletedValue == ulong.MaxValue)
                    {
                        Debug.LogWarning("StreamingManager", "GPU device lost during Flush — aborting wait");
                        return;
                    }
                    if (--timeout <= 0)
                    {
                        Debug.LogWarning("StreamingManager", $"Flush timed out (fence {_fence.CompletedValue} < {lastSubmitted})");
                        return;
                    }
                    Thread.Sleep(1);
                }
            }
        }

        public void Dispose()
        {
            _cts.Cancel();
            _uploadThread.Join();
            _fence.Dispose();
            _uploadHeap.Dispose();
        }
    }
}
