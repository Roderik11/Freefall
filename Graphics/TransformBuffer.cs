using System;
using System.Collections.Generic;
using System.Numerics;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// Global persistent GPU buffer for all entity transforms.
    /// Entities allocate slots and write directly, avoiding per-frame dictionary lookups.
    /// </summary>
    public class TransformBuffer : IDisposable
    {
        public const int MaxSlots = 100_000;
        private const int FrameCount = 3;
        private const int TransformSize = 64; // Matrix4x4 = 16 floats * 4 bytes

        // GPU buffers (upload heap for CPU writes)
        private ID3D12Resource[] _buffers = new ID3D12Resource[FrameCount];
        private uint[] _srvIndices = new uint[FrameCount];
        
        // MaterialID buffers indexed by TransformSlot (for order-independent per-instance materials)
        private ID3D12Resource[] _materialBuffers = new ID3D12Resource[FrameCount];
        private uint[] _materialSrvIndices = new uint[FrameCount];
        
        // Slot allocation
        private readonly Stack<int> _freeSlots = new();
        private int _nextSlot = 0;
        private int _activeSlots = 0;
        
        // Thread-safety lock for parallel SetTransform calls
        private readonly object _transformLock = new object();
        
        // CPU-side staging
        private Matrix4x4[] _transforms = new Matrix4x4[MaxSlots];
        private uint[] _materialIds = new uint[MaxSlots];
        
        // Per-frame dirty tracking - each frame buffer has its own dirty set
        // When a slot is modified, it's marked dirty for ALL frames
        // When uploading to frame N, only clear dirty flags for that frame
        private bool[][] _dirtyPerFrame = new bool[FrameCount][];
        private bool[][] _materialDirtyPerFrame = new bool[FrameCount][];
        private bool[] _anyDirtyPerFrame = new bool[FrameCount];
        private bool[] _anyMaterialDirtyPerFrame = new bool[FrameCount];

        // Singleton instance
        public static TransformBuffer Instance { get; private set; } = null!;

        public int ActiveSlots => _activeSlots;
        
        /// <summary>
        /// Get the SRV bindless index for the current frame's transform buffer.
        /// Used by shaders to read transforms directly.
        /// </summary>
        public uint SrvIndex => _srvIndices[Engine.FrameIndex % FrameCount];
        
        /// <summary>
        /// Get the SRV bindless index for the current frame's materialID buffer.
        /// </summary>
        public uint MaterialSrvIndex => _materialSrvIndices[Engine.FrameIndex % FrameCount];

        public static void Initialize(GraphicsDevice device)
        {
            Instance = new TransformBuffer(device);
            Debug.Log($"[TransformBuffer] Initialized with {MaxSlots} slots");
        }

        private TransformBuffer(GraphicsDevice device)
        {
            int bufferSize = MaxSlots * TransformSize;
            int materialBufferSize = MaxSlots * sizeof(uint);
            
            // Initialize per-frame dirty tracking
            for (int f = 0; f < FrameCount; f++)
            {
                _dirtyPerFrame[f] = new bool[MaxSlots];
                _materialDirtyPerFrame[f] = new bool[MaxSlots];
            }
            
            for (int i = 0; i < FrameCount; i++)
            {
                // Transform buffers
                _buffers[i] = device.CreateUploadBuffer(bufferSize);
                _srvIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferSRV(_buffers[i], MaxSlots, TransformSize, _srvIndices[i]);
                
                // MaterialID buffers (indexed by TransformSlot for order-independent materials)
                _materialBuffers[i] = device.CreateUploadBuffer(materialBufferSize);
                _materialSrvIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferSRV(_materialBuffers[i], MaxSlots, sizeof(uint), _materialSrvIndices[i]);
            }
        }

        /// <summary>
        /// Allocate a slot for an entity. Returns the slot index.
        /// Thread-safe: can be called from parallel Draw().
        /// </summary>
        public int AllocateSlot()
        {
            lock (_transformLock)
            {
                int slot;
                if (_freeSlots.Count > 0)
                {
                    slot = _freeSlots.Pop();
                }
                else
                {
                    if (_nextSlot >= MaxSlots)
                        throw new InvalidOperationException($"TransformBuffer: exceeded max slots ({MaxSlots})");
                    slot = _nextSlot++;
                }
                
                _activeSlots++;
                _transforms[slot] = Matrix4x4.Identity;
                // Mark dirty for ALL frames - each frame buffer needs this update
                for (int f = 0; f < FrameCount; f++)
                {
                    _dirtyPerFrame[f][slot] = true;
                    _anyDirtyPerFrame[f] = true;
                }
                return slot;
            }
        }

        /// <summary>
        /// Release a slot when entity is destroyed.
        /// </summary>
        public void ReleaseSlot(int slot)
        {
            if (slot < 0 || slot >= _nextSlot) return;
            _freeSlots.Push(slot);
            _activeSlots--;
            _transforms[slot] = Matrix4x4.Identity;
        }

        /// <summary>
        /// Set transform for a slot. Marks dirty for GPU upload.
        /// </summary>
        public void SetTransform(int slot, Matrix4x4 world)
        {
            if (slot < 0 || slot >= _nextSlot) return;
            
            // Thread-safe: lock to prevent race conditions during parallel Draw()
            lock (_transformLock)
            {
                _transforms[slot] = Matrix4x4.Transpose(world); // Pre-transpose for GPU
                // Mark dirty for ALL frames - each frame buffer needs this update
                for (int f = 0; f < FrameCount; f++)
                {
                    _dirtyPerFrame[f][slot] = true;
                    _anyDirtyPerFrame[f] = true;
                }
            }
        }

        /// <summary>
        /// Set materialID for a slot. Enables per-instance materials indexed by TransformSlot.
        /// </summary>
        public void SetMaterialId(int slot, uint materialId)
        {
            if (slot < 0 || slot >= _nextSlot) return;
            
            // Thread-safe: lock to prevent race conditions during parallel Draw()
            lock (_transformLock)
            {
                _materialIds[slot] = materialId;
                // Mark dirty for ALL frames - each frame buffer needs this update
                for (int f = 0; f < FrameCount; f++)
                {
                    _materialDirtyPerFrame[f][slot] = true;
                    _anyMaterialDirtyPerFrame[f] = true;
                }
            }
        }

        /// <summary>
        /// Get the SRV index for current frame's transform buffer.
        /// </summary>
        public uint GetSRVIndex()
        {
            int frameIndex = Engine.FrameIndex % FrameCount;
            return _srvIndices[frameIndex];
        }
        
        /// <summary>
        /// Get transform directly from CPU-side staging array (already transposed).
        /// Used by GPUInstanceBatch for fast copy without dictionary lookups.
        /// </summary>
        public Matrix4x4 GetTransformDirect(int slot)
        {
            if (slot < 0 || slot >= _nextSlot) return Matrix4x4.Identity;
            return _transforms[slot];
        }

        /// <summary>
        /// Upload dirty transforms and materialIDs to current frame's GPU buffer only.
        /// Uses per-frame dirty tracking to avoid race conditions with GPU reading
        /// from other frame buffers.
        /// </summary>
        public void Upload()
        {
            int frameIndex = Engine.FrameIndex % FrameCount;
            
            unsafe
            {
                // Upload transforms to CURRENT FRAME BUFFER ONLY
                // This avoids race conditions with GPU reading from other frame buffers
                if (_anyDirtyPerFrame[frameIndex])
                {
                    byte* pData;
                    _buffers[frameIndex].Map(0, null, (void**)&pData);
                    if (pData == null)
                    {
                        try
                        {
                            var reason = Engine.Device.NativeDevice.DeviceRemovedReason;
                            Debug.LogAlways($"[TransformBuffer] Map returned null! DeviceRemovedReason: {reason}");
                        }
                        catch { Debug.LogAlways("[TransformBuffer] Map returned null! Could not query DeviceRemovedReason."); }
                        return;
                    }
                    Matrix4x4* pTransforms = (Matrix4x4*)pData;
                    
                    var dirtyFlags = _dirtyPerFrame[frameIndex];
                    for (int i = 0; i < _nextSlot; i++)
                    {
                        if (dirtyFlags[i])
                        {
                            pTransforms[i] = _transforms[i];
                            dirtyFlags[i] = false;
                        }
                    }
                    
                    _buffers[frameIndex].Unmap(0);
                    _anyDirtyPerFrame[frameIndex] = false;
                }
                
                // Upload materialIDs to CURRENT FRAME BUFFER ONLY
                if (_anyMaterialDirtyPerFrame[frameIndex])
                {
                    byte* pData;
                    _materialBuffers[frameIndex].Map(0, null, (void**)&pData);
                    uint* pMaterials = (uint*)pData;
                    
                    var dirtyFlags = _materialDirtyPerFrame[frameIndex];
                    for (int i = 0; i < _nextSlot; i++)
                    {
                        if (dirtyFlags[i])
                        {
                            pMaterials[i] = _materialIds[i];
                            dirtyFlags[i] = false;
                        }
                    }
                    
                    _materialBuffers[frameIndex].Unmap(0);
                    _anyMaterialDirtyPerFrame[frameIndex] = false;
                }
            }
        }

        /// <summary>
        /// Bulk upload all transforms (for initial or full refresh).
        /// </summary>
        public void UploadAll()
        {
            int frameIndex = Engine.FrameIndex % FrameCount;
            
            unsafe
            {
                byte* pData;
                _buffers[frameIndex].Map(0, null, (void**)&pData);
                
                fixed (Matrix4x4* pSrc = _transforms)
                {
                    Buffer.MemoryCopy(pSrc, pData, MaxSlots * TransformSize, _nextSlot * TransformSize);
                }
                
                _buffers[frameIndex].Unmap(0);
            }
            
            // Clear dirty flags for current frame (all slots were uploaded)
            Array.Clear(_dirtyPerFrame[frameIndex], 0, _nextSlot);
            _anyDirtyPerFrame[frameIndex] = false;
        }

        public void Dispose()
        {
            for (int i = 0; i < FrameCount; i++)
            {
                _buffers[i]?.Dispose();
                _materialBuffers[i]?.Dispose();
                if (_srvIndices[i] != 0)
                    Engine.Device.ReleaseBindlessIndex(_srvIndices[i]);
                if (_materialSrvIndices[i] != 0)
                    Engine.Device.ReleaseBindlessIndex(_materialSrvIndices[i]);
            }
        }
    }
}
