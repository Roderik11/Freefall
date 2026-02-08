using System;
using System.Collections.Generic;
using System.Numerics;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// Manages a global bone matrix buffer for all skinned meshes.
    /// Batches bone matrix uploads to reduce Map/Unmap overhead.
    /// </summary>
    public static class BoneMatrixManager
    {
        private const int FrameCount = 3;
        private const int MaxBoneMatrices = 10000; // Support up to 10k bone matrices total
        
        private static ID3D12Resource[] _boneBuffers;
        private static uint[] _boneBufferIndices;
        private static int _currentOffset = 0;
        private static bool _initialized = false;
        
        // Pending uploads for this frame
        private static List<(int offset, Matrix4x4[] matrices)> _pendingUploads = new List<(int, Matrix4x4[])>();
        
        public static void Initialize(GraphicsDevice device)
        {
            if (_initialized) return;
            
            int bufferSize = MaxBoneMatrices * 64; // sizeof(Matrix4x4)
            
            _boneBuffers = new ID3D12Resource[FrameCount];
            _boneBufferIndices = new uint[FrameCount];
            
            for (int i = 0; i < FrameCount; i++)
            {
                _boneBuffers[i] = device.CreateUploadBuffer(bufferSize);
                _boneBufferIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferSRV(_boneBuffers[i], MaxBoneMatrices, 64, _boneBufferIndices[i]);
            }
            
            _initialized = true;
            Debug.Log($"[BoneMatrixManager] Initialized with {MaxBoneMatrices} bone matrix slots");
        }
        
        /// <summary>
        /// Allocates a range of bone matrix slots for a skinned mesh.
        /// Returns the offset into the global bone buffer.
        /// </summary>
        public static int AllocateBoneSlots(int boneCount)
        {
            if (!_initialized)
                throw new InvalidOperationException("BoneMatrixManager not initialized");
                
            int offset = _currentOffset;
            _currentOffset += boneCount;
            
            if (_currentOffset > MaxBoneMatrices)
                throw new InvalidOperationException($"Exceeded max bone matrices ({MaxBoneMatrices})");
                
            return offset;
        }
        
        /// <summary>
        /// Queues bone matrices for upload. Actual upload happens in FlushUploads().
        /// </summary>
        public static void QueueBoneMatrices(int offset, Matrix4x4[] matrices)
        {
            _pendingUploads.Add((offset, matrices));
        }
        
        /// <summary>
        /// Uploads all queued bone matrices in a single Map/Unmap operation.
        /// Call this once per frame after all SkinnedMeshRenderer.Update() calls.
        /// </summary>
        public static void FlushUploads()
        {
            if (_pendingUploads.Count == 0) return;
            
            int frameIndex = Engine.FrameIndex % FrameCount;
            var buffer = _boneBuffers[frameIndex];
            
            unsafe
            {
                void* pData;
                buffer.Map(0, null, &pData);
                if (pData != null)
                {
                    var span = new Span<Matrix4x4>(pData, MaxBoneMatrices);
                    
                    // Copy all pending uploads
                    foreach (var (offset, matrices) in _pendingUploads)
                    {
                        for (int i = 0; i < matrices.Length; i++)
                        {
                            span[offset + i] = matrices[i];
                        }
                    }
                    
                    buffer.Unmap(0);
                }
            }
            
            _pendingUploads.Clear();
        }
        
        /// <summary>
        /// Gets the bindless buffer index for the current frame.
        /// </summary>
        public static uint GetBufferIndex()
        {
            int frameIndex = Engine.FrameIndex % FrameCount;
            return _boneBufferIndices[frameIndex];
        }
        
        /// <summary>
        /// Resets allocation offset. Call at start of frame or when reloading scene.
        /// </summary>
        public static void Reset()
        {
            _currentOffset = 0;
            _pendingUploads.Clear();
        }
    }
}
