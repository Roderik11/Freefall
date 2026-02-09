using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Components;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// GPU-driven instance batch. All command generation happens on the GPU.
    /// Flow: unsorted draws → GPU frustum culling → cache-friendly grouping → indirect commands
    /// </summary>
    public sealed class InstanceBatch : IDisposable
    {
        public const int FrameCount = 3;

        // Deferred disposal queue — old GPU buffers kept alive until GPU is done with them
        private static readonly List<(ID3D12Resource Resource, uint BindlessIndex, int FrameToDispose)> _deferredDisposals = new();

        /// <summary>
        /// Call once per frame (after Present) to release GPU resources from previous resizes.
        /// </summary>
        public static void FlushDeferredDisposals()
        {
            int currentFrame = Engine.TickCount;
            for (int i = _deferredDisposals.Count - 1; i >= 0; i--)
            {
                if (currentFrame >= _deferredDisposals[i].FrameToDispose)
                {
                    _deferredDisposals[i].Resource?.Dispose();
                    if (_deferredDisposals[i].BindlessIndex != 0)
                        Engine.Device?.ReleaseBindlessIndex(_deferredDisposals[i].BindlessIndex);
                    _deferredDisposals.RemoveAt(i);
                }
            }
        }

        private static void DeferDispose(ID3D12Resource? resource, uint bindlessIndex)
        {
            if (resource == null && bindlessIndex == 0) return;
            _deferredDisposals.Add((resource, bindlessIndex, Engine.TickCount + FrameCount + 1));
        }
        
        public BatchKey Key;
        public Material Material;

        public struct RawDraw
        {
            public Mesh Mesh;
            public int PartIndex;
            public MaterialBlock Block;
            public int TransformSlot;
            public uint MaterialId;
            public int MeshPartId;
        }

        internal int _activeFrame = -1;
        private RawDraw[] _draws = Array.Empty<RawDraw>();
        private HashSet<int> _uniqueMeshPartIds = new HashSet<int>();

        public int DrawCount => _drawCount;
        public int SubBatchCount { get; private set; }

        private int capacity = 0;

        
        /// <summary>
        /// Per-instance GPU buffer for a single parameter (SoA pattern).
        /// Self-describing: carries its own PushConstantSlot and stride.
        /// </summary>
        private class PerInstanceBuffer
        {
            public int ParamHash;
            public int PushConstantSlot; // which push constant slot to bind SRV to
            public ID3D12Resource[] Buffers = new ID3D12Resource[FrameCount];
            public uint[] SRVIndices = new uint[FrameCount];
            public int Capacity; // in slots
            public int ElementsPerInstance; // e.g. numBones
            public int ElementStride; // bytes per element
            public bool Dirty;
            
            // CPU staging buffer — filled during MergeFromBucket, uploaded in one MemoryCopy
            public byte[] StagingData = Array.Empty<byte>();
            public int StagingCapacity; // in slots (matches Capacity growth)
        }
        
        // Active per-instance buffers for this batch
        private Dictionary<int, PerInstanceBuffer> _perInstanceBuffers = new();
        
        // Hardcoded compute push constant slots for per-instance buffers.
        // These are core pipeline constants — compute layout is fixed in cull_instances.hlsl.
        // Graphics slots (from shader resource bindings) are separate and used in Draw/DrawShadow.
        private static readonly Dictionary<int, int> ComputeSlots = new()
        {
            { "Bones".GetHashCode(), 30 },              // Indices[7].z = BoneBufferIdx
            { DrawBucket.BoundingSpheresHash, 2 },       // Indices[0].z = BoundingSpheresIdx
            { DrawBucket.DescriptorsHash, 4 },            // Indices[1].x = DescriptorBufferIdx
            { DrawBucket.SubbatchIdsHash, 23 },           // Indices[5].w = SubbatchIdsIdx
        };
        
        // Cached array to avoid per-frame allocation
        private ID3D12DescriptorHeap[]? _cachedSrvHeapArray;

        // GPU output: visible instance indices (after culling)
        private ID3D12Resource[] visibleIndicesBuffers = new ID3D12Resource[FrameCount];
        private uint[] visibleIndicesUAVIndices = new uint[FrameCount];
        private uint[] visibleIndicesSRVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] visibleIndicesCPUHandles = new CpuDescriptorHandle[FrameCount];
        

        
        // GPU output: indirect commands (compute shader writes)
        private ID3D12Resource[] gpuCommandBuffers = new ID3D12Resource[FrameCount];
        private uint[] gpuCommandUAVIndices = new uint[FrameCount];

        // Per-subbatch counters (atomic increments during culling)
        private ID3D12Resource[] counterBuffers = new ID3D12Resource[FrameCount];
        private uint[] counterBufferUAVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] counterBufferCPUHandles = new CpuDescriptorHandle[FrameCount];
        
        // Visibility flags buffer (1=visible, 0=not)
        private ID3D12Resource[] visibilityFlagsBuffers = new ID3D12Resource[FrameCount];
        private uint[] visibilityFlagsUAVIndices = new uint[FrameCount];
        
        private const int MaxSubBatches = 4096;
        private bool _cullerInitialized = false;
        private ID3D12DescriptorHeap? _cpuHeap;
        
        // Histogram buffer for GPU-driven per-MeshPartId instance counting
        private ID3D12Resource[] histogramBuffers = new ID3D12Resource[FrameCount];
        private uint[] histogramUAVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] histogramCPUHandles = new CpuDescriptorHandle[FrameCount];
        
        // Shadow cascade buffers - separate culling pipeline per cascade (4 cascades)
        private const int ShadowCascadeCount = 4;
        private ID3D12Resource[,] shadowVisibilityBuffers = new ID3D12Resource[FrameCount, ShadowCascadeCount];
        private uint[,] shadowVisibilityUAVIndices = new uint[FrameCount, ShadowCascadeCount];
        private ID3D12Resource[,] shadowVisibleIndicesBuffers = new ID3D12Resource[FrameCount, ShadowCascadeCount];
        private uint[,] shadowVisibleIndicesUAVIndices = new uint[FrameCount, ShadowCascadeCount];
        private uint[,] shadowVisibleIndicesSRVIndices = new uint[FrameCount, ShadowCascadeCount];

        private ID3D12Resource[,] shadowCommandBuffers = new ID3D12Resource[FrameCount, ShadowCascadeCount];
        private uint[,] shadowCommandUAVIndices = new uint[FrameCount, ShadowCascadeCount];
        private ID3D12Resource[,] shadowCounterBuffers = new ID3D12Resource[FrameCount, ShadowCascadeCount];
        private uint[,] shadowCounterUAVIndices = new uint[FrameCount, ShadowCascadeCount];
        private ID3D12Resource[,] shadowHistogramBuffers = new ID3D12Resource[FrameCount, ShadowCascadeCount];
        private uint[,] shadowHistogramUAVIndices = new uint[FrameCount, ShadowCascadeCount];
        private bool _shadowCullerInitialized = false;
        private ID3D12DescriptorHeap? _shadowCpuHeap;
        private CpuDescriptorHandle[,] shadowCounterCPUHandles = new CpuDescriptorHandle[FrameCount, ShadowCascadeCount];
        private CpuDescriptorHandle[,] shadowVisibleIndicesCPUHandles = new CpuDescriptorHandle[FrameCount, ShadowCascadeCount];
        private CpuDescriptorHandle[,] shadowHistogramCPUHandles = new CpuDescriptorHandle[FrameCount, ShadowCascadeCount];




        private int _drawCount = 0;
        private int _maxTransformSlot = 0;
        
        public InstanceBatch(BatchKey key, Material material)
        {
            Key = key;
            Material = material;
            ResizeBuffers(128);
        }

        public void Clear()
        {
            _drawCount = 0;
            _uniqueMeshPartIds.Clear();
            SubBatchCount = 0;
            _maxTransformSlot = 0;
            foreach (var buf in _perInstanceBuffers.Values)
                buf.Dirty = false;
        }

        public void MergeFromBucket(DrawBucket bucket)
        {
            if (bucket.Count == 0) return;
            
            int startIdx = _drawCount;
            int count = bucket.Count;
            int newTotal = startIdx + count;
            
            // Ensure capacity for all new draws
            if (newTotal > capacity)
                ResizeBuffers(Math.Max(capacity * 2, newTotal));
            
            // Block-copy RawDraws using Span (slice to actual count)
            var srcDraws = CollectionsMarshal.AsSpan(bucket.Draws).Slice(0, count);
            srcDraws.CopyTo(_draws.AsSpan(startIdx));
            
            foreach (var id in bucket.UniqueMeshPartIds)
                _uniqueMeshPartIds.Add(id);
            SubBatchCount = _uniqueMeshPartIds.Count;
            
            // Block-copy all pre-staged per-instance data from bucket (staged at Enqueue time)
            // This includes core channels (Descriptors, BoundingSpheres, SubbatchIds) and
            // optional channels (TerrainPatchData, PointLightData, Bones, etc.)
            foreach (var (hash, staging) in bucket.PerInstanceData)
            {
                if (staging.Count == 0) continue;
                
                if (!_perInstanceBuffers.TryGetValue(hash, out var pib))
                {
                    pib = new PerInstanceBuffer
                    {
                        ParamHash = hash,
                        PushConstantSlot = staging.PushConstantSlot,
                        ElementStride = staging.ElementStride,
                        ElementsPerInstance = staging.ElementsPerInstance
                    };
                    _perInstanceBuffers[hash] = pib;
                }
                
                // Ensure staging capacity (dense, indexed by instance)
                int bytesPerInst = staging.BytesPerInstance;
                int requiredBytes = newTotal * bytesPerInst;
                if (pib.StagingData.Length < requiredBytes)
                {
                    int newCap = Math.Max(pib.StagingData.Length * 2, requiredBytes);
                    Array.Resize(ref pib.StagingData, newCap);
                }
                
                // Block-copy from bucket staging into batch staging
                int srcBytes = staging.Count * bytesPerInst;
                Buffer.BlockCopy(staging.Data, 0, pib.StagingData, startIdx * bytesPerInst, srcBytes);
                pib.Dirty = true;
            }
            
            // Track max transform slot for bone buffer sizing
            for (int i = startIdx; i < newTotal; i++)
            {
                if (_draws[i].TransformSlot > _maxTransformSlot)
                    _maxTransformSlot = _draws[i].TransformSlot;
            }
            
            _drawCount = newTotal;
        }

        private void ResizeBuffers(int newCapacity)
        {
            if (newCapacity <= capacity) return;
            capacity = newCapacity;

            // Defer disposal of old GPU buffers instead of immediate destroy
            DeferDisposeBuffers();

            Array.Resize(ref _draws, capacity);

            int slotSize = capacity * 4;
            int commandSize = MaxSubBatches * IndirectDrawSizes.IndirectCommandSize;

            for (int i = 0; i < FrameCount; ++i)
            {
                // GPU output: visible indices (compute shader writes)
                visibleIndicesBuffers[i] = Engine.Device.CreateDefaultBuffer(slotSize);
                visibleIndicesUAVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferUAV(visibleIndicesBuffers[i], (uint)capacity, 4, visibleIndicesUAVIndices[i]);
                visibleIndicesSRVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferSRV(visibleIndicesBuffers[i], (uint)capacity, 4, visibleIndicesSRVIndices[i]);

                // GPU output: indirect commands (compute shader writes)
                gpuCommandBuffers[i] = Engine.Device.CreateDefaultBuffer(commandSize);
                gpuCommandUAVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferUAV(gpuCommandBuffers[i], MaxSubBatches,
                    (uint)IndirectDrawSizes.IndirectCommandSize, gpuCommandUAVIndices[i]);
            }
        }

        public void UploadInstanceData(GraphicsDevice device)
        {
            if (_drawCount == 0) return;
            int frameIndex = Engine.FrameIndex % FrameCount;

            // All per-instance data (descriptors, bounding spheres, subbatch IDs,
            // terrain data, bones, etc.) is now uploaded through the unified path.
            UploadPerInstanceBuffers(frameIndex);
        }

        private void UploadPerInstanceBuffers(int frameIndex)
        {
            if (_drawCount == 0) return;
            int requiredSlots = _drawCount; // Dense: per-instance data is indexed by instance, not TransformSlot

            foreach (var (hash, pib) in _perInstanceBuffers)
            {
                if (!pib.Dirty) continue;
                if (pib.ElementStride == 0 || pib.ElementsPerInstance == 0) continue;

                // Ensure GPU buffer capacity
                EnsurePerInstanceBuffer(pib, requiredSlots);

                // Single MemoryCopy from pre-staged byte array — no iteration, no reflection
                int bytesPerSlot = pib.ElementsPerInstance * pib.ElementStride;
                int uploadBytes = requiredSlots * bytesPerSlot;
                if (uploadBytes > pib.StagingData.Length)
                    uploadBytes = pib.StagingData.Length;

                var buf = pib.Buffers[frameIndex];
                unsafe
                {
                    byte* pData;
                    buf.Map(0, null, (void**)&pData);
                    fixed (byte* pSrc = pib.StagingData)
                        Buffer.MemoryCopy(pSrc, pData, uploadBytes, uploadBytes);
                    buf.Unmap(0);
                }
            }
        }

        private void EnsurePerInstanceBuffer(PerInstanceBuffer pib, int requiredSlots)
        {
            if (pib.Capacity >= requiredSlots) return;

            // Defer disposal of old buffers — GPU may still be reading them from in-flight frames
            for (int i = 0; i < FrameCount; ++i)
            {
                DeferDispose(pib.Buffers[i], pib.SRVIndices[i]);
            }

            int totalElements = requiredSlots * pib.ElementsPerInstance;
            int bufferSize = totalElements * pib.ElementStride;

            for (int i = 0; i < FrameCount; ++i)
            {
                pib.Buffers[i] = Engine.Device.CreateUploadBuffer(bufferSize);
                pib.SRVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferSRV(
                    pib.Buffers[i],
                    (uint)totalElements,
                    (uint)pib.ElementStride,
                    pib.SRVIndices[i]);
            }

            pib.Capacity = requiredSlots;
        }

        public void Build(GraphicsDevice device, ID3D12GraphicsCommandList commandList, ulong frustumBufferAddress)
        {
            int count = _drawCount;
            if (count == 0) return;

            MeshRegistry.Upload(device);
            SubBatchCount = MeshRegistry.Count;
        }

        #region GPU Culler

        private void InitializeCullerResources()
        {
            if (_cullerInitialized) return;
            _cullerInitialized = true;

            var device = Engine.Device;
            int counterBufferSize = MaxSubBatches * sizeof(uint);
            int instanceBufferSize = capacity * sizeof(uint);

            var cpuHeapDesc = new DescriptorHeapDescription
            {
                Type = DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView,
                DescriptorCount = FrameCount * 3,
                Flags = DescriptorHeapFlags.None
            };
            _cpuHeap = device.NativeDevice.CreateDescriptorHeap(cpuHeapDesc);
            uint handleIncrementSize = device.NativeDevice.GetDescriptorHandleIncrementSize(
                DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView);

            for (int i = 0; i < FrameCount; i++)
            {
                counterBuffers[i] = device.CreateDefaultBuffer(counterBufferSize);
                counterBufferUAVIndices[i] = device.AllocateBindlessIndex();

                var uavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView
                    {
                        FirstElement = 0,
                        NumElements = MaxSubBatches,
                        StructureByteStride = sizeof(uint),
                        Flags = BufferUnorderedAccessViewFlags.None
                    }
                };
                device.NativeDevice.CreateUnorderedAccessView(
                    counterBuffers[i], null, uavDesc, device.GetCpuHandle(counterBufferUAVIndices[i]));

                var cpuHandle = _cpuHeap.GetCPUDescriptorHandleForHeapStart() + ((int)(i * handleIncrementSize));
                device.NativeDevice.CreateUnorderedAccessView(counterBuffers[i], null, uavDesc, cpuHandle);
                counterBufferCPUHandles[i] = cpuHandle;

                var visibleIndicesCpuHandle = _cpuHeap.GetCPUDescriptorHandleForHeapStart() + ((int)((FrameCount + i) * handleIncrementSize));
                var visibleUavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView
                    {
                        FirstElement = 0,
                        NumElements = (uint)capacity,
                        StructureByteStride = sizeof(uint),
                        Flags = BufferUnorderedAccessViewFlags.None
                    }
                };
                device.NativeDevice.CreateUnorderedAccessView(visibleIndicesBuffers[i], null, visibleUavDesc, visibleIndicesCpuHandle);
                visibleIndicesCPUHandles[i] = visibleIndicesCpuHandle;

                visibilityFlagsBuffers[i] = device.CreateDefaultBuffer(instanceBufferSize);
                visibilityFlagsUAVIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(visibilityFlagsBuffers[i], (uint)capacity, sizeof(uint), 
                    visibilityFlagsUAVIndices[i]);

                int histogramSize = MeshRegistry.MaxMeshParts * sizeof(uint);
                histogramBuffers[i] = device.CreateDefaultBuffer(histogramSize);
                histogramUAVIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(histogramBuffers[i], (uint)MeshRegistry.MaxMeshParts, sizeof(uint),
                    histogramUAVIndices[i]);
                
                var histogramCpuHandle = _cpuHeap.GetCPUDescriptorHandleForHeapStart() + ((int)((2 * FrameCount + i) * handleIncrementSize));
                var histogramUavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView
                    {
                        FirstElement = 0,
                        NumElements = (uint)MeshRegistry.MaxMeshParts,
                        StructureByteStride = sizeof(uint),
                        Flags = BufferUnorderedAccessViewFlags.None
                    }
                };
                device.NativeDevice.CreateUnorderedAccessView(histogramBuffers[i], null, histogramUavDesc, histogramCpuHandle);
                histogramCPUHandles[i] = histogramCpuHandle;
                
            }
        }

        public void Cull(ID3D12GraphicsCommandList commandList, ulong frustumBufferGPUAddress, GPUCuller? culler)
        {
            if (SubBatchCount == 0) return;
            if (culler?.VisibilityPSO == null || culler?.MainPSO == null || culler?.HistogramPSO == null) return;

            InitializeCullerResources();

            int frameIndex = Engine.FrameIndex % FrameCount;
            var device = Engine.Device;
            int totalInstances = _drawCount;

            commandList.SetComputeRootSignature(device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            commandList.SetComputeRootConstantBufferView(1, frustumBufferGPUAddress);

            commandList.SetComputeRoot32BitConstant(0, MeshRegistry.SrvIndex, 0);
            commandList.SetComputeRoot32BitConstant(0, gpuCommandUAVIndices[frameIndex], 1);
            
            commandList.SetComputeRoot32BitConstant(0, visibleIndicesUAVIndices[frameIndex], 5);
            commandList.SetComputeRoot32BitConstant(0, counterBufferUAVIndices[frameIndex], 6);
            commandList.SetComputeRoot32BitConstant(0, (uint)SubBatchCount, 7);
            
            commandList.SetComputeRoot32BitConstant(0, visibleIndicesSRVIndices[frameIndex], 9);
            commandList.SetComputeRoot32BitConstant(0, TransformBuffer.Instance?.SrvIndex ?? 0, 10);
            commandList.SetComputeRoot32BitConstant(0, visibilityFlagsUAVIndices[frameIndex], 11);
            commandList.SetComputeRoot32BitConstant(0, (uint)totalInstances, 13);
            commandList.SetComputeRoot32BitConstant(0, Material.MaterialsBufferIndex, 25);
            commandList.SetComputeRoot32BitConstant(0, histogramUAVIndices[frameIndex], 26);
            commandList.SetComputeRoot32BitConstant(0, (uint)MeshRegistry.Count, 27);


            // Bind per-instance buffer SRV indices to COMPUTE push constants (hardcoded slots).
            // Compute and graphics slot namespaces differ — can't reuse PushConstantSlot here.
            foreach (var (hash, pib) in _perInstanceBuffers)
            {
                if (!ComputeSlots.TryGetValue(hash, out int computeSlot)) continue;
                uint srvIdx = pib.SRVIndices[frameIndex];
                commandList.SetComputeRoot32BitConstant(0, srvIdx, (uint)computeSlot);
            }

            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(counterBufferUAVIndices[frameIndex]),
                counterBufferCPUHandles[frameIndex],
                counterBuffers[frameIndex],
                new Vortice.Mathematics.Int4(0, 0, 0, 0));
            
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(visibleIndicesUAVIndices[frameIndex]),
                visibleIndicesCPUHandles[frameIndex],
                visibleIndicesBuffers[frameIndex],
                new Vortice.Mathematics.Int4(unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF)));
            
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(histogramUAVIndices[frameIndex]),
                histogramCPUHandles[frameIndex],
                histogramBuffers[frameIndex],
                new Vortice.Mathematics.Int4(0, 0, 0, 0));
            

            
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Pass 1: CSVisibility
            commandList.SetPipelineState(culler.VisibilityPSO);
            commandList.Dispatch((uint)((totalInstances + 255) / 256), 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Pass 2: CSHistogram
            commandList.SetPipelineState(culler.HistogramPSO);
            commandList.Dispatch((uint)((totalInstances + 63) / 64), 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Pass 3: CSHistogramPrefixSum
            commandList.SetPipelineState(culler.HistogramPrefixSumPSO);
            commandList.Dispatch(1, 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Pass 4: CSGlobalScatter
            commandList.SetPipelineState(culler.GlobalScatterPSO);
            commandList.Dispatch((uint)((totalInstances + 255) / 256), 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Pass 5: CSMain
            commandList.SetPipelineState(culler.MainPSO);
            commandList.Dispatch((uint)((MeshRegistry.Count + 63) / 64), 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
        }

        private void InitializeShadowCullerResources()
        {
            if (_shadowCullerInitialized) return;
            _shadowCullerInitialized = true;

            var device = Engine.Device;
            int counterBufferSize = MaxSubBatches * sizeof(uint);
            int instanceBufferSize = capacity * sizeof(uint);
            int commandSize = MaxSubBatches * IndirectDrawSizes.IndirectCommandSize;
            int histogramSize = MeshRegistry.MaxMeshParts * sizeof(uint);

            // CPU descriptor heap for ClearUnorderedAccessViewUint (4 buffers per frame per cascade)
            int totalShadowDescriptors = FrameCount * ShadowCascadeCount * 3;
            var cpuHeapDesc = new DescriptorHeapDescription
            {
                Type = DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView,
                DescriptorCount = (uint)totalShadowDescriptors,
                Flags = DescriptorHeapFlags.None
            };
            _shadowCpuHeap = device.NativeDevice.CreateDescriptorHeap(cpuHeapDesc);
            uint handleIncrement = device.NativeDevice.GetDescriptorHandleIncrementSize(
                DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView);
            int descriptorIdx = 0;

            for (int f = 0; f < FrameCount; f++)
            {
                for (int c = 0; c < ShadowCascadeCount; c++)
                {
                    // Visibility flags
                    shadowVisibilityBuffers[f, c] = device.CreateDefaultBuffer(instanceBufferSize);
                    shadowVisibilityUAVIndices[f, c] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferUAV(shadowVisibilityBuffers[f, c], (uint)capacity, sizeof(uint), shadowVisibilityUAVIndices[f, c]);

                    // Visible indices output
                    shadowVisibleIndicesBuffers[f, c] = device.CreateDefaultBuffer(instanceBufferSize);
                    shadowVisibleIndicesUAVIndices[f, c] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferUAV(shadowVisibleIndicesBuffers[f, c], (uint)capacity, sizeof(uint), shadowVisibleIndicesUAVIndices[f, c]);
                    shadowVisibleIndicesSRVIndices[f, c] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferSRV(shadowVisibleIndicesBuffers[f, c], (uint)capacity, sizeof(uint), shadowVisibleIndicesSRVIndices[f, c]);

                    // CPU handle for visible indices clear
                    var visibleCpuHandle = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(descriptorIdx * handleIncrement);
                    var visibleUavDesc = new UnorderedAccessViewDescription
                    {
                        Format = Vortice.DXGI.Format.Unknown,
                        ViewDimension = UnorderedAccessViewDimension.Buffer,
                        Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = (uint)capacity, StructureByteStride = sizeof(uint) }
                    };
                    device.NativeDevice.CreateUnorderedAccessView(shadowVisibleIndicesBuffers[f, c], null, visibleUavDesc, visibleCpuHandle);
                    shadowVisibleIndicesCPUHandles[f, c] = visibleCpuHandle;
                    descriptorIdx++;



                    // Indirect commands
                    shadowCommandBuffers[f, c] = device.CreateDefaultBuffer(commandSize);
                    shadowCommandUAVIndices[f, c] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferUAV(shadowCommandBuffers[f, c], MaxSubBatches, (uint)IndirectDrawSizes.IndirectCommandSize, shadowCommandUAVIndices[f, c]);

                    // Counters
                    shadowCounterBuffers[f, c] = device.CreateDefaultBuffer(counterBufferSize);
                    shadowCounterUAVIndices[f, c] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferUAV(shadowCounterBuffers[f, c], MaxSubBatches, sizeof(uint), shadowCounterUAVIndices[f, c]);

                    // CPU handle for counter clear
                    var counterCpuHandle = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(descriptorIdx * handleIncrement);
                    var counterUavDesc = new UnorderedAccessViewDescription
                    {
                        Format = Vortice.DXGI.Format.Unknown,
                        ViewDimension = UnorderedAccessViewDimension.Buffer,
                        Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = MaxSubBatches, StructureByteStride = sizeof(uint) }
                    };
                    device.NativeDevice.CreateUnorderedAccessView(shadowCounterBuffers[f, c], null, counterUavDesc, counterCpuHandle);
                    shadowCounterCPUHandles[f, c] = counterCpuHandle;
                    descriptorIdx++;

                    // Histogram
                    shadowHistogramBuffers[f, c] = device.CreateDefaultBuffer(histogramSize);
                    shadowHistogramUAVIndices[f, c] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferUAV(shadowHistogramBuffers[f, c], (uint)MeshRegistry.MaxMeshParts, sizeof(uint), shadowHistogramUAVIndices[f, c]);

                    // CPU handle for histogram clear
                    var histogramCpuHandle = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(descriptorIdx * handleIncrement);
                    var histogramUavDesc = new UnorderedAccessViewDescription
                    {
                        Format = Vortice.DXGI.Format.Unknown,
                        ViewDimension = UnorderedAccessViewDimension.Buffer,
                        Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = (uint)MeshRegistry.MaxMeshParts, StructureByteStride = sizeof(uint) }
                    };
                    device.NativeDevice.CreateUnorderedAccessView(shadowHistogramBuffers[f, c], null, histogramUavDesc, histogramCpuHandle);
                    shadowHistogramCPUHandles[f, c] = histogramCpuHandle;
                    descriptorIdx++;
                }
            }
        }

        /// <summary>
        /// Cull instances for a shadow cascade using GPU-driven pipeline.
        /// </summary>
        public void CullShadow(ID3D12GraphicsCommandList commandList, int cascadeIndex, ulong shadowCascadeBufferAddress, GPUCuller? culler)
        {
            if (SubBatchCount == 0 || cascadeIndex < 0 || cascadeIndex >= ShadowCascadeCount) return;
            if (culler?.VisibilityShadowPSO == null || culler?.MainPSO == null || culler?.HistogramPSO == null) return;

            InitializeShadowCullerResources();

            int frameIndex = Engine.FrameIndex % FrameCount;
            var device = Engine.Device;
            int totalInstances = _drawCount;

            commandList.SetComputeRootSignature(device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            
            // Bind shadow cascade planes cbuffer (slot 2 -> register b1)
            commandList.SetComputeRootConstantBufferView(2, shadowCascadeBufferAddress);

            // Set push constants for shadow culling (using shadow-specific buffers)
            commandList.SetComputeRoot32BitConstant(0, MeshRegistry.SrvIndex, 0);
            commandList.SetComputeRoot32BitConstant(0, shadowCommandUAVIndices[frameIndex, cascadeIndex], 1);
            commandList.SetComputeRoot32BitConstant(0, shadowVisibleIndicesUAVIndices[frameIndex, cascadeIndex], 5);
            commandList.SetComputeRoot32BitConstant(0, shadowCounterUAVIndices[frameIndex, cascadeIndex], 6);
            commandList.SetComputeRoot32BitConstant(0, (uint)SubBatchCount, 7);
            commandList.SetComputeRoot32BitConstant(0, shadowVisibleIndicesSRVIndices[frameIndex, cascadeIndex], 9);
            commandList.SetComputeRoot32BitConstant(0, TransformBuffer.Instance?.SrvIndex ?? 0, 10);
            commandList.SetComputeRoot32BitConstant(0, shadowVisibilityUAVIndices[frameIndex, cascadeIndex], 11);
            commandList.SetComputeRoot32BitConstant(0, (uint)totalInstances, 13);
            commandList.SetComputeRoot32BitConstant(0, Material.MaterialsBufferIndex, 25);
            commandList.SetComputeRoot32BitConstant(0, shadowHistogramUAVIndices[frameIndex, cascadeIndex], 26);
            commandList.SetComputeRoot32BitConstant(0, (uint)MeshRegistry.Count, 27);

            // Bind per-instance buffer SRV indices to COMPUTE push constants (hardcoded slots).
            foreach (var (hash, pib) in _perInstanceBuffers)
            {
                if (!ComputeSlots.TryGetValue(hash, out int computeSlot)) continue;
                uint srvIdx = pib.SRVIndices[frameIndex];
                commandList.SetComputeRoot32BitConstant(0, srvIdx, (uint)computeSlot);
            }
            commandList.SetComputeRoot32BitConstant(0, (uint)cascadeIndex, 31); // ShadowCascadeIdx

            // Clear UAV buffers before dispatching (matching regular Cull pattern)
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(shadowCounterUAVIndices[frameIndex, cascadeIndex]),
                shadowCounterCPUHandles[frameIndex, cascadeIndex],
                shadowCounterBuffers[frameIndex, cascadeIndex],
                new Vortice.Mathematics.Int4(0, 0, 0, 0));
            
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(shadowVisibleIndicesUAVIndices[frameIndex, cascadeIndex]),
                shadowVisibleIndicesCPUHandles[frameIndex, cascadeIndex],
                shadowVisibleIndicesBuffers[frameIndex, cascadeIndex],
                new Vortice.Mathematics.Int4(unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF)));
            
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(shadowHistogramUAVIndices[frameIndex, cascadeIndex]),
                shadowHistogramCPUHandles[frameIndex, cascadeIndex],
                shadowHistogramBuffers[frameIndex, cascadeIndex],
                new Vortice.Mathematics.Int4(0, 0, 0, 0));
            

            
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Pass 1: CSVisibilityShadow (uses shadow cascade frustum)
            commandList.SetPipelineState(culler.VisibilityShadowPSO);
            commandList.Dispatch((uint)((totalInstances + 255) / 256), 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Pass 2: CSHistogram  
            commandList.SetPipelineState(culler.HistogramPSO);
            commandList.Dispatch((uint)((totalInstances + 63) / 64), 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Pass 3: CSHistogramPrefixSum
            commandList.SetPipelineState(culler.HistogramPrefixSumPSO);
            commandList.Dispatch(1, 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Pass 4: CSGlobalScatter
            commandList.SetPipelineState(culler.GlobalScatterPSO);
            commandList.Dispatch((uint)((totalInstances + 255) / 256), 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Pass 5: CSMain (generate commands)
            commandList.SetPipelineState(culler.MainPSO);
            commandList.Dispatch((uint)((MeshRegistry.Count + 63) / 64), 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
        }

        /// <summary>
        /// Draw shadow cascade using GPU-driven indirect commands.
        /// Uses a dedicated shadow SceneConstants CBV to avoid overwriting the opaque pass's CBV.
        /// </summary>
        public void DrawShadow(ID3D12GraphicsCommandList commandList, int cascadeIndex, ulong shadowSceneCBVAddress)
        {
            int frameIndex = Engine.FrameIndex % FrameCount;
            if (!_shadowCullerInitialized || SubBatchCount == 0 || cascadeIndex < 0 || cascadeIndex >= ShadowCascadeCount)
            {
                if (Engine.FrameIndex % 60 == 0 && cascadeIndex == 0)
                    Debug.Log($"[Shadow] DrawShadow SKIP: cullerInit={_shadowCullerInitialized}, subBatches={SubBatchCount}, cascade={cascadeIndex}");
                return;
            }

            var commandBuffer = shadowCommandBuffers[frameIndex, cascadeIndex];
            if (commandBuffer == null)
            {
                if (Engine.FrameIndex % 60 == 0 && cascadeIndex == 0)
                    Debug.Log($"[Shadow] DrawShadow SKIP: commandBuffer is null for frame={frameIndex}, cascade={cascadeIndex}");
                return;
            }
            
            // Skip if material doesn't have a Shadow pass
            if (!Material.HasPass(RenderPass.Shadow))
            {
                if (Engine.FrameIndex % 60 == 0 && cascadeIndex == 0)
                    Debug.Log($"[Shadow] DrawShadow SKIP: No Shadow pass on material");
                return;
            }
            
            if (Engine.FrameIndex % 60 == 0 && cascadeIndex == 0)
                Debug.Log($"[Shadow] DrawShadow: subBatches={SubBatchCount}, cbvAddr=0x{shadowSceneCBVAddress:X}, frame={frameIndex}");
            
            // Use Material.Apply for full graphics state setup (PSO, root sig, descriptors,
            // resource bindings, Materials buffer). Then override the SceneConstants CBV with our
            // dedicated shadow buffer to prevent overwriting the camera matrices used by opaque pass.
            Material.SetPass(RenderPass.Shadow);
            Material.Apply(commandList, Engine.Device, null);
            
            // OVERRIDE: Rebind SceneConstants CBV (slot 1 = register b0) with shadow-specific
            // buffer containing light View/Projection. Material.Apply just committed the camera's
            // SceneConstants, but the shadow VS needs the light matrices.
            commandList.SetGraphicsRootConstantBufferView(1, shadowSceneCBVAddress);

            // Bind per-instance buffer SRV indices to GRAPHICS push constants (same as Draw).
            foreach (var (hash, pib) in _perInstanceBuffers)
            {
                if (pib.PushConstantSlot < 0) continue;
                uint srvIdx = pib.SRVIndices[frameIndex];
                commandList.SetGraphicsRoot32BitConstant(0, srvIdx, (uint)pib.PushConstantSlot);
            }

            // Transition buffers for reading
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(shadowVisibleIndicesBuffers[frameIndex, cascadeIndex],
                    ResourceStates.UnorderedAccess,
                    ResourceStates.NonPixelShaderResource)));
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(commandBuffer,
                    ResourceStates.UnorderedAccess,
                    ResourceStates.IndirectArgument)));

            commandList.ExecuteIndirect(
                Engine.Device.BindlessCommandSignature,
                (uint)SubBatchCount,
                commandBuffer,
                0,
                null,
                0
            );

            // Transition back for next frame
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(commandBuffer,
                    ResourceStates.IndirectArgument,
                    ResourceStates.UnorderedAccess)));
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(shadowVisibleIndicesBuffers[frameIndex, cascadeIndex],
                    ResourceStates.NonPixelShaderResource,
                    ResourceStates.UnorderedAccess)));
        }

        #endregion

        public void Draw(ID3D12GraphicsCommandList commandList, GraphicsDevice device)
        {
            int frameIndex = Engine.FrameIndex % FrameCount;
            if (SubBatchCount == 0) return;
            
            if (Engine.FrameIndex % 60 == 0)
                Debug.Log($"[Batch.Draw] Effect={Material?.Effect?.Name ?? "null"} DrawCount={_drawCount} SubBatches={SubBatchCount} Desc0={(_drawCount > 0 ? $"slot={_draws[0].TransformSlot},mat={_draws[0].MaterialId}" : "N/A")}");

            var commandBuffer = gpuCommandBuffers[frameIndex];
            if (commandBuffer == null) return;

            // Bind per-instance buffer SRV indices to GRAPHICS push constants.
            // The command signature only writes slots 2-15 per draw.
            // Per-instance buffers (e.g. TerrainPatchData at slot 1) use slots outside
            // the command signature range and must be set explicitly.
            foreach (var (hash, pib) in _perInstanceBuffers)
            {
                if (pib.PushConstantSlot < 0) continue;
                uint srvIdx = pib.SRVIndices[frameIndex];
                commandList.SetGraphicsRoot32BitConstant(0, srvIdx, (uint)pib.PushConstantSlot);
            }

            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(visibleIndicesBuffers[frameIndex],
                    ResourceStates.UnorderedAccess,
                    ResourceStates.NonPixelShaderResource)));

            
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(commandBuffer,
                    ResourceStates.UnorderedAccess,
                    ResourceStates.IndirectArgument)));

            commandList.ExecuteIndirect(
                Engine.Device.BindlessCommandSignature,
                (uint)SubBatchCount,
                commandBuffer,
                0,
                null,
                0
            );

            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(commandBuffer,
                    ResourceStates.IndirectArgument,
                    ResourceStates.UnorderedAccess)));
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(visibleIndicesBuffers[frameIndex],
                    ResourceStates.NonPixelShaderResource,
                    ResourceStates.UnorderedAccess)));

        }

        /// <summary>
        /// Reset buffer states after draw for next frame. Currently empty.
        /// </summary>
        public void ResetBufferState(ID3D12GraphicsCommandList commandList) { }

        /// <summary>
        /// Queue old buffers for deferred disposal (GPU may still be using them).
        /// </summary>
        private void DeferDisposeBuffers()
        {
            for (int i = 0; i < FrameCount; ++i)
            {

                DeferDispose(visibleIndicesBuffers[i], visibleIndicesUAVIndices[i]);
                DeferDispose(null, visibleIndicesSRVIndices[i]);
                visibleIndicesBuffers[i] = null; visibleIndicesUAVIndices[i] = 0; visibleIndicesSRVIndices[i] = 0;



                DeferDispose(gpuCommandBuffers[i], gpuCommandUAVIndices[i]);
                gpuCommandBuffers[i] = null; gpuCommandUAVIndices[i] = 0;

                DeferDispose(counterBuffers[i], counterBufferUAVIndices[i]);
                counterBuffers[i] = null; counterBufferUAVIndices[i] = 0;

                DeferDispose(visibilityFlagsBuffers[i], visibilityFlagsUAVIndices[i]);
                visibilityFlagsBuffers[i] = null; visibilityFlagsUAVIndices[i] = 0;

                DeferDispose(histogramBuffers[i], histogramUAVIndices[i]);
                histogramBuffers[i] = null; histogramUAVIndices[i] = 0;
            }

            _cpuHeap?.Dispose();
            _cpuHeap = null;
            _cullerInitialized = false;

            // Defer shadow cascade buffers too
            if (_shadowCullerInitialized)
            {
                for (int f = 0; f < FrameCount; f++)
                {
                    for (int c = 0; c < ShadowCascadeCount; c++)
                    {
                        DeferDispose(shadowVisibilityBuffers[f, c], shadowVisibilityUAVIndices[f, c]);
                        shadowVisibilityBuffers[f, c] = null; shadowVisibilityUAVIndices[f, c] = 0;

                        DeferDispose(shadowVisibleIndicesBuffers[f, c], shadowVisibleIndicesUAVIndices[f, c]);
                        DeferDispose(null, shadowVisibleIndicesSRVIndices[f, c]);
                        shadowVisibleIndicesBuffers[f, c] = null; shadowVisibleIndicesUAVIndices[f, c] = 0; shadowVisibleIndicesSRVIndices[f, c] = 0;



                        DeferDispose(shadowCommandBuffers[f, c], shadowCommandUAVIndices[f, c]);
                        shadowCommandBuffers[f, c] = null; shadowCommandUAVIndices[f, c] = 0;

                        DeferDispose(shadowCounterBuffers[f, c], shadowCounterUAVIndices[f, c]);
                        shadowCounterBuffers[f, c] = null; shadowCounterUAVIndices[f, c] = 0;

                        DeferDispose(shadowHistogramBuffers[f, c], shadowHistogramUAVIndices[f, c]);
                        shadowHistogramBuffers[f, c] = null; shadowHistogramUAVIndices[f, c] = 0;
                    }
                }
                _shadowCullerInitialized = false;
                _shadowCpuHeap?.Dispose();
                _shadowCpuHeap = null;
            }
        }

        private void DisposeBuffers()
        {
            for (int i = 0; i < FrameCount; ++i)
            {
                visibleIndicesBuffers[i]?.Dispose();
                if (visibleIndicesUAVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(visibleIndicesUAVIndices[i]);
                if (visibleIndicesSRVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(visibleIndicesSRVIndices[i]);
                
                gpuCommandBuffers[i]?.Dispose();
                if (gpuCommandUAVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(gpuCommandUAVIndices[i]);
                
                counterBuffers[i]?.Dispose();
                if (counterBufferUAVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(counterBufferUAVIndices[i]);

                visibleIndicesUAVIndices[i] = 0;
                visibleIndicesSRVIndices[i] = 0;
                gpuCommandUAVIndices[i] = 0;
                counterBufferUAVIndices[i] = 0;
            }

            _cpuHeap?.Dispose();
            _cpuHeap = null;
            _cullerInitialized = false;
            
            // Dispose shadow cascade buffers
            if (_shadowCullerInitialized)
            {
                for (int f = 0; f < FrameCount; f++)
                {
                    for (int c = 0; c < ShadowCascadeCount; c++)
                    {
                        shadowVisibilityBuffers[f, c]?.Dispose();
                        if (shadowVisibilityUAVIndices[f, c] != 0) Engine.Device.ReleaseBindlessIndex(shadowVisibilityUAVIndices[f, c]);
                        
                        shadowVisibleIndicesBuffers[f, c]?.Dispose();
                        if (shadowVisibleIndicesUAVIndices[f, c] != 0) Engine.Device.ReleaseBindlessIndex(shadowVisibleIndicesUAVIndices[f, c]);
                        if (shadowVisibleIndicesSRVIndices[f, c] != 0) Engine.Device.ReleaseBindlessIndex(shadowVisibleIndicesSRVIndices[f, c]);
                        

                        
                        shadowCommandBuffers[f, c]?.Dispose();
                        if (shadowCommandUAVIndices[f, c] != 0) Engine.Device.ReleaseBindlessIndex(shadowCommandUAVIndices[f, c]);
                        
                        shadowCounterBuffers[f, c]?.Dispose();
                        if (shadowCounterUAVIndices[f, c] != 0) Engine.Device.ReleaseBindlessIndex(shadowCounterUAVIndices[f, c]);
                        
                        shadowHistogramBuffers[f, c]?.Dispose();  
                        if (shadowHistogramUAVIndices[f, c] != 0) Engine.Device.ReleaseBindlessIndex(shadowHistogramUAVIndices[f, c]);
                    }
                }
                _shadowCullerInitialized = false;
            }
        }

        public void Dispose()
        {
            DisposeBuffers();
            foreach (var kvp in _perInstanceBuffers)
            {
                for (int i = 0; i < FrameCount; ++i)
                {
                    kvp.Value.Buffers[i]?.Dispose();
                    if (kvp.Value.SRVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(kvp.Value.SRVIndices[i]);
                }
            }
            _perInstanceBuffers.Clear();
        }
    }
}
