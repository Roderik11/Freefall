using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Components;
using Vortice.Direct3D;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// GPU-driven instance batch. All command generation happens on the GPU.
    /// Flow: unsorted draws â†’ GPU frustum culling â†’ cache-friendly grouping â†’ indirect commands
    /// </summary>
    public sealed class InstanceBatch : IDisposable
    {
        public const int FrameCount = 3;

        // Deferred disposal queue â€” old GPU buffers kept alive until GPU is done with them
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
            
            // CPU staging buffer â€” filled during MergeFromBucket, uploaded in one MemoryCopy
            public byte[] StagingData = Array.Empty<byte>();
            public int StagingCapacity; // in slots (matches Capacity growth)
        }
        
        // Active per-instance buffers for this batch
        private Dictionary<int, PerInstanceBuffer> _perInstanceBuffers = new();
        
        // Hardcoded compute push constant slots for per-instance buffers.
        // These are core pipeline constants â€” compute layout is fixed in cull_instances.hlsl.
        // Graphics slots (from shader resource bindings) are separate and used in Draw/DrawShadow.
        private static readonly Dictionary<int, int> ComputeSlots = new()
        {
            { "Bones".GetHashCode(), 30 },              // Indices[7].z = BoneBufferIdx
            // BoundingSpheres removed â€” culler reads bounds from MeshRegistry
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
        private uint[] visibilityFlagsSRVIndices = new uint[FrameCount]; // SRV for reading previous frame's flags
        
        private const int MaxSubBatches = 4096;
        private bool _cullerInitialized = false;
        private ID3D12DescriptorHeap? _cpuHeap;
        
        // Histogram buffer for GPU-driven per-MeshPartId instance counting
        private ID3D12Resource[] histogramBuffers = new ID3D12Resource[FrameCount];
        private uint[] histogramUAVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] histogramCPUHandles = new CpuDescriptorHandle[FrameCount];
        
        // Shadow culling â€” union visibility + cascade mask (single-pass rendering)


        private ID3D12Resource[] shadowCombinedVisBuffers = new ID3D12Resource[FrameCount];
        private uint[] shadowCombinedVisUAVIndices = new uint[FrameCount];
        private ID3D12Resource[] shadowCascadeMaskBuffers = new ID3D12Resource[FrameCount];
        private uint[] shadowCascadeMaskUAVIndices = new uint[FrameCount];
        private uint[] shadowCascadeMaskSRVIndices = new uint[FrameCount]; // For VS cascade early-out
        private ID3D12Resource[] shadowVisibleIndicesBuffers = new ID3D12Resource[FrameCount];
        private uint[] shadowVisibleIndicesUAVIndices = new uint[FrameCount];
        private uint[] shadowVisibleIndicesSRVIndices = new uint[FrameCount];
        private ID3D12Resource[] shadowCommandBuffers = new ID3D12Resource[FrameCount];
        private uint[] shadowCommandUAVIndices = new uint[FrameCount];
        private ID3D12Resource[] shadowCounterBuffers = new ID3D12Resource[FrameCount];
        private uint[] shadowCounterUAVIndices = new uint[FrameCount];
        private ID3D12Resource[] shadowHistogramBuffers = new ID3D12Resource[FrameCount];
        private uint[] shadowHistogramUAVIndices = new uint[FrameCount];
        private bool _shadowCullerInitialized = false;
        private ID3D12DescriptorHeap? _shadowCpuHeap;
        private CpuDescriptorHandle[] shadowCounterCPUHandles = new CpuDescriptorHandle[FrameCount];
        private CpuDescriptorHandle[] shadowVisibleIndicesCPUHandles = new CpuDescriptorHandle[FrameCount];
        private CpuDescriptorHandle[] shadowHistogramCPUHandles = new CpuDescriptorHandle[FrameCount];

        // Shadow cascade expansion buffer — stores (cascadeIdx<<30 | instanceIdx) per visible cascade
        private ID3D12Resource[] shadowExpansionBuffers = new ID3D12Resource[FrameCount];
        private uint[] shadowExpansionUAVIndices = new uint[FrameCount];
        private uint[] shadowExpansionSRVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] shadowExpansionCPUHandles = new CpuDescriptorHandle[FrameCount];



        private int _drawCount = 0;
        private int _maxTransformSlot = 0;
        internal bool _isGPUSourced = false;

        public PrimitiveTopology Topology { get; private set; }

        public InstanceBatch(BatchKey key, Material material)
        {
            Key = key;
            Material = material;
            Topology = material.HasTessellation
                        ? PrimitiveTopology.PatchListWith3ControlPoints
                        : PrimitiveTopology.TriangleList;

            ResizeBuffers(8192);
        }

        public void Clear()
        {
            _drawCount = 0;
            _uniqueMeshPartIds.Clear();
            SubBatchCount = 0;
            _maxTransformSlot = 0;
            _isGPUSourced = false;
            foreach (var buf in _perInstanceBuffers.Values)
                buf.Dirty = false;
        }

        /// <summary>
        /// Descriptor for a GPU-generated per-instance buffer channel.
        /// </summary>
        public struct GPUBufferBinding
        {
            public int ParamHash;
            public int PushConstantSlot;
            public int ElementStride;
            public uint SrvIndex;
        }

        /// <summary>
        /// Attach GPU-generated per-instance buffers (bypasses CPU staging/upload).
        /// The provided SRV indices point at buffers filled by a compute shader.
        /// descriptorsSRV/subbatchIdsSRV are core culler infrastructure.
        /// customBindings contains additional per-instance data channels (e.g. TerrainPatchData).
        /// </summary>
        public void AttachGPUData(
            uint descriptorsSRV, uint subbatchIdsSRV,
            int instanceCount, int meshPartId,
            ReadOnlySpan<GPUBufferBinding> customBindings = default)
        {
            _isGPUSourced = true;
            _drawCount = instanceCount;
            
            // Ensure capacity is large enough for culler buffers
            if (instanceCount > capacity)
                ResizeBuffers(Math.Max(instanceCount, capacity * 2));
            
            // Register the mesh part so MeshRegistry has it
            _uniqueMeshPartIds.Add(meshPartId);
            SubBatchCount = _uniqueMeshPartIds.Count;
            
            int frameIndex = Engine.FrameIndex % FrameCount;
            
            // Core culler infrastructure buffers (bounding spheres now come from MeshRegistry)
            EnsureGPUPerInstanceBuffer(DrawBucket.DescriptorsHash, -1, 12, 1, frameIndex, descriptorsSRV);
            EnsureGPUPerInstanceBuffer(DrawBucket.SubbatchIdsHash, -1, 4, 1, frameIndex, subbatchIdsSRV);
            
            // Custom per-instance data channels (caller provides hash, slot, stride)
            foreach (ref readonly var binding in customBindings)
            {
                EnsureGPUPerInstanceBuffer(binding.ParamHash, binding.PushConstantSlot,
                    binding.ElementStride, 1, frameIndex, binding.SrvIndex);
            }
        }

        private void EnsureGPUPerInstanceBuffer(int hash, int pushConstantSlot, int elementStride, int elementsPerInstance, int frameIndex, uint srvIndex)
        {
            if (!_perInstanceBuffers.TryGetValue(hash, out var pib))
            {
                pib = new PerInstanceBuffer
                {
                    ParamHash = hash,
                    PushConstantSlot = pushConstantSlot,
                    ElementStride = elementStride,
                    ElementsPerInstance = elementsPerInstance,
                };
                _perInstanceBuffers[hash] = pib;
            }
            // Override the SRV index for this frame with the GPU-generated buffer
            pib.SRVIndices[frameIndex] = srvIndex;
            pib.Dirty = false; // Don't upload CPU data
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
            // This includes core channels (Descriptors, SubbatchIds) and
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
            if (_drawCount == 0 || _isGPUSourced) return;
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

                // Single MemoryCopy from pre-staged byte array â€” no iteration, no reflection
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

            // Defer disposal of old buffers â€” GPU may still be reading them from in-flight frames
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
            if (!_isGPUSourced)
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
                // SRV for reading previous frame's visibility flags (feedback loop breaker)
                visibilityFlagsSRVIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferSRV(visibilityFlagsBuffers[i], (uint)capacity, sizeof(uint),
                    visibilityFlagsSRVIndices[i]);

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
            // Slot 2: previous frame's visibility flags SRV (feedback loop breaker)
            int prevFrameIndex = (frameIndex + FrameCount - 1) % FrameCount;
            commandList.SetComputeRoot32BitConstant(0, visibilityFlagsSRVIndices[prevFrameIndex], 2);
            commandList.SetComputeRoot32BitConstant(0, 1u, 3);  // InstanceMultiplier = 1 (shadow sets 4; must reset)
            
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
            // Compute and graphics slot namespaces differ â€” can't reuse PushConstantSlot here.
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

            // Barrier after clears: only the 3 cleared buffers need sync
            commandList.ResourceBarrier(new[] {
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(counterBuffers[frameIndex])),
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(visibleIndicesBuffers[frameIndex])),
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(histogramBuffers[frameIndex]))
            });

            // Pass 1: CSVisibility â€” writes visibilityFlags
            commandList.SetPipelineState(culler.VisibilityPSO);
            commandList.Dispatch((uint)((totalInstances + 255) / 256), 1, 1);
            // Only visibilityFlags was written; next pass (Histogram) reads it
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(visibilityFlagsBuffers[frameIndex])));

            // Pass 2: CSHistogram â€” reads visibilityFlags, writes histogram
            commandList.SetPipelineState(culler.HistogramPSO);
            commandList.Dispatch((uint)((totalInstances + 63) / 64), 1, 1);
            // Only histogram was written; next pass (PrefixSum) reads it
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(histogramBuffers[frameIndex])));

            // Pass 3: CSHistogramPrefixSum â€” reads histogram, writes counters
            commandList.SetPipelineState(culler.HistogramPrefixSumPSO);
            commandList.Dispatch(1, 1, 1);
            // Only counters was written; next pass (Scatter) reads it
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(counterBuffers[frameIndex])));

            // Pass 4: CSGlobalScatter â€” reads visibilityFlags+counters, writes visibleIndices+counters
            commandList.SetPipelineState(culler.GlobalScatterPSO);
            commandList.Dispatch((uint)((totalInstances + 255) / 256), 1, 1);
            // visibleIndices and counters were written; next pass (CSMain) reads both
            commandList.ResourceBarrier(new[] {
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(visibleIndicesBuffers[frameIndex])),
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(counterBuffers[frameIndex]))
            });

            // Pass 5: CSMain â€” reads histogram+counters, writes commands
            commandList.SetPipelineState(culler.MainPSO);
            commandList.Dispatch((uint)((MeshRegistry.Count + 63) / 64), 1, 1);
            // Commands buffer transitions to IndirectArgument in Draw(), no UAV barrier needed here
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

            // CPU descriptor heap for ClearUnorderedAccessViewUint (4 buffers per frame: counter, visible indices, histogram, expansion)
            int totalShadowDescriptors = FrameCount * 4;
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
                // Combined visibility flags (union of all cascades)
                shadowCombinedVisBuffers[f] = device.CreateDefaultBuffer(instanceBufferSize);
                shadowCombinedVisUAVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(shadowCombinedVisBuffers[f], (uint)capacity, sizeof(uint), shadowCombinedVisUAVIndices[f]);

                // Cascade mask (4-bit per instance, for VS early-out)
                shadowCascadeMaskBuffers[f] = device.CreateDefaultBuffer(instanceBufferSize);
                shadowCascadeMaskUAVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(shadowCascadeMaskBuffers[f], (uint)capacity, sizeof(uint), shadowCascadeMaskUAVIndices[f]);
                shadowCascadeMaskSRVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferSRV(shadowCascadeMaskBuffers[f], (uint)capacity, sizeof(uint), shadowCascadeMaskSRVIndices[f]);

                // Visible indices output (ONE buffer, union-compacted)
                shadowVisibleIndicesBuffers[f] = device.CreateDefaultBuffer(instanceBufferSize);
                shadowVisibleIndicesUAVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(shadowVisibleIndicesBuffers[f], (uint)capacity, sizeof(uint), shadowVisibleIndicesUAVIndices[f]);
                shadowVisibleIndicesSRVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferSRV(shadowVisibleIndicesBuffers[f], (uint)capacity, sizeof(uint), shadowVisibleIndicesSRVIndices[f]);

                // CPU handle for visible indices clear
                var visibleCpuHandle = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(descriptorIdx * handleIncrement);
                var visibleUavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = (uint)capacity, StructureByteStride = sizeof(uint) }
                };
                device.NativeDevice.CreateUnorderedAccessView(shadowVisibleIndicesBuffers[f], null, visibleUavDesc, visibleCpuHandle);
                shadowVisibleIndicesCPUHandles[f] = visibleCpuHandle;
                descriptorIdx++;

                // Indirect commands (ONE buffer)
                shadowCommandBuffers[f] = device.CreateDefaultBuffer(commandSize);
                shadowCommandUAVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(shadowCommandBuffers[f], MaxSubBatches, (uint)IndirectDrawSizes.IndirectCommandSize, shadowCommandUAVIndices[f]);

                // Counters (ONE buffer)
                shadowCounterBuffers[f] = device.CreateDefaultBuffer(counterBufferSize);
                shadowCounterUAVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(shadowCounterBuffers[f], MaxSubBatches, sizeof(uint), shadowCounterUAVIndices[f]);

                // CPU handle for counter clear
                var counterCpuHandle = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(descriptorIdx * handleIncrement);
                var counterUavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = MaxSubBatches, StructureByteStride = sizeof(uint) }
                };
                device.NativeDevice.CreateUnorderedAccessView(shadowCounterBuffers[f], null, counterUavDesc, counterCpuHandle);
                shadowCounterCPUHandles[f] = counterCpuHandle;
                descriptorIdx++;

                // Histogram (ONE buffer)
                shadowHistogramBuffers[f] = device.CreateDefaultBuffer(histogramSize);
                shadowHistogramUAVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(shadowHistogramBuffers[f], (uint)MeshRegistry.MaxMeshParts, sizeof(uint), shadowHistogramUAVIndices[f]);

                // CPU handle for histogram clear
                var histogramCpuHandle = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(descriptorIdx * handleIncrement);
                var histogramUavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = (uint)MeshRegistry.MaxMeshParts, StructureByteStride = sizeof(uint) }
                };
                device.NativeDevice.CreateUnorderedAccessView(shadowHistogramBuffers[f], null, histogramUavDesc, histogramCpuHandle);
                shadowHistogramCPUHandles[f] = histogramCpuHandle;
                descriptorIdx++;

                // Expansion buffer (capacity × 4 = worst case when every instance is in all 4 cascades)
                int expansionBufferSize = capacity * 4 * sizeof(uint);
                shadowExpansionBuffers[f] = device.CreateDefaultBuffer(expansionBufferSize);
                shadowExpansionUAVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(shadowExpansionBuffers[f], (uint)(capacity * 4), sizeof(uint), shadowExpansionUAVIndices[f]);
                shadowExpansionSRVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferSRV(shadowExpansionBuffers[f], (uint)(capacity * 4), sizeof(uint), shadowExpansionSRVIndices[f]);

                // CPU handle for expansion clear
                var expansionCpuHandle = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(descriptorIdx * handleIncrement);
                var expansionUavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = (uint)(capacity * 4), StructureByteStride = sizeof(uint) }
                };
                device.NativeDevice.CreateUnorderedAccessView(shadowExpansionBuffers[f], null, expansionUavDesc, expansionCpuHandle);
                shadowExpansionCPUHandles[f] = expansionCpuHandle;
                descriptorIdx++;
            }
        }

        /// <summary>
        /// Cull all shadow cascades in a single pass using union visibility.
        /// CSVisibilityShadow4 writes combined visibility (union) + cascade mask.
        /// Compaction pipeline runs ONCE on the union set. CSMain sets DrawInstanceCount × 4.
        /// </summary>
        public void CullShadowAll(ID3D12GraphicsCommandList commandList, uint cascadeBufferSrv, ulong cascadeBufferGpuAddress, uint shadowHiZSrv, GPUCuller? culler)
        {
            if (SubBatchCount == 0) return;
            if (culler?.VisibilityShadow4PSO == null || culler?.MainPSO == null || culler?.HistogramPSO == null) return;

            InitializeShadowCullerResources();

            int frameIndex = Engine.FrameIndex % FrameCount;
            var device = Engine.Device;
            int totalInstances = _drawCount;
            
            if (Engine.FrameIndex % 300 == 0)
                Debug.Log("ShadowCull", $"batch={Material?.Name} drawCount={totalInstances} subBatches={SubBatchCount} perInstBufs={_perInstanceBuffers.Count}");

            commandList.SetComputeRootSignature(device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            
            // Root signature requires all CBV slots bound — bind cascade buffer at slot 2 (register b1)
            // Shader reads cascade data via bindless SRV, but root signature still defines this CBV slot
            commandList.SetComputeRootConstantBufferView(2, cascadeBufferGpuAddress);

            // Set push constants
            commandList.SetComputeRoot32BitConstant(0, MeshRegistry.SrvIndex, 0);
            commandList.SetComputeRoot32BitConstant(0, shadowCommandUAVIndices[frameIndex], 1);
            commandList.SetComputeRoot32BitConstant(0, shadowHiZSrv, 2);                                 // ShadowHiZSrvIdx (Indices[0].z)
            commandList.SetComputeRoot32BitConstant(0, 1u, 3);                                           // InstanceMultiplier = 1
            commandList.SetComputeRoot32BitConstant(0, shadowVisibleIndicesUAVIndices[frameIndex], 5);
            commandList.SetComputeRoot32BitConstant(0, shadowCounterUAVIndices[frameIndex], 6);
            commandList.SetComputeRoot32BitConstant(0, (uint)SubBatchCount, 7);
            commandList.SetComputeRoot32BitConstant(0, shadowVisibleIndicesSRVIndices[frameIndex], 9);
            commandList.SetComputeRoot32BitConstant(0, TransformBuffer.Instance?.SrvIndex ?? 0, 10);
            commandList.SetComputeRoot32BitConstant(0, shadowCombinedVisUAVIndices[frameIndex], 11);  // VisibilityFlagsIdx = combined
            commandList.SetComputeRoot32BitConstant(0, (uint)totalInstances, 13);
            commandList.SetComputeRoot32BitConstant(0, shadowCombinedVisUAVIndices[frameIndex], 24);  // CombinedShadowVisIdx
            commandList.SetComputeRoot32BitConstant(0, Material.MaterialsBufferIndex, 25);
            commandList.SetComputeRoot32BitConstant(0, shadowHistogramUAVIndices[frameIndex], 26);
            commandList.SetComputeRoot32BitConstant(0, (uint)MeshRegistry.Count, 27);
            commandList.SetComputeRoot32BitConstant(0, shadowCascadeMaskUAVIndices[frameIndex], 28);  // CascadeMaskUAVIdx
            commandList.SetComputeRoot32BitConstant(0, shadowExpansionUAVIndices[frameIndex], 29);    // ExpansionUAVIdx
            commandList.SetComputeRoot32BitConstant(0, cascadeBufferSrv, 31);                         // CascadeBufferSRVIdx (Indices[7].w)

            foreach (var (hash, pib) in _perInstanceBuffers)
            {
                if (!ComputeSlots.TryGetValue(hash, out int computeSlot)) continue;
                commandList.SetComputeRoot32BitConstant(0, pib.SRVIndices[frameIndex], (uint)computeSlot);
            }

            // Clear union buffers
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(shadowCounterUAVIndices[frameIndex]),
                shadowCounterCPUHandles[frameIndex],
                shadowCounterBuffers[frameIndex],
                new Vortice.Mathematics.Int4(0, 0, 0, 0));
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(shadowVisibleIndicesUAVIndices[frameIndex]),
                shadowVisibleIndicesCPUHandles[frameIndex],
                shadowVisibleIndicesBuffers[frameIndex],
                new Vortice.Mathematics.Int4(unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF)));
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(shadowHistogramUAVIndices[frameIndex]),
                shadowHistogramCPUHandles[frameIndex],
                shadowHistogramBuffers[frameIndex],
                new Vortice.Mathematics.Int4(0, 0, 0, 0));

            commandList.ResourceBarrier(new[] {
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(shadowCounterBuffers[frameIndex])),
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(shadowVisibleIndicesBuffers[frameIndex])),
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(shadowHistogramBuffers[frameIndex]))
            });

            // PHASE 1: Unified visibility (combined vis + cascade mask)
            commandList.SetPipelineState(culler.VisibilityShadow4PSO);
            commandList.Dispatch((uint)((totalInstances + 255) / 256), 1, 1);

            commandList.ResourceBarrier(new[] {
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(shadowCombinedVisBuffers[frameIndex])),
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(shadowCascadeMaskBuffers[frameIndex]))
            });

            // PHASE 2: Single compaction pipeline (runs ONCE, not 4Ã—)
            commandList.SetPipelineState(culler.HistogramPSO);
            commandList.Dispatch((uint)((totalInstances + 63) / 64), 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(shadowHistogramBuffers[frameIndex])));

            commandList.SetPipelineState(culler.HistogramPrefixSumPSO);
            commandList.Dispatch(1, 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(shadowCounterBuffers[frameIndex])));

            commandList.SetPipelineState(culler.GlobalScatterPSO);
            commandList.Dispatch((uint)((totalInstances + 255) / 256), 1, 1);
            commandList.ResourceBarrier(new[] {
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(shadowVisibleIndicesBuffers[frameIndex])),
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(shadowCounterBuffers[frameIndex]))
            });

            // CSMain - generates commands with DrawInstanceCount x 1
            commandList.SetPipelineState(culler.MainPSO);
            commandList.Dispatch((uint)((MeshRegistry.Count + 63) / 64), 1, 1);

            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(shadowCommandBuffers[frameIndex])));

            // PHASE 3: Cascade expansion - expand compacted instances by cascade mask
            // Re-clear counters for atomic append tracking
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(shadowCounterUAVIndices[frameIndex]),
                shadowCounterCPUHandles[frameIndex],
                shadowCounterBuffers[frameIndex],
                new Vortice.Mathematics.Int4(0, 0, 0, 0));
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(shadowCounterBuffers[frameIndex])));

            // Dispatch CSExpandCascades: 2D dispatch (x = instance groups, y = mesh parts)
            uint maxVisiblePerPart = (uint)((totalInstances + 255) / 256);
            commandList.SetPipelineState(culler.ExpandCascadesPSO);
            commandList.Dispatch(maxVisiblePerPart, (uint)MeshRegistry.Count, 1);

            commandList.ResourceBarrier(new[] {
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(shadowExpansionBuffers[frameIndex])),
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(shadowCounterBuffers[frameIndex]))
            });

            // Patch indirect args with expanded counts and updated base offsets
            commandList.SetPipelineState(culler.PatchExpandedPSO);
            commandList.Dispatch((uint)((MeshRegistry.Count + 63) / 64), 1, 1);

            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(shadowCommandBuffers[frameIndex])));
        }









        /// <summary>
        /// Single-pass shadow draw: 1 ExecuteIndirect for all cascades.
        /// Material/PSO/descriptors must already be set by the caller.
        /// Expansion buffer encodes (cascadeIdx, instanceIdx) — no cascade mask needed in VS.
        /// </summary>
        public void DrawShadowSinglePass(ID3D12GraphicsCommandList commandList, uint shadowVPSrv)
        {
            int frameIndex = Engine.FrameIndex % FrameCount;
            if (!_shadowCullerInitialized || SubBatchCount == 0) return;

            var commandBuffer = shadowCommandBuffers[frameIndex];
            if (commandBuffer == null) return;



            // Bind per-instance buffer SRV indices (may use any slots including high ones)
            foreach (var (hash, pib) in _perInstanceBuffers)
            {
                if (pib.PushConstantSlot < 0) continue;
                commandList.SetGraphicsRoot32BitConstant(0, pib.SRVIndices[frameIndex], (uint)pib.PushConstantSlot);
            }

            // Set expansion buffer + VP buffer
            commandList.SetGraphicsRoot32BitConstant(0, shadowExpansionSRVIndices[frameIndex], 20); // ExpansionBufferIdx
            commandList.SetGraphicsRoot32BitConstant(0, shadowVPSrv, 21);    // ShadowVPBufferIdx

            // Transition buffers for reading
            commandList.ResourceBarrier(new[] {
                new ResourceBarrier(new ResourceTransitionBarrier(shadowExpansionBuffers[frameIndex],
                    ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)),
                new ResourceBarrier(new ResourceTransitionBarrier(commandBuffer,
                    ResourceStates.UnorderedAccess, ResourceStates.IndirectArgument))
            });

            // 1 ExecuteIndirect — commands have expanded DrawInstanceCount from CSPatchExpandedCounts
            commandList.ExecuteIndirect(
                Engine.Device.BindlessCommandSignature,
                (uint)SubBatchCount,
                commandBuffer,
                0,
                null,
                0
            );

            // Transition back
            commandList.ResourceBarrier(new[] {
                new ResourceBarrier(new ResourceTransitionBarrier(commandBuffer,
                    ResourceStates.IndirectArgument, ResourceStates.UnorderedAccess)),
                new ResourceBarrier(new ResourceTransitionBarrier(shadowExpansionBuffers[frameIndex],
                    ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess))
            });
        }

        #endregion


        public void Draw(ID3D12GraphicsCommandList commandList, GraphicsDevice device)
        {
            int frameIndex = Engine.FrameIndex % FrameCount;
            if (SubBatchCount == 0) return;
            
            
            var commandBuffer = gpuCommandBuffers[frameIndex];
            if (commandBuffer == null) return;

            // Set topology: tessellation shaders require patch topology
            commandList.IASetPrimitiveTopology(Topology);

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
                DeferDispose(null, visibilityFlagsSRVIndices[i]);
                visibilityFlagsBuffers[i] = null; visibilityFlagsUAVIndices[i] = 0; visibilityFlagsSRVIndices[i] = 0;

                DeferDispose(histogramBuffers[i], histogramUAVIndices[i]);
                histogramBuffers[i] = null; histogramUAVIndices[i] = 0;
            }

            _cpuHeap?.Dispose();
            _cpuHeap = null;
            _cullerInitialized = false;

            // Defer shadow union buffers
            if (_shadowCullerInitialized)
            {
                for (int f = 0; f < FrameCount; f++)
                {
                    DeferDispose(shadowCombinedVisBuffers[f], shadowCombinedVisUAVIndices[f]);
                    shadowCombinedVisBuffers[f] = null; shadowCombinedVisUAVIndices[f] = 0;

                    DeferDispose(shadowCascadeMaskBuffers[f], shadowCascadeMaskUAVIndices[f]);
                    DeferDispose(null, shadowCascadeMaskSRVIndices[f]);
                    shadowCascadeMaskBuffers[f] = null; shadowCascadeMaskUAVIndices[f] = 0; shadowCascadeMaskSRVIndices[f] = 0;

                    DeferDispose(shadowVisibleIndicesBuffers[f], shadowVisibleIndicesUAVIndices[f]);
                    DeferDispose(null, shadowVisibleIndicesSRVIndices[f]);
                    shadowVisibleIndicesBuffers[f] = null; shadowVisibleIndicesUAVIndices[f] = 0; shadowVisibleIndicesSRVIndices[f] = 0;

                    DeferDispose(shadowCommandBuffers[f], shadowCommandUAVIndices[f]);
                    shadowCommandBuffers[f] = null; shadowCommandUAVIndices[f] = 0;

                    DeferDispose(shadowCounterBuffers[f], shadowCounterUAVIndices[f]);
                    shadowCounterBuffers[f] = null; shadowCounterUAVIndices[f] = 0;

                    DeferDispose(shadowHistogramBuffers[f], shadowHistogramUAVIndices[f]);
                    shadowHistogramBuffers[f] = null; shadowHistogramUAVIndices[f] = 0;

                    DeferDispose(shadowExpansionBuffers[f], shadowExpansionUAVIndices[f]);
                    DeferDispose(null, shadowExpansionSRVIndices[f]);
                    shadowExpansionBuffers[f] = null; shadowExpansionUAVIndices[f] = 0; shadowExpansionSRVIndices[f] = 0;
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
            
            // Dispose shadow union buffers
            if (_shadowCullerInitialized)
            {
                for (int f = 0; f < FrameCount; f++)
                {
                    shadowCombinedVisBuffers[f]?.Dispose();
                    if (shadowCombinedVisUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowCombinedVisUAVIndices[f]);

                    shadowCascadeMaskBuffers[f]?.Dispose();
                    if (shadowCascadeMaskUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowCascadeMaskUAVIndices[f]);
                    if (shadowCascadeMaskSRVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowCascadeMaskSRVIndices[f]);

                    shadowVisibleIndicesBuffers[f]?.Dispose();
                    if (shadowVisibleIndicesUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowVisibleIndicesUAVIndices[f]);
                    if (shadowVisibleIndicesSRVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowVisibleIndicesSRVIndices[f]);

                    shadowCommandBuffers[f]?.Dispose();
                    if (shadowCommandUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowCommandUAVIndices[f]);

                    shadowCounterBuffers[f]?.Dispose();
                    if (shadowCounterUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowCounterUAVIndices[f]);

                    shadowHistogramBuffers[f]?.Dispose();
                    if (shadowHistogramUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowHistogramUAVIndices[f]);

                    shadowExpansionBuffers[f]?.Dispose();
                    if (shadowExpansionUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowExpansionUAVIndices[f]);
                    if (shadowExpansionSRVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowExpansionSRVIndices[f]);
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
