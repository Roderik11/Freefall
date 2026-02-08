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
        private bool IsSkinned { get; set; }
        
        // Cached array to avoid per-frame allocation
        private ID3D12DescriptorHeap[]? _cachedSrvHeapArray;

        // Transform slots buffer (indices into global TransformBuffer)
        private ID3D12Resource[] transformSlotBuffers = new ID3D12Resource[FrameCount];
        private uint[] transformSlotSRVIndices = new uint[FrameCount];
        
        // Material ID buffer
        private ID3D12Resource[] materialIdBuffers = new ID3D12Resource[FrameCount];
        private uint[] materialIdSRVIndices = new uint[FrameCount];

        // Bounding spheres for GPU culling
        private ID3D12Resource[] boundingSpheresBuffers = new ID3D12Resource[FrameCount];
        private uint[] boundingSpheresSRVIndices = new uint[FrameCount];

        // Subbatch ID per instance (for GPU to determine command without contiguous ranges)
        private ID3D12Resource[] subbatchIdBuffers = new ID3D12Resource[FrameCount];
        private uint[] subbatchIdSRVIndices = new uint[FrameCount];

        // GPU output: visible instance indices (after culling)
        private ID3D12Resource[] visibleIndicesBuffers = new ID3D12Resource[FrameCount];
        private uint[] visibleIndicesUAVIndices = new uint[FrameCount];
        private uint[] visibleIndicesSRVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] visibleIndicesCPUHandles = new CpuDescriptorHandle[FrameCount];
        
        // GPU output: scattered material IDs (same order as visibleIndices)
        private ID3D12Resource[] scatteredMaterialBuffers = new ID3D12Resource[FrameCount];
        private uint[] scatteredMaterialUAVIndices = new uint[FrameCount];
        private uint[] scatteredMaterialSRVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] scatteredMaterialCPUHandles = new CpuDescriptorHandle[FrameCount];
        
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
        private ID3D12Resource[,] shadowScatteredMaterialBuffers = new ID3D12Resource[FrameCount, ShadowCascadeCount];
        private uint[,] shadowScatteredMaterialUAVIndices = new uint[FrameCount, ShadowCascadeCount];
        private uint[,] shadowScatteredMaterialSRVIndices = new uint[FrameCount, ShadowCascadeCount];
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
        private CpuDescriptorHandle[,] shadowScatteredMaterialCPUHandles = new CpuDescriptorHandle[FrameCount, ShadowCascadeCount];

        // Bone buffers per mesh
        private Dictionary<Mesh, (ID3D12Resource[] Buffers, uint[] Indices, int Capacity)> meshBoneBuffers = new();

        // Contiguous arrays for O(1) block-copy upload (stored in draw-order)
        private uint[] _transformSlots = Array.Empty<uint>();
        private uint[] _materialIds = Array.Empty<uint>();
        private Vector4[] _boundingSpheres = Array.Empty<Vector4>();
        private uint[] _subbatchIds = Array.Empty<uint>();
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
            IsSkinned = false;
            _maxTransformSlot = 0;
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
            
            // Block-copy GPU arrays using Span (slice to actual count)
            var srcTransforms = CollectionsMarshal.AsSpan(bucket.TransformSlots).Slice(0, count);
            var srcMaterials = CollectionsMarshal.AsSpan(bucket.MaterialIds).Slice(0, count);
            var srcSpheres = CollectionsMarshal.AsSpan(bucket.BoundingSpheres).Slice(0, count);
            var srcPartIds = CollectionsMarshal.AsSpan(bucket.MeshPartIds).Slice(0, count);
            
            srcTransforms.CopyTo(_transformSlots.AsSpan(startIdx));
            srcMaterials.CopyTo(_materialIds.AsSpan(startIdx));
            srcSpheres.CopyTo(_boundingSpheres.AsSpan(startIdx));
            srcPartIds.CopyTo(_subbatchIds.AsSpan(startIdx));
            
            // Propagate skinned flag from bucket
            if (bucket.HasSkinnedMesh)
                IsSkinned = true;
            
            // Track max transform slot for bone buffer sizing
            var slots = srcTransforms;
            for (int i = 0; i < slots.Length; i++)
            {
                if ((int)slots[i] > _maxTransformSlot)
                    _maxTransformSlot = (int)slots[i];
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
            Array.Resize(ref _transformSlots, capacity);
            Array.Resize(ref _materialIds, capacity);
            Array.Resize(ref _boundingSpheres, capacity);
            Array.Resize(ref _subbatchIds, capacity);

            int slotSize = capacity * 4;
            int sphereSize = capacity * 16;
            int commandSize = MaxSubBatches * IndirectDrawSizes.IndirectCommandSize;

            for (int i = 0; i < FrameCount; ++i)
            {
                // Input buffers (CPU → GPU upload)
                transformSlotBuffers[i] = Engine.Device.CreateUploadBuffer(slotSize);
                transformSlotSRVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferSRV(transformSlotBuffers[i], (uint)capacity, 4, transformSlotSRVIndices[i]);

                materialIdBuffers[i] = Engine.Device.CreateUploadBuffer(slotSize);
                materialIdSRVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferSRV(materialIdBuffers[i], (uint)capacity, 4, materialIdSRVIndices[i]);

                boundingSpheresBuffers[i] = Engine.Device.CreateUploadBuffer(sphereSize);
                boundingSpheresSRVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferSRV(boundingSpheresBuffers[i], (uint)capacity, 16, boundingSpheresSRVIndices[i]);

                // Subbatch ID per instance (GPU reads to determine command)
                subbatchIdBuffers[i] = Engine.Device.CreateUploadBuffer(slotSize);
                subbatchIdSRVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferSRV(subbatchIdBuffers[i], (uint)capacity, 4, subbatchIdSRVIndices[i]);

                // GPU output: visible indices (compute shader writes)
                visibleIndicesBuffers[i] = Engine.Device.CreateDefaultBuffer(slotSize);
                visibleIndicesUAVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferUAV(visibleIndicesBuffers[i], (uint)capacity, 4, visibleIndicesUAVIndices[i]);
                visibleIndicesSRVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferSRV(visibleIndicesBuffers[i], (uint)capacity, 4, visibleIndicesSRVIndices[i]);
                
                // GPU output: scattered material IDs (compute shader writes, same order as visibleIndices)
                scatteredMaterialBuffers[i] = Engine.Device.CreateDefaultBuffer(slotSize);
                scatteredMaterialUAVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferUAV(scatteredMaterialBuffers[i], (uint)capacity, 4, scatteredMaterialUAVIndices[i]);
                scatteredMaterialSRVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferSRV(scatteredMaterialBuffers[i], (uint)capacity, 4, scatteredMaterialSRVIndices[i]);

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

            unsafe
            {
                byte* pSlots;
                transformSlotBuffers[frameIndex].Map(0, null, (void**)&pSlots);
                fixed (uint* pSrc = _transformSlots)
                    Buffer.MemoryCopy(pSrc, pSlots, _drawCount * 4, _drawCount * 4);
                transformSlotBuffers[frameIndex].Unmap(0);

                byte* pMatId;
                materialIdBuffers[frameIndex].Map(0, null, (void**)&pMatId);
                fixed (uint* pMatSrc = _materialIds)
                    Buffer.MemoryCopy(pMatSrc, pMatId, _drawCount * 4, _drawCount * 4);
                materialIdBuffers[frameIndex].Unmap(0);

                byte* pSpheres;
                boundingSpheresBuffers[frameIndex].Map(0, null, (void**)&pSpheres);
                fixed (Vector4* pSphereSrc = _boundingSpheres)
                    Buffer.MemoryCopy(pSphereSrc, pSpheres, _drawCount * 16, _drawCount * 16);
                boundingSpheresBuffers[frameIndex].Unmap(0);

                byte* pSubbatchIds;
                subbatchIdBuffers[frameIndex].Map(0, null, (void**)&pSubbatchIds);
                fixed (uint* pSubbatchSrc = _subbatchIds)
                    Buffer.MemoryCopy(pSubbatchSrc, pSubbatchIds, _drawCount * 4, _drawCount * 4);
                subbatchIdBuffers[frameIndex].Unmap(0);
            }

            if(IsSkinned)
                UploadBoneBuffers(frameIndex);
        }

        private void UploadBoneBuffers(int frameIndex)
        {
            if (_drawCount == 0) return;

            // All skinned meshes in a batch share the same skeleton
            var mesh = _draws[0].Mesh;
            if (mesh.Bones == null) return;

            int requiredSlots = _maxTransformSlot + 1;

            EnsureBoneBuffer(mesh, requiredSlots);
            var boneData = meshBoneBuffers[mesh];
            var boneBuf = boneData.Buffers[frameIndex];

            // Single Map/Unmap for the whole batch — bones indexed by transform slot
            // This makes bone lookup scatter-safe: the shader uses the transform slot
            // (which is already scattered correctly) to index into the bone buffer
            unsafe
            {
                byte* pBone;
                boneBuf.Map(0, null, (void**)&pBone);
                var boneMatrixSpan = new Span<Matrix4x4>(pBone, boneData.Capacity * mesh.Bones.Length);

                for (int i = 0; i < _drawCount; ++i)
                {
                    var d = _draws[i];
                    var bones = d.Block?.GetValue<Matrix4x4[]>("Bones");
                    if (bones != null)
                    {
                        int baseIdx = d.TransformSlot * mesh.Bones.Length;
                        for (int b = 0; b < bones.Length; ++b)
                            boneMatrixSpan[baseIdx + b] = bones[b];
                    }
                }

                boneBuf.Unmap(0);
            }
        }

        private void EnsureBoneBuffer(Mesh mesh, int requiredSlots)
        {
            if (!meshBoneBuffers.TryGetValue(mesh, out var boneData) || boneData.Capacity < requiredSlots)
            {
                if (boneData.Buffers != null)
                {
                    for (int i = 0; i < FrameCount; ++i)
                    {
                        boneData.Buffers[i]?.Dispose();
                        if (boneData.Indices[i] != 0) Engine.Device.ReleaseBindlessIndex(boneData.Indices[i]);
                    }
                }

                var newBuffers = new ID3D12Resource[FrameCount];
                var newIndices = new uint[FrameCount];
                int boneCount = mesh.Bones!.Length;
                int bufferSize = requiredSlots * boneCount * 64;

                for (int i = 0; i < FrameCount; ++i)
                {
                    newBuffers[i] = Engine.Device.CreateUploadBuffer(bufferSize);
                    newIndices[i] = Engine.Device.AllocateBindlessIndex();
                    Engine.Device.CreateStructuredBufferSRV(newBuffers[i], (uint)(requiredSlots * boneCount), 64, newIndices[i]);
                }

                meshBoneBuffers[mesh] = (newBuffers, newIndices, requiredSlots);
            }
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
                DescriptorCount = FrameCount * 4,
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
                
                var scatteredMaterialCpuHandle = _cpuHeap.GetCPUDescriptorHandleForHeapStart() + ((int)((3 * FrameCount + i) * handleIncrementSize));
                var scatteredMaterialUavDesc = new UnorderedAccessViewDescription
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
                device.NativeDevice.CreateUnorderedAccessView(scatteredMaterialBuffers[i], null, scatteredMaterialUavDesc, scatteredMaterialCpuHandle);
                scatteredMaterialCPUHandles[i] = scatteredMaterialCpuHandle;
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
            commandList.SetComputeRoot32BitConstant(0, boundingSpheresSRVIndices[frameIndex], 2);
            
            commandList.SetComputeRoot32BitConstant(0, transformSlotSRVIndices[frameIndex], 4);
            commandList.SetComputeRoot32BitConstant(0, visibleIndicesUAVIndices[frameIndex], 5);
            commandList.SetComputeRoot32BitConstant(0, counterBufferUAVIndices[frameIndex], 6);
            commandList.SetComputeRoot32BitConstant(0, (uint)SubBatchCount, 7);
            
            commandList.SetComputeRoot32BitConstant(0, visibleIndicesSRVIndices[frameIndex], 9);
            commandList.SetComputeRoot32BitConstant(0, TransformBuffer.Instance?.SrvIndex ?? 0, 10);
            commandList.SetComputeRoot32BitConstant(0, visibilityFlagsUAVIndices[frameIndex], 11);
            commandList.SetComputeRoot32BitConstant(0, (uint)totalInstances, 13);
            commandList.SetComputeRoot32BitConstant(0, subbatchIdSRVIndices[frameIndex], 23);
            commandList.SetComputeRoot32BitConstant(0, materialIdSRVIndices[frameIndex], 24);
            commandList.SetComputeRoot32BitConstant(0, Material.MaterialsBufferIndex, 25);
            commandList.SetComputeRoot32BitConstant(0, histogramUAVIndices[frameIndex], 26);
            commandList.SetComputeRoot32BitConstant(0, (uint)MeshRegistry.Count, 27);
            commandList.SetComputeRoot32BitConstant(0, scatteredMaterialUAVIndices[frameIndex], 28);
            commandList.SetComputeRoot32BitConstant(0, scatteredMaterialSRVIndices[frameIndex], 29);
            
            // Bone buffer for skinned batches (0 for static)
            uint boneBufferIdx = 0;
            if (IsSkinned && _drawCount > 0)
            {
                var firstMesh = _draws[0].Mesh;
                if (meshBoneBuffers.TryGetValue(firstMesh, out var boneData))
                    boneBufferIdx = boneData.Indices[frameIndex];
            }
            commandList.SetComputeRoot32BitConstant(0, boneBufferIdx, 30);

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
            
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(scatteredMaterialUAVIndices[frameIndex]),
                scatteredMaterialCPUHandles[frameIndex],
                scatteredMaterialBuffers[frameIndex],
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
            int totalShadowDescriptors = FrameCount * ShadowCascadeCount * 4;
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

                    // Scattered material IDs
                    shadowScatteredMaterialBuffers[f, c] = device.CreateDefaultBuffer(instanceBufferSize);
                    shadowScatteredMaterialUAVIndices[f, c] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferUAV(shadowScatteredMaterialBuffers[f, c], (uint)capacity, sizeof(uint), shadowScatteredMaterialUAVIndices[f, c]);
                    shadowScatteredMaterialSRVIndices[f, c] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferSRV(shadowScatteredMaterialBuffers[f, c], (uint)capacity, sizeof(uint), shadowScatteredMaterialSRVIndices[f, c]);

                    // CPU handle for scattered material clear
                    var scatteredCpuHandle = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(descriptorIdx * handleIncrement);
                    var scatteredUavDesc = new UnorderedAccessViewDescription
                    {
                        Format = Vortice.DXGI.Format.Unknown,
                        ViewDimension = UnorderedAccessViewDimension.Buffer,
                        Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = (uint)capacity, StructureByteStride = sizeof(uint) }
                    };
                    device.NativeDevice.CreateUnorderedAccessView(shadowScatteredMaterialBuffers[f, c], null, scatteredUavDesc, scatteredCpuHandle);
                    shadowScatteredMaterialCPUHandles[f, c] = scatteredCpuHandle;
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
            commandList.SetComputeRoot32BitConstant(0, boundingSpheresSRVIndices[frameIndex], 2);
            commandList.SetComputeRoot32BitConstant(0, transformSlotSRVIndices[frameIndex], 4);
            commandList.SetComputeRoot32BitConstant(0, shadowVisibleIndicesUAVIndices[frameIndex, cascadeIndex], 5);
            commandList.SetComputeRoot32BitConstant(0, shadowCounterUAVIndices[frameIndex, cascadeIndex], 6);
            commandList.SetComputeRoot32BitConstant(0, (uint)SubBatchCount, 7);
            commandList.SetComputeRoot32BitConstant(0, shadowVisibleIndicesSRVIndices[frameIndex, cascadeIndex], 9);
            commandList.SetComputeRoot32BitConstant(0, TransformBuffer.Instance?.SrvIndex ?? 0, 10);
            commandList.SetComputeRoot32BitConstant(0, shadowVisibilityUAVIndices[frameIndex, cascadeIndex], 11);
            commandList.SetComputeRoot32BitConstant(0, (uint)totalInstances, 13);
            commandList.SetComputeRoot32BitConstant(0, subbatchIdSRVIndices[frameIndex], 23);
            commandList.SetComputeRoot32BitConstant(0, materialIdSRVIndices[frameIndex], 24);
            commandList.SetComputeRoot32BitConstant(0, Material.MaterialsBufferIndex, 25);
            commandList.SetComputeRoot32BitConstant(0, shadowHistogramUAVIndices[frameIndex, cascadeIndex], 26);
            commandList.SetComputeRoot32BitConstant(0, (uint)MeshRegistry.Count, 27);
            commandList.SetComputeRoot32BitConstant(0, shadowScatteredMaterialUAVIndices[frameIndex, cascadeIndex], 28);
            commandList.SetComputeRoot32BitConstant(0, shadowScatteredMaterialSRVIndices[frameIndex, cascadeIndex], 29);
            // Bone buffer for skinned batches (0 for static)
            uint boneBufferIdx = 0;
            if (IsSkinned && _drawCount > 0)
            {
                var firstMesh = _draws[0].Mesh;
                if (meshBoneBuffers.TryGetValue(firstMesh, out var boneData))
                    boneBufferIdx = boneData.Indices[frameIndex];
            }
            commandList.SetComputeRoot32BitConstant(0, boneBufferIdx, 30);
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
            
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(shadowScatteredMaterialUAVIndices[frameIndex, cascadeIndex]),
                shadowScatteredMaterialCPUHandles[frameIndex, cascadeIndex],
                shadowScatteredMaterialBuffers[frameIndex, cascadeIndex],
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

            // Transition buffers for reading
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(shadowVisibleIndicesBuffers[frameIndex, cascadeIndex],
                    ResourceStates.UnorderedAccess,
                    ResourceStates.NonPixelShaderResource)));
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(shadowScatteredMaterialBuffers[frameIndex, cascadeIndex],
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
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(shadowScatteredMaterialBuffers[frameIndex, cascadeIndex],
                    ResourceStates.NonPixelShaderResource,
                    ResourceStates.UnorderedAccess)));
        }

        #endregion

        public void Draw(ID3D12GraphicsCommandList commandList, GraphicsDevice device)
        {
            int frameIndex = Engine.FrameIndex % FrameCount;
            if (SubBatchCount == 0) return;

            var commandBuffer = gpuCommandBuffers[frameIndex];
            if (commandBuffer == null) return;

            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(visibleIndicesBuffers[frameIndex],
                    ResourceStates.UnorderedAccess,
                    ResourceStates.NonPixelShaderResource)));
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(scatteredMaterialBuffers[frameIndex],
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
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(scatteredMaterialBuffers[frameIndex],
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
                DeferDispose(transformSlotBuffers[i], transformSlotSRVIndices[i]);
                transformSlotBuffers[i] = null; transformSlotSRVIndices[i] = 0;

                DeferDispose(materialIdBuffers[i], materialIdSRVIndices[i]);
                materialIdBuffers[i] = null; materialIdSRVIndices[i] = 0;

                DeferDispose(boundingSpheresBuffers[i], boundingSpheresSRVIndices[i]);
                boundingSpheresBuffers[i] = null; boundingSpheresSRVIndices[i] = 0;

                DeferDispose(subbatchIdBuffers[i], subbatchIdSRVIndices[i]);
                subbatchIdBuffers[i] = null; subbatchIdSRVIndices[i] = 0;

                DeferDispose(visibleIndicesBuffers[i], visibleIndicesUAVIndices[i]);
                DeferDispose(null, visibleIndicesSRVIndices[i]);
                visibleIndicesBuffers[i] = null; visibleIndicesUAVIndices[i] = 0; visibleIndicesSRVIndices[i] = 0;

                DeferDispose(scatteredMaterialBuffers[i], scatteredMaterialUAVIndices[i]);
                DeferDispose(null, scatteredMaterialSRVIndices[i]);
                scatteredMaterialBuffers[i] = null; scatteredMaterialUAVIndices[i] = 0; scatteredMaterialSRVIndices[i] = 0;

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

                        DeferDispose(shadowScatteredMaterialBuffers[f, c], shadowScatteredMaterialUAVIndices[f, c]);
                        DeferDispose(null, shadowScatteredMaterialSRVIndices[f, c]);
                        shadowScatteredMaterialBuffers[f, c] = null; shadowScatteredMaterialUAVIndices[f, c] = 0; shadowScatteredMaterialSRVIndices[f, c] = 0;

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
                transformSlotBuffers[i]?.Dispose();
                if (transformSlotSRVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(transformSlotSRVIndices[i]);
                
                materialIdBuffers[i]?.Dispose();
                if (materialIdSRVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(materialIdSRVIndices[i]);
                
                boundingSpheresBuffers[i]?.Dispose();
                if (boundingSpheresSRVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(boundingSpheresSRVIndices[i]);
                
                visibleIndicesBuffers[i]?.Dispose();
                if (visibleIndicesUAVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(visibleIndicesUAVIndices[i]);
                if (visibleIndicesSRVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(visibleIndicesSRVIndices[i]);
                
                gpuCommandBuffers[i]?.Dispose();
                if (gpuCommandUAVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(gpuCommandUAVIndices[i]);
                
                counterBuffers[i]?.Dispose();
                if (counterBufferUAVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(counterBufferUAVIndices[i]);

                transformSlotSRVIndices[i] = 0;
                materialIdSRVIndices[i] = 0;
                boundingSpheresSRVIndices[i] = 0;
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
                        
                        shadowScatteredMaterialBuffers[f, c]?.Dispose();
                        if (shadowScatteredMaterialUAVIndices[f, c] != 0) Engine.Device.ReleaseBindlessIndex(shadowScatteredMaterialUAVIndices[f, c]);
                        if (shadowScatteredMaterialSRVIndices[f, c] != 0) Engine.Device.ReleaseBindlessIndex(shadowScatteredMaterialSRVIndices[f, c]);
                        
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
            foreach (var kvp in meshBoneBuffers)
            {
                foreach (var b in kvp.Value.Buffers) b?.Dispose();
                foreach (var idx in kvp.Value.Indices) Engine.Device.ReleaseBindlessIndex(idx);
            }
            meshBoneBuffers.Clear();
        }
    }
}
