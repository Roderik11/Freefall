using System;
using System.Numerics;
using Vortice.Direct3D12;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>
    /// GPU-driven culler that reads from SceneBuffers.
    /// Replaces InstanceBatch for static/mesh renderers.
    /// Same cull_instances.hlsl shaders, same push constant layout — just reads SceneBuffers SRVs.
    /// </summary>
    public sealed class SceneCuller : IDisposable
    {
        public const int FrameCount = 3;
        private const int MaxSubBatches = 4096;
        private const int ShadowCascadeCount = 4;

        private int _capacity;
        private bool _cullerInitialized;
        private bool _shadowCullerInitialized;

        // GPU output: visible instance indices (after culling)
        private ID3D12Resource[] _visibleIndicesBuffers = new ID3D12Resource[FrameCount];
        private uint[] _visibleIndicesUAVIndices = new uint[FrameCount];
        private uint[] _visibleIndicesSRVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] _visibleIndicesCPUHandles = new CpuDescriptorHandle[FrameCount];

        // GPU output: indirect commands (compute shader writes)
        private ID3D12Resource[] _gpuCommandBuffers = new ID3D12Resource[FrameCount];
        private uint[] _gpuCommandUAVIndices = new uint[FrameCount];

        // Per-subbatch counters (atomic increments during culling)
        private ID3D12Resource[] _counterBuffers = new ID3D12Resource[FrameCount];
        private uint[] _counterBufferUAVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] _counterBufferCPUHandles = new CpuDescriptorHandle[FrameCount];

        // Visibility flags buffer (1=visible, 0=not)
        private ID3D12Resource[] _visibilityFlagsBuffers = new ID3D12Resource[FrameCount];
        private uint[] _visibilityFlagsUAVIndices = new uint[FrameCount];

        // Histogram buffer for GPU-driven per-MeshPartId instance counting
        private ID3D12Resource[] _histogramBuffers = new ID3D12Resource[FrameCount];
        private uint[] _histogramUAVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] _histogramCPUHandles = new CpuDescriptorHandle[FrameCount];

        private ID3D12DescriptorHeap? _cpuHeap;

        // Shadow cascade buffers
        private ID3D12Resource[,] _shadowVisibilityBuffers = new ID3D12Resource[FrameCount, ShadowCascadeCount];
        private uint[,] _shadowVisibilityUAVIndices = new uint[FrameCount, ShadowCascadeCount];
        private ID3D12Resource[,] _shadowVisibleIndicesBuffers = new ID3D12Resource[FrameCount, ShadowCascadeCount];
        private uint[,] _shadowVisibleIndicesUAVIndices = new uint[FrameCount, ShadowCascadeCount];
        private uint[,] _shadowVisibleIndicesSRVIndices = new uint[FrameCount, ShadowCascadeCount];
        private ID3D12Resource[,] _shadowCommandBuffers = new ID3D12Resource[FrameCount, ShadowCascadeCount];
        private uint[,] _shadowCommandUAVIndices = new uint[FrameCount, ShadowCascadeCount];
        private ID3D12Resource[,] _shadowCounterBuffers = new ID3D12Resource[FrameCount, ShadowCascadeCount];
        private uint[,] _shadowCounterUAVIndices = new uint[FrameCount, ShadowCascadeCount];
        private ID3D12Resource[,] _shadowHistogramBuffers = new ID3D12Resource[FrameCount, ShadowCascadeCount];
        private uint[,] _shadowHistogramUAVIndices = new uint[FrameCount, ShadowCascadeCount];
        private CpuDescriptorHandle[,] _shadowCounterCPUHandles = new CpuDescriptorHandle[FrameCount, ShadowCascadeCount];
        private CpuDescriptorHandle[,] _shadowVisibleIndicesCPUHandles = new CpuDescriptorHandle[FrameCount, ShadowCascadeCount];
        private CpuDescriptorHandle[,] _shadowHistogramCPUHandles = new CpuDescriptorHandle[FrameCount, ShadowCascadeCount];
        private ID3D12DescriptorHeap? _shadowCpuHeap;

        // Cached array
        private ID3D12DescriptorHeap[]? _cachedSrvHeapArray;

        public SceneCuller(int initialCapacity = 4096)
        {
            _capacity = initialCapacity;
        }

        private void EnsureCapacity(int needed)
        {
            if (needed <= _capacity) return;
            int newCap = Math.Max(_capacity * 2, needed);
            // TODO: resize GPU buffers with deferred disposal
            _capacity = newCap;
        }

        #region Main Camera Culling

        private void InitializeCullerResources()
        {
            if (_cullerInitialized) return;
            _cullerInitialized = true;

            var device = Engine.Device;
            int counterBufferSize = MaxSubBatches * sizeof(uint);
            int instanceBufferSize = _capacity * sizeof(uint);
            int commandSize = MaxSubBatches * IndirectDrawSizes.IndirectCommandSize;

            var cpuHeapDesc = new DescriptorHeapDescription
            {
                Type = DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView,
                DescriptorCount = FrameCount * 3,
                Flags = DescriptorHeapFlags.None
            };
            _cpuHeap = device.NativeDevice.CreateDescriptorHeap(cpuHeapDesc);
            uint handleInc = device.NativeDevice.GetDescriptorHandleIncrementSize(
                DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView);

            for (int i = 0; i < FrameCount; i++)
            {
                // Visible indices
                _visibleIndicesBuffers[i] = device.CreateDefaultBuffer(instanceBufferSize);
                _visibleIndicesUAVIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(_visibleIndicesBuffers[i], (uint)_capacity, 4, _visibleIndicesUAVIndices[i]);
                _visibleIndicesSRVIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferSRV(_visibleIndicesBuffers[i], (uint)_capacity, 4, _visibleIndicesSRVIndices[i]);

                var viCpu = _cpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)((FrameCount + i) * handleInc);
                var viUav = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView { NumElements = (uint)_capacity, StructureByteStride = sizeof(uint) }
                };
                device.NativeDevice.CreateUnorderedAccessView(_visibleIndicesBuffers[i], null, viUav, viCpu);
                _visibleIndicesCPUHandles[i] = viCpu;

                // Commands
                _gpuCommandBuffers[i] = device.CreateDefaultBuffer(commandSize);
                _gpuCommandUAVIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(_gpuCommandBuffers[i], MaxSubBatches,
                    (uint)IndirectDrawSizes.IndirectCommandSize, _gpuCommandUAVIndices[i]);

                // Counters
                _counterBuffers[i] = device.CreateDefaultBuffer(counterBufferSize);
                _counterBufferUAVIndices[i] = device.AllocateBindlessIndex();
                var ctrUav = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView { NumElements = MaxSubBatches, StructureByteStride = sizeof(uint) }
                };
                device.NativeDevice.CreateUnorderedAccessView(
                    _counterBuffers[i], null, ctrUav, device.GetCpuHandle(_counterBufferUAVIndices[i]));
                var ctrCpu = _cpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(i * handleInc);
                device.NativeDevice.CreateUnorderedAccessView(_counterBuffers[i], null, ctrUav, ctrCpu);
                _counterBufferCPUHandles[i] = ctrCpu;

                // Visibility flags
                _visibilityFlagsBuffers[i] = device.CreateDefaultBuffer(instanceBufferSize);
                _visibilityFlagsUAVIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(_visibilityFlagsBuffers[i], (uint)_capacity, sizeof(uint),
                    _visibilityFlagsUAVIndices[i]);

                // Histogram
                int histogramSize = MeshRegistry.MaxMeshParts * sizeof(uint);
                _histogramBuffers[i] = device.CreateDefaultBuffer(histogramSize);
                _histogramUAVIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(_histogramBuffers[i], (uint)MeshRegistry.MaxMeshParts, sizeof(uint),
                    _histogramUAVIndices[i]);

                var histCpu = _cpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)((2 * FrameCount + i) * handleInc);
                var histUav = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView { NumElements = (uint)MeshRegistry.MaxMeshParts, StructureByteStride = sizeof(uint) }
                };
                device.NativeDevice.CreateUnorderedAccessView(_histogramBuffers[i], null, histUav, histCpu);
                _histogramCPUHandles[i] = histCpu;
            }
        }

        /// <summary>
        /// 5-pass GPU frustum culling over all SceneBuffers instances.
        /// </summary>
        public void Cull(ID3D12GraphicsCommandList commandList, ulong frustumBufferGPUAddress, GPUCuller? culler)
        {
            int totalInstances = SceneBuffers.ActiveSlotCount;
            int subBatchCount = MeshRegistry.Count;
            if (totalInstances == 0 || subBatchCount == 0) return;
            if (culler?.VisibilityPSO == null || culler?.MainPSO == null || culler?.HistogramPSO == null) return;

            InitializeCullerResources();

            int f = Engine.FrameIndex % FrameCount;
            var device = Engine.Device;

            commandList.SetComputeRootSignature(device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            commandList.SetComputeRootConstantBufferView(1, frustumBufferGPUAddress);

            // Push constants — same layout as InstanceBatch, but SRVs point to SceneBuffers
            commandList.SetComputeRoot32BitConstant(0, MeshRegistry.SrvIndex, 0);
            commandList.SetComputeRoot32BitConstant(0, _gpuCommandUAVIndices[f], 1);
            commandList.SetComputeRoot32BitConstant(0, SceneBuffers.BoundingSphereSrvIndex, 2);   // was per-instance buffer
            commandList.SetComputeRoot32BitConstant(0, SceneBuffers.DescriptorSrvIndex, 4);        // was per-instance buffer
            commandList.SetComputeRoot32BitConstant(0, _visibleIndicesUAVIndices[f], 5);
            commandList.SetComputeRoot32BitConstant(0, _counterBufferUAVIndices[f], 6);
            commandList.SetComputeRoot32BitConstant(0, (uint)subBatchCount, 7);
            commandList.SetComputeRoot32BitConstant(0, _visibleIndicesSRVIndices[f], 9);
            commandList.SetComputeRoot32BitConstant(0, SceneBuffers.TransformSrvIndex, 10);        // was TransformBuffer.Instance.SrvIndex
            commandList.SetComputeRoot32BitConstant(0, _visibilityFlagsUAVIndices[f], 11);
            commandList.SetComputeRoot32BitConstant(0, (uint)totalInstances, 13);
            commandList.SetComputeRoot32BitConstant(0, SceneBuffers.MeshPartIdSrvIndex, 23);       // was per-instance buffer
            commandList.SetComputeRoot32BitConstant(0, Material.MaterialsBufferIndex, 25);
            commandList.SetComputeRoot32BitConstant(0, _histogramUAVIndices[f], 26);
            commandList.SetComputeRoot32BitConstant(0, (uint)MeshRegistry.Count, 27);

            // Clear UAVs
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(_counterBufferUAVIndices[f]),
                _counterBufferCPUHandles[f],
                _counterBuffers[f],
                new Int4(0, 0, 0, 0));
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(_visibleIndicesUAVIndices[f]),
                _visibleIndicesCPUHandles[f],
                _visibleIndicesBuffers[f],
                new Int4(unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF)));
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(_histogramUAVIndices[f]),
                _histogramCPUHandles[f],
                _histogramBuffers[f],
                new Int4(0, 0, 0, 0));

            commandList.ResourceBarrier(new[] {
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_counterBuffers[f])),
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_visibleIndicesBuffers[f])),
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_histogramBuffers[f]))
            });

            // Pass 1: Visibility
            commandList.SetPipelineState(culler.VisibilityPSO);
            commandList.Dispatch((uint)((totalInstances + 255) / 256), 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_visibilityFlagsBuffers[f])));

            // Pass 2: Histogram
            commandList.SetPipelineState(culler.HistogramPSO);
            commandList.Dispatch((uint)((totalInstances + 63) / 64), 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_histogramBuffers[f])));

            // Pass 3: PrefixSum
            commandList.SetPipelineState(culler.HistogramPrefixSumPSO);
            commandList.Dispatch(1, 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_counterBuffers[f])));

            // Pass 4: Scatter
            commandList.SetPipelineState(culler.GlobalScatterPSO);
            commandList.Dispatch((uint)((totalInstances + 255) / 256), 1, 1);
            commandList.ResourceBarrier(new[] {
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_visibleIndicesBuffers[f])),
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_counterBuffers[f]))
            });

            // Pass 5: Generate commands
            commandList.SetPipelineState(culler.MainPSO);
            commandList.Dispatch((uint)((MeshRegistry.Count + 63) / 64), 1, 1);
        }

        /// <summary>
        /// Execute indirect draw commands generated by Cull().
        /// Material must be applied by caller before this.
        /// </summary>
        public void Draw(ID3D12GraphicsCommandList commandList)
        {
            int f = Engine.FrameIndex % FrameCount;
            int subBatchCount = MeshRegistry.Count;
            if (!_cullerInitialized || subBatchCount == 0) return;

            var commandBuffer = _gpuCommandBuffers[f];
            if (commandBuffer == null) return;

            // Transition for reading
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_visibleIndicesBuffers[f],
                    ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)));
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(commandBuffer,
                    ResourceStates.UnorderedAccess, ResourceStates.IndirectArgument)));

            commandList.ExecuteIndirect(
                Engine.Device.BindlessCommandSignature,
                (uint)subBatchCount,
                commandBuffer, 0, null, 0);

            // Transition back
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(commandBuffer,
                    ResourceStates.IndirectArgument, ResourceStates.UnorderedAccess)));
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_visibleIndicesBuffers[f],
                    ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess)));
        }

        #endregion

        #region Shadow Culling

        private void InitializeShadowCullerResources()
        {
            if (_shadowCullerInitialized) return;
            _shadowCullerInitialized = true;

            var device = Engine.Device;
            int counterBufSize = MaxSubBatches * sizeof(uint);
            int instanceBufSize = _capacity * sizeof(uint);
            int commandSize = MaxSubBatches * IndirectDrawSizes.IndirectCommandSize;
            int histogramSize = MeshRegistry.MaxMeshParts * sizeof(uint);

            int totalDescs = FrameCount * ShadowCascadeCount * 3;
            var cpuHeapDesc = new DescriptorHeapDescription
            {
                Type = DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView,
                DescriptorCount = (uint)totalDescs,
                Flags = DescriptorHeapFlags.None
            };
            _shadowCpuHeap = device.NativeDevice.CreateDescriptorHeap(cpuHeapDesc);
            uint hi = device.NativeDevice.GetDescriptorHandleIncrementSize(
                DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView);
            int idx = 0;

            for (int f = 0; f < FrameCount; f++)
            {
                for (int c = 0; c < ShadowCascadeCount; c++)
                {
                    // Visibility flags
                    _shadowVisibilityBuffers[f, c] = device.CreateDefaultBuffer(instanceBufSize);
                    _shadowVisibilityUAVIndices[f, c] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferUAV(_shadowVisibilityBuffers[f, c], (uint)_capacity, sizeof(uint),
                        _shadowVisibilityUAVIndices[f, c]);

                    // Visible indices
                    _shadowVisibleIndicesBuffers[f, c] = device.CreateDefaultBuffer(instanceBufSize);
                    _shadowVisibleIndicesUAVIndices[f, c] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferUAV(_shadowVisibleIndicesBuffers[f, c], (uint)_capacity, sizeof(uint),
                        _shadowVisibleIndicesUAVIndices[f, c]);
                    _shadowVisibleIndicesSRVIndices[f, c] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferSRV(_shadowVisibleIndicesBuffers[f, c], (uint)_capacity, sizeof(uint),
                        _shadowVisibleIndicesSRVIndices[f, c]);

                    var viCpu = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(idx * hi);
                    var viUav = new UnorderedAccessViewDescription
                    {
                        Format = Vortice.DXGI.Format.Unknown,
                        ViewDimension = UnorderedAccessViewDimension.Buffer,
                        Buffer = new BufferUnorderedAccessView { NumElements = (uint)_capacity, StructureByteStride = sizeof(uint) }
                    };
                    device.NativeDevice.CreateUnorderedAccessView(_shadowVisibleIndicesBuffers[f, c], null, viUav, viCpu);
                    _shadowVisibleIndicesCPUHandles[f, c] = viCpu;
                    idx++;

                    // Commands
                    _shadowCommandBuffers[f, c] = device.CreateDefaultBuffer(commandSize);
                    _shadowCommandUAVIndices[f, c] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferUAV(_shadowCommandBuffers[f, c], MaxSubBatches,
                        (uint)IndirectDrawSizes.IndirectCommandSize, _shadowCommandUAVIndices[f, c]);

                    // Counters
                    _shadowCounterBuffers[f, c] = device.CreateDefaultBuffer(counterBufSize);
                    _shadowCounterUAVIndices[f, c] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferUAV(_shadowCounterBuffers[f, c], MaxSubBatches, sizeof(uint),
                        _shadowCounterUAVIndices[f, c]);

                    var cCpu = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(idx * hi);
                    var cUav = new UnorderedAccessViewDescription
                    {
                        Format = Vortice.DXGI.Format.Unknown,
                        ViewDimension = UnorderedAccessViewDimension.Buffer,
                        Buffer = new BufferUnorderedAccessView { NumElements = MaxSubBatches, StructureByteStride = sizeof(uint) }
                    };
                    device.NativeDevice.CreateUnorderedAccessView(_shadowCounterBuffers[f, c], null, cUav, cCpu);
                    _shadowCounterCPUHandles[f, c] = cCpu;
                    idx++;

                    // Histogram
                    _shadowHistogramBuffers[f, c] = device.CreateDefaultBuffer(histogramSize);
                    _shadowHistogramUAVIndices[f, c] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferUAV(_shadowHistogramBuffers[f, c], (uint)MeshRegistry.MaxMeshParts, sizeof(uint),
                        _shadowHistogramUAVIndices[f, c]);

                    var hCpu = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(idx * hi);
                    var hUav = new UnorderedAccessViewDescription
                    {
                        Format = Vortice.DXGI.Format.Unknown,
                        ViewDimension = UnorderedAccessViewDimension.Buffer,
                        Buffer = new BufferUnorderedAccessView { NumElements = (uint)MeshRegistry.MaxMeshParts, StructureByteStride = sizeof(uint) }
                    };
                    device.NativeDevice.CreateUnorderedAccessView(_shadowHistogramBuffers[f, c], null, hUav, hCpu);
                    _shadowHistogramCPUHandles[f, c] = hCpu;
                    idx++;
                }
            }
        }

        /// <summary>
        /// Cull all 4 shadow cascades with unified visibility pass.
        /// </summary>
        public void CullShadowAll(ID3D12GraphicsCommandList commandList, ulong shadowCascadeBufferAddress, GPUCuller? culler)
        {
            int totalInstances = SceneBuffers.ActiveSlotCount;
            int subBatchCount = MeshRegistry.Count;
            if (totalInstances == 0 || subBatchCount == 0) return;
            if (culler?.VisibilityShadow4PSO == null || culler?.MainPSO == null || culler?.HistogramPSO == null) return;

            InitializeShadowCullerResources();

            int f = Engine.FrameIndex % FrameCount;
            var device = Engine.Device;

            // Set shared compute state
            commandList.SetComputeRootSignature(device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            commandList.SetComputeRootConstantBufferView(2, shadowCascadeBufferAddress);

            // Push constants shared across all passes — SceneBuffers SRVs
            commandList.SetComputeRoot32BitConstant(0, MeshRegistry.SrvIndex, 0);
            commandList.SetComputeRoot32BitConstant(0, SceneBuffers.BoundingSphereSrvIndex, 2);
            commandList.SetComputeRoot32BitConstant(0, SceneBuffers.DescriptorSrvIndex, 4);
            commandList.SetComputeRoot32BitConstant(0, (uint)subBatchCount, 7);
            commandList.SetComputeRoot32BitConstant(0, SceneBuffers.TransformSrvIndex, 10);
            commandList.SetComputeRoot32BitConstant(0, (uint)totalInstances, 13);
            commandList.SetComputeRoot32BitConstant(0, SceneBuffers.MeshPartIdSrvIndex, 23);
            commandList.SetComputeRoot32BitConstant(0, Material.MaterialsBufferIndex, 25);
            commandList.SetComputeRoot32BitConstant(0, (uint)MeshRegistry.Count, 27);

            // Phase 1: Unified visibility (all 4 cascades)
            commandList.SetComputeRoot32BitConstant(0, _shadowVisibilityUAVIndices[f, 0], 24);
            commandList.SetComputeRoot32BitConstant(0, _shadowVisibilityUAVIndices[f, 1], 28);
            commandList.SetComputeRoot32BitConstant(0, _shadowVisibilityUAVIndices[f, 2], 29);
            commandList.SetComputeRoot32BitConstant(0, _shadowVisibilityUAVIndices[f, 3], 31);

            commandList.SetPipelineState(culler.VisibilityShadow4PSO);
            commandList.Dispatch((uint)((totalInstances + 255) / 256), 1, 1);

            commandList.ResourceBarrier(new[] {
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_shadowVisibilityBuffers[f, 0])),
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_shadowVisibilityBuffers[f, 1])),
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_shadowVisibilityBuffers[f, 2])),
                new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_shadowVisibilityBuffers[f, 3]))
            });

            // Phase 2: Per-cascade Histogram → PrefixSum → Scatter → Main
            for (int c = 0; c < ShadowCascadeCount; c++)
            {
                // Clear per-cascade buffers
                commandList.ClearUnorderedAccessViewUint(
                    device.GetGpuHandle(_shadowCounterUAVIndices[f, c]),
                    _shadowCounterCPUHandles[f, c],
                    _shadowCounterBuffers[f, c],
                    new Int4(0, 0, 0, 0));
                commandList.ClearUnorderedAccessViewUint(
                    device.GetGpuHandle(_shadowVisibleIndicesUAVIndices[f, c]),
                    _shadowVisibleIndicesCPUHandles[f, c],
                    _shadowVisibleIndicesBuffers[f, c],
                    new Int4(unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF)));
                commandList.ClearUnorderedAccessViewUint(
                    device.GetGpuHandle(_shadowHistogramUAVIndices[f, c]),
                    _shadowHistogramCPUHandles[f, c],
                    _shadowHistogramBuffers[f, c],
                    new Int4(0, 0, 0, 0));

                commandList.ResourceBarrier(new[] {
                    new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_shadowCounterBuffers[f, c])),
                    new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_shadowVisibleIndicesBuffers[f, c])),
                    new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_shadowHistogramBuffers[f, c]))
                });

                // Rebind per-cascade buffer indices
                commandList.SetComputeRoot32BitConstant(0, _shadowCommandUAVIndices[f, c], 1);
                commandList.SetComputeRoot32BitConstant(0, _shadowVisibleIndicesUAVIndices[f, c], 5);
                commandList.SetComputeRoot32BitConstant(0, _shadowCounterUAVIndices[f, c], 6);
                commandList.SetComputeRoot32BitConstant(0, _shadowVisibleIndicesSRVIndices[f, c], 9);
                commandList.SetComputeRoot32BitConstant(0, _shadowVisibilityUAVIndices[f, c], 11);
                commandList.SetComputeRoot32BitConstant(0, _shadowHistogramUAVIndices[f, c], 26);

                // Histogram
                commandList.SetPipelineState(culler.HistogramPSO);
                commandList.Dispatch((uint)((totalInstances + 63) / 64), 1, 1);
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_shadowHistogramBuffers[f, c])));

                // PrefixSum
                commandList.SetPipelineState(culler.HistogramPrefixSumPSO);
                commandList.Dispatch(1, 1, 1);
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_shadowCounterBuffers[f, c])));

                // Scatter
                commandList.SetPipelineState(culler.GlobalScatterPSO);
                commandList.Dispatch((uint)((totalInstances + 255) / 256), 1, 1);
                commandList.ResourceBarrier(new[] {
                    new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_shadowVisibleIndicesBuffers[f, c])),
                    new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_shadowCounterBuffers[f, c]))
                });

                // Main (command generation)
                commandList.SetPipelineState(culler.MainPSO);
                commandList.Dispatch((uint)((MeshRegistry.Count + 63) / 64), 1, 1);
            }
        }

        /// <summary>
        /// Execute indirect draw for a shadow cascade.
        /// Material must be applied by caller before this.
        /// </summary>
        public void DrawShadow(ID3D12GraphicsCommandList commandList, int cascadeIndex, ulong shadowCBVAddress)
        {
            int f = Engine.FrameIndex % FrameCount;
            int subBatchCount = MeshRegistry.Count;
            if (!_shadowCullerInitialized || subBatchCount == 0 || cascadeIndex < 0 || cascadeIndex >= ShadowCascadeCount)
                return;

            var commandBuffer = _shadowCommandBuffers[f, cascadeIndex];
            if (commandBuffer == null) return;

            // Transition for reading
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_shadowVisibleIndicesBuffers[f, cascadeIndex],
                    ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)));
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(commandBuffer,
                    ResourceStates.UnorderedAccess, ResourceStates.IndirectArgument)));

            commandList.ExecuteIndirect(
                Engine.Device.BindlessCommandSignature,
                (uint)subBatchCount,
                commandBuffer, 0, null, 0);

            // Transition back
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(commandBuffer,
                    ResourceStates.IndirectArgument, ResourceStates.UnorderedAccess)));
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_shadowVisibleIndicesBuffers[f, cascadeIndex],
                    ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess)));
        }

        #endregion

        public void Dispose()
        {
            // Main culler
            for (int i = 0; i < FrameCount; i++)
            {
                _visibleIndicesBuffers[i]?.Dispose();
                _gpuCommandBuffers[i]?.Dispose();
                _counterBuffers[i]?.Dispose();
                _visibilityFlagsBuffers[i]?.Dispose();
                _histogramBuffers[i]?.Dispose();
            }
            _cpuHeap?.Dispose();

            // Shadow culler
            for (int f = 0; f < FrameCount; f++)
            {
                for (int c = 0; c < ShadowCascadeCount; c++)
                {
                    _shadowVisibilityBuffers[f, c]?.Dispose();
                    _shadowVisibleIndicesBuffers[f, c]?.Dispose();
                    _shadowCommandBuffers[f, c]?.Dispose();
                    _shadowCounterBuffers[f, c]?.Dispose();
                    _shadowHistogramBuffers[f, c]?.Dispose();
                }
            }
            _shadowCpuHeap?.Dispose();
        }
    }
}
