using System;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// GPU-based frustum culling using compute shaders.
    /// Two-pass dispatch:
    /// 1. CSCount - per-instance frustum test, increments per-sub-batch counters
    /// 2. CSMain - copies templates with culled instance counts
    /// </summary>
    public class GPUCuller : IDisposable
    {
        private const int FrameCount = 3;
        private const int MaxSubBatches = 4096;
        
        // Compute PSOs for GPU-driven rendering pipeline
        private ID3D12PipelineState? _clearPSO;         // CSClear - counter/histogram initialization
        private ID3D12PipelineState? _visibilityPSO;    // CSVisibility - write visibility flags
        private ID3D12PipelineState? _histogramPSO;     // CSHistogram - count per MeshPartId
        private ID3D12PipelineState? _histogramPrefixSumPSO;  // CSHistogramPrefixSum - prefix sum
        private ID3D12PipelineState? _localScanPSO;     // CSLocalScan - local prefix sum per block
        private ID3D12PipelineState? _blockScanPSO;     // CSBlockScan - scan block sums
        private ID3D12PipelineState? _globalScatterPSO; // CSGlobalScatter - add prefix, scatter
        private ID3D12PipelineState? _bitonicSortPSO;   // CSBitonicSort - sort for determinism
        private ID3D12PipelineState? _sortIndirectionPSO; // CSSortIndirection - per-subbatch indirection sort
        private ID3D12PipelineState? _mainPSO;          // CSMain - generate commands from histogram
        private ID3D12PipelineState? _visibilityShadowPSO; // CSVisibilityShadow - shadow cascade culling
        private ID3D12PipelineState? _downsamplePSO;    // CSDownsample - Hi-Z depth pyramid generation
        
        // SDSM depth analysis PSOs
        private ID3D12PipelineState? _depthReducePSO;    // CSDepthReduce - min/max depth reduction
        private ID3D12PipelineState? _depthHistogramPSO; // CSDepthHistogram - depth histogram
        private ID3D12PipelineState? _computeSplitsPSO;  // CSComputeSplits - percentile split computation
        
        // Frustum constants buffer per frame (just planes)
        private ID3D12Resource[] _frustumConstantsBuffers = new ID3D12Resource[FrameCount];
        
        // Atomic counter buffer (one uint per sub-batch)
        private ID3D12Resource[] _counterBuffers = new ID3D12Resource[FrameCount];
        private uint[] _counterBufferUAVs = new uint[FrameCount];
        private CpuDescriptorHandle[] _counterBufferCPUHandles = new CpuDescriptorHandle[FrameCount];
        
        // Non-shader-visible heap for ClearUAV CPU handles (D3D12 requirement)
        private ID3D12DescriptorHeap? _clearUAVHeap;
        
        // Instance range buffer (StartInstance, InstanceCount per sub-batch)
        private ID3D12Resource[] _rangeBuffers = new ID3D12Resource[FrameCount];
        private uint[] _rangeBufferSRVs = new uint[FrameCount];
        
        // Shadow cascade frustum planes buffer (6 planes * 4 cascades = 24 float4s)
        private ID3D12Resource[] _shadowCascadeBuffers = new ID3D12Resource[FrameCount];
        
        // SDSM depth analysis buffers
        private ID3D12Resource? _depthMinMaxBuffer;           // 2 uints (min, max as float bits), GPU default
        private uint _depthMinMaxUAV;
        private ID3D12Resource? _depthHistogramBuffer;        // 256 uints, GPU default
        private uint _depthHistogramUAV;
        private ID3D12Resource? _depthSplitsBuffer;           // 4 floats, GPU default
        private uint _depthSplitsUAV;
        private ID3D12Resource[]? _depthSplitsReadbackBuffers; // Per-frame readback (HeapType.Readback)
        private IntPtr[]? _depthSplitsReadbackPtrs;           // Persistently mapped pointers
        private bool _sdsmInitialized;
        private int _sdsmValidFrames;                         // Frames since SDSM started producing data
        
        private GraphicsDevice _device;
        private bool _initialized;
        
        // Cull stats readback (2 uints: [0]=visible, [1]=hi-z occluded)
        private ID3D12Resource? _cullStatsBuffer;           // GPU default heap
        private uint _cullStatsUAV;
        private ID3D12Resource[]? _cullStatsReadbackBuffers; // Per-frame readback
        private IntPtr[]? _cullStatsReadbackPtrs;            // Persistently mapped
        private CpuDescriptorHandle _cullStatsClearCPU;      // For ClearUAV
        
        /// <summary>Number of instances that passed both frustum and Hi-Z tests (1 frame behind).</summary>
        public int LastVisibleCount { get; private set; }
        /// <summary>Number of instances that passed frustum but were occluded by Hi-Z (1 frame behind).</summary>
        public int LastHiZOccludedCount { get; private set; }
        
        // Cached array to avoid per-frame allocation
        private ID3D12DescriptorHeap[]? _cachedSrvHeapArray;
        
        // Public PSO access for GPU-driven culling pipeline
        public ID3D12PipelineState? VisibilityPSO => _visibilityPSO;
        public ID3D12PipelineState? HistogramPSO => _histogramPSO;
        public ID3D12PipelineState? HistogramPrefixSumPSO => _histogramPrefixSumPSO;
        public ID3D12PipelineState? LocalScanPSO => _localScanPSO;
        public ID3D12PipelineState? BlockScanPSO => _blockScanPSO;
        public ID3D12PipelineState? GlobalScatterPSO => _globalScatterPSO;
        public ID3D12PipelineState? BitonicSortPSO => _bitonicSortPSO;
        public ID3D12PipelineState? SortIndirectionPSO => _sortIndirectionPSO;
        public ID3D12PipelineState? MainPSO => _mainPSO;
        public ID3D12PipelineState? ClearPSO => _clearPSO;
        public ID3D12PipelineState? VisibilityShadowPSO => _visibilityShadowPSO;
        public ID3D12PipelineState? DownsamplePSO => _downsamplePSO;
        
        // Hi-Z depth pyramid resources
        private ID3D12Resource? _hiZPyramid;
        public ID3D12Resource HiZTexture => _hiZPyramid;
        private uint[] _hiZMipUAVs = Array.Empty<uint>();     // UAV per mip level
        private CpuDescriptorHandle[] _hiZMipUAVCPU = Array.Empty<CpuDescriptorHandle>(); // CPU handles for UAV clears
        private uint[] _hiZMipSRVs = Array.Empty<uint>();     // SRV per mip level (for reading previous mip)
        private uint _hiZPyramidSRV;                          // Full-pyramid SRV (all mips)
        private int _hiZMipCount;
        private int _hiZWidth;
        private int _hiZHeight;
        private bool _hiZReady;  // True after pyramid has been generated at least once
        
        /// <summary>Bindless SRV index of the full Hi-Z pyramid (all mips). 0 if not created.</summary>
        public uint HiZPyramidSRV => _hiZPyramidSRV;
        public int HiZWidth => _hiZWidth;
        public int HiZHeight => _hiZHeight;
        public int HiZMipCount => _hiZMipCount;
        /// <summary>True after the pyramid has been generated at least once (safe to use for occlusion).</summary>
        public bool HiZReady => _hiZReady;
        
        /// <summary>True after SDSM buffers are created and ready for dispatch.</summary>
        public bool SdsmReady => _sdsmInitialized;

        /// <summary>
        /// Frustum planes uploaded to the compute shader.
        /// Must match cbuffer CullParams in cull_instances.hlsl.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct FrustumConstants
        {
            public Vector4 Plane0;
            public Vector4 Plane1;
            public Vector4 Plane2;
            public Vector4 Plane3;
            public Vector4 Plane4;
            public Vector4 Plane5;
            // Hi-Z occlusion culling parameters
            public Matrix4x4 OcclusionProjection; // Projection matrix for sphere-to-screen
            public uint HiZSrvIdx;       // Bindless SRV index of Hi-Z pyramid (0 = disabled)
            public float HiZWidth;       // Mip 0 width
            public float HiZHeight;      // Mip 0 height
            public uint HiZMipCount;     // Number of mip levels in pyramid
            public float NearPlane;      // Camera near plane for depth margin calculation
            public uint CullStatsUAVIdx; // UAV index for cull stats (0 = disabled)
            public uint DebugMode;       // Debug visualization mode (4 = x-ray occlusion)
            public float _pad1;
        }
        
        /// <summary>
        /// Per-sub-batch instance range.
        /// Must match InstanceRange in cull_instances.hlsl.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct InstanceRange
        {
            public uint StartInstance;
            public uint InstanceCount;
            public uint MeshPartId;       // Index into MeshRegistry
            public uint Reserved;         // Padding for 16-byte alignment
        }
        
        /// <summary>
        /// Shadow cascade frustum planes (6 planes * 4 cascades = 24 float4s).
        /// Must match cbuffer ShadowCascadePlanes in cull_instances.hlsl.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public unsafe struct ShadowCascadeConstants
        {
            public fixed float Planes[24 * 4]; // 24 float4s = 24 * 4 floats
            
            public void SetCascadePlanes(int cascade, Vector4[] planes)
            {
                if (cascade < 0 || cascade > 3 || planes.Length < 6) return;
                int offset = cascade * 6 * 4;
                for (int i = 0; i < 6; i++)
                {
                    fixed (float* p = Planes)
                    {
                        p[offset + i * 4 + 0] = planes[i].X;
                        p[offset + i * 4 + 1] = planes[i].Y;
                        p[offset + i * 4 + 2] = planes[i].Z;
                        p[offset + i * 4 + 3] = planes[i].W;
                    }
                }
            }
        }

        public GPUCuller(GraphicsDevice device)
        {
            _device = device;
        }

        /// <summary>
        /// Initialize the GPU culler by compiling the compute shader and creating buffers.
        /// Called lazily on first use.
        /// </summary>
        public void Initialize()
        {
            if (_initialized) return;
            
            try
            {
                // Load and compile compute shader
                string shaderPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Resources", "Shaders", "cull_instances.hlsl");
                if (!File.Exists(shaderPath))
                {
                    Debug.LogError("GPUCuller", $"Compute shader not found: {shaderPath}");
                    return;
                }
                
                string shaderSource = File.ReadAllText(shaderPath);
                
                // Compile all GPU-driven rendering pipeline passes
                
                // Pass 0: CSClear - zero counter buffer and histogram
                var clearShader = new Shader(shaderSource, "CSClear", "cs_6_6");
                _clearPSO = _device.CreateComputePipelineState(clearShader.Bytecode);
                clearShader.Dispose();
                
                // Pass 1: CSVisibility - write visibility flags
                var visibilityShader = new Shader(shaderSource, "CSVisibility", "cs_6_6");
                _visibilityPSO = _device.CreateComputePipelineState(visibilityShader.Bytecode);
                visibilityShader.Dispose();
                
                // Pass 1b: CSHistogram - count visible instances per MeshPartId
                var histogramShader = new Shader(shaderSource, "CSHistogram", "cs_6_6");
                _histogramPSO = _device.CreateComputePipelineState(histogramShader.Bytecode);
                histogramShader.Dispose();
                
                // Pass 2: CSLocalScan - local prefix sum per block, output block sums
                var localScanShader = new Shader(shaderSource, "CSLocalScan", "cs_6_6");
                _localScanPSO = _device.CreateComputePipelineState(localScanShader.Bytecode);
                localScanShader.Dispose();
                
                // Pass 3: CSBlockScan - scan the block sums
                var blockScanShader = new Shader(shaderSource, "CSBlockScan", "cs_6_6");
                _blockScanPSO = _device.CreateComputePipelineState(blockScanShader.Bytecode);
                blockScanShader.Dispose();
                
                // Pass 4: CSGlobalScatter - add block prefix and scatter to output
                var globalScatterShader = new Shader(shaderSource, "CSGlobalScatter", "cs_6_6");
                _globalScatterPSO = _device.CreateComputePipelineState(globalScatterShader.Bytecode);
                globalScatterShader.Dispose();
                
                // Pass 5: CSBitonicSort - sort for deterministic ordering
                var bitonicSortShader = new Shader(shaderSource, "CSBitonicSort", "cs_6_6");
                _bitonicSortPSO = _device.CreateComputePipelineState(bitonicSortShader.Bytecode);
                bitonicSortShader.Dispose();
                
                // Pass 5b: CSSortIndirection - per-subbatch indirection sort
                var sortIndirectionShader = new Shader(shaderSource, "CSSortIndirection", "cs_6_6");
                _sortIndirectionPSO = _device.CreateComputePipelineState(sortIndirectionShader.Bytecode);
                sortIndirectionShader.Dispose();
                
                // Pass 5c: CSHistogramPrefixSum - prefix sum of histogram for StartInstance offsets
                var histogramPrefixSumShader = new Shader(shaderSource, "CSHistogramPrefixSum", "cs_6_6");
                _histogramPrefixSumPSO = _device.CreateComputePipelineState(histogramPrefixSumShader.Bytecode);
                histogramPrefixSumShader.Dispose();
                
                // Pass 6: CSMain - generate final indirect draw commands from histogram
                var mainShader = new Shader(shaderSource, "CSMain", "cs_6_6");
                _mainPSO = _device.CreateComputePipelineState(mainShader.Bytecode);
                mainShader.Dispose();
                
                // Pass 7: CSVisibilityShadow - shadow cascade frustum culling
                var visibilityShadowShader = new Shader(shaderSource, "CSVisibilityShadow", "cs_6_6");
                _visibilityShadowPSO = _device.CreateComputePipelineState(visibilityShadowShader.Bytecode);
                visibilityShadowShader.Dispose();
                
                // Hi-Z depth pyramid downsampler (separate shader file)
                string pyramidPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Resources", "Shaders", "depth_pyramid.hlsl");
                if (File.Exists(pyramidPath))
                {
                    string pyramidSource = File.ReadAllText(pyramidPath);
                    var downsampleShader = new Shader(pyramidSource, "CSDownsample", "cs_6_6");
                    _downsamplePSO = _device.CreateComputePipelineState(downsampleShader.Bytecode);
                    downsampleShader.Dispose();
                    Debug.Log("GPUCuller", "CSDownsample compiled for Hi-Z pyramid generation");
                }
                else
                {
                    Debug.LogWarning("GPUCuller", $"depth_pyramid.hlsl not found: {pyramidPath} — Hi-Z disabled");
                }
                
                // SDSM depth analysis shaders (separate file)
                string analysisPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Resources", "Shaders", "depth_analysis.hlsl");
                if (File.Exists(analysisPath))
                {
                    string analysisSource = File.ReadAllText(analysisPath);
                    var reduceShader = new Shader(analysisSource, "CSDepthReduce", "cs_6_6");
                    _depthReducePSO = _device.CreateComputePipelineState(reduceShader.Bytecode);
                    reduceShader.Dispose();
                    
                    var histShader = new Shader(analysisSource, "CSDepthHistogram", "cs_6_6");
                    _depthHistogramPSO = _device.CreateComputePipelineState(histShader.Bytecode);
                    histShader.Dispose();
                    
                    var splitsShader = new Shader(analysisSource, "CSComputeSplits", "cs_6_6");
                    _computeSplitsPSO = _device.CreateComputePipelineState(splitsShader.Bytecode);
                    splitsShader.Dispose();
                    
                    // Create SDSM GPU buffers
                    // MinMax buffer: 8 bytes (2 uints), raw byte address buffer
                    _depthMinMaxBuffer = _device.CreateDefaultBuffer(8);
                    _depthMinMaxUAV = _device.AllocateBindlessIndex();
                    var minMaxUavDesc = new UnorderedAccessViewDescription
                    {
                        Format = Format.R32_Typeless,
                        ViewDimension = UnorderedAccessViewDimension.Buffer,
                        Buffer = new BufferUnorderedAccessView
                        {
                            FirstElement = 0,
                            NumElements = 2,
                            Flags = BufferUnorderedAccessViewFlags.Raw
                        }
                    };
                    _device.NativeDevice.CreateUnorderedAccessView(_depthMinMaxBuffer, null, minMaxUavDesc, _device.GetCpuHandle(_depthMinMaxUAV));
                    
                    // Histogram buffer: 256 uints
                    _depthHistogramBuffer = _device.CreateDefaultBuffer(256 * sizeof(uint));
                    _depthHistogramUAV = _device.AllocateBindlessIndex();
                    _device.CreateStructuredBufferUAV(_depthHistogramBuffer, 256, sizeof(uint), _depthHistogramUAV);
                    
                    // Splits buffer: 4 floats
                    _depthSplitsBuffer = _device.CreateDefaultBuffer(4 * sizeof(float));
                    _depthSplitsUAV = _device.AllocateBindlessIndex();
                    _device.CreateStructuredBufferUAV(_depthSplitsBuffer, 4, sizeof(float), _depthSplitsUAV);
                    
                    // Readback buffers: one per frame, persistently mapped
                    _depthSplitsReadbackBuffers = new ID3D12Resource[FrameCount];
                    _depthSplitsReadbackPtrs = new IntPtr[FrameCount];
                    for (int rb = 0; rb < FrameCount; rb++)
                    {
                        _depthSplitsReadbackBuffers[rb] = _device.NativeDevice.CreateCommittedResource(
                            new HeapProperties(HeapType.Readback),
                            HeapFlags.None,
                            ResourceDescription.Buffer(4 * sizeof(float)),
                            ResourceStates.CopyDest,
                            null);
                        unsafe
                        {
                            void* pData;
                            _depthSplitsReadbackBuffers[rb].Map(0, null, &pData);
                            _depthSplitsReadbackPtrs[rb] = (IntPtr)pData;
                        }
                    }
                    
                    _sdsmInitialized = true;
                    Debug.Log("GPUCuller", "SDSM depth analysis initialized: CSDepthReduce + CSDepthHistogram + CSComputeSplits");
                }
                else
                {
                    Debug.LogWarning("GPUCuller", $"depth_analysis.hlsl not found: {analysisPath} — SDSM disabled");
                }
                
                Debug.Log("GPUCuller", "Compute shaders compiled: CSClear + CSVisibility + CSHistogram + CSLocalScan + CSBlockScan + CSGlobalScatter + CSBitonicSort + CSSortIndirection + CSHistogramPrefixSum + CSMain + CSVisibilityShadow");
                
                // Create per-frame frustum constant buffers
                int frustumConstantsSize = (Marshal.SizeOf<FrustumConstants>() + 255) & ~255; // 256-byte aligned
                
                for (int i = 0; i < FrameCount; i++)
                {
                    _frustumConstantsBuffers[i] = _device.CreateUploadBuffer(frustumConstantsSize);
                    
                    // Counter buffer: one uint per sub-batch (GPU writable)
                    _counterBuffers[i] = _device.CreateDefaultBuffer(MaxSubBatches * sizeof(uint));
                    _counterBufferUAVs[i] = _device.AllocateBindlessIndex();
                    _device.CreateStructuredBufferUAV(_counterBuffers[i], (uint)MaxSubBatches, sizeof(uint), _counterBufferUAVs[i]);
                    
                    // Range buffer: StartInstance + InstanceCount per sub-batch (upload for CPU write)
                    int rangeStride = Marshal.SizeOf<InstanceRange>();
                    _rangeBuffers[i] = _device.CreateUploadBuffer(MaxSubBatches * rangeStride);
                    _rangeBufferSRVs[i] = _device.AllocateBindlessIndex();
                    _device.CreateStructuredBufferSRV(_rangeBuffers[i], (uint)MaxSubBatches, (uint)rangeStride, _rangeBufferSRVs[i]);
                    
                    // Shadow cascade buffer: 24 float4s (6 planes * 4 cascades)
                    int shadowCascadeSize = (24 * 16 + 255) & ~255; // 256-byte aligned
                    _shadowCascadeBuffers[i] = _device.CreateUploadBuffer(shadowCascadeSize);
                    
                    Debug.Log("GPUCuller", $"Frame {i}: CounterBufferUAV={_counterBufferUAVs[i]}, RangeBufferSRV={_rangeBufferSRVs[i]}");
                }
                
                // Create non-shader-visible heap for ClearUAV CPU handles
                // D3D12 ClearUnorderedAccessViewUint requires CPU handle from non-shader-visible heap
                var clearHeapDesc = new DescriptorHeapDescription
                {
                    Type = DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView,
                    DescriptorCount = FrameCount,
                    Flags = DescriptorHeapFlags.None  // Non-shader-visible
                };
                _clearUAVHeap = _device.NativeDevice.CreateDescriptorHeap(clearHeapDesc);
                
                // Create UAV descriptors in non-shader-visible heap for ClearUAV
                int uavDescriptorSize = (int)_device.NativeDevice.GetDescriptorHandleIncrementSize(DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView);
                for (int i = 0; i < FrameCount; i++)
                {
                    var cpuHandle = _clearUAVHeap.GetCPUDescriptorHandleForHeapStart() + (i * uavDescriptorSize);
                    // IMPORTANT: Use same format as shader-visible heap (structured buffer, not raw)
                    // ClearUAV requires matching descriptor format to clear the correct memory layout
                    var uavDesc = new UnorderedAccessViewDescription
                    {
                        Format = Vortice.DXGI.Format.Unknown, // Structured buffer uses Unknown format
                        ViewDimension = UnorderedAccessViewDimension.Buffer,
                        Buffer = new BufferUnorderedAccessView
                        {
                            FirstElement = 0,
                            NumElements = (uint)MaxSubBatches,
                            StructureByteStride = sizeof(uint), // Match structured buffer stride
                            Flags = BufferUnorderedAccessViewFlags.None // Not raw
                        }
                    };
                    _device.NativeDevice.CreateUnorderedAccessView(_counterBuffers[i], null, uavDesc, cpuHandle);
                    _counterBufferCPUHandles[i] = cpuHandle;
                }
                
                // --- Cull stats buffer (2 uints: visible, hi-z occluded) ---
                _cullStatsBuffer = _device.CreateDefaultBuffer(2 * sizeof(uint));
                _cullStatsUAV = _device.AllocateBindlessIndex();
                var statsUavDesc = new UnorderedAccessViewDescription
                {
                    Format = Format.R32_Typeless,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView
                    {
                        FirstElement = 0,
                        NumElements = 2,
                        Flags = BufferUnorderedAccessViewFlags.Raw
                    }
                };
                _device.NativeDevice.CreateUnorderedAccessView(_cullStatsBuffer, null, statsUavDesc, _device.GetCpuHandle(_cullStatsUAV));
                
                // Non-shader-visible descriptor for ClearUAV
                var statsClearHeapDesc = new DescriptorHeapDescription
                {
                    Type = DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView,
                    DescriptorCount = 1,
                    Flags = DescriptorHeapFlags.None
                };
                var statsClearHeap = _device.NativeDevice.CreateDescriptorHeap(statsClearHeapDesc);
                _cullStatsClearCPU = statsClearHeap.GetCPUDescriptorHandleForHeapStart();
                _device.NativeDevice.CreateUnorderedAccessView(_cullStatsBuffer, null, statsUavDesc, _cullStatsClearCPU);
                
                // Readback buffers (per-frame, persistently mapped)
                _cullStatsReadbackBuffers = new ID3D12Resource[FrameCount];
                _cullStatsReadbackPtrs = new IntPtr[FrameCount];
                for (int rb = 0; rb < FrameCount; rb++)
                {
                    _cullStatsReadbackBuffers[rb] = _device.NativeDevice.CreateCommittedResource(
                        new HeapProperties(HeapType.Readback),
                        HeapFlags.None,
                        ResourceDescription.Buffer(2 * sizeof(uint)),
                        ResourceStates.CopyDest,
                        null);
                    unsafe
                    {
                        void* pData;
                        _cullStatsReadbackBuffers[rb].Map(0, null, &pData);
                        _cullStatsReadbackPtrs[rb] = (IntPtr)pData;
                    }
                }
                Debug.Log("GPUCuller", $"Cull stats buffer created: UAV={_cullStatsUAV}");
                
                _initialized = true;
                Debug.Log("GPUCuller", "Initialized for GPU frustum culling");
            }
            catch (Exception ex)
            {
                Debug.LogError("GPUCuller", $"Initialization failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Clear the counter buffer once before processing all batches.
        /// Call this before the batch loop, not per-batch.
        /// </summary>
        public void ClearCounterBuffer(ID3D12GraphicsCommandList commandList)
        {
            if (!_initialized) return;
            
            int frameIndex = Engine.FrameIndex % FrameCount;
            
            var gpuHandle = _device.SrvHeap.GetGPUDescriptorHandleForHeapStart() + 
                (int)(_counterBufferUAVs[frameIndex] * _device.NativeDevice.GetDescriptorHandleIncrementSize(
                    Vortice.Direct3D12.DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView));
            commandList.ClearUnorderedAccessViewUint(
                gpuHandle,
                _counterBufferCPUHandles[frameIndex],
                _counterBuffers[frameIndex],
                new Vortice.Mathematics.Int4(0, 0, 0, 0));
            
            // UAV barrier - ensure counters are zeroed before any CSCount runs
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
        }
        
        /// <summary>UAV index for cull stats buffer (passed in FrustumConstants).</summary>
        public uint CullStatsUAV => _cullStatsUAV;
        
        /// <summary>
        /// Clear cull stats buffer and read back previous frame's results.
        /// Call once per frame before the batch loop.
        /// </summary>
        public void ClearCullStats(ID3D12GraphicsCommandList commandList)
        {
            if (!_initialized || _cullStatsBuffer == null) return;
            
            // Read back N-2 frame's stats (safely past GPU completion)
            int readbackFrame = (Engine.FrameIndex + 1) % FrameCount;
            if (_cullStatsReadbackPtrs != null)
            {
                unsafe
                {
                    uint* pStats = (uint*)_cullStatsReadbackPtrs[readbackFrame];
                    LastVisibleCount = (int)pStats[0];
                    LastHiZOccludedCount = (int)pStats[1];
                }
            }
            
            // Clear stats buffer for this frame
            var gpuHandle = _device.SrvHeap.GetGPUDescriptorHandleForHeapStart() + 
                (int)(_cullStatsUAV * _device.NativeDevice.GetDescriptorHandleIncrementSize(
                    DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView));
            commandList.ClearUnorderedAccessViewUint(
                gpuHandle, _cullStatsClearCPU, _cullStatsBuffer,
                new Vortice.Mathematics.Int4(0, 0, 0, 0));
            
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
        }
        
        /// <summary>
        /// Copy cull stats from GPU buffer to readback buffer.
        /// Call once per frame after all batches have been culled.
        /// </summary>
        public void CopyCullStatsToReadback(ID3D12GraphicsCommandList commandList)
        {
            if (!_initialized || _cullStatsBuffer == null || _cullStatsReadbackBuffers == null) return;
            
            int frameIndex = Engine.FrameIndex % FrameCount;
            
            // Transition stats buffer for copy source
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_cullStatsBuffer, 
                    ResourceStates.UnorderedAccess, ResourceStates.CopySource)));
            
            commandList.CopyResource(_cullStatsReadbackBuffers[frameIndex], _cullStatsBuffer);
            
            // Transition back to UAV for next frame
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_cullStatsBuffer, 
                    ResourceStates.CopySource, ResourceStates.UnorderedAccess)));
        }

        /// <summary>
        /// Dispatch frustum culling: two passes.
        /// Pass 1: Per-instance frustum test (CSCount) - counts visible per sub-batch
        /// Pass 2: Copy templates with culled counts (CSMain)
        /// </summary>
        public void DispatchCull(
            ID3D12GraphicsCommandList commandList,
            Vector4[] frustumPlanes,
            int subBatchCount,
            int totalInstances,
            InstanceRange[] ranges,
            uint templateBufferSRV,
            uint outputBufferUAV,
            uint boundingSpheresSRV,
            uint sortedIndicesInSRV,
            uint counterBufferUAV,
            int counterOffset,
            uint transformBufferSRV,      // World matrices for culling sphere transforms
            uint transformSlotsSRV)       // TransformSlot buffer for order-independent output
        {
            if (!_initialized || _visibilityPSO == null || _mainPSO == null || subBatchCount == 0) return;
            
            int frameIndex = Engine.FrameIndex % FrameCount;
            
            // Upload frustum planes
            var frustumConstants = new FrustumConstants
            {
                Plane0 = frustumPlanes[0],
                Plane1 = frustumPlanes[1],
                Plane2 = frustumPlanes[2],
                Plane3 = frustumPlanes[3],
                Plane4 = frustumPlanes[4],
                Plane5 = frustumPlanes[5]
            };
            
            unsafe
            {
                void* pData;
                _frustumConstantsBuffers[frameIndex].Map(0, null, &pData);
                *(FrustumConstants*)pData = frustumConstants;
                _frustumConstantsBuffers[frameIndex].Unmap(0);
            }
            
            // Upload instance ranges
            unsafe
            {
                void* pData;
                _rangeBuffers[frameIndex].Map(0, null, &pData);
                var span = new Span<InstanceRange>(pData, subBatchCount);
                for (int i = 0; i < subBatchCount; i++)
                    span[i] = ranges[i];
                _rangeBuffers[frameIndex].Unmap(0);
            }
            
            // No counter buffer transition needed - we write directly to output buffer
            
            // Set root signature and descriptor heap (shared by all passes)
            commandList.SetComputeRootSignature(_device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            
            // Bind frustum constant buffer (slot 1)
            commandList.SetComputeRootConstantBufferView(1, _frustumConstantsBuffers[frameIndex].GPUVirtualAddress);
            
            // Three-pass approach with separate counter buffer:
            // 1. CSClear - zero the counter buffer
            // 2. CSCount - atomically increment counters[subBatch] for visible instances
            // 3. CSMain - copy templates and set DrawInstanceCount from counters
            
            // Push constants used by all passes
            commandList.SetComputeRoot32BitConstant(0, templateBufferSRV, 0);
            commandList.SetComputeRoot32BitConstant(0, outputBufferUAV, 1);
            commandList.SetComputeRoot32BitConstant(0, boundingSpheresSRV, 2);
            commandList.SetComputeRoot32BitConstant(0, _rangeBufferSRVs[frameIndex], 3);
            commandList.SetComputeRoot32BitConstant(0, sortedIndicesInSRV, 4);
            commandList.SetComputeRoot32BitConstant(0, 0, 5);  // SortedIndicesOutIdx unused
            commandList.SetComputeRoot32BitConstant(0, _counterBufferUAVs[frameIndex], 6);  // Counter buffer UAV
            commandList.SetComputeRoot32BitConstant(0, (uint)subBatchCount, 7);
            commandList.SetComputeRoot32BitConstant(0, (uint)counterOffset, 8);  // Counter offset for this batch
            commandList.SetComputeRoot32BitConstant(0, 0, 9);  // Slot 9: unused (VisibleIndicesSRVIdx set elsewhere)
            commandList.SetComputeRoot32BitConstant(0, transformBufferSRV, 10);  // Slot 10: CullTransformIdx for world-space sphere transform
            commandList.SetComputeRoot32BitConstant(0, transformSlotsSRV, 11);   // Slot 11: TransformSlotsIdx for order-independent output
            
            uint threadGroupsSubBatches = ((uint)subBatchCount + 63) / 64;
            uint threadGroupsInstances = ((uint)totalInstances + 63) / 64;
            
            // NOTE: Counter buffer cleared by ClearCounterBuffer() before batch loop, not here
            
            // --- Pass 1: CSVisibility (write visibility flags and count visible) ---
            commandList.SetPipelineState(_visibilityPSO);
            commandList.Dispatch(threadGroupsInstances, 1, 1);
            
            // UAV barrier - ensure all atomic writes complete before CSMain reads
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
            
            // --- Pass 3: CSMain (copy templates and read DrawInstanceCount from counters) ---
            commandList.SetPipelineState(_mainPSO);
            commandList.Dispatch(threadGroupsSubBatches, 1, 1);
            
            // UAV barrier to ensure all writes complete before graphics use
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
        }
        
        /// <summary>
        /// Upload shadow cascade frustum planes and dispatch CSVisibilityShadow for a single cascade.
        /// Call this once per cascade before the full culling pipeline.
        /// </summary>
        /// <param name="commandList">Command list to record to</param>
        /// <param name="cascadePlanes">Array of 4 cascade planes (each containing 6 frustum planes)</param>
        /// <param name="cascadeIndex">Which cascade to cull for (0-3)</param>
        /// <param name="totalInstances">Total instance count</param>
        /// <param name="boundingSpheresSRV">SRV for bounding spheres</param>
        /// <param name="transformSlotsSRV">SRV for transform slot indices</param>
        /// <param name="transformBufferSRV">SRV for world matrices</param>
        /// <param name="visibilityFlagsUAV">UAV for output visibility flags</param>
        public void DispatchShadowVisibility(
            ID3D12GraphicsCommandList commandList,
            Vector4[][] cascadePlanes,
            int cascadeIndex,
            int totalInstances,
            uint boundingSpheresSRV,
            uint transformSlotsSRV,
            uint transformBufferSRV,
            uint visibilityFlagsUAV)
        {
            if (!_initialized || _visibilityShadowPSO == null || cascadeIndex < 0 || cascadeIndex > 3) return;
            
            int frameIndex = Engine.FrameIndex % FrameCount;
            
            // Upload all 4 cascade frustum planes to the shadow cascade buffer
            unsafe
            {
                var shadowConstants = new ShadowCascadeConstants();
                for (int c = 0; c < 4 && c < cascadePlanes.Length; c++)
                {
                    shadowConstants.SetCascadePlanes(c, cascadePlanes[c]);
                }
                
                void* pData;
                _shadowCascadeBuffers[frameIndex].Map(0, null, &pData);
                *(ShadowCascadeConstants*)pData = shadowConstants;
                _shadowCascadeBuffers[frameIndex].Unmap(0);
            }
            
            // Set root signature and descriptor heap
            commandList.SetComputeRootSignature(_device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            
            // Bind shadow cascade constant buffer (slot 2 -> register b1)
            commandList.SetComputeRootConstantBufferView(2, _shadowCascadeBuffers[frameIndex].GPUVirtualAddress);
            
            // Set push constants for CSVisibilityShadow
            commandList.SetComputeRoot32BitConstant(0, boundingSpheresSRV, 2);   // BoundingSpheresIdx
            commandList.SetComputeRoot32BitConstant(0, transformSlotsSRV, 4);    // TransformSlotsIdx (Indices[1].x)
            commandList.SetComputeRoot32BitConstant(0, transformBufferSRV, 10);  // GlobalTransformsIdx (Indices[2].z)
            commandList.SetComputeRoot32BitConstant(0, visibilityFlagsUAV, 11);  // VisibilityFlagsIdx (Indices[2].w)
            commandList.SetComputeRoot32BitConstant(0, (uint)totalInstances, 13); // TotalInstances (Indices[3].y)
            commandList.SetComputeRoot32BitConstant(0, (uint)cascadeIndex, 31);   // ShadowCascadeIdx (Indices[7].w)
            
            // Dispatch CSVisibilityShadow
            commandList.SetPipelineState(_visibilityShadowPSO);
            uint threadGroups = ((uint)totalInstances + 255) / 256;
            commandList.Dispatch(threadGroups, 1, 1);
            
            // UAV barrier - ensure visibility flags are written before downstream passes
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
        }
        
        /// <summary>
        /// Simple pass-through dispatch (no culling, for comparison/fallback).
        /// </summary>
        public void DispatchPassThrough(
            ID3D12GraphicsCommandList commandList,
            int subBatchCount,
            uint templateBufferSRV,
            uint outputBufferUAV)
        {
            if (!_initialized || _mainPSO == null || subBatchCount == 0) return;
            
            // Set compute PSO and root signature
            commandList.SetPipelineState(_mainPSO);
            commandList.SetComputeRootSignature(_device.GlobalRootSignature);
            
            // Bind descriptor heap
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            
            // Push constants for buffer indices
            // Indices[0] = (templateSRV, outputUAV, unused, unused)
            commandList.SetComputeRoot32BitConstant(0, templateBufferSRV, 0);
            commandList.SetComputeRoot32BitConstant(0, outputBufferUAV, 1);
            commandList.SetComputeRoot32BitConstant(0, 0, 2);
            commandList.SetComputeRoot32BitConstant(0, 0, 3);
            
            // Indices[1] = (unused, unused, counterUAV=0 for pass-through, subBatchCount)
            commandList.SetComputeRoot32BitConstant(0, 0, 4);
            commandList.SetComputeRoot32BitConstant(0, 0, 5);
            commandList.SetComputeRoot32BitConstant(0, 0, 6); // CounterBufferIdx = 0 means pass-through (no culling)
            commandList.SetComputeRoot32BitConstant(0, (uint)subBatchCount, 7);
            
            // Dispatch enough thread groups to cover all sub-batches
            uint threadGroupsX = ((uint)subBatchCount + 63) / 64;
            commandList.Dispatch(threadGroupsX, 1, 1);
        }

        /// <summary>
        /// Prepare buffers for GPU command generation (transitions).
        /// Call before dispatching sub-batches.
        /// </summary>
        public void BeginCommandGeneration(
            ID3D12GraphicsCommandList commandList,
            ID3D12Resource outputBuffer,
            ID3D12Resource sortedIndicesBuffer,
            bool isFirstUseOutput,
            bool isFirstUseSortedIndices)
        {
            // Transition to UAV for compute write
            // On first use, buffer is in Common state. On subsequent uses, it's in the state from EndCommandGeneration.
            var outputSourceState = isFirstUseOutput ? ResourceStates.Common : ResourceStates.IndirectArgument;
            var sortedSourceState = isFirstUseSortedIndices ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
            
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(outputBuffer, outputSourceState, ResourceStates.UnorderedAccess)));
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(sortedIndicesBuffer, sortedSourceState, ResourceStates.UnorderedAccess)));
        }

        /// <summary>
        /// Finish GPU command generation (transitions for graphics use).
        /// Call after all sub-batch dispatches complete.
        /// </summary>
        public void EndCommandGeneration(
            ID3D12GraphicsCommandList commandList,
            ID3D12Resource outputBuffer,
            ID3D12Resource sortedIndicesBuffer)
        {
            // Transition for graphics use
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(outputBuffer, ResourceStates.UnorderedAccess, ResourceStates.IndirectArgument)));
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(sortedIndicesBuffer, ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)));
        }

        /// <summary>
        /// Reset buffer state after ExecuteIndirect completes.
        /// Call after Draw() to prepare buffers for next frame's cycle.
        /// </summary>
        public void ResetBufferState(
            ID3D12GraphicsCommandList commandList,
            ID3D12Resource outputBuffer,
            ID3D12Resource sortedIndicesBuffer)
        {
            // Transition back to Common for next frame's BeginCommandGeneration
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(outputBuffer, ResourceStates.IndirectArgument, ResourceStates.Common)));
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(sortedIndicesBuffer, ResourceStates.NonPixelShaderResource, ResourceStates.Common)));
        }

        /// <summary>
        /// Create the Hi-Z depth pyramid texture and per-mip descriptors.
        /// Call once when the depth buffer dimensions are known (or on resize).
        /// </summary>
        public void CreateHiZPyramid(int depthWidth, int depthHeight)
        {
            if (!_initialized) return;
            
            // Dispose previous pyramid if resizing
            _hiZPyramid?.Dispose();
            _hiZReady = false;  // New pyramid needs to be generated before use
            
            // Hi-Z is half-res of depth buffer
            _hiZWidth = depthWidth / 2;
            _hiZHeight = depthHeight / 2;
            
            // Calculate mip count
            _hiZMipCount = 1 + (int)Math.Floor(Math.Log2(Math.Max(_hiZWidth, _hiZHeight)));
            
            // Create texture: R32_Float with UAV support, full mip chain
            _hiZPyramid = _device.CreateTexture2D(
                Vortice.DXGI.Format.R32_Float,
                _hiZWidth, _hiZHeight,
                1, _hiZMipCount,
                ResourceFlags.AllowUnorderedAccess,
                ResourceStates.Common);
            
            // Allocate per-mip UAVs and SRVs
            _hiZMipUAVs = new uint[_hiZMipCount];
            _hiZMipUAVCPU = new CpuDescriptorHandle[_hiZMipCount];
            _hiZMipSRVs = new uint[_hiZMipCount];
            
            for (int i = 0; i < _hiZMipCount; i++)
            {
                // UAV for writing this mip level
                _hiZMipUAVs[i] = _device.AllocateBindlessIndex();
                var uavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.R32_Float,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = (uint)i }
                };
                _device.NativeDevice.CreateUnorderedAccessView(_hiZPyramid, null, uavDesc, _device.GetCpuHandle(_hiZMipUAVs[i]));
                
                // SRV for reading this mip level (used as input for next level's downsample)
                _hiZMipSRVs[i] = _device.AllocateBindlessIndex();
                var srvDesc = new ShaderResourceViewDescription
                {
                    Format = Vortice.DXGI.Format.R32_Float,
                    ViewDimension = Vortice.Direct3D12.ShaderResourceViewDimension.Texture2D,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2D = new Texture2DShaderResourceView
                    {
                        MostDetailedMip = (uint)i,
                        MipLevels = 1
                    }
                };
                _device.NativeDevice.CreateShaderResourceView(_hiZPyramid, srvDesc, _device.GetCpuHandle(_hiZMipSRVs[i]));
            }
            
            // Full-pyramid SRV (all mips, for CSVisibility to sample with Load at any mip)
            _hiZPyramidSRV = _device.AllocateBindlessIndex();
            var fullSrvDesc = new ShaderResourceViewDescription
            {
                Format = Vortice.DXGI.Format.R32_Float,
                ViewDimension = Vortice.Direct3D12.ShaderResourceViewDimension.Texture2D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2D = new Texture2DShaderResourceView
                {
                    MostDetailedMip = 0,
                    MipLevels = (uint)_hiZMipCount
                }
            };
            _device.NativeDevice.CreateShaderResourceView(_hiZPyramid, fullSrvDesc, _device.GetCpuHandle(_hiZPyramidSRV));
            
            Debug.Log("GPUCuller", $"Hi-Z pyramid created: {_hiZWidth}x{_hiZHeight}, {_hiZMipCount} mips, SRV={_hiZPyramidSRV}");
        }
        
        /// <summary>
        /// Generate the Hi-Z depth pyramid from the current depth buffer.
        /// Call after GBuffer pass, when depth is in PixelShaderResource state.
        /// </summary>
        /// <param name="commandList">Command list to record to</param>
        /// <param name="depthSrvIndex">Bindless SRV index of the source depth buffer (R32_Float view)</param>
        public void GenerateHiZPyramid(ID3D12GraphicsCommandList commandList, uint depthSrvIndex)
        {
            if (_downsamplePSO == null || _hiZPyramid == null || _hiZMipCount == 0) return;
            
            bool firstGeneration = !_hiZReady;
            _hiZReady = true;  // Pyramid will be valid after this dispatch
            
            commandList.SetComputeRootSignature(_device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            commandList.SetPipelineState(_downsamplePSO);
            
            int w = _hiZWidth;
            int h = _hiZHeight;
            
            for (int mip = 0; mip < _hiZMipCount; mip++)
            {
                // Transition this mip to UAV for writing
                // First generation: ALL mips start in Common (just created); subsequent: all mips in NonPixelShaderResource
                var beforeState = firstGeneration ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
                commandList.ResourceBarrier(new ResourceBarrier(
                    new ResourceTransitionBarrier(_hiZPyramid,
                        beforeState,
                        ResourceStates.UnorderedAccess,
                        (uint)mip)));
                
                // Set push constants: InputMipIdx, OutputMipIdx, OutputWidth, OutputHeight
                uint inputSrv = mip == 0 ? depthSrvIndex : _hiZMipSRVs[mip - 1];
                commandList.SetComputeRoot32BitConstant(0, inputSrv, 0);
                commandList.SetComputeRoot32BitConstant(0, _hiZMipUAVs[mip], 1);
                commandList.SetComputeRoot32BitConstant(0, (uint)w, 2);
                commandList.SetComputeRoot32BitConstant(0, (uint)h, 3);
                
                // Dispatch
                uint groupsX = ((uint)w + 7) / 8;
                uint groupsY = ((uint)h + 7) / 8;
                commandList.Dispatch(groupsX, groupsY, 1);
                
                // Transition this mip to SRV for reading by next level
                commandList.ResourceBarrier(new ResourceBarrier(
                    new ResourceTransitionBarrier(_hiZPyramid,
                        ResourceStates.UnorderedAccess,
                        ResourceStates.NonPixelShaderResource,
                        (uint)mip)));
                
                // Next mip is half size
                w = Math.Max(1, w / 2);
                h = Math.Max(1, h / 2);
            }
            
            // Final transition: all subresources to PixelShaderResource | NonPixelShaderResource
            // so CSVisibility can sample via Load
            // Note: individual mip subresources are already in NonPixelShaderResource from the loop above
        }

        /// <summary>
        /// Dispatch SDSM depth analysis compute shaders.
        /// Call after Hi-Z generation, when DepthGBuffer is in NonPixelShaderResource state.
        /// </summary>
        public void AnalyzeDepth(ID3D12GraphicsCommandList commandList, uint depthSrvIndex, int texWidth, int texHeight, float nearPlane, float farPlane)
        {
            if (!_sdsmInitialized || _depthReducePSO == null || _depthHistogramPSO == null || _computeSplitsPSO == null) return;
            
            int frameIndex = Engine.FrameIndex % FrameCount;
            
            commandList.SetComputeRootSignature(_device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            
            // Create the persistent clear buffer (once)
            unsafe
            {
                if (_sdsmClearBuffer == null)
                {
                    _sdsmClearBuffer = _device.CreateUploadBuffer(8 + 256 * sizeof(uint));
                    void* pData;
                    _sdsmClearBuffer.Map(0, null, &pData);
                    uint* pUint = (uint*)pData;
                    pUint[0] = 0x7F7FFFFF; // min = max float
                    pUint[1] = 0;           // max = 0
                    for (int i = 0; i < 256; i++)
                        pUint[2 + i] = 0;   // histogram = all zeros
                    _sdsmClearBuffer.Unmap(0);
                }
            }
            
            // Step 1: Transition buffers to CopyDest for clear copies
            var minMaxFrom = _sdsmValidFrames == 0 ? ResourceStates.Common : ResourceStates.CopySource;
            var histFrom = _sdsmValidFrames == 0 ? ResourceStates.Common : ResourceStates.UnorderedAccess;
            var splitsFrom = _sdsmValidFrames == 0 ? ResourceStates.Common : ResourceStates.CopySource;
            
            commandList.ResourceBarrierTransition(_depthMinMaxBuffer!, minMaxFrom, ResourceStates.CopyDest);
            commandList.ResourceBarrierTransition(_depthHistogramBuffer!, histFrom, ResourceStates.CopyDest);
            commandList.ResourceBarrierTransition(_depthSplitsBuffer!, splitsFrom, ResourceStates.CopyDest);
            
            // Step 2: Copy clear values from upload buffer
            commandList.CopyBufferRegion(_depthMinMaxBuffer!, 0, _sdsmClearBuffer!, 0, 8);
            commandList.CopyBufferRegion(_depthHistogramBuffer!, 0, _sdsmClearBuffer!, 8, 256 * sizeof(uint));
            
            // Step 3: Transition to UAV for compute
            commandList.ResourceBarrierTransition(_depthMinMaxBuffer!, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
            commandList.ResourceBarrierTransition(_depthHistogramBuffer!, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
            commandList.ResourceBarrierTransition(_depthSplitsBuffer!, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
            
            // Set push constants shared by all passes
            commandList.SetComputeRoot32BitConstant(0, depthSrvIndex, 0);        // DepthTexIdx
            commandList.SetComputeRoot32BitConstant(0, _depthMinMaxUAV, 1);      // MinMaxUAVIdx
            commandList.SetComputeRoot32BitConstant(0, _depthHistogramUAV, 2);   // HistogramUAVIdx
            commandList.SetComputeRoot32BitConstant(0, _depthSplitsUAV, 3);      // SplitsUAVIdx
            commandList.SetComputeRoot32BitConstant(0, (uint)texWidth, 4);       // TexWidth
            commandList.SetComputeRoot32BitConstant(0, (uint)texHeight, 5);      // TexHeight
            
            uint nearAsUint;
            uint farAsUint;
            unsafe
            {
                nearAsUint = *(uint*)&nearPlane;
                farAsUint = *(uint*)&farPlane;
            }
            commandList.SetComputeRoot32BitConstant(0, nearAsUint, 6);           // NearPlane
            commandList.SetComputeRoot32BitConstant(0, farAsUint, 7);            // FarPlane
            
            uint groupsX = ((uint)texWidth + 15) / 16;
            uint groupsY = ((uint)texHeight + 15) / 16;
            
            // --- Pass 1: CSDepthReduce ---
            commandList.SetPipelineState(_depthReducePSO);
            commandList.Dispatch(groupsX, groupsY, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
            
            // --- Pass 2: CSDepthHistogram ---
            commandList.SetPipelineState(_depthHistogramPSO);
            commandList.Dispatch(groupsX, groupsY, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
            
            // --- Pass 3: CSComputeSplits ---
            commandList.SetPipelineState(_computeSplitsPSO);
            commandList.Dispatch(1, 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
            
            // Step 4: Transition splits + minmax to CopySource for readback
            commandList.ResourceBarrierTransition(_depthSplitsBuffer!,
                ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            commandList.ResourceBarrierTransition(_depthMinMaxBuffer!,
                ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            // Histogram stays in UAV (next frame will transition it to CopyDest)
            
            // Step 5: Copy to readback buffer for this frame
            commandList.CopyBufferRegion(
                _depthSplitsReadbackBuffers![frameIndex], 0,
                _depthSplitsBuffer!, 0,
                4 * sizeof(float));
            
            _sdsmValidFrames++;
        }
        
        // Persistent upload buffer for clearing SDSM buffers
        private ID3D12Resource? _sdsmClearBuffer;
        
        /// <summary>
        /// Read the adaptive cascade splits computed by the previous frame's depth analysis.
        /// Returns null if no valid data is available yet.
        /// </summary>
        public unsafe float[]? ReadAdaptiveSplits()
        {
            if (!_sdsmInitialized || _depthSplitsReadbackPtrs == null || _sdsmValidFrames < 2)
                return null;
            
            // Read from the PREVIOUS frame's readback buffer (guaranteed complete)
            int readFrame = (Engine.FrameIndex - 1 + FrameCount) % FrameCount;
            float* pSplits = (float*)_depthSplitsReadbackPtrs[readFrame];
            
            // Check for invalid data (zeros = no scene geometry)
            if (pSplits[0] <= 0)
                return null;
            
            return new float[] { pSplits[0], pSplits[1], pSplits[2], pSplits[3] };
        }

        public void Dispose()
        {
            _clearPSO?.Dispose();
            _visibilityPSO?.Dispose();
            _visibilityShadowPSO?.Dispose();
            _histogramPSO?.Dispose();
            _histogramPrefixSumPSO?.Dispose();
            _localScanPSO?.Dispose();
            _blockScanPSO?.Dispose();
            _globalScatterPSO?.Dispose();
            _bitonicSortPSO?.Dispose();
            _sortIndirectionPSO?.Dispose();
            _mainPSO?.Dispose();
            _downsamplePSO?.Dispose();
            _depthReducePSO?.Dispose();
            _depthHistogramPSO?.Dispose();
            _computeSplitsPSO?.Dispose();
            _clearUAVHeap?.Dispose();
            _hiZPyramid?.Dispose();
            _depthMinMaxBuffer?.Dispose();
            _depthHistogramBuffer?.Dispose();
            _depthSplitsBuffer?.Dispose();
            _sdsmClearBuffer?.Dispose();
            
            if (_depthSplitsReadbackBuffers != null)
                for (int i = 0; i < FrameCount; i++)
                    _depthSplitsReadbackBuffers[i]?.Dispose();
            
            for (int i = 0; i < FrameCount; i++)
            {
                _frustumConstantsBuffers[i]?.Dispose();
                _counterBuffers[i]?.Dispose();
                _rangeBuffers[i]?.Dispose();
                _shadowCascadeBuffers[i]?.Dispose();
            }
        }
    }
}
