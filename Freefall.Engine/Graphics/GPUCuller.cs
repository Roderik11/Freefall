using System;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Components;
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
        private ID3D12PipelineState? _visibilityShadow4PSO; // CSVisibilityShadow4 - unified multi-cascade culling
        private ID3D12PipelineState? _expandCascadesPSO;  // CSExpandCascades - expand visible instances per cascade
        private ID3D12PipelineState? _patchExpandedPSO;   // CSPatchExpandedCounts - patch indirect args after expansion
        private ID3D12PipelineState? _downsamplePSO;    // CSSinglePassDownsample - SPD mips 0-5
        private ID3D12PipelineState? _downsamplePerMipPSO; // CSDownsample - per-mip fallback for mips 6+
        private ID3D12PipelineState? _shadowDownsamplePSO;    // CSSinglePassDownsampleShadow - shadow SPD
        private ID3D12PipelineState? _shadowDownsamplePerMipPSO; // CSDownsampleShadow - shadow per-mip fallback
        
        // SDSM depth analysis PSOs
        private ID3D12PipelineState? _depthReducePSO;    // CSDepthReduce - min/max depth reduction
        private ID3D12PipelineState? _depthHistogramPSO; // CSDepthHistogram - depth histogram
        private ID3D12PipelineState? _computeSplitsPSO;  // CSComputeSplits - percentile split computation
        
        // GPU-driven cascade computation PSO
        private ID3D12PipelineState? _cascadeComputePSO; // CSComputeCascadeMatrices - GPU cascade matrix computation
        
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
        

        
        // SDSM depth analysis buffers
        private ID3D12Resource? _depthMinMaxBuffer;           // 2 uints (min, max as float bits), GPU default
        private uint _depthMinMaxUAV;
        private ID3D12Resource? _depthHistogramBuffer;        // 256 uints, GPU default
        private uint _depthHistogramUAV;
        private ID3D12Resource? _depthSplitsBuffer;           // 4 floats, GPU default
        private uint _depthSplitsUAV;
        private uint _depthSplitsSRV;                         // SRV for cascade compute to read splits
        private ID3D12Resource[]? _depthSplitsReadbackBuffers; // Per-frame readback (HeapType.Readback)
        private IntPtr[]? _depthSplitsReadbackPtrs;           // Persistently mapped pointers
        private bool _sdsmInitialized;
        private int _sdsmValidFrames;                         // Frames since SDSM started producing data
        
        // GPU-driven cascade computation buffers
        private bool _cascadeComputeInitialized;
        private ID3D12Resource? _lightingCascadeBuffer;       // GPU default: LightingCascadeData[MaxCascades]
        private uint _lightingCascadeUAV;                     // UAV for compute write
        private uint _lightingCascadeSRV;                     // SRV for lighting pass read
        private ID3D12Resource? _prevVPBuffer;                // Upload: previous frame VP matrices [MaxCascades]
        private IntPtr _prevVPBufferPtr;                      // Persistently mapped
        private uint _prevVPBufferSRV;                        // SRV for cascade compute read
        private ID3D12Resource[]? _cascadeParamsCBs;          // Per-frame cbuffer for cascade compute params
        private IntPtr[]? _cascadeParamsCBPtrs;               // Persistently mapped
        private ID3D12Resource? _smoothedSplitsBuffer;        // GPU default: 4 floats, persists between frames
        private uint _smoothedSplitsUAV;                      // UAV for cascade compute read/write
        
        private GraphicsDevice _device;
        private bool _initialized;
        public bool Initialized => _initialized;
        public string? InitError { get; private set; }
        
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
        public ID3D12PipelineState? VisibilityShadow4PSO => _visibilityShadow4PSO;
        public ID3D12PipelineState? ExpandCascadesPSO => _expandCascadesPSO;
        public ID3D12PipelineState? PatchExpandedPSO => _patchExpandedPSO;
        public ID3D12PipelineState? DownsamplePSO => _downsamplePSO;
        
        // Hi-Z pyramid resources are now owned by HiZPyramid (managed by DeferredRenderer)
        
        /// <summary>True after SDSM buffers are created and ready for dispatch.</summary>
        public bool SdsmReady => _sdsmInitialized;
        
        /// <summary>True after cascade compute PSO and buffers are ready.</summary>
        public bool CascadeComputeReady => _cascadeComputeInitialized;
        
        /// <summary>SRV index for the GPU-computed lighting cascade data (for light_directional.fx).</summary>
        public uint LightingCascadeSRV => _lightingCascadeSRV;

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
            public uint DebugMode;       // Debug visualization mode (unused by culler, kept for layout compatibility)
            public float ProjScale;      // Projection._m22 = cot(fovY/2) for sphere→screen size
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
        /// Per-cascade data for shadow rendering (StructuredBuffer element).
        /// Must match CascadeData struct in HLSL.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct CascadeData
        {
            // 6 frustum planes (float4 each) 
            public Vector4 Plane0, Plane1, Plane2, Plane3, Plane4, Plane5;
            // Current frame VP matrix (for rendering)
            public Matrix4x4 VP;
            // Previous frame VP matrix (for Hi-Z occlusion)
            public Matrix4x4 PrevVP;
            // Cascade split distances: X=near, Y=far, ZW=unused
            public Vector4 SplitDistances;
            
            public void SetPlanes(Vector4[] planes)
            {
                if (planes.Length < 6) return;
                Plane0 = planes[0];
                Plane1 = planes[1];
                Plane2 = planes[2];
                Plane3 = planes[3];
                Plane4 = planes[4];
                Plane5 = planes[5];
            }
        }

        /// <summary>
        /// Shadow cascade frustum planes for compute culler cbuffer (register b1).
        /// 4 cascades × 6 planes = 24 Vector4s. Must match cbuffer ShadowCascadePlanes in HLSL.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct ShadowCascadeConstants
        {
            public Vector4 C0P0, C0P1, C0P2, C0P3, C0P4, C0P5;
            public Vector4 C1P0, C1P1, C1P2, C1P3, C1P4, C1P5;
            public Vector4 C2P0, C2P1, C2P2, C2P3, C2P4, C2P5;
            public Vector4 C3P0, C3P1, C3P2, C3P3, C3P4, C3P5;
        }

        /// <summary>
        /// Per-cascade lighting data for the light_directional pixel shader.
        /// Must match LightingCascadeData struct in cascade_compute.hlsl and light_directional.fx.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct LightingCascadeData
        {
            public Matrix4x4 VP;       // Camera-relative light VP (64 bytes)
            public Vector4 Cascade;    // X=near, Y=far (16 bytes)
        }

        /// <summary>
        /// Parameters cbuffer for cascade compute shader (register b1).
        /// Must match CascadeParams cbuffer in cascade_compute.hlsl.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct CascadeComputeParams
        {
            public Matrix4x4 CameraView;         // 64 bytes
            public Matrix4x4 CameraProjection;   // 64 bytes
            public Vector3 CameraPosition;        // 12 bytes
            public float _pad0;                   // 4 bytes
            public Vector3 CameraForward;          // 12 bytes
            public float _pad1;                   // 4 bytes
            public Vector3 CameraUp;               // 12 bytes
            public float _pad2;                   // 4 bytes
            public Vector3 LightForward;           // 12 bytes
            public float _pad3;                   // 4 bytes
            public Vector4 CascadeSplits;          // 16 bytes (unused for GPU path, splits come from buffer)
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
                
                // Pass 8: CSVisibilityShadow4 - unified multi-cascade visibility
                var visShadow4Shader = new Shader(shaderSource, "CSVisibilityShadow4", "cs_6_6");
                _visibilityShadow4PSO = _device.CreateComputePipelineState(visShadow4Shader.Bytecode);
                visShadow4Shader.Dispose();
                
                // Pass 9: CSExpandCascades - expand visible instances per cascade
                var expandShader = new Shader(shaderSource, "CSExpandCascades", "cs_6_6");
                _expandCascadesPSO = _device.CreateComputePipelineState(expandShader.Bytecode);
                expandShader.Dispose();
                
                // Pass 10: CSPatchExpandedCounts - patch indirect args after expansion
                var patchShader = new Shader(shaderSource, "CSPatchExpandedCounts", "cs_6_6");
                _patchExpandedPSO = _device.CreateComputePipelineState(patchShader.Bytecode);
                patchShader.Dispose();
                
                // Hi-Z depth pyramid downsampler (separate shader file)
                string pyramidPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Resources", "Shaders", "depth_pyramid.hlsl");
                if (File.Exists(pyramidPath))
                {
                    string pyramidSource = File.ReadAllText(pyramidPath);
                    var spdShader = new Shader(pyramidSource, "CSSinglePassDownsample", "cs_6_6");
                    _downsamplePSO = _device.CreateComputePipelineState(spdShader.Bytecode);
                    spdShader.Dispose();
                    Debug.Log("GPUCuller", "CSSinglePassDownsample compiled for Hi-Z pyramid (mips 0-5)");
                    
                    var perMipShader = new Shader(pyramidSource, "CSDownsample", "cs_6_6");
                    _downsamplePerMipPSO = _device.CreateComputePipelineState(perMipShader.Bytecode);
                    perMipShader.Dispose();
                    Debug.Log("GPUCuller", "CSDownsample compiled for Hi-Z pyramid (mips 6+)");
                    
                    var shadowSpdShader = new Shader(pyramidSource, "CSSinglePassDownsampleShadow", "cs_6_6");
                    _shadowDownsamplePSO = _device.CreateComputePipelineState(shadowSpdShader.Bytecode);
                    shadowSpdShader.Dispose();
                    Debug.Log("GPUCuller", "CSSinglePassDownsampleShadow compiled");
                    
                    var shadowPerMipShader = new Shader(pyramidSource, "CSDownsampleShadow", "cs_6_6");
                    _shadowDownsamplePerMipPSO = _device.CreateComputePipelineState(shadowPerMipShader.Bytecode);
                    shadowPerMipShader.Dispose();
                    Debug.Log("GPUCuller", "CSDownsampleShadow compiled");
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
                    _depthSplitsSRV = _device.AllocateBindlessIndex();
                    _device.CreateStructuredBufferSRV(_depthSplitsBuffer, 4, sizeof(float), _depthSplitsSRV);
                    
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
                
                // GPU-driven cascade matrix computation (depends on SDSM being initialized)
                if (_sdsmInitialized)
                {
                    string cascadePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Resources", "Shaders", "cascade_compute.hlsl");
                    if (File.Exists(cascadePath))
                    {
                        string cascadeSource = File.ReadAllText(cascadePath);
                        var cascadeShader = new Shader(cascadeSource, "CSComputeCascadeMatrices", "cs_6_6");
                        _cascadeComputePSO = _device.CreateComputePipelineState(cascadeShader.Bytecode);
                        cascadeShader.Dispose();
                        
                        int maxCascades = 8;
                        int lightingStride = Marshal.SizeOf<LightingCascadeData>();
                        int vpStride = Marshal.SizeOf<Matrix4x4>();
                        
                        // Lighting cascade output buffer (GPU default, compute writes)
                        _lightingCascadeBuffer = _device.CreateDefaultBuffer(lightingStride * maxCascades);
                        _lightingCascadeUAV = _device.AllocateBindlessIndex();
                        _device.CreateStructuredBufferUAV(_lightingCascadeBuffer, (uint)maxCascades, (uint)lightingStride, _lightingCascadeUAV);
                        _lightingCascadeSRV = _device.AllocateBindlessIndex();
                        _device.CreateStructuredBufferSRV(_lightingCascadeBuffer, (uint)maxCascades, (uint)lightingStride, _lightingCascadeSRV);
                        
                        // Previous VP buffer (upload, persistently mapped)
                        _prevVPBuffer = _device.CreateUploadBuffer(vpStride * maxCascades);
                        _prevVPBufferSRV = _device.AllocateBindlessIndex();
                        _device.CreateStructuredBufferSRV(_prevVPBuffer, (uint)maxCascades, (uint)vpStride, _prevVPBufferSRV);
                        unsafe
                        {
                            void* pData;
                            _prevVPBuffer.Map(0, null, &pData);
                            _prevVPBufferPtr = (IntPtr)pData;
                            // Initialize to identity matrices
                            for (int m = 0; m < maxCascades; m++)
                                ((Matrix4x4*)pData)[m] = Matrix4x4.Identity;
                        }
                        
                        // Cascade compute params cbuffer (per-frame, 256-byte aligned)
                        int paramsSize = (Marshal.SizeOf<CascadeComputeParams>() + 255) & ~255;
                        _cascadeParamsCBs = new ID3D12Resource[FrameCount];
                        _cascadeParamsCBPtrs = new IntPtr[FrameCount];
                        for (int f = 0; f < FrameCount; f++)
                        {
                            _cascadeParamsCBs[f] = _device.CreateUploadBuffer(paramsSize);
                            unsafe
                            {
                                void* pData;
                                _cascadeParamsCBs[f].Map(0, null, &pData);
                                _cascadeParamsCBPtrs[f] = (IntPtr)pData;
                            }
                        }
                        
                        // Smoothed splits buffer (GPU default, persistent between frames)
                        _smoothedSplitsBuffer = _device.CreateDefaultBuffer(4 * sizeof(float));
                        _smoothedSplitsUAV = _device.AllocateBindlessIndex();
                        _device.CreateStructuredBufferUAV(_smoothedSplitsBuffer, 4, sizeof(float), _smoothedSplitsUAV);
                        
                        // DISABLED: GPU cascade compute produces incorrect VP matrices for lighting.
                        // The CPU cascade path with SDSM GPU readback works correctly.
                        // TODO: Debug lightVP computation in cascade_compute.hlsl
                        // _cascadeComputeInitialized = true;
                        Debug.Log("GPUCuller", "CSComputeCascadeMatrices initialized: GPU-driven cascade computation ready");
                    }
                    else
                    {
                        Debug.LogWarning("GPUCuller", $"cascade_compute.hlsl not found: {cascadePath} — GPU cascade compute disabled");
                    }
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
                InitError = ex.Message;
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
            uint sortedIndicesInSRV,
            uint counterBufferUAV,
            int counterOffset,
            uint transformBufferSRV,      // World matrices for culling sphere transforms
            uint descriptorSRV)            // InstanceDescriptor buffer for TransformSlot lookup
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
            // Slot 2 (BoundingSpheresIdx) no longer used — culler reads from MeshRegistry
            commandList.SetComputeRoot32BitConstant(0, _rangeBufferSRVs[frameIndex], 3);
            commandList.SetComputeRoot32BitConstant(0, sortedIndicesInSRV, 4);
            commandList.SetComputeRoot32BitConstant(0, 0, 5);  // SortedIndicesOutIdx unused
            commandList.SetComputeRoot32BitConstant(0, _counterBufferUAVs[frameIndex], 6);  // Counter buffer UAV
            commandList.SetComputeRoot32BitConstant(0, (uint)subBatchCount, 7);
            commandList.SetComputeRoot32BitConstant(0, (uint)counterOffset, 8);  // Counter offset for this batch
            commandList.SetComputeRoot32BitConstant(0, 0, 9);  // Slot 9: unused (VisibleIndicesSRVIdx set elsewhere)
            commandList.SetComputeRoot32BitConstant(0, transformBufferSRV, 10);  // Slot 10: CullTransformIdx for world-space sphere transform
            commandList.SetComputeRoot32BitConstant(0, descriptorSRV, 11);   // Slot 11: DescriptorBufferIdx for order-independent output
            
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
        /// <param name="cascadePlanes">Array of cascade planes (each containing 6 frustum planes)</param>
        /// <param name="cascadeIndex">Which cascade to cull for (0-3)</param>
        /// <param name="totalInstances">Total instance count</param>
        /// <param name="descriptorSRV">SRV for InstanceDescriptor buffer</param>
        /// <param name="transformBufferSRV">SRV for world matrices</param>
        /// <param name="visibilityFlagsUAV">UAV for output visibility flags</param>
        public void DispatchShadowVisibility(
            ID3D12GraphicsCommandList commandList,
            uint cascadeBufferSrv,
            int cascadeIndex,
            int totalInstances,
            uint descriptorSRV,
            uint transformBufferSRV,
            uint visibilityFlagsUAV)
        {
            if (!_initialized || _visibilityShadowPSO == null) return;
            
            // Set root signature and descriptor heap
            commandList.SetComputeRootSignature(_device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            
            // Set push constants for CSVisibilityShadow
            commandList.SetComputeRoot32BitConstant(0, descriptorSRV, 4);    // DescriptorBufferIdx
            commandList.SetComputeRoot32BitConstant(0, transformBufferSRV, 10);  // GlobalTransformsIdx
            commandList.SetComputeRoot32BitConstant(0, visibilityFlagsUAV, 11);  // VisibilityFlagsIdx
            commandList.SetComputeRoot32BitConstant(0, (uint)totalInstances, 13); // TotalInstances
            commandList.SetComputeRoot32BitConstant(0, cascadeBufferSrv, 31);    // CascadeBufferSRVIdx
            
            // Dispatch CSVisibilityShadow
            commandList.SetPipelineState(_visibilityShadowPSO);
            uint threadGroups = ((uint)totalInstances + 255) / 256;
            commandList.Dispatch(threadGroups, 1, 1);
            
            // UAV barrier
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
        /// Generate the Hi-Z depth pyramid from the current depth buffer.
        /// Call after GBuffer pass, when depth is in PixelShaderResource state.
        /// </summary>
        /// <param name="commandList">Command list to record to</param>
        /// <param name="depthSrvIndex">Bindless SRV index of the source depth buffer (R32_Float view)</param>
        /// <param name="pyramid">Hi-Z pyramid to write into (owned by DeferredRenderer)</param>
        public void GenerateHiZPyramid(ID3D12GraphicsCommandList commandList, uint depthSrvIndex, HiZPyramid pyramid)
        {
            if (_downsamplePerMipPSO == null || pyramid?.Texture == null || pyramid.MipCount == 0) return;
            
            bool firstGeneration = !pyramid.Ready;
            pyramid.Ready = true;  // Pyramid will be valid after this dispatch
            
            commandList.SetComputeRootSignature(_device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            commandList.SetPipelineState(_downsamplePerMipPSO);
            
            int w = pyramid.Width;
            int h = pyramid.Height;
            
            for (int mip = 0; mip < pyramid.MipCount; mip++)
            {
                // Transition this mip to UAV for writing
                var beforeState = firstGeneration ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
                commandList.ResourceBarrier(new ResourceBarrier(
                    new ResourceTransitionBarrier(pyramid.Texture,
                        beforeState,
                        ResourceStates.UnorderedAccess,
                        (uint)mip)));
                
                // Push constants: InputSrv, OutputUav, Width, Height
                uint inputSrv = mip == 0 ? depthSrvIndex : pyramid.MipSRVs[mip - 1];
                commandList.SetComputeRoot32BitConstant(0, inputSrv, 0);
                commandList.SetComputeRoot32BitConstant(0, pyramid.MipUAVs[mip], 1);
                commandList.SetComputeRoot32BitConstant(0, (uint)w, 2);
                commandList.SetComputeRoot32BitConstant(0, (uint)h, 3);
                
                commandList.Dispatch(((uint)w + 7) / 8, ((uint)h + 7) / 8, 1);
                
                // Transition to SRV for next level
                commandList.ResourceBarrier(new ResourceBarrier(
                    new ResourceTransitionBarrier(pyramid.Texture,
                        ResourceStates.UnorderedAccess,
                        ResourceStates.NonPixelShaderResource,
                        (uint)mip)));
                
                w = Math.Max(1, w / 2);
                h = Math.Max(1, h / 2);
            }
        }

        /// <summary>
        /// Generate Hi-Z pyramid from shadow depth maps (all cascades).
        /// Call after shadow rendering, when shadow texture is in PixelShaderResource state.
        /// </summary>
        public void GenerateShadowHiZPyramid(
            ID3D12GraphicsCommandList commandList,
            uint shadowArraySrvIndex,
            ShadowHiZPyramid pyramid)
        {
            if (!_initialized || _shadowDownsamplePSO == null || pyramid?.Texture == null) return;

            bool firstGeneration = !pyramid.Ready;
            pyramid.Ready = true;

            int sourceW = pyramid.SourceWidth;
            int sourceH = pyramid.SourceHeight;
            int spdMipCount = Math.Min(pyramid.MipCount, 6);
            uint numGroupsX = ((uint)sourceW + 63) / 64;
            uint numGroupsY = ((uint)sourceH + 63) / 64;

            commandList.SetComputeRootSignature(_device.GlobalRootSignature);
            commandList.SetPipelineState(_shadowDownsamplePSO);

            // Process each cascade slice
            for (int slice = 0; slice < pyramid.SliceCount; slice++)
            {
                // Transition SPD mips (0 to spdMipCount-1) for this slice to UAV
                var beforeState = firstGeneration ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
                var barriers = new ResourceBarrier[spdMipCount];
                for (int mip = 0; mip < spdMipCount; mip++)
                {
                    // Subresource index for Texture2DArray: mip + slice * MipCount
                    uint subresource = (uint)(mip + slice * pyramid.MipCount);
                    barriers[mip] = new ResourceBarrier(
                        new ResourceTransitionBarrier(pyramid.Texture,
                            beforeState,
                            ResourceStates.UnorderedAccess,
                            subresource));
                }
                commandList.ResourceBarrier(barriers);

                // Push constants — same layout as camera SPD
                commandList.SetComputeRoot32BitConstant(0, shadowArraySrvIndex, 0); // SourceSrvIdx (Texture2DArray)
                commandList.SetComputeRoot32BitConstant(0, 0u, 1);                  // CounterUavIdx (unused)
                commandList.SetComputeRoot32BitConstant(0, (uint)sourceW, 2);
                commandList.SetComputeRoot32BitConstant(0, (uint)sourceH, 3);
                commandList.SetComputeRoot32BitConstant(0, (uint)spdMipCount, 4);
                commandList.SetComputeRoot32BitConstant(0, numGroupsX, 5);
                commandList.SetComputeRoot32BitConstant(0, numGroupsY, 6);
                // Mip 0 UAV
                commandList.SetComputeRoot32BitConstant(0, pyramid.GetMipUAV(slice, 0), 7);
                // Mip 1-5 UAVs
                for (int m = 1; m < spdMipCount && m <= 12; m++)
                {
                    int slot = 8 + (m - 1);
                    commandList.SetComputeRoot32BitConstant(0, pyramid.GetMipUAV(slice, m), (uint)slot);
                }
                // SliceIndex at [5].x = slot 20
                commandList.SetComputeRoot32BitConstant(0, (uint)slice, 20);

                commandList.Dispatch(numGroupsX, numGroupsY, 1);

                // Transition SPD mips back to SRV
                for (int mip = 0; mip < spdMipCount; mip++)
                {
                    uint subresource = (uint)(mip + slice * pyramid.MipCount);
                    barriers[mip] = new ResourceBarrier(
                        new ResourceTransitionBarrier(pyramid.Texture,
                            ResourceStates.UnorderedAccess,
                            ResourceStates.NonPixelShaderResource,
                            subresource));
                }
                commandList.ResourceBarrier(barriers);

                // Mips 6+ via per-mip fallback
                if (pyramid.MipCount > 6 && _shadowDownsamplePerMipPSO != null)
                {
                    commandList.SetPipelineState(_shadowDownsamplePerMipPSO);

                    int mw = pyramid.Width;
                    int mh = pyramid.Height;
                    for (int skip = 0; skip < 5; skip++) { mw = Math.Max(1, mw / 2); mh = Math.Max(1, mh / 2); }

                    for (int mip = 6; mip < pyramid.MipCount; mip++)
                    {
                        mw = Math.Max(1, mw / 2);
                        mh = Math.Max(1, mh / 2);

                        uint sub = (uint)(mip + slice * pyramid.MipCount);
                        commandList.ResourceBarrier(new ResourceBarrier(
                            new ResourceTransitionBarrier(pyramid.Texture,
                                firstGeneration ? ResourceStates.Common : ResourceStates.NonPixelShaderResource,
                                ResourceStates.UnorderedAccess, sub)));

                        commandList.SetComputeRoot32BitConstant(0, pyramid.GetMipSRV(slice, mip - 1), 0);
                        commandList.SetComputeRoot32BitConstant(0, pyramid.GetMipUAV(slice, mip), 1);
                        commandList.SetComputeRoot32BitConstant(0, (uint)mw, 2);
                        commandList.SetComputeRoot32BitConstant(0, (uint)mh, 3);

                        commandList.Dispatch(((uint)mw + 7) / 8, ((uint)mh + 7) / 8, 1);

                        commandList.ResourceBarrier(new ResourceBarrier(
                            new ResourceTransitionBarrier(pyramid.Texture,
                                ResourceStates.UnorderedAccess,
                                ResourceStates.NonPixelShaderResource, sub)));
                    }

                    // Restore SPD PSO for next slice
                    if (slice < pyramid.SliceCount - 1)
                        commandList.SetPipelineState(_shadowDownsamplePSO);
                }
            }
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

        /// <summary>
        /// Dispatch GPU cascade matrix computation.
        /// Reads splits from the SDSM splits buffer (previous frame) and computes all cascade data.
        /// </summary>
        public unsafe void ComputeCascadeMatrices(
            ID3D12GraphicsCommandList commandList,
            Camera camera,
            Vector3 cameraPosition,
            Vector3 lightForward,
            int shadowMapResolution,
            uint cascadeOutUAV,          // UAV for CascadeData structured buffer
            uint cascadeOutSRV)          // SRV for CascadeData (will be read by culler)
        {
            if (!_cascadeComputeInitialized || _cascadeComputePSO == null) return;
            
            int frameIndex = Engine.FrameIndex % FrameCount;
            
            // Upload cascade compute params
            var cameraView = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, camera.Forward, camera.Up);
            
            var cbParams = new CascadeComputeParams
            {
                CameraView = cameraView,
                CameraProjection = camera.Projection,
                CameraPosition = cameraPosition,
                CameraForward = camera.Forward,
                CameraUp = camera.Up,
                LightForward = lightForward,
                CascadeSplits = Vector4.Zero  // Unused — splits come from GPU buffer
            };
            *(CascadeComputeParams*)_cascadeParamsCBPtrs![frameIndex] = cbParams;
            
            // Transition lighting cascade buffer to UAV for compute write
            // (On first use it's Common, subsequent uses it's NonPixelShaderResource)
            var lightingFromState = _sdsmValidFrames > 1 ? ResourceStates.NonPixelShaderResource : ResourceStates.Common;
            commandList.ResourceBarrierTransition(_lightingCascadeBuffer!,
                lightingFromState, ResourceStates.UnorderedAccess);
            
            // Transition splits buffer from CopySource (left by AnalyzeDepth) to SRV for reading
            if (_sdsmValidFrames > 0)
            {
                commandList.ResourceBarrierTransition(_depthSplitsBuffer!,
                    ResourceStates.CopySource, ResourceStates.NonPixelShaderResource);
            }
            
            // Set compute state
            commandList.SetComputeRootSignature(_device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            commandList.SetPipelineState(_cascadeComputePSO);
            
            // Bind params cbuffer (slot 2 → register b1)
            commandList.SetComputeRootConstantBufferView(2, _cascadeParamsCBs![frameIndex].GPUVirtualAddress);
            
            // Push constants
            commandList.SetComputeRoot32BitConstant(0, _depthSplitsSRV, 0);          // SplitsBufferIdx
            commandList.SetComputeRoot32BitConstant(0, cascadeOutUAV, 1);             // CascadeOutUAVIdx
            commandList.SetComputeRoot32BitConstant(0, _lightingCascadeUAV, 2);       // LightingOutUAVIdx
            commandList.SetComputeRoot32BitConstant(0, (uint)shadowMapResolution, 3); // ShadowMapRes
            
            // NearPlane as float bits
            uint nearAsUint;
            float nearPlane = camera.NearPlane;
            nearAsUint = *(uint*)&nearPlane;
            commandList.SetComputeRoot32BitConstant(0, nearAsUint, 4);               // NearPlane
            commandList.SetComputeRoot32BitConstant(0, 4u, 5);                        // CascadeCount
            commandList.SetComputeRoot32BitConstant(0, _prevVPBufferSRV, 6);          // PrevVPBufferIdx
            commandList.SetComputeRoot32BitConstant(0, _smoothedSplitsUAV, 7);         // SmoothedSplitsIdx
            
            // Dispatch: 1 thread group, 4 threads (one per cascade)
            commandList.Dispatch(1, 1, 1);
            
            // UAV barriers for cascade and lighting outputs
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
            
            // Transition splits buffer back to CopySource (AnalyzeDepth expects this state)
            if (_sdsmValidFrames > 0)
            {
                commandList.ResourceBarrierTransition(_depthSplitsBuffer!,
                    ResourceStates.NonPixelShaderResource, ResourceStates.CopySource);
            }
            
            // Transition lighting cascade buffer to SRV for lighting pass
            commandList.ResourceBarrierTransition(_lightingCascadeBuffer!,
                ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
            
            // Copy current frame's shadow VPs to prevVP buffer for next frame
            // The cascade compute wrote shadow VPs to cascadeOutUAV — we need to
            // read them back. Since cascadeOutUAV is an upload buffer (writable from CPU),
            // this is handled by DirectionalLight after the compute runs.
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
            _cascadeComputePSO?.Dispose();
            _clearUAVHeap?.Dispose();
            _depthMinMaxBuffer?.Dispose();
            _depthHistogramBuffer?.Dispose();
            _depthSplitsBuffer?.Dispose();
            _sdsmClearBuffer?.Dispose();
            _lightingCascadeBuffer?.Dispose();
            _prevVPBuffer?.Dispose();
            
            if (_cascadeParamsCBs != null)
                for (int i = 0; i < FrameCount; i++)
                    _cascadeParamsCBs[i]?.Dispose();
            
            if (_depthSplitsReadbackBuffers != null)
                for (int i = 0; i < FrameCount; i++)
                    _depthSplitsReadbackBuffers[i]?.Dispose();
            
            for (int i = 0; i < FrameCount; i++)
            {
                _frustumConstantsBuffers[i]?.Dispose();
                _counterBuffers[i]?.Dispose();
                _rangeBuffers[i]?.Dispose();

            }
        }
    }
}
