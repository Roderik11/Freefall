using System;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Components;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;

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
        
        // Compute shaders (multi-kernel, reflection-based binding)
        private ComputeShader? _cullShader;          // cull_instances.hlsl — culling pipeline
        private ComputeShader? _pyramidShader;       // depth_pyramid.hlsl — Hi-Z pyramid
        private ComputeShader? _depthAnalysisShader; // depth_analysis.hlsl — SDSM analysis
        private ComputeShader? _cascadeShader;       // cascade_compute.hlsl — GPU cascade matrices
        
        // Cached kernel indices for cull_instances.hlsl
        private int _clearKernel, _visibilityKernel, _histogramKernel, _histogramPrefixSumKernel;
        private int _localScanKernel, _blockScanKernel, _globalScatterKernel;
        private int _bitonicSortKernel, _sortIndirectionKernel, _mainKernel;
        private int _visibilityShadowKernel, _visibilityShadow4Kernel;
        private int _expandCascadesKernel, _patchExpandedKernel;
        
        // Cached kernel indices for depth_pyramid.hlsl
        private int _spdKernel, _perMipKernel, _shadowSpdKernel, _shadowPerMipKernel;
        
        // Cached kernel indices for depth_analysis.hlsl
        private int _depthReduceKernel, _depthHistogramKernel, _computeSplitsKernel;
        
        // Cached kernel index for cascade_compute.hlsl
        private int _cascadeComputeKernel;

        
        // Frustum constants buffer per frame (just planes)
        private ID3D12Resource[] _frustumConstantsBuffers = new ID3D12Resource[FrameCount];
        
        // Atomic counter buffer (one uint per sub-batch)
        private GraphicsBuffer[] _counterBuffers = new GraphicsBuffer[FrameCount];
        
        // Instance range buffer (StartInstance, InstanceCount per sub-batch)
        private GraphicsBuffer[] _rangeBuffers = new GraphicsBuffer[FrameCount];
        

        
        // SDSM depth analysis buffers
        private GraphicsBuffer? _depthMinMaxBuffer;            // 2 uints (min, max as float bits)
        private GraphicsBuffer? _depthMinMaxInitBuffer;        // Static upload buffer with init values {MAX_FLOAT, 0}
        private GraphicsBuffer? _depthHistogramBuffer;         // 256 uints
        private GraphicsBuffer? _depthSplitsBuffer;            // 4 floats (SRV + UAV)
        private GraphicsBuffer[]? _depthSplitsReadbackBuffers;  // Per-frame readback
        private bool _sdsmInitialized;
        private int _sdsmValidFrames;                          // Frames since SDSM started producing data
        
        // GPU-driven cascade computation buffers
        private bool _cascadeComputeInitialized;
        private GraphicsBuffer? _lightingCascadeBuffer;        // LightingCascadeData[MaxCascades] (SRV + UAV)
        private GraphicsBuffer? _prevVPBuffer;                // Upload: previous frame VP matrices [MaxCascades]
        private GraphicsBuffer[]? _cascadeParamsCBs;          // Per-frame cbuffer for cascade compute params
        private GraphicsBuffer? _smoothedSplitsBuffer;         // 4 floats, persists between frames (UAV)
        
        private GraphicsDevice _device;
        private bool _initialized;
        public bool Initialized => _initialized;
        public string? InitError { get; private set; }
        
        // Cull stats readback (2 uints: [0]=visible, [1]=hi-z occluded)
        private GraphicsBuffer? _cullStatsBuffer;
        private GraphicsBuffer[]? _cullStatsReadbackBuffers;  // Per-frame readback
        
        /// <summary>Number of instances that passed both frustum and Hi-Z tests (1 frame behind).</summary>
        public int LastVisibleCount { get; private set; }
        /// <summary>Number of instances that passed frustum but were occluded by Hi-Z (1 frame behind).</summary>
        public int LastHiZOccludedCount { get; private set; }
        
        // Cached array to avoid per-frame allocation
        private ID3D12DescriptorHeap[]? _cachedSrvHeapArray;
        
        // Public PSO access for GPU-driven culling pipeline (callers use SetPipelineState directly)
        public ID3D12PipelineState? VisibilityPSO => _cullShader?.GetPSO(_visibilityKernel);
        public ID3D12PipelineState? HistogramPSO => _cullShader?.GetPSO(_histogramKernel);
        public ID3D12PipelineState? HistogramPrefixSumPSO => _cullShader?.GetPSO(_histogramPrefixSumKernel);
        public ID3D12PipelineState? LocalScanPSO => _cullShader?.GetPSO(_localScanKernel);
        public ID3D12PipelineState? BlockScanPSO => _cullShader?.GetPSO(_blockScanKernel);
        public ID3D12PipelineState? GlobalScatterPSO => _cullShader?.GetPSO(_globalScatterKernel);
        public ID3D12PipelineState? BitonicSortPSO => _cullShader?.GetPSO(_bitonicSortKernel);
        public ID3D12PipelineState? SortIndirectionPSO => _cullShader?.GetPSO(_sortIndirectionKernel);
        public ID3D12PipelineState? MainPSO => _cullShader?.GetPSO(_mainKernel);
        public ID3D12PipelineState? ClearPSO => _cullShader?.GetPSO(_clearKernel);
        public ID3D12PipelineState? VisibilityShadowPSO => _cullShader?.GetPSO(_visibilityShadowKernel);
        public ID3D12PipelineState? VisibilityShadow4PSO => _cullShader?.GetPSO(_visibilityShadow4Kernel);
        public ID3D12PipelineState? ExpandCascadesPSO => _cullShader?.GetPSO(_expandCascadesKernel);
        public ID3D12PipelineState? PatchExpandedPSO => _cullShader?.GetPSO(_patchExpandedKernel);
        public ID3D12PipelineState? DownsamplePSO => _pyramidShader?.GetPSO(_spdKernel);
        
        /// <summary>True after SDSM buffers are created and ready for dispatch.</summary>
        public bool SdsmReady => _sdsmInitialized;
        
        /// <summary>True after cascade compute PSO and buffers are ready.</summary>
        public bool CascadeComputeReady => _cascadeComputeInitialized;
        
        /// <summary>SRV index for the GPU-computed lighting cascade data (for light_directional.fx).</summary>
        public uint LightingCascadeSRV => _lightingCascadeBuffer?.SrvIndex ?? 0;

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
                // Compile all GPU-driven rendering pipeline passes via ComputeShader
                _cullShader = new ComputeShader("cull_instances.hlsl");
                _clearKernel = _cullShader.FindKernel("CSClear");
                _visibilityKernel = _cullShader.FindKernel("CSVisibility");
                _histogramKernel = _cullShader.FindKernel("CSHistogram");
                _histogramPrefixSumKernel = _cullShader.FindKernel("CSHistogramPrefixSum");
                _localScanKernel = _cullShader.FindKernel("CSLocalScan");
                _blockScanKernel = _cullShader.FindKernel("CSBlockScan");
                _globalScatterKernel = _cullShader.FindKernel("CSGlobalScatter");
                _bitonicSortKernel = _cullShader.FindKernel("CSBitonicSort");
                _sortIndirectionKernel = _cullShader.FindKernel("CSSortIndirection");
                _mainKernel = _cullShader.FindKernel("CSMain");
                _visibilityShadowKernel = _cullShader.FindKernel("CSVisibilityShadow");
                _visibilityShadow4Kernel = _cullShader.FindKernel("CSVisibilityShadow4");
                _expandCascadesKernel = _cullShader.FindKernel("CSExpandCascades");
                _patchExpandedKernel = _cullShader.FindKernel("CSPatchExpandedCounts");
                
                // Hi-Z depth pyramid downsampler
                _pyramidShader = new ComputeShader("depth_pyramid.hlsl");
                _spdKernel = _pyramidShader.FindKernel("CSSinglePassDownsample");
                _perMipKernel = _pyramidShader.FindKernel("CSDownsample");
                _shadowSpdKernel = _pyramidShader.FindKernel("CSSinglePassDownsampleShadow");
                _shadowPerMipKernel = _pyramidShader.FindKernel("CSDownsampleShadow");
                Debug.Log("GPUCuller", "depth_pyramid.hlsl compiled: 4 kernels");
                
                // SDSM depth analysis
                _depthAnalysisShader = new ComputeShader("depth_analysis.hlsl");
                _depthReduceKernel = _depthAnalysisShader.FindKernel("CSDepthReduce");
                _depthHistogramKernel = _depthAnalysisShader.FindKernel("CSDepthHistogram");
                _computeSplitsKernel = _depthAnalysisShader.FindKernel("CSComputeSplits");
                
                // Create SDSM GPU buffers
                _depthMinMaxBuffer = GraphicsBuffer.CreateRaw(2, uav: true, clearable: true);
                // Static upload buffer with init values: min=MAX_FLOAT (0x7F7FFFFF), max=0
                _depthMinMaxInitBuffer = GraphicsBuffer.CreateUpload<uint>(2, mapped: true);
                unsafe
                {
                    var pInit = _depthMinMaxInitBuffer.WritePtr<uint>();
                    pInit[0] = 0x7F7FFFFF;  // min = MAX_FLOAT
                    pInit[1] = 0;           // max = 0
                }
                _depthHistogramBuffer = GraphicsBuffer.CreateStructured<uint>(256, uav: true);
                _depthSplitsBuffer = GraphicsBuffer.CreateStructured<float>(4, srv: true, uav: true);
                
                // Readback buffers: one per frame
                _depthSplitsReadbackBuffers = new GraphicsBuffer[FrameCount];
                for (int rb = 0; rb < FrameCount; rb++)
                    _depthSplitsReadbackBuffers[rb] = GraphicsBuffer.CreateReadback<float>(4);
                
                _sdsmInitialized = true;
                Debug.Log("GPUCuller", "SDSM depth analysis initialized via ComputeShader: 3 kernels");
                
                // GPU-driven cascade matrix computation (depends on SDSM being initialized)
                if (_sdsmInitialized)
                {
                    _cascadeShader = new ComputeShader("cascade_compute.hlsl");
                    _cascadeComputeKernel = _cascadeShader.FindKernel("CSComputeCascadeMatrices");
                    
                    int maxCascades = 8;
                    int lightingStride = Marshal.SizeOf<LightingCascadeData>();
                    int vpStride = Marshal.SizeOf<Matrix4x4>();
                    
                    // Lighting cascade output buffer (GPU default, compute writes)
                    _lightingCascadeBuffer = GraphicsBuffer.CreateStructured(maxCascades, lightingStride, srv: true, uav: true);
                    
                    // Previous VP buffer (upload with SRV, persistently mapped)
                    _prevVPBuffer = GraphicsBuffer.CreateUpload<Matrix4x4>(maxCascades, mapped: true);
                    unsafe
                    {
                        // Initialize to identity matrices
                        var pData = _prevVPBuffer.WritePtr<Matrix4x4>();
                        for (int m = 0; m < maxCascades; m++)
                            pData[m] = Matrix4x4.Identity;
                    }
                    
                    // Cascade compute params cbuffer (per-frame, 256-byte aligned)
                    _cascadeParamsCBs = new GraphicsBuffer[FrameCount];
                    for (int f = 0; f < FrameCount; f++)
                        _cascadeParamsCBs[f] = GraphicsBuffer.CreateConstantBuffer<CascadeComputeParams>();
                    
                    // Smoothed splits buffer (GPU default, persistent between frames)
                    _smoothedSplitsBuffer = GraphicsBuffer.CreateStructured<float>(4, uav: true);
                    
                    // DISABLED: GPU cascade compute produces incorrect VP matrices for lighting.
                    // _cascadeComputeInitialized = true;
                    Debug.Log("GPUCuller", "cascade_compute.hlsl compiled via ComputeShader: 1 kernel");
                }
                
                Debug.Log("GPUCuller", "Compute shaders compiled: CSClear + CSVisibility + CSHistogram + CSLocalScan + CSBlockScan + CSGlobalScatter + CSBitonicSort + CSSortIndirection + CSHistogramPrefixSum + CSMain + CSVisibilityShadow");
                
                // Create per-frame frustum constant buffers
                int frustumConstantsSize = (Marshal.SizeOf<FrustumConstants>() + 255) & ~255; // 256-byte aligned
                
                for (int i = 0; i < FrameCount; i++)
                {
                    _frustumConstantsBuffers[i] = _device.CreateUploadBuffer(frustumConstantsSize);
                    
                    // Counter buffer: one uint per sub-batch (GPU writable, clearable for ClearUAV)
                    _counterBuffers[i] = GraphicsBuffer.CreateStructured<uint>(MaxSubBatches, uav: true);
                    
                    // Range buffer: StartInstance + InstanceCount per sub-batch (upload for CPU write)
                    int rangeStride = Marshal.SizeOf<InstanceRange>();
                    _rangeBuffers[i] = GraphicsBuffer.CreateUpload<InstanceRange>(MaxSubBatches);
                    
                    Debug.Log("GPUCuller", $"Frame {i}: CounterBufferUAV={_counterBuffers[i].UavIndex}, RangeBufferSRV={_rangeBuffers[i].SrvIndex}");
                }
                
                // --- Cull stats buffer (2 uints: visible, hi-z occluded) ---
                _cullStatsBuffer = GraphicsBuffer.CreateRaw(2, uav: true, clearable: true);
                
                // Readback buffers (per-frame)
                _cullStatsReadbackBuffers = new GraphicsBuffer[FrameCount];
                for (int rb = 0; rb < FrameCount; rb++)
                    _cullStatsReadbackBuffers[rb] = GraphicsBuffer.CreateReadback<uint>(2);
                Debug.Log("GPUCuller", $"Cull stats buffer created: UAV={_cullStatsBuffer.UavIndex}");
                
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
            
            _counterBuffers[frameIndex].ClearUAV(commandList, new Vortice.Mathematics.Int4(0, 0, 0, 0));
            
            // UAV barrier - ensure counters are zeroed before any CSCount runs
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
        }
        
        /// <summary>UAV index for cull stats buffer (passed in FrustumConstants).</summary>
        public uint CullStatsUAV => _cullStatsBuffer?.UavIndex ?? 0;
        
        /// <summary>
        /// Clear cull stats buffer and read back previous frame's results.
        /// Call once per frame before the batch loop.
        /// </summary>
        public void ClearCullStats(ID3D12GraphicsCommandList commandList)
        {
            if (!_initialized || _cullStatsBuffer == null) return;
            
            // Read back N-2 frame's stats (safely past GPU completion)
            int readbackFrame = (Engine.FrameIndex + 1) % FrameCount;
            if (_cullStatsReadbackBuffers != null)
            {
                unsafe
                {
                    uint* pStats = _cullStatsReadbackBuffers[readbackFrame].ReadPtr<uint>();
                    LastVisibleCount = (int)pStats[0];
                    LastHiZOccludedCount = (int)pStats[1];
                }
            }
            
            // Clear stats buffer for this frame
            _cullStatsBuffer.ClearUAV(commandList, new Vortice.Mathematics.Int4(0, 0, 0, 0));
            
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
                new ResourceTransitionBarrier(_cullStatsBuffer.Native, 
                    ResourceStates.UnorderedAccess, ResourceStates.CopySource)));
            
            commandList.CopyResource(_cullStatsReadbackBuffers[frameIndex].Native, _cullStatsBuffer.Native);
            
            // Transition back to UAV for next frame
            commandList.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_cullStatsBuffer.Native, 
                    ResourceStates.CopySource, ResourceStates.UnorderedAccess)));
        }

        // NOTE: DispatchCull, DispatchShadowVisibility, DispatchPassThrough,
        // BeginCommandGeneration, EndCommandGeneration were removed (dead code).
        // InstanceBatch and SceneCuller drive the culling pipeline directly.


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
            if (_pyramidShader == null || pyramid?.Texture == null || pyramid.MipCount == 0) return;
            
            bool firstGeneration = !pyramid.Ready;
            pyramid.Ready = true;  // Pyramid will be valid after this dispatch
            
            commandList.SetComputeRootSignature(_device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            
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
                var ps = _pyramidShader!;
                ps.SetPushConstant(_perMipKernel, "SourceSrv", inputSrv);
                ps.SetPushConstant(_perMipKernel, "CounterUAV", pyramid.MipUAVs[mip]);
                ps.SetPushConstant(_perMipKernel, "SourceWidth", (uint)w);
                ps.SetPushConstant(_perMipKernel, "SourceHeight", (uint)h);
                
                ps.Dispatch(_perMipKernel, commandList, ((uint)w + 7) / 8, ((uint)h + 7) / 8, 1);
                
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
            if (!_initialized || _pyramidShader == null || pyramid?.Texture == null) return;

            bool firstGeneration = !pyramid.Ready;
            pyramid.Ready = true;

            int sourceW = pyramid.SourceWidth;
            int sourceH = pyramid.SourceHeight;
            int spdMipCount = Math.Min(pyramid.MipCount, 6);
            uint numGroupsX = ((uint)sourceW + 63) / 64;
            uint numGroupsY = ((uint)sourceH + 63) / 64;

            commandList.SetComputeRootSignature(_device.GlobalRootSignature);

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

                // Push constants via ComputeShader API
                var ps = _pyramidShader!;
                ps.SetPushConstant(_shadowSpdKernel, "SourceSrv", shadowArraySrvIndex);
                ps.SetPushConstant(_shadowSpdKernel, "CounterUAV", 0u);
                ps.SetPushConstant(_shadowSpdKernel, "SourceWidth", (uint)sourceW);
                ps.SetPushConstant(_shadowSpdKernel, "SourceHeight", (uint)sourceH);
                ps.SetParam(_shadowSpdKernel, "MipCount", (uint)spdMipCount);
                ps.SetParam(_shadowSpdKernel, "NumGroupsX", numGroupsX);
                ps.SetParam(_shadowSpdKernel, "NumGroupsY", numGroupsY);
                // Mip UAVs
                ps.SetPushConstant(_shadowSpdKernel, "Mip0UAV", pyramid.GetMipUAV(slice, 0));
                for (int m = 1; m < spdMipCount && m <= 12; m++)
                    ps.SetPushConstant(_shadowSpdKernel, $"Mip{m}UAV", pyramid.GetMipUAV(slice, m));
                ps.SetPushConstant(_shadowSpdKernel, "SliceIndex", (uint)slice);

                ps.Dispatch(_shadowSpdKernel, commandList, numGroupsX, numGroupsY, 1);

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
                if (pyramid.MipCount > 6 && _pyramidShader != null)
                {
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

                        ps.SetPushConstant(_shadowPerMipKernel, "SourceSrv", pyramid.GetMipSRV(slice, mip - 1));
                        ps.SetPushConstant(_shadowPerMipKernel, "CounterUAV", pyramid.GetMipUAV(slice, mip));
                        ps.SetPushConstant(_shadowPerMipKernel, "SourceWidth", (uint)mw);
                        ps.SetPushConstant(_shadowPerMipKernel, "SourceHeight", (uint)mh);

                        ps.Dispatch(_shadowPerMipKernel, commandList, ((uint)mw + 7) / 8, ((uint)mh + 7) / 8, 1);

                        commandList.ResourceBarrier(new ResourceBarrier(
                            new ResourceTransitionBarrier(pyramid.Texture,
                                ResourceStates.UnorderedAccess,
                                ResourceStates.NonPixelShaderResource, sub)));
                    }
                }
            }
        }

        /// <summary>
        /// Dispatch SDSM depth analysis compute shaders.
        /// Call after Hi-Z generation, when DepthGBuffer is in NonPixelShaderResource state.
        /// </summary>
        public void AnalyzeDepth(ID3D12GraphicsCommandList commandList, uint depthSrvIndex, int texWidth, int texHeight, float nearPlane, float farPlane)
        {
            if (!_sdsmInitialized || _depthAnalysisShader == null) return;
            
            int frameIndex = Engine.FrameIndex % FrameCount;
            
            commandList.SetComputeRootSignature(_device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            
            // Clear SDSM buffers
            var minMaxFrom = _sdsmValidFrames == 0 ? ResourceStates.Common : ResourceStates.CopySource;
            var histFrom = _sdsmValidFrames == 0 ? ResourceStates.Common : ResourceStates.UnorderedAccess;
            var splitsFrom = _sdsmValidFrames == 0 ? ResourceStates.Common : ResourceStates.CopySource;
            
            // MinMax buffer needs two different values {MAX_FLOAT, 0} — can't use ClearUAV (fills all elements with first component)
            commandList.ResourceBarrierTransition(_depthMinMaxBuffer!.Native, minMaxFrom, ResourceStates.CopyDest);
            commandList.CopyBufferRegion(_depthMinMaxBuffer!.Native, 0, _depthMinMaxInitBuffer!.Native, 0, 8);
            commandList.ResourceBarrierTransition(_depthMinMaxBuffer!.Native, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
            
            // Histogram: uniform zero clear — ClearUAV is perfect for this
            if (histFrom != ResourceStates.UnorderedAccess)
                commandList.ResourceBarrierTransition(_depthHistogramBuffer!.Native, histFrom, ResourceStates.UnorderedAccess);
            _depthHistogramBuffer!.ClearUAV(commandList, new Int4(0, 0, 0, 0));
            
            commandList.ResourceBarrierTransition(_depthSplitsBuffer!.Native, splitsFrom, ResourceStates.UnorderedAccess);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
            
            // Set constants via ComputeShader API
            var da = _depthAnalysisShader!;
            // Push constants (resource indices)
            da.SetPushConstant("DepthTex", depthSrvIndex);
            da.SetBuffer("MinMaxUAV", _depthMinMaxBuffer!);
            da.SetBuffer("HistogramUAV", _depthHistogramBuffer!);
            da.SetBuffer("SplitsUAV", _depthSplitsBuffer!);
            // Params cbuffer
            da.SetParam("TexWidth", (uint)texWidth);
            da.SetParam("TexHeight", (uint)texHeight);
            da.SetParam("NearPlane", nearPlane);
            da.SetParam("FarPlane", farPlane);
            
            uint groupsX = ((uint)texWidth + 15) / 16;
            uint groupsY = ((uint)texHeight + 15) / 16;
            
            // --- Pass 1: CSDepthReduce ---
            da.Dispatch(_depthReduceKernel, commandList, groupsX, groupsY, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
            
            // --- Pass 2: CSDepthHistogram ---
            da.Dispatch(_depthHistogramKernel, commandList, groupsX, groupsY, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
            
            // --- Pass 3: CSComputeSplits ---
            da.Dispatch(_computeSplitsKernel, commandList, 1, 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
            
            // Step 4: Transition splits + minmax to CopySource for readback
            commandList.ResourceBarrierTransition(_depthSplitsBuffer!.Native,
                ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            commandList.ResourceBarrierTransition(_depthMinMaxBuffer!.Native,
                ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            // Histogram stays in UAV (next frame will transition it to CopyDest)
            
            // Step 5: Copy to readback buffer for this frame
            commandList.CopyBufferRegion(
                _depthSplitsReadbackBuffers![frameIndex].Native, 0,
                _depthSplitsBuffer!.Native, 0,
                4 * sizeof(float));
            
            _sdsmValidFrames++;
        }
        
        /// <summary>
        /// Read the adaptive cascade splits computed by the previous frame's depth analysis.
        /// Returns null if no valid data is available yet.
        /// </summary>
        public unsafe float[]? ReadAdaptiveSplits()
        {
            if (!_sdsmInitialized || _depthSplitsReadbackBuffers == null || _sdsmValidFrames < 2)
                return null;
            
            // Read from the PREVIOUS frame's readback buffer (guaranteed complete)
            int readFrame = (Engine.FrameIndex - 1 + FrameCount) % FrameCount;
            float* pSplits = _depthSplitsReadbackBuffers[readFrame].ReadPtr<float>();
            
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
            if (!_cascadeComputeInitialized || _cascadeShader == null) return;
            
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
            *_cascadeParamsCBs![frameIndex].WritePtr<CascadeComputeParams>() = cbParams;
            
            // Transition lighting cascade buffer to UAV for compute write
            // (On first use it's Common, subsequent uses it's NonPixelShaderResource)
            var lightingFromState = _sdsmValidFrames > 1 ? ResourceStates.NonPixelShaderResource : ResourceStates.Common;
            commandList.ResourceBarrierTransition(_lightingCascadeBuffer!.Native,
                lightingFromState, ResourceStates.UnorderedAccess);
            
            // Transition splits buffer from CopySource (left by AnalyzeDepth) to SRV for reading
            if (_sdsmValidFrames > 0)
            {
                commandList.ResourceBarrierTransition(_depthSplitsBuffer!.Native,
                    ResourceStates.CopySource, ResourceStates.NonPixelShaderResource);
            }
            
            // Set compute state
            commandList.SetComputeRootSignature(_device.GlobalRootSignature);
            _cachedSrvHeapArray ??= new[] { _device.SrvHeap };
            commandList.SetDescriptorHeaps(1, _cachedSrvHeapArray);
            
            // Bind params cbuffer (slot 2 → register b1, externally managed)
            commandList.SetComputeRootConstantBufferView(2, _cascadeParamsCBs![frameIndex].Native.GPUVirtualAddress);
            
            // Push constants via ComputeShader API
            var cs = _cascadeShader!;
            cs.SetSRV(_cascadeComputeKernel, "SplitsBuffer", _depthSplitsBuffer!);
            cs.SetPushConstant(_cascadeComputeKernel, "CascadeOutUAV", cascadeOutUAV);
            cs.SetUAV(_cascadeComputeKernel, "LightingOutUAV", _lightingCascadeBuffer!);
            cs.SetParam(_cascadeComputeKernel, "ShadowMapRes", (uint)shadowMapResolution);
            cs.SetParam(_cascadeComputeKernel, "NearPlane", camera.NearPlane);
            cs.SetParam(_cascadeComputeKernel, "CascadeCount", 4u);
            cs.SetPushConstant(_cascadeComputeKernel, "PrevVPBuffer", _prevVPBuffer!.SrvIndex);
            cs.SetUAV(_cascadeComputeKernel, "SmoothedSplits", _smoothedSplitsBuffer!);
            
            // Dispatch: 1 thread group, 4 threads (one per cascade)
            cs.Dispatch(_cascadeComputeKernel, commandList, 1, 1, 1);
            
            // UAV barriers for cascade and lighting outputs
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
            
            // Transition splits buffer back to CopySource (AnalyzeDepth expects this state)
            if (_sdsmValidFrames > 0)
            {
                commandList.ResourceBarrierTransition(_depthSplitsBuffer!.Native,
                    ResourceStates.NonPixelShaderResource, ResourceStates.CopySource);
            }
            
            // Transition lighting cascade buffer to SRV for lighting pass
            commandList.ResourceBarrierTransition(_lightingCascadeBuffer!.Native,
                ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
            
            // Copy current frame's shadow VPs to prevVP buffer for next frame
            // The cascade compute wrote shadow VPs to cascadeOutUAV — we need to
            // read them back. Since cascadeOutUAV is an upload buffer (writable from CPU),
            // this is handled by DirectionalLight after the compute runs.
        }

        public void Dispose()
        {
            _cullShader?.Dispose();
            _pyramidShader?.Dispose();
            _depthAnalysisShader?.Dispose();
            _cascadeShader?.Dispose();
            _cullStatsBuffer?.Dispose();
            _depthMinMaxBuffer?.Dispose();
            _depthMinMaxInitBuffer?.Dispose();
            _depthHistogramBuffer?.Dispose();
            _depthSplitsBuffer?.Dispose();

            _lightingCascadeBuffer?.Dispose();
            _prevVPBuffer?.Dispose();
            
            if (_cascadeParamsCBs != null)
                for (int i = 0; i < FrameCount; i++)
                    _cascadeParamsCBs[i]?.Dispose();
            
            if (_depthSplitsReadbackBuffers != null)
                for (int i = 0; i < FrameCount; i++)
                    _depthSplitsReadbackBuffers[i]?.Dispose();
            
            if (_cullStatsReadbackBuffers != null)
                for (int i = 0; i < FrameCount; i++)
                    _cullStatsReadbackBuffers[i]?.Dispose();
            
            for (int i = 0; i < FrameCount; i++)
            {
                _frustumConstantsBuffers[i]?.Dispose();
                _counterBuffers[i]?.Dispose();
                _rangeBuffers[i]?.Dispose();
            }
        }
    }
}
