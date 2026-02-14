using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Freefall.Graphics;
using Freefall.Base;

namespace Freefall.Components
{
    /// <summary>
    /// GPU-driven quadtree terrain. A single compute dispatch evaluates the entire
    /// quadtree and produces per-patch data directly into InstanceBatch-compatible buffers.
    /// The compute dispatch runs as a custom action on the renderer's command list.
    /// </summary>
    public class GPUTerrain : Freefall.Base.Component, IDraw, IHeightProvider
    {
        public static bool ComputeReady { get; set; }
        public static string ComputeError { get; set; } = "";

        // ───── Public Parameters ──────────────────────────────────────────
        public Texture Heightmap = null!;
        public Material Material = null!;
        public Vector2 TerrainSize = new Vector2(1700, 1700);
        public float MaxHeight = 600;
        public float[,]? HeightField;
        public float PixelErrorThreshold = 2.0f;
        public int MaxDepth = 7;
        public int MaxPatches = 8192;
        public Texture?[] ControlMaps = new Texture?[4];
        public List<TextureLayer>? Layers;

        public class TextureLayer
        {
            public Texture Diffuse;
            public Texture Normals;
            public Vector2 Tiling = Vector2.One;
        }

        private Vector4[] _layerTiling = new Vector4[32];

        // ───── Internal State ─────────────────────────────────────────────
        private const int FrameCount = 3;

        // Compute pipeline — three-pass restricted quadtree + one-time mip pyramid builder
        private ID3D12PipelineState _buildMinMaxMipPSO = null!;
        private ID3D12PipelineState _markSplitsPSO = null!;
        private ID3D12PipelineState _emitLeavesPSO = null!;
        private ID3D12PipelineState _buildDrawArgsPSO = null!;
        private bool _computeInitialized;

        // GPU output buffers — per-frame
        private ID3D12Resource[] _descriptorBuffers = new ID3D12Resource[FrameCount];
        private ID3D12Resource[] _sphereBuffers = new ID3D12Resource[FrameCount];
        private ID3D12Resource[] _subbatchIdBuffers = new ID3D12Resource[FrameCount];
        private ID3D12Resource[] _terrainDataBuffers = new ID3D12Resource[FrameCount];
        private ID3D12Resource[] _counterBuffers = new ID3D12Resource[FrameCount];
        private ID3D12Resource[] _splitFlagsBuffers = new ID3D12Resource[FrameCount];
        private ID3D12Resource[] _indirectArgsBuffers = new ID3D12Resource[FrameCount];
        private uint[] _indirectArgsUAVs = new uint[FrameCount];
        
        // Frustum + Hi-Z constant buffers per frame (for compute shader culling)
        private ID3D12Resource[] _frustumConstantBuffers = new ID3D12Resource[FrameCount];
        private Matrix4x4 _previousFrameViewProjection;
        
        // Height range mip pyramid (one-time, not per-frame)
        private ID3D12Resource _heightRangePyramid = null!;
        private uint[] _heightRangeMipUAVs = null!;
        private uint[] _heightRangeMipSRVs = null!;
        private uint _heightRangePyramidSRV;
        private int _heightRangeMipCount;
        private bool _heightRangePyramidBuilt;

        // UAV indices (for compute write)
        private uint[] _descriptorUAVs = new uint[FrameCount];
        private uint[] _sphereUAVs = new uint[FrameCount];
        private uint[] _subbatchIdUAVs = new uint[FrameCount];
        private uint[] _terrainDataUAVs = new uint[FrameCount];
        private uint[] _counterUAVs = new uint[FrameCount];
        private uint[] _splitFlagsUAVs = new uint[FrameCount];

        // SRV indices (for culler/draw read)
        private uint[] _descriptorSRVs = new uint[FrameCount];
        private uint[] _sphereSRVs = new uint[FrameCount];
        private uint[] _subbatchIdSRVs = new uint[FrameCount];
        private uint[] _terrainDataSRVs = new uint[FrameCount];

        // CPU descriptor handles for UAV clears
        private ID3D12DescriptorHeap _cpuHeap = null!;
        private CpuDescriptorHandle[] _counterCPUHandles = new CpuDescriptorHandle[FrameCount];
        private CpuDescriptorHandle[] _splitFlagsCPUHandles = new CpuDescriptorHandle[FrameCount];

        // Mesh and registration
        private Mesh _patchMesh = null!;
        private int _meshPartId;
        private int _transformSlot = -1;

        // Texture arrays
        private Texture? ControlMapsArray;
        private Texture? DiffuseMapsArray;
        private Texture? NormalMapsArray;

        private int _totalNodes;
        private bool[] _buffersInSRV = new bool[FrameCount]; // track per-frame state
        private bool[] _indirectArgsInArgState = new bool[FrameCount]; // indirect args in IndirectArgument state

        // ───── Lifecycle ──────────────────────────────────────────────────

        protected override void Awake()
        {
            _patchMesh = Mesh.CreatePatch(Engine.Device);
            _meshPartId = MeshRegistry.Register(_patchMesh, 0);
            _transformSlot = TransformBuffer.Instance!.AllocateSlot();

            OnTransformChanged();
            Entity.Transform.OnChanged += OnTransformChanged;

            if (Layers != null && Layers.Count > 0)
                SetupLayerTiling();

            _totalNodes = CalculateTotalNodes(MaxDepth);
            try
            {
                InitializeCompute();
            }
            catch (Exception ex)
            {
                ComputeError = ex.Message;
                Debug.LogError("[GPUTerrain]", $"Failed to initialize compute: {ex.Message}");
            }

            Debug.Log($"[GPUTerrain] MaxDepth={MaxDepth} TotalNodes={_totalNodes} MaxPatches={MaxPatches} ComputeInit={_computeInitialized}");
            ComputeReady = _computeInitialized;
        }

        public override void Destroy()
        {
            Entity.Transform.OnChanged -= OnTransformChanged;
        }

        private void OnTransformChanged()
        {
            TransformBuffer.Instance!.SetTransform(_transformSlot, Transform.WorldMatrix);
        }

        public void Draw()
        {
            if (Camera.Main == null || !_computeInitialized) return;

            int frameIndex = Engine.FrameIndex % FrameCount;

            // Set shared material params
            Material.SetParameter("CameraPos", Camera.Main.Position);
            Material.SetParameter("HeightTexel", Heightmap != null ? 1.0f / Heightmap.Native.Description.Width : 1.0f / 1024.0f);
            Material.SetParameter("MaxHeight", MaxHeight);
            Material.SetParameter("TerrainSize", TerrainSize);
            Material.SetParameter("TerrainOrigin", new Vector2(Transform.Position.X, Transform.Position.Z));

            Material.SetParameter("LayerTiling", _layerTiling);

            // Bind textures
            if (Heightmap != null) Material.SetTexture("HeightTex", Heightmap);
            if (ControlMapsArray != null) Material.SetTexture("ControlMaps", ControlMapsArray);
            if (DiffuseMapsArray != null) Material.SetTexture("DiffuseMaps", DiffuseMapsArray);
            if (NormalMapsArray != null) Material.SetTexture("NormalMaps", NormalMapsArray);

            // Capture values for lambda closure
            int fi = frameIndex;
            var self = this;

            // Enqueue compute dispatch as custom action (runs first in Execute, before batch processing)
            CommandBuffer.Enqueue(RenderPass.Opaque, (list) => self.DispatchQuadtreeEval(list, fi));

            // Enqueue self-draw action: terrain draws itself via ExecuteIndirect
            // This runs during the draw phase (after batches are drawn)
            CommandBuffer.Enqueue(RenderPass.Opaque, (list) => self.DrawTerrain(list, fi));
        }

        // ───── Compute Pipeline ───────────────────────────────────────────

        private void InitializeCompute()
        {
            var device = Engine.Device;

            string shaderPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Resources", "Shaders", "terrain_quadtree.hlsl");
            if (!File.Exists(shaderPath))
            {
                Debug.LogError("[GPUTerrain]", $"Compute shader not found: {shaderPath}");
                return;
            }

            string source = File.ReadAllText(shaderPath);

            // Pass 1: Mark splits + enforce restricted quadtree
            var markShader = new Shader(source, "CSMarkSplits", "cs_6_6");
            _markSplitsPSO = device.CreateComputePipelineState(markShader.Bytecode);
            markShader.Dispose();

            // Pass 2: Emit leaves using restriction-enforced split flags
            var emitShader = new Shader(source, "CSEmitLeaves", "cs_6_6");
            _emitLeavesPSO = device.CreateComputePipelineState(emitShader.Bytecode);
            emitShader.Dispose();

            // Pass 3: Build DrawInstanced indirect args from counter
            var buildDrawArgsShader = new Shader(source, "CSBuildDrawArgs", "cs_6_6");
            _buildDrawArgsPSO = device.CreateComputePipelineState(buildDrawArgsShader.Bytecode);
            buildDrawArgsShader.Dispose();

            // One-time: Build height range mip pyramid
            var buildMinMaxMipShader = new Shader(source, "CSBuildMinMaxMip", "cs_6_6");
            _buildMinMaxMipPSO = device.CreateComputePipelineState(buildMinMaxMipShader.Bytecode);
            buildMinMaxMipShader.Dispose();

            // CPU heap for UAV clears
            var cpuHeapDesc = new DescriptorHeapDescription
            {
                Type = DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView,
                DescriptorCount = (uint)(FrameCount * 2), // counter + splitFlags per frame
                Flags = DescriptorHeapFlags.None
            };
            _cpuHeap = device.NativeDevice.CreateDescriptorHeap(cpuHeapDesc);
            uint incSize = device.NativeDevice.GetDescriptorHandleIncrementSize(
                DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView);

            for (int i = 0; i < FrameCount; i++)
            {
                CreateFrameBuffers(device, i, incSize);
            }

            // Create frustum constant buffers (for compute shader Hi-Z culling)
            int frustumCBSize = Marshal.SizeOf<GPUCuller.FrustumConstants>();
            for (int i = 0; i < FrameCount; i++)
                _frustumConstantBuffers[i] = device.CreateUploadBuffer(frustumCBSize);

            // Create height range mip pyramid texture
            CreateHeightRangePyramid(device);

            _computeInitialized = true;
        }

        private void CreateFrameBuffers(GraphicsDevice device, int i, uint incSize)
        {
            // InstanceDescriptor: 12 bytes
            _descriptorBuffers[i] = device.CreateDefaultBuffer(MaxPatches * 12);
            _descriptorUAVs[i] = device.AllocateBindlessIndex();
            device.CreateStructuredBufferUAV(_descriptorBuffers[i], (uint)MaxPatches, 12, _descriptorUAVs[i]);
            _descriptorSRVs[i] = device.AllocateBindlessIndex();
            device.CreateStructuredBufferSRV(_descriptorBuffers[i], (uint)MaxPatches, 12, _descriptorSRVs[i]);

            // BoundingSphere: 16 bytes
            _sphereBuffers[i] = device.CreateDefaultBuffer(MaxPatches * 16);
            _sphereUAVs[i] = device.AllocateBindlessIndex();
            device.CreateStructuredBufferUAV(_sphereBuffers[i], (uint)MaxPatches, 16, _sphereUAVs[i]);
            _sphereSRVs[i] = device.AllocateBindlessIndex();
            device.CreateStructuredBufferSRV(_sphereBuffers[i], (uint)MaxPatches, 16, _sphereSRVs[i]);

            // SubbatchId: 4 bytes
            _subbatchIdBuffers[i] = device.CreateDefaultBuffer(MaxPatches * 4);
            _subbatchIdUAVs[i] = device.AllocateBindlessIndex();
            device.CreateStructuredBufferUAV(_subbatchIdBuffers[i], (uint)MaxPatches, 4, _subbatchIdUAVs[i]);
            _subbatchIdSRVs[i] = device.AllocateBindlessIndex();
            device.CreateStructuredBufferSRV(_subbatchIdBuffers[i], (uint)MaxPatches, 4, _subbatchIdSRVs[i]);

            // TerrainPatchData: 32 bytes
            _terrainDataBuffers[i] = device.CreateDefaultBuffer(MaxPatches * 32);
            _terrainDataUAVs[i] = device.AllocateBindlessIndex();
            device.CreateStructuredBufferUAV(_terrainDataBuffers[i], (uint)MaxPatches, 32, _terrainDataUAVs[i]);
            _terrainDataSRVs[i] = device.AllocateBindlessIndex();
            device.CreateStructuredBufferSRV(_terrainDataBuffers[i], (uint)MaxPatches, 32, _terrainDataSRVs[i]);

            // Counter: raw uint for atomic append
            _counterBuffers[i] = device.CreateDefaultBuffer(4);
            _counterUAVs[i] = device.AllocateBindlessIndex();
            var rawUavDesc = new UnorderedAccessViewDescription
            {
                Format = Format.R32_Typeless,
                ViewDimension = UnorderedAccessViewDimension.Buffer,
                Buffer = new BufferUnorderedAccessView
                {
                    FirstElement = 0,
                    NumElements = 1,
                    Flags = BufferUnorderedAccessViewFlags.Raw,
                }
            };
            device.NativeDevice.CreateUnorderedAccessView(
                _counterBuffers[i], null, rawUavDesc, device.GetCpuHandle(_counterUAVs[i]));

            var cpuHandle = _cpuHeap.GetCPUDescriptorHandleForHeapStart() + ((int)(i * incSize));
            device.NativeDevice.CreateUnorderedAccessView(
                _counterBuffers[i], null, rawUavDesc, cpuHandle);
            _counterCPUHandles[i] = cpuHandle;

            // Indirect args buffer: 16 bytes for DrawInstancedArguments
            _indirectArgsBuffers[i] = device.CreateDefaultBuffer(16);
            _indirectArgsUAVs[i] = device.AllocateBindlessIndex();
            var argsUavDesc = new UnorderedAccessViewDescription
            {
                Format = Format.R32_Typeless,
                ViewDimension = UnorderedAccessViewDimension.Buffer,
                Buffer = new BufferUnorderedAccessView
                {
                    FirstElement = 0,
                    NumElements = 4,
                    Flags = BufferUnorderedAccessViewFlags.Raw,
                }
            };
            device.NativeDevice.CreateUnorderedAccessView(
                _indirectArgsBuffers[i], null, argsUavDesc, device.GetCpuHandle(_indirectArgsUAVs[i]));

            // SplitFlags buffer: 1 uint per node for atomic writes
            uint splitFlagsSize = (uint)(_totalNodes * 4);
            _splitFlagsBuffers[i] = device.CreateDefaultBuffer((int)splitFlagsSize);
            _splitFlagsUAVs[i] = device.AllocateBindlessIndex();
            var splitFlagsUavDesc = new UnorderedAccessViewDescription
            {
                Format = Format.R32_Typeless,
                ViewDimension = UnorderedAccessViewDimension.Buffer,
                Buffer = new BufferUnorderedAccessView
                {
                    FirstElement = 0,
                    NumElements = (uint)_totalNodes,
                    Flags = BufferUnorderedAccessViewFlags.Raw,
                }
            };
            device.NativeDevice.CreateUnorderedAccessView(
                _splitFlagsBuffers[i], null, splitFlagsUavDesc, device.GetCpuHandle(_splitFlagsUAVs[i]));

            // CPU handle for splitFlags clear
            var splitFlagsCpuHandle = _cpuHeap.GetCPUDescriptorHandleForHeapStart() + ((int)((FrameCount + i) * incSize));
            device.NativeDevice.CreateUnorderedAccessView(
                _splitFlagsBuffers[i], null, splitFlagsUavDesc, splitFlagsCpuHandle);
            _splitFlagsCPUHandles[i] = splitFlagsCpuHandle;

        }

        /// <summary>
        /// Dispatches the quadtree evaluation compute shader on the renderer's command list.
        /// Called as a custom action before batch processing in Pass.Execute.
        /// </summary>
        private void DispatchQuadtreeEval(ID3D12GraphicsCommandList commandList, int frameIndex)
        {
            var device = Engine.Device;

            // If buffers were left in SRV state from the previous frame, transition back to UAV
            if (_buffersInSRV[frameIndex])
            {
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceTransitionBarrier(_descriptorBuffers[frameIndex], ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess)));
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceTransitionBarrier(_sphereBuffers[frameIndex], ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess)));
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceTransitionBarrier(_subbatchIdBuffers[frameIndex], ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess)));
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceTransitionBarrier(_terrainDataBuffers[frameIndex], ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess)));
                _buffersInSRV[frameIndex] = false;
            }

            // Transition indirect args back to UAV if it was used as IndirectArgument last frame
            if (_indirectArgsInArgState[frameIndex])
            {
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceTransitionBarrier(_indirectArgsBuffers[frameIndex], ResourceStates.IndirectArgument, ResourceStates.UnorderedAccess)));
                _indirectArgsInArgState[frameIndex] = false;
            }

            // Set compute root signature and descriptor heap
            commandList.SetComputeRootSignature(device.GlobalRootSignature);
            commandList.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            // ── Upload frustum + Hi-Z constants for compute shader culling ──
            UploadFrustumConstants(frameIndex);
            commandList.SetComputeRootConstantBufferView(1, _frustumConstantBuffers[frameIndex].GPUVirtualAddress);

            // Clear counter to zero
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(_counterUAVs[frameIndex]),
                _counterCPUHandles[frameIndex],
                _counterBuffers[frameIndex],
                new Vortice.Mathematics.Int4(0, 0, 0, 0));

            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Clear splitFlags buffer to zero
            commandList.ClearUnorderedAccessViewUint(
                device.GetGpuHandle(_splitFlagsUAVs[frameIndex]),
                _splitFlagsCPUHandles[frameIndex],
                _splitFlagsBuffers[frameIndex],
                new Vortice.Mathematics.Int4(0, 0, 0, 0));

            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Compute root/extents (LOCAL SPACE)
            Vector3 cameraPos = Camera.Main!.Position - Transform.Position;
            Vector3 rootCenter = new Vector3(TerrainSize.X * 0.5f, 0, TerrainSize.Y * 0.5f);
            Vector3 rootExtents = new Vector3(TerrainSize.X * 0.5f, MaxHeight, TerrainSize.Y * 0.5f);

            // Screen-space error: compute TanHalfFov and ScreenHeight
            var camera = Camera.Main!;
            float vFovRad = camera.FieldOfView * (MathF.PI / 180f);
            float tanHalfFov = MathF.Tan(vFovRad * 0.5f);
            float screenHeight = camera.Target?.Height ?? 1080f;

            // Push constants (Indices[0..7] = 32 dwords) — shared by all passes
            // Indices[0]: UAV outputs
            commandList.SetComputeRoot32BitConstant(0, _descriptorUAVs[frameIndex], 0);     // Indices[0].x
            commandList.SetComputeRoot32BitConstant(0, _sphereUAVs[frameIndex], 1);          // Indices[0].y
            commandList.SetComputeRoot32BitConstant(0, _subbatchIdUAVs[frameIndex], 2);      // Indices[0].z
            commandList.SetComputeRoot32BitConstant(0, _terrainDataUAVs[frameIndex], 3);     // Indices[0].w
            
            // Indices[1]: control params
            commandList.SetComputeRoot32BitConstant(0, _counterUAVs[frameIndex], 4);         // Indices[1].x
            commandList.SetComputeRoot32BitConstant(0, (uint)MaxDepth, 5);                   // Indices[1].y
            commandList.SetComputeRoot32BitConstant(0, (uint)_transformSlot, 6);             // Indices[1].z
            commandList.SetComputeRoot32BitConstant(0, (uint)(Material?.MaterialID ?? 0), 7);// Indices[1].w
            
            // Indices[2]: more control params
            commandList.SetComputeRoot32BitConstant(0, (uint)_meshPartId, 8);                // Indices[2].x
            commandList.SetComputeRoot32BitConstant(0, 0u, 9);                                // Indices[2].y (unused, was LODRange0)
            commandList.SetComputeRoot32BitConstant(0, (uint)_totalNodes, 10);               // Indices[2].z
            commandList.SetComputeRoot32BitConstant(0, Heightmap?.BindlessIndex ?? 0u, 11);  // Indices[2].w = HeightTexIdx

            // Indices[3]: camera pos + root center x
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(cameraPos.X), 12);    // Indices[3].x
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(cameraPos.Y), 13);    // Indices[3].y
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(cameraPos.Z), 14);    // Indices[3].z
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(rootCenter.X), 15);   // Indices[3].w

            // Indices[4]: root center yz + root extents xy
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(rootCenter.Y), 16);   // Indices[4].x
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(rootCenter.Z), 17);   // Indices[4].y
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(rootExtents.X), 18);  // Indices[4].z
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(rootExtents.Y), 19);  // Indices[4].w

            // Indices[5]: root extents z + max patches + splitFlags UAV + maxHeight
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(rootExtents.Z), 20);  // Indices[5].x
            commandList.SetComputeRoot32BitConstant(0, (uint)MaxPatches, 21);                                // Indices[5].y
            commandList.SetComputeRoot32BitConstant(0, _splitFlagsUAVs[frameIndex], 22);                    // Indices[5].z
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(MaxHeight), 23);     // Indices[5].w

            // Indices[6]: HeightRangeSRV + TerrainSize + (BuildMip unused at runtime)
            commandList.SetComputeRoot32BitConstant(0, _heightRangePyramidSRV, 24);                          // Indices[6].x
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(TerrainSize.X), 25);  // Indices[6].y
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(TerrainSize.Y), 26);  // Indices[6].z
            commandList.SetComputeRoot32BitConstant(0, 0u, 27);                                              // Indices[6].w = unused

            // Indices[7]: PixelErrorThreshold + ScreenHeight + TanHalfFov (slots 29-30 overwritten for mip build)
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(PixelErrorThreshold), 28); // Indices[7].x
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(screenHeight), 29);        // Indices[7].y
            commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(tanHalfFov), 30);          // Indices[7].z

            // ── One-time: Build height range mip pyramid ──
            if (!_heightRangePyramidBuilt)
            {
                BuildHeightRangePyramid(commandList);
                _heightRangePyramidBuilt = true;

                // Mip build overwrites slots 29-30 — restore ScreenHeight/TanHalfFov
                commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(screenHeight), 29);
                commandList.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(tanHalfFov), 30);
            }

            uint threadGroups = (uint)((_totalNodes + 255) / 256);

            // ── Pass 1: Mark splits + enforce restricted quadtree ──
            commandList.SetPipelineState(_markSplitsPSO);
            commandList.Dispatch(threadGroups, 1, 1);

            // UAV barrier — splitFlags must be visible to Pass 2
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // ── Pass 2: Emit leaves with inline frustum + Hi-Z culling ──
            commandList.SetPipelineState(_emitLeavesPSO);
            commandList.Dispatch(threadGroups, 1, 1);

            // UAV barrier — counter and output buffers must be visible to Pass 3
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // ── Pass 3: Build DrawInstanced indirect args from counter ──
            // Set slot 29 (MipInputSRV) = mesh index count for CSBuildDrawArgs
            commandList.SetComputeRoot32BitConstant(0, (uint)_patchMesh.IndexCount, 29);  // Indices[7].y = vertex count per instance
            commandList.SetComputeRoot32BitConstant(0, _indirectArgsUAVs[frameIndex], 31); // Indices[7].w = IndirectArgsUAV
            commandList.SetPipelineState(_buildDrawArgsPSO);
            commandList.Dispatch(1, 1, 1);

            // UAV barrier — indirect args must be visible before ExecuteIndirect
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Transition output buffers from UAV to SRV for vertex shader reads
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceTransitionBarrier(_descriptorBuffers[frameIndex], ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)));
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceTransitionBarrier(_sphereBuffers[frameIndex], ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)));
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceTransitionBarrier(_subbatchIdBuffers[frameIndex], ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)));
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceTransitionBarrier(_terrainDataBuffers[frameIndex], ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)));
            _buffersInSRV[frameIndex] = true;

            // Transition indirect args buffer for ExecuteIndirect
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceTransitionBarrier(_indirectArgsBuffers[frameIndex], ResourceStates.UnorderedAccess, ResourceStates.IndirectArgument)));
            _indirectArgsInArgState[frameIndex] = true;

            // Store VP for next frame's Hi-Z occlusion projection
            _previousFrameViewProjection = Camera.Main.ViewProjection;
        }

        /// <summary>
        /// Self-draw via ExecuteIndirect. Sets root constants, applies material PSO,
        /// and calls ExecuteIndirect with the GPU-generated draw args.
        /// </summary>
        private void DrawTerrain(ID3D12GraphicsCommandList commandList, int frameIndex)
        {
            var device = Engine.Device;

            // Apply material PSO and textures
            Material.Apply(commandList, device);

            // Set topology
            commandList.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleList);

            // Set descriptor heap (Material.Apply may have changed it)
            commandList.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            // Set root constants for push constant slots used by the vertex shader
            // Slot 1: TerrainPatchData SRV
            commandList.SetGraphicsRoot32BitConstant(0, _terrainDataSRVs[frameIndex], 1);
            // Slot 2: InstanceDescriptor SRV
            commandList.SetGraphicsRoot32BitConstant(0, _descriptorSRVs[frameIndex], 2);

            // Slot 7: Index buffer SRV
            commandList.SetGraphicsRoot32BitConstant(0, _patchMesh.IndexBufferIndex, 7);
            // Slot 8: BaseIndex (always 0 for terrain patch mesh)
            commandList.SetGraphicsRoot32BitConstant(0, 0u, 8);
            // Slot 9: Position buffer SRV
            commandList.SetGraphicsRoot32BitConstant(0, _patchMesh.PosBufferIndex, 9);
            // Slot 15: GlobalTransformBuffer SRV
            commandList.SetGraphicsRoot32BitConstant(0, TransformBuffer.Instance!.SrvIndex, 15);
            // Slot 16: Debug mode
            commandList.SetGraphicsRoot32BitConstant(0, (uint)Engine.Settings.DebugVisualizationMode, 16);

            // ExecuteIndirect with DrawInstancedSignature — args written by CSBuildDrawArgs
            commandList.ExecuteIndirect(
                device.DrawInstancedSignature,
                1,
                _indirectArgsBuffers[frameIndex],
                0,
                null,
                0);
        }

        /// <summary>
        /// Upload frustum planes + Hi-Z occlusion data to the per-frame constant buffer.
        /// This is bound to compute root slot 1 (register b0) for inline culling.
        /// </summary>
        private void UploadFrustumConstants(int frameIndex)
        {
            var vpMatrix = Engine.Settings.FreezeFrustum
                ? Engine.Settings.FrozenViewProjection
                : Camera.Main!.ViewProjection;
            var frustum = new Frustum(vpMatrix);
            var planes = frustum.GetPlanesAsVector4();

            // The compute shader works in terrain LOCAL space, but frustum planes
            // are in WORLD space.  Transform planes to local space:
            //   plane_local = (n, d + dot(n, terrainPos))
            // because x_world = x_local + terrainPos  =>  n·x_local + (d + n·P) = 0
            var terrainPos = Transform.Position;
            for (int i = 0; i < planes.Length; i++)
            {
                var n = new Vector3(planes[i].X, planes[i].Y, planes[i].Z);
                planes[i].W += Vector3.Dot(n, terrainPos);
            }

            var constants = new GPUCuller.FrustumConstants
            {
                Plane0 = planes[0],
                Plane1 = planes[1],
                Plane2 = planes[2],
                Plane3 = planes[3],
                Plane4 = planes[4],
                Plane5 = planes[5],
            };

            // Hi-Z occlusion data — VP must also expect local-space input:
            //   clip = x_local * Translate(terrainPos) * VP
            var culler = CommandBuffer.Culler;
            if (culler != null && culler.HiZPyramidSRV != 0 && culler.HiZReady && !Engine.Settings.DisableHiZ)
            {
                var occVP = Engine.Settings.FreezeFrustum
                    ? Engine.Settings.FrozenViewProjection
                    : _previousFrameViewProjection;
                // Premultiply by the terrain's world translation so the matrix
                // accepts local-space positions directly:
                var localToWorld = Matrix4x4.CreateTranslation(terrainPos);
                constants.OcclusionProjection = localToWorld * occVP;

                constants.HiZSrvIdx = culler.HiZPyramidSRV;
                constants.HiZWidth = culler.HiZWidth;
                constants.HiZHeight = culler.HiZHeight;
                constants.HiZMipCount = (uint)culler.HiZMipCount;
                constants.NearPlane = Camera.Main!.NearPlane;
            }

            unsafe
            {
                void* pData;
                _frustumConstantBuffers[frameIndex].Map(0, null, &pData);
                *(GPUCuller.FrustumConstants*)pData = constants;
                _frustumConstantBuffers[frameIndex].Unmap(0);
            }
        }

        // ───── Height Range Mip Pyramid ───────────────────────────────────

        /// <summary>
        /// Creates the RG32F mip pyramid texture and per-mip descriptors.
        /// Called once during InitializeCompute.
        /// </summary>
        private void CreateHeightRangePyramid(GraphicsDevice device)
        {
            // Pyramid size: power-of-2 that covers the finest quadtree level.
            // At maxDepth, we have 2^maxDepth nodes per axis.
            int pyramidSize = 1 << MaxDepth;
            _heightRangeMipCount = MaxDepth + 1;

            _heightRangePyramid = device.CreateTexture2D(
                Format.R32G32_Float,
                pyramidSize, pyramidSize,
                1, _heightRangeMipCount,
                ResourceFlags.AllowUnorderedAccess,
                ResourceStates.Common);

            _heightRangeMipUAVs = new uint[_heightRangeMipCount];
            _heightRangeMipSRVs = new uint[_heightRangeMipCount];

            for (int mip = 0; mip < _heightRangeMipCount; mip++)
            {
                // Per-mip UAV for writing during build
                _heightRangeMipUAVs[mip] = device.AllocateBindlessIndex();
                var uavDesc = new UnorderedAccessViewDescription
                {
                    Format = Format.R32G32_Float,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = (uint)mip }
                };
                device.NativeDevice.CreateUnorderedAccessView(
                    _heightRangePyramid, null, uavDesc, device.GetCpuHandle(_heightRangeMipUAVs[mip]));

                // Per-mip SRV for reading during build (input to next mip level)
                _heightRangeMipSRVs[mip] = device.AllocateBindlessIndex();
                var srvDesc = new ShaderResourceViewDescription
                {
                    Format = Format.R32G32_Float,
                    ViewDimension = Vortice.Direct3D12.ShaderResourceViewDimension.Texture2D,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2D = new Texture2DShaderResourceView
                    {
                        MostDetailedMip = (uint)mip,
                        MipLevels = 1
                    }
                };
                device.NativeDevice.CreateShaderResourceView(
                    _heightRangePyramid, srvDesc, device.GetCpuHandle(_heightRangeMipSRVs[mip]));
            }

            // Full-pyramid SRV for runtime sampling (all mips)
            _heightRangePyramidSRV = device.AllocateBindlessIndex();
            var fullSrvDesc = new ShaderResourceViewDescription
            {
                Format = Format.R32G32_Float,
                ViewDimension = Vortice.Direct3D12.ShaderResourceViewDimension.Texture2D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2D = new Texture2DShaderResourceView
                {
                    MostDetailedMip = 0,
                    MipLevels = (uint)_heightRangeMipCount
                }
            };
            device.NativeDevice.CreateShaderResourceView(
                _heightRangePyramid, fullSrvDesc, device.GetCpuHandle(_heightRangePyramidSRV));

            Debug.Log($"[GPUTerrain] Height range pyramid: {pyramidSize}x{pyramidSize}, {_heightRangeMipCount} mips, SRV={_heightRangePyramidSRV}");
        }

        /// <summary>
        /// Dispatches CSBuildMinMaxMip per mip level to build the height range pyramid.
        /// Called once on first frame (inside DispatchQuadtreeEval).
        /// </summary>
        private void BuildHeightRangePyramid(ID3D12GraphicsCommandList commandList)
        {
            commandList.SetPipelineState(_buildMinMaxMipPSO);

            int w = 1 << MaxDepth;
            int h = w;

            for (int mip = 0; mip < _heightRangeMipCount; mip++)
            {
                // Transition this mip to UAV for writing
                commandList.ResourceBarrier(new ResourceBarrier(
                    new ResourceTransitionBarrier(_heightRangePyramid,
                        ResourceStates.Common,
                        ResourceStates.UnorderedAccess,
                        (uint)mip)));

                // Push constants for this mip level
                commandList.SetComputeRoot32BitConstant(0, (uint)mip, 27);               // BuildMip
                commandList.SetComputeRoot32BitConstant(0, mip > 0 ? _heightRangeMipSRVs[mip - 1] : 0u, 29); // MipInputSRV
                commandList.SetComputeRoot32BitConstant(0, _heightRangeMipUAVs[mip], 30); // MipOutputUAV

                // Dispatch 8x8 threadgroups
                uint groupsX = (uint)((w + 7) / 8);
                uint groupsY = (uint)((h + 7) / 8);
                commandList.Dispatch(groupsX, groupsY, 1);

                // Transition to SRV for the next mip to read
                commandList.ResourceBarrier(new ResourceBarrier(
                    new ResourceTransitionBarrier(_heightRangePyramid,
                        ResourceStates.UnorderedAccess,
                        ResourceStates.NonPixelShaderResource,
                        (uint)mip)));

                w = Math.Max(1, w / 2);
                h = Math.Max(1, h / 2);
            }

            // All mips are now in NonPixelShaderResource — ready for SampleLevel in CSMarkSplits/CSEmitLeaves
        }

        // ───── Helpers ────────────────────────────────────────────────────

        private static int CalculateTotalNodes(int maxDepth)
        {
            int total = 0;
            int levelSize = 1;
            for (int d = 0; d <= maxDepth; d++)
            {
                total += levelSize;
                levelSize *= 4;
            }
            return total;
        }

        private void SetupLayerTiling()
        {
            if (Layers == null || Layers.Count == 0) return;

            var diffuseList = new List<Texture>();
            var normalList = new List<Texture>();

            for (int i = 0; i < Layers.Count && i < _layerTiling.Length; i++)
            {
                var layer = Layers[i];
                if (layer.Tiling.X != 0 && layer.Tiling.Y != 0)
                    _layerTiling[i] = new Vector4(TerrainSize.X / layer.Tiling.X, TerrainSize.Y / layer.Tiling.Y, 0, 0);
                else
                    _layerTiling[i] = Vector4.One;

                if (layer.Diffuse != null) diffuseList.Add(layer.Diffuse);
                if (layer.Normals != null) normalList.Add(layer.Normals);
            }

            var device = Engine.Device;
            if (diffuseList.Count > 0)
                DiffuseMapsArray = Texture.CreateTexture2DArray(device, diffuseList);
            if (normalList.Count > 0)
                NormalMapsArray = Texture.CreateTexture2DArray(device, normalList);

            // Build ControlMapsArray from splatmaps
            var controlList = new List<Texture>();
            for (int i = 0; i < ControlMaps.Length; i++)
            {
                if (ControlMaps[i] != null) controlList.Add(ControlMaps[i]!);
            }
            if (controlList.Count > 0)
                ControlMapsArray = Texture.CreateTexture2DArray(device, controlList);
        }

        // ───── IHeightProvider ────────────────────────────────────────────

        public float GetHeight(Vector3 position)
        {
            if (HeightField == null) return Transform?.Position.Y ?? 0;
            if (Transform == null) return 0;

            position -= Transform.Position;

            var dimx = HeightField.GetLength(0) - 1;
            var dimy = HeightField.GetLength(1) - 1;

            var fx = (dimx + 1) / TerrainSize.X;
            var fy = (dimy + 1) / TerrainSize.Y;

            position.X *= fx;
            position.Z *= fy;

            int x = (int)Math.Floor(position.X);
            int z = (int)Math.Floor(position.Z);

            if (x < 0 || x > dimx) return Transform.Position.Y;
            if (z < 0 || z > dimy) return Transform.Position.Y;

            float xf = position.X - x;
            float zf = position.Z - z;

            float h1 = HeightField[x, z];
            float h2 = HeightField[Math.Min(dimx, x + 1), z];
            float h3 = HeightField[x, Math.Min(dimy, z + 1)];

            float height = h1;
            height += (h2 - h1) * xf;
            height += (h3 - h1) * zf;

            return Transform.Position.Y + height * MaxHeight;
        }
    }
}
