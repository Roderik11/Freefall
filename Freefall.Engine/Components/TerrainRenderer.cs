using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Freefall.Assets;
using Freefall.Graphics;
using Freefall.Base;

namespace Freefall.Components
{
    /// <summary>
    /// GPU-driven quadtree terrain renderer. A single compute dispatch evaluates the entire
    /// quadtree and produces per-patch data directly into InstanceBatch-compatible buffers.
    /// The compute dispatch runs as a custom action on the renderer's command list.
    /// Resource data (heightmap, material, layers, splatmaps) lives in the Terrain asset.
    /// </summary>
    public class TerrainRenderer : Freefall.Base.Component, IDraw, IHeightProvider
    {
        public static bool ComputeReady { get; set; }
        public static string ComputeError { get; set; } = "";

        // ───── Asset Reference ────────────────────────────────────────────
        public Terrain? Terrain;
        public Material? Material;
        public Material? DecoratorMaterial = InternalAssets.DecoratorMaterial;

        // ───── Rendering Parameters ───────────────────────────────────────
        public float PixelErrorThreshold = 2.0f;
        public int MaxDepth = 7;
        public int MaxPatches = 8192;

        private Vector4[] _layerTiling = new Vector4[32];

        // ───── Internal State ─────────────────────────────────────────────
        private const int FrameCount = 3;

        // Compute pipeline — restricted quadtree (auto-discovers all #pragma kernel entries)
        private ComputeShader? _quadtreeCS;
        private bool _computeInitialized;
        private bool _firstDispatch = true; // skip Hi-Z on first frame (no valid _previousFrameViewProjection yet)

        // GPU output buffers — per-frame (GraphicsBuffer with auto-managed SRV/UAV/state)
        private GraphicsBuffer[] _descriptorBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _sphereBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _subbatchIdBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _terrainDataBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _counterBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _splitFlagsBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _indirectArgsBuffers = new GraphicsBuffer[FrameCount];

        private GraphicsBuffer[] _shadowArgsBuffers = new GraphicsBuffer[FrameCount];

        // Shadow emit output buffers (CSEmitLeavesShadow with per-cascade frustum culling)
        private GraphicsBuffer[] _shadowDescriptorBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _shadowTerrainDataBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _shadowSphereBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _shadowSubbatchIdBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _shadowCounterBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _shadowCascadeIdxBuffers = new GraphicsBuffer[FrameCount];
        private ID3D12Resource[] _shadowCascadeBuffers = new ID3D12Resource[FrameCount]; // StructuredBuffer<CascadeData> (local-space planes)
        private IntPtr[] _shadowCascadeBufferPtrs = new IntPtr[FrameCount];
        private uint[] _shadowCascadeBufferSrvs = new uint[FrameCount];
        private ID3D12Resource? _shadowCounterReadback;
        private ID3D12Resource? _shadowArgsReadback;
        private int _shadowReadbackFrame = -1;
        
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

        // Mesh and registration
        private Mesh _patchMesh = null!;
        private int _meshPartId;

        // Texture arrays
        private Texture? ControlMapsArray;
        private Texture? DecoMapsArray;
        private Texture? DiffuseMapsArray;
        private Texture? NormalMapsArray;

        private int _totalNodes;

        // ───── Ground Coverage Decorator ──────────────────────────────────
        private GraphicsBuffer? _decoratorHeadersBuffer;
        private GraphicsBuffer? _decoratorSlotsBuffer;
        private GraphicsBuffer? _decoratorLODTableBuffer;
        private bool _decoratorBuffersBuilt;
        private bool _decoratorDispatched;
        private int _decoratorStructVersion = -1;
        private int _decoratorValueVersion = -1;
        private int _decoratorDispatchGroupsX;
        private int _decoratorDispatchGroupsY;



        // ───── Decoration Control Prepass ─────────────────────────────────
        private ComputeShader? _decoPrepassCS;
        private ID3D12Resource? _decoControlTex;     // RGBA16_UINT, 2 slices
        private uint _decoControlUAV;
        private uint _decoControlSRV;
        private bool _decoControlDirty = true;

        // ───── Baked Terrain Albedo ───────────────────────────────────────
        private ComputeShader? _albedoBakeCS;
        private ID3D12Resource? _bakedAlbedoTex;     // RGBA8, 256×256
        private uint _bakedAlbedoUAV;
        private uint _bakedAlbedoSRV;
        private GraphicsBuffer? _tilingBuffer;       // StructuredBuffer<float4>, 32 entries
        private bool _bakedAlbedoDirty = true;
        private const int BakedAlbedoSize = 256;

        // ───── Lifecycle ──────────────────────────────────────────────────

        protected override void Awake()
        {
            // GPU resource init — matches TerrainGPU.Awake() exactly.
            // Called before YAML properties are applied, but MaxDepth defaults to 7
            // and CreateHeightRangePyramid doesn't read the heightmap —
            // BuildHeightRangePyramid is deferred to the first DispatchQuadtreeEval.
            _patchMesh = Mesh.CreatePatch(Engine.Device);
            _meshPartId = MeshRegistry.Register(_patchMesh, 0);
            _totalNodes = CalculateTotalNodes(MaxDepth);
            try { InitializeCompute(); }
            catch (Exception ex)
            {
                ComputeError = ex.Message;
                Debug.LogError("[TerrainRenderer]", $"Failed to initialize compute: {ex.Message}");
            }
            Debug.Log($"[TerrainRenderer] Awake: MaxDepth={MaxDepth} TotalNodes={_totalNodes} MaxPatches={MaxPatches} ComputeInit={_computeInitialized}");
            ComputeReady = _computeInitialized;
        }

        private bool _texturesInitialized;

        private void LazyInitTextures()
        {
            if (_texturesInitialized) return;
            if (Terrain?.Layers == null || Terrain.Layers.Count == 0) return;
            SetupLayerTiling();
            _texturesInitialized = true;
        }

        public void Draw()
        {
            if (Camera.Main == null || Terrain == null || !_computeInitialized) return;

            LazyInitTextures();

            var material = Material;
            var heightmap = Terrain.Heightmap;
            if (material == null || material.Effect == null) return;

            int frameIndex = Engine.FrameIndex % FrameCount;

            // Set shared material params
            material.SetParameter("CameraPos", Camera.Main.Position);
            material.SetParameter("HeightTexel", heightmap != null ? 1.0f / heightmap.Native.Description.Width : 1.0f / 1024.0f);
            material.SetParameter("MaxHeight", Terrain.MaxHeight);
            material.SetParameter("TerrainSize", Terrain.TerrainSize);
            material.SetParameter("TerrainOrigin", new Vector2(Transform.WorldPosition.X, Transform.WorldPosition.Z));

            material.SetParameter("LayerTiling", _layerTiling);

            // Bind textures — texture arrays MUST be valid Texture2DArray resources.
            // The PS samples them as Texture2DArray with array index 0..3.
            // A Texture2D fallback would cause a GPU fault (TDR).
            // Bind textures if available (terrain still renders without splatmaps)

            if (heightmap != null) material.SetTexture("HeightTex", heightmap);
            if (ControlMapsArray != null) material.SetTexture("ControlMaps", ControlMapsArray);
            if (DiffuseMapsArray != null) material.SetTexture("DiffuseMaps", DiffuseMapsArray);
            if (NormalMapsArray != null) material.SetTexture("NormalMaps", NormalMapsArray);

            // Capture values for lambda closure
            int fi = frameIndex;
            var self = this;

            // Enqueue compute dispatch as custom action (runs first in Execute, before batch processing)
            CommandBuffer.Enqueue(RenderPass.Opaque, (list) => self.DispatchQuadtreeEval(list, fi));

            // Enqueue self-draw action: terrain draws itself via ExecuteIndirect
            CommandBuffer.Enqueue(RenderPass.Opaque, (list) => self.DrawTerrain(list, fi));

            // Enqueue single-pass terrain shadow draw
            CommandBuffer.Enqueue(RenderPass.Shadow, (list) => self.DrawTerrainShadow(list, fi));

            // Enqueue ground coverage decorator if configured
            if (Terrain.DrawDetail && Terrain.Decorations.Count > 0 && DecoratorMaterial?.Effect != null)
            {
                // Structural hash: only mesh names and count (triggers buffer rebuild)
                int structHash = Terrain.Decorations.Count;
                foreach (var d in Terrain.Decorations)
                    structHash = HashCode.Combine(structHash, d.Mesh?.Name);

                // Value hash: density, scale, rotation, etc. (triggers data update only)
                int valueHash = HashCode.Combine(Terrain.DecorationRadius);
                foreach (var d in Terrain.Decorations)
                {
                    valueHash = HashCode.Combine(valueHash, d.Density, d.HeightRange, d.WidthRange,
                        d.RootRotation, d.SlopeBias);
                }

                if (structHash != _decoratorStructVersion)
                {
                    // Full rebuild: create buffers + upload data
                    BuildDecoratorBuffers();
                    _decoratorStructVersion = structHash;
                    _decoratorValueVersion = valueHash;
                }
                else if (valueHash != _decoratorValueVersion)
                {
                    // Value-only change: just update data in existing buffers
                    UpdateDecoratorBufferData();
                    _decoratorValueVersion = valueHash;
                }

                CommandBuffer.Enqueue(RenderPass.Opaque, (list) => self.DispatchDecorator(list, fi, RenderPass.Opaque));
                CommandBuffer.Enqueue(RenderPass.Shadow, (list) => self.DispatchDecorator(list, fi, RenderPass.Shadow));
            }
        }

        // ───── Compute Pipeline ───────────────────────────────────────────

        private void InitializeCompute()
        {
            var device = Engine.Device;

            _quadtreeCS = new ComputeShader("terrain_quadtree.hlsl");

            for (int i = 0; i < FrameCount; i++)
            {
                CreateFrameBuffers(i);
            }

            // Create frustum constant buffers (for compute shader Hi-Z culling)
            int frustumCBSize = Marshal.SizeOf<GPUCuller.FrustumConstants>();
            int cascadeDataSize = Marshal.SizeOf<GPUCuller.CascadeData>();
            int cascadeBufferSize = cascadeDataSize * DirectionalLight.MaxCascades;
            for (int i = 0; i < FrameCount; i++)
            {
                _frustumConstantBuffers[i] = device.CreateUploadBuffer(frustumCBSize);
                _shadowCascadeBuffers[i] = device.CreateUploadBuffer(cascadeBufferSize);
                _shadowCascadeBufferSrvs[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferSRV(_shadowCascadeBuffers[i], (uint)DirectionalLight.MaxCascades, (uint)cascadeDataSize, _shadowCascadeBufferSrvs[i]);
                unsafe
                {
                    void* pData;
                    _shadowCascadeBuffers[i].Map(0, null, &pData);
                    _shadowCascadeBufferPtrs[i] = (IntPtr)pData;
                }
            }

            // Create height range mip pyramid texture
            CreateHeightRangePyramid(device);

            _computeInitialized = true;
        }

        private void CreateFrameBuffers(int i)
        {
            int shadowCapacity = MaxPatches * DirectionalLight.CascadeCount;

            // Structured buffers with SRV + UAV
            _descriptorBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 12, srv: true, uav: true);
            _sphereBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 16, srv: true, uav: true);
            _subbatchIdBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 4, srv: true, uav: true);
            _terrainDataBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 32, srv: true, uav: true);

            // Raw R32 buffers
            _counterBuffers[i] = GraphicsBuffer.CreateRaw(1, uav: true, clearable: true);
            _splitFlagsBuffers[i] = GraphicsBuffer.CreateRaw(_totalNodes, uav: true, clearable: true);
            _indirectArgsBuffers[i] = GraphicsBuffer.CreateRaw(4, srv: true, uav: true);
            _shadowArgsBuffers[i] = GraphicsBuffer.CreateRaw(4, uav: true);

            // Shadow structured buffers
            _shadowDescriptorBuffers[i] = GraphicsBuffer.CreateStructured(shadowCapacity, 12, srv: true, uav: true);
            _shadowTerrainDataBuffers[i] = GraphicsBuffer.CreateStructured(shadowCapacity, 32, srv: true, uav: true);
            _shadowSphereBuffers[i] = GraphicsBuffer.CreateStructured(shadowCapacity, 16, uav: true);
            _shadowSubbatchIdBuffers[i] = GraphicsBuffer.CreateStructured(shadowCapacity, 4, uav: true);
            _shadowCascadeIdxBuffers[i] = GraphicsBuffer.CreateStructured(shadowCapacity, 4, srv: true, uav: true);
            _shadowCounterBuffers[i] = GraphicsBuffer.CreateRaw(1, uav: true, clearable: true);
        }

        /// <summary>
        /// Dispatches the quadtree evaluation compute shader on the renderer's command list.
        /// Called as a custom action before batch processing in Pass.Execute.
        /// </summary>
        private void DispatchQuadtreeEval(ID3D12GraphicsCommandList commandList, int frameIndex)
        {
            var device = Engine.Device;
            var cs = _quadtreeCS!;

            // Transition buffers to UAV
            _descriptorBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);
            _sphereBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);
            _subbatchIdBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);
            _terrainDataBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);
            _indirectArgsBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);

            // Upload frustum + Hi-Z constants
            UploadFrustumConstants(frameIndex);
            cs.Begin(commandList);  // Sets root sig + descriptor heaps once
            commandList.SetComputeRootConstantBufferView(1, _frustumConstantBuffers[frameIndex].GPUVirtualAddress);

            // Clear counter and splitFlags
            _counterBuffers[frameIndex].ClearUAV(commandList, new Vortice.Mathematics.Int4(0, 0, 0, 0));
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
            _splitFlagsBuffers[frameIndex].ClearUAV(commandList, new Vortice.Mathematics.Int4(0, 0, 0, 0));
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Compute root/extents (LOCAL SPACE)
            var terrainSize = Terrain!.TerrainSize;
            var maxHeight = Terrain.MaxHeight;
            Vector3 cameraPos = Camera.Main!.Position - Transform.Position;
            Vector3 rootCenter = new Vector3(terrainSize.X * 0.5f, 0, terrainSize.Y * 0.5f);
            Vector3 rootExtents = new Vector3(terrainSize.X * 0.5f, maxHeight, terrainSize.Y * 0.5f);

            // Screen-space error params
            var camera = Camera.Main!;
            float vFovRad = camera.FieldOfView * (MathF.PI / 180f);
            float tanHalfFov = MathF.Tan(vFovRad * 0.5f);
            float screenHeight = camera.Target?.Height ?? 1080f;

            // Set named push constants — shared by all passes
            // We use SetKernel("CSMarkSplits") to set constants on kernal 0, which all passes read
            cs.SetKernel("CSMarkSplits");

            // UAV outputs
            cs.Set("OutputDescriptorsUAV", _descriptorBuffers[frameIndex].UavIndex);
            cs.Set("OutputSpheresUAV", _sphereBuffers[frameIndex].UavIndex);
            cs.Set("OutputSubbatchIdsUAV", _subbatchIdBuffers[frameIndex].UavIndex);
            cs.Set("OutputTerrainDataUAV", _terrainDataBuffers[frameIndex].UavIndex);

            // Control params
            cs.Set("CounterUAV", _counterBuffers[frameIndex].UavIndex);
            cs.Set("MaxDepth", (uint)MaxDepth);
            cs.Set("TransformSlot", (uint)Transform.TransformSlot);
            cs.Set("MaterialId", (uint)(Material?.MaterialID ?? 0));
            cs.Set("MeshPartId", (uint)_meshPartId);
            cs.Set("TotalNodes", (uint)_totalNodes);
            cs.Set("HeightTex", Terrain?.Heightmap?.BindlessIndex ?? 0u);

            // Camera/bounds (float via uint bits)
            cs.Set("CameraPosX", cameraPos.X);
            cs.Set("CameraPosY", cameraPos.Y);
            cs.Set("CameraPosZ", cameraPos.Z);
            cs.Set("RootCenterX", rootCenter.X);
            cs.Set("RootCenterY", rootCenter.Y);
            cs.Set("RootCenterZ", rootCenter.Z);
            cs.Set("RootExtentsX", rootExtents.X);
            cs.Set("RootExtentsY", rootExtents.Y);
            cs.Set("RootExtentsZ", rootExtents.Z);

            // More control
            cs.Set("MaxPatches", (uint)MaxPatches);
            cs.Set("SplitFlagsUAV", _splitFlagsBuffers[frameIndex].UavIndex);
            cs.Set("MaxHeight", maxHeight);
            cs.Set("HeightRangeSRV", _heightRangePyramidSRV);
            cs.Set("TerrainSizeX", terrainSize.X);
            cs.Set("TerrainSizeY", terrainSize.Y);

            // Screen-space error
            cs.Set("PixelErrorThreshold", PixelErrorThreshold);
            cs.Set("ScreenHeight", screenHeight);
            cs.Set("TanHalfFov", tanHalfFov);

            // ── One-time: Build height range mip pyramid ──
            if (!_heightRangePyramidBuilt)
            {
                StreamingManager.Instance?.Flush();
                Engine.Device.WaitForCopyQueue();

                BuildHeightRangePyramid(commandList);
                _heightRangePyramidBuilt = true;

                // Mip build overwrites slots 29-30 — restore
                cs.Set("ScreenHeight", screenHeight);
                cs.Set("TanHalfFov", tanHalfFov);
            }

            uint threadGroups = (uint)((_totalNodes + 255) / 256);

            // ── Pass 1: Mark splits + enforce restricted quadtree ──
            cs.Dispatch(commandList, threadGroups);

            // UAV barrier
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // ── Pass 2: Emit leaves with inline frustum + Hi-Z culling ──
            cs.SetKernel("CSEmitLeaves");
            cs.Dispatch(commandList, threadGroups);

            // UAV barrier
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // ── Pass 3: Build DrawInstanced indirect args from counter ──
            cs.SetKernel("CSBuildDrawArgs");
            cs.Set("ScreenHeight", (uint)_patchMesh.IndexCount);  // Repurposed: vertex count
            cs.Set("IndirectArgsUAV", _indirectArgsBuffers[frameIndex].UavIndex);
            cs.Dispatch(commandList, 1);

            // UAV barrier
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Transition indirect args buffer for ExecuteIndirect
            _indirectArgsBuffers[frameIndex].Transition(commandList, ResourceStates.IndirectArgument);

            // Transition output buffers from UAV to SRV for vertex shader reads
            _descriptorBuffers[frameIndex].Transition(commandList, ResourceStates.NonPixelShaderResource);
            _sphereBuffers[frameIndex].Transition(commandList, ResourceStates.NonPixelShaderResource);
            _subbatchIdBuffers[frameIndex].Transition(commandList, ResourceStates.NonPixelShaderResource);
            _terrainDataBuffers[frameIndex].Transition(commandList, ResourceStates.NonPixelShaderResource);

            // Store VP for next frame's Hi-Z occlusion projection
            _previousFrameViewProjection = Camera.Main.ViewProjection;
            _firstDispatch = false;
        }

        /// <summary>
        /// Self-draw via ExecuteIndirect. Sets root constants, applies material PSO,
        /// and calls ExecuteIndirect with the GPU-generated draw args.
        /// </summary>
        private void DrawTerrain(ID3D12GraphicsCommandList commandList, int frameIndex)
        {
            var device = Engine.Device;

            // Apply material PSO and textures (explicitly set Opaque pass in case shadow changed it)
            Material!.SetPass(RenderPass.Shadow); // reset to first pass — Opaque
            Material!.SetPass(RenderPass.Opaque);
            Material!.Apply(commandList, device);

            // Set topology
            commandList.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleList);

            // Set descriptor heap (Material.Apply may have changed it)
            commandList.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            // Set root constants for push constant slots used by the vertex shader
            // Slot 1: TerrainPatchData SRV
            commandList.SetGraphicsRoot32BitConstant(0, _terrainDataBuffers[frameIndex].SrvIndex, 1);
            // Slot 2: InstanceDescriptor SRV
            commandList.SetGraphicsRoot32BitConstant(0, _descriptorBuffers[frameIndex].SrvIndex, 2);

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
                _indirectArgsBuffers[frameIndex].Native,
                0,
                null,
                0);
        }

        /// <summary>
        /// Single-pass shadow render: dispatches CSEmitLeavesShadow with per-cascade frustum
        /// culling, builds draw args from compacted counter, then ExecuteIndirect.
        /// VS_Shadow reads cascadeIdx from per-entry buffer — no instance expansion.
        /// Called as a custom action during RenderPass.Shadow.
        /// </summary>
        private unsafe void DrawTerrainShadow(ID3D12GraphicsCommandList commandList, int frameIndex)
        {
            if (Material == null || !_computeInitialized) return;

            var device = Engine.Device;
            var allPlanes = DirectionalLight.GetAllCascadeFrustumPlanes();
            if (allPlanes == null) return;
            // With SDSM adaptive splits, all cascades may cover nearby geometry — use them all.
            // With fixed splits, skip outermost cascade (perf: outer cascade is most expensive, least visible detail).
            int cascadeCount = Engine.Settings.UseAdaptiveSplits
                ? DirectionalLight.CascadeCount
                : Math.Max(1, DirectionalLight.CascadeCount - 1);

            // ════════════════════════════════════════════════════════════════
            // Phase 1: CSEmitLeavesShadow — per-cascade frustum culling
            // ════════════════════════════════════════════════════════════════

            // Upload local-space cascade planes to StructuredBuffer<CascadeData>
            var terrainPos = Transform.Position;
            GPUCuller.CascadeData* cascadePtr = (GPUCuller.CascadeData*)_shadowCascadeBufferPtrs[frameIndex];
            for (int c = 0; c < cascadeCount; c++)
            {
                var localPlanes = new Vector4[6];
                for (int p = 0; p < 6; p++)
                {
                    var plane = allPlanes[c][p];
                    var n = new Vector3(plane.X, plane.Y, plane.Z);
                    plane.W += Vector3.Dot(n, terrainPos);
                    localPlanes[p] = plane;
                }
                cascadePtr[c] = default;
                cascadePtr[c].SetPlanes(localPlanes);
            }

            // Transition shadow output buffers to UAV (no-op if already in UAV)
            _shadowDescriptorBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);
            _shadowTerrainDataBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);
            _shadowCascadeIdxBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);

            // Readback shadow args (transition through CopySource if needed)
            if (_shadowArgsBuffers[frameIndex].CurrentState == ResourceStates.IndirectArgument && _shadowArgsReadback != null)
            {
                _shadowArgsBuffers[frameIndex].Transition(commandList, ResourceStates.CopySource);
                commandList.CopyResource(_shadowArgsReadback, _shadowArgsBuffers[frameIndex].Native);
            }
            _shadowArgsBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);

            // Switch to compute pipeline
            var cs = _quadtreeCS!;
            cs.Begin(commandList);

            // Clear shadow counter
            _shadowCounterBuffers[frameIndex].ClearUAV(commandList, new Vortice.Mathematics.Int4(0, 0, 0, 0));

            // Set named push constants for CSEmitLeavesShadow
            cs.SetKernel("CSEmitLeavesShadow");

            // Shadow output UAVs
            cs.Set("OutputDescriptorsUAV", _shadowDescriptorBuffers[frameIndex].UavIndex);
            cs.Set("OutputSpheresUAV", _shadowSphereBuffers[frameIndex].UavIndex);
            cs.Set("OutputSubbatchIdsUAV", _shadowSubbatchIdBuffers[frameIndex].UavIndex);
            cs.Set("OutputTerrainDataUAV", _shadowTerrainDataBuffers[frameIndex].UavIndex);
            cs.Set("CounterUAV", _shadowCounterBuffers[frameIndex].UavIndex);

            // Quadtree constants
            cs.Set("MaxDepth", (uint)MaxDepth);
            cs.Set("TransformSlot", (uint)Transform.TransformSlot);
            cs.Set("MaterialId", (uint)(Material?.MaterialID ?? 0));
            cs.Set("MeshPartId", (uint)_meshPartId);
            cs.Set("CascadeIdxUAV", _shadowCascadeIdxBuffers[frameIndex].UavIndex);
            cs.Set("TotalNodes", (uint)_totalNodes);
            cs.Set("HeightTex", Terrain!.Heightmap?.BindlessIndex ?? 0u);

            // Camera position in LOCAL space
            var camPos = Camera.Main!.Position - Transform.Position;
            cs.Set("CameraPosX", camPos.X);
            cs.Set("CameraPosY", camPos.Y);
            cs.Set("CameraPosZ", camPos.Z);

            var terrainSz = Terrain.TerrainSize;
            cs.Set("RootCenterX", terrainSz.X * 0.5f);
            cs.Set("RootCenterY", 0f);
            cs.Set("RootCenterZ", terrainSz.Y * 0.5f);
            cs.Set("RootExtentsX", terrainSz.X * 0.5f);
            cs.Set("RootExtentsY", Terrain.MaxHeight);
            cs.Set("RootExtentsZ", terrainSz.Y * 0.5f);

            int shadowMaxPatches = MaxPatches * cascadeCount;
            cs.Set("MaxPatches", (uint)shadowMaxPatches);
            cs.Set("SplitFlagsUAV", _splitFlagsBuffers[frameIndex].UavIndex);
            cs.Set("MaxHeight", Terrain.MaxHeight);
            cs.Set("HeightRangeSRV", _heightRangePyramidSRV);

            cs.Set("TerrainSizeX", terrainSz.X);
            cs.Set("TerrainSizeY", terrainSz.Y);
            cs.Set("TanHalfFov", (uint)cascadeCount);  // Slot 30 reused as CascadeCount

            // Bind FrustumPlanes CB at root slot 1 (required by root signature)
            commandList.SetComputeRootConstantBufferView(1, _frustumConstantBuffers[frameIndex].GPUVirtualAddress);

            // Bind terrain-local cascade buffer via push constant (CascadeBufferSRVIdx, slot 31)
            cs.Set("IndirectArgsUAV", _shadowCascadeBufferSrvs[frameIndex]);  // Slot 31 reused for CascadeBufferSRV

            // Dispatch CSEmitLeavesShadow
            uint threadGroups = (uint)((_totalNodes + 255) / 256);
            cs.Dispatch(commandList, threadGroups);

            // UAV barrier
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Build draw args from shadow counter (reuse CSBuildDrawArgs)
            cs.SetKernel("CSBuildDrawArgs");
            cs.Set("ScreenHeight", (uint)_patchMesh.IndexCount);  // Repurposed: vertex count
            cs.Set("IndirectArgsUAV", _shadowArgsBuffers[frameIndex].UavIndex);
            cs.Dispatch(commandList, 1);

            // UAV barrier
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Transition shadow output to SRV for VS_Shadow
            _shadowDescriptorBuffers[frameIndex].Transition(commandList, ResourceStates.NonPixelShaderResource);
            _shadowTerrainDataBuffers[frameIndex].Transition(commandList, ResourceStates.NonPixelShaderResource);
            _shadowCascadeIdxBuffers[frameIndex].Transition(commandList, ResourceStates.NonPixelShaderResource);

            // ════════════════════════════════════════════════════════════════
            // Phase 2: Draw terrain shadows (graphics)
            // ════════════════════════════════════════════════════════════════

            Material.SetPass(RenderPass.Shadow);
            Material.Apply(commandList, device);

            commandList.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleList);
            commandList.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            // Bind shadow patch data
            commandList.SetGraphicsRoot32BitConstant(0, _shadowTerrainDataBuffers[frameIndex].SrvIndex, 1);  // TerrainDataIdx
            commandList.SetGraphicsRoot32BitConstant(0, _shadowDescriptorBuffers[frameIndex].SrvIndex, 2);   // DescriptorBufIdx
            commandList.SetGraphicsRoot32BitConstant(0, _patchMesh.IndexBufferIndex, 7);          // IndexBufferIdx
            commandList.SetGraphicsRoot32BitConstant(0, 0u, 8);                                   // BaseIndex
            commandList.SetGraphicsRoot32BitConstant(0, _patchMesh.PosBufferIndex, 9);            // PosBufferIdx
            commandList.SetGraphicsRoot32BitConstant(0, TransformBuffer.Instance!.SrvIndex, 15);  // GlobalTransformBufferIdx

            // Shadow-specific push constants
            commandList.SetGraphicsRoot32BitConstant(0, DirectionalLight.CurrentCascadeSrvIndex, 20); // CascadeBufferSRVIdx
            commandList.SetGraphicsRoot32BitConstant(0, (uint)cascadeCount, 21);                       // ShadowCascadeCount
            commandList.SetGraphicsRoot32BitConstant(0, _shadowCascadeIdxBuffers[frameIndex].SrvIndex, 22);        // CascadeIdxBufIdx

            // Transition shadow args to IndirectArgument
            _shadowArgsBuffers[frameIndex].Transition(commandList, ResourceStates.IndirectArgument);

            // ExecuteIndirect — instance count is compacted per-cascade count
            commandList.ExecuteIndirect(
                device.DrawInstancedSignature,
                1,
                _shadowArgsBuffers[frameIndex].Native,
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
            var pyramid = DeferredRenderer.Current?.HiZPyramid;
            if (!_firstDispatch && pyramid != null && pyramid.FullSRV != 0 && pyramid.Ready && !Engine.Settings.DisableHiZ)

            {
                var occVP = Engine.Settings.FreezeFrustum
                    ? Engine.Settings.FrozenViewProjection
                    : _previousFrameViewProjection;
                // Premultiply by the terrain's world translation so the matrix
                // accepts local-space positions directly:
                var localToWorld = Matrix4x4.CreateTranslation(terrainPos);
                constants.OcclusionProjection = localToWorld * occVP;

                constants.HiZSrvIdx = pyramid.FullSRV;
                constants.HiZWidth = pyramid.Width;
                constants.HiZHeight = pyramid.Height;
                constants.HiZMipCount = (uint)pyramid.MipCount;
                constants.NearPlane = Camera.Main!.NearPlane;
            }

            try
            {
                unsafe
                {
                    void* pData;
                    _frustumConstantBuffers[frameIndex].Map(0, null, &pData);
                    *(GPUCuller.FrustumConstants*)pData = constants;
                    _frustumConstantBuffers[frameIndex].Unmap(0);
                }
            }
            catch (Exception ex)
            {
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

           // Debug.Log($"[TerrainRenderer] Height range pyramid: {pyramidSize}x{pyramidSize}, {_heightRangeMipCount} mips, SRV={_heightRangePyramidSRV}");
        }

        /// <summary>
        /// Dispatches CSBuildMinMaxMip per mip level to build the height range pyramid.
        /// Called once on first frame (inside DispatchQuadtreeEval).
        /// </summary>
        private void BuildHeightRangePyramid(ID3D12GraphicsCommandList commandList)
        {
            var cs = _quadtreeCS!;
            cs.SetKernel("CSBuildMinMaxMip");

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
                cs.Set("BuildMip", (uint)mip);
                cs.Set("ScreenHeight", mip > 0 ? _heightRangeMipSRVs[mip - 1] : 0u);  // MipInputSRV (reused slot)
                cs.Set("TanHalfFov", _heightRangeMipUAVs[mip]);                         // MipOutputUAV (reused slot)

                // Dispatch 8x8 threadgroups
                uint groupsX = (uint)((w + 7) / 8);
                uint groupsY = (uint)((h + 7) / 8);
                cs.Dispatch(commandList, groupsX, groupsY);

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
            var terrainSize = Terrain?.TerrainSize ?? Vector2.One;
            var layers = Terrain?.Layers;
            var diffuseList = new List<Texture>();
            var normalList = new List<Texture>();

            if (layers != null)
            {
                for (int i = 0; i < layers.Count && i < _layerTiling.Length; i++)
                {
                    var layer = layers[i];
                    if (layer.Tiling.X != 0 && layer.Tiling.Y != 0)
                        _layerTiling[i] = new Vector4(terrainSize.X / layer.Tiling.X, terrainSize.Y / layer.Tiling.Y, 0, 0);
                    else
                        _layerTiling[i] = Vector4.One;

                    if (layer.Diffuse != null && layer.Diffuse.Native != null) diffuseList.Add(layer.Diffuse);
                    if (layer.Normals != null && layer.Normals.Native != null) normalList.Add(layer.Normals);
                }
            }

            var device = Engine.Device;
            
            // Diffuse Fallback
            if (diffuseList.Count > 0)
                DiffuseMapsArray = Texture.CreateTexture2DArray(device, diffuseList);
            else
                DiffuseMapsArray = InternalAssets.BlackArray;

            // Normal Fallback
            if (normalList.Count > 0)
                NormalMapsArray = Texture.CreateTexture2DArray(device, normalList);
            else
                NormalMapsArray = InternalAssets.FlatNormalArray;

            // Build ControlMapsArray from splatmaps
            var controlList = new List<Texture>();
            if (Terrain?.ControlMaps != null)
            {
                var controlMaps = Terrain.ControlMaps;
                for (int i = 0; i < controlMaps.Count; i++)
                {
                    if (controlMaps[i] != null && controlMaps[i]!.Native != null) controlList.Add(controlMaps[i]!);
                }
            }
            
            // Control Fallback
            if (controlList.Count > 0)
                ControlMapsArray = Texture.CreateTexture2DArray(device, controlList);
            else
                ControlMapsArray = InternalAssets.BlackArray;

            // Build DecoMapsArray from decorator density maps
            var decoList = new List<Texture>();
            if (Terrain?.DecoMaps != null)
            {
                foreach (var dm in Terrain.DecoMaps)
                {
                    if (dm != null && dm.Native != null) decoList.Add(dm);
                }
            }
            if (decoList.Count > 0)
                DecoMapsArray = Texture.CreateTexture2DArray(device, decoList);
            else
                DecoMapsArray = null;
        }

        // ───── IHeightProvider ────────────────────────────────────────────

        public float GetHeight(Vector3 position)
        {
            var heightField = Terrain?.HeightField;
            if (heightField == null) return Transform?.Position.Y ?? 0;
            if (Transform == null) return 0;

            var terrainSize = Terrain!.TerrainSize;
            var maxHeight = Terrain.MaxHeight;

            position -= Transform.Position;

            var dimx = heightField.GetLength(0) - 1;
            var dimy = heightField.GetLength(1) - 1;

            var fx = (dimx + 1) / terrainSize.X;
            var fy = (dimy + 1) / terrainSize.Y;

            position.X *= fx;
            position.Z *= fy;

            int x = (int)Math.Floor(position.X);
            int z = (int)Math.Floor(position.Z);

            if (x < 0 || x > dimx) return Transform.Position.Y;
            if (z < 0 || z > dimy) return Transform.Position.Y;

            float xf = position.X - x;
            float zf = position.Z - z;

            float h1 = heightField[x, z];
            float h2 = heightField[Math.Min(dimx, x + 1), z];
            float h3 = heightField[x, Math.Min(dimy, z + 1)];

            float height = h1;
            height += (h2 - h1) * xf;
            height += (h3 - h1) * zf;

            return Transform.Position.Y + height * maxHeight;
        }

        // ───── Ground Coverage Decorator ──────────────────────────────────

        [StructLayout(LayoutKind.Sequential)]
        private struct ChannelHeader
        {
            public uint StartIndex;
            public uint Count;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct DecoratorSlotGPU
        {
            public float Density;     // absolute density (instances/m²)
            public float MinH, MaxH;
            public float MinW, MaxW;
            public uint LODCount;
            public uint LODTableOffset;
            // Precomputed root rotation matrix (CPU computes cos/sin, GPU reads directly)
            public float Rot00, Rot01, Rot02;
            public float Rot10, Rot11, Rot12;
            public float Rot20, Rot21, Rot22;
            public float SlopeBias;  // 0=upright, 1=fully slope-aligned
            public uint DecoMapSlice;     // slice in DecoMaps Texture2DArray (0xFFFFFFFF = none)
            public uint _pad0;            // unused (was ControlChannel)
            public uint Mode;             // 0=Mesh, 1=Billboard, 2=Cross
            public uint TextureIdx;       // bindless index (billboard/cross)
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct LODTableEntry
        {
            public uint MeshPartId;
            public float MaxDistance;
            public uint MaterialId;
            public uint _pad;
        }

        /// <summary>
        /// Builds GPU-side structured buffers from Terrain.Decorations.
        /// Collects unique DensityMaps into a DecoMapsArray, builds per-decoration
        /// DecoratorSlots with LOD table offsets and auto-resolved DecoMap slices,
        /// and registers all LOD meshes in MeshRegistry.
        /// </summary>
        public void BuildDecoratorBuffers()
        {
            if (Terrain == null) return;
            var decorations = Terrain.Decorations;
            if (decorations.Count == 0) return;

            Debug.Log($"[TerrainRenderer]", $"Building decorator buffers for {decorations.Count} decorations");

            var device = Engine.Device;

            // Dispose old buffers if rebuilding
            _decoratorHeadersBuffer?.Dispose();
            _decoratorSlotsBuffer?.Dispose();
            _decoratorLODTableBuffer?.Dispose();

            // Collect unique density maps and build DecoMapsArray
            var uniqueDensityMaps = new List<Texture>();
            var densityMapToSlice = new Dictionary<Texture, uint>();
            foreach (var deco in decorations)
            {
                if (deco.DensityMap != null && deco.DensityMap.Native != null && !densityMapToSlice.ContainsKey(deco.DensityMap))
                {
                    densityMapToSlice[deco.DensityMap] = (uint)uniqueDensityMaps.Count;
                    uniqueDensityMaps.Add(deco.DensityMap);
                }
            }
            if (uniqueDensityMaps.Count > 0)
                DecoMapsArray = Texture.CreateTexture2DArray(device, uniqueDensityMaps);
            else
                DecoMapsArray = null;

            // Flat list — all decorators go into a single header.
            var headers = new List<ChannelHeader>();
            var slots = new List<DecoratorSlotGPU>();
            var lodTable = new List<LODTableEntry>();

            for (int i = 0; i < decorations.Count; i++)
            {
                var deco = decorations[i];
                var mode = deco.Mode;

                // Mesh mode requires a StaticMesh
                if (mode == DecoratorMode.Mesh && deco.Mesh == null) continue;
                // Billboard/Cross mode requires a Texture
                if (mode != DecoratorMode.Mesh && deco.Texture == null && deco.Mesh == null) continue;

                uint lodTableOffset = (uint)lodTable.Count;
                uint lodCount = 0;
                uint textureIdx = 0;

                if (mode == DecoratorMode.Mesh)
                {
                    var mesh = deco.Mesh!;

                    // LOD0 = base mesh
                    if (mesh.Mesh != null)
                    {
                        foreach (var part in mesh.MeshParts)
                        {
                            int partId = MeshRegistry.Register(mesh.Mesh, part.MeshPartIndex);
                            uint matId = part.Material != null ? (uint)part.Material.MaterialID : 0;
                            float maxDist = mesh.LODGroup?.Ranges?.Count > 0
                                ? mesh.LODGroup.Ranges[0] * 1000f
                                : 100f;
                            lodTable.Add(new LODTableEntry { MeshPartId = (uint)partId, MaxDistance = maxDist, MaterialId = matId });
                            lodCount++;
                        }
                    }

                    // Additional LODs
                    for (int lod = 0; lod < mesh.LODs.Count; lod++)
                    {
                        var lodMesh = mesh.LODs[lod];
                        if (lodMesh.Mesh == null) continue;
                        foreach (var part in lodMesh.MeshParts)
                        {
                            int partId = MeshRegistry.Register(lodMesh.Mesh, part.MeshPartIndex);
                            uint matId = part.Material != null ? (uint)part.Material.MaterialID : 0;
                            float maxDist = mesh.LODGroup?.Ranges != null && lod + 1 < mesh.LODGroup.Ranges.Count
                                ? mesh.LODGroup.Ranges[lod + 1] * 1000f
                                : 50f * (lod + 1);
                            lodTable.Add(new LODTableEntry { MeshPartId = (uint)partId, MaxDistance = maxDist, MaterialId = matId });
                            lodCount++;
                        }
                    }
                }
                else
                {
                    // Billboard / Cross: use Texture's material or create a dummy LOD entry
                    // The texture is bound via the material system (Option A)
                    uint matId = 0;
                    if (deco.Texture != null)
                        textureIdx = (uint)deco.Texture.BindlessIndex;

                    // If there's a mesh reference, use its material for the billboard texture
                    if (deco.Mesh?.MeshParts?.Count > 0 && deco.Mesh.MeshParts[0].Material != null)
                        matId = (uint)deco.Mesh.MeshParts[0].Material!.MaterialID;

                    lodTable.Add(new LODTableEntry { MeshPartId = 0, MaxDistance = 200f, MaterialId = matId });
                    lodCount = 1;
                }

                float degToRad = MathF.PI / 180f;
                float rx = deco.RootRotation.X * degToRad;
                float ry = deco.RootRotation.Y * degToRad;
                float rz = deco.RootRotation.Z * degToRad;
                float cx = MathF.Cos(rx), sx = MathF.Sin(rx);
                float cy = MathF.Cos(ry), sy = MathF.Sin(ry);
                float cz = MathF.Cos(rz), sz = MathF.Sin(rz);

                slots.Add(new DecoratorSlotGPU
                {
                    Density = deco.Density * Terrain.DecorationDensity,
                    MinH = deco.HeightRange.X,
                    MaxH = deco.HeightRange.Y,
                    MinW = deco.WidthRange.X,
                    MaxW = deco.WidthRange.Y,
                    LODCount = lodCount,
                    LODTableOffset = lodTableOffset,
                    Rot00 = cy*cz,              Rot01 = cy*sz,              Rot02 = -sy,
                    Rot10 = sx*sy*cz - cx*sz,   Rot11 = sx*sy*sz + cx*cz,   Rot12 = sx*cy,
                    Rot20 = cx*sy*cz + sx*sz,   Rot21 = cx*sy*sz - sx*cz,   Rot22 = cx*cy,
                    SlopeBias = deco.SlopeBias,
                    DecoMapSlice = deco.DensityMap != null && densityMapToSlice.TryGetValue(deco.DensityMap, out var slice)
                        ? slice : 0xFFFFFFFF,
                    _pad0 = 0,
                    Mode = (uint)mode,
                    TextureIdx = textureIdx
                });
            }

            // Single header covering all slots
            headers.Add(new ChannelHeader
            {
                StartIndex = 0,
                Count = (uint)slots.Count
            });



            // Store absolute densities — shader does weighted selection proportional to each

            // Upload to GPU (creates new buffers + SRVs — only on structural changes)
            _decoratorHeadersBuffer = CreateAndUpload(headers);
            _decoratorSlotsBuffer = CreateAndUpload(slots);
            _decoratorLODTableBuffer = CreateAndUpload(lodTable);

            _decoratorBuffersBuilt = true;
            _decoControlDirty = true;  // trigger prepass rebuild
        }

        /// <summary>
        /// Creates the decoration control texture (RGBA16_UINT, 2 slices) and dispatches
        /// the prepass compute shader to build it from density maps.
        /// </summary>
        private void BuildDecorationControlTexture(ID3D12GraphicsCommandList cmd)
        {
            if (!_decoControlDirty || DecoMapsArray == null || _decoratorSlotsBuffer == null)
                return;

            var device = Engine.Device;
            var decoDesc = DecoMapsArray.Native.Description;
            int width = (int)decoDesc.Width;
            int height = (int)decoDesc.Height;

            // Create or recreate control texture
            _decoControlTex?.Dispose();
            _decoControlTex = device.CreateTexture2D(
                Format.R16G16B16A16_UInt, width, height, 2, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _decoControlUAV = device.AllocateBindlessIndex();
            var uavDesc = new UnorderedAccessViewDescription
            {
                Format = Format.R16G16B16A16_UInt,
                ViewDimension = UnorderedAccessViewDimension.Texture2DArray,
                Texture2DArray = new Texture2DArrayUnorderedAccessView
                {
                    MipSlice = 0,
                    FirstArraySlice = 0,
                    ArraySize = 2
                }
            };
            device.NativeDevice.CreateUnorderedAccessView(_decoControlTex, null, uavDesc, device.GetCpuHandle(_decoControlUAV));

            _decoControlSRV = device.AllocateBindlessIndex();
            var srvDesc = new ShaderResourceViewDescription
            {
                Format = Format.R16G16B16A16_UInt,
                ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2DArray = new Texture2DArrayShaderResourceView
                {
                    MostDetailedMip = 0,
                    MipLevels = 1,
                    FirstArraySlice = 0,
                    ArraySize = 2
                }
            };
            device.NativeDevice.CreateShaderResourceView(_decoControlTex, srvDesc, device.GetCpuHandle(_decoControlSRV));

            // Count valid slots
            uint slotCount = 0;
            if (Terrain?.Decorations != null)
            {
                foreach (var d in Terrain.Decorations)
                {
                    if (d.Mode == DecoratorMode.Mesh && d.Mesh == null) continue;
                    if (d.Mode != DecoratorMode.Mesh && d.Texture == null && d.Mesh == null) continue;
                    slotCount++;
                }
            }

            // Dispatch via ComputeShader
            _decoPrepassCS ??= new ComputeShader("decoration_prepass.hlsl");
            _decoPrepassCS.Begin(cmd);
            _decoPrepassCS.Set("DecoMaps", DecoMapsArray);              // Texture → BindlessIndex
            _decoPrepassCS.SetSRV("Slots", _decoratorSlotsBuffer!);     // GraphicsBuffer → SrvIndex
            _decoPrepassCS.Set("ControlUAV", _decoControlUAV);          // uint (raw UAV index)
            _decoPrepassCS.Set("SlotCount", slotCount);                  // uint
            _decoPrepassCS.Dispatch(cmd, (uint)((width + 7) / 8), (uint)((height + 7) / 8));

            cmd.ResourceBarrierUnorderedAccessView(_decoControlTex);

            _decoControlDirty = false;
            _bakedAlbedoDirty = true; // Rebuild albedo when control changes
        }

        /// <summary>
        /// Bakes terrain splatmap layers into a single 256×256 albedo texture
        /// for ground color blending in vegetation shaders.
        /// </summary>
        private void BakeTerrainAlbedo(ID3D12GraphicsCommandList cmd)
        {
            if (!_bakedAlbedoDirty) return;
            if (ControlMapsArray == null || DiffuseMapsArray == null) return;

            var device = Engine.Device;

            // Upload tiling data as structured buffer
            _tilingBuffer ??= GraphicsBuffer.CreateUpload<Vector4>(32);
            _tilingBuffer.Upload<Vector4>(_layerTiling.AsSpan());

            // Create or recreate baked albedo texture
            if (_bakedAlbedoTex == null)
            {
                _bakedAlbedoTex = device.CreateTexture2D(
                    Format.R8G8B8A8_UNorm, BakedAlbedoSize, BakedAlbedoSize, 1, 1,
                    ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

                _bakedAlbedoUAV = device.AllocateBindlessIndex();
                var uavDesc = new UnorderedAccessViewDescription
                {
                    Format = Format.R8G8B8A8_UNorm,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
                };
                device.NativeDevice.CreateUnorderedAccessView(_bakedAlbedoTex, null, uavDesc, device.GetCpuHandle(_bakedAlbedoUAV));

                _bakedAlbedoSRV = device.AllocateBindlessIndex();
                var srvDesc = new ShaderResourceViewDescription
                {
                    Format = Format.R8G8B8A8_UNorm,
                    ViewDimension = ShaderResourceViewDimension.Texture2D,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2D = new Texture2DShaderResourceView
                    {
                        MostDetailedMip = 0,
                        MipLevels = 1
                    }
                };
                device.NativeDevice.CreateShaderResourceView(_bakedAlbedoTex, srvDesc, device.GetCpuHandle(_bakedAlbedoSRV));
            }

            // Dispatch via ComputeShader
            _albedoBakeCS ??= new ComputeShader("terrain_albedo_bake.hlsl");
            _albedoBakeCS.Begin(cmd);
            _albedoBakeCS.Set("ControlMaps", ControlMapsArray);      // Texture → BindlessIndex
            _albedoBakeCS.Set("DiffuseMaps", DiffuseMapsArray);      // Texture → BindlessIndex
            _albedoBakeCS.Set("OutputUAV", _bakedAlbedoUAV);         // uint (raw UAV index)
            _albedoBakeCS.SetSRV("TilingBuf", _tilingBuffer);        // GraphicsBuffer → SrvIndex

            uint groups = (uint)((BakedAlbedoSize + 7) / 8);
            _albedoBakeCS.Dispatch(cmd, groups, groups);

            cmd.ResourceBarrierUnorderedAccessView(_bakedAlbedoTex);

            _bakedAlbedoDirty = false;
        }

        /// <summary>
        /// Updates slot data in existing GPU buffers without creating new resources.
        /// Called when density, scale, rotation, or radius change (but decoration structure unchanged).
        /// </summary>
        private unsafe void UpdateDecoratorBufferData()
        {
            if (Terrain == null || _decoratorSlotsBuffer == null) return;

            var decorations = Terrain.Decorations;
            if (decorations.Count == 0) return;

            // Rebuild slot data with current values
            var slots = new List<DecoratorSlotGPU>();

            // Read existing LOD/structural data from the current buffer
            var pRead = _decoratorSlotsBuffer.Map<DecoratorSlotGPU>();
            int validCount = 0;
            foreach (var d in decorations)
            {
                if (d.Mode == DecoratorMode.Mesh && d.Mesh == null) continue;
                if (d.Mode != DecoratorMode.Mesh && d.Texture == null && d.Mesh == null) continue;
                validCount++;
            }
            var existing = new Span<DecoratorSlotGPU>(pRead, validCount);
            var existingLodCounts = new uint[validCount];
            var existingLodOffsets = new uint[validCount];
            var existingDecoSlices = new uint[validCount];
            var existingTextureIdx = new uint[validCount];
            for (int i = 0; i < validCount; i++)
            {
                existingLodCounts[i] = existing[i].LODCount;
                existingLodOffsets[i] = existing[i].LODTableOffset;
                existingDecoSlices[i] = existing[i].DecoMapSlice;
                existingTextureIdx[i] = existing[i].TextureIdx;
            }
            _decoratorSlotsBuffer.Unmap();

            float degToRad = MathF.PI / 180f;
            int slotIdx = 0;

            foreach (var deco in decorations)
            {
                // Same filter as BuildDecoratorBuffers
                if (deco.Mode == DecoratorMode.Mesh && deco.Mesh == null) continue;
                if (deco.Mode != DecoratorMode.Mesh && deco.Texture == null && deco.Mesh == null) continue;

                float rx = deco.RootRotation.X * degToRad;
                float ry = deco.RootRotation.Y * degToRad;
                float rz = deco.RootRotation.Z * degToRad;
                float cx = MathF.Cos(rx), sx = MathF.Sin(rx);
                float cy = MathF.Cos(ry), sy = MathF.Sin(ry);
                float cz = MathF.Cos(rz), sz = MathF.Sin(rz);

                slots.Add(new DecoratorSlotGPU
                {
                    Density = deco.Density * Terrain.DecorationDensity,
                    MinH = deco.HeightRange.X,
                    MaxH = deco.HeightRange.Y,
                    MinW = deco.WidthRange.X,
                    MaxW = deco.WidthRange.Y,
                    LODCount = slotIdx < existingLodCounts.Length ? existingLodCounts[slotIdx] : 0,
                    LODTableOffset = slotIdx < existingLodOffsets.Length ? existingLodOffsets[slotIdx] : 0,
                    Rot00 = cy*cz,              Rot01 = cy*sz,              Rot02 = -sy,
                    Rot10 = sx*sy*cz - cx*sz,   Rot11 = sx*sy*sz + cx*cz,   Rot12 = sx*cy,
                    Rot20 = cx*sy*cz + sx*sz,   Rot21 = cx*sy*sz - sx*cz,   Rot22 = cx*cy,
                    SlopeBias = deco.SlopeBias,
                    DecoMapSlice = slotIdx < existingDecoSlices.Length ? existingDecoSlices[slotIdx] : 0xFFFFFFFF,
                    _pad0 = 0,
                    Mode = (uint)deco.Mode,
                    TextureIdx = slotIdx < existingTextureIdx.Length ? existingTextureIdx[slotIdx] : 0
                });
                slotIdx++;
            }

            // Re-upload slot data
            if (slots.Count > 0)
            {
                var span = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(slots);
                _decoratorSlotsBuffer.Upload<DecoratorSlotGPU>(span);
            }
        }

        /// <summary>
        /// Create an upload GraphicsBuffer from a list and upload the data.
        /// </summary>
        private static GraphicsBuffer CreateAndUpload<T>(List<T> data) where T : unmanaged
        {
            int count = Math.Max(1, data.Count);
            var buffer = GraphicsBuffer.CreateUpload<T>(count);
            if (data.Count > 0)
            {
                var span = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(data);
                buffer.Upload<T>(span);
            }
            return buffer;
        }

        private void DispatchDecorator(ID3D12GraphicsCommandList commandList, int frameIndex, RenderPass pass = RenderPass.Opaque)
        {
            if (DecoratorMaterial?.Effect == null || Terrain == null) return;

            // Build control texture on first frame or after decoration changes
            BuildDecorationControlTexture(commandList);
            // Bake terrain albedo for ground color blending in vegetation
            BakeTerrainAlbedo(commandList);

            var device = Engine.Device;
            var camPos = Camera.Main!.Position;

            float range = Terrain.DecorationRadius;
            const int TILE_SIZE = 8;

            // Tile size = one CONTROL pixel; cell size = tile / 8
            float controlWidth = _decoControlTex != null
                ? _decoControlTex.Description.Width : 1024f;
            float tileSize = Terrain.TerrainSize.X / controlWidth;
            float cs = tileSize / TILE_SIZE;
            int tilesPerSide = Math.Max(1, (int)MathF.Ceiling(2.0f * range / tileSize));

            //Debug.Log($"[Grass] tileSize={tileSize:F3} cellSize={cs:F3} range={range:F0} tiles={tilesPerSide}x{tilesPerSide}={tilesPerSide*tilesPerSide} decoMapsIdx={DecoMapsArray?.BindlessIndex ?? 0}");

            // Select correct PSO pass and apply
            DecoratorMaterial.SetPass(pass);
            DecoratorMaterial.Apply(commandList, device);

            // Push constants
            commandList.SetGraphicsRoot32BitConstant(0, _decoratorHeadersBuffer!.SrvIndex, 0);
            commandList.SetGraphicsRoot32BitConstant(0, _decoratorSlotsBuffer!.SrvIndex, 1);
            commandList.SetGraphicsRoot32BitConstant(0, _decoratorLODTableBuffer!.SrvIndex, 2);
            commandList.SetGraphicsRoot32BitConstant(0, MeshRegistry.SrvIndex, 3);
            if (Terrain.Heightmap != null)
                commandList.SetGraphicsRoot32BitConstant(0, Terrain.Heightmap.BindlessIndex, 4);
            commandList.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(Terrain.TerrainSize.X), 5);
            commandList.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(Terrain.TerrainSize.Y), 6);
            commandList.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(Terrain.MaxHeight), 7);
            commandList.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(camPos.X), 8);
            commandList.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(camPos.Y), 9);
            commandList.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(camPos.Z), 10);
            commandList.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(Transform.WorldPosition.X), 11);
            commandList.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(Transform.WorldPosition.Z), 12);
            commandList.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(Transform.WorldPosition.Y), 13);
            commandList.SetGraphicsRoot32BitConstant(0, Graphics.Material.MaterialsBufferIndex, 14);
            // Slot 15: cell size (tile / 8, locked to CONTROL resolution)
            commandList.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(cs), 15);
            // Slot 16: decoration radius
            commandList.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(range), 16);
            // Slot 17: wind animation time
            commandList.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(Base.Time.TotalTime), 17);
            // Slot 18: decoration control texture SRV (from prepass compute)
            commandList.SetGraphicsRoot32BitConstant(0, _decoControlSRV, 18);
            // Slot 19: baked terrain albedo SRV (for ground color blending)
            commandList.SetGraphicsRoot32BitConstant(0, _bakedAlbedoSRV, 19);

            // Shadow single-pass: bind VP structured buffer and cascade count
            if (pass == RenderPass.Shadow)
            {
                commandList.SetGraphicsRoot32BitConstant(0, DirectionalLight.CurrentCascadeSrvIndex, 20);
                // With SDSM, all cascades may cover nearby geometry — use them all.
                // With fixed splits, skip outermost cascade (grass detail not visible at that distance).
                int grassShadowCascades = Engine.Settings.UseAdaptiveSplits
                    ? DirectionalLight.CascadeCount
                    : Math.Max(1, DirectionalLight.CascadeCount - 1);
                commandList.SetGraphicsRoot32BitConstant(0, (uint)grassShadowCascades, 21);
                // Shadow Hi-Z pyramid SRV for per-instance occlusion culling
                var shadowPyramid = DeferredRenderer.Current?.ShadowHiZPyramid;
                uint shadowHiZSrv = (shadowPyramid?.Ready == true) ? shadowPyramid.FullSRV : 0u;
                commandList.SetGraphicsRoot32BitConstant(0, shadowHiZSrv, 22);
            }

            using var commandList6 = commandList.QueryInterface<ID3D12GraphicsCommandList6>();
            commandList6.DispatchMesh((uint)tilesPerSide, (uint)tilesPerSide, 1);
        }
    }
}
