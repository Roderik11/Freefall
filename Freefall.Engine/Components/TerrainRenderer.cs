using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.ComponentModel;
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
    [Icon("icon_terrain.png")]
    public class TerrainRenderer : Freefall.Base.Component, IDraw, IHeightProvider
    {
        public static bool ComputeReady { get; set; }
        public static string ComputeError { get; set; } = "";

        // ───── Asset Reference ────────────────────────────────────────────
        public Terrain? Terrain;

        [Browsable(false)]
        public Material? Material = InternalAssets.TerrainMaterial;

        [Browsable(false)]
        public Material? DecoratorMaterial = InternalAssets.DecoratorMaterial;

        // ───── Rendering Parameters ───────────────────────────────────────
        public int MaxDepth = 7;
        private int MaxPatches = 32768;
        private const int MinPatches = 32768;

        private Vector4[] _layerTiling = new Vector4[32];

        // ───── Internal State ─────────────────────────────────────────────
        private const int FrameCount = 3;

        // Per-instance baker — owns all GPU bake resources for this terrain
        private TerrainBaker _baker;
        /// <summary>Expose the baker for save/readback operations (TerrainLoader).</summary>
        public TerrainBaker Baker => _baker;

        // Compute pipeline — restricted quadtree (auto-discovers all #pragma kernel entries)
        private ComputeShader? _quadtreeCS;
        private int _kMarkSplits, _kEmitLeaves, _kBuildMinMaxMip, _kBuildDrawArgs, _kEmitLeavesShadow;
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
        
        // Constant buffers per frame — split for b0 (frustum), b1 (Hi-Z), b2 (terrain params)
        private ID3D12Resource[] _frustumPlaneBuffers = new ID3D12Resource[FrameCount]; // b0 = root slot 1
        private ID3D12Resource[] _hizParamBuffers = new ID3D12Resource[FrameCount];     // b1 = root slot 2
        private ID3D12Resource[] _terrainParamBuffers = new ID3D12Resource[FrameCount]; // b2 = root slot 3 (main pass)
        private ID3D12Resource[] _shadowTerrainParamBuffers = new ID3D12Resource[FrameCount]; // b2 = root slot 3 (shadow pass)
        private Matrix4x4 _previousFrameViewProjection;

        // Must match cbuffer FrustumPlanes : register(b0)
        [StructLayout(LayoutKind.Sequential)]
        private struct FrustumPlanesData
        {
            public Vector4 Plane0, Plane1, Plane2, Plane3, Plane4, Plane5;
        }

        // Must match cbuffer HiZParams : register(b1)
        [StructLayout(LayoutKind.Sequential)]
        private struct HiZParamsData
        {
            public Matrix4x4 OcclusionProjection;
            public uint HiZSrvIdx;
            public float HiZWidth, HiZHeight;
            public uint HiZMipCount;
            public float NearPlane;
            public uint CullStatsUAVIdx;
            public uint FrustumDebugMode;
            public float Pad;
        }

        // Must match cbuffer TerrainParams : register(b2)
        [StructLayout(LayoutKind.Sequential)]
        private struct TerrainParamsData
        {
            public Vector3 CameraPos;  public float MaxHeight;
            public Vector3 RootCenter; public uint  MaxDepth;
            public Vector3 RootExtents;public uint  TotalNodes;
            public Vector2 TerrainSize;public float PixelErrorThreshold; public float ScreenHeight;
            public float TanHalfFov;   public uint  TransformSlot;       public uint  MaterialId; public uint MeshPartId;
            public uint  MaxPatches;   public uint  HeightTexIdx;         public uint  Pad0, Pad1;
        }
        
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
        private Texture? DiffuseMapsArray;
        private Texture? NormalMapsArray;
        private Texture? HeightMapsArray;

        private int _totalNodes;

        // ───── Ground Coverage Decorator ──────────────────────────────────
        private GraphicsBuffer? _decoratorHeadersBuffer;
        private GraphicsBuffer? _decoratorSlotsBuffer;
        private GraphicsBuffer? _decoratorLODTableBuffer;
        private GraphicsBuffer? _decoratorGroupsBuffer;  // one (variant offset, count) per decorator slot
        private int _decoSlotCapacity, _decoLodCapacity, _decoGroupCapacity;
        private int _decoVariantCount;
        private bool _decoratorBuffersBuilt;
        private bool _decoratorDispatched;

        private int _decoratorDispatchGroupsX;
        private int _decoratorDispatchGroupsY;

        // ───── Compute Prepass (grass_compute.hlsl) ──────────────────────
        private ComputeShader? _grassCS;
        private int _kBakeNormals, _kSpawnInstances, _kBuildDecoDrawArgs, _kBinMeshInstances;
        private GraphicsBuffer? _decoInstanceBuffer;    // StructuredBuffer<DecoInstance> = 64 bytes
        private GraphicsBuffer? _instanceCounterBuffer; // RWByteAddressBuffer (2 uints: billboard count + mesh count)
        private GraphicsBuffer? _decoDispatchArgsBuffer; // RWByteAddressBuffer (4 uints: 3 for DispatchMesh + 1 mesh count)
        private bool _decoBuffersCreated;
        private bool _decoDispatchLogged;
        private int _maxDecoInstances;

        // ───── Mesh-Mode Decorators ──────────────────────────────────────────
        private GraphicsBuffer? _meshDecoInstanceBuffer;    // unsorted mesh instances (DecoInstance, 64 bytes)
        private GraphicsBuffer? _sortedMeshInstanceBuffer;  // sorted by mesh type (DecoInstance, 64 bytes)
        private GraphicsBuffer? _meshDrawArgsBuffer;        // BindlessDrawCommand per mesh type (72 bytes × 32)
        private GraphicsBuffer? _meshDrawCountBuffer;       // RWByteAddressBuffer (1 uint: draw count)
        private Material? _meshDecoratorMaterial;           // grass_mesh.fx

        // ───── Debug Stats (read by editor SettingsControls) ─────────────
        public static int LastInstanceCount { get; set; }
        public static int LastMeshInstanceCount { get; set; }
        public static int LastMeshDrawCount { get; set; }
        public static int LastMaxInstances { get; set; }
        public static int LastDispatchN { get; set; }
        private ID3D12Resource? _instanceCounterReadback;
        private IntPtr _instanceCounterReadbackPtr;

        // ───── Baked Terrain Normals (one-time) ──────────────────────────
        private ID3D12Resource? _bakedNormalTex;        // R16G16_SNORM
        private uint _bakedNormalUAV;
        private uint _bakedNormalSRV;
        private bool _bakedNormalsDirty = true;

        // ───── Decoration Control Prepass ─────────────────────────────────
        private ID3D12Resource? _decoControlTex;     // RGBA16_UINT, 2 slices
        private uint _decoControlUAV;
        private uint _decoControlSRV;

        // ───── Baked Terrain Albedo ───────────────────────────────────────
        private ComputeShader? _albedoBakeCS;
        private ID3D12Resource? _bakedAlbedoTex;     // RGBA8, 256×256
        private uint _bakedAlbedoUAV;
        private uint _bakedAlbedoSRV;
        private GraphicsBuffer? _tilingBuffer;       // StructuredBuffer<float4>, 32 entries
        private const int BakedAlbedoSize = 256;

        // ───── Palette ──────────────────────────────────────────────────
        // What this terrain renders is derived from the stamps in scope, never stored: the layers its
        // splat stamps reference (one channel of the packed control array each, in bake order) and the
        // decorators its deco stamps add (one slot of the decoration control texture each).
        private List<TerrainLayer> _layerPalette = new();
        private List<TerrainDecorator> _decoratorPalette = new();
        private List<SplatStamp> _splatStamps = new();   // in bake order, collected with the palette
        private List<DecoStamp> _decoStamps = new();
        private readonly List<string> _layerWarnings = new();
        private readonly List<string> _decoWarnings = new();

        /// <summary>The layers this terrain renders, by control-array channel.</summary>
        [Browsable(false)]
        public IReadOnlyList<TerrainLayer> LayerPalette => _layerPalette;

        /// <summary>The decorators this terrain renders, by control-texture slot.</summary>
        [Browsable(false)]
        public IReadOnlyList<TerrainDecorator> DecoratorPalette => _decoratorPalette;

        /// <summary>Problems found while resolving the palette (too many layers, stamps that cannot apply).</summary>
        [Browsable(false)]
        public IReadOnlyList<string> PaletteWarnings => _layerWarnings.Concat(_decoWarnings).ToList();

        // Decoration coverage bake prepared on the main thread, run by the next decorator dispatch
        private TerrainBaker.CoveragePlan? _pendingDecoPlan;

        // ───── Height Bake (GPU layer compositor) ─────────────────────────
        private bool _needHeightFieldReadback;

        // ───── Cached Packed Splatmap Array ────────────────────────────────
        private ID3D12Resource _packedControlArray;
        private Texture _packedControlTexture;     // stable wrapper for ControlMapsArray
        private uint _packedControlSRV;
        private uint[] _packedSliceUAVs;            // per-slice UAVs for direct packing
        private uint _packedControlArrayUAV;        // full-array UAV for stamp overlay
        private int _packedArrayResolution;
        private int _packedSliceCount;

        // ───── Lifecycle ──────────────────────────────────────────────────

        protected override void Awake()
        {
            // GPU resource init — matches TerrainGPU.Awake() exactly.
            // Called before YAML properties are applied, but MaxDepth defaults to 7
            // and CreateHeightRangePyramid doesn't read the heightmap —
            // BuildHeightRangePyramid is deferred to the first DispatchQuadtreeEval.
            _baker = new TerrainBaker();
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

            MessageDispatcher.AddListener(EngineMsg.StampChanged, OnStampChanged);
            MessageDispatcher.AddListener(EngineMsg.SplineChanged, OnSplineChanged);
            MessageDispatcher.AddListener("AssetDirty", OnAssetDirty);
        }

        public override void Destroy()
        {
            Debug.Log($"[TerrainRenderer] Destroy: UID={UID} Entity={Entity?.Name} _computeInitialized={_computeInitialized}");
            _computeInitialized = false;
            var device = Engine.Device;

            // Nothing below is released here. The command lists of the frames in flight still bind these
            // buffers, textures and pipeline states, copy into the readback buffers and index the descriptor
            // heap with the bindless slots; releasing them now risks removing the device (see
            // Animator.Destroy). Everything goes to the device, which releases it once those frames are
            // done. A raw resource's slots go the same way, or the next allocation could rewrite a
            // descriptor those frames still read.

            // ── Per-frame GraphicsBuffer arrays ──
            for (int i = 0; i < FrameCount; i++)
            {
                device?.DeferDispose(_descriptorBuffers[i]);
                device?.DeferDispose(_sphereBuffers[i]);
                device?.DeferDispose(_subbatchIdBuffers[i]);
                device?.DeferDispose(_terrainDataBuffers[i]);
                device?.DeferDispose(_counterBuffers[i]);
                device?.DeferDispose(_splitFlagsBuffers[i]);
                device?.DeferDispose(_indirectArgsBuffers[i]);
                device?.DeferDispose(_shadowArgsBuffers[i]);

                device?.DeferDispose(_shadowDescriptorBuffers[i]);
                device?.DeferDispose(_shadowTerrainDataBuffers[i]);
                device?.DeferDispose(_shadowSphereBuffers[i]);
                device?.DeferDispose(_shadowSubbatchIdBuffers[i]);
                device?.DeferDispose(_shadowCounterBuffers[i]);
                device?.DeferDispose(_shadowCascadeIdxBuffers[i]);
            }

            // ── Per-frame ID3D12Resource constant buffers + shadow cascade ──
            for (int i = 0; i < FrameCount; i++)
            {
                device?.DeferDispose(_frustumPlaneBuffers[i]);
                _frustumPlaneBuffers[i] = null;
                device?.DeferDispose(_hizParamBuffers[i]);
                _hizParamBuffers[i] = null;
                device?.DeferDispose(_terrainParamBuffers[i]);
                _terrainParamBuffers[i] = null;
                device?.DeferDispose(_shadowTerrainParamBuffers[i]);
                _shadowTerrainParamBuffers[i] = null;
                device?.DeferDispose(_shadowCascadeBuffers[i]);
                _shadowCascadeBuffers[i] = null;

                if (_shadowCascadeBufferSrvs[i] != 0)
                    device?.DeferReleaseBindlessIndex(_shadowCascadeBufferSrvs[i]);
                _shadowCascadeBufferSrvs[i] = 0;
            }

            // ── Readback buffers ──
            device?.DeferDispose(_shadowCounterReadback);
            device?.DeferDispose(_shadowArgsReadback);
            device?.DeferDispose(_instanceCounterReadback);

            // ── Height range mip pyramid ──
            device?.DeferDispose(_heightRangePyramid);
            if (_heightRangeMipUAVs != null)
                foreach (var idx in _heightRangeMipUAVs)
                    if (idx != 0) device?.DeferReleaseBindlessIndex(idx);
            if (_heightRangeMipSRVs != null)
                foreach (var idx in _heightRangeMipSRVs)
                    if (idx != 0) device?.DeferReleaseBindlessIndex(idx);
            if (_heightRangePyramidSRV != 0)
                device?.DeferReleaseBindlessIndex(_heightRangePyramidSRV);

            // ── Baked normals ──
            device?.DeferDispose(_bakedNormalTex);
            if (_bakedNormalUAV != 0) device?.DeferReleaseBindlessIndex(_bakedNormalUAV);
            if (_bakedNormalSRV != 0) device?.DeferReleaseBindlessIndex(_bakedNormalSRV);

            // ── Decoration control prepass ──
            device?.DeferDispose(_decoControlTex);
            if (_decoControlUAV != 0) device?.DeferReleaseBindlessIndex(_decoControlUAV);
            if (_decoControlSRV != 0) device?.DeferReleaseBindlessIndex(_decoControlSRV);

            // ── Baked albedo ──
            device?.DeferDispose(_bakedAlbedoTex);
            if (_bakedAlbedoUAV != 0) device?.DeferReleaseBindlessIndex(_bakedAlbedoUAV);
            if (_bakedAlbedoSRV != 0) device?.DeferReleaseBindlessIndex(_bakedAlbedoSRV);
            device?.DeferDispose(_tilingBuffer);

            // ── Packed control array ──
            device?.DeferDispose(_packedControlArray);
            if (_packedControlSRV != 0) device?.DeferReleaseBindlessIndex(_packedControlSRV);
            if (_packedSliceUAVs != null)
                foreach (var idx in _packedSliceUAVs)
                    if (idx != 0) device?.DeferReleaseBindlessIndex(idx);
            if (_packedControlArrayUAV != 0) device?.DeferReleaseBindlessIndex(_packedControlArrayUAV);

            // ── Decorator buffers ──
            device?.DeferDispose(_decoratorHeadersBuffer);
            device?.DeferDispose(_decoratorSlotsBuffer);
            device?.DeferDispose(_decoratorLODTableBuffer);
            device?.DeferDispose(_decoratorGroupsBuffer);

            // ── Deco compute buffers ──
            device?.DeferDispose(_decoInstanceBuffer);
            device?.DeferDispose(_instanceCounterBuffer);
            device?.DeferDispose(_decoDispatchArgsBuffer);

            // ── Mesh-mode decorator buffers ──
            device?.DeferDispose(_meshDecoInstanceBuffer);
            device?.DeferDispose(_sortedMeshInstanceBuffer);
            device?.DeferDispose(_meshDrawArgsBuffer);
            device?.DeferDispose(_meshDrawCountBuffer);

            // ── Compute shaders (pipeline states + their constant buffers) ──
            device?.DeferDispose(_quadtreeCS);
            device?.DeferDispose(_grassCS);
            device?.DeferDispose(_albedoBakeCS);

            // ── Patch mesh ──
            device?.DeferDispose(_patchMesh);

            // ── Baker: release scratch buffers (stamp descriptors, spline points) ──
            // The height texture is NOT released — it is owned by the
            // cached Terrain asset and persist for reuse on next scene load.
            device?.DeferDispose(_baker);
            _baker = null;

            MessageDispatcher.RemoveListener(EngineMsg.StampChanged, OnStampChanged);
            MessageDispatcher.RemoveListener(EngineMsg.SplineChanged, OnSplineChanged);
            MessageDispatcher.RemoveListener("AssetDirty", OnAssetDirty);
        }

        private bool _textureArraysInitialized;
        private bool _initialBakeRequested;

        /// <summary>
        /// Inspector / command-server edits (e.g. assigning a different Terrain at runtime) must rebuild
        /// everything derived from the terrain asset, as a scene load would.
        /// </summary>
        public override void OnMemberChanged()
        {
            _textureArraysInitialized = false;
            Terrain?.MarkForUpdate(TerrainDirtyFlags.All);
        }

        public void Draw()
        {
            if (Camera.Main == null || Terrain == null || !_computeInitialized) return;

            var material = Material;
            if (material == null || material.Effect == null) return;

            int frameIndex = Engine.FrameIndex % FrameCount;

            // Upload saved baked heightmap from cache (skips GPU compositor if present)
            if (Terrain.PendingBakedHeightmapBytes != null)
            {
                var baker = _baker;
                var bytes = Terrain.PendingBakedHeightmapBytes;
                Terrain.PendingBakedHeightmapBytes = null; // consumed

                int expectedRes = Terrain.EffectiveHeightmapResolution;
                int cachedRes = (int)Math.Sqrt(bytes.Length / 2); // R16_UNorm = 2 bpp

                if (cachedRes == expectedRes)
                {
                    var tex = baker.UploadBakedHeightmap(bytes, expectedRes);
                    if (tex != null)
                    {
                        Terrain.BakedHeightmap = tex;
                        Terrain.ConsumeFlags(TerrainDirtyFlags.HeightBake); // loaded from cache, skip bake
                        _heightRangePyramidBuilt = false;
                        Terrain.MarkForUpdate(TerrainDirtyFlags.AlbedoBake);
                        _needHeightFieldReadback = true;
                    }
                }
                else
                {
                    // Resolution mismatch (migration from power-of-2) — discard cache, force rebake
                    Debug.Log($"[TerrainRenderer] Cached heightmap {cachedRes}x{cachedRes} != expected {expectedRes}x{expectedRes}, forcing rebake");
                    Terrain.MarkForUpdate(TerrainDirtyFlags.HeightBake);
                }
            }

            // The Terrain asset is shared across scene loads, so its BakedHeightmap may hold another scene's
            // (or an older) stamp result, and stamps only send StampChanged when edited, not when loaded.
            // Bake once per renderer so this scene's stamps always win; the cached heightmap above only
            // bridges the first frames.
            if (!_initialBakeRequested)
            {
                _initialBakeRequested = true;
                Terrain.MarkForUpdate(TerrainDirtyFlags.HeightBake | TerrainDirtyFlags.SplatPack |
                                      TerrainDirtyFlags.AlbedoBake | TerrainDirtyFlags.DecoPrepass);
            }

            // ── Height: lay out the bake from the height stamps in scope ──
            bool heightBake = Terrain.ConsumeFlags(TerrainDirtyFlags.HeightBake);
            TerrainBaker.HeightPlan? heightPlan = heightBake ? _baker.PrepareHeight(Terrain, this) : null;
            bool hasHeightWork = heightPlan is { HasWork: true };
            if (heightBake)
            {
                // A bake nobody gave a region for (first bake, terrain settings) may change anything
                if (!_heightRegionMarked) _heightChangeAll = true;
                _heightRegionMarked = false;
            }

            // ── Palette: the layers and decorators this terrain renders are whatever its stamps reference ──
            // Resolved before anything built from it (texture arrays, layer params, decorator buffers).
            // New heights change what the stamps' height/slope filters let through, so they rebake both.
            bool splatDirty = Terrain.ConsumeFlags(TerrainDirtyFlags.SplatPack) || hasHeightWork;
            bool decoDirty = Terrain.ConsumeFlags(TerrainDirtyFlags.DecoPrepass) || splatDirty;

            if (splatDirty || decoDirty)
            {
                if (splatDirty) RefreshLayerPalette();
                RefreshDecoratorPalette();
            }

            if (Terrain.ConsumeFlags(TerrainDirtyFlags.TextureArrays) || !_textureArraysInitialized)
            {
                try
                {
                    RebuildTextureArrays();
                    _textureArraysInitialized = true;
                }
                catch (Exception ex)
                {
                    Debug.LogError("[TerrainRenderer]", $"RebuildTextureArrays failed: {ex.Message}");
                    if (Engine.Device.IsDeviceLost) return;
                }
            }

            if (Terrain.ConsumeFlags(TerrainDirtyFlags.LayerParams))
                UpdateLayerParams();

            // GPU height bake (runs before any heightmap access)
            if (hasHeightWork)
            {
                var baker = _baker;
                var terrain = Terrain;
                var plan = heightPlan!;
                var renderer = this;
                System.Threading.Interlocked.Increment(ref _heightBakesPending);
                CommandBuffer.Enqueue(RenderPass.Opaque, (list) =>
                {
                    System.Threading.Interlocked.Decrement(ref renderer._heightBakesPending);
                    baker.BakeHeight(terrain, plan, list);
                    _heightRangePyramidBuilt = false; // force rebuild with new heights
                    terrain.MarkForUpdate(TerrainDirtyFlags.AlbedoBake); // re-bake albedo with new terrain shape
                    _needHeightFieldReadback = true; // trigger CPU-side heightfield rebuild next frame
                    renderer._bakedNormalsDirty = true; // normals depend on height
                    renderer.BakeTerrainNormals(list);
                });
            }

            // Capture heightmap AFTER cache upload / bake — BakedHeightmap may have just been set above
            var heightmap = Terrain.Heightmap;

            // Debounced CPU heightfield readback — only when baking has settled (not while a stamp is dragged)
            if (_needHeightFieldReadback && !Terrain.NeedsUpdate(TerrainDirtyFlags.HeightBake))
            {
                _needHeightFieldReadback = false;
                var heights = _baker.ReadbackHeightmap();
                if (heights != null && Terrain != null)
                {
                    Terrain.SetHeightField(heights);
                    // Surface-snapped geometry (RuntimeMesh) and projected PCG output sampled the old heights —
                    // let whatever lies in the changed region rebuild
                    var change = new TerrainHeightsChange(Terrain, _heightChangeAll, _heightChangeMin, _heightChangeMax);

                    // A bake that is enqueued but has not run yet (at load: this readback is of the cached
                    // heightmap) still owes its region to the readback that follows it.
                    if (System.Threading.Volatile.Read(ref _heightBakesPending) == 0)
                    {
                        _heightChangeAll = false;
                        _heightChangeMin = new Vector2(float.MaxValue);
                        _heightChangeMax = new Vector2(float.MinValue);
                        SnapshotStampRegions();
                    }
                    MessageDispatcher.Send(EngineMsg.TerrainHeightsChanged, change);
                }
            }

            // ── Splat: composite the splat stamps into the packed layer weights ──
            if (splatDirty)
            {
                if (_layerPalette.Count > 0)
                {
                    int res = Terrain.EffectiveSplatmapResolution;
                    int sliceCount = (_layerPalette.Count + 3) / 4;

                    // Ensure packed array exists at correct resolution/slice count
                    EnsurePackedControlArray(res, sliceCount);
                    ControlMapsArray = _packedControlTexture;

                    var plan = _baker.PrepareSplat(Terrain, this, _splatStamps, _layerPalette);
                    var baker = _baker;
                    var terrain = Terrain;
                    var packedArray = _packedControlArray;
                    var arrayUAV = _packedControlArrayUAV;
                    CommandBuffer.Enqueue(RenderPass.Opaque, (list) =>
                    {
                        // The height bake enqueued above has run by now, so this reads the new heights
                        var hm = terrain.Heightmap;
                        uint hmSrv = hm != null ? (uint)hm.BindlessIndex : 0;

                        list.ResourceBarrierTransition(packedArray,
                            ResourceStates.Common, ResourceStates.UnorderedAccess);

                        baker.BakeSplat(plan, list, terrain, hmSrv, arrayUAV, res, sliceCount);
                        list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(packedArray)));

                        // Back to common for shader reads
                        list.ResourceBarrierTransition(packedArray,
                            ResourceStates.UnorderedAccess, ResourceStates.Common);

                        terrain.MarkForUpdate(TerrainDirtyFlags.AlbedoBake);
                    });
                }
                else
                {
                    // No splat stamp references a layer: nothing to render the surface with
                    ControlMapsArray = InternalAssets.BlackArray;
                }
            }

            // ── Decoration coverage: laid out here, run by the decorator dispatch below ──
            if (decoDirty)
            {
                var decoPlan = _baker.PrepareDeco(Terrain, this, _decoStamps, _decoratorPalette, _layerPalette);
                System.Threading.Interlocked.Exchange(ref _pendingDecoPlan, decoPlan);
            }

            // Set shared material params
            material.SetParameter("CameraPos", Camera.Main.Position);
            material.SetParameter("HeightTexel", 1.0f / (heightmap != null ? Terrain.EffectiveHeightmapResolution : 1024));
            material.SetParameter("MaxHeight", Terrain.MaxHeight);
            material.SetParameter("BlendDepth", Terrain.LayerBlendDepth);
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
            if (HeightMapsArray != null) material.SetTexture("HeightMaps", HeightMapsArray);

            // Capture values for lambda closure
            int fi = frameIndex;
            var self = this;

            // Enqueue compute dispatch as custom action (runs first in Execute, before batch processing)
            CommandBuffer.Enqueue(RenderPass.Opaque, (list) => self.DispatchQuadtreeEval(list, fi));

            // Enqueue self-draw action: terrain draws itself via ExecuteIndirect
            CommandBuffer.Enqueue(RenderPass.Opaque, (list) => self.DrawTerrain(list, fi));

            // Enqueue single-pass terrain shadow draw
            CommandBuffer.Enqueue(RenderPass.Shadow, (list) => self.DrawTerrainShadow(list, fi));

            // Decoration pipeline — CPU-side setup
            if (Terrain.DrawDetail && _decoratorPalette.Count > 0 && DecoratorMaterial?.Effect != null)
            {
                if (Terrain.ConsumeFlags(TerrainDirtyFlags.DecoStructure | TerrainDirtyFlags.DecoParams) || _decoratorSlotsBuffer == null)
                    RebuildDecoSlots();

                if (_decoVariantCount > 0)
                {
                    EnsureDecoRenderBuffers();

                    CommandBuffer.Enqueue(RenderPass.Opaque, (list) => self.DispatchDecorator(list, fi, RenderPass.Opaque));
                    CommandBuffer.Enqueue(RenderPass.Shadow, (list) => self.DispatchDecorator(list, fi, RenderPass.Shadow));
                }
            }
        }

        // ───── Palette ────────────────────────────────────────────────────

        /// <summary>
        /// Collect the splat stamps that apply to this terrain and derive the layers it renders: each
        /// distinct TerrainLayer gets a channel, in the order the stamps first use them. Raises the
        /// texture-array rebuild when the set changes.
        /// </summary>
        private void RefreshLayerPalette()
        {
            _layerWarnings.Clear();
            _splatStamps = TerrainBaker.CollectStamps<SplatStamp>(this);

            var palette = new List<TerrainLayer>();
            var dropped = new HashSet<TerrainLayer>();
            int unassigned = 0;
            foreach (var stamp in _splatStamps)
            {
                var layer = stamp.Layer;
                if (layer == null) { unassigned++; continue; }
                if (palette.Contains(layer)) continue;

                if (palette.Count >= TerrainBaker.MaxLayers) dropped.Add(layer);
                else palette.Add(layer);
            }

            if (unassigned > 0)
                _layerWarnings.Add($"{unassigned} splat stamp(s) have no Layer and paint nothing.");
            if (dropped.Count > 0)
                _layerWarnings.Add($"Too many terrain layers: {palette.Count + dropped.Count} are referenced, " +
                                   $"{TerrainBaker.MaxLayers} can be rendered. Not rendered: " +
                                   string.Join(", ", dropped.Select(l => l.Name)));
            WarnAboutStrayGlobals(ComponentCache<SplatStamp>.All, _layerWarnings);
            ReportWarnings(_layerWarnings, ref _lastLayerWarning);

            if (!palette.SequenceEqual(_layerPalette))
            {
                _layerPalette = palette;
                Terrain!.MarkForUpdate(TerrainDirtyFlags.TextureArrays | TerrainDirtyFlags.LayerParams);
            }
        }

        /// <summary>
        /// Collect the deco stamps that apply to this terrain and derive the decorators it renders:
        /// each decorator some stamp adds gets a slot. Raises the decorator buffer rebuild when the
        /// set changes.
        /// </summary>
        private void RefreshDecoratorPalette()
        {
            _decoWarnings.Clear();
            _decoStamps = TerrainBaker.CollectStamps<DecoStamp>(this);

            var palette = new List<TerrainDecorator>();
            var dropped = new HashSet<TerrainDecorator>();
            foreach (var stamp in _decoStamps)
            {
                // Only an Add stamp puts a decorator on the terrain; Multiply merely scales what is there
                var decorator = stamp.Decorator;
                if (decorator == null || stamp.Op != DecoOp.Add || palette.Contains(decorator)) continue;

                if (palette.Count >= TerrainBaker.MaxDecorators) dropped.Add(decorator);
                else palette.Add(decorator);
            }

            if (dropped.Count > 0)
                _decoWarnings.Add($"Too many terrain decorators: {palette.Count + dropped.Count} are placed, " +
                                  $"{TerrainBaker.MaxDecorators} can be rendered. Not rendered: " +
                                  string.Join(", ", dropped.Select(d => d.Name)));
            foreach (var decorator in palette)
                if (decorator.Variants.Count > TerrainDecorator.MaxVariants)
                    _decoWarnings.Add($"Decorator '{decorator.Name}' has {decorator.Variants.Count} variants; " +
                                      $"only the first {TerrainDecorator.MaxVariants} are scattered.");
            WarnAboutStrayGlobals(ComponentCache<DecoStamp>.All, _decoWarnings);
            ReportWarnings(_decoWarnings, ref _lastDecoWarning);

            if (!palette.SequenceEqual(_decoratorPalette))
            {
                _decoratorPalette = palette;
                Terrain!.MarkForUpdate(TerrainDirtyFlags.DecoStructure);
            }
        }

        /// <summary>A global stamp only applies to the terrain it is parented under; one that sits elsewhere does nothing.</summary>
        private static void WarnAboutStrayGlobals<T>(IReadOnlyList<T> stamps, List<string> warnings) where T : TerrainStamp
        {
            for (int i = 0; i < stamps.Count; i++)
            {
                var stamp = stamps[i];
                if (stamp is { IsGlobal: true, Enabled: true } && stamp.Entity != null && stamp.FindOwningTerrain() == null)
                    warnings.Add($"Global {stamp.GetType().Name} on '{stamp.Entity.Name}' is not parented under a terrain and applies to nothing.");
            }
        }

        /// <summary>Log palette warnings when they change, not on every bake.</summary>
        private static void ReportWarnings(List<string> warnings, ref string last)
        {
            string joined = string.Join("\n", warnings);
            if (joined == last) return;
            last = joined;
            foreach (var warning in warnings)
                Debug.LogWarning("TerrainRenderer", warning);
        }

        private string _lastLayerWarning = "";
        private string _lastDecoWarning = "";

        /// <summary>
        /// A layer or decorator asset was edited: refresh what this terrain built from it.
        /// </summary>
        private void OnAssetDirty(Message msg)
        {
            if (Terrain == null) return;

            if (msg.Data is TerrainLayer layer && _layerPalette.Contains(layer))
                Terrain.MarkForUpdate(TerrainDirtyFlags.TextureArrays | TerrainDirtyFlags.LayerParams | TerrainDirtyFlags.AlbedoBake);
            else if (msg.Data is TerrainDecorator decorator && _decoratorPalette.Contains(decorator))
                Terrain.MarkForUpdate(TerrainDirtyFlags.DecoStructure | TerrainDirtyFlags.DecoParams);
        }

        /// <summary>
        /// Ensures the packed RGBA Texture2DArray exists at the right resolution/slice count.
        /// Only recreates when the configuration changes (layer count or resolution).
        /// </summary>
        private void EnsurePackedControlArray(int resolution, int sliceCount)
        {
            if (_packedControlArray != null && _packedArrayResolution == resolution && _packedSliceCount == sliceCount)
                return;

            var device = Engine.Device;

            // Retire the old array. Deferred: the splat bake and the terrain draws of the frames in flight still
            // reference it. Its slots go the same way; every reader takes them from the fields again each
            // frame (they used to be overwritten below without being returned).
            device.DeferDispose(_packedControlArray);
            device.DeferReleaseBindlessIndex(_packedControlSRV);
            if (_packedSliceUAVs != null)
                foreach (var idx in _packedSliceUAVs)
                    device.DeferReleaseBindlessIndex(idx);
            device.DeferReleaseBindlessIndex(_packedControlArrayUAV);

            // Create Texture2DArray: RGBA8, resolution × resolution, sliceCount array slices
            var desc = ResourceDescription.Texture2D(
                Format.R8G8B8A8_UNorm, (uint)resolution, (uint)resolution,
                (ushort)sliceCount, 1, 1, 0, ResourceFlags.AllowUnorderedAccess);

            _packedControlArray = device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Default), HeapFlags.None, desc, ResourceStates.Common, null);

            // Create SRV over full array
            _packedControlSRV = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(_packedControlArray,
                new ShaderResourceViewDescription
                {
                    Format = Format.R8G8B8A8_UNorm,
                    ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2DArray = new Texture2DArrayShaderResourceView
                    {
                        MipLevels = 1,
                        ArraySize = (uint)sliceCount,
                        FirstArraySlice = 0
                    }
                }, device.GetCpuHandle(_packedControlSRV));

            // Create per-slice UAVs for compute packing
            _packedSliceUAVs = new uint[sliceCount];
            for (int i = 0; i < sliceCount; i++)
            {
                _packedSliceUAVs[i] = device.AllocateBindlessIndex();
                device.NativeDevice.CreateUnorderedAccessView(_packedControlArray, null,
                    new UnorderedAccessViewDescription
                    {
                        Format = Format.R8G8B8A8_UNorm,
                        ViewDimension = UnorderedAccessViewDimension.Texture2DArray,
                        Texture2DArray = new Texture2DArrayUnorderedAccessView
                        {
                            MipSlice = 0,
                            FirstArraySlice = (uint)i,
                            ArraySize = 1
                        }
                    }, device.GetCpuHandle(_packedSliceUAVs[i]));
            }

            // Full-array UAV for stamp overlay (covers all slices)
            _packedControlArrayUAV = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(_packedControlArray, null,
                new UnorderedAccessViewDescription
                {
                    Format = Format.R8G8B8A8_UNorm,
                    ViewDimension = UnorderedAccessViewDimension.Texture2DArray,
                    Texture2DArray = new Texture2DArrayUnorderedAccessView
                    {
                        MipSlice = 0,
                        FirstArraySlice = 0,
                        ArraySize = (uint)sliceCount
                    }
                }, device.GetCpuHandle(_packedControlArrayUAV));

            // Create stable Texture wrapper (never replaced unless array is recreated)
            _packedControlTexture = Texture.WrapNative(_packedControlArray, _packedControlSRV);
            _packedArrayResolution = resolution;
            _packedSliceCount = sliceCount;
        }

        // ───── Compute Pipeline ───────────────────────────────────────────

        private void InitializeCompute()
        {
            var device = Engine.Device;

            _quadtreeCS = new ComputeShader("terrain_quadtree.hlsl");
            _kMarkSplits       = _quadtreeCS.FindKernel("CSMarkSplits");
            _kEmitLeaves       = _quadtreeCS.FindKernel("CSEmitLeaves");
            _kBuildMinMaxMip   = _quadtreeCS.FindKernel("CSBuildMinMaxMip");
            _kBuildDrawArgs    = _quadtreeCS.FindKernel("CSBuildDrawArgs");
            _kEmitLeavesShadow = _quadtreeCS.FindKernel("CSEmitLeavesShadow");

            for (int i = 0; i < FrameCount; i++)
            {
                CreateFrameBuffers(i);
            }

            // Create constant buffers: split into b0 (planes), b1 (Hi-Z), b2 (terrain params)
            int planesCBSize = ((Marshal.SizeOf<FrustumPlanesData>() + 255) & ~255);
            int hizCBSize = ((Marshal.SizeOf<HiZParamsData>() + 255) & ~255);
            int terrainCBSize = ((Marshal.SizeOf<TerrainParamsData>() + 255) & ~255);
            int cascadeDataSize = Marshal.SizeOf<GPUCuller.CascadeData>();
            int cascadeBufferSize = cascadeDataSize * DirectionalLight.MaxCascades;
            for (int i = 0; i < FrameCount; i++)
            {
                _frustumPlaneBuffers[i] = device.CreateUploadBuffer(planesCBSize);
                _hizParamBuffers[i] = device.CreateUploadBuffer(hizCBSize);
                _terrainParamBuffers[i] = device.CreateUploadBuffer(terrainCBSize);
                _shadowTerrainParamBuffers[i] = device.CreateUploadBuffer(terrainCBSize);
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
            _descriptorBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 20, srv: true, uav: true);
            _sphereBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 16, srv: true, uav: true);
            _subbatchIdBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 4, srv: true, uav: true);
            _terrainDataBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 32, srv: true, uav: true);

            // Raw R32 buffers
            _counterBuffers[i] = GraphicsBuffer.CreateRaw(1, uav: true, clearable: true);
            _splitFlagsBuffers[i] = GraphicsBuffer.CreateRaw(_totalNodes, uav: true, clearable: true);
            _indirectArgsBuffers[i] = GraphicsBuffer.CreateRaw(4, srv: true, uav: true);
            _shadowArgsBuffers[i] = GraphicsBuffer.CreateRaw(4, uav: true);

            // Shadow structured buffers
            _shadowDescriptorBuffers[i] = GraphicsBuffer.CreateStructured(shadowCapacity, 20, srv: true, uav: true);
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
            if (!_computeInitialized) return; // Destroyed between enqueue and execute

            var device = Engine.Device;
            var cs = _quadtreeCS!;

            // Transition buffers to UAV
            _descriptorBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);
            _sphereBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);
            _subbatchIdBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);
            _terrainDataBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);
            _indirectArgsBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);

            // Upload frustum + Hi-Z constants (b0, b1)
            UploadFrustumConstants(frameIndex);

            // Upload terrain params (b2)
            var terrainSize = Terrain!.TerrainSize;
            var maxHeight = Terrain.MaxHeight;
            Vector3 cameraPos = Camera.Main!.Position - Transform.Position;
            Vector3 rootCenter = new Vector3(terrainSize.X * 0.5f, 0, terrainSize.Y * 0.5f);
            Vector3 rootExtents = new Vector3(terrainSize.X * 0.5f, maxHeight, terrainSize.Y * 0.5f);
            var camera = Camera.Main!;
            float vFovRad = camera.FieldOfView * (MathF.PI / 180f);

            var terrainParams = new TerrainParamsData
            {
                CameraPos = cameraPos,
                MaxHeight = maxHeight,
                RootCenter = rootCenter,
                MaxDepth = (uint)MaxDepth,
                RootExtents = rootExtents,
                TotalNodes = (uint)_totalNodes,
                TerrainSize = terrainSize,
                PixelErrorThreshold = Engine.Settings.PixelErrorThreshold,
                ScreenHeight = camera.Target?.Height ?? 1080f,
                TanHalfFov = MathF.Tan(vFovRad * 0.5f),
                TransformSlot = (uint)Transform.TransformSlot,
                MaterialId = (uint)(Material?.MaterialID ?? 0),
                MeshPartId = (uint)_meshPartId,
                MaxPatches = (uint)Math.Max(MaxPatches, MinPatches),
                HeightTexIdx = Terrain?.Heightmap?.BindlessIndex ?? 0u,
            };
            UploadBuffer(_terrainParamBuffers[frameIndex], terrainParams);

            // Bind root sig + descriptor heaps + all cbuffers once (before any compute operations)
            commandList.SetComputeRootSignature(device.GlobalRootSignature);
            commandList.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            commandList.SetComputeRootConstantBufferView(1, _frustumPlaneBuffers[frameIndex].GPUVirtualAddress);
            commandList.SetComputeRootConstantBufferView(2, _hizParamBuffers[frameIndex].GPUVirtualAddress);
            commandList.SetComputeRootConstantBufferView(3, _terrainParamBuffers[frameIndex].GPUVirtualAddress);

            // Clear counter and splitFlags
            _counterBuffers[frameIndex].ClearUAV(commandList, new Vortice.Mathematics.Int4(0, 0, 0, 0));
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
            _splitFlagsBuffers[frameIndex].ClearUAV(commandList, new Vortice.Mathematics.Int4(0, 0, 0, 0));
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Push constants — bindless indices only (slots 0-14)
            cs.SetBuffer("OutputDescriptorsUAV", _descriptorBuffers[frameIndex]);
            cs.SetBuffer("OutputSpheresUAV", _sphereBuffers[frameIndex]);
            cs.SetBuffer("OutputSubbatchIdsUAV", _subbatchIdBuffers[frameIndex]);
            cs.SetBuffer("OutputTerrainDataUAV", _terrainDataBuffers[frameIndex]);
            cs.SetBuffer("CounterUAV", _counterBuffers[frameIndex]);
            cs.SetBuffer("SplitFlagsUAV", _splitFlagsBuffers[frameIndex]);
            cs.SetPushConstant("HeightRangeSRV", _heightRangePyramidSRV);

            // Per-kernel overrides for CSBuildDrawArgs
            cs.SetPushConstant(_kBuildDrawArgs, "VertexCount", (uint)_patchMesh.IndexCount);
            cs.SetBuffer(_kBuildDrawArgs, "IndirectArgsUAV", _indirectArgsBuffers[frameIndex]);

            // ── One-time: Build height range mip pyramid ──
            if (!_heightRangePyramidBuilt)
            {
                StreamingManager.Instance?.Flush();
                Engine.Device.WaitForCopyQueue();

                var hmIdx = Terrain?.Heightmap?.BindlessIndex ?? 0u;
                Debug.Log($"[TerrainRenderer] Pyramid build in QuadtreeEval: HeightTexIdx={terrainParams.HeightTexIdx} HM.Idx={hmIdx} BakedHM={Terrain?.BakedHeightmap != null}");

                BuildHeightRangePyramid(commandList);
                _heightRangePyramidBuilt = true;

                // Re-bind root sig + cbuffers (mip builder may have changed PSO state)
                commandList.SetComputeRootSignature(device.GlobalRootSignature);
                commandList.SetDescriptorHeaps(1, new[] { device.SrvHeap });
                commandList.SetComputeRootConstantBufferView(1, _frustumPlaneBuffers[frameIndex].GPUVirtualAddress);
                commandList.SetComputeRootConstantBufferView(2, _hizParamBuffers[frameIndex].GPUVirtualAddress);
                commandList.SetComputeRootConstantBufferView(3, _terrainParamBuffers[frameIndex].GPUVirtualAddress);
            }

            uint threadGroups = (uint)((_totalNodes + 255) / 256);

            // ── Pass 1: Mark splits + enforce restricted quadtree ──
            cs.Dispatch(_kMarkSplits, commandList, threadGroups);

            // UAV barrier
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // ── Pass 2: Emit leaves with inline frustum + Hi-Z culling ──
            cs.Dispatch(_kEmitLeaves, commandList, threadGroups);

            // UAV barrier
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // ── Pass 3: Build DrawInstanced indirect args from counter ──
            cs.Dispatch(_kBuildDrawArgs, commandList, 1);

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
            if (!_computeInitialized) return; // Destroyed between enqueue and execute

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
            // Slot 21: Deco control map SRV (for debug mode 5)
            commandList.SetGraphicsRoot32BitConstant(0, _decoControlSRV, 21);
            // Slot 26: HeightMapsArray SRV (SSDM per-layer displacement)
            commandList.SetGraphicsRoot32BitConstant(0, (uint)(HeightMapsArray?.BindlessIndex ?? 0), 26);

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
            int kShadow = _kEmitLeavesShadow;
            int kArgs = _kBuildDrawArgs;

            // Upload shadow-specific TerrainParams (b2, root slot 3) — only MaxPatches differs
            var terrainSz = Terrain.TerrainSize;
            int shadowMaxPatches = MaxPatches * cascadeCount;
            var camPos = Camera.Main!.Position - Transform.Position;
            float vFovRad = Camera.Main!.FieldOfView * (MathF.PI / 180f);
            var shadowTerrainParams = new TerrainParamsData
            {
                CameraPos = camPos,
                MaxHeight = Terrain.MaxHeight,
                RootCenter = new Vector3(terrainSz.X * 0.5f, 0, terrainSz.Y * 0.5f),
                MaxDepth = (uint)MaxDepth,
                RootExtents = new Vector3(terrainSz.X * 0.5f, Terrain.MaxHeight, terrainSz.Y * 0.5f),
                TotalNodes = (uint)_totalNodes,
                TerrainSize = terrainSz,
                PixelErrorThreshold = Engine.Settings.PixelErrorThreshold,
                ScreenHeight = Camera.Main.Target?.Height ?? 1080f,
                TanHalfFov = MathF.Tan(vFovRad * 0.5f),
                TransformSlot = (uint)Transform.TransformSlot,
                MaterialId = (uint)(Material?.MaterialID ?? 0),
                MeshPartId = (uint)_meshPartId,
                MaxPatches = (uint)shadowMaxPatches,
                HeightTexIdx = Terrain.Heightmap?.BindlessIndex ?? 0u,
            };
            UploadBuffer(_shadowTerrainParamBuffers[frameIndex], shadowTerrainParams);

            // Clear shadow counter
            commandList.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            commandList.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            _shadowCounterBuffers[frameIndex].ClearUAV(commandList, new Vortice.Mathematics.Int4(0, 0, 0, 0));

            // Bind cbuffers (b0, b1 already uploaded by main pass; b2 updated above)
            commandList.SetComputeRootConstantBufferView(1, _frustumPlaneBuffers[frameIndex].GPUVirtualAddress);
            commandList.SetComputeRootConstantBufferView(2, _hizParamBuffers[frameIndex].GPUVirtualAddress);
            commandList.SetComputeRootConstantBufferView(3, _shadowTerrainParamBuffers[frameIndex].GPUVirtualAddress);

            // Push constants — bindless indices only
            cs.SetBuffer(kShadow, "OutputDescriptorsUAV", _shadowDescriptorBuffers[frameIndex]);
            cs.SetBuffer(kShadow, "OutputSpheresUAV", _shadowSphereBuffers[frameIndex]);
            cs.SetBuffer(kShadow, "OutputSubbatchIdsUAV", _shadowSubbatchIdBuffers[frameIndex]);
            cs.SetBuffer(kShadow, "OutputTerrainDataUAV", _shadowTerrainDataBuffers[frameIndex]);
            cs.SetBuffer(kShadow, "CounterUAV", _shadowCounterBuffers[frameIndex]);
            cs.SetBuffer(kShadow, "CascadeIdxUAV", _shadowCascadeIdxBuffers[frameIndex]);
            cs.SetBuffer(kShadow, "SplitFlagsUAV", _splitFlagsBuffers[frameIndex]);
            cs.SetPushConstant(kShadow, "HeightRangeSRV", _heightRangePyramidSRV);
            cs.SetPushConstant(kShadow, "CascadeCount", (uint)cascadeCount);
            cs.SetPushConstant(kShadow, "CascadeBufferSRV", _shadowCascadeBufferSrvs[frameIndex]);

            // Dispatch CSEmitLeavesShadow
            uint threadGroups = (uint)((_totalNodes + 255) / 256);
            cs.Dispatch(kShadow, commandList, threadGroups);

            // UAV barrier
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Build draw args from shadow counter (CSBuildDrawArgs with shadow-specific overrides)
            cs.SetPushConstant(kArgs, "VertexCount", (uint)_patchMesh.IndexCount);
            cs.SetBuffer(kArgs, "IndirectArgsUAV", _shadowArgsBuffers[frameIndex]);
            cs.SetBuffer(kArgs, "CounterUAV", _shadowCounterBuffers[frameIndex]);
            cs.Dispatch(kArgs, commandList, 1);

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
            commandList.SetGraphicsRoot32BitConstant(0, DirectionalLight.CurrentCascadeSrvIndex, 23); // CascadeBufferSRVIdx
            commandList.SetGraphicsRoot32BitConstant(0, (uint)cascadeCount, 24);                       // ShadowCascadeCount
            commandList.SetGraphicsRoot32BitConstant(0, _shadowCascadeIdxBuffers[frameIndex].SrvIndex, 25);        // CascadeIdxBufIdx

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

        private System.Numerics.Matrix4x4 FrozenViewProjection; // VP matrix when frustum frozen

        /// <summary>
        /// Upload frustum planes + Hi-Z occlusion data to the per-frame constant buffer.
        /// This is bound to compute root slot 1 (register b0) for inline culling.
        /// </summary>
        private void UploadFrustumConstants(int frameIndex)
        {
            var vpMatrix = Engine.Settings.FreezeFrustum
                ? FrozenViewProjection
                : Camera.Main!.ViewProjection;
            var frustum = new Frustum(vpMatrix);
            var planes = frustum.GetPlanesAsVector4();

            // Transform frustum planes from world to terrain local space
            var terrainPos = Transform.Position;
            for (int i = 0; i < planes.Length; i++)
            {
                var n = new Vector3(planes[i].X, planes[i].Y, planes[i].Z);
                planes[i].W += Vector3.Dot(n, terrainPos);
            }

            // Upload FrustumPlanes (b0)
            var frustumData = new FrustumPlanesData
            {
                Plane0 = planes[0], Plane1 = planes[1], Plane2 = planes[2],
                Plane3 = planes[3], Plane4 = planes[4], Plane5 = planes[5],
            };
            if (_frustumPlaneBuffers[frameIndex] == null)
            {
                Debug.LogError("[TerrainRenderer]", $"UploadFrustumConstants: null buffer! frameIndex={frameIndex} _computeInitialized={_computeInitialized} Entity={Entity?.Name} UID={UID}");
                return;
            }
            UploadBuffer(_frustumPlaneBuffers[frameIndex], frustumData);

            // Upload HiZParams (b1)
            var hizData = new HiZParamsData();
            var pyramid = DeferredRenderer.Current?.HiZPyramid;
            if (!_firstDispatch && pyramid != null && pyramid.FullSRV != 0 && pyramid.Ready && !Engine.Settings.DisableHiZ)
            {
                var occVP = Engine.Settings.FreezeFrustum
                    ? FrozenViewProjection
                    : _previousFrameViewProjection;
                var localToWorld = Matrix4x4.CreateTranslation(terrainPos);
                hizData.OcclusionProjection = localToWorld * occVP;
                hizData.HiZSrvIdx = pyramid.FullSRV;
                hizData.HiZWidth = pyramid.Width;
                hizData.HiZHeight = pyramid.Height;
                hizData.HiZMipCount = (uint)pyramid.MipCount;
                hizData.NearPlane = Camera.Main!.NearPlane;
            }
            UploadBuffer(_hizParamBuffers[frameIndex], hizData);
        }

        private static unsafe void UploadBuffer<T>(ID3D12Resource buffer, T data) where T : unmanaged
        {
            void* pData = null;
            var hr = buffer.Map(0, null, &pData);
            if (hr.Failure || pData == null)
            {
                // Map fails once the device is removed; writing through the null pointer used to kill the editor
                // with an NRE here, hiding the real (GPU-side) cause. Report it once instead.
                Engine.Device.LogDeviceRemoved($"TerrainRenderer.UploadBuffer (Map 0x{hr.Code:X8})");
                return;
            }
            *(T*)pData = data;
            buffer.Unmap(0);
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
            int kMip = _kBuildMinMaxMip;

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

                // Per-kernel push constants for this mip level
                cs.SetParam(kMip, "BuildMip", (uint)mip);
                cs.SetParam(kMip, "MipInputSRV", mip > 0 ? _heightRangeMipSRVs[mip - 1] : 0u);
                cs.SetParam(kMip, "MipOutputUAV", _heightRangeMipUAVs[mip]);

                // Dispatch 8x8 threadgroups
                uint groupsX = (uint)((w + 7) / 8);
                uint groupsY = (uint)((h + 7) / 8);
                cs.Dispatch(kMip, commandList, groupsX, groupsY);

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

        /// <summary>
        /// Lightweight: refresh the per-channel tiling from the palette's layer assets.
        /// No GPU allocations. Called when LayerParams flag is consumed.
        /// </summary>
        private void UpdateLayerParams()
        {
            var terrainSize = Terrain?.TerrainSize ?? Vector2.One;
            Array.Clear(_layerTiling);

            for (int i = 0; i < _layerPalette.Count && i < _layerTiling.Length; i++)
            {
                var layer = _layerPalette[i];
                float hasHeight = (layer.Height?.Native != null) ? 1.0f : 0.0f;
                if (layer.Tiling.X != 0 && layer.Tiling.Y != 0)
                    _layerTiling[i] = new Vector4(terrainSize.X / layer.Tiling.X, terrainSize.Y / layer.Tiling.Y, hasHeight, layer.HeightScale);
                else
                    _layerTiling[i] = new Vector4(1, 1, hasHeight, layer.HeightScale);
            }
        }

        /// <summary>
        /// Heavy: rebuild DiffuseMapsArray, NormalMapsArray, HeightMapsArray from the palette.
        /// Only called when TextureArrays flag is consumed (the set of layers or a layer's textures changed).
        /// Also refreshes tiling since channels may have moved.
        /// </summary>
        private void RebuildTextureArrays()
        {
            var diffuseList = new List<Texture>();
            var normalList = new List<Texture>();
            var heightList = new List<Texture>();

            foreach (var layer in _layerPalette)
            {
                if (layer.Diffuse != null && layer.Diffuse.Native != null) diffuseList.Add(layer.Diffuse);
                if (layer.Normals != null && layer.Normals.Native != null) normalList.Add(layer.Normals);
                if (layer.Height != null && layer.Height.Native != null)
                    heightList.Add(layer.Height);
            }
            Debug.Log($"[Terrain] RebuildTextureArrays: {diffuseList.Count} diffuse, {normalList.Count} normals, {heightList.Count} heights from {_layerPalette.Count} layers");

            // The arrays are indexed by channel, so a layer without a texture would shift every layer after it
            if (diffuseList.Count != _layerPalette.Count)
                Debug.LogWarning("TerrainRenderer", "A terrain layer has no Diffuse texture: the layers after it render with the wrong textures.");
            if (normalList.Count != 0 && normalList.Count != _layerPalette.Count)
                Debug.LogWarning("TerrainRenderer", "Some terrain layers have no Normals texture: the layers after them render with the wrong normals.");
            if (heightList.Count != 0 && heightList.Count != _layerPalette.Count)
                Debug.LogWarning("TerrainRenderer", "Some terrain layers have no Height texture: the layers after them blend with the wrong heights.");

            // Also refresh tiling since channels may have moved
            UpdateLayerParams();

            var device = Engine.Device;

            // Diffuse Fallback
            if (diffuseList.Count > 0)
                DiffuseMapsArray = Texture.CreateTexture2DArray(device, diffuseList);
            else
                DiffuseMapsArray = InternalAssets.BlackArray;

            // Normal Fallback
            if (normalList.Count > 0)
                NormalMapsArray = Texture.CreateTexture2DArray(device, normalList, stripSrgb: true);
            else
                NormalMapsArray = InternalAssets.FlatNormalArray;

            // Height Fallback (SSDM per-layer displacement heights)
            if (heightList.Count > 0)
                HeightMapsArray = Texture.CreateTexture2DArray(device, heightList, stripSrgb: true);
            else
                HeightMapsArray = null; // no height data = no displacement

            // The layer weights are baked separately (SplatPack); until the first bake nothing shows
            ControlMapsArray = _packedControlTexture ?? InternalAssets.BlackArray;
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

        /// <summary>
        /// One decorator variant on the GPU ("DecoratorSlot" in the shaders). Instances carry the index
        /// of this record. Must match DecoratorSlot in grass_compute.hlsl / grass.fx / grass_mesh.fx.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct DecoratorSlotGPU
        {
            public float Density;     // instances/m² at full coverage (the spawn kernel applies DecorationDensity)
            public float MinH, MaxH;
            public float MinW, MaxW;
            public uint LODCount;
            public uint LODTableOffset;
            // Precomputed root rotation matrix (CPU computes cos/sin, GPU reads directly)
            public float Rot00, Rot01, Rot02;
            public float Rot10, Rot11, Rot12;
            public float Rot20, Rot21, Rot22;
            public float SlopeBias;  // 0=upright, 1=fully slope-aligned
            public uint Seed;             // decorrelates this variant's scatter and noise from the others
            public uint _unused0;
            public uint Mode;             // 0=Mesh, 1=Billboard, 2=Cross
            public uint TextureIdx;       // bindless index (billboard/cross)
            // Color tint (Unity detail prototype healthy/dry colors)
            public Vector3 HealthyColor;
            public Vector3 DryColor;
            public float NoiseSpread;
            public uint _unused1;
            public float _unused2;
            public float ClusterScale;    // world size of density clumps (m), 0 = off
            public float ClusterAmount;   // 0 = uniform, 1 = full clumps with bare gaps
            public uint AlphaClip;        // mesh mode: 1 = alpha-test albedo, 0 = opaque material
            public float HeightNoiseScale;  // world size of short vs. lush patches (m), 0 = off
            public float HeightNoiseAmount; // 0 = none, 1 = 0.4x .. 1.4x height
        }

        /// <summary>
        /// One TerrainDecorator on the GPU: the run of variants a control-texture slot stands for.
        /// Must match DecoratorGroup in grass_compute.hlsl.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct DecoratorGroupGPU
        {
            public uint VariantOffset;
            public uint VariantCount;
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
        /// Builds the decoration GPU buffers from the decorator palette: one group per decorator, one
        /// slot per renderable variant, and the LOD table the variants point into.
        /// Runs on structural changes (decorators or variants added/removed, mesh swapped) and on
        /// parameter edits alike; when the element counts are unchanged the existing buffers are
        /// refilled in place, so dragging a slider allocates nothing.
        /// </summary>
        public void RebuildDecoSlots()
        {
            if (Terrain == null) return;

            // Flat list — all variants go into a single header.
            var headers = new List<ChannelHeader>();
            var slots = new List<DecoratorSlotGPU>();
            var groups = new List<DecoratorGroupGPU>();
            var lodTable = new List<LODTableEntry>();

            foreach (var decorator in _decoratorPalette)
            {
                uint variantOffset = (uint)slots.Count;
                int variantLimit = Math.Min(decorator.Variants.Count, TerrainDecorator.MaxVariants);

                for (int v = 0; v < variantLimit; v++)
                {
                    var variant = decorator.Variants[v];
                    if (variant == null || !variant.IsRenderable) continue;

                    var mode = variant.Mode;
                    uint lodTableOffset = (uint)lodTable.Count;
                    uint lodCount = 0;
                    uint textureIdx = 0;

                    if (mode == DecoratorMode.Mesh)
                    {
                        var mesh = variant.Mesh!;
                        uint meshMatId = (uint)(variant.Material?.MaterialID ?? 0);

                        // Determine LOD0 part indices
                        int[] lod0Indices;
                        if (mesh.LODs.Count > 0 && mesh.LODs[0].MeshPartIndices != null)
                            lod0Indices = mesh.LODs[0].MeshPartIndices;
                        else
                            lod0Indices = Enumerable.Range(0, mesh.MeshParts.Count).ToArray();

                        // LOD0 parts
                        foreach (var partIdx in lod0Indices)
                        {
                            if (partIdx >= mesh.MeshParts.Count) continue;
                            int partId = MeshRegistry.Register(mesh, partIdx);
                            lodTable.Add(new LODTableEntry { MeshPartId = (uint)partId, MaxDistance = 100f, MaterialId = meshMatId });
                            lodCount++;
                        }

                        // Additional LOD levels
                        for (int lod = 1; lod < mesh.LODs.Count; lod++)
                        {
                            var lodLevel = mesh.LODs[lod];
                            if (lodLevel.MeshPartIndices == null) continue;
                            foreach (var partIdx in lodLevel.MeshPartIndices)
                            {
                                if (partIdx >= mesh.MeshParts.Count) continue;
                                int partId = MeshRegistry.Register(mesh, partIdx);
                                float maxDist = 50f * (lod + 1);
                                lodTable.Add(new LODTableEntry { MeshPartId = (uint)partId, MaxDistance = maxDist, MaterialId = meshMatId });
                                lodCount++;
                            }
                        }
                    }
                    else
                    {
                        // Billboard / Cross: one dummy LOD entry
                        textureIdx = (uint)variant.Texture!.BindlessIndex;
                        lodTable.Add(new LODTableEntry { MeshPartId = 0, MaxDistance = 200f, MaterialId = 0 });
                        lodCount = 1;
                    }

                    float degToRad = MathF.PI / 180f;
                    float rx = variant.RootRotation.X * degToRad;
                    float ry = variant.RootRotation.Y * degToRad;
                    float rz = variant.RootRotation.Z * degToRad;
                    float cx = MathF.Cos(rx), sx = MathF.Sin(rx);
                    float cy = MathF.Cos(ry), sy = MathF.Sin(ry);
                    float cz = MathF.Cos(rz), sz = MathF.Sin(rz);

                    slots.Add(new DecoratorSlotGPU
                    {
                        Density = variant.Density,
                        MinH = variant.HeightRange.X,
                        MaxH = variant.HeightRange.Y,
                        MinW = variant.WidthRange.X,
                        MaxW = variant.WidthRange.Y,
                        LODCount = lodCount,
                        LODTableOffset = lodTableOffset,
                        Rot00 = cy*cz,              Rot01 = cy*sz,              Rot02 = -sy,
                        Rot10 = sx*sy*cz - cx*sz,   Rot11 = sx*sy*sz + cx*cz,   Rot12 = sx*cy,
                        Rot20 = cx*sy*cz + sx*sz,   Rot21 = cx*sy*sz - sx*cz,   Rot22 = cx*cy,
                        SlopeBias = variant.SlopeBias,
                        Seed = (uint)Math.Clamp(variant.Seed, 0, 255),
                        Mode = (uint)mode,
                        TextureIdx = textureIdx,
                        HealthyColor = new Vector3(variant.HealthyColor.X, variant.HealthyColor.Y, variant.HealthyColor.Z),
                        DryColor = new Vector3(variant.DryColor.X, variant.DryColor.Y, variant.DryColor.Z),
                        NoiseSpread = variant.NoiseSpread,
                        ClusterScale = variant.ClusterScale,
                        ClusterAmount = variant.ClusterAmount,
                        AlphaClip = variant.Material?.Effect?.Name == "gbuffer" ? 0u : 1u,
                        HeightNoiseScale = variant.HeightNoiseScale,
                        HeightNoiseAmount = variant.HeightNoiseAmount,
                    });
                }

                groups.Add(new DecoratorGroupGPU
                {
                    VariantOffset = variantOffset,
                    VariantCount = (uint)slots.Count - variantOffset,
                });
            }

            // Single header covering all slots
            headers.Add(new ChannelHeader
            {
                StartIndex = 0,
                Count = (uint)slots.Count
            });

            bool sameShape = _decoratorSlotsBuffer != null && _decoratorLODTableBuffer != null && _decoratorGroupsBuffer != null
                             && _decoratorHeadersBuffer != null
                             && slots.Count == _decoSlotCapacity && lodTable.Count == _decoLodCapacity
                             && groups.Count == _decoGroupCapacity;

            if (sameShape)
            {
                // Same element counts: refill in place (no GPU allocations)
                Refill(_decoratorHeadersBuffer!, headers);
                Refill(_decoratorSlotsBuffer!, slots);
                Refill(_decoratorGroupsBuffer!, groups);
                Refill(_decoratorLODTableBuffer!, lodTable);
            }
            else
            {
                Debug.Log($"[TerrainRenderer] Building decorator buffers: {groups.Count} decorators, {slots.Count} variants, {lodTable.Count} LOD entries");

                // Deferred: the decorator dispatches and draws of the frames in flight still read the old buffers
                var device = Engine.Device;
                device.DeferDispose(_decoratorHeadersBuffer);
                device.DeferDispose(_decoratorSlotsBuffer);
                device.DeferDispose(_decoratorGroupsBuffer);
                device.DeferDispose(_decoratorLODTableBuffer);

                _decoratorHeadersBuffer = CreateAndUpload(headers);
                _decoratorSlotsBuffer = CreateAndUpload(slots);
                _decoratorGroupsBuffer = CreateAndUpload(groups);
                _decoratorLODTableBuffer = CreateAndUpload(lodTable);

                _decoSlotCapacity = slots.Count;
                _decoLodCapacity = lodTable.Count;
                _decoGroupCapacity = groups.Count;
            }

            _decoVariantCount = slots.Count;
            _decoratorBuffersBuilt = true;
        }

        private static void Refill<T>(GraphicsBuffer buffer, List<T> data) where T : unmanaged
        {
            if (data.Count == 0) return;
            buffer.Upload<T>(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(data));
        }

        /// <summary>
        /// Runs the decoration coverage bake when one is pending: composites the deco stamps (laid out
        /// on the main thread, see Draw) into the top-8 RGBA16_UINT control texture the spawn kernel reads.
        /// </summary>
        private void DispatchDecoControlPrepass(ID3D12GraphicsCommandList cmd)
        {
            var plan = System.Threading.Interlocked.Exchange(ref _pendingDecoPlan, null);
            if (plan == null || Terrain == null || _baker == null) return;

            var device = Engine.Device;
            int resolution = Terrain.EffectiveDecorationMapResolution;

            // Create or recreate control texture only when resolution changes
            bool freshTexture = false;
            if (_decoControlTex == null || (int)_decoControlTex.Description.Width != resolution)
            {
                freshTexture = true;
                device.DeferDispose(_decoControlTex); // frames in flight may still read it
                _decoControlTex = device.CreateTexture2D(
                    Format.R16G16B16A16_UInt, resolution, resolution, 2, 1,
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
            }

            // Transition control texture back to UAV for writing
            // Fresh textures start in Common (implicit promotion to UAV).
            // Re-dispatches need explicit SRV → UAV transition.
            if (!freshTexture)
            {
                cmd.ResourceBarrier(new ResourceBarrier(
                    new ResourceTransitionBarrier(_decoControlTex,
                        ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess)));
            }

            // Heightmap for the stamps' height/slope filters
            var heightmap = Terrain.Heightmap;
            uint heightSrv = heightmap != null ? (uint)heightmap.BindlessIndex : 0;

            // Packed layer weights for the stamps' layer filters — the same array the surface shader
            // samples, already rebaked this frame if the splat stamps changed
            uint controlMapsSrv = plan.LayerCount > 0 && _packedControlTexture != null
                ? (uint)_packedControlTexture.BindlessIndex : 0;

            _baker.BakeDecoControl(plan, cmd, Terrain, heightSrv, controlMapsSrv, _decoControlUAV, resolution);

            cmd.ResourceBarrierUnorderedAccessView(_decoControlTex);

            // Transition from UAV → SRV so the spawn shader and terrain debug overlay can read correctly
            cmd.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_decoControlTex,
                    ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)));
        }

        /// <summary>
        /// Bakes terrain splatmap layers into a single 256×256 albedo texture
        /// for ground color blending in vegetation shaders.
        /// </summary>
        private void BakeTerrainAlbedo(ID3D12GraphicsCommandList cmd)
        {
            return;

            if (!Terrain.ConsumeFlags(TerrainDirtyFlags.AlbedoBake)) return;
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
            cmd.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            _albedoBakeCS.SetTexture("ControlMaps", ControlMapsArray);    // Texture → BindlessIndex
            _albedoBakeCS.SetTexture("DiffuseMaps", DiffuseMapsArray);    // Texture → BindlessIndex
            _albedoBakeCS.SetPushConstant("OutputUAV", _bakedAlbedoUAV); // uint (raw UAV index)
            _albedoBakeCS.SetBuffer("TilingBuf", _tilingBuffer);         // GraphicsBuffer → auto SRV

            uint groups = (uint)((BakedAlbedoSize + 7) / 8);
            _albedoBakeCS.Dispatch(0, cmd, groups, groups);

            cmd.ResourceBarrierUnorderedAccessView(_bakedAlbedoTex);

            // AlbedoBake flag already consumed in the guard above
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

        /// <summary>
        /// One-time init: creates render-side GPU buffers (instance buffer, counters,
        /// dispatch args) and initializes the compute shader + kernel indices.
        /// Sized from HeightmapResolution. Idempotent — only runs once.
        /// </summary>
        private void EnsureDecoRenderBuffers()
        {
            if (_decoBuffersCreated) return;

            int resolution = Terrain?.EffectiveDecorationMapResolution ?? 256;
            int maxTiles = resolution * resolution;

            // Worst-case instances: ~64 per tile max (64 threads/group)
            // But realistically with density, far fewer. Use maxTiles * 8 as reasonable upper bound.
            _maxDecoInstances = Math.Min(maxTiles * 8, 1024 * 1024); // cap at 1M

            _decoInstanceBuffer = GraphicsBuffer.CreateStructured(_maxDecoInstances, 64, srv: true, uav: true); // 64 bytes per DecoInstance
            _instanceCounterBuffer = GraphicsBuffer.CreateRaw(2, uav: true, clearable: true); // 2 uints: billboard + mesh counts
            _decoDispatchArgsBuffer = GraphicsBuffer.CreateRaw(4, uav: true); // 4 uints: 3 DispatchMesh args + 1 mesh count

            // Mesh-mode decorator buffers
            _meshDecoInstanceBuffer = GraphicsBuffer.CreateStructured(_maxDecoInstances, 64, srv: true, uav: true);
            _sortedMeshInstanceBuffer = GraphicsBuffer.CreateStructured(_maxDecoInstances, 64, srv: true, uav: true);
            _meshDrawArgsBuffer = GraphicsBuffer.CreateRaw(32 * 18, uav: true); // 32 mesh types × 72 bytes = 32 × 18 uints
            _meshDrawCountBuffer = GraphicsBuffer.CreateRaw(1, uav: true, clearable: true);

            // Initialize compute shader and find kernels
            _grassCS ??= new ComputeShader("grass_compute.hlsl");
            _kBakeNormals       = _grassCS.FindKernel("CS_BakeTerrainNormals");
            _kSpawnInstances    = _grassCS.FindKernel("CS_SpawnInstances");
            _kBuildDecoDrawArgs = _grassCS.FindKernel("CS_BuildDrawArgs");
            _kBinMeshInstances  = _grassCS.FindKernel("CS_BinMeshInstances");

            // Load mesh decorator material
            _meshDecoratorMaterial ??= new Material(new Effect("grass_mesh"));

            Debug.Log($"[TerrainRenderer] Deco buffers created: maxTiles={maxTiles} maxInstances={_maxDecoInstances} resolution={resolution}");

            // Readback buffer for instance counter + mesh draw count (12 bytes for 3 uints)
            _instanceCounterReadback = Engine.Device.NativeDevice.CreateCommittedResource(
                new Vortice.Direct3D12.HeapProperties(Vortice.Direct3D12.HeapType.Readback),
                Vortice.Direct3D12.HeapFlags.None,
                Vortice.Direct3D12.ResourceDescription.Buffer(12),
                Vortice.Direct3D12.ResourceStates.CopyDest,
                null);
            unsafe
            {
                void* pData;
                _instanceCounterReadback.Map(0, null, &pData);
                _instanceCounterReadbackPtr = (IntPtr)pData;
            }

            _decoBuffersCreated = true;
        }

        /// <summary>
        /// One-time bake: heightmap → R16G16_SNORM terrain normal map.
        /// Eliminates 5-tap normal computation per instance in the spawn kernel.
        /// </summary>
        private void BakeTerrainNormals(ID3D12GraphicsCommandList cmd)
        {
            if (!_bakedNormalsDirty) return;
            if (Terrain?.Heightmap == null) return;

            var device = Engine.Device;
            var hmDesc = Terrain.Heightmap.Native.Description;
            int hmW = (int)hmDesc.Width;
            int hmH = (int)hmDesc.Height;

            // Create normal map texture (R16G16_SNORM: stores XZ, Y reconstructed)
            if (_bakedNormalTex == null)
            {
                _bakedNormalTex = device.CreateTexture2D(
                    Format.R16G16_SNorm, hmW, hmH, 1, 1,
                    ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

                _bakedNormalUAV = device.AllocateBindlessIndex();
                var uavDesc = new UnorderedAccessViewDescription
                {
                    Format = Format.R16G16_SNorm,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
                };
                device.NativeDevice.CreateUnorderedAccessView(_bakedNormalTex, null, uavDesc, device.GetCpuHandle(_bakedNormalUAV));

                _bakedNormalSRV = device.AllocateBindlessIndex();
                var srvDesc = new ShaderResourceViewDescription
                {
                    Format = Format.R16G16_SNorm,
                    ViewDimension = ShaderResourceViewDimension.Texture2D,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2D = new Texture2DShaderResourceView
                    {
                        MostDetailedMip = 0,
                        MipLevels = 1
                    }
                };
                device.NativeDevice.CreateShaderResourceView(_bakedNormalTex, srvDesc, device.GetCpuHandle(_bakedNormalSRV));
            }

            // Dispatch CS_BakeTerrainNormals
            // Ensure compute shader is initialized (independent of deco buffers)
            _grassCS ??= new ComputeShader("grass_compute.hlsl");
            if (_kBakeNormals == 0) _kBakeNormals = _grassCS.FindKernel("CS_BakeTerrainNormals");
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            _grassCS!.SetTexture(_kBakeNormals, "Heightmap", Terrain.Heightmap);
            _grassCS.SetPushConstant(_kBakeNormals, "BakedNormalUAV", _bakedNormalUAV);
            _grassCS.SetParam("TerrainSize", new Vector2(Terrain.TerrainSize.X, Terrain.TerrainSize.Y));
            _grassCS.SetParam("MaxHeight", Terrain.MaxHeight);
            _grassCS.SetParam("HeightmapSize", new Vortice.Mathematics.UInt2((uint)hmW, (uint)hmH));

            uint groupsX = (uint)((hmW + 7) / 8);
            uint groupsY = (uint)((hmH + 7) / 8);
            _grassCS.Dispatch(_kBakeNormals, cmd, groupsX, groupsY);

            cmd.ResourceBarrierUnorderedAccessView(_bakedNormalTex);
            _bakedNormalsDirty = false;
            Debug.Log($"[TerrainRenderer] Terrain normals baked: {hmW}x{hmH}");
        }

        /// <summary>
        /// Dispatches the compute prepass (3 stages) + lean AS/MS for decoration rendering.
        /// </summary>
        private void DispatchDecorator(ID3D12GraphicsCommandList commandList, int frameIndex, RenderPass pass = RenderPass.Opaque)
        {
            if (!_computeInitialized) return; // Destroyed between enqueue and execute
            if (DecoratorMaterial?.Effect == null || Terrain == null) return;
            // A freshly created terrain has no heightmap until its first bake; decorators need it for placement.
            if (Terrain.Heightmap == null) return;

            // GPU prepass: composite the deco stamps into the control texture (when they changed)
            DispatchDecoControlPrepass(commandList);

            if (!_decoBuffersCreated) return;
            if (_decoratorSlotsBuffer == null || _decoratorLODTableBuffer == null || _decoratorGroupsBuffer == null) return;
            // Nothing to spawn from until the first coverage bake has run
            if (_decoControlTex == null) return;

            var device = Engine.Device;
            var camPos = Camera.Main!.Position;
            var cs = _grassCS!;

            float range = Terrain.DecorationRadius;
            int controlW = _decoControlTex != null ? (int)_decoControlTex.Description.Width : Terrain.EffectiveDecorationMapResolution;
            int controlH = _decoControlTex != null ? (int)_decoControlTex.Description.Height : Terrain.EffectiveDecorationMapResolution;
            float tileSize = Terrain.TerrainSize.X / controlW;


            // ════════════════════════════════════════════════════════════════
            // Phase 1: Compute Prepass (only on opaque pass — shadow reuses instances)
            // ════════════════════════════════════════════════════════════════

            if (pass == RenderPass.Opaque)
            {
                // Transition buffers to UAV
                _decoInstanceBuffer!.Transition(commandList, ResourceStates.UnorderedAccess);
                _meshDecoInstanceBuffer!.Transition(commandList, ResourceStates.UnorderedAccess);
                _sortedMeshInstanceBuffer!.Transition(commandList, ResourceStates.UnorderedAccess);
                _meshDrawArgsBuffer!.Transition(commandList, ResourceStates.UnorderedAccess);
                _instanceCounterBuffer!.Transition(commandList, ResourceStates.UnorderedAccess);
                _decoDispatchArgsBuffer!.Transition(commandList, ResourceStates.UnorderedAccess);

                // Clear instance counter
                commandList.SetComputeRootSignature(device.GlobalRootSignature);
                commandList.SetDescriptorHeaps(1, new[] { device.SrvHeap });
                _instanceCounterBuffer.ClearUAV(commandList, new Vortice.Mathematics.Int4(0, 0, 0, 0));
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

                // ── Push constants: bindless resource indices only ──
                cs.SetBuffer("DecoratorSlots", _decoratorSlotsBuffer!);
                cs.SetBuffer("LODTable", _decoratorLODTableBuffer!);
                cs.SetBuffer("DecoratorGroups", _decoratorGroupsBuffer!);
                cs.SetPushConstant("MeshRegistry", MeshRegistry.SrvIndex);
                if (Terrain.Heightmap != null) cs.SetTexture("Heightmap", Terrain.Heightmap);
                cs.SetPushConstant("DecoControl", _decoControlSRV);
                cs.SetPushConstant("BakedNormal", _bakedNormalSRV);

                // ── cbuffer DecoParams ──
                cs.SetParam("TerrainSize", new Vector2(Terrain.TerrainSize.X, Terrain.TerrainSize.Y));
                cs.SetParam("MaxHeight", Terrain.MaxHeight);
                cs.SetParam("DecoRadius", range);
                cs.SetParam("CamPos", camPos);
                cs.SetParam("TileSize", tileSize);
                cs.SetParam("TerrainOrigin", new Vector3(Transform.WorldPosition.X, Transform.WorldPosition.Z, Transform.WorldPosition.Y));
                cs.SetParam("DecorationDensity", Terrain.DecorationDensity);

                // Camera forward direction for half-space culling (normalized XZ)
                var camFwd = Camera.Main?.Transform?.Forward ?? Vector3.UnitZ;
                float fwdLen = MathF.Sqrt(camFwd.X * camFwd.X + camFwd.Z * camFwd.Z);
                if (fwdLen > 0.001f) { camFwd.X /= fwdLen; camFwd.Z /= fwdLen; }
                cs.SetParam("CamFwd", new Vector2(camFwd.X, camFwd.Z));

                cs.SetParam("ControlWidth", (uint)controlW);
                cs.SetParam("ControlHeight", (uint)controlH);
                cs.SetParam("SlotCount", (uint)_decoVariantCount);
                var hmDesc2 = Terrain.Heightmap!.Native.Description;
                cs.SetParam("HeightmapSize", new Vortice.Mathematics.UInt2((uint)hmDesc2.Width, (uint)hmDesc2.Height));

                // ── Camera-centered grid base tile ──
                int camTileX = (int)MathF.Floor((camPos.X - Transform.WorldPosition.X) / tileSize);
                int camTileZ = (int)MathF.Floor((camPos.Z - Transform.WorldPosition.Z) / tileSize);
                int N = (int)MathF.Ceiling(2 * range / tileSize) + 1;
                int baseTileX = camTileX - N / 2;
                int baseTileZ = camTileZ - N / 2;

                cs.SetParam("BaseTileX", baseTileX);
                cs.SetParam("BaseTileZ", baseTileZ);
                cs.SetParam("MaxInstances", (uint)_maxDecoInstances);

                // ── Per-kernel push constants (resource indices only) ──
                cs.SetBuffer(_kSpawnInstances, "DecoInstanceUAV", _decoInstanceBuffer);
                cs.SetBuffer(_kSpawnInstances, "InstanceCounterUAV", _instanceCounterBuffer);
                cs.SetBuffer(_kSpawnInstances, "MeshDecoInstanceUAV", _meshDecoInstanceBuffer);

                cs.SetBuffer(_kBuildDecoDrawArgs, "InstanceCounterUAV", _instanceCounterBuffer);
                cs.SetBuffer(_kBuildDecoDrawArgs, "DispatchArgsUAV", _decoDispatchArgsBuffer);

                // ── Stage 1: CS_SpawnInstances (single-pass, camera-centered grid) ──
                // Bind Hi-Z occlusion cbuffer (same _hizParamBuffers used by terrain quadtree)
                commandList.SetComputeRootConstantBufferView(2, _hizParamBuffers[frameIndex].GPUVirtualAddress);
                LastDispatchN = N;
                LastMaxInstances = _maxDecoInstances;
                cs.Dispatch(_kSpawnInstances, commandList, (uint)N, (uint)N);
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

                // ── Stage 2: CS_BuildDrawArgs ──
                cs.Dispatch(_kBuildDecoDrawArgs, commandList, 1);
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

                // Transition billboard instance buffer to SRV for MS reads
                _decoInstanceBuffer.Transition(commandList, ResourceStates.NonPixelShaderResource);

                // ── Stage 3: CS_BinMeshInstances ──
                // Transition mesh instance buffer to SRV for binning reads
                _meshDecoInstanceBuffer!.Transition(commandList, ResourceStates.NonPixelShaderResource);
                // DispatchArgs is still UAV — binning reads mesh count from offset 12
                cs.SetBuffer(_kBinMeshInstances, "DecoratorSlots", _decoratorSlotsBuffer!);
                cs.SetBuffer(_kBinMeshInstances, "LODTable", _decoratorLODTableBuffer!);
                cs.SetPushConstant(_kBinMeshInstances, "MeshRegistry", MeshRegistry.SrvIndex);
                cs.SetPushConstant(_kBinMeshInstances, "MeshDecoInstanceUAV", _meshDecoInstanceBuffer!.SrvIndex);
                cs.SetBuffer(_kBinMeshInstances, "DispatchArgsUAV", _decoDispatchArgsBuffer);
                cs.SetBuffer(_kBinMeshInstances, "SortedMeshInstanceUAV", _sortedMeshInstanceBuffer);
                cs.SetBuffer(_kBinMeshInstances, "MeshDrawArgsUAV", _meshDrawArgsBuffer);
                cs.SetBuffer(_kBinMeshInstances, "MeshDrawCountUAV", _meshDrawCountBuffer);

                // SRV indices to embed in draw commands (read by binning kernel, written to each command)
                cs.SetBuffer(_kBinMeshInstances, "DrawSortedSRV", _sortedMeshInstanceBuffer!);
                cs.SetBuffer(_kBinMeshInstances, "DrawSlotsSRV", _decoratorSlotsBuffer!);
                cs.SetBuffer(_kBinMeshInstances, "DrawLODSRV", _decoratorLODTableBuffer!);
                cs.SetPushConstant(_kBinMeshInstances, "DrawMeshRegSRV", MeshRegistry.SrvIndex);
                cs.SetPushConstant(_kBinMeshInstances, "DrawMaterialsSRV", Graphics.Material.MaterialsBufferIndex);

                cs.Dispatch(_kBinMeshInstances, commandList, 1);
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

                // Transition all buffers for rendering
                _decoDispatchArgsBuffer.Transition(commandList, ResourceStates.IndirectArgument);
                _sortedMeshInstanceBuffer!.Transition(commandList, ResourceStates.NonPixelShaderResource);
                _meshDrawArgsBuffer!.Transition(commandList, ResourceStates.IndirectArgument);

                // Readback instance counter for debug stats
                if (_instanceCounterReadback != null)
                {
                    unsafe
                    {
                        uint* pData = (uint*)_instanceCounterReadbackPtr;
                        LastInstanceCount = (int)pData[0];
                        LastMeshInstanceCount = (int)pData[1];
                        LastMeshDrawCount = (int)pData[2];
                    }

                    _instanceCounterBuffer.Transition(commandList, ResourceStates.CopySource);
                    commandList.CopyBufferRegion(_instanceCounterReadback, 0, _instanceCounterBuffer.Native, 0, 8);
                    _instanceCounterBuffer.Transition(commandList, ResourceStates.UnorderedAccess);

                    _meshDrawCountBuffer!.Transition(commandList, ResourceStates.CopySource);
                    commandList.CopyBufferRegion(_instanceCounterReadback, 8, _meshDrawCountBuffer.Native, 0, 4);
                }

                // The mesh-mode ExecuteIndirect reads its draw count from this buffer, so it must be in
                // IndirectArgument state (it was left in UnorderedAccess: no mesh decorator ever drew).
                _meshDrawCountBuffer!.Transition(commandList, ResourceStates.IndirectArgument);
            }

            // ════════════════════════════════════════════════════════════════
            // Phase 2: Lean MS dispatch (reads pre-computed instances)
            // ════════════════════════════════════════════════════════════════

            DecoratorMaterial.SetPass(pass);
            DecoratorMaterial.Apply(commandList, device);

            // Push constants for the lean MS
            commandList.SetGraphicsRoot32BitConstant(0, _decoratorSlotsBuffer!.SrvIndex, 0);  // DecoratorSlotsIdx
            commandList.SetGraphicsRoot32BitConstant(0, _decoratorLODTableBuffer!.SrvIndex, 1); // LODTableIdx
            commandList.SetGraphicsRoot32BitConstant(0, MeshRegistry.SrvIndex, 2);              // MeshRegistryIdx
            commandList.SetGraphicsRoot32BitConstant(0, _decoInstanceBuffer!.SrvIndex, 3);      // DecoInstanceSRV
            // InstanceCount — worst-case upper bound (AS will clamp based on actual count)
            commandList.SetGraphicsRoot32BitConstant(0, (uint)_maxDecoInstances, 4);
            commandList.SetGraphicsRoot32BitConstant(0, Graphics.Material.MaterialsBufferIndex, 5); // MaterialsBufferIdx
            commandList.SetGraphicsRoot32BitConstant(0, _bakedAlbedoSRV, 6);                    // BakedAlbedoIdx

            if (pass == RenderPass.Shadow)
            {
                commandList.SetGraphicsRoot32BitConstant(0, DirectionalLight.CurrentCascadeSrvIndex, 11);
                int grassShadowCascades = 0;// Math.Max(1, DirectionalLight.CascadeCount - 1);
                commandList.SetGraphicsRoot32BitConstant(0, (uint)grassShadowCascades, 12);
            }

            // ExecuteIndirect with DispatchMesh args written by CS_BuildDrawArgs
            using var commandList6 = commandList.QueryInterface<ID3D12GraphicsCommandList6>();
            commandList6.ExecuteIndirect(device.DispatchMeshSignature,1,_decoDispatchArgsBuffer!.Native, 0, null, 0);

            // ════════════════════════════════════════════════════════════════
            // Phase 3: Mesh-mode decorators via VS/PS + BindlessCommandSignature
            // ════════════════════════════════════════════════════════════════

            if (_meshDecoratorMaterial?.Effect != null && pass == RenderPass.Opaque)
            {
                _meshDecoratorMaterial.SetPass(pass);
                _meshDecoratorMaterial.Apply(commandList, device);

                // Push constants slots 0-1 are NOT overwritten by BindlessCommandSignature
                // (it only writes slots 2-15). All slots 2-15 are embedded in each draw command
                // by the binning kernel — no additional SetGraphicsRoot32BitConstant needed.

                commandList.ExecuteIndirect(
                    device.MeshDecoCommandSignature, // 72-byte commands from CS_BinMeshInstances
                    32,  // max 32 mesh types
                    _meshDrawArgsBuffer!.Native,
                    0,
                    _meshDrawCountBuffer!.Native,  // count buffer: actual number of draws
                    0);
            }

            // Back to UAV for next frame's binning pass (written by CS_BinMeshInstances)
            if (pass == RenderPass.Opaque)
                _meshDrawCountBuffer!.Transition(commandList, ResourceStates.UnorderedAccess);
        }

        // ── Terrain Stamp message handlers ─────────────────────────

        /// <summary>Diagnostics: stamp / spline change events that requested a re-bake (editor debug stats).</summary>
        public static int RebakeRequestCount;

        // ── Changed region ──
        // Listeners of TerrainHeightsChanged (PCG, surface meshes) only rebuild if they touch the region the
        // stamps changed: where each edited stamp was at the last readback, plus where it is now.

        private readonly Dictionary<TerrainStamp, (Vector2 min, Vector2 max)> _stampRegions = new();
        private bool _heightChangeAll = true;   // the first bake covers everything
        private bool _heightRegionMarked;       // a stamp gave a region for the pending bake
        private int _heightBakesPending;        // bakes enqueued whose render callback has not run yet
        private Vector2 _heightChangeMin = new(float.MaxValue);
        private Vector2 _heightChangeMax = new(float.MinValue);

        private static bool TryGetStampRegion(TerrainStamp stamp, out Vector2 min, out Vector2 max)
        {
            min = max = default;
            // A global stamp has no region: whatever it changes, it can change anywhere
            if (stamp.IsGlobal) return false;
            if (stamp.IsSplineMode && stamp.GetSpline().Points.Count < 2) return false;

            var bounds = stamp.GetWorldBounds();
            float pad = 2f + (stamp.EnableNoise ? stamp.NoiseAmplitude : 0f);
            min = new Vector2(bounds.Min.X - pad, bounds.Min.Z - pad);
            max = new Vector2(bounds.Max.X + pad, bounds.Max.Z + pad);
            return true;
        }

        private void GrowHeightChange(Vector2 min, Vector2 max)
        {
            _heightChangeMin = Vector2.Min(_heightChangeMin, min);
            _heightChangeMax = Vector2.Max(_heightChangeMax, max);
        }

        private void MarkStampRegion(TerrainStamp stamp)
        {
            _heightRegionMarked = true;

            if (_stampRegions.TryGetValue(stamp, out var old))
                GrowHeightChange(old.min, old.max);

            if (TryGetStampRegion(stamp, out var min, out var max))
            {
                GrowHeightChange(min, max);
                _stampRegions[stamp] = (min, max);
            }
            else
            {
                _heightChangeAll = true;
            }
        }

        private void SnapshotStampRegions()
        {
            _stampRegions.Clear();
            foreach (var stamp in ComponentCache<HeightStamp>.All) Snapshot(stamp);
            foreach (var stamp in ComponentCache<HeightNoiseStamp>.All) Snapshot(stamp);
            foreach (var stamp in ComponentCache<HeightErosionStamp>.All) Snapshot(stamp);
            foreach (var stamp in ComponentCache<SplatStamp>.All) Snapshot(stamp);
            foreach (var stamp in ComponentCache<DecoStamp>.All) Snapshot(stamp);

            void Snapshot(TerrainStamp stamp)
            {
                if (TryGetStampRegion(stamp, out var min, out var max))
                    _stampRegions[stamp] = (min, max);
            }
        }

        /// <summary>
        /// Request the rebakes a changed stamp needs.
        ///
        /// Height stamps rebake everything: the splat and deco stamps' height/slope filters depend on them.
        /// Local splat and deco stamps also go through the height path, although they leave the heights
        /// alone: its readback is what tells PCG and surface meshes which region changed, and PCG's
        /// ExcludeStamps depends on where the splat stamps are.
        /// Global splat and deco stamps skip it. They have no region, so they would regenerate every
        /// PCG component on the terrain for a change none of them can see.
        /// </summary>
        private void RequestRebake(TerrainStamp stamp)
        {
            RebakeRequestCount++;

            if (stamp is CoverageStamp { IsGlobal: true })
            {
                Terrain?.MarkForUpdate(stamp is DecoStamp
                    ? TerrainDirtyFlags.DecoPrepass
                    : TerrainDirtyFlags.SplatPack | TerrainDirtyFlags.AlbedoBake | TerrainDirtyFlags.DecoPrepass);
                return;
            }

            MarkStampRegion(stamp);
            Terrain?.MarkForUpdate(
                TerrainDirtyFlags.HeightBake |
                TerrainDirtyFlags.SplatPack |
                TerrainDirtyFlags.AlbedoBake |
                TerrainDirtyFlags.DecoPrepass);
        }

        private void OnStampChanged(Message msg)
        {
            if (msg.Data is TerrainStamp stamp)
            {
                RequestRebake(stamp);
                return;
            }

            // Sender unknown: rebake everything
            RebakeRequestCount++;
            Terrain?.MarkForUpdate(
                TerrainDirtyFlags.HeightBake |
                TerrainDirtyFlags.SplatPack |
                TerrainDirtyFlags.AlbedoBake |
                TerrainDirtyFlags.DecoPrepass);
        }

        private void OnSplineChanged(Message msg)
        {
            // Rebake if the changed spline has any sibling stamp component. Components are cached by their
            // concrete type, so GetComponent<TerrainStamp>() never finds a HeightStamp / SplatStamp / DecoStamp.
            if (msg.Data is not Spline spline || spline.Entity == null) return;

            foreach (var component in spline.Entity.Components)
                if (component is TerrainStamp stamp)
                    RequestRebake(stamp);
        }
    }
}
