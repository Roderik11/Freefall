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
        private Vector4[] _layerTint = Enumerable.Repeat(Vector4.One, 32).ToArray();   // TerrainLayer.Tint per channel
        private Vector4[] _layerPlantTint = Enumerable.Repeat(Vector4.One, 32).ToArray();   // TerrainLayer.PlantTint per channel
        private GraphicsBuffer? _plantTintBuffer;   // _layerPlantTint for the decorator spawn (grass_compute.hlsl LayerPlantTint)

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
            public uint  ShadowReceiversIdx;    // shadow pass: this frame's sun shadow receivers (0 = no test)
            public uint  Pad2, Pad3, Pad4;
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

        // The layer weights (texture array, baked); the layers' own textures by bindless index per channel
        private Texture? ControlMapsArray;
        private LayerTextureIds[] _layerTextures = new LayerTextureIds[32];
        private bool _layerHeights;     // some layer has a height map (push slot 26)

        // Must match gputerrain.fx LayerTextures (cbuffer tiling)
        [StructLayout(LayoutKind.Sequential)]
        private struct LayerTextureIds
        {
            public uint Diffuse, Normals, Height, Pad;
        }

        private int _totalNodes;

        // ───── Ground Coverage Decorator ──────────────────────────────────
        private GraphicsBuffer? _decoratorHeadersBuffer;
        private GraphicsBuffer? _decoratorSlotsBuffer;
        private GraphicsBuffer? _decoratorLODTableBuffer;
        private GraphicsBuffer? _decoratorGroupsBuffer;  // one (variant offset, count) per decorator slot
        private int _decoSlotCapacity, _decoLodCapacity, _decoGroupCapacity;
        private int _decoVariantCount;
        private float _decoMaxReach;  // farthest any decorator reaches from its root (m): sizes the spawn's cell cull sphere
        private bool _decoratorBuffersBuilt;
        private bool _decoratorDispatched;

        private int _decoratorDispatchGroupsX;
        private int _decoratorDispatchGroupsY;

        // ───── Compute Prepass (grass_compute.hlsl) ──────────────────────
        private ComputeShader? _grassCS;
        private int _kBakeNormals, _kSpawnInstances, _kBuildDecoDrawArgs, _kBinMeshInstances;
        private GraphicsBuffer? _decoInstanceBuffer;    // StructuredBuffer<DecoInstance> = 64 bytes
        private GraphicsBuffer? _instanceCounterBuffer; // RWByteAddressBuffer (3 uints: card, mesh, blade clump counts)
        private GraphicsBuffer? _decoDispatchArgsBuffer; // RWByteAddressBuffer (7 uints: card DispatchMesh, mesh count, blade DispatchMesh at 16)
        private bool _decoBuffersCreated;
        private bool _decoDispatchLogged;
        private int _maxDecoInstances;

        // ───── Mesh-Mode Decorators ──────────────────────────────────────────
        private GraphicsBuffer? _meshDecoInstanceBuffer;    // unsorted mesh instances (DecoInstance, 64 bytes)
        private GraphicsBuffer? _sortedMeshInstanceBuffer;  // sorted by mesh type (DecoInstance, 64 bytes)
        private GraphicsBuffer? _meshDrawArgsBuffer;        // BindlessDrawCommand per mesh type (72 bytes × 32)
        private GraphicsBuffer? _meshDrawCountBuffer;       // RWByteAddressBuffer (1 uint: draw count)
        private Material? _meshDecoratorMaterial;           // grass_mesh.fx

        // ───── Blade Clumps (card grass near the camera, grass_blades.fx) ─────
        private const int MaxBladeInstances = 128 * 1024;
        private const int BladeColorsPerVariant = 6;        // = COLORS_PER_VARIANT in the shaders: 4 height bands, foliage, bloom
        private GraphicsBuffer? _bladeInstanceBuffer;       // DecoInstance, 64 bytes; padded to whole AS groups
        private GraphicsBuffer? _bladeColorBuffer;          // float4 per variant x BladeColorsPerVariant (CS_BladeColors)
        private GraphicsBuffer? _decoratorTintBuffer;       // 2 float4 per decorator: ground tint and clumps, CS_DecoratorTints
        private bool _bladeColorsDirty = true;
        private bool _decoHasBlades;                        // some variant draws a band as Blades (grass_blades.fx)
        private bool _decoHasPlants;                        // ... as Flowers or Spikes (grass_plants.fx)
        private Material? _bladeMaterial;
        private Material? _plantMaterial;
        private int _kBladeColors, _kDecoratorTints;

        // ───── Debug Stats (read by editor SettingsControls) ─────────────
        public static int LastInstanceCount { get; set; }
        public static int LastBladeInstanceCount { get; set; }
        public static int LastMeshInstanceCount { get; set; }
        public static int LastMeshDrawCount { get; set; }
        public static int LastMaxInstances { get; set; }
        public static int LastDispatchN { get; set; }
        private ID3D12Resource? _instanceCounterReadback;
        private IntPtr _instanceCounterReadbackPtr;

        // ───── Baked Terrain Normals (one-time) ──────────────────────────
        private ID3D12Resource? _bakedNormalTex;        // R16G16_SNORM
        private bool _bakedNormalReadable;              // false: still in Common as created; true: in AllShaderResource
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

        // ───── Top Layer Map (which layer is where, for GetLayer) ──────────
        // One texel per control texel: the palette channel that shows most there. The splat bake
        // writes it with the weights; the CPU copy is fetched only once somebody has asked.
        private ID3D12Resource? _topLayerTex;      // R8_UINT
        private uint _topLayerUAV;
        private byte[]? _topLayers;                // row by row, as the control array lies (Y-flipped)
        private int _topLayersResolution;
        private List<TerrainLayer> _topLayersPalette = new();   // what the channels in _topLayers mean
        private List<TerrainLayer> _bakedPalette = new();       // what they mean in the texture
        private int _bakedTopLayerResolution;
        private bool _topLayersWanted;
        private volatile bool _topLayersStale;     // a bake has run since the CPU copy was fetched

        // ───── Holes (HoleStamp) ───────────────────────────────────────────
        // The mask itself is Terrain.Holes, baked on the CPU (TerrainBaker.BakeHoles) because the PhysX height
        // field, the navmesh and PCG need it there; this is its copy for the shaders (terrain_holes.hlsli).
        private ID3D12Resource? _holeTex;          // R8_UNorm, one texel per height sample
        private uint _holeSRV;
        private int _holeTexResolution;
        private List<TerrainBaker.HoleRect>? _holeRects;   // what the last hole bake cut; null = not known (mask from the cache)

        /// <summary>The hole mask the shaders read (bindless SRV), 0 when nothing opens the terrain.</summary>
        private uint HoleSrv => Terrain is { HasHoles: true } && _holeTex != null ? _holeSRV : 0u;

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
            MessageDispatcher.AddListener("AssetReloaded", OnAssetReloaded);
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
            device?.DeferDispose(_plantTintBuffer);

            // ── Packed control array ──
            device?.DeferDispose(_packedControlArray);
            if (_packedControlSRV != 0) device?.DeferReleaseBindlessIndex(_packedControlSRV);
            if (_packedSliceUAVs != null)
                foreach (var idx in _packedSliceUAVs)
                    if (idx != 0) device?.DeferReleaseBindlessIndex(idx);
            if (_packedControlArrayUAV != 0) device?.DeferReleaseBindlessIndex(_packedControlArrayUAV);

            // ── Top layer map ──
            device?.DeferDispose(_topLayerTex);
            if (_topLayerUAV != 0) device?.DeferReleaseBindlessIndex(_topLayerUAV);
            _topLayers = null;

            // ── Hole mask (the CPU mask stays with the Terrain asset, like its heights) ──
            device?.DeferDispose(_holeTex);
            if (_holeSRV != 0) device?.DeferReleaseBindlessIndex(_holeSRV);
            _holeTex = null;
            _holeSRV = 0;

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

            // ── Blade clumps ──
            device?.DeferDispose(_bladeInstanceBuffer);
            device?.DeferDispose(_bladeColorBuffer);
            device?.DeferDispose(_decoratorTintBuffer);

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
            MessageDispatcher.RemoveListener("AssetReloaded", OnAssetReloaded);
        }

        private bool _textureArraysInitialized;
        private bool _initialBakeRequested;

        // The first Draw after a load does every bake: these show which part of it costs what
        private static readonly ProfilerMarker _uploadMarker = ProfilerMarker.Get("Terrain heightmap upload");
        private static readonly ProfilerMarker _holesMarker = ProfilerMarker.Get("Terrain holes");
        private static readonly ProfilerMarker _planMarker = ProfilerMarker.Get("Terrain height plan");
        private static readonly ProfilerMarker _paletteMarker = ProfilerMarker.Get("Terrain palettes");
        private static readonly ProfilerMarker _arraysMarker = ProfilerMarker.Get("Terrain texture arrays");
        private static readonly ProfilerMarker _readbackMarker = ProfilerMarker.Get("Terrain height readback");
        private static readonly ProfilerMarker _heightsChangedMarker = ProfilerMarker.Get("Terrain heights changed");
        private static readonly ProfilerMarker _splatPlanMarker = ProfilerMarker.Get("Terrain splat plan");
        private static readonly ProfilerMarker _decoPlanMarker = ProfilerMarker.Get("Terrain deco plan");

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

            // While a load wakes the scene its stamps are still arriving: the first bakes wait for the load's terrain
            // step, so they see every stamp. Before, a bake during the wake-up cut the holes of the stamps awake so far
            // (none, often), then again with all of them, and the editor took the half-cut mask for a change to save.
            if (SceneManager.Loading is { Current: < SceneLoad.Step.Terrain }) return;

            int frameIndex = Engine.FrameIndex % FrameCount;

            // Upload saved baked heightmap from cache (skips GPU compositor if present)
            if (Terrain.PendingBakedHeightmapBytes != null)
            {
                using var _ = _uploadMarker.Auto();
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
                        Terrain.HeightmapStampsHash = Terrain.BakedStampsHash;
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
            // Bake once per renderer so this scene's stamps always win, unless the heights in place were baked from
            // stamps that hash the same (below: KeepsHeights).
            if (!_initialBakeRequested)
            {
                _initialBakeRequested = true;
                _initialHeightCheck = true;
                Terrain.MarkForUpdate(TerrainDirtyFlags.HeightBake | TerrainDirtyFlags.SplatPack |
                                      TerrainDirtyFlags.AlbedoBake | TerrainDirtyFlags.DecoPrepass |
                                      TerrainDirtyFlags.HoleBake);
            }

            // ── Holes: cut on the CPU, where physics, the navmesh and PCG read them; the shaders get a copy ──
            // What sampled the ground there hears of it once the stamps have settled (not while one is dragged)
            if (Terrain.ConsumeFlags(TerrainDirtyFlags.HoleBake))
            {
                using (_holesMarker.Auto())
                    RebakeHoles();
            }
            else if (_holeChanges.Count > 0)
            {
                NotifyHolesChanged(_holeChanges);
                _holeChanges.Clear();
            }

            // ── Height: lay out the bake from the height stamps in scope ──
            bool heightBake = Terrain.ConsumeFlags(TerrainDirtyFlags.HeightBake);
            TerrainBaker.HeightPlan? heightPlan = null;
            if (heightBake)
            {
                using (_planMarker.Auto())
                    heightPlan = _baker.PrepareHeight(Terrain, this);
            }
            bool hasHeightWork = heightPlan is { HasWork: true };
            if (heightBake)
            {
                // A bake nobody gave a region for (first bake, terrain settings) may change anything
                if (!_heightRegionMarked) _heightChangeAll = true;
                _heightRegionMarked = false;
            }

            // The first bake of this renderer, of a map that is already right: keep it
            if (heightBake && _initialHeightCheck)
            {
                _initialHeightCheck = false;
                if (hasHeightWork && KeepsHeights(heightPlan!))
                {
                    hasHeightWork = false;
                    KeepHeights();
                }
            }

            // ── Palette: the layers and decorators this terrain renders are whatever its stamps reference ──
            // Resolved before anything built from it (texture arrays, layer params, decorator buffers).
            // New heights change what the stamps' height/slope filters let through, so they rebake both.
            bool splatAsked = Terrain.ConsumeFlags(TerrainDirtyFlags.SplatPack);
            bool splatDirty = splatAsked || hasHeightWork;
            bool decoDirty = Terrain.ConsumeFlags(TerrainDirtyFlags.DecoPrepass) || splatDirty;
            // Where the splat bake composites again: the regions local stamps' edits gave, or everywhere. New heights
            // that no stamp asked for (terrain settings) may have moved any stamp's height or slope filter.
            bool splatLocal = Terrain.TakeSplatRegion(out var splatRegionMin, out var splatRegionMax) && splatAsked;

            if (splatDirty || decoDirty)
            {
                using var _ = _paletteMarker.Auto();
                var paletteBefore = _layerPalette;
                if (splatDirty) RefreshLayerPalette();
                if (_layerPalette != paletteBefore) splatLocal = false;   // other channels: every texel changes
                RefreshDecoratorPalette();
            }

            if (Terrain.ConsumeFlags(TerrainDirtyFlags.TextureArrays) || !_textureArraysInitialized)
            {
                try
                {
                    using var _ = _arraysMarker.Auto();
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
                    PixMarker.Begin(list, "Terrain height bake");
                    baker.BakeHeight(terrain, plan, list);
                    PixMarker.End(list);
                    terrain.HeightmapStampsHash = plan.Hash;
                    _heightRangePyramidBuilt = false; // force rebuild with new heights
                    terrain.MarkForUpdate(TerrainDirtyFlags.AlbedoBake); // re-bake albedo with new terrain shape
                    _needHeightFieldReadback = true; // trigger CPU-side heightfield rebuild next frame
                    renderer._bakedNormalsDirty = true; // normals depend on height
                    PixMarker.Begin(list, "Terrain normals bake");
                    renderer.BakeTerrainNormals(list);
                    PixMarker.End(list);
                });
            }

            // Capture heightmap AFTER cache upload / bake — BakedHeightmap may have just been set above
            var heightmap = Terrain.Heightmap;

            // Debounced CPU heightfield readback — only when baking has settled (not while a stamp is dragged)
            if (_needHeightFieldReadback && !Terrain.NeedsUpdate(TerrainDirtyFlags.HeightBake))
            {
                _needHeightFieldReadback = false;
                using var readbackScope = _readbackMarker.Auto();
                // Only where the stamps changed it since the last readback, when a field of this size is there to
                // copy (in texels: sample i stands at (i + 0.5) · size / N, see GetHeight; a texel more each side)
                float[,]? heights;
                var previousField = Terrain.HeightField;
                if (!_heightChangeAll && previousField != null && Transform != null && _heightChangeMax.X >= _heightChangeMin.X)
                {
                    var origin = Transform.WorldPosition;
                    var size = Terrain.TerrainSize;
                    int n = previousField.GetLength(0);
                    heights = _baker.ReadbackHeightmap(previousField,
                        (int)MathF.Floor((_heightChangeMin.X - origin.X) / size.X * n - 0.5f) - 1,
                        (int)MathF.Floor((_heightChangeMin.Y - origin.Z) / size.Y * n - 0.5f) - 1,
                        (int)MathF.Ceiling((_heightChangeMax.X - origin.X) / size.X * n - 0.5f) + 2,
                        (int)MathF.Ceiling((_heightChangeMax.Y - origin.Z) / size.Y * n - 0.5f) + 2);
                }
                else
                    heights = _baker.ReadbackHeightmap();
                if (heights != null && Terrain != null)
                {
                    Terrain.SetHeightField(heights);
                    Terrain.HeightFieldStampsHash = Terrain.HeightmapStampsHash;   // read back from the map as it is now
                    Terrain.CheckMapsAgainstSaved();    // on Save Changed Assets when the bake no longer equals the saved heightmap
                    _heightsReadBack = true;
                    // Surface-snapped geometry (RuntimeMesh) and projected PCG output sampled the old heights —
                    // let whatever lies in the changed region rebuild
                    var change = new TerrainHeightsChange(Terrain, _heightChangeAll, _heightChangeMin, _heightChangeMax);
                    Debug.Log($"[Terrain] Heights read back (tick {Engine.TickCount}): " +
                              (change.All ? "all" : $"x {change.Min.X:0}..{change.Max.X:0}, z {change.Min.Y:0}..{change.Max.Y:0}") +
                              $", {System.Threading.Volatile.Read(ref _heightBakesPending)} bake(s) pending");

                    // A bake that is enqueued but has not run yet (at load: this readback is of the cached
                    // heightmap) still owes its region to the readback that follows it.
                    if (System.Threading.Volatile.Read(ref _heightBakesPending) == 0)
                    {
                        _heightChangeAll = false;
                        _heightChangeMin = new Vector2(float.MaxValue);
                        _heightChangeMax = new Vector2(float.MinValue);
                        SnapshotStampRegions();
                    }
                    using (_heightsChangedMarker.Auto())
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
                    var arrayBefore = _packedControlArray;
                    EnsurePackedControlArray(res, sliceCount);
                    ControlMapsArray = _packedControlTexture;

                    // A local edit redoes only its region (control texels; the map is Y-flipped against the terrain's
                    // UV), a texel more each side. A new array has nothing outside it yet.
                    (int x, int y, int width, int height)? region = null;
                    if (splatLocal && arrayBefore == _packedControlArray && arrayBefore != null)
                    {
                        var origin = Transform?.WorldPosition ?? Vector3.Zero;
                        var size = Terrain.TerrainSize;
                        int x0 = (int)MathF.Floor((splatRegionMin.X - origin.X) / size.X * res) - 1;
                        int x1 = (int)MathF.Ceiling((splatRegionMax.X - origin.X) / size.X * res) + 1;
                        int y0 = (int)MathF.Floor((1f - (splatRegionMax.Y - origin.Z) / size.Y) * res) - 1;
                        int y1 = (int)MathF.Ceiling((1f - (splatRegionMin.Y - origin.Z) / size.Y) * res) + 1;
                        x0 = Math.Clamp(x0, 0, res); x1 = Math.Clamp(x1, 0, res);
                        y0 = Math.Clamp(y0, 0, res); y1 = Math.Clamp(y1, 0, res);
                        region = (x0, y0, Math.Max(0, x1 - x0), Math.Max(0, y1 - y0));
                    }
                    SplatBakeTexels = region is { } r ? (long)r.width * r.height : (long)res * res;

                    TerrainBaker.CoveragePlan plan;
                    using (_splatPlanMarker.Auto())
                        plan = _baker.PrepareSplat(Terrain, this, _splatStamps, _layerPalette);
                    var baker = _baker;
                    var terrain = Terrain;
                    var packedArray = _packedControlArray;
                    var arrayUAV = _packedControlArrayUAV;
                    var topLayerTex = _topLayerTex!;
                    var topLayerUAV = _topLayerUAV;
                    var palette = _layerPalette;   // replaced, never changed, when the layers change
                    var renderer = this;
                    CommandBuffer.Enqueue(RenderPass.Opaque, (list) =>
                    {
                        // The height bake enqueued above has run by now, so this reads the new heights
                        var hm = terrain.Heightmap;
                        uint hmSrv = hm != null ? (uint)hm.BindlessIndex : 0;

                        list.ResourceBarrierTransition(packedArray,
                            ResourceStates.Common, ResourceStates.UnorderedAccess);
                        list.ResourceBarrierTransition(topLayerTex,
                            ResourceStates.Common, ResourceStates.UnorderedAccess);

                        PixMarker.Begin(list, "Terrain splat bake");
                        baker.BakeSplat(plan, list, terrain, hmSrv, arrayUAV, topLayerUAV, res, sliceCount, region);
                        PixMarker.End(list);
                        list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(packedArray)));
                        list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(topLayerTex)));

                        // Back to common for shader reads (and for the copy GetLayer's readback makes)
                        list.ResourceBarrierTransition(packedArray,
                            ResourceStates.UnorderedAccess, ResourceStates.Common);
                        list.ResourceBarrierTransition(topLayerTex,
                            ResourceStates.UnorderedAccess, ResourceStates.Common);

                        terrain.MarkForUpdate(TerrainDirtyFlags.AlbedoBake);

                        renderer._bakedPalette = palette;
                        renderer._bakedTopLayerResolution = res;
                        renderer._topLayersStale = true;
                    });
                }
                else
                {
                    // No splat stamp references a layer: nothing to render the surface with
                    ControlMapsArray = InternalAssets.BlackArray;
                    _topLayers = null;
                    _topLayersStale = false;
                }
            }

            // The CPU copy of which layer is where, once somebody has asked (GetLayer) and the bakes have
            // settled: the same blocking fetch as the height field's, so not while a stamp is dragged.
            if (_topLayersWanted && _topLayersStale && !splatDirty && _topLayerTex != null
                && _bakedTopLayerResolution == _packedArrayResolution)
            {
                _topLayersStale = false;
                var palette = _bakedPalette;
                var layers = TerrainBaker.ReadbackTopLayers(_topLayerTex, _bakedTopLayerResolution);
                if (layers != null)
                {
                    _topLayersPalette = palette;
                    _topLayersResolution = _bakedTopLayerResolution;
                    _topLayers = layers;
                }
            }

            // ── Decoration coverage: laid out here, run by the decorator dispatch below ──
            if (decoDirty)
            {
                using var _ = _decoPlanMarker.Auto();
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
            material.SetParameter("LayerTint", _layerTint);
            material.SetParameter("LayerPlantTint", _layerPlantTint);
            material.SetParameter("LayerTextures", _layerTextures);
            material.SetParameter("DecoRadius", Terrain.DecorationRadius);
            material.SetParameter("DecoGroundTint", Terrain.DrawDetail ? Terrain.DecorationGroundTint : 0f);

            // Bind textures — the control map MUST be a valid Texture2DArray (the PS samples it with a slice index;
            // a Texture2D would cause a GPU fault). The layers' own textures go by index (LayerTextures above).
            // Bind textures if available (terrain still renders without splatmaps)

            if (heightmap != null) material.SetTexture("HeightTex", heightmap);
            if (ControlMapsArray != null) material.SetTexture("ControlMaps", ControlMapsArray);

            // Capture values for lambda closure
            int fi = frameIndex;
            var self = this;

            // Enqueue compute dispatch as custom action (runs first in Execute, before batch processing)
            CommandBuffer.Enqueue(RenderPass.Opaque, (list) => { PixMarker.Begin(list, "Terrain quadtree"); self.DispatchQuadtreeEval(list, fi); PixMarker.End(list); });

            // Self-draw via ExecuteIndirect, in two steps: depth only before the mesh batches (the terrain still hides
            // what lies behind hills), the splat shading after them, so only terrain pixels that stay visible pay for
            // it (gputerrain.fx technique). In one draw before the batches, a town plaza view paid ~1.4 ms of splat
            // shading for terrain the paving and houses covered.
            CommandBuffer.Enqueue(RenderPass.Opaque, (list) => { PixMarker.Begin(list, "Terrain depth"); self.DrawTerrain(list, fi, "DepthPrepass"); PixMarker.End(list); });
            CommandBuffer.EnqueueLate(RenderPass.Opaque, (list) => { PixMarker.Begin(list, "Terrain"); self.DrawTerrain(list, fi, "Shade"); PixMarker.End(list); });

            // Enqueue single-pass terrain shadow draw
            CommandBuffer.Enqueue(RenderPass.Shadow, (list) => { PixMarker.Begin(list, "Terrain"); self.DrawTerrainShadow(list, fi); PixMarker.End(list); });

            // Decoration pipeline — CPU-side setup
            if (Terrain.DrawDetail && _decoratorPalette.Count > 0 && DecoratorMaterial?.Effect != null)
            {
                if (Terrain.ConsumeFlags(TerrainDirtyFlags.DecoStructure | TerrainDirtyFlags.DecoParams) || _decoratorSlotsBuffer == null)
                    RebuildDecoSlots();

                if (_decoVariantCount > 0)
                {
                    EnsureDecoRenderBuffers();

                    CommandBuffer.Enqueue(RenderPass.Opaque, (list) => { PixMarker.Begin(list, "Decorators"); self.DispatchDecorator(list, fi, RenderPass.Opaque); PixMarker.End(list); });
                    CommandBuffer.Enqueue(RenderPass.Shadow, (list) => { PixMarker.Begin(list, "Decorators"); self.DispatchDecorator(list, fi, RenderPass.Shadow); PixMarker.End(list); });
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
        /// An asset was reimported in place. The layer arrays hold copies of a texture's old pixels, so a layer
        /// texture changed on disk (a reprocessed height map) never showed until the layer itself was edited; a layer
        /// or decorator file edited on disk is the same as an edit in the inspector.
        /// </summary>
        private void OnAssetReloaded(Message msg)
        {
            if (Terrain == null) return;
            if (msg.Data is TerrainLayer or TerrainDecorator)
            {
                OnAssetDirty(msg);
                return;
            }
            if (msg.Data is not Texture texture) return;

            foreach (var layer in _layerPalette)
                if (layer.Diffuse == texture || layer.Normals == texture || layer.Height == texture)
                {
                    Terrain.MarkForUpdate(TerrainDirtyFlags.TextureArrays | TerrainDirtyFlags.AlbedoBake);
                    return;
                }
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

            // The top layer map goes with the array: same texels, filled by the same bake
            device.DeferDispose(_topLayerTex);
            device.DeferReleaseBindlessIndex(_topLayerUAV);

            _topLayerTex = device.CreateTexture2D(Format.R8_UInt, resolution, resolution, 1, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
            _topLayerUAV = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(_topLayerTex, null,
                new UnorderedAccessViewDescription
                {
                    Format = Format.R8_UInt,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
                }, device.GetCpuHandle(_topLayerUAV));
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
            int shadowCapacity = MaxPatches * DirectionalLight.MaxCascades;

            // Structured buffers with SRV + UAV
            _descriptorBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 20, srv: true, uav: true);
            _sphereBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 16, srv: true, uav: true);
            _subbatchIdBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 4, srv: true, uav: true);
            _terrainDataBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 32, srv: true, uav: true);

            // Raw R32 buffers
            _counterBuffers[i] = GraphicsBuffer.CreateRaw(1, uav: true, clearable: true);
            _splitFlagsBuffers[i] = GraphicsBuffer.CreateRaw(_totalNodes, uav: true, clearable: true);
            // D3D12_DRAW_INDEXED_ARGUMENTS (5 uints, CSBuildDrawArgs)
            _indirectArgsBuffers[i] = GraphicsBuffer.CreateRaw(5, srv: true, uav: true);
            _shadowArgsBuffers[i] = GraphicsBuffer.CreateRaw(5, uav: true);

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
            commandList.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            commandList.SetComputeRootSignature(device.GlobalRootSignature);
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
                commandList.SetDescriptorHeaps(1, new[] { device.SrvHeap });
                commandList.SetComputeRootSignature(device.GlobalRootSignature);
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
        /// <param name="pass">Technique pass by name: "DepthPrepass" (before the mesh batches) or "Shade" (after them)</param>
        private void DrawTerrain(ID3D12GraphicsCommandList commandList, int frameIndex, string pass)
        {
            if (!_computeInitialized) return; // Destroyed between enqueue and execute

            var device = Engine.Device;

            // Holes: the depth prepass needs its variant that can discard (see the technique in gputerrain.fx)
            uint holes = HoleSrv;
            if (holes != 0 && pass == "DepthPrepass") pass = "DepthPrepass_Holes";

            // Apply material PSO and textures (the pass explicitly: the shadow draw changes it)
            Material!.SetPass(pass);
            Material!.Apply(commandList, device);

            // Set topology; the patch mesh's indices, drawn indexed (SV_VertexID is the grid vertex): the post-transform
            // cache shades a vertex the neighbouring triangles share once, not once per corner
            commandList.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleList);
            commandList.IASetIndexBuffer(_patchMesh.IndexBufferView);

            // Set descriptor heap (Material.Apply may have changed it)
            commandList.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            // Set root constants for push constant slots used by the vertex shader
            // Slot 1: TerrainPatchData SRV
            commandList.SetGraphicsRoot32BitConstant(0, _terrainDataBuffers[frameIndex].SrvIndex, 1);
            // Slot 2: InstanceDescriptor SRV
            commandList.SetGraphicsRoot32BitConstant(0, _descriptorBuffers[frameIndex].SrvIndex, 2);

            // Slot 9: Position buffer SRV
            commandList.SetGraphicsRoot32BitConstant(0, _patchMesh.PosBufferIndex, 9);
            // Slot 15: GlobalTransformBuffer SRV
            commandList.SetGraphicsRoot32BitConstant(0, TransformBuffer.Instance!.SrvIndex, 15);
            // Slot 16: Debug mode
            commandList.SetGraphicsRoot32BitConstant(0, (uint)Engine.Settings.DebugVisualizationMode, 16);
            // Slot 21: Deco control map SRV (ground tint, debug mode 5); slot 22: the tint per decorator
            commandList.SetGraphicsRoot32BitConstant(0, _decoControlSRV, 21);
            commandList.SetGraphicsRoot32BitConstant(0, _decoratorTintBuffer?.SrvIndex ?? 0u, 22);
            // Slot 26: some layer has a height map (blend and SSDM; LayerTextures holds them)
            commandList.SetGraphicsRoot32BitConstant(0, _layerHeights ? 1u : 0u, 26);
            // Slot 27: hole mask (HoleStamp), 0 = none
            commandList.SetGraphicsRoot32BitConstant(0, holes, 27);

            // ExecuteIndirect with DrawIndexedInstancedSignature — args written by CSBuildDrawArgs
            commandList.ExecuteIndirect(
                device.DrawIndexedInstancedSignature,
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

            // Upload local-space cascade planes to StructuredBuffer<CascadeData>, and the cascades' VPs from terrain-local
            // space (the receiver test, CSEmitLeavesShadow)
            var terrainPos = Transform.Position;
            var cascadeVPs = DirectionalLight.CascadeViewProjections;
            var localToWorld = Matrix4x4.CreateTranslation(terrainPos);
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
                if (cascadeVPs != null && c < cascadeVPs.Length)
                    cascadePtr[c].VP = localToWorld * cascadeVPs[c];
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
                // A patch casts into a cascade only where a surface the camera sees through it lies behind
                // (receiver_mask_culling.md); the receivers were built from this frame's depth before this action ran
                ShadowReceiversIdx = cascadeVPs != null ? DirectionalLight.CurrentShadowReceiversUav : 0u,
            };
            UploadBuffer(_shadowTerrainParamBuffers[frameIndex], shadowTerrainParams);

            // Clear shadow counter
            commandList.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            commandList.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
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

            // Where a hole opens the terrain it casts no shadow: light falls into the cave mouth
            uint holes = HoleSrv;
            if (holes != 0) Material.SetPass("Shadow_Holes");
            else Material.SetPass(RenderPass.Shadow);
            Material.Apply(commandList, device);

            commandList.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleList);
            commandList.IASetIndexBuffer(_patchMesh.IndexBufferView);                             // drawn indexed, as DrawTerrain
            commandList.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            commandList.SetGraphicsRoot32BitConstant(0, holes, 27);                                // HoleTexIdx

            // Bind shadow patch data
            commandList.SetGraphicsRoot32BitConstant(0, _shadowTerrainDataBuffers[frameIndex].SrvIndex, 1);  // TerrainDataIdx
            commandList.SetGraphicsRoot32BitConstant(0, _shadowDescriptorBuffers[frameIndex].SrvIndex, 2);   // DescriptorBufIdx
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
                device.DrawIndexedInstancedSignature,
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
            Array.Fill(_layerTint, Vector4.One);
            Array.Fill(_layerPlantTint, Vector4.One);

            for (int i = 0; i < _layerPalette.Count && i < _layerTiling.Length; i++)
            {
                var layer = _layerPalette[i];
                _layerTint[i] = new Vector4(layer.Tint.R, layer.Tint.G, layer.Tint.B, layer.TileRotation * MathF.PI / 180f);
                _layerPlantTint[i] = new Vector4(layer.PlantTint.R, layer.PlantTint.G, layer.PlantTint.B, 1f);
                float hasHeight = (layer.Height?.Native != null) ? 1.0f : 0.0f;
                if (layer.Tiling.X != 0 && layer.Tiling.Y != 0)
                    _layerTiling[i] = new Vector4(terrainSize.X / layer.Tiling.X, terrainSize.Y / layer.Tiling.Y, hasHeight, layer.HeightScale);
                else
                    _layerTiling[i] = new Vector4(1, 1, hasHeight, layer.HeightScale);
            }

            // Written only when a layer changes; a frame in flight may still read the previous tints, which is
            // harmless for a colour
            _plantTintBuffer ??= GraphicsBuffer.CreateUpload<Vector4>(32);
            _plantTintBuffer.Upload<Vector4>(_layerPlantTint.AsSpan());
        }

        /// <summary>
        /// Point each palette channel at its layer's own textures (bindless indices, gputerrain.fx LayerTextures).
        /// Called when the TextureArrays flag is consumed (the set of layers or a layer's textures changed).
        /// Also refreshes tiling since channels may have moved.
        /// Until 2026-10-24 the textures were copied into three Texture2DArrays. Layers whose textures differ in size or
        /// format (Eldermoor: 1024 and 2048 px; BC7, BC5, BC4, BC1) sent every array down CreateTexture2DArray's compute
        /// path: uncompressed RGBA8 at the largest size, ~270 MB each for 12 layers, never released, and 4 to 8 times the
        /// bytes per texel the splat shader fetched. Sampled where they are, each texture keeps its own size and format.
        /// </summary>
        private void RebuildTextureArrays()
        {
            // Indexed by channel, so a layer without a texture gets a stand-in: skipped, it shifted every layer after it
            // onto its neighbour's textures (2026-10-17: a cobble layer that lost its diffuse rendered the next layer's
            // grass scan, untinted). No diffuse = the layer's Tint alone, no normals = flat, no height = never sampled
            // (LayerTiling.z is 0 for it, UpdateLayerParams).
            var missingDiffuse = new List<string>();
            var missingNormals = new List<string>();
            int normalCount = 0;
            Array.Clear(_layerTextures);
            _layerHeights = false;

            for (int i = 0; i < _layerPalette.Count && i < _layerTextures.Length; i++)
            {
                var layer = _layerPalette[i];
                var diffuse = layer.Diffuse?.Native != null ? layer.Diffuse : InternalAssets.White;
                if (diffuse != layer.Diffuse) missingDiffuse.Add(layer.Name);

                var normals = layer.Normals?.Native != null ? layer.Normals : InternalAssets.FlatNormal;
                if (normals != layer.Normals) missingNormals.Add(layer.Name);
                else normalCount++;

                var height = layer.Height?.Native != null ? layer.Height : InternalAssets.Black;
                _layerHeights |= height == layer.Height;

                _layerTextures[i] = new LayerTextureIds
                {
                    Diffuse = (uint)diffuse.BindlessIndex,
                    Normals = (uint)normals.BindlessIndex,
                    Height = (uint)height.BindlessIndex,
                };
            }

            if (missingDiffuse.Count > 0)
                Debug.LogWarning("TerrainRenderer", $"Terrain layer(s) without a Diffuse texture render as their Tint: {string.Join(", ", missingDiffuse)}");
            if (missingNormals.Count > 0 && normalCount > 0)
                Debug.LogWarning("TerrainRenderer", $"Terrain layer(s) without a Normals texture render flat: {string.Join(", ", missingNormals)}");

            // Also refresh tiling since channels may have moved
            UpdateLayerParams();

            // The layer weights are baked separately (SplatPack); until the first bake nothing shows
            ControlMapsArray = _packedControlTexture ?? InternalAssets.BlackArray;
        }

        // ───── Holes ───────────────────────────────────────────────────────

        /// <summary>
        /// Re-cut the hole stamps into Terrain.Holes and bring the shaders' copy up to date. Only the samples the
        /// stamps reach now or reached at the last bake are touched, so dragging a hole stamp costs its own area,
        /// not the terrain's. Main thread.
        /// </summary>
        private void RebakeHoles()
        {
            var terrain = Terrain!;
            var plan = TerrainBaker.PrepareHoles(terrain, this);
            int res = plan.Resolution;
            var mask = terrain.Holes;
            bool sameMask = mask != null && terrain.HoleResolution == res;
            // Where stamps cut at the last bake; null = not known (first bake: the mask came from the cache or
            // another scene). No mask after a bake of this renderer = nothing was cut.
            var cutBefore = _holeRects == null ? null
                : mask == null ? new List<TerrainBaker.HoleRect>()
                : sameMask ? _holeRects : null;

            if (!plan.HasHoles)
            {
                if (mask != null)
                {
                    terrain.SetHoles(null, 0);
                    terrain.CheckMapsAgainstSaved();
                    if (cutBefore != null) _holeChanges.AddRange(cutBefore);
                }
                _holeRects = new List<TerrainBaker.HoleRect>();
                return;
            }

            var cutNow = plan.Rects.ToList();
            if (!sameMask)
            {
                mask = new byte[res * res];
                TerrainBaker.BakeHoles(plan, mask, Array.Empty<TerrainBaker.HoleRect>());
            }
            else
            {
                // A mask of unknown origin (the saved one, or another scene's) is cleared whole once
                TerrainBaker.BakeHoles(plan, mask!, cutBefore ?? [new TerrainBaker.HoleRect(0, 0, res - 1, res - 1)]);
            }
            terrain.SetHoles(mask, res);
            terrain.CheckMapsAgainstSaved();
            _holeRects = cutNow;

            // What changed: where stamps cut before and where they cut now (all of it when that is not known)
            var changed = cutBefore != null ? cutBefore.Concat(cutNow).Distinct().ToList() : null;
            // The texture holds only what was uploaded since it was made, and nothing that was cut while no
            // mask existed: all of it then
            bool created = EnsureHoleTexture(res);
            UploadHoles(mask!, res, created || !sameMask ? null : changed);
            if (changed != null)
                foreach (var rect in changed)
                    if (!_holeChanges.Contains(rect)) _holeChanges.Add(rect);
        }

        // Rects whose holes changed and whose listeners have not heard of it yet (sent once the bakes settle)
        private readonly List<TerrainBaker.HoleRect> _holeChanges = new();

        /// <summary>The R8 texture for the hole mask; true if it was (re)created and holds nothing yet.</summary>
        private bool EnsureHoleTexture(int resolution)
        {
            if (_holeTex != null && _holeTexResolution == resolution) return false;

            var device = Engine.Device;
            device.DeferDispose(_holeTex);
            if (_holeSRV != 0) device.DeferReleaseBindlessIndex(_holeSRV);

            _holeTex = device.CreateTexture2D(Format.R8_UNorm, resolution, resolution, 1, 1,
                ResourceFlags.None, ResourceStates.Common);
            _holeSRV = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(_holeTex,
                new ShaderResourceViewDescription
                {
                    Format = Format.R8_UNorm,
                    ViewDimension = ShaderResourceViewDimension.Texture2D,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
                }, device.GetCpuHandle(_holeSRV));
            _holeTexResolution = resolution;
            return true;
        }

        /// <summary>
        /// Copy rects of the mask into the hole texture (null = all of it). The rows are copied into upload
        /// buffers here, so the render callback never reads the mask while the next bake changes it.
        /// </summary>
        private unsafe void UploadHoles(byte[] mask, int res, List<TerrainBaker.HoleRect>? rects)
        {
            var device = Engine.Device;
            var tex = _holeTex!;
            rects ??= [new TerrainBaker.HoleRect(0, 0, res - 1, res - 1)];

            foreach (var r in rects)
            {
                int w = r.X1 - r.X0 + 1, h = r.Z1 - r.Z0 + 1;
                int rowPitch = (w + 255) & ~255;
                var upload = device.NativeDevice.CreateCommittedResource(
                    new HeapProperties(HeapType.Upload), HeapFlags.None,
                    ResourceDescription.Buffer((ulong)(rowPitch * h)), ResourceStates.GenericRead, null);

                void* data;
                upload.Map(0, null, &data);
                fixed (byte* src = mask)
                    for (int z = 0; z < h; z++)
                        Buffer.MemoryCopy(src + (long)(r.Z0 + z) * res + r.X0, (byte*)data + (long)z * rowPitch, w, w);
                upload.Unmap(0);

                var footprint = new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R8_UNorm, (uint)w, (uint)h, 1, (uint)rowPitch)
                };
                uint x0 = (uint)r.X0, z0 = (uint)r.Z0;
                CommandBuffer.Enqueue(RenderPass.Opaque, list =>
                {
                    list.ResourceBarrierTransition(tex, ResourceStates.Common, ResourceStates.CopyDest);
                    list.CopyTextureRegion(new TextureCopyLocation(tex, 0), x0, z0, 0, new TextureCopyLocation(upload, footprint));
                    list.ResourceBarrierTransition(tex, ResourceStates.CopyDest, ResourceStates.Common);
                });
                device.DeferDispose(upload);   // until the frames that copy from it are done
            }
        }

        /// <summary>
        /// Tell what samples the ground (PCG's TerrainProjection drops points in holes) that the holes changed in
        /// these rects. A first bake after a load records no rects: its height readback announces the whole
        /// terrain anyway.
        /// </summary>
        private void NotifyHolesChanged(List<TerrainBaker.HoleRect> rects)
        {
            if (Terrain == null || Transform == null) return;

            var origin = Transform.WorldPosition;
            float step = Terrain.TerrainSize.X / (Terrain.EffectiveHeightmapResolution - 1);
            foreach (var r in rects)
            {
                var min = new Vector2(origin.X + r.X0 * step, origin.Z + r.Z0 * step);
                var max = new Vector2(origin.X + r.X1 * step, origin.Z + r.Z1 * step);
                MessageDispatcher.Send(EngineMsg.TerrainHeightsChanged, new TerrainHeightsChange(Terrain, false, min, max));
            }
        }

        /// <summary>
        /// True where a HoleStamp opens the terrain (X and Z of the position): no ground is drawn there and nothing
        /// collides with it. GetHeight still answers with the surface the hole was cut out of.
        /// </summary>
        public bool IsHole(Vector3 position)
        {
            var terrain = Terrain;
            if (terrain is not { HasHoles: true } || Transform == null) return false;
            var origin = Transform.Position;
            return terrain.IsHole(position.X - origin.X, position.Z - origin.Z);
        }

        // ───── IHeightProvider ────────────────────────────────────────────

        /// <summary>
        /// The terrain's height at a point, as it is drawn: the height map sampled bilinearly, the way the terrain's
        /// vertex shader samples it. Texel (i, j) holds the height of the point ((i + 0.5) · size / N, (j + 0.5) · size / N)
        /// from the terrain's corner: the height bake works at texel centres, and stamps are placed at
        /// uv = (world − origin) / size. Outside the terrain, its own height.
        ///
        /// Until 2026-10-24 this took sample i at i · size / N, half a texel off, and the plane through three corners
        /// of the cell, which extrapolates in its other half: on bumpy ground tens of centimetres from what is drawn
        /// (animals stood in the ground). The physics height field (TerrainCollider) and the navmesh bake
        /// (TerrainSource.SampleHeight) put their samples at the same places; the hole mask keeps a grid of its own
        /// and is read by place (terrain_system/overview.md).
        /// </summary>
        public float GetHeight(Vector3 position)
        {
            var heightField = Terrain?.HeightField;
            if (heightField == null) return Transform?.Position.Y ?? 0;
            if (Transform == null) return 0;

            var terrainSize = Terrain!.TerrainSize;
            position -= Transform.Position;
            if (position.X < 0f || position.X >= terrainSize.X || position.Z < 0f || position.Z >= terrainSize.Y)
                return Transform.Position.Y;

            int nx = heightField.GetLength(0), nz = heightField.GetLength(1);

            // In texels, with the texel centres on whole numbers; clamped at the edges like the GPU's sampler
            float tx = Math.Clamp(position.X / terrainSize.X * nx - 0.5f, 0f, nx - 1);
            float tz = Math.Clamp(position.Z / terrainSize.Y * nz - 0.5f, 0f, nz - 1);
            int x0 = (int)tx, z0 = (int)tz;
            int x1 = Math.Min(x0 + 1, nx - 1), z1 = Math.Min(z0 + 1, nz - 1);
            float fx = tx - x0, fz = tz - z0;

            float near = heightField[x0, z0] + (heightField[x1, z0] - heightField[x0, z0]) * fx;
            float far = heightField[x0, z1] + (heightField[x1, z1] - heightField[x0, z1]) * fx;
            return Transform.Position.Y + (near + (far - near) * fz) * Terrain.MaxHeight;
        }

        /// <summary>
        /// The layer that shows most at a place on the terrain (X and Z of the position), or null:
        /// off the terrain, nothing painted there, or not known yet. The answer comes from a CPU copy
        /// of the splat bake's result that is only fetched once somebody asks, so the first calls after
        /// a load return null for a frame or two; after the stamps changed, the old answer stands until
        /// the new one has been fetched. One array lookup; main thread.
        /// </summary>
        public TerrainLayer? GetLayer(Vector3 position)
        {
            _topLayersWanted = true;

            var layers = _topLayers;
            if (layers == null || Terrain == null || Transform == null) return null;

            var size = Terrain.TerrainSize;
            var origin = Transform.Position;
            float u = (position.X - origin.X) / size.X;
            float v = (position.Z - origin.Z) / size.Y;
            if (!(u >= 0f && u < 1f && v >= 0f && v < 1f)) return null;

            // The control array is Y-flipped relative to terrain UV (see terrain_splat_bake.hlsl)
            int res = _topLayersResolution;
            int x = Math.Min((int)(u * res), res - 1);
            int y = Math.Min((int)((1f - v) * res), res - 1);

            int channel = layers[y * res + x];
            var palette = _topLayersPalette;
            return channel < palette.Count ? palette[channel] : null;
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
            public uint BandModes;        // distance bands near → far: DecoratorMode per byte, 0xFF = no band
            public uint Mode;             // 0=Mesh, 1=Billboard, 2=Cross
            public uint TextureIdx;       // bindless index (billboard/cross)
            // Color tint (Unity detail prototype healthy/dry colors)
            public Vector3 HealthyColor;
            public Vector3 DryColor;
            public float NoiseSpread;
            public uint BandEnds01;       // far ends of bands 0 and 1 (m), two halves
            public uint BandEnds23;       // bands 2 and 3
            public float ClusterScale;    // world size of density clumps (m), 0 = off
            public float ClusterAmount;   // 0 = uniform, 1 = full clumps with bare gaps
            public uint AlphaClip;        // mesh mode: 1 = alpha-test albedo, 0 = opaque material
            public float HeightNoiseScale;  // world size of short vs. lush patches (m), 0 = off
            public float HeightNoiseAmount; // 0 = none, 1 = 0.4x .. 1.4x height
            public uint PlantPartsIdx;      // Flowers / Spikes / Ferns / Grain: bindless index of the parts atlas, 0 = none
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
            float maxReach = 0;
            bool hasBlades = false, hasPlants = false;

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

                    // How far the variant reaches from its root at its largest: cards go up by H (plus wind sway)
                    // and out by W/2; a mesh is scaled by (W, H, W). Height noise stretches H by up to 1.4x.
                    float maxH = variant.HeightRange.Y * (1f + 0.4f * Math.Clamp(variant.HeightNoiseAmount, 0f, 1f));
                    float maxW = variant.WidthRange.Y;
                    float reach = 1.3f * maxH + 0.5f * maxW;
                    if (mode == DecoratorMode.Mesh)
                    {
                        var bounds = variant.Mesh!.LocalBoundingSphere;
                        reach = (new Vector3(bounds.X, bounds.Y, bounds.Z).Length() + bounds.W) * MathF.Max(maxH, maxW) * 1.2f;
                    }
                    maxReach = MathF.Max(maxReach, reach);

                    // Distance bands, near → far. A card variant draws each band as its mode (a Mesh band
                    // would have no mesh: Cross); a mesh variant stays a mesh and takes only the range.
                    uint bandModes = 0xFFFFFFFFu, bandEnds01 = 0, bandEnds23 = 0;
                    if (variant.Bands is { Count: > 0 } bands)
                    {
                        var sorted = bands.Where(b => b != null).OrderBy(b => b.MaxDistance).Take(TerrainDecorator.MaxBands).ToList();
                        for (int b = 0; b < sorted.Count; b++)
                        {
                            var bandMode = mode == DecoratorMode.Mesh ? DecoratorMode.Mesh
                                : sorted[b].Mode == DecoratorMode.Mesh ? DecoratorMode.Cross : sorted[b].Mode;
                            bandModes = (bandModes & ~(0xFFu << (8 * b))) | ((uint)bandMode << (8 * b));
                            uint half = BitConverter.HalfToUInt16Bits((Half)MathF.Max(0f, sorted[b].MaxDistance));
                            if (b < 2) bandEnds01 |= half << (16 * b);
                            else bandEnds23 |= half << (16 * (b - 2));
                            if (bandMode == DecoratorMode.Blades) hasBlades = true; else if (bandMode > DecoratorMode.Blades) hasPlants = true;
                        }
                    }
                    if (mode == DecoratorMode.Blades) hasBlades = true; else if (mode > DecoratorMode.Blades) hasPlants = true;

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
                        BandModes = bandModes,
                        BandEnds01 = bandEnds01,
                        BandEnds23 = bandEnds23,
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
                        PlantPartsIdx = (uint)(variant.PlantParts?.BindlessIndex ?? 0),
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
            _decoMaxReach = maxReach;
            _decoHasBlades = hasBlades;
            _decoHasPlants = hasPlants;
            _decoratorBuffersBuilt = true;

            // Blade colours are taken from the variants' textures again (CS_BladeColors, next decorator dispatch)
            int bladeColorCount = Math.Max(1, slots.Count) * BladeColorsPerVariant;
            if (_bladeColorBuffer == null || _bladeColorBuffer.ElementCount != bladeColorCount)
            {
                Engine.Device.DeferDispose(_bladeColorBuffer);
                _bladeColorBuffer = GraphicsBuffer.CreateStructured(bladeColorCount, 16, srv: true, uav: true);
            }
            int tintCount = Math.Max(1, groups.Count) * 2;     // colour + even weight, clumps (CS_DecoratorTints)
            if (_decoratorTintBuffer == null || _decoratorTintBuffer.ElementCount != tintCount)
            {
                Engine.Device.DeferDispose(_decoratorTintBuffer);
                _decoratorTintBuffer = GraphicsBuffer.CreateStructured(tintCount, 16, srv: true, uav: true);
            }
            _bladeColorsDirty = true;
        }

        /// <summary>One procedural decorator pipeline over the procedural instance buffer (grass_procedural.hlsli).</summary>
        private void DrawProcedural(ID3D12GraphicsCommandList6 commandList, RenderPass pass, Material? material, string marker)
        {
            if (material?.Effect == null || _bladeColorBuffer == null) return;
            var device = Engine.Device;
            material.SetPass(pass);
            material.Apply(commandList, device);
            commandList.SetGraphicsRoot32BitConstant(0, _decoratorSlotsBuffer!.SrvIndex, 0);
            commandList.SetGraphicsRoot32BitConstant(0, _bladeInstanceBuffer!.SrvIndex, 1);
            commandList.SetGraphicsRoot32BitConstant(0, _bladeColorBuffer.SrvIndex, 2);
            commandList.SetGraphicsRoot32BitConstant(0, (uint)Math.Clamp(Engine.Settings.GrassBladesPerClump, 1, 28), 3);
            if (pass == RenderPass.Shadow)
                commandList.SetGraphicsRoot32BitConstant(0, DirectionalLight.CurrentCascadeSrvIndex, 4);

            PixMarker.Begin(commandList, marker);
            commandList.ExecuteIndirect(device.DispatchMeshSignature, 1, _decoDispatchArgsBuffer!.Native, 16, null, 0);
            PixMarker.End(commandList);
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

            // Transition the control texture to UAV for writing: from Common when fresh (a texture is never promoted
            // to UnorderedAccess implicitly), from the SRV state on a re-dispatch
            cmd.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_decoControlTex,
                    freshTexture ? ResourceStates.Common : ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess)));

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
            _instanceCounterBuffer = GraphicsBuffer.CreateRaw(3, uav: true, clearable: true); // card, mesh, blade clump counts
            _decoDispatchArgsBuffer = GraphicsBuffer.CreateRaw(7, uav: true); // card DispatchMesh (3), mesh count, blade DispatchMesh (3)
            _bladeInstanceBuffer = GraphicsBuffer.CreateStructured(MaxBladeInstances, 64, srv: true, uav: true);

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
            _kBladeColors       = _grassCS.FindKernel("CS_BladeColors");
            _kDecoratorTints    = _grassCS.FindKernel("CS_DecoratorTints");

            // Load mesh decorator material
            _meshDecoratorMaterial ??= new Material(new Effect("grass_mesh"));
            _bladeMaterial ??= new Material(new Effect("grass_blades"));
            _plantMaterial ??= new Material(new Effect("grass_plants"));

            Debug.Log($"[TerrainRenderer] Deco buffers created: maxTiles={maxTiles} maxInstances={_maxDecoInstances} resolution={resolution}");

            // Readback buffer: card count, mesh count, mesh draw count, blade clump count (4 uints)
            _instanceCounterReadback = Engine.Device.NativeDevice.CreateCommittedResource(
                new Vortice.Direct3D12.HeapProperties(Vortice.Direct3D12.HeapType.Readback),
                Vortice.Direct3D12.HeapFlags.None,
                Vortice.Direct3D12.ResourceDescription.Buffer(16),
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
                _bakedNormalReadable = false;

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
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            cmd.SetComputeRootSignature(device.GlobalRootSignature);

            _grassCS!.SetTexture(_kBakeNormals, "Heightmap", Terrain.Heightmap);
            _grassCS.SetPushConstant(_kBakeNormals, "BakedNormalUAV", _bakedNormalUAV);
            _grassCS.SetParam("TerrainSize", new Vector2(Terrain.TerrainSize.X, Terrain.TerrainSize.Y));
            _grassCS.SetParam("MaxHeight", Terrain.MaxHeight);
            _grassCS.SetParam("HeightmapSize", new Vortice.Mathematics.UInt2((uint)hmW, (uint)hmH));

            uint groupsX = (uint)((hmW + 7) / 8);
            uint groupsY = (uint)((hmH + 7) / 8);
            // A texture is never promoted to UnorderedAccess implicitly; after the bake the decoration compute and
            // the terrain's pixel shaders read it
            cmd.ResourceBarrierTransition(_bakedNormalTex,
                _bakedNormalReadable ? ResourceStates.AllShaderResource : ResourceStates.Common, ResourceStates.UnorderedAccess);
            _grassCS.Dispatch(_kBakeNormals, cmd, groupsX, groupsY);
            cmd.ResourceBarrierTransition(_bakedNormalTex, ResourceStates.UnorderedAccess, ResourceStates.AllShaderResource);
            _bakedNormalReadable = true;
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
            PixMarker.Begin(commandList, "Control prepass");
            DispatchDecoControlPrepass(commandList);
            PixMarker.End(commandList);

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
                _bladeInstanceBuffer!.Transition(commandList, ResourceStates.UnorderedAccess);
                _meshDecoInstanceBuffer!.Transition(commandList, ResourceStates.UnorderedAccess);
                _sortedMeshInstanceBuffer!.Transition(commandList, ResourceStates.UnorderedAccess);
                _meshDrawArgsBuffer!.Transition(commandList, ResourceStates.UnorderedAccess);
                _instanceCounterBuffer!.Transition(commandList, ResourceStates.UnorderedAccess);
                _decoDispatchArgsBuffer!.Transition(commandList, ResourceStates.UnorderedAccess);

                // Clear instance counter
                commandList.SetDescriptorHeaps(1, new[] { device.SrvHeap });
                commandList.SetComputeRootSignature(device.GlobalRootSignature);
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

                // The layer weights and the layers' plant tints: each plant takes the colour of the ground it stands on
                bool plantTints = ControlMapsArray != null && _plantTintBuffer != null && _layerPalette.Count > 0;
                if (plantTints)
                {
                    cs.SetTexture("SplatControl", ControlMapsArray!);
                    cs.SetBuffer("LayerPlantTint", _plantTintBuffer!);
                }
                cs.SetPushConstant("SplatLayerCount", plantTints ? (uint)_layerPalette.Count : 0u);
                cs.SetPushConstant("HoleTex", HoleSrv);   // nothing spawns in a hole

                // ── cbuffer DecoParams ──
                cs.SetParam("TerrainSize", new Vector2(Terrain.TerrainSize.X, Terrain.TerrainSize.Y));
                cs.SetParam("MaxHeight", Terrain.MaxHeight);
                cs.SetParam("DecoRadius", range);
                cs.SetParam("CamPos", camPos);
                cs.SetParam("TileSize", tileSize);
                cs.SetParam("TerrainOrigin", new Vector3(Transform.WorldPosition.X, Transform.WorldPosition.Z, Transform.WorldPosition.Y));
                cs.SetParam("DecorationDensity", Terrain.DecorationDensity);

                // Cell culling against the camera frustum (planes at b0). A cell just outside the view can still
                // throw its shadow into it, so the cull sphere is swept along the light by the longest shadow
                // a decorator casts on flat ground (a few metres; more when the sun is low).
                cs.SetParam("DecoReach", _decoMaxReach);
                var shadowSweep = Vector3.Zero;
                if (Engine.Settings.EnableShadows)
                {
                    var lightDir = DirectionalLight.GetShadowDirection();
                    if (lightDir.Y < -0.001f)
                        shadowSweep = lightDir * MathF.Min(_decoMaxReach / MathF.Max(-lightDir.Y, 0.1f), 20f);
                }
                cs.SetParam("ShadowSweep", shadowSweep);

                cs.SetParam("ControlWidth", (uint)controlW);
                cs.SetParam("ControlHeight", (uint)controlH);
                cs.SetParam("SlotCount", (uint)_decoVariantCount);
                var hmDesc2 = Terrain.Heightmap!.Native.Description;
                cs.SetParam("HeightmapSize", new Vortice.Mathematics.UInt2((uint)hmDesc2.Width, (uint)hmDesc2.Height));

                // ── Camera-centered grid: K rings of cells around the camera's cell (the spawn walks it as a spiral) ──
                int camTileX = (int)MathF.Floor((camPos.X - Transform.WorldPosition.X) / tileSize);
                int camTileZ = (int)MathF.Floor((camPos.Z - Transform.WorldPosition.Z) / tileSize);
                int rings = (int)MathF.Ceiling(range / tileSize) + 1;
                int N = 2 * rings + 1;
                int baseTileX = camTileX - rings;
                int baseTileZ = camTileZ - rings;

                cs.SetParam("BaseTileX", baseTileX);
                cs.SetParam("BaseTileZ", baseTileZ);
                cs.SetParam("GridSize", (uint)N);
                cs.SetParam("MaxInstances", (uint)_maxDecoInstances);
                cs.SetParam("MaxBladeInstances", (uint)MaxBladeInstances);

                // The snow buries the grass and flowers (grass_compute.hlsl SnowBurial), less where the cover map keeps it off
                var shelter = ShelterMap.Current;
                cs.SetParam("SnowCover", Weather.SnowCover);
                cs.SetParam("ShelterMinY", shelter.MinY);
                cs.SetParam("ShelterRangeY", shelter.RangeY);
                cs.SetParam("ShelterLevel0", new Vector3(shelter.L0.Left, shelter.L0.Top, shelter.L0.Size));
                cs.SetParam("ShelterMap0", shelter.L0.Map);
                cs.SetParam("ShelterLevel1", new Vector3(shelter.L1.Left, shelter.L1.Top, shelter.L1.Size));
                cs.SetParam("ShelterMap1", shelter.L1.Map);

                // ── Per-kernel push constants (resource indices only) ──
                cs.SetBuffer(_kSpawnInstances, "DecoInstanceUAV", _decoInstanceBuffer);
                cs.SetBuffer(_kSpawnInstances, "InstanceCounterUAV", _instanceCounterBuffer);
                cs.SetBuffer(_kSpawnInstances, "MeshDecoInstanceUAV", _meshDecoInstanceBuffer);
                cs.SetBuffer(_kSpawnInstances, "BladeInstanceUAV", _bladeInstanceBuffer);

                cs.SetBuffer(_kBuildDecoDrawArgs, "InstanceCounterUAV", _instanceCounterBuffer);
                cs.SetBuffer(_kBuildDecoDrawArgs, "DispatchArgsUAV", _decoDispatchArgsBuffer);
                cs.SetBuffer(_kBuildDecoDrawArgs, "BladeInstanceUAV", _bladeInstanceBuffer);

                // ── Blade colours from the card textures (when the variants changed) ──
                if (_bladeColorsDirty && _bladeColorBuffer != null)
                {
                    _bladeColorBuffer.Transition(commandList, ResourceStates.UnorderedAccess);
                    cs.SetBuffer(_kBladeColors, "BladeColorsUAV", _bladeColorBuffer);
                    PixMarker.Begin(commandList, "Blade colours");
                    cs.Dispatch(_kBladeColors, commandList, (uint)_decoVariantCount);
                    commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
                    if (_decoratorTintBuffer != null)
                    {
                        // The ground tint per decorator (gputerrain.fx) from those colours
                        _decoratorTintBuffer.Transition(commandList, ResourceStates.UnorderedAccess);
                        cs.SetBuffer(_kDecoratorTints, "BladeColorsUAV", _bladeColorBuffer);
                        cs.SetBuffer(_kDecoratorTints, "DecoratorTintsUAV", _decoratorTintBuffer);
                        cs.Dispatch(_kDecoratorTints, commandList, (uint)((_decoratorTintBuffer.ElementCount + 63) / 64));
                        commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
                        _decoratorTintBuffer.Transition(commandList, ResourceStates.PixelShaderResource);
                    }
                    PixMarker.End(commandList);
                    _bladeColorBuffer.Transition(commandList, ResourceStates.NonPixelShaderResource);
                    _bladeColorsDirty = false;
                }

                // ── Stage 1: CS_SpawnInstances (single-pass, camera-centered grid) ──
                // Bind the camera frustum (terrain-local) and Hi-Z cbuffers the terrain quadtree uploaded this frame
                commandList.SetComputeRootConstantBufferView(1, _frustumPlaneBuffers[frameIndex].GPUVirtualAddress);
                commandList.SetComputeRootConstantBufferView(2, _hizParamBuffers[frameIndex].GPUVirtualAddress);
                LastDispatchN = N;
                LastMaxInstances = _maxDecoInstances;
                PixMarker.Begin(commandList, "Spawn");
                cs.Dispatch(_kSpawnInstances, commandList, (uint)N, (uint)N);
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
                PixMarker.End(commandList);

                // ── Stage 2: CS_BuildDrawArgs ──
                PixMarker.Begin(commandList, "Build args");
                cs.Dispatch(_kBuildDecoDrawArgs, commandList, 1);
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
                PixMarker.End(commandList);

                // Transition billboard and blade clump instance buffers to SRV for MS reads
                _decoInstanceBuffer.Transition(commandList, ResourceStates.NonPixelShaderResource);
                _bladeInstanceBuffer.Transition(commandList, ResourceStates.NonPixelShaderResource);

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

                PixMarker.Begin(commandList, "Bin meshes");
                cs.Dispatch(_kBinMeshInstances, commandList, 1);
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
                PixMarker.End(commandList);

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
                        LastBladeInstanceCount = (int)pData[3];
                    }

                    _instanceCounterBuffer.Transition(commandList, ResourceStates.CopySource);
                    commandList.CopyBufferRegion(_instanceCounterReadback, 0, _instanceCounterBuffer.Native, 0, 8);
                    commandList.CopyBufferRegion(_instanceCounterReadback, 12, _instanceCounterBuffer.Native, 8, 4);
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
                // Grass casts into the nearest cascade only (further out screen-space shadows carry it);
                // AS_Shadow also skips the instances outside it
                const int grassShadowCascades = 1;
                commandList.SetGraphicsRoot32BitConstant(0, (uint)grassShadowCascades, 12);
            }

            // ExecuteIndirect with DispatchMesh args written by CS_BuildDrawArgs
            using var commandList6 = commandList.QueryInterface<ID3D12GraphicsCommandList6>();
            PixMarker.Begin(commandList, "Grass draw");
            commandList6.ExecuteIndirect(device.DispatchMeshSignature,1,_decoDispatchArgsBuffer!.Native, 0, null, 0);
            PixMarker.End(commandList);

            // The procedural bands (DispatchMesh args at offset 16): blade clumps (grass_blades.fx), then flowers and
            // spikes (grass_plants.fx). Both walk the same instances, each building only its own kinds: the plants'
            // alpha-tested parts would cost the blades their early depth test in one pipeline.
            DrawProcedural(commandList6, pass, _decoHasBlades ? _bladeMaterial : null, "Grass blades");
            DrawProcedural(commandList6, pass, _decoHasPlants ? _plantMaterial : null, "Flowers");

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

                PixMarker.Begin(commandList, "Mesh decorators");
                commandList.ExecuteIndirect(
                    device.MeshDecoCommandSignature, // 72-byte commands from CS_BinMeshInstances
                    32,  // max 32 mesh types
                    _meshDrawArgsBuffer!.Native,
                    0,
                    _meshDrawCountBuffer!.Native,  // count buffer: actual number of draws
                    0);
                PixMarker.End(commandList);
            }

            // Back to UAV for next frame's binning pass (written by CS_BinMeshInstances)
            if (pass == RenderPass.Opaque)
                _meshDrawCountBuffer!.Transition(commandList, ResourceStates.UnorderedAccess);
        }

        // ── Terrain Stamp message handlers ─────────────────────────

        /// <summary>Diagnostics: stamp / spline change events that requested a re-bake (editor debug stats).</summary>
        public static int RebakeRequestCount;

        /// <summary>Diagnostics: control texels the last splat bake composited (the whole map, or a local edit's region).</summary>
        [Reflection.DontSerialize]
        public long SplatBakeTexels { get; private set; }

        // ── Changed region ──
        // Listeners of TerrainHeightsChanged (PCG, surface meshes) only rebuild if they touch the region the
        // stamps changed: where each edited stamp was at the last readback, plus where it is now.

        /// <summary>
        /// The CPU height field is missing or about to be replaced: a height bake or its readback is still
        /// outstanding. Whatever samples heights now (PCG graphs) gets flat ground or values that change in
        /// a frame or two; TerrainHeightsChanged follows once they are in.
        /// </summary>
        [System.ComponentModel.Browsable(false)]
        public bool HeightsPending
            => Enabled && Terrain != null
            && (Terrain.HeightField == null || !_heightsReadBack || _needHeightFieldReadback
                || System.Threading.Volatile.Read(ref _heightBakesPending) > 0
                || Terrain.NeedsUpdate(TerrainDirtyFlags.HeightBake));

        // This renderer has read heights back at least once. Until then a Terrain asset still cached from the
        // scene before a reload looks ready, although this renderer's first readback is yet to come.
        private bool _heightsReadBack;

        // The first height bake of this renderer is yet to be laid out: it may keep the heights in place
        private bool _initialHeightCheck;

        /// <summary>
        /// The GPU heightmap and the CPU height field both hold what this plan would bake (the saved map at a cold load,
        /// the last bake at a reload): equal <see cref="TerrainBaker.HeightPlan.Hash"/>es bake equal maps.
        /// </summary>
        private bool KeepsHeights(TerrainBaker.HeightPlan plan)
        {
            if (Terrain.Heightmap == null || Terrain.HeightField == null) return false;
            ulong hash = plan.Hash;
            return hash == Terrain.HeightmapStampsHash && hash == Terrain.HeightFieldStampsHash;
        }

        /// <summary>
        /// Skip the first bake: no GPU bake, no readback and no TerrainHeightsChanged (nothing changed). What the bake
        /// would have left behind is set up here: the heights count as read back, the stamp regions are where the
        /// stamps are, and this renderer's normals are baked from the map in place. At load this saves a full-terrain
        /// bake (~1.5 s of GPU on Eldermoor) and the readback that waited for it.
        /// </summary>
        private void KeepHeights()
        {
            _needHeightFieldReadback = false;
            _heightsReadBack = true;
            _heightChangeAll = false;
            _heightChangeMin = new Vector2(float.MaxValue);
            _heightChangeMax = new Vector2(float.MinValue);
            SnapshotStampRegions();

            _bakedNormalsDirty = true;
            var renderer = this;
            CommandBuffer.Enqueue(RenderPass.Opaque, (list) =>
            {
                PixMarker.Begin(list, "Terrain normals bake");
                renderer.BakeTerrainNormals(list);
                PixMarker.End(list);
            });
            Debug.LogAlways($"[Terrain] '{Terrain.Name}': heights kept, the stamps are those they were baked from");
        }

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

        /// <summary>Where a changed local stamp was and is: the region the height readback reports and the splat bake redoes.</summary>
        private void MarkStampRegion(TerrainStamp stamp)
        {
            _heightRegionMarked = true;

            if (_stampRegions.TryGetValue(stamp, out var old))
            {
                GrowHeightChange(old.min, old.max);
                Terrain?.MarkSplatRegion(old.min, old.max);
            }

            if (TryGetStampRegion(stamp, out var min, out var max))
            {
                GrowHeightChange(min, max);
                Terrain?.MarkSplatRegion(min, max);
                _stampRegions[stamp] = (min, max);
            }
            else
            {
                _heightChangeAll = true;
                Terrain?.MarkForUpdate(TerrainDirtyFlags.SplatPack);
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
        /// Height stamps rebake everything: the splat and deco stamps' height/slope filters depend on them. The
        /// splat bake redoes only the stamp's region (where it was and where it is), the height readback tells PCG
        /// and surface meshes which region changed.
        /// Splat and deco stamps leave the heights alone and skip the height path (on Eldermoor's 4097² map a height
        /// bake and its readback cost ~50 ms of GPU and ~200 ms of CPU per edit; until 2026-10-26 every paint edit
        /// paid it to have PCG hear of the region). A local splat stamp rebakes the paint of its region and tells
        /// PCG at once (TerrainPaintChanged: ExcludeStamps tests the stamps' shapes on the CPU, not the bake); a local
        /// deco stamp rebakes the ground cover. Global ones have no region: everywhere, and no message, which would
        /// regenerate every PCG component on the terrain for a change none of them can see.
        /// </summary>
        private void RequestRebake(TerrainStamp stamp)
        {
            RebakeRequestCount++;

            // A hole changes nothing the bakes on the GPU produce; it announces its own region (RebakeHoles)
            if (stamp is HoleStamp)
            {
                Terrain?.MarkForUpdate(TerrainDirtyFlags.HoleBake);
                return;
            }

            if (stamp is CoverageStamp { IsGlobal: true })
            {
                Terrain?.MarkForUpdate(stamp is DecoStamp
                    ? TerrainDirtyFlags.DecoPrepass
                    : TerrainDirtyFlags.SplatPack | TerrainDirtyFlags.AlbedoBake | TerrainDirtyFlags.DecoPrepass);
                return;
            }

            if (stamp is DecoStamp)
            {
                Terrain?.MarkForUpdate(TerrainDirtyFlags.DecoPrepass);
                return;
            }

            if (stamp is SplatStamp)
            {
                var change = MarkPaintRegion(stamp);
                Terrain?.MarkForUpdate(TerrainDirtyFlags.AlbedoBake | TerrainDirtyFlags.DecoPrepass);
                MessageDispatcher.Send(EngineMsg.TerrainPaintChanged, change);
                return;
            }

            MarkStampRegion(stamp);     // the splat rebake, of the stamp's region
            Terrain?.MarkForUpdate(
                TerrainDirtyFlags.HeightBake |
                TerrainDirtyFlags.AlbedoBake |
                TerrainDirtyFlags.DecoPrepass);
        }

        /// <summary>
        /// Where a changed local splat stamp was and is: the splat bake's region, and the region the paint message
        /// gives (everywhere when the stamp has none, e.g. a spline that lost its points).
        /// </summary>
        private TerrainHeightsChange MarkPaintRegion(TerrainStamp stamp)
        {
            var min = new Vector2(float.MaxValue);
            var max = new Vector2(float.MinValue);
            if (_stampRegions.TryGetValue(stamp, out var old))
            {
                min = Vector2.Min(min, old.min);
                max = Vector2.Max(max, old.max);
            }

            if (TryGetStampRegion(stamp, out var newMin, out var newMax))
            {
                min = Vector2.Min(min, newMin);
                max = Vector2.Max(max, newMax);
                _stampRegions[stamp] = (newMin, newMax);
                Terrain?.MarkSplatRegion(min, max);
                return new TerrainHeightsChange(Terrain, false, min, max);
            }

            _stampRegions.Remove(stamp);
            Terrain?.MarkForUpdate(TerrainDirtyFlags.SplatPack);
            return new TerrainHeightsChange(Terrain, true, min, max);
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

            // Round height stamps on its children are confined to it (HeightStamp.MaskSpline), or were until this edit
            // opened the spline
            if (spline.Transform == null) return;
            foreach (var child in spline.Transform)
            {
                if (child.Entity == null) continue;
                foreach (var component in child.Entity.Components)
                    if (component is HeightStamp { IsGlobal: false, IsSplineMode: false } height)
                        RequestRebake(height);
            }
        }
    }
}
