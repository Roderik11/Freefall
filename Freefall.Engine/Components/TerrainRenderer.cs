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
        // The quadtree's depth: Terrain.QuadtreeDepth, from the terrain's size and MetresPerVertex (EnsureQuadtreeDepth).
        // Not a setting since 2026-11-07 (TERRAIN_PAGES_PLAN.md stage 1); scenes that still hold one are not read.
        private int MaxDepth = 7;
        private bool _gridWarned;
        private bool _poolWarned;
        // The walk packs a node as (depth << 28) | its index in the level (≤ 14), and the split flags take 4 bytes a node of
        // the whole tree: 22 MB a frame at depth 11 (32 km at 0.5 m), 4 times that a level deeper
        private const int MaxQuadtreeDepth = 12;
        private int MaxPatches = 32768;
        private const int MinPatches = 32768;
        // Entries in each of the walk's lists: a level's nodes, the leaves (terrain_quadtree.hlsl CSWalk)
        private const int WalkCapacity = 32768;

        private Vector4[] _layerTiling = new Vector4[32];

        // The bindless heap as SetDescriptorHeaps takes it, made once instead of per dispatch
        private ID3D12DescriptorHeap[]? _srvHeaps;
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
        private int _kWalk, _kEmitLeaves, _kBuildMinMaxMip, _kBuildDrawArgs, _kEmitLeavesShadow;
        private bool _computeInitialized;
        private bool _firstDispatch = true; // skip Hi-Z on first frame (no valid _previousFrameViewProjection yet)

        // GPU output buffers — per-frame (GraphicsBuffer with auto-managed SRV/UAV/state)
        private GraphicsBuffer[] _descriptorBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _sphereBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _subbatchIdBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _terrainDataBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _counterBuffers = new GraphicsBuffer[FrameCount];
        private GraphicsBuffer[] _splitFlagsBuffers = new GraphicsBuffer[FrameCount];

        // The quadtree's walk (terrain_quadtree.hlsl CSWalk): its lists and its leaves, which the shadow pass takes too
        private GraphicsBuffer[] _walkBuffers = new GraphicsBuffer[FrameCount];
        // Split flags hold the tag of the evaluation that split a node (MarkSplit), so they are cleared only when made
        private readonly bool[] _splitFlagsFresh = new bool[FrameCount];
        private uint _quadtreeTag;
        private readonly uint[] _frameTags = new uint[FrameCount];
        // The main and shadow patch counts of each frame, read back when its slot comes round again
        private readonly ID3D12Resource?[] _countReadbacks = new ID3D12Resource?[FrameCount];
        private readonly IntPtr[] _countReadbackPtrs = new IntPtr[FrameCount];

        /// <summary>Terrain patches drawn in the last frame read back (a few frames old).</summary>
        [Reflection.DontSerialize]
        public uint Patches { get; private set; }

        /// <summary>Terrain shadow entries (patch × cascade) drawn in the last frame read back.</summary>
        [Reflection.DontSerialize]
        public uint ShadowPatches { get; private set; }

        /// <summary>
        /// This frame's <c>TerrainPagesParams</c> for this terrain (terrain_pages.hlsli <c>LoadTerrain</c>): its bindless
        /// index, 0 while its heights are not in a pool yet. The scene's terrain goes to every shader as
        /// Graphics.Atmosphere.TerrainIdx; this terrain's own passes use their own.
        /// </summary>
        public uint GpuTerrainIdx()
        {
            if (_baker == null || Terrain == null || Transform == null) return 0;
            // The paint pages ride along (terrain_paint.hlsli LoadPaint): 0 while there is no paint
            uint paintIdx = _splatPlan != null && _layerPalette.Count > 0 ? _paintPool.WriteParams(_layerPalette.Count) : 0u;
            return _baker.Pool.WriteParams(Transform.WorldPosition, Terrain.TerrainSize, Terrain.MaxHeight,
                                           (uint)TerrainBaker.HeightVersion, paintIdx);
        }
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
            public uint  MaxPatches;   public uint  TerrainPagesIdx;      public uint  FrameTag, WalkCapacity;
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

        // The layers' own textures by bindless index per channel (their weights: the paint pages, _paintPool)
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
        private int _kSpawnInstances, _kBuildDecoDrawArgs, _kBinMeshInstances;
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

        // ───── Decoration pages (TERRAIN_PAGES_PLAN.md 5c) ─────────────────
        // The top 8 decorators per metre as pages in a pool, baked from the deco stamps as pages arrive round the camera or
        // change (EngineSettings.TerrainDecoPages; 0 = every level 0 page, the whole map as before 5c). The decorations'
        // normals come from the height pages in the spawn (the baked normal map went with 5c).
        private readonly Graphics.DecoPool _decoPool = new();
        private Graphics.TerrainMapResidency? _decoResidency;
        private TerrainBaker.CoveragePlan? _decoPlan;             // the stamps every decoration page bakes from
        private bool _decoPoolWarned;
        private static readonly ProfilerMarker _decoResidencyMarker = ProfilerMarker.Get("Terrain deco residency");

        private Graphics.TerrainMapResidency DecoResidency => _decoResidency ??= new(_decoPool);

        /// <summary>This frame's TerrainDecoParams (terrain_deco.hlsli), 0 while there are no decoration pages.</summary>
        private uint DecoParamsIdx() => _decoPlan != null ? _decoPool.WriteParams() : 0u;

        // ───── Baked Terrain Albedo ───────────────────────────────────────
        private ComputeShader? _albedoBakeCS;
        private ID3D12Resource? _bakedAlbedoTex;     // RGBA8, 256×256
        private uint _bakedAlbedoUAV;
        private uint _bakedAlbedoSRV;
        private GraphicsBuffer? _tilingBuffer;       // StructuredBuffer<float4>, 32 entries
        private const int BakedAlbedoSize = 256;

        // ───── Palette ──────────────────────────────────────────────────
        // What this terrain renders is derived from the stamps in scope, never stored: the layers its
        // splat stamps and biome stamps' rules reference (one channel of the packed control array each, in bake
        // order) and the decorators its deco stamps and biome rules add (one slot of the decoration control texture each).
        private List<TerrainLayer> _layerPalette = new();
        private List<TerrainDecorator> _decoratorPalette = new();
        private List<SplatEntry> _splatEntries = new();   // in bake order, collected with the palette (CoverageEntries)
        private List<DecoEntry> _decoEntries = new();
        private readonly List<string> _layerWarnings = new();
        private readonly List<string> _decoWarnings = new();

        /// <summary>The layers this terrain renders, by control-array channel.</summary>
        [Browsable(false)]
        public IReadOnlyList<TerrainLayer> LayerPalette => _layerPalette;

        /// <summary>The decorators this terrain renders, by control-texture slot.</summary>
        [Browsable(false)]
        public IReadOnlyList<TerrainDecorator> DecoratorPalette => _decoratorPalette;

        /// <summary>
        /// The decorators' far colours (CS_DecoratorTints) as a bindless SRV: two float4 per palette slot, the colour
        /// of its plants seen from afar with their strength, and its clumps. What the ground under a decorator takes,
        /// and a field's far canopy (CropCanopy). 0 until the decorator buffers are built.
        /// </summary>
        [Browsable(false)]
        public uint DecoratorTintSrv => _decoratorTintBuffer?.SrvIndex ?? 0u;

        /// <summary>A decorator's slot in <see cref="DecoratorPalette"/> (its group of the tint buffer); -1 if it grows nowhere here.</summary>
        public int DecoratorSlotOf(TerrainDecorator decorator) => _decoratorPalette.IndexOf(decorator);

        /// <summary>
        /// The colours the decorators' variants are drawn with (CS_BladeColors) as a bindless SRV: six float4 (rgb,
        /// coverage) per variant slot, four height bands root → tip, the foliage and the bloom. Read in vertex, mesh and
        /// compute stages only (it rests in NonPixelShaderResource). 0 until built.
        /// </summary>
        [Browsable(false)]
        public uint BladeColorsSrv => _bladeColorBuffer?.SrvIndex ?? 0u;

        /// <summary>A variant's slot in the decorator buffers (its colours in <see cref="BladeColorsSrv"/>); -1 if not drawn here.</summary>
        public int VariantSlotOf(DecoratorVariant variant) => _variantSlots.TryGetValue(variant, out int slot) ? slot : -1;
        private readonly Dictionary<DecoratorVariant, int> _variantSlots = new();

        /// <summary>
        /// After the decorator buffers were rebuilt: the palette, a decorator's variants or the terrain's decoration
        /// settings changed, and the tint buffer may be a new one. What reads them (CropCanopy) takes them again.
        /// </summary>
        public static event Action<TerrainRenderer>? DecoratorsRebuilt;

        /// <summary>Problems found while resolving the palette (too many layers, stamps that cannot apply).</summary>
        [Browsable(false)]
        public IReadOnlyList<string> PaletteWarnings => _layerWarnings.Concat(_decoWarnings).ToList();

        // ───── Height Bake (GPU layer compositor) ─────────────────────────
        private bool _needHeightFieldReadback;

        // ───── Paint pages (TERRAIN_PAGES_PLAN.md 5b) ───────────────────────
        // The layer weights as pages in a pool, baked from the splat stamps as pages arrive round the camera or change
        // (EngineSettings.TerrainPaintPages; 0 = every level 0 page, the whole map at full paint as before 5b).
        private readonly Graphics.PaintPool _paintPool = new();
        private Graphics.TerrainMapResidency? _paintResidency;
        private TerrainBaker.CoveragePlan? _splatPlan;            // the stamps every paint page bakes from
        private List<TerrainLayer> _paintPalette = new();         // the palette _splatPlan was laid out for
        private bool _paintPoolWarned;
        private static readonly ProfilerMarker _paintResidencyMarker = ProfilerMarker.Get("Terrain paint residency");

        private Graphics.TerrainMapResidency PaintResidency => _paintResidency ??= new(_paintPool);

        // Per slot: a serial bumped each time a bake of it is put on a list, and the palette it was baked with (what its
        // channels mean), so GetLayer's copies know when they are old
        private long[] _paintSlotSerial = Array.Empty<long>();
        private List<TerrainLayer>?[] _paintSlotPalette = Array.Empty<List<TerrainLayer>?>();
        private long _paintSerial;

        // ───── Top layers (which layer is where, for GetLayer) ──────────────
        // Per resident paint page asked about: its top layer (the channel that shows most per texel), read back from the
        // pool once somebody has asked about a place on it and its bake has gone to the GPU.
        private sealed class TopPage
        {
            public required byte[] Layers;               // 256 x 256, row by row along +Z
            public required List<TerrainLayer> Palette;  // what the channels mean
            public int Slot;
            public long Serial;
        }
        private readonly Dictionary<(int level, int px, int py), TopPage> _topPages = new();
        private readonly HashSet<(int level, int px, int py)> _topWanted = new();
        private const int TopPagesPerFrame = 4;

        // ───── Holes (HoleStamp) ───────────────────────────────────────────
        // The mask itself is in Terrain.Pages, baked on the CPU (TerrainBaker.BakeHoles) because the PhysX height
        // field, the navmesh and PCG need it there; this is its copy for the shaders (terrain_holes.hlsli).
        private readonly Graphics.HolePool _holePool = new();   // the pages a hole cuts (since 2026-11-08; one flat R8 map before)
        private List<TerrainBaker.HoleRect>? _holeRects;   // what the last hole bake cut; null = not known (mask from the cache)

        /// <summary>The hole mask the shaders read (bindless SRV), 0 when nothing opens the terrain.</summary>
        private uint HoleSrv() => Terrain is { HasHoles: true } ? _holePool.WriteParams() : 0u;

        // ───── Lifecycle ──────────────────────────────────────────────────

        protected override void Awake()
        {
            // GPU resource init — matches TerrainGPU.Awake() exactly.
            // Called before YAML properties are applied, so the Terrain is not known yet: the quadtree's buffers are made
            // for depth 7 and made again by EnsureQuadtreeDepth when the terrain asks for another; CreateHeightRangePyramid
            // doesn't read the heightmap — BuildHeightRangePyramid is deferred to the first DispatchQuadtreeEval.
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
                device?.DeferDispose(_walkBuffers[i]);
                device?.DeferDispose(_countReadbacks[i]);
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

            // ── Decoration pages ──
            if (device != null) _decoPool.Dispose();
            _decoResidency = null;
            _decoPlan = null;

            // ── Baked albedo ──
            device?.DeferDispose(_bakedAlbedoTex);
            if (_bakedAlbedoUAV != 0) device?.DeferReleaseBindlessIndex(_bakedAlbedoUAV);
            if (_bakedAlbedoSRV != 0) device?.DeferReleaseBindlessIndex(_bakedAlbedoSRV);
            device?.DeferDispose(_tilingBuffer);
            device?.DeferDispose(_plantTintBuffer);

            // ── Paint pages and the top layers read from them ──
            if (device != null) _paintPool.Dispose();
            _paintResidency = null;
            _splatPlan = null;
            _topPages.Clear();
            _topWanted.Clear();

            // ── Hole mask (the CPU mask stays with the Terrain asset, like its heights) ──
            if (device != null) _holePool.Dispose();

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

            EnsureQuadtreeDepth(Math.Min(Terrain.QuadtreeDepth, MaxQuadtreeDepth));

            // The pool's layout: every level 0 page, or the windows round the camera (EngineSettings.TerrainDetailRadius,
            // TERRAIN_PAGES_PLAN.md 5a)
            int poolCapacity = Graphics.TerrainHeightResidency.CapacityFor(Terrain.EffectiveHeightmapResolution, Terrain.VertexSpacing,
                Engine.Settings.TerrainDetailRadius);
            _baker.PoolCapacity = poolCapacity;

            // With every page resident a pool past D3D12's array cap cannot be made (16 km at 0.5 m per vertex needs 4,096
            // slices), so nothing is drawn rather than the device failing; a detail radius brings it down to the windows
            int slices = poolCapacity == 0
                ? Graphics.HeightPool.SlicesFor(Terrain.EffectiveHeightmapResolution)
                : (poolCapacity + Graphics.HeightPool.ScratchSlots + 3) / 4;
            if (slices > Graphics.HeightPool.MaxSlices)
            {
                if (!_poolWarned)
                {
                    _poolWarned = true;
                    Debug.LogError("[TerrainRenderer]", $"{Terrain.Name}: {Terrain.TerrainSize.X} m at {Terrain.VertexSpacing} m per vertex needs " +
                                   $"{slices} slices of height pages, past D3D12's {Graphics.HeightPool.MaxSlices}: not drawn. " +
                                   "Set a TerrainDetailRadius (the pages stream round the camera) or a coarser MetresPerVertex.");
                }
                return;
            }

            if (!Terrain.GridIsExact && !_gridWarned)
            {
                _gridWarned = true;
                Debug.LogWarning("[TerrainRenderer]", $"{Terrain.Name}: {Terrain.TerrainSize.X} x {Terrain.TerrainSize.Y} m is not a square of " +
                                 $"{Assets.Terrain.PatchQuads} x {Terrain.VertexSpacing} m x a power of two: the vertices miss the height samples");
            }

            // The heights live in the terrain asset's pool, which outlives this renderer (a reload keeps them)
            _baker.AttachPool(Terrain);

            // While a load wakes the scene its stamps are still arriving: the first bakes wait for the load's terrain
            // step, so they see every stamp. Before, a bake during the wake-up cut the holes of the stamps awake so far
            // (none, often), then again with all of them, and the editor took the half-cut mask for a change to save.
            if (SceneManager.Loading is { Current: < SceneLoad.Step.Terrain }) return;

            int frameIndex = Engine.FrameIndex % FrameCount;

            // The saved heights, read from the pages file at load, go up into the pool instead of a bake
            if (Terrain.PendingPagesUpload)
            {
                using var _ = _uploadMarker.Auto();
                Terrain.PendingPagesUpload = false; // consumed

                // The loader keeps only pages of the terrain's own density
                if (FillPoolFromCpu())
                {
                    Terrain.ConsumeFlags(TerrainDirtyFlags.HeightBake); // loaded from the file, skip bake
                    Terrain.MarkForUpdate(TerrainDirtyFlags.AlbedoBake);
                    _needHeightFieldReadback = true;
                }
                else
                    Terrain.MarkForUpdate(TerrainDirtyFlags.HeightBake);
            }
            // The detail radius changed: a pool of the new layout, filled from the CPU store (a bake owing pages to it bakes
            // again: what the old pool owed is gone with it)
            else if (_baker.Pool.Texture != null && !_baker.Pool.Matches(Terrain.EffectiveHeightmapResolution, poolCapacity))
            {
                bool owed = _baker.OwedPages > 0;
                if (!FillPoolFromCpu() || owed)
                    Terrain.MarkForUpdate(TerrainDirtyFlags.HeightBake);
            }

            // Streaming: the windows follow the camera, a few pages a frame, on this frame's list before its bake; a pool a
            // bake made with nothing in it is filled once the heights are read back
            if (poolCapacity != 0 && _baker.Pool.Texture != null
                && Terrain.Pages is { HasHeights: true } cpuPages && cpuPages.Resolution == _baker.Pool.Resolution)
            {
                if (!_baker.Pool.Filled)
                    FillPoolFromCpu();
                else
                {
                    using (_residencyMarker.Auto())
                        Residency.Update(cpuPages, CameraLocal(), Terrain.VertexSpacing, Engine.Settings.TerrainDetailRadius,
                            Engine.Settings.TerrainPageUploadsPerFrame);
                    if (Residency.HasWork)
                    {
                        var residency = Residency;
                        CommandBuffer.Enqueue(RenderPass.Opaque, list => residency.Record(list));
                    }
                }
            }

            // The Terrain asset is shared across scene loads, so its height pool may hold another scene's
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
            (int px0, int py0, int px1, int py1)? heightPages = null;
            if (heightBake)
            {
                // A bake nobody gave a region for (first bake, terrain settings) may change anything
                if (!_heightRegionMarked) { _heightChangeAll = true; _heightBakeAll = true; }
                _heightRegionMarked = false;

                // The pages where the stamps changed since the last bake was laid out, or all of them
                if (heightPlan != null && !_heightBakeAll && _heightBakeMax.X >= _heightBakeMin.X)
                    heightPages = heightPlan.PagesOf(_heightBakeMin, _heightBakeMax);
                _heightBakeAll = false;
                _heightBakeMin = new Vector2(float.MaxValue);
                _heightBakeMax = new Vector2(float.MinValue);
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
            bool decoAsked = Terrain.ConsumeFlags(TerrainDirtyFlags.DecoPrepass);
            bool decoDirty = decoAsked || splatDirty;
            // Where the decoration pages bake again: the regions local stamps' edits gave (Terrain.MarkDecoRegion), or
            // everywhere
            bool decoLocal = Terrain.TakeDecoRegion(out var decoRegionMin, out var decoRegionMax) && decoAsked;
            // Where the splat bake composites again: the regions local stamps' edits gave, or everywhere. New heights
            // that no stamp asked for (terrain settings) may have moved any stamp's height or slope filter.
            bool splatLocal = Terrain.TakeSplatRegion(out var splatRegionMin, out var splatRegionMax) && splatAsked;

            bool decoEverywhere = false;
            if (splatDirty || decoDirty)
            {
                using var _ = _paletteMarker.Auto();
                var paletteBefore = _layerPalette;
                var decoPaletteBefore = _decoratorPalette;
                if (splatDirty) RefreshLayerPalette();
                if (_layerPalette != paletteBefore) splatLocal = false;   // other channels: every texel changes
                RefreshDecoratorPalette();
                // The decoration pages bake again everywhere for a deco edit, new heights or other layers or decorators;
                // a splat edit alone reaches them through the paint pages it bakes (UpdatePaintPages)
                decoEverywhere = (decoAsked && !decoLocal) || (hasHeightWork && !splatLocal) ||
                                 _layerPalette != paletteBefore || _decoratorPalette != decoPaletteBefore;
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
                var pages = heightPages;
                HeightBakePages = pages is { } p ? (p.px1 - p.px0) * (p.py1 - p.py0) : plan.Pages * plan.Pages;
                System.Threading.Interlocked.Increment(ref _heightBakesPending);
                CommandBuffer.Enqueue(RenderPass.Opaque, (list) =>
                {
                    System.Threading.Interlocked.Decrement(ref renderer._heightBakesPending);
                    PixMarker.Begin(list, "Terrain height bake");
                    baker.BakeHeight(terrain, plan, list, pages);
                    PixMarker.End(list);
                    terrain.HeightmapStampsHash = plan.Hash;
                    _heightRangePyramidBuilt = false; // force rebuild with new heights
                    terrain.MarkForUpdate(TerrainDirtyFlags.AlbedoBake); // re-bake albedo with new terrain shape
                    _needHeightFieldReadback = true; // read the heights back into the CPU store next frame
                });
            }

            // Debounced readback into the CPU store (Terrain.Pages) — only when baking has settled (not while a stamp is dragged)
            if (_needHeightFieldReadback && !Terrain.NeedsUpdate(TerrainDirtyFlags.HeightBake))
                ReadBackHeights();

            // ── Paint: the splat stamps laid out once, baked into the paint pool page by page (TERRAIN_PAGES_PLAN.md 5b) ──
            // An edit marks the pages over its region dirty; new heights or another palette mark every page (they may move
            // any stamp's height or slope filter, or what every channel means). The pages bake below as they arrive or
            // come up dirty.
            if (splatDirty)
            {
                if (_layerPalette.Count > 0)
                {
                    using (_splatPlanMarker.Auto())
                        _splatPlan = _baker.PrepareSplat(Terrain, this, _splatEntries, _layerPalette);
                    if (splatLocal && ReferenceEquals(_paintPalette, _layerPalette))
                    {
                        var origin = Transform?.WorldPosition ?? Vector3.Zero;
                        var size = Terrain.TerrainSize;
                        int res = Terrain.EffectiveSplatmapResolution;
                        int x0 = (int)MathF.Floor((splatRegionMin.X - origin.X) / size.X * res) - 1;
                        int x1 = (int)MathF.Ceiling((splatRegionMax.X - origin.X) / size.X * res) + 1;
                        int z0 = (int)MathF.Floor((splatRegionMin.Y - origin.Z) / size.Y * res) - 1;
                        int z1 = (int)MathF.Ceiling((splatRegionMax.Y - origin.Z) / size.Y * res) + 1;
                        PaintResidency.MarkDirty(Math.Clamp(x0, 0, res), Math.Clamp(z0, 0, res), Math.Clamp(x1, 0, res), Math.Clamp(z1, 0, res));
                    }
                    else
                        PaintResidency.MarkAllDirty();
                    _paintPalette = _layerPalette;   // replaced, never changed, when the layers change
                }
                else
                {
                    // No splat stamp references a layer: nothing to render the surface with
                    _splatPlan = null;
                    _paintPalette = _layerPalette;
                }
            }

            if (_splatPlan != null && _layerPalette.Count > 0)
                UpdatePaintPages();

            // GetLayer's copies of the pages it was asked about, a few a frame, once their bakes have gone to the GPU (not
            // while a stamp is dragged: the page bakes again every frame and the old answer stands)
            if (_topWanted.Count > 0)
                FetchTopPages();

            // ── Decoration pages: the deco stamps laid out once, baked into the decoration pool page by page (5c) ──
            if (decoDirty)
            {
                if (_decoratorPalette.Count > 0)
                {
                    using (_decoPlanMarker.Auto())
                        _decoPlan = _baker.PrepareDeco(Terrain, this, _decoEntries, _decoratorPalette, _layerPalette);
                    if (decoEverywhere) DecoResidency.MarkAllDirty();
                    else if (decoLocal && decoRegionMax.X >= decoRegionMin.X)
                    {
                        var origin = Transform?.WorldPosition ?? Vector3.Zero;
                        var size = Terrain.TerrainSize;
                        MarkDecoOver(new Vector2((decoRegionMin.X - origin.X) / size.X, (decoRegionMin.Y - origin.Z) / size.Y),
                                     new Vector2((decoRegionMax.X - origin.X) / size.X, (decoRegionMax.Y - origin.Z) / size.Y), 0);
                    }
                }
                else
                    _decoPlan = null;
            }
            if (_decoPlan != null && Terrain.DrawDetail)
                UpdateDecoPages();

            // Set shared material params
            material.SetParameter("CameraPos", Camera.Main.Position);
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

            // The heights: this terrain's pages (terrain_pages.hlsli), this frame's parameters
            // (and through them the paint pages, TerrainPagesParams.PaintIdx)
            material.SetTextureIndex("TerrainPages", GpuTerrainIdx());

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
        /// Collect the splat stamps and biome stamps that apply to this terrain into the paint entries, and derive the
        /// layers it renders: each distinct TerrainLayer gets a channel, in the order the entries first use them.
        /// Raises the texture-array rebuild when the set changes.
        /// </summary>
        private void RefreshLayerPalette()
        {
            _layerWarnings.Clear();
            var biomes = TerrainBaker.CollectStamps<BiomeStamp>(this);
            _splatEntries = CoverageEntries.Splat(TerrainBaker.CollectStamps<SplatStamp>(this), biomes);

            var palette = new List<TerrainLayer>();
            var dropped = new HashSet<TerrainLayer>();
            int unassigned = 0, unassignedRules = 0;
            foreach (var entry in _splatEntries)
            {
                var layer = entry.Layer;
                if (layer == null)
                {
                    if (entry.Shape is BiomeStamp) unassignedRules++;
                    else unassigned++;
                    continue;
                }
                if (palette.Contains(layer)) continue;

                if (palette.Count >= TerrainBaker.MaxLayers) dropped.Add(layer);
                else palette.Add(layer);
            }

            if (unassigned > 0)
                _layerWarnings.Add($"{unassigned} splat stamp(s) have no Layer and paint nothing.");
            if (unassignedRules > 0)
                _layerWarnings.Add($"{unassignedRules} biome layer rule(s) have no Layer and paint nothing.");
            int noBiome = biomes.Count(b => b.Biome == null);
            if (noBiome > 0)
                _layerWarnings.Add($"{noBiome} biome stamp(s) have no Biome and place nothing.");
            if (dropped.Count > 0)
                _layerWarnings.Add($"Too many terrain layers: {palette.Count + dropped.Count} are referenced, " +
                                   $"{TerrainBaker.MaxLayers} can be rendered. Not rendered: " +
                                   string.Join(", ", dropped.Select(l => l.Name)));
            WarnAboutStrayGlobals(ComponentCache<SplatStamp>.All, _layerWarnings);
            WarnAboutStrayGlobals(ComponentCache<BiomeStamp>.All, _layerWarnings);
            ReportWarnings(_layerWarnings, ref _lastLayerWarning);

            if (!palette.SequenceEqual(_layerPalette))
            {
                _layerPalette = palette;
                Terrain!.MarkForUpdate(TerrainDirtyFlags.TextureArrays | TerrainDirtyFlags.LayerParams);
            }
        }

        /// <summary>
        /// Collect the deco stamps and biome stamps that apply to this terrain into the ground cover entries, and
        /// derive the decorators it renders: each decorator some entry adds gets a slot. Raises the decorator buffer
        /// rebuild when the set changes.
        /// </summary>
        private void RefreshDecoratorPalette()
        {
            _decoWarnings.Clear();
            _decoEntries = CoverageEntries.Deco(TerrainBaker.CollectStamps<DecoStamp>(this),
                                                TerrainBaker.CollectStamps<BiomeStamp>(this));

            var palette = new List<TerrainDecorator>();
            var dropped = new HashSet<TerrainDecorator>();
            foreach (var entry in _decoEntries)
            {
                // Only an Add entry puts a decorator on the terrain; Multiply merely scales what is there
                var decorator = entry.Decorator;
                if (decorator == null || entry.Op != DecoOp.Add || palette.Contains(decorator)) continue;

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
            else if (msg.Data is Biome biome && PlacesBiome(biome))
            {
                // Its rules are paint and ground cover wherever its stamps are: the palettes again (SplatPack refreshes
                // both), the paint and the ground cover everywhere, since a rule's layer may take another channel
                RebakeRequestCount++;
                Terrain.MarkForUpdate(TerrainDirtyFlags.SplatPack | TerrainDirtyFlags.AlbedoBake | TerrainDirtyFlags.DecoPrepass);
            }
        }

        /// <summary>Whether a biome stamp that applies to this terrain places the biome.</summary>
        private bool PlacesBiome(Biome biome)
        {
            var stamps = ComponentCache<BiomeStamp>.All;
            for (int i = 0; i < stamps.Count; i++)
                if (stamps[i] is { } stamp && stamp.Biome == biome && stamp.AppliesTo(this))
                    return true;
            return false;
        }

        /// <summary>
        /// An asset was reimported in place. The layer arrays hold copies of a texture's old pixels, so a layer
        /// texture changed on disk (a reprocessed height map) never showed until the layer itself was edited; a layer
        /// or decorator file edited on disk is the same as an edit in the inspector.
        /// </summary>
        private void OnAssetReloaded(Message msg)
        {
            if (Terrain == null) return;
            if (msg.Data is TerrainLayer or TerrainDecorator or Biome)
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
        /// This frame's paint pages (TERRAIN_PAGES_PLAN.md 5b): the pool made at the layout EngineSettings.TerrainPaintPages
        /// asks (made again when the map's size, the layers' slices or the layout change, and then baked whole), the pages
        /// round the camera given slots, and those arriving or dirty baked on this frame's list after the heights. Main
        /// thread.
        /// </summary>
        private void UpdatePaintPages()
        {
            int texels = Terrain.EffectiveSplatmapResolution;
            int layerSlices = (_layerPalette.Count + 3) / 4;
            int pagesPerSide = Math.Max(0, Engine.Settings.TerrainPaintPages);
            int capacity = Graphics.TerrainMapResidency.CapacityFor(texels, pagesPerSide);

            if (!_paintPool.Matches(texels, layerSlices, capacity))
            {
                int across = Graphics.PaintPool.PagesAcrossFor(texels);
                int slots = capacity > 0 ? capacity : across * across;
                int slices = Graphics.PaintPool.WeightSlicesFor(slots, layerSlices);
                if (slices > Graphics.HeightPool.MaxSlices)
                {
                    if (!_paintPoolWarned)
                    {
                        _paintPoolWarned = true;
                        Debug.LogError("[TerrainRenderer]", $"{Terrain.Name}: {slots} paint pages of {_layerPalette.Count} layers need {slices} " +
                                       $"array slices, past D3D12's {Graphics.HeightPool.MaxSlices}: not painted. Set TerrainPaintPages " +
                                       "(the pages stream round the camera) or a lower PaintPixelsPerMetre.");
                    }
                    return;
                }
                _paintPool.Ensure(texels, layerSlices, capacity, null);   // the table goes up on the bake's list
                _paintSlotSerial = new long[_paintPool.Capacity];
                _paintSlotPalette = new List<TerrainLayer>?[_paintPool.Capacity];
                _topPages.Clear();
                Debug.Log("[Terrain]", $"'{Terrain.Name}': paint pool {(capacity == 0 ? "of every page" : $"of {capacity} slots, {pagesPerSide} pages a side per level")}, " +
                          $"{across}x{across} pages of {Graphics.PaintPool.PageTexels} texels, {slices} slices");
            }

            IReadOnlyList<(int level, int px, int py, int slot)> bakes;
            using (_paintResidencyMarker.Auto())
                bakes = PaintResidency.Update(CameraTexels(texels), pagesPerSide, Math.Max(1, Engine.Settings.TerrainPaintBakesPerFrame));
            if (bakes.Count == 0 && !_paintPool.Slots.TableDirty) return;

            var pages = new (int level, int px, int py, int slot)[bakes.Count];
            for (int i = 0; i < pages.Length; i++)
            {
                var b = bakes[i];
                pages[i] = b;
                _paintSlotSerial[b.slot] = ++_paintSerial;
                _paintSlotPalette[b.slot] = _layerPalette;
                // A level 0 paint page that arrives: the level 0 decoration pages over its ground bake again from it (their
                // layer filters read the paint, the coarse paint before). An edit marks its own
                // decoration region at every level (Terrain.MarkDecoRegion); marking the decoration pages under every paint
                // page an edit re-baked, coarse ones included (the top one covers the terrain), re-baked every coarse
                // decoration page per edit: 196 decoration pages for 4 moves of a 7.5 m stamp (2026-11-08)
                if (_decoPlan != null && b.level == 0 && !_paintPool.Slots.Written(b.slot))   // arrived: nothing wrote the slot yet
                {
                    float m = Graphics.PaintPool.PageTexels / (float)texels;
                    MarkDecoOver(new Vector2(b.px * m, b.py * m), new Vector2((b.px + 1) * m, (b.py + 1) * m), 0, 0);
                }
            }
            SplatBakeTexels = (long)pages.Length * Graphics.PaintPool.SlotTexels * Graphics.PaintPool.SlotTexels;
            PaintPagesBaked += pages.Length;

            var plan = _splatPlan!;
            var pool = _paintPool;
            var baker = _baker;
            var terrain = Terrain;
            var renderer = this;
            CommandBuffer.Enqueue(RenderPass.Opaque, list =>
            {
                pool.Slots.RecordTable(list);
                if (pages.Length == 0) return;
                // The height bake enqueued before has run by now, so the filters read the new heights
                PixMarker.Begin(list, "Terrain paint bake");
                baker.BakeSplatPages(plan, list, terrain, renderer.GpuTerrainIdx(), pool, pages);
                PixMarker.End(list);
            });
        }

        /// <summary>
        /// This frame's decoration pages (TERRAIN_PAGES_PLAN.md 5c), as the paint's: the pool at the layout
        /// EngineSettings.TerrainDecoPages asks, the pages round the camera given slots, and those arriving or dirty baked on
        /// this frame's list after the paint (whose layer weights their filters read). Main thread.
        /// </summary>
        private void UpdateDecoPages()
        {
            int texels = Terrain.EffectiveDecorationMapResolution;
            int pagesPerSide = Math.Max(0, Engine.Settings.TerrainDecoPages);
            int capacity = Graphics.TerrainMapResidency.CapacityFor(texels, pagesPerSide);

            if (!_decoPool.Matches(texels, capacity))
            {
                int across = Graphics.TerrainMapResidency.PagesAcrossFor(texels);
                int slots = capacity > 0 ? capacity : across * across;
                int slices = Graphics.DecoPool.SlicesFor(slots);
                if (slices > Graphics.HeightPool.MaxSlices)
                {
                    if (!_decoPoolWarned)
                    {
                        _decoPoolWarned = true;
                        Debug.LogError("[TerrainRenderer]", $"{Terrain.Name}: {slots} decoration pages need {slices} array slices, past " +
                                       $"D3D12's {Graphics.HeightPool.MaxSlices}: no decorations. Set TerrainDecoPages (the pages stream " +
                                       "round the camera).");
                    }
                    return;
                }
                _decoPool.Ensure(texels, capacity);   // the table goes up on the bake's list
                Debug.Log("[Terrain]", $"'{Terrain.Name}': decoration pool {(capacity == 0 ? "of every page" : $"of {capacity} slots, {pagesPerSide} pages a side per level")}, " +
                          $"{across}x{across} pages of {Graphics.DecoPool.PageTexels} texels, {slices} slices");
            }

            IReadOnlyList<(int level, int px, int py, int slot)> bakes;
            using (_decoResidencyMarker.Auto())
                bakes = DecoResidency.Update(CameraTexels(texels), pagesPerSide, Math.Max(1, Engine.Settings.TerrainDecoBakesPerFrame));
            if (bakes.Count == 0 && !_decoPool.Slots.TableDirty) return;

            var pages = new (int level, int px, int py, int slot)[bakes.Count];
            for (int i = 0; i < pages.Length; i++) pages[i] = bakes[i];
            if (pages.Length > 0)
                DecoBakeTexels = (long)pages.Length * Graphics.DecoPool.SlotTexels * Graphics.DecoPool.SlotTexels;
            DecoPagesBaked += pages.Length;

            var plan = _decoPlan!;
            var pool = _decoPool;
            var baker = _baker;
            var terrain = Terrain;
            var renderer = this;
            CommandBuffer.Enqueue(RenderPass.Opaque, list =>
            {
                pool.Slots.RecordTable(list);
                if (pages.Length == 0) return;
                PixMarker.Begin(list, "Terrain deco bake");
                baker.BakeDecoPages(plan, list, terrain, renderer.GpuTerrainIdx(), pool, pages);
                PixMarker.End(list);
            });
        }

        // The decoration pages over this ground (terrain UV) bake again, at the levels from minLevel to maxLevel
        private void MarkDecoOver(Vector2 uvMin, Vector2 uvMax, int minLevel, int maxLevel = int.MaxValue)
        {
            if (_decoPool.Control == null) return;
            int texels = _decoPool.Texels;
            int x0 = Math.Clamp((int)MathF.Floor(uvMin.X * texels) - 1, 0, texels);
            int x1 = Math.Clamp((int)MathF.Ceiling(uvMax.X * texels) + 1, 0, texels);
            int z0 = Math.Clamp((int)MathF.Floor(uvMin.Y * texels) - 1, 0, texels);
            int z1 = Math.Clamp((int)MathF.Ceiling(uvMax.Y * texels) + 1, 0, texels);
            DecoResidency.MarkDirty(x0, z0, x1, z1, minLevel, maxLevel);
        }

        // The camera in level 0 texels of a map of this many texels a side (terrain UV × texels)
        private Vector2 CameraTexels(int texels)
        {
            var local = CameraLocal();
            return new Vector2(local.X / Terrain.TerrainSize.X * texels, local.Y / Terrain.TerrainSize.Y * texels);
        }

        // GetLayer's copies of the pages it asked about: read back once the page's last bake has gone to the GPU
        private void FetchTopPages()
        {
            if (_paintPool.TopLayer == null) { _topWanted.Clear(); return; }
            var slots = _paintPool.Slots;
            int fetched = 0;
            List<(int, int, int)>? done = null;
            foreach (var key in _topWanted)
            {
                if (fetched >= TopPagesPerFrame) break;
                int slot = slots.SlotOf(key.level, key.px, key.py);
                if (slot < 0) { (done ??= new()).Add(key); continue; }   // gone: asked about again when it is back
                if (!slots.Settled(slot)) continue;
                long serial = _paintSlotSerial[slot];
                var palette = _paintSlotPalette[slot];
                var layers = TerrainBaker.ReadbackTopPage(_paintPool, slot);
                fetched++;
                (done ??= new()).Add(key);
                if (layers == null || palette == null) continue;
                _topPages[key] = new TopPage { Layers = layers, Palette = palette, Slot = slot, Serial = serial };
            }
            if (done != null)
                foreach (var key in done) _topWanted.Remove(key);

            // Copies of pages that left the pool go (64 KB each; a flight over the map would keep them all)
            if (_topPages.Count > 64)
            {
                List<(int, int, int)>? gone = null;
                foreach (var (key, page) in _topPages)
                    if (slots.SlotOf(key.level, key.px, key.py) != page.Slot) (gone ??= new()).Add(key);
                if (gone != null)
                    foreach (var key in gone) _topPages.Remove(key);
            }
        }

        // ───── Compute Pipeline ───────────────────────────────────────────

        private void InitializeCompute()
        {
            var device = Engine.Device;

            _quadtreeCS = new ComputeShader("terrain_quadtree.hlsl");
            _kWalk             = _quadtreeCS.FindKernel("CSWalk");
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
            _splitFlagsFresh[i] = true;
            // A header of 4 uints and four lists of WalkCapacity entries (terrain_quadtree.hlsl WalkList)
            _walkBuffers[i] = GraphicsBuffer.CreateRaw(4 + 4 * WalkCapacity, uav: true);
            _countReadbacks[i] = Engine.Device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Readback), HeapFlags.None, ResourceDescription.Buffer(8), ResourceStates.CopyDest, null);
            unsafe
            {
                void* pData;
                _countReadbacks[i]!.Map(0, null, &pData);
                _countReadbackPtrs[i] = (IntPtr)pData;
            }
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

            // The counts this slot's last frame drew (the GPU is done with it: this slot's buffers are being reused)
            unsafe
            {
                if (_countReadbackPtrs[frameIndex] != IntPtr.Zero)
                {
                    uint* counts = (uint*)_countReadbackPtrs[frameIndex];
                    Patches = Math.Min(counts[0], (uint)Math.Max(MaxPatches, MinPatches));
                    ShadowPatches = counts[1];
                }
            }

            // This evaluation's tag: a split flag holding it is split (terrain_quadtree.hlsl MarkSplit); never 0
            _quadtreeTag = _quadtreeTag == uint.MaxValue ? 1u : _quadtreeTag + 1u;
            _frameTags[frameIndex] = _quadtreeTag;

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
                TerrainPagesIdx = GpuTerrainIdx(),
                FrameTag = _quadtreeTag,
                WalkCapacity = WalkCapacity,
            };
            UploadBuffer(_terrainParamBuffers[frameIndex], terrainParams);

            // Bind root sig + descriptor heaps + all cbuffers once (before any compute operations)
            commandList.SetDescriptorHeaps(1, _srvHeaps ??= new[] { Engine.Device.SrvHeap });
            commandList.SetComputeRootSignature(device.GlobalRootSignature);
            commandList.SetComputeRootConstantBufferView(1, _frustumPlaneBuffers[frameIndex].GPUVirtualAddress);
            commandList.SetComputeRootConstantBufferView(2, _hizParamBuffers[frameIndex].GPUVirtualAddress);
            commandList.SetComputeRootConstantBufferView(3, _terrainParamBuffers[frameIndex].GPUVirtualAddress);

            // Clear the counter; the split flags only once, when made (they hold a tag, MarkSplit). The counter is put in
            // UnorderedAccess first: the clear would promote it there unseen by its tracked state, and the count's copy
            // below would then name Common as the state it leaves (GBV: ResourceBarrierBeforeAfterMismatch on frame 1)
            _counterBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);
            _counterBuffers[frameIndex].ClearUAV(commandList, new Vortice.Mathematics.Int4(0, 0, 0, 0));
            if (_splitFlagsFresh[frameIndex])
            {
                _splitFlagsBuffers[frameIndex].ClearUAV(commandList, new Vortice.Mathematics.Int4(0, 0, 0, 0));
                _splitFlagsFresh[frameIndex] = false;
            }
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Push constants — bindless indices only (slots 0-14)
            cs.SetBuffer("OutputDescriptorsUAV", _descriptorBuffers[frameIndex]);
            cs.SetBuffer("OutputSpheresUAV", _sphereBuffers[frameIndex]);
            cs.SetBuffer("OutputSubbatchIdsUAV", _subbatchIdBuffers[frameIndex]);
            cs.SetBuffer("OutputTerrainDataUAV", _terrainDataBuffers[frameIndex]);
            cs.SetBuffer("CounterUAV", _counterBuffers[frameIndex]);
            cs.SetBuffer("SplitFlagsUAV", _splitFlagsBuffers[frameIndex]);
            cs.SetBuffer("WalkUAV", _walkBuffers[frameIndex]);
            cs.SetPushConstant("HeightRangeSRV", _heightRangePyramidSRV);

            // Per-kernel overrides for CSBuildDrawArgs
            cs.SetPushConstant(_kBuildDrawArgs, "VertexCount", (uint)_patchMesh.IndexCount);
            cs.SetBuffer(_kBuildDrawArgs, "IndirectArgsUAV", _indirectArgsBuffers[frameIndex]);

            // ── Height range mip pyramid: from the CPU store while the pages stream, else built from the pool ──
            if (_baker.PoolCapacity != 0)
            {
                if (_cpuRangesPending && Terrain?.Pages is { HasHeights: true } rangePages
                    && rangePages.Resolution == Terrain.EffectiveHeightmapResolution)
                {
                    UploadCpuRanges(commandList, rangePages);
                    _cpuRangesPending = false;
                }
                _heightRangePyramidBuilt = true;
            }
            else if (!_heightRangePyramidBuilt)
            {
                StreamingManager.Instance?.Flush();
                Engine.Device.WaitForCopyQueue();

                Debug.Log($"[TerrainRenderer] Pyramid build in QuadtreeEval: TerrainPagesIdx={terrainParams.TerrainPagesIdx} pool filled={_baker?.Pool.Filled}");

                BuildHeightRangePyramid(commandList);
                _heightRangePyramidBuilt = true;

                // Re-bind root sig + cbuffers (mip builder may have changed PSO state)
                commandList.SetDescriptorHeaps(1, _srvHeaps ??= new[] { Engine.Device.SrvHeap });
                commandList.SetComputeRootSignature(device.GlobalRootSignature);
                commandList.SetComputeRootConstantBufferView(1, _frustumPlaneBuffers[frameIndex].GPUVirtualAddress);
                commandList.SetComputeRootConstantBufferView(2, _hizParamBuffers[frameIndex].GPUVirtualAddress);
                commandList.SetComputeRootConstantBufferView(3, _terrainParamBuffers[frameIndex].GPUVirtualAddress);
            }

            // ── Pass 1: the walk from the root into a list of leaves (one group: terrain_quadtree.hlsl CSWalk) ──
            cs.Dispatch(_kWalk, commandList, 1);

            // UAV barrier
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // ── Pass 2: Emit leaves with inline frustum + Hi-Z culling (a thread a leaf; the rest leave at once) ──
            cs.Dispatch(_kEmitLeaves, commandList, (WalkCapacity + 255) / 256);

            // UAV barrier
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // ── Pass 3: Build DrawInstanced indirect args from counter ──
            cs.Dispatch(_kBuildDrawArgs, commandList, 1);

            // UAV barrier
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Transition indirect args buffer for ExecuteIndirect
            _indirectArgsBuffers[frameIndex].Transition(commandList, ResourceStates.IndirectArgument);

            // The patch count, for Patches when this slot comes round again
            if (_countReadbacks[frameIndex] is { } countReadback)
            {
                _counterBuffers[frameIndex].Transition(commandList, ResourceStates.CopySource);
                commandList.CopyBufferRegion(countReadback, 0, _counterBuffers[frameIndex].Native, 0, 4);
                _counterBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);
            }

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
            uint holes = HoleSrv();
            if (holes != 0 && pass == "DepthPrepass") pass = "DepthPrepass_Holes";

            // Apply material PSO and textures (the pass explicitly: the shadow draw changes it)
            Material!.SetPass(pass);
            Material!.Apply(commandList, device);

            // Set topology; the patch mesh's indices, drawn indexed (SV_VertexID is the grid vertex): the post-transform
            // cache shades a vertex the neighbouring triangles share once, not once per corner
            commandList.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleList);
            commandList.IASetIndexBuffer(_patchMesh.IndexBufferView);

            // Set descriptor heap (Material.Apply may have changed it)
            commandList.SetDescriptorHeaps(1, _srvHeaps ??= new[] { Engine.Device.SrvHeap });

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
            // Slot 21: the decoration pages' TerrainDecoParams (ground tint, debug mode 5); slot 22: the tint per decorator
            commandList.SetGraphicsRoot32BitConstant(0, DecoParamsIdx(), 21);
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
            Span<Vector4> localPlanes = stackalloc Vector4[6];
            for (int c = 0; c < cascadeCount; c++)
            {
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
                TerrainPagesIdx = GpuTerrainIdx(),
                // This frame's flags and leaves, from the walk in the opaque pass (DispatchQuadtreeEval)
                FrameTag = _frameTags[frameIndex],
                WalkCapacity = WalkCapacity,
                // A patch casts into a cascade only where a surface the camera sees through it lies behind
                // (receiver_mask_culling.md); the receivers were built from this frame's depth before this action ran
                ShadowReceiversIdx = cascadeVPs != null ? DirectionalLight.CurrentShadowReceiversUav : 0u,
            };
            UploadBuffer(_shadowTerrainParamBuffers[frameIndex], shadowTerrainParams);

            // Clear shadow counter
            commandList.SetDescriptorHeaps(1, _srvHeaps ??= new[] { Engine.Device.SrvHeap });
            commandList.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            _shadowCounterBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);   // as the main counter
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
            cs.SetBuffer(kShadow, "WalkUAV", _walkBuffers[frameIndex]);
            cs.SetPushConstant(kShadow, "HeightRangeSRV", _heightRangePyramidSRV);
            cs.SetPushConstant(kShadow, "CascadeCount", (uint)cascadeCount);
            cs.SetPushConstant(kShadow, "CascadeBufferSRV", _shadowCascadeBufferSrvs[frameIndex]);

            // Dispatch CSEmitLeavesShadow: a thread a leaf of the walk's list
            uint threadGroups = (WalkCapacity + 255) / 256;
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

            // The shadow entry count, for ShadowPatches when this slot comes round again
            if (_countReadbacks[frameIndex] is { } countReadback)
            {
                _shadowCounterBuffers[frameIndex].Transition(commandList, ResourceStates.CopySource);
                commandList.CopyBufferRegion(countReadback, 4, _shadowCounterBuffers[frameIndex].Native, 0, 4);
                _shadowCounterBuffers[frameIndex].Transition(commandList, ResourceStates.UnorderedAccess);
            }

            // Transition shadow output to SRV for VS_Shadow
            _shadowDescriptorBuffers[frameIndex].Transition(commandList, ResourceStates.NonPixelShaderResource);
            _shadowTerrainDataBuffers[frameIndex].Transition(commandList, ResourceStates.NonPixelShaderResource);
            _shadowCascadeIdxBuffers[frameIndex].Transition(commandList, ResourceStates.NonPixelShaderResource);

            // ════════════════════════════════════════════════════════════════
            // Phase 2: Draw terrain shadows (graphics)
            // ════════════════════════════════════════════════════════════════

            // Where a hole opens the terrain it casts no shadow: light falls into the cave mouth
            uint holes = HoleSrv();
            if (holes != 0) Material.SetPass("Shadow_Holes");
            else Material.SetPass(RenderPass.Shadow);
            Material.Apply(commandList, device);

            commandList.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleList);
            commandList.IASetIndexBuffer(_patchMesh.IndexBufferView);                             // drawn indexed, as DrawTerrain
            commandList.SetDescriptorHeaps(1, _srvHeaps ??= new[] { Engine.Device.SrvHeap });
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
            Span<Vector4> planes = stackalloc Vector4[6];
            new Frustum(vpMatrix).GetPlanes(planes);

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
        /// Called during InitializeCompute and again by EnsureQuadtreeDepth.
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

            // A new pyramid holds nothing. While the pages stream it is filled from the CPU store, so every node's ranges
            // are owed. FillPoolFromCpu asked for them only when it filled the pool, and the pool is the terrain asset's:
            // a renderer made over a pool already filled (Play loading the scene the editor had open) never filled it and
            // kept a pyramid of zeros, every node judged at height 0, so the walk chose coarse patches everywhere and a
            // summit was cut off below the player standing on it.
            _cpuRangesAll = true;
            _cpuRangesPending = true;

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
                // Transition this mip to UAV for writing, from where the last build or upload left it
                commandList.ResourceBarrier(new ResourceBarrier(
                    new ResourceTransitionBarrier(_heightRangePyramid,
                        _pyramidState,
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
            _pyramidState = ResourceStates.NonPixelShaderResource;
        }

        // The state every mip of the height-range pyramid is in (all mips move together): Common as made
        private ResourceStates _pyramidState = ResourceStates.Common;

        // ───── Streaming heights (TERRAIN_PAGES_PLAN.md 5a) ──────────────

        private Graphics.TerrainHeightResidency? _residency;
        private static readonly ProfilerMarker _residencyMarker = ProfilerMarker.Get("Terrain height residency");
        private static readonly ProfilerMarker _cpuRangesMarker = ProfilerMarker.Get("Terrain height ranges on the CPU");

        // The residency of the pool the baker holds now (a terrain asset's pool outlives its renderers)
        private Graphics.TerrainHeightResidency Residency =>
            _residency != null && ReferenceEquals(_residency.Pool, _baker.Pool) ? _residency : (_residency = new(_baker.Pool));

        // The camera in terrain-local metres on the ground plane
        private Vector2 CameraLocal()
        {
            var p = (Camera.Main?.Position ?? Vector3.Zero) - (Transform?.WorldPosition ?? Vector3.Zero);
            return new Vector2(p.X, p.Z);
        }

        /// <summary>
        /// The pool made, or made again, at the layout the detail radius asks (<see cref="TerrainBaker.PoolCapacity"/>), and
        /// filled from the CPU store: every page, or the windows round the camera. False when the store has no heights of
        /// the terrain's size.
        /// </summary>
        private bool FillPoolFromCpu()
        {
            if (Terrain.Pages is not { HasHeights: true } cpu || cpu.Resolution != Terrain.EffectiveHeightmapResolution) return false;
            if (_baker.PoolCapacity == 0)
            {
                if (!_baker.UploadPages(cpu)) return false;
            }
            else
            {
                _baker.EnsurePoolNow(cpu.Resolution);
                Residency.FillNow(cpu, CameraLocal(), Terrain.VertexSpacing, Engine.Settings.TerrainDetailRadius);
                if (!_baker.Pool.Filled) return false;
                TerrainBaker.NoteHeightsChanged();
                _cpuRangesAll = true;
                _cpuRangesPending = true;
                Debug.LogAlways($"[Terrain] '{Terrain.Name}': height pages stream, {Residency.UploadedLastFrame} filled round the camera " +
                                $"({_baker.Pool.Capacity} slots, {_baker.Pool.Slices} slices, detail radius {Engine.Settings.TerrainDetailRadius} m)");
            }
            Terrain.HeightmapStampsHash = Terrain.PagesStampsHash;
            _heightRangePyramidBuilt = false;
            return true;
        }

        // The quadtree's height ranges built on the CPU from the CPU store while the pages stream: the GPU build reads every
        // sample through the pool, which away from the camera holds coarse pages only. Per mip, (min, max) per texel.
        private float[][]? _cpuRanges;
        private bool _cpuRangesPending;
        private bool _cpuRangesAll = true;
        private (int x0, int z0, int x1, int z1)? _cpuRangesBox;

        // The heights changed over these samples (x0, z0 inclusive, x1, z1 exclusive): their ranges are built again
        private void MarkCpuRanges(int x0, int z0, int x1, int z1)
        {
            _cpuRangesBox = _cpuRangesBox is { } b
                ? (Math.Min(b.x0, x0), Math.Min(b.z0, z0), Math.Max(b.x1, x1), Math.Max(b.z1, z1))
                : (x0, z0, x1, z1);
            _cpuRangesPending = true;
        }

        /// <summary>
        /// The height-range pyramid from the CPU store: mip 0 one texel per finest node, the min and max (× MaxHeight) of
        /// the samples its patch's vertices read, as CSBuildMinMaxMip takes them (samples floor(uvMin · cells) ..
        /// ceil(uvMax · cells)); the mips above min of mins and max of maxes. Only the nodes over the changed samples when
        /// there was a build before. Copied up on this list.
        /// </summary>
        private unsafe void UploadCpuRanges(ID3D12GraphicsCommandList cmd, TerrainPages pages)
        {
            using var scope = _cpuRangesMarker.Auto();
            int size = 1 << MaxDepth;
            int mips = _heightRangeMipCount;
            if (_cpuRanges == null || _cpuRanges.Length != mips || _cpuRanges[0].Length != size * size * 2)
            {
                _cpuRanges = new float[mips][];
                for (int m = 0; m < mips; m++)
                {
                    int w = Math.Max(1, size >> m);
                    _cpuRanges[m] = new float[w * w * 2];
                }
                _cpuRangesAll = true;
            }

            long cells = pages.Resolution - 1;
            float maxH = Terrain.MaxHeight;
            int lx0 = 0, lz0 = 0, lx1 = size - 1, lz1 = size - 1;
            if (!_cpuRangesAll && _cpuRangesBox is { } b)
            {
                lx0 = Math.Clamp((int)(b.x0 * (long)size / cells) - 1, 0, size - 1);
                lz0 = Math.Clamp((int)(b.z0 * (long)size / cells) - 1, 0, size - 1);
                lx1 = Math.Clamp((int)((b.x1 * (long)size + cells - 1) / cells) + 1, 0, size - 1);
                lz1 = Math.Clamp((int)((b.z1 * (long)size + cells - 1) / cells) + 1, 0, size - 1);
            }
            var mip0 = _cpuRanges[0];
            System.Threading.Tasks.Parallel.For(lz0, lz1 + 1, z =>
            {
                int sz0 = (int)(z * cells / size), sz1 = (int)(((z + 1) * cells + size - 1) / size);
                for (int x = lx0; x <= lx1; x++)
                {
                    int sx0 = (int)(x * cells / size), sx1 = (int)(((x + 1) * cells + size - 1) / size);
                    var (lo, hi) = pages.MinMax(sx0, sz0, sx1, sz1);
                    mip0[(z * size + x) * 2] = lo / 65535.0f * maxH;
                    mip0[(z * size + x) * 2 + 1] = hi / 65535.0f * maxH;
                }
            });
            for (int m = 1; m < mips; m++)
            {
                var src = _cpuRanges[m - 1];
                var dst = _cpuRanges[m];
                int sw = Math.Max(1, size >> (m - 1)), w = Math.Max(1, size >> m);
                for (int z = 0; z < w; z++)
                    for (int x = 0; x < w; x++)
                    {
                        int a = ((2 * z) * sw + 2 * x) * 2, c = ((2 * z + 1) * sw + 2 * x) * 2;
                        dst[(z * w + x) * 2] = MathF.Min(MathF.Min(src[a], src[a + 2]), MathF.Min(src[c], src[c + 2]));
                        dst[(z * w + x) * 2 + 1] = MathF.Max(MathF.Max(src[a + 1], src[a + 3]), MathF.Max(src[c + 1], src[c + 3]));
                    }
            }
            _cpuRangesAll = false;
            _cpuRangesBox = null;

            // Every mip up, each at a 512-byte boundary with 256-byte rows
            var offsets = new long[mips];
            var pitches = new int[mips];
            long total = 0;
            for (int m = 0; m < mips; m++)
            {
                int w = Math.Max(1, size >> m);
                pitches[m] = (w * 8 + 255) & ~255;
                offsets[m] = total;
                total += ((long)pitches[m] * w + 511) & ~511L;
            }
            var device = Engine.Device;
            var upload = device.NativeDevice.CreateCommittedResource(new HeapProperties(HeapType.Upload), HeapFlags.None,
                ResourceDescription.Buffer((ulong)total), ResourceStates.GenericRead, null);
            void* data;
            upload.Map(0, null, &data);
            for (int m = 0; m < mips; m++)
            {
                int w = Math.Max(1, size >> m);
                for (int z = 0; z < w; z++)
                    fixed (float* row = &_cpuRanges[m][z * w * 2])
                        Buffer.MemoryCopy(row, (byte*)data + offsets[m] + (long)z * pitches[m], w * 8, w * 8);
            }
            upload.Unmap(0);
            for (int m = 0; m < mips; m++)
            {
                int w = Math.Max(1, size >> m);
                cmd.ResourceBarrier(new ResourceBarrier(new ResourceTransitionBarrier(_heightRangePyramid, _pyramidState, ResourceStates.CopyDest, (uint)m)));
                cmd.CopyTextureRegion(new TextureCopyLocation(_heightRangePyramid, (uint)m), 0, 0, 0,
                    new TextureCopyLocation(upload, new PlacedSubresourceFootPrint
                    {
                        Offset = (ulong)offsets[m],
                        Footprint = new SubresourceFootPrint(Format.R32G32_Float, (uint)w, (uint)w, 1, (uint)pitches[m])
                    }), null);
                cmd.ResourceBarrier(new ResourceBarrier(new ResourceTransitionBarrier(_heightRangePyramid, ResourceStates.CopyDest,
                    ResourceStates.NonPixelShaderResource, (uint)m)));
            }
            _pyramidState = ResourceStates.NonPixelShaderResource;
            device.DeferDispose(upload);   // read when the list runs
        }

        // ───── Helpers ────────────────────────────────────────────────────

        /// <summary>
        /// Makes the quadtree's depth-sized resources (the split flags, the height-range pyramid) for this depth when they
        /// were made for another: Awake makes them before the Terrain is known, and the depth comes from the terrain.
        /// </summary>
        private void EnsureQuadtreeDepth(int depth)
        {
            if (depth == MaxDepth) return;
            var device = Engine.Device;

            for (int i = 0; i < FrameCount; i++)
                device.DeferDispose(_splitFlagsBuffers[i]);
            device.DeferDispose(_heightRangePyramid);
            foreach (var idx in _heightRangeMipUAVs) if (idx != 0) device.DeferReleaseBindlessIndex(idx);
            foreach (var idx in _heightRangeMipSRVs) if (idx != 0) device.DeferReleaseBindlessIndex(idx);
            if (_heightRangePyramidSRV != 0) device.DeferReleaseBindlessIndex(_heightRangePyramidSRV);

            MaxDepth = depth;
            _totalNodes = CalculateTotalNodes(depth);
            for (int i = 0; i < FrameCount; i++)
            {
                _splitFlagsBuffers[i] = GraphicsBuffer.CreateRaw(_totalNodes, uav: true, clearable: true);
                _splitFlagsFresh[i] = true;
            }
            CreateHeightRangePyramid(device);
            _heightRangePyramidBuilt = false;
            _pyramidState = ResourceStates.Common;
            Debug.Log($"[TerrainRenderer] Quadtree depth {depth} ({Terrain?.Name}): {_totalNodes} nodes");
        }

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

            // Also refresh tiling since channels may have moved (the weights are the paint pages, baked on SplatPack)
            UpdateLayerParams();
        }

        // ───── Holes ───────────────────────────────────────────────────────

        /// <summary>
        /// Re-cut the hole stamps into the terrain's CPU store (Terrain.Pages: a new store with the hole pages they reach
        /// made anew) and bring the shaders' copy up to date. Only the samples the stamps reach now or reached at the
        /// last bake are touched, so dragging a hole stamp costs its own area, not the terrain's. Main thread.
        /// </summary>
        private void RebakeHoles()
        {
            var terrain = Terrain!;
            var plan = TerrainBaker.PrepareHoles(terrain, this);
            int res = plan.Resolution;
            var pages = terrain.Pages;

            // Heights of another size are still to be replaced by a bake at this one (the density changed): the holes
            // are cut into the store that bake's readback makes
            if (pages is { HasHeights: true } && pages.Resolution != res && HeightsPending)
            {
                terrain.MarkForUpdate(TerrainDirtyFlags.HoleBake);
                return;
            }

            bool samePages = pages != null && pages.Resolution == res;
            bool hadHoles = samePages && pages!.HasHoles;
            // Where stamps cut at the last bake; null = not known (first bake: the holes came from the cache or
            // another scene). No holes after a bake of this renderer = nothing was cut.
            var cutBefore = _holeRects == null ? null
                : pages is not { HasHoles: true } ? new List<TerrainBaker.HoleRect>()
                : samePages ? _holeRects : null;

            if (!plan.HasHoles)
            {
                if (pages is { HasHoles: true })
                {
                    var none = pages.Edit();
                    none.ClearHoles();
                    terrain.SetPages(none.Done());
                    terrain.CheckMapsAgainstSaved();
                    if (cutBefore != null) _holeChanges.AddRange(cutBefore);
                }
                _holePool.Clear();
                _holeRects = new List<TerrainBaker.HoleRect>();
                return;
            }

            var cutNow = plan.Rects.ToList();
            TerrainPages baked;
            if (!hadHoles)
            {
                // Nothing cut at this size yet (a store of another size has no heights here: the readback brings them)
                var basis = samePages ? pages! : TerrainPages.Empty(res);
                baked = TerrainBaker.BakeHoles(plan, basis, Array.Empty<TerrainBaker.HoleRect>());
            }
            else
            {
                // Holes of unknown origin (the saved ones, or another scene's) are cleared whole once
                baked = TerrainBaker.BakeHoles(plan, pages!, cutBefore ?? [new TerrainBaker.HoleRect(0, 0, res - 1, res - 1)]);
            }
            terrain.SetPages(baked);
            terrain.CheckMapsAgainstSaved();
            _holeRects = cutNow;

            // What changed: where stamps cut before and where they cut now (all of it when that is not known)
            var changed = cutBefore != null ? cutBefore.Concat(cutNow).Distinct().ToList() : null;
            // The shaders' hole pages: those the changed rects reach go up again (all of them when that is not known, or
            // nothing was cut before), new cut pages get a slice, pages left without holes leave
            HashSet<int>? changedPages = null;
            if (changed != null && hadHoles)
            {
                changedPages = new HashSet<int>();
                foreach (var r in changed)
                {
                    var (px0, pz0, px1, pz1) = baked.PagesOf(r.X0, r.Z0, r.X1 + 1, r.Z1 + 1);
                    for (int pz = pz0; pz < pz1; pz++)
                        for (int px = px0; px < px1; px++)
                            changedPages.Add(pz * baked.PagesAcross + px);
                }
            }
            _holePool.Sync(baked, changedPages);
            if (changed != null)
                foreach (var rect in changed)
                    if (!_holeChanges.Contains(rect)) _holeChanges.Add(rect);
        }

        // Rects whose holes changed and whose listeners have not heard of it yet (sent once the bakes settle)
        private readonly List<TerrainBaker.HoleRect> _holeChanges = new();

        /// <summary>Diagnostics: what the last Check Hole Pages found.</summary>
        [Reflection.DontSerialize]
        public string HolePagesCheck { get; private set; } = "";

        /// <summary>Diagnostics: pages a hole cuts, in the shaders' hole pool.</summary>
        [Reflection.DontSerialize]
        public int HolePages => _holePool.Count;

        /// <summary>
        /// The hole pages (Graphics.HolePool, terrain_holes.hlsli) against the whole-map R8 texture there was before
        /// 2026-11-08, built from the CPU store for the check and read the old way: whether each is open, at 16 points per
        /// cell over the whole terrain (terrain_holes_check.hlsl). Only a terrain of at most 16,384 samples a side (the
        /// flat map cannot be made larger). Blocks.
        /// </summary>
        [Button("Check Hole Pages")]
        [System.ComponentModel.Description("Compare the hole pages with the whole-map hole texture they replaced, at 16 points per cell")]
        public unsafe void CheckHolePages()
        {
            var pages = Terrain?.Pages;
            if (pages is not { HasHoles: true }) { HolePagesCheck = "no holes"; return; }
            int res = pages.Resolution;
            if (res > MaxTextureSize) { HolePagesCheck = $"{res} samples a side: the flat map cannot be made"; return; }
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var device = Engine.Device;
            const int steps = 4;

            // The old flat map, from the store
            var flat = device.CreateTexture2D(Format.R8_UNorm, res, res, 1, 1, ResourceFlags.None, ResourceStates.Common);
            uint flatSrv = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(flat, new ShaderResourceViewDescription
            {
                Format = Format.R8_UNorm,
                ViewDimension = ShaderResourceViewDimension.Texture2D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
            }, device.GetCpuHandle(flatSrv));
            int rowPitch = (res + 255) & ~255;
            var upload = device.NativeDevice.CreateCommittedResource(new HeapProperties(HeapType.Upload), HeapFlags.None,
                ResourceDescription.Buffer((ulong)rowPitch * (ulong)res), ResourceStates.GenericRead, null);
            void* data;
            upload.Map(0, null, &data);
            pages.CopyHoles(0, 0, res, res, new Span<byte>(data, rowPitch * res), rowPitch);
            upload.Unmap(0);

            var result = GraphicsBuffer.CreateRaw(4, uav: true, clearable: true);
            var readback = device.NativeDevice.CreateCommittedResource(new HeapProperties(HeapType.Readback), HeapFlags.None,
                ResourceDescription.Buffer(16), ResourceStates.CopyDest, null);
            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            var cmd = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, allocator, null);
            try
            {
                cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
                cmd.SetComputeRootSignature(device.GlobalRootSignature);
                cmd.ResourceBarrierTransition(flat, ResourceStates.Common, ResourceStates.CopyDest);
                cmd.CopyTextureRegion(new TextureCopyLocation(flat, 0), 0, 0, 0, new TextureCopyLocation(upload, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R8_UNorm, (uint)res, (uint)res, 1, (uint)rowPitch)
                }));
                cmd.ResourceBarrierTransition(flat, ResourceStates.CopyDest, ResourceStates.NonPixelShaderResource);
                result.Transition(cmd, ResourceStates.UnorderedAccess);
                result.ClearUAV(cmd, new Vortice.Mathematics.Int4(0, 0, 0, 0));
                cmd.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(result.Native)));

                _holeCheckCS ??= new ComputeShader("terrain_holes_check.hlsl");
                int k = _holeCheckCS.FindKernel("CS_CheckHoles");
                _holeCheckCS.SetPushConstant(k, "HoleParams", _holePool.WriteParams());
                _holeCheckCS.SetPushConstant(k, "Flat", flatSrv);
                _holeCheckCS.SetPushConstant(k, "Result", result.UavIndex);
                _holeCheckCS.SetPushConstant(k, "Cells", (uint)(res - 1));
                _holeCheckCS.SetPushConstant(k, "Steps", (uint)steps);
                _holeCheckCS.SetParam(k, "SizeX", Terrain!.TerrainSize.X);
                _holeCheckCS.SetParam(k, "SizeZ", Terrain.TerrainSize.Y);
                uint groups = (uint)(((res - 1) * steps + 7) / 8);
                _holeCheckCS.Dispatch(k, cmd, groups, groups);

                cmd.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(result.Native)));
                result.Transition(cmd, ResourceStates.CopySource);
                cmd.CopyBufferRegion(readback, 0, result.Native, 0, 16);
                cmd.ResourceBarrierTransition(flat, ResourceStates.NonPixelShaderResource, ResourceStates.Common);
                cmd.Close();
                device.SubmitAndWait(cmd);

                uint* counts;
                readback.Map(0, null, (void**)&counts);
                uint differ = counts[0], open = counts[1], far = counts[2];
                readback.Unmap(0);
                long points = (long)(res - 1) * steps * (res - 1) * steps;
                HolePagesCheck = (differ == 0 ? "equal" : far == 0 ? "equal but at the edge" : "DIFFERENT") + $": {points:N0} points over {res - 1}² cells, {open:N0} open by the " +
                                 $"flat map, {differ:N0} differ ({far:N0} of them more than 2/256 from the edge, the rest where the hardware filter's 1/256 steps tip it); {_holePool.Count} cut pages in the pool; {sw.ElapsedMilliseconds} ms";
            }
            finally
            {
                cmd.Dispose();
                allocator.Dispose();
                readback.Dispose();
                device.DeferDispose(upload);
                device.DeferDispose(flat);
                device.DeferReleaseBindlessIndex(flatSrv);
                device.DeferDispose(result);
            }
            Debug.LogAlways($"[Terrain] '{Terrain.Name}' hole pages check: {HolePagesCheck}");
        }

        private ComputeShader? _holeCheckCS;

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
        /// vertex shader samples it. Sample (i, j) holds the height of the point (i · s, j · s) from the terrain's corner,
        /// s = size / (N − 1) = Terrain.VertexSpacing: one sample on every vertex (terrain_grid.hlsli, TERRAIN_PAGES_PLAN.md
        /// stage 1). Stamps are placed at uv = (world − origin) / size. Outside the terrain, its own height.
        ///
        /// Until 2026-10-24 this took sample i at i · size / N and the plane through three corners of the cell, which
        /// extrapolates in its other half (animals stood in the ground); until 2026-11-07 the bake put sample i at
        /// (i + 0.5) · size / N, between the vertices. The physics height field (TerrainCollider) and the navmesh bake
        /// (TerrainSource.SampleHeight) put their samples at the same places; the hole mask is read by place
        /// (terrain_system/overview.md).
        /// </summary>
        public float GetHeight(Vector3 position)
        {
            var pages = Terrain?.Pages;
            if (pages is not { HasHeights: true }) return Transform?.Position.Y ?? 0;
            if (Transform == null) return 0;

            var terrainSize = Terrain!.TerrainSize;
            position -= Transform.Position;
            if (position.X < 0f || position.X > terrainSize.X || position.Z < 0f || position.Z > terrainSize.Y)
                return Transform.Position.Y;

            // In samples: sample i at i · size / (N − 1), the first and the last on the terrain's edges; the four round
            // it always lie on one page (TerrainPages)
            int last = pages.Resolution - 1;
            float tx = Math.Clamp(position.X / terrainSize.X * last, 0f, last);
            float tz = Math.Clamp(position.Z / terrainSize.Y * last, 0f, last);
            return Transform.Position.Y + pages.Height01(tx, tz) * Terrain.MaxHeight;
        }

        /// <summary>
        /// The layer that shows most at a place on the terrain (X and Z of the position), or null:
        /// off the terrain, nothing painted there, or not known yet. The answer comes from a CPU copy of the finest paint
        /// page resident over the place (TERRAIN_PAGES_PLAN.md 5b), fetched only once somebody asks about a place on it,
        /// so the first calls about a page return null for a frame or two; after the stamps changed, the old answer stands
        /// until the new one has been fetched. A dictionary lookup; main thread.
        /// </summary>
        public TerrainLayer? GetLayer(Vector3 position)
        {
            if (Terrain == null || Transform == null || _paintPool.Weights == null || _splatPlan == null) return null;

            var size = Terrain.TerrainSize;
            var origin = Transform.Position;
            float u = (position.X - origin.X) / size.X;
            float v = (position.Z - origin.Z) / size.Y;
            if (!(u >= 0f && u < 1f && v >= 0f && v < 1f)) return null;

            // The finest resident page over the place, and the texel in it (pages lie in terrain UV, +Z up). Its copy is asked
            // for when there is none or it is old; until it comes, a coarser resident page's copy answers (a page that
            // arrives at the camera replaces a coarse one), and an old copy of the same page stands.
            var slots = _paintPool.Slots;
            int texels = _paintPool.Texels;
            int x = Math.Min((int)(u * texels), texels - 1), z = Math.Min((int)(v * texels), texels - 1);
            int px0 = x / Graphics.PaintPool.PageTexels, pz0 = z / Graphics.PaintPool.PageTexels;
            bool asked = false;
            for (int level = 0; level < slots.Levels; level++)
            {
                int px = px0 >> level, pz = pz0 >> level;
                int slot = slots.SlotOf(level, px, pz);
                if (slot < 0) continue;

                var key = (level, px, pz);
                bool known = _topPages.TryGetValue(key, out var page) && page!.Slot == slot;
                if (!asked && (!known || page!.Serial != _paintSlotSerial[slot]))
                    _topWanted.Add(key);
                asked = true;
                if (!known) continue;

                int lx = (x >> level) - px * Graphics.PaintPool.PageTexels, lz = (z >> level) - pz * Graphics.PaintPool.PageTexels;
                int channel = page!.Layers[lz * Graphics.PaintPool.PageTexels + lx];
                return channel < page.Palette.Count ? page.Palette[channel] : null;
            }
            return null;
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
            _variantSlots.Clear();
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

                    _variantSlots[variant] = slots.Count;
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
            DecoratorsRebuilt?.Invoke(this);
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
            long maxTiles = (long)resolution * resolution;

            // Worst-case instances: ~64 per tile max (64 threads/group)
            // But realistically with density, far fewer. Use maxTiles * 8 as reasonable upper bound. In longs: 16 km
            // (16,384 tiles a side) times 8 is one past int.MaxValue, and the negative count failed every Draw (2026-11-08)
            _maxDecoInstances = (int)Math.Min(maxTiles * 8, 1024 * 1024); // cap at 1M

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

        /// <summary>D3D12's largest 2D texture, texels a side.</summary>
        private const int MaxTextureSize = 16384;

        /// <summary>
        /// Dispatches the compute prepass (3 stages) + lean AS/MS for decoration rendering.
        /// </summary>
        private void DispatchDecorator(ID3D12GraphicsCommandList commandList, int frameIndex, RenderPass pass = RenderPass.Opaque)
        {
            if (!_computeInitialized) return; // Destroyed between enqueue and execute
            if (DecoratorMaterial?.Effect == null || Terrain == null) return;
            // A freshly created terrain has no heights until its first bake; decorators need them for placement.
            if (_baker?.Pool.Texture == null) return;

            if (!_decoBuffersCreated) return;
            if (_decoratorSlotsBuffer == null || _decoratorLODTableBuffer == null || _decoratorGroupsBuffer == null) return;
            // Nothing to spawn from until the decoration pages are there (baked in Draw, UpdateDecoPages)
            if (_decoPool.Control == null || _decoPlan == null) return;

            var device = Engine.Device;
            var camPos = Camera.Main!.Position;
            var cs = _grassCS!;

            float range = Terrain.DecorationRadius;
            int controlW = _decoPool.Texels, controlH = _decoPool.Texels;
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
                commandList.SetDescriptorHeaps(1, _srvHeaps ??= new[] { Engine.Device.SrvHeap });
                commandList.SetComputeRootSignature(device.GlobalRootSignature);
                _instanceCounterBuffer.ClearUAV(commandList, new Vortice.Mathematics.Int4(0, 0, 0, 0));
                commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

                // The registry as this frame has it: a custom action runs before its pass's batches, whose Build is
                // what uploads it otherwise, so in a load's first world frame its index was still 0 (GBV)
                MeshRegistry.Upload(device);

                // ── Push constants: bindless resource indices only ──
                cs.SetBuffer("DecoratorSlots", _decoratorSlotsBuffer!);
                cs.SetBuffer("LODTable", _decoratorLODTableBuffer!);
                cs.SetBuffer("DecoratorGroups", _decoratorGroupsBuffer!);
                cs.SetPushConstant("MeshRegistry", MeshRegistry.SrvIndex);
                cs.SetPushConstant("TerrainPages", GpuTerrainIdx());
                cs.SetPushConstant("DecoControl", DecoParamsIdx());   // the decoration pages' TerrainDecoParams

                // The layers' plant tints over the paint pages (through TerrainPages): each plant takes the colour of the
                // ground it stands on
                bool plantTints = _splatPlan != null && _plantTintBuffer != null && _layerPalette.Count > 0;
                if (plantTints)
                    cs.SetBuffer("LayerPlantTint", _plantTintBuffer!);
                cs.SetPushConstant("SplatLayerCount", plantTints ? (uint)_layerPalette.Count : 0u);
                cs.SetPushConstant("HoleTex", HoleSrv());   // nothing spawns in a hole
                cs.SetPushConstant("DecoRows", _baker?.DecoRowsSrv ?? 0u);   // crops in rows (DecoStamp.RowSpacing)

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

        /// <summary>Diagnostics: paint texels the last paint bake composited (its pages', border included).</summary>
        [Reflection.DontSerialize]
        public long SplatBakeTexels { get; private set; }

        /// <summary>Diagnostics: paint pages and decoration pages baked since this renderer began (arrivals and edits).</summary>
        [Reflection.DontSerialize]
        public long PaintPagesBaked { get; private set; }
        [Reflection.DontSerialize]
        public long DecoPagesBaked { get; private set; }

        /// <summary>Diagnostics: decoration control texels the last deco bake composited (its pages', border included).</summary>
        [Reflection.DontSerialize]
        public long DecoBakeTexels { get; private set; }

        /// <summary>Diagnostics: decoration pages resident, waiting for a slot or bake, and baked last frame.</summary>
        [Reflection.DontSerialize]
        public string DecoPages => _decoPool.Control == null ? "none"
            : $"{(_decoPool.Slots.AllResident ? _decoPool.Capacity : _decoPool.Capacity - _decoPool.Slots.FreeSlots)} of {_decoPool.Capacity} slots, " +
              $"{DecoResidency.Waiting} waiting, {DecoResidency.BakedLastFrame} baked last frame";

        /// <summary>Diagnostics: what the last Check Deco Pages found.</summary>
        [Reflection.DontSerialize]
        public string DecoPagesCheck { get; private set; } = "";

        /// <summary>
        /// Every level 0 decoration page resident (all of them with EngineSettings.TerrainDecoPages 0) against the whole-map
        /// prepass there was before 5c, baked over the same texels: both slices value for value (TerrainBaker.CheckDecoPages).
        /// Blocks for the bakes.
        /// </summary>
        [Button("Check Deco Pages")]
        [System.ComponentModel.Description("Compare the level 0 decoration pages in place with the whole-map prepass over the same texels")]
        public void CheckDecoPages()
        {
            if (Terrain == null || _decoPlan == null) { DecoPagesCheck = "no decorations"; return; }
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var (texels, pages, compared, checkedPages) = _baker.CheckDecoPages(_decoPlan, Terrain, GpuTerrainIdx(), _decoPool);
            DecoPagesCheck = texels < 0 ? "the pool is not ready"
                : (texels == 0 ? "equal" : "DIFFERENT") + $": {checkedPages} of {_decoPool.Slots.PagesAcross * _decoPool.Slots.PagesAcross} level 0 " +
                  $"pages in place, {compared:N0} texels compared, {texels:N0} differ on {pages} pages; {sw.ElapsedMilliseconds} ms";
            Debug.LogAlways($"[Terrain] '{Terrain.Name}' deco pages check: {DecoPagesCheck}");
        }

        /// <summary>Diagnostics: paint pages resident, waiting for a slot or bake, and baked last frame.</summary>
        [Reflection.DontSerialize]
        public string PaintPages => _paintPool.Weights == null ? "none"
            : $"{(_paintPool.Slots.AllResident ? _paintPool.Capacity : _paintPool.Capacity - _paintPool.Slots.FreeSlots)} of {_paintPool.Capacity} slots, " +
              $"{PaintResidency.Waiting} waiting, {PaintResidency.BakedLastFrame} baked last frame";

        /// <summary>Diagnostics: height pages the last height bake went over (all of them, or a local edit's).</summary>
        [Reflection.DontSerialize]
        public int HeightBakePages { get; private set; }

        /// <summary>Diagnostics: what the last Check Page Bake found.</summary>
        [Reflection.DontSerialize]
        public string PageBakeCheck { get; private set; } = "";

        /// <summary>
        /// Bake the heights three ways and compare them sample by sample with each other and with the map in place
        /// (TerrainBaker.CheckPageBake): every stamp over the whole map, every page at once with its own stamps, and page
        /// after page alone. They must be equal to the bit (TERRAIN_PAGES_PLAN.md stage 3); the map in place too, after
        /// edits that rebaked only their pages. Blocks for the bakes.
        /// </summary>
        [Button("Check Page Bake")]
        [System.ComponentModel.Description("Bake the heights over the whole map and page by page, and compare them sample by sample")]
        public void CheckPageBake()
        {
            if (Terrain == null) return;
            var plan = _baker.PrepareHeight(Terrain, this);
            var (inPlace, bDiffers, cDiffers, borderDiffers, maxStep, aMs, bMs) = _baker.CheckPageBake(Terrain, plan);
            int pages = plan.Pages * plan.Pages;
            PageBakeCheck = (inPlace == 0 && bDiffers == 0 && cDiffers == 0 && borderDiffers == 0 ? "equal" : "DIFFERENT") +
                            $": {pages} pages; against every stamp on every page, {inPlace} texels of the pool in place " +
                            $"differ, all pages at once {bDiffers}, page by page {cDiffers}; {borderDiffers} border or " +
                            "shared-edge texels differ from the page their sample belongs to" +
                            (maxStep > 0 ? $" (up to {maxStep} steps of 65535)" : "") +
                            $"; whole map with every stamp {aMs:0.0} ms, pages with their own {bMs:0.0} ms (on the queue, " +
                            "less a copy alone)";
            if (Transform != null)
            {
                var (atSamples, between) = _baker.CompareReads(Terrain, Transform.WorldPosition);
                PageBakeCheck += $"; reads through the pool against the same bilinear in float on the CPU: {atSamples * 1000f:0.####} mm " +
                                 $"at most on samples, {between * 1000f:0.####} mm between them";
            }
            Debug.LogAlways($"[Terrain] '{Terrain.Name}' page bake check: {PageBakeCheck}");
        }

        /// <summary>Diagnostics: <see cref="GetLayer"/> under the main camera (asked again every time it is read, so the
        /// page under it is fetched; "?" until it has been).</summary>
        [Reflection.DontSerialize]
        public string LayerUnderCamera => Camera.Main == null ? "" : GetLayer(Camera.Main.Position)?.Name ?? "?";

        /// <summary>Diagnostics: what the last Check Paint Pages found.</summary>
        [Reflection.DontSerialize]
        public string PaintPagesCheck { get; private set; } = "";

        /// <summary>
        /// Every level 0 paint page resident (all of them with EngineSettings.TerrainPaintPages 0) against the whole-map splat
        /// bake there was before 5b, baked over the same texels: their weights byte for byte (TerrainBaker.CheckPaintPages).
        /// Blocks for the bakes.
        /// </summary>
        [Button("Check Paint Pages")]
        [System.ComponentModel.Description("Compare the level 0 paint pages in place with the whole-map splat bake over the same texels")]
        public void CheckPaintPages()
        {
            if (Terrain == null || _splatPlan == null) { PaintPagesCheck = "no paint"; return; }
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var (texels, pages, maxStep, compared, checkedPages) = _baker.CheckPaintPages(_splatPlan, Terrain, GpuTerrainIdx(), _paintPool);
            PaintPagesCheck = texels < 0 ? "the pool is not ready"
                : (texels == 0 ? "equal" : "DIFFERENT") + $": {checkedPages} of {_paintPool.Slots.PagesAcross * _paintPool.Slots.PagesAcross} level 0 " +
                  $"pages in place, {compared:N0} texel slices compared, {texels:N0} differ on {pages} pages" +
                  (maxStep > 0 ? $" (up to {maxStep} of 255)" : "") + $"; {sw.ElapsedMilliseconds} ms";
            Debug.LogAlways($"[Terrain] '{Terrain.Name}' paint pages check: {PaintPagesCheck}");
        }

        // ── Changed region ──
        // Listeners of TerrainHeightsChanged (PCG, surface meshes) only rebuild if they touch the region the
        // stamps changed: where each edited stamp was at the last readback, plus where it is now. The readback goes
        // over that region and tells them only where samples came out different (ChangedSamplesRegion).

        /// <summary>
        /// The world XZ region a regional readback changed: the box of samples that differ from the field before
        /// (<see cref="TerrainBaker.ReadbackPages"/>), widened by how far from a sample its height is read. A height
        /// between samples comes from the samples either side (<see cref="GetHeight"/>), so a sample reaches a spacing
        /// each way, and normals are taken from heights up to half a metre apart (TerrainProjection): a metre more.
        /// Null when no sample differs.
        /// </summary>
        private TerrainHeightsChange? ChangedSamplesRegion((int x0, int z0, int x1, int z1)? changed, int cellsX, int cellsZ)
        {
            if (changed is not { } box) return null;
            const float normalReach = 1f;
            var origin = Transform!.WorldPosition;
            var size = Terrain.TerrainSize;
            float sx = size.X / cellsX, sz = size.Y / cellsZ;
            var min = new Vector2(origin.X + (box.x0 - 1) * sx - normalReach, origin.Z + (box.z0 - 1) * sz - normalReach);
            var max = new Vector2(origin.X + box.x1 * sx + normalReach, origin.Z + box.z1 * sz + normalReach);
            return new TerrainHeightsChange(Terrain, false, min, max);
        }

        /// <summary>
        /// Brings Terrain.Pages up to what the GPU holds, for a save (TerrainLoader writes the pages file from it): the
        /// readback that is due runs now instead of on the next Draw. Throws while a bake is still on its way (stamps
        /// changed and not baked yet, or baked into a command list that has not run): those heights are nowhere to save
        /// yet. Until 4c a save read the whole map back from the GPU itself, whatever the CPU held.
        /// </summary>
        internal void FinishHeightsForSave()
        {
            if (Terrain == null) return;
            if (System.Threading.Volatile.Read(ref _heightBakesPending) > 0
                || Terrain.NeedsUpdate(TerrainDirtyFlags.HeightBake) || Terrain.NeedsUpdate(TerrainDirtyFlags.HoleBake))
                throw new InvalidOperationException($"'{Terrain.Name}' is still baking its heights or holes: save again once it has");
            if (_needHeightFieldReadback && _baker != null)
                ReadBackHeights();
        }

        /// <summary>
        /// Reads the baked heights back into the CPU store (Terrain.Pages): only the pages the stamps changed since the
        /// last readback, when there is one of this size to compare with. Tells whoever sampled the ground where it
        /// changed (TerrainHeightsChanged).
        /// </summary>
        private void ReadBackHeights()
        {
            _needHeightFieldReadback = false;
            using var readbackScope = _readbackMarker.Auto();
            // Only the pages where the stamps changed it since the last readback, when a store of this size holds
            // heights to compare with (in samples: sample i stands at i · size / (N − 1), see GetHeight; a sample
            // more each side). A page that comes back the same stays the one the store had.
            (int x0, int z0, int x1, int z1)? changed;
            var previousPages = Terrain.Pages;
            bool regional = !_heightChangeAll && previousPages is { HasHeights: true } && Transform != null && _heightChangeMax.X >= _heightChangeMin.X;
            TerrainPages? heights;
            if (regional)
            {
                var origin = Transform!.WorldPosition;
                var size = Terrain.TerrainSize;
                int cells = previousPages!.Resolution - 1;
                heights = _baker.ReadbackPages(previousPages, false, out changed,
                    (int)MathF.Floor((_heightChangeMin.X - origin.X) / size.X * cells) - 1,
                    (int)MathF.Floor((_heightChangeMin.Y - origin.Z) / size.Y * cells) - 1,
                    (int)MathF.Ceiling((_heightChangeMax.X - origin.X) / size.X * cells) + 2,
                    (int)MathF.Ceiling((_heightChangeMax.Y - origin.Z) / size.Y * cells) + 2);
            }
            else
                heights = _baker.ReadbackPages(previousPages, true, out changed);
            if (heights != null && Terrain != null)
            {
                Terrain.SetPages(heights);
                Terrain.PagesStampsHash = Terrain.HeightmapStampsHash;   // read back from the map as it is now
                // Streaming: the resident pages over what changed go up again (the coarse ones and those that came in
                // from the store before this readback), and the height ranges there are built again
                if (_baker.PoolCapacity != 0 && changed is { } moved)
                {
                    Residency.MarkStale(moved.x0, moved.z0, moved.x1, moved.z1);
                    MarkCpuRanges(moved.x0, moved.z0, moved.x1, moved.z1);
                }
                // Holes cut at another size do not go over to a store of this one: the stamps cut them again
                if (previousPages is { HasHoles: true } && previousPages.Resolution != heights.Resolution)
                    Terrain.MarkForUpdate(TerrainDirtyFlags.HoleBake);
                Terrain.CheckMapsAgainstSaved();    // on Save Changed Assets when the bake no longer equals the saved heightmap
                _heightsReadBack = true;
                // Surface-snapped geometry (RuntimeMesh) and projected PCG output sampled the old heights —
                // let whatever lies in the changed region rebuild. A regional readback names only the samples that
                // differ from the field they sampled, or nothing when none does: a stamp edit that bakes the same
                // heights (Opacity 1 → 1 on Eldermoor's rolling ground: 196 pages) regenerated every PCG component
                // over its region before, 2.7 s.
                var change = regional
                    ? ChangedSamplesRegion(changed, heights.Resolution - 1, heights.Resolution - 1)
                    : new TerrainHeightsChange(Terrain, _heightChangeAll, _heightChangeMin, _heightChangeMax);
                Debug.Log($"[Terrain] Heights read back (tick {Engine.TickCount}): " +
                          (change == null ? "unchanged" : change.All ? "all" : $"x {change.Min.X:0}..{change.Max.X:0}, z {change.Min.Y:0}..{change.Max.Y:0}") +
                          (regional ? $" (stamps x {_heightChangeMin.X:0}..{_heightChangeMax.X:0}, z {_heightChangeMin.Y:0}..{_heightChangeMax.Y:0})" : "") +
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
                if (change != null)
                {
                    using (_heightsChangedMarker.Auto())
                        MessageDispatcher.Send(EngineMsg.TerrainHeightsChanged, change);
                }
            }
        }

        /// <summary>
        /// The CPU heights (Terrain.Pages) are missing or about to be replaced: a height bake or its readback is still
        /// outstanding. Whatever samples heights now (PCG graphs) gets flat ground or values that change in
        /// a frame or two; TerrainHeightsChanged follows once they are in. Not at a load that keeps the saved heights
        /// (<see cref="KeepHeights"/>: nothing changed): there the load's <see cref="SceneLoad.HeightsIn"/> says so.
        /// </summary>
        [System.ComponentModel.Browsable(false)]
        public bool HeightsPending
            => Enabled && Terrain != null
            && (Terrain.Pages is not { HasHeights: true } || !_heightsReadBack || _needHeightFieldReadback
                || System.Threading.Volatile.Read(ref _heightBakesPending) > 0
                || Terrain.NeedsUpdate(TerrainDirtyFlags.HeightBake));

        // This renderer has read heights back at least once. Until then a Terrain asset still cached from the
        // scene before a reload looks ready, although this renderer's first readback is yet to come.
        private bool _heightsReadBack;

        // The first height bake of this renderer is yet to be laid out: it may keep the heights in place
        private bool _initialHeightCheck;

        /// <summary>
        /// The GPU pool and the CPU store both hold what this plan would bake (the saved map at a cold load,
        /// the last bake at a reload): equal <see cref="TerrainBaker.HeightPlan.Hash"/>es bake equal maps.
        /// </summary>
        private bool KeepsHeights(TerrainBaker.HeightPlan plan)
        {
            string? why =
                _baker?.Pool is not { Filled: true } ? "the GPU pool is not filled"
                : _baker.Pool.Resolution != plan.Resolution ? $"the GPU pool is {_baker.Pool.Resolution}², the plan {plan.Resolution}²"
                : Terrain.Pages is not { HasHeights: true } ? "no CPU heights"
                : Terrain.Pages.Resolution != plan.Resolution ? $"the CPU heights are {Terrain.Pages.Resolution}², the plan {plan.Resolution}²"
                : plan.Hash != Terrain.HeightmapStampsHash ? $"the stamps hash {plan.Hash}, the GPU heights came from {Terrain.HeightmapStampsHash}"
                : plan.Hash != Terrain.PagesStampsHash ? $"the stamps hash {plan.Hash}, the CPU heights came from {Terrain.PagesStampsHash}"
                : null;
            if (why != null)
                Debug.LogAlways($"[Terrain] '{Terrain.Name}': heights baked, not kept: {why}");
            return why == null;
        }

        /// <summary>
        /// Skip the first bake: no GPU bake, no readback and no TerrainHeightsChanged (nothing changed). What the bake
        /// would have left behind is set up here: the heights count as read back, the stamp regions are where the
        /// stamps are. At load this saves a full-terrain
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
            Debug.LogAlways($"[Terrain] '{Terrain.Name}': heights kept, the stamps are those they were baked from");
        }

        private readonly Dictionary<TerrainStamp, (Vector2 min, Vector2 max)> _stampRegions = new();
        private bool _heightChangeAll = true;   // the first bake covers everything
        private bool _heightRegionMarked;       // a stamp gave a region for the pending bake
        private int _heightBakesPending;        // bakes enqueued whose render callback has not run yet
        private Vector2 _heightChangeMin = new(float.MaxValue);
        private Vector2 _heightChangeMax = new(float.MinValue);

        // The pages the next height bake goes over: where stamps changed since the last bake was laid out. The readback's
        // region above gathers until the heights are read back, over several bakes while a stamp is dragged.
        private bool _heightBakeAll = true;
        private Vector2 _heightBakeMin = new(float.MaxValue);
        private Vector2 _heightBakeMax = new(float.MinValue);

        private static bool TryGetStampRegion(TerrainStamp stamp, out Vector2 min, out Vector2 max)
        {
            min = max = default;
            // A global stamp has no region: whatever it changes, it can change anywhere. A removed one is nowhere now
            // (its spline may already be gone with it): only where it was counts
            if (stamp.IsGlobal || stamp.IsDestroyed) return false;
            if (stamp.IsSplineMode && stamp.GetSpline().Points.Count < 2) return false;
            // Nor does a stamp covering what lies beyond its outline (the seabed): its spline's box is where it is not
            if (stamp is HeightStamp { Outside: true, IsSplineMode: true } outside && outside.GetSpline() is { Closed: true })
                return false;

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
            _heightBakeMin = Vector2.Min(_heightBakeMin, min);
            _heightBakeMax = Vector2.Max(_heightBakeMax, max);
        }

        /// <summary>Where a changed local stamp was and is: the region the height readback reports and the splat bake redoes.</summary>
        private void MarkStampRegion(TerrainStamp stamp)
        {
            _heightRegionMarked = true;

            if (_stampRegions.TryGetValue(stamp, out var old))
            {
                GrowHeightChange(old.min, old.max);
                Terrain?.MarkSplatRegion(old.min, old.max);
                Terrain?.MarkDecoRegion(old.min, old.max);
            }

            if (TryGetStampRegion(stamp, out var min, out var max))
            {
                GrowHeightChange(min, max);
                Terrain?.MarkSplatRegion(min, max);
                Terrain?.MarkDecoRegion(min, max);
                _stampRegions[stamp] = (min, max);
            }
            else if (stamp.IsDestroyed && _stampRegions.Remove(stamp))
            {
                // removed: where it was, marked above
            }
            else
            {
                _heightChangeAll = true;
                _heightBakeAll = true;
                Terrain?.MarkForUpdate(TerrainDirtyFlags.SplatPack | TerrainDirtyFlags.DecoPrepass);
            }
        }

        private void SnapshotStampRegions()
        {
            _stampRegions.Clear();
            foreach (var stamp in ComponentCache<HeightStamp>.All) Snapshot(stamp);
            foreach (var stamp in ComponentCache<HeightNoiseStamp>.All) Snapshot(stamp);
            foreach (var stamp in ComponentCache<SplatStamp>.All) Snapshot(stamp);
            foreach (var stamp in ComponentCache<BiomeStamp>.All) Snapshot(stamp);
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

            // A decoration stamp: its decoration pages where it was and is (it changes nothing else)
            if (stamp is DecoStamp)
            {
                MarkDecoStampRegion(stamp);
                return;
            }

            // A biome stamp is paint and ground cover too (its rules), never the height path below
            if (stamp is SplatStamp or BiomeStamp)
            {
                var change = MarkPaintRegion(stamp);
                Terrain?.MarkForUpdate(TerrainDirtyFlags.AlbedoBake);
                // The decoration pages there (a biome's ground cover rules; a splat stamp reaches them through the paint
                // pages it bakes as well)
                if (change.All) Terrain?.MarkForUpdate(TerrainDirtyFlags.DecoPrepass);
                else Terrain?.MarkDecoRegion(change.Min, change.Max);
                MessageDispatcher.Send(EngineMsg.TerrainPaintChanged, change);
                return;
            }

            MarkStampRegion(stamp);     // the splat and decoration rebakes, of the stamp's region
            Terrain?.MarkForUpdate(
                TerrainDirtyFlags.HeightBake |
                TerrainDirtyFlags.AlbedoBake);
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

            if (stamp.IsDestroyed && _stampRegions.Remove(stamp))
            {
                // Removed: where it was
                Terrain?.MarkSplatRegion(min, max);
                return new TerrainHeightsChange(Terrain, false, min, max);
            }

            _stampRegions.Remove(stamp);
            Terrain?.MarkForUpdate(TerrainDirtyFlags.SplatPack);
            return new TerrainHeightsChange(Terrain, true, min, max);
        }

        /// <summary>Where a changed local deco stamp was and is: the decoration pages that bake again (all of them when it
        /// has no region).</summary>
        private void MarkDecoStampRegion(TerrainStamp stamp)
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
                _stampRegions[stamp] = (newMin, newMax);
                Terrain?.MarkDecoRegion(Vector2.Min(min, newMin), Vector2.Max(max, newMax));
                return;
            }
            if (stamp.IsDestroyed && _stampRegions.Remove(stamp))
            {
                Terrain?.MarkDecoRegion(min, max);   // removed: where it was
                return;
            }
            _stampRegions.Remove(stamp);
            Terrain?.MarkForUpdate(TerrainDirtyFlags.DecoPrepass);
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
