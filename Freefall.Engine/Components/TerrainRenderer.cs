using System;
using System.Collections.Generic;
using System.IO;
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
        private float PixelErrorThreshold = 128.0f;
        public int MaxDepth = 7;
        private int MaxPatches = 32768;
        private const int MinPatches = 32768;

        private Vector4[] _layerTiling = new Vector4[32];

        // ───── Internal State ─────────────────────────────────────────────
        private const int FrameCount = 3;

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

        // ───── Height Bake (GPU layer compositor) ─────────────────────────
        private bool _heightBakeDirty = true;
        private bool _splatPackDirty = true;
        private bool _needHeightFieldReadback;

        /// <summary>Marks the baked heightmap as dirty, triggering a rebake next frame.</summary>
        public void MarkHeightDirty() => _heightBakeDirty = true;
        public void MarkSplatDirty() => _splatPackDirty = true;

        /// <summary>Forces full rebuild of all texture arrays + height bake + splat pack next frame.</summary>
        public void MarkLayersDirty()
        {
            _heightBakeDirty = true;
            _splatPackDirty = true;
            _bakedAlbedoDirty = true;
            _textureArraysDirty = true;
        }

        private bool _textureArraysDirty;
        private List<Texture> _pendingPackedSlices;

        /// <summary>
        /// Enqueues a brush stroke to be dispatched on the render thread.
        /// Points are in terrain UV space [0..1].
        /// </summary>
        public void EnqueueBrushStroke(Vector2[] strokePoints, int pointCount,
                                       uint mode, float strength,
                                       float radius, float falloff,
                                       float targetHeight = 0,
                                       TerrainBaker.ControlMapTarget target = TerrainBaker.ControlMapTarget.Height,
                                       int layerIndex = 0)
        {
            if (Terrain == null || pointCount == 0) return;

            // Resolve target: determine the Action<Texture> setter
            Action<Texture> setControlMap = null;
            switch (target)
            {
                case TerrainBaker.ControlMapTarget.Height:
                {
                    var paintLayer = Terrain.HeightLayers.OfType<PaintHeightLayer>().FirstOrDefault();
                    if (paintLayer == null)
                    {
                        paintLayer = new PaintHeightLayer();
                        Terrain.HeightLayers.Add(paintLayer);
                    }
                    var layer = paintLayer;
                    setControlMap = tex => layer.ControlMap = tex;
                    break;
                }
                case TerrainBaker.ControlMapTarget.Splatmap:
                    if (Terrain.Layers != null && layerIndex >= 0 && layerIndex < Terrain.Layers.Count)
                    {
                        var layer = Terrain.Layers[layerIndex];
                        setControlMap = tex => layer.ControlMap = tex;
                    }
                    break;
                case TerrainBaker.ControlMapTarget.Density:
                    if (Terrain.Decorations != null && layerIndex >= 0 && layerIndex < Terrain.Decorations.Count)
                    {
                        var deco = Terrain.Decorations[layerIndex];
                        setControlMap = tex => deco.ControlMap = tex;
                    }
                    break;
            }

            if (setControlMap == null) return;

            // Capture references for the lambda
            var terrain = Terrain;
            var baker = TerrainBaker.Instance;
            var pts = strokePoints;
            int count = pointCount;
            var tgt = target;
            int idx = layerIndex;
            var setter = setControlMap;
            bool isHeightTarget = target == TerrainBaker.ControlMapTarget.Height;

            CommandBuffer.Enqueue(RenderPass.Opaque, (list) =>
            {
                Debug.Log($"[TerrainRenderer] Brush lambda executing: target={tgt} layer={idx} mode={mode} pts={count}");
                baker.PaintBrush(terrain, tgt, idx, setter, list,
                                 pts, count, mode, strength,
                                 radius, falloff, targetHeight);
                if (isHeightTarget)
                {
                    _heightBakeDirty = true;
                    _heightRangePyramidBuilt = false;
                }
                if (tgt == TerrainBaker.ControlMapTarget.Splatmap)
                    _splatPackDirty = true;
            });
        }

        /// <summary>
        /// GPU-only brush: enqueues a compute raycast against the baked heightmap
        /// followed by painting at the hit UV. No CPU heightfield involvement.
        /// </summary>
        public void EnqueueBrushRaycastAndPaint(Vector3 rayOrigin, Vector3 rayDir,
                                                 uint mode, float strength,
                                                 float radius, float falloff,
                                                 float targetHeight = 0,
                                                 TerrainBaker.ControlMapTarget target = TerrainBaker.ControlMapTarget.Height,
                                                 int layerIndex = 0)
        {
            if (Terrain == null) return;

            // Resolve the ControlMap setter (same pattern as EnqueueBrushStroke)
            Action<Texture> setControlMap = null;
            switch (target)
            {
                case TerrainBaker.ControlMapTarget.Height:
                {
                    var paintLayer = Terrain.HeightLayers.OfType<PaintHeightLayer>().FirstOrDefault();
                    if (paintLayer == null)
                    {
                        paintLayer = new PaintHeightLayer();
                        Terrain.HeightLayers.Add(paintLayer);
                    }
                    var layer = paintLayer;
                    setControlMap = tex => layer.ControlMap = tex;
                    break;
                }
                case TerrainBaker.ControlMapTarget.Splatmap:
                    if (Terrain.Layers != null && layerIndex >= 0 && layerIndex < Terrain.Layers.Count)
                    {
                        var layer = Terrain.Layers[layerIndex];
                        setControlMap = tex => layer.ControlMap = tex;
                    }
                    break;
                case TerrainBaker.ControlMapTarget.Density:
                    if (Terrain.Decorations != null && layerIndex >= 0 && layerIndex < Terrain.Decorations.Count)
                    {
                        var deco = Terrain.Decorations[layerIndex];
                        setControlMap = tex => deco.ControlMap = tex;
                    }
                    break;
            }

            if (setControlMap == null) return;

            // Capture for lambda
            var terrain = Terrain;
            var baker = TerrainBaker.Instance;
            var origin = Transform?.Position ?? Vector3.Zero;
            var size = terrain.TerrainSize;
            var maxH = terrain.MaxHeight;
            var setter = setControlMap;
            var tgt = target;
            int idx = layerIndex;
            bool isHeightTarget = target == TerrainBaker.ControlMapTarget.Height;

            CommandBuffer.Enqueue(RenderPass.Opaque, (list) =>
            {
                baker.BrushRaycastAndPaint(terrain, tgt, idx, setter, list,
                    rayOrigin, rayDir, origin, size, maxH,
                    mode, strength, radius, falloff, targetHeight);

                if (isHeightTarget)
                {
                    _heightBakeDirty = true;
                    _heightRangePyramidBuilt = false;
                }
                if (tgt == TerrainBaker.ControlMapTarget.Splatmap)
                    _splatPackDirty = true;
            });
        }

        /// <summary>
        /// Enqueues a channel import from a source texture into a ControlMap.
        /// channelIndex: 0=R, 1=G, 2=B, 3=A
        /// </summary>
        public void EnqueueImportChannel(Texture sourceTexture, int channelIndex,
                                         TerrainBaker.ControlMapTarget target, int layerIndex)
        {
            if (Terrain == null || sourceTexture == null) return;

            // Resolve setter
            Action<Texture> setControlMap = null;
            switch (target)
            {
                case TerrainBaker.ControlMapTarget.Height:
                {
                    var paintLayer = Terrain.HeightLayers.OfType<PaintHeightLayer>().FirstOrDefault();
                    if (paintLayer == null)
                    {
                        paintLayer = new PaintHeightLayer();
                        Terrain.HeightLayers.Add(paintLayer);
                    }
                    var layer = paintLayer;
                    setControlMap = tex => layer.ControlMap = tex;
                    break;
                }
                case TerrainBaker.ControlMapTarget.Splatmap:
                    if (Terrain.Layers != null && layerIndex >= 0 && layerIndex < Terrain.Layers.Count)
                    {
                        var layer = Terrain.Layers[layerIndex];
                        setControlMap = tex => layer.ControlMap = tex;
                    }
                    break;
                case TerrainBaker.ControlMapTarget.Density:
                    if (Terrain.Decorations != null && layerIndex >= 0 && layerIndex < Terrain.Decorations.Count)
                    {
                        var deco = Terrain.Decorations[layerIndex];
                        setControlMap = tex => deco.ControlMap = tex;
                    }
                    break;
            }

            if (setControlMap == null) return;

            var terrain = Terrain;
            var baker = TerrainBaker.Instance;
            var src = sourceTexture;
            int ch = channelIndex;
            var tgt = target;
            int idx = layerIndex;
            var setter = setControlMap;
            bool isHeightTarget = target == TerrainBaker.ControlMapTarget.Height;

            CommandBuffer.Enqueue(RenderPass.Opaque, (list) =>
            {
                baker.ImportChannel(terrain, tgt, idx, setter, list, src, ch);
                if (isHeightTarget)
                {
                    _heightBakeDirty = true;
                    _heightRangePyramidBuilt = false;
                }
                if (tgt == TerrainBaker.ControlMapTarget.Splatmap)
                    _splatPackDirty = true;
            });
        }

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
            if (_textureArraysDirty || !_texturesInitialized)
            {
                if (Terrain?.Layers == null || Terrain.Layers.Count == 0)
                {
                    if (!_texturesInitialized) return; // no layers yet on first init — skip
                    // Layers removed — clear to fallbacks
                }
                try
                {
                    SetupLayerTiling();
                    _texturesInitialized = true;
                }
                catch (Exception ex)
                {
                    Debug.LogError("[TerrainRenderer]", $"SetupLayerTiling failed: {ex.Message}");
                }
                _textureArraysDirty = false; // always clear — MarkLayersDirty() re-sets if needed
            }
        }

        public void Draw()
        {
            if (Camera.Main == null || Terrain == null || !_computeInitialized) return;

            LazyInitTextures();

            var material = Material;
            var heightmap = Terrain.Heightmap;
            if (material == null || material.Effect == null) return;

            int frameIndex = Engine.FrameIndex % FrameCount;

            // GPU height layer bake (runs before any heightmap access)
            if (_heightBakeDirty && Terrain.HeightLayers.Count > 0)
            {
                var baker = TerrainBaker.Instance;

                // Upload any pending ControlMap data loaded from cache
                foreach (var layer in Terrain.HeightLayers)
                {
                    if (layer is PaintHeightLayer paint && paint.PendingControlMapBytes != null)
                    {
                        var p = paint; // capture for lambda
                        baker.UploadControlMap(TerrainBaker.ControlMapTarget.Height, 0,
                            paint.PendingControlMapBytes, Terrain.HeightmapResolution,
                            tex => p.ControlMap = tex);
                        paint.PendingControlMapBytes = null; // consumed
                    }
                }

                CommandBuffer.Enqueue(RenderPass.Opaque, (list) =>
                {
                    baker.Bake(Terrain, list);
                    _heightBakeDirty = false;
                    _heightRangePyramidBuilt = false; // force rebuild with new heights
                    _bakedAlbedoDirty = true; // re-bake albedo with new terrain shape
                    _needHeightFieldReadback = true; // trigger CPU-side heightfield rebuild next frame
                });
            }

            // Debounced CPU heightfield readback — only when baking has settled (not during active painting)
            if (_needHeightFieldReadback && !_heightBakeDirty)
            {
                _needHeightFieldReadback = false;
                var heights = TerrainBaker.Instance.ReadbackHeightmap();
                if (heights != null && Terrain != null)
                    Terrain.SetHeightField(heights);
            }

            // Upload any pending splatmap/density ControlMap data loaded from cache
            if (Terrain?.Layers != null)
            {
                var baker2 = TerrainBaker.Instance;
                for (int i = 0; i < Terrain.Layers.Count; i++)
                {
                    var layer = Terrain.Layers[i];
                    if (layer.PendingControlMapBytes != null)
                    {
                        var l = layer;
                        int idx = i;
                        baker2.UploadControlMap(TerrainBaker.ControlMapTarget.Splatmap, idx,
                            layer.PendingControlMapBytes, Terrain.HeightmapResolution,
                            tex => l.ControlMap = tex);
                        layer.PendingControlMapBytes = null;
                        _splatPackDirty = true;
                    }
                }
            }

            if (Terrain?.Decorations != null)
            {
                var baker2 = TerrainBaker.Instance;
                for (int i = 0; i < Terrain.Decorations.Count; i++)
                {
                    var deco = Terrain.Decorations[i];
                    if (deco.PendingControlMapBytes != null)
                    {
                        var d = deco;
                        int idx = i;
                        baker2.UploadControlMap(TerrainBaker.ControlMapTarget.Density, idx,
                            deco.PendingControlMapBytes, Terrain.HeightmapResolution,
                            tex => d.ControlMap = tex);
                        deco.PendingControlMapBytes = null;
                    }
                }
            }

            // GPU splatmap packing — deferred: process pending packed slices from last frame
            if (_pendingPackedSlices != null)
            {
                try
                {
                    ControlMapsArray = Texture.CreateTexture2DArray(Engine.Device, _pendingPackedSlices);
                    _bakedAlbedoDirty = true;
                }
                catch (Exception ex)
                {
                    Debug.LogError("[TerrainRenderer]", $"ControlMapsArray creation failed: {ex.Message}");
                }
                _pendingPackedSlices = null;
            }

            // GPU splatmap packing — pack per-layer R16 ControlMaps into RGBA slices
            if (_splatPackDirty && Terrain?.Layers != null && Terrain.Layers.Count > 0)
            {
                // Check if any layer has a ControlMap with a valid GPU resource
                var srvIndices = new uint[Terrain.Layers.Count];
                bool hasAny = false;
                for (int i = 0; i < Terrain.Layers.Count; i++)
                {
                    var cm = Terrain.Layers[i].ControlMap;
                    if (cm != null && cm.BindlessIndex != 0)
                    {
                        srvIndices[i] = cm.BindlessIndex;
                        hasAny = true;
                    }
                }

                if (hasAny)
                {
                    int res = Terrain.HeightmapResolution;
                    var indices = srvIndices;
                    var renderer = this;
                    CommandBuffer.Enqueue(RenderPass.Opaque, (list) =>
                    {
                        var packedSlices = TerrainBaker.Instance.PackControlMaps(list, indices, res);
                        if (packedSlices.Count > 0)
                            renderer._pendingPackedSlices = packedSlices;
                        renderer._splatPackDirty = false;
                    });
                }
                else
                {
                    _splatPackDirty = false;
                }
            }

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
            _descriptorBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 16, srv: true, uav: true);
            _sphereBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 16, srv: true, uav: true);
            _subbatchIdBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 4, srv: true, uav: true);
            _terrainDataBuffers[i] = GraphicsBuffer.CreateStructured(MaxPatches, 32, srv: true, uav: true);

            // Raw R32 buffers
            _counterBuffers[i] = GraphicsBuffer.CreateRaw(1, uav: true, clearable: true);
            _splitFlagsBuffers[i] = GraphicsBuffer.CreateRaw(_totalNodes, uav: true, clearable: true);
            _indirectArgsBuffers[i] = GraphicsBuffer.CreateRaw(4, srv: true, uav: true);
            _shadowArgsBuffers[i] = GraphicsBuffer.CreateRaw(4, uav: true);

            // Shadow structured buffers
            _shadowDescriptorBuffers[i] = GraphicsBuffer.CreateStructured(shadowCapacity, 16, srv: true, uav: true);
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
                PixelErrorThreshold = PixelErrorThreshold,
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
                PixelErrorThreshold = PixelErrorThreshold,
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

            // Transform frustum planes from world → terrain local space
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
            void* pData;
            buffer.Map(0, null, &pData);
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
                Debug.Log($"[Terrain] SetupLayerTiling: {diffuseList.Count} diffuse, {normalList.Count} normals from {layers.Count} layers");
            }

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
            {
                Debug.Log("[Terrain] Normals missing! Using flat normals.");
                NormalMapsArray = InternalAssets.FlatNormalArray;
            }

            // Build ControlMapsArray for GPU splatmap sampling
            // The GPU expects ceil(layerCount/4) RGBA slices, each channel = one layer weight.
            // Legacy path: old RGBA splatmaps are already packed correctly.
            // New path: per-layer R16 ControlMaps are packed by CS_PackChannels in Draw().
            if (Terrain?._legacyControlMaps != null && Terrain._legacyControlMaps.Count > 0)
            {
                var controlList = new List<Texture>();
                foreach (var splatmap in Terrain._legacyControlMaps)
                {
                    if (splatmap?.Native != null)
                        controlList.Add(splatmap);
                }
                if (controlList.Count > 0)
                    ControlMapsArray = Texture.CreateTexture2DArray(device, controlList);
                else
                    ControlMapsArray = InternalAssets.BlackArray;
            }
            else
            {
                // Per-layer R16 ControlMaps — pack lazily in Draw()
                _splatPackDirty = true;
                ControlMapsArray = InternalAssets.BlackArray;
            }

            // DecoMapsArray is built by BuildDecoratorBuffers() from per-Decoration ControlMaps
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
            // Color tint (Unity detail prototype healthy/dry colors)
            public Vector3 HealthyColor;
            public Vector3 DryColor;
            public float NoiseSpread;
            public uint _colorPad;        // pad to 16-byte alignment
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
        /// Collects unique ControlMaps into a DecoMapsArray, builds per-decoration
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

            // Collect unique control maps and build DecoMapsArray
            var uniqueControlMaps = new List<Texture>();
            var controlMapToSlice = new Dictionary<Texture, uint>();
            foreach (var deco in decorations)
            {
                if (deco.ControlMap != null && deco.ControlMap.Native != null && !controlMapToSlice.ContainsKey(deco.ControlMap))
                {
                    controlMapToSlice[deco.ControlMap] = (uint)uniqueControlMaps.Count;
                    uniqueControlMaps.Add(deco.ControlMap);
                }
            }
            if (uniqueControlMaps.Count > 0)
                DecoMapsArray = Texture.CreateTexture2DArray(device, uniqueControlMaps);
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
                    Density = deco.Density,
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
                    DecoMapSlice = deco.ControlMap != null && controlMapToSlice.TryGetValue(deco.ControlMap, out var slice)
                        ? slice : 0xFFFFFFFF,
                    _pad0 = 0,
                    Mode = (uint)mode,
                    TextureIdx = textureIdx,
                    HealthyColor = new Vector3(deco.HealthyColor.X, deco.HealthyColor.Y, deco.HealthyColor.Z),
                    DryColor = new Vector3(deco.DryColor.X, deco.DryColor.Y, deco.DryColor.Z),
                    NoiseSpread = deco.NoiseSpread,
                    _colorPad = 0
                });

                bool hasControlMap = deco.ControlMap != null && controlMapToSlice.ContainsKey(deco.ControlMap);
                float logMaxDist = lodCount > 0 ? lodTable[^1].MaxDistance : 0;
                Debug.Log($"[Deco] Slot {slots.Count - 1}: mode={mode} lodCount={lodCount} density={deco.Density} controlMap={hasControlMap} slice={slots[^1].DecoMapSlice} maxDist={logMaxDist:F0}");
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
            cmd.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            _decoPrepassCS.SetTexture("DecoMaps", DecoMapsArray);             // Texture → BindlessIndex
            _decoPrepassCS.SetBuffer("Slots", _decoratorSlotsBuffer!);        // GraphicsBuffer → auto SRV
            _decoPrepassCS.SetPushConstant("ControlUAV", _decoControlUAV);   // uint (raw UAV index)
            _decoPrepassCS.SetPushConstant("SlotCount", slotCount);           // uint
            _decoPrepassCS.Dispatch(0, cmd, (uint)((width + 7) / 8), (uint)((height + 7) / 8));

            cmd.ResourceBarrierUnorderedAccessView(_decoControlTex);
            // Transition from UAV → SRV so the spawn shader and terrain debug overlay can read correctly
            cmd.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_decoControlTex,
                    ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)));

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
            cmd.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            _albedoBakeCS.SetTexture("ControlMaps", ControlMapsArray);    // Texture → BindlessIndex
            _albedoBakeCS.SetTexture("DiffuseMaps", DiffuseMapsArray);    // Texture → BindlessIndex
            _albedoBakeCS.SetPushConstant("OutputUAV", _bakedAlbedoUAV); // uint (raw UAV index)
            _albedoBakeCS.SetBuffer("TilingBuf", _tilingBuffer);         // GraphicsBuffer → auto SRV

            uint groups = (uint)((BakedAlbedoSize + 7) / 8);
            _albedoBakeCS.Dispatch(0, cmd, groups, groups);

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
                    TextureIdx = slotIdx < existingTextureIdx.Length ? existingTextureIdx[slotIdx] : 0,
                    HealthyColor = new Vector3(deco.HealthyColor.X, deco.HealthyColor.Y, deco.HealthyColor.Z),
                    DryColor = new Vector3(deco.DryColor.X, deco.DryColor.Y, deco.DryColor.Z),
                    NoiseSpread = deco.NoiseSpread,
                    _colorPad = 0
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

        /// <summary>
        /// Creates GPU buffers for the compute prepass pipeline.
        /// Called once; worst-case sizing based on control texture dimensions.
        /// </summary>
        private void CreateDecoBuffers()
        {
            if (_decoBuffersCreated) return;
            if (_decoControlTex == null) return;

            var decoDesc = _decoControlTex.Description;
            int controlW = (int)decoDesc.Width;
            int controlH = (int)decoDesc.Height;
            int maxTiles = controlW * controlH;

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

            Debug.Log($"[TerrainRenderer] Deco buffers created: maxTiles={maxTiles} maxInstances={_maxDecoInstances} controlSize={controlW}x{controlH}");

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
            CreateDecoBuffers(); // ensures _grassCS is initialized
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
            if (DecoratorMaterial?.Effect == null || Terrain == null) return;

            // One-time bakes
            BuildDecorationControlTexture(commandList);
            BakeTerrainAlbedo(commandList);
            BakeTerrainNormals(commandList);
            CreateDecoBuffers();

            if (!_decoBuffersCreated || _decoControlTex == null) return;

            var device = Engine.Device;
            var camPos = Camera.Main!.Position;
            var cs = _grassCS!;

            float range = Terrain.DecorationRadius;
            var decoDesc = _decoControlTex.Description;
            int controlW = (int)decoDesc.Width;
            int controlH = (int)decoDesc.Height;
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
                cs.SetPushConstant("MeshRegistry", MeshRegistry.SrvIndex);
                if (Terrain.Heightmap != null) cs.SetTexture("Heightmap", Terrain.Heightmap);
                cs.SetPushConstant("DecoControl", _decoControlSRV);
                cs.SetPushConstant("BakedNormal", _bakedNormalSRV);
                if (DecoMapsArray != null) cs.SetTexture("DecoMaps", DecoMapsArray);

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
                cs.SetParam("SlotCount", (uint)Terrain.Decorations.Count);
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
                    _meshDrawCountBuffer.Transition(commandList, ResourceStates.UnorderedAccess);
                }
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
                int grassShadowCascades = Math.Max(1, DirectionalLight.CascadeCount - 1);
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
                    device.BindlessCommandSignature,
                    32,  // max 32 mesh types
                    _meshDrawArgsBuffer!.Native,
                    0,
                    _meshDrawCountBuffer!.Native,  // count buffer: actual number of draws
                    0);
            }
        }
    }
}
