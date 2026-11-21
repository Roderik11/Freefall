using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Base;
using Freefall.Components;
using Vortice.Direct3D;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// The sea's grid as a quadtree round the camera (2026-11-21). The ocean was one 128² grid stretched over its entity's
    /// scale: made for a 1 km scene, it gave TestWorld's 65 km sea 512 m cells, and since no tessellation goes past 64
    /// segments an edge the waves round a ship stood in 10 m facets. Now <see cref="OceanRenderer"/> draws the leaves of a
    /// quadtree over that square, walked each frame on the GPU (ocean_quadtree.hlsl CSWalk): split by distance, so the cells
    /// next to the eye are about <see cref="FinestCell"/> and double with every level outward; a node out of view, behind
    /// the opaque scene (Hi-Z) or over land (the sea mask) is dropped with all it holds before the hull shader runs. Every
    /// leaf is <see cref="LeafCells"/>² cells drawn with one ExecuteIndirect at the head of the forward pass
    /// (ocean_patch.hlsli VS_Leaf), tessellated by size on screen as before; its edges against a coarser neighbour snap to
    /// that neighbour's control points, so the levels meet without cracks. EngineSettings.OceanUniformGrid draws the old
    /// grid through the same path, to compare. Docs: .agent/knowledge/ocean_system/overview.md "The grid".
    /// </summary>
    internal sealed class OceanQuadtree : IDisposable
    {
        /// <summary>Cells a side of a leaf (ocean_leaf.hlsli OCEAN_LEAF_CELLS).</summary>
        public const int LeafCells = 8;

        /// <summary>The leaves' cells next to the eye are at most this (m): tessellated to 48 segments, 4 cm edges.</summary>
        public const float FinestCell = 2f;

        /// <summary>The deepest level a node can sit at (12 bits a coordinate, ocean_leaf.hlsli).</summary>
        public const int MaxDepthCap = 12;

        /// <summary>
        /// A node splits while the camera is nearer to it than this times its size. At least √2, so neighbouring leaves are
        /// never two levels apart (ocean_quadtree.hlsl); 2 puts every cell at least 16 cells from the eye, ~75 px at
        /// 1250 px and 45°: tessellation factors of 4-8 there, low enough for AMD's tessellator.
        /// </summary>
        public const float SplitDistance = 2f;

        /// <summary>How far the waves move the surface, every way (m): ocean_patch.hlsli PatchDisplacementBound.</summary>
        public const float WaveBound = 20f;

        /// <summary>The most leaves drawn (a view at sea level draws a few hundred).</summary>
        public const int MaxLeaves = 4096;

        // The old grid (OceanUniformGrid): 2^4 leaves of 8 cells, 128 a side
        private const int UniformDepth = 4;

        // One entry per view and frame in flight: the walk's parameters and the OceanData the draw reads
        private const int FramesInFlight = 3;
        private const int ViewsPerFrame = 4;
        private const int Slots = FramesInFlight * ViewsPerFrame;

        /// <summary>ocean_quadtree.hlsl OceanWalk, field for field.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct WalkParams
        {
            public Matrix4x4 ViewProjection;
            public Vector3 Camera;
            public float SeaLevel;
            public Vector2 RootMin;
            public float RootSize;
            public uint MaxDepth;
            public Vector4 Extent;
            public Vector4 MaskRect;
            public uint HiZSrv;
            public uint ReachSrv;
            public uint Uniform;
            public uint MaxLeaves;
            public float SplitDistance;
            public float Bound;
            public Vector2 Pad;
        }

        // The sea's reach (ocean_quadtree.hlsl REACH_*): bit levels from 256² over the terrain down to 1
        public const int ReachRes = 256;
        public const int ReachLevels = 9;

        private ComputeShader? _shader;
        private int _kWalk;
        private GraphicsBuffer? _walkParams, _oceanData, _leaves, _args;
        private readonly GraphicsBuffer?[] _readback = new GraphicsBuffer?[FramesInFlight];
        private GraphicsBuffer? _reach;
        private ID3D12DescriptorHeap[]? _heaps;
        private bool _disposed;

        // The entry of this tick's newest Draw, and what its leaves are measured in: the forward pass records the view
        // its Draw prepared before the next view draws (DeferredRenderer.Render: FillGBuffer, then RenderForward)
        private int _slotTick = -1, _viewInTick;
        private int _slot;
        private Vector2 _rootMin;
        private float _finestCell;
        private uint _maxDepth;
        private Material? _material;
        private readonly Action<ID3D12GraphicsCommandList> _record;

        // Where the split is measured from while the frustum is frozen (so the frozen tree can be looked at from outside)
        private Vector3 _frozenCamera;
        private bool _wasFrozen;

        /// <summary>Leaves drawn in a frame read back (a few frames old); <see cref="LeafCells"/>² × 2 patches each.</summary>
        public uint Leaves { get; private set; }

        public OceanQuadtree()
        {
            _record = Record;
        }

        /// <summary>
        /// Draw: this view's walk and the ocean's draw, at the head of the forward pass, with <paramref name="data"/> as the
        /// draw's OceanData. Main thread.
        /// </summary>
        public unsafe void Queue(Material material, in OceanRenderer.OceanData data, Vector3 oceanPosition, Vector3 oceanScale,
                                 Camera camera, TerrainRenderer? terrain)
        {
            if (_disposed) return;
            EnsureResources();

            if (_slotTick != Engine.TickCount) { _slotTick = Engine.TickCount; _viewInTick = 0; }
            else _viewInTick = Math.Min(_viewInTick + 1, ViewsPerFrame - 1);
            _slot = Engine.FrameIndex % FramesInFlight * ViewsPerFrame + _viewInTick;

            // The root: the ocean's square (its entity's scale is the half size), the old grid's -1..1 times the scale
            float halfX = MathF.Abs(oceanScale.X), halfZ = MathF.Abs(oceanScale.Z);
            float rootSize = 2f * MathF.Max(MathF.Max(halfX, halfZ), 1f);
            bool uniform = Engine.Settings.OceanUniformGrid;
            int depth = UniformDepth;
            if (!uniform)
            {
                depth = 0;
                while (depth < MaxDepthCap && rootSize / (LeafCells * (float)(1 << depth)) > FinestCell) depth++;
            }
            _maxDepth = (uint)depth;
            _finestCell = rootSize / (LeafCells * (float)(1 << depth));
            _rootMin = new Vector2(oceanPosition.X - rootSize * 0.5f, oceanPosition.Z - rootSize * 0.5f);
            _material = material;

            // Frozen frustum (EngineSettings.FreezeFrustum): culled and split as it was when frozen, no Hi-Z
            bool frozen = Engine.Settings.FreezeFrustum;
            if (frozen && !_wasFrozen) _frozenCamera = camera.Position;
            _wasFrozen = frozen;

            var walk = new WalkParams
            {
                ViewProjection = frozen ? Engine.Settings.FrozenViewProjection : camera.ViewProjection,
                Camera = frozen ? _frozenCamera : camera.Position,
                SeaLevel = oceanPosition.Y,
                RootMin = _rootMin,
                RootSize = rootSize,
                MaxDepth = _maxDepth,
                Extent = new Vector4(oceanPosition.X - halfX, oceanPosition.Z - halfZ, oceanPosition.X + halfX, oceanPosition.Z + halfZ),
                MaskRect = new Vector4(data.TerrainOriginX, data.TerrainOriginZ, MathF.Max(data.TerrainSizeX, 1f), MathF.Max(data.TerrainSizeZ, 1f)),
                HiZSrv = data.HiZSRV,
                ReachSrv = data.SeaMaskSRV != 0 && _reach != null ? _reach.SrvIndex : 0u,
                Uniform = uniform ? 1u : 0u,
                MaxLeaves = MaxLeaves,
                SplitDistance = SplitDistance,
                Bound = WaveBound,
            };
            _walkParams!.WritePtr<WalkParams>()[_slot] = walk;
            _oceanData!.WritePtr<OceanRenderer.OceanData>()[_slot] = data;

            CommandBuffer.Enqueue(RenderPass.Forward, _record);
        }

        // The forward pass, first thing: the walk, then the leaves drawn by it
        private void Record(ID3D12GraphicsCommandList cmd)
        {
            if (_disposed || _shader == null || _material == null) return;
            var device = Engine.Device;
            int frame = Engine.FrameIndex % FramesInFlight;

            // The leaves this slot's last frame drew (the GPU is done with it: its back buffer came round again)
            if (_readback[frame] is { } readback)
                unsafe { Leaves = *readback.ReadPtr<uint>(); }

            PixMarker.Begin(cmd, "Ocean quadtree");
            cmd.SetDescriptorHeaps(1, _heaps ??= new[] { device.SrvHeap });
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            _leaves!.Transition(cmd, ResourceStates.UnorderedAccess);
            _args!.Transition(cmd, ResourceStates.UnorderedAccess);
            _shader.SetPushConstant(_kWalk, "ParamsSrv", _walkParams!.SrvIndex);
            _shader.SetPushConstant(_kWalk, "ParamsSlot", (uint)_slot);
            _shader.SetBuffer(_kWalk, "LeavesUAV", _leaves);
            _shader.SetBuffer(_kWalk, "ArgsUAV", _args);
            _shader.Dispatch(_kWalk, cmd, 1);
            _args.Transition(cmd, ResourceStates.CopySource);
            if (_readback[frame] is { } target)
                cmd.CopyBufferRegion(target.Native, 0, _args.Native, 4, 4);   // InstanceCount
            _args.Transition(cmd, ResourceStates.IndirectArgument);
            _leaves.Transition(cmd, ResourceStates.NonPixelShaderResource);
            PixMarker.End(cmd);

            // The draw: the ocean's forward pass, every leaf an instance of LeafCells² cells (VS_Leaf)
            var material = _material;
            material.SetPass(RenderPass.Forward);
            if (material.PipelineState?.Native == null) return;   // its effect not compiled yet: nothing drawn
            PixMarker.Begin(cmd, "Ocean");
            material.Apply(cmd, device);
            cmd.IASetPrimitiveTopology(PrimitiveTopology.PatchListWith3ControlPoints);
            var renderer = DeferredRenderer.Current;
            cmd.SetGraphicsRoot32BitConstant(0, renderer?.ShadowMapIndex ?? 0u, 0);
            cmd.SetGraphicsRoot32BitConstant(0, renderer?.CompositeSnapshot?.BindlessIndex ?? 0u, 1);
            cmd.SetGraphicsRoot32BitConstant(0, _oceanData!.SrvIndex, 9);                       // OceanDataIdx
            cmd.SetGraphicsRoot32BitConstant(0, _leaves.SrvIndex, 10);                         // OceanLeavesIdx
            cmd.SetGraphicsRoot32BitConstant(0, (uint)_slot, 11);                              // OceanSlot
            cmd.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(_rootMin.X), 12);
            cmd.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(_rootMin.Y), 13);
            cmd.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(_finestCell), 15);
            cmd.SetGraphicsRoot32BitConstant(0, (uint)Engine.Settings.DebugVisualizationMode, 16);
            cmd.SetGraphicsRoot32BitConstant(0, _maxDepth, 17);                                // OceanMaxDepth
            cmd.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(
                Math.Clamp(Engine.Settings.OceanPixelsPerEdge, 4f, 64f)), 18);                  // OceanPixelsPerEdge
            cmd.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(
                Math.Clamp(Engine.Settings.OceanFineDistance, 0f, 2000f)), 19);                 // OceanFineDistance
            cmd.ExecuteIndirect(device.DrawInstancedSignature, 1, _args.Native, 0, null, 0);
            PixMarker.End(cmd);
        }

        /// <summary>
        /// The sea's reach from a sea mask, on a worker: a bit per cell of <see cref="ReachRes"/>² over the terrain where any
        /// sea lies in that cell or the next (the pixel shader reads the mask bilinear), then levels of 2 x 2 down to one,
        /// packed one after the other (ocean_quadtree.hlsl SeaWithin).
        /// </summary>
        public static uint[] BuildReach(SeaMask.Result mask)
        {
            int size = mask.Size;
            var cells = mask.Cells;
            var sea = new bool[ReachRes * ReachRes];
            if (size > 1 && cells.Length >= size * size)
            {
                // Mask cell i lies at u = i / (size - 1) (the terrain's edges at its first and last cells)
                var column = new int[size];
                for (int i = 0; i < size; i++)
                    column[i] = Math.Min(i * ReachRes / (size - 1), ReachRes - 1);
                for (int j = 0; j < size; j++)
                {
                    int row = column[j] * ReachRes, from = j * size;
                    for (int i = 0; i < size; i++)
                        if (cells[from + i] >= 128) sea[row + column[i]] = true;
                }
            }

            int total = 0;
            for (int l = 0; l < ReachLevels; l++) total += (ReachRes >> l) * (ReachRes >> l);
            var bits = new uint[(total + 31) / 32];

            // Level 0: the cells with sea and their neighbours
            var level = new bool[ReachRes * ReachRes];
            for (int z = 0; z < ReachRes; z++)
            for (int x = 0; x < ReachRes; x++)
            {
                bool any = false;
                for (int dz = -1; dz <= 1 && !any; dz++)
                for (int dx = -1; dx <= 1 && !any; dx++)
                {
                    int nx = x + dx, nz = z + dz;
                    any = nx >= 0 && nz >= 0 && nx < ReachRes && nz < ReachRes && sea[nz * ReachRes + nx];
                }
                level[z * ReachRes + x] = any;
            }

            int offset = 0;
            for (int l = 0, res = ReachRes; l < ReachLevels; l++, res >>= 1)
            {
                if (l > 0)
                {
                    int finer = res * 2;
                    var coarser = new bool[res * res];
                    for (int z = 0; z < res; z++)
                    for (int x = 0; x < res; x++)
                        coarser[z * res + x] = level[2 * z * finer + 2 * x] || level[2 * z * finer + 2 * x + 1]
                                            || level[(2 * z + 1) * finer + 2 * x] || level[(2 * z + 1) * finer + 2 * x + 1];
                    level = coarser;
                }
                for (int i = 0; i < res * res; i++)
                    if (level[i]) bits[(offset + i) >> 5] |= 1u << ((offset + i) & 31);
                offset += res * res;
            }
            return bits;
        }

        /// <summary>The reach of a new sea mask (BuildReach), for the walks from now on. Main thread.</summary>
        public unsafe void UploadReach(uint[] bits)
        {
            if (_disposed) return;
            var buffer = GraphicsBuffer.CreateUpload<uint>(bits.Length, mapped: true);
            var dst = buffer.WritePtr<uint>();
            for (int i = 0; i < bits.Length; i++) dst[i] = bits[i];
            Engine.Device.DeferDispose(_reach);   // the frames in flight may still walk with it
            _reach = buffer;
        }

        private void EnsureResources()
        {
            if (_shader != null) return;
            _shader = new ComputeShader("ocean_quadtree.hlsl");
            _kWalk = _shader.FindKernel("CSWalk");
            _walkParams = GraphicsBuffer.CreateUpload<WalkParams>(Slots, mapped: true);
            _oceanData = GraphicsBuffer.CreateUpload<OceanRenderer.OceanData>(Slots, mapped: true);
            _leaves = GraphicsBuffer.CreateStructured<uint>(MaxLeaves, srv: true, uav: true);
            _args = GraphicsBuffer.CreateRaw(4, uav: true);
            for (int i = 0; i < FramesInFlight; i++)
                _readback[i] = GraphicsBuffer.CreateReadback<uint>(1);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            // Deferred: the frames in flight still walk and draw with them
            var device = Engine.Device;
            device.DeferDispose(_shader);
            device.DeferDispose(_walkParams);
            device.DeferDispose(_oceanData);
            device.DeferDispose(_leaves);
            device.DeferDispose(_args);
            device.DeferDispose(_reach);
            foreach (var r in _readback) device.DeferDispose(r);
        }
    }
}
