using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Components;
using Vortice.Direct3D;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// A far forest: trees past the far tier's start (<see cref="FarTier"/>) drawn as the grass is. The trees are a static
    /// GPU buffer of 16 bytes each (position, yaw, two scales, kind), held together by far tier cell. Each frame a compute
    /// pass (far_forest_cull.hlsl) keeps the cells on the far side of the band, in view and not behind last frame's depth
    /// (Hi-Z), then puts their trees that are in view and big enough on screen on their kind's list; one indirect draw
    /// renders every kind through gbuffer_crossmesh.fx's FarForest pass: the trees' own cards, shaded as their instances
    /// are, dithered against them over the band. The same for the sun's cascades (Shadow_FarForest), from half way
    /// through the band.
    ///
    /// Nothing per tree lives on the CPU or in the engine's instance batches: no transform slot, no draw record. A kind
    /// is a mesh part (a tree's coarsest LOD, its cards) and its material; every kind's material must be of
    /// gbuffer_crossmesh. Call <see cref="Draw"/> every frame it should draw; <see cref="Dispose"/> when done.
    /// </summary>
    public sealed class FarForest : IDisposable
    {
        public struct Tree
        {
            public Vector3 Position;
            public float Yaw;            // radians about up
            public float ScaleAcross;    // 0..3.175, in steps of 0.025
            public float ScaleUp;
            public int Kind;             // 0..63
        }

        public sealed class Kind
        {
            public Mesh Mesh = null!;
            public int Part;
            public Material Material = null!;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct GpuTree   // far_forest.hlsli FarTree
        {
            public Vector3 Position;
            public uint Packed;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct GpuKind   // far_forest.hlsli FarKind (64 bytes)
        {
            public uint MeshPartId, IndexCount, VisibleBase, ShadowBase, ShadowCapacity, MaterialId;
            public float CentreY, Radius;
            public uint IndexAddressLo, IndexAddressHi, IndexBytes, BaseIndex;
            public float HalfWidth;
            public uint Pad0, Pad1, Pad2;
        }

        /// <summary>The most trees one forest holds: a shadow list entry keeps 22 bits for the tree (far_forest.hlsli).</summary>
        public const int MaxTrees = 1 << 22;

        [StructLayout(LayoutKind.Sequential)]
        private struct GpuCell   // far_forest.hlsli FarCell (48 bytes)
        {
            public Vector3 Min;
            public uint First;
            public Vector3 Max;
            public uint Count;
            public Vector2 Centre;
            public uint Pad0, Pad1;
        }

        private const float ScaleStep = 0.025f;
        private const int ShadowEntriesPerTree = 2;   // room per tree in the shadow lists (a tree in more cascades loses the rest)
        private const int CommandUints = 12;          // IndexedCommandSignature: 2 constants + index buffer view + DrawIndexed

        public int TreeCount { get; }
        public int KindCount { get; }
        public int CellCount { get; }
        /// <summary>The GPU memory it holds: trees, kinds, cells, lists and draw commands.</summary>
        public long VramBytes { get; }
        /// <summary>A tree whose bounding sphere's radius is under this many pixels is not drawn.</summary>
        public float MinPixels = 1f;
        public bool CastShadows = true;

        /// <summary>
        /// Thinning, as the grass's plants: a tree whose crown is narrower on screen than this many pixels is kept with the
        /// share (width / ThinPixels)², so the trees per bit of screen stay the same however far, and the kept ones grow to
        /// cover what the others did (far_forest.hlsli). Which trees stay is fixed by where they stand. 0 = never, the
        /// default: measured, it made the far forest dearer (its trees cost their pixels, which thinning keeps; vram.md).
        /// </summary>
        public float ThinPixels = 0f;

        /// <summary>How far a kept tree may grow (at most 4); past it the canopy thins out.</summary>
        public float MaxGrow = 2f;

        private readonly List<Kind> _kinds;
        private readonly Material _material;
        private GraphicsBuffer? _trees, _kindsBuffer, _cells, _visible, _args, _shadow, _shadowArgs;
        private GraphicsBuffer? _cellList, _cellArgs, _shadowCellList, _shadowCellArgs;
        private GraphicsBuffer? _treesUpload, _kindsUpload, _cellsUpload;
        private bool _uploaded, _disposed;
        private Matrix4x4 _lastViewProjection;
        private bool _hasLastView;

        private static ComputeShader? s_cull;
        private static int s_resetCamera, s_cullCells, s_cullTrees, s_resetShadow, s_cullShadowCells, s_cullShadowTrees, s_clampShadow;

        /// <param name="cellSize">The far tier's cell (<see cref="FarTier.CellSize"/>): the trees are culled and faded by it.</param>
        public FarForest(IReadOnlyList<Kind> kinds, ReadOnlySpan<Tree> trees, float cellSize)
        {
            if (kinds.Count == 0 || kinds.Count > 64) throw new ArgumentException("1 to 64 kinds", nameof(kinds));
            if (trees.Length > MaxTrees) throw new ArgumentException($"at most {MaxTrees} trees", nameof(trees));
            _kinds = new List<Kind>(kinds);
            _material = kinds[0].Material;
            TreeCount = trees.Length;
            KindCount = kinds.Count;

            // Each kind's lists in one buffer each: its share of the trees
            var perKind = new int[KindCount];
            foreach (var t in trees) perKind[t.Kind]++;
            var gpuKinds = new GpuKind[KindCount];
            uint visibleBase = 0, shadowBase = 0;
            for (int k = 0; k < KindCount; k++)
            {
                var kind = kinds[k];
                int id = kind.Mesh.GetMeshPartId(kind.Part);
                if (id < 0) { kind.Mesh.RegisterMeshParts(); id = kind.Mesh.GetMeshPartId(kind.Part); }
                var part = kind.Mesh.MeshParts[kind.Part];
                var sphere = part.BoundingSphere;
                var indices = kind.Mesh.IndexBufferView;
                var box = part.BoundingBox;
                gpuKinds[k] = new GpuKind
                {
                    HalfWidth = 0.5f * MathF.Max(box.Max.X - box.Min.X, box.Max.Z - box.Min.Z),
                    MeshPartId = (uint)id,
                    IndexCount = (uint)part.NumIndices,
                    IndexAddressLo = (uint)indices.BufferLocation,
                    IndexAddressHi = (uint)(indices.BufferLocation >> 32),
                    IndexBytes = indices.SizeInBytes,
                    BaseIndex = (uint)part.BaseIndex,
                    VisibleBase = visibleBase,
                    ShadowBase = shadowBase,
                    ShadowCapacity = (uint)(perKind[k] * ShadowEntriesPerTree),
                    MaterialId = (uint)kind.Material.MaterialID,
                    CentreY = sphere.Y,
                    Radius = sphere.W + MathF.Sqrt(sphere.X * sphere.X + sphere.Z * sphere.Z),
                };
                visibleBase += (uint)perKind[k];
                shadowBase += (uint)(perKind[k] * ShadowEntriesPerTree);
                kind.Material.Acquire();   // its textures on the GPU while it draws (Material.OnDemand)
            }

            // The trees by cell: each cell's trees together, and its box round their spheres
            var order = new int[trees.Length];
            var keys = new (int X, int Z)[trees.Length];
            for (int i = 0; i < trees.Length; i++)
            {
                order[i] = i;
                keys[i] = ((int)MathF.Floor(trees[i].Position.X / cellSize), (int)MathF.Floor(trees[i].Position.Z / cellSize));
            }
            Array.Sort(order, (a, b) => keys[a].X != keys[b].X ? keys[a].X.CompareTo(keys[b].X) : keys[a].Z.CompareTo(keys[b].Z));
            var cells = new List<GpuCell>();

            int count = Math.Max(TreeCount, 1);
            _treesUpload = GraphicsBuffer.CreateUpload<GpuTree>(count, mapped: true);
            _kindsUpload = GraphicsBuffer.CreateUpload<GpuKind>(KindCount, mapped: true);
            unsafe
            {
                var tp = _treesUpload.WritePtr<GpuTree>();
                for (int n = 0; n < order.Length; n++)
                {
                    var t = trees[order[n]];
                    var key = keys[order[n]];
                    if (cells.Count == 0 || n == 0 || keys[order[n - 1]] != key)
                        cells.Add(new GpuCell
                        {
                            First = (uint)n,
                            Min = new Vector3(float.MaxValue), Max = new Vector3(float.MinValue),
                            Centre = new Vector2((key.X + 0.5f) * cellSize, (key.Z + 0.5f) * cellSize),
                        });
                    var cell = cells[^1];
                    var sphere = gpuKinds[t.Kind];
                    float scale = MathF.Max(t.ScaleAcross, t.ScaleUp);
                    var centre = t.Position + new Vector3(0, sphere.CentreY * scale, 0);
                    var reach = new Vector3(sphere.Radius * scale);
                    cell.Min = Vector3.Min(cell.Min, centre - reach);
                    cell.Max = Vector3.Max(cell.Max, centre + reach);
                    cell.Count++;
                    cells[^1] = cell;

                    float turns = t.Yaw / (2f * MathF.PI);
                    turns -= MathF.Floor(turns);
                    uint yaw = (uint)(turns * 4096f + 0.5f) & 0xFFFu;
                    uint across = (uint)Math.Clamp((int)MathF.Round(t.ScaleAcross / ScaleStep), 1, 127);
                    uint up = (uint)Math.Clamp((int)MathF.Round(t.ScaleUp / ScaleStep), 1, 127);
                    tp[n] = new GpuTree { Position = t.Position, Packed = yaw | (across << 12) | (up << 19) | ((uint)t.Kind << 26) };
                }
                var kp = _kindsUpload.WritePtr<GpuKind>();
                for (int k = 0; k < KindCount; k++) kp[k] = gpuKinds[k];
            }
            CellCount = cells.Count;
            int cellCount = Math.Max(CellCount, 1);
            _cellsUpload = GraphicsBuffer.CreateUpload<GpuCell>(cellCount, mapped: true);
            unsafe
            {
                var cp = _cellsUpload.WritePtr<GpuCell>();
                for (int c = 0; c < CellCount; c++) cp[c] = cells[c];
            }

            _trees = GraphicsBuffer.CreateStructured<GpuTree>(count, srv: true);
            _kindsBuffer = GraphicsBuffer.CreateStructured<GpuKind>(KindCount, srv: true);
            _cells = GraphicsBuffer.CreateStructured<GpuCell>(cellCount, srv: true);
            _visible = GraphicsBuffer.CreateStructured<uint>(count, srv: true, uav: true);
            _shadow = GraphicsBuffer.CreateStructured<uint>(count * ShadowEntriesPerTree, srv: true, uav: true);
            _cellList = GraphicsBuffer.CreateStructured<uint>(cellCount, uav: true);
            _shadowCellList = GraphicsBuffer.CreateStructured<uint>(cellCount, uav: true);
            _args = GraphicsBuffer.CreateRaw(KindCount * CommandUints, uav: true);
            _shadowArgs = GraphicsBuffer.CreateRaw(KindCount * CommandUints, uav: true);
            _cellArgs = GraphicsBuffer.CreateRaw(3, uav: true);
            _shadowCellArgs = GraphicsBuffer.CreateRaw(3, uav: true);
            VramBytes = count * 16L + KindCount * 64L + cellCount * 48L + count * 4L + count * ShadowEntriesPerTree * 4L
                        + 2L * cellCount * 4 + 2L * KindCount * CommandUints * 4 + 24;

            if (s_cull == null)
            {
                s_cull = new ComputeShader("far_forest_cull.hlsl");
                s_resetCamera = s_cull.FindKernel("CSResetCamera");
                s_cullCells = s_cull.FindKernel("CSCullCells");
                s_cullTrees = s_cull.FindKernel("CSCullTrees");
                s_resetShadow = s_cull.FindKernel("CSResetShadow");
                s_cullShadowCells = s_cull.FindKernel("CSCullShadowCells");
                s_cullShadowTrees = s_cull.FindKernel("CSCullShadowTrees");
                s_clampShadow = s_cull.FindKernel("CSClampShadow");
            }
        }

        /// <summary>Draw it this frame: into the G-buffer, and into the sun's cascades when it casts.</summary>
        public void Draw()
        {
            if (_disposed || TreeCount == 0 || FarTier.Band <= 0) return;
            foreach (var kind in _kinds) kind.Material.Touch();
            CommandBuffer.Enqueue(RenderPass.Opaque, DrawCamera);
            if (CastShadows) CommandBuffer.Enqueue(RenderPass.Shadow, DrawShadow);
        }

        private void Upload(ID3D12GraphicsCommandList cmd)
        {
            if (_uploaded) return;
            _uploaded = true;
            Copy(cmd, _trees!, _treesUpload!);
            Copy(cmd, _kindsBuffer!, _kindsUpload!);
            Copy(cmd, _cells!, _cellsUpload!);
            _treesUpload = _kindsUpload = _cellsUpload = null;

            static void Copy(ID3D12GraphicsCommandList cmd, GraphicsBuffer to, GraphicsBuffer from)
            {
                to.Transition(cmd, ResourceStates.CopyDest);
                cmd.CopyBufferRegion(to.Native, 0, from.Native, 0, (ulong)to.SizeInBytes);
                to.Transition(cmd, ResourceStates.AllShaderResource);
                Engine.Device.DeferDispose(from);
            }
        }

        private void BindCull()
        {
            // Push constants by name; a name ending in "Idx" is known without it (ComputeShader's reflection)
            var cs = s_cull!;
            cs.SetPushConstant("Trees", _trees!.SrvIndex);
            cs.SetPushConstant("TreeCount", (uint)TreeCount);
            cs.SetPushConstant("Kinds", _kindsBuffer!.SrvIndex);
            cs.SetPushConstant("KindCount", (uint)KindCount);
            cs.SetPushConstant("VisibleUAV", _visible!.UavIndex);
            cs.SetPushConstant("ArgsUAV", _args!.UavIndex);
            cs.SetPushConstant("ShadowUAV", _shadow!.UavIndex);
            cs.SetPushConstant("ShadowArgsUAV", _shadowArgs!.UavIndex);
            cs.SetPushConstant("Cells", _cells!.SrvIndex);
            cs.SetPushConstant("CellCount", (uint)CellCount);
            cs.SetPushConstant("CellListUAV", _cellList!.UavIndex);
            cs.SetPushConstant("CellArgsUAV", _cellArgs!.UavIndex);
            cs.SetPushConstant("ShadowCellListUAV", _shadowCellList!.UavIndex);
            cs.SetPushConstant("ShadowCellArgsUAV", _shadowCellArgs!.UavIndex);

            var camera = Camera.Main;
            var proj = camera.Projection;
            float height = Math.Max(camera.Target?.Height ?? 1, 1);
            cs.SetParam("ViewProjection", camera.ViewProjection);
            cs.SetParam("CamPos", camera.Position);
            cs.SetParam("ClipSlackX", MathF.Abs(proj.M11) + 1f);
            cs.SetParam("ClipSlackY", MathF.Abs(proj.M22) + 1f);
            cs.SetParam("PixelsPerMetre", 0.5f * height * MathF.Abs(proj.M22));
            cs.SetParam("MinPixels", MinPixels);
            cs.SetParam("FarFrom", FarTier.From);
            cs.SetParam("FarBand", FarTier.Band);
            float shadowWidth = DeferredRenderer.Current?.ShadowTextureArray?.Width ?? 2048f;
            cs.SetParam("MinCasterClip", MathF.Max(0f, Engine.Settings.ShadowMinCasterTexels) * 2f / shadowWidth);

            // Last frame's depth, as the grass's spawn reads it (TerrainRenderer: the pyramid and the view it was drawn with)
            var pyramid = DeferredRenderer.Current?.HiZPyramid;
            bool hiZ = _hasLastView && pyramid != null && pyramid.FullSRV != 0 && pyramid.Ready && !Engine.Settings.DisableHiZ;
            cs.SetParam("OcclusionProjection", _lastViewProjection);
            cs.SetParam("HiZTexture", hiZ ? pyramid!.FullSRV : 0u);
            cs.SetParam("HiZSize", hiZ ? new Vector2(pyramid!.Width, pyramid.Height) : Vector2.One);
            cs.SetParam("HiZMipCount", hiZ ? (uint)pyramid!.MipCount : 1u);
            cs.SetParam("ThinPixels", MathF.Max(0f, ThinPixels));
            cs.SetParam("MaxGrow", Math.Clamp(MaxGrow, 1f, 4f));
        }

        // In the G-buffer pass, before its batches: pick the cells, then their trees, then draw them
        private void DrawCamera(ID3D12GraphicsCommandList cmd)
        {
            if (_disposed || Camera.Main == null) return;
            var device = Engine.Device;
            Upload(cmd);
            PixMarker.Begin(cmd, "Far forest cull");

            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            BindCull();
            _args!.Transition(cmd, ResourceStates.UnorderedAccess);
            _visible!.Transition(cmd, ResourceStates.UnorderedAccess);
            _cellArgs!.Transition(cmd, ResourceStates.UnorderedAccess);
            _cellList!.Transition(cmd, ResourceStates.UnorderedAccess);
            s_cull!.Dispatch(s_resetCamera, cmd, (uint)(KindCount + 63) / 64);
            _cellArgs.UAVBarrier(cmd);
            _args.UAVBarrier(cmd);
            s_cull.Dispatch(s_cullCells, cmd, (uint)(CellCount + 63) / 64);
            _cellList.UAVBarrier(cmd);
            _cellArgs.Transition(cmd, ResourceStates.IndirectArgument);
            s_cull.BindKernel(s_cullTrees, cmd);
            cmd.ExecuteIndirect(device.DispatchSignature, 1, _cellArgs.Native, 0, null, 0);
            _args.Transition(cmd, ResourceStates.IndirectArgument);
            _visible.Transition(cmd, ResourceStates.NonPixelShaderResource);
            PixMarker.End(cmd);

            _lastViewProjection = Camera.Main.ViewProjection;
            _hasLastView = true;

            PixMarker.Begin(cmd, "Far forest draw");
            _material.SetPass("FarForest");
            _material.Apply(cmd, device);
            BindDraw(cmd, _visible.SrvIndex);
            cmd.ExecuteIndirect(device.IndexedCommandSignature, (uint)KindCount, _args.Native, 0, null, 0);
            PixMarker.End(cmd);
        }

        // In the sun's shadow pass, after its batches (the cascades' full-array depth bound)
        private void DrawShadow(ID3D12GraphicsCommandList cmd)
        {
            if (_disposed || Camera.Main == null || !_uploaded) return;
            uint cascades = DirectionalLight.CurrentCascadeSrvIndex;
            int cascadeCount = DirectionalLight.CascadeCount;
            if (cascades == 0 || cascadeCount <= 0) return;
            var device = Engine.Device;
            PixMarker.Begin(cmd, "Far forest cull");

            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            BindCull();
            s_cull!.SetPushConstant("Cascade", cascades);
            s_cull.SetPushConstant("CascadeCount", (uint)Math.Min(cascadeCount, 8));
            _shadowArgs!.Transition(cmd, ResourceStates.UnorderedAccess);
            _shadow!.Transition(cmd, ResourceStates.UnorderedAccess);
            _shadowCellArgs!.Transition(cmd, ResourceStates.UnorderedAccess);
            _shadowCellList!.Transition(cmd, ResourceStates.UnorderedAccess);
            s_cull.Dispatch(s_resetShadow, cmd, (uint)(KindCount + 63) / 64);
            _shadowCellArgs.UAVBarrier(cmd);
            _shadowArgs.UAVBarrier(cmd);
            s_cull.Dispatch(s_cullShadowCells, cmd, (uint)(CellCount + 63) / 64);
            _shadowCellList.UAVBarrier(cmd);
            _shadowCellArgs.Transition(cmd, ResourceStates.IndirectArgument);
            s_cull.BindKernel(s_cullShadowTrees, cmd);
            cmd.ExecuteIndirect(device.DispatchSignature, 1, _shadowCellArgs.Native, 0, null, 0);
            _shadowArgs.UAVBarrier(cmd);
            s_cull.Dispatch(s_clampShadow, cmd, (uint)(KindCount + 63) / 64);
            _shadowArgs.Transition(cmd, ResourceStates.IndirectArgument);
            _shadow.Transition(cmd, ResourceStates.NonPixelShaderResource);
            PixMarker.End(cmd);

            PixMarker.Begin(cmd, "Far forest draw");
            _material.SetPass("Shadow_FarForest");
            _material.Apply(cmd, device);
            BindDraw(cmd, _shadow.SrvIndex);
            cmd.SetGraphicsRoot32BitConstant(0, cascades, 21);   // CascadeBufferSRVIdx
            cmd.ExecuteIndirect(device.IndexedCommandSignature, (uint)KindCount, _shadowArgs.Native, 0, null, 0);
            PixMarker.End(cmd);
        }

        // gbuffer_crossmesh.fx VS_FarForest's push constants (the per-draw two come with each command)
        private void BindDraw(ID3D12GraphicsCommandList cmd, uint listSrv)
        {
            cmd.IASetPrimitiveTopology(PrimitiveTopology.TriangleList);
            cmd.SetGraphicsRoot32BitConstant(0, _trees!.SrvIndex, 4);             // the trees
            cmd.SetGraphicsRoot32BitConstant(0, listSrv, 5);                       // the list drawn
            cmd.SetGraphicsRoot32BitConstant(0, MeshRegistry.SrvIndex, 6);
            cmd.SetGraphicsRoot32BitConstant(0, Material.MaterialsBufferIndex, 7);
            cmd.SetGraphicsRoot32BitConstant(0, _kindsBuffer!.SrvIndex, 8);        // the kinds
            cmd.SetGraphicsRoot32BitConstant(0, 1u, 17);                           // IndexedDraw: the commands bind the indices
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var kind in _kinds) kind.Material.Release();
            var device = Engine.Device;
            device?.DeferDispose(_trees);
            device?.DeferDispose(_kindsBuffer);
            device?.DeferDispose(_cells);
            device?.DeferDispose(_visible);
            device?.DeferDispose(_shadow);
            device?.DeferDispose(_args);
            device?.DeferDispose(_shadowArgs);
            device?.DeferDispose(_cellList);
            device?.DeferDispose(_cellArgs);
            device?.DeferDispose(_shadowCellList);
            device?.DeferDispose(_shadowCellArgs);
            device?.DeferDispose(_treesUpload);
            device?.DeferDispose(_kindsUpload);
            device?.DeferDispose(_cellsUpload);
        }
    }
}
