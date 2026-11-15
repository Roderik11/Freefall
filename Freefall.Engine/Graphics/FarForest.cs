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
    /// The far forest: every tree past its hand-over (<see cref="FarTier"/> profiles), of every owner that keeps one, drawn
    /// as the grass is, from one set of GPU buffers. Each frame one compute pass (far_forest_cull.hlsl) keeps the cells on
    /// the far side of their band, in view and not behind last frame's depth (Hi-Z), then puts their trees that are in view,
    /// past their own kind's hand-over and big enough on screen on their kind's list; one indirect draw renders every kind
    /// through gbuffer_crossmesh.fx's FarForest pass: the trees' own cards, shaded as their instances are, dithered against
    /// them over the band. The same for the sun's cascades (Shadow_FarForest), from half way through the band.
    ///
    /// <para><b>Owners</b> (the world cells' generators) give their trees as a block (<see cref="Set"/>, <see cref="Remove"/>).
    /// A block is uploaded into a range of its own in the tree buffer, so a re-run uploads only its own trees; the buffer
    /// is compacted into a larger one when a block finds no room. The cells (each owner's trees by far tier cell) and the
    /// kinds are small tables made again on any change. A kind is a mesh part (a tree's coarsest LOD, its cards), its
    /// material and the profile it hands over at, shared by every owner that draws it; at most <see cref="MaxKinds"/>.</para>
    ///
    /// Nothing per tree lives on the CPU or in the engine's instance batches: no transform slot, no draw record. Every
    /// kind's material must be of gbuffer_crossmesh. Call <see cref="Draw"/> once per view while it should draw.
    /// </summary>
    public sealed class FarForest : IDisposable
    {
        public struct Tree
        {
            public Vector3 Position;
            public float Yaw;            // radians about up
            public float ScaleAcross;    // 0..3.175, in steps of 0.025
            public float ScaleUp;
            public int Kind;             // into the kinds given with it
        }

        /// <summary>A kind of tree: a mesh part (its cards), its material and the far tier profile it hands over at.</summary>
        public readonly record struct Kind(Mesh Mesh, int Part, Material Material, int Profile);

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
            public uint Profile, Pad1, Pad2;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct GpuCell   // far_forest.hlsli FarCell (48 bytes)
        {
            public Vector3 Min;
            public uint First;
            public Vector3 Max;
            public uint Count;
            public Vector2 Centre;
            public uint Profile, Pad;   // its trees' nearest hand-over: what the cell cull goes by
        }

        /// <summary>The most trees it holds: a shadow list entry keeps 22 bits for the tree (far_forest.hlsli).</summary>
        public const int MaxTrees = 1 << 22;

        /// <summary>The most kinds: a tree keeps 8 bits for its kind (far_forest.hlsli).</summary>
        public const int MaxKinds = 256;

        private const float ScaleStep = 0.025f;
        private const int ShadowEntriesPerTree = 2;   // room per tree in the shadow lists (a tree in more cascades loses the rest)
        private const int CommandUints = 12;          // IndexedCommandSignature: 2 constants + index buffer view + DrawIndexed

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

        private sealed class KindSlot
        {
            public Kind Kind;
            public int Refs;        // owners drawing it
            public int Trees;       // their trees of it
            public GpuKind Gpu;     // without its list bases
        }

        private sealed class Block
        {
            public int First = -1;  // in the tree buffer; -1 = not placed yet
            public int Count;
            public GpuCell[] Cells = Array.Empty<GpuCell>();   // First relative to the block
            public int[] Slots = Array.Empty<int>();           // the kinds it draws
            public int[] SlotTrees = Array.Empty<int>();       // its trees of each
            public GraphicsBuffer? Upload;                     // its trees, until copied into place
        }

        private readonly float _cellSize;
        private readonly KindSlot?[] _slots = new KindSlot?[MaxKinds];
        private readonly Dictionary<Kind, int> _slotOf = new();
        private readonly Dictionary<object, Block> _blocks = new();
        private readonly List<(int First, int Count)> _free = new();   // gaps in the tree buffer, by First
        private int _capacity;      // trees the tree buffer holds
        private int _commands;      // draw commands: the highest kind slot in use + 1
        private int _cellCount, _treeCount;
        private bool _dirty;
        private bool _warnedFull;

        private GraphicsBuffer? _trees, _kindsBuffer, _cells, _visible, _args, _shadow, _shadowArgs;
        private GraphicsBuffer? _cellList, _cellArgs, _shadowCellList, _shadowCellArgs;
        private int _cellsCapacity, _listCapacity, _cellListCapacity;
        private Matrix4x4 _lastViewProjection;
        private bool _hasLastView;

        private static ComputeShader? s_cull;
        private static int s_resetCamera, s_cullCells, s_cullTrees, s_resetShadow, s_cullShadowCells, s_cullShadowTrees, s_clampShadow;

        /// <param name="cellSize">The far tier's cell (<see cref="FarTier.CellSize"/>): the trees are culled and faded by it.</param>
        public FarForest(float cellSize)
        {
            _cellSize = cellSize;
        }

        /// <summary>Trees it holds, of every owner.</summary>
        public int TreeCount => _treeCount;

        /// <summary>Kinds in use.</summary>
        public int KindCount { get; private set; }

        /// <summary>Cells with trees, of every owner (an owner's trees in one far tier cell).</summary>
        public int CellCount => _cellCount;

        /// <summary>Owners with trees in it.</summary>
        public int OwnerCount => _blocks.Count;

        /// <summary>The GPU memory it holds: trees (with room to spare), kinds, cells, lists and draw commands.</summary>
        public long VramBytes =>
            _capacity * 16L + (_kindsBuffer != null ? MaxKinds * 64L + 2L * MaxKinds * CommandUints * 4 : 0)
            + _cellsCapacity * 48L + _listCapacity * (4L + ShadowEntriesPerTree * 4L) + 2L * _cellListCapacity * 4 + 24;

        /// <summary>The trees an owner has in it.</summary>
        public int TreesOf(object owner) => _blocks.TryGetValue(owner, out var b) ? b.Count : 0;

        /// <summary>
        /// The owner's trees, in place of what it had: <paramref name="trees"/>' kinds index <paramref name="kinds"/>.
        /// Trees of a kind that finds no room among the <see cref="MaxKinds"/> are left out (a warning, once). Main thread.
        /// </summary>
        public void Set(object owner, IReadOnlyList<Kind> kinds, ReadOnlySpan<Tree> trees)
        {
            // Its kinds first (before letting go of the old ones, so a kind it keeps keeps its slot and its textures)
            var slots = new int[kinds.Count];
            for (int k = 0; k < kinds.Count; k++) slots[k] = AcquireSlot(kinds[k]);
            if (_blocks.TryGetValue(owner, out var old)) Release(old);
            _blocks.Remove(owner);

            int count = 0;
            foreach (var t in trees) if (t.Kind >= 0 && t.Kind < slots.Length && slots[t.Kind] >= 0) count++;
            if (count > MaxTrees - (_treeCount))
            {
                Debug.LogWarning("FarForest", $"{_treeCount + count:N0} far trees, more than the far forest holds ({MaxTrees:N0}); '{owner}' has no far trees");
                count = 0;
            }
            if (count == 0)
            {
                foreach (int s in slots) if (s >= 0) ReleaseSlot(s);
                _dirty = true;
                if (_blocks.Count == 0) FreeBuffers();
                return;
            }

            // The trees by cell: each cell's trees together, and its box round their spheres
            var order = new int[count];
            var keys = new (int X, int Z)[trees.Length];
            int n0 = 0;
            for (int i = 0; i < trees.Length; i++)
            {
                var t = trees[i];
                if (t.Kind < 0 || t.Kind >= slots.Length || slots[t.Kind] < 0) continue;
                order[n0++] = i;
                keys[i] = ((int)MathF.Floor(t.Position.X / _cellSize), (int)MathF.Floor(t.Position.Z / _cellSize));
            }
            Array.Sort(order, (a, b) => keys[a].X != keys[b].X ? keys[a].X.CompareTo(keys[b].X) : keys[a].Z.CompareTo(keys[b].Z));

            var block = new Block { Count = count, Slots = slots, SlotTrees = new int[slots.Length] };
            var cells = new List<GpuCell>();
            var cellFrom = new List<float>();
            block.Upload = GraphicsBuffer.CreateUpload<GpuTree>(count, mapped: true);
            unsafe
            {
                var tp = block.Upload.WritePtr<GpuTree>();
                for (int n = 0; n < count; n++)
                {
                    var t = trees[order[n]];
                    var key = keys[order[n]];
                    int slot = slots[t.Kind];
                    var kind = _slots[slot]!;
                    block.SlotTrees[t.Kind]++;
                    if (n == 0 || keys[order[n - 1]] != key)
                    {
                        cells.Add(new GpuCell
                        {
                            First = (uint)n,
                            Min = new Vector3(float.MaxValue), Max = new Vector3(float.MinValue),
                            Centre = new Vector2((key.X + 0.5f) * _cellSize, (key.Z + 0.5f) * _cellSize),
                            Profile = (uint)kind.Kind.Profile,
                        });
                        cellFrom.Add(FarTier.FromOf(kind.Kind.Profile));
                    }
                    var cell = cells[^1];
                    float from = FarTier.FromOf(kind.Kind.Profile);
                    if (from < cellFrom[^1]) { cellFrom[^1] = from; cell.Profile = (uint)kind.Kind.Profile; }
                    float scale = MathF.Max(t.ScaleAcross, t.ScaleUp);
                    var centre = t.Position + new Vector3(0, kind.Gpu.CentreY * scale, 0);
                    var reach = new Vector3(kind.Gpu.Radius * scale);
                    cell.Min = Vector3.Min(cell.Min, centre - reach);
                    cell.Max = Vector3.Max(cell.Max, centre + reach);
                    cell.Count++;
                    cells[^1] = cell;

                    float turns = t.Yaw / (2f * MathF.PI);
                    turns -= MathF.Floor(turns);
                    uint yaw = (uint)(turns * 1024f + 0.5f) & 0x3FFu;
                    uint across = (uint)Math.Clamp((int)MathF.Round(t.ScaleAcross / ScaleStep), 1, 127);
                    uint up = (uint)Math.Clamp((int)MathF.Round(t.ScaleUp / ScaleStep), 1, 127);
                    tp[n] = new GpuTree { Position = t.Position, Packed = yaw | (across << 10) | (up << 17) | ((uint)slot << 24) };
                }
            }
            block.Cells = cells.ToArray();
            for (int k = 0; k < slots.Length; k++)
            {
                if (slots[k] < 0) continue;
                if (block.SlotTrees[k] == 0) { ReleaseSlot(slots[k]); slots[k] = -1; continue; }   // a kind it gave but placed none of
                _slots[slots[k]]!.Trees += block.SlotTrees[k];
            }
            _blocks[owner] = block;
            _treeCount += count;
            _dirty = true;
        }

        /// <summary>The owner's trees go. Main thread.</summary>
        public void Remove(object owner)
        {
            if (!_blocks.Remove(owner, out var block)) return;
            Release(block);
            _dirty = true;
            if (_blocks.Count == 0) FreeBuffers();
        }

        // Its range back to the free list, its kinds released, its pending upload dropped
        private void Release(Block block)
        {
            if (block.First >= 0) AddFree(block.First, block.Count);
            for (int k = 0; k < block.Slots.Length; k++)
            {
                if (block.Slots[k] < 0) continue;
                _slots[block.Slots[k]]!.Trees -= block.SlotTrees[k];
                ReleaseSlot(block.Slots[k]);
            }
            _treeCount -= block.Count;
            if (block.Upload != null) Engine.Device?.DeferDispose(block.Upload);
            block.Upload = null;
        }

        private int AcquireSlot(Kind kind)
        {
            if (_slotOf.TryGetValue(kind, out int slot))
            {
                _slots[slot]!.Refs++;
                return slot;
            }
            slot = Array.IndexOf(_slots, null);
            if (slot < 0)
            {
                if (!_warnedFull) Debug.LogWarning("FarForest", $"more than {MaxKinds} kinds of far trees; '{kind.Mesh.Name}' has no far form");
                _warnedFull = true;
                return -1;
            }
            int id = kind.Mesh.GetMeshPartId(kind.Part);
            if (id < 0) { kind.Mesh.RegisterMeshParts(); id = kind.Mesh.GetMeshPartId(kind.Part); }
            var part = kind.Mesh.MeshParts[kind.Part];
            var sphere = part.BoundingSphere;
            var indices = kind.Mesh.IndexBufferView;
            var box = part.BoundingBox;
            _slots[slot] = new KindSlot
            {
                Kind = kind,
                Refs = 1,
                Gpu = new GpuKind
                {
                    HalfWidth = 0.5f * MathF.Max(box.Max.X - box.Min.X, box.Max.Z - box.Min.Z),
                    Profile = (uint)kind.Profile,
                    MeshPartId = (uint)id,
                    IndexCount = (uint)part.NumIndices,
                    IndexAddressLo = (uint)indices.BufferLocation,
                    IndexAddressHi = (uint)(indices.BufferLocation >> 32),
                    IndexBytes = indices.SizeInBytes,
                    BaseIndex = (uint)part.BaseIndex,
                    MaterialId = (uint)kind.Material.MaterialID,
                    CentreY = sphere.Y,
                    Radius = sphere.W + MathF.Sqrt(sphere.X * sphere.X + sphere.Z * sphere.Z),
                },
            };
            _slotOf[kind] = slot;
            kind.Material.Acquire();   // its textures on the GPU while it draws (Material.OnDemand)
            return slot;
        }

        private void ReleaseSlot(int slot)
        {
            var s = _slots[slot]!;
            if (--s.Refs > 0) return;
            s.Kind.Material.Release();
            _slotOf.Remove(s.Kind);
            _slots[slot] = null;
        }

        private void AddFree(int first, int count)
        {
            int i = 0;
            while (i < _free.Count && _free[i].First < first) i++;
            _free.Insert(i, (first, count));
            // Merge with the neighbours
            if (i + 1 < _free.Count && _free[i].First + _free[i].Count == _free[i + 1].First)
            {
                _free[i] = (_free[i].First, _free[i].Count + _free[i + 1].Count);
                _free.RemoveAt(i + 1);
            }
            if (i > 0 && _free[i - 1].First + _free[i - 1].Count == _free[i].First)
            {
                _free[i - 1] = (_free[i - 1].First, _free[i - 1].Count + _free[i].Count);
                _free.RemoveAt(i);
            }
        }

        private bool TakeFree(int count, out int first)
        {
            for (int i = 0; i < _free.Count; i++)
            {
                var (f, c) = _free[i];
                if (c < count) continue;
                first = f;
                if (c == count) _free.RemoveAt(i);
                else _free[i] = (f + count, c - count);
                return true;
            }
            first = -1;
            return false;
        }

        /// <summary>Draw it in this view: into the G-buffer, and into the sun's cascades when it casts.</summary>
        public void Draw()
        {
            if (_treeCount == 0 || FarTier.Band <= 0) return;
            foreach (var s in _slots) s?.Kind.Material.Touch();
            CommandBuffer.Enqueue(RenderPass.Opaque, DrawCamera);
            if (CastShadows) CommandBuffer.Enqueue(RenderPass.Shadow, DrawShadow);
        }

        // What changed reaches the GPU: blocks into place (the buffer compacted into a larger one when one finds no room),
        // then the cells and the kinds made again, and the lists grown to hold every tree
        private void Sync(ID3D12GraphicsCommandList cmd)
        {
            _dirty = false;
            var device = Engine.Device;

            // Room for the blocks not placed yet
            var pending = new List<Block>();
            foreach (var b in _blocks.Values) if (b.First < 0) pending.Add(b);
            bool fits = true;
            foreach (var b in pending)
                if (!TakeFree(b.Count, out b.First)) { fits = false; break; }
            if (!fits || _trees == null)
            {
                // Compact into a new buffer: the placed blocks first, as they are on the GPU, then the new ones
                int live = 0;
                foreach (var b in _blocks.Values) live += b.Count;
                int capacity = Math.Min(MaxTrees, Math.Max(live + live / 4, 1024));
                var trees = GraphicsBuffer.CreateStructured<GpuTree>(capacity, srv: true);
                trees.Transition(cmd, ResourceStates.CopyDest);
                _trees?.Transition(cmd, ResourceStates.CopySource);
                int at = 0;
                foreach (var b in _blocks.Values)
                {
                    if (b.Upload == null && b.First >= 0 && _trees != null)
                        cmd.CopyBufferRegion(trees.Native, (ulong)at * 16, _trees.Native, (ulong)b.First * 16, (ulong)b.Count * 16);
                    b.First = at;
                    at += b.Count;
                }
                if (_trees != null) device.DeferDispose(_trees);
                _trees = trees;
                _capacity = capacity;
                _free.Clear();
                if (at < capacity) _free.Add((at, capacity - at));
            }
            else _trees.Transition(cmd, ResourceStates.CopyDest);
            foreach (var b in _blocks.Values)
            {
                if (b.Upload == null) continue;
                cmd.CopyBufferRegion(_trees.Native, (ulong)b.First * 16, b.Upload.Native, 0, (ulong)b.Count * 16);
                device.DeferDispose(b.Upload);
                b.Upload = null;
            }
            _trees.Transition(cmd, ResourceStates.AllShaderResource);

            // The cells of every block, where its trees now are
            _cellCount = 0;
            foreach (var b in _blocks.Values) _cellCount += b.Cells.Length;
            if (_cellCount > _cellsCapacity || _cells == null)
            {
                if (_cells != null) device.DeferDispose(_cells);
                _cellsCapacity = Math.Max(_cellCount + _cellCount / 4, 256);
                _cells = GraphicsBuffer.CreateStructured<GpuCell>(_cellsCapacity, srv: true);
            }
            if (_cellCount > 0)
            {
                var upload = GraphicsBuffer.CreateUpload<GpuCell>(_cellCount, mapped: true);
                unsafe
                {
                    var cp = upload.WritePtr<GpuCell>();
                    int c = 0;
                    foreach (var b in _blocks.Values)
                        foreach (var cell in b.Cells)
                        {
                            var placed = cell;
                            placed.First += (uint)b.First;
                            cp[c++] = placed;
                        }
                }
                Copy(cmd, _cells, upload, _cellCount * 48);
            }

            // The kinds: each one's lists after the one before, a slot not in use drawing nothing
            KindCount = 0;
            _commands = 0;
            int firstLive = -1;
            for (int s = 0; s < MaxKinds; s++)
                if (_slots[s] != null) { KindCount++; _commands = s + 1; if (firstLive < 0) firstLive = s; }
            _kindsBuffer ??= GraphicsBuffer.CreateStructured<GpuKind>(MaxKinds, srv: true);
            _args ??= GraphicsBuffer.CreateRaw(MaxKinds * CommandUints, uav: true);
            _shadowArgs ??= GraphicsBuffer.CreateRaw(MaxKinds * CommandUints, uav: true);
            _cellArgs ??= GraphicsBuffer.CreateRaw(3, uav: true);
            _shadowCellArgs ??= GraphicsBuffer.CreateRaw(3, uav: true);
            if (_commands > 0)
            {
                var upload = GraphicsBuffer.CreateUpload<GpuKind>(_commands, mapped: true);
                unsafe
                {
                    var kp = upload.WritePtr<GpuKind>();
                    uint visibleBase = 0, shadowBase = 0;
                    for (int s = 0; s < _commands; s++)
                    {
                        var slot = _slots[s];
                        var gpu = (slot ?? _slots[firstLive]!).Gpu;   // a valid index buffer for a slot no tree names
                        int trees = slot?.Trees ?? 0;
                        gpu.VisibleBase = visibleBase;
                        gpu.ShadowBase = shadowBase;
                        gpu.ShadowCapacity = (uint)(trees * ShadowEntriesPerTree);
                        kp[s] = gpu;
                        visibleBase += (uint)trees;
                        shadowBase += (uint)(trees * ShadowEntriesPerTree);
                    }
                }
                Copy(cmd, _kindsBuffer, upload, _commands * 64);
            }

            // The lists: room for every tree, every cell
            if (_treeCount > _listCapacity || _visible == null)
            {
                if (_visible != null) device.DeferDispose(_visible);
                if (_shadow != null) device.DeferDispose(_shadow);
                _listCapacity = Math.Max(_treeCount + _treeCount / 4, 1024);
                _visible = GraphicsBuffer.CreateStructured<uint>(_listCapacity, srv: true, uav: true);
                _shadow = GraphicsBuffer.CreateStructured<uint>(_listCapacity * ShadowEntriesPerTree, srv: true, uav: true);
            }
            if (_cellCount > _cellListCapacity || _cellList == null)
            {
                if (_cellList != null) device.DeferDispose(_cellList);
                if (_shadowCellList != null) device.DeferDispose(_shadowCellList);
                _cellListCapacity = Math.Max(_cellCount + _cellCount / 4, 256);
                _cellList = GraphicsBuffer.CreateStructured<uint>(_cellListCapacity, uav: true);
                _shadowCellList = GraphicsBuffer.CreateStructured<uint>(_cellListCapacity, uav: true);
            }

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

            static void Copy(ID3D12GraphicsCommandList cmd, GraphicsBuffer to, GraphicsBuffer from, long bytes)
            {
                to.Transition(cmd, ResourceStates.CopyDest);
                cmd.CopyBufferRegion(to.Native, 0, from.Native, 0, (ulong)bytes);
                to.Transition(cmd, ResourceStates.AllShaderResource);
                Engine.Device.DeferDispose(from);
            }
        }

        private void BindCull()
        {
            // Push constants by name; a name ending in "Idx" is known without it (ComputeShader's reflection)
            var cs = s_cull!;
            cs.SetPushConstant("Trees", _trees!.SrvIndex);
            cs.SetPushConstant("TreeCount", (uint)_treeCount);
            cs.SetPushConstant("Kinds", _kindsBuffer!.SrvIndex);
            cs.SetPushConstant("KindCount", (uint)_commands);
            cs.SetPushConstant("VisibleUAV", _visible!.UavIndex);
            cs.SetPushConstant("ArgsUAV", _args!.UavIndex);
            cs.SetPushConstant("ShadowUAV", _shadow!.UavIndex);
            cs.SetPushConstant("ShadowArgsUAV", _shadowArgs!.UavIndex);
            cs.SetPushConstant("Cells", _cells!.SrvIndex);
            cs.SetPushConstant("CellCount", (uint)_cellCount);
            cs.SetPushConstant("CellListUAV", _cellList!.UavIndex);
            cs.SetPushConstant("CellArgsUAV", _cellArgs!.UavIndex);
            cs.SetPushConstant("ShadowCellListUAV", _shadowCellList!.UavIndex);
            cs.SetPushConstant("ShadowCellArgsUAV", _shadowCellArgs!.UavIndex);
            cs.SetPushConstant("Profiles", FarTier.ProfilesSrv(Engine.Device));
            cs.SetPushConstant("MinPixels", BitConverter.SingleToUInt32Bits(MinPixels));
            cs.SetPushConstant("ThinPixels", BitConverter.SingleToUInt32Bits(MathF.Max(0f, ThinPixels)));
            cs.SetPushConstant("MaxGrow", BitConverter.SingleToUInt32Bits(Math.Clamp(MaxGrow, 1f, 4f)));

            var camera = Camera.Main;
            var proj = camera.Projection;
            float height = Math.Max(camera.Target?.Height ?? 1, 1);
            cs.SetParam("ViewProjection", camera.ViewProjection);
            cs.SetParam("CamPos", camera.Position);
            cs.SetParam("ClipSlackX", MathF.Abs(proj.M11) + 1f);
            cs.SetParam("ClipSlackY", MathF.Abs(proj.M22) + 1f);
            cs.SetParam("PixelsPerMetre", 0.5f * height * MathF.Abs(proj.M22));
            float shadowWidth = DeferredRenderer.Current?.ShadowTextureArray?.Width ?? 2048f;
            cs.SetParam("MinCasterClip", MathF.Max(0f, Engine.Settings.ShadowMinCasterTexels) * 2f / shadowWidth);

            // Last frame's depth, as the grass's spawn reads it (TerrainRenderer: the pyramid and the view it was drawn with)
            var pyramid = DeferredRenderer.Current?.HiZPyramid;
            bool hiZ = _hasLastView && pyramid != null && pyramid.FullSRV != 0 && pyramid.Ready && !Engine.Settings.DisableHiZ;
            cs.SetParam("OcclusionProjection", _lastViewProjection);
            cs.SetParam("HiZTexture", hiZ ? pyramid!.FullSRV : 0u);
            cs.SetParam("HiZSize", hiZ ? new Vector2(pyramid!.Width, pyramid.Height) : Vector2.One);
            cs.SetParam("HiZMipCount", hiZ ? (uint)pyramid!.MipCount : 1u);
        }

        // In the G-buffer pass, before its batches: pick the cells, then their trees, then draw them
        private void DrawCamera(ID3D12GraphicsCommandList cmd)
        {
            if (Camera.Main == null || _treeCount == 0) return;
            var device = Engine.Device;
            if (_dirty || _trees == null) Sync(cmd);
            if (_commands == 0 || _cellCount == 0) return;
            PixMarker.Begin(cmd, "Far forest cull");

            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            BindCull();
            _args!.Transition(cmd, ResourceStates.UnorderedAccess);
            _visible!.Transition(cmd, ResourceStates.UnorderedAccess);
            _cellArgs!.Transition(cmd, ResourceStates.UnorderedAccess);
            _cellList!.Transition(cmd, ResourceStates.UnorderedAccess);
            s_cull!.Dispatch(s_resetCamera, cmd, (uint)(_commands + 63) / 64);
            _cellArgs.UAVBarrier(cmd);
            _args.UAVBarrier(cmd);
            s_cull.Dispatch(s_cullCells, cmd, (uint)(_cellCount + 63) / 64);
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
            var material = PassMaterial();
            material.SetPass("FarForest");
            material.Apply(cmd, device);
            BindDraw(cmd, _visible.SrvIndex);
            cmd.ExecuteIndirect(device.IndexedCommandSignature, (uint)_commands, _args.Native, 0, null, 0);
            PixMarker.End(cmd);
        }

        // In the sun's shadow pass, after its batches (the cascades' full-array depth bound)
        private void DrawShadow(ID3D12GraphicsCommandList cmd)
        {
            if (Camera.Main == null || _treeCount == 0 || _dirty || _trees == null || _commands == 0 || _cellCount == 0) return;
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
            s_cull.Dispatch(s_resetShadow, cmd, (uint)(_commands + 63) / 64);
            _shadowCellArgs.UAVBarrier(cmd);
            _shadowArgs.UAVBarrier(cmd);
            s_cull.Dispatch(s_cullShadowCells, cmd, (uint)(_cellCount + 63) / 64);
            _shadowCellList.UAVBarrier(cmd);
            _shadowCellArgs.Transition(cmd, ResourceStates.IndirectArgument);
            s_cull.BindKernel(s_cullShadowTrees, cmd);
            cmd.ExecuteIndirect(device.DispatchSignature, 1, _shadowCellArgs.Native, 0, null, 0);
            _shadowArgs.UAVBarrier(cmd);
            s_cull.Dispatch(s_clampShadow, cmd, (uint)(_commands + 63) / 64);
            _shadowArgs.Transition(cmd, ResourceStates.IndirectArgument);
            _shadow.Transition(cmd, ResourceStates.NonPixelShaderResource);
            PixMarker.End(cmd);

            PixMarker.Begin(cmd, "Far forest draw");
            var material = PassMaterial();
            material.SetPass("Shadow_FarForest");
            material.Apply(cmd, device);
            BindDraw(cmd, _shadow.SrvIndex);
            cmd.SetGraphicsRoot32BitConstant(0, cascades, 21);   // CascadeBufferSRVIdx
            cmd.ExecuteIndirect(device.IndexedCommandSignature, (uint)_commands, _shadowArgs.Native, 0, null, 0);
            PixMarker.End(cmd);
        }

        // The pass's state comes from any kind's material: they are all gbuffer_crossmesh, each kind's own through its id
        private Material PassMaterial()
        {
            foreach (var s in _slots) if (s != null) return s.Kind.Material;
            throw new InvalidOperationException("no kinds");
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

        // Nothing left: the GPU memory goes (the next owner makes it again)
        private void FreeBuffers()
        {
            var device = Engine.Device;
            foreach (var b in new[] { _trees, _kindsBuffer, _cells, _visible, _args, _shadow, _shadowArgs, _cellList, _cellArgs, _shadowCellList, _shadowCellArgs })
                if (b != null) device?.DeferDispose(b);
            _trees = _kindsBuffer = _cells = _visible = _args = _shadow = _shadowArgs = null;
            _cellList = _cellArgs = _shadowCellList = _shadowCellArgs = null;
            _capacity = _cellsCapacity = _listCapacity = _cellListCapacity = 0;
            _cellCount = _commands = 0;
            KindCount = 0;
            _free.Clear();
            _hasLastView = false;
        }

        public void Dispose()
        {
            foreach (var owner in new List<object>(_blocks.Keys)) Remove(owner);
            FreeBuffers();
        }
    }
}
