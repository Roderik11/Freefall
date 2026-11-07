using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// A terrain's heights as pages in one Texture2DArray, the height pool (TERRAIN_PAGES_PLAN.md stages 4a and 5a).
    ///
    /// A page is <see cref="Assets.Terrain.PageCells"/> (256) cells a side and holds 257 samples, its cells and the edge
    /// it shares with the next page, inside a border of 2: <see cref="PageTexels"/> (261) texels. Four pages to a slice,
    /// 2 × 2; a page sits at a slot (slice × 4 + quarter). A level L page holds every 2^L-th level 0 sample over 2^L × 2^L
    /// level 0 pages (copies, so it agrees with level 0 on its own grid). Which page is in which slot, and the page table
    /// shaders read, are <see cref="PageSlots"/>'s. Shaders read the pool only through terrain_pages.hlsli. Every pool and
    /// table is reached by one descriptor each, the same in every lane: the slice and the place in it are plain data, so
    /// no read is non-uniform (AMD).
    ///
    /// Two layouts (<see cref="Ensure"/>): every level 0 page resident at its own index (capacity 0, as in stage 4), or
    /// streaming (TerrainHeightResidency): a fixed number of slots for the pages round the camera at each level, plus
    /// <see cref="ScratchSlots"/> the readback bakes owed pages in.
    ///
    /// <see cref="WriteParams"/> fills this frame's <c>TerrainPagesParams</c>, the one index shaders are handed
    /// (<c>LoadTerrain</c>).
    /// </summary>
    public sealed class HeightPool : IDisposable
    {
        /// <summary>Texels round a page's samples, so a read near its edge stays in it.</summary>
        public const int Border = 2;
        /// <summary>Samples a page holds a side: its cells and the edge it shares with the next page.</summary>
        public const int PageSamples = Assets.Terrain.PageCells + 1;
        /// <summary>Texels a page takes a side in its slice, border included.</summary>
        public const int PageTexels = PageSamples + 2 * Border;
        /// <summary>Texels a slice has a side: 2 × 2 pages.</summary>
        public const int SliceTexels = 2 * PageTexels;
        /// <summary>Slots a streaming pool keeps aside for pages baked only to be read back (TerrainBaker.ReadbackPages).</summary>
        public const int ScratchSlots = 64;

        private const int FrameCount = 3;

        /// <summary>terrain_pages.hlsli TerrainPagesParams. 64 bytes.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Params
        {
            public uint HeightPool, PageTable, PagesAcross, Cells;
            public Vector3 Origin;
            public float Spacing;
            public Vector2 Size;
            public float MaxHeight;
            public uint Version;
            public uint PaintIdx;   // the terrain's TerrainPaintParams (terrain_paint.hlsli), 0 = none
            public Vector3 Pad;
        }

        /// <summary>Which page is in which slot, and the page table.</summary>
        public PageSlots Slots { get; } = new();

        /// <summary>Samples a side of the terrain the pool was made for (0 before <see cref="Ensure"/>).</summary>
        public int Resolution { get; private set; }

        /// <summary>Level 0 pages a side.</summary>
        public int PagesAcross => Slots.PagesAcross;

        /// <summary>Levels of pages: 1 for <see cref="AllResident"/>, else up to the level of one page.</summary>
        public int Levels => Slots.Levels;

        /// <summary>Slots, scratch included.</summary>
        public int Capacity => Texture != null ? Slots.Capacity : 0;

        /// <summary>Slices of the array.</summary>
        public int Slices { get; private set; }

        /// <summary>Every level 0 page is resident at the slot of its own index (no levels, no scratch).</summary>
        public bool AllResident => Slots.AllResident;

        /// <summary>The Texture2DArray (R16_UNorm). Rests in Common between uses.</summary>
        public ID3D12Resource? Texture { get; private set; }

        public uint Srv { get; private set; }
        public uint Uav { get; private set; }

        /// <summary>The page table (StructuredBuffer&lt;uint&gt;), read only.</summary>
        public uint TableSrv => Slots.TableSrv;

        /// <summary>Heights are in it: a bake or an upload has filled it since it was made.</summary>
        public bool Filled { get; set; }

        private readonly GraphicsBuffer?[] _params = new GraphicsBuffer?[FrameCount];

        /// <summary>D3D12's cap on a texture array's slices (D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION).</summary>
        public const int MaxSlices = 2048;

        /// <summary>Array slices a pool of every page takes for a terrain of this many samples a side (four pages a slice).</summary>
        public static int SlicesFor(int resolution)
        {
            int across = PagesAcrossFor(resolution);
            return (across * across + 3) / 4;
        }

        /// <summary>Level 0 pages a side for a terrain of this many samples.</summary>
        public static int PagesAcrossFor(int resolution) =>
            Math.Max(1, (Math.Max(1, resolution - 1) + Assets.Terrain.PageCells - 1) / Assets.Terrain.PageCells);

        /// <summary>Levels from 0 up to the one whose single page covers the terrain.</summary>
        public static int LevelsFor(int pagesAcross) => PageSlots.LevelsFor(pagesAcross);

        /// <summary>Pages a side at level L of a terrain of this many level 0 pages a side.</summary>
        public static int PagesAcrossAt(int pagesAcross, int level) => PageSlots.PagesAcrossAt(pagesAcross, level);

        /// <summary>Pages a side at level L.</summary>
        public int PagesAcrossAt(int level) => Slots.PagesAcrossAt(level);

        /// <summary>The pool is made, for a terrain of this many samples a side and with this many page slots (0: every
        /// level 0 page resident at its own index).</summary>
        public bool Matches(int resolution, int capacity)
        {
            bool all = capacity <= 0;
            int across = PagesAcrossFor(resolution);
            return Texture != null && Resolution == resolution && AllResident == all
                   && Slots.Capacity == (all ? across * across : capacity + ScratchSlots);
        }

        /// <summary>
        /// The pool for a terrain of this many samples a side, with this many page slots (0: every level 0 page resident at
        /// its own index): made, and its page table written on this list, when there is none of that size and layout. True
        /// when it was made just now, so nothing is in it yet.
        /// </summary>
        public bool Ensure(int resolution, int capacity, ID3D12GraphicsCommandList cmd)
        {
            if (Matches(resolution, capacity)) return false;
            bool all = capacity <= 0;
            int across = PagesAcrossFor(resolution);
            int slots = all ? across * across : capacity + ScratchSlots;
            int slices = (slots + 3) / 4;
            if (slices > MaxSlices)
                throw new InvalidOperationException($"A height pool of {slots} pages ({resolution}² samples) needs {slices} array " +
                                                    $"slices: D3D12 allows {MaxSlices}");
            Release();
            Filled = false;
            Slices = slices;

            var device = Engine.Device;
            Texture = device.CreateTexture2D(Format.R16_UNorm, SliceTexels, SliceTexels, Slices, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
            Texture.Name = "Terrain height pool";

            Srv = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(Texture, new ShaderResourceViewDescription
            {
                Format = Format.R16_UNorm,
                ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2DArray = new Texture2DArrayShaderResourceView { MostDetailedMip = 0, MipLevels = 1, FirstArraySlice = 0, ArraySize = (uint)Slices },
            }, device.GetCpuHandle(Srv));

            Uav = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(Texture, null, new UnorderedAccessViewDescription
            {
                Format = Format.R16_UNorm,
                ViewDimension = UnorderedAccessViewDimension.Texture2DArray,
                Texture2DArray = new Texture2DArrayUnorderedAccessView { MipSlice = 0, FirstArraySlice = 0, ArraySize = (uint)Slices },
            }, device.GetCpuHandle(Uav));

            Slots.Reset(across, all ? 0 : capacity, ScratchSlots, all);
            Slots.RecordTable(cmd);
            Resolution = resolution;
            return true;
        }

        // ── Residency (PageSlots) ──

        /// <summary>The slot of page (px, py) at a level, or -1 when it is not resident.</summary>
        public int SlotOf(int level, int px, int py) => Slots.SlotOf(level, px, py);

        public int FreeSlots => Slots.FreeSlots;
        public (int level, int px, int py)? PageIn(int slot) => Slots.PageIn(slot);
        public int Add(int level, int px, int py) => Slots.Add(level, px, py);
        public void Remove(int level, int px, int py) => Slots.Remove(level, px, py);
        public int ScratchSlot(int i) => Slots.ScratchSlot(i);
        public void MarkWritten(int slot, bool submitted = false) => Slots.MarkWritten(slot, submitted);
        public bool Settled(int slot) => Slots.Settled(slot);
        public void RecordTable(ID3D12GraphicsCommandList cmd) => Slots.RecordTable(cmd);
        public bool TableDirty => Slots.TableDirty;

        /// <summary>A slot's slice and the texel of its quarter's corner. Sample (i, j) of the page in it lies at texel
        /// (x + Border + i, y + Border + j).</summary>
        public static (int slice, int x, int y) SlotCorner(int slot)
        {
            int q = slot & 3;
            return (slot >> 2, (q & 1) * PageTexels, (q >> 1) * PageTexels);
        }

        /// <summary>The slot of resident level 0 page (px, py): its slice and the texel of its quarter's corner. Throws when the
        /// page is not resident.</summary>
        public (int slice, int x, int y) PageCorner(int px, int py)
        {
            int slot = SlotOf(0, px, py);
            if (slot < 0) throw new InvalidOperationException($"height page ({px}, {py}) is not resident");
            return SlotCorner(slot);
        }

        /// <summary>
        /// This frame's <c>TerrainPagesParams</c> (terrain_pages.hlsli) for the terrain the pool belongs to: its bindless
        /// index, for <c>LoadTerrain</c>. One buffer per frame in flight, so a frame still on the GPU keeps what it read.
        /// 0 before the pool is made. <paramref name="paintIdx"/>: the terrain's paint params this frame (0 = none).
        /// </summary>
        public uint WriteParams(Vector3 origin, Vector2 size, float maxHeight, uint version, uint paintIdx = 0)
        {
            if (Texture == null) return 0;
            int slot = Engine.FrameIndex % FrameCount;
            var buffer = _params[slot] ??= GraphicsBuffer.CreateUpload<Params>(1, mapped: true);
            unsafe { *buffer.WritePtr<Params>() = MakeParams(origin, size, maxHeight, version, paintIdx); }
            return buffer.SrvIndex;
        }

        /// <summary>The <c>TerrainPagesParams</c> of this pool for a terrain at this place.</summary>
        public Params MakeParams(Vector3 origin, Vector2 size, float maxHeight, uint version, uint paintIdx = 0) => new()
        {
            HeightPool = Srv,
            PageTable = TableSrv,
            PagesAcross = (uint)PagesAcross,
            Cells = (uint)(Resolution - 1),
            Origin = origin,
            Spacing = size.X / (Resolution - 1),
            Size = size,
            MaxHeight = maxHeight,
            Version = version,
            PaintIdx = paintIdx,
        };

        private void Release()
        {
            var device = Engine.Device;
            if (Texture != null)
            {
                device.DeferDispose(Texture);   // frames in flight may still read it
                device.DeferReleaseBindlessIndex(Srv);
                device.DeferReleaseBindlessIndex(Uav);
            }
            Slots.Release();
            Texture = null;
            Srv = Uav = 0;
            Resolution = 0;
        }

        public void Dispose()
        {
            Release();
            for (int i = 0; i < _params.Length; i++)
            {
                Engine.Device.DeferDispose(_params[i]);
                _params[i] = null;
            }
        }
    }
}
