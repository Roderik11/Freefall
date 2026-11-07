using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// A terrain's heights as pages in one Texture2DArray, the height pool (TERRAIN_PAGES_PLAN.md stage 4a, 2026-11-07).
    ///
    /// A page is <see cref="Assets.Terrain.PageCells"/> (256) cells a side and holds 257 samples, its cells and the edge
    /// it shares with the next page, inside a border of 2: <see cref="PageTexels"/> (261) texels. Four pages to a slice,
    /// 2 × 2. The page table names each level 0 page's slot (slice × 4 + quarter); for now every page is loaded and its
    /// slot is its index, row by row. The bake writes the pages, borders included (TerrainBaker.BakeHeight), and shaders
    /// read them only through terrain_pages.hlsli, which this mirrors. Every pool and table is reached by one descriptor
    /// each, the same in every lane: the slice and the place in it are plain data, so no read is non-uniform (AMD).
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
            public Vector4 Pad;
        }

        /// <summary>Samples a side of the terrain the pool was made for (0 before <see cref="Ensure"/>).</summary>
        public int Resolution { get; private set; }

        /// <summary>Level 0 pages a side.</summary>
        public int PagesAcross { get; private set; }

        /// <summary>Slices of the array.</summary>
        public int Slices { get; private set; }

        /// <summary>The Texture2DArray (R16_UNorm). Rests in Common between uses.</summary>
        public ID3D12Resource? Texture { get; private set; }

        public uint Srv { get; private set; }
        public uint Uav { get; private set; }

        /// <summary>The page table (StructuredBuffer&lt;uint&gt;), read only.</summary>
        public uint TableSrv => _table?.SrvIndex ?? 0u;

        /// <summary>Heights are in it: a bake or an upload has filled it since it was made.</summary>
        public bool Filled { get; set; }

        private GraphicsBuffer? _table;
        private readonly GraphicsBuffer?[] _params = new GraphicsBuffer?[FrameCount];

        /// <summary>
        /// The pool for a terrain of this many samples a side: made (and its page table written, on this list) when there is
        /// none of that size. True when it was made just now, so nothing is in it yet.
        /// </summary>
        /// <summary>D3D12's cap on a texture array's slices (D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION).</summary>
        public const int MaxSlices = 2048;

        /// <summary>Array slices a pool of every page takes for a terrain of this many samples a side (four pages a slice).</summary>
        public static int SlicesFor(int resolution)
        {
            int across = Math.Max(1, (Math.Max(1, resolution - 1) + Assets.Terrain.PageCells - 1) / Assets.Terrain.PageCells);
            return (across * across + 3) / 4;
        }

        public bool Ensure(int resolution, ID3D12GraphicsCommandList cmd)
        {
            if (Texture != null && Resolution == resolution) return false;
            Release();
            Filled = false;

            var device = Engine.Device;
            int cells = Math.Max(1, resolution - 1);
            PagesAcross = Math.Max(1, (cells + Assets.Terrain.PageCells - 1) / Assets.Terrain.PageCells);
            int pages = PagesAcross * PagesAcross;
            Slices = (pages + 3) / 4;
            if (Slices > MaxSlices)
                throw new InvalidOperationException($"A terrain of {resolution}² samples needs {pages} height pages, {Slices} array " +
                                                    $"slices: D3D12 allows {MaxSlices}");

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

            // Every page loaded, at the slot of its own index
            _table = GraphicsBuffer.CreateStructured<uint>(pages, srv: true);
            var upload = GraphicsBuffer.CreateUpload<uint>(pages, mapped: true);
            unsafe
            {
                var entries = upload.WritePtr<uint>();
                for (int i = 0; i < pages; i++) entries[i] = (uint)i;   // slot i, level 0
            }
            _table.Transition(cmd, ResourceStates.CopyDest);
            cmd.CopyBufferRegion(_table.Native, 0, upload.Native, 0, (ulong)(pages * sizeof(uint)));
            _table.Transition(cmd, ResourceStates.AllShaderResource);
            device.DeferDispose(upload);   // read when the list runs

            Resolution = resolution;
            return true;
        }

        /// <summary>The slot of level 0 page (px, py): its slice and the texel of its quarter's corner. Sample (i, j) of the
        /// page lies at texel (x + Border + i, y + Border + j).</summary>
        public (int slice, int x, int y) PageCorner(int px, int py)
        {
            int slot = py * PagesAcross + px;
            int q = slot & 3;
            return (slot >> 2, (q & 1) * PageTexels, (q >> 1) * PageTexels);
        }

        /// <summary>
        /// This frame's <c>TerrainPagesParams</c> (terrain_pages.hlsli) for the terrain the pool belongs to: its bindless
        /// index, for <c>LoadTerrain</c>. One buffer per frame in flight, so a frame still on the GPU keeps what it read.
        /// 0 before the pool is made.
        /// </summary>
        public uint WriteParams(Vector3 origin, Vector2 size, float maxHeight, uint version)
        {
            if (Texture == null) return 0;
            int slot = Engine.FrameIndex % FrameCount;
            var buffer = _params[slot] ??= GraphicsBuffer.CreateUpload<Params>(1, mapped: true);
            unsafe { *buffer.WritePtr<Params>() = MakeParams(origin, size, maxHeight, version); }
            return buffer.SrvIndex;
        }

        /// <summary>The <c>TerrainPagesParams</c> of this pool for a terrain at this place.</summary>
        public Params MakeParams(Vector3 origin, Vector2 size, float maxHeight, uint version) => new()
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
            device.DeferDispose(_table);
            Texture = null;
            _table = null;
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
