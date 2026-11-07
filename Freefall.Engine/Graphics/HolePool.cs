using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Freefall.Assets;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// The shaders' copy of a terrain's holes (HoleStamp) as pages (2026-11-08): only the pages a hole cuts, all of them all
    /// the time, one 257² R8 slice each (the page's samples, its shared edges included, as the CPU store holds them:
    /// TerrainPages.HolePage), and a table with one entry per page: 0 for none, else its slice + 1. Hole stamps are few, so
    /// the pool is small (66 KB a page) and needs no streaming; a read where no hole cuts is one table lookup. Until then one
    /// R8 texture held a texel per height sample of the whole terrain, which D3D12 cannot make past 16,384 (16 km at 1 m per
    /// vertex is 16,385) and would be 1 GB at 32 km. Shaders read it only through terrain_holes.hlsli
    /// (<c>TerrainHoleAt</c>), from <c>TerrainHoleParams</c> by push constant.
    /// </summary>
    public sealed class HolePool : IDisposable
    {
        public const int PageSamples = TerrainPages.PageSamples;
        private const int RowPitch = (PageSamples + 255) & ~255;   // 512
        private const int FrameCount = 3;

        /// <summary>terrain_holes.hlsli TerrainHoleParams. 16 bytes.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Params
        {
            public uint Pool, Table, PagesAcross, Cells;
        }

        /// <summary>The pages (Texture2DArray R8_UNorm, a slice each). Rests in Common.</summary>
        public ID3D12Resource? Texture { get; private set; }
        public uint Srv { get; private set; }

        /// <summary>Slices the texture has room for.</summary>
        public int Capacity { get; private set; }

        /// <summary>Pages with holes in the pool.</summary>
        public int Count => _sliceOf.Count;

        public int PagesAcross { get; private set; }
        public int Cells { get; private set; }

        private readonly Dictionary<int, int> _sliceOf = new();     // page index → slice
        private readonly Stack<int> _free = new();
        private uint[] _entries = Array.Empty<uint>();
        private bool _entriesDirty;
        private GraphicsBuffer? _table;
        private readonly GraphicsBuffer?[] _params = new GraphicsBuffer?[FrameCount];

        /// <summary>
        /// Bring the pool to the store's holes: pages that lost theirs are let go, pages that have holes and are not in the
        /// pool or are named in <paramref name="changed"/> (page indices; null = every page with holes) are uploaded on the
        /// frame's list, and the table follows. A store of another layout, or more cut pages than there is room for, makes
        /// the pool again and uploads every cut page. Main thread.
        /// </summary>
        public void Sync(TerrainPages store, ICollection<int>? changed)
        {
            int across = store.PagesAcross, cells = store.Resolution - 1;
            bool layout = across != PagesAcross || cells != Cells || _table == null;

            // The pages with holes now
            var cut = new List<int>();
            for (int i = 0, n = across * across; i < n; i++)
                if (store.PageIdentity(i % across, i / across).holes != null) cut.Add(i);

            if (layout || cut.Count > Capacity)
            {
                Make(across, cells, Math.Max(cut.Count, layout ? 0 : Capacity));
                changed = null;
            }

            // Pages without holes now leave
            if (_sliceOf.Count > 0)
            {
                var cutSet = new HashSet<int>(cut);
                var gone = new List<int>();
                foreach (var page in _sliceOf.Keys)
                    if (!cutSet.Contains(page)) gone.Add(page);
                foreach (var page in gone)
                {
                    _free.Push(_sliceOf[page]);
                    _sliceOf.Remove(page);
                    _entries[page] = 0;
                    _entriesDirty = true;
                }
            }

            // Pages with holes: new ones get a slice, and new or changed ones go up
            List<(int page, int slice)>? uploads = null;
            foreach (int page in cut)
            {
                bool known = _sliceOf.TryGetValue(page, out int slice);
                if (!known)
                {
                    slice = _free.Pop();
                    _sliceOf[page] = slice;
                    _entries[page] = (uint)slice + 1;
                    _entriesDirty = true;
                }
                if (!known || changed == null || changed.Contains(page))
                    (uploads ??= new()).Add((page, slice));
            }
            if (uploads != null) Upload(store, uploads);
            RecordTable();
        }

        // A texture of room for at least this many pages (a little more, so a few new cuts do not make it again), every
        // slice free, the table empty
        private void Make(int across, int cells, int pages)
        {
            Release();
            PagesAcross = across;
            Cells = cells;
            Capacity = Math.Max(16, pages + pages / 4);
            if (Capacity > HeightPool.MaxSlices)
                throw new InvalidOperationException($"{pages} pages cut by holes: D3D12 allows {HeightPool.MaxSlices} slices");

            var device = Engine.Device;
            Texture = device.CreateTexture2D(Format.R8_UNorm, PageSamples, PageSamples, Capacity, 1, ResourceFlags.None, ResourceStates.Common);
            Texture.Name = "Terrain hole pages";
            Srv = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(Texture, new ShaderResourceViewDescription
            {
                Format = Format.R8_UNorm,
                ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2DArray = new Texture2DArrayShaderResourceView { MostDetailedMip = 0, MipLevels = 1, FirstArraySlice = 0, ArraySize = (uint)Capacity },
            }, device.GetCpuHandle(Srv));

            for (int s = Capacity - 1; s >= 0; s--) _free.Push(s);
            _entries = new uint[across * across];
            _entriesDirty = true;
            _table = GraphicsBuffer.CreateStructured<uint>(_entries.Length, srv: true);
        }

        // The pages' bytes into their slices, on the frame's list (before the terrain draws)
        private unsafe void Upload(TerrainPages store, List<(int page, int slice)> pages)
        {
            var device = Engine.Device;
            var texture = Texture!;
            int across = store.PagesAcross;
            ulong pageBytes = (ulong)(RowPitch * PageSamples + 511) & ~511UL;
            var upload = device.NativeDevice.CreateCommittedResource(new HeapProperties(HeapType.Upload), HeapFlags.None,
                ResourceDescription.Buffer(pageBytes * (ulong)pages.Count), ResourceStates.GenericRead, null);
            void* data;
            upload.Map(0, null, &data);
            for (int i = 0; i < pages.Count; i++)
            {
                int px = pages[i].page % across, pz = pages[i].page / across;
                var holes = store.HolePage(px, pz);
                byte* dst = (byte*)data + (long)pageBytes * i;
                for (int z = 0; z < PageSamples; z++)
                {
                    var row = new Span<byte>(dst + (long)z * RowPitch, PageSamples);
                    if (holes.IsEmpty) row.Clear();
                    else holes.Slice(z * PageSamples, PageSamples).CopyTo(row);
                }
            }
            upload.Unmap(0);

            var list = pages.ToArray();
            CommandBuffer.Enqueue(RenderPass.Opaque, cmd =>
            {
                cmd.ResourceBarrierTransition(texture, ResourceStates.Common, ResourceStates.CopyDest);
                for (int i = 0; i < list.Length; i++)
                {
                    var footprint = new PlacedSubresourceFootPrint
                    {
                        Offset = pageBytes * (ulong)i,
                        Footprint = new SubresourceFootPrint(Format.R8_UNorm, PageSamples, PageSamples, 1, RowPitch)
                    };
                    cmd.CopyTextureRegion(new TextureCopyLocation(texture, (uint)list[i].slice), 0, 0, 0, new TextureCopyLocation(upload, footprint));
                }
                cmd.ResourceBarrierTransition(texture, ResourceStates.CopyDest, ResourceStates.Common);
            });
            device.DeferDispose(upload);   // until the frames that copy from it are done
        }

        // The table up on the frame's list when it changed
        private void RecordTable()
        {
            if (!_entriesDirty || _table == null) return;
            _entriesDirty = false;
            var upload = GraphicsBuffer.CreateUpload<uint>(_entries.Length, mapped: true);
            unsafe
            {
                var dst = upload.WritePtr<uint>();
                for (int i = 0; i < _entries.Length; i++) dst[i] = _entries[i];
            }
            var table = _table;
            int bytes = _entries.Length * sizeof(uint);
            CommandBuffer.Enqueue(RenderPass.Opaque, cmd =>
            {
                table.Transition(cmd, ResourceStates.CopyDest);
                cmd.CopyBufferRegion(table.Native, 0, upload.Native, 0, (ulong)bytes);
                table.Transition(cmd, ResourceStates.AllShaderResource);
            });
            Engine.Device.DeferDispose(upload);
        }

        /// <summary>This frame's <c>TerrainHoleParams</c>: its bindless index, 0 when no page is cut.</summary>
        public uint WriteParams()
        {
            if (Texture == null || _table == null || _sliceOf.Count == 0) return 0;
            int slot = Engine.FrameIndex % FrameCount;   // the buffer of this frame in flight
            var buffer = _params[slot] ??= GraphicsBuffer.CreateUpload<Params>(1, mapped: true);
            unsafe
            {
                *buffer.WritePtr<Params>() = new Params { Pool = Srv, Table = _table.SrvIndex, PagesAcross = (uint)PagesAcross, Cells = (uint)Cells };
            }
            return buffer.SrvIndex;
        }

        /// <summary>No holes any more: the pages and the table go.</summary>
        public void Clear() => Release();

        private void Release()
        {
            var device = Engine.Device;
            if (Texture != null)
            {
                device.DeferDispose(Texture);
                device.DeferReleaseBindlessIndex(Srv);
            }
            device?.DeferDispose(_table);
            Texture = null;
            _table = null;
            Srv = 0;
            Capacity = 0;
            PagesAcross = Cells = 0;
            _sliceOf.Clear();
            _free.Clear();
            _entries = Array.Empty<uint>();
            _entriesDirty = false;
        }

        public void Dispose()
        {
            Release();
            for (int i = 0; i < _params.Length; i++)
            {
                Engine.Device?.DeferDispose(_params[i]);
                _params[i] = null;
            }
        }
    }
}
