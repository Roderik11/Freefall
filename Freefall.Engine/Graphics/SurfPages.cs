using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// The surf round the camera at the terrain's own detail (2026-11-23): the whole terrain's warp (<see cref="ShoreWarp"/>)
    /// is 1025² cells at most, 16 m on a 16 km terrain, and its stairs showed in the swash and along the breakers. Here the
    /// warp is baked again in pages on the terrain's page grid (<see cref="ShoreWarp.PageTexels"/> texels a side; level 0 at
    /// the height samples' spacing, each level twice the last, up to the last level finer than the whole terrain's cells),
    /// <see cref="Window"/> pages a side per level round the camera, as the paint is kept (<see cref="TerrainMapResidency"/>).
    /// Only pages with a coast in them (<see cref="ShoreWarp.PageWanted"/>); elsewhere, and until a page arrives, the ocean
    /// reads the whole terrain's warp. Each page is baked on a worker (<see cref="ShoreWarp.BakePage"/>: the whole terrain's
    /// travel time and height, its coast strip marched at the page's own cells) and copied into its slot on the main
    /// thread's next frame. A new whole-terrain bake
    /// (the wind turned, new heights) bakes every page again, nearest first; until then each page keeps what it was baked
    /// with, its bands' headings beside it in the slot buffer, so it stays whole.
    ///
    /// ocean_surf.hlsli SampleBandWarp reads it: the page table (one entry per level 0 page, the finest page over it, slot |
    /// level &lt;&lt; 24, 0xFFFFFFFF none), per slot two slices of RGBA16F (per warped band: lag − base, amplitude, g.x, g.z)
    /// and one R16F (the beach's hollows), and two float4s (base lag of each band, their deep headings, their ocean bands).
    /// </summary>
    public sealed class SurfPages : IDisposable
    {
        /// <summary>Pages a side per level round the camera (odd; as EngineSettings.TerrainPaintPages).</summary>
        public const int Window = 3;

        /// <summary>Pages baking at once, on workers.</summary>
        private static readonly int MaxInFlight = Math.Max(1, Environment.ProcessorCount / 4);

        private const int SlotTexels = ShoreWarp.SlotTexels;
        private const uint Empty = 0xFFFFFFFFu;

        private readonly PageSlots _slots = new() { EmptyEntry = Empty };
        private ID3D12Resource? _dataTex, _hollowTex;
        private uint _dataSrv, _hollowSrv;
        private GraphicsBuffer? _info;
        private Vector4[] _infoCpu = Array.Empty<Vector4>();
        private bool _infoDirty;
        private ShoreWarp.Seeds?[] _slotSeeds = Array.Empty<ShoreWarp.Seeds?>();

        private int _levels, _pagesAcross, _samples;
        private Vector2 _terrainSize;
        private ShoreWarp.Seeds? _seeds;
        private readonly Dictionary<(int, int, int), bool> _wanted = new();
        private readonly HashSet<(int, int, int)> _inFlight = new();
        // Baked pages with the pool they were started for (a pool made again drops what was baking for the old one)
        private readonly ConcurrentQueue<(int pool, ShoreWarp.Page page)> _done = new();
        private int _pool;
        private readonly List<ShoreWarp.Page> _uploads = new();
        // Pages to bake: those with none first, coarsest first; then those baked from an older whole-terrain bake; each nearest first
        private readonly List<(int group, int order, float distance, int level, int px, int py)> _candidates = new();
        private int _bakedInBurst;
        private double _burstMs;

        /// <summary>The page table (StructuredBuffer&lt;uint&gt;); 0 = no pages.</summary>
        public uint TableSrv => _dataTex != null ? _slots.TableSrv : 0u;
        public uint DataSrv => _dataSrv;
        public uint HollowSrv => _hollowSrv;
        public uint InfoSrv => _info?.SrvIndex ?? 0u;
        /// <summary>Level 0 pages a side.</summary>
        public int PagesAcross => _pagesAcross;
        /// <summary>Height samples a side: level 0 texel i lies on sample i.</summary>
        public int Samples => _samples;
        /// <summary>Levels baked in pages (the coarser ones are the whole terrain's warp).</summary>
        public int Levels => _levels;
        public int InFlight => _inFlight.Count;
        public int Resident { get; private set; }

        /// <summary>
        /// The frame's work, main thread: take baked pages into their slots, let go of pages beyond their window, and start
        /// bakes of the pages the windows want. <paramref name="seeds"/>: the whole terrain's last bake (null: none yet).
        /// </summary>
        public void Update(Vector3 camera, ShoreWarp.Seeds? seeds, Assets.TerrainPages? heights, SeaMask.Result? mask,
                           float seaLevel01, float maxHeight, Vector2 terrainSize, Vector2 origin)
        {
            if (seeds == null || heights == null || !heights.HasHeights || heights.Resolution < 2) return;
            if (heights.Resolution != _samples || terrainSize != _terrainSize || _dataTex == null)
                Reset(heights.Resolution, terrainSize, seeds);
            if (_levels == 0) return;

            if (!ReferenceEquals(seeds, _seeds))
            {
                _seeds = seeds;
                _wanted.Clear();
            }

            float cell0 = terrainSize.X / (_samples - 1);
            var c = new Vector2((camera.X - origin.X) / cell0, (camera.Z - origin.Y) / cell0);
            int h = (Window - 1) / 2;

            // Pages beyond their window let go (half a page of hysteresis)
            for (int l = 0; l < _levels; l++)
            {
                int n = _slots.PagesAcrossAt(l);
                float span = ShoreWarp.PageTexels << l, keep = (h + 0.5f) * span;
                for (int py = 0; py < n; py++)
                    for (int px = 0; px < n; px++)
                    {
                        int slot = _slots.SlotOf(l, px, py);
                        if (slot >= 0 && Distance(c, px, py, span) > keep)
                        {
                            _slots.Remove(l, px, py);
                            _slotSeeds[slot] = null;
                        }
                    }
            }

            // Baked pages into their slots (a page let go meanwhile is dropped)
            _uploads.Clear();
            while (_done.TryDequeue(out var done))
            {
                var page = done.page;
                if (done.pool != _pool) continue;
                _inFlight.Remove((page.Level, page.Px, page.Py));
                if (page.Data.Length == 0)
                {
                    _wanted[(page.Level, page.Px, page.Py)] = false;   // failed (logged): not tried again until the next bake
                    continue;
                }
                _bakedInBurst++;
                _burstMs += page.Milliseconds;
                float span = ShoreWarp.PageTexels << page.Level;
                if (Distance(c, page.Px, page.Py, span) > (h + 0.5f) * span) continue;
                int slot = _slots.SlotOf(page.Level, page.Px, page.Py);
                if (slot < 0) slot = _slots.Add(page.Level, page.Px, page.Py);
                if (slot < 0) continue;
                _slotSeeds[slot] = page.Seeds;
                SetInfo(slot, page);
                _uploads.Add(page);
            }
            if (_inFlight.Count == 0 && _bakedInBurst > 0)
            {
                Debug.LogAlways($"[SurfPages] {_bakedInBurst} page(s) baked, {_burstMs / _bakedInBurst:0} ms each on a worker");
                _bakedInBurst = 0;
                _burstMs = 0;
            }

            int resident = 0;
            for (int s = 0; s < _slotSeeds.Length; s++) if (_slotSeeds[s] != null) resident++;
            Resident = resident;

            Record();

            // New bakes: pages the windows want that have none, coarsest and nearest first, then pages baked from an older
            // whole-terrain bake, nearest first
            if (_inFlight.Count >= MaxInFlight) return;
            _candidates.Clear();
            for (int l = _levels - 1; l >= 0; l--)
            {
                int n = _slots.PagesAcrossAt(l);
                float span = ShoreWarp.PageTexels << l, want = h * span;
                int cx = (int)MathF.Floor(c.X / span), cz = (int)MathF.Floor(c.Y / span);
                for (int py = Math.Max(0, cz - h - 1); py <= Math.Min(n - 1, cz + h + 1); py++)
                    for (int px = Math.Max(0, cx - h - 1); px <= Math.Min(n - 1, cx + h + 1); px++)
                    {
                        float d = Distance(c, px, py, span);
                        if (d > want || _inFlight.Contains((l, px, py))) continue;
                        int slot = _slots.SlotOf(l, px, py);
                        bool stale = slot >= 0 && _slotSeeds[slot] != _seeds;
                        if (slot >= 0 && !stale) continue;
                        if (!Wanted(l, px, py)) continue;
                        _candidates.Add(slot < 0 ? (0, -l, d, l, px, py) : (1, 0, d, l, px, py));
                    }
            }
            if (_candidates.Count == 0) return;
            _candidates.Sort((a, b) => a.group != b.group ? a.group.CompareTo(b.group)
                                     : a.order != b.order ? a.order.CompareTo(b.order) : a.distance.CompareTo(b.distance));
            int pool = _pool;
            foreach (var (_, _, _, l, px, py) in _candidates)
            {
                if (_inFlight.Count >= MaxInFlight) break;
                if (_slots.SlotOf(l, px, py) < 0 && _slots.FreeSlots <= _inFlight.Count) break;   // no room for it when done
                var job = new ShoreWarp.PageJob
                {
                    Level = l, Px = px, Py = py, Seeds = _seeds!, Heights = heights, Mask = mask,
                    SeaLevel01 = seaLevel01, MaxHeight = maxHeight, TerrainSize = terrainSize, Origin = origin,
                };
                _inFlight.Add((l, px, py));
                Task.Run(() =>
                {
                    try { _done.Enqueue((pool, ShoreWarp.BakePage(job))); }
                    catch (Exception e)
                    {
                        Debug.LogError("SurfPages", $"page {job.Level}/{job.Px},{job.Py} failed: {e}");
                        _done.Enqueue((pool, new ShoreWarp.Page { Level = job.Level, Px = job.Px, Py = job.Py, Seeds = null! }));
                    }
                });
            }
        }

        private bool Wanted(int level, int px, int py)
        {
            if (_wanted.TryGetValue((level, px, py), out bool w)) return w;
            w = ShoreWarp.PageWanted(_seeds!, level, px, py, _samples, _terrainSize);
            _wanted[(level, px, py)] = w;
            return w;
        }

        // The camera's distance to a page's square, in level 0 texels, along the farther axis (0 inside it)
        private static float Distance(Vector2 c, int px, int py, float span)
        {
            float dx = MathF.Max(0f, MathF.Max(px * span - c.X, c.X - (px + 1) * span));
            float dy = MathF.Max(0f, MathF.Max(py * span - c.Y, c.Y - (py + 1) * span));
            return MathF.Max(dx, dy);
        }

        // The pool for a terrain of this many height samples a side: the levels finer than the whole terrain's cells
        private void Reset(int samples, Vector2 terrainSize, ShoreWarp.Seeds seeds)
        {
            Release();
            _samples = samples;
            _terrainSize = terrainSize;
            _pagesAcross = Math.Max(1, (samples - 1 + ShoreWarp.PageTexels - 1) / ShoreWarp.PageTexels);
            float cell0 = terrainSize.X / (samples - 1);
            int levels = 0;
            while (levels < PageSlots.LevelsFor(_pagesAcross) && cell0 * (1 << levels) < 0.75f * seeds.Hx) levels++;
            _levels = levels;
            if (levels == 0) return;

            int capacity = 0;
            for (int l = 0; l < levels; l++)
            {
                int n = PageSlots.PagesAcrossAt(_pagesAcross, l);
                int s = Math.Min(Window + 1, n);
                capacity += s * s;
            }
            _slots.Reset(_pagesAcross, capacity, 0, allResident: false);
            _slotSeeds = new ShoreWarp.Seeds?[capacity];
            _infoCpu = new Vector4[capacity * 2];
            _infoDirty = true;

            var device = Engine.Device;
            _dataTex = device.CreateTexture2D(Format.R16G16B16A16_Float, SlotTexels, SlotTexels, capacity * ShoreWarp.MaxBands, 1,
                ResourceFlags.None, ResourceStates.Common);
            _hollowTex = device.CreateTexture2D(Format.R16_Float, SlotTexels, SlotTexels, capacity, 1, ResourceFlags.None, ResourceStates.Common);
            _dataTex.Name = "SurfPages data";
            _hollowTex.Name = "SurfPages hollows";
            _dataSrv = device.AllocateBindlessIndex();
            _hollowSrv = device.AllocateBindlessIndex();
            CreateArraySrv(device, _dataTex, Format.R16G16B16A16_Float, capacity * ShoreWarp.MaxBands, _dataSrv);
            CreateArraySrv(device, _hollowTex, Format.R16_Float, capacity, _hollowSrv);
            _info = GraphicsBuffer.CreateStructured<Vector4>(capacity * 2, srv: true);
            Debug.LogAlways($"[SurfPages] {levels} level(s) of {_pagesAcross} pages a side, {capacity} slots, " +
                            $"{(long)SlotTexels * SlotTexels * capacity * (8 * ShoreWarp.MaxBands + 2) / (1024 * 1024)} MiB");
        }

        // A slot's float4s: (base lag of slice 0, of slice 1, slice 0's deep heading), (slice 1's heading, its bands + 1)
        private void SetInfo(int slot, ShoreWarp.Page page)
        {
            var bands = page.Seeds?.Bands ?? Array.Empty<ShoreWarp.Band>();
            Vector2 d0 = bands.Length > 0 ? bands[0].Direction : Vector2.UnitX, d1 = bands.Length > 1 ? bands[1].Direction : Vector2.UnitX;
            _infoCpu[slot * 2] = new Vector4(page.Base[0], page.Base[1], d0.X, d0.Y);
            _infoCpu[slot * 2 + 1] = new Vector4(d1.X, d1.Y, bands.Length > 0 ? bands[0].Index + 1 : 0, bands.Length > 1 ? bands[1].Index + 1 : 0);
            _infoDirty = true;
        }

        // This frame's copies (pages into their slots, the slot buffer, the table) on the opaque pass's list, before the
        // forward pass draws the sea
        private unsafe void Record()
        {
            if (_uploads.Count == 0 && !_infoDirty && !_slots.TableDirty) return;
            var device = Engine.Device;
            var dataTex = _dataTex!;
            var hollowTex = _hollowTex!;
            var copies = new List<(ID3D12Resource data, int dataPitch, ID3D12Resource hollow, int hollowPitch, int slot)>(_uploads.Count);
            foreach (var page in _uploads)
            {
                int slot = _slots.SlotOf(page.Level, page.Px, page.Py);
                var data = Stage(device, page.Data.AsSpan(), 8, ShoreWarp.MaxBands, out int dataPitch);
                var hollow = Stage(device, page.Hollow.AsSpan(), 2, 1, out int hollowPitch);
                copies.Add((data, dataPitch, hollow, hollowPitch, slot));
                device.DeferDispose(data);
                device.DeferDispose(hollow);
            }
            GraphicsBuffer? infoUpload = null;
            if (_infoDirty && _info != null)
            {
                infoUpload = GraphicsBuffer.CreateUpload<Vector4>(_infoCpu.Length, mapped: true);
                var dst = infoUpload.WritePtr<Vector4>();
                for (int i = 0; i < _infoCpu.Length; i++) dst[i] = _infoCpu[i];
                _infoDirty = false;
            }
            var info = _info;
            var slots = _slots;
            CommandBuffer.Enqueue(RenderPass.Opaque, list =>
            {
                if (copies.Count > 0)
                {
                    list.ResourceBarrierTransition(dataTex, ResourceStates.Common, ResourceStates.CopyDest);
                    list.ResourceBarrierTransition(hollowTex, ResourceStates.Common, ResourceStates.CopyDest);
                    foreach (var (data, dataPitch, hollow, hollowPitch, slot) in copies)
                    {
                        for (int s = 0; s < ShoreWarp.MaxBands; s++)
                            list.CopyTextureRegion(new TextureCopyLocation(dataTex, (uint)(slot * ShoreWarp.MaxBands + s)), 0, 0, 0,
                                new TextureCopyLocation(data, new PlacedSubresourceFootPrint
                                {
                                    Offset = (ulong)s * (ulong)dataPitch * SlotTexels,
                                    Footprint = new SubresourceFootPrint(Format.R16G16B16A16_Float, SlotTexels, SlotTexels, 1, (uint)dataPitch)
                                }));
                        list.CopyTextureRegion(new TextureCopyLocation(hollowTex, (uint)slot), 0, 0, 0,
                            new TextureCopyLocation(hollow, new PlacedSubresourceFootPrint
                            {
                                Footprint = new SubresourceFootPrint(Format.R16_Float, SlotTexels, SlotTexels, 1, (uint)hollowPitch)
                            }));
                    }
                    list.ResourceBarrierTransition(dataTex, ResourceStates.CopyDest, ResourceStates.Common);
                    list.ResourceBarrierTransition(hollowTex, ResourceStates.CopyDest, ResourceStates.Common);
                }
                if (infoUpload != null && info != null)
                {
                    info.Transition(list, ResourceStates.CopyDest);
                    list.CopyBufferRegion(info.Native, 0, infoUpload.Native, 0, (ulong)(info.ElementCount * 16));
                    info.Transition(list, ResourceStates.AllShaderResource);
                }
                slots.RecordTable(list);
            });
            if (infoUpload != null) device.DeferDispose(infoUpload);
        }

        // An upload buffer holding a page's slices, rows padded to the copy pitch
        private static unsafe ID3D12Resource Stage<T>(GraphicsDevice device, Span<T> data, int texelBytes, int slices, out int pitch)
            where T : unmanaged
        {
            int rowBytes = SlotTexels * texelBytes;
            pitch = (rowBytes + 255) & ~255;
            var upload = device.CreateCommittedResource(
                new HeapProperties(HeapType.Upload), HeapFlags.None,
                ResourceDescription.Buffer((ulong)pitch * SlotTexels * (ulong)slices), ResourceStates.GenericRead, null);
            void* mapped;
            upload.Map(0, null, &mapped);
            fixed (T* src = data)
            {
                byte* s = (byte*)src;
                for (int row = 0; row < SlotTexels * slices; row++)
                    Buffer.MemoryCopy(s + (long)row * rowBytes, (byte*)mapped + (long)row * pitch, rowBytes, rowBytes);
            }
            upload.Unmap(0);
            return upload;
        }

        private static void CreateArraySrv(GraphicsDevice device, ID3D12Resource tex, Format format, int slices, uint srv)
        {
            device.NativeDevice.CreateShaderResourceView(tex, new ShaderResourceViewDescription
            {
                Format = format,
                ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2DArray = new Texture2DArrayShaderResourceView { MostDetailedMip = 0, MipLevels = 1, FirstArraySlice = 0, ArraySize = (uint)slices }
            }, device.GetCpuHandle(srv));
        }

        private void Release()
        {
            _pool++;
            _inFlight.Clear();
            var device = Engine.Device;
            device?.DeferDispose(_dataTex);
            device?.DeferDispose(_hollowTex);
            device?.DeferDispose(_info);
            if (_dataSrv != 0) device?.DeferReleaseBindlessIndex(_dataSrv);
            if (_hollowSrv != 0) device?.DeferReleaseBindlessIndex(_hollowSrv);
            _dataTex = _hollowTex = null;
            _info = null;
            _dataSrv = _hollowSrv = 0;
            _slots.Release();
            _levels = 0;
            _wanted.Clear();
            _seeds = null;
        }

        public void Dispose() => Release();
    }
}
