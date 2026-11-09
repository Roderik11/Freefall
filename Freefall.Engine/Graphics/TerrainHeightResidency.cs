using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Assets;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Which height pages a streaming <see cref="HeightPool"/> holds (TERRAIN_PAGES_PLAN.md 5a): windows round the camera,
    /// level 0 within <c>EngineSettings.TerrainDetailRadius</c> R, level 1 within 2R, level 2 within 4R, every page of
    /// level 3 and up; a page is let go half a page beyond its window, so one does not come and go at a window's edge.
    /// Pages come from the CPU store (Terrain.Pages, whole in 5a; a level L page is decimated from it), the nearest and
    /// coarsest first, <c>TerrainPageUploadsPerFrame</c> a frame, copied on the frame's own list before its terrain reads,
    /// the page table after them. A page arrives over a coarser one already there, so there is never a hole.
    /// </summary>
    public sealed class TerrainHeightResidency
    {
        private readonly HeightPool _pool;
        private ID3D12Resource? _poolTexture;   // the pool this was set up for: a new pool starts empty

        // Pages whose upload is staged on the CPU, to be recorded on this frame's list
        private readonly List<(int level, int px, int py, int slot)> _staged = new();
        private ID3D12Resource? _stagedBuffer;
        // Resident pages whose heights changed on the CPU since they went up (a readback): uploaded again
        private readonly HashSet<(int level, int px, int py)> _stale = new();
        private readonly List<(float key, int level, int px, int py)> _candidates = new();
        // Where the camera was, and the radius, when the windows were last looked at
        private Vector2 _chosenAt = new(float.NaN);
        private float _chosenRadius = float.NaN;

        public TerrainHeightResidency(HeightPool pool) => _pool = pool;

        /// <summary>The pool this keeps.</summary>
        public HeightPool Pool => _pool;

        /// <summary>Pages are staged or the page table changed: <see cref="Record"/> has something to put on the list.</summary>
        public bool HasWork => _staged.Count > 0 || _pool.TableDirty;

        /// <summary>Pages a frame went up last frame, and how many are waiting (for the profiler and the log).</summary>
        public int UploadedLastFrame { get; private set; }
        public int Waiting { get; private set; }

        // Texels and bytes of a page in an upload buffer: rows 256-byte aligned, each page at a 512-byte boundary
        private const int RowPitch = (HeightPool.PageTexels * 2 + 255) & ~255;
        private const int PageBytes = (RowPitch * HeightPool.PageTexels + 511) & ~511;

        // Levels with a window; the rest are whole
        private const int WindowedLevels = 3;

        /// <summary>
        /// Page slots (scratch not counted) the windows can hold at once for a terrain of this many samples, spacing and
        /// detail radius: 0 when the radius is 0 or its level 0 window would take the whole terrain anyway (every page
        /// resident, HeightPool.AllResident).
        /// </summary>
        public static int CapacityFor(int resolution, float spacing, float radius)
        {
            if (radius <= 0f || resolution < 2) return 0;
            int across = HeightPool.PagesAcrossFor(resolution);
            float page0 = Terrain.PageCells * spacing;
            if (WindowSide(radius, page0, across) >= across) return 0;
            int levels = HeightPool.LevelsFor(across);
            int total = 0;
            for (int l = 0; l < levels; l++)
            {
                int n = HeightPool.PagesAcrossAt(across, l);
                int side = l < WindowedLevels ? WindowSide(radius * (1 << l), page0 * (1 << l), n) : n;
                total += side * side;
            }
            return total;
        }

        // Pages a side that lie within R plus half a page of a point (the hysteresis's reach), capped at the level's count
        private static int WindowSide(float radius, float pageMetres, int n) =>
            Math.Min(n, (int)MathF.Ceiling((2f * radius + pageMetres) / pageMetres) + 1);

        // The camera's distance to a page's square at a level, along the farther axis (0 inside it), in metres
        private static float Distance(Vector2 camera, int px, int py, float pageMetres)
        {
            float dx = MathF.Max(0f, MathF.Max(px * pageMetres - camera.X, camera.X - (px + 1) * pageMetres));
            float dy = MathF.Max(0f, MathF.Max(py * pageMetres - camera.Y, camera.Y - (py + 1) * pageMetres));
            return MathF.Max(dx, dy);
        }

        /// <summary>The heights at <paramref name="pages"/> have changed over these samples (x0, z0 inclusive, x1, z1
        /// exclusive): the resident pages over them, at every level, go up again.</summary>
        public void MarkStale(int x0, int z0, int x1, int z1)
        {
            if (_pool.Texture == null || _pool.AllResident) return;
            for (int l = 0; l < _pool.Levels; l++)
            {
                int n = _pool.PagesAcrossAt(l);
                int span = Terrain.PageCells << l;
                // A sample on a page's edge is held by both pages, and coarse pages' borders reach further
                int px0 = Math.Clamp((x0 - 2 * (1 << l)) / span, 0, n - 1), px1 = Math.Clamp((x1 + 2 * (1 << l)) / span, 0, n - 1);
                int pz0 = Math.Clamp((z0 - 2 * (1 << l)) / span, 0, n - 1), pz1 = Math.Clamp((z1 + 2 * (1 << l)) / span, 0, n - 1);
                for (int py = pz0; py <= pz1; py++)
                    for (int px = px0; px <= px1; px++)
                        if (_pool.SlotOf(l, px, py) >= 0) _stale.Add((l, px, py));
            }
        }

        /// <summary>
        /// Fill the windows round the camera at once (a load, a new pool): every page they want, from the CPU store, on
        /// lists of their own that have run when this returns. The pool is <see cref="HeightPool.Filled"/> after.
        /// </summary>
        public void FillNow(TerrainPages pages, Vector2 cameraLocal, float spacing, float radius)
        {
            if (_pool.Texture == null || _pool.AllResident || pages is not { HasHeights: true }) return;
            _poolTexture = _pool.Texture;
            _staged.Clear();
            _stale.Clear();
            Choose(cameraLocal, spacing, radius, int.MaxValue);
            var device = Engine.Device;
            const int batch = 128;
            var buffer = device.CreateCommittedResource(new HeapProperties(HeapType.Upload), HeapFlags.None,
                ResourceDescription.Buffer((ulong)PageBytes * batch), ResourceStates.GenericRead, null);
            var all = new List<(int level, int px, int py, int slot)>(_staged);
            _staged.Clear();
            try
            {
                for (int first = 0; first < all.Count; first += batch)
                {
                    int n = Math.Min(batch, all.Count - first);
                    Fill(pages, buffer, all, first, n);
                    var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
                    var cmd = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, allocator, null);
                    try
                    {
                        RecordCopies(cmd, buffer, all, first, n);
                        if (first + n >= all.Count) _pool.RecordTable(cmd);
                        cmd.Close();
                        device.SubmitAndWait(cmd);
                    }
                    finally
                    {
                        cmd.Dispose();
                        allocator.Dispose();
                    }
                    for (int k = 0; k < n; k++) _pool.MarkWritten(all[first + k].slot, submitted: true);
                }
            }
            finally
            {
                buffer.Dispose();
            }
            _pool.Filled = true;
            UploadedLastFrame = all.Count;
            Waiting = 0;
        }

        /// <summary>
        /// This frame's changes: pages beyond their window let go, the nearest wanted ones given slots and staged from the
        /// CPU store, stale ones staged again, up to <paramref name="budget"/> pages. <see cref="Record"/> puts them on the
        /// frame's list. Main thread, before the frame's terrain is drawn.
        /// </summary>
        public void Update(TerrainPages pages, Vector2 cameraLocal, float spacing, float radius, int budget)
        {
            _staged.Clear();
            if (_pool.Texture == null || _pool.AllResident || pages is not { HasHeights: true }) return;
            if (!ReferenceEquals(_poolTexture, _pool.Texture))
            {
                // A new pool: nothing in it yet
                _poolTexture = _pool.Texture;
                _stale.Clear();
                _chosenAt = new Vector2(float.NaN);
            }

            // The windows only change when the camera has moved an eighth of a level 0 page since they were last looked at,
            // or pages are still waiting: the scan of every page of the windowed levels cost 0.5 ms a frame on a 16 km
            // terrain with nothing to do
            float eighth = 0.125f * Terrain.PageCells * spacing;
            if (Waiting > 0 || radius != _chosenRadius || !(Vector2.DistanceSquared(cameraLocal, _chosenAt) < eighth * eighth))
            {
                Choose(cameraLocal, spacing, radius, budget);
                _chosenAt = cameraLocal;
                _chosenRadius = radius;
            }

            // Stale pages, nearest first, in what budget is left
            if (_stale.Count > 0 && _staged.Count < budget)
            {
                _candidates.Clear();
                foreach (var (l, px, py) in _stale)
                    _candidates.Add((Distance(cameraLocal, px, py, Terrain.PageCells * spacing * (1 << l)) - l * 1e6f, l, px, py));
                _candidates.Sort((a, b) => a.key.CompareTo(b.key));
                foreach (var (_, l, px, py) in _candidates)
                {
                    if (_staged.Count >= budget) break;
                    _stale.Remove((l, px, py));
                    int slot = _pool.SlotOf(l, px, py);
                    if (slot >= 0) _staged.Add((l, px, py, slot));
                }
            }

            if (_staged.Count == 0) return;
            var device = Engine.Device;
            _stagedBuffer = device.CreateCommittedResource(new HeapProperties(HeapType.Upload), HeapFlags.None,
                ResourceDescription.Buffer((ulong)PageBytes * (ulong)_staged.Count), ResourceStates.GenericRead, null);
            Fill(pages, _stagedBuffer, _staged, 0, _staged.Count);
        }

        /// <summary>The staged pages' copies, then the page table, on the frame's list (CommandBuffer, before its terrain
        /// reads). The slots count as written by this frame.</summary>
        public void Record(ID3D12GraphicsCommandList cmd)
        {
            if (_staged.Count > 0 && _stagedBuffer != null)
            {
                RecordCopies(cmd, _stagedBuffer, _staged, 0, _staged.Count);
                foreach (var s in _staged) _pool.MarkWritten(s.slot);
                Engine.Device.DeferDispose(_stagedBuffer);   // read when the list runs
            }
            UploadedLastFrame = _staged.Count;
            _staged.Clear();
            _stagedBuffer = null;
            _pool.RecordTable(cmd);
        }

        // Let go of what is beyond its window, then give slots to the wanted pages not resident, coarsest and nearest
        // first, up to the budget, staging each
        private void Choose(Vector2 camera, float spacing, float radius, int budget)
        {
            float page0 = Terrain.PageCells * spacing;
            for (int l = 0; l < _pool.Levels && l < WindowedLevels; l++)
            {
                int n = _pool.PagesAcrossAt(l);
                float m = page0 * (1 << l), keep = radius * (1 << l) + 0.5f * m;
                for (int py = 0; py < n; py++)
                    for (int px = 0; px < n; px++)
                        if (_pool.SlotOf(l, px, py) >= 0 && Distance(camera, px, py, m) > keep)
                        {
                            _pool.Remove(l, px, py);
                            _stale.Remove((l, px, py));
                        }
            }

            _candidates.Clear();
            for (int l = _pool.Levels - 1; l >= 0; l--)
            {
                int n = _pool.PagesAcrossAt(l);
                float m = page0 * (1 << l), want = radius * (1 << l);
                for (int py = 0; py < n; py++)
                    for (int px = 0; px < n; px++)
                    {
                        if (_pool.SlotOf(l, px, py) >= 0) continue;
                        float d = Distance(camera, px, py, m);
                        if (l < WindowedLevels && d > want) continue;
                        // Coarser levels first (a page always lands over one), then nearest
                        _candidates.Add((l * -1e7f + d, l, px, py));
                    }
            }
            Waiting = _candidates.Count;
            if (_candidates.Count == 0) return;
            _candidates.Sort((a, b) => a.key.CompareTo(b.key));
            foreach (var (_, l, px, py) in _candidates)
            {
                if (_staged.Count >= budget) break;
                int slot = _pool.Add(l, px, py);
                if (slot < 0) break;   // full: the windows' capacity was sized for them, so only while the camera jumped
                _staged.Add((l, px, py, slot));
                Waiting--;
            }
        }

        // The texels of staged pages first .. first + count into the buffer, page k at k · PageBytes, side by side
        private static unsafe void Fill(TerrainPages pages, ID3D12Resource buffer, List<(int level, int px, int py, int slot)> list, int first, int count)
        {
            void* data;
            buffer.Map(0, null, &data);
            var basePtr = (IntPtr)data;
            System.Threading.Tasks.Parallel.For(0, count, k =>
            {
                var (level, px, py, _) = list[first + k];
                var span = new Span<ushort>((byte*)basePtr + (long)k * PageBytes, PageBytes / 2);
                pages.FillPoolPage(level, px, py, span, RowPitch / 2, HeightPool.Border, HeightPool.PageTexels);
            });
            buffer.Unmap(0);
        }

        // Copies of staged pages first .. first + count from the buffer into their slots, the pool from Common and back
        private void RecordCopies(ID3D12GraphicsCommandList cmd, ID3D12Resource buffer, List<(int level, int px, int py, int slot)> list, int first, int count)
        {
            cmd.ResourceBarrierTransition(_pool.Texture, ResourceStates.Common, ResourceStates.CopyDest);
            for (int k = 0; k < count; k++)
            {
                var (slice, cx, cy) = HeightPool.SlotCorner(list[first + k].slot);
                var src = new TextureCopyLocation(buffer, new PlacedSubresourceFootPrint
                {
                    Offset = (ulong)((long)k * PageBytes),
                    Footprint = new SubresourceFootPrint(Format.R16_UNorm, HeightPool.PageTexels, HeightPool.PageTexels, 1, RowPitch)
                });
                cmd.CopyTextureRegion(new TextureCopyLocation(_pool.Texture, (uint)slice), (uint)cx, (uint)cy, 0, src, null);
            }
            cmd.ResourceBarrierTransition(_pool.Texture, ResourceStates.CopyDest, ResourceStates.Common);
        }
    }
}
