using System;
using System.Collections.Generic;
using System.Numerics;

namespace Freefall.Graphics
{
    /// <summary>
    /// A terrain map kept as pages of <see cref="TerrainMapResidency.PageTexels"/> texels a side in a pool (the paint,
    /// <see cref="PaintPool"/>; the decoration control, <see cref="DecoPool"/>): which page is in which slot, the border
    /// round each page, and the resource whose replacement means a new, empty pool.
    /// </summary>
    public interface IPagedMap
    {
        PageSlots Slots { get; }
        /// <summary>The pool's texture: another object means the pool was made again and holds nothing yet.</summary>
        object? Resource { get; }
        /// <summary>Texels round a page.</summary>
        int Border { get; }
    }

    /// <summary>
    /// Which pages of a paged terrain map (<see cref="IPagedMap"/>: the paint, TERRAIN_PAGES_PLAN.md 5b; the decoration
    /// control, 5c) its pool holds and which to bake this frame: N pages a side per level round the camera
    /// (<c>EngineSettings.TerrainPaintPages</c>, <c>TerrainDecoPages</c>), every page of the levels that have fewer; a page
    /// is let go half a page beyond its window. Pages are baked on arrival from the stamps, the coarsest and nearest first,
    /// a budget a frame; a page arrives over a coarser one already baked, so the ground is never without the map once the
    /// first fill is done. Pages over an edit (or new heights) are dirty and bake again: in the level 0 window at once, so
    /// an edit shows as it is dragged, the rest within the budget. With N = 0 every level 0 page is resident (the whole
    /// map at full detail).
    /// </summary>
    public sealed class TerrainMapResidency
    {
        /// <summary>Texels a page holds a side, at every level and for every map.</summary>
        public const int PageTexels = 256;

        private readonly IPagedMap _pool;
        private readonly HashSet<(int level, int px, int py)> _dirty = new();
        private readonly List<(int level, int px, int py, int slot)> _bakes = new();
        private readonly List<(float key, int level, int px, int py)> _candidates = new();
        private object? _poolTexture;
        private Vector2 _chosenAt = new(float.NaN);
        private int _chosenSide = -1;

        public TerrainMapResidency(IPagedMap pool) => _pool = pool;

        public IPagedMap Pool => _pool;

        /// <summary>Pages waiting for a slot or a bake after the last <see cref="Update"/>.</summary>
        public int Waiting { get; private set; }

        /// <summary>Pages baked last frame.</summary>
        public int BakedLastFrame { get; private set; }

        /// <summary>Level 0 pages a side for a map of this many texels.</summary>
        public static int PagesAcrossFor(int texels) => Math.Max(1, (texels + PageTexels - 1) / PageTexels);

        /// <summary>
        /// Page slots the windows can hold for a map of this many texels and N pages a side per level: 0 when N is 0 or
        /// the level 0 window would take every level 0 page anyway (every page resident).
        /// </summary>
        public static int CapacityFor(int texels, int pagesPerSide)
        {
            if (pagesPerSide <= 0 || texels <= 0) return 0;
            int across = PagesAcrossFor(texels);
            int side = pagesPerSide + 1;   // half a page of hysteresis each side
            if (side >= across) return 0;
            int levels = PageSlots.LevelsFor(across), total = 0;
            for (int l = 0; l < levels; l++)
            {
                int n = PageSlots.PagesAcrossAt(across, l);
                int s = Math.Min(side, n);
                total += s * s;
            }
            return total;
        }

        // The camera's distance to a page's square, in level 0 texels, along the farther axis (0 inside it)
        private static float Distance(Vector2 c, int px, int py, float m)
        {
            float dx = MathF.Max(0f, MathF.Max(px * m - c.X, c.X - (px + 1) * m));
            float dy = MathF.Max(0f, MathF.Max(py * m - c.Y, c.Y - (py + 1) * m));
            return MathF.Max(dx, dy);
        }

        /// <summary>The map changed over these level 0 texels (x0, z0 inclusive, x1, z1 exclusive; terrain UV × texels): the
        /// resident pages over them, at every level from <paramref name="minLevel"/> to <paramref name="maxLevel"/>, bake
        /// again.</summary>
        public void MarkDirty(int x0, int z0, int x1, int z1, int minLevel = 0, int maxLevel = int.MaxValue)
        {
            var slots = _pool.Slots;
            if (_pool.Resource == null) return;
            for (int l = minLevel; l < slots.Levels && l <= maxLevel; l++)
            {
                int n = slots.PagesAcrossAt(l), span = PageTexels << l, border = _pool.Border << l;
                int px0 = Math.Clamp((x0 - border) / span, 0, n - 1), px1 = Math.Clamp((x1 + border) / span, 0, n - 1);
                int pz0 = Math.Clamp((z0 - border) / span, 0, n - 1), pz1 = Math.Clamp((z1 + border) / span, 0, n - 1);
                for (int py = pz0; py <= pz1; py++)
                    for (int px = px0; px <= px1; px++)
                        if (slots.SlotOf(l, px, py) >= 0) _dirty.Add((l, px, py));
            }
        }

        /// <summary>Every resident page bakes again (a palette change, every stamp, new heights everywhere).</summary>
        public void MarkAllDirty()
        {
            var slots = _pool.Slots;
            if (_pool.Resource == null) return;
            for (int l = 0; l < slots.Levels; l++)
            {
                int n = slots.PagesAcrossAt(l);
                for (int py = 0; py < n; py++)
                    for (int px = 0; px < n; px++)
                        if (slots.SlotOf(l, px, py) >= 0) _dirty.Add((l, px, py));
            }
        }

        /// <summary>
        /// This frame's work: pages beyond their window let go, wanted ones given slots, and the list of pages to bake
        /// (arrivals and dirty ones), each with its slot. A new pool fills within the budget too, coarsest first, so the
        /// top page is there from its first frame. Main thread.
        /// <paramref name="camera"/>: the camera in level 0 texels (terrain UV × texels).
        /// </summary>
        public IReadOnlyList<(int level, int px, int py, int slot)> Update(Vector2 camera, int pagesPerSide, int budget)
        {
            _bakes.Clear();
            var slots = _pool.Slots;
            if (_pool.Resource == null) return _bakes;
            bool fresh = !ReferenceEquals(_poolTexture, _pool.Resource);
            if (fresh)
            {
                _poolTexture = _pool.Resource;
                _dirty.Clear();
                _chosenAt = new Vector2(float.NaN);
            }
            // A new pool used to bake every page it wanted in one frame. On a 16 km terrain whose biome outlines are walked
            // per rule, a coarse paint page is ~40 ms of GPU, and a whole window of them (~120 at 2 px a metre) is several
            // seconds in one command list: past Windows' 2 s timeout, which removes the device (TestWorld, 2026-11-08)
            int limit = budget;

            if (slots.AllResident)
            {
                // Every level 0 page: those never baked, then the dirty ones, all at once
                int n = slots.PagesAcross;
                for (int py = 0; py < n; py++)
                    for (int px = 0; px < n; px++)
                    {
                        int slot = slots.SlotOf(0, px, py);
                        if (!slots.Written(slot) || _dirty.Remove((0, px, py))) _bakes.Add((0, px, py, slot));
                    }
                _dirty.Clear();
                Waiting = 0;
                BakedLastFrame = _bakes.Count;
                return _bakes;
            }

            int h = (pagesPerSide - 1) / 2;
            if (fresh || Waiting > 0 || pagesPerSide != _chosenSide || !(Vector2.DistanceSquared(camera, _chosenAt) < 32f * 32f))
            {
                Choose(camera, h, limit);
                _chosenAt = camera;
                _chosenSide = pagesPerSide;
            }

            // Dirty pages: those in the level 0 window at once, the rest within what the budget leaves, nearest first
            if (_dirty.Count > 0)
            {
                _candidates.Clear();
                foreach (var (l, px, py) in _dirty)
                    _candidates.Add((Distance(camera, px, py, PageTexels << l) - l * 1e7f, l, px, py));
                _candidates.Sort((a, b) => a.key.CompareTo(b.key));
                float near = (h + 0.5f) * PageTexels;
                foreach (var (key, l, px, py) in _candidates)
                {
                    bool now = l == 0 && key <= near;
                    if (!now && _bakes.Count >= limit) continue;
                    _dirty.Remove((l, px, py));
                    int slot = slots.SlotOf(l, px, py);
                    if (slot >= 0) _bakes.Add((l, px, py, slot));
                }
            }
            BakedLastFrame = _bakes.Count;
            return _bakes;
        }

        // Let go of what is beyond its window, then give slots to the wanted pages not resident, coarsest and nearest first
        private void Choose(Vector2 camera, int h, int limit)
        {
            var slots = _pool.Slots;
            for (int l = 0; l < slots.Levels; l++)
            {
                int n = slots.PagesAcrossAt(l);
                float m = PageTexels << l, keep = (h + 0.5f) * m;
                for (int py = 0; py < n; py++)
                    for (int px = 0; px < n; px++)
                        if (slots.SlotOf(l, px, py) >= 0 && Distance(camera, px, py, m) > keep)
                        {
                            slots.Remove(l, px, py);
                            _dirty.Remove((l, px, py));
                        }
            }

            _candidates.Clear();
            for (int l = slots.Levels - 1; l >= 0; l--)
            {
                int n = slots.PagesAcrossAt(l);
                float m = PageTexels << l, want = h * m;
                for (int py = 0; py < n; py++)
                    for (int px = 0; px < n; px++)
                    {
                        if (slots.SlotOf(l, px, py) >= 0) continue;
                        float d = Distance(camera, px, py, m);
                        if (d > want) continue;
                        _candidates.Add((l * -1e7f + d, l, px, py));
                    }
            }
            Waiting = _candidates.Count;
            if (_candidates.Count == 0) return;
            _candidates.Sort((a, b) => a.key.CompareTo(b.key));
            foreach (var (_, l, px, py) in _candidates)
            {
                if (_bakes.Count >= limit) break;
                int slot = slots.Add(l, px, py);
                if (slot < 0) break;
                _bakes.Add((l, px, py, slot));
                Waiting--;
            }
        }
    }
}
