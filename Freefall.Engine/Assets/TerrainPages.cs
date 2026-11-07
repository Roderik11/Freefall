using System;
using System.Numerics;
using System.Runtime.InteropServices;
using PhysX;
using Freefall.Base;

namespace Freefall.Assets
{
    /// <summary>
    /// The terrain's heights and holes on the CPU, in pages of <see cref="Terrain.PageCells"/> cells
    /// (TERRAIN_PAGES_PLAN.md "CPU", stage 4b): what GetHeight, IsHole, the physics cook, the navmesh bake, the sea mask
    /// and the project's scripts read. Until 4b this was Terrain.HeightField, a float[,] of the whole map (67 MB on
    /// Eldermoor, 4.3 GB at 32 km); pages of the pool's own 16-bit values take half that, and pages are what streams.
    ///
    /// A page holds its cells and the edge it shares with the next page, <see cref="PageSamples"/> a side, so the four
    /// samples of a bilinear read always lie on one page; a sample on a shared edge is held by both pages, equal. Page
    /// (px, pz) holds samples px · PageCells … px · PageCells + PageCells (as far as the terrain goes), row by row along
    /// +Z, X along a row. Heights are the GPU pool's R16 values as they are (0..65535 of MaxHeight above the terrain's
    /// base: terrain_pages.hlsli reads the same numbers), holes one byte per sample (0 = ground, 255 = cut; the ground is
    /// open where the bilinear weight is 0.5 or more), and a page without a hole holds no hole bytes.
    ///
    /// <b>Never changed once handed out.</b> A readback or a hole bake makes a new store that shares the pages it left as
    /// they were (<see cref="Edit"/>), so whoever took one (the sea mask's worker, the navmesh bake, the collider cook)
    /// reads the terrain as it was when it took it, and nothing is copied for them. No device in it: a server reads the
    /// same.
    /// </summary>
    public sealed class TerrainPages
    {
        /// <summary>Cells along a side of a page.</summary>
        public const int PageCells = Terrain.PageCells;

        /// <summary>Samples along a side of a page: its cells and the edge it shares with the next one.</summary>
        public const int PageSamples = PageCells + 1;

        /// <summary>Samples in a page.</summary>
        public const int PageLength = PageSamples * PageSamples;

        private const int PageShift = 8;   // log2(PageCells)

        /// <summary>Samples along a side of the terrain, the first and the last on its edges.</summary>
        public int Resolution { get; }

        /// <summary>Pages along a side.</summary>
        public int PagesAcross { get; }

        /// <summary>The heights are in: read back from the GPU or loaded with the asset. Without them every height is 0.</summary>
        public bool HasHeights { get; }

        /// <summary>A hole stamp cuts the terrain somewhere.</summary>
        public bool HasHoles => _holePages > 0;

        /// <summary>Pages that hold hole bytes.</summary>
        public int HolePages => _holePages;

        private readonly ushort[]?[] _heights;
        private readonly byte[]?[] _holes;
        private int _holePages;

        private TerrainPages(int resolution, bool hasHeights, ushort[]?[] heights, byte[]?[] holes)
        {
            Resolution = resolution;
            PagesAcross = PagesFor(resolution);
            HasHeights = hasHeights;
            _heights = heights;
            _holes = holes;
            foreach (var page in holes) if (page != null) _holePages++;
        }

        /// <summary>Pages along a side for a terrain of this many samples.</summary>
        public static int PagesFor(int resolution) => Math.Max(1, (Math.Max(1, resolution - 1) + PageCells - 1) / PageCells);

        /// <summary>A store of this size with no heights and no holes yet.</summary>
        public static TerrainPages Empty(int resolution)
        {
            int pages = PagesFor(resolution);
            return new TerrainPages(resolution, false, new ushort[]?[pages * pages], new byte[]?[pages * pages]);
        }

        /// <summary>
        /// A store of these pages (TerrainPagesFile.Read): one entry per page, row by row along +Z, each PageLength long or
        /// null. They are the store's from then on.
        /// </summary>
        internal static TerrainPages FromPages(int resolution, bool hasHeights, ushort[]?[] heights, byte[]?[] holes)
        {
            int count = PagesFor(resolution) * PagesFor(resolution);
            if (heights.Length != count || holes.Length != count)
                throw new ArgumentException($"{heights.Length} height and {holes.Length} hole pages for {count}");
            return new TerrainPages(resolution, hasHeights, heights, holes);
        }

        /// <summary>The samples page (px, pz) holds: its first sample and how many along X and Z (PageSamples, fewer on a
        /// terrain that ends inside its last page).</summary>
        public (int x0, int z0, int width, int height) PageExtent(int px, int pz)
        {
            int x0 = px << PageShift, z0 = pz << PageShift;
            return (x0, z0, Math.Min(PageSamples, Resolution - x0), Math.Min(PageSamples, Resolution - z0));
        }

        /// <summary>
        /// The pages that hold any sample of the box (x0, z0 inclusive, x1, z1 exclusive), as px0, pz0 inclusive, px1,
        /// pz1 exclusive: a sample on a shared edge is on both pages.
        /// </summary>
        public (int px0, int pz0, int px1, int pz1) PagesOf(int x0, int z0, int x1, int z1)
        {
            int last = PagesAcross;
            return (Math.Clamp((x0 - 1) >> PageShift, 0, last), Math.Clamp((z0 - 1) >> PageShift, 0, last),
                    Math.Clamp(((x1 - 1) >> PageShift) + 1, 0, last), Math.Clamp(((z1 - 1) >> PageShift) + 1, 0, last));
        }

        /// <summary>Page (px, pz)'s heights (PageSamples² R16 values, row by row along +Z), or null before any are in.</summary>
        public ReadOnlySpan<ushort> HeightPage(int px, int pz) => _heights[pz * PagesAcross + px];

        /// <summary>Page (px, pz)'s hole bytes, or null where no hole cuts it.</summary>
        public ReadOnlySpan<byte> HolePage(int px, int pz) => _holes[pz * PagesAcross + px];

        /// <summary>
        /// What page (px, pz) is made of: its height and hole arrays. A store is never changed once made and an edit makes a
        /// new array for every page it changes, so another reference means the page changed (TerrainCollider cooks again
        /// only those).
        /// </summary>
        internal (object? heights, object? holes) PageIdentity(int px, int pz) => (_heights[pz * PagesAcross + px], _holes[pz * PagesAcross + px]);

        // The page that holds the cell (x, z) to (x + 1, z + 1) and the cell's corner on it; the far edge's samples
        // fall on the last page
        private int PageOf(int x, int z, out int local)
        {
            int px = Math.Min(x >> PageShift, PagesAcross - 1), pz = Math.Min(z >> PageShift, PagesAcross - 1);
            local = (z - (pz << PageShift)) * PageSamples + x - (px << PageShift);
            return pz * PagesAcross + px;
        }

        // ── Heights ──

        /// <summary>Sample (x, z) as the R16 value the GPU holds, clamped to the terrain; 0 before the heights are in.</summary>
        public ushort Sample(int x, int z)
        {
            int last = Resolution - 1;
            x = Math.Clamp(x, 0, last);
            z = Math.Clamp(z, 0, last);
            var page = _heights[PageOf(x, z, out int i)];
            return page != null ? page[i] : (ushort)0;
        }

        /// <summary>Sample (x, z) in 0..1 of MaxHeight above the terrain's base, clamped to the terrain.</summary>
        public float Sample01(int x, int z) => Sample(x, z) / 65535.0f;

        /// <summary>
        /// The height at a place given in samples (sample i at i · VertexSpacing from the terrain's corner), bilinear between
        /// the four samples round it, in 0..1 of MaxHeight: TerrainRenderer.GetHeight's sum, the same in float as
        /// terrain_pages.hlsli's TerrainHeight01AtSample and, sample for sample, as the float[,] the CPU held before 4b.
        /// Clamped to the terrain.
        /// </summary>
        public float Height01(float sx, float sz)
        {
            int last = Resolution - 1;
            sx = Math.Clamp(sx, 0f, last);
            sz = Math.Clamp(sz, 0f, last);
            int x0 = (int)sx, z0 = (int)sz;
            int x1 = Math.Min(x0 + 1, last), z1 = Math.Min(z0 + 1, last);
            float fx = sx - x0, fz = sz - z0;

            var page = _heights[PageOf(x0, z0, out int i)];
            if (page == null) return 0f;
            int dx = x1 - x0, dz = (z1 - z0) * PageSamples;
            float h00 = page[i] / 65535.0f, h10 = page[i + dx] / 65535.0f;
            float h01 = page[i + dz] / 65535.0f, h11 = page[i + dz + dx] / 65535.0f;
            float near = h00 + (h10 - h00) * fx;
            float far = h01 + (h11 - h01) * fx;
            return near + (far - near) * fz;
        }

        /// <summary>The height at a terrain-local position (metres from the terrain's corner), in 0..1 of MaxHeight.</summary>
        public float Height01(Vector2 terrainSize, float localX, float localZ)
        {
            int last = Resolution - 1;
            return Height01(localX / terrainSize.X * last, localZ / terrainSize.Y * last);
        }

        /// <summary>
        /// The texels of height pool page (px, py) at a level (Graphics.HeightPool: <paramref name="texels"/> a side, the
        /// page's 257 samples inside a border of <paramref name="border"/>) into rows of <paramref name="pitch"/> values: a
        /// level L page holds every 2^L-th sample from (px, py) · 256 · 2^L, its border the samples beyond its edges at
        /// that spacing, clamped to the terrain (past its edge nothing reads them). Level 0 copies its own rows. Any thread.
        /// </summary>
        public void FillPoolPage(int level, int px, int py, Span<ushort> into, int pitch, int border, int texels)
        {
            int step = 1 << level;
            int x0 = (px << PageShift) << level, z0 = (py << PageShift) << level;
            if (level == 0 && px < PagesAcross && py < PagesAcross)
            {
                var (_, _, w, h) = PageExtent(px, py);
                var own = HeightPage(px, py);
                for (int tz = 0; tz < texels; tz++)
                {
                    var row = into.Slice(tz * pitch, texels);
                    int z = tz - border;
                    if (z >= 0 && z < h && !own.IsEmpty)
                    {
                        own.Slice(z * PageSamples, w).CopyTo(row.Slice(border, w));
                        for (int tx = 0; tx < border; tx++)
                            row[tx] = Sample(x0 + tx - border, z0 + z);
                        for (int tx = border + w; tx < texels; tx++)
                            row[tx] = Sample(x0 + tx - border, z0 + z);
                    }
                    else
                        for (int tx = 0; tx < texels; tx++)
                            row[tx] = Sample(x0 + tx - border, z0 + z);
                }
                return;
            }
            for (int tz = 0; tz < texels; tz++)
            {
                var row = into.Slice(tz * pitch, texels);
                int sz = z0 + (tz - border) * step;
                for (int tx = 0; tx < texels; tx++)
                    row[tx] = Sample(x0 + (tx - border) * step, sz);
            }
        }

        /// <summary>The lowest and highest sample of a box (x0, z0 to x1, z1, all inclusive, clamped to the terrain), as the
        /// R16 values held. (65535, 0) when the box is empty or no heights are in. Any thread.</summary>
        public (ushort min, ushort max) MinMax(int x0, int z0, int x1, int z1)
        {
            int last = Resolution - 1;
            x0 = Math.Clamp(x0, 0, last); x1 = Math.Clamp(x1, 0, last);
            z0 = Math.Clamp(z0, 0, last); z1 = Math.Clamp(z1, 0, last);
            ushort lo = ushort.MaxValue, hi = 0;
            for (int z = z0; z <= z1; z++)
            {
                int pz = Math.Min(z >> PageShift, PagesAcross - 1);
                int x = x0;
                while (x <= x1)
                {
                    int px = Math.Min(x >> PageShift, PagesAcross - 1);
                    int pageEnd = px == PagesAcross - 1 ? last : ((px + 1) << PageShift) - 1;
                    int run = Math.Min(x1, pageEnd) - x + 1;
                    var page = _heights[pz * PagesAcross + px];
                    if (page != null)
                    {
                        var span = page.AsSpan((z - (pz << PageShift)) * PageSamples + x - (px << PageShift), run);
                        foreach (var v in span)
                        {
                            if (v < lo) lo = v;
                            if (v > hi) hi = v;
                        }
                    }
                    x += run;
                }
            }
            return (lo, hi);
        }

        // ── Holes ──

        /// <summary>
        /// How far the holes cut at a terrain-local position (0..1, bilinear between the samples, as the terrain shader
        /// reads them). The ground is open where this is 0.5 or more. 0 off the terrain.
        /// </summary>
        public float HoleWeight(Vector2 terrainSize, float localX, float localZ)
        {
            if (_holePages == 0) return 0f;

            int last = Resolution - 1;
            float fx = localX / terrainSize.X * last;
            float fz = localZ / terrainSize.Y * last;
            if (!(fx >= 0f && fz >= 0f && fx <= last && fz <= last)) return 0f;

            int x0 = Math.Min((int)fx, last - 1), z0 = Math.Min((int)fz, last - 1);
            float tx = fx - x0, tz = fz - z0;
            var page = _holes[PageOf(x0, z0, out int i)];
            if (page == null) return 0f;
            float h0 = page[i] + (page[i + 1] - page[i]) * tx;
            float h1 = page[i + PageSamples] + (page[i + PageSamples + 1] - page[i + PageSamples]) * tx;
            return (h0 + (h1 - h0) * tz) * (1f / 255f);
        }

        /// <summary>True where the ground is open at a terrain-local position.</summary>
        public bool IsHole(Vector2 terrainSize, float localX, float localZ) => HoleWeight(terrainSize, localX, localZ) >= 0.5f;

        /// <summary>
        /// Whether the height-field cell with sample (x, z) as its low corner is open: the holes at the cell's middle, read
        /// as the terrain shader reads them (PhysX cook, PhysicsHelper.IsHoleCell before 4b, the same sums).
        /// </summary>
        public bool IsHoleCell(int x, int z)
        {
            int last = Resolution - 1;
            if (_holePages == 0 || x < 0 || z < 0 || x >= last || z >= last) return false;
            return HoleWeight(Vector2.One, (x + 0.5f) / last, (z + 0.5f) / last) >= 0.5f;
        }

        /// <summary>Hole samples that are open (128 or more).</summary>
        public long OpenHoleSamples()
        {
            long open = 0;
            for (int pz = 0; pz < PagesAcross; pz++)
                for (int px = 0; px < PagesAcross; px++)
                {
                    var page = _holes[pz * PagesAcross + px];
                    if (page == null) continue;
                    var (_, _, w, h) = PageExtent(px, pz);
                    // A shared edge counts once: on the page it begins
                    int ownW = px == PagesAcross - 1 ? w : w - 1, ownH = pz == PagesAcross - 1 ? h : h - 1;
                    for (int z = 0; z < ownH; z++)
                        for (int x = 0; x < ownW; x++)
                            if (page[z * PageSamples + x] >= 128) open++;
                }
            return open;
        }

        /// <summary>
        /// Copy a box of hole samples (x0, z0 and width × height) into a flat map, rows of <paramref name="pitch"/> bytes
        /// (the GPU's hole texture, the saved mask). Where no hole cuts, zeros.
        /// </summary>
        public void CopyHoles(int x0, int z0, int width, int height, Span<byte> destination, int pitch)
        {
            for (int z = 0; z < height; z++)
            {
                int gz = z0 + z;
                int pz = Math.Min(gz >> PageShift, PagesAcross - 1);
                var row = destination.Slice(z * pitch, width);
                int x = 0;
                while (x < width)
                {
                    int gx = x0 + x;
                    int px = Math.Min(gx >> PageShift, PagesAcross - 1);
                    int pageEnd = px == PagesAcross - 1 ? Resolution : (px + 1) << PageShift;
                    int run = Math.Min(width - x, pageEnd - gx);
                    var page = _holes[pz * PagesAcross + px];
                    if (page == null) row.Slice(x, run).Clear();
                    else page.AsSpan((gz - (pz << PageShift)) * PageSamples + gx - (px << PageShift), run).CopyTo(row.Slice(x, run));
                    x += run;
                }
            }
        }

        // ── PhysX ──

        /// <summary>
        /// Page (px, pz)'s PhysX height field samples, the page's own extent (<see cref="PageExtent"/>: w × h, row i = x,
        /// column j = z, so the field lies along the terrain's X and Z): each height as a 15-bit share of MaxHeight, each
        /// cell's triangles left out where a hole opens its middle. A page's field and its neighbour's share their edge
        /// samples, so the ground has no seam. Until 4c (2026-11-07) one field held the whole terrain (ToSamples). Any
        /// thread.
        ///
        /// PhysX.Net's HeightFieldSample is a class, an object per sample: <paramref name="reuse"/>, an array this gave
        /// before whose cook is done (PhysX.Net copies the samples out when it cooks), is filled again when it is the
        /// size this page needs. Allocated anew, the samples took 10.5 of a worker's 17 s cooking a 16 km terrain at 1 m
        /// per vertex (268 million objects).
        /// </summary>
        public HeightFieldSample[] PhysXSamples(int px, int pz, HeightFieldSample[]? reuse = null)
        {
            var (x0, z0, w, h) = PageExtent(px, pz);
            var page = _heights[pz * PagesAcross + px];
            var samples = reuse != null && reuse.Length == w * h ? reuse : new HeightFieldSample[w * h];
            for (int i = 0; i < w; i++)
                for (int j = 0; j < h; j++)
                {
                    // As from the float[,] before 4b: the share as a float, then to 15 bits
                    float share = page != null ? page[j * PageSamples + i] / 65535.0f : 0f;
                    short storedHeight = (short)(share * short.MaxValue);

                    // A sample's two materials are those of the cell it is the low corner of; material 127
                    // (PxHeightFieldMaterial::eHOLE) leaves the triangle out. The far row and column have no cell here.
                    byte material = i < w - 1 && j < h - 1 && IsHoleCell(x0 + i, z0 + j) ? PhysicsHelper.HoleMaterial : (byte)0;
                    var sample = samples[i * h + j] ??= new HeightFieldSample();
                    sample.MaterialIndex0 = new BitAndByte(material, true);
                    sample.MaterialIndex1 = new BitAndByte(material, false);
                    sample.Height = storedHeight;
                }
            return samples;
        }

        // ── Hash ──

        /// <summary>FNV-1a over the heights and the holes, page by page (a page without holes counts as such).</summary>
        public ulong Hash()
        {
            ulong h = 14695981039346656037UL;
            void Mix(ReadOnlySpan<byte> bytes)
            {
                var words = MemoryMarshal.Cast<byte, ulong>(bytes);
                foreach (var w in words) h = (h ^ w) * 1099511628211UL;
                for (int i = words.Length * 8; i < bytes.Length; i++) h = (h ^ bytes[i]) * 1099511628211UL;
                h = (h ^ (ulong)bytes.Length) * 1099511628211UL;
            }
            h = (h ^ (ulong)Resolution) * 1099511628211UL;
            foreach (var page in _heights)
                if (page != null) Mix(MemoryMarshal.AsBytes(page.AsSpan()));
                else h = (h ^ 1UL) * 1099511628211UL;
            foreach (var page in _holes)
                if (page != null) Mix(page);
                else h = (h ^ 2UL) * 1099511628211UL;
            return h;
        }

        // ── Making a new one ──

        /// <summary>A new store to be made from this one: pages set on it replace this one's, the rest are shared.</summary>
        public Builder Edit() => new(this);

        /// <summary>Builds the store that replaces another (<see cref="Edit"/>). The pages handed to it are its own from then on.</summary>
        public sealed class Builder
        {
            internal readonly TerrainPages Store;
            private readonly ushort[]?[] _heights;
            private readonly byte[]?[] _holes;

            internal Builder(TerrainPages from)
            {
                Store = from;
                _heights = (ushort[]?[])from._heights.Clone();
                _holes = (byte[]?[])from._holes.Clone();
            }

            public int Resolution => Store.Resolution;
            public int PagesAcross => Store.PagesAcross;

            /// <summary>Page (px, pz)'s heights from now on (PageLength values).</summary>
            public void SetHeights(int px, int pz, ushort[] page)
            {
                if (page.Length != PageLength) throw new ArgumentException($"a page holds {PageLength} samples");
                _heights[pz * PagesAcross + px] = page;
            }

            /// <summary>Page (px, pz)'s holes from now on (PageLength bytes), null for none.</summary>
            public void SetHoles(int px, int pz, byte[]? page)
            {
                if (page != null && page.Length != PageLength) throw new ArgumentException($"a page holds {PageLength} samples");
                _holes[pz * PagesAcross + px] = page;
            }

            /// <summary>No hole anywhere from now on.</summary>
            public void ClearHoles() => Array.Clear(_holes);

            /// <summary>Page (px, pz)'s holes as they stand in this edit.</summary>
            public byte[]? Holes(int px, int pz) => _holes[pz * PagesAcross + px];

            /// <summary>The new store.</summary>
            public TerrainPages Done(bool? hasHeights = null) =>
                new(Resolution, hasHeights ?? Store.HasHeights, _heights, _holes);
        }
    }
}
