using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using System.Text.Json.Serialization;
using System.ComponentModel;
using Freefall.Base;
using Freefall.Graphics;
using Freefall.Reflection;

namespace Freefall.Assets
{
    /// <summary>
    /// A terrain density: half, one or two per metre (or metres per one). Powers of two only, since the quadtree halves
    /// (TERRAIN_PAGES_PLAN.md).
    /// </summary>
    public enum TerrainDensity
    {
        Half,
        One,
        Two,
    }

    /// <summary>
    /// Granular dirty flags consumed by TerrainRenderer each frame.
    /// Multiple flags can be set simultaneously; renderer clears them after processing.
    /// </summary>
    [Flags]
    public enum TerrainDirtyFlags
    {
        None           = 0,

        // ── Atomic flags ──
        /// <summary>Height stamps changed — rebake the heightmap.</summary>
        HeightBake     = 1 << 0,
        /// <summary>Splat stamps (or the height they filter on) changed — rebake the layer weights.</summary>
        SplatPack      = 1 << 1,
        /// <summary>Layer parameters changed (tiling, height scale) — re-upload buffers.</summary>
        LayerParams    = 1 << 2,
        /// <summary>The set of layers or their textures changed — rebuild Texture2DArrays.</summary>
        TextureArrays  = 1 << 3,
        /// <summary>Baked albedo is stale — re-dispatch albedo bake compute.</summary>
        AlbedoBake     = 1 << 4,
        /// <summary>The set of decorators or their variants changed (add/remove/mesh swap) — rebuild buffers.</summary>
        DecoStructure  = 1 << 5,
        /// <summary>Decorator parameters changed (density, scale, etc.) — re-upload data.</summary>
        DecoParams     = 1 << 6,
        /// <summary>Deco stamps (or the splat result they filter on) changed — rebake decoration coverage.</summary>
        DecoPrepass    = 1 << 7,
        /// <summary>Hole stamps changed — rebake the hole mask (CPU) and upload it.</summary>
        HoleBake       = 1 << 8,

        // ── Combinations ──
        /// <summary>Height change → rebake heightmap + albedo (slope/height masks shift).</summary>
        HeightAll      = HeightBake | AlbedoBake,
        /// <summary>Splat layer visuals changed → rebake + rebake albedo + refresh decorators.</summary>
        SplatAll       = LayerParams | AlbedoBake | DecoPrepass | SplatPack,
        /// <summary>Full decorator rebuild.</summary>
        DecoAll        = DecoStructure | DecoParams | DecoPrepass,
        /// <summary>Everything.</summary>
        All            = HeightBake | SplatPack | LayerParams | TextureArrays | AlbedoBake | DecoStructure | DecoParams | DecoPrepass | HoleBake,
    }

    /// <summary>
    /// Terrain asset — dimensions, resolutions and the cache of the last bake.
    ///
    /// A terrain holds no authored content. Its height, its ground materials and its ground cover are
    /// the result of compositing the terrain stamps in the scene (HeightStamp, SplatStamp, DecoStamp, ...),
    /// which reference <see cref="TerrainLayer"/> and <see cref="TerrainDecorator"/> assets. The layers
    /// and decorators a terrain renders are derived from those stamps each bake (TerrainRenderer.Palette).
    /// GPU rendering logic and Material live on TerrainRenderer (Component).
    /// </summary>
    [CreateAsset("Terrain")]
    public class Terrain : Asset
    {
        // ── Dirty Flags (thread-safe: render-thread lambdas set, main-thread Draw consumes) ──

        [Reflection.DontSerialize]
        [System.Text.Json.Serialization.JsonIgnore]
        private int _dirtyFlags = (int)TerrainDirtyFlags.All; // rebuild everything on first load

        // Where the next splat bake has to composite again (world XZ): what local stamps' edits touched. A SplatPack
        // asked for without a region (MarkForUpdate) is everywhere.
        [Reflection.DontSerialize]
        [System.Text.Json.Serialization.JsonIgnore]
        private readonly object _splatRegionLock = new();
        [Reflection.DontSerialize]
        [System.Text.Json.Serialization.JsonIgnore]
        private bool _splatEverywhere = true;
        [Reflection.DontSerialize]
        [System.Text.Json.Serialization.JsonIgnore]
        private System.Numerics.Vector2 _splatRegionMin = new(float.MaxValue), _splatRegionMax = new(float.MinValue);

        /// <summary>
        /// Atomically set one or more dirty flags. Safe to call from any thread. A splat rebake asked for here
        /// redoes the whole map (<see cref="MarkSplatRegion"/> for one place).
        /// </summary>
        public void MarkForUpdate(TerrainDirtyFlags flags)
        {
            if ((flags & TerrainDirtyFlags.SplatPack) != 0)
                lock (_splatRegionLock) _splatEverywhere = true;
            System.Threading.Interlocked.Or(ref _dirtyFlags, (int)flags);
        }

        /// <summary>
        /// A splat rebake of one place (world XZ, min and max): a local stamp's edit, where it was and where it is now.
        /// The bake composites only there; regions asked for before the bake runs add up.
        /// </summary>
        public void MarkSplatRegion(System.Numerics.Vector2 min, System.Numerics.Vector2 max)
        {
            lock (_splatRegionLock)
            {
                _splatRegionMin = System.Numerics.Vector2.Min(_splatRegionMin, min);
                _splatRegionMax = System.Numerics.Vector2.Max(_splatRegionMax, max);
            }
            System.Threading.Interlocked.Or(ref _dirtyFlags, (int)TerrainDirtyFlags.SplatPack);
        }

        /// <summary>
        /// Takes where the splat bake has to composite (world XZ; empty when nothing gave a region) and starts the next
        /// one empty. False: everywhere. Call it together with consuming SplatPack.
        /// </summary>
        public bool TakeSplatRegion(out System.Numerics.Vector2 min, out System.Numerics.Vector2 max)
        {
            lock (_splatRegionLock)
            {
                bool local = !_splatEverywhere;
                min = _splatRegionMin;
                max = _splatRegionMax;
                _splatEverywhere = false;
                _splatRegionMin = new System.Numerics.Vector2(float.MaxValue);
                _splatRegionMax = new System.Numerics.Vector2(float.MinValue);
                return local;
            }
        }

        /// <summary>
        /// Check if specific flags are set (non-consuming, momentary snapshot).
        /// </summary>
        public bool NeedsUpdate(TerrainDirtyFlags flags)
            => (System.Threading.Interlocked.CompareExchange(ref _dirtyFlags, 0, 0) & (int)flags) != 0;

        /// <summary>
        /// Atomically consume (clear) specific flags. Returns true if any were set.
        /// Uses Interlocked.And to avoid racing with render-thread MarkForUpdate.
        /// </summary>
        public bool ConsumeFlags(TerrainDirtyFlags flags)
        {
            int old = System.Threading.Interlocked.And(ref _dirtyFlags, ~(int)flags);
            return (old & (int)flags) != 0;
        }

        /// <summary>
        /// Inspector property changes — re-upload layer params and rebake the weights (cheap).
        /// </summary>
        public override void MarkDirty()
        {
            base.MarkDirty();
            MarkForUpdate(TerrainDirtyFlags.SplatAll);
        }

        // Hash of the heights and holes as last saved: read from the Cache at load, or written by a save
        private ulong _savedMapsHash;
        private bool _checkAfterLoad;

        /// <summary>The heights and holes as they are now are the saved ones (TerrainLoader: after a load, after a save).</summary>
        internal void MarkMapsSaved() => _savedMapsHash = MapsHash();

        /// <summary>The heights and holes are those last read from the pages file or written to it.</summary>
        internal bool MapsAreSaved => MapsHash() == _savedMapsHash;

        /// <summary>
        /// After a bake: the heights and holes are saved with this asset (its pages file: Save Changed Assets writes it,
        /// a published game and its ground collider load it), so when they no longer equal the saved ones the asset goes
        /// on Save Changed Assets. A terrain saved before 4c has no pages file, so its first bake puts it there. Compared, not assumed from an edit: stamps and the generators that move them
        /// also rebake while a scene loads, to the same maps. Editor only, without the splat rebake <see cref="MarkDirty"/> adds.
        /// </summary>
        internal void CheckMapsAgainstSaved()
        {
            if (!Engine.IsEditor || IsDirty) return;

            // Mid-load the maps can pass through states the scene does not end in: compared once it is ready
            if (SceneManager.Loading is { } load)
            {
                if (!_checkAfterLoad)
                {
                    _checkAfterLoad = true;
                    load.Completed += _ =>
                    {
                        _checkAfterLoad = false;
                        CheckMapsAgainstSaved();
                    };
                }
                return;
            }

            if (MapsHash() != _savedMapsHash)
            {
                base.MarkDirty();
                return;
            }

            // The bake equals the saved maps, but the stamps it came from are not what the asset says (a terrain saved
            // before the hash existed): now they are known. Saved once, a load can keep the map without baking.
            if (PagesStampsHash != 0 && BakedStampsHash != PagesStampsHash)
            {
                BakedStampsHash = PagesStampsHash;
                base.MarkDirty();
            }
        }

        // FNV-1a over the CPU store's heights and holes, page by page (TerrainPages.Hash)
        private ulong MapsHash() => Pages?.Hash() ?? 14695981039346656037UL;

        // ───── Densities (TERRAIN_PAGES_PLAN.md stage 1, 2026-11-07) ─────
        // The only resolution settings. Every map's size follows from them and the terrain's size, and so does the
        // quadtree's depth: the user, "having to chose the texture resolutions also needs to go".

        /// <summary>
        /// Metres between the terrain's vertices, with one height sample on every vertex: sample i lies at i × this from
        /// the terrain's corner, so the height map holds size / this + 1 samples a side (terrain_grid.hlsli).
        /// </summary>
        [Category("Density")]
        [Description("Metres between vertices, one height sample on each: sets the height map's samples and the quadtree's depth")]
        public TerrainDensity MetresPerVertex = TerrainDensity.One;

        /// <summary>Texels per metre of the layer weights (paint) and of the top-layer map GetLayer reads.</summary>
        [Description("Texels per metre of the layer weights (paint)")]
        public TerrainDensity PaintPixelsPerMetre = TerrainDensity.One;

        /// <summary>Quads along a side of the quadtree's patch (33 × 33 vertices): a leaf node spans this many vertices.</summary>
        public const int PatchQuads = 32;

        /// <summary>
        /// Cells along a side of a page, whatever the density (TERRAIN_PAGES_PLAN.md "Pages"): 256 m at 1 m per vertex. The
        /// height bake goes page by page (TerrainBaker.HeightPlan): page p holds samples p × PageCells up to the next
        /// page's first, the last page in a row also the terrain's far edge.
        /// </summary>
        public const int PageCells = 256;

        /// <summary>0.5, 1 or 2.</summary>
        public static float DensityValue(TerrainDensity density) => density switch
        {
            TerrainDensity.Half => 0.5f,
            TerrainDensity.Two => 2f,
            _ => 1f,
        };

        /// <summary>Metres between vertices and between height samples (<see cref="MetresPerVertex"/>).</summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public float VertexSpacing => DensityValue(MetresPerVertex);

        /// <summary>Height samples along a side: one on every vertex, the first and the last on the terrain's edges.</summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public int EffectiveHeightmapResolution => Math.Max(2, (int)MathF.Round(TerrainSize.X / VertexSpacing) + 1);

        /// <summary>Layer weight texels along a side, each at the middle of its cell (<see cref="PaintPixelsPerMetre"/>).</summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public int EffectiveSplatmapResolution => Math.Max(1, (int)MathF.Round(TerrainSize.X * DensityValue(PaintPixelsPerMetre)));

        /// <summary>
        /// Decoration control texels along a side: one a metre, whatever the densities, since a texel is one decoration
        /// tile and another size would change what grows and what it costs.
        /// </summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public int EffectiveDecorationMapResolution => Math.Max(1, (int)MathF.Round(TerrainSize.X));

        /// <summary>
        /// The quadtree's depth: a leaf node spans <see cref="PatchQuads"/> vertices. Exact only when the terrain's size is
        /// a power of two times PatchQuads × VertexSpacing (<see cref="GridIsExact"/>).
        /// </summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public int QuadtreeDepth => Math.Max(0, (int)MathF.Round(MathF.Log2(MathF.Max(1f, TerrainSize.X / (PatchQuads * VertexSpacing)))));

        /// <summary>Whether every vertex of the leaves sits on a height sample: a square terrain of PatchQuads × VertexSpacing × 2^depth.</summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public bool GridIsExact =>
            TerrainSize.X == TerrainSize.Y && MathF.Abs(PatchQuads * VertexSpacing * (1 << QuadtreeDepth) - TerrainSize.X) < 1e-3f;

        /// <summary>
        /// The baked heights on the GPU: pages in one array (terrain_pages.hlsli, TERRAIN_PAGES_PLAN.md 4a; until
        /// 2026-11-07 the flat BakedHeightmap texture). Runtime only, kept with the asset as that texture was, so a scene
        /// reloaded with the same stamps keeps its heights without a bake (TerrainRenderer.KeepsHeights). Made by the
        /// renderer's baker (TerrainBaker.AttachPool).
        /// </summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        internal Graphics.HeightPool? HeightPool;

        /// <summary>
        /// The GUID of the file the heights and holes are saved in, page by page (<see cref="TerrainPagesFile"/>, the
        /// Cache root's {guid}.pages): what a load reads instead of baking, and a game without a renderer (the server)
        /// has its ground from. Null until the first save. Since 4c (2026-11-07); before, BakedHeightmapRef and
        /// BakedHolesRef named a flat R16 heightmap and a flat hole mask, and a cooked PhysX height field of the whole
        /// terrain was saved beside them.
        /// </summary>
        [Browsable(false)]
        public string? PagesGuid;

        /// <summary>
        /// The height stamps the saved heights were baked from, as <see cref="TerrainBaker.HeightPlan.Hash"/>. A scene
        /// whose stamps hash the same keeps the saved heights as they are: no bake and no readback at load. 0 = not known.
        /// </summary>
        [Browsable(false)]
        public ulong BakedStampsHash;

        /// <summary>What the GPU heightmap was baked from: set when a bake runs, or to <see cref="BakedStampsHash"/> when the saved one is uploaded. 0 = not known.</summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        internal ulong HeightmapStampsHash;

        /// <summary>What the heights of <see cref="Pages"/> were baked from: set when the heights are read back, or by the loader from the saved map. 0 = not known.</summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        internal ulong PagesStampsHash;

        /// <summary>
        /// The pages in <see cref="Pages"/> came from the saved file and the GPU pool has yet to get them: TerrainRenderer
        /// uploads them (TerrainBaker.UploadPages) on its next Draw instead of baking.
        /// </summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        internal bool PendingPagesUpload;

        // ── Terrain dimensions ──
        public Vector2 TerrainSize = new(1700, 1700);
        public float MaxHeight = 600;

        /// <summary>
        /// Softness of transitions between texture layers, in splat-weight space.
        /// Low = crisp edge shaped by the layer height maps, 1 = fade spans the whole
        /// weight ramp (stamp Falloff / filter blends).
        /// </summary>
        [ValueRange(0.01f, 1f)]
        public float LayerBlendDepth = 0.2f;

        // ── Ground Coverage ──

        /// <summary>Multiplies the density of every decorator variant on this terrain.</summary>
        [DirtyFlag(TerrainDirtyFlags.DecoParams)]
        [ValueRange(0.1f, 10)]
        public float DecorationDensity = 0.1f;

        public float DecorationRadius = 100f;

        /// <summary>
        /// How much of the ground under decorators becomes the sward (0 = off, 1 = all of it): the plants' colour,
        /// tinted by the layers there (TerrainLayer.PlantTint) and lit as the plants are, at every distance, so the
        /// grass neither speckles with light ground near the camera nor ends at a line where its cards end. Hollows,
        /// crests and the large-scale variation still show through. A quarter less up close, where the scanned
        /// ground gives texture between the blades.
        /// </summary>
        [ValueRange(0f, 1f)]
        public float DecorationGroundTint = 0.8f;

        public bool DrawDetail = true;

        // ── The heights and holes on the CPU ──

        /// <summary>
        /// The heights and holes on the CPU, in pages (<see cref="TerrainPages"/>, TERRAIN_PAGES_PLAN.md 4b): what
        /// GetHeight, IsHole, the physics cook, the navmesh and the sea mask read. Set from the saved maps at load, then
        /// replaced (never changed) by every height readback and hole bake, so a reader that took one keeps a consistent
        /// terrain. Null before anything is known. Until 2026-11-07 the heights were HeightField, a float[,] of the whole
        /// map, and the holes a flat byte mask.
        /// </summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public TerrainPages? Pages { get; private set; }

        /// <summary>Replaces the CPU store (a readback, a hole bake, the loader).</summary>
        internal void SetPages(TerrainPages? pages) => Pages = pages;

        // ── Holes (HoleStamp) ──

        // Where the hole stamps cut the terrain is in Pages: one byte per height sample, 0 = ground, 255 = fully cut, the
        // edge of a hole where it crosses 128; a page no hole cuts holds none. Baked on the CPU (TerrainBaker.BakeHoles)
        // because physics, the navmesh and PCG need it here. Until 2026-11-07 a flat mask, Holes / HoleResolution.

        /// <summary>True if any hole stamp cuts this terrain.</summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public bool HasHoles => Pages is { HasHoles: true };

        /// <summary>
        /// How far the holes cut at a terrain-local position (0..1, bilinear between the samples, as the terrain
        /// shader reads it). The ground is open where this is 0.5 or more.
        /// </summary>
        public float HoleWeight(float localX, float localZ) => Pages?.HoleWeight(TerrainSize, localX, localZ) ?? 0f;

        /// <summary>True where the ground is open at a terrain-local position.</summary>
        public bool IsHole(float localX, float localZ) => HoleWeight(localX, localZ) >= 0.5f;
    }
}
