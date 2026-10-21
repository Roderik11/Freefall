using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text.Json.Serialization;
using System.ComponentModel;
using Freefall.Base;
using Freefall.Graphics;
using Freefall.Reflection;

namespace Freefall.Assets
{
    /// <summary>Power-of-2+1 heightmap resolutions for edge-vertex alignment.</summary>
    public enum HeightmapResolution
    {
        _129  = 129,
        _257  = 257,
        _513  = 513,
        _1025 = 1025,
        _2049 = 2049,
        _4097 = 4097,
    }

    /// <summary>Power-of-2 control map resolutions. Inherit = use heightmap resolution.</summary>
    public enum ControlMapResolution
    {
        Inherit = 0,
        _256  = 256,
        _512  = 512,
        _1024 = 1024,
        _2048 = 2048,
        _4096 = 4096,
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

        /// <summary>
        /// Atomically set one or more dirty flags. Safe to call from any thread.
        /// </summary>
        public void MarkForUpdate(TerrainDirtyFlags flags)
            => System.Threading.Interlocked.Or(ref _dirtyFlags, (int)flags);

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

        /// <summary>The heights and holes as they are now are the saved ones (TerrainLoader: after a load, after a save).</summary>
        internal void MarkMapsSaved() => _savedMapsHash = MapsHash();

        /// <summary>
        /// After a bake: the heightmap and the hole mask are saved with this asset (Save Changed Assets writes them; a
        /// published game and its ground collider load them), so when they no longer equal the saved ones the asset
        /// goes on Save Changed Assets. Compared, not assumed from an edit: stamps and the generators that move them
        /// also rebake while a scene loads, to the same maps. Editor only, without the splat rebake <see cref="MarkDirty"/> adds.
        /// </summary>
        internal void CheckMapsAgainstSaved()
        {
            if (Engine.IsEditor && !IsDirty && MapsHash() != _savedMapsHash)
                base.MarkDirty();
        }

        // FNV-1a over the bits of the CPU height field and the hole mask (~10 ms for 4097²)
        private ulong MapsHash()
        {
            ulong h = 14695981039346656037UL;
            void Mix(ReadOnlySpan<byte> bytes)
            {
                var words = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ulong>(bytes);
                foreach (var w in words) h = (h ^ w) * 1099511628211UL;
                for (int i = words.Length * 8; i < bytes.Length; i++) h = (h ^ bytes[i]) * 1099511628211UL;
                h = (h ^ (ulong)bytes.Length) * 1099511628211UL;
            }
            if (HeightField is { } heights)
                Mix(System.Runtime.InteropServices.MemoryMarshal.CreateReadOnlySpan(
                    ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(heights), heights.Length * sizeof(float)));
            if (Holes is { } holes)
                Mix(holes);
            return h;
        }

        /// <summary>
        /// Resolution of the baked heightmap (power-of-2+1). Configurable per-terrain.
        /// </summary>
        public HeightmapResolution HeightmapResolution = HeightmapResolution._1025;

        /// <summary>Resolution of the baked layer weights. Inherit = use heightmap resolution.</summary>
        public ControlMapResolution SplatmapResolution = ControlMapResolution.Inherit;

        /// <summary>Resolution of the baked decoration coverage. Inherit = use heightmap resolution.</summary>
        public ControlMapResolution DecorationMapResolution = ControlMapResolution.Inherit;

        /// <summary>Effective heightmap resolution as int.</summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public int EffectiveHeightmapResolution => (int)HeightmapResolution;

        /// <summary>Effective splatmap resolution (falls back to heightmap).</summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public int EffectiveSplatmapResolution =>
            SplatmapResolution != ControlMapResolution.Inherit ? (int)SplatmapResolution : (int)HeightmapResolution;

        /// <summary>Effective decoration map resolution (falls back to heightmap).</summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public int EffectiveDecorationMapResolution =>
            DecorationMapResolution != ControlMapResolution.Inherit ? (int)DecorationMapResolution : (int)HeightmapResolution;

        /// <summary>Migrate old power-of-2 heightmap resolution to nearest power-of-2+1.</summary>
        internal void MigrateResolution()
        {
            int raw = (int)HeightmapResolution;
            if (raw > 1 && !Enum.IsDefined(typeof(HeightmapResolution), HeightmapResolution))
            {
                // Snap to next power-of-2+1: e.g. 1024 → 1025, 512 → 513
                int po2 = 1;
                while (po2 < raw) po2 <<= 1;
                var migrated = (HeightmapResolution)(po2 + 1);
                Debug.Log($"[Terrain] Migrating HeightmapResolution {raw} → {(int)migrated}");
                HeightmapResolution = migrated;
            }
        }

        /// <summary>
        /// GPU-baked heightmap texture (runtime only, regenerated from the height stamps in the scene).
        /// </summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public Texture BakedHeightmap;

        /// <summary>
        /// Serialized GUID reference to the saved baked heightmap DDS subasset.
        /// TerrainLoader loads this on startup so the game doesn't need to rebake.
        /// </summary>
        [Browsable(false)]
        public Texture BakedHeightmapRef;

        /// <summary>
        /// Pending baked heightmap bytes loaded from cache, awaiting GPU upload.
        /// Consumed by TerrainRenderer on the next Draw frame.
        /// </summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        internal byte[] PendingBakedHeightmapBytes;

        /// <summary>
        /// The final heightmap. All consumers (renderer, physics, decorators) should read this.
        /// Null until the first bake (or cache upload) has run.
        /// </summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public Texture Heightmap => BakedHeightmap;

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

        // ── HeightField — built internally from Heightmap ──
        [JsonIgnore]
        public float[,] HeightField { get; private set; }

        /// <summary>
        /// Pre-cooked PhysX height field. Populated during asset loading from the
        /// collision subasset so that RigidBody.Awake() doesn't need to cook on the main thread.
        /// </summary>
        [JsonIgnore]
        public PhysX.HeightField CookedHeightField { get; private set; }

        /// <summary>
        /// Set the pre-cooked PhysX height field. Called by TerrainLoader.
        /// </summary>
        internal void SetCookedHeightField(PhysX.HeightField hf) => CookedHeightField = hf;

        /// <summary>
        /// Sets the CPU-side height field directly from readback data.
        /// Called by TerrainRenderer after GPU heightmap readback.
        /// </summary>
        internal void SetHeightField(float[,] heights) => HeightField = heights;

        // ── Holes (HoleStamp) ──

        /// <summary>
        /// Serialized GUID reference to the saved hole mask (R8 DDS subasset, saved next to the baked heightmap).
        /// Null when the terrain had no holes when it was saved.
        /// </summary>
        [Browsable(false)]
        public Texture BakedHolesRef;

        /// <summary>
        /// Where the hole stamps cut the terrain, one byte per height sample: row by row along +Z, X along a row,
        /// sample (x, z) at terrain-local (x, z) * TerrainSize / (HoleResolution - 1), the PhysX height field's
        /// mapping. 0 = ground, 255 = fully cut; the edge of a hole is where it crosses 128. Null = no holes.
        /// Baked on the CPU (TerrainBaker.BakeHoles) because physics, the navmesh and PCG need it here.
        /// </summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public byte[] Holes { get; private set; }

        [Reflection.DontSerialize]
        [JsonIgnore]
        public int HoleResolution { get; private set; }

        /// <summary>True if any hole stamp cuts this terrain.</summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public bool HasHoles => Holes != null;

        /// <summary>Replaces the hole mask (null = no holes). Called by TerrainRenderer after a hole bake, TerrainLoader on load.</summary>
        internal void SetHoles(byte[] holes, int resolution)
        {
            Holes = holes;
            HoleResolution = holes != null ? resolution : 0;
        }

        /// <summary>
        /// How far the holes cut at a terrain-local position (0..1, bilinear between the samples, as the terrain
        /// shader reads it). The ground is open where this is 0.5 or more.
        /// </summary>
        public float HoleWeight(float localX, float localZ) => HoleWeight(Holes, HoleResolution, TerrainSize, localX, localZ);

        /// <summary>HoleWeight on a copy of a mask (the navmesh bake takes one, so it never reads one being re-cut).</summary>
        public static float HoleWeight(byte[] holes, int resolution, Vector2 terrainSize, float localX, float localZ)
        {
            if (holes == null) return 0f;

            int last = resolution - 1;
            float fx = localX / terrainSize.X * last;
            float fz = localZ / terrainSize.Y * last;
            if (!(fx >= 0f && fz >= 0f && fx <= last && fz <= last)) return 0f;

            int x0 = Math.Min((int)fx, last - 1), z0 = Math.Min((int)fz, last - 1);
            float tx = fx - x0, tz = fz - z0;
            int i = z0 * resolution + x0;
            float h0 = holes[i] + (holes[i + 1] - holes[i]) * tx;
            float h1 = holes[i + resolution] + (holes[i + resolution + 1] - holes[i + resolution]) * tx;
            return (h0 + (h1 - h0) * tz) * (1f / 255f);
        }

        /// <summary>True where the ground is open at a terrain-local position.</summary>
        public bool IsHole(float localX, float localZ) => HoleWeight(localX, localZ) >= 0.5f;

        /// <summary>
        /// True if the height-field cell between samples (x, z) and (x + 1, z + 1) is open: its centre lies in a
        /// hole. What the PhysX height field and the navmesh leave out.
        /// </summary>
        public bool IsHoleCell(int x, int z)
        {
            var holes = Holes;
            if (holes == null || x < 0 || z < 0 || x >= HoleResolution - 1 || z >= HoleResolution - 1) return false;
            int i = z * HoleResolution + x;
            return holes[i] + holes[i + 1] + holes[i + HoleResolution] + holes[i + HoleResolution + 1] >= 2 * 255;
        }
    }
}
