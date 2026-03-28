using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;
using System.ComponentModel;
using Freefall.Base;
using Freefall.Graphics;
using Freefall.Reflection;
using Vortice.DXGI;

namespace Freefall.Assets
{
    public enum DecoratorMode { Mesh, Billboard, Cross }

    /// <summary>
    /// Terrain asset — holds all resource data (heightmap, layers, splatmaps).
    /// GPU rendering logic and Material live on TerrainRenderer (Component).
    /// </summary>
    /// <summary>
    /// Granular dirty flags consumed by TerrainRenderer each frame.
    /// Multiple flags can be set simultaneously; renderer clears them after processing.
    /// </summary>
    [Flags]
    public enum TerrainDirtyFlags
    {
        None           = 0,

        // ── Atomic flags ──
        /// <summary>Height layers changed — rebake heightmap via GPU compositor.</summary>
        HeightBake     = 1 << 0,
        /// <summary>Splatmap control maps changed — repack into RGBA slices.</summary>
        SplatPack      = 1 << 1,
        /// <summary>Layer parameters changed (tiling, auto-mask) — re-upload buffers.</summary>
        LayerParams    = 1 << 2,
        /// <summary>Layer textures added/removed/swapped — rebuild Texture2DArrays.</summary>
        TextureArrays  = 1 << 3,
        /// <summary>Baked albedo is stale — re-dispatch albedo bake compute.</summary>
        AlbedoBake     = 1 << 4,
        /// <summary>Decorator structure changed (add/remove/mesh swap) — rebuild buffers.</summary>
        DecoStructure  = 1 << 5,
        /// <summary>Decorator parameters changed (density, scale, etc.) — re-upload data.</summary>
        DecoParams     = 1 << 6,
        /// <summary>Density maps painted — re-dispatch control prepass.</summary>
        DecoPrepass    = 1 << 7,

        // ── Combinations ──
        /// <summary>Height change → rebake heightmap + albedo (slope/height masks shift).</summary>
        HeightAll      = HeightBake | AlbedoBake,
        /// <summary>Splat layer visuals changed → repack + rebake albedo.</summary>
        SplatAll       = LayerParams | AlbedoBake,
        /// <summary>Full decorator rebuild.</summary>
        DecoAll        = DecoStructure | DecoParams | DecoPrepass,
        /// <summary>Everything.</summary>
        All            = HeightBake | SplatPack | LayerParams | TextureArrays | AlbedoBake | DecoStructure | DecoParams | DecoPrepass,
    }

    /// <summary>
    /// Terrain asset — holds all resource data (heightmap, layers, splatmaps).
    /// GPU rendering logic and Material live on TerrainRenderer (Component).
    /// </summary>
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
        /// Inspector property changes — re-upload layer params (cheap).
        /// Structural changes (add/remove layers) should explicitly call
        /// MarkForUpdate(TextureArrays) at the call site.
        /// </summary>
        public override void MarkDirty()
        {
            base.MarkDirty();
            MarkForUpdate(TerrainDirtyFlags.SplatAll);
        }

        /// <summary>
        /// Resolution of the baked heightmap (power of 2). Configurable per-terrain.
        /// </summary>
        public int HeightmapResolution = 1024;

        /// <summary>
        /// Non-destructive height layer stack. Composited bottom-to-top via GPU bake.
        /// </summary>
        public List<HeightLayer> HeightLayers = [];

        /// <summary>
        /// GPU-baked heightmap texture (runtime only, regenerated from HeightLayers).
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
        /// Stamp groups — decal-like height placements grouped by brush texture.
        /// Baked after HeightLayers, one GPU pass per group.
        /// </summary>
        public List<StampGroup> StampGroups = [];

        /// <summary>
        /// The final heightmap: baked result if available, otherwise the first ImportHeightLayer source.
        /// All consumers (renderer, physics, decorators) should read this.
        /// </summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public Texture Heightmap
        {
            get
            {
                if (BakedHeightmap != null) return BakedHeightmap;

                // Fallback: first enabled ImportHeightLayer source
                foreach (var layer in HeightLayers)
                {
                    if (layer is ImportHeightLayer import && import.Enabled && import.Source != null)
                        return import.Source;
                }

                return null;
            }
        }

        // ── Terrain dimensions ──
        public Vector2 TerrainSize = new(1700, 1700);
        public float MaxHeight = 600;
        public List<TextureLayer> Layers= [];

        /// <summary>
        /// Migration only: catches the old flat ControlMaps list during deserialization.
        /// Used directly by TerrainRenderer for RGBA-packed GPU splatmap array.
        /// Will be replaced by per-layer R16 ControlMaps after channel-split migration.
        /// </summary>
        [FormerlySerializedAs("ControlMaps")]
        [DontSerialize]
        public List<Texture> _legacyControlMaps;

        // ── Ground Coverage ──
        public List<Decoration> Decorations = [];
        
        [ValueRange(0.1f, 10)]
        public float DecorationDensity = 0.1f;

        public float DecorationRadius = 100f;

        public bool DrawDetail = true;

        [JsonIgnore] public int DecorationVersion { get; private set; }

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
        /// Builds the CPU-side height field from the current Heightmap texture.
        /// Must be called after Heightmap is assigned (e.g. during loading).
        /// </summary>
        public void BuildHeightField(string heightmapPath)
        {
            HeightField = Texture.ReadHeightField(heightmapPath);
        }

        /// <summary>
        /// Sets the CPU-side height field directly from readback data.
        /// Called by TerrainRenderer after GPU heightmap readback.
        /// </summary>
        internal void SetHeightField(float[,] heights) => HeightField = heights;

        // ── Texture Layer ──
        [Serializable]
        public class TextureLayer
        {
            [DirtyFlag(TerrainDirtyFlags.TextureArrays | TerrainDirtyFlags.AlbedoBake)]
            public Texture Diffuse;

            [DirtyFlag(TerrainDirtyFlags.TextureArrays | TerrainDirtyFlags.AlbedoBake)]
            public Texture Normals;

            [DirtyFlag(TerrainDirtyFlags.SplatAll)]
            public Vector2 Tiling = Vector2.One;

            /// <summary>Splatmap controlling where this layer paints (R8, hidden subasset).</summary>
            ///
            [Browsable(false)]
            public Texture ControlMap;

            /// <summary>Staging: raw R16 bytes loaded from cache, consumed by GPU upload.</summary>
            [Reflection.DontSerialize]
            [JsonIgnore]
            public byte[] PendingControlMapBytes;

            // ── Procedural Auto-Mask ──────────────────────────────────

            /// <summary>Height range for procedural mask (normalized 0..1 of MaxHeight).</summary>
            [DirtyFlag(TerrainDirtyFlags.SplatAll)]
            public Vector2 HeightRange = new(0, 1);

            /// <summary>Slope range for procedural mask (degrees, 0=flat, 90=cliff).</summary>
            [DirtyFlag(TerrainDirtyFlags.SplatAll)]
            public Vector2 SlopeRange = new(0, 90);

            /// <summary>Smooth blend width at height range edges (normalized 0..1).</summary>
            [DirtyFlag(TerrainDirtyFlags.SplatAll)]
            [ValueRange(0f, 0.5f)]
            public float HeightBlend = 0.05f;

            /// <summary>Smooth blend width at slope range edges (degrees).</summary>
            [DirtyFlag(TerrainDirtyFlags.SplatAll)]
            [ValueRange(0f, 30f)]
            public float SlopeBlend = 5.0f;

            /// <summary>Procedural auto-mask strength. 0=paint only, 1=full procedural.
            /// Final weight = max(painted, procedural * this).</summary>
            [DirtyFlag(TerrainDirtyFlags.SplatAll)]
            [ValueRange(0f, 1f)]
            public float ProceduralWeight = 0.0f;
        }

        // ── Ground Coverage Decoration ──
        /// <summary>
        /// A single decoration entry. References a StaticMesh or Texture for rendering,
        /// plus an optional ControlMap (grayscale) that controls placement density.
        /// At runtime, all unique ControlMaps are packed into a Texture2DArray.
        /// </summary>
        [Serializable]
        public class Decoration
        {
            [DirtyFlag(TerrainDirtyFlags.DecoStructure)]
            public DecoratorMode Mode = DecoratorMode.Mesh;

            [DirtyFlag(TerrainDirtyFlags.DecoStructure)]
            public StaticMesh Mesh;       // Mesh mode: geometry + LODs + material

            [DirtyFlag(TerrainDirtyFlags.DecoStructure)]
            public Texture Texture;       // Billboard/Cross mode: alpha-tested texture

            /// <summary>
            /// Controls placement density (R8, hidden subasset).
            /// Without a control map, the decorator renders everywhere.
            /// </summary>
            [FormerlySerializedAs("DensityMap")]
            [Browsable(false)]
            public Texture ControlMap;

            /// <summary>Staging: raw R16 bytes loaded from cache, consumed by GPU upload.</summary>
            [Reflection.DontSerialize]
            [JsonIgnore]
            public byte[] PendingControlMapBytes;

            /// <summary>Instances per square meter.</summary>
            [DirtyFlag(TerrainDirtyFlags.DecoParams)]
            [ValueRange(.1f, 10)]
            public float Density = 1.0f;

            [DirtyFlag(TerrainDirtyFlags.DecoParams)]
            [ValueRange(0f, 1f)]
            public float Weight = 1.0f;

            [DirtyFlag(TerrainDirtyFlags.DecoParams)]
            public Vector2 HeightRange = new(0.3f, 0.6f);

            [DirtyFlag(TerrainDirtyFlags.DecoParams)]
            public Vector2 WidthRange = new(0.2f, 0.4f);

            /// <summary>Root rotation applied to mesh vertices (euler degrees).</summary>
            [DirtyFlag(TerrainDirtyFlags.DecoParams)]
            public Vector3 RootRotation = new(-90, 0, 0);

            /// <summary>Blend factor for aligning to terrain slope (0=upright, 1=fully aligned).</summary>
            [DirtyFlag(TerrainDirtyFlags.DecoParams)]
            [ValueRange(-1, 1)]
            public float SlopeBias = 0.0f;

            /// <summary>Tint color for "healthy" instances (multiplicative). Alpha=0 means no tinting.</summary>
            [DirtyFlag(TerrainDirtyFlags.DecoParams)]
            public Vector4 HealthyColor = new(1, 1, 1, 1);

            /// <summary>Tint color for "dry" instances (multiplicative). Alpha=0 means no tinting.</summary>
            [DirtyFlag(TerrainDirtyFlags.DecoParams)]
            public Vector4 DryColor = new(1, 1, 1, 1);

            /// <summary>World-space noise frequency for healthy/dry blend. 0 = uniform healthy color.</summary>
            [DirtyFlag(TerrainDirtyFlags.DecoParams)]
            [ValueRange(0f, 1f)]
            public float NoiseSpread = 1.0f;
        }
    }
}
