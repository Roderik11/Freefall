using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;

using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Assets
{
    /// <summary>
    /// GPU compositor for terrain stamps. Turns the stamps that apply to a terrain into its three
    /// baked results:
    ///   height   — R16_UNorm heightmap                    (HeightStamp, HeightNoiseStamp)
    ///   splat    — packed layer-weight array              (SplatStamp)
    ///   deco     — decoration control texture             (DecoStamp)
    ///
    /// Each bake is two steps. Prepare* runs on the main thread: it collects the stamps in scope, sorts
    /// them by priority and copies everything the GPU needs into a plan, so the render thread never reads
    /// a component. Bake* runs inside a render callback and only uploads the plan and dispatches.
    ///
    /// The height bake goes page by page over a box of pages (the pages a stamp edit touched, or all of them); the
    /// splat bake over a box of texels. Stamps composite in ascending Priority (TerrainStamp.CompareBakeOrder).
    /// </summary>
    public class TerrainBaker : IDisposable
    {
        // ── Static: shared compute shaders + noise LUT (initialized once) ──
        private static ComputeShader _cs;
        private static int _kernelClear;
        private static int _kernelImport;
        private static int _kernelNoiseLayer;
        private static int _kernelInfluenceLayer;
        private static int _kernelCompareReads;
        private static ID3D12Resource _noiseLUTTex;
        private static uint _noiseLUTSRV;
        private static bool _initialized;

        private static ComputeShader _splatCS;
        private static int _kernelSplatBake;      // CS_SplatBakePage: one paint page into its slot
        private static int _kernelSplatBakeFlat;  // CS_SplatBakeFlat: the whole map into a flat array (Check Paint Pages)
        private static ComputeShader _decoCS;
        private static int _kernelDecoControl;   // the whole-map prepass over a region, for Check Deco Pages
        private static int _kernelDecoPage;
        private static bool _coverageInitialized;

        // ── Instance: per-terrain GPU resources ──

        // The heights: the pool of pages the bake writes and every reader reads (TERRAIN_PAGES_PLAN.md stage 4a). It is
        // the terrain asset's (Terrain.HeightPool), handed over by the renderer, so it outlives this baker as the flat
        // heightmap texture did
        private HeightPool _pool = new();

        /// <summary>The terrain's heights as pages (terrain_pages.hlsli).</summary>
        public HeightPool Pool => _pool;

        /// <summary>Bake into and read from this terrain's pool, made with the asset if it has none yet.</summary>
        public void AttachPool(Terrain terrain)
        {
            if (terrain == null) return;
            _pool = terrain.HeightPool ??= new HeightPool();
        }

        // Upload buffers (reused across bakes)
        private GraphicsBuffer _heightStampBuffer;
        private int _heightStampBufferCapacity;
        private GraphicsBuffer _heightStampSplineBuffer;
        private int _heightStampSplineCapacity;
        private GraphicsBuffer _heightPageListBuffer;
        private int _heightPageListCapacity;
        private GraphicsBuffer _heightStepBuffer;
        private int _heightStepCapacity;
        private HeightPlan _uploadedHeightPlan;    // the plan the four height buffers hold
        private (int, Vector2, float, float)? _bakedGrid;   // the last height bake's resolution, origin, spacing, MaxHeight

        private GraphicsBuffer _splatStampBuffer;
        private int _splatStampBufferCapacity;
        private GraphicsBuffer _splatSplineBuffer;
        private int _splatSplineCapacity;

        private GraphicsBuffer _decoStampBuffer;
        private int _decoStampBufferCapacity;
        private GraphicsBuffer _decoSplineBuffer;
        private int _decoSplineCapacity;
        private GraphicsBuffer _decoRowBuffer;
        private int _decoRowCapacity;

        /// <summary>
        /// The last decoration bake's row stamps (StructuredBuffer of DecoRowGPU) for the spawn, 0 when none grows in
        /// rows: a control texel whose channel holds slot 254 names one of them in its low byte.
        /// </summary>
        public uint DecoRowsSrv => _decoRowBuffer?.SrvIndex ?? 0u;

        /// <summary>Layers a terrain can render: 8 RGBA slices of the packed control array.</summary>
        public const int MaxLayers = 32;

        /// <summary>Decorators a terrain can render: slots of the decoration control texture.</summary>
        public const int MaxDecorators = 32;

        // ── GPU structs ──

        /// <summary>GPU struct matching HLSL HeightStampDescriptor. 132 bytes (33 x uint/float).</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct HeightStampDescriptorGPU
        {
            public Vector2 Center;           // terrain UV center (radial)
            public float Radius;             // UV-space inner radius
            public float Falloff;            // UV-space falloff width
            public float TargetHeight;       // normalized [0..1]
            public uint InvertShape;         // bit 0: the heightmap's relief digs instead of raising; bit 1: Outside
            public uint SplinePointOffset;   // into spline point buffer (0xFFFFFFFF = radial)
            public uint SplinePointCount;    // 0 = radial, high bit = closed area
            public float NoiseFreq;          // edge noise frequency (0 = disabled)
            public float NoiseAmp;           // edge noise amplitude in UV space
            public uint NoiseSeed;
            public uint HeightmapIdx;        // bindless SRV index (0 = no heightmap)
            public float HeightmapStrength;  // normalized strength (worldStrength / maxHeight)
            public float RotationSin;        // the heightmap's turn: UV delta → texture frame
            public float RotationCos;
            public uint BlendMode;           // HeightBlendMode
            public Vector2 HeightmapCenter;  // terrain UV of the heightmap's centre
            public Vector2 HeightmapExtent;  // UV half-size of the heightmap along its turned axes
            public uint MaskPointOffset;     // HeightStamp.MaskSpline's samples (a round stamp under a closed spline)
            public uint MaskPointCount;      // 0 = no mask, else count | closed bit
            public Vector2 BoundsMin;        // terrain UV box outside which the stamp's weight is 0 (StampBounds)
            public Vector2 BoundsMax;
            public Vector2 RidgeAcross;      // ridge and furrow (HeightStamp.RidgeSpacing): unit vector across the ridges
            public float RidgeSpacing;       // UV between crests (0 = no ridges)
            public float RidgeHeight;        // normalized furrow-to-crest height
            public float RidgeBend;          // UV the ridges swing aside at the field's ends
            public Vector2 RidgeAlong;       // the stamp's extent along the ridges: min, max of dot(uv, along)
        }

        /// <summary>GPU struct matching HLSL StampSplinePoint. 16 bytes.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct StampSplinePointGPU
        {
            public Vector2 UV;       // terrain UV position
            public float Height;     // normalized [0..1]
            public float HalfWidth;  // UV-space half-width
            public float Fade;       // the stamp's start / end fade here (TerrainStamp.StartFade, EndFade), 1 = none
        }

        /// <summary>GPU struct matching HLSL CoverageStamp (terrain_coverage.hlsli). 116 bytes.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct CoverageStampGPU
        {
            public Vector2 Center;
            public float Radius;
            public float Falloff;
            public uint SplinePointOffset;
            public uint SplinePointCount;
            public float NoiseFreq;
            public float NoiseAmp;
            public uint NoiseSeed;
            public uint Flags;               // CoverageGlobal | CoverageFilter | (op << 8)
            public uint Target;              // layer channel / decorator slot (AllTargets = every one)
            public float Strength;
            public float HeightMin, HeightMax, HeightBlend;
            public float SlopeMin, SlopeMax, SlopeBlend;
            public uint RequireMask;
            public uint ExcludeMask;
            public float PatchSize;          // metres, 0 = no patches
            public float PatchCoverage;
            public float PatchSoftness;
            public uint PatchSeed;
            public Vector4 Bounds;           // terrain UV min x, min y, max x, max y the shape can reach (CoverageBounds)
            public uint Rows;                // deco: 1 + index into CoveragePlan.Rows (0 = scattered)
        }

        /// <summary>
        /// GPU struct matching HLSL DecoRow (grass_compute.hlsl): one DecoStamp's rows. 32 bytes. The deco control
        /// texture names the one a texel holds; the spawn moves that decorator's plants onto them.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct DecoRowGPU
        {
            public Vector2 Normal;           // world XZ, across the rows
            public float Spacing;            // metres between row centres
            public float Width;              // metres, the band a row's plants stand in
            public uint Slot;                // the decorator slot the rows apply to
            public uint _pad0, _pad1, _pad2;
        }

        /// <summary>The most row stamps a decoration bake holds (texel marker 254, index in the low byte).</summary>
        private const int MaxDecoRows = 255;

        private const uint CoverageGlobal = 1u;
        private const uint CoverageFilter = 2u;
        private const uint CoverageSameShape = 4u;   // COVERAGE_SAME_SHAPE: reuse the previous stamp's shape weight
        private const uint AllTargets = 0xFFFFFFFFu;

        private static void EnsureInitialized()
        {
            if (_initialized) return;
            _cs = new ComputeShader("terrain_height_bake.hlsl");
            _kernelClear = _cs.FindKernel("CS_Clear");
            _kernelImport = _cs.FindKernel("CS_ImportLayer");
            _kernelNoiseLayer = _cs.FindKernel("CS_NoiseLayer");
            _kernelCompareReads = _cs.FindKernel("CS_CompareReads");
            _kernelInfluenceLayer = _cs.FindKernel("CS_InfluenceLayer");
            _initialized = true;
        }

        private static void EnsureCoverageInitialized()
        {
            if (_coverageInitialized) return;
            _splatCS = new ComputeShader("terrain_splat_bake.hlsl");
            _kernelSplatBake = _splatCS.FindKernel("CS_SplatBakePage");
            _kernelSplatBakeFlat = _splatCS.FindKernel("CS_SplatBakeFlat");
            _decoCS = new ComputeShader("decoration_prepass.hlsl");
            _kernelDecoControl = _decoCS.FindKernel("CSBuildDecoControl");
            _kernelDecoPage = _decoCS.FindKernel("CSBuildDecoPage");
            _coverageInitialized = true;
        }

        // ═══════════════════════════════════════════════════════════════════
        // ── Stamp collection ──
        // ═══════════════════════════════════════════════════════════════════

        /// <summary>
        /// The stamps of one type that take part in a terrain's bake, in bake order.
        /// </summary>
        public static List<T> CollectStamps<T>(TerrainRenderer renderer) where T : TerrainStamp
        {
            var result = new List<T>();
            var all = ComponentCache<T>.All;
            for (int i = 0; i < all.Count; i++)
            {
                var stamp = all[i];
                if (stamp != null && stamp.AppliesTo(renderer))
                    result.Add(stamp);
            }
            result.Sort(TerrainStamp.CompareBakeOrder);
            return result;
        }

        // ═══════════════════════════════════════════════════════════════════
        // ── Height ──
        // ═══════════════════════════════════════════════════════════════════

        internal enum HeightStepKind { Influence, Import, Noise }

        internal struct HeightStep
        {
            public HeightStepKind Kind;

            // Influence: a run of descriptors, and where its per-page lists start in HeightPlan.PageRuns
            public int Start, Count;
            public int PageRuns;

            // Import / Noise
            public uint BlendMode;
            public float Opacity;

            // Import: value × Scale + Bias over a world rectangle (metres: its first sample, 1 / its size)
            public uint SourceSrv;
            public float Scale, Bias;
            public Vector2 ImportMin, ImportInvSize;

            // Noise, in world metres: Frequency in LUT units per metre, Offset, MaskCenter and MaskRadius in metres
            public uint NoiseType, Octaves, Seed, TerraceSteps;
            public float Frequency, Amplitude, Lacunarity, Persistence, TerraceSmoothness;
            public Vector2 Offset, MaskCenter;
            public float MaskRadius, MaskFalloff;

            // Noise along the entity's spline instead of the circle (HeightNoiseStamp on a spline): its samples in the
            // plan's spline points, radius and falloff in terrain UV (scaled by the spline's widths as any spline stamp's),
            // and the UV box it reaches; pages outside the box are not dispatched
            public uint MaskSplineOffset, MaskSplineCount;
            public float MaskSplineRadius, MaskSplineFalloff;
            public Vector2 MaskBoundsMin, MaskBoundsMax;
            public uint Centred;            // HeightNoiseStamp.Centred: the noise around 0

            public HeightStepGPU ToGpu() => new()
            {
                BlendMode = BlendMode, Opacity = Opacity,
                Source = SourceSrv, ImportScale = Scale, ImportBias = Bias, ImportMin = ImportMin, ImportInvSize = ImportInvSize,
                NoiseType = NoiseType, Octaves = Octaves, Frequency = Frequency, Amplitude = Amplitude,
                Lacunarity = Lacunarity, Persistence = Persistence, Offset = Offset, Seed = Seed,
                TerraceSteps = TerraceSteps, TerraceSmoothness = TerraceSmoothness,
                MaskCenter = MaskCenter, MaskRadius = MaskRadius, MaskFalloff = MaskFalloff,
                MaskSplineOffset = MaskSplineOffset, MaskSplineCount = MaskSplineCount,
                MaskSplineRadius = MaskSplineRadius, MaskSplineFalloff = MaskSplineFalloff,
                MaskBoundsMin = MaskBoundsMin, MaskBoundsMax = MaskBoundsMax,
                NoiseCentred = Centred,
            };
        }

        /// <summary>A step's parameters for the GPU: terrain_height_bake.hlsl BakeStep, 144 bytes.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct HeightStepGPU
        {
            public uint BlendMode;
            public float Opacity;
            public uint Source;
            public float ImportScale, ImportBias;
            public Vector2 ImportMin, ImportInvSize;
            public uint NoiseType, Octaves;
            public float Frequency, Amplitude, Lacunarity, Persistence;
            public Vector2 Offset;
            public uint Seed, TerraceSteps;
            public float TerraceSmoothness;
            public Vector2 MaskCenter;
            public float MaskRadius, MaskFalloff;
            public uint MaskSplineOffset, MaskSplineCount;
            public float MaskSplineRadius, MaskSplineFalloff;
            public Vector2 MaskBoundsMin, MaskBoundsMax;
            public uint NoiseCentred;
            public uint _pad0, _pad1, _pad2;
        }

        /// <summary>
        /// Everything a height bake needs, captured on the main thread. The bake goes page by page
        /// (<see cref="Terrain.PageCells"/>): each page runs every step, and an influence run only with the descriptors
        /// whose bounds reach the page (its list in <see cref="PageStamps"/>). A sample depends on its index in the grid
        /// and on the stamps alone, so a page baked by itself equals the same samples of a whole bake to the bit.
        /// </summary>
        public sealed class HeightPlan
        {
            internal readonly List<HeightStep> Steps = new();
            internal readonly List<HeightStampDescriptorGPU> Descriptors = new();
            internal readonly List<StampSplinePointGPU> SplinePoints = new();
            internal readonly List<string> Sources = new();   // GUIDs of the heightmap textures, in the order the plan uses them
            internal int Resolution;
            internal Vector2 TerrainSize;
            internal float MaxHeight;
            internal Vector2 Origin;        // world X, Z of sample (0, 0)
            internal float Spacing;         // metres between samples

            // Pages: PagesAcross² of them, row by row. Each influence step has PagesAcross² entries in PageRuns from its
            // HeightStep.PageRuns on: the page's descriptors of that run, as a range of PageStamps (descriptor indices
            // in bake order).
            internal int PagesAcross;
            internal readonly List<uint> PageStamps = new();
            internal readonly List<(int Start, int Count)> PageRuns = new();
            internal int AllStamps;         // where PageStamps holds every descriptor once, in order (a run over the whole map)

            /// <summary>True if any stamp contributes; a bake without work is skipped so a terrain
            /// with no height stamps keeps the heightmap it was saved with.</summary>
            public bool HasWork => Steps.Count > 0;

            /// <summary>Pages along a side.</summary>
            public int Pages => PagesAcross;

            /// <summary>The samples of page (px, py): x0, y0 inclusive, x1, y1 exclusive. The last page in a row or a
            /// column also holds the far edge's samples.</summary>
            public (int x0, int y0, int x1, int y1) PageSamples(int px, int py) =>
                PageSamples(px, py, PagesAcross, Resolution);

            internal static (int x0, int y0, int x1, int y1) PageSamples(int px, int py, int pagesAcross, int resolution)
            {
                int x0 = px * Terrain.PageCells, y0 = py * Terrain.PageCells;
                int x1 = px == pagesAcross - 1 ? resolution : Math.Min(resolution, x0 + Terrain.PageCells);
                int y1 = py == pagesAcross - 1 ? resolution : Math.Min(resolution, y0 + Terrain.PageCells);
                return (x0, y0, x1, y1);
            }

            /// <summary>The samples page (px, py) holds in the pool: its 257 and the border round them, x0, y0 inclusive
            /// (before the terrain at its first page), x1, y1 exclusive.</summary>
            internal (int x0, int y0, int x1, int y1) PageBox(int px, int py)
            {
                int x0 = px * Terrain.PageCells - Graphics.HeightPool.Border, y0 = py * Terrain.PageCells - Graphics.HeightPool.Border;
                return (x0, y0, x0 + Graphics.HeightPool.PageTexels, y0 + Graphics.HeightPool.PageTexels);
            }

            /// <summary>The page box (px0, py0 inclusive, px1, py1 exclusive) holding the samples of a world XZ box, or
            /// every page for null.</summary>
            public (int px0, int py0, int px1, int py1) PagesOf(Vector2? worldMin, Vector2? worldMax)
            {
                if (worldMin is not { } min || worldMax is not { } max || Spacing <= 0)
                    return (0, 0, PagesAcross, PagesAcross);
                int Page(float world, float origin) =>
                    (int)Math.Clamp(MathF.Floor((world - origin) / Spacing / Terrain.PageCells), 0f, PagesAcross - 1);
                // A sample more each side against rounding at the box's edges, and the border: a page's border holds its
                // neighbours' samples, so it is baked again when they change
                float pad = (1 + Graphics.HeightPool.Border) * Spacing;
                return (Page(min.X - pad, Origin.X), Page(min.Y - pad, Origin.Y),
                        Page(max.X + pad, Origin.X) + 1, Page(max.Y + pad, Origin.Y) + 1);
            }

            /// <summary>
            /// What the bake of this plan comes out as, as a hash: every step, descriptor and spline point (the bindless
            /// indices left out, the textures they name by GUID), the resolution, the terrain's size and where its first
            /// sample lies, and the bake shader's source. Equal hashes bake equal heightmaps, so a heightmap baked from
            /// an equal hash can be kept (Terrain.BakedStampsHash). A heightmap texture reimported under the same GUID is
            /// not seen.
            /// </summary>
            public ulong Hash => _hash ??= ComputeHash();
            private ulong? _hash;

            private ulong ComputeHash()
            {
                ulong h = Fnv.Start;
                h = Fnv.Mix(h, Resolution);
                h = Fnv.Mix(h, TerrainSize.X); h = Fnv.Mix(h, TerrainSize.Y); h = Fnv.Mix(h, MaxHeight);
                h = Fnv.Mix(h, Origin.X); h = Fnv.Mix(h, Origin.Y); h = Fnv.Mix(h, Spacing);
                h = Fnv.Mix(h, ShaderHash);

                var steps = Steps.ToArray();
                for (int i = 0; i < steps.Length; i++) steps[i].SourceSrv = 0;
                h = Fnv.Mix(h, MemoryMarshal.AsBytes(steps.AsSpan()));

                var descriptors = Descriptors.ToArray();
                for (int i = 0; i < descriptors.Length; i++) descriptors[i].HeightmapIdx = descriptors[i].HeightmapIdx != 0 ? 1u : 0u;
                h = Fnv.Mix(h, MemoryMarshal.AsBytes(descriptors.AsSpan()));

                h = Fnv.Mix(h, MemoryMarshal.AsBytes(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(SplinePoints)));
                foreach (var source in Sources)
                    h = Fnv.Mix(h, System.Text.Encoding.UTF8.GetBytes(source));
                return h == 0 ? 1 : h;   // 0 means "not known" on the terrain
            }
        }

        // The height bake shader and everything it includes, hashed once per process: a changed kernel bakes
        // differently from the same stamps
        private static ulong? _shaderHash;
        private static ulong ShaderHash => _shaderHash ??= HashShader("terrain_height_bake.hlsl");

        private static ulong HashShader(string file)
        {
            ulong h = Fnv.Start;
            var path = Graphics.ShaderHotReload.ResolvePath(file);
            if (path == null) return h;
            var files = new List<string>(Graphics.ShaderHotReload.CollectDependencies(path)) { path };
            files.Sort(StringComparer.OrdinalIgnoreCase);
            foreach (var f in files.Distinct(StringComparer.OrdinalIgnoreCase))
                if (EngineFiles.Exists(f))
                    h = Fnv.Mix(h, EngineFiles.ReadAllBytes(f));
            return h;
        }

        /// <summary>64-bit FNV-1a, for the bake hashes.</summary>
        internal static class Fnv
        {
            public const ulong Start = 14695981039346656037UL;
            private const ulong Prime = 1099511628211UL;

            public static ulong Mix(ulong h, ReadOnlySpan<byte> bytes)
            {
                foreach (var b in bytes) h = (h ^ b) * Prime;
                return (h ^ (ulong)bytes.Length) * Prime;
            }

            public static ulong Mix(ulong h, ulong value) => (h ^ value) * Prime;
            public static ulong Mix(ulong h, int value) => Mix(h, (ulong)(uint)value);
            public static ulong Mix(ulong h, float value) => Mix(h, (ulong)BitConverter.SingleToUInt32Bits(value));
        }

        /// <summary>
        /// Collect the height stamps that apply to the terrain and lay out the bake: consecutive
        /// HeightStamps become one step, and every whole-terrain operation (import, noise) is a step of its
        /// own at its place in the priority order. Main thread.
        /// </summary>
        public HeightPlan PrepareHeight(Terrain terrain, TerrainRenderer renderer)
        {
            var terrainSize = terrain.TerrainSize;
            float maxHeight = terrain.MaxHeight;
            var terrainOrigin = renderer.Transform?.WorldPosition ?? Vector3.Zero;
            int resolution = terrain.EffectiveHeightmapResolution;
            var plan = new HeightPlan
            {
                Resolution = resolution,
                TerrainSize = terrainSize,
                MaxHeight = maxHeight,
                Origin = new Vector2(terrainOrigin.X, terrainOrigin.Z),
                // Sample i at i / (N - 1) of the terrain, as the influence kernel and every reader have it
                Spacing = terrainSize.X / (resolution - 1),
            };

            var stamps = new List<TerrainStamp>();
            stamps.AddRange(CollectStamps<HeightStamp>(renderer));
            stamps.AddRange(CollectStamps<HeightNoiseStamp>(renderer));
            stamps.Sort(TerrainStamp.CompareBakeOrder);

            int runStart = 0;
            void CloseRun()
            {
                int count = plan.Descriptors.Count - runStart;
                if (count > 0)
                    plan.Steps.Add(new HeightStep { Kind = HeightStepKind.Influence, Start = runStart, Count = count });
                runStart = plan.Descriptors.Count;
            }

            foreach (var stamp in stamps)
            {
                switch (stamp)
                {
                    case HeightStamp height when height.IsGlobal:
                    {
                        // Whole-terrain import: nothing to do without a heightmap
                        if (height.Heightmap == null || height.Heightmap.BindlessIndex == 0) break;
                        CloseRun();
                        plan.Sources.Add(height.Heightmap.Guid ?? height.Heightmap.Name ?? "");
                        // World Y, as the target height of a local stamp (see TryBuildHeightDescriptor)
                        float baseY = (height.Transform?.WorldPosition.Y ?? 0f) + height.HeightOffset - terrainOrigin.Y;
                        plan.Steps.Add(new HeightStep
                        {
                            Kind = HeightStepKind.Import,
                            BlendMode = (uint)HeightBlendMode.Set,
                            Opacity = 1f,
                            SourceSrv = height.Heightmap.BindlessIndex,
                            Scale = height.Strength / maxHeight * (height.InvertShape ? -1f : 1f),
                            Bias = baseY / maxHeight,
                            // Laid over the terrain as it is, in world metres: its first sample on the terrain's
                            // corner, its last on the far one
                            ImportMin = plan.Origin,
                            ImportInvSize = new Vector2(1f / terrainSize.X, 1f / terrainSize.Y),
                        });
                        break;
                    }

                    case HeightStamp height:
                        if (TryBuildHeightDescriptor(height, terrainOrigin, terrainSize, maxHeight, plan.SplinePoints, out var desc))
                        {
                            plan.Descriptors.Add(desc);
                            if (height.Heightmap != null) plan.Sources.Add(height.Heightmap.Guid ?? height.Heightmap.Name ?? "");
                        }
                        break;

                    case HeightNoiseStamp noise:
                    {
                        CloseRun();
                        var step = new HeightStep
                        {
                            Kind = HeightStepKind.Noise,
                            BlendMode = (uint)noise.BlendMode,
                            Opacity = noise.Opacity,
                            NoiseType = (uint)noise.Type,
                            Octaves = (uint)Math.Clamp(noise.Octaves, 1, 12),
                            // The LUT holds NoiseLUTPeriod cells per unit; a cell of the first octave is FeatureSize m
                            Frequency = 1f / (NoiseLUTPeriod * Math.Max(noise.FeatureSize, 0.01f)),
                            Amplitude = noise.Amplitude,
                            Lacunarity = noise.Lacunarity,
                            Persistence = noise.Persistence,
                            Offset = noise.Offset,
                            Seed = (uint)noise.Seed,
                            TerraceSteps = (uint)Math.Max(0, noise.TerraceSteps),
                            TerraceSmoothness = noise.TerraceSmoothness,
                            Centred = noise.Centred ? 1u : 0u,
                        };
                        if (!noise.IsGlobal && noise.IsSplineMode)
                        {
                            // Along the entity's spline: its corridor (or closed area) with Radius and Falloff scaled by
                            // the spline's widths, weighed as any spline stamp's shape (EvaluateStampWeightEx), e.g. pools
                            // and bars along a river's bed. A spline with too few points makes no noise at all
                            if (!BuildSplinePoints(noise, terrainOrigin, terrainSize, maxHeight, 0f, plan.SplinePoints,
                                    out uint offset, out uint count, out float uvRadius))
                                break;
                            float uvScale = 1f / Math.Max(terrainSize.X, terrainSize.Y);
                            step.MaskSplineOffset = offset;
                            step.MaskSplineCount = count;
                            step.MaskSplineRadius = uvRadius;
                            step.MaskSplineFalloff = noise.Falloff * uvScale;
                            var reach = new HeightStampDescriptorGPU
                            {
                                Radius = uvRadius, Falloff = step.MaskSplineFalloff,
                                SplinePointOffset = offset, SplinePointCount = count,
                            };
                            StampBounds(ref reach, plan.SplinePoints, false);
                            step.MaskBoundsMin = reach.BoundsMin;
                            step.MaskBoundsMax = reach.BoundsMax;
                        }
                        else if (!noise.IsGlobal)
                        {
                            // Radial fade around the entity: full strength at the centre, gone at Radius + Falloff.
                            // The kernel's mask is 1 - (dist / radius)^exponent; a short falloff means a hard edge.
                            var center = noise.Transform?.WorldPosition ?? Vector3.Zero;
                            float extent = noise.Radius + noise.Falloff;
                            step.MaskCenter = new Vector2(center.X, center.Z);
                            step.MaskRadius = Math.Max(extent, 0.01f);
                            step.MaskFalloff = Math.Clamp(2f * extent / Math.Max(noise.Falloff, 0.01f), 1f, 64f);
                        }
                        plan.Steps.Add(step);
                        break;
                    }

                }
            }
            CloseRun();

            BuildPageLists(plan);
            return plan;
        }

        /// <summary>
        /// Every influence run's descriptors page by page: a descriptor goes into the list of every page its bounds
        /// (StampBounds) reach, a sample more each side, in bake order. The kernel tests the bounds per sample as well,
        /// so a page's list only has to hold every stamp that can touch it; an Outside stamp is in every list. After the
        /// lists, every descriptor once in order (HeightPlan.AllStamps), for a run over the whole map.
        /// </summary>
        private static void BuildPageLists(HeightPlan plan)
        {
            int cells = plan.Resolution - 1;
            int pages = Math.Max(1, (cells + Terrain.PageCells - 1) / Terrain.PageCells);
            plan.PagesAcross = pages;

            List<uint>[] buckets = null;
            for (int s = 0; s < plan.Steps.Count; s++)
            {
                var step = plan.Steps[s];
                if (step.Kind != HeightStepKind.Influence) continue;

                buckets ??= new List<uint>[pages * pages];
                foreach (var bucket in buckets) bucket?.Clear();
                for (int d = step.Start; d < step.Start + step.Count; d++)
                {
                    var desc = plan.Descriptors[d];
                    if (!PageSpan(desc.BoundsMin.X, desc.BoundsMax.X, cells, pages, out int px0, out int px1) ||
                        !PageSpan(desc.BoundsMin.Y, desc.BoundsMax.Y, cells, pages, out int py0, out int py1))
                        continue;   // reaches no sample of the terrain
                    for (int py = py0; py <= py1; py++)
                        for (int px = px0; px <= px1; px++)
                            (buckets[py * pages + px] ??= new List<uint>()).Add((uint)d);
                }

                step.PageRuns = plan.PageRuns.Count;
                for (int p = 0; p < pages * pages; p++)
                {
                    var bucket = buckets[p];
                    int count = bucket?.Count ?? 0;
                    plan.PageRuns.Add((plan.PageStamps.Count, count));
                    if (count > 0) plan.PageStamps.AddRange(bucket);
                }
                plan.Steps[s] = step;
            }

            plan.AllStamps = plan.PageStamps.Count;
            for (int d = 0; d < plan.Descriptors.Count; d++)
                plan.PageStamps.Add((uint)d);
        }

        /// <summary>The pages (first and last, inclusive) along one axis whose samples, border included, a terrain UV span
        /// reaches, a sample more each side; false when it reaches no sample.</summary>
        private static bool PageSpan(float uvMin, float uvMax, int cells, int pages, out int first, out int last)
        {
            first = last = 0;
            const float margin = 1 + Graphics.HeightPool.Border;
            float s0 = uvMin * cells - margin, s1 = uvMax * cells + margin;
            // Empty, NaN, or beyond the first and last pages' borders (s0, s1 already hold the margin)
            if (!(s1 >= s0) || s1 < 0f || s0 > cells) return false;
            first = (int)Math.Clamp(MathF.Floor(s0 / Terrain.PageCells), 0f, pages - 1);
            last = (int)Math.Clamp(MathF.Floor(s1 / Terrain.PageCells), 0f, pages - 1);
            return true;
        }

        private static bool TryBuildHeightDescriptor(HeightStamp stamp, Vector3 terrainOrigin, Vector2 terrainSize,
            float maxHeight, List<StampSplinePointGPU> splinePoints, out HeightStampDescriptorGPU desc)
        {
            desc = new HeightStampDescriptorGPU();
            float uvScale = 1f / Math.Max(terrainSize.X, terrainSize.Y);

            // Outside: a closed spline's stamp covers what lies beyond its outline (HeightStamp.Outside)
            bool outside = stamp.Outside && stamp.IsSplineMode && stamp.GetSpline() is { Closed: true };
            desc.InvertShape = (stamp.InvertShape ? 1u : 0u) | (outside ? 2u : 0u);
            desc.Falloff = stamp.Falloff * uvScale;
            desc.BlendMode = (uint)stamp.BlendMode;

            // Add raises the ground by an amount, so where the entity sits vertically must not count
            bool relative = stamp.BlendMode == HeightBlendMode.Add;

            if (stamp.EnableNoise)
            {
                desc.NoiseFreq = stamp.NoiseFrequency;
                desc.NoiseAmp = stamp.NoiseAmplitude * uvScale;
                desc.NoiseSeed = (uint)stamp.NoiseSeed;
            }

            if (stamp.IsSplineMode)
            {
                if (!BuildSplinePoints(stamp, terrainOrigin, terrainSize, maxHeight, stamp.HeightOffset, splinePoints,
                        out desc.SplinePointOffset, out desc.SplinePointCount, out desc.Radius, useWorldY: !relative))
                    return false;

                desc.TargetHeight = 0;
                desc.Center = Vector2.Zero;
            }
            else
            {
                var center = stamp.Transform?.WorldPosition ?? Vector3.Zero;
                desc.Center = new Vector2(
                    (center.X - terrainOrigin.X) / terrainSize.X,
                    (center.Z - terrainOrigin.Z) / terrainSize.Y);
                desc.Radius = stamp.Radius * uvScale;
                // A world height, measured up from the terrain's base (its entity's Y) as the heightmap is
                desc.TargetHeight = ((relative ? 0f : center.Y - terrainOrigin.Y) + stamp.HeightOffset) / maxHeight;
                desc.SplinePointOffset = 0xFFFFFFFF;
                desc.SplinePointCount = 0;

                // Under a closed spline: confined to its outline (the samples carry no width: the outline is not grown)
                // (a spline that cannot be sampled leaves the count at 0: no mask)
                if (stamp.MaskSpline is { } mask)
                    BuildSplinePoints(mask, 0f, terrainOrigin, terrainSize, maxHeight, 0f, splinePoints,
                        out desc.MaskPointOffset, out desc.MaskPointCount, out _);
            }

            if (stamp.Heightmap != null)
            {
                desc.HeightmapIdx = stamp.Heightmap.BindlessIndex;
                desc.HeightmapStrength = stamp.Strength / maxHeight;

                if (stamp.IsSplineMode)
                {
                    // The texture turns with the spline (the entity's world yaw) and spans its footprint
                    var world = stamp.Transform?.Matrix ?? Matrix4x4.Identity;
                    float yaw = MathF.Atan2(-world.M13, world.M11);
                    desc.RotationSin = -MathF.Sin(yaw);
                    desc.RotationCos = MathF.Cos(yaw);
                    if (outside)
                    {
                        // What it covers reaches to the terrain's border: the texture spans the terrain, turned
                        float e = 0.5f * (MathF.Abs(desc.RotationCos) + MathF.Abs(desc.RotationSin));
                        desc.HeightmapCenter = new Vector2(0.5f);
                        desc.HeightmapExtent = new Vector2(e);
                    }
                    else
                        FitHeightmapToSpline(ref desc, splinePoints);
                }
                else
                {
                    // A radial stamp's texture spans Radius + Falloff each way. It has always turned against its
                    // entity (sin taken as the inverse's); kept so rotated stamps in existing scenes keep their shape.
                    var rot = stamp.Transform?.Rotation ?? Quaternion.Identity;
                    float yaw = MathF.Atan2(2f * (rot.W * rot.Y + rot.X * rot.Z),
                                            1f - 2f * (rot.Y * rot.Y + rot.Z * rot.Z));
                    desc.RotationSin = MathF.Sin(yaw);
                    desc.RotationCos = MathF.Cos(yaw);
                    desc.HeightmapCenter = desc.Center;
                    desc.HeightmapExtent = new Vector2(desc.Radius + desc.Falloff);
                }
            }

            StampBounds(ref desc, splinePoints, outside);

            // Ridge and furrow: across the ridges, and how far the stamp reaches along them (for the bend at the ends)
            if (stamp.RidgeSpacing > 0.5f && stamp.RidgeHeight > 0f)
            {
                var along = AlongLongestEdge(stamp, stamp.RidgeAngle);
                desc.RidgeAcross = new Vector2(-along.Y, along.X);
                desc.RidgeSpacing = stamp.RidgeSpacing * uvScale;
                desc.RidgeHeight = stamp.RidgeHeight / maxHeight;
                desc.RidgeBend = stamp.RidgeBend * uvScale;
                float from = float.MaxValue, to = float.MinValue;
                if (desc.SplinePointOffset != 0xFFFFFFFF)
                {
                    int start = (int)desc.SplinePointOffset;
                    int end = start + (int)(desc.SplinePointCount & 0x7FFFFFFF);
                    for (int i = start; i < end; i++)
                    {
                        float d = Vector2.Dot(splinePoints[i].UV, along);
                        from = MathF.Min(from, d);
                        to = MathF.Max(to, d);
                    }
                }
                else
                {
                    float d = Vector2.Dot(desc.Center, along);
                    from = d - desc.Radius - desc.Falloff;
                    to = d + desc.Radius + desc.Falloff;
                }
                desc.RidgeAlong = new Vector2(from, to);
            }
            return true;
        }

        /// <summary>
        /// The terrain UV box beyond which a stamp's weight is 0, so the bake skips it there without walking its spline:
        /// every texel used to evaluate every stamp, and a 4 km terrain with its roads and rivers took ~1.5 s per bake
        /// (Eldermoor, 2026-10-19), close to the GPU's 2 s timeout. The reach is EvaluateStampWeight's early-out: a
        /// circle's Radius + Falloff, a closed outline grown by its width (the falloff lies inside), an open spline's
        /// corridor plus its falloff, each plus the edge noise. An Outside stamp reaches everywhere.
        /// </summary>
        private static void StampBounds(ref HeightStampDescriptorGPU desc, List<StampSplinePointGPU> splinePoints, bool outside)
        {
            if (outside)
            {
                desc.BoundsMin = new Vector2(-1e9f);
                desc.BoundsMax = new Vector2(1e9f);
                return;
            }

            float noise = desc.NoiseFreq > 0 ? desc.NoiseAmp : 0f;
            if (desc.SplinePointOffset == 0xFFFFFFFF)
            {
                // Radial (a mask under a parent's spline only takes from the circle)
                var reach = new Vector2(desc.Radius + desc.Falloff + noise);
                desc.BoundsMin = desc.Center - reach;
                desc.BoundsMax = desc.Center + reach;
                return;
            }

            int start = (int)desc.SplinePointOffset;
            int end = start + (int)(desc.SplinePointCount & 0x7FFFFFFF);
            bool closed = (desc.SplinePointCount & 0x80000000) != 0;
            var min = new Vector2(float.MaxValue);
            var max = new Vector2(float.MinValue);
            for (int i = start; i < end; i++)
            {
                var p = splinePoints[i];
                float falloff = closed ? 0f : desc.Radius > 1e-7f ? desc.Falloff * p.HalfWidth / desc.Radius : desc.Falloff;
                var reach = new Vector2(Math.Max(p.HalfWidth, 0f) + falloff + noise);
                min = Vector2.Min(min, p.UV - reach);
                max = Vector2.Max(max, p.UV + reach);
            }
            desc.BoundsMin = min;
            desc.BoundsMax = max;
        }

        /// <summary>
        /// A spline stamp's heightmap covers what the stamp reaches, as a radial stamp's covers its Radius + Falloff:
        /// the rectangle, in the texture's turned frame, around every sample's reach. On a closed spline that is the
        /// outline grown by Radius (the falloff lies inside it), on an open one the corridor plus its falloff.
        /// </summary>
        private static void FitHeightmapToSpline(ref HeightStampDescriptorGPU desc, List<StampSplinePointGPU> splinePoints)
        {
            int start = (int)desc.SplinePointOffset;
            int end = start + (int)(desc.SplinePointCount & 0x7FFFFFFF);
            bool closed = (desc.SplinePointCount & 0x80000000) != 0;
            float s = desc.RotationSin, c = desc.RotationCos;

            var min = new Vector2(float.MaxValue);
            var max = new Vector2(float.MinValue);
            for (int i = start; i < end; i++)
            {
                var p = splinePoints[i];
                // The falloff tapers with the sample's width, as in EvaluateStampWeight
                float falloff = closed ? 0f : desc.Radius > 1e-7f ? desc.Falloff * p.HalfWidth / desc.Radius : desc.Falloff;
                var reach = new Vector2(Math.Max(p.HalfWidth, 0f) + falloff + desc.NoiseAmp);
                var local = new Vector2(p.UV.X * c + p.UV.Y * s, -p.UV.X * s + p.UV.Y * c);
                min = Vector2.Min(min, local - reach);
                max = Vector2.Max(max, local + reach);
            }

            // Back from the turned frame (the transpose of the turn above) to terrain UV
            var mid = (min + max) * 0.5f;
            desc.HeightmapCenter = new Vector2(mid.X * c - mid.Y * s, mid.X * s + mid.Y * c);
            desc.HeightmapExtent = Vector2.Max((max - min) * 0.5f, new Vector2(1e-6f));
        }

        /// <summary>
        /// Run a height plan into the terrain's height pool over a box of pages (px0, py0 inclusive, px1, py1 exclusive:
        /// <see cref="HeightPlan.PagesOf"/>), or all of them for null. Each page is baked whole, border included: cleared,
        /// then every step, an influence run with the page's own stamps. A new pool or flat map (the first bake, another
        /// resolution) is baked whole. Render callback.
        /// </summary>
        public void BakeHeight(Terrain terrain, HeightPlan plan, ID3D12GraphicsCommandList cmd,
            (int px0, int py0, int px1, int py1)? pages = null)
        {
            if (plan == null || !plan.HasWork) return;

            EnsureInitialized();
            var device = Engine.Device;
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            cmd.SetComputeRootSignature(device.GlobalRootSignature);

            bool fresh = _pool.Ensure(plan.Resolution, PoolCapacity, cmd);
            // Another grid or height range than the last bake's moves every sample, whatever region the caller saw
            var grid = (plan.Resolution, plan.Origin, plan.Spacing, plan.MaxHeight);
            bool gridChanged = _bakedGrid is { } last && last != grid;
            _bakedGrid = grid;
            var box = fresh || gridChanged || pages == null ? (0, 0, plan.PagesAcross, plan.PagesAcross) : pages.Value;

            UploadHeightPlan(plan);
            _lastPlan = plan;

            // The resident pages bake in place; the others are owed to the readback, which bakes them into scratch slots
            // (a streaming pool, TERRAIN_PAGES_PLAN.md 5a)
            var resident = new List<(int px, int py, int slot)>((box.Item3 - box.Item1) * (box.Item4 - box.Item2));
            for (int py = box.Item2; py < box.Item4; py++)
                for (int px = box.Item1; px < box.Item3; px++)
                {
                    int slot = _pool.SlotOf(0, px, py);
                    if (slot >= 0)
                    {
                        resident.Add((px, py, slot));
                        _pool.MarkWritten(slot);
                    }
                    else
                        _owed.Add(py * plan.PagesAcross + px);
                }

            // The pool rests in Common between bakes, and a texture is never promoted to UnorderedAccess implicitly (only
            // to the read and copy states); back to Common, so its readers promote it to SRV implicitly
            cmd.ResourceBarrierTransition(_pool.Texture, ResourceStates.Common, ResourceStates.UnorderedAccess);
            RecordHeight(plan, cmd, resident, wholeRuns: false);
            cmd.ResourceBarrierTransition(_pool.Texture, ResourceStates.UnorderedAccess, ResourceStates.Common);
            if (_pool.AllResident) _pool.Filled = true;
            HeightVersion++;
        }

        /// <summary>
        /// Page slots the pool is made with (TerrainHeightResidency.CapacityFor): 0 = every level 0 page resident at its
        /// own index. Set by the renderer before it bakes; a change makes a new pool.
        /// </summary>
        public int PoolCapacity { get; set; }

        /// <summary>
        /// Make the pool for this many samples a side at <see cref="PoolCapacity"/> now, on a list of its own, when the one
        /// there is not that (the detail radius changed). True when it was made: nothing is in it, and the owed pages go
        /// with the old one (the renderer fills the new one from the CPU store). Blocks. Main thread.
        /// </summary>
        public bool EnsurePoolNow(int resolution)
        {
            if (_pool.Matches(resolution, PoolCapacity)) return false;
            var device = Engine.Device;
            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            var cmd = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, allocator, null);
            try
            {
                _pool.Ensure(resolution, PoolCapacity, cmd);
                cmd.Close();
                device.SubmitAndWait(cmd);
            }
            finally
            {
                cmd.Dispose();
                allocator.Dispose();
            }
            _owed.Clear();
            return true;
        }

        // The plan of the last bake: the readback bakes owed pages with it
        private HeightPlan? _lastPlan;

        // Level 0 pages (index py · across + px) a bake did not write, not being resident: their CPU heights are behind
        // until a readback bakes them into scratch slots and reads them back
        private readonly HashSet<int> _owed = new();

        /// <summary>Level 0 pages a bake owes the CPU store (not resident when it ran).</summary>
        public int OwedPages => _owed.Count;

        /// <summary>
        /// The plan's steps, descriptors, spline points and page lists, into upload buffers of their own once per plan.
        /// Every dispatch of a bake reads its own range of them (an upload buffer rewritten between dispatches of one
        /// command list would show every dispatch the last write), and a bake recorded before the last one's frames have
        /// run gets new buffers (EnsureUpload).
        /// </summary>
        private void UploadHeightPlan(HeightPlan plan)
        {
            if (_uploadedHeightPlan == plan) return;
            _uploadedHeightPlan = plan;
            var steps = new List<HeightStepGPU>(plan.Steps.Count);
            foreach (var step in plan.Steps) steps.Add(step.ToGpu());
            EnsureUpload<HeightStepGPU>(ref _heightStepBuffer, ref _heightStepCapacity, Math.Max(1, steps.Count), 8);
            Upload(_heightStepBuffer, steps);
            if (plan.Descriptors.Count > 0)
            {
                EnsureUpload<HeightStampDescriptorGPU>(ref _heightStampBuffer, ref _heightStampBufferCapacity, plan.Descriptors.Count, 16);
                Upload(_heightStampBuffer, plan.Descriptors);
            }
            if (plan.SplinePoints.Count > 0)
            {
                EnsureUpload<StampSplinePointGPU>(ref _heightStampSplineBuffer, ref _heightStampSplineCapacity, plan.SplinePoints.Count, 128);
                Upload(_heightStampSplineBuffer, plan.SplinePoints);
            }
            if (plan.PageStamps.Count > 0)
            {
                EnsureUpload<uint>(ref _heightPageListBuffer, ref _heightPageListCapacity, plan.PageStamps.Count, 64);
                Upload(_heightPageListBuffer, plan.PageStamps);
            }
        }

        /// <summary>The samples of a box of pages, borders left out: x0, y0 inclusive, x1, y1 exclusive.</summary>
        private static (int x0, int y0, int x1, int y1) BoxSamples(HeightPlan plan, (int px0, int py0, int px1, int py1) pages)
        {
            if (pages.px1 <= pages.px0 || pages.py1 <= pages.py0) return default;
            var first = plan.PageSamples(pages.px0, pages.py0);
            var last = plan.PageSamples(pages.px1 - 1, pages.py1 - 1);
            return (first.x0, first.y0, last.x1, last.y1);
        }

        /// <summary>Every page of a box with the slot it is resident at (every page of an AllResident pool).</summary>
        private List<(int px, int py, int slot)> ResidentPages((int px0, int py0, int px1, int py1) pages)
        {
            var list = new List<(int px, int py, int slot)>();
            for (int py = pages.py0; py < pages.py1; py++)
                for (int px = pages.px0; px < pages.px1; px++)
                    if (_pool.SlotOf(0, px, py) is var slot && slot >= 0) list.Add((px, py, slot));
            return list;
        }

        /// <summary>
        /// Record a plan's bake of these pages, each into its slot, into the pool, which is in UnorderedAccess: each page
        /// cleared, then every step over each page, border included. An influence run takes the page's list, or, with
        /// <paramref name="wholeRuns"/>, every descriptor of the run (CheckPageBake). Pages write texels of their own, so
        /// there is a barrier between steps only. The slot need not be the page's resident one: the readback bakes owed
        /// pages into scratch slots.
        /// </summary>
        private void RecordHeight(HeightPlan plan, ID3D12GraphicsCommandList cmd,
            IReadOnlyList<(int px, int py, int slot)> pages, bool wholeRuns)
        {
            if (pages.Count == 0) return;

            // Where the samples lie, the same for every kernel
            _cs.SetPushConstant("Output", _pool.Uav);
            _cs.SetPushConstant("GridSamples", (uint)plan.Resolution);
            _cs.SetParam("OriginX", plan.Origin.X);
            _cs.SetParam("OriginZ", plan.Origin.Y);
            _cs.SetParam("Spacing", plan.Spacing);
            _cs.SetBuffer("StepBuf", _heightStepBuffer);

            _cs.SetParam(_kernelClear, "ClearValue", 0f);
            foreach (var (px, py, slot) in pages)
                DispatchPage(_kernelClear, cmd, plan, px, py, slot);
            cmd.ResourceBarrierUnorderedAccessView(_pool.Texture);

            for (int s = 0; s < plan.Steps.Count; s++)
            {
                var step = plan.Steps[s];
                if (step.Kind == HeightStepKind.Influence)
                    DispatchInfluence(cmd, plan, step, pages, wholeRuns);
                else
                {
                    var k = step.Kind == HeightStepKind.Noise ? _kernelNoiseLayer : _kernelImport;
                    if (step.Kind == HeightStepKind.Noise)
                    {
                        EnsureNoiseLUT();
                        _cs.SetPushConstant(k, "NoiseLUT", _noiseLUTSRV);
                        _cs.SetPushConstant(k, "SplineBuf",
                            plan.SplinePoints.Count > 0 && _heightStampSplineBuffer != null ? _heightStampSplineBuffer.SrvIndex : 0u);
                    }
                    _cs.SetPushConstant(k, "StepIndex", (uint)s);
                    float lastSample = Math.Max(plan.Resolution - 1, 1);
                    foreach (var (px, py, slot) in pages)
                    {
                        // Noise along a spline: only the pages its corridor reaches (a river's walk over every
                        // sample of the map would cost as much as every stamp of it together)
                        if (step.MaskSplineCount != 0)
                        {
                            var box = plan.PageBox(px, py);
                            if (box.x1 / lastSample < step.MaskBoundsMin.X || box.x0 / lastSample > step.MaskBoundsMax.X ||
                                box.y1 / lastSample < step.MaskBoundsMin.Y || box.y0 / lastSample > step.MaskBoundsMax.Y)
                                continue;
                        }
                        DispatchPage(k, cmd, plan, px, py, slot);
                    }
                }
                cmd.ResourceBarrierUnorderedAccessView(_pool.Texture);
            }
        }

        /// <summary>One dispatch of a kernel over page (px, py) with its border, written at a slot.</summary>
        private void DispatchPage(int k, ID3D12GraphicsCommandList cmd, HeightPlan plan, int px, int py, int slot)
        {
            var box = plan.PageBox(px, py);
            var (slice, cx, cy) = HeightPool.SlotCorner(slot);
            // Sample s goes to texel s + Out: the box's first sample, a border before the page's, at the slot's corner
            _cs.SetPushConstant(k, "OutX", unchecked((uint)(cx - box.x0)));
            _cs.SetPushConstant(k, "OutY", unchecked((uint)(cy - box.y0)));
            _cs.SetPushConstant(k, "OutSlice", (uint)slice);
            DispatchBox(k, cmd, box);
        }

        /// <summary>One dispatch of a kernel over a box of samples (x0, y0 inclusive, x1, y1 exclusive).</summary>
        private static void DispatchBox(int k, ID3D12GraphicsCommandList cmd, (int x0, int y0, int x1, int y1) box)
        {
            int w = box.x1 - box.x0, h = box.y1 - box.y0;
            if (w <= 0 || h <= 0) return;
            _cs.SetPushConstant(k, "RegionX", unchecked((uint)box.x0));
            _cs.SetPushConstant(k, "RegionY", unchecked((uint)box.y0));
            _cs.SetPushConstant(k, "RegionW", (uint)w);
            _cs.SetPushConstant(k, "RegionH", (uint)h);
            _cs.Dispatch(k, cmd, (uint)((w + 7) / 8), (uint)((h + 7) / 8));
        }

        private void DispatchInfluence(ID3D12GraphicsCommandList cmd, HeightPlan plan, in HeightStep step,
            IReadOnlyList<(int px, int py, int slot)> pages, bool wholeRuns)
        {
            var k = _kernelInfluenceLayer;
            _cs.SetBuffer(k, "StampBuf", _heightStampBuffer);
            _cs.SetBuffer(k, "StampList", _heightPageListBuffer);
            _cs.SetPushConstant(k, "SplineBuf",
                plan.SplinePoints.Count > 0 && _heightStampSplineBuffer != null ? _heightStampSplineBuffer.SrvIndex : 0u);

            foreach (var (px, py, slot) in pages)
            {
                var (start, count) = wholeRuns
                    ? (plan.AllStamps + step.Start, step.Count)
                    : plan.PageRuns[step.PageRuns + py * plan.PagesAcross + px];
                if (count == 0) continue;
                _cs.SetPushConstant(k, "ListStart", (uint)start);
                _cs.SetPushConstant(k, "ListCount", (uint)count);
                DispatchPage(k, cmd, plan, px, py, slot);
            }
        }

        /// <summary>
        /// Bake a plan three ways and compare them with each other and with the pool in place (TERRAIN_PAGES_PLAN.md stages
        /// 3 and 4a), every texel of every page, borders included: (A) every page with every descriptor of each run, as the
        /// bake was before page lists; (B) every page at once with its own lists, as <see cref="BakeHeight"/> does; (C) page
        /// after page, each alone, onto a pool first filled with a value no bake leaves, in a shuffled order. The pool in
        /// place is what the bakes so far left: the first whole one and the pages of the edits since. Also checks A's
        /// borders and shared edges: each texel that holds a sample of the terrain equals that sample in the page it
        /// belongs to. Leaves C in the pool and the flat map. Blocks until the GPU is done. Main thread.
        /// </summary>
        public unsafe (long inPlace, long bDiffers, long cDiffers, long borderDiffers, int maxStep, double aMs, double bMs)
            CheckPageBake(Terrain terrain, HeightPlan plan)
        {
            if (plan == null || !plan.HasWork) return default;
            if (PoolCapacity != 0 || (_pool.Texture != null && !_pool.AllResident))
            {
                Debug.LogWarning("TerrainBaker", "Check Page Bake compares every page: set TerrainDetailRadius to 0 (every page resident) first");
                return default;
            }
            EnsureInitialized();
            EnsureNoiseLUT();
            var device = Engine.Device;
            int res = plan.Resolution;
            int pagesAcross = plan.PagesAcross;
            var all = new List<(int px, int py, int slot)>(pagesAcross * pagesAcross);
            for (int py = 0; py < pagesAcross; py++)
                for (int px = 0; px < pagesAcross; px++)
                    all.Add((px, py, py * pagesAcross + px));   // every page at its own index (AllResident)

            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            ID3D12Resource[] readbacks = null;
            var times = new double[3];
            try
            {
                int rowPitch = (HeightPool.SliceTexels * 2 + 255) & ~255;
                ulong sliceBytes = (ulong)rowPitch * HeightPool.SliceTexels;   // 512-aligned: rowPitch is a multiple of 256, 522 even

                // Pass 0 twice: the first waits for the frames still queued, the second times a submit and a copy alone
                for (int run = -1; run < 4; run++)
                {
                    int pass = Math.Max(run, 0);
                    var cmd = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, allocator, null);
                    cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
                    cmd.SetComputeRootSignature(device.GlobalRootSignature);
                    _pool.Ensure(res, 0, cmd);
                    if (readbacks == null)
                    {
                        readbacks = new ID3D12Resource[4];
                        for (int i = 0; i < readbacks.Length; i++)
                            readbacks[i] = device.NativeDevice.CreateCommittedResource(new HeapProperties(HeapType.Readback),
                                HeapFlags.None, ResourceDescription.Buffer(sliceBytes * (ulong)_pool.Slices), ResourceStates.CopyDest, null);
                    }
                    UploadHeightPlan(plan);
                    cmd.ResourceBarrierTransition(_pool.Texture, ResourceStates.Common, ResourceStates.UnorderedAccess);

                    if (pass == 0) { }   // the pool in place
                    else if (pass < 3)
                        RecordHeight(plan, cmd, all, wholeRuns: pass == 1);
                    else
                    {
                        // The top of the range, which no sample of a real terrain holds along a whole row, then the
                        // pages one by one in a shuffled order: a texel no page bakes keeps it
                        _cs.SetPushConstant("Output", _pool.Uav);
                        _cs.SetParam(_kernelClear, "ClearValue", 1f);
                        foreach (var (px, py, slot) in all)
                            DispatchPage(_kernelClear, cmd, plan, px, py, slot);
                        cmd.ResourceBarrierUnorderedAccessView(_pool.Texture);
                        var order = Enumerable.Range(0, pagesAcross * pagesAcross).ToArray();
                        new Random(7).Shuffle(order);
                        foreach (int p in order)
                            RecordHeight(plan, cmd, new[] { all[p] }, wholeRuns: false);
                    }

                    cmd.ResourceBarrierTransition(_pool.Texture, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    for (int slice = 0; slice < _pool.Slices; slice++)
                        cmd.CopyTextureRegion(
                            new TextureCopyLocation(readbacks[pass], new PlacedSubresourceFootPrint
                            {
                                Offset = sliceBytes * (ulong)slice,
                                Footprint = new SubresourceFootPrint(Format.R16_UNorm, (uint)HeightPool.SliceTexels,
                                    (uint)HeightPool.SliceTexels, 1, (uint)rowPitch)
                            }), 0, 0, 0, new TextureCopyLocation(_pool.Texture, (uint)slice), null);
                    cmd.ResourceBarrierTransition(_pool.Texture, ResourceStates.CopySource, ResourceStates.Common);
                    cmd.Close();

                    var watch = System.Diagnostics.Stopwatch.StartNew();
                    device.SubmitAndWait(cmd);
                    if (run >= 0 && pass < 3) times[pass] = watch.Elapsed.TotalMilliseconds;
                    cmd.Dispose();
                    allocator.Reset();
                }

                long inPlace = 0, bDiffers = 0, cDiffers = 0, borderDiffers = 0;
                int maxStep = 0;
                unsafe
                {
                    void* d, a, b, c;
                    readbacks[0].Map(0, null, &d);
                    readbacks[1].Map(0, null, &a);
                    readbacks[2].Map(0, null, &b);
                    readbacks[3].Map(0, null, &c);
                    ushort* At(void* data, int px, int py, int tx, int ty)
                    {
                        var (slice, cx, cy) = _pool.PageCorner(px, py);
                        return (ushort*)((byte*)data + (long)slice * (long)sliceBytes + (long)(cy + ty) * rowPitch) + cx + tx;
                    }
                    for (int py = 0; py < pagesAcross; py++)
                        for (int px = 0; px < pagesAcross; px++)
                            for (int ty = 0; ty < HeightPool.PageTexels; ty++)
                                for (int tx = 0; tx < HeightPool.PageTexels; tx++)
                                {
                                    ushort va = *At(a, px, py, tx, ty);
                                    ushort vd = *At(d, px, py, tx, ty), vb = *At(b, px, py, tx, ty), vc = *At(c, px, py, tx, ty);
                                    if (va != vb) { bDiffers++; maxStep = Math.Max(maxStep, Math.Abs(va - vb)); }
                                    if (va != vc) { cDiffers++; maxStep = Math.Max(maxStep, Math.Abs(va - vc)); }

                                    // The rest only where the texel holds a sample of the terrain: a pool filled from the
                                    // saved pages (UploadPages) holds the edge sample again in the border beyond the
                                    // terrain's edge, where the bake writes what its stamps give there; nothing reads it
                                    int sx = px * Terrain.PageCells - HeightPool.Border + tx;
                                    int sy = py * Terrain.PageCells - HeightPool.Border + ty;
                                    if (sx < 0 || sy < 0 || sx >= res || sy >= res) continue;
                                    if (va != vd) { inPlace++; maxStep = Math.Max(maxStep, Math.Abs(va - vd)); }

                                    // A texel holding a sample of the terrain equals that sample in the page it belongs to
                                    int ox = Math.Min(sx / Terrain.PageCells, pagesAcross - 1);
                                    int oy = Math.Min(sy / Terrain.PageCells, pagesAcross - 1);
                                    if (ox == px && oy == py) continue;
                                    ushort owner = *At(a, ox, oy, sx - ox * Terrain.PageCells + HeightPool.Border,
                                                                  sy - oy * Terrain.PageCells + HeightPool.Border);
                                    if (owner != va) borderDiffers++;
                                }
                    for (int i = 0; i < readbacks.Length; i++) readbacks[i].Unmap(0);
                }

                HeightVersion++;
                // The bakes' time on the queue: less what a submit and a copy take by themselves
                return (inPlace, bDiffers, cDiffers, borderDiffers, maxStep, times[1] - times[0], times[2] - times[0]);
            }
            finally
            {
                allocator.Dispose();
                if (readbacks != null)
                    foreach (var r in readbacks) r?.Dispose();
            }
        }

        /// <summary>
        /// The height at a million places read through terrain_pages.hlsli against the same bilinear in float on the CPU
        /// from the CPU store (TerrainPages.Height01: TerrainRenderer.GetHeight's sum), on the terrain and 3 m round it,
        /// every second place on a sample (CS_CompareReads). Returns the largest difference in metres on samples and
        /// between them. The store must hold the pool's heights (read back after the last bake). Blocks. Main thread.
        /// </summary>
        public unsafe (float atSamples, float between) CompareReads(Terrain terrain, Vector3 origin)
        {
            var pages = terrain.Pages;
            if (_pool.Texture == null || pages is not { HasHeights: true }) return default;
            EnsureInitialized();
            const int count = 1 << 20;
            var device = Engine.Device;
            var paramsBuffer = GraphicsBuffer.CreateUpload<HeightPool.Params>(1, mapped: true);
            *paramsBuffer.WritePtr<HeightPool.Params>() = _pool.MakeParams(origin, terrain.TerrainSize, terrain.MaxHeight, 0);
            var output = GraphicsBuffer.CreateStructured<Vector4>(count, uav: true);
            var readback = GraphicsBuffer.CreateReadback<Vector4>(count);
            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            var cmd = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, allocator, null);
            try
            {
                cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
                cmd.SetComputeRootSignature(device.GlobalRootSignature);
                output.Transition(cmd, ResourceStates.UnorderedAccess);
                var k = _kernelCompareReads;
                _cs.SetPushConstant(k, "TerrainParams", paramsBuffer.SrvIndex);
                _cs.SetBuffer(k, "CompareUAV", output);
                _cs.SetPushConstant(k, "CompareCount", (uint)count);
                _cs.Dispatch(k, cmd, (uint)(count / 64));
                output.Transition(cmd, ResourceStates.CopySource);
                cmd.CopyBufferRegion(readback.Native, 0, output.Native, 0, (ulong)(count * sizeof(Vector4)));
                cmd.Close();
                device.SubmitAndWait(cmd);

                float atSamples = 0, between = 0;
                var r = readback.ReadPtr<Vector4>();
                int n = pages.Resolution;
                float spacing = terrain.TerrainSize.X / Math.Max(1, n - 1);
                for (int i = 0; i < count; i++)
                {
                    // The same place in float on the CPU, from the samples read back
                    float tx = (r[i].Z - origin.X) / spacing;
                    float tz = (r[i].W - origin.Z) / spacing;
                    float d = MathF.Abs(r[i].X - pages.Height01(tx, tz)) * terrain.MaxHeight;
                    if ((i & 1) != 0) atSamples = MathF.Max(atSamples, d);
                    else between = MathF.Max(between, d);
                }
                return (atSamples, between);
            }
            finally
            {
                cmd.Dispose();
                allocator.Dispose();
                device.DeferDispose(paramsBuffer);
                device.DeferDispose(output);
                device.DeferDispose(readback);
            }
        }


        // ── Noise LUT generation ──────────────────────────────────────────

        private const int NoiseLUTSize = 256;
        // Noise tiling period in cells — small for many texels per cell (smooth interpolation).
        // Per-octave UV rotation in the shader breaks visible tiling.
        private const int NoiseLUTPeriod = 16;

        private static void EnsureNoiseLUT()
        {
            if (_noiseLUTTex != null) return;

            var device = Engine.Device;

            // Generate CPU-side tileable Perlin noise as R16G16_Float (2 independent channels)
            var pixels = new Half[NoiseLUTSize * NoiseLUTSize * 2]; // RG16F = 2 halfs per texel
            GenerateTileableNoise(pixels, NoiseLUTSize, NoiseLUTPeriod, seed1: 0, seed2: 137);

            // Create GPU texture (R16G16_Float — 4 bytes per texel)
            _noiseLUTTex = device.CreateTexture2D(
                Format.R16G16_Float, NoiseLUTSize, NoiseLUTSize, 1, 1,
                ResourceFlags.None, ResourceStates.Common);

            // Create SRV
            _noiseLUTSRV = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(_noiseLUTTex,
                new ShaderResourceViewDescription
                {
                    Format = Format.R16G16_Float,
                    ViewDimension = ShaderResourceViewDimension.Texture2D,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
                }, device.GetCpuHandle(_noiseLUTSRV));

            // Upload pixel data (R16G16_Float = 4 bytes per texel)
            int bytesPerPixel = 4;
            int rowPitch = (NoiseLUTSize * bytesPerPixel + 255) & ~255; // 256-byte aligned
            int totalBytes = rowPitch * NoiseLUTSize;

            var uploadResource = device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Upload),
                HeapFlags.None,
                ResourceDescription.Buffer((ulong)totalBytes),
                ResourceStates.GenericRead,
                null);

            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            var cmdList = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(
                0, CommandListType.Direct, allocator, null);

            try
            {
                unsafe
                {
                    void* pData;
                    uploadResource.Map(0, null, &pData);
                    int srcRowBytes = NoiseLUTSize * bytesPerPixel;
                    var dstPtr = (byte*)pData;
                    fixed (Half* srcBase = pixels)
                    {
                        var srcBytePtr = (byte*)srcBase;
                        for (int y = 0; y < NoiseLUTSize; y++)
                            Buffer.MemoryCopy(srcBytePtr + y * srcRowBytes,
                                dstPtr + y * rowPitch, srcRowBytes, srcRowBytes);
                    }
                    uploadResource.Unmap(0);
                }

                cmdList.ResourceBarrierTransition(_noiseLUTTex,
                    ResourceStates.Common, ResourceStates.CopyDest);

                var src = new TextureCopyLocation(uploadResource, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R16G16_Float,
                        (uint)NoiseLUTSize, (uint)NoiseLUTSize, 1, (uint)rowPitch)
                });
                var dst = new TextureCopyLocation(_noiseLUTTex, 0);
                cmdList.CopyTextureRegion(dst, 0, 0, 0, src);

                cmdList.ResourceBarrierTransition(_noiseLUTTex,
                    ResourceStates.CopyDest, ResourceStates.Common);

                cmdList.Close();
                device.SubmitAndWait(cmdList);

                Debug.Log($"[TerrainBaker] Noise LUT uploaded: {NoiseLUTSize}x{NoiseLUTSize} R16G16_Float (period={NoiseLUTPeriod})");
            }
            finally
            {
                cmdList.Dispose();
                allocator.Dispose();
                uploadResource.Dispose();
            }
        }

        /// <summary>
        /// Generates tileable 2D Perlin noise into a Half[] buffer (R16G16 layout).
        /// Channel R = noise with seed1, channel G = noise with seed2.
        /// </summary>
        private static void GenerateTileableNoise(Half[] pixels, int size, int period, int seed1, int seed2)
        {
            var perm1 = BuildPermutation(seed1);
            var perm2 = BuildPermutation(seed2);

            // 12 gradient directions (classic Perlin)
            ReadOnlySpan<(float, float)> grads = stackalloc (float, float)[]
            {
                ( 1, 0), (-1, 0), ( 0, 1), ( 0,-1),
                ( 1, 1), (-1, 1), ( 1,-1), (-1,-1),
                ( 0.7071f, 0.7071f), (-0.7071f, 0.7071f),
                ( 0.7071f,-0.7071f), (-0.7071f,-0.7071f),
            };

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float fx = (float)x / size;
                    float fy = (float)y / size;

                    float n1 = TileablePerlin(fx, fy, period, perm1, grads);
                    float n2 = TileablePerlin(fx, fy, period, perm2, grads);

                    int idx = (y * size + x) * 2; // 2 half-floats per texel
                    pixels[idx + 0] = (Half)n1;
                    pixels[idx + 1] = (Half)n2;
                }
            }
        }

        private static int[] BuildPermutation(int seed)
        {
            var rng = new Random(seed);
            var p = new int[512];
            var base256 = new int[256];
            for (int i = 0; i < 256; i++) base256[i] = i;
            // Fisher-Yates shuffle
            for (int i = 255; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (base256[i], base256[j]) = (base256[j], base256[i]);
            }
            for (int i = 0; i < 512; i++) p[i] = base256[i & 255];
            return p;
        }

        /// <summary>
        /// Tileable Perlin noise: wraps integer coordinates modulo <paramref name="period"/>.
        /// Input (fx,fy) should be in [0,1), scaled by period internally.
        /// </summary>
        private static float TileablePerlin(float fx, float fy, int period,
            int[] perm, ReadOnlySpan<(float, float)> grads)
        {
            // Scale to noise-space grid [0, period)
            float px = fx * period;
            float py = fy * period;

            int ix = (int)MathF.Floor(px);
            int iy = (int)MathF.Floor(py);
            float dx = px - ix;
            float dy = py - iy;

            // Wrap for tiling
            int ix0 = ix % period;
            int iy0 = iy % period;
            int ix1 = (ix + 1) % period;
            int iy1 = (iy + 1) % period;

            // Gradient indices via permutation table
            int gi00 = perm[perm[ix0] + iy0] % 12;
            int gi10 = perm[perm[ix1] + iy0] % 12;
            int gi01 = perm[perm[ix0] + iy1] % 12;
            int gi11 = perm[perm[ix1] + iy1] % 12;

            // Dot products
            float n00 = grads[gi00].Item1 * dx       + grads[gi00].Item2 * dy;
            float n10 = grads[gi10].Item1 * (dx - 1) + grads[gi10].Item2 * dy;
            float n01 = grads[gi01].Item1 * dx        + grads[gi01].Item2 * (dy - 1);
            float n11 = grads[gi11].Item1 * (dx - 1) + grads[gi11].Item2 * (dy - 1);

            // Quintic interpolation (C2 continuous)
            float u = dx * dx * dx * (dx * (dx * 6f - 15f) + 10f);
            float v = dy * dy * dy * (dy * (dy * 6f - 15f) + 10f);

            float nx0 = n00 + u * (n10 - n00);
            float nx1 = n01 + u * (n11 - n01);
            float result = nx0 + v * (nx1 - nx0);

            return result * 0.5f + 0.5f; // map [-1,1] → [0,1]
        }

        /// <summary>
        /// Uploads the CPU store's heights (Terrain.Pages, read from the pages file at load) into the pool, page by page,
        /// each with its border of <see cref="HeightPool.Border"/> samples from its neighbours (clamped to the terrain's
        /// edges, past which nothing reads). Pages go up <see cref="UploadBatch"/> at a time through one upload buffer,
        /// each in its own place in it, so no copy is wider than a page: until 4c (2026-11-07) the flat map Save wrote went
        /// up through one footprint as wide as the map, and D3D12 takes none wider than 16,384 texels (16 km at 1 m per
        /// vertex is 16,385 samples). Used at load in place of a bake.
        /// </summary>
        public bool UploadPages(TerrainPages pages)
        {
            if (pages is not { HasHeights: true }) return false;
            if (PoolCapacity != 0) return false;   // streaming: TerrainHeightResidency fills the windows

            var device = Engine.Device;
            int resolution = pages.Resolution;
            int across = pages.PagesAcross, count = across * across;
            const int texels = HeightPool.PageTexels;
            int rowPitch = (texels * 2 + 255) & ~255;
            int pageBytes = (rowPitch * texels + 511) & ~511;
            int batch = Math.Min(UploadBatch, count);

            var uploadResource = device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Upload),
                HeapFlags.None,
                ResourceDescription.Buffer((ulong)pageBytes * (ulong)batch),
                ResourceStates.GenericRead,
                null);

            try
            {
                for (int first = 0; first < count; first += batch)
                {
                    int n = Math.Min(batch, count - first);
                    unsafe
                    {
                        void* pData;
                        uploadResource.Map(0, null, &pData);
                        var basePtr = (IntPtr)pData;
                        System.Threading.Tasks.Parallel.For(0, n, k =>
                            pages.FillPoolPage(0, (first + k) % across, (first + k) / across, new Span<ushort>((byte*)basePtr + (long)k * pageBytes, pageBytes / 2), rowPitch / 2, HeightPool.Border, HeightPool.PageTexels));
                        uploadResource.Unmap(0);
                    }

                    var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
                    var cmdList = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, allocator, null);
                    try
                    {
                        if (first == 0) _pool.Ensure(resolution, 0, cmdList);
                        cmdList.ResourceBarrierTransition(_pool.Texture, ResourceStates.Common, ResourceStates.CopyDest);
                        for (int k = 0; k < n; k++)
                        {
                            int i = first + k;
                            var (slice, cx, cy) = _pool.PageCorner(i % across, i / across);
                            var src = new TextureCopyLocation(uploadResource, new PlacedSubresourceFootPrint
                            {
                                Offset = (ulong)((long)k * pageBytes),
                                Footprint = new SubresourceFootPrint(Format.R16_UNorm, texels, texels, 1, (uint)rowPitch)
                            });
                            cmdList.CopyTextureRegion(new TextureCopyLocation(_pool.Texture, (uint)slice), (uint)cx, (uint)cy, 0, src, null);
                        }
                        cmdList.ResourceBarrierTransition(_pool.Texture, ResourceStates.CopyDest, ResourceStates.Common);
                        cmdList.Close();
                        device.SubmitAndWait(cmdList);
                        for (int k = 0; k < n; k++)
                            _pool.MarkWritten(first + k, submitted: true);   // every page at its own index
                    }
                    finally
                    {
                        cmdList.Dispose();
                        allocator.Dispose();
                    }
                }

                Debug.Log($"[TerrainBaker] Uploaded {count} pages: {resolution}x{resolution}");
                _pool.Filled = true;
                HeightVersion++;
                return true;
            }
            finally
            {
                uploadResource.Dispose();
            }
        }

        /// <summary>Pages that go up through the upload buffer at a time (<see cref="UploadPages"/>): 128 pages, 25 MB.</summary>
        private const int UploadBatch = 128;

        private static readonly ProfilerMarker _copyMarker = ProfilerMarker.Get("Terrain height copy and compare");

        /// <summary>
        /// Reads the baked heights (the pool) back into the CPU store (<see cref="TerrainPages"/>, TERRAIN_PAGES_PLAN.md
        /// 4b) and returns the store that replaces <paramref name="previous"/>, which is left as it is: workers may still
        /// read it. Null if no heights have been baked.
        ///
        /// Page by page, each page's 257 samples straight out of its slot. Only the pages holding a sample of the box
        /// (x0, z0 inclusive, x1, z1 exclusive) are read, unless <paramref name="whole"/> or <paramref name="previous"/> has
        /// no heights of this size. A page whose samples all equal the one before is kept as that one, so a store shares
        /// what did not change; the holes go over to the new store as they are. On Eldermoor's 4097² map a whole readback
        /// took ~190 ms of the frame it ran in, after every height stamp edit (2026-10-26).
        ///
        /// <paramref name="changed"/> is the box of the samples (x0, z0 inclusive, x1, z1 exclusive) whose value is not the
        /// one 'previous' held, compared as they are copied, or null when every sample read back is as it was. A stamp edit
        /// that bakes the same heights (an Opacity set to what it was) then changes nothing on the CPU. Without heights of
        /// this size to compare with, it is the whole map.
        /// </summary>
        public TerrainPages? ReadbackPages(TerrainPages? previous, bool whole, out (int x0, int z0, int x1, int z1)? changed,
            int x0 = 0, int z0 = 0, int x1 = int.MaxValue, int z1 = int.MaxValue)
        {
            changed = null;
            if (_pool.Texture == null)
                return null;

            int res = _pool.Resolution;
            bool compare = previous is { HasHeights: true } && previous.Resolution == res;
            var basis = previous != null && previous.Resolution == res ? previous : TerrainPages.Empty(res);
            if (whole || !compare)
            {
                x0 = 0; z0 = 0; x1 = res; z1 = res;
            }
            x0 = Math.Clamp(x0, 0, res); x1 = Math.Clamp(x1, 0, res);
            z0 = Math.Clamp(z0, 0, res); z1 = Math.Clamp(z1, 0, res);
            if (compare && (x1 <= x0 || z1 <= z0))
                return previous;

            var (px0, pz0, px1, pz1) = basis.PagesOf(x0, z0, x1, z1);
            int across = basis.PagesAcross;
            var inBox = new HashSet<int>();
            var pageList = new List<(int px, int pz, int slot)>((px1 - px0) * (pz1 - pz0));
            for (int pz = pz0; pz < pz1; pz++)
                for (int px = px0; px < px1; px++)
                {
                    inBox.Add(pz * across + px);
                    pageList.Add((px, pz, -1));
                }
            // Pages a bake owed the CPU store, wherever they are
            foreach (int owed in _owed)
                if (owed < across * across && inBox.Add(owed))
                    pageList.Add((owed % across, owed / across, -1));

            // Where each page is read from: its slot when it is resident, written on a list already handed to the GPU and
            // not skipped by a bake since; else it is baked into a scratch slot with the last bake's plan first
            // (TERRAIN_PAGES_PLAN.md 5a). Without that plan such a page keeps what the store had.
            var plan = _lastPlan is { } p && p.Resolution == res ? p : null;
            int skipped = 0;
            for (int i = pageList.Count - 1; i >= 0; i--)
            {
                var (px, pz, _) = pageList[i];
                int slot = _pool.SlotOf(0, px, pz);
                bool readable = _pool.AllResident
                    ? slot >= 0
                    : slot >= 0 && _pool.Settled(slot) && !_owed.Contains(pz * across + px);
                if (readable) pageList[i] = (px, pz, slot);
                else if (plan == null) { pageList.RemoveAt(i); skipped++; }
            }
            if (pageList.Count == 0)
                return compare ? previous : null;

            // A page's rows 256-byte aligned, each page at a 512-byte boundary (the placed footprint's alignment); a
            // batch of pages per copy, so a large terrain needs no readback buffer the size of its map
            const int rowPitch = (TerrainPages.PageSamples * sizeof(ushort) + 255) & ~255;
            const int pageBytes = (rowPitch * TerrainPages.PageSamples + 511) & ~511;
            const int batch = 64;

            var device = Engine.Device;
            var edit = basis.Edit();
            int cx0 = int.MaxValue, cx1 = int.MinValue, cz0 = int.MaxValue, cz1 = int.MinValue;
            int replaced = 0;
            var readbackResource = device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Readback),
                HeapFlags.None,
                ResourceDescription.Buffer((ulong)(pageBytes * Math.Min(batch, pageList.Count))),
                ResourceStates.CopyDest,
                null);

            try
            {
                int baked = 0;
                for (int first = 0; first < pageList.Count; first += batch)
                {
                    int count = Math.Min(batch, pageList.Count - first);

                    // The batch's pages with no readable slot: baked into scratch slots on a list of their own first
                    var bake = new List<(int px, int py, int slot)>();
                    for (int k = 0; k < count; k++)
                    {
                        var (px, pz, slot) = pageList[first + k];
                        if (slot >= 0) continue;
                        int scratch = _pool.ScratchSlot(bake.Count);
                        bake.Add((px, pz, scratch));
                        pageList[first + k] = (px, pz, scratch);
                    }
                    if (bake.Count > 0)
                    {
                        BakeIntoScratch(plan!, bake);
                        baked += bake.Count;
                    }

                    ReadPoolPages(readbackResource, pageList, first, count, rowPitch, pageBytes, basis);
                    for (int k = 0; k < count; k++)
                        _owed.Remove(pageList[first + k].pz * across + pageList[first + k].px);

                    unsafe
                    {
                        void* pData;
                        readbackResource.Map(0, null, &pData);
                        using (_copyMarker.Auto())
                        {
                            for (int k = 0; k < count; k++)
                            {
                                var (px, pz, _) = pageList[first + k];
                                var (sx, sz, w, h) = basis.PageExtent(px, pz);
                                var before = compare ? basis.HeightPage(px, pz) : default;
                                var src = (byte*)pData + (long)k * pageBytes;

                                // Compared row by row with the page before; a page that differs anywhere is made anew,
                                // and the rows that differ widen the box
                                ushort[]? page = null;
                                for (int z = 0; z < h; z++)
                                {
                                    var row = new ReadOnlySpan<ushort>(src + (long)z * rowPitch, w);
                                    if (!before.IsEmpty)
                                    {
                                        var old = before.Slice(z * TerrainPages.PageSamples, w);
                                        int firstDiff = row.CommonPrefixLength(old);
                                        if (firstDiff == w) continue;
                                        int lastDiff = w - 1;
                                        while (lastDiff > firstDiff && row[lastDiff] == old[lastDiff]) lastDiff--;
                                        cx0 = Math.Min(cx0, sx + firstDiff);
                                        cx1 = Math.Max(cx1, sx + lastDiff + 1);
                                        cz0 = Math.Min(cz0, sz + z);
                                        cz1 = Math.Max(cz1, sz + z + 1);
                                        if (page == null)
                                        {
                                            page = new ushort[TerrainPages.PageLength];
                                            before.CopyTo(page);
                                        }
                                    }
                                    else page ??= new ushort[TerrainPages.PageLength];
                                    row.CopyTo(page.AsSpan(z * TerrainPages.PageSamples, w));
                                }
                                if (page != null)
                                {
                                    edit.SetHeights(px, pz, page);
                                    replaced++;
                                }
                            }
                        }
                        readbackResource.Unmap(0);
                    }
                }

                var pages = edit.Done(hasHeights: true);
                if (!compare) changed = (0, 0, res, res);
                else if (cx1 > cx0) changed = (cx0, cz0, cx1, cz1);
                Debug.Log($"[TerrainBaker] Heights read back: {pageList.Count} of {pages.PagesAcross * pages.PagesAcross} pages, " +
                          $"{replaced} new, {baked} baked in scratch slots, {skipped} not resident and no plan to bake them, " +
                          (changed is { } box ? $"changed x {box.x0}..{box.x1}, z {box.z0}..{box.z1}" : "nothing changed"));
                return pages;
            }
            finally
            {
                readbackResource.Dispose();
            }
        }

        /// <summary>
        /// Bake these pages of a plan into the scratch slots they were given (a streaming pool, TERRAIN_PAGES_PLAN.md 5a), on
        /// a list of their own: the readback's way to the heights of pages that are not resident. Blocks until the GPU is
        /// done.
        /// </summary>
        private void BakeIntoScratch(HeightPlan plan, List<(int px, int py, int slot)> pages)
        {
            var device = Engine.Device;
            EnsureInitialized();
            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            var cmd = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, allocator, null);
            try
            {
                cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
                cmd.SetComputeRootSignature(device.GlobalRootSignature);
                UploadHeightPlan(plan);
                cmd.ResourceBarrierTransition(_pool.Texture, ResourceStates.Common, ResourceStates.UnorderedAccess);
                RecordHeight(plan, cmd, pages, wholeRuns: false);
                cmd.ResourceBarrierTransition(_pool.Texture, ResourceStates.UnorderedAccess, ResourceStates.Common);
                cmd.Close();
                device.SubmitAndWait(cmd);
            }
            finally
            {
                cmd.Dispose();
                allocator.Dispose();
            }
        }

        /// <summary>
        /// Copy whole pages of the pool (each its own samples, no border) from their slots into a readback buffer, page k at
        /// k × pageBytes, rows of <paramref name="rowPitch"/> bytes. Its own command list; blocks until the GPU is done.
        /// </summary>
        private void ReadPoolPages(ID3D12Resource readback, List<(int px, int pz, int slot)> pages, int first, int count,
            int rowPitch, int pageBytes, TerrainPages layout)
        {
            var device = Engine.Device;
            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            var cmdList = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, allocator, null);
            try
            {
                cmdList.ResourceBarrierTransition(_pool.Texture, ResourceStates.Common, ResourceStates.CopySource);
                for (int k = 0; k < count; k++)
                {
                    var (px, pz, slot) = pages[first + k];
                    var (_, _, w, h) = layout.PageExtent(px, pz);
                    var (slice, cx, cy) = HeightPool.SlotCorner(slot);
                    int tx = cx + HeightPool.Border, tz = cy + HeightPool.Border;
                    var dst = new TextureCopyLocation(readback, new PlacedSubresourceFootPrint
                    {
                        Offset = (ulong)((long)k * pageBytes),
                        Footprint = new SubresourceFootPrint(Format.R16_UNorm, (uint)w, (uint)h, 1, (uint)rowPitch)
                    });
                    cmdList.CopyTextureRegion(dst, 0, 0, 0, new TextureCopyLocation(_pool.Texture, (uint)slice),
                        new Vortice.Mathematics.Box(tx, tz, 0, tx + w, tz + h, 1));
                }
                cmdList.ResourceBarrierTransition(_pool.Texture, ResourceStates.CopySource, ResourceStates.Common);
                cmdList.Close();
                device.SubmitAndWait(cmdList);
            }
            finally
            {
                cmdList.Dispose();
                allocator.Dispose();
            }
        }

        /// <summary>Samples a side of the baked heights (the pool), or 0 if none.</summary>
        public int BakedResolution => _pool.Texture != null ? _pool.Resolution : 0;

        /// <summary>Counts the baked heightmaps' changes (a bake, an upload from the cache): what is made from the heights
        /// (Graphics.TerrainLightShadow) is made again when it moves on.</summary>
        public static int HeightVersion { get; private set; }

        /// <summary>The pool's heights changed some other way than a bake (filled from the CPU store).</summary>
        internal static void NoteHeightsChanged() => HeightVersion++;

        // ═══════════════════════════════════════════════════════════════════
        // ── Coverage (splat + decoration) ──
        // ═══════════════════════════════════════════════════════════════════

        /// <summary>Stamp descriptors for a splat or decoration bake, captured on the main thread.</summary>
        public sealed class CoveragePlan
        {
            internal readonly List<CoverageStampGPU> Stamps = new();
            internal readonly List<StampSplinePointGPU> SplinePoints = new();
            internal readonly List<DecoRowGPU> Rows = new();     // deco: the stamps that grow in rows

            /// <summary>Channels (splat) or slots (deco) the stamps index into.</summary>
            public int TargetCount;

            /// <summary>Number of layer channels the filters' layer masks refer to.</summary>
            public int LayerCount;

            public int StampCount => Stamps.Count;
        }

        /// <summary>
        /// Lay out a splat bake from the terrain's paint entries (already in bake order: the splat stamps and the biome
        /// stamps' layer rules, <see cref="CoverageEntries.Splat"/>).
        /// 'layers' is the terrain's palette: an entry's Layer becomes the index of its channel.
        /// Main thread.
        /// </summary>
        public CoveragePlan PrepareSplat(Terrain terrain, TerrainRenderer renderer,
            IReadOnlyList<SplatEntry> entries, IReadOnlyList<TerrainLayer> layers)
        {
            var plan = new CoveragePlan { TargetCount = layers.Count, LayerCount = layers.Count };
            var layerIndex = BuildIndex(layers);
            var terrainOrigin = renderer.Transform?.WorldPosition ?? Vector3.Zero;
            TerrainStamp? lastShape = null;

            foreach (var entry in entries)
            {
                if (entry.Layer == null || !layerIndex.TryGetValue(entry.Layer, out int channel)) continue;

                int pointsBefore = plan.SplinePoints.Count;
                if (!TryBuildCoverage(entry.Shape, entry.Filter, terrain, terrainOrigin, layerIndex, plan.SplinePoints, out var desc))
                    continue;
                if (ReferenceEquals(entry.Shape, lastShape)) SameShapeAsLast(plan, ref desc, pointsBefore);

                desc.Flags |= (uint)entry.Op << 8;
                desc.Target = (uint)channel;
                desc.Strength = entry.Strength;
                plan.Stamps.Add(desc);
                lastShape = entry.Shape;
            }

            return plan;
        }

        /// <summary>
        /// Lay out a decoration coverage bake from the terrain's ground cover entries (already in bake order: the deco
        /// stamps and the biome stamps' decorator rules, <see cref="CoverageEntries.Deco"/>).
        /// 'decorators' is the terrain's decorator palette; 'layers' its layer palette (for filters).
        /// Main thread.
        /// </summary>
        public CoveragePlan PrepareDeco(Terrain terrain, TerrainRenderer renderer,
            IReadOnlyList<DecoEntry> entries, IReadOnlyList<TerrainDecorator> decorators,
            IReadOnlyList<TerrainLayer> layers)
        {
            var plan = new CoveragePlan { TargetCount = decorators.Count, LayerCount = layers.Count };
            var layerIndex = BuildIndex(layers);
            var decoratorIndex = BuildIndex(decorators);
            var terrainOrigin = renderer.Transform?.WorldPosition ?? Vector3.Zero;
            TerrainStamp? lastShape = null;

            foreach (var entry in entries)
            {
                uint target;
                if (entry.Decorator == null)
                {
                    // No decorator: a Multiply entry thins or boosts everything; an Add entry has nothing to add
                    if (entry.Op != DecoOp.Multiply) continue;
                    target = AllTargets;
                }
                else if (decoratorIndex.TryGetValue(entry.Decorator, out int slot))
                {
                    target = (uint)slot;
                }
                else continue;

                int pointsBefore = plan.SplinePoints.Count;
                if (!TryBuildCoverage(entry.Shape, entry.Filter, terrain, terrainOrigin, layerIndex, plan.SplinePoints, out var desc))
                    continue;
                if (ReferenceEquals(entry.Shape, lastShape)) SameShapeAsLast(plan, ref desc, pointsBefore);

                desc.Flags |= (uint)entry.Op << 8;
                desc.Target = target;
                desc.Strength = entry.Weight;
                if (entry is { Op: DecoOp.Add, Rows: { RowSpacing: > 0.05f } rows } && plan.Rows.Count < MaxDecoRows)
                {
                    plan.Rows.Add(new DecoRowGPU
                    {
                        Normal = RowNormal(rows),
                        Spacing = rows.RowSpacing,
                        Width = Math.Clamp(rows.RowWidth, 0f, rows.RowSpacing),
                        Slot = target,
                    });
                    desc.Rows = (uint)plan.Rows.Count;
                }
                plan.Stamps.Add(desc);
                lastShape = entry.Shape;
            }

            return plan;
        }

        /// <summary>
        /// The entry just built has the shape of the stamp added before it (a biome stamp's next rule): the shaders reuse
        /// that stamp's shape weight (COVERAGE_SAME_SHAPE) rather than walk the outline again for every rule, every sample.
        /// On TestWorld a biome's 4-7 rules each walked its 113-176-point outline, 16 times a texel on a coarse page: about
        /// 100 ms of GPU a page, 0.4 s a frame at four pages (2026-11-08). The copy of its spline points is dropped.
        /// </summary>
        private static void SameShapeAsLast(CoveragePlan plan, ref CoverageStampGPU desc, int pointsBefore)
        {
            var previous = plan.Stamps[^1];
            plan.SplinePoints.RemoveRange(pointsBefore, plan.SplinePoints.Count - pointsBefore);
            desc.SplinePointOffset = previous.SplinePointOffset;
            desc.SplinePointCount = previous.SplinePointCount;
            desc.Flags |= CoverageSameShape;
        }

        /// <summary>
        /// Across a row stamp's rows in world XZ: the rows run along its spline's longest control edge (a field is
        /// ploughed and sown along its length), or along world +X without a spline, turned by RowAngle.
        /// </summary>
        internal static Vector2 RowNormal(DecoStamp stamp)
        {
            var along = AlongLongestEdge(stamp, stamp.RowAngle);
            return new Vector2(-along.Y, along.X);
        }

        /// <summary>
        /// Along a stamp's spline's longest control edge in world XZ (a field is ploughed and sown along its length), or
        /// world +X without a spline, turned by <paramref name="angleDegrees"/>: the direction of a field's rows and ridges.
        /// </summary>
        internal static Vector2 AlongLongestEdge(TerrainStamp stamp, float angleDegrees)
        {
            var along = Vector2.UnitX;
            var spline = stamp.IsSplineMode ? stamp.GetSpline() : null;
            if (spline != null && spline.Points.Count >= 2 && spline.Transform != null)
            {
                var world = spline.Transform.Matrix;
                int n = spline.Points.Count, edges = spline.Closed ? n : n - 1;
                float longest = 0f;
                for (int i = 0; i < edges; i++)
                {
                    var a = Vector3.Transform(spline.Points[i], world);
                    var b = Vector3.Transform(spline.Points[(i + 1) % n], world);
                    var edge = new Vector2(b.X - a.X, b.Z - a.Z);
                    float lengthSq = edge.LengthSquared();
                    if (lengthSq > longest) { longest = lengthSq; along = edge / MathF.Sqrt(lengthSq); }
                }
            }
            float turn = angleDegrees * MathF.PI / 180f;
            float c = MathF.Cos(turn), s = MathF.Sin(turn);
            return new Vector2(along.X * c - along.Y * s, along.X * s + along.Y * c);
        }

        private static Dictionary<T, int> BuildIndex<T>(IReadOnlyList<T> items) where T : class
        {
            var index = new Dictionary<T, int>(items.Count);
            for (int i = 0; i < items.Count; i++)
                index.TryAdd(items[i], i);
            return index;
        }

        /// <summary>
        /// Shape and edge noise of a coverage entry's stamp (a splat or deco stamp, or the biome stamp of a rule) and the
        /// entry's filter. Op, target and strength are the caller's.
        /// </summary>
        private static bool TryBuildCoverage(TerrainStamp stamp, StampFilter filter, Terrain terrain, Vector3 terrainOrigin,
            Dictionary<TerrainLayer, int> layerIndex, List<StampSplinePointGPU> splinePoints, out CoverageStampGPU desc)
        {
            desc = new CoverageStampGPU();
            var terrainSize = terrain.TerrainSize;
            float uvScale = 1f / Math.Max(terrainSize.X, terrainSize.Y);

            if (stamp.IsGlobal)
            {
                desc.Flags |= CoverageGlobal;
                desc.SplinePointOffset = 0xFFFFFFFF;
                desc.Bounds = new Vector4(float.MinValue, float.MinValue, float.MaxValue, float.MaxValue);
            }
            else
            {
                desc.Falloff = stamp.Falloff * uvScale;

                if (stamp.EnableNoise)
                {
                    desc.NoiseFreq = stamp.NoiseFrequency;
                    desc.NoiseAmp = stamp.NoiseAmplitude * uvScale;
                    desc.NoiseSeed = (uint)stamp.NoiseSeed;
                }

                if (stamp.IsSplineMode)
                {
                    if (!BuildSplinePoints(stamp, terrainOrigin, terrainSize, terrain.MaxHeight, 0f, splinePoints,
                            out desc.SplinePointOffset, out desc.SplinePointCount, out desc.Radius))
                    {
                        // A spline without enough points covers nothing
                        return false;
                    }
                }
                else
                {
                    var pos = stamp.Transform?.WorldPosition ?? Vector3.Zero;
                    desc.Center = new Vector2(
                        (pos.X - terrainOrigin.X) / terrainSize.X,
                        (pos.Z - terrainOrigin.Z) / terrainSize.Y);
                    desc.Radius = stamp.Radius * uvScale;
                    desc.SplinePointOffset = 0xFFFFFFFF;
                }
                desc.Bounds = CoverageBounds(desc, splinePoints, uvScale);
            }

            // Filter. The ranges are always written so the shader never reads garbage.
            desc.HeightMin = filter.HeightRange.X;
            desc.HeightMax = filter.HeightRange.Y;
            desc.HeightBlend = filter.HeightBlend;
            desc.SlopeMin = filter.SlopeRange.X;
            desc.SlopeMax = filter.SlopeRange.Y;
            desc.SlopeBlend = filter.SlopeBlend;
            desc.PatchSize = filter.PatchSize;
            desc.PatchCoverage = filter.PatchCoverage;
            desc.PatchSoftness = filter.PatchSoftness;
            desc.PatchSeed = (uint)filter.PatchSeed;

            if (filter.HasFilter)
            {
                desc.Flags |= CoverageFilter;
                desc.RequireMask = LayerMask(filter.RequireLayers, layerIndex, out bool anyRequired);
                desc.ExcludeMask = LayerMask(filter.ExcludeLayers, layerIndex, out _);

                // Requires layers, but none of them is on this terrain: the entry can never apply
                if (anyRequired && desc.RequireMask == 0)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// The terrain UV box a local coverage stamp's shape can reach: its centre or its spline's samples, grown by
        /// what EvaluateStampWeight lets it reach past them (radius and falloff, both scaled by a sample's width;
        /// a radius of 0 or less leaves the falloff as it is), the edge noise and a texel. The bakes skip the shape
        /// outside it.
        /// </summary>
        private static Vector4 CoverageBounds(in CoverageStampGPU desc, List<StampSplinePointGPU> splinePoints, float uvScale)
        {
            float reachOut = desc.NoiseAmp + uvScale;
            if (desc.SplinePointOffset == 0xFFFFFFFF)
            {
                float r = Math.Max(desc.Radius, 0f) + desc.Falloff + reachOut;
                return new Vector4(desc.Center.X - r, desc.Center.Y - r, desc.Center.X + r, desc.Center.Y + r);
            }

            int first = (int)desc.SplinePointOffset;
            int count = (int)(desc.SplinePointCount & 0x7FFFFFFF);
            var min = new Vector2(float.MaxValue);
            var max = new Vector2(float.MinValue);
            float halfWidth = 0f;
            for (int i = first; i < first + count; i++)
            {
                var p = splinePoints[i];
                min = Vector2.Min(min, p.UV);
                max = Vector2.Max(max, p.UV);
                halfWidth = Math.Max(halfWidth, p.HalfWidth);
            }
            float falloff = desc.Radius > 1e-7f ? desc.Falloff * halfWidth / desc.Radius : desc.Falloff;
            float reach = Math.Max(halfWidth, 0f) + falloff + reachOut;
            return new Vector4(min.X - reach, min.Y - reach, max.X + reach, max.Y + reach);
        }

        /// <summary>Bit N set = layer at channel N. Layers that are not in the palette have no weight anywhere.</summary>
        private static uint LayerMask(List<TerrainLayer> layers, Dictionary<TerrainLayer, int> layerIndex, out bool any)
        {
            any = false;
            if (layers == null) return 0;

            uint mask = 0;
            foreach (var layer in layers)
            {
                if (layer == null) continue;
                any = true;
                if (layerIndex.TryGetValue(layer, out int channel))
                    mask |= 1u << channel;
            }
            return mask;
        }

        /// <summary>
        /// Bake these paint pages into their slots of the paint pool (TERRAIN_PAGES_PLAN.md 5b): every texel of a page,
        /// border included, composited from all the stamps (a coarse page averages a 4 × 4 grid over each texel), every
        /// layer slice and the top layer written. The plan's stamps go up once per plan. Render callback, on the frame's
        /// list after the heights; the slots count as written by this frame.
        /// </summary>
        public void BakeSplatPages(CoveragePlan plan, ID3D12GraphicsCommandList cmd, Terrain terrain, uint terrainPagesIdx,
            Graphics.PaintPool pool, IReadOnlyList<(int level, int px, int py, int slot)> pages)
        {
            if (pages.Count == 0 || pool.Weights == null) return;
            EnsureCoverageInitialized();
            if (!ReferenceEquals(_uploadedSplatPlan, plan))
            {
                UploadCoverage(plan, ref _splatStampBuffer, ref _splatStampBufferCapacity,
                    ref _splatSplineBuffer, ref _splatSplineCapacity);
                _uploadedSplatPlan = plan;
            }

            var device = Engine.Device;
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.ResourceBarrierTransition(pool.Weights, ResourceStates.Common, ResourceStates.UnorderedAccess);
            cmd.ResourceBarrierTransition(pool.TopLayer, ResourceStates.Common, ResourceStates.UnorderedAccess);

            var k = _kernelSplatBake;
            _splatCS.SetBuffer(k, "StampBuf", _splatStampBuffer);
            _splatCS.SetPushConstant(k, "Output", pool.WeightsUav);
            _splatCS.SetPushConstant(k, "TerrainPages", terrainPagesIdx);
            _splatCS.SetPushConstant(k, "SplineBuf", _splatSplineBuffer.SrvIndex);
            _splatCS.SetPushConstant(k, "StampCount", (uint)plan.StampCount);
            _splatCS.SetPushConstant(k, "Resolution", (uint)pool.Texels);
            _splatCS.SetPushConstant(k, "SliceCount", (uint)pool.LayerSlices);
            _splatCS.SetPushConstant(k, "LayerCount", (uint)plan.TargetCount);
            _splatCS.SetParam(k, "MaxHeight", terrain.MaxHeight);
            _splatCS.SetParam(k, "TerrainSizeX", terrain.TerrainSize.X);
            _splatCS.SetPushConstant(k, "TopLayer", pool.TopUav);
            uint groups = (uint)((Graphics.PaintPool.SlotTexels + 7) / 8);
            foreach (var (level, px, py, slot) in pages)
            {
                var (x, y, weightSlice, topSlice) = pool.SlotPlace(slot);
                _splatCS.SetPushConstant(k, "PageX", unchecked((uint)px));
                _splatCS.SetPushConstant(k, "PageY", unchecked((uint)py));
                _splatCS.SetPushConstant(k, "Level", (uint)level);
                _splatCS.SetPushConstant(k, "OutX", (uint)x);
                _splatCS.SetPushConstant(k, "OutY", (uint)y);
                _splatCS.SetPushConstant(k, "OutSlice", (uint)weightSlice);
                _splatCS.SetPushConstant(k, "OutTopSlice", (uint)topSlice);
                _splatCS.Dispatch(k, cmd, groups, groups);
                pool.Slots.MarkWritten(slot);
            }

            cmd.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(pool.Weights)));
            cmd.ResourceBarrierTransition(pool.Weights, ResourceStates.UnorderedAccess, ResourceStates.Common);
            cmd.ResourceBarrierTransition(pool.TopLayer, ResourceStates.UnorderedAccess, ResourceStates.Common);
        }

        // The splat plan whose stamps are in the upload buffers
        private CoveragePlan? _uploadedSplatPlan;

        /// <summary>
        /// A paint page's top layer (the channel that shows most per texel, 255 = none) read back from its slot: 256 × 256
        /// bytes, row by row along +Z, the page's own texels without the border. Blocks until the GPU is done; only for a
        /// slot whose bake was handed to the GPU before this frame. Main thread.
        /// </summary>
        public static byte[]? ReadbackTopPage(Graphics.PaintPool pool, int slot)
        {
            if (pool.TopLayer == null || !pool.Slots.Settled(slot)) return null;
            const int texels = Graphics.PaintPool.PageTexels;
            const int rowPitch = (texels + 255) & ~255;
            var device = Engine.Device;
            var readback = device.NativeDevice.CreateCommittedResource(new HeapProperties(HeapType.Readback), HeapFlags.None,
                ResourceDescription.Buffer((ulong)(rowPitch * texels)), ResourceStates.CopyDest, null);
            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            var cmd = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, allocator, null);
            try
            {
                var (x, y, _, topSlice) = pool.SlotPlace(slot);
                int tx = x + Graphics.PaintPool.Border, ty = y + Graphics.PaintPool.Border;
                cmd.ResourceBarrierTransition(pool.TopLayer, ResourceStates.Common, ResourceStates.CopySource);
                cmd.CopyTextureRegion(new TextureCopyLocation(readback, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R8_UInt, texels, texels, 1, rowPitch)
                }), 0, 0, 0, new TextureCopyLocation(pool.TopLayer, (uint)topSlice),
                    new Vortice.Mathematics.Box(tx, ty, 0, tx + texels, ty + texels, 1));
                cmd.ResourceBarrierTransition(pool.TopLayer, ResourceStates.CopySource, ResourceStates.Common);
                cmd.Close();
                device.SubmitAndWait(cmd);

                var layers = new byte[texels * texels];
                unsafe
                {
                    void* data;
                    readback.Map(0, null, &data);
                    fixed (byte* dst = layers)
                        for (int r = 0; r < texels; r++)
                            Buffer.MemoryCopy((byte*)data + (long)r * rowPitch, dst + (long)r * texels, texels, texels);
                    readback.Unmap(0);
                }
                return layers;
            }
            finally
            {
                cmd.Dispose();
                allocator.Dispose();
                readback.Dispose();
            }
        }

        /// <summary>
        /// Every level 0 paint page resident in the pool (all of them with EngineSettings.TerrainPaintPages 0) against the
        /// whole-map bake there was before 5b (CS_SplatBakeFlat, Y-flipped), baked over the same texels: their weights byte for
        /// byte. Returns the texels and pages that differ, the largest difference of a channel (of 255), the texel slices
        /// compared and the pages checked. Blocks. Main thread.
        /// </summary>
        public unsafe (long texels, int pages, int maxStep, long compared, int checkedPages) CheckPaintPages(CoveragePlan plan, Terrain terrain,
            uint terrainPagesIdx, Graphics.PaintPool pool)
        {
            if (pool.Weights == null) return (-1, -1, -1, 0, 0);
            EnsureCoverageInitialized();
            UploadCoverage(plan, ref _splatStampBuffer, ref _splatStampBufferCapacity, ref _splatSplineBuffer, ref _splatSplineCapacity);
            _uploadedSplatPlan = plan;

            const int texels = Graphics.PaintPool.PageTexels;
            int res = pool.Texels, g = pool.LayerSlices, across = pool.Slots.PagesAcross;
            const int rowPitch = texels * 4;   // 1024, 256-aligned
            const int sliceBytes = rowPitch * texels;
            var device = Engine.Device;
            var flat = device.CreateTexture2D(Format.R8G8B8A8_UNorm, texels, texels, g, 1, ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
            uint flatUav = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(flat, null, new UnorderedAccessViewDescription
            {
                Format = Format.R8G8B8A8_UNorm,
                ViewDimension = UnorderedAccessViewDimension.Texture2DArray,
                Texture2DArray = new Texture2DArrayUnorderedAccessView { MipSlice = 0, FirstArraySlice = 0, ArraySize = (uint)g },
            }, device.GetCpuHandle(flatUav));
            var readback = device.NativeDevice.CreateCommittedResource(new HeapProperties(HeapType.Readback), HeapFlags.None,
                ResourceDescription.Buffer((ulong)sliceBytes * (ulong)g * 2), ResourceStates.CopyDest, null);
            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            long differ = 0, compared = 0;
            int differPages = 0, maxStep = 0, checkedPages = 0;
            try
            {
                for (int pz = 0; pz < across; pz++)
                    for (int px = 0; px < across; px++)
                    {
                        int x0 = px * texels, j0 = pz * texels;
                        int w = Math.Min(texels, res - x0), h = Math.Min(texels, res - j0);
                        if (w <= 0 || h <= 0) continue;
                        int y0 = res - j0 - h;   // the flat array's rows of these texel rows (flipped)
                        int slot = pool.Slots.SlotOf(0, px, pz);
                        if (slot < 0 || !pool.Slots.Settled(slot)) continue;   // not resident at level 0 (streaming), or its bake not run
                        checkedPages++;
                        var (sx, sy, weightSlice, _) = pool.SlotPlace(slot);

                        var cmd = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, allocator, null);
                        cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
                        cmd.SetComputeRootSignature(device.GlobalRootSignature);
                        cmd.ResourceBarrierTransition(flat, ResourceStates.Common, ResourceStates.UnorderedAccess);
                        var k = _kernelSplatBakeFlat;
                        _splatCS.SetBuffer(k, "StampBuf", _splatStampBuffer);
                        _splatCS.SetPushConstant(k, "Output", flatUav);
                        _splatCS.SetPushConstant(k, "TerrainPages", terrainPagesIdx);
                        _splatCS.SetPushConstant(k, "SplineBuf", _splatSplineBuffer.SrvIndex);
                        _splatCS.SetPushConstant(k, "StampCount", (uint)plan.StampCount);
                        _splatCS.SetPushConstant(k, "Resolution", (uint)res);
                        _splatCS.SetPushConstant(k, "SliceCount", (uint)g);
                        _splatCS.SetPushConstant(k, "LayerCount", (uint)plan.TargetCount);
                        _splatCS.SetParam(k, "MaxHeight", terrain.MaxHeight);
                        _splatCS.SetParam(k, "TerrainSizeX", terrain.TerrainSize.X);
                        _splatCS.SetPushConstant(k, "TopLayer", 0u);
                        _splatCS.SetPushConstant(k, "RegionX", (uint)x0);
                        _splatCS.SetPushConstant(k, "RegionY", (uint)y0);
                        _splatCS.SetPushConstant(k, "RegionW", (uint)w);
                        _splatCS.SetPushConstant(k, "RegionH", (uint)h);
                        _splatCS.Dispatch(k, cmd, (uint)((w + 7) / 8), (uint)((h + 7) / 8));
                        cmd.ResourceBarrierTransition(flat, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                        cmd.ResourceBarrierTransition(pool.Weights, ResourceStates.Common, ResourceStates.CopySource);
                        for (int s = 0; s < g; s++)
                        {
                            cmd.CopyTextureRegion(new TextureCopyLocation(readback, new PlacedSubresourceFootPrint
                            {
                                Offset = (ulong)((long)s * sliceBytes),
                                Footprint = new SubresourceFootPrint(Format.R8G8B8A8_UNorm, texels, texels, 1, rowPitch)
                            }), 0, 0, 0, new TextureCopyLocation(flat, (uint)s), new Vortice.Mathematics.Box(0, 0, 0, w, h, 1));
                            int bx = sx + Graphics.PaintPool.Border, by = sy + Graphics.PaintPool.Border;
                            cmd.CopyTextureRegion(new TextureCopyLocation(readback, new PlacedSubresourceFootPrint
                            {
                                Offset = (ulong)((long)(g + s) * sliceBytes),
                                Footprint = new SubresourceFootPrint(Format.R8G8B8A8_UNorm, texels, texels, 1, rowPitch)
                            }), 0, 0, 0, new TextureCopyLocation(pool.Weights, (uint)(weightSlice + s)),
                                new Vortice.Mathematics.Box(bx, by, 0, bx + w, by + h, 1));
                        }
                        cmd.ResourceBarrierTransition(flat, ResourceStates.CopySource, ResourceStates.Common);
                        cmd.ResourceBarrierTransition(pool.Weights, ResourceStates.CopySource, ResourceStates.Common);
                        cmd.Close();
                        device.SubmitAndWait(cmd);
                        cmd.Dispose();
                        allocator.Reset();

                        void* data;
                        readback.Map(0, null, &data);
                        var bytes = (byte*)data;
                        long before = differ;
                        for (int s = 0; s < g; s++)
                            for (int ty = 0; ty < h; ty++)
                            {
                                // Flat region row ty holds texel row j0 + h - 1 - ty: the page's local row h - 1 - ty
                                byte* f = bytes + (long)s * sliceBytes + (long)ty * rowPitch;
                                byte* p = bytes + (long)(g + s) * sliceBytes + (long)(h - 1 - ty) * rowPitch;
                                for (int tx = 0; tx < w; tx++)
                                {
                                    bool same = true;
                                    for (int c = 0; c < 4; c++)
                                    {
                                        int d = Math.Abs(f[tx * 4 + c] - p[tx * 4 + c]);
                                        if (d != 0) { same = false; maxStep = Math.Max(maxStep, d); }
                                    }
                                    if (!same) differ++;
                                    compared++;
                                }
                            }
                        readback.Unmap(0);
                        if (differ > before) differPages++;
                    }
                return (differ, differPages, maxStep, compared, checkedPages);
            }
            finally
            {
                allocator.Dispose();
                readback.Dispose();
                device.DeferDispose(flat);
                device.DeferReleaseBindlessIndex(flatUav);
            }
        }

        /// <summary>
        /// Bake these decoration pages into their slots of the decoration pool (TERRAIN_PAGES_PLAN.md 5c): every texel of a
        /// page, border included, the top 8 decorators (a coarse page averages each decorator's coverage over a 4 × 4 grid
        /// per texel). The layer filters read the paint pages through <paramref name="terrainPagesIdx"/>
        /// (TerrainPagesParams.PaintIdx). The plan's stamps and its rows (the spawn's DecoRowsSrv) go up once per plan.
        /// Render callback, on the frame's list after the paint; the slots count as written by this frame.
        /// </summary>
        public void BakeDecoPages(CoveragePlan plan, ID3D12GraphicsCommandList cmd, Terrain terrain, uint terrainPagesIdx,
            Graphics.DecoPool pool, IReadOnlyList<(int level, int px, int py, int slot)> pages)
        {
            if (pages.Count == 0 || pool.Control == null) return;
            UploadDecoPlan(plan);

            var device = Engine.Device;
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.ResourceBarrierTransition(pool.Control, ResourceStates.Common, ResourceStates.UnorderedAccess);

            var k = _kernelDecoPage;
            SetDecoConstants(k, plan, terrain, terrainPagesIdx, pool.ControlUav, pool.Texels);
            uint groups = (uint)((Graphics.DecoPool.SlotTexels + 7) / 8);
            foreach (var (level, px, py, slot) in pages)
            {
                var (x, y, slice) = pool.SlotPlace(slot);
                _decoCS.SetPushConstant(k, "PageX", unchecked((uint)px));
                _decoCS.SetPushConstant(k, "PageY", unchecked((uint)py));
                _decoCS.SetPushConstant(k, "Level", (uint)level);
                _decoCS.SetPushConstant(k, "OutX", (uint)x);
                _decoCS.SetPushConstant(k, "OutY", (uint)y);
                _decoCS.SetPushConstant(k, "OutSlice", (uint)slice);
                _decoCS.Dispatch(k, cmd, groups, groups);
                pool.Slots.MarkWritten(slot);
            }

            cmd.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(pool.Control)));
            cmd.ResourceBarrierTransition(pool.Control, ResourceStates.UnorderedAccess, ResourceStates.Common);
        }

        // The decoration plan whose stamps and rows are in the upload buffers
        private CoveragePlan? _uploadedDecoPlan;

        private void UploadDecoPlan(CoveragePlan plan)
        {
            EnsureCoverageInitialized();
            if (ReferenceEquals(_uploadedDecoPlan, plan)) return;
            _uploadedDecoPlan = plan;
            UploadCoverage(plan, ref _decoStampBuffer, ref _decoStampBufferCapacity, ref _decoSplineBuffer, ref _decoSplineCapacity);

            // The row stamps the pages name, for the spawn (DecoRowsSrv) until the next plan
            if (plan.Rows.Count > 0)
            {
                EnsureUpload<DecoRowGPU>(ref _decoRowBuffer, ref _decoRowCapacity, plan.Rows.Count, 8);
                Upload(_decoRowBuffer, plan.Rows);
            }
            else
            {
                Engine.Device.DeferDispose(_decoRowBuffer);
                _decoRowBuffer = null;
            }
        }

        private void SetDecoConstants(int k, CoveragePlan plan, Terrain terrain, uint terrainPagesIdx, uint outputUav, int texels)
        {
            _decoCS.SetBuffer(k, "StampBuf", _decoStampBuffer);
            _decoCS.SetPushConstant(k, "ControlUAV", outputUav);
            _decoCS.SetPushConstant(k, "DecoratorCount", (uint)plan.TargetCount);
            _decoCS.SetPushConstant(k, "Resolution", (uint)texels);
            _decoCS.SetPushConstant(k, "TerrainPages", terrainPagesIdx);
            _decoCS.SetPushConstant(k, "SplineBuf", _decoSplineBuffer.SrvIndex);
            _decoCS.SetPushConstant(k, "LayerCount", (uint)plan.LayerCount);
            _decoCS.SetParam(k, "MaxHeight", terrain.MaxHeight);
            _decoCS.SetParam(k, "TerrainSizeX", terrain.TerrainSize.X);
            _decoCS.SetPushConstant(k, "StampCount", (uint)plan.StampCount);
        }

        /// <summary>
        /// Every level 0 decoration page resident and settled in the pool against the whole-map prepass there was before 5c
        /// (CSBuildDecoControl, Y-flipped), baked over the same texels: both slices value for value. Returns the texels and
        /// pages that differ, the texels compared and the pages checked. Blocks. Main thread.
        /// </summary>
        public unsafe (long texels, int pages, long compared, int checkedPages) CheckDecoPages(CoveragePlan plan, Terrain terrain,
            uint terrainPagesIdx, Graphics.DecoPool pool)
        {
            if (pool.Control == null) return (-1, -1, 0, 0);
            UploadDecoPlan(plan);

            const int texels = Graphics.DecoPool.PageTexels;
            int res = pool.Texels, across = pool.Slots.PagesAcross;
            const int rowPitch = texels * 8;   // RGBA16: 2048, 256-aligned
            const int sliceBytes = rowPitch * texels;
            var device = Engine.Device;
            var flat = device.CreateTexture2D(Format.R16G16B16A16_UInt, texels, texels, 2, 1, ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
            uint flatUav = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(flat, null, new UnorderedAccessViewDescription
            {
                Format = Format.R16G16B16A16_UInt,
                ViewDimension = UnorderedAccessViewDimension.Texture2DArray,
                Texture2DArray = new Texture2DArrayUnorderedAccessView { MipSlice = 0, FirstArraySlice = 0, ArraySize = 2 },
            }, device.GetCpuHandle(flatUav));
            var readback = device.NativeDevice.CreateCommittedResource(new HeapProperties(HeapType.Readback), HeapFlags.None,
                ResourceDescription.Buffer((ulong)sliceBytes * 4), ResourceStates.CopyDest, null);
            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            long differ = 0, compared = 0;
            int differPages = 0, checkedPages = 0;
            try
            {
                for (int pz = 0; pz < across; pz++)
                    for (int px = 0; px < across; px++)
                    {
                        int x0 = px * texels, j0 = pz * texels;
                        int w = Math.Min(texels, res - x0), h = Math.Min(texels, res - j0);
                        if (w <= 0 || h <= 0) continue;
                        int slot = pool.Slots.SlotOf(0, px, pz);
                        if (slot < 0 || !pool.Slots.Settled(slot)) continue;   // not resident at level 0, or its bake not run
                        checkedPages++;
                        int y0 = res - j0 - h;   // the flat bake's rows of these texel rows (flipped)
                        var (sx, sy, slice) = pool.SlotPlace(slot);

                        var cmd = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, allocator, null);
                        cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
                        cmd.SetComputeRootSignature(device.GlobalRootSignature);
                        cmd.ResourceBarrierTransition(flat, ResourceStates.Common, ResourceStates.UnorderedAccess);
                        var k = _kernelDecoControl;
                        SetDecoConstants(k, plan, terrain, terrainPagesIdx, flatUav, res);
                        _decoCS.SetPushConstant(k, "RegionX", (uint)x0);
                        _decoCS.SetPushConstant(k, "RegionY", (uint)y0);
                        _decoCS.SetPushConstant(k, "RegionW", (uint)w);
                        _decoCS.SetPushConstant(k, "RegionH", (uint)h);
                        _decoCS.Dispatch(k, cmd, (uint)((w + 7) / 8), (uint)((h + 7) / 8));
                        cmd.ResourceBarrierTransition(flat, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                        cmd.ResourceBarrierTransition(pool.Control, ResourceStates.Common, ResourceStates.CopySource);
                        for (int s = 0; s < 2; s++)
                        {
                            cmd.CopyTextureRegion(new TextureCopyLocation(readback, new PlacedSubresourceFootPrint
                            {
                                Offset = (ulong)((long)s * sliceBytes),
                                Footprint = new SubresourceFootPrint(Format.R16G16B16A16_UInt, texels, texels, 1, rowPitch)
                            }), 0, 0, 0, new TextureCopyLocation(flat, (uint)s), new Vortice.Mathematics.Box(0, 0, 0, w, h, 1));
                            int bx = sx + Graphics.DecoPool.Border, by = sy + Graphics.DecoPool.Border;
                            cmd.CopyTextureRegion(new TextureCopyLocation(readback, new PlacedSubresourceFootPrint
                            {
                                Offset = (ulong)((long)(2 + s) * sliceBytes),
                                Footprint = new SubresourceFootPrint(Format.R16G16B16A16_UInt, texels, texels, 1, rowPitch)
                            }), 0, 0, 0, new TextureCopyLocation(pool.Control, (uint)(slice + s)),
                                new Vortice.Mathematics.Box(bx, by, 0, bx + w, by + h, 1));
                        }
                        cmd.ResourceBarrierTransition(flat, ResourceStates.CopySource, ResourceStates.Common);
                        cmd.ResourceBarrierTransition(pool.Control, ResourceStates.CopySource, ResourceStates.Common);
                        cmd.Close();
                        device.SubmitAndWait(cmd);
                        cmd.Dispose();
                        allocator.Reset();

                        void* data;
                        readback.Map(0, null, &data);
                        var bytes = (byte*)data;
                        long before = differ;
                        for (int ty = 0; ty < h; ty++)
                        {
                            // Flat region row ty holds texel row j0 + h - 1 - ty: the page's local row h - 1 - ty
                            for (int tx = 0; tx < w; tx++)
                            {
                                bool same = true;
                                for (int s = 0; s < 2 && same; s++)
                                {
                                    ulong* f = (ulong*)(bytes + (long)s * sliceBytes + (long)ty * rowPitch) + tx;
                                    ulong* p = (ulong*)(bytes + (long)(2 + s) * sliceBytes + (long)(h - 1 - ty) * rowPitch) + tx;
                                    if (*f != *p) same = false;
                                }
                                if (!same) differ++;
                                compared++;
                            }
                        }
                        readback.Unmap(0);
                        if (differ > before) differPages++;
                    }
                return (differ, differPages, compared, checkedPages);
            }
            finally
            {
                allocator.Dispose();
                readback.Dispose();
                device.DeferDispose(flat);
                device.DeferReleaseBindlessIndex(flatUav);
            }
        }

        private void UploadCoverage(CoveragePlan plan,
            ref GraphicsBuffer stampBuffer, ref int stampCapacity,
            ref GraphicsBuffer splineBuffer, ref int splineCapacity)
        {
            // Both buffers always exist so the kernels have something valid to bind, even with no stamps
            EnsureUpload<CoverageStampGPU>(ref stampBuffer, ref stampCapacity, plan.Stamps.Count, 16);
            EnsureUpload<StampSplinePointGPU>(ref splineBuffer, ref splineCapacity, plan.SplinePoints.Count, 128);
            Upload(stampBuffer, plan.Stamps);
            Upload(splineBuffer, plan.SplinePoints);
        }

        // ═══════════════════════════════════════════════════════════════════
        // ── Holes (HoleStamp) ──
        // ═══════════════════════════════════════════════════════════════════
        //
        // Baked on the CPU, not in a compute pass: the PhysX height field, the navmesh and PCG all need the mask
        // on the CPU, and hole stamps are few and small, so only the samples under each stamp are evaluated. The
        // shape is EvaluateStampWeight's (terrain_stamp_common.hlsli), ported below, edge noise included.

        /// <summary>One hole stamp's shape in terrain UV, and the mask samples it can reach.</summary>
        internal struct HoleShape
        {
            public Vector2 Center;
            public float Radius, Falloff;
            public int PointStart, PointCount;   // PointCount 0 = radial
            public bool Closed;
            public float NoiseFreq, NoiseAmp;
            public uint NoiseSeed;
            public HoleRect Rect;
        }

        /// <summary>Mask samples x0..x1, z0..z1 (inclusive).</summary>
        public readonly record struct HoleRect(int X0, int Z0, int X1, int Z1)
        {
            public bool Overlaps(HoleRect o) => X0 <= o.X1 && X1 >= o.X0 && Z0 <= o.Z1 && Z1 >= o.Z0;
        }

        /// <summary>The hole stamps of a terrain, captured on the main thread.</summary>
        public sealed class HolePlan
        {
            internal readonly List<HoleShape> Shapes = new();
            internal readonly List<StampSplinePointGPU> Points = new();
            public int Resolution;

            public bool HasHoles => Shapes.Count > 0;

            /// <summary>The samples each stamp can reach: what a later bake has to clear again.</summary>
            public IEnumerable<HoleRect> Rects
            {
                get { foreach (var shape in Shapes) yield return shape.Rect; }
            }
        }

        /// <summary>Collect the hole stamps that cut a terrain. Global ones would remove all of it and are skipped.</summary>
        public static HolePlan PrepareHoles(Terrain terrain, TerrainRenderer renderer)
        {
            var plan = new HolePlan { Resolution = terrain.EffectiveHeightmapResolution };
            var origin = renderer.Transform?.WorldPosition ?? Vector3.Zero;
            var size = terrain.TerrainSize;
            float uvScale = 1f / Math.Max(size.X, size.Y);
            int last = plan.Resolution - 1;

            foreach (var stamp in CollectStamps<HoleStamp>(renderer))
            {
                if (stamp.IsGlobal) continue;

                var shape = new HoleShape { Falloff = Math.Max(stamp.Falloff, 0f) * uvScale };
                if (stamp.EnableNoise && stamp.NoiseAmplitude > 0f)
                {
                    shape.NoiseFreq = stamp.NoiseFrequency;
                    shape.NoiseAmp = stamp.NoiseAmplitude * uvScale;
                    shape.NoiseSeed = (uint)stamp.NoiseSeed;
                }

                // How far the shape reaches in UV, around its centre or around every spline sample
                Vector2 min, max;
                if (stamp.IsSplineMode)
                {
                    shape.PointStart = plan.Points.Count;
                    if (!BuildSplinePoints(stamp, origin, size, terrain.MaxHeight, 0f, plan.Points,
                            out _, out uint count, out shape.Radius))
                        continue;
                    shape.Closed = (count & 0x80000000) != 0;
                    shape.PointCount = (int)(count & 0x7FFFFFFF);

                    min = new Vector2(float.MaxValue);
                    max = new Vector2(float.MinValue);
                    for (int i = shape.PointStart; i < shape.PointStart + shape.PointCount; i++)
                    {
                        var p = plan.Points[i];
                        float falloff = shape.Radius > 1e-7f ? shape.Falloff * p.HalfWidth / shape.Radius : shape.Falloff;
                        float reach = p.HalfWidth + falloff + shape.NoiseAmp;
                        min = Vector2.Min(min, p.UV - new Vector2(reach));
                        max = Vector2.Max(max, p.UV + new Vector2(reach));
                    }
                }
                else
                {
                    var pos = stamp.Transform?.WorldPosition ?? Vector3.Zero;
                    shape.Center = new Vector2((pos.X - origin.X) / size.X, (pos.Z - origin.Z) / size.Y);
                    shape.Radius = Math.Max(stamp.Radius, 0f) * uvScale;
                    float reach = shape.Radius + shape.Falloff + shape.NoiseAmp;
                    min = shape.Center - new Vector2(reach);
                    max = shape.Center + new Vector2(reach);
                }

                // Sample (x, z) lies at UV (x, z) / last
                int x0 = Math.Max(0, (int)MathF.Floor(min.X * last)), z0 = Math.Max(0, (int)MathF.Floor(min.Y * last));
                int x1 = Math.Min(last, (int)MathF.Ceiling(max.X * last)), z1 = Math.Min(last, (int)MathF.Ceiling(max.Y * last));
                if (x0 > x1 || z0 > z1) continue;   // off the terrain
                shape.Rect = new HoleRect(x0, z0, x1, z1);
                plan.Shapes.Add(shape);
            }

            return plan;
        }

        /// <summary>
        /// The store with its holes cut again: the given rects cleared, then every stamp of the plan cut in (the union
        /// of the stamps, by the largest weight). Rects a stamp covered at the last bake must be among 'clear', or it
        /// stays cut there. Page by page (TerrainPages): only the pages the rects and the stamps reach are made anew, a
        /// sample on a shared edge is cut on both its pages alike, and a page left without a hole holds none. The store
        /// passed in is left as it is; it must be of the plan's resolution.
        /// </summary>
        public static TerrainPages BakeHoles(HolePlan plan, TerrainPages pages, IEnumerable<HoleRect> clear)
        {
            int res = plan.Resolution;
            if (pages.Resolution != res)
                throw new ArgumentException($"holes of {res}² into a store of {pages.Resolution}²");
            var clears = clear as IList<HoleRect> ?? clear.ToList();

            // The pages to make anew: those any rect to clear or any stamp reaches
            int across = pages.PagesAcross;
            var touched = new SortedSet<int>();
            void Touch(HoleRect r)
            {
                var (px0, pz0, px1, pz1) = pages.PagesOf(r.X0, r.Z0, r.X1 + 1, r.Z1 + 1);
                for (int pz = pz0; pz < pz1; pz++)
                    for (int px = px0; px < px1; px++)
                        touched.Add(pz * across + px);
            }
            foreach (var r in clears) Touch(r);
            foreach (var shape in plan.Shapes) Touch(shape.Rect);

            float inv = 1f / (res - 1);
            var points = CollectionsMarshal.AsSpan(plan.Points);
            var edit = pages.Edit();
            foreach (int index in touched)
            {
                int px = index % across, pz = index / across;
                var (sx, sz, w, h) = pages.PageExtent(px, pz);
                var pageRect = new HoleRect(sx, sz, sx + w - 1, sz + h - 1);
                var page = edit.Holes(px, pz) is { } old ? (byte[])old.Clone() : new byte[TerrainPages.PageLength];

                foreach (var r in clears)
                {
                    if (!r.Overlaps(pageRect)) continue;
                    int ax = Math.Max(r.X0, sx), bx = Math.Min(r.X1, pageRect.X1);
                    for (int z = Math.Max(r.Z0, sz); z <= Math.Min(r.Z1, pageRect.Z1); z++)
                        Array.Clear(page, (z - sz) * TerrainPages.PageSamples + ax - sx, bx - ax + 1);
                }

                foreach (var shape in plan.Shapes)
                {
                    var r = shape.Rect;
                    if (!r.Overlaps(pageRect)) continue;
                    int ax = Math.Max(r.X0, sx), bx = Math.Min(r.X1, pageRect.X1);
                    for (int z = Math.Max(r.Z0, sz); z <= Math.Min(r.Z1, pageRect.Z1); z++)
                    {
                        int row = (z - sz) * TerrainPages.PageSamples - sx;
                        for (int x = ax; x <= bx; x++)
                        {
                            float weight = HoleWeight(shape, points, new Vector2(x * inv, z * inv));
                            if (weight <= 0f) continue;
                            byte b = (byte)MathF.Round(weight * 255f);
                            if (b > page[row + x]) page[row + x] = b;
                        }
                    }
                }

                edit.SetHoles(px, pz, page.AsSpan().IndexOfAnyExcept((byte)0) >= 0 ? page : null);
            }
            return edit.Done();
        }

        /// <summary>EvaluateStampWeight (terrain_stamp_common.hlsli) on the CPU.</summary>
        private static float HoleWeight(in HoleShape s, ReadOnlySpan<StampSplinePointGPU> points, Vector2 uv)
        {
            float dist, radius = s.Radius, falloff = s.Falloff;

            if (s.PointCount > 0)
            {
                int n = s.PointCount;
                int segments = s.Closed ? n : n - 1;
                float best = float.MaxValue, bestHalfWidth = radius;
                bool inside = false;

                for (int i = 0; i < segments; i++)
                {
                    var p0 = points[s.PointStart + i];
                    var p1 = points[s.PointStart + (i + 1) % n];

                    var ab = p1.UV - p0.UV;
                    float lenSq = ab.LengthSquared();
                    float t = lenSq < 1e-10f ? 0f : Math.Clamp(Vector2.Dot(uv - p0.UV, ab) / lenSq, 0f, 1f);
                    float d = Vector2.Distance(uv, p0.UV + ab * t);
                    if (d < best)
                    {
                        best = d;
                        bestHalfWidth = p0.HalfWidth + (p1.HalfWidth - p0.HalfWidth) * t;
                    }

                    if (s.Closed)
                    {
                        var a = p0.UV;
                        var b = p1.UV;
                        if ((a.Y <= uv.Y && b.Y > uv.Y) || (b.Y <= uv.Y && a.Y > uv.Y))
                        {
                            float xHit = a.X + (uv.Y - a.Y) / (b.Y - a.Y) * (b.X - a.X);
                            if (uv.X < xHit) inside = !inside;
                        }
                    }
                }

                dist = s.Closed && inside ? 0f : best;
                if (radius > 1e-7f) falloff *= bestHalfWidth / radius;
                radius = bestHalfWidth;
            }
            else
            {
                dist = Vector2.Distance(uv, s.Center);
            }

            float total = radius + falloff;
            if (dist >= total + (s.NoiseFreq > 0f ? s.NoiseAmp : 0f)) return 0f;

            if (s.NoiseFreq > 0f && s.NoiseAmp > 0f)
            {
                float scale = s.NoiseFreq / MathF.Max(total, 0.001f);
                var coord = uv * scale + new Vector2(s.NoiseSeed * 1.37f, s.NoiseSeed * 0.73f);
                dist = MathF.Max(dist + StampNoise(coord) * s.NoiseAmp, 0f);
            }

            if (dist >= total) return 0f;
            if (dist <= radius) return 1f;
            float k = (dist - radius) / MathF.Max(falloff, 0.001f);
            return 1f - k * k * (3f - 2f * k);
        }

        /// <summary>stamp_noise (terrain_stamp_common.hlsli): gradient noise in -1..1.</summary>
        private static float StampNoise(Vector2 p)
        {
            var i = new Vector2(MathF.Floor(p.X), MathF.Floor(p.Y));
            var f = p - i;
            var u = f * f * (new Vector2(3f) - 2f * f);

            float n00 = Vector2.Dot(StampHash(i), f);
            float n10 = Vector2.Dot(StampHash(i + new Vector2(1, 0)), f - new Vector2(1, 0));
            float n01 = Vector2.Dot(StampHash(i + new Vector2(0, 1)), f - new Vector2(0, 1));
            float n11 = Vector2.Dot(StampHash(i + new Vector2(1, 1)), f - new Vector2(1, 1));

            float nx0 = n00 + (n10 - n00) * u.X;
            float nx1 = n01 + (n11 - n01) * u.X;
            return nx0 + (nx1 - nx0) * u.Y;
        }

        private static Vector2 StampHash(Vector2 p)
        {
            float a = Vector2.Dot(p, new Vector2(127.1f, 311.7f));
            float b = Vector2.Dot(p, new Vector2(269.5f, 183.3f));
            static float Frac(float v) => v - MathF.Floor(v);
            return new Vector2(-1f + 2f * Frac(MathF.Sin(a) * 43758.5453123f), -1f + 2f * Frac(MathF.Sin(b) * 43758.5453123f));
        }

        // ── Shared helpers ─────────────────────────────────────────────

        /// <summary>
        /// Sample a stamp's spline into terrain UV points for the GPU. Returns false when the spline
        /// has fewer than two points.
        /// </summary>
        private static bool BuildSplinePoints(
            TerrainStamp stamp, Vector3 terrainOrigin, Vector2 terrainSize, float maxHeight, float heightOffset,
            List<StampSplinePointGPU> splinePoints,
            out uint splinePointOffset, out uint splinePointCount, out float radius, bool useWorldY = true)
            => BuildSplinePoints(stamp.GetSpline(), stamp.Radius, terrainOrigin, terrainSize, maxHeight, heightOffset,
                splinePoints, out splinePointOffset, out splinePointCount, out radius, useWorldY, stamp.StartFade, stamp.EndFade);

        /// <summary>
        /// Sample a spline into terrain UV points; each carries <paramref name="worldRadius"/> × its width, and on an
        /// open spline how far the stamp has faded in from its start and out towards its end (metres along it, flat
        /// in XZ; 1 where neither fade reaches).
        /// </summary>
        private static bool BuildSplinePoints(
            Spline? spline, float worldRadius, Vector3 terrainOrigin, Vector2 terrainSize, float maxHeight, float heightOffset,
            List<StampSplinePointGPU> splinePoints,
            out uint splinePointOffset, out uint splinePointCount, out float radius, bool useWorldY = true,
            float startFade = 0f, float endFade = 0f)
        {
            splinePointOffset = 0xFFFFFFFF;
            splinePointCount = 0;
            radius = 0;

            if (spline == null || spline.Points.Count < 2)
                return false;

            splinePointOffset = (uint)splinePoints.Count;
            float uvRadius = worldRadius / Math.Max(terrainSize.X, terrainSize.Y);

            int sampleCount = Math.Max(spline.TotalSegments, spline.Points.Count * 4);
            int first = splinePoints.Count;
            float along = 0f;
            var previous = Vector3.Zero;
            for (int i = 0; i <= sampleCount; i++)
            {
                float t = (float)i / sampleCount;
                var worldPos = spline.GetWorldPoint(t);
                if (i > 0) along += MathF.Sqrt((worldPos.X - previous.X) * (worldPos.X - previous.X) + (worldPos.Z - previous.Z) * (worldPos.Z - previous.Z));
                previous = worldPos;

                splinePoints.Add(new StampSplinePointGPU
                {
                    UV = new Vector2(
                        (worldPos.X - terrainOrigin.X) / terrainSize.X,
                        (worldPos.Z - terrainOrigin.Z) / terrainSize.Y),
                    Height = ((useWorldY ? worldPos.Y - terrainOrigin.Y : 0f) + heightOffset) / maxHeight,
                    HalfWidth = uvRadius * spline.GetWidth(t),
                    Fade = along,       // the distance along it for now, the fade below
                });
            }

            float length = along;
            for (int i = first; i < splinePoints.Count; i++)
            {
                var point = splinePoints[i];
                float fade = 1f;
                if (!spline.Closed)
                {
                    if (startFade > 0f) fade *= SmoothStep01(point.Fade / startFade);
                    if (endFade > 0f) fade *= SmoothStep01((length - point.Fade) / endFade);
                }
                point.Fade = fade;
                splinePoints[i] = point;
            }

            uint ptCount = (uint)(sampleCount + 1);
            if (spline.Closed) ptCount |= 0x80000000;

            splinePointCount = ptCount;
            // The stamp's own radius stays the reference: the shader scales the falloff by
            // (sample half-width / Radius), see EvaluateStampWeight.
            radius = uvRadius;
            return true;
        }

        private static float SmoothStep01(float x)
        {
            x = Math.Clamp(x, 0f, 1f);
            return x * x * (3f - 2f * x);
        }

        private static void EnsureUpload<T>(ref GraphicsBuffer buffer, ref int capacity, int count, int minimum)
            where T : unmanaged
        {
            // A bake's dispatches run on the GPU frames after they are recorded, and the next bake can be
            // recorded before that (a stamp being dragged rebakes every frame). Every bake gets its own
            // buffer and the previous one lives until its frames are done: rewriting or freeing it here
            // fed the earlier bake's shader a stamp list that was changing under it (GPU hang).
            Engine.Device.DeferDispose(buffer);
            capacity = Math.Max(count, minimum);
            buffer = GraphicsBuffer.CreateUpload<T>(capacity, mapped: true);
        }

        private static unsafe void Upload<T>(GraphicsBuffer buffer, List<T> items) where T : unmanaged
        {
            if (items.Count == 0) return;
            var dst = buffer.WritePtr<T>();
            var src = CollectionsMarshal.AsSpan(items);
            for (int i = 0; i < src.Length; i++)
                dst[i] = src[i];
        }

        /// <summary>
        /// Release all per-instance GPU resources.
        /// Static compute shaders and the noise LUT are shared and live for the process lifetime.
        /// </summary>
        public void Dispose()
        {
            // Only dispose internal scratch buffers. The height texture is an output owned by the
            // Terrain asset — it persists in the AssetManager cache.
            _heightStampBuffer?.Dispose();
            _heightStampSplineBuffer?.Dispose();
            _heightPageListBuffer?.Dispose();
            _heightStepBuffer?.Dispose();
            // The pool is the terrain asset's: it stays (Terrain.HeightPool)
            _splatStampBuffer?.Dispose();
            _splatSplineBuffer?.Dispose();
            _decoStampBuffer?.Dispose();
            _decoSplineBuffer?.Dispose();
            _decoRowBuffer?.Dispose();
        }
    }
}
