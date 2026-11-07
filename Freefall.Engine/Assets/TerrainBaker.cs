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
        private static int _kernelSplatBake;
        private static ComputeShader _decoCS;
        private static int _kernelDecoControl;
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
            _kernelSplatBake = _splatCS.FindKernel("CS_SplatBake");
            _decoCS = new ComputeShader("decoration_prepass.hlsl");
            _kernelDecoControl = _decoCS.FindKernel("CSBuildDecoControl");
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

            public HeightStepGPU ToGpu() => new()
            {
                BlendMode = BlendMode, Opacity = Opacity,
                Source = SourceSrv, ImportScale = Scale, ImportBias = Bias, ImportMin = ImportMin, ImportInvSize = ImportInvSize,
                NoiseType = NoiseType, Octaves = Octaves, Frequency = Frequency, Amplitude = Amplitude,
                Lacunarity = Lacunarity, Persistence = Persistence, Offset = Offset, Seed = Seed,
                TerraceSteps = TerraceSteps, TerraceSmoothness = TerraceSmoothness,
                MaskCenter = MaskCenter, MaskRadius = MaskRadius, MaskFalloff = MaskFalloff,
            };
        }

        /// <summary>A step's parameters for the GPU: terrain_height_bake.hlsl BakeStep, 96 bytes.</summary>
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
                        };
                        if (!noise.IsGlobal)
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

            bool fresh = _pool.Ensure(plan.Resolution, cmd);
            // Another grid or height range than the last bake's moves every sample, whatever region the caller saw
            var grid = (plan.Resolution, plan.Origin, plan.Spacing, plan.MaxHeight);
            bool gridChanged = _bakedGrid is { } last && last != grid;
            _bakedGrid = grid;
            var box = fresh || gridChanged || pages == null ? (0, 0, plan.PagesAcross, plan.PagesAcross) : pages.Value;

            UploadHeightPlan(plan);

            // The pool rests in Common between bakes, and a texture is never promoted to UnorderedAccess implicitly (only
            // to the read and copy states); back to Common, so its readers promote it to SRV implicitly
            cmd.ResourceBarrierTransition(_pool.Texture, ResourceStates.Common, ResourceStates.UnorderedAccess);
            RecordHeight(plan, cmd, box, wholeRuns: false);
            cmd.ResourceBarrierTransition(_pool.Texture, ResourceStates.UnorderedAccess, ResourceStates.Common);
            _pool.Filled = true;
            HeightVersion++;
        }

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

        /// <summary>
        /// Record a plan's bake over a box of pages into the pool, which is in UnorderedAccess: each page cleared, then
        /// every step over each page, border included. An influence run takes the page's list, or, with
        /// <paramref name="wholeRuns"/>, every descriptor of the run (CheckPageBake). Pages write texels of their own, so
        /// there is a barrier between steps only.
        /// </summary>
        private void RecordHeight(HeightPlan plan, ID3D12GraphicsCommandList cmd,
            (int px0, int py0, int px1, int py1) pages, bool wholeRuns)
        {
            if (pages.px1 <= pages.px0 || pages.py1 <= pages.py0) return;

            // Where the samples lie, the same for every kernel
            _cs.SetPushConstant("Output", _pool.Uav);
            _cs.SetPushConstant("GridSamples", (uint)plan.Resolution);
            _cs.SetParam("OriginX", plan.Origin.X);
            _cs.SetParam("OriginZ", plan.Origin.Y);
            _cs.SetParam("Spacing", plan.Spacing);
            _cs.SetBuffer("StepBuf", _heightStepBuffer);

            _cs.SetParam(_kernelClear, "ClearValue", 0f);
            for (int py = pages.py0; py < pages.py1; py++)
                for (int px = pages.px0; px < pages.px1; px++)
                    DispatchPage(_kernelClear, cmd, plan, px, py);
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
                    }
                    _cs.SetPushConstant(k, "StepIndex", (uint)s);
                    for (int py = pages.py0; py < pages.py1; py++)
                        for (int px = pages.px0; px < pages.px1; px++)
                            DispatchPage(k, cmd, plan, px, py);
                }
                cmd.ResourceBarrierUnorderedAccessView(_pool.Texture);
            }
        }

        /// <summary>One dispatch of a kernel over page (px, py) with its border, written at the page's slot.</summary>
        private void DispatchPage(int k, ID3D12GraphicsCommandList cmd, HeightPlan plan, int px, int py)
        {
            var box = plan.PageBox(px, py);
            var (slice, cx, cy) = _pool.PageCorner(px, py);
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
            (int px0, int py0, int px1, int py1) pages, bool wholeRuns)
        {
            var k = _kernelInfluenceLayer;
            _cs.SetBuffer(k, "StampBuf", _heightStampBuffer);
            _cs.SetBuffer(k, "StampList", _heightPageListBuffer);
            _cs.SetPushConstant(k, "SplineBuf",
                plan.SplinePoints.Count > 0 && _heightStampSplineBuffer != null ? _heightStampSplineBuffer.SrvIndex : 0u);

            for (int py = pages.py0; py < pages.py1; py++)
                for (int px = pages.px0; px < pages.px1; px++)
                {
                    var (start, count) = wholeRuns
                        ? (plan.AllStamps + step.Start, step.Count)
                        : plan.PageRuns[step.PageRuns + py * plan.PagesAcross + px];
                    if (count == 0) continue;
                    _cs.SetPushConstant(k, "ListStart", (uint)start);
                    _cs.SetPushConstant(k, "ListCount", (uint)count);
                    DispatchPage(k, cmd, plan, px, py);
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
            EnsureInitialized();
            EnsureNoiseLUT();
            var device = Engine.Device;
            int res = plan.Resolution;
            int pagesAcross = plan.PagesAcross;
            var all = (0, 0, pagesAcross, pagesAcross);

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
                    _pool.Ensure(res, cmd);
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
                        for (int py = 0; py < pagesAcross; py++)
                            for (int px = 0; px < pagesAcross; px++)
                                DispatchPage(_kernelClear, cmd, plan, px, py);
                        cmd.ResourceBarrierUnorderedAccessView(_pool.Texture);
                        var order = Enumerable.Range(0, pagesAcross * pagesAcross).ToArray();
                        new Random(7).Shuffle(order);
                        foreach (int p in order)
                        {
                            int px = p % pagesAcross, py = p / pagesAcross;
                            RecordHeight(plan, cmd, (px, py, px + 1, py + 1), wholeRuns: false);
                        }
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
                                    // saved map (UploadBakedHeightmap) leaves the border beyond the edge unwritten, as
                                    // nothing reads it
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
        /// Uploads pre-saved R16_UNorm baked heightmap bytes (the flat map Save writes) into the pool, page by page, each
        /// with the samples of its border that lie on the terrain (the border beyond the terrain's edge is never read:
        /// terrain_pages.hlsli clamps to the edge). Used at load time when the baked heightmap was persisted to cache.
        /// </summary>
        public bool UploadBakedHeightmap(byte[] pixels, int resolution)
        {
            if (pixels == null || pixels.Length == 0) return false;

            var device = Engine.Device;
            int bytesPerPixel = 2; // R16_UNorm
            int rowPitch = (resolution * bytesPerPixel + 255) & ~255;
            int totalBytes = rowPitch * resolution;

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

                    int srcRowBytes = resolution * bytesPerPixel;
                    var dstPtr = (byte*)pData;
                    for (int y = 0; y < resolution; y++)
                        Marshal.Copy(pixels, y * srcRowBytes, (IntPtr)(dstPtr + y * rowPitch), srcRowBytes);

                    uploadResource.Unmap(0);
                }

                var src = new TextureCopyLocation(uploadResource, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R16_UNorm, (uint)resolution, (uint)resolution, 1, (uint)rowPitch)
                });

                _pool.Ensure(resolution, cmdList);
                cmdList.ResourceBarrierTransition(_pool.Texture, ResourceStates.Common, ResourceStates.CopyDest);
                for (int py = 0; py < _pool.PagesAcross; py++)
                    for (int px = 0; px < _pool.PagesAcross; px++)
                    {
                        int bx = px * Terrain.PageCells - HeightPool.Border, by = py * Terrain.PageCells - HeightPool.Border;
                        int x0 = Math.Max(bx, 0), y0 = Math.Max(by, 0);
                        int x1 = Math.Min(bx + HeightPool.PageTexels, resolution), y1 = Math.Min(by + HeightPool.PageTexels, resolution);
                        if (x1 <= x0 || y1 <= y0) continue;
                        var (slice, cx, cy) = _pool.PageCorner(px, py);
                        cmdList.CopyTextureRegion(new TextureCopyLocation(_pool.Texture, (uint)slice),
                            (uint)(cx + x0 - bx), (uint)(cy + y0 - by), 0, src,
                            new Vortice.Mathematics.Box(x0, y0, 0, x1, y1, 1));
                    }
                cmdList.ResourceBarrierTransition(_pool.Texture, ResourceStates.CopyDest, ResourceStates.Common);

                cmdList.Close();
                device.SubmitAndWait(cmdList);

                Debug.Log($"[TerrainBaker] Uploaded baked heightmap from cache: {resolution}x{resolution} ({pixels.Length} bytes)");
                _pool.Filled = true;
                HeightVersion++;
                return true;
            }
            finally
            {
                cmdList.Dispose();
                allocator.Dispose();
                uploadResource.Dispose();
            }
        }

        /// <summary>
        /// Read a box of samples (x0, z0 inclusive, x1, z1 exclusive) out of the pool into a readback buffer laid out as a
        /// flat map of the box, rows of <paramref name="rowPitch"/> bytes: each page's own samples (no border; the last page
        /// in a row or column also the far edge). Its own command list; blocks until the GPU is done.
        /// </summary>
        private void ReadPoolBox(ID3D12Resource readback, int rowPitch, int x0, int z0, int x1, int z1)
        {
            var device = Engine.Device;
            int res = _pool.Resolution;
            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            var cmdList = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, allocator, null);
            try
            {
                var dst = new TextureCopyLocation(readback, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R16_UNorm, (uint)(x1 - x0), (uint)(z1 - z0), 1, (uint)rowPitch)
                });
                cmdList.ResourceBarrierTransition(_pool.Texture, ResourceStates.Common, ResourceStates.CopySource);
                int pages = _pool.PagesAcross;
                for (int py = 0; py < pages; py++)
                    for (int px = 0; px < pages; px++)
                    {
                        var own = HeightPlan.PageSamples(px, py, pages, res);
                        int ax = Math.Max(own.x0, x0), az = Math.Max(own.y0, z0);
                        int bx = Math.Min(own.x1, x1), bz = Math.Min(own.y1, z1);
                        if (bx <= ax || bz <= az) continue;
                        var (slice, cx, cy) = _pool.PageCorner(px, py);
                        int tx = cx + HeightPool.Border + ax - own.x0, tz = cy + HeightPool.Border + az - own.y0;
                        cmdList.CopyTextureRegion(dst, (uint)(ax - x0), (uint)(az - z0), 0,
                            new TextureCopyLocation(_pool.Texture, (uint)slice),
                            new Vortice.Mathematics.Box(tx, tz, 0, tx + bx - ax, tz + bz - az, 1));
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
            var pageList = new List<(int px, int pz)>((px1 - px0) * (pz1 - pz0));
            for (int pz = pz0; pz < pz1; pz++)
                for (int px = px0; px < px1; px++)
                    pageList.Add((px, pz));

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
                for (int first = 0; first < pageList.Count; first += batch)
                {
                    int count = Math.Min(batch, pageList.Count - first);
                    ReadPoolPages(readbackResource, pageList, first, count, rowPitch, pageBytes, basis);

                    unsafe
                    {
                        void* pData;
                        readbackResource.Map(0, null, &pData);
                        using (_copyMarker.Auto())
                        {
                            for (int k = 0; k < count; k++)
                            {
                                var (px, pz) = pageList[first + k];
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
                          $"{replaced} new, " +
                          (changed is { } box ? $"changed x {box.x0}..{box.x1}, z {box.z0}..{box.z1}" : "nothing changed"));
                return pages;
            }
            finally
            {
                readbackResource.Dispose();
            }
        }

        /// <summary>
        /// Copy whole pages of the pool (each its own samples, no border) into a readback buffer, page k at k × pageBytes,
        /// rows of <paramref name="rowPitch"/> bytes. Its own command list; blocks until the GPU is done.
        /// </summary>
        private void ReadPoolPages(ID3D12Resource readback, List<(int px, int pz)> pages, int first, int count,
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
                    var (px, pz) = pages[first + k];
                    var (_, _, w, h) = layout.PageExtent(px, pz);
                    var (slice, cx, cy) = _pool.PageCorner(px, pz);
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

        /// <summary>
        /// Reads back the baked heights (the pool) as one flat map of raw R16_UNorm bytes, for DDS persistence (Save).
        /// Returns null if no heights have been baked.
        /// </summary>
        public byte[] ReadbackBakedHeightmapBytes()
        {
            if (_pool.Texture == null)
                return null;

            var device = Engine.Device;
            int res = _pool.Resolution;
            int bytesPerPixel = 2; // R16_UNorm
            int rowPitch = (res * bytesPerPixel + 255) & ~255;
            int totalBytes = rowPitch * res;

            var readbackResource = device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Readback),
                HeapFlags.None,
                ResourceDescription.Buffer((ulong)totalBytes),
                ResourceStates.CopyDest,
                null);

            try
            {
                ReadPoolBox(readbackResource, rowPitch, 0, 0, res, res);

                int srcRowBytes = res * bytesPerPixel;
                byte[] pixels = new byte[srcRowBytes * res];
                unsafe
                {
                    void* pData;
                    readbackResource.Map(0, null, &pData);
                    var srcPtr = (byte*)pData;
                    for (int y = 0; y < res; y++)
                        Marshal.Copy((IntPtr)(srcPtr + y * rowPitch), pixels, y * srcRowBytes, srcRowBytes);
                    readbackResource.Unmap(0);
                }

                Debug.Log($"[TerrainBaker] Baked heightmap bytes readback: {res}x{res}, {pixels.Length} bytes");
                return pixels;
            }
            finally
            {
                readbackResource.Dispose();
            }
        }

        /// <summary>Samples a side of the baked heights (the pool), or 0 if none.</summary>
        public int BakedResolution => _pool.Texture != null ? _pool.Resolution : 0;

        /// <summary>Counts the baked heightmaps' changes (a bake, an upload from the cache): what is made from the heights
        /// (Graphics.TerrainLightShadow) is made again when it moves on.</summary>
        public static int HeightVersion { get; private set; }

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
        /// Lay out a splat bake from the terrain's splat stamps (already in bake order).
        /// 'layers' is the terrain's palette: a stamp's Layer becomes the index of its channel.
        /// Main thread.
        /// </summary>
        public CoveragePlan PrepareSplat(Terrain terrain, TerrainRenderer renderer,
            IReadOnlyList<SplatStamp> stamps, IReadOnlyList<TerrainLayer> layers)
        {
            var plan = new CoveragePlan { TargetCount = layers.Count, LayerCount = layers.Count };
            var layerIndex = BuildIndex(layers);
            var terrainOrigin = renderer.Transform?.WorldPosition ?? Vector3.Zero;

            foreach (var stamp in stamps)
            {
                if (stamp.Layer == null || !layerIndex.TryGetValue(stamp.Layer, out int channel)) continue;

                if (!TryBuildCoverage(stamp, terrain, terrainOrigin, layerIndex, plan.SplinePoints, out var desc))
                    continue;

                desc.Flags |= (uint)stamp.Op << 8;
                desc.Target = (uint)channel;
                desc.Strength = stamp.Strength;
                plan.Stamps.Add(desc);
            }

            return plan;
        }

        /// <summary>
        /// Lay out a decoration coverage bake from the terrain's deco stamps (already in bake order).
        /// 'decorators' is the terrain's decorator palette; 'layers' its layer palette (for filters).
        /// Main thread.
        /// </summary>
        public CoveragePlan PrepareDeco(Terrain terrain, TerrainRenderer renderer,
            IReadOnlyList<DecoStamp> stamps, IReadOnlyList<TerrainDecorator> decorators,
            IReadOnlyList<TerrainLayer> layers)
        {
            var plan = new CoveragePlan { TargetCount = decorators.Count, LayerCount = layers.Count };
            var layerIndex = BuildIndex(layers);
            var decoratorIndex = BuildIndex(decorators);
            var terrainOrigin = renderer.Transform?.WorldPosition ?? Vector3.Zero;

            foreach (var stamp in stamps)
            {
                uint target;
                if (stamp.Decorator == null)
                {
                    // No decorator: a Multiply stamp thins or boosts everything; an Add stamp has nothing to add
                    if (stamp.Op != DecoOp.Multiply) continue;
                    target = AllTargets;
                }
                else if (decoratorIndex.TryGetValue(stamp.Decorator, out int slot))
                {
                    target = (uint)slot;
                }
                else continue;

                if (!TryBuildCoverage(stamp, terrain, terrainOrigin, layerIndex, plan.SplinePoints, out var desc))
                    continue;

                desc.Flags |= (uint)stamp.Op << 8;
                desc.Target = target;
                desc.Strength = stamp.Weight;
                if (stamp.Op == DecoOp.Add && stamp.RowSpacing > 0.05f && plan.Rows.Count < MaxDecoRows)
                {
                    plan.Rows.Add(new DecoRowGPU
                    {
                        Normal = RowNormal(stamp),
                        Spacing = stamp.RowSpacing,
                        Width = Math.Clamp(stamp.RowWidth, 0f, stamp.RowSpacing),
                        Slot = target,
                    });
                    desc.Rows = (uint)plan.Rows.Count;
                }
                plan.Stamps.Add(desc);
            }

            return plan;
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

        /// <summary>Shape, edge noise and filter of a coverage stamp. Op, target and strength are the caller's.</summary>
        private static bool TryBuildCoverage(CoverageStamp stamp, Terrain terrain, Vector3 terrainOrigin,
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
            desc.HeightMin = stamp.HeightRange.X;
            desc.HeightMax = stamp.HeightRange.Y;
            desc.HeightBlend = stamp.HeightBlend;
            desc.SlopeMin = stamp.SlopeRange.X;
            desc.SlopeMax = stamp.SlopeRange.Y;
            desc.SlopeBlend = stamp.SlopeBlend;
            desc.PatchSize = stamp.PatchSize;
            desc.PatchCoverage = stamp.PatchCoverage;
            desc.PatchSoftness = stamp.PatchSoftness;
            desc.PatchSeed = (uint)stamp.PatchSeed;

            if (stamp.HasFilter)
            {
                desc.Flags |= CoverageFilter;
                desc.RequireMask = LayerMask(stamp.RequireLayers, layerIndex, out bool anyRequired);
                desc.ExcludeMask = LayerMask(stamp.ExcludeLayers, layerIndex, out _);

                // Requires layers, but none of them is on this terrain: the stamp can never apply
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
        /// Composite the splat stamps into the packed control array (ceil(layers / 4) RGBA slices).
        /// Clears what an earlier bake left, so a plan without stamps yields an empty array.
        /// 'topLayerUAV' (0 = none) gets the channel that shows most per texel, see ReadbackTopLayers.
        /// 'region' (control texels: x, y, width, height; null = the whole map) is all that is redone: every texel in
        /// it is composited from all the stamps again, the rest keeps what the last bake wrote. Only for a bake into the
        /// same array with the same palette, after stamps changed only inside it.
        /// Render callback.
        /// </summary>
        public void BakeSplat(CoveragePlan plan, ID3D12GraphicsCommandList cmd, Terrain terrain,
            uint terrainPagesIdx, uint controlArrayUAV, uint topLayerUAV, int resolution, int sliceCount,
            (int x, int y, int width, int height)? region = null)
        {
            var r = region ?? (0, 0, resolution, resolution);
            EnsureCoverageInitialized();

            UploadCoverage(plan, ref _splatStampBuffer, ref _splatStampBufferCapacity,
                ref _splatSplineBuffer, ref _splatSplineCapacity);

            var device = Engine.Device;
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            cmd.SetComputeRootSignature(device.GlobalRootSignature);

            var k = _kernelSplatBake;
            _splatCS.SetBuffer(k, "StampBuf", _splatStampBuffer);
            _splatCS.SetPushConstant(k, "Output", controlArrayUAV);
            _splatCS.SetPushConstant(k, "TerrainPages", terrainPagesIdx);
            _splatCS.SetPushConstant(k, "SplineBuf", _splatSplineBuffer.SrvIndex);
            _splatCS.SetPushConstant(k, "StampCount", (uint)plan.StampCount);
            _splatCS.SetPushConstant(k, "Resolution", (uint)resolution);
            _splatCS.SetPushConstant(k, "SliceCount", (uint)sliceCount);
            _splatCS.SetPushConstant(k, "LayerCount", (uint)plan.TargetCount);
            _splatCS.SetParam(k, "MaxHeight", terrain.MaxHeight);
            _splatCS.SetParam(k, "TerrainSizeX", terrain.TerrainSize.X);
            _splatCS.SetPushConstant(k, "TopLayer", topLayerUAV);
            _splatCS.SetPushConstant(k, "RegionX", (uint)r.x);
            _splatCS.SetPushConstant(k, "RegionY", (uint)r.y);
            _splatCS.SetPushConstant(k, "RegionW", (uint)r.width);
            _splatCS.SetPushConstant(k, "RegionH", (uint)r.height);

            _splatCS.Dispatch(k, cmd, (uint)((r.width + 7) / 8), (uint)((r.height + 7) / 8));
        }

        /// <summary>
        /// Reads back the splat bake's top-layer map (R8_UINT, in Common state): one byte per control
        /// texel, row by row, holding the palette channel that shows most there (255 = none).
        /// Blocks until the GPU has caught up, like <see cref="ReadbackHeightmap"/>. Main thread.
        /// </summary>
        public static byte[] ReadbackTopLayers(ID3D12Resource topLayerTex, int res)
        {
            if (topLayerTex == null || res <= 0)
                return null;

            var device = Engine.Device;
            int rowPitch = (res + 255) & ~255; // 256-byte aligned
            int totalBytes = rowPitch * res;

            var readbackResource = device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Readback),
                HeapFlags.None,
                ResourceDescription.Buffer((ulong)totalBytes),
                ResourceStates.CopyDest,
                null);

            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            var cmdList = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(
                0, CommandListType.Direct, allocator, null);

            try
            {
                cmdList.ResourceBarrierTransition(topLayerTex,
                    ResourceStates.Common, ResourceStates.CopySource);

                var src = new TextureCopyLocation(topLayerTex, 0);
                var dst = new TextureCopyLocation(readbackResource, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R8_UInt, (uint)res, (uint)res, 1, (uint)rowPitch)
                });
                cmdList.CopyTextureRegion(dst, 0, 0, 0, src);

                cmdList.ResourceBarrierTransition(topLayerTex,
                    ResourceStates.CopySource, ResourceStates.Common);

                cmdList.Close();
                device.SubmitAndWait(cmdList);

                var layers = new byte[res * res];
                unsafe
                {
                    void* pData;
                    readbackResource.Map(0, null, &pData);

                    fixed (byte* dstPtr = layers)
                    {
                        for (int y = 0; y < res; y++)
                            Buffer.MemoryCopy((byte*)pData + (long)y * rowPitch, dstPtr + (long)y * res, res, res);
                    }

                    readbackResource.Unmap(0);
                }

                return layers;
            }
            finally
            {
                cmdList.Dispose();
                allocator.Dispose();
                readbackResource.Dispose();
            }
        }

        /// <summary>
        /// Composite the deco stamps into the decoration control texture (top 8 decorators per texel).
        /// 'controlMapsSrv' is the finished splat result the layer filters read (0 = no layers).
        /// Render callback; the control texture must be in UnorderedAccess state.
        /// </summary>
        public void BakeDecoControl(CoveragePlan plan, ID3D12GraphicsCommandList cmd, Terrain terrain,
            uint terrainPagesIdx, uint controlMapsSrv, uint decoControlUAV, int resolution)
        {
            EnsureCoverageInitialized();

            UploadCoverage(plan, ref _decoStampBuffer, ref _decoStampBufferCapacity,
                ref _decoSplineBuffer, ref _decoSplineCapacity);

            // The row stamps the control texture will name, for the spawn (DecoRowsSrv) until the next bake
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

            var device = Engine.Device;
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            cmd.SetComputeRootSignature(device.GlobalRootSignature);

            var k = _kernelDecoControl;
            _decoCS.SetBuffer(k, "StampBuf", _decoStampBuffer);
            _decoCS.SetPushConstant(k, "ControlUAV", decoControlUAV);
            _decoCS.SetPushConstant(k, "DecoratorCount", (uint)plan.TargetCount);
            _decoCS.SetPushConstant(k, "Resolution", (uint)resolution);
            _decoCS.SetPushConstant(k, "TerrainPages", terrainPagesIdx);
            _decoCS.SetPushConstant(k, "SplineBuf", _decoSplineBuffer.SrvIndex);
            _decoCS.SetPushConstant(k, "LayerCount", (uint)plan.LayerCount);
            _decoCS.SetPushConstant(k, "ControlMaps", controlMapsSrv);
            _decoCS.SetParam(k, "MaxHeight", terrain.MaxHeight);
            _decoCS.SetParam(k, "TerrainSizeX", terrain.TerrainSize.X);
            _decoCS.SetPushConstant(k, "StampCount", (uint)plan.StampCount);

            uint groups = (uint)((resolution + 7) / 8);
            _decoCS.Dispatch(k, cmd, groups, groups);
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
