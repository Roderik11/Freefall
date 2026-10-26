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
    ///   height   — R16_UNorm heightmap                    (HeightStamp, HeightNoiseStamp, HeightErosionStamp)
    ///   splat    — packed layer-weight array              (SplatStamp)
    ///   deco     — decoration control texture             (DecoStamp)
    ///
    /// Each bake is two steps. Prepare* runs on the main thread: it collects the stamps in scope, sorts
    /// them by priority and copies everything the GPU needs into a plan, so the render thread never reads
    /// a component. Bake* runs inside a render callback and only uploads the plan and dispatches.
    ///
    /// A bake always covers the whole terrain. Stamps composite in ascending Priority
    /// (TerrainStamp.CompareBakeOrder).
    /// </summary>
    public class TerrainBaker : IDisposable
    {
        // ── Static: shared compute shaders + noise LUT (initialized once) ──
        private static ComputeShader _cs;
        private static int _kernelClear;
        private static int _kernelImport;
        private static int _kernelNoiseLayer;
        private static int _kernelErosionFilter;
        private static int _kernelInfluenceLayer;
        private static ID3D12Resource _noiseLUTTex;
        private static uint _noiseLUTSRV;
        private static bool _initialized;

        private static ComputeShader _splatCS;
        private static int _kernelSplatBake;
        private static ComputeShader _decoCS;
        private static int _kernelDecoControl;
        private static bool _coverageInitialized;

        // ── Instance: per-terrain GPU resources ──

        // GPU resources for the baked heightmap
        private ID3D12Resource _heightTexture;
        private uint _heightUAV;
        private uint _heightSRV;
        private int _currentResolution;

        // Upload buffers (reused across bakes)
        private GraphicsBuffer _heightStampBuffer;
        private int _heightStampBufferCapacity;
        private GraphicsBuffer _heightStampSplineBuffer;
        private int _heightStampSplineCapacity;

        private GraphicsBuffer _splatStampBuffer;
        private int _splatStampBufferCapacity;
        private GraphicsBuffer _splatSplineBuffer;
        private int _splatSplineCapacity;

        private GraphicsBuffer _decoStampBuffer;
        private int _decoStampBufferCapacity;
        private GraphicsBuffer _decoSplineBuffer;
        private int _decoSplineCapacity;

        /// <summary>Layers a terrain can render: 8 RGBA slices of the packed control array.</summary>
        public const int MaxLayers = 32;

        /// <summary>Decorators a terrain can render: slots of the decoration control texture.</summary>
        public const int MaxDecorators = 32;

        // ── GPU structs ──

        /// <summary>GPU struct matching HLSL HeightStampDescriptor. 104 bytes (26 x uint/float).</summary>
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

        /// <summary>GPU struct matching HLSL CoverageStamp (terrain_coverage.hlsli). 112 bytes.</summary>
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
        }

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
            _kernelErosionFilter = _cs.FindKernel("CS_ErosionFilter");
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

        internal enum HeightStepKind { Influence, Import, Noise, Erosion }

        internal struct HeightStep
        {
            public HeightStepKind Kind;

            // Influence: a run of descriptors
            public int Start, Count;

            // Import / Noise / Erosion
            public uint BlendMode;
            public float Opacity;

            // Import
            public uint SourceSrv;
            public float Scale, Bias;

            // Noise
            public uint NoiseType, Octaves, Seed, TerraceSteps;
            public float Frequency, Amplitude, Lacunarity, Persistence, TerraceSmoothness;
            public Vector2 Offset, MaskCenter;
            public float MaskRadius, MaskFalloff;

            // Erosion
            public float EroScale, EroStrength, EroGullyWeight, EroDetail, EroLacunarity, EroGain, EroCellScale;
            public float EroNormalization, EroRidgeRounding, EroCreaseRounding, EroSlopeOnset;
            public float EroAssumedSlope, EroAssumedSlopeAmount;
            public uint EroOctaves;
        }

        /// <summary>Everything a height bake needs, captured on the main thread.</summary>
        public sealed class HeightPlan
        {
            internal readonly List<HeightStep> Steps = new();
            internal readonly List<HeightStampDescriptorGPU> Descriptors = new();
            internal readonly List<StampSplinePointGPU> SplinePoints = new();
            internal readonly List<string> Sources = new();   // GUIDs of the heightmap textures, in the order the plan uses them
            internal int Resolution;
            internal Vector2 TerrainSize;
            internal float MaxHeight;

            /// <summary>True if any stamp contributes; a bake without work is skipped so a terrain
            /// with no height stamps keeps the heightmap it was saved with.</summary>
            public bool HasWork => Steps.Count > 0;

            /// <summary>
            /// What the bake of this plan comes out as, as a hash: every step, descriptor and spline point (the bindless
            /// indices left out, the textures they name by GUID), the resolution and the terrain's size, and the bake
            /// shader's source. Equal hashes bake equal heightmaps, so a heightmap baked from an equal hash can be kept
            /// (Terrain.BakedStampsHash). A heightmap texture reimported under the same GUID is not seen.
            /// </summary>
            public ulong Hash => _hash ??= ComputeHash();
            private ulong? _hash;

            private ulong ComputeHash()
            {
                ulong h = Fnv.Start;
                h = Fnv.Mix(h, Resolution);
                h = Fnv.Mix(h, TerrainSize.X); h = Fnv.Mix(h, TerrainSize.Y); h = Fnv.Mix(h, MaxHeight);
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
        /// HeightStamps become one dispatch, and every whole-terrain operation (import, noise, erosion)
        /// is a dispatch of its own at its place in the priority order. Main thread.
        /// </summary>
        public HeightPlan PrepareHeight(Terrain terrain, TerrainRenderer renderer)
        {
            var plan = new HeightPlan { Resolution = terrain.EffectiveHeightmapResolution, TerrainSize = terrain.TerrainSize, MaxHeight = terrain.MaxHeight };

            var stamps = new List<TerrainStamp>();
            stamps.AddRange(CollectStamps<HeightStamp>(renderer));
            stamps.AddRange(CollectStamps<HeightNoiseStamp>(renderer));
            stamps.AddRange(CollectStamps<HeightErosionStamp>(renderer));
            stamps.Sort(TerrainStamp.CompareBakeOrder);

            var terrainSize = terrain.TerrainSize;
            float maxHeight = terrain.MaxHeight;
            float uvScale = 1f / Math.Max(terrainSize.X, terrainSize.Y);
            var terrainOrigin = renderer.Transform?.WorldPosition ?? Vector3.Zero;

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
                            Frequency = noise.Frequency,
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
                            step.MaskCenter = new Vector2(
                                (center.X - terrainOrigin.X) / terrainSize.X,
                                (center.Z - terrainOrigin.Z) / terrainSize.Y);
                            step.MaskRadius = Math.Max(extent, 0.01f) * uvScale;
                            step.MaskFalloff = Math.Clamp(2f * extent / Math.Max(noise.Falloff, 0.01f), 1f, 64f);
                        }
                        plan.Steps.Add(step);
                        break;
                    }

                    case HeightErosionStamp erosion:
                        CloseRun();
                        plan.Steps.Add(new HeightStep
                        {
                            Kind = HeightStepKind.Erosion,
                            BlendMode = (uint)erosion.BlendMode,
                            Opacity = erosion.Opacity,
                            EroScale = erosion.Scale,
                            EroStrength = erosion.Strength,
                            EroGullyWeight = erosion.GullyWeight,
                            EroDetail = erosion.Detail,
                            EroLacunarity = erosion.Lacunarity,
                            EroGain = erosion.Gain,
                            EroCellScale = erosion.CellScale,
                            EroOctaves = (uint)Math.Clamp(erosion.Octaves, 1, 8),
                            EroNormalization = erosion.Normalization,
                            EroRidgeRounding = erosion.RidgeRounding,
                            EroCreaseRounding = erosion.CreaseRounding,
                            EroSlopeOnset = erosion.SlopeOnset,
                            EroAssumedSlope = erosion.AssumedSlope,
                            EroAssumedSlopeAmount = erosion.AssumedSlopeAmount,
                        });
                        break;
                }
            }
            CloseRun();

            return plan;
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
        /// Run a height plan: clear, then one dispatch per step, into the terrain's BakedHeightmap.
        /// Render callback.
        /// </summary>
        public void BakeHeight(Terrain terrain, HeightPlan plan, ID3D12GraphicsCommandList cmd)
        {
            if (plan == null || !plan.HasWork) return;

            EnsureInitialized();
            EnsureTexture(plan.Resolution);

            var device = Engine.Device;
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            cmd.SetComputeRootSignature(device.GlobalRootSignature);

            uint groups = (uint)((plan.Resolution + 7) / 8);

            // Every influence run of this bake shares one upload: each dispatch reads its own range.
            // (An upload buffer rewritten between dispatches of one command list would show every
            // dispatch the last write.)
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

            // The heightmap rests in Common between bakes, and a texture is never promoted to UnorderedAccess
            // implicitly (only to the read and copy states)
            cmd.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_heightTexture,
                    ResourceStates.Common, ResourceStates.UnorderedAccess)));

            _cs.SetPushConstant(_kernelClear, "Output", _heightUAV);
            _cs.Dispatch(_kernelClear, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(_heightTexture);

            foreach (var step in plan.Steps)
            {
                switch (step.Kind)
                {
                    case HeightStepKind.Influence: DispatchInfluence(cmd, plan, step, groups); break;
                    case HeightStepKind.Import:    DispatchImport(cmd, step, groups); break;
                    case HeightStepKind.Noise:     DispatchNoise(cmd, step, groups); break;
                    case HeightStepKind.Erosion:   DispatchErosion(cmd, step, groups); break;
                }
                cmd.ResourceBarrierUnorderedAccessView(_heightTexture);
            }

            // Transition heightmap from UAV back to Common so it can be
            // implicitly promoted to SRV by the height-range pyramid builder
            cmd.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_heightTexture,
                    ResourceStates.UnorderedAccess, ResourceStates.Common)));

            // Wrap the baked texture for the rendering pipeline
            terrain.BakedHeightmap = Texture.WrapNative(_heightTexture, _heightSRV);
            HeightVersion++;
        }

        private void DispatchInfluence(ID3D12GraphicsCommandList cmd, HeightPlan plan, in HeightStep step, uint groups)
        {
            var k = _kernelInfluenceLayer;
            _cs.SetPushConstant(k, "Output", _heightUAV);
            _cs.SetBuffer(k, "StampBuf", _heightStampBuffer);
            _cs.SetPushConstant(k, "StampStart", (uint)step.Start);
            _cs.SetPushConstant(k, "StampCount", (uint)step.Count);
            // BrushRadius carries the spline point buffer's SRV index in this kernel
            _cs.SetPushConstant(k, "BrushRadius",
                plan.SplinePoints.Count > 0 && _heightStampSplineBuffer != null ? _heightStampSplineBuffer.SrvIndex : 0u);
            _cs.Dispatch(k, cmd, groups, groups);
        }

        private void DispatchImport(ID3D12GraphicsCommandList cmd, in HeightStep step, uint groups)
        {
            var k = _kernelImport;
            _cs.SetPushConstant(k, "Source", step.SourceSrv);
            _cs.SetPushConstant(k, "Output", _heightUAV);
            _cs.SetPushConstant(k, "BlendMode", step.BlendMode);
            _cs.SetParam(k, "Opacity", step.Opacity);
            _cs.SetParam(k, "Amplitude", step.Scale);
            _cs.SetParam(k, "BrushTargetHeight", step.Bias);
            _cs.Dispatch(k, cmd, groups, groups);
        }

        private void DispatchNoise(ID3D12GraphicsCommandList cmd, in HeightStep step, uint groups)
        {
            EnsureNoiseLUT();

            var k = _kernelNoiseLayer;
            _cs.SetPushConstant(k, "Output", _heightUAV);
            _cs.SetPushConstant(k, "BlendMode", step.BlendMode);
            _cs.SetParam(k, "Opacity", step.Opacity);
            _cs.SetPushConstant(k, "NoiseType", step.NoiseType);
            _cs.SetPushConstant(k, "Octaves", step.Octaves);
            _cs.SetParam(k, "Frequency", step.Frequency);
            _cs.SetParam(k, "Amplitude", step.Amplitude);
            _cs.SetParam(k, "Lacunarity", step.Lacunarity);
            _cs.SetParam(k, "Persistence", step.Persistence);
            _cs.SetParam(k, "OffsetX", step.Offset.X);
            _cs.SetParam(k, "OffsetY", step.Offset.Y);
            _cs.SetPushConstant(k, "NoiseSeed", step.Seed);
            // Bind noise LUT SRV via the ErosionMode slot (aliased as NoiseLUTIdx in shader)
            _cs.SetPushConstant(k, "ErosionMode", _noiseLUTSRV);
            _cs.SetPushConstant(k, "TerraceSteps", step.TerraceSteps);
            _cs.SetParam(k, "TerraceSmoothness", step.TerraceSmoothness);
            // Spatial mask (radius 0 = whole terrain)
            _cs.SetParam(k, "MaskCenterX", step.MaskCenter.X);
            _cs.SetParam(k, "MaskCenterY", step.MaskCenter.Y);
            _cs.SetParam(k, "MaskRadius", step.MaskRadius);
            _cs.SetParam(k, "MaskFalloff", step.MaskFalloff);
            _cs.Dispatch(k, cmd, groups, groups);
        }

        private void DispatchErosion(ID3D12GraphicsCommandList cmd, in HeightStep step, uint groups)
        {
            // Single-dispatch erosion filter — read accumulated height, write eroded result directly
            var k = _kernelErosionFilter;

            _cs.SetPushConstant(k, "Source", _heightSRV);
            _cs.SetPushConstant(k, "Output", _heightUAV);
            _cs.SetPushConstant(k, "BlendMode", step.BlendMode);
            _cs.SetParam(k, "Opacity", step.Opacity);

            // Core erosion params (aliased onto existing push constant slots)
            _cs.SetParam(k, "BrushRadius", step.EroScale);          // EFScale
            _cs.SetParam(k, "Frequency", step.EroStrength);         // EFStrength
            _cs.SetParam(k, "Amplitude", step.EroGullyWeight);      // EFGullyWeight
            _cs.SetParam(k, "Lacunarity", step.EroDetail);          // EFDetail
            _cs.SetParam(k, "Persistence", step.EroLacunarity);     // EFLacunarity
            _cs.SetParam(k, "OffsetX", step.EroGain);               // EFGain
            _cs.SetParam(k, "OffsetY", step.EroCellScale);          // EFCellScale
            _cs.SetPushConstant(k, "NoiseSeed", step.EroOctaves);   // EFOctaves

            // Float values sent through uint slots — reinterpret bits
            _cs.SetPushConstant(k, "ErosionMode", BitConverter.SingleToUInt32Bits(step.EroNormalization));   // EFNormalization
            _cs.SetPushConstant(k, "TerraceSteps", BitConverter.SingleToUInt32Bits(step.EroRidgeRounding)); // EFRidgeRounding

            _cs.SetParam(k, "TerraceSmoothness", step.EroCreaseRounding);  // EFCreaseRounding
            _cs.SetParam(k, "MaskCenterX", 0.1f);                          // EFRoundInputMul
            _cs.SetParam(k, "MaskCenterY", 2.0f);                          // EFRoundOctMul (= lacunarity)
            _cs.SetParam(k, "MaskRadius", step.EroSlopeOnset);             // EFOnsetInput
            _cs.SetParam(k, "MaskFalloff", step.EroSlopeOnset);            // EFOnsetOctave

            // Assumed slope override
            _cs.SetParam(k, "BrushFalloff", step.EroAssumedSlope);             // EFAssumedVal
            _cs.SetParam(k, "BrushTargetHeight", step.EroAssumedSlopeAmount);  // EFAssumedAmt

            _cs.Dispatch(k, cmd, groups, groups);
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

        private void EnsureTexture(int resolution)
        {
            if (_heightTexture != null && _currentResolution == resolution)
                return;

            var device = Engine.Device;
            device.DeferDispose(_heightTexture); // frames in flight may still read it

            _heightTexture = device.CreateTexture2D(
                Format.R16_UNorm, resolution, resolution, 1, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _heightUAV = device.AllocateBindlessIndex();
            var uavDesc = new UnorderedAccessViewDescription
            {
                Format = Format.R16_UNorm,
                ViewDimension = UnorderedAccessViewDimension.Texture2D,
                Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
            };
            device.NativeDevice.CreateUnorderedAccessView(_heightTexture, null, uavDesc, device.GetCpuHandle(_heightUAV));

            _heightSRV = device.AllocateBindlessIndex();
            var srvDesc = new ShaderResourceViewDescription
            {
                Format = Format.R16_UNorm,
                ViewDimension = ShaderResourceViewDimension.Texture2D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2D = new Texture2DShaderResourceView
                {
                    MostDetailedMip = 0,
                    MipLevels = 1
                }
            };
            device.NativeDevice.CreateShaderResourceView(_heightTexture, srvDesc, device.GetCpuHandle(_heightSRV));

            _currentResolution = resolution;
        }

        /// <summary>
        /// Uploads pre-saved R16_UNorm baked heightmap bytes directly to the GPU texture.
        /// Used at load time when the baked heightmap was persisted to cache.
        /// </summary>
        public Texture UploadBakedHeightmap(byte[] pixels, int resolution)
        {
            if (pixels == null || pixels.Length == 0) return null;

            EnsureTexture(resolution);

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

                cmdList.ResourceBarrierTransition(_heightTexture,
                    ResourceStates.Common, ResourceStates.CopyDest);

                var src = new TextureCopyLocation(uploadResource, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R16_UNorm, (uint)resolution, (uint)resolution, 1, (uint)rowPitch)
                });
                var dst = new TextureCopyLocation(_heightTexture, 0);
                cmdList.CopyTextureRegion(dst, 0, 0, 0, src);

                cmdList.ResourceBarrierTransition(_heightTexture,
                    ResourceStates.CopyDest, ResourceStates.Common);

                cmdList.Close();
                device.SubmitAndWait(cmdList);

                Debug.Log($"[TerrainBaker] Uploaded baked heightmap from cache: {resolution}x{resolution} ({pixels.Length} bytes)");
                HeightVersion++;
                return Texture.WrapNative(_heightTexture, _heightSRV);
            }
            finally
            {
                cmdList.Dispose();
                allocator.Dispose();
                uploadResource.Dispose();
            }
        }

        /// <summary>
        /// Reads back the baked R16_UNorm heightmap into a CPU float[,] array ([x, z], texel rows along +Z).
        /// Values are normalized [0..1] — caller multiplies by MaxHeight.
        /// Returns null if no heightmap has been baked.
        ///
        /// With <paramref name="previous"/> (the field read back last, at this resolution) and a texel box (x0, z0
        /// inclusive, x1, z1 exclusive), only the box is fetched: the result is a copy of 'previous' with the box
        /// replaced. 'previous' itself is left as it is, since workers may still read it. On Eldermoor's 4097² map a
        /// whole readback took ~190 ms of the frame it ran in, after every height stamp edit (2026-10-26).
        /// </summary>
        public float[,] ReadbackHeightmap(float[,] previous = null, int x0 = 0, int z0 = 0, int x1 = int.MaxValue, int z1 = int.MaxValue)
        {
            if (_heightTexture == null || _currentResolution == 0)
                return null;

            var device = Engine.Device;
            int res = _currentResolution;
            if (previous == null || previous.GetLength(0) != res || previous.GetLength(1) != res)
            {
                previous = null;
                x0 = 0; z0 = 0; x1 = res; z1 = res;
            }
            x0 = Math.Clamp(x0, 0, res); x1 = Math.Clamp(x1, 0, res);
            z0 = Math.Clamp(z0, 0, res); z1 = Math.Clamp(z1, 0, res);
            if (previous != null && (x1 <= x0 || z1 <= z0))
                return previous;
            int width = x1 - x0, height = z1 - z0;
            int bytesPerPixel = 2; // R16_UNorm
            int rowPitch = (width * bytesPerPixel + 255) & ~255; // 256-byte aligned
            int totalBytes = rowPitch * height;

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
                cmdList.ResourceBarrierTransition(_heightTexture,
                    ResourceStates.Common, ResourceStates.CopySource);

                var src = new TextureCopyLocation(_heightTexture, 0);
                var dst = new TextureCopyLocation(readbackResource, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R16_UNorm, (uint)width, (uint)height, 1, (uint)rowPitch)
                });
                cmdList.CopyTextureRegion(dst, 0, 0, 0, src, new Vortice.Mathematics.Box(x0, z0, 0, x1, z1, 1));

                cmdList.ResourceBarrierTransition(_heightTexture,
                    ResourceStates.CopySource, ResourceStates.Common);

                cmdList.Close();
                device.SubmitAndWait(cmdList);

                var heights = previous != null ? (float[,])previous.Clone() : new float[res, res];
                unsafe
                {
                    void* pData;
                    readbackResource.Map(0, null, &pData);

                    var srcPtr = (byte*)pData;
                    for (int z = 0; z < height; z++)
                    {
                        var rowStart = (byte*)(srcPtr + z * rowPitch);
                        for (int x = 0; x < width; x++)
                        {
                            ushort raw = *(ushort*)(rowStart + x * 2);
                            heights[x0 + x, z0 + z] = raw / 65535.0f;
                        }
                    }

                    readbackResource.Unmap(0);
                }

                Debug.Log($"[TerrainBaker] HeightField readback: {width}x{height} of {res}x{res}");
                return heights;
            }
            finally
            {
                cmdList.Dispose();
                allocator.Dispose();
                readbackResource.Dispose();
            }
        }

        /// <summary>
        /// Reads back the baked heightmap as raw R16_UNorm bytes for DDS persistence.
        /// Returns null if no heightmap has been baked.
        /// </summary>
        public byte[] ReadbackBakedHeightmapBytes()
        {
            if (_heightTexture == null || _currentResolution == 0)
                return null;

            var device = Engine.Device;
            int res = _currentResolution;
            int bytesPerPixel = 2; // R16_UNorm
            int rowPitch = (res * bytesPerPixel + 255) & ~255;
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
                cmdList.ResourceBarrierTransition(_heightTexture,
                    ResourceStates.Common, ResourceStates.CopySource);

                var src = new TextureCopyLocation(_heightTexture, 0);
                var dst = new TextureCopyLocation(readbackResource, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R16_UNorm, (uint)res, (uint)res, 1, (uint)rowPitch)
                });
                cmdList.CopyTextureRegion(dst, 0, 0, 0, src);

                cmdList.ResourceBarrierTransition(_heightTexture,
                    ResourceStates.CopySource, ResourceStates.Common);

                cmdList.Close();
                device.SubmitAndWait(cmdList);

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
                cmdList.Dispose();
                allocator.Dispose();
                readbackResource.Dispose();
            }
        }

        /// <summary>Resolution of the current baked heightmap texture, or 0 if none.</summary>
        public int BakedResolution => _currentResolution;

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
                plan.Stamps.Add(desc);
            }

            return plan;
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
            uint heightSrv, uint controlArrayUAV, uint topLayerUAV, int resolution, int sliceCount,
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
            _splatCS.SetPushConstant(k, "HeightTex", heightSrv);
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
            uint heightSrv, uint controlMapsSrv, uint decoControlUAV, int resolution)
        {
            EnsureCoverageInitialized();

            UploadCoverage(plan, ref _decoStampBuffer, ref _decoStampBufferCapacity,
                ref _decoSplineBuffer, ref _decoSplineCapacity);

            var device = Engine.Device;
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            cmd.SetComputeRootSignature(device.GlobalRootSignature);

            var k = _kernelDecoControl;
            _decoCS.SetBuffer(k, "StampBuf", _decoStampBuffer);
            _decoCS.SetPushConstant(k, "ControlUAV", decoControlUAV);
            _decoCS.SetPushConstant(k, "DecoratorCount", (uint)plan.TargetCount);
            _decoCS.SetPushConstant(k, "Resolution", (uint)resolution);
            _decoCS.SetPushConstant(k, "HeightTex", heightSrv);
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
        /// Clear the given rects of the mask, then cut every stamp of the plan into it (the union of the stamps, by
        /// the largest weight). Rects a stamp covered at the last bake must be among 'clear', or it stays cut there.
        /// </summary>
        public static void BakeHoles(HolePlan plan, byte[] mask, IEnumerable<HoleRect> clear)
        {
            int res = plan.Resolution;
            foreach (var r in clear)
                for (int z = r.Z0; z <= r.Z1; z++)
                    Array.Clear(mask, z * res + r.X0, r.X1 - r.X0 + 1);

            float inv = 1f / (res - 1);
            var points = CollectionsMarshal.AsSpan(plan.Points);
            foreach (var shape in plan.Shapes)
            {
                var r = shape.Rect;
                for (int z = r.Z0; z <= r.Z1; z++)
                {
                    int row = z * res;
                    for (int x = r.X0; x <= r.X1; x++)
                    {
                        float w = HoleWeight(shape, points, new Vector2(x * inv, z * inv));
                        if (w <= 0f) continue;
                        byte b = (byte)MathF.Round(w * 255f);
                        if (b > mask[row + x]) mask[row + x] = b;
                    }
                }
            }
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
            _splatStampBuffer?.Dispose();
            _splatSplineBuffer?.Dispose();
            _decoStampBuffer?.Dispose();
            _decoSplineBuffer?.Dispose();
        }
    }
}
