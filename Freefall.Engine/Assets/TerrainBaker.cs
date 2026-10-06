using System;
using System.Collections.Generic;
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

        /// <summary>GPU struct matching HLSL HeightStampDescriptor. 64 bytes (16 x uint/float).</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct HeightStampDescriptorGPU
        {
            public Vector2 Center;           // terrain UV center (radial)
            public float Radius;             // UV-space inner radius
            public float Falloff;            // UV-space falloff width
            public float TargetHeight;       // normalized [0..1]
            public uint InvertShape;         // 0 = flatten toward, 1 = push away
            public uint SplinePointOffset;   // into spline point buffer (0xFFFFFFFF = radial)
            public uint SplinePointCount;    // 0 = radial, high bit = closed area
            public float NoiseFreq;          // edge noise frequency (0 = disabled)
            public float NoiseAmp;           // edge noise amplitude in UV space
            public uint NoiseSeed;
            public uint HeightmapIdx;        // bindless SRV index (0 = no heightmap)
            public float HeightmapStrength;  // normalized strength (worldStrength / maxHeight)
            public float RotationSin;        // sin(entity Y rotation)
            public float RotationCos;        // cos(entity Y rotation)
            public uint BlendMode;           // HeightBlendMode
        }

        /// <summary>GPU struct matching HLSL StampSplinePoint. 16 bytes.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct StampSplinePointGPU
        {
            public Vector2 UV;       // terrain UV position
            public float Height;     // normalized [0..1]
            public float HalfWidth;  // UV-space half-width
        }

        /// <summary>GPU struct matching HLSL CoverageStamp (terrain_coverage.hlsli). 80 bytes.</summary>
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
            internal int Resolution;

            /// <summary>True if any stamp contributes; a bake without work is skipped so a terrain
            /// with no height stamps keeps the heightmap it was saved with.</summary>
            public bool HasWork => Steps.Count > 0;
        }

        /// <summary>
        /// Collect the height stamps that apply to the terrain and lay out the bake: consecutive
        /// HeightStamps become one dispatch, and every whole-terrain operation (import, noise, erosion)
        /// is a dispatch of its own at its place in the priority order. Main thread.
        /// </summary>
        public HeightPlan PrepareHeight(Terrain terrain, TerrainRenderer renderer)
        {
            var plan = new HeightPlan { Resolution = terrain.EffectiveHeightmapResolution };

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
                        // World Y, as the target height of a local stamp (see TryBuildHeightDescriptor)
                        float baseY = (height.Transform?.WorldPosition.Y ?? 0f) + height.HeightOffset;
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
                            plan.Descriptors.Add(desc);
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

            desc.InvertShape = stamp.InvertShape ? 1u : 0u;
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
                desc.TargetHeight = ((relative ? 0f : center.Y) + stamp.HeightOffset) / maxHeight;
                desc.SplinePointOffset = 0xFFFFFFFF;
                desc.SplinePointCount = 0;
            }

            if (stamp.Heightmap != null)
            {
                desc.HeightmapIdx = stamp.Heightmap.BindlessIndex;
                desc.HeightmapStrength = stamp.Strength / maxHeight;

                // Extract Y rotation from entity transform
                var rot = stamp.Transform?.Rotation ?? Quaternion.Identity;
                float yaw = MathF.Atan2(2f * (rot.W * rot.Y + rot.X * rot.Z),
                                        1f - 2f * (rot.Y * rot.Y + rot.Z * rot.Z));
                desc.RotationSin = MathF.Sin(yaw);
                desc.RotationCos = MathF.Cos(yaw);
            }

            return true;
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
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });

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

            _heightTexture?.Release();

            var device = Engine.Device;
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
        /// Reads back the baked R16_UNorm heightmap into a CPU float[,] array.
        /// Values are normalized [0..1] — caller multiplies by MaxHeight.
        /// Returns null if no heightmap has been baked.
        /// </summary>
        public float[,] ReadbackHeightmap()
        {
            if (_heightTexture == null || _currentResolution == 0)
                return null;

            var device = Engine.Device;
            int res = _currentResolution;
            int bytesPerPixel = 2; // R16_UNorm
            int rowPitch = (res * bytesPerPixel + 255) & ~255; // 256-byte aligned
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

                var heights = new float[res, res];
                unsafe
                {
                    void* pData;
                    readbackResource.Map(0, null, &pData);

                    var srcPtr = (byte*)pData;
                    for (int y = 0; y < res; y++)
                    {
                        var rowStart = (byte*)(srcPtr + y * rowPitch);
                        for (int x = 0; x < res; x++)
                        {
                            ushort raw = *(ushort*)(rowStart + x * 2);
                            heights[x, y] = raw / 65535.0f;
                        }
                    }

                    readbackResource.Unmap(0);
                }

                Debug.Log($"[TerrainBaker] HeightField readback: {res}x{res}");
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
            }

            // Filter. The ranges are always written so the shader never reads garbage.
            desc.HeightMin = stamp.HeightRange.X;
            desc.HeightMax = stamp.HeightRange.Y;
            desc.HeightBlend = stamp.HeightBlend;
            desc.SlopeMin = stamp.SlopeRange.X;
            desc.SlopeMax = stamp.SlopeRange.Y;
            desc.SlopeBlend = stamp.SlopeBlend;

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
        /// Render callback.
        /// </summary>
        public void BakeSplat(CoveragePlan plan, ID3D12GraphicsCommandList cmd, Terrain terrain,
            uint heightSrv, uint controlArrayUAV, int resolution, int sliceCount)
        {
            EnsureCoverageInitialized();

            UploadCoverage(plan, ref _splatStampBuffer, ref _splatStampBufferCapacity,
                ref _splatSplineBuffer, ref _splatSplineCapacity);

            var device = Engine.Device;
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });

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

            uint groups = (uint)((resolution + 7) / 8);
            _splatCS.Dispatch(k, cmd, groups, groups);
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
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });

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

        // ── Shared helpers ─────────────────────────────────────────────

        /// <summary>
        /// Sample a stamp's spline into terrain UV points for the GPU. Returns false when the spline
        /// has fewer than two points.
        /// </summary>
        private static bool BuildSplinePoints(
            TerrainStamp stamp, Vector3 terrainOrigin, Vector2 terrainSize, float maxHeight, float heightOffset,
            List<StampSplinePointGPU> splinePoints,
            out uint splinePointOffset, out uint splinePointCount, out float radius, bool useWorldY = true)
        {
            splinePointOffset = 0xFFFFFFFF;
            splinePointCount = 0;
            radius = 0;

            var spline = stamp.GetSpline();
            if (spline == null || spline.Points.Count < 2)
                return false;

            splinePointOffset = (uint)splinePoints.Count;
            float uvRadius = stamp.Radius / Math.Max(terrainSize.X, terrainSize.Y);

            int sampleCount = Math.Max(spline.TotalSegments, spline.Points.Count * 4);
            for (int i = 0; i <= sampleCount; i++)
            {
                float t = (float)i / sampleCount;
                var worldPos = spline.GetWorldPoint(t);

                splinePoints.Add(new StampSplinePointGPU
                {
                    UV = new Vector2(
                        (worldPos.X - terrainOrigin.X) / terrainSize.X,
                        (worldPos.Z - terrainOrigin.Z) / terrainSize.Y),
                    Height = ((useWorldY ? worldPos.Y : 0f) + heightOffset) / maxHeight,
                    HalfWidth = uvRadius * spline.GetWidth(t),
                });
            }

            uint ptCount = (uint)(sampleCount + 1);
            if (spline.Closed) ptCount |= 0x80000000;

            splinePointCount = ptCount;
            // The stamp's own radius stays the reference: the shader scales the falloff by
            // (sample half-width / Radius), see EvaluateStampWeight.
            radius = uvRadius;
            return true;
        }

        private static void EnsureUpload<T>(ref GraphicsBuffer buffer, ref int capacity, int count, int minimum)
            where T : unmanaged
        {
            if (buffer != null && capacity >= count) return;
            buffer?.Dispose();
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
