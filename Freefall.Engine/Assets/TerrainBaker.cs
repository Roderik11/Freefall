using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;

using Freefall.Base;
using Freefall.Graphics;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Assets
{
    /// <summary>
    /// GPU-based height layer + stamp compositor. Iterates HeightLayers bottom-to-top,
    /// then applies StampGroups (one dispatch per group with batched instances).
    /// Output: R16_Float heightmap at configurable resolution.
    /// </summary>
    public class TerrainBaker : IDisposable
    {
        /// <summary>Engine-wide singleton — one compute shader, shared buffers.</summary>
        public static TerrainBaker Instance { get; } = new();

        private ComputeShader _cs;
        private int _kernelClear;
        private int _kernelImport;
        private int _kernelStampGroup;
        private int _kernelPaintBrush;
        private int _kernelClearDelta;
        private int _kernelImportChannel;
        private int _kernelPackChannels;
        private int _kernelBrushRaycast;
        private int _kernelNoiseLayer;
        private int _kernelErosionInit;
        private int _kernelErosionStep;

        // GPU resources for the baked heightmap
        private ID3D12Resource _heightTexture;
        private uint _heightUAV;
        private uint _heightSRV;
        private int _currentResolution;

        // Stamp instance upload buffer (reused across groups)
        private GraphicsBuffer _stampBuffer;
        private int _stampBufferCapacity;

        // Brush stroke point upload buffer (reused across strokes)
        private GraphicsBuffer _strokeBuffer;
        private int _strokeBufferCapacity;

        // GPU raycast result buffer — written by CS_BrushRaycast, read by CS_PaintBrush
        private GraphicsBuffer _raycastResultBuffer;

        // Shared erosion auxiliary textures (reused across all ErosionHeightLayers)
        private ID3D12Resource _erosionHeightTex;  // R32_Float working height
        private uint _erosionHeightUAV, _erosionHeightSRV;
        private ID3D12Resource _erosionWaterTex;    // R32_Float water map
        private uint _erosionWaterUAV;
        private ID3D12Resource _erosionSedimentTex; // R32_Float sediment map
        private uint _erosionSedimentUAV;
        private int _erosionTexResolution;

        // Pre-baked tileable noise LUT (256x256, RGBA8, 2 independent noise channels)
        private ID3D12Resource _noiseLUTTex;
        private uint _noiseLUTSRV;

        private bool _initialized;

        /// <summary>Which ControlMap category to paint.</summary>
        public enum ControlMapTarget { Height, Splatmap, Density }

        /// <summary>GPU struct matching HLSL StampData. Must be 32 bytes (8-float aligned).</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct StampDataGPU
        {
            public Vector2 Position;
            public float Radius;
            public float Strength;
            public float Falloff;
            public float Rotation;
            public Vector2 _pad;
        }

        private void EnsureInitialized()
        {
            if (_initialized) return;
            _cs = new ComputeShader("terrain_height_bake.hlsl");
            _kernelClear = _cs.FindKernel("CS_Clear");
            _kernelImport = _cs.FindKernel("CS_ImportLayer");
            _kernelStampGroup = _cs.FindKernel("CS_StampGroup");
            _kernelPaintBrush = _cs.FindKernel("CS_PaintBrush");
            _kernelClearDelta = _cs.FindKernel("CS_ClearDelta");
            _kernelImportChannel = _cs.FindKernel("CS_ImportChannel");
            _kernelPackChannels = _cs.FindKernel("CS_PackChannels");
            _kernelBrushRaycast = _cs.FindKernel("CS_BrushRaycast");
            _kernelNoiseLayer = _cs.FindKernel("CS_NoiseLayer");
            _kernelErosionInit = _cs.FindKernel("CS_ErosionInit");
            _kernelErosionStep = _cs.FindKernel("CS_ErosionStep");
            _initialized = true;
        }

        /// <summary>
        /// Bake all HeightLayers + StampGroups into the terrain's BakedHeightmap.
        /// </summary>
        public void Bake(Terrain terrain, ID3D12GraphicsCommandList cmd)
        {
            if (terrain.HeightLayers.Count == 0 && terrain.StampGroups.Count == 0) return;

            EnsureInitialized();
            EnsureTexture(terrain.HeightmapResolution);

            var device = Engine.Device;
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            int res = terrain.HeightmapResolution;
            uint groups = (uint)((res + 7) / 8);

            // Phase 1: Clear
            _cs.SetPushConstant(_kernelClear, "Output", _heightUAV);
            _cs.Dispatch(_kernelClear, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(_heightTexture);

            // Phase 2: Height layers (bottom-to-top)
            foreach (var layer in terrain.HeightLayers)
            {
                if (!layer.Enabled) continue;
                if (layer is ImportHeightLayer import)
                    DispatchImport(cmd, import, groups);
                else if (layer is PaintHeightLayer paint)
                    DispatchPaint(cmd, paint, res, groups);
                else if (layer is NoiseHeightLayer noise)
                    DispatchNoise(cmd, noise, groups);
                else if (layer is ErosionHeightLayer erosion)
                    DispatchErosion(cmd, erosion, res, groups);
            }

            // Phase 3: Stamp groups (after layers)
            foreach (var group in terrain.StampGroups)
            {
                if (!group.Enabled || group.Instances.Count == 0 || group.Brush == null) continue;
                DispatchStampGroup(cmd, group, groups);
            }

            // Transition heightmap from UAV back to Common so it can be
            // implicitly promoted to SRV by the height-range pyramid builder
            cmd.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_heightTexture,
                    ResourceStates.UnorderedAccess, ResourceStates.Common)));

            // Wrap the baked texture for the rendering pipeline
            terrain.BakedHeightmap = Texture.WrapNative(_heightTexture, _heightSRV);
        }

        private void DispatchImport(ID3D12GraphicsCommandList cmd, ImportHeightLayer layer, uint groups)
        {
            if (layer.Source == null) return;

            _cs.SetPushConstant(_kernelImport, "Source", layer.Source.BindlessIndex);
            _cs.SetPushConstant(_kernelImport, "Output", _heightUAV);
            _cs.SetPushConstant(_kernelImport, "BlendMode", (uint)layer.BlendMode);
            _cs.SetParam(_kernelImport, "Opacity", layer.Opacity);
            _cs.Dispatch(_kernelImport, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(_heightTexture);
        }

        private void DispatchPaint(ID3D12GraphicsCommandList cmd, PaintHeightLayer layer, int resolution, uint groups)
        {
            if (layer.ControlMap == null || layer.ControlMap.BindlessIndex == 0) return;


            _cs.SetPushConstant(_kernelImport, "Source", layer.ControlMap.BindlessIndex);
            _cs.SetPushConstant(_kernelImport, "Output", _heightUAV);
            _cs.SetPushConstant(_kernelImport, "BlendMode", (uint)layer.BlendMode);
            _cs.SetParam(_kernelImport, "Opacity", layer.Opacity);
            _cs.Dispatch(_kernelImport, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(_heightTexture);
        }

        private void DispatchStampGroup(ID3D12GraphicsCommandList cmd, StampGroup group, uint groups)
        {
            int count = group.Instances.Count;
            EnsureStampBuffer(count);

            // Upload stamp instances
            unsafe
            {
                var dst = _stampBuffer.WritePtr<StampDataGPU>();
                for (int i = 0; i < count; i++)
                {
                    var s = group.Instances[i];
                    dst[i] = new StampDataGPU
                    {
                        Position = s.Position,
                        Radius = s.Radius,
                        Strength = s.Strength,
                        Falloff = s.Falloff,
                        Rotation = s.Rotation,
                    };
                }
            }

            _cs.SetPushConstant(_kernelStampGroup, "Source", group.Brush.BindlessIndex);
            _cs.SetPushConstant(_kernelStampGroup, "Output", _heightUAV);
            _cs.SetBuffer(_kernelStampGroup, "StampBuf", _stampBuffer);
            _cs.SetPushConstant(_kernelStampGroup, "BlendMode", (uint)group.BlendMode);
            _cs.SetParam(_kernelStampGroup, "Opacity", group.Opacity);
            _cs.SetPushConstant(_kernelStampGroup, "StampCount", (uint)count);
            _cs.Dispatch(_kernelStampGroup, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(_heightTexture);
        }

        private void DispatchNoise(ID3D12GraphicsCommandList cmd, NoiseHeightLayer layer, uint groups)
        {
            EnsureNoiseLUT();

            _cs.SetPushConstant(_kernelNoiseLayer, "Output", _heightUAV);
            _cs.SetPushConstant(_kernelNoiseLayer, "BlendMode", (uint)layer.BlendMode);
            _cs.SetParam(_kernelNoiseLayer, "Opacity", layer.Opacity);
            _cs.SetPushConstant(_kernelNoiseLayer, "NoiseType", (uint)layer.Type);
            _cs.SetPushConstant(_kernelNoiseLayer, "Octaves", (uint)layer.Octaves);
            _cs.SetParam(_kernelNoiseLayer, "Frequency", layer.Frequency);
            _cs.SetParam(_kernelNoiseLayer, "Amplitude", layer.Amplitude);
            _cs.SetParam(_kernelNoiseLayer, "Lacunarity", layer.Lacunarity);
            _cs.SetParam(_kernelNoiseLayer, "Persistence", layer.Persistence);
            _cs.SetParam(_kernelNoiseLayer, "OffsetX", layer.Offset.X);
            _cs.SetParam(_kernelNoiseLayer, "OffsetY", layer.Offset.Y);
            _cs.SetPushConstant(_kernelNoiseLayer, "NoiseSeed", (uint)layer.Seed);
            // Bind noise LUT SRV via the ErosionMode slot (aliased as NoiseLUTIdx in shader)
            _cs.SetPushConstant(_kernelNoiseLayer, "ErosionMode", _noiseLUTSRV);
            // Terrace params
            _cs.SetPushConstant(_kernelNoiseLayer, "TerraceSteps", (uint)layer.TerraceSteps);
            _cs.SetParam(_kernelNoiseLayer, "TerraceSmoothness", layer.TerraceSmoothness);
            // Spatial mask params
            _cs.SetParam(_kernelNoiseLayer, "MaskCenterX", layer.MaskCenter.X);
            _cs.SetParam(_kernelNoiseLayer, "MaskCenterY", layer.MaskCenter.Y);
            _cs.SetParam(_kernelNoiseLayer, "MaskRadius", layer.MaskRadius);
            _cs.SetParam(_kernelNoiseLayer, "MaskFalloff", layer.MaskFalloff);
            _cs.Dispatch(_kernelNoiseLayer, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(_heightTexture);
        }

        private void DispatchErosion(ID3D12GraphicsCommandList cmd, ErosionHeightLayer layer, int resolution, uint groups)
        {
            EnsureErosionTextures(resolution);

            // Phase 1: Init — copy current baked height into working buffer, clear water/sediment
            _cs.SetPushConstant(_kernelErosionInit, "Source", _heightSRV);
            _cs.SetPushConstant(_kernelErosionInit, "Output", _erosionHeightUAV);
            _cs.SetPushConstant(_kernelErosionInit, "StampBuf", _erosionWaterUAV);     // WaterIdx
            _cs.SetPushConstant(_kernelErosionInit, "StampCount", _erosionSedimentUAV); // SedimentIdx
            _cs.Dispatch(_kernelErosionInit, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(_erosionHeightTex);
            cmd.ResourceBarrierUnorderedAccessView(_erosionWaterTex);
            cmd.ResourceBarrierUnorderedAccessView(_erosionSedimentTex);

            // Phase 2: Iterative erosion steps
            _cs.SetPushConstant(_kernelErosionStep, "Output", _erosionHeightUAV);
            _cs.SetPushConstant(_kernelErosionStep, "StampBuf", _erosionWaterUAV);
            _cs.SetPushConstant(_kernelErosionStep, "StampCount", _erosionSedimentUAV);
            _cs.SetPushConstant(_kernelErosionStep, "ErosionMode", (uint)layer.Mode);
            _cs.SetPushConstant(_kernelErosionStep, "NoiseSeed", (uint)layer.Seed);

            // Erosion params (aliased onto noise float slots)
            _cs.SetParam(_kernelErosionStep, "Frequency", layer.RainRate);           // RainRate
            _cs.SetParam(_kernelErosionStep, "Amplitude", layer.SedimentCapacity);   // SedimentCap
            _cs.SetParam(_kernelErosionStep, "Lacunarity", layer.DepositionRate);    // DepositRate
            _cs.SetParam(_kernelErosionStep, "Persistence", layer.DissolutionRate);  // DissolveRate
            _cs.SetParam(_kernelErosionStep, "OffsetX", layer.Evaporation);          // Evaporation
            _cs.SetParam(_kernelErosionStep, "OffsetY", layer.TalusAngle);           // TalusAngleDeg
            _cs.SetParam(_kernelErosionStep, "BrushRadius", layer.ThermalRate);      // ThermalRate

            for (int i = 0; i < layer.Iterations; i++)
            {
                _cs.Dispatch(_kernelErosionStep, cmd, groups, groups);
                cmd.ResourceBarrierUnorderedAccessView(_erosionHeightTex);
                cmd.ResourceBarrierUnorderedAccessView(_erosionWaterTex);
                cmd.ResourceBarrierUnorderedAccessView(_erosionSedimentTex);
            }

            // Phase 3: Blend eroded result back into main heightmap
            // Re-use CS_ImportLayer to composite erosionHeight → _heightTexture
            _cs.SetPushConstant(_kernelImport, "Source", _erosionHeightSRV);
            _cs.SetPushConstant(_kernelImport, "Output", _heightUAV);
            _cs.SetPushConstant(_kernelImport, "BlendMode", (uint)layer.BlendMode);
            _cs.SetParam(_kernelImport, "Opacity", layer.Opacity);
            _cs.Dispatch(_kernelImport, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(_heightTexture);
        }

        private void EnsureErosionTextures(int resolution)
        {
            if (_erosionHeightTex != null && _erosionTexResolution == resolution)
                return;

            // Release old
            _erosionHeightTex?.Release();
            _erosionWaterTex?.Release();
            _erosionSedimentTex?.Release();

            var device = Engine.Device;
            _erosionTexResolution = resolution;

            // Working height (R32_Float for full precision during simulation)
            _erosionHeightTex = device.CreateTexture2D(
                Format.R32_Float, resolution, resolution, 1, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _erosionHeightUAV = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(_erosionHeightTex, null,
                new UnorderedAccessViewDescription
                {
                    Format = Format.R32_Float,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
                }, device.GetCpuHandle(_erosionHeightUAV));

            _erosionHeightSRV = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(_erosionHeightTex,
                new ShaderResourceViewDescription
                {
                    Format = Format.R32_Float,
                    ViewDimension = ShaderResourceViewDimension.Texture2D,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
                }, device.GetCpuHandle(_erosionHeightSRV));

            // Water map
            _erosionWaterTex = device.CreateTexture2D(
                Format.R32_Float, resolution, resolution, 1, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _erosionWaterUAV = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(_erosionWaterTex, null,
                new UnorderedAccessViewDescription
                {
                    Format = Format.R32_Float,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
                }, device.GetCpuHandle(_erosionWaterUAV));

            // Sediment map
            _erosionSedimentTex = device.CreateTexture2D(
                Format.R32_Float, resolution, resolution, 1, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _erosionSedimentUAV = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(_erosionSedimentTex, null,
                new UnorderedAccessViewDescription
                {
                    Format = Format.R32_Float,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
                }, device.GetCpuHandle(_erosionSedimentUAV));
        }

        // ── Noise LUT generation ──────────────────────────────────────────

        private const int NoiseLUTSize = 256;
        // Noise tiling period in cells — small for many texels per cell (smooth interpolation).
        // Per-octave UV rotation in the shader breaks visible tiling.
        private const int NoiseLUTPeriod = 16;

        private void EnsureNoiseLUT()
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

        private void EnsureStampBuffer(int requiredCount)
        {
            if (_stampBuffer != null && _stampBufferCapacity >= requiredCount) return;

            _stampBuffer?.Dispose();
            _stampBufferCapacity = Math.Max(requiredCount, 64); // min 64 to reduce reallocs
            _stampBuffer = GraphicsBuffer.CreateUpload<StampDataGPU>(_stampBufferCapacity, mapped: true);
        }

        private void EnsureTexture(int resolution)
        {
            if (_heightTexture != null && _currentResolution == resolution)
                return;

            _heightTexture?.Release();

            var device = Engine.Device;
            _heightTexture = device.CreateTexture2D(
                Format.R16_Float, resolution, resolution, 1, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _heightUAV = device.AllocateBindlessIndex();
            var uavDesc = new UnorderedAccessViewDescription
            {
                Format = Format.R16_Float,
                ViewDimension = UnorderedAccessViewDimension.Texture2D,
                Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
            };
            device.NativeDevice.CreateUnorderedAccessView(_heightTexture, null, uavDesc, device.GetCpuHandle(_heightUAV));

            _heightSRV = device.AllocateBindlessIndex();
            var srvDesc = new ShaderResourceViewDescription
            {
                Format = Format.R16_Float,
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

        // ── Brush Painting (Multi-Target) ────────────────────────────────

        /// <summary>GPU state for a single ControlMap target.</summary>
        private struct ControlMapGPU
        {
            public ID3D12Resource Texture;
            public uint UAV, SRV;
            public int Resolution;
            public bool NeedsInitialClear;
            /// <summary>Cached Texture wrapper — avoids WrapNative allocation per call.</summary>
            public Texture Wrapper;
        }

        private readonly Dictionary<(ControlMapTarget, int), ControlMapGPU> _controlMaps = new();

        /// <summary>
        /// Dispatches a brush stroke on any ControlMap target.
        /// setControlMap is called to wire the resulting Texture back to the owner.
        /// Points are in terrain UV space [0..1].
        /// </summary>
        public void PaintBrush(Terrain terrain, ControlMapTarget target, int layerIndex,
                               Action<Texture> setControlMap,
                               ID3D12GraphicsCommandList cmd,
                               Vector2[] strokePoints, int pointCount,
                               uint mode, float strength,
                               float radius, float falloff,
                               float targetHeight = 0)
        {
            if (pointCount == 0 || strokePoints == null) return;

            EnsureInitialized();
            int res = terrain.HeightmapResolution;
            var key = (target, layerIndex);
            var gpu = EnsureControlMap(key, res, setControlMap);

            var device = Engine.Device;
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            uint groups = (uint)((res + 7) / 8);

            // Clear on first use
            if (gpu.NeedsInitialClear)
            {
                _cs.SetPushConstant(_kernelClearDelta, "Output", gpu.UAV);
                _cs.Dispatch(_kernelClearDelta, cmd, groups, groups);
                cmd.ResourceBarrierUnorderedAccessView(gpu.Texture);
                gpu.NeedsInitialClear = false;
                _controlMaps[key] = gpu;
            }

            // Upload stroke points
            EnsureStrokeBuffer(pointCount);
            unsafe
            {
                var dst = _strokeBuffer.WritePtr<Vector2>();
                for (int i = 0; i < pointCount; i++)
                    dst[i] = strokePoints[i];
            }

            // Convert world radius to UV radius
            float uvRadius = radius / Math.Max(terrain.TerrainSize.X, terrain.TerrainSize.Y);

            // Normalize target height
            float normalizedTarget = targetHeight / terrain.MaxHeight;

            // Push constants
            _cs.SetPushConstant(_kernelPaintBrush, "Source", _heightSRV);  // Current baked heightmap for flatten/smooth
            _cs.SetPushConstant(_kernelPaintBrush, "Output", gpu.UAV);
            _cs.SetBuffer(_kernelPaintBrush, "StampBuf", _strokeBuffer);
            _cs.SetPushConstant(_kernelPaintBrush, "BlendMode", mode);
            _cs.SetParam(_kernelPaintBrush, "Opacity", strength);
            _cs.SetPushConstant(_kernelPaintBrush, "StampCount", (uint)pointCount);

            _cs.SetParam(_kernelPaintBrush, "BrushRadius", uvRadius);
            _cs.SetParam(_kernelPaintBrush, "BrushFalloff", falloff);
            _cs.SetParam(_kernelPaintBrush, "BrushTargetHeight", normalizedTarget);
            _cs.SetPushConstant(_kernelPaintBrush, "FlipV", target != ControlMapTarget.Height ? 1u : 0u);

            _cs.Dispatch(_kernelPaintBrush, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(gpu.Texture);
        }

        /// <summary>
        /// GPU-only brush: dispatches CS_BrushRaycast (ray march against heightmap)
        /// then chains CS_PaintBrush (paints at the hit UV). Zero CPU readback.
        /// </summary>
        public void BrushRaycastAndPaint(Terrain terrain, ControlMapTarget target, int layerIndex,
                                         Action<Texture> setControlMap,
                                         ID3D12GraphicsCommandList cmd,
                                         Vector3 rayOrigin, Vector3 rayDir,
                                         Vector3 terrainOrigin, Vector2 terrainSize, float maxHeight,
                                         uint mode, float strength,
                                         float radius, float falloff,
                                         float targetHeight = 0)
        {
            EnsureInitialized();
            //Debug.Log($"[BrushRaycast] heightSRV={_heightSRV}, ray={rayOrigin}/{rayDir}, " +
            //          $"terrainOrig={terrainOrigin}, size={terrainSize}, maxH={maxHeight}, " +
            //          $"mode={mode}, str={strength}, rad={radius}");
            int res = terrain.HeightmapResolution;
            var key = (target, layerIndex);
            var gpu = EnsureControlMap(key, res, setControlMap);

            var device = Engine.Device;
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            // Clear on first use
            uint groups = (uint)((res + 7) / 8);
            if (gpu.NeedsInitialClear)
            {
                _cs.SetPushConstant(_kernelClearDelta, "Output", gpu.UAV);
                _cs.Dispatch(_kernelClearDelta, cmd, groups, groups);
                cmd.ResourceBarrierUnorderedAccessView(gpu.Texture);
                gpu.NeedsInitialClear = false;
                _controlMaps[key] = gpu;
            }

            // ── Phase 1: GPU Raycast ──
            // Create/reuse raycast result buffer (1 float2, SRV+UAV on GPU default heap)
            if (_raycastResultBuffer == null)
                _raycastResultBuffer = GraphicsBuffer.CreateStructured<Vector2>(1, srv: true, uav: true);

            _cs.SetPushConstant(_kernelBrushRaycast, "Source", _heightSRV);
            _cs.SetUAV(_kernelBrushRaycast, "StampBuf", _raycastResultBuffer);
            _cs.SetParam(_kernelBrushRaycast, "RayOriginX", rayOrigin.X);
            _cs.SetParam(_kernelBrushRaycast, "RayOriginY", rayOrigin.Y);
            _cs.SetParam(_kernelBrushRaycast, "RayOriginZ", rayOrigin.Z);
            _cs.SetParam(_kernelBrushRaycast, "RayDirX", rayDir.X);
            _cs.SetParam(_kernelBrushRaycast, "RayDirY", rayDir.Y);
            _cs.SetParam(_kernelBrushRaycast, "RayDirZ", rayDir.Z);
            _cs.SetParam(_kernelBrushRaycast, "TerrainOriginX", terrainOrigin.X);
            _cs.SetParam(_kernelBrushRaycast, "TerrainOriginZ", terrainOrigin.Z);
            _cs.SetParam(_kernelBrushRaycast, "TerrainSizeX", terrainSize.X);
            _cs.SetParam(_kernelBrushRaycast, "TerrainSizeZ", terrainSize.Y);
            _cs.SetParam(_kernelBrushRaycast, "TerrainMaxHeight", maxHeight);

            _cs.Dispatch(_kernelBrushRaycast, cmd, 1, 1, 1);
            _raycastResultBuffer.UAVBarrier(cmd);

            // ── Phase 2: Paint at hit UV ──
            // Convert world radius to UV radius
            float uvRadius = radius / Math.Max(terrainSize.X, terrainSize.Y);
            float normalizedTarget = targetHeight / maxHeight;

            _cs.SetPushConstant(_kernelPaintBrush, "Source", _heightSRV);
            _cs.SetPushConstant(_kernelPaintBrush, "Output", gpu.UAV);
            _cs.SetSRV(_kernelPaintBrush, "StampBuf", _raycastResultBuffer);
            _cs.SetPushConstant(_kernelPaintBrush, "BlendMode", mode);
            _cs.SetParam(_kernelPaintBrush, "Opacity", strength);
            _cs.SetPushConstant(_kernelPaintBrush, "StampCount", 1u);
            _cs.SetParam(_kernelPaintBrush, "BrushRadius", uvRadius);
            _cs.SetParam(_kernelPaintBrush, "BrushFalloff", falloff);
            _cs.SetParam(_kernelPaintBrush, "BrushTargetHeight", normalizedTarget);
            _cs.SetPushConstant(_kernelPaintBrush, "FlipV", target != ControlMapTarget.Height ? 1u : 0u);

            _cs.Dispatch(_kernelPaintBrush, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(gpu.Texture);
        }

        /// <summary>
        /// Clears a specific ControlMap to zero.
        /// </summary>
        public void ClearControlMap(ControlMapTarget target, int layerIndex, ID3D12GraphicsCommandList cmd, int resolution)
        {
            if (!_controlMaps.TryGetValue((target, layerIndex), out var gpu)) return;

            EnsureInitialized();
            var device = Engine.Device;
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            uint groups = (uint)((resolution + 7) / 8);

            _cs.SetPushConstant(_kernelClearDelta, "Output", gpu.UAV);
            _cs.Dispatch(_kernelClearDelta, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(gpu.Texture);
        }

        /// <summary>
        /// Imports a single channel from a source texture into a ControlMap.
        /// channelIndex: 0=R, 1=G, 2=B, 3=A
        /// </summary>
        public void ImportChannel(Terrain terrain, ControlMapTarget target, int layerIndex,
                                  Action<Texture> setControlMap,
                                  ID3D12GraphicsCommandList cmd,
                                  Texture sourceTexture, int channelIndex)
        {
            if (sourceTexture == null || sourceTexture.BindlessIndex == 0) return;

            EnsureInitialized();
            int res = terrain.HeightmapResolution;
            var key = (target, layerIndex);
            var gpu = EnsureControlMap(key, res, setControlMap);
            gpu.NeedsInitialClear = false;
            _controlMaps[key] = gpu;

            var device = Engine.Device;
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            uint groups = (uint)((res + 7) / 8);

            _cs.SetPushConstant(_kernelImportChannel, "Source", sourceTexture.BindlessIndex);
            _cs.SetPushConstant(_kernelImportChannel, "Output", gpu.UAV);
            _cs.SetPushConstant(_kernelImportChannel, "BlendMode", (uint)channelIndex);
            _cs.Dispatch(_kernelImportChannel, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(gpu.Texture);
        }

        /// <summary>
        /// Ensures a ControlMap GPU texture exists for the given key.
        /// Format: R8_UNorm for Splatmap/Density, R16_Float for Height.
        /// Creates the resource + UAV/SRV on first call.
        /// </summary>
        private ControlMapGPU EnsureControlMap((ControlMapTarget, int) key, int resolution, Action<Texture> setControlMap)
        {
            if (_controlMaps.TryGetValue(key, out var existing) && existing.Resolution == resolution)
            {
                // Reuse cached wrapper — no allocation
                setControlMap?.Invoke(existing.Wrapper);
                return existing;
            }

            existing.Texture?.Release();

            var device = Engine.Device;
            var gpu = new ControlMapGPU
            {
                Resolution = resolution,
                NeedsInitialClear = true
            };

            // Height painted layers need R16_Float for signed additive data.
            // Splatmaps and density maps are 0..1 — R8_UNorm is sufficient.
            var format = key.Item1 == ControlMapTarget.Height
                ? Format.R16_Float
                : Format.R8_UNorm;

            gpu.Texture = device.CreateTexture2D(
                format, resolution, resolution, 1, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            gpu.UAV = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(gpu.Texture, null,
                new UnorderedAccessViewDescription
                {
                    Format = format,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
                }, device.GetCpuHandle(gpu.UAV));

            gpu.SRV = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(gpu.Texture,
                new ShaderResourceViewDescription
                {
                    Format = format,
                    ViewDimension = ShaderResourceViewDimension.Texture2D,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
                }, device.GetCpuHandle(gpu.SRV));

            // Create and cache the wrapper once
            gpu.Wrapper = Texture.WrapNative(gpu.Texture, gpu.SRV);
            _controlMaps[key] = gpu;
            setControlMap?.Invoke(gpu.Wrapper);

            return gpu;
        }

        private void EnsureStrokeBuffer(int requiredCount)
        {
            if (_strokeBuffer != null && _strokeBufferCapacity >= requiredCount) return;

            _strokeBuffer?.Dispose();
            _strokeBufferCapacity = Math.Max(requiredCount, 64);
            _strokeBuffer = GraphicsBuffer.CreateUpload<Vector2>(_strokeBufferCapacity, mapped: true);
        }

        /// <summary>
        /// Uploads pre-saved R16_Float baked heightmap bytes directly to the GPU texture.
        /// Used at load time when the baked heightmap was persisted to cache.
        /// </summary>
        public Texture UploadBakedHeightmap(byte[] pixels, int resolution)
        {
            if (pixels == null || pixels.Length == 0) return null;

            EnsureTexture(resolution);

            var device = Engine.Device;
            int bytesPerPixel = 2; // R16_Float
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
                    Footprint = new SubresourceFootPrint(Format.R16_Float, (uint)resolution, (uint)resolution, 1, (uint)rowPitch)
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
        /// Reads back the baked R16_Float heightmap into a CPU float[,] array.
        /// Values are normalized [0..1] — caller multiplies by MaxHeight.
        /// Returns null if no heightmap has been baked.
        /// </summary>
        public float[,] ReadbackHeightmap()
        {
            if (_heightTexture == null || _currentResolution == 0)
                return null;

            var device = Engine.Device;
            int res = _currentResolution;
            int bytesPerPixel = 2; // R16_Float
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
                    Footprint = new SubresourceFootPrint(Format.R16_Float, (uint)res, (uint)res, 1, (uint)rowPitch)
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
                            Half h = *(Half*)(rowStart + x * 2);
                            heights[x, y] = (float)h;
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
        /// Reads back the baked heightmap as raw R16_Float bytes for DDS persistence.
        /// Returns null if no heightmap has been baked.
        /// </summary>
        public byte[] ReadbackBakedHeightmapBytes()
        {
            if (_heightTexture == null || _currentResolution == 0)
                return null;

            var device = Engine.Device;
            int res = _currentResolution;
            int bytesPerPixel = 2; // R16_Float
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
                    Footprint = new SubresourceFootPrint(Format.R16_Float, (uint)res, (uint)res, 1, (uint)rowPitch)
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

        // ── ControlMap Persistence ─────────────────────────────────────────

        /// <summary>
        /// Reads back a specific ControlMap GPU texture to CPU memory as raw pixel bytes.
        /// Returns R8 (1 bpp) for Splatmap/Density, R16 (2 bpp) for Height.
        /// Returns null if the target doesn't exist.
        /// </summary>
        public byte[] ReadbackControlMap(ControlMapTarget target, int layerIndex)
        {
            if (!_controlMaps.TryGetValue((target, layerIndex), out var gpu) || gpu.Texture == null)
                return null;

            var device = Engine.Device;
            int res = gpu.Resolution;
            var format = target == ControlMapTarget.Height ? Format.R16_Float : Format.R8_UNorm;
            int bytesPerPixel = target == ControlMapTarget.Height ? 2 : 1;
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
                cmdList.ResourceBarrierTransition(gpu.Texture,
                    ResourceStates.Common, ResourceStates.CopySource);

                var src = new TextureCopyLocation(gpu.Texture, 0);
                var dst = new TextureCopyLocation(readbackResource, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(format, (uint)res, (uint)res, 1, (uint)rowPitch)
                });
                cmdList.CopyTextureRegion(dst, 0, 0, 0, src);

                cmdList.ResourceBarrierTransition(gpu.Texture,
                    ResourceStates.CopySource, ResourceStates.Common);

                cmdList.Close();
                device.SubmitAndWait(cmdList);

                unsafe
                {
                    void* pData;
                    readbackResource.Map(0, null, &pData);

                    int srcRowBytes = res * bytesPerPixel;
                    byte[] pixels = new byte[srcRowBytes * res];
                    var srcPtr = (byte*)pData;
                    for (int y = 0; y < res; y++)
                        Marshal.Copy((IntPtr)(srcPtr + y * rowPitch), pixels, y * srcRowBytes, srcRowBytes);

                    readbackResource.Unmap(0);
                    return pixels;
                }
            }
            finally
            {
                cmdList.Dispose();
                allocator.Dispose();
                readbackResource.Dispose();
            }
        }

        /// <summary>
        /// Uploads ControlMap pixel data (raw bytes from cache) to a specific target.
        /// Handles format conversion: source may be R8/R16/R32, target is R8 (splatmap/density) or R16 (height).
        /// Creates the GPU texture if needed.
        /// </summary>
        public void UploadControlMap(ControlMapTarget target, int layerIndex, byte[] pixels, int resolution, Action<Texture> setControlMap)
        {
            if (pixels == null || pixels.Length == 0) return;

            // Strip DDS header if present ("DDS " magic = 0x20534444)
            int offset = 0;
            if (pixels.Length > 128 && BitConverter.ToInt32(pixels, 0) == 0x20534444)
            {
                offset = 128;
                // Check for DX10 extended header
                if (pixels.Length > 148 && BitConverter.ToInt32(pixels, 84) == 0x30315844)
                    offset = 148;
            }

            int pixelDataLen = pixels.Length - offset;

            // Detect source format: try R8 (1 bpp), then R16 (2 bpp), then R32 (4 bpp)
            int srcBpp = 1;
            int srcRes = (int)Math.Sqrt(pixelDataLen);
            if (srcRes * srcRes != pixelDataLen)
            {
                srcBpp = 2;
                srcRes = (int)Math.Sqrt(pixelDataLen / 2);
                if (srcRes * srcRes * 2 != pixelDataLen)
                {
                    srcBpp = 4;
                    srcRes = (int)Math.Sqrt(pixelDataLen / 4);
                }
            }

            if (srcRes * srcRes * srcBpp != pixelDataLen)
            {
                Debug.LogWarning("TerrainBaker", $"UploadControlMap {target}[{layerIndex}]: cannot determine resolution from {pixelDataLen} pixel bytes (offset={offset}). Skipping.");
                return;
            }

            // Determine destination format
            int dstBpp = target == ControlMapTarget.Height ? 2 : 1;
            var dstFormat = target == ControlMapTarget.Height ? Format.R16_Float : Format.R8_UNorm;

            int res = srcRes;
            var key = (target, layerIndex);
            var gpu = EnsureControlMap(key, res, setControlMap);
            gpu.NeedsInitialClear = false;
            _controlMaps[key] = gpu;

            // Convert source data to destination format if needed
            byte[] uploadPixels;
            if (srcBpp == dstBpp)
            {
                uploadPixels = pixels;
                // offset stays as-is
            }
            else
            {
                Debug.Log($"[TerrainBaker] Converting {srcBpp}bpp→{dstBpp}bpp for {target}[{layerIndex}] ({srcRes}x{srcRes})");
                uploadPixels = new byte[res * res * dstBpp];
                for (int i = 0; i < res * res; i++)
                {
                    // Read source as float
                    float val;
                    if (srcBpp == 4)
                        val = BitConverter.ToSingle(pixels, offset + i * 4);
                    else if (srcBpp == 2)
                        val = (float)BitConverter.ToHalf(pixels, offset + i * 2);
                    else
                        val = pixels[offset + i] / 255.0f;

                    // Write to destination format
                    if (dstBpp == 2)
                    {
                        var halfBytes = BitConverter.GetBytes((Half)val);
                        uploadPixels[i * 2] = halfBytes[0];
                        uploadPixels[i * 2 + 1] = halfBytes[1];
                    }
                    else
                    {
                        uploadPixels[i] = (byte)Math.Clamp(val * 255.0f + 0.5f, 0, 255);
                    }
                }
                offset = 0;
            }

            var device = Engine.Device;
            int bytesPerPixel = dstBpp;
            int rowPitch = (res * bytesPerPixel + 255) & ~255;
            int totalBytes = rowPitch * res;

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

                    int srcRowBytes = res * bytesPerPixel;
                    var dstPtr = (byte*)pData;
                    for (int y = 0; y < res; y++)
                        Marshal.Copy(uploadPixels, offset + y * srcRowBytes, (IntPtr)(dstPtr + y * rowPitch), srcRowBytes);

                    uploadResource.Unmap(0);
                }

                cmdList.ResourceBarrierTransition(gpu.Texture,
                    ResourceStates.Common, ResourceStates.CopyDest);

                var src = new TextureCopyLocation(uploadResource, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(dstFormat, (uint)res, (uint)res, 1, (uint)rowPitch)
                });
                var dst = new TextureCopyLocation(gpu.Texture, 0);
                cmdList.CopyTextureRegion(dst, 0, 0, 0, src);

                cmdList.ResourceBarrierTransition(gpu.Texture,
                    ResourceStates.CopyDest, ResourceStates.Common);

                cmdList.Close();
                device.SubmitAndWait(cmdList);

                Debug.Log($"[TerrainBaker] Uploaded ControlMap {target}[{layerIndex}]: {res}x{res} {dstFormat} ({pixels.Length} bytes, src={srcBpp}bpp)");
            }
            finally
            {
                cmdList.Dispose();
                allocator.Dispose();
                uploadResource.Dispose();
            }
        }

        // ── RGBA Packing ─────────────────────────────────────────────

        private GraphicsBuffer _packIndexBuffer;

        /// <summary>
        /// Packs per-layer R16 ControlMaps directly into caller-owned Texture2DArray slices.
        /// sliceUAVs: per-slice UAV bindless indices (created by caller for each array slice).
        /// Stateless: no GPU resources are created or cached here.
        /// </summary>
        public void PackControlMaps(ID3D12GraphicsCommandList cmd, uint[] layerSrvIndices, uint[] sliceUAVs, int resolution)
        {
            if (layerSrvIndices == null || layerSrvIndices.Length == 0 || sliceUAVs == null) return;

            EnsureInitialized();
            var device = Engine.Device;
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            int layerCount = layerSrvIndices.Length;
            int sliceCount = (layerCount + 3) / 4;
            uint groups = (uint)((resolution + 7) / 8);

            // Ensure index buffer (4 uint per slice)
            if (_packIndexBuffer == null || _packIndexBuffer.ElementCount < 4)
            {
                _packIndexBuffer?.Dispose();
                _packIndexBuffer = GraphicsBuffer.CreateUpload<uint>(4, mapped: true);
            }

            for (int slice = 0; slice < sliceCount && slice < sliceUAVs.Length; slice++)
            {
                // Upload source indices for this slice
                int channelCount = Math.Min(4, layerCount - slice * 4);
                unsafe
                {
                    var ptr = _packIndexBuffer.WritePtr<uint>();
                    for (int c = 0; c < 4; c++)
                    {
                        int layerIdx = slice * 4 + c;
                        ptr[c] = layerIdx < layerCount ? layerSrvIndices[layerIdx] : 0;
                    }
                }

                _cs.SetPushConstant(_kernelPackChannels, "Output", sliceUAVs[slice]);
                _cs.SetBuffer(_kernelPackChannels, "StampBuf", _packIndexBuffer);
                _cs.SetPushConstant(_kernelPackChannels, "BlendMode", (uint)channelCount);
                _cs.Dispatch(_kernelPackChannels, cmd, groups, groups);
            }
        }

        public void Dispose()
        {
            _heightTexture?.Release();
            foreach (var gpu in _controlMaps.Values)
                gpu.Texture?.Release();
            _controlMaps.Clear();
            _stampBuffer?.Dispose();
            _strokeBuffer?.Dispose();
            _packIndexBuffer?.Dispose();
            _erosionHeightTex?.Release();
            _erosionWaterTex?.Release();
            _erosionSedimentTex?.Release();
            _cs?.Dispose();
        }
    }
}
