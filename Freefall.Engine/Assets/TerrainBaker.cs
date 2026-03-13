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
    /// Output: R32_Float heightmap at configurable resolution.
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

        private bool _initialized;

        /// <summary>Brush mode for CS_PaintBrush dispatch.</summary>
        public enum BrushMode : uint { Raise = 0, Lower = 1, Flatten = 2, Smooth = 3 }

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
            }

            // Phase 3: Stamp groups (after layers)
            foreach (var group in terrain.StampGroups)
            {
                if (!group.Enabled || group.Instances.Count == 0 || group.Brush == null) continue;
                DispatchStampGroup(cmd, group, groups);
            }

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

            Debug.Log($"[TerrainBaker] DispatchPaint: ControlMap SRV={layer.ControlMap.BindlessIndex} BlendMode={layer.BlendMode}");
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
                Format.R32_Float, resolution, resolution, 1, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _heightUAV = device.AllocateBindlessIndex();
            var uavDesc = new UnorderedAccessViewDescription
            {
                Format = Format.R32_Float,
                ViewDimension = UnorderedAccessViewDimension.Texture2D,
                Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
            };
            device.NativeDevice.CreateUnorderedAccessView(_heightTexture, null, uavDesc, device.GetCpuHandle(_heightUAV));

            _heightSRV = device.AllocateBindlessIndex();
            var srvDesc = new ShaderResourceViewDescription
            {
                Format = Format.R32_Float,
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
                               BrushMode mode, float strength,
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
            _cs.SetPushConstant(_kernelPaintBrush, "BlendMode", (uint)mode);
            _cs.SetParam(_kernelPaintBrush, "Opacity", strength);
            _cs.SetPushConstant(_kernelPaintBrush, "StampCount", (uint)pointCount);

            _cs.SetParam(_kernelPaintBrush, "BrushRadius", uvRadius);
            _cs.SetParam(_kernelPaintBrush, "BrushFalloff", falloff);
            _cs.SetParam(_kernelPaintBrush, "BrushTargetHeight", normalizedTarget);

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
        /// Ensures a ControlMap R16_Float GPU texture exists for the given key.
        /// Creates the resource + UAV/SRV on first call.
        /// </summary>
        private ControlMapGPU EnsureControlMap((ControlMapTarget, int) key, int resolution, Action<Texture> setControlMap)
        {
            if (_controlMaps.TryGetValue(key, out var existing) && existing.Resolution == resolution)
            {
                setControlMap?.Invoke(Texture.WrapNative(existing.Texture, existing.SRV));
                return existing;
            }

            existing.Texture?.Release();

            var device = Engine.Device;
            var gpu = new ControlMapGPU
            {
                Resolution = resolution,
                NeedsInitialClear = true
            };

            gpu.Texture = device.CreateTexture2D(
                Format.R16_Float, resolution, resolution, 1, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            gpu.UAV = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(gpu.Texture, null,
                new UnorderedAccessViewDescription
                {
                    Format = Format.R16_Float,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
                }, device.GetCpuHandle(gpu.UAV));

            gpu.SRV = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(gpu.Texture,
                new ShaderResourceViewDescription
                {
                    Format = Format.R16_Float,
                    ViewDimension = ShaderResourceViewDimension.Texture2D,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
                }, device.GetCpuHandle(gpu.SRV));

            _controlMaps[key] = gpu;
            setControlMap?.Invoke(Texture.WrapNative(gpu.Texture, gpu.SRV));

            return gpu;
        }

        private void EnsureStrokeBuffer(int requiredCount)
        {
            if (_strokeBuffer != null && _strokeBufferCapacity >= requiredCount) return;

            _strokeBuffer?.Dispose();
            _strokeBufferCapacity = Math.Max(requiredCount, 64);
            _strokeBuffer = GraphicsBuffer.CreateUpload<Vector2>(_strokeBufferCapacity, mapped: true);
        }

        // ── ControlMap Persistence ─────────────────────────────────────────

        /// <summary>
        /// Reads back a specific ControlMap GPU texture to CPU memory as raw pixel bytes.
        /// Returns null if the target doesn't exist.
        /// </summary>
        public byte[] ReadbackControlMap(ControlMapTarget target, int layerIndex)
        {
            if (!_controlMaps.TryGetValue((target, layerIndex), out var gpu) || gpu.Texture == null)
                return null;

            var device = Engine.Device;
            int res = gpu.Resolution;
            int bytesPerPixel = 2; // R16_Float = 2 bytes
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
                    Footprint = new SubresourceFootPrint(Format.R16_Float, (uint)res, (uint)res, 1, (uint)rowPitch)
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
        /// Creates the GPU texture if needed.
        /// </summary>
        public void UploadControlMap(ControlMapTarget target, int layerIndex, byte[] pixels, int resolution, Action<Texture> setControlMap)
        {
            if (pixels == null || pixels.Length == 0) return;

            int res = resolution;
            var key = (target, layerIndex);
            var gpu = EnsureControlMap(key, res, setControlMap);
            gpu.NeedsInitialClear = false;
            _controlMaps[key] = gpu;

            var device = Engine.Device;
            int bytesPerPixel = 2;
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
                        Marshal.Copy(pixels, y * srcRowBytes, (IntPtr)(dstPtr + y * rowPitch), srcRowBytes);

                    uploadResource.Unmap(0);
                }

                cmdList.ResourceBarrierTransition(gpu.Texture,
                    ResourceStates.Common, ResourceStates.CopyDest);

                var src = new TextureCopyLocation(uploadResource, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R16_Float, (uint)res, (uint)res, 1, (uint)rowPitch)
                });
                var dst = new TextureCopyLocation(gpu.Texture, 0);
                cmdList.CopyTextureRegion(dst, 0, 0, 0, src);

                cmdList.ResourceBarrierTransition(gpu.Texture,
                    ResourceStates.CopyDest, ResourceStates.Common);

                cmdList.Close();
                device.SubmitAndWait(cmdList);

                Debug.Log($"[TerrainBaker] Uploaded ControlMap {target}[{layerIndex}]: {res}x{res} ({pixels.Length} bytes)");
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
        /// Packs per-layer R16 ControlMaps into ceil(N/4) RGBA slices for the shader.
        /// layers: SRV bindless indices of the R16 ControlMaps (0 = no data for that layer).
        /// Returns a list of packed RGBA textures suitable for Texture.CreateTexture2DArray.
        /// </summary>
        public List<Texture> PackControlMaps(ID3D12GraphicsCommandList cmd, uint[] layerSrvIndices, int resolution)
        {
            if (layerSrvIndices == null || layerSrvIndices.Length == 0)
                return new List<Texture>();

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

            var result = new List<Texture>(sliceCount);

            for (int slice = 0; slice < sliceCount; slice++)
            {
                // Create RGBA texture for this slice
                var rgbaTexture = device.CreateTexture2D(
                    Format.R8G8B8A8_UNorm, resolution, resolution, 1, 1,
                    ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

                uint uav = device.AllocateBindlessIndex();
                device.NativeDevice.CreateUnorderedAccessView(rgbaTexture, null,
                    new UnorderedAccessViewDescription
                    {
                        Format = Format.R8G8B8A8_UNorm,
                        ViewDimension = UnorderedAccessViewDimension.Texture2D,
                        Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
                    }, device.GetCpuHandle(uav));

                uint srv = device.AllocateBindlessIndex();
                device.NativeDevice.CreateShaderResourceView(rgbaTexture,
                    new ShaderResourceViewDescription
                    {
                        Format = Format.R8G8B8A8_UNorm,
                        ViewDimension = ShaderResourceViewDimension.Texture2D,
                        Shader4ComponentMapping = ShaderComponentMapping.Default,
                        Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
                    }, device.GetCpuHandle(srv));

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

                _cs.SetPushConstant(_kernelPackChannels, "Output", uav);
                _cs.SetBuffer(_kernelPackChannels, "StampBuf", _packIndexBuffer);
                _cs.SetPushConstant(_kernelPackChannels, "BlendMode", (uint)channelCount);
                _cs.Dispatch(_kernelPackChannels, cmd, groups, groups);
                cmd.ResourceBarrierUnorderedAccessView(rgbaTexture);

                result.Add(Texture.WrapNative(rgbaTexture, srv));
            }

            return result;
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
            _cs?.Dispose();
        }
    }
}
