using System;
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
        private ComputeShader _cs;
        private int _kernelClear;
        private int _kernelImport;
        private int _kernelStampGroup;
        private int _kernelPaintBrush;
        private int _kernelClearDelta;

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

        // ── Brush Painting ────────────────────────────────────────────────

        // ControlMap UAV resources (R16_Float, created on first brush stroke)
        private ID3D12Resource _controlMapTexture;
        private uint _controlMapUAV;
        private uint _controlMapSRV;
        private int _controlMapResolution;
        private bool _controlMapNeedsInitialClear = true;

        /// <summary>
        /// Dispatches a brush stroke on the PaintHeightLayer's ControlMap.
        /// Points are in terrain UV space [0..1]. Creates the ControlMap if needed.
        /// Must be called on the render thread with an active command list.
        /// </summary>
        public void PaintBrush(Terrain terrain, PaintHeightLayer layer,
                               ID3D12GraphicsCommandList cmd,
                               Vector2[] strokePoints, int pointCount,
                               BrushMode mode, float strength,
                               float radius, float falloff,
                               float targetHeight = 0)
        {
            if (pointCount == 0 || strokePoints == null) return;

            EnsureInitialized();
            int res = terrain.HeightmapResolution;
            EnsureControlMap(layer, res);

            var device = Engine.Device;
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            uint groups = (uint)((res + 7) / 8);

            // Clear on first use
            if (_controlMapNeedsInitialClear)
            {
                Debug.Log("[TerrainBaker] Clearing ControlMap (first use)");
                _cs.SetPushConstant(_kernelClearDelta, "Output", _controlMapUAV);
                _cs.Dispatch(_kernelClearDelta, cmd, groups, groups);
                cmd.ResourceBarrierUnorderedAccessView(_controlMapTexture);
                _controlMapNeedsInitialClear = false;
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

            Debug.Log($"[TerrainBaker] PaintBrush: mode={mode} pts={pointCount} uvR={uvRadius:F4} str={strength} controlMapUAV={_controlMapUAV} heightSRV={_heightSRV}");

            // Push constants (reusing existing PushConstants layout)
            _cs.SetPushConstant(_kernelPaintBrush, "Source", _heightSRV);  // Current baked heightmap for flatten/smooth
            _cs.SetPushConstant(_kernelPaintBrush, "Output", _controlMapUAV);   // ControlMap UAV
            _cs.SetBuffer(_kernelPaintBrush, "StampBuf", _strokeBuffer);
            _cs.SetPushConstant(_kernelPaintBrush, "BlendMode", (uint)mode);
            _cs.SetParam(_kernelPaintBrush, "Opacity", strength);
            _cs.SetPushConstant(_kernelPaintBrush, "StampCount", (uint)pointCount);

            // BrushParams cbuffer
            _cs.SetParam(_kernelPaintBrush, "BrushRadius", uvRadius);
            _cs.SetParam(_kernelPaintBrush, "BrushFalloff", falloff);
            _cs.SetParam(_kernelPaintBrush, "BrushTargetHeight", normalizedTarget);

            _cs.Dispatch(_kernelPaintBrush, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(_controlMapTexture);
        }

        /// <summary>
        /// Clears the ControlMap to zero (removes all paint edits).
        /// </summary>
        public void ClearControlMap(Terrain terrain, PaintHeightLayer layer, ID3D12GraphicsCommandList cmd)
        {
            if (_controlMapTexture == null) return;

            EnsureInitialized();
            var device = Engine.Device;
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });

            int res = terrain.HeightmapResolution;
            uint groups = (uint)((res + 7) / 8);

            _cs.SetPushConstant(_kernelClearDelta, "Output", _controlMapUAV);
            _cs.Dispatch(_kernelClearDelta, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(_controlMapTexture);
        }

        /// <summary>
        /// Ensures a ControlMap R16_Float texture exists for the paint layer.
        /// Creates the GPU resource + UAV/SRV on first call.
        /// </summary>
        private void EnsureControlMap(PaintHeightLayer layer, int resolution)
        {
            if (_controlMapTexture != null && _controlMapResolution == resolution)
            {
                // Already created — just make sure layer has the reference
                if (layer.ControlMap == null || layer.ControlMap.BindlessIndex != _controlMapSRV)
                    layer.ControlMap = Texture.WrapNative(_controlMapTexture, _controlMapSRV);
                return;
            }

            _controlMapTexture?.Release();

            var device = Engine.Device;
            _controlMapTexture = device.CreateTexture2D(
                Format.R16_Float, resolution, resolution, 1, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _controlMapUAV = device.AllocateBindlessIndex();
            var uavDesc = new UnorderedAccessViewDescription
            {
                Format = Format.R16_Float,
                ViewDimension = UnorderedAccessViewDimension.Texture2D,
                Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
            };
            device.NativeDevice.CreateUnorderedAccessView(_controlMapTexture, null, uavDesc, device.GetCpuHandle(_controlMapUAV));

            _controlMapSRV = device.AllocateBindlessIndex();
            var srvDesc = new ShaderResourceViewDescription
            {
                Format = Format.R16_Float,
                ViewDimension = ShaderResourceViewDimension.Texture2D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
            };
            device.NativeDevice.CreateShaderResourceView(_controlMapTexture, srvDesc, device.GetCpuHandle(_controlMapSRV));

            _controlMapResolution = resolution;
            layer.ControlMap = Texture.WrapNative(_controlMapTexture, _controlMapSRV);

            // Clear the new texture
            // Note: Cleared on next Bake since we might not have a cmd list here
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
        /// Reads back the ControlMap GPU texture to CPU memory as raw pixel bytes.
        /// Returns null if no ControlMap has been created yet.
        /// This is a synchronous GPU operation — call from the main thread only.
        /// </summary>
        public byte[] ReadbackControlMap()
        {
            if (_controlMapTexture == null || _controlMapResolution == 0)
                return null;

            var device = Engine.Device;
            int res = _controlMapResolution;
            int bytesPerPixel = 2; // R16_Float = 2 bytes
            int rowPitch = (res * bytesPerPixel + 255) & ~255; // 256-byte aligned
            int totalBytes = rowPitch * res;

            // Create a readback heap resource
            var readbackResource = device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Readback),
                HeapFlags.None,
                ResourceDescription.Buffer((ulong)totalBytes),
                ResourceStates.CopyDest,
                null);

            // Create a Direct command allocator + list for barriers + copy
            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            var cmdList = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(
                0, CommandListType.Direct, allocator, null);

            try
            {
                // Transition control map texture to CopySource
                cmdList.ResourceBarrierTransition(_controlMapTexture,
                    ResourceStates.Common, ResourceStates.CopySource);

                // Copy texture to readback buffer
                var src = new TextureCopyLocation(_controlMapTexture, 0);
                var dst = new TextureCopyLocation(readbackResource, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R16_Float, (uint)res, (uint)res, 1, (uint)rowPitch)
                });
                cmdList.CopyTextureRegion(dst, 0, 0, 0, src);

                // Transition back to Common (UAV-compatible)
                cmdList.ResourceBarrierTransition(_controlMapTexture,
                    ResourceStates.CopySource, ResourceStates.Common);

                // Close, submit, and GPU-wait
                cmdList.Close();
                device.SubmitAndWait(cmdList);

                // Read from the mapped readback buffer
                unsafe
                {
                    void* pData;
                    readbackResource.Map(0, null, &pData);

                    // Copy rows, stripping alignment padding
                    int srcRowBytes = res * bytesPerPixel;
                    byte[] pixels = new byte[srcRowBytes * res];
                    var srcPtr = (byte*)pData;
                    for (int y = 0; y < res; y++)
                    {
                        Marshal.Copy((IntPtr)(srcPtr + y * rowPitch), pixels, y * srcRowBytes, srcRowBytes);
                    }

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
        /// Uploads ControlMap pixel data (raw bytes from cache) to the GPU
        /// and wires up the SRV so the bake shader can read it.
        /// Creates the ControlMap texture if needed.
        /// </summary>
        public void UploadControlMap(byte[] pixels, int resolution, PaintHeightLayer layer)
        {
            if (pixels == null || pixels.Length == 0) return;

            int res = resolution;
            int bpp = 2; // R16_Float = 2 bytes per pixel

            // Create the ControlMap GPU texture (UAV + SRV)
            EnsureControlMap(layer, res);
            _controlMapNeedsInitialClear = false; // we're uploading saved data, no clear needed

            var device = Engine.Device;
            int bytesPerPixel = 2; // R16_Float
            int rowPitch = (res * bytesPerPixel + 255) & ~255; // 256-byte aligned
            int totalBytes = rowPitch * res;

            // Create upload heap resource
            var uploadResource = device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Upload),
                HeapFlags.None,
                ResourceDescription.Buffer((ulong)totalBytes),
                ResourceStates.GenericRead,
                null);

            // Create a Direct command allocator + list for barriers + copy
            var allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            var cmdList = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(
                0, CommandListType.Direct, allocator, null);

            try
            {
                // Fill upload buffer (with row pitch alignment padding)
                unsafe
                {
                    void* pData;
                    uploadResource.Map(0, null, &pData);

                    int srcRowBytes = res * bytesPerPixel;
                    var dstPtr = (byte*)pData;
                    for (int y = 0; y < res; y++)
                    {
                        Marshal.Copy(pixels, y * srcRowBytes, (IntPtr)(dstPtr + y * rowPitch), srcRowBytes);
                    }

                    uploadResource.Unmap(0);
                }

                // GPU copy: upload buffer → control map texture
                cmdList.ResourceBarrierTransition(_controlMapTexture,
                    ResourceStates.Common, ResourceStates.CopyDest);

                var src = new TextureCopyLocation(uploadResource, new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R16_Float, (uint)res, (uint)res, 1, (uint)rowPitch)
                });
                var dst = new TextureCopyLocation(_controlMapTexture, 0);
                cmdList.CopyTextureRegion(dst, 0, 0, 0, src);

                // Transition back to Common (UAV-compatible)
                cmdList.ResourceBarrierTransition(_controlMapTexture,
                    ResourceStates.CopyDest, ResourceStates.Common);

                // Close, submit, and GPU-wait
                cmdList.Close();
                device.SubmitAndWait(cmdList);

                Debug.Log($"[TerrainBaker] Uploaded ControlMap: {res}x{res} ({pixels.Length} bytes)");
            }
            finally
            {
                cmdList.Dispose();
                allocator.Dispose();
                uploadResource.Dispose();
            }
        }

        public void Dispose()
        {
            _heightTexture?.Release();
            _controlMapTexture?.Release();
            _stampBuffer?.Dispose();
            _strokeBuffer?.Dispose();
            _cs?.Dispose();
        }
    }
}
