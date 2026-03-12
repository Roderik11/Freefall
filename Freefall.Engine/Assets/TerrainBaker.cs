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

        // GPU resources for the baked heightmap
        private ID3D12Resource _heightTexture;
        private uint _heightUAV;
        private uint _heightSRV;
        private int _currentResolution;

        // Stamp instance upload buffer (reused across groups)
        private GraphicsBuffer _stampBuffer;
        private int _stampBufferCapacity;

        private bool _initialized;

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

        public void Dispose()
        {
            _heightTexture?.Release();
            _stampBuffer?.Dispose();
            _cs?.Dispose();
        }
    }
}
