using System;
using System.Numerics;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Where the surf's breaking lips stand round the camera, decided once per frame (ocean_lip_map.hlsl): per texel
    /// the lip amount of the crest the point's strip belongs to (LipStripAmount) and the phase's mean rate back to it
    /// (k), smoothed over a few metres. The ocean's PS leaves its face out where the amount is above LIP_MIN and the
    /// lip pass (ocean_lip.fx) bends its strip by the same texels, so the two agree by construction. Decided per
    /// pixel and per vertex instead, wherever the crest search changed faster than the vertices lay apart (crests
    /// wrapping into a cove, steep fronts by the shore) the two parted: the seabed showed through slivers between
    /// them and the sheet stood in teeth and scallops.
    ///
    /// <see cref="Size"/>² texels of <see cref="Texel"/> m, snapped to whole texels round the camera (the lip's grid
    /// and its fade reach 128 m). RGBA16F: amount, k (0 outside a strip), the crest's age (LipAmount), unused. OceanRenderer
    /// runs it after the FFT.
    /// </summary>
    public sealed class LipMap : IDisposable
    {
        /// <summary>Texels along a side (LIP_MAP_SIZE in ocean_surf.hlsli).</summary>
        public const int Size = 256;

        /// <summary>A texel's size in metres (LIP_MAP_TEXEL).</summary>
        public const float Texel = 1.0f;

        private ID3D12Resource? _raw, _map;
        private uint _rawUav, _rawSrv, _mapUav, _mapSrv;
        private ComputeShader? _shader;
        private int _kBuild, _kSmooth;
        private ID3D12DescriptorHeap[]? _heaps;

        /// <summary>Bindless SRV of the smoothed map; 0 until <see cref="EnsureCreated"/>.</summary>
        public uint Srv => _mapSrv;

        /// <summary>The map's corner (world xz) round a camera: snapped to whole texels, or it would swim.</summary>
        public static Vector2 OriginFor(Vector3 camera)
        {
            float half = Size * Texel * 0.5f;
            return new Vector2(MathF.Floor(camera.X / Texel) * Texel - half, MathF.Floor(camera.Z / Texel) * Texel - half);
        }

        /// <summary>Creates the textures and kernels the first time. Main thread.</summary>
        public void EnsureCreated(GraphicsDevice device)
        {
            if (_map != null) return;
            _raw = CreateTexture(device, out _rawUav, out _rawSrv);
            _map = CreateTexture(device, out _mapUav, out _mapSrv);
            _shader = new ComputeShader("ocean_lip_map.hlsl");
            _kBuild = _shader.FindKernel("CSLipMapBuild");
            _kSmooth = _shader.FindKernel("CSLipMapSmooth");
            _heaps = new[] { device.SrvHeap };
        }

        private static ID3D12Resource CreateTexture(GraphicsDevice device, out uint uav, out uint srv)
        {
            var texture = device.CreateTexture2D(Format.R16G16B16A16_Float, Size, Size, 1, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
            uav = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(texture, null, new UnorderedAccessViewDescription
            {
                Format = Format.R16G16B16A16_Float,
                ViewDimension = UnorderedAccessViewDimension.Texture2D,
                Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
            }, device.GetCpuHandle(uav));
            srv = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(texture, new ShaderResourceViewDescription
            {
                Format = Format.R16G16B16A16_Float,
                ViewDimension = ShaderResourceViewDimension.Texture2D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
            }, device.GetCpuHandle(srv));
            return texture;
        }

        /// <summary>
        /// Decide and smooth the map from this frame's OceanData (a StructuredBuffer of one, its LipMapOrigin set from
        /// <see cref="OriginFor"/>). After the FFT, before the ocean draws.
        /// </summary>
        public void Dispatch(ID3D12GraphicsCommandList cmd, GraphicsDevice device, uint oceanDataSrv)
        {
            if (_shader == null || _raw == null || _map == null || oceanDataSrv == 0) return;
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, _heaps!);
            uint groups = (uint)((Size + 7) / 8);

            _shader.SetPushConstant("OceanData", oceanDataSrv);
            _shader.SetPushConstant("RawUAV", _rawUav);
            _shader.SetPushConstant("RawSRV", _rawSrv);
            _shader.SetPushConstant("MapUAV", _mapUav);
            _shader.Dispatch(_kBuild, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(_raw);
            _shader.Dispatch(_kSmooth, cmd, groups, groups);
            cmd.ResourceBarrierUnorderedAccessView(_map);
        }

        /// <summary>Deferred: the frames in flight still read the map and run its kernels.</summary>
        public void Dispose()
        {
            var device = Engine.Device;
            device.DeferDispose(_raw);
            device.DeferDispose(_map);
            device.DeferReleaseBindlessIndex(_rawUav);
            device.DeferReleaseBindlessIndex(_rawSrv);
            device.DeferReleaseBindlessIndex(_mapUav);
            device.DeferReleaseBindlessIndex(_mapSrv);
            device.DeferDispose(_shader);
            _raw = _map = null;
            _shader = null;
            _rawUav = _rawSrv = _mapUav = _mapSrv = 0;
        }
    }
}
