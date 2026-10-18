using System;
using System.Numerics;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// The sky probe (ATMOSPHERE_PLAN stage 4): the sky round the camera as a cubemap, every frame, with the cloud layer
    /// and the lit ground below the horizon (sky_probe_cs.hlsl), its mips for blurred reflections and an SH9 of it for
    /// the ambient light (sky_common.fx SkyAmbient, SkyReflection). Made by <see cref="Atmosphere.Render"/> after the
    /// ambient kernel, whose ground light it shows below the horizon. Docs: physical_atmosphere.md §12.
    /// </summary>
    internal static class SkyProbe
    {
        public const int Size = 64;                 // texels per face edge at mip 0
        public const int Mips = 7;                  // 64 .. 1
        private const int SHMip = 2;                // 16²: the SH is made from it (1536 texels)
        private const int SHCoefficients = 9;

        private static ComputeShader? _cs;
        private static int _kFaces, _kDownsample, _kSH;
        private static ID3D12Resource? _cube;
        private static uint[] _mipUavs = Array.Empty<uint>();
        private static uint _cubeSrv;
        private static GraphicsBuffer? _sh;
        private static bool _written;               // first use: the cube starts in Common

        /// <summary>Whether the probe is made and read (Engine.Settings.SkyProbe); off = the hemisphere ambient.</summary>
        public static bool Enabled => Engine.Settings.SkyProbe;

        /// <summary>For the atmosphere's parameters: the cube (TextureCube SRV), the SH buffer, the mip count; zeros when off.</summary>
        public static (uint Cube, uint SH, float Mips) Indices
        {
            get
            {
                if (!Enabled) return (0, 0, 0);
                EnsureResources();
                return (_cubeSrv, _sh!.SrvIndex, Mips);
            }
        }

        private static void EnsureResources()
        {
            if (_cs != null) return;
            var device = Engine.Device;
            _cs = new ComputeShader("sky_probe_cs.hlsl");
            _kFaces = _cs.FindKernel("CSProbeFaces");
            _kDownsample = _cs.FindKernel("CSProbeDownsample");
            _kSH = _cs.FindKernel("CSProbeSH");

            // RGBA16F: pre-exposed light, as the HDR targets hold it
            const Format format = Format.R16G16B16A16_Float;
            _cube = device.CreateTexture2D(format, Size, Size, 6, Mips, ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
            _mipUavs = new uint[Mips];
            for (int mip = 0; mip < Mips; mip++)
            {
                _mipUavs[mip] = device.AllocateBindlessIndex();
                device.NativeDevice.CreateUnorderedAccessView(_cube, null, new UnorderedAccessViewDescription
                {
                    Format = format,
                    ViewDimension = UnorderedAccessViewDimension.Texture2DArray,
                    Texture2DArray = new Texture2DArrayUnorderedAccessView { MipSlice = (uint)mip, FirstArraySlice = 0, ArraySize = 6 }
                }, device.GetCpuHandle(_mipUavs[mip]));
            }
            _cubeSrv = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(_cube, new ShaderResourceViewDescription
            {
                Format = format,
                ViewDimension = ShaderResourceViewDimension.TextureCube,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                TextureCube = new TextureCubeShaderResourceView { MostDetailedMip = 0, MipLevels = Mips }
            }, device.GetCpuHandle(_cubeSrv));

            _sh = GraphicsBuffer.CreateStructured<Vector4>(SHCoefficients, srv: true, uav: true);
        }

        /// <summary>
        /// Make the probe for the view being drawn: faces, mips, SH. Compute root signature, heap and scene constants must
        /// be bound (Atmosphere.Render), and the ambient buffer readable.
        /// </summary>
        public static void Render(ID3D12GraphicsCommandList list)
        {
            if (!Enabled) return;
            EnsureResources();
            var cs = _cs!;
            var shaderRead = ResourceStates.NonPixelShaderResource | ResourceStates.PixelShaderResource;

            PixMarker.Begin(list, "Sky probe");
            list.ResourceBarrierTransition(_cube!, _written ? shaderRead : ResourceStates.Common, ResourceStates.UnorderedAccess);

            cs.SetPushConstant("OutputUAV", _mipUavs[0]);
            cs.SetPushConstant("Size", (uint)Size);
            cs.Dispatch(_kFaces, list, (Size + 7) / 8, (Size + 7) / 8, 6);
            UavBarrier(list);

            for (int mip = 1; mip < Mips; mip++)
            {
                int size = Size >> mip;
                cs.SetPushConstant("OutputUAV", _mipUavs[mip]);
                cs.SetPushConstant("SourceUAV", _mipUavs[mip - 1]);
                cs.SetPushConstant("Size", (uint)size);
                cs.Dispatch(_kDownsample, list, (uint)(size + 7) / 8, (uint)(size + 7) / 8, 6);
                UavBarrier(list);
            }

            _sh!.Transition(list, ResourceStates.UnorderedAccess);
            cs.SetPushConstant("SourceUAV", _mipUavs[SHMip]);
            cs.SetPushConstant("Size", (uint)(Size >> SHMip));
            cs.SetPushConstant("SHUAV", _sh!.UavIndex);
            cs.Dispatch(_kSH, list, 1);

            list.ResourceBarrierTransition(_cube, ResourceStates.UnorderedAccess, shaderRead);
            _sh.Transition(list, shaderRead);
            _written = true;
            PixMarker.End(list);
        }

        private static void UavBarrier(ID3D12GraphicsCommandList list) =>
            list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
    }
}
