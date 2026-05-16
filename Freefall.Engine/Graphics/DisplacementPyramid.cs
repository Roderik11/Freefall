using System;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Mip pyramid for SSDM displacement vectors.
    /// Two instances needed: Pyramid A (displacement vectors) and Pyramid B (source UVs).
    /// </summary>
    public class DisplacementPyramid : IDisposable
    {
        public ID3D12Resource? Texture { get; private set; }
        public uint[] MipUAVs { get; private set; } = Array.Empty<uint>();
        public uint[] MipSRVs { get; private set; } = Array.Empty<uint>();
        public uint FullChainSrv { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int MipCount { get; private set; }

        private const int MaxMips = 4; // level 0 (full res) + 3 coarse levels

        public void Create(GraphicsDevice device, int width, int height, Format format = Format.R16G16_Float)
        {
            Texture?.Dispose();

            Width = width;
            Height = height;
            MipCount = Math.Min(MaxMips, 1 + (int)Math.Floor(Math.Log2(Math.Max(width, height))));

            Texture = device.CreateTexture2D(
                format,
                Width, Height,
                1, MipCount,
                ResourceFlags.AllowUnorderedAccess,
                ResourceStates.Common);

            var fmt = format;

            MipUAVs = new uint[MipCount];
            MipSRVs = new uint[MipCount];

            for (int i = 0; i < MipCount; i++)
            {
                MipUAVs[i] = device.AllocateBindlessIndex();
                var uavDesc = new UnorderedAccessViewDescription
                {
                    Format = fmt,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = (uint)i }
                };
                device.NativeDevice.CreateUnorderedAccessView(Texture, null, uavDesc, device.GetCpuHandle(MipUAVs[i]));

                MipSRVs[i] = device.AllocateBindlessIndex();
                var srvDesc = new ShaderResourceViewDescription
                {
                    Format = fmt,
                    ViewDimension = ShaderResourceViewDimension.Texture2D,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2D = new Texture2DShaderResourceView
                    {
                        MostDetailedMip = (uint)i,
                        MipLevels = 1
                    }
                };
                device.NativeDevice.CreateShaderResourceView(Texture, srvDesc, device.GetCpuHandle(MipSRVs[i]));
            }

            // Full-chain SRV (all mips visible, for CSRefine multi-mip sampling)
            FullChainSrv = device.AllocateBindlessIndex();
            var fullSrvDesc = new ShaderResourceViewDescription
            {
                Format = fmt,
                ViewDimension = ShaderResourceViewDimension.Texture2D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2D = new Texture2DShaderResourceView
                {
                    MostDetailedMip = 0,
                    MipLevels = (uint)MipCount
                }
            };
            device.NativeDevice.CreateShaderResourceView(Texture, fullSrvDesc, device.GetCpuHandle(FullChainSrv));
        }

        public void Dispose()
        {
            Texture?.Dispose();
            Texture = null;
        }
    }
}
