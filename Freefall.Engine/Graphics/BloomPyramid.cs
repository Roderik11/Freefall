using System;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Owns the bloom mip-chain texture and per-mip descriptors.
    /// </summary>
    public class BloomPyramid : IDisposable
    {
        public ID3D12Resource? Texture { get; private set; }
        public uint[] MipUAVs { get; private set; } = Array.Empty<uint>();
        public uint[] MipSRVs { get; private set; } = Array.Empty<uint>();
        public uint FullSRV { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int MipCount { get; private set; }

        /// <summary>
        /// Create or recreate the pyramid to match the given source dimensions.
        /// Disposes previous resources if resizing.
        /// </summary>
        public void Create(GraphicsDevice device, int sourceWidth, int sourceHeight)
        {
            // Dispose previous pyramid if resizing
            Dispose();

            // Bloom starts at half-res
            Width = Math.Max(1, sourceWidth / 2);
            Height = Math.Max(1, sourceHeight / 2);

            // Cap at 6 mips
            MipCount = Math.Min(6, 1 + (int)Math.Floor(Math.Log2(Math.Max(Width, Height))));

            // Create texture: R11G11B10_Float with UAV support
            Texture = device.CreateTexture2D(
                Format.R11G11B10_Float,
                Width, Height,
                1, MipCount,
                ResourceFlags.AllowUnorderedAccess,
                ResourceStates.Common);

            // Allocate per-mip UAVs and SRVs
            MipUAVs = new uint[MipCount];
            MipSRVs = new uint[MipCount];

            for (int i = 0; i < MipCount; i++)
            {
                // UAV for writing this mip level
                MipUAVs[i] = device.AllocateBindlessIndex();
                var uavDesc = new UnorderedAccessViewDescription
                {
                    Format = Format.R11G11B10_Float,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = (uint)i }
                };
                device.NativeDevice.CreateUnorderedAccessView(Texture, null, uavDesc, device.GetCpuHandle(MipUAVs[i]));

                // SRV for reading this mip level
                MipSRVs[i] = device.AllocateBindlessIndex();
                var srvDesc = new ShaderResourceViewDescription
                {
                    Format = Format.R11G11B10_Float,
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

            // Full-pyramid SRV (all mips)
            FullSRV = device.AllocateBindlessIndex();
            var fullSrvDesc = new ShaderResourceViewDescription
            {
                Format = Format.R11G11B10_Float,
                ViewDimension = ShaderResourceViewDimension.Texture2D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2D = new Texture2DShaderResourceView
                {
                    MostDetailedMip = 0,
                    MipLevels = (uint)MipCount
                }
            };
            device.NativeDevice.CreateShaderResourceView(Texture, fullSrvDesc, device.GetCpuHandle(FullSRV));
        }

        public void Dispose()
        {
            Texture?.Dispose();
            Texture = null;

            foreach (var idx in MipUAVs)
                Engine.Device.ReleaseBindlessIndex(idx);

            foreach (var idx in MipSRVs)
                Engine.Device.ReleaseBindlessIndex(idx);

            if (FullSRV != 0)
            {
                Engine.Device.ReleaseBindlessIndex(FullSRV);
                FullSRV = 0;
            }

            MipUAVs = Array.Empty<uint>();
            MipSRVs = Array.Empty<uint>();
        }
    }
}
