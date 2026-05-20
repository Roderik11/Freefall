using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// GPU-only 3D texture with SRV (for sampling) and UAV (for compute write).
    /// Used for precomputed noise LUTs, volume textures, etc.
    /// </summary>
    public class RenderTexture3D : Texture
    {
        public uint UavIndex { get; private set; }

        public RenderTexture3D(GraphicsDevice device, int width, int height, int depth, Format format)
        {
            _resource = device.CreateTexture3D(format, width, height, depth, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            // SRV for shader sampling
            var cpuSrv = device.AllocateSrv(out var gpuSrv, out uint srvIndex);
            SrvHandle = gpuSrv;
            SrvCpuHandle = cpuSrv;
            BindlessIndex = srvIndex;
            device.NativeDevice.CreateShaderResourceView(_resource, new ShaderResourceViewDescription
            {
                Format = format,
                ViewDimension = ShaderResourceViewDimension.Texture3D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture3D = new Texture3DShaderResourceView { MipLevels = 1, MostDetailedMip = 0 }
            }, cpuSrv);

            // UAV for compute write
            UavIndex = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(_resource, null,
                new UnorderedAccessViewDescription
                {
                    Format = format,
                    ViewDimension = UnorderedAccessViewDimension.Texture3D,
                    Texture3D = new Texture3DUnorderedAccessView
                    {
                        MipSlice = 0,
                        FirstWSlice = 0,
                        WSize = (uint)depth
                    }
                },
                device.GetCpuHandle(UavIndex));
        }

        public new void Dispose()
        {
            var device = Engine.Device;
            if (BindlessIndex != 0)
            {
                device.ReleaseBindlessIndex(BindlessIndex);
                BindlessIndex = 0;
            }
            if (UavIndex != 0)
            {
                device.ReleaseBindlessIndex(UavIndex);
                UavIndex = 0;
            }
            base.Dispose();
        }
    }
}
