using System;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    public class BufferTexture : Texture
    {
        public BufferTexture(GraphicsDevice device, ID3D12Resource buffer, int elementCount, int stride)
        {
            _resource = buffer;
            
            // Allocate Bindless Index
            BindlessIndex = device.AllocateBindlessIndex();
            SrvCpuHandle = device.GetCpuHandle(BindlessIndex);
            SrvHandle = device.GetGpuHandle(BindlessIndex);

            var srvDesc = new ShaderResourceViewDescription
            {
                ViewDimension = ShaderResourceViewDimension.Buffer,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Buffer = new BufferShaderResourceView
                {
                    FirstElement = 0,
                    NumElements = (uint)elementCount,
                    StructureByteStride = (uint)stride,
                    Flags = BufferShaderResourceViewFlags.None
                }
            };

            device.NativeDevice.CreateShaderResourceView(buffer, srvDesc, SrvCpuHandle);
        }

        // Override Dispose to avoid double disposal if we don't own the resource?
        // Texture base disposes _resource. 
        // If Bucket owns logic, passing ownership here is fine.
    }
}
