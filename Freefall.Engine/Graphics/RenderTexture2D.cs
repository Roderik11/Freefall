using System;
using System.Collections.Generic;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    public class RenderTexture2D : Texture
    {
        public CpuDescriptorHandle RtvHandle { get; private set; }
        
        // Stack logic removed for simplicity in initial port
        
        public RenderTexture2D(GraphicsDevice device, int width, int height, Format format, bool useMultiSampling = false, bool randomWrite = false)
        {
            ResourceFlags flags = ResourceFlags.AllowRenderTarget;
            // if (randomWrite) flags |= ResourceFlags.AllowUnorderedAccess; // TODO: UAV support later

            // Must start in Common state when no optimized clear value is provided
            _resource = device.CreateTexture2D(format, width, height, 1, 1, flags, ResourceStates.Common, null);

            // RTV
            RtvHandle = device.AllocateRtv();
            device.CreateRenderTargetView(_resource, new RenderTargetViewDescription
            {
                Format = format,
                ViewDimension = RenderTargetViewDimension.Texture2D
            }, RtvHandle);

            // SRV
            var cpuSrv = device.AllocateSrv(out var gpuSrv, out uint index);
            SrvHandle = gpuSrv;
            SrvCpuHandle = cpuSrv;
            BindlessIndex = index;
            device.CreateShaderResourceView(_resource, null, cpuSrv);
        }
    }

    public class DepthTexture2D : Texture
    {
        public CpuDescriptorHandle DsvHandle { get; private set; }

        // Typeless for SRV reading of Depth? DX12 often needs Type-less resource, specific view formats
        // Apex uses R32_Typeless for Res, D32_Float for DSV, R32_Float for SRV
        // The original constructor is removed        public CpuDescriptorHandle DsvHandle { get; private set; }

        public DepthTexture2D(int width, int height, Format format = Format.D32_Float, bool shaderResource = false) 
            : base() // Invoke protected ctor
        {
            var device = Engine.Device;
            Format resFormat = format;
            if (shaderResource)
            {
                if (format == Format.D32_Float) resFormat = Format.R32_Typeless;
                else if (format == Format.D16_UNorm) resFormat = Format.R16_Typeless;
                else if (format == Format.D24_UNorm_S8_UInt) resFormat = Format.R24G8_Typeless;
            }

             // Typeless resources must start in Common state, transition to DepthWrite before use
             var initialState = shaderResource ? ResourceStates.Common : ResourceStates.DepthWrite;
             // Don't pass explicit clearValue - GraphicsDevice.CreateTexture2D handles depth clear values automatically
             _resource = device.CreateTexture2D(resFormat, width, height, 1, 1, ResourceFlags.AllowDepthStencil, initialState, null);

            // Create DSV
            var dsvDesc = new DepthStencilViewDescription
            {
                Format = format,
                ViewDimension = DepthStencilViewDimension.Texture2D,
                Texture2D = new Texture2DDepthStencilView { MipSlice = 0 }
            };

            DsvHandle = device.AllocateDsv();
            device.NativeDevice.CreateDepthStencilView(_resource, dsvDesc, DsvHandle);

            // Create SRV if requested
            if (shaderResource)
            {
                var cpuHandle = device.AllocateSrv(out var gpuHandle, out uint index);
                SrvHandle = gpuHandle;
                SrvCpuHandle = cpuHandle;
                BindlessIndex = index;
                
                var srvDesc = new ShaderResourceViewDescription
                {
                    Format = Format.R32_Float,
                    ViewDimension = ShaderResourceViewDimension.Texture2D,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2D = new Texture2DShaderResourceView { MipLevels = 1 }
                };
                device.NativeDevice.CreateShaderResourceView(_resource, srvDesc, cpuHandle);
            }
        }
    }

    public class DepthTextureArray2D : Texture
    {
        public List<CpuDescriptorHandle> SliceDsvHandles { get; private set; } = new List<CpuDescriptorHandle>();
        public CpuDescriptorHandle[] SliceDsvs => SliceDsvHandles.ToArray(); // Helper if needed
        
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int Slices { get; private set; }

        public DepthTextureArray2D(int width, int height, int slices)
        {
            Width = width;
            Height = height;
            Slices = slices;
            
            var device = Engine.Device;
            _resource = device.CreateTexture2D(Format.R32_Typeless, width, height, slices, 1, ResourceFlags.AllowDepthStencil, ResourceStates.DepthWrite); // Initial State?

            // DSVs for each slice
            for(int i=0; i<slices; i++)
            {
                var dsv = device.AllocateDsv();
                SliceDsvHandles.Add(dsv);
                
                device.NativeDevice.CreateDepthStencilView(_resource, new DepthStencilViewDescription
                {
                     Format = Format.D32_Float,
                     ViewDimension = DepthStencilViewDimension.Texture2DArray,
                     Texture2DArray = new Texture2DArrayDepthStencilView
                     {
                         ArraySize = 1,
                         FirstArraySlice = (uint)i,
                         MipSlice = 0
                     }
                }, dsv);
            }

            // SRV for whole array
            var cpuHandle = device.AllocateSrvRange(1, out var gpuHandle, out uint index); // Simplified but correct index
            SrvHandle = gpuHandle;
            SrvCpuHandle = cpuHandle;
            BindlessIndex = index;
            
            device.NativeDevice.CreateShaderResourceView(_resource, new ShaderResourceViewDescription
            {
                Format = Format.R32_Float,
                ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2DArray = new Texture2DArrayShaderResourceView
                {
                    ArraySize = (uint)slices,
                    FirstArraySlice = 0,
                    MipLevels = 1,
                    MostDetailedMip = 0
                }
            }, cpuHandle);
        }
    }
}
