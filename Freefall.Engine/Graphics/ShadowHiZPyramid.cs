using System;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Owns the shadow Hi-Z pyramid — a Texture2DArray with mip chain for all cascades.
    /// Uses min() reduction (standard Z: near=small, far=1.0).
    /// GPUCuller.GenerateShadowHiZPyramid dispatches compute into this pyramid.
    /// </summary>
    public class ShadowHiZPyramid : IDisposable
    {
        public ID3D12Resource? Texture { get; private set; }
        
        /// <summary>Per-slice, per-mip UAVs. Indexed as [slice * MipCount + mip].</summary>
        public uint[] MipUAVs { get; private set; } = Array.Empty<uint>();
        
        /// <summary>Per-slice, per-mip SRVs (single-mip read). Indexed as [slice * MipCount + mip].</summary>
        public uint[] MipSRVs { get; private set; } = Array.Empty<uint>();
        
        /// <summary>Full-array SRV with all mips (for consumer sampling). Texture2DArray view.</summary>
        public uint FullSRV { get; private set; }
        
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int MipCount { get; private set; }
        public int SliceCount { get; private set; }
        public int SourceWidth { get; private set; }
        public int SourceHeight { get; private set; }
        
        /// <summary>True after first generation (safe for occlusion testing).</summary>
        public bool Ready { get; set; }

        /// <summary>
        /// Create or recreate the pyramid for the given shadow map dimensions.
        /// </summary>
        public void Create(GraphicsDevice device, int shadowWidth, int shadowHeight, int slices)
        {
            Texture?.Dispose();
            Ready = false;

            SourceWidth = shadowWidth;
            SourceHeight = shadowHeight;
            SliceCount = slices;
            
            // Half-res of shadow map
            Width = Math.Max(1, shadowWidth / 2);
            Height = Math.Max(1, shadowHeight / 2);
            MipCount = 1 + (int)Math.Floor(Math.Log2(Math.Max(Width, Height)));
            
            // Create Texture2DArray: R32_Float with UAV support, full mip chain, N slices
            Texture = device.CreateTexture2D(
                Format.R32_Float,
                Width, Height,
                slices, MipCount,
                ResourceFlags.AllowUnorderedAccess,
                ResourceStates.Common);
            
            // Allocate per-slice, per-mip UAVs and SRVs
            int totalMips = slices * MipCount;
            MipUAVs = new uint[totalMips];
            MipSRVs = new uint[totalMips];
            
            for (int slice = 0; slice < slices; slice++)
            {
                for (int mip = 0; mip < MipCount; mip++)
                {
                    int idx = slice * MipCount + mip;
                    
                    // UAV — single slice, single mip
                    MipUAVs[idx] = device.AllocateBindlessIndex();
                    var uavDesc = new UnorderedAccessViewDescription
                    {
                        Format = Format.R32_Float,
                        ViewDimension = UnorderedAccessViewDimension.Texture2DArray,
                        Texture2DArray = new Texture2DArrayUnorderedAccessView
                        {
                            MipSlice = (uint)mip,
                            FirstArraySlice = (uint)slice,
                            ArraySize = 1
                        }
                    };
                    device.NativeDevice.CreateUnorderedAccessView(Texture, null, uavDesc, device.GetCpuHandle(MipUAVs[idx]));
                    
                    // SRV — single slice, single mip (for per-mip fallback reads)
                    MipSRVs[idx] = device.AllocateBindlessIndex();
                    var srvDesc = new ShaderResourceViewDescription
                    {
                        Format = Format.R32_Float,
                        ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                        Shader4ComponentMapping = ShaderComponentMapping.Default,
                        Texture2DArray = new Texture2DArrayShaderResourceView
                        {
                            MostDetailedMip = (uint)mip,
                            MipLevels = 1,
                            FirstArraySlice = (uint)slice,
                            ArraySize = 1
                        }
                    };
                    device.NativeDevice.CreateShaderResourceView(Texture, srvDesc, device.GetCpuHandle(MipSRVs[idx]));
                }
            }
            
            // Full-array SRV (all slices, all mips) for consumer sampling
            FullSRV = device.AllocateBindlessIndex();
            var fullSrvDesc = new ShaderResourceViewDescription
            {
                Format = Format.R32_Float,
                ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2DArray = new Texture2DArrayShaderResourceView
                {
                    MostDetailedMip = 0,
                    MipLevels = (uint)MipCount,
                    FirstArraySlice = 0,
                    ArraySize = (uint)slices
                }
            };
            device.NativeDevice.CreateShaderResourceView(Texture, fullSrvDesc, device.GetCpuHandle(FullSRV));
            
            Debug.Log("ShadowHiZPyramid", $"Created: {Width}x{Height}x{slices}, {MipCount} mips, SRV={FullSRV}");
        }
        
        /// <summary>Get UAV index for a specific slice and mip level.</summary>
        public uint GetMipUAV(int slice, int mip) => MipUAVs[slice * MipCount + mip];
        
        /// <summary>Get SRV index for a specific slice and mip level.</summary>
        public uint GetMipSRV(int slice, int mip) => MipSRVs[slice * MipCount + mip];

        public void Dispose()
        {
            Texture?.Dispose();
            Texture = null;
        }
    }
}
