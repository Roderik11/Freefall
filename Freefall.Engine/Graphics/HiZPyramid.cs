using System;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Owns the Hi-Z depth pyramid texture and per-mip descriptors.
    /// Created and resized by DeferredRenderer to match its depth buffer.
    /// GPUCuller.GenerateHiZPyramid dispatches compute into this pyramid.
    /// </summary>
    public class HiZPyramid : IDisposable
    {
        public ID3D12Resource? Texture { get; private set; }
        public uint[] MipUAVs { get; private set; } = Array.Empty<uint>();
        public CpuDescriptorHandle[] MipUAVCPU { get; private set; } = Array.Empty<CpuDescriptorHandle>();
        public uint[] MipSRVs { get; private set; } = Array.Empty<uint>();
        public uint FullSRV { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int MipCount { get; private set; }
        public int SourceWidth { get; private set; }   // Full-res depth buffer width
        public int SourceHeight { get; private set; }  // Full-res depth buffer height

        // SPD atomic counter for last-group-standing synchronization
        public ID3D12Resource? CounterBuffer { get; private set; }
        public uint CounterUAV { get; private set; }
        public CpuDescriptorHandle CounterCPUHandle { get; private set; }
        private ID3D12DescriptorHeap? _counterCpuHeap;
        
        /// <summary>True after the pyramid has been generated at least once (safe to use for occlusion).</summary>
        public bool Ready { get; set; }

        /// <summary>
        /// Create or recreate the pyramid to match the given depth buffer dimensions.
        /// Disposes previous resources if resizing.
        /// </summary>
        public void Create(GraphicsDevice device, int depthWidth, int depthHeight)
        {
            // Dispose previous pyramid if resizing
            Texture?.Dispose();
            Ready = false;  // New pyramid needs to be generated before use

            // Store exact source dimensions (before integer division truncation)
            SourceWidth = depthWidth;
            SourceHeight = depthHeight;

            // Hi-Z is half-res of depth buffer (minimum 1 to avoid 0-size texture)
            Width = Math.Max(1, depthWidth / 2);
            Height = Math.Max(1, depthHeight / 2);

            // Calculate mip count
            MipCount = 1 + (int)Math.Floor(Math.Log2(Math.Max(Width, Height)));

            // Create texture: R32_Float with UAV support, full mip chain
            Texture = device.CreateTexture2D(
                Format.R32_Float,
                Width, Height,
                1, MipCount,
                ResourceFlags.AllowUnorderedAccess,
                ResourceStates.Common);

            // Allocate per-mip UAVs and SRVs
            MipUAVs = new uint[MipCount];
            MipUAVCPU = new CpuDescriptorHandle[MipCount];
            MipSRVs = new uint[MipCount];

            for (int i = 0; i < MipCount; i++)
            {
                // UAV for writing this mip level
                MipUAVs[i] = device.AllocateBindlessIndex();
                var uavDesc = new UnorderedAccessViewDescription
                {
                    Format = Format.R32_Float,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = (uint)i }
                };
                device.NativeDevice.CreateUnorderedAccessView(Texture, null, uavDesc, device.GetCpuHandle(MipUAVs[i]));

                // SRV for reading this mip level (used as input for next level's downsample)
                MipSRVs[i] = device.AllocateBindlessIndex();
                var srvDesc = new ShaderResourceViewDescription
                {
                    Format = Format.R32_Float,
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

            // Full-pyramid SRV (all mips, for CSVisibility to sample with Load at any mip)
            FullSRV = device.AllocateBindlessIndex();
            var fullSrvDesc = new ShaderResourceViewDescription
            {
                Format = Format.R32_Float,
                ViewDimension = ShaderResourceViewDimension.Texture2D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2D = new Texture2DShaderResourceView
                {
                    MostDetailedMip = 0,
                    MipLevels = (uint)MipCount
                }
            };
            device.NativeDevice.CreateShaderResourceView(Texture, fullSrvDesc, device.GetCpuHandle(FullSRV));

            // SPD atomic counter buffer (single uint, cleared to 0 before each dispatch)
            CounterBuffer?.Dispose();
            _counterCpuHeap?.Dispose();
            CounterBuffer = device.CreateDefaultBuffer(4); // 1 × uint
            CounterUAV = device.AllocateBindlessIndex();
            var counterUavDesc = new UnorderedAccessViewDescription
            {
                Format = Format.R32_Typeless,
                ViewDimension = UnorderedAccessViewDimension.Buffer,
                Buffer = new BufferUnorderedAccessView { NumElements = 1, Flags = BufferUnorderedAccessViewFlags.Raw }
            };
            device.NativeDevice.CreateUnorderedAccessView(CounterBuffer, null, counterUavDesc, device.GetCpuHandle(CounterUAV));

            // CPU-side descriptor for ClearUnorderedAccessViewUint
            var cpuHeapDesc = new DescriptorHeapDescription
            {
                Type = DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView,
                DescriptorCount = 1,
                Flags = DescriptorHeapFlags.None
            };
            _counterCpuHeap = device.NativeDevice.CreateDescriptorHeap(cpuHeapDesc);
            CounterCPUHandle = _counterCpuHeap.GetCPUDescriptorHandleForHeapStart();
            device.NativeDevice.CreateUnorderedAccessView(CounterBuffer, null, counterUavDesc, CounterCPUHandle);

            //Debug.Log("HiZPyramid", $"Created: {Width}x{Height}, {MipCount} mips, SRV={FullSRV}");
        }

        public void Dispose()
        {
            Texture?.Dispose();
            Texture = null;
            CounterBuffer?.Dispose();
            CounterBuffer = null;
            _counterCpuHeap?.Dispose();
            _counterCpuHeap = null;
        }
    }
}
