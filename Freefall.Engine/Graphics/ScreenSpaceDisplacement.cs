using System;
using System.Numerics;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// SSDM: Screen Space Displacement Mapping (Lobel 2008).
    /// Two modes: simple pass-through, or full pyramid refinement.
    /// Toggle via Engine.Settings.UseSSDMPyramid.
    /// </summary>
    public class ScreenSpaceDisplacement : IDisposable
    {
        private ComputeShader _shader;
        private int _displaceKernel;
        private int _scaleAKernel;
        private int _buildMipKernel;
        private int _refineKernel;
        private int _displaceDepthKernel;

        // Pyramids (reuse existing DisplacementPyramid class)
        private DisplacementPyramid _pyramidA = new();
        private DisplacementPyramid _pyramidB = new();

        // Simple mode output (single texture, no mips)
        private ID3D12Resource? _simpleOutput;
        private uint _simpleOutputUav;
        private uint _simpleOutputSrv;

        // Displaced depth output
        private ID3D12Resource? _displacedDepth;
        private uint _displacedDepthUav;
        private uint _displacedDepthSrv;
        private bool _displacedDepthFirstFrame = true;

        private int _width, _height;
        private bool _firstFrame = true;
        private bool _pyramidFirstFrame = true;

        public uint OutputSrvIndex { get; private set; }
        public uint DisplacedDepthSrvIndex { get; private set; }

        public ScreenSpaceDisplacement()
        {
            _shader = new ComputeShader("screenspace_displacement.hlsl");
            _displaceKernel = _shader.FindKernel("CSDisplace");
            _scaleAKernel = _shader.FindKernel("CSScaleA");
            _buildMipKernel = _shader.FindKernel("CSBuildMip");
            _refineKernel = _shader.FindKernel("CSRefine");
            _displaceDepthKernel = _shader.FindKernel("CSDisplaceDepth");
        }

        public void Execute(
            ID3D12GraphicsCommandList cmd,
            uint displacementSrvIndex,
            uint depthSrvIndex,
            int texWidth, int texHeight,
            Matrix4x4 viewProjection)
        {
            if (displacementSrvIndex == 0) return;

            EnsureResources(texWidth, texHeight);

            cmd.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });

            if (Engine.Settings.UseSSDMPyramid)
                ExecutePyramid(cmd, displacementSrvIndex, depthSrvIndex, texWidth, texHeight);
            else
                ExecuteSimple(cmd, displacementSrvIndex, depthSrvIndex, texWidth, texHeight);
        }

        private void ExecuteSimple(ID3D12GraphicsCommandList cmd,
            uint displacementSrvIndex, uint depthSrvIndex, int texWidth, int texHeight)
        {
            var from = _firstFrame ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
            cmd.ResourceBarrierTransition(_simpleOutput!, from, ResourceStates.UnorderedAccess);
            _firstFrame = false;

            _shader.SetPushConstant(_displaceKernel, "SrcMip", displacementSrvIndex);
            _shader.SetPushConstant(_displaceKernel, "DstMip", _simpleOutputUav);
            _shader.SetParam("DstWidth", (uint)texWidth);
            _shader.SetParam("DstHeight", (uint)texHeight);
            _shader.SetParam("HeightScale", Engine.Settings.SSDMHeightScale);

            uint gx = (uint)((texWidth + 7) / 8);
            uint gy = (uint)((texHeight + 7) / 8);
            _shader.Dispatch(_displaceKernel, cmd, gx, gy);

            cmd.ResourceBarrierTransition(_simpleOutput!,
                ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);

            OutputSrvIndex = _simpleOutputSrv;

            // Displace depth for simple mode too
            DispatchDisplaceDepth(cmd, depthSrvIndex, texWidth, texHeight);
        }

        private void ExecutePyramid(ID3D12GraphicsCommandList cmd,
            uint displacementSrvIndex, uint depthSrvIndex, int texWidth, int texHeight)
        {
            var beforeState = _pyramidFirstFrame ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
            _pyramidFirstFrame = false;

            int mipCount = _pyramidA.MipCount;

            // Phase 0: Bake HeightScale into A[0]
            cmd.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_pyramidA.Texture!,
                    beforeState, ResourceStates.UnorderedAccess, 0)));

            _shader.SetPushConstant(_scaleAKernel, "SrcMip", displacementSrvIndex);
            _shader.SetPushConstant(_scaleAKernel, "DstMip", _pyramidA.MipUAVs[0]);
            _shader.SetParam("DstWidth", (uint)texWidth);
            _shader.SetParam("DstHeight", (uint)texHeight);
            _shader.SetParam("HeightScale", Engine.Settings.SSDMHeightScale);

            _shader.Dispatch(_scaleAKernel, cmd,
                (uint)((texWidth + 7) / 8), (uint)((texHeight + 7) / 8));

            cmd.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_pyramidA.Texture!,
                    ResourceStates.UnorderedAccess,
                    ResourceStates.NonPixelShaderResource, 0)));

            // Phase 1: Build A mip chain from A[0]
            uint prevSrv = _pyramidA.MipSRVs[0];
            for (int mip = 1; mip < mipCount; mip++)
            {
                int mipW = Math.Max(1, texWidth >> mip);
                int mipH = Math.Max(1, texHeight >> mip);

                cmd.ResourceBarrier(new ResourceBarrier(
                    new ResourceTransitionBarrier(_pyramidA.Texture!,
                        beforeState, ResourceStates.UnorderedAccess, (uint)mip)));

                _shader.SetPushConstant(_buildMipKernel, "SrcMip", prevSrv);
                _shader.SetPushConstant(_buildMipKernel, "DstMip", _pyramidA.MipUAVs[mip]);
                _shader.SetParam("DstWidth", (uint)mipW);
                _shader.SetParam("DstHeight", (uint)mipH);

                _shader.Dispatch(_buildMipKernel, cmd,
                    (uint)((mipW + 7) / 8), (uint)((mipH + 7) / 8));

                cmd.ResourceBarrier(new ResourceBarrier(
                    new ResourceTransitionBarrier(_pyramidA.Texture!,
                        ResourceStates.UnorderedAccess,
                        ResourceStates.NonPixelShaderResource, (uint)mip)));

                prevSrv = _pyramidA.MipSRVs[mip];
            }

            // Phase 2: Iterative Newton inversion at full resolution
            cmd.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_pyramidB.Texture!,
                    beforeState, ResourceStates.UnorderedAccess, 0)));

            _shader.SetPushConstant(_refineKernel, "SrcMip", _pyramidA.FullChainSrv);
            _shader.SetPushConstant(_refineKernel, "DstMip", _pyramidB.MipUAVs[0]);
            _shader.SetParam("DstWidth", (uint)texWidth);
            _shader.SetParam("DstHeight", (uint)texHeight);
            _shader.SetParam("MipCount", (uint)_pyramidA.MipCount);

            _shader.Dispatch(_refineKernel, cmd,
                (uint)((texWidth + 7) / 8), (uint)((texHeight + 7) / 8));

            cmd.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(_pyramidB.Texture!,
                    ResourceStates.UnorderedAccess,
                    ResourceStates.NonPixelShaderResource, 0)));

            OutputSrvIndex = _pyramidB.MipSRVs[0];

            // Phase 3: Displace depth buffer using B map
            DispatchDisplaceDepth(cmd, depthSrvIndex, texWidth, texHeight);
        }

        private void EnsureResources(int width, int height)
        {
            if (_simpleOutput != null && _width == width && _height == height)
                return;

            _width = width;
            _height = height;
            var device = Engine.Device;

            // Simple output texture
            _simpleOutput?.Dispose();
            _simpleOutput = device.CreateTexture2D(
                Format.R16G16_Float, width, height, 1, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _simpleOutputUav = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(_simpleOutput, null,
                new UnorderedAccessViewDescription
                {
                    Format = Format.R16G16_Float,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
                }, device.GetCpuHandle(_simpleOutputUav));

            _simpleOutputSrv = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(_simpleOutput,
                new ShaderResourceViewDescription
                {
                    Format = Format.R16G16_Float,
                    ViewDimension = ShaderResourceViewDimension.Texture2D,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
                }, device.GetCpuHandle(_simpleOutputSrv));

            // Displaced depth texture (R32_Float, same resolution)
            _displacedDepth?.Dispose();
            _displacedDepth = device.CreateTexture2D(
                Format.R32_Float, width, height, 1, 1,
                ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);

            _displacedDepthUav = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(_displacedDepth, null,
                new UnorderedAccessViewDescription
                {
                    Format = Format.R32_Float,
                    ViewDimension = UnorderedAccessViewDimension.Texture2D,
                    Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
                }, device.GetCpuHandle(_displacedDepthUav));

            _displacedDepthSrv = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(_displacedDepth,
                new ShaderResourceViewDescription
                {
                    Format = Format.R32_Float,
                    ViewDimension = ShaderResourceViewDimension.Texture2D,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
                }, device.GetCpuHandle(_displacedDepthSrv));

            // Pyramids
            _pyramidA.Create(device, width, height);
            _pyramidB.Create(device, width, height);

            _firstFrame = true;
            _displacedDepthFirstFrame = true;
        }

        private void DispatchDisplaceDepth(ID3D12GraphicsCommandList cmd,
            uint depthSrvIndex, int texWidth, int texHeight)
        {
            if (depthSrvIndex == 0) return;

            var beforeState = _displacedDepthFirstFrame ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
            _displacedDepthFirstFrame = false;

            cmd.ResourceBarrierTransition(_displacedDepth!,
                beforeState, ResourceStates.UnorderedAccess);

            // SrcMipIdx = B buffer SRV, PrevBMipIdx = depth input SRV, DstMipIdx = displaced depth UAV
            _shader.SetPushConstant(_displaceDepthKernel, "SrcMip", OutputSrvIndex);
            _shader.SetPushConstant(_displaceDepthKernel, "DstMip", _displacedDepthUav);
            _shader.SetPushConstant(_displaceDepthKernel, "PrevBMip", depthSrvIndex);
            _shader.SetParam("DstWidth", (uint)texWidth);
            _shader.SetParam("DstHeight", (uint)texHeight);

            _shader.Dispatch(_displaceDepthKernel, cmd,
                (uint)((texWidth + 7) / 8), (uint)((texHeight + 7) / 8));

            cmd.ResourceBarrierTransition(_displacedDepth!,
                ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);

            DisplacedDepthSrvIndex = _displacedDepthSrv;
        }

        public void Dispose()
        {
            _simpleOutput?.Dispose();
            _displacedDepth?.Dispose();
            _pyramidA.Dispose();
            _pyramidB.Dispose();
            _shader?.Dispose();
        }
    }
}
