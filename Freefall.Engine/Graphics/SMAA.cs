using System;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// SMAA 1x — Enhanced Subpixel Morphological Anti-Aliasing.
    /// Three compute passes: edge detection, blend weight calculation, neighborhood blend.
    /// Uses analytical blend weights (no LUT textures needed for SMAA 1x).
    /// </summary>
    public class SMAA : IDisposable
    {
        private ComputeShader _shader;
        private int _kEdgeDetection;
        private int _kBlendWeights;
        private int _kNeighborhoodBlend;

        private RenderTexture2D? _edgesTex;    // RG8_UNorm — edge flags
        private RenderTexture2D? _blendTex;    // RGBA8_UNorm — blend weights
        private RenderTexture2D? _outputTex;   // R8G8B8A8_UNorm — final output

        private int _width, _height;
        private bool _firstFrame = true;

        /// <summary>Native resource for blit to backbuffer.</summary>
        public ID3D12Resource? BlitSource => _outputTex?.Native;

        /// <summary>True when SMAA textures have been allocated.</summary>
        public bool IsReady => _outputTex != null;

        public SMAA()
        {
            _shader = new ComputeShader("smaa_cs.hlsl");
            _kEdgeDetection = _shader.FindKernel("CSEdgeDetection");
            _kBlendWeights = _shader.FindKernel("CSBlendWeights");
            _kNeighborhoodBlend = _shader.FindKernel("CSNeighborhoodBlend");
        }

        public void Execute(
            ID3D12GraphicsCommandList list,
            ID3D12Resource inputResource,
            uint inputSrvIndex,
            int texWidth, int texHeight)
        {
            EnsureResources(texWidth, texHeight);

            // Composite: PixelShaderResource → NonPixelShaderResource for compute reads
            list.ResourceBarrierTransition(inputResource,
                ResourceStates.PixelShaderResource, ResourceStates.NonPixelShaderResource);

            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });

            uint gx = (uint)((texWidth + 7) / 8);
            uint gy = (uint)((texHeight + 7) / 8);

            // ── Pass 1: Edge Detection ──
            {
                var from = _firstFrame ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
                list.ResourceBarrierTransition(_edgesTex!.Native, from, ResourceStates.UnorderedAccess);

                _shader.SetPushConstant(_kEdgeDetection, "InputTex", inputSrvIndex);
                _shader.SetPushConstant(_kEdgeDetection, "EdgesUAV", _edgesTex!.UavIndex);
                _shader.SetPushConstant(_kEdgeDetection, "ScreenWidth", (uint)texWidth);
                _shader.SetPushConstant(_kEdgeDetection, "ScreenHeight", (uint)texHeight);

                _shader.Dispatch(_kEdgeDetection, list, gx, gy);

                list.ResourceBarrierTransition(_edgesTex!.Native,
                    ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
            }

            // ── Pass 2: Blend Weight Calculation ──
            {
                var from = _firstFrame ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
                list.ResourceBarrierTransition(_blendTex!.Native, from, ResourceStates.UnorderedAccess);

                _shader.SetPushConstant(_kBlendWeights, "BlendUAV", _blendTex!.UavIndex);
                _shader.SetPushConstant(_kBlendWeights, "EdgesSRV", _edgesTex!.BindlessIndex);
                _shader.SetPushConstant(_kBlendWeights, "ScreenWidth", (uint)texWidth);
                _shader.SetPushConstant(_kBlendWeights, "ScreenHeight", (uint)texHeight);

                _shader.Dispatch(_kBlendWeights, list, gx, gy);

                list.ResourceBarrierTransition(_blendTex!.Native,
                    ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
            }

            // ── Pass 3: Neighborhood Blending ──
            {
                var from = _firstFrame ? ResourceStates.Common : ResourceStates.PixelShaderResource;
                list.ResourceBarrierTransition(_outputTex!.Native, from, ResourceStates.UnorderedAccess);

                _shader.SetPushConstant(_kNeighborhoodBlend, "InputTex", inputSrvIndex);
                _shader.SetPushConstant(_kNeighborhoodBlend, "OutputUAV", _outputTex!.UavIndex);
                _shader.SetPushConstant(_kNeighborhoodBlend, "BlendSRV", _blendTex!.BindlessIndex);
                _shader.SetPushConstant(_kNeighborhoodBlend, "ScreenWidth", (uint)texWidth);
                _shader.SetPushConstant(_kNeighborhoodBlend, "ScreenHeight", (uint)texHeight);

                _shader.Dispatch(_kNeighborhoodBlend, list, gx, gy);

                list.ResourceBarrierTransition(_outputTex!.Native,
                    ResourceStates.UnorderedAccess, ResourceStates.PixelShaderResource);
            }

            // Restore Composite state
            list.ResourceBarrierTransition(inputResource,
                ResourceStates.NonPixelShaderResource, ResourceStates.PixelShaderResource);

            _firstFrame = false;
        }

        private void EnsureResources(int width, int height)
        {
            if (_edgesTex != null && _width == width && _height == height)
                return;

            _edgesTex?.Dispose();
            _blendTex?.Dispose();
            _outputTex?.Dispose();

            var device = Engine.Device;
            _width = width;
            _height = height;

            _edgesTex = new RenderTexture2D(device, width, height, Format.R8G8_UNorm, false, true);
            _blendTex = new RenderTexture2D(device, width, height, Format.R8G8B8A8_UNorm, false, true);
            _outputTex = new RenderTexture2D(device, width, height, Format.R8G8B8A8_UNorm, false, true);

            _firstFrame = true;
        }

        public void Dispose()
        {
            _edgesTex?.Dispose();
            _blendTex?.Dispose();
            _outputTex?.Dispose();
            _shader?.Dispose();
        }
    }
}
