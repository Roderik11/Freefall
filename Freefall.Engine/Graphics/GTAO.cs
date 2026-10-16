using System;
using System.Numerics;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Ground Truth Ambient Occlusion (Jimenez et al. 2016, after Intel XeGTAO).
    /// Spatial-only: a 4x4 interleaved noise pattern plus a depth-aware 4x4 blur, no temporal pass.
    /// Traces and blurs at half resolution, then a depth-aware upsample fills a single-channel visibility
    /// mask at screen resolution (1 = unoccluded). Half resolution was judged indistinguishable from full
    /// at about half the cost (2026-10-16).
    /// </summary>
    public class GTAO : IDisposable
    {
        private ComputeShader _shader;
        private int _kGTAO;
        private int _kDenoise;
        private int _kDownsampleDepth;
        private int _kUpsample;
        private RenderTexture2D? _halfDepth; // one screen pixel's depth per 2x2 block
        private RenderTexture2D? _raw;       // half resolution, noisy
        private RenderTexture2D? _denoised;  // half resolution
        private RenderTexture2D? _output;    // screen size, read by composition
        private int _width, _height;
        private bool _firstFrame = true;

        private const uint StepCount = 4; // per side of each slice

        public uint OutputSrvIndex => _output?.BindlessIndex ?? 0;

        public GTAO()
        {
            _shader = new ComputeShader("gtao.hlsl");
            _kGTAO = _shader.FindKernel("CSGTAO");
            _kDenoise = _shader.FindKernel("CSDenoise");
            _kDownsampleDepth = _shader.FindKernel("CSDownsampleDepth");
            _kUpsample = _shader.FindKernel("CSUpsample");
        }

        /// <summary>
        /// Execute AO generation. Call after the GBuffer is populated, before composition.
        /// </summary>
        /// <param name="viewRotation">Zero-translation view matrix (world → view rotation).</param>
        public void Execute(
            ID3D12GraphicsCommandList commandList,
            uint depthSrvIndex,
            uint normalSrvIndex,
            int texWidth, int texHeight,
            Matrix4x4 viewRotation,
            Matrix4x4 projection)
        {
            EnsureTextures(texWidth, texHeight);

            // Common on first frame after creation
            var fromState = _firstFrame ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
            _firstFrame = false;

            commandList.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            commandList.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });

            var settings = Engine.Settings;

            // Camera basis vectors are the columns of the view rotation
            _shader.SetParam("ViewRight", new Vector4(viewRotation.M11, viewRotation.M21, viewRotation.M31, 0f));
            _shader.SetParam("ViewUp", new Vector4(viewRotation.M12, viewRotation.M22, viewRotation.M32, 0f));
            _shader.SetParam("ViewForward", new Vector4(viewRotation.M13, viewRotation.M23, viewRotation.M33, 0f));
            _shader.SetParam("ProjScale", new Vector2(projection.M11, projection.M22));
            _shader.SetParam("Radius", settings.GTAORadius);
            _shader.SetParam("Power", settings.GTAOPower);
            _shader.SetParam("Intensity", settings.GTAOIntensity);
            _shader.SetParam("SliceCount", (uint)Math.Clamp(settings.GTAOSlices, 1, 8));
            _shader.SetParam("StepCount", StepCount);

            int aoWidth = (texWidth + 1) / 2;
            int aoHeight = (texHeight + 1) / 2;

            _shader.SetPushConstant("DepthTex", _halfDepth!.BindlessIndex);
            _shader.SetPushConstant("NormalTex", normalSrvIndex);
            _shader.SetPushConstant("FullDepthTex", depthSrvIndex);
            _shader.SetPushConstant("ScreenWidth", (uint)aoWidth);
            _shader.SetPushConstant("ScreenHeight", (uint)aoHeight);
            _shader.SetPushConstant("FullWidth", (uint)texWidth);
            _shader.SetPushConstant("FullHeight", (uint)texHeight);

            uint groupsX = ((uint)aoWidth + 7) / 8;
            uint groupsY = ((uint)aoHeight + 7) / 8;

            // Pass 0: one screen pixel's depth per 2x2 block
            PixMarker.Begin(commandList, "Depth");
            commandList.ResourceBarrierTransition(_halfDepth.Native, fromState, ResourceStates.UnorderedAccess);
            _shader.SetPushConstant(_kDownsampleDepth, "OutputUAV", _halfDepth.UavIndex);
            _shader.Dispatch(_kDownsampleDepth, commandList, groupsX, groupsY);
            commandList.ResourceBarrierTransition(_halfDepth.Native,
                ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
            PixMarker.End(commandList);

            // Pass 1: horizon search → raw (noisy) AO
            PixMarker.Begin(commandList, "Trace");
            commandList.ResourceBarrierTransition(_raw!.Native, fromState, ResourceStates.UnorderedAccess);
            _shader.SetPushConstant(_kGTAO, "OutputUAV", _raw.UavIndex);
            _shader.Dispatch(_kGTAO, commandList, groupsX, groupsY);
            commandList.ResourceBarrierTransition(_raw.Native,
                ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
            PixMarker.End(commandList);

            // Pass 2: depth-aware 4x4 blur
            PixMarker.Begin(commandList, "Denoise");
            commandList.ResourceBarrierTransition(_denoised!.Native, fromState, ResourceStates.UnorderedAccess);
            _shader.SetPushConstant(_kDenoise, "AOInput", _raw.BindlessIndex);
            _shader.SetPushConstant(_kDenoise, "OutputUAV", _denoised.UavIndex);
            _shader.Dispatch(_kDenoise, commandList, groupsX, groupsY);
            commandList.ResourceBarrierTransition(_denoised.Native,
                ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
            PixMarker.End(commandList);

            // Pass 3: depth-aware upsample → screen-size AO
            PixMarker.Begin(commandList, "Upsample");
            commandList.ResourceBarrierTransition(_output!.Native, fromState, ResourceStates.UnorderedAccess);
            _shader.SetPushConstant(_kUpsample, "AOInput", _denoised.BindlessIndex);
            _shader.SetPushConstant(_kUpsample, "OutputUAV", _output.UavIndex);
            _shader.Dispatch(_kUpsample, commandList, ((uint)texWidth + 7) / 8, ((uint)texHeight + 7) / 8);
            // Transition output back to SRV for composition
            commandList.ResourceBarrierTransition(_output.Native,
                ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
            PixMarker.End(commandList);
        }

        private void EnsureTextures(int width, int height)
        {
            if (_output != null && _width == width && _height == height)
                return;

            // Frames in flight may still read the old set
            ReleaseTextures();

            int aoWidth = (width + 1) / 2;
            int aoHeight = (height + 1) / 2;
            _halfDepth = new RenderTexture2D(Engine.Device, aoWidth, aoHeight, Format.R32_Float, false, true);
            _raw = new RenderTexture2D(Engine.Device, aoWidth, aoHeight, Format.R8_UNorm, false, true);
            _denoised = new RenderTexture2D(Engine.Device, aoWidth, aoHeight, Format.R8_UNorm, false, true);
            _output = new RenderTexture2D(Engine.Device, width, height, Format.R8_UNorm, false, true);
            _width = width;
            _height = height;
            _firstFrame = true;
        }

        private void ReleaseTextures()
        {
            // RenderTexture2D.Dispose hides Texture.Dispose, so wrap it or DeferDispose would skip the descriptors
            foreach (var texture in new[] { _halfDepth, _raw, _denoised, _output })
                if (texture != null)
                    Engine.Device.DeferDispose(new DeferredTexture(texture));
            _halfDepth = _raw = _denoised = _output = null;
        }

        private sealed class DeferredTexture(RenderTexture2D texture) : IDisposable
        {
            public void Dispose() => texture.Dispose();
        }

        public void Dispose()
        {
            _halfDepth?.Dispose();
            _raw?.Dispose();
            _denoised?.Dispose();
            _output?.Dispose();
            _shader?.Dispose();
        }
    }
}
