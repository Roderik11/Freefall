using System;
using System.Numerics;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Ground Truth Ambient Occlusion (Jimenez et al. 2016, after Intel XeGTAO).
    /// Spatial-only: a 4x4 interleaved noise pattern plus a depth-aware 4x4 blur, no temporal pass.
    /// Produces a single-channel visibility mask at screen resolution (1 = unoccluded).
    /// </summary>
    public class GTAO : IDisposable
    {
        private ComputeShader _shader;
        private int _kGTAO;
        private int _kDenoise;
        private RenderTexture2D? _raw;
        private RenderTexture2D? _output;
        private int _width, _height;
        private bool _firstFrame = true;

        private const uint StepCount = 4; // per side of each slice

        public uint OutputSrvIndex => _output?.BindlessIndex ?? 0;

        public GTAO()
        {
            _shader = new ComputeShader("gtao.hlsl");
            _kGTAO = _shader.FindKernel("CSGTAO");
            _kDenoise = _shader.FindKernel("CSDenoise");
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

            _shader.SetPushConstant("DepthTex", depthSrvIndex);
            _shader.SetPushConstant("NormalTex", normalSrvIndex);
            _shader.SetPushConstant("ScreenWidth", (uint)texWidth);
            _shader.SetPushConstant("ScreenHeight", (uint)texHeight);

            uint groupsX = ((uint)texWidth + 7) / 8;
            uint groupsY = ((uint)texHeight + 7) / 8;

            // Pass 1: horizon search → raw (noisy) AO
            commandList.ResourceBarrierTransition(_raw!.Native, fromState, ResourceStates.UnorderedAccess);
            _shader.SetPushConstant(_kGTAO, "OutputUAV", _raw.UavIndex);
            _shader.Dispatch(_kGTAO, commandList, groupsX, groupsY);
            commandList.ResourceBarrierTransition(_raw.Native,
                ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);

            // Pass 2: depth-aware 4x4 blur → final AO
            commandList.ResourceBarrierTransition(_output!.Native, fromState, ResourceStates.UnorderedAccess);
            _shader.SetPushConstant(_kDenoise, "AOInput", _raw.BindlessIndex);
            _shader.SetPushConstant(_kDenoise, "OutputUAV", _output.UavIndex);
            _shader.Dispatch(_kDenoise, commandList, groupsX, groupsY);

            // Transition output back to SRV for composition
            commandList.ResourceBarrierTransition(_output.Native,
                ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
        }

        private void EnsureTextures(int width, int height)
        {
            if (_output != null && _width == width && _height == height)
                return;

            _raw?.Dispose();
            _output?.Dispose();
            _raw = new RenderTexture2D(Engine.Device, width, height, Format.R8_UNorm, false, true);
            _output = new RenderTexture2D(Engine.Device, width, height, Format.R8_UNorm, false, true);
            _width = width;
            _height = height;
            _firstFrame = true;
        }

        public void Dispose()
        {
            _raw?.Dispose();
            _output?.Dispose();
            _shader?.Dispose();
        }
    }
}
