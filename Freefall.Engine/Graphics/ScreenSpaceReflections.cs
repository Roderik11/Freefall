using System;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Screen-space reflections (ssr_cs.hlsl): for every surface smoother than roughness 0.45, one ray along the mirror
    /// direction through the linear depth, at half resolution after the lighting; the hit pixel lit as the composition
    /// lights it (its direct light, the lanterns' included, the sky's ambient, emissive glass, the air in front). The
    /// composition lays it over the sky probe's reflection by its confidence (EngineSettings.ScreenSpaceReflections), so
    /// wet streets and puddles mirror the houses and the lanterns, not only the sky. Spatial only: one sharp ray, no
    /// history; rougher surfaces keep the probe's blurred reflection.
    /// </summary>
    public sealed class ScreenSpaceReflections : IDisposable
    {
        private readonly ComputeShader _shader;
        private readonly int _kReflect;
        private RenderTexture2D? _output;
        private int _width, _height;
        private bool _firstFrame = true;

        /// <summary>The half-size reflections (RGBA16F: rgb × confidence, confidence), 0 before the first frame.</summary>
        public uint OutputSrvIndex => _output?.BindlessIndex ?? 0;

        public ScreenSpaceReflections()
        {
            _shader = new ComputeShader("ssr_cs.hlsl");
            _kReflect = _shader.FindKernel("CSReflect");
        }

        /// <summary>After the lighting, before the composition; the G-buffer and the light buffer readable by compute. The
        /// scene constants (camera, atmosphere) are bound by <paramref name="bindSceneConstants"/>.</summary>
        public void Execute(ID3D12GraphicsCommandList list, DeferredRenderer renderer, Action<ID3D12GraphicsCommandList> bindSceneConstants)
        {
            var desc = renderer.Albedo.Native.Description;
            int width = (int)desc.Width, height = (int)desc.Height;
            EnsureTexture(width, height);
            int outW = (width + 1) / 2, outH = (height + 1) / 2;

            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            bindSceneConstants(list);

            _shader.SetPushConstant("AlbedoTex", renderer.Albedo.BindlessIndex);
            _shader.SetPushConstant("NormalTex", renderer.Normals.BindlessIndex);
            _shader.SetPushConstant("DataTex", renderer.Data.BindlessIndex);
            _shader.SetPushConstant("LightTex", renderer.LightBuffer.BindlessIndex);
            _shader.SetPushConstant("DepthTex", renderer.Depth.BindlessIndex);
            _shader.SetPushConstant("LinearDepthTex", renderer.DepthGBuffer.BindlessIndex);
            _shader.SetPushConstant("OutputUAV", _output!.UavIndex);
            _shader.SetPushConstant("OutWidth", (uint)outW);
            _shader.SetPushConstant("OutHeight", (uint)outH);
            _shader.SetPushConstant("FullWidth", (uint)width);
            _shader.SetPushConstant("FullHeight", (uint)height);

            PixMarker.Begin(list, "Screen-space reflections");
            var fromState = _firstFrame ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
            _firstFrame = false;
            list.ResourceBarrierTransition(_output.Native, fromState, ResourceStates.UnorderedAccess);
            _shader.Dispatch(_kReflect, list, ((uint)outW + 7) / 8, ((uint)outH + 7) / 8);
            list.ResourceBarrierTransition(_output.Native, ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
            PixMarker.End(list);
        }

        private void EnsureTexture(int width, int height)
        {
            if (_output != null && _width == width && _height == height)
                return;
            if (_output != null)
                Engine.Device.DeferDispose(new DeferredTexture(_output));
            _output = new RenderTexture2D(Engine.Device, (width + 1) / 2, (height + 1) / 2, Format.R16G16B16A16_Float, false, true);
            _width = width;
            _height = height;
            _firstFrame = true;
        }

        // RenderTexture2D.Dispose hides Texture.Dispose, so wrap it or DeferDispose would skip the descriptors (as GTAO)
        private sealed class DeferredTexture(RenderTexture2D texture) : IDisposable
        {
            public void Dispose() => texture.Dispose();
        }

        public void Dispose()
        {
            _output?.Dispose();
            _shader.Dispose();
        }
    }
}
