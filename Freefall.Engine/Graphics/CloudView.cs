using System;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// A view's volumetric clouds (clouds_cs.hlsl CSCloudMarch): the clouds' light in front of the sky (the air between
    /// included, × Exposure) and what they let through, at half the view's size each way. One per
    /// <see cref="DeferredRenderer"/>; the sky dome reads it through SceneConstants.CloudTargetIdx (0 = the 2D layer).
    /// Stage 1 of VOLUMETRIC_CLOUDS_PLAN.md: every texel marched every frame, no history yet.
    /// </summary>
    public sealed class CloudView : IDisposable
    {
        private RenderTexture2D? _target;
        private ResourceStates _state = ResourceStates.Common;
        private int _kMarch = -1;
        private uint _frame;

        public int Width { get; private set; }
        public int Height { get; private set; }

        /// <summary>Fit the target to a view of this size and return its bindless index for the frame (0 when the
        /// volumetric clouds are off: the sky draws the 2D layer).</summary>
        public uint Prepare(int viewWidth, int viewHeight)
        {
            if (!Clouds.Active)
                return 0;
            int w = Math.Max(1, (viewWidth + 1) / 2), h = Math.Max(1, (viewHeight + 1) / 2);
            if (_target == null || w != Width || h != Height)
            {
                _target?.Dispose();
                // Pre-exposed light and a transmittance
                _target = new RenderTexture2D(Engine.Device, w, h, Format.R16G16B16A16_Float, randomWrite: true);
                _state = ResourceStates.Common;
                Width = w;
                Height = h;
            }
            return _target.BindlessIndex;
        }

        /// <summary>March this view's clouds: after Clouds.Render (the bake) and the atmosphere's LUTs, before the sky is
        /// drawn. The scene constants must be bound for compute by <paramref name="bindSceneConstants"/>.</summary>
        public void Render(ID3D12GraphicsCommandList list, Action<ID3D12GraphicsCommandList> bindSceneConstants)
        {
            if (!Clouds.Active || _target == null)
                return;
            var cs = Clouds.Shader;
            if (_kMarch < 0)
                _kMarch = cs.FindKernel("CSCloudMarch");
            var device = Engine.Device;

            PixMarker.Begin(list, "Clouds");
            list.SetComputeRootSignature(device.GlobalRootSignature);
            list.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            bindSceneConstants(list);

            if (_state != ResourceStates.UnorderedAccess)
                list.ResourceBarrierTransition(_target.Native, _state, ResourceStates.UnorderedAccess);
            cs.SetPushConstant(_kMarch, "TargetUAV", _target.UavIndex);
            cs.SetPushConstant(_kMarch, "TargetWidth", (uint)Width);
            cs.SetPushConstant(_kMarch, "TargetHeight", (uint)Height);
            cs.SetPushConstant(_kMarch, "FrameIndex", _frame++);
            cs.Dispatch(_kMarch, list, (uint)(Width + 7) / 8, (uint)(Height + 7) / 8);
            var shaderRead = ResourceStates.NonPixelShaderResource | ResourceStates.PixelShaderResource;
            list.ResourceBarrierTransition(_target.Native, ResourceStates.UnorderedAccess, shaderRead);
            _state = shaderRead;
            PixMarker.End(list);
        }

        public void Dispose()
        {
            _target?.Dispose();
            _target = null;
        }
    }
}
