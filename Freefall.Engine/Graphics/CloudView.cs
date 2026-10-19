using System;
using System.Numerics;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// A view's volumetric clouds (clouds_cs.hlsl): their light in front of the sky (the air between included, ×
    /// Exposure), what they let through and how far they are, at half the view's size each way. Each frame marches one
    /// texel of every 2×2 block (CSCloudMarch, sparse, Bayer order: all of them every 4 frames) and CSCloudReconstruct
    /// fills the rest from last frame's, reprojected for the camera's move and the wind's and clamped to this frame's
    /// marched neighbours. Without a history (the first frame, a resize, a jump, <see cref="EngineSettings.VolumetricCloudsFullRate"/>)
    /// it marches every texel. One per <see cref="DeferredRenderer"/>, with its own ComputeShader (its Params buffer
    /// holds this view's reprojection); the sky dome reads it through SceneConstants.CloudTargetIdx (0 = the 2D layer).
    /// Docs: .agent/knowledge/atmospheric_effects/volumetric_clouds.md.
    /// </summary>
    public sealed class CloudView : IDisposable
    {
        private const float TeleportDistance = 500f;    // metres in a frame: the history is dropped

        private sealed class Target
        {
            public RenderTexture2D Texture = null!;
            public ResourceStates State = ResourceStates.Common;
        }

        private ComputeShader? _cs;
        private int _kMarch, _kReconstruct;
        private Target? _sparse, _sparseDepth;
        private readonly Target?[] _color = new Target?[2], _depth = new Target?[2];
        private uint _frame;
        private bool _historyValid;
        private Matrix4x4 _prevRelativeVP;
        private Vector3 _prevCameraPos;
        private float _prevExposure;

        public int Width { get; private set; }
        public int Height { get; private set; }

        private const ResourceStates ShaderRead = ResourceStates.NonPixelShaderResource | ResourceStates.PixelShaderResource;

        /// <summary>Fit the targets to a view of this size and return the bindless index of the one this frame's clouds
        /// go into (0 when the volumetric clouds are off: the sky draws the 2D layer).</summary>
        public uint Prepare(int viewWidth, int viewHeight)
        {
            if (!Clouds.Active)
            {
                _historyValid = false;
                return 0;
            }
            int w = Math.Max(1, (viewWidth + 1) / 2), h = Math.Max(1, (viewHeight + 1) / 2);
            if (_color[0] == null || w != Width || h != Height)
            {
                DisposeTargets();
                var device = Engine.Device;
                Target Make(int tw, int th, Format format) =>
                    new() { Texture = new RenderTexture2D(device, tw, th, format, randomWrite: true) };
                for (int i = 0; i < 2; i++)
                {
                    _color[i] = Make(w, h, Format.R16G16B16A16_Float);     // pre-exposed light, transmittance
                    _depth[i] = Make(w, h, Format.R16_Float);              // km
                }
                _sparse = Make((w + 1) / 2, (h + 1) / 2, Format.R16G16B16A16_Float);
                _sparseDepth = Make((w + 1) / 2, (h + 1) / 2, Format.R16_Float);
                Width = w;
                Height = h;
                _historyValid = false;
            }
            return _color[_frame & 1]!.Texture.BindlessIndex;
        }

        /// <summary>
        /// March this view's clouds: after Clouds.Render (the bake, the light) and the atmosphere's LUTs, before the sky
        /// is drawn. The scene constants must be bound for compute by <paramref name="bindSceneConstants"/>;
        /// <paramref name="relativeVP"/> is the view-projection without translation.
        /// </summary>
        public void Render(ID3D12GraphicsCommandList list, Vector3 cameraPos, Matrix4x4 relativeVP,
            Action<ID3D12GraphicsCommandList> bindSceneConstants)
        {
            if (!Clouds.Active || _color[0] == null)
                return;
            if (_cs == null)
            {
                _cs = new ComputeShader("clouds_cs.hlsl");
                _kMarch = _cs.FindKernel("CSCloudMarch");
                _kReconstruct = _cs.FindKernel("CSCloudReconstruct");
            }
            var cs = _cs;
            var device = Engine.Device;
            int cur = (int)(_frame & 1), prev = cur ^ 1;
            float exposure = Atmosphere.Exposure;
            var move = cameraPos - _prevCameraPos;
            bool full = !_historyValid || Engine.Settings.VolumetricCloudsFullRate || _prevExposure <= 0f
                || move.LengthSquared() > TeleportDistance * TeleportDistance;

            PixMarker.Begin(list, "Clouds");
            list.SetComputeRootSignature(device.GlobalRootSignature);
            list.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            bindSceneConstants(list);

            cs.SetParam("PrevCameraRelativeVP", _prevRelativeVP);
            cs.SetParam("CameraMove", move);
            cs.SetParam("ExposureRatio", _prevExposure > 0f ? exposure / _prevExposure : 1f);
            cs.SetParam("CloudStep", Clouds.LastStep);

            var color = _color[cur]!; var depth = _depth[cur]!;
            if (full)
            {
                // Every texel, straight into this frame's target
                Transition(list, color, ResourceStates.UnorderedAccess);
                Transition(list, depth, ResourceStates.UnorderedAccess);
                SetMarch(cs, 0u, color.Texture.UavIndex, depth.Texture.UavIndex);
                cs.Dispatch(_kMarch, list, (uint)(Width + 7) / 8, (uint)(Height + 7) / 8);
            }
            else
            {
                // One texel of each 2×2 block into the sparse targets, then the reconstruct from them and the history
                Transition(list, _sparse!, ResourceStates.UnorderedAccess);
                Transition(list, _sparseDepth!, ResourceStates.UnorderedAccess);
                SetMarch(cs, 1u, _sparse!.Texture.UavIndex, _sparseDepth!.Texture.UavIndex);
                cs.Dispatch(_kMarch, list, (uint)((Width + 1) / 2 + 7) / 8, (uint)((Height + 1) / 2 + 7) / 8);

                Transition(list, _sparse, ShaderRead);
                Transition(list, _sparseDepth, ShaderRead);
                Transition(list, _color[prev]!, ShaderRead);
                Transition(list, color, ResourceStates.UnorderedAccess);
                Transition(list, depth, ResourceStates.UnorderedAccess);
                cs.SetPushConstant(_kReconstruct, "TargetUAV", color.Texture.UavIndex);
                cs.SetPushConstant(_kReconstruct, "DepthUAV", depth.Texture.UavIndex);
                cs.SetPushConstant(_kReconstruct, "TargetWidth", (uint)Width);
                cs.SetPushConstant(_kReconstruct, "TargetHeight", (uint)Height);
                cs.SetPushConstant(_kReconstruct, "FrameIndex", _frame);
                cs.SetPushConstant(_kReconstruct, "SparseTex", _sparse.Texture.BindlessIndex);
                cs.SetPushConstant(_kReconstruct, "SparseDepthTex", _sparseDepth.Texture.BindlessIndex);
                cs.SetPushConstant(_kReconstruct, "HistoryTex", _color[prev]!.Texture.BindlessIndex);
                cs.Dispatch(_kReconstruct, list, (uint)(Width + 7) / 8, (uint)(Height + 7) / 8);
            }
            Transition(list, color, ShaderRead);
            Transition(list, depth, ShaderRead);
            PixMarker.End(list);

            _prevRelativeVP = relativeVP;
            _prevCameraPos = cameraPos;
            _prevExposure = exposure;
            _historyValid = true;
            _frame++;
        }

        private void SetMarch(ComputeShader cs, uint mode, uint colorUav, uint depthUav)
        {
            cs.SetPushConstant(_kMarch, "MarchMode", mode);
            cs.SetPushConstant(_kMarch, "TargetUAV", colorUav);
            cs.SetPushConstant(_kMarch, "DepthUAV", depthUav);
            cs.SetPushConstant(_kMarch, "TargetWidth", (uint)Width);
            cs.SetPushConstant(_kMarch, "TargetHeight", (uint)Height);
            cs.SetPushConstant(_kMarch, "FrameIndex", _frame);
        }

        private static void Transition(ID3D12GraphicsCommandList list, Target target, ResourceStates to)
        {
            if (target.State == to)
            {
                if (to == ResourceStates.UnorderedAccess)
                    list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(target.Texture.Native)));
                return;
            }
            list.ResourceBarrierTransition(target.Texture.Native, target.State, to);
            target.State = to;
        }

        private void DisposeTargets()
        {
            for (int i = 0; i < 2; i++)
            {
                _color[i]?.Texture.Dispose(); _color[i] = null;
                _depth[i]?.Texture.Dispose(); _depth[i] = null;
            }
            _sparse?.Texture.Dispose(); _sparse = null;
            _sparseDepth?.Texture.Dispose(); _sparseDepth = null;
        }

        public void Dispose()
        {
            DisposeTargets();
            _cs?.Dispose();
        }
    }
}
