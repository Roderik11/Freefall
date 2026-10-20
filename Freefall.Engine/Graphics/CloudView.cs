using System;
using System.Numerics;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// A view's volumetric clouds (clouds_cs.hlsl): their own light (× Exposure; the air in front of them is the sky
    /// dome's, from the aerial perspective volume: sky_common.fx AtmosphereSkyWithClouds), what they let through and how
    /// far they are (<see cref="DepthIndex"/>), at half the view's size each way. Each frame marches one
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
        private int _kMarch, _kReconstruct, _kSceneDepth;
        private Target? _sparse, _sparseDepth;
        private readonly Target?[] _color = new Target?[2], _depth = new Target?[2], _sceneDepth = new Target?[2];
        private uint _frame;
        private bool _historyValid;
        private bool _prevSceneValid;      // last frame's scene depth was made
        private Matrix4x4 _prevRelativeVP;
        private Vector3 _prevCameraPos;
        private float _prevExposure;

        public int Width { get; private set; }
        public int Height { get; private set; }

        /// <summary>Bindless index of this frame's cloud depth (R16F, km), the target <see cref="Prepare"/> returned
        /// goes with: SceneConstants.CloudDepthIdx (the sky dome's shadows over the sky).</summary>
        public uint DepthIndex => _depth[_frame & 1]?.Texture.BindlessIndex ?? 0u;

        /// <summary>Bindless index of this frame's half-size scene depth (R32F, m of view depth, 0 = the sky): what each texel
        /// was marched to, a checkerboard of each 2×2's nearest and farthest surface (SceneConstants.CloudSceneDepthIdx).</summary>
        public uint SceneDepthIndex => _sceneDepth[_frame & 1]?.Texture.BindlessIndex ?? 0u;

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
                    _sceneDepth[i] = Make(w, h, Format.R32_Float);         // m of view depth
                }
                _sparse = Make((w + 1) / 2, (h + 1) / 2, Format.R16G16B16A16_Float);
                _sparseDepth = Make((w + 1) / 2, (h + 1) / 2, Format.R16_Float);
                Width = w;
                Height = h;
                _historyValid = false;
                _prevSceneValid = false;
            }
            return _color[_frame & 1]!.Texture.BindlessIndex;
        }

        /// <summary>
        /// March this view's clouds: after Clouds.Render (the bake, the light) and the atmosphere's LUTs, before the sky
        /// is drawn. The scene constants must be bound for compute by <paramref name="bindSceneConstants"/>;
        /// <paramref name="relativeVP"/> is the view-projection without translation; <paramref name="sceneDepthIdx"/> the
        /// G-buffer's linear depth (readable by compute; 0 = march to the slab's end): the march stops at the surface.
        /// </summary>
        public void Render(ID3D12GraphicsCommandList list, Vector3 cameraPos, Matrix4x4 relativeVP,
            Action<ID3D12GraphicsCommandList> bindSceneConstants, uint sceneDepthIdx)
        {
            if (!Clouds.Active || _color[0] == null)
                return;
            if (_cs == null)
            {
                _cs = new ComputeShader("clouds_cs.hlsl");
                _kMarch = _cs.FindKernel("CSCloudMarch");
                _kReconstruct = _cs.FindKernel("CSCloudReconstruct");
                _kSceneDepth = _cs.FindKernel("CSCloudSceneDepth");
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

            // The scene's depth at half size, what each texel is marched to (a checkerboard of each 2×2's nearest and farthest)
            var scene = _sceneDepth[cur]!;
            uint sceneIdx = 0;
            if (sceneDepthIdx != 0)
            {
                Transition(list, scene, ResourceStates.UnorderedAccess);
                cs.SetPushConstant(_kSceneDepth, "SourceTex", sceneDepthIdx);
                cs.SetPushConstant(_kSceneDepth, "TargetUAV", scene.Texture.UavIndex);
                cs.SetPushConstant(_kSceneDepth, "TargetWidth", (uint)Width);
                cs.SetPushConstant(_kSceneDepth, "TargetHeight", (uint)Height);
                cs.Dispatch(_kSceneDepth, list, (uint)(Width + 7) / 8, (uint)(Height + 7) / 8);
                Transition(list, scene, ShaderRead);
                sceneIdx = scene.Texture.BindlessIndex;
            }
            cs.SetPushConstant(_kMarch, "SceneDepthTex", sceneIdx);

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
                Transition(list, _depth[prev]!, ShaderRead);
                cs.SetPushConstant(_kReconstruct, "HistoryDepthTex", _depth[prev]!.Texture.BindlessIndex);
                // Last frame's scene depth: the history is dropped where the surface there was another (a silhouette moved)
                Transition(list, _sceneDepth[prev]!, ShaderRead);
                cs.SetPushConstant(_kReconstruct, "SceneDepthTex", sceneIdx);
                cs.SetPushConstant(_kReconstruct, "PrevSceneDepthTex", sceneIdx != 0 && _prevSceneValid ? _sceneDepth[prev]!.Texture.BindlessIndex : 0u);
                cs.Dispatch(_kReconstruct, list, (uint)(Width + 7) / 8, (uint)(Height + 7) / 8);
            }
            Transition(list, color, ShaderRead);
            Transition(list, depth, ShaderRead);
            PixMarker.End(list);

            _prevRelativeVP = relativeVP;
            _prevCameraPos = cameraPos;
            _prevExposure = exposure;
            _historyValid = true;
            _prevSceneValid = sceneIdx != 0;
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
                _sceneDepth[i]?.Texture.Dispose(); _sceneDepth[i] = null;
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
