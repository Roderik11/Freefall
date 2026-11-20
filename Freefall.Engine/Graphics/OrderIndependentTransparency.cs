using System;
using Freefall.Components;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>
    /// Order-independent transparency (EngineSettings.Transparency; oit.hlsli, oit_resolve_cs.hlsl): glass, particles,
    /// trails and vfx drawn in any order after the Forward pass's water, into two accumulation targets (RenderPass.OIT),
    /// then laid over the composite. A third target of that pass is the temporal upscaler's reactive mask
    /// (ComposeReactive). With Moments a pass before it (RenderPass.OITMoments) sums the moments of the
    /// absorbance over depth, so each fragment is weighed by what lies in front of it. The water stays the one ordered
    /// layer: it refracts the scene behind it and writes depth. A split compare (TransparencySplit) draws the left half
    /// with another mode in the same frame, the sorted draws kept to their side by the scissor.
    /// Docs: .agent/knowledge/rendering_engine/features/order_independent_transparency.md
    /// </summary>
    public sealed class OrderIndependentTransparency : IDisposable
    {
        // This view's columns (Begin), read by the draws that run in its passes

        /// <summary>Some of this view's pixels take the transparents through the accumulation.</summary>
        public static bool Active { get; private set; }

        /// <summary>Some show the sorted draws of the Transparent and Forward passes: while Active these keep to SortedScissor.</summary>
        public static bool Sorted { get; private set; } = true;

        /// <summary>The moments pass runs this view.</summary>
        public static bool MomentsDue { get; private set; }

        public static RectI SortedScissor { get; private set; }
        public static RectI FullScissor { get; private set; }

        private RenderTexture2D? _accum, _extra, _zeroth, _moments;
        private bool _accumNew, _momentsNew;            // just made: still in Common

        // The reactive mask: the accumulation's third target, two in turn so last frame's is still there
        // (oit_reactive_cs.hlsl), and the mask composed from both that the upscaler reads
        private readonly RenderTexture2D?[] _reactive = new RenderTexture2D?[2];
        private readonly ResourceStates[] _reactiveState = { ResourceStates.Common, ResourceStates.Common };
        private RenderTexture2D? _reactiveMask;
        private ResourceStates _maskState = ResourceStates.Common;
        private int _reactiveNow;                       // which of the two this frame's accumulation wrote
        private int _drawnTick = int.MinValue, _prevDrawnTick = int.MinValue;   // Engine.TickCount of the last two
        private bool _reactiveDrawn;                    // the accumulation ran for this view
        private bool _maskComposed;                     // ComposeReactive ran for this view

        private int _width, _height;
        private int _resolveFrom, _resolveTo, _momentsFrom, _momentsTo;
        private float _logNear, _logScale;
        private readonly ComputeShader _resolve, _reactiveCS;
        private readonly int _kResolve, _kReactive, _kReactiveView;

        public OrderIndependentTransparency()
        {
            _resolve = new ComputeShader("oit_resolve_cs.hlsl");
            _kResolve = _resolve.FindKernel("CSResolve");
            _reactiveCS = new ComputeShader("oit_reactive_cs.hlsl");
            _kReactive = _reactiveCS.FindKernel("CSReactive");
            _kReactiveView = _reactiveCS.FindKernel("CSReactiveView");
        }

        /// <summary>
        /// This view's split, before anything is drawn: which columns show which mode, so the draw systems enqueue for
        /// the right passes and the sorted draws keep to their side.
        /// </summary>
        public void Begin(Camera camera, int width, int height)
        {
            var settings = Engine.Settings;
            int split = settings.TransparencySplit ? width / 2 : 0;
            var left = settings.TransparencySplitLeft;
            var right = settings.Transparency;

            (int from, int to) Columns(Func<TransparencyMode, bool> wanted)
            {
                bool l = split > 0 && wanted(left), r = wanted(right);
                return l && r ? (0, width) : l ? (0, split) : r ? (split, width) : (0, 0);
            }
            var sorted = Columns(m => m == TransparencyMode.Sorted);
            (_resolveFrom, _resolveTo) = Columns(m => m != TransparencyMode.Sorted);
            (_momentsFrom, _momentsTo) = Columns(m => m == TransparencyMode.Moments);

            Active = _resolveTo > _resolveFrom;
            _reactiveDrawn = false;
            _maskComposed = false;
            Sorted = sorted.to > sorted.from;
            MomentsDue = _momentsTo > _momentsFrom;
            FullScissor = new RectI(0, 0, width, height);
            SortedScissor = RectI.FromLTRB(sorted.from, 0, sorted.to, height);

            // The moments' depth, logarithmic from the near plane to the far one
            float near = MathF.Max(camera.NearPlane, 0.01f);
            float far = MathF.Max(camera.FarPlane, near * 2f);
            _logNear = MathF.Log(near);
            _logScale = 2f / (MathF.Log(far) - _logNear);

            if (Active) EnsureTargets(width, height);
        }

        /// <summary>
        /// After the camera's motion pass, before the temporal upscaler: the reactive mask it reads (R8_UNorm at the
        /// render size, left in NonPixelShaderResource), how far to trust this frame over its history. This frame's
        /// mask from the accumulation (where particles, trails and vfx are now, oit.hlsli OITReactive; 0 elsewhere,
        /// glass included), with last frame's taken where this pixel's history is read from (oit_reactive_cs.hlsl), so
        /// what has moved on leaves no fading trail. Null when no pixel took the transparents through the accumulation
        /// (EngineSettings.Transparency Sorted). <paramref name="motion"/>: the Motion target, readable by compute;
        /// <paramref name="reset"/>: the history starts over (a cut), last frame's mask is not taken.
        /// </summary>
        public RenderTexture2D? ComposeReactive(ID3D12GraphicsCommandList list, RenderTexture2D motion, bool reset)
        {
            if (!_reactiveDrawn || _reactiveMask == null) return null;
            var now = _reactive[_reactiveNow]!;
            var prev = _reactive[_reactiveNow ^ 1];
            bool hasPrev = !reset && prev != null && _prevDrawnTick == _drawnTick - 1
                           && _reactiveState[_reactiveNow ^ 1] == ResourceStates.NonPixelShaderResource;

            PixMarker.Begin(list, "Reactive mask");
            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            if (_maskState != ResourceStates.UnorderedAccess)
                list.ResourceBarrierTransition(_reactiveMask.Native, _maskState, ResourceStates.UnorderedAccess);
            _reactiveCS.SetPushConstant(_kReactive, "Now", now.BindlessIndex);
            _reactiveCS.SetPushConstant(_kReactive, "Prev", hasPrev ? prev!.BindlessIndex : 0u);
            _reactiveCS.SetPushConstant(_kReactive, "Motion", motion.BindlessIndex);
            _reactiveCS.SetPushConstant(_kReactive, "OutUAV", _reactiveMask.UavIndex);
            _reactiveCS.SetPushConstant(_kReactive, "RenderWidth", (uint)_width);
            _reactiveCS.SetPushConstant(_kReactive, "RenderHeight", (uint)_height);
            _reactiveCS.Dispatch(_kReactive, list, (uint)(_width + 7) / 8, (uint)(_height + 7) / 8);
            list.ResourceBarrierTransition(_reactiveMask.Native, ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
            _maskState = ResourceStates.NonPixelShaderResource;
            PixMarker.End(list);
            _maskComposed = true;
            return _reactiveMask;
        }

        /// <summary>
        /// DebugVizMode.Reactive: the composed mask painted red over the tonemapped picture (<paramref name="ldr"/>,
        /// display size, in PixelShaderResource before and after). Nothing when no mask was composed this view (no
        /// temporal upscaler, or Sorted).
        /// </summary>
        public void ReactiveView(ID3D12GraphicsCommandList list, RenderTexture2D ldr, int displayWidth, int displayHeight)
        {
            if (!_maskComposed || _reactiveMask == null) return;
            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            list.ResourceBarrierTransition(ldr.Native, ResourceStates.PixelShaderResource, ResourceStates.UnorderedAccess);
            _reactiveCS.SetPushConstant(_kReactiveView, "Mask", _reactiveMask.BindlessIndex);
            _reactiveCS.SetPushConstant(_kReactiveView, "OutputUAV", ldr.UavIndex);
            _reactiveCS.SetPushConstant(_kReactiveView, "RenderWidth", (uint)_width);
            _reactiveCS.SetPushConstant(_kReactiveView, "RenderHeight", (uint)_height);
            _reactiveCS.SetPushConstant(_kReactiveView, "DisplayWidth", (uint)displayWidth);
            _reactiveCS.SetPushConstant(_kReactiveView, "DisplayHeight", (uint)displayHeight);
            _reactiveCS.Dispatch(_kReactiveView, list, (uint)(displayWidth + 7) / 8, (uint)(displayHeight + 7) / 8);
            list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(ldr.Native)));
            list.ResourceBarrierTransition(ldr.Native, ResourceStates.UnorderedAccess, ResourceStates.PixelShaderResource);
        }

        /// <summary>The scene constants every effect reads (common.fx SceneConstants, oit.hlsli).</summary>
        public void SetParameters(Effect effect)
        {
            bool moments = Active && MomentsDue && _zeroth != null;
            effect.SetParameter("OITZerothIdx", moments ? _zeroth!.BindlessIndex : 0u);
            effect.SetParameter("OITMomentsIdx", moments ? _moments!.BindlessIndex : 0u);
            effect.SetParameter("OITMomentsFrom", (float)_momentsFrom);
            effect.SetParameter("OITMomentsTo", (float)_momentsTo);
            effect.SetParameter("OITLogNear", _logNear);
            effect.SetParameter("OITLogScale", _logScale);
        }

        private void EnsureTargets(int width, int height)
        {
            if (width != _width || height != _height)
            {
                ReleaseTextures();
                _width = width;
                _height = height;
            }
            if (_accum == null)
            {
                _accum = new RenderTexture2D(Engine.Device, width, height, Format.R16G16B16A16_Float);
                _extra = new RenderTexture2D(Engine.Device, width, height, Format.R16G16B16A16_Float);
                for (int i = 0; i < 2; i++)
                {
                    _reactive[i] = new RenderTexture2D(Engine.Device, width, height, Format.R8_UNorm);
                    _reactiveState[i] = ResourceStates.Common;
                }
                _reactiveMask = new RenderTexture2D(Engine.Device, width, height, Format.R8_UNorm, randomWrite: true);
                _maskState = ResourceStates.Common;
                _drawnTick = _prevDrawnTick = int.MinValue;
                _accumNew = true;
            }
            if (MomentsDue && _zeroth == null)
            {
                _zeroth = new RenderTexture2D(Engine.Device, width, height, Format.R32_Float);
                _moments = new RenderTexture2D(Engine.Device, width, height, Format.R32G32B32A32_Float);
                _momentsNew = true;
            }
        }

        /// <summary>
        /// The passes and the resolve, right after the Forward pass's draws: <paramref name="composite"/> bound as a
        /// render target with <paramref name="depth"/> (DepthWrite) on entry, in RenderTarget again on return (the caller
        /// binds it again).
        /// </summary>
        public void Render(ID3D12GraphicsCommandList list, RenderTexture2D composite, DepthTexture2D depth)
        {
            var device = Engine.Device;
            var viewport = new Viewport(0, 0, _width, _height);
            var clear = new Color4(0, 0, 0, 0);
            PixMarker.Begin(list, "OIT");

            if (MomentsDue)
            {
                PixMarker.Begin(list, "Moments");
                var from = _momentsNew ? ResourceStates.Common : ResourceStates.PixelShaderResource;
                list.ResourceBarrierTransition(_zeroth!.Native, from, ResourceStates.RenderTarget);
                list.ResourceBarrierTransition(_moments!.Native, from, ResourceStates.RenderTarget);
                _momentsNew = false;
                list.ClearRenderTargetView(_zeroth.RtvHandle, clear);
                list.ClearRenderTargetView(_moments.RtvHandle, clear);
                list.OMSetRenderTargets(new[] { _zeroth.RtvHandle, _moments.RtvHandle }, depth.DsvHandle);
                list.RSSetViewport(viewport);
                list.RSSetScissorRect(FullScissor);
                CommandBuffer.Execute(RenderPass.OITMoments, list, device);
                list.ResourceBarrierTransition(_zeroth.Native, ResourceStates.RenderTarget, ResourceStates.PixelShaderResource);
                list.ResourceBarrierTransition(_moments.Native, ResourceStates.RenderTarget, ResourceStates.PixelShaderResource);
                PixMarker.End(list);
            }

            PixMarker.Begin(list, "Accumulate");
            var accumFrom = _accumNew ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
            list.ResourceBarrierTransition(_accum!.Native, accumFrom, ResourceStates.RenderTarget);
            list.ResourceBarrierTransition(_extra!.Native, accumFrom, ResourceStates.RenderTarget);
            // This frame's mask goes into the other of the two: last frame's stays for ComposeReactive
            _reactiveNow ^= 1;
            _prevDrawnTick = _drawnTick;
            _drawnTick = Engine.TickCount;
            var reactive = _reactive[_reactiveNow]!;
            list.ResourceBarrierTransition(reactive.Native, _reactiveState[_reactiveNow], ResourceStates.RenderTarget);
            _accumNew = false;
            list.ClearRenderTargetView(_accum.RtvHandle, clear);
            list.ClearRenderTargetView(_extra.RtvHandle, clear);
            list.ClearRenderTargetView(reactive.RtvHandle, clear);
            list.OMSetRenderTargets(new[] { _accum.RtvHandle, _extra.RtvHandle, reactive.RtvHandle }, depth.DsvHandle);
            list.RSSetViewport(viewport);
            list.RSSetScissorRect(FullScissor);
            CommandBuffer.Execute(RenderPass.OIT, list, device);
            list.ResourceBarrierTransition(_accum.Native, ResourceStates.RenderTarget, ResourceStates.NonPixelShaderResource);
            list.ResourceBarrierTransition(_extra.Native, ResourceStates.RenderTarget, ResourceStates.NonPixelShaderResource);
            list.ResourceBarrierTransition(reactive.Native, ResourceStates.RenderTarget, ResourceStates.NonPixelShaderResource);
            _reactiveState[_reactiveNow] = ResourceStates.NonPixelShaderResource;
            _reactiveDrawn = true;
            PixMarker.End(list);

            PixMarker.Begin(list, "Resolve");
            list.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            list.SetComputeRootSignature(device.GlobalRootSignature);
            _resolve.SetPushConstant("CompositeUAV", composite.UavIndex);
            _resolve.SetPushConstant("Accum", _accum.BindlessIndex);
            _resolve.SetPushConstant("Extra", _extra.BindlessIndex);
            _resolve.SetPushConstant("ResolveFrom", (uint)_resolveFrom);
            _resolve.SetPushConstant("ResolveTo", (uint)_resolveTo);
            _resolve.SetPushConstant("Height", (uint)_height);
            list.ResourceBarrierTransition(composite.Native, ResourceStates.RenderTarget, ResourceStates.UnorderedAccess);
            _resolve.Dispatch(_kResolve, list, (uint)(_resolveTo + 7) / 8, (uint)(_height + 7) / 8);
            list.ResourceBarrierTransition(composite.Native, ResourceStates.UnorderedAccess, ResourceStates.RenderTarget);
            PixMarker.End(list);

            PixMarker.End(list);
        }

        private void ReleaseTextures()
        {
            // RenderTexture2D.Dispose hides Texture.Dispose, so wrap it or DeferDispose would skip the descriptors
            foreach (var texture in new[] { _accum, _extra, _reactive[0], _reactive[1], _reactiveMask, _zeroth, _moments })
                if (texture != null)
                    Engine.Device.DeferDispose(new DeferredTexture(texture));
            _accum = _extra = _reactiveMask = _zeroth = _moments = null;
            _reactive[0] = _reactive[1] = null;
            _reactiveDrawn = _maskComposed = false;
        }

        private sealed class DeferredTexture(RenderTexture2D texture) : IDisposable
        {
            public void Dispose() => texture.Dispose();
        }

        public void Dispose()
        {
            _accum?.Dispose();
            _extra?.Dispose();
            _reactive[0]?.Dispose();
            _reactive[1]?.Dispose();
            _reactiveMask?.Dispose();
            _reactiveCS.Dispose();
            _zeroth?.Dispose();
            _moments?.Dispose();
            _resolve.Dispose();
        }
    }
}
