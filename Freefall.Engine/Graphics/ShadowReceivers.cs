using System;
using System.Collections.Generic;
using System.Numerics;
using Vortice.Direct3D12;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// Sun shadow receivers: where this frame's visible surfaces sample each cascade, as the deepest receiver light depth
    /// per tile of a 64×64 grid over the cascade (shadow_receivers.hlsl; layout in shadow_receivers.hlsli). The cascade
    /// culling (CSVisibilityShadow4, push constant slot 39) tests every cascade entry against it: a caster whose nearest
    /// point lies deeper than every receiver under its footprint darkens nothing the camera sees through that cascade
    /// (receiver-mask caster culling, Bittner et al. 2011).
    /// Built in DirectionalLight.DrawShadows, between the opaque pass (this frame's linear depth exists) and the
    /// cascade culling. EngineSettings.ShadowReceiverCulling: Off, Count (stats only) or On (such entries are left out).
    /// </summary>
    public sealed class ShadowReceivers : IDisposable
    {
        // shadow_receivers.hlsli
        private const int HeaderWords = 32;
        private const int CascadeWords = 5461;
        private const int MaxCascades = 4;          // CSVisibilityShadow4 tests four
        private const int HeaderInstances = 2, HeaderInstancesCut = 3, HeaderTiles = 4, HeaderEntries = 8,
                          HeaderEntriesCut = 12, HeaderTris = 16, HeaderTrisCut = 20;
        private const uint ModeCount = 1, ModeCull = 2;
        private const int MaxOceanBands = 8;            // RECEIVER_MAX_BANDS
        private const uint MaxVisibleLights = 4096;     // PointLightBuffer.MaxVisible

        // How far the lighting's shadow lookup reaches from a receiver, in texels: GetShadowFactor's PCF disk
        // (2.5 texels from cascade 2 on) plus the comparison sampler's bilinear footprint, rounded up
        private const float FilterReachTexels = 4f;

        // Metres added to the ocean's measured wave heights (plus its shore surge): the heights are this frame's,
        // the surface the forward pass draws later in the frame moves on by a frame at most
        private const float OceanSlack = 0.25f;

        private const int FrameCount = 3;

        private readonly ComputeShader _shader;
        private readonly int _waveBoundKernel, _tilesKernel, _spheresKernel, _lightsKernel, _pyramidKernel;
        private GraphicsBuffer? _buffer;
        private readonly GraphicsBuffer?[] _readback = new GraphicsBuffer?[FrameCount];
        private readonly bool[] _readbackFilled = new bool[FrameCount];
        private int _lastBuildTick = -1;

        // Header of the last build read back (frames behind), null before the first
        private uint[]? _stats;

        // One per renderer (scene view, game view): the stats route asks all of them
        private static readonly List<ShadowReceivers> _instances = new();

        public ShadowReceivers()
        {
            _shader = new ComputeShader("shadow_receivers.hlsl");
            _waveBoundKernel = _shader.FindKernel("CSOceanWaveBound");
            _tilesKernel = _shader.FindKernel("CSReceiverTiles");
            _spheresKernel = _shader.FindKernel("CSReceiverSpheres");
            _lightsKernel = _shader.FindKernel("CSReceiverLights");
            _pyramidKernel = _shader.FindKernel("CSReceiverPyramid");
            lock (_instances) _instances.Add(this);
        }

        /// <summary>Describe() of every renderer that has built receivers, for engine_stats.</summary>
        public static object DescribeAll()
        {
            lock (_instances)
            {
                var result = new List<object>(_instances.Count);
                foreach (var instance in _instances)
                    if (instance.Describe() is { } description)
                        result.Add(description);
                return result;
            }
        }

        /// <summary>
        /// Builds this frame's receiver tiles and returns the UAV for push constant slot 39 of the cascade culling
        /// (0 = off: the culling tests nothing). Receivers: the linear depth's pixels, the ocean's wave band along each
        /// view ray, the forward instances (glass, water bodies) the camera saw last frame, and the sun samples of the
        /// point light shadow selection around each visible light (the light cull runs before the shadow pass).
        /// </summary>
        /// <param name="renderer">Its DepthGBuffer (linear view depth) is in RenderTarget state and left there.</param>
        /// <param name="cascadeSrv">This frame's StructuredBuffer&lt;CascadeData&gt; (VP, split distances).</param>
        public uint Build(ID3D12GraphicsCommandList cmd, Camera camera, DeferredRenderer renderer, uint cascadeSrv,
            int cascadeCount, int shadowResolution)
        {
            var mode = Engine.Settings.ShadowReceiverCulling;
            if (mode == ShadowReceiverCulling.Off)
                return 0;

            EnsureBuffers();
            var buffer = _buffer!;
            var depth = renderer.DepthGBuffer;
            int f = Engine.FrameIndex % FrameCount;

            if (_lastBuildTick != Engine.TickCount)
            {
                // The stats copied two frames ago are complete on the CPU side now
                int done = (Engine.FrameIndex + 1) % FrameCount;
                if (_readbackFilled[done])
                    ReadStats(done);

                // The last build's header, all batches culled: the buffer has decayed to Common after that frame's
                // command list, so the copy promotes it to CopySource on its own
                if (_lastBuildTick >= 0)
                {
                    cmd.CopyBufferRegion(_readback[f]!.Native, 0, buffer.Native, 0, HeaderWords * sizeof(uint));
                    cmd.ResourceBarrierTransition(buffer.Native, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                    _readbackFilled[f] = true;
                }
            }
            else
            {
                // A second view this frame: after the first one's culling read it
                buffer.UAVBarrier(cmd);
            }
            _lastBuildTick = Engine.TickCount;

            PixMarker.Begin(cmd, "Shadow receivers");
            buffer.ClearUAV(cmd, new Vortice.Mathematics.Int4(0, 0, 0, 0));
            buffer.UAVBarrier(cmd);

            cmd.ResourceBarrierTransition(depth.Native, ResourceStates.RenderTarget, ResourceStates.NonPixelShaderResource);

            cmd.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });

            int width = (int)depth.Native.Description.Width;
            int height = (int)depth.Native.Description.Height;

            // Camera-relative rays as the lighting rebuilds positions: view rotation from forward and up
            var view = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, camera.Forward, camera.Up);
            var projection = camera.Projection;
            _shader.SetParam("CameraPosition", new Vector4(camera.Position, 0f));
            _shader.SetParam("RayRight", new Vector4(view.M11, view.M21, view.M31, 0f) / projection.M11);
            _shader.SetParam("RayUp", new Vector4(view.M12, view.M22, view.M32, 0f) / projection.M22);
            _shader.SetParam("RayForward", new Vector4(view.M13, view.M23, view.M33, 0f));

            int cascades = Math.Min(cascadeCount, MaxCascades);
            _shader.SetPushConstant("DepthTex", depth.BindlessIndex);
            _shader.SetPushConstant("Cascades", cascadeSrv);
            _shader.SetPushConstant("ReceiversUAV", buffer.UavIndex);
            _shader.SetPushConstant("CascadeCount", (uint)cascades);
            _shader.SetPushConstant("ScreenWidth", (uint)width);
            _shader.SetPushConstant("ScreenHeight", (uint)height);
            _shader.SetPushConstant("Mode", mode == ShadowReceiverCulling.On ? ModeCull : ModeCount);
            _shader.SetPushConstant("MarginUV", BitConverter.SingleToUInt32Bits(FilterReachTexels / shadowResolution));

            // The ocean: its bands' largest wave heights first, for the band each view ray crosses
            var ocean = OceanRenderer.DrawnThisTick;
            int bands = Math.Min(ocean?.BandCount ?? 0, MaxOceanBands);
            uint displacement = bands > 0 ? ocean!.DisplacementSrv : 0u;
            _shader.SetPushConstant("OceanDisplacement", displacement);
            _shader.SetPushConstant("OceanBandCount", (uint)bands);
            if (displacement != 0)
            {
                _shader.SetPushConstant("OceanPlaneY", BitConverter.SingleToUInt32Bits(ocean!.PlaneY));
                _shader.SetPushConstant("OceanMargin", BitConverter.SingleToUInt32Bits(MathF.Abs(ocean.ShoreSurge) + OceanSlack));
                _shader.Dispatch(_waveBoundKernel, cmd, OceanFFT.N / 8, OceanFFT.N / 8, (uint)bands);
                buffer.UAVBarrier(cmd);
            }

            // The pixels (and the ocean band), the forward instances and the light samples only max-merge into level 0
            _shader.Dispatch(_tilesKernel, cmd, (uint)(width + 7) / 8, (uint)(height + 7) / 8);
            MarkForwardInstances(cmd, ocean);
            var (lights, lightCount) = renderer.VisiblePointLights;
            if (lights != 0)
            {
                _shader.SetPushConstant("VisibleLights", lights);
                _shader.SetPushConstant("VisibleLightCount", lightCount);
                _shader.Dispatch(_lightsKernel, cmd, MaxVisibleLights / 64);
            }
            buffer.UAVBarrier(cmd);
            _shader.Dispatch(_pyramidKernel, cmd, (uint)cascades);
            buffer.UAVBarrier(cmd);

            cmd.ResourceBarrierTransition(depth.Native, ResourceStates.NonPixelShaderResource, ResourceStates.RenderTarget);
            PixMarker.End(cmd);

            return buffer.UavIndex;
        }

        // The forward batches' instances the camera saw last frame (glass, water bodies; not the ocean, whose sphere would
        // cover everything: its band is marked per pixel). Forward batches are culled in the forward pass, after the
        // shadows, so last frame's culling is the newest there is; an instance seen for the first time this frame is
        // not marked, which can leave its shadow out for one frame.
        private void MarkForwardInstances(ID3D12GraphicsCommandList cmd, OceanRenderer? ocean)
        {
            var batches = CommandBuffer.GetAllBatches(RenderPass.Forward);
            if (batches == null)
                return;
            foreach (var batch in batches)
            {
                if (ocean != null && ReferenceEquals(batch.Material.Effect, ocean.SurfaceEffect))
                    continue;
                if (!batch.RecentCameraCull(out uint bounds, out uint flags, out int count))
                    continue;
                _shader.SetPushConstant("SphereBounds", bounds);
                _shader.SetPushConstant("SphereFlags", flags);
                _shader.SetPushConstant("SphereCount", (uint)count);
                _shader.Dispatch(_spheresKernel, cmd, (uint)(count + 63) / 64);
            }
        }

        private void EnsureBuffers()
        {
            if (_buffer != null) return;
            _buffer = GraphicsBuffer.CreateStructured<uint>(HeaderWords + MaxCascades * CascadeWords, uav: true);
            for (int i = 0; i < FrameCount; i++)
                _readback[i] = GraphicsBuffer.CreateReadback<uint>(HeaderWords);
        }

        private unsafe void ReadStats(int index)
        {
            _stats ??= new uint[HeaderWords];
            uint* src = _readback[index]!.ReadPtr<uint>();
            for (int i = 0; i < HeaderWords; i++)
                _stats[i] = src[i];
        }

        /// <summary>
        /// What the receiver test found in a frame a few frames back, for engine_stats: per cascade the tiles with a
        /// receiver (of 4096), the cascade entries (instance × cascade) and their triangles, and how many of them shade
        /// no receiver. Null while off or before the first readback.
        /// </summary>
        public object? Describe()
        {
            if (_stats == null || Engine.Settings.ShadowReceiverCulling == ShadowReceiverCulling.Off)
                return null;
            var s = _stats;

            var cascades = new object[MaxCascades];
            ulong entries = 0, entriesCut = 0, tris = 0, trisCut = 0;
            for (int c = 0; c < MaxCascades; c++)
            {
                entries += s[HeaderEntries + c];
                entriesCut += s[HeaderEntriesCut + c];
                tris += s[HeaderTris + c];
                trisCut += s[HeaderTrisCut + c];
                cascades[c] = new
                {
                    tilesWithReceivers = s[HeaderTiles + c],
                    entries = s[HeaderEntries + c],
                    entriesCut = s[HeaderEntriesCut + c],
                    entriesCutPct = Percent(s[HeaderEntriesCut + c], s[HeaderEntries + c]),
                    triangles = s[HeaderTris + c],
                    trianglesCut = s[HeaderTrisCut + c],
                    trianglesCutPct = Percent(s[HeaderTrisCut + c], s[HeaderTris + c]),
                };
            }

            return new
            {
                mode = Engine.Settings.ShadowReceiverCulling.ToString(),
                instances = s[HeaderInstances],
                instancesCut = s[HeaderInstancesCut],
                entries,
                entriesCut,
                entriesCutPct = Percent(entriesCut, entries),
                triangles = tris,
                trianglesCut = trisCut,
                trianglesCutPct = Percent(trisCut, tris),
                cascades,
            };
        }

        private static double Percent(ulong part, ulong whole) => whole == 0 ? 0 : Math.Round(100.0 * part / whole, 1);

        public void Dispose()
        {
            lock (_instances) _instances.Remove(this);
            var device = Engine.Device;
            if (_buffer != null)
                device.DeferDispose(_buffer);
            for (int i = 0; i < FrameCount; i++)
                if (_readback[i] != null)
                    device.DeferDispose(_readback[i]!);
            _buffer = null;
            _shader.Dispose();
        }
    }
}
