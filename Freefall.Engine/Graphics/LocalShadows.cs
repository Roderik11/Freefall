using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.Mathematics;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// Point light shadows, picked and laid out on the GPU (local_shadow_select.hlsl) from the renderer's visible
    /// light list (point_light_cull.hlsl). Every light in view competes: the GPU ranks them by brightness × the
    /// screen share of their sphere (lower when their centre is hidden), gives the first
    /// EngineSettings.MaxShadowedPointLights 512², 256² or 128² tiles for their visible cube faces, fades shadows
    /// in and out, and builds the faces' views. Render() then culls every opaque batch against all faces at once
    /// and draws each batch once (LocalShadow pass); the point light shader samples through the light's face table.
    ///
    /// The CPU only dispatches; it learns what was picked 2-3 frames later, for the editor's stats.
    /// The atlas is not part of the sun's cascade array: the cascade pass is skipped whenever no directional
    /// light casts, and that must not take the torches' shadows with it.
    /// Shader side: local_shadows.hlsli, point_lights.hlsli, local_shadow_select.hlsl, cull_instances.hlsl
    /// (CS*LocalShadows), common.fx (ShadowClipPosition), light_point_cs.hlsl (sampling).
    /// </summary>
    public sealed class LocalShadows : IDisposable
    {
        public const uint NoShadow = 0xFFFFFFFF;

        public const int PageSize = 2048;
        public const int PageCount = 4;

        /// <summary>Views (cube faces) per frame: the culler's view mask is a uint4.</summary>
        public const int MaxViews = 128;
        public const int MaxLights = 32;

        private const int MaxCandidates = 1024;
        private const int ScoreBins = 64;
        private const int CountSlots = 16;
        private const int FrameCount = 3;

        // local_shadow_select.hlsl COUNT_*
        private const int CountLights = 0, CountViews = 1, CountSlotsUsed = 2, CountFacesTier0 = 3, CountCandidates = 9;
        private const int SlotCount = 64;
        private static readonly int[] TierSize = { 512, 256, 128 };

        /// <summary>One rendered cube face. Must match LocalShadowView in local_shadows.hlsli (256 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct View
        {
            public Vector4 Plane0, Plane1, Plane2, Plane3, Plane4, Plane5;
            public Matrix4x4 VP;
            public Matrix4x4 SampleVP;
            public Vector4 Tile;
            public uint Page;
            public uint Light;
            public float TexelPerMetre;
            private uint _pad;
        }

        /// <summary>A light with a shadow this frame. Must match LocalShadowLight in local_shadows.hlsli (48 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct ShadowLight
        {
            public Vector3 Position;
            public float Range;
            public uint View0, View1, View2, View3, View4, View5;
            public float Strength;
            private float _pad;
        }

        /// <summary>Per registry slot. Must match LightShadowState in local_shadow_select.hlsl (16 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct LightState
        {
            public float Fade;
            public uint Tier;
            public uint Seen;
            public float SunFactor;
        }

        /// <summary>Must match ShadowPick in local_shadow_select.hlsl (16 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct Pick
        {
            public uint VisibleIndex;
            public uint Slot;
            public uint Packed;     // faces | tier << 8 | wanted << 16
            public float Fade;
        }

        public DepthTextureArray2D Atlas { get; }

        private readonly ComputeShader _cs;
        private readonly int _kScore, _kGather, _kSelect, _kBuildViews;

        private GraphicsBuffer? _states;
        private int _stateCapacity;
        private bool _clearStates;
        private readonly List<(GraphicsBuffer Buffer, long DisposeAfterTick)> _retired = new();
        private readonly GraphicsBuffer _histogram;
        private readonly GraphicsBuffer _candidates;
        private readonly GraphicsBuffer _candidateCount;
        private readonly GraphicsBuffer _picks;
        private readonly GraphicsBuffer _faces;
        private readonly GraphicsBuffer _counts;
        private readonly GraphicsBuffer _shadowLights;
        private readonly GraphicsBuffer _views;

        // Counts, picks and the visible light count of each frame, read back for the stats
        private const int StatsPicksOffset = CountSlots * 4;
        private const int StatsVisibleOffset = StatsPicksOffset + MaxLights * 16;
        private const int StatsBytes = StatsVisibleOffset + 8; // lights in the list, lights in view
        private readonly GraphicsBuffer[] _statsReadback = new GraphicsBuffer[FrameCount];

        private ResourceStates _atlasState = ResourceStates.DepthWrite;
        private int _selectCount;
        private uint _recentLights = uint.MaxValue;
        private bool _selected;     // Select ran this frame: views and shadow lights are in NonPixelShaderResource
        private bool _rendered;
        private string? _cameraName;

        /// <summary>The atlas rendered this frame, or 0 when none was: the point light shader then skips shadows.</summary>
        public uint AtlasIndex => _rendered ? Atlas.BindlessIndex : 0u;
        public uint ViewBufferSrv => _views.SrvIndex;
        public uint LightBufferSrv => _shadowLights.SrvIndex;

        // Every renderer's instance, for the editor's stats (the scene view is not the only one rendering)
        private static readonly List<LocalShadows> _instances = new();

        private static readonly ProfilerMarker _marker = ProfilerMarker.Get("Point light shadows");

        public LocalShadows()
        {
            Atlas = new DepthTextureArray2D(PageSize, PageSize, PageCount);

            _cs = new ComputeShader("local_shadow_select.hlsl");
            _kScore = _cs.FindKernel("CSScore");
            _kGather = _cs.FindKernel("CSGather");
            _kSelect = _cs.FindKernel("CSSelect");
            _kBuildViews = _cs.FindKernel("CSBuildViews");

            _histogram = GraphicsBuffer.CreateStructured<uint>(ScoreBins, uav: true);
            _candidates = GraphicsBuffer.CreateStructured(MaxCandidates, 16, uav: true);
            _candidateCount = GraphicsBuffer.CreateRaw(1, uav: true, clearable: true);
            _picks = GraphicsBuffer.CreateStructured<Pick>(MaxLights, uav: true);
            _faces = GraphicsBuffer.CreateStructured(MaxViews, 16, uav: true);
            _counts = GraphicsBuffer.CreateStructured<uint>(CountSlots, uav: true);
            _shadowLights = GraphicsBuffer.CreateStructured<ShadowLight>(MaxLights, srv: true, uav: true);
            _views = GraphicsBuffer.CreateStructured<View>(MaxViews, srv: true, uav: true);
            for (int i = 0; i < FrameCount; i++)
                _statsReadback[i] = GraphicsBuffer.CreateReadback<byte>(StatsBytes);

            lock (_instances) _instances.Add(this);
        }

        /// <summary>
        /// Picks this frame's shadowed lights out of the visible list and builds their faces, all on the GPU.
        /// Writes each picked light's shadow index into the visible list for the point light shader.
        /// </summary>
        /// <param name="frameParamsSrv">StructuredBuffer&lt;PointLightFrame&gt; (budget, strength, fade step, camera).</param>
        /// <param name="visibleUav">The visible list (RWStructuredBuffer&lt;PointLightData&gt;).</param>
        /// <param name="visibleCount">Its count (RWByteAddressBuffer; read back for the stats too).</param>
        /// <param name="depthSrv">Linear depth (view-space Z) for the hidden-centre test, readable by compute.</param>
        /// <param name="sunShadowMap">The sun's cascades rendered this frame (DeferredRenderer.ShadowMapIndex), 0 = none.</param>
        /// <param name="sunCascades">Their CascadeData buffer (DirectionalLight.CurrentCascadeSrvIndex).</param>
        public void Select(ID3D12GraphicsCommandList list, Camera camera, uint frameParamsSrv, uint visibleUav,
            GraphicsBuffer visibleCount, uint depthSrv, uint sunShadowMap, uint sunCascades, int sunCascadeCount)
        {
            uint visibleCountUav = visibleCount.UavIndex;
            _cameraName = camera.Entity?.Name;
            EnsureStates(list);

            // Lights picked 2 frames ago (the newest readback the GPU has finished): Render skips the culling and the
            // draws while that stays 0, e.g. at a clear noon. The fade-in covers the 2 frames a new pick waits.
            _selectCount++;
            _recentLights = _selectCount > FrameCount
                ? ReadCount(_statsReadback[(Engine.FrameIndex + 1) % FrameCount], CountLights)
                : uint.MaxValue;

            PixMarker.Begin(list, "Point light shadow selection");

            _histogram.ClearUAV(list, new Int4(0, 0, 0, 0));
            _candidateCount.ClearUAV(list, new Int4(0, 0, 0, 0));
            UavBarrier(list);

            _cs.SetPushConstant("FrameParams", frameParamsSrv);
            _cs.SetPushConstant("VisibleUAV", visibleUav);
            _cs.SetPushConstant("VisibleCountUAV", visibleCountUav);
            _cs.SetPushConstant("StateUAV", _states!.UavIndex);
            _cs.SetPushConstant("StateCount", (uint)_stateCapacity);
            _cs.SetPushConstant("HistogramUAV", _histogram.UavIndex);
            _cs.SetPushConstant("CandidatesUAV", _candidates.UavIndex);
            _cs.SetPushConstant("CandidateCountUAV", _candidateCount.UavIndex);
            _cs.SetPushConstant("PicksUAV", _picks.UavIndex);
            _cs.SetPushConstant("FacesUAV", _faces.UavIndex);
            _cs.SetPushConstant("CountsUAV", _counts.UavIndex);
            _cs.SetPushConstant("ShadowLightsUAV", _shadowLights.UavIndex);
            _cs.SetPushConstant("ViewsUAV", _views.UavIndex);
            _cs.SetPushConstant("DepthTex", depthSrv);
            _cs.SetPushConstant("SunShadowMap", sunShadowMap);
            _cs.SetPushConstant("SunCascades", sunCascades);
            _cs.SetPushConstant("SunCascadeCount", sunShadowMap != 0 ? (uint)sunCascadeCount : 0u);

            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });

            uint visibleGroups = (uint)(PointLightBuffer.MaxVisible / 64);
            _cs.Dispatch(_kScore, list, visibleGroups);
            UavBarrier(list);
            _cs.Dispatch(_kGather, list, visibleGroups);
            UavBarrier(list);
            _cs.Dispatch(_kSelect, list, 1);
            UavBarrier(list);
            _cs.Dispatch(_kBuildViews, list, (uint)(MaxViews / 64));
            UavBarrier(list);

            // Views and shadow lights are read through SRVs by the culler, the shadow vertex shaders and the
            // point light shader until EndFrame
            list.Barriers(
                new ResourceBarrier(new ResourceTransitionBarrier(_views.Native, ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)),
                new ResourceBarrier(new ResourceTransitionBarrier(_shadowLights.Native, ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)));
            _selected = true;

            // What was picked, for the stats (read 2 frames later)
            var readback = _statsReadback[Engine.FrameIndex % FrameCount].Native;
            list.Barriers(
                new ResourceBarrier(new ResourceTransitionBarrier(_counts.Native, ResourceStates.UnorderedAccess, ResourceStates.CopySource)),
                new ResourceBarrier(new ResourceTransitionBarrier(_picks.Native, ResourceStates.UnorderedAccess, ResourceStates.CopySource)),
                new ResourceBarrier(new ResourceTransitionBarrier(visibleCount.Native, ResourceStates.UnorderedAccess, ResourceStates.CopySource)));
            list.CopyBufferRegion(readback, 0, _counts.Native, 0, CountSlots * 4);
            list.CopyBufferRegion(readback, StatsPicksOffset, _picks.Native, 0, MaxLights * 16);
            list.CopyBufferRegion(readback, StatsVisibleOffset, visibleCount.Native, 0, 8);
            list.Barriers(
                new ResourceBarrier(new ResourceTransitionBarrier(_counts.Native, ResourceStates.CopySource, ResourceStates.UnorderedAccess)),
                new ResourceBarrier(new ResourceTransitionBarrier(_picks.Native, ResourceStates.CopySource, ResourceStates.UnorderedAccess)),
                new ResourceBarrier(new ResourceTransitionBarrier(visibleCount.Native, ResourceStates.CopySource, ResourceStates.UnorderedAccess)));

            PixMarker.End(list);
        }

        // The state buffer follows the registry's capacity; a new one starts every light without a shadow
        private void EnsureStates(ID3D12GraphicsCommandList list)
        {
            for (int i = _retired.Count - 1; i >= 0; i--)
            {
                if (Engine.TickCount < _retired[i].DisposeAfterTick) continue;
                _retired[i].Buffer.Dispose();
                _retired.RemoveAt(i);
            }

            int capacity = PointLightBuffer.Instance?.Capacity ?? 0;
            if (_states == null || _stateCapacity < capacity)
            {
                // Frames still in flight may use the old one
                if (_states != null)
                    _retired.Add((_states, Engine.TickCount + FrameCount + 1));
                _stateCapacity = Math.Max(capacity, 1024);
                _states = GraphicsBuffer.CreateStructured<LightState>(_stateCapacity, uav: true);
                _clearStates = true;
            }
            if (_clearStates)
            {
                _states.ClearUAV(list, new Int4(0, 0, 0, 0));
                UavBarrier(list);
                _clearStates = false;
            }
        }

        /// <summary>
        /// Renders the picked faces into the atlas: one cull and one draw per opaque batch for all faces. The CPU does
        /// not know how many were picked: the culler reads the count, and all pages are cleared.
        /// Leaves render targets and viewport changed (the caller restores them).
        /// </summary>
        /// <param name="placeholderCbAddress">Any CBV address for root slot 2 (the culler binds it, its kernels do not read it).</param>
        public bool Render(ID3D12GraphicsCommandList list, ulong placeholderCbAddress)
        {
            _rendered = false;
            var culler = CommandBuffer.Culler;
            var batches = CommandBuffer.GetAllBatches(RenderPass.Opaque);
            if (!_selected || _recentLights == 0 || culler == null || batches == null)
                return false;

            using var _ = _marker.Auto();
            PixMarker.Begin(list, "Point light shadows");

            if (_atlasState != ResourceStates.DepthWrite)
                list.ResourceBarrierTransition(Atlas.Native, _atlasState, ResourceStates.DepthWrite);

            list.RSSetViewport(new Viewport(0, 0, PageSize, PageSize));
            list.RSSetScissorRect(new RectI(0, 0, PageSize, PageSize));
            for (int p = 0; p < PageCount; p++)
                list.ClearDepthStencilView(Atlas.SliceDsvHandles[p], ClearFlags.Depth, 1.0f, 0);

            PixMarker.Begin(list, "Culling");
            InstanceBatch.CullLocalShadowAll(list, batches, _views.SrvIndex, _shadowLights.SrvIndex, _counts.UavIndex,
                placeholderCbAddress, culler);
            PixMarker.End(list);

            list.OMSetRenderTargets(0, (CpuDescriptorHandle[]?)null, false, Atlas.FullArrayDsvHandle);

            PixMarker.Begin(list, "Draws");
            foreach (var batch in batches)
            {
                if (!batch.Material.HasPass(RenderPass.LocalShadow)) continue;
                batch.Material.SetPass(RenderPass.LocalShadow);
                batch.Material.Apply(list, Engine.Device, null);
                PixMarker.Begin(list, batch.ProfileName);
                batch.DrawShadowSinglePass(list, _views.SrvIndex);
                PixMarker.End(list);
            }
            PixMarker.End(list);

            // Read by the point light compute shader
            list.ResourceBarrierTransition(Atlas.Native, ResourceStates.DepthWrite, ResourceStates.AllShaderResource);
            _atlasState = ResourceStates.AllShaderResource;
            _rendered = true;

            PixMarker.End(list);
            return true;
        }

        /// <summary>A frame without point light shadows (daylight, none budgeted): the shader skips them.</summary>
        public void Skip()
        {
            _rendered = false;
            _selected = false;
        }

        /// <summary>After the point light shader: views and shadow lights go back to UAV for the next selection.</summary>
        public void EndFrame(ID3D12GraphicsCommandList list)
        {
            if (!_selected) return;
            list.Barriers(
                new ResourceBarrier(new ResourceTransitionBarrier(_views.Native, ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess)),
                new ResourceBarrier(new ResourceTransitionBarrier(_shadowLights.Native, ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess)));
            _selected = false;
        }

        private static unsafe uint ReadCount(GraphicsBuffer readback, int index) => ((uint*)readback.ReadPtr<byte>())[index];

        private static void UavBarrier(ID3D12GraphicsCommandList list) =>
            list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

        /// <summary>
        /// The registry slots of the lights this renderer ranked highest 2 frames ago (the readback the GPU has
        /// finished): its wanted picks in rank order, not those fading out. Returns how many were written.
        /// </summary>
        public unsafe int TopPicks(Span<uint> slots)
        {
            if (_selectCount <= FrameCount)
                return 0;
            var bytes = _statsReadback[(Engine.FrameIndex + 1) % FrameCount].ReadPtr<byte>();
            uint count = Math.Min(((uint*)bytes)[CountLights], (uint)MaxLights);
            var picks = (Pick*)(bytes + StatsPicksOffset);
            int written = 0;
            for (int i = 0; i < count && written < slots.Length; i++)
                if ((picks[i].Packed & (1u << 16)) != 0)
                    slots[written++] = picks[i].Slot;
            return written;
        }

        /// <summary>Each renderer's picks of 2 frames ago, for the editor's stats.</summary>
        public static object DescribeAll()
        {
            lock (_instances)
            {
                var result = new List<object>(_instances.Count);
                foreach (var instance in _instances)
                    result.Add(instance.Describe());
                return result;
            }
        }

        /// <summary>This renderer's picks of 2 frames ago (the readback the GPU has finished).</summary>
        public unsafe object Describe()
        {
            var bytes = _statsReadback[(Engine.FrameIndex + 1) % FrameCount].ReadPtr<byte>();
            var counts = (uint*)bytes;
            var picks = (Pick*)(bytes + StatsPicksOffset);
            uint visibleLights = *(uint*)(bytes + StatsVisibleOffset);
            uint lightsInView = *(uint*)(bytes + StatsVisibleOffset + 4);
            int lightCount = (int)Math.Min(counts[CountLights], MaxLights);

            var lights = new List<object>(lightCount);
            for (int s = 0; s < lightCount; s++)
            {
                var pick = picks[s];
                var light = PointLightBuffer.Instance?.LightAt((int)pick.Slot);
                int tier = (int)((pick.Packed >> 8) & 3);
                lights.Add(new
                {
                    entity = light?.Entity?.Id ?? 0,
                    name = light?.Entity?.Name,
                    tile = TierSize[Math.Min(tier, 2)],
                    faces = BitOperations.PopCount(pick.Packed & 0x3F),
                    fade = MathF.Round(pick.Fade, 2),
                    fadingOut = (pick.Packed & (1u << 16)) == 0,
                });
            }
            return new
            {
                camera = _cameraName,
                rendered = _rendered,
                // lights in view, and those in the visible list: past PointLightBuffer.MaxVisible the least important are left out
                lightsInView,
                visibleLights = Math.Min(visibleLights, (uint)PointLightBuffer.MaxVisible),
                visibleCap = PointLightBuffer.MaxVisible,
                lights = lightCount,
                faces = counts[CountViews],
                faces512 = counts[CountFacesTier0],
                faces256 = counts[CountFacesTier0 + 1],
                faces128 = counts[CountFacesTier0 + 2],
                slotsUsed = counts[CountSlotsUsed],
                slots = SlotCount,
                candidates = counts[CountCandidates],
                picked = lights,
            };
        }

        public void Dispose()
        {
            lock (_instances) _instances.Remove(this);
            Atlas.Dispose();
            _cs.Dispose();
            _states?.Dispose();
            foreach (var (buffer, _) in _retired) buffer.Dispose();
            _histogram.Dispose();
            _candidates.Dispose();
            _candidateCount.Dispose();
            _picks.Dispose();
            _faces.Dispose();
            _counts.Dispose();
            _shadowLights.Dispose();
            _views.Dispose();
            foreach (var b in _statsReadback) b.Dispose();
        }
    }
}
