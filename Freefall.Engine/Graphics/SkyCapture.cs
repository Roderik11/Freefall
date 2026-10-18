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
    /// Sky visibility past the meshes (ATMOSPHERE_PLAN stage 4, part 2): the scene's meshes around the camera drawn as
    /// depth maps seen from 16 directions spread over the sky, so a place sees the sky in a direction when nothing in
    /// that direction's map lies between it and the sky. Walls, roofs, tree crowns (with their gaps: the leaves are
    /// alpha-tested) and the cave kit all count; a room is dark but for what its lamps give. A top-down height capture
    /// cannot see walls: a closed room under it was lit like an open hall.
    ///
    /// The maps cover a square of <see cref="Size"/> metres round the camera, from <see cref="Below"/> under to
    /// <see cref="Above"/> over its height, its centre snapped to a <see cref="Step"/> metre grid. A direction is drawn
    /// again when the centre moves to another grid cell or <see cref="Invalidate"/> is called, one per frame (all of them
    /// in one frame cost 2-2.8 ms at Eldermoor) through the sun shadows' machinery: InstanceBatch.CullShadowAll with this
    /// direction as its first view (the other three cull nothing) and the Shadow pass with a small, clamped slope bias
    /// (Material.SkyCaptureRaster). Characters (skinned batches), the terrain and the grass are left out:
    /// sky_visibility.hlsli does the terrain. Shader side: sky_capture.hlsli; sky_visibility_cs.hlsl for the opaque
    /// surfaces, sky_common.fx SkyVisibility for the rest. Docs: physical_atmosphere.md §11.
    /// </summary>
    public static class SkyCapture
    {
        public const int Resolution = 1024;
        public const int ViewCount = 16;            // sky_capture.hlsli SKY_CAPTURE_VIEWS
        private const int Rings = 4;
        public const float Size = 256f;             // metres the captured square spans
        public const float Step = 32f;              // its centre snaps to this grid
        public const float Below = 48f, Above = 96f; // metres captured under and over the snapped camera height
        private const float HeightStep = 16f;
        private const float MinCasterTexels = 2f;   // smaller casters are left out
        private const int CullViews = 4;            // the cascade culling's view count
        private const int FrameCount = 3;

        /// <summary>One direction. Must match SkyCaptureView in sky_capture.hlsli (80 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct View
        {
            public Matrix4x4 VP;
            public Vector3 Direction;
            public float Valid;
        }

        private static readonly Vector3[] Directions = MakeDirections();
        private static readonly View[] _views = new View[ViewCount];

        private static DepthTextureArray2D? _depth;
        private static readonly GraphicsBuffer?[] _viewBuffers = new GraphicsBuffer?[FrameCount];
        private static readonly GraphicsBuffer?[] _cullBuffers = new GraphicsBuffer?[FrameCount];
        private static ResourceStates _depthState = ResourceStates.DepthWrite;
        private static readonly List<InstanceBatch> _batches = new();
        private static readonly ProfilerMarker _marker = ProfilerMarker.Get("Sky captures");

        private static Vector3 _centre = new(float.NaN);
        private static int _pending;                // directions still to draw round the current centre (bit per view)
        private static bool _invalid = true;
        private static int _renderedTick = -1;
        private static int _viewsDrawn;             // for the stats
        private static int _lastBatches;

        /// <summary>For engine_stats: whether on, the captured square's centre, the directions still due (bit mask),
        /// the directions drawn since start, the batches the last one drew, the tick it was drawn.</summary>
        public static object Describe() => new
        {
            enabled = Enabled,
            centre = float.IsNaN(_centre.X) ? null : new[] { _centre.X, _centre.Y, _centre.Z },
            pending = _pending,
            viewsDrawn = _viewsDrawn,
            lastBatches = _lastBatches,
            lastTick = _renderedTick,
        };

        /// <summary>Whether the captures are drawn and read (EngineSettings.SkyVisibility and SkyCaptures).</summary>
        public static bool Enabled => Engine.Settings.SkyVisibility && Engine.Settings.SkyCaptures;

        /// <summary>Draw every direction again (a scene was loaded, meshes were placed or moved).</summary>
        public static void Invalidate() => _invalid = true;

        // Cosine-weighted about the zenith: ring r at sin²(elevation) = (r + 0.5) / Rings (21°, 38°, 52°, 69°), so each
        // direction counts alike for a surface facing up; four azimuths per ring, each ring turned 22.5° on from the last.
        // Drawn in index order: one azimuth of every ring, then the next.
        private static Vector3[] MakeDirections()
        {
            var dirs = new Vector3[ViewCount];
            int perRing = ViewCount / Rings;
            for (int k = 0; k < perRing; k++)
                for (int r = 0; r < Rings; r++)
                {
                    float elevation = MathF.Asin(MathF.Sqrt((r + 0.5f) / Rings));
                    float azimuth = (k * 360f / perRing + r * 22.5f + 11.25f) * MathF.PI / 180f;
                    dirs[k * Rings + r] = new Vector3(
                        MathF.Cos(elevation) * MathF.Sin(azimuth), MathF.Sin(elevation), MathF.Cos(elevation) * MathF.Cos(azimuth));
                }
            return dirs;
        }

        private static void EnsureResources()
        {
            if (_depth != null) return;
            _depth = new DepthTextureArray2D(Resolution, Resolution, ViewCount);
            _depthState = ResourceStates.DepthWrite;
            for (int i = 0; i < FrameCount; i++)
            {
                _viewBuffers[i] = GraphicsBuffer.CreateUpload<View>(ViewCount, mapped: true);
                _cullBuffers[i] = GraphicsBuffer.CreateUpload<GPUCuller.CascadeData>(CullViews, mapped: true);
            }
            for (int v = 0; v < ViewCount; v++)
                _views[v] = new View { Direction = Directions[v] };
        }

        /// <summary>
        /// For the atmosphere's parameters (once per frame): this frame's view buffer (written now and again when a
        /// direction is drawn this frame), the depth maps and the direction count; zeros when off.
        /// </summary>
        public static (uint Views, uint Depth, uint Count) PrepareFrame(int slot)
        {
            if (!Enabled)
            {
                _invalid = true;    // the scene may change meanwhile
                return Current = (0, 0, 0);
            }
            EnsureResources();
            WriteViews(slot);
            return Current = (_viewBuffers[slot]!.SrvIndex, _depth!.BindlessIndex, ViewCount);
        }

        /// <summary>What <see cref="PrepareFrame"/> handed on this frame (zeros when off): the views are this frame's once
        /// <see cref="Render"/> has run (PointLightBuffer.UpdateLamps reads them after it).</summary>
        public static (uint Views, uint Depth, uint Count) Current { get; private set; }

        private static unsafe void WriteViews(int slot)
        {
            var dst = _viewBuffers[slot]!.WritePtr<View>();
            for (int v = 0; v < ViewCount; v++)
                dst[v] = _views[v];
        }

        /// <summary>
        /// Draws the next direction that is due, once per frame (the first view drawn this frame). Call after the shadow
        /// passes: it reuses the batches' shadow culling buffers. Leaves render targets and viewport changed when it
        /// returns true (the caller restores them).
        /// </summary>
        public static bool Render(ID3D12GraphicsCommandList list, Camera camera)
        {
            if (!Enabled || _renderedTick == Engine.TickCount || _depth == null)
                return false;
            var culler = CommandBuffer.Culler;
            var all = CommandBuffer.GetAllBatches(RenderPass.Opaque);
            if (culler == null || all == null)
                return false;

            var eye = (Camera.Main ?? camera).Position;
            var centre = new Vector3(MathF.Round(eye.X / Step) * Step, MathF.Round(eye.Y / HeightStep) * HeightStep,
                MathF.Round(eye.Z / Step) * Step);
            if (_invalid || centre != _centre)
            {
                _centre = centre;
                _pending = (1 << ViewCount) - 1;
                _invalid = false;
            }
            if (_pending == 0)
                return false;
            _renderedTick = Engine.TickCount;

            int view = BitOperations.TrailingZeroCount(_pending);
            _pending &= ~(1 << view);

            _batches.Clear();
            foreach (var batch in all)
                if (batch.Material.HasPass(RenderPass.Shadow) && batch.Key.Effect?.Name != "gbuffer_skinned")
                    _batches.Add(batch);

            using var _ = _marker.Auto();
            PixMarker.Begin(list, "Sky captures");

            // The culling's four views: this direction, then three that hold nothing (every plane rejects)
            int slot = Engine.FrameIndex % FrameCount;
            var vp = ViewProjection(Directions[view], _centre);
            unsafe
            {
                var cull = _cullBuffers[slot]!.WritePtr<GPUCuller.CascadeData>();
                var data = new GPUCuller.CascadeData();
                data.SetPlanes(new Frustum(vp).GetPlanesAsVector4());
                data.VP = vp;
                data.PrevVP = vp;
                cull[0] = data;
                var none = new Vector4(0, 0, 0, float.MaxValue);
                for (int k = 1; k < CullViews; k++)
                    cull[k] = new GPUCuller.CascadeData { Plane0 = none, Plane1 = none, Plane2 = none, Plane3 = none, Plane4 = none, Plane5 = none };
            }
            _views[view] = new View { VP = vp, Direction = Directions[view], Valid = 1 };
            WriteViews(slot);

            if (_depthState != ResourceStates.DepthWrite)
                list.ResourceBarrierTransition(_depth.Native, _depthState, ResourceStates.DepthWrite);
            var dsv = _depth.SliceDsvHandles[view];
            list.RSSetViewport(new Viewport(0, 0, Resolution, Resolution));
            list.RSSetScissorRect(new RectI(0, 0, Resolution, Resolution));
            list.ClearDepthStencilView(dsv, ClearFlags.Depth, 1.0f, 0);

            var cullBuffer = _cullBuffers[slot]!;
            PixMarker.Begin(list, "Culling");
            InstanceBatch.CullShadowAll(list, _batches, cullBuffer.SrvIndex, cullBuffer.Native.GPUVirtualAddress,
                shadowHiZSrv: 0, culler, minCasterClip: -MinCasterTexels * 2f / Resolution, receiversUav: 0);
            PixMarker.End(list);

            // A single-slice DSV: the vertex shader routes the view to array index 0
            list.OMSetRenderTargets(0, (CpuDescriptorHandle[]?)null, false, dsv);
            PixMarker.Begin(list, "Draws");
            Material.SkyCaptureRaster = true;
            try
            {
                foreach (var batch in _batches)
                {
                    batch.Material.SetPass(RenderPass.Shadow);
                    batch.Material.Apply(list, Engine.Device, null);
                    PixMarker.Begin(list, batch.ProfileName);
                    batch.DrawShadowSinglePass(list, cullBuffer.SrvIndex);
                    PixMarker.End(list);
                }
            }
            finally
            {
                Material.SkyCaptureRaster = false;
            }
            PixMarker.End(list);

            // Read by the sky visibility pass (compute), the forward passes and the particles' vertex shaders
            list.ResourceBarrierTransition(_depth.Native, ResourceStates.DepthWrite, ResourceStates.AllShaderResource);
            _depthState = ResourceStates.AllShaderResource;
            PixMarker.End(list);
            _viewsDrawn++;
            _lastBatches = _batches.Count;
            return true;
        }

        // Absolute world -> clip of the direction toward the sky `dir`, orthographic, over the captured box round
        // `centre`. Its texel grid is snapped in light space, so the same place lands on the same texels whatever the
        // centre (no shimmer when the square moves).
        private static Matrix4x4 ViewProjection(Vector3 dir, Vector3 centre)
        {
            var rotation = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, -dir, Vector3.UnitY);
            float half = Size * 0.5f;
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            for (int c = 0; c < 8; c++)
            {
                var corner = new Vector3(
                    centre.X + ((c & 1) != 0 ? half : -half),
                    centre.Y + ((c & 2) != 0 ? Above : -Below),
                    centre.Z + ((c & 4) != 0 ? half : -half));
                var ls = Vector3.Transform(corner, rotation);
                min = Vector3.Min(min, ls);
                max = Vector3.Max(max, ls);
            }
            // Two texels to spare for the snap
            float texelX = (max.X - min.X) / (Resolution - 2), texelY = (max.Y - min.Y) / (Resolution - 2);
            float left = MathF.Floor(min.X / texelX) * texelX, bottom = MathF.Floor(min.Y / texelY) * texelY;
            var projection = Matrix4x4.CreateOrthographicOffCenterLeftHanded(
                left, left + Resolution * texelX, bottom, bottom + Resolution * texelY, min.Z - 2f, max.Z + 2f);
            return rotation * projection;
        }
    }
}
