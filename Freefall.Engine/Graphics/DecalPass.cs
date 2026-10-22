using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// Decals and the weather on the opaque G-buffer, per renderer (DECALS_PLAN.md Parts A and D): right after the opaque
    /// pass, the registry (DecalBuffer) is culled and sorted for this view (decal_cull.hlsl), binned into 16 px tiles, and
    /// one compute pass reads each pixel's G-buffer values, applies the decals of its tile in order and then the weather
    /// (wet surfaces, snow: Graphics.Weather), and writes them back (decals_cs.hlsl). Needs typed unordered-access loads of
    /// the G-buffer's formats; without them the pass is skipped (logged once).
    /// </summary>
    public sealed class DecalPass : IDisposable
    {
        private const int FrameCount = 3;
        private const int Tile = 16;                                    // decals.hlsli DECAL_TILE
        private const int MaskWords = DecalBuffer.MaxVisible / 32;      // DECAL_MASK_WORDS
        private const int ImportanceBins = 64;                          // decal_cull.hlsl IMPORTANCE_BINS

        /// <summary>The view and its weather. Must match DecalFrame in decals.hlsli (240 bytes).</summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct Frame
        {
            public Vector4 Plane0, Plane1, Plane2, Plane3, Plane4;
            public Vector3 CameraPosition; public float PixelsPerMetre;
            public Vector3 CameraForward; public uint TilesX;
            public Matrix4x4 ShelterVP;
            public uint ShelterDepth, ShelterSlice, TerrainHeightmap; public float TerrainMaxHeight;
            public Vector3 TerrainOrigin; public float Wetness;
            public Vector2 TerrainSize; public float PuddleFill, SnowCover;
            public float Rain; public uint TilesY, WeatherOn, _pad;
        }

        private static ComputeShader? _cullCS, _applyCS;
        private static int _kClear, _kCount, _kCull, _kSort, _kTile, _kApply;
        private static bool? _supported;

        private readonly GraphicsBuffer?[] _frames = new GraphicsBuffer?[FrameCount];
        private GraphicsBuffer? _histogram, _count, _visible, _sorted, _masks;
        private int _tilesX, _tilesY;
        private readonly ProfilerMarker _marker = ProfilerMarker.Get("Decals");

        /// <summary>Decals and weather applied this frame (for the stats): whether the pass ran, the slots walked.</summary>
        public bool RanThisFrame { get; private set; }

        /// <summary>
        /// Whether the GPU loads every G-buffer format through unordered access (RGBA16F and R32F always; RGBA16 SNorm,
        /// RGBA8 UNorm and RG16F need TypedUAVLoadAdditionalFormats and the format's own support).
        /// </summary>
        public static bool Supported
        {
            get
            {
                if (_supported.HasValue) return _supported.Value;
                var device = Engine.Device.NativeDevice;
                var options = new FeatureDataD3D12Options();
                bool additional = device.CheckFeatureSupport(Vortice.Direct3D12.Feature.Options, ref options)
                                  && options.TypedUAVLoadAdditionalFormats;
                bool ok = additional;
                foreach (var format in new[] { Format.R16G16B16A16_SNorm, Format.R8G8B8A8_UNorm, Format.R16G16_Float })
                {
                    var support = new FeatureDataFormatSupport { Format = format };
                    ok &= device.CheckFeatureSupport(Vortice.Direct3D12.Feature.FormatSupport, ref support)
                          && (support.Support2 & FormatSupport2.UnorderedAccessViewTypedLoad) != 0;
                }
                if (!ok)
                    Debug.LogWarning("Decals", "The GPU cannot load the G-buffer's formats through unordered access: no decals, no wet or snowy surfaces");
                _supported = ok;
                return ok;
            }
        }

        private void EnsureResources(int width, int height)
        {
            if (_cullCS == null)
            {
                _cullCS = new ComputeShader("decal_cull.hlsl");
                _kClear = _cullCS.FindKernel("CSClearDecals");
                _kCount = _cullCS.FindKernel("CSCountDecals");
                _kCull = _cullCS.FindKernel("CSCullDecals");
                _kSort = _cullCS.FindKernel("CSSortDecals");
                _applyCS = new ComputeShader("decals_cs.hlsl");
                _kTile = _applyCS.FindKernel("CSTileDecals");
                _kApply = _applyCS.FindKernel("CSDecals");
            }
            if (_histogram == null)
            {
                for (int i = 0; i < FrameCount; i++)
                    _frames[i] = GraphicsBuffer.CreateUpload<Frame>(1, mapped: true);
                _histogram = GraphicsBuffer.CreateStructured<uint>(ImportanceBins, uav: true);
                _count = GraphicsBuffer.CreateRaw(4, uav: true);
                _visible = GraphicsBuffer.CreateStructured<Vector2>(DecalBuffer.MaxVisible, uav: true);    // uint2
                _sorted = GraphicsBuffer.CreateStructured<Vector2>(DecalBuffer.MaxVisible, srv: true, uav: true);
            }
            int tilesX = (width + Tile - 1) / Tile, tilesY = (height + Tile - 1) / Tile;
            if (_masks == null || tilesX != _tilesX || tilesY != _tilesY)
            {
                Engine.Device.DeferDispose(_masks);
                _masks = GraphicsBuffer.CreateRaw(tilesX * tilesY * MaskWords, uav: true);
                _tilesX = tilesX;
                _tilesY = tilesY;
            }
        }

        /// <summary>
        /// Culls, sorts and bins the decals in view and applies them and the weather to the G-buffer. Call right after the
        /// opaque pass with the G-buffer bound as render targets (RenderTarget state, the hardware depth in DepthWrite);
        /// they are left in those states. True when it ran: the caller binds the G-buffer targets and viewport again.
        /// </summary>
        public bool Render(ID3D12GraphicsCommandList list, Camera camera, DeferredRenderer renderer,
            Action<ID3D12GraphicsCommandList> bindSceneConstants)
        {
            RanThisFrame = false;
            var registry = DecalBuffer.Instance;
            int slots = registry?.SlotCount ?? 0;
            bool weather = Weather.SurfaceWetness > 0.001f || Weather.SnowCover > 0.001f;
            if ((slots == 0 && !weather) || !Supported)
                return false;
            if (renderer.Albedo.UavIndex == 0)
                return false;   // targets made without unordered access

            using var _ = _marker.Auto();
            var desc = renderer.Albedo.Native.Description;
            int width = (int)desc.Width, height = (int)desc.Height;
            EnsureResources(width, height);
            registry?.Upload();
            int slot = Engine.FrameIndex % FrameCount;
            WriteFrame(slot, camera, height, weather);

            var device = Engine.Device;
            list.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            list.SetComputeRootSignature(device.GlobalRootSignature);
            bindSceneConstants(list);

            // Cull and sort for this view
            PixMarker.Begin(list, "Decal cull");
            var cull = _cullCS!;
            cull.SetPushConstant("Records", registry?.SrvIndex ?? 0u);
            cull.SetPushConstant("RecordCount", (uint)slots);
            cull.SetPushConstant("Transforms", TransformBuffer.Instance.SrvIndex);
            cull.SetPushConstant("Frame", _frames[slot]!.SrvIndex);
            cull.SetPushConstant("VisibleUAV", _visible!.UavIndex);
            cull.SetPushConstant("CountUAV", _count!.UavIndex);
            cull.SetPushConstant("HistogramUAV", _histogram!.UavIndex);
            cull.SetPushConstant("SortedUAV", _sorted!.UavIndex);
            _histogram.Transition(list, ResourceStates.UnorderedAccess);
            _count.Transition(list, ResourceStates.UnorderedAccess);
            _visible.Transition(list, ResourceStates.UnorderedAccess);
            _sorted.Transition(list, ResourceStates.UnorderedAccess);
            cull.Dispatch(_kClear, list, 1);
            UavBarrier(list, _histogram, _count);
            if (slots > 0)
            {
                uint groups = (uint)((slots + 63) / 64);
                cull.Dispatch(_kCount, list, groups);
                UavBarrier(list, _histogram, _count);
                cull.Dispatch(_kCull, list, groups);
                UavBarrier(list, _visible, _count);
                cull.Dispatch(_kSort, list, 1);
                UavBarrier(list, _sorted);
            }
            _sorted.Transition(list, ResourceStates.NonPixelShaderResource);
            PixMarker.End(list);

            // The G-buffer: read and written by the pass; the hardware depth and the surfaces' classes only read
            var shaderRead = ResourceStates.NonPixelShaderResource;
            list.ResourceBarrierTransition(renderer.Albedo.Native, ResourceStates.RenderTarget, ResourceStates.UnorderedAccess);
            list.ResourceBarrierTransition(renderer.Normals.Native, ResourceStates.RenderTarget, ResourceStates.UnorderedAccess);
            list.ResourceBarrierTransition(renderer.Data.Native, ResourceStates.RenderTarget, ResourceStates.UnorderedAccess);
            list.ResourceBarrierTransition(renderer.DepthGBuffer.Native, ResourceStates.RenderTarget, ResourceStates.UnorderedAccess);
            list.ResourceBarrierTransition(renderer.HeightBuffer.Native, ResourceStates.RenderTarget, ResourceStates.UnorderedAccess);
            list.ResourceBarrierTransition(renderer.EntityIdBuffer.Native, ResourceStates.RenderTarget, shaderRead);
            list.ResourceBarrierTransition(renderer.Depth.Native, ResourceStates.DepthWrite, shaderRead);

            PixMarker.Begin(list, "Decals");
            var apply = _applyCS!;
            apply.SetPushConstant("Records", registry?.SrvIndex ?? 0u);
            apply.SetPushConstant("Transforms", TransformBuffer.Instance.SrvIndex);
            apply.SetPushConstant("Frame", _frames[slot]!.SrvIndex);
            apply.SetPushConstant("Sorted", _sorted.SrvIndex);
            apply.SetPushConstant("CountUAV", _count.UavIndex);
            apply.SetPushConstant("MasksUAV", _masks!.UavIndex);
            apply.SetPushConstant("Materials", Material.MaterialsBufferIndex);
            apply.SetPushConstant("DepthTex", renderer.Depth.BindlessIndex);
            apply.SetPushConstant("EntityIdTex", renderer.EntityIdBuffer.BindlessIndex);
            apply.SetPushConstant("AlbedoUAV", renderer.Albedo.UavIndex);
            apply.SetPushConstant("NormalUAV", renderer.Normals.UavIndex);
            apply.SetPushConstant("DataUAV", renderer.Data.UavIndex);
            apply.SetPushConstant("LinearDepthUAV", renderer.DepthGBuffer.UavIndex);
            apply.SetPushConstant("HeightUAV", renderer.HeightBuffer.UavIndex);
            _masks.Transition(list, ResourceStates.UnorderedAccess);
            if (slots > 0)
            {
                apply.Dispatch(_kTile, list, (uint)_tilesX, (uint)_tilesY);
                UavBarrier(list, _masks);
            }
            apply.Dispatch(_kApply, list, (uint)_tilesX, (uint)_tilesY);
            PixMarker.End(list);

            list.ResourceBarrierTransition(renderer.Albedo.Native, ResourceStates.UnorderedAccess, ResourceStates.RenderTarget);
            list.ResourceBarrierTransition(renderer.Normals.Native, ResourceStates.UnorderedAccess, ResourceStates.RenderTarget);
            list.ResourceBarrierTransition(renderer.Data.Native, ResourceStates.UnorderedAccess, ResourceStates.RenderTarget);
            list.ResourceBarrierTransition(renderer.DepthGBuffer.Native, ResourceStates.UnorderedAccess, ResourceStates.RenderTarget);
            list.ResourceBarrierTransition(renderer.HeightBuffer.Native, ResourceStates.UnorderedAccess, ResourceStates.RenderTarget);
            list.ResourceBarrierTransition(renderer.EntityIdBuffer.Native, shaderRead, ResourceStates.RenderTarget);
            list.ResourceBarrierTransition(renderer.Depth.Native, shaderRead, ResourceStates.DepthWrite);
            RanThisFrame = true;
            return true;
        }

        private static void UavBarrier(ID3D12GraphicsCommandList list, params GraphicsBuffer[] buffers)
        {
            foreach (var b in buffers)
                list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(b.Native)));
        }

        private unsafe void WriteFrame(int slot, Camera camera, int height, bool weather)
        {
            // As the point lights' frame: the view without translation, its planes camera-relative with normals in
            var cvp = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, camera.Forward, camera.Up) * camera.Projection;
            var c0 = new Vector4(cvp.M11, cvp.M21, cvp.M31, cvp.M41);
            var c1 = new Vector4(cvp.M12, cvp.M22, cvp.M32, cvp.M42);
            var c3 = new Vector4(cvp.M14, cvp.M24, cvp.M34, cvp.M44);
            var shelter = SkyCapture.Shelter;
            var terrain = Atmosphere.TerrainCover;
            *_frames[slot]!.WritePtr<Frame>() = new Frame
            {
                Plane0 = NormalisePlane(c3 + c0),
                Plane1 = NormalisePlane(c3 - c0),
                Plane2 = NormalisePlane(c3 + c1),
                Plane3 = NormalisePlane(c3 - c1),
                Plane4 = new Vector4(Vector3.Normalize(camera.Forward), -camera.NearPlane),
                CameraPosition = camera.Position,
                PixelsPerMetre = 0.5f * height * camera.Projection.M22,
                CameraForward = Vector3.Normalize(camera.Forward),
                TilesX = (uint)_tilesX,
                ShelterVP = shelter.VP,
                ShelterDepth = shelter.Depth,
                ShelterSlice = shelter.Slice,
                TerrainHeightmap = terrain.Heightmap,
                TerrainMaxHeight = terrain.MaxHeight,
                TerrainOrigin = terrain.Origin,
                Wetness = Weather.SurfaceWetness,
                TerrainSize = terrain.Size,
                PuddleFill = Weather.PuddleFill,
                SnowCover = Weather.SnowCover,
                Rain = Weather.Rain,
                TilesY = (uint)_tilesY,
                WeatherOn = weather ? 1u : 0u,
            };
        }

        private static Vector4 NormalisePlane(Vector4 plane) =>
            plane / new Vector3(plane.X, plane.Y, plane.Z).Length();

        public void Dispose()
        {
            foreach (var f in _frames) f?.Dispose();
            _histogram?.Dispose();
            _count?.Dispose();
            _visible?.Dispose();
            _sorted?.Dispose();
            _masks?.Dispose();
        }
    }
}
