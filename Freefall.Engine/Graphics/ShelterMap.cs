using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// What keeps the falling rain and snow off the ground as far as it falls (WEATHER_PLAN stage 2): the meshes round the
    /// camera drawn straight down, in three nested levels like shadow cascades, each texel holding the height of the
    /// highest cover over it and the share of the texel it covers. The decal pass reads it for how wet and how white a
    /// surface gets (weather_cover.hlsli WeatherOpenness). One top-down capture of 256 m (SkyCapture's slice) ended in a
    /// square round the camera, with bare ground under the trees inside it and snow under them past it (user, 2026-10-23).
    ///
    /// Levels of <see cref="Sizes"/> metres at <see cref="Resolution"/>² texels (25 cm, 1 m, 4 m), each drawn at twice that
    /// (the share of a texel from 2 × 2 samples, so a far crown covers part of a texel instead of all or nothing) and
    /// drawn again only when the camera's snapped cell changes (<see cref="Steps"/>: 32, 128, 512 m) or the scene does
    /// (<see cref="Invalidate"/>), one level per frame, the finest first. Until then a level keeps the square it was
    /// drawn for (the shader reads that one), so moving on leaves its far side uncovered for a frame, never wrong.
    ///
    /// Drawn through the sun shadows' machinery as SkyCapture: InstanceBatch.CullShadowAll with the level as its one view,
    /// the Shadow pass with Material.SkyCaptureRaster, characters (skinned) left out; casters under two drawn texels are
    /// culled, so the 4 km level holds trees and houses. Twice per level: the solid meshes, then the vegetation (the
    /// foliage and cross-mesh effects), whose crowns let <see cref="CrownLeak"/> of the fall through. shelter_map_cs.hlsl
    /// reduces each into the level's map and drops cover lower than <see cref="MinAboveGround"/> over the terrain (grass,
    /// shrubs, crates: the snow buries them). The terrain itself covers by its height map (caves), not through this map.
    /// The rain particles and their splashes still read SkyCapture's slice. Docs: atmospheric_effects/weather.md §2.
    /// </summary>
    public static class ShelterMap
    {
        public const int Levels = 3;                 // weather_cover.hlsli SHELTER_LEVELS
        public const int Resolution = 1024;          // a level's texels a side (SHELTER_RESOLUTION)
        private const int DrawResolution = 2 * Resolution;
        public static readonly float[] Sizes = { 256f, 1024f, 4096f };
        public static readonly float[] Steps = { 32f, 128f, 512f };
        private const float MinCasterTexels = 2f;    // drawn texels; smaller casters are left out
        public const float MinAboveGround = 1f;      // m over the terrain cover must reach
        public const float CrownLeak = 0.5f;         // share of the fall that comes through the vegetation's crowns
        private const float BelowGround = 20f, AboveGround = 120f;  // m drawn under the terrain's lowest and over its highest
        private const float NoTerrainBelow = 100f, NoTerrainAbove = 200f;   // ... and round the camera without a terrain

        /// <summary>A level as the decal pass reads it (weather_cover.hlsli ShelterLevel, 16 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct LevelData
        {
            public float Left;      // world x at the square's left edge
            public float Top;       // world z at its top edge (the map's v 0)
            public float Size;      // metres it spans; 0 = not drawn
            public uint Map;        // Texture2D<uint>: (height << 16) | share
        }

        private static readonly LevelData[] _drawn = new LevelData[Levels];
        private static readonly Vector2[] _centres = new Vector2[Levels];
        private static readonly RenderTexture2D?[] _maps = new RenderTexture2D?[Levels];
        private static readonly bool[] _mapReadable = new bool[Levels];
        private static DepthTextureArray2D? _depth;
        private static ComputeShader? _reduce;
        private static int _kReduce;
        private static readonly GraphicsBuffer?[] _cullBuffers = new GraphicsBuffer?[FrameCount];
        private const int FrameCount = 3;
        private static int _pending;                 // levels still to draw (bit per level)
        private static bool _invalid = true;
        private static float _minY = float.NaN, _maxY = float.NaN;
        private static int _renderedTick = -1;
        private static int _levelsDrawn;
        private static readonly List<InstanceBatch> _solid = new(), _vegetation = new();
        private static readonly ProfilerMarker _marker = ProfilerMarker.Get("Cover map");

        // The finest level on the CPU (OpenAt): copied into _readback after it is drawn, taken into _cpuMap when that
        // frame's slot comes round again (the GPU is done with it then), with the square and heights it was drawn for
        private static GraphicsBuffer? _readback;
        private static bool _copyWanted;
        private static int _copyTick = -1, _copySlot;
        private static LevelData _copyLevel;
        private static float _copyMinY, _copyMaxY;
        private static uint[]? _cpuMap;
        private static LevelData _cpuLevel;
        private static float _cpuMinY, _cpuMaxY;
        private const int OpenAround = 2;            // OpenAt averages (2 × this + 1)² texels: 1.25 m
        private const float CoverAbove = 0.25f;      // m over the place cover must be (not the bridge it stands on)

        /// <summary>Whether it is drawn and read (with the sky captures: EngineSettings.SkyVisibility and SkyCaptures).</summary>
        public static bool Enabled => SkyCapture.Enabled;

        /// <summary>Draw every level again (a scene was loaded, meshes were placed or moved).</summary>
        public static void Invalidate()
        {
            _invalid = true;
            _cpuLevel = _copyLevel = default;   // a copy on its way is of the old scene
        }

        /// <summary>
        /// How much of the falling rain and snow reaches a place (0..1), as the decal pass has it: 1 in the open, 0 under a
        /// roof, 1 − <see cref="CrownLeak"/> under a full crown. For game code that has to agree with the picture: the ground's
        /// sounds in snow and rain (a ray up meets roofs but not crowns, which have no collider). The share of the texels
        /// round the place (1.25 m) covered from over it, in the finest level (256 m round the camera) as last drawn, read
        /// back each time it is drawn and a few frames late. 1 outside it, while the map is off and until the first read
        /// back. The terrain's own cover (a cave) is not in the map: the decal pass tests the height map for it.
        /// </summary>
        public static float OpenAt(Vector3 place)
        {
            var map = _cpuMap;
            var level = _cpuLevel;
            if (map == null || level.Size <= 0f || !Enabled)
                return 1f;
            float texel = level.Size / Resolution;
            int cx = (int)MathF.Floor((place.X - level.Left) / texel);
            int cy = (int)MathF.Floor((level.Top - place.Z) / texel);
            if (cx < OpenAround || cy < OpenAround || cx >= Resolution - OpenAround || cy >= Resolution - OpenAround)
                return 1f;

            // Cover counts over the place only: the packed height has to be above this
            float over = (place.Y + CoverAbove - _cpuMinY) / MathF.Max(_cpuMaxY - _cpuMinY, 1e-3f);
            uint above = over <= 0f ? 0u : over >= 1f ? 65535u : (uint)(over * 65535f);
            uint covered = 0;
            for (int y = cy - OpenAround; y <= cy + OpenAround; y++)
                for (int x = cx - OpenAround; x <= cx + OpenAround; x++)
                {
                    uint v = map[y * Resolution + x];
                    if ((v >> 16) > above) covered += v & 0xFFFFu;
                }
            const int Texels = (2 * OpenAround + 1) * (2 * OpenAround + 1);
            return 1f - covered / (65535f * Texels);
        }

        /// <summary>
        /// Where the levels are centred while <paramref name="camera"/> draws: the main camera's place (the game's view
        /// also in the editor's scene view). The shader blends the levels by the distance from here, not by their
        /// snapped squares (weather_cover.hlsli ShelterBlend).
        /// </summary>
        public static Vector3 EyeFor(Camera camera) => (Camera.Main ?? camera).Position;

        /// <summary>The levels as drawn (Size 0 where not yet or when off) and the heights their maps span.</summary>
        public static (LevelData L0, LevelData L1, LevelData L2, float MinY, float RangeY) Current =>
            Enabled ? (_drawn[0], _drawn[1], _drawn[2], _minY, _maxY - _minY) : (default, default, default, 0f, 1f);

        /// <summary>For engine_stats: whether on, each level's square as drawn (x, z of its centre, size), the levels
        /// still due, how many have been drawn.</summary>
        public static object Describe() => new
        {
            enabled = Enabled,
            levels = Array.ConvertAll(_drawn, l => l.Size > 0f ? new[] { l.Left + l.Size * 0.5f, l.Top - l.Size * 0.5f, l.Size } : null),
            heights = float.IsNaN(_minY) ? null : new[] { _minY, _maxY },
            pending = _pending,
            levelsDrawn = _levelsDrawn,
        };

        private static void EnsureResources()
        {
            if (_depth != null) return;
            _depth = new DepthTextureArray2D(DrawResolution, DrawResolution, 1);
            for (int i = 0; i < Levels; i++)
                _maps[i] = new RenderTexture2D(Engine.Device, Resolution, Resolution, Format.R32_UInt, false, true);
            for (int i = 0; i < FrameCount; i++)
                _cullBuffers[i] = GraphicsBuffer.CreateUpload<GPUCuller.CascadeData>(1, mapped: true);
            _reduce = new ComputeShader("shelter_map_cs.hlsl");
            _kReduce = _reduce.FindKernel("CSReduce");
        }

        /// <summary>
        /// Draws the next level that is due, at most one per frame. Call after the shadow passes (it reuses the batches'
        /// shadow culling buffers). Leaves render targets and viewport changed when it returns true (the caller restores
        /// them).
        /// </summary>
        public static bool Render(ID3D12GraphicsCommandList list, Camera camera)
        {
            if (!Enabled)
            {
                _invalid = true;    // the scene may change meanwhile
                return false;
            }
            if (_renderedTick == Engine.TickCount)
                return false;
            ReadBack(list);
            var culler = CommandBuffer.Culler;
            var all = CommandBuffer.GetAllBatches(RenderPass.Opaque);
            if (culler == null || all == null)
                return false;

            var eye = EyeFor(camera);
            var terrain = Atmosphere.TerrainCover;
            float minY = terrain.Heightmap != 0 ? terrain.Origin.Y - BelowGround : MathF.Floor(eye.Y / 64f) * 64f - NoTerrainBelow;
            float maxY = terrain.Heightmap != 0 ? terrain.Origin.Y + terrain.MaxHeight + AboveGround : minY + NoTerrainBelow + NoTerrainAbove;
            if (_invalid || minY != _minY || maxY != _maxY)
            {
                _minY = minY;
                _maxY = maxY;
                _pending = (1 << Levels) - 1;
                _invalid = false;
            }
            for (int l = 0; l < Levels; l++)
            {
                var centre = new Vector2(MathF.Round(eye.X / Steps[l]) * Steps[l], MathF.Round(eye.Z / Steps[l]) * Steps[l]);
                if (centre != _centres[l] || _drawn[l].Size == 0f)
                {
                    _centres[l] = centre;
                    _pending |= 1 << l;
                }
            }
            if (_pending == 0)
                return false;
            EnsureResources();
            _renderedTick = Engine.TickCount;

            int level = BitOperations.TrailingZeroCount(_pending);
            _pending &= ~(1 << level);

            _solid.Clear();
            _vegetation.Clear();
            foreach (var batch in all)
            {
                if (!batch.Material.HasPass(RenderPass.Shadow)) continue;
                string? effect = batch.Key.Effect?.Name;
                if (effect == "gbuffer_skinned") continue;
                (effect == "gbuffer_foliage" || effect == "gbuffer_crossmesh" ? _vegetation : _solid).Add(batch);
            }

            using var _ = _marker.Auto();
            PixMarker.Begin(list, "Cover map");

            // The level's square, its corner snapped to its texels in world space (the same place, the same texels)
            float size = Sizes[level], texel = size / Resolution;
            float left = MathF.Floor((_centres[level].X - size * 0.5f) / texel) * texel;
            float bottom = MathF.Floor((_centres[level].Y - size * 0.5f) / texel) * texel;
            // Looking down: the view's x is the world's x, its y the world's z, its depth the height from the top down
            var view = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, -Vector3.UnitY, Vector3.UnitZ);
            var vp = view * Matrix4x4.CreateOrthographicOffCenterLeftHanded(left, left + size, bottom, bottom + size, -_maxY, -_minY);

            int slot = Engine.FrameIndex % FrameCount;
            var cullBuffer = _cullBuffers[slot]!;
            unsafe
            {
                var data = new GPUCuller.CascadeData();
                data.SetPlanes(new Frustum(vp).GetPlanesAsVector4());
                data.VP = vp;
                data.PrevVP = vp;
                *cullBuffer.WritePtr<GPUCuller.CascadeData>() = data;
            }

            var map = _maps[level]!;
            list.ResourceBarrierTransition(map.Native, _mapReadable[level] ? ResourceStates.NonPixelShaderResource : ResourceStates.Common,
                ResourceStates.UnorderedAccess);
            _mapReadable[level] = false;

            Draw(list, _solid, cullBuffer, culler);
            Reduce(list, map, merge: false, left, bottom + size, size, terrain);
            list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(map.Native)));
            Draw(list, _vegetation, cullBuffer, culler);
            Reduce(list, map, merge: true, left, bottom + size, size, terrain);

            // Read by the decal pass (compute)
            list.ResourceBarrierTransition(map.Native, ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
            _mapReadable[level] = true;
            _drawn[level] = new LevelData { Left = left, Top = bottom + size, Size = size, Map = map.BindlessIndex };
            if (level == 0)
            {
                _copyWanted = true;
                ReadBack(list);
            }
            PixMarker.End(list);
            _levelsDrawn++;
            return true;
        }

        /// <summary>
        /// The finest level for <see cref="OpenAt"/>: a copy taken in once its frame's slot has come round (the frame's
        /// fence was waited for then, as AutoExposure's histogram), and the level copied to the readback buffer when it has
        /// been drawn again and no copy is on its way. 4 MB, only when the camera has moved on 32 m or the scene changed.
        /// </summary>
        private static unsafe void ReadBack(ID3D12GraphicsCommandList list)
        {
            int slot = Engine.FrameIndex % FrameCount;
            if (_copyTick >= 0 && Engine.TickCount > _copyTick && slot == _copySlot)
            {
                _cpuMap ??= new uint[Resolution * Resolution];
                long bytes = (long)_cpuMap.Length * sizeof(uint);
                fixed (uint* to = _cpuMap)
                    Buffer.MemoryCopy(_readback!.ReadPtr<uint>(), to, bytes, bytes);
                _cpuLevel = _copyLevel;
                _cpuMinY = _copyMinY;
                _cpuMaxY = _copyMaxY;
                _copyTick = -1;
            }
            if (!_copyWanted || _copyTick >= 0 || _maps[0] == null || !_mapReadable[0])
                return;

            _readback ??= GraphicsBuffer.CreateReadback<uint>(Resolution * Resolution);
            var map = _maps[0]!.Native;
            list.ResourceBarrierTransition(map, ResourceStates.NonPixelShaderResource, ResourceStates.CopySource);
            var footprint = new PlacedSubresourceFootPrint
            {
                Offset = 0,
                Footprint = new SubresourceFootPrint(Format.R32_UInt, Resolution, Resolution, 1, Resolution * sizeof(uint))
            };
            list.CopyTextureRegion(new TextureCopyLocation(_readback.Native, footprint), 0, 0, 0, new TextureCopyLocation(map, 0), null);
            list.ResourceBarrierTransition(map, ResourceStates.CopySource, ResourceStates.NonPixelShaderResource);
            _copyWanted = false;
            _copyTick = Engine.TickCount;
            _copySlot = slot;
            _copyLevel = _drawn[0];
            _copyMinY = _minY;
            _copyMaxY = _maxY;
        }

        private static void Draw(ID3D12GraphicsCommandList list, List<InstanceBatch> batches, GraphicsBuffer cullBuffer, GPUCuller culler)
        {
            var dsv = _depth!.SliceDsvHandles[0];
            list.RSSetViewport(new Viewport(0, 0, DrawResolution, DrawResolution));
            list.RSSetScissorRect(new RectI(0, 0, DrawResolution, DrawResolution));
            list.ClearDepthStencilView(dsv, ClearFlags.Depth, 1.0f, 0);
            if (batches.Count > 0)
            {
                PixMarker.Begin(list, "Culling");
                InstanceBatch.CullShadowAll(list, batches, cullBuffer.SrvIndex, cullBuffer.Native.GPUVirtualAddress,
                    shadowHiZSrv: 0, culler, cascadeCount: 1, minCasterClip: -MinCasterTexels * 2f / DrawResolution, receiversUav: 0);
                PixMarker.End(list);

                // A single-slice DSV: the vertex shader routes the view to array index 0
                list.OMSetRenderTargets(0, (CpuDescriptorHandle[]?)null, false, dsv);
                PixMarker.Begin(list, "Draws");
                Material.SkyCaptureRaster = true;
                try
                {
                    foreach (var batch in batches)
                    {
                        batch.Material.SetPass(RenderPass.Shadow);
                        batch.Material.Apply(list, Engine.Device, null);
                        batch.DrawShadowSinglePass(list, cullBuffer.SrvIndex);
                    }
                }
                finally
                {
                    Material.SkyCaptureRaster = false;
                }
                PixMarker.End(list);
            }
            list.ResourceBarrierTransition(_depth.Native, ResourceStates.DepthWrite, ResourceStates.NonPixelShaderResource);
        }

        private static void Reduce(ID3D12GraphicsCommandList list, RenderTexture2D map, bool merge, float left, float top, float size,
            (uint Heightmap, float MaxHeight, Vector3 Origin, Vector2 Size) terrain)
        {
            var cs = _reduce!;
            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            cs.SetPushConstant("Depth", _depth!.BindlessIndex);
            cs.SetPushConstant("OutputUAV", map.UavIndex);
            cs.SetPushConstant("Merge", merge ? 1u : 0u);
            cs.SetPushConstant("Resolution", (uint)Resolution);
            cs.SetPushConstant("TerrainHeightmap", terrain.Heightmap);
            cs.SetParam("TerrainMaxHeight", terrain.MaxHeight);
            cs.SetParam("TerrainOriginX", terrain.Origin.X);
            cs.SetParam("TerrainOriginY", terrain.Origin.Y);
            cs.SetParam("TerrainOriginZ", terrain.Origin.Z);
            cs.SetParam("TerrainSizeX", terrain.Size.X);
            cs.SetParam("TerrainSizeZ", terrain.Size.Y);
            cs.SetParam("Left", left);
            cs.SetParam("Top", top);
            cs.SetParam("Size", size);
            cs.SetParam("MinY", _minY);
            cs.SetParam("MaxY", _maxY);
            cs.SetParam("MinAboveGround", MinAboveGround);
            cs.SetParam("CrownLeak", CrownLeak);
            cs.Dispatch(_kReduce, list, (uint)(Resolution + 7) / 8, (uint)(Resolution + 7) / 8);
            list.ResourceBarrierTransition(_depth.Native, ResourceStates.NonPixelShaderResource, ResourceStates.DepthWrite);
        }
    }
}
