using System;
using System.Numerics;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// Which of this frame's visible point lights reach a place in the view, for what is not an opaque surface: the air
    /// (<see cref="VolumetricFog"/>), particles and ribbons. Z-binning: the lights sorted by view depth, the first and last
    /// of them reaching into each of 1024 bins of view depth, and a bitmask over them per 32-pixel tile; no list has a cap
    /// (light_clusters.hlsli). One per <see cref="DeferredRenderer"/>.
    ///
    /// Built after the point light shadow selection (which writes each light's shadow into the visible list), before the
    /// fog; read through SceneConstants.LightClustersIdx / ClusterLightsIdx until <see cref="EndFrame"/>.
    /// Docs: .agent/knowledge/rendering_engine/features/lighting/point_lights.md "Light clusters".
    /// </summary>
    public sealed class LightClusters : IDisposable
    {
        public const int TileShift = 5;                                 // light_clusters.hlsli LIGHT_CLUSTER_TILE_SHIFT
        public const int TileSize = 1 << TileShift;
        public const int ZBins = 1024;                                  // LIGHT_ZBINS
        private const int Words = PointLightBuffer.MaxVisible / 32;     // LIGHT_CLUSTER_WORDS
        private const int MasksOffset = 16 + ZBins + PointLightBuffer.MaxVisible;  // LC_MASKS

        private const ResourceStates ShaderRead = ResourceStates.NonPixelShaderResource | ResourceStates.PixelShaderResource;

        private readonly ComputeShader _cs;
        private readonly int _kSort, _kBins, _kTiles;
        private readonly GraphicsBuffer _lights;
        private GraphicsBuffer? _clusters;
        private int _tilesX, _tilesY, _width, _height;
        private bool _readable;     // built this frame: both buffers are in ShaderRead until EndFrame

        /// <summary>SceneConstants.ClusterLightsIdx: the visible lights in view-depth order (StructuredBuffer&lt;PointLightData&gt;).</summary>
        public uint LightsSrv => _lights.SrvIndex;

        /// <summary>SceneConstants.LightClustersIdx: header, z-bins and tile masks (ByteAddressBuffer); 0 before <see cref="Prepare"/>.</summary>
        public uint ClustersSrv => _clusters?.SrvIndex ?? 0u;

        public LightClusters()
        {
            _cs = new ComputeShader("light_clusters_cs.hlsl");
            _kSort = _cs.FindKernel("CSClusterSort");
            _kBins = _cs.FindKernel("CSClusterBins");
            _kTiles = _cs.FindKernel("CSClusterTiles");
            _lights = GraphicsBuffer.CreateStructured<PointLightBuffer.VisibleLight>(PointLightBuffer.MaxVisible, srv: true, uav: true);
        }

        /// <summary>Size the tile masks to the view (pixels of the target the passes draw into).</summary>
        public void Prepare(int width, int height)
        {
            width = Math.Max(1, width);
            height = Math.Max(1, height);
            int tx = (width + TileSize - 1) / TileSize, ty = (height + TileSize - 1) / TileSize;
            if (_clusters == null || tx != _tilesX || ty != _tilesY)
            {
                // Frames in flight may still read the old one
                if (_clusters != null)
                    Engine.Device.DeferDispose(_clusters);
                _clusters = GraphicsBuffer.CreateRaw(MasksOffset + tx * ty * Words, srv: true, uav: true);
                _tilesX = tx;
                _tilesY = ty;
            }
            _width = width;
            _height = height;
        }

        /// <summary>
        /// Sort the visible lights and build the bins and the tile masks. <paramref name="visibleCountUav"/> 0 = no lights
        /// this frame (the header then says none). This frame's point light shadows and the directional light's cascades
        /// (<paramref name="sunShadowMap"/> 0 = none drawn) go into the header for the readers: particles have no other
        /// way to them.
        /// </summary>
        public void Build(ID3D12GraphicsCommandList list, uint frameParamsSrv, uint visibleUav, uint visibleCountUav,
            uint shadowAtlas, uint shadowViews, uint shadowLights, uint sunShadowMap, uint sunCascades, int sunCascadeCount,
            Vector3 sunDirection)
        {
            if (_clusters == null) return;

            // Buffers decay to Common between command lists: the first write below promotes them to UAV
            PixMarker.Begin(list, "Light clusters");
            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);

            _cs.SetPushConstant("VisibleUAV", visibleUav);
            _cs.SetPushConstant("VisibleCountUAV", visibleCountUav);
            _cs.SetPushConstant("FrameParams", frameParamsSrv);
            _cs.SetPushConstant("LightsUAV", _lights.UavIndex);
            _cs.SetPushConstant("ClustersUAV", _clusters.UavIndex);
            _cs.SetPushConstant("ViewWidth", (uint)_width);
            _cs.SetPushConstant("ViewHeight", (uint)_height);
            _cs.SetPushConstant("TilesX", (uint)_tilesX);
            _cs.SetPushConstant("ShadowAtlas", shadowAtlas);
            _cs.SetPushConstant("ShadowViews", shadowAtlas != 0 ? shadowViews : 0u);
            _cs.SetPushConstant("ShadowLights", shadowAtlas != 0 ? shadowLights : 0u);
            bool sun = sunShadowMap != 0 && sunCascades != 0 && sunCascadeCount > 0;
            _cs.SetPushConstant("SunShadowMap", sun ? sunShadowMap : 0u);
            _cs.SetPushConstant("SunCascades", sun ? sunCascades : 0u);
            _cs.SetPushConstant("SunCascadeCount", sun ? (uint)sunCascadeCount : 0u);
            _cs.SetPushConstant("SunDirectionX", BitConverter.SingleToUInt32Bits(sunDirection.X));
            _cs.SetPushConstant("SunDirectionY", BitConverter.SingleToUInt32Bits(sunDirection.Y));
            _cs.SetPushConstant("SunDirectionZ", BitConverter.SingleToUInt32Bits(sunDirection.Z));

            _cs.Dispatch(_kSort, list, 1);
            UavBarrier(list);
            if (visibleCountUav != 0)
            {
                _cs.Dispatch(_kBins, list, (uint)(ZBins / 64));
                _cs.Dispatch(_kTiles, list, (uint)_tilesX, (uint)_tilesY);
                UavBarrier(list);
            }

            list.Barriers(
                new ResourceBarrier(new ResourceTransitionBarrier(_lights.Native, ResourceStates.UnorderedAccess, ShaderRead)),
                new ResourceBarrier(new ResourceTransitionBarrier(_clusters.Native, ResourceStates.UnorderedAccess, ShaderRead)));
            _readable = true;
            PixMarker.End(list);
        }

        /// <summary>After the last pass that reads them (the forward pass): back to UAV for the next build.</summary>
        public void EndFrame(ID3D12GraphicsCommandList list)
        {
            if (!_readable || _clusters == null) return;
            list.Barriers(
                new ResourceBarrier(new ResourceTransitionBarrier(_lights.Native, ShaderRead, ResourceStates.UnorderedAccess)),
                new ResourceBarrier(new ResourceTransitionBarrier(_clusters.Native, ShaderRead, ResourceStates.UnorderedAccess)));
            _readable = false;
        }

        private static void UavBarrier(ID3D12GraphicsCommandList list) =>
            list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

        public void Dispose()
        {
            _clusters?.Dispose();
            _lights.Dispose();
            _cs.Dispose();
        }
    }
}
