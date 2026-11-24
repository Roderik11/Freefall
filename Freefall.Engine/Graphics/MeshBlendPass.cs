using System;
using Vortice.Direct3D12;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// Mesh blending, per renderer: where a mesh cuts into the ground or into another mesh (rocks in the terrain, the cliff
    /// kit's rocks pushed into each other), near the seam it takes on the look of what it meets, so the cut reads as a
    /// soft join. Right after the opaque pass and before the decals, on the G-buffer (mesh_blend_cs.hlsl): the seams are
    /// found from EntityId and the depth, the 16 px tiles they reach are listed, the surfaces there that blend keep what
    /// they were (so the other side of a seam is read as it was), and each of their pixels goes toward the surface across
    /// its nearest seam, in place. A surface blends by its material's <see cref="Material.BlendRadius"/>, found through
    /// <see cref="TransformBuffer.OfferMaterial"/>. Shading only: the outline where the seam is the silhouette stays sharp.
    /// Needs the G-buffer's unordered access (<see cref="DecalPass.Supported"/>); skipped while no material has a radius.
    /// What is kept: albedo and AO in <see cref="DeferredRenderer.Composite"/>, which nothing reads until composition writes
    /// all of it; the normal and roughness in a buffer of 4 bytes a pixel, the pass's only memory of its own besides the
    /// tile lists. Docs: .agent/knowledge/rendering_engine/features/mesh_blending.md.
    /// </summary>
    public sealed class MeshBlendPass : IDisposable
    {
        private const int Tile = 16;            // mesh_blend_cs.hlsl BLEND_TILE

        private static ComputeShader? _cs;
        private static int _kSeams, _kActive, _kKeep, _kBlend;

        private GraphicsBuffer? _reach, _args, _list, _keptNormals;
        private int _tilesX, _tilesY, _width, _height;
        private readonly ProfilerMarker _marker = ProfilerMarker.Get("Mesh blend");

        private void EnsureResources(int width, int height)
        {
            if (_cs == null)
            {
                _cs = new ComputeShader("mesh_blend_cs.hlsl");
                _kSeams = _cs.FindKernel("CSSeams");
                _kActive = _cs.FindKernel("CSActive");
                _kKeep = _cs.FindKernel("CSKeep");
                _kBlend = _cs.FindKernel("CSBlend");
            }
            _args ??= GraphicsBuffer.CreateRaw(3, uav: true);
            if (_reach != null && width == _width && height == _height)
                return;
            var device = Engine.Device;
            device.DeferDispose(_reach);
            device.DeferDispose(_list);
            device.DeferDispose(_keptNormals);
            _width = width;
            _height = height;
            _tilesX = (width + Tile - 1) / Tile;
            _tilesY = (height + Tile - 1) / Tile;
            _reach = GraphicsBuffer.CreateRaw(_tilesX * _tilesY, uav: true);
            _list = GraphicsBuffer.CreateRaw(_tilesX * _tilesY, uav: true);     // every tile can be listed
            _keptNormals = GraphicsBuffer.CreateStructured(width * height, sizeof(uint), uav: true);
        }

        /// <summary>
        /// Blends the G-buffer at the seams in view. Call right after the opaque pass with the G-buffer bound as render
        /// targets (RenderTarget state, the hardware depth in DepthWrite); they are left in those states, and the Composite
        /// target in NonPixelShaderResource as composition expects it. True when it ran: the caller binds the G-buffer
        /// targets and viewport again.
        /// </summary>
        public bool Render(ID3D12GraphicsCommandList list, Camera camera, DeferredRenderer renderer,
            Action<ID3D12GraphicsCommandList> bindSceneConstants)
        {
            if (!Engine.Settings.MeshBlending || Material.BlendingMaterials <= 0 || !DecalPass.Supported)
                return false;
            // Targets made without unordered access; or just made, the Composite target still in Common
            if (renderer.Albedo.UavIndex == 0 || renderer.Composite.UavIndex == 0 || renderer.IsFirstFrame)
                return false;

            using var _ = _marker.Auto();
            var desc = renderer.Albedo.Native.Description;
            int width = (int)desc.Width, height = (int)desc.Height;
            EnsureResources(width, height);

            var device = Engine.Device;
            list.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            list.SetComputeRootSignature(device.GlobalRootSignature);
            bindSceneConstants(list);

            var shaderRead = ResourceStates.NonPixelShaderResource;
            list.ResourceBarrierTransition(renderer.Albedo.Native, ResourceStates.RenderTarget, ResourceStates.UnorderedAccess);
            list.ResourceBarrierTransition(renderer.Normals.Native, ResourceStates.RenderTarget, ResourceStates.UnorderedAccess);
            list.ResourceBarrierTransition(renderer.Data.Native, ResourceStates.RenderTarget, ResourceStates.UnorderedAccess);
            list.ResourceBarrierTransition(renderer.EntityIdBuffer.Native, ResourceStates.RenderTarget, shaderRead);
            list.ResourceBarrierTransition(renderer.Depth.Native, ResourceStates.DepthWrite, shaderRead);
            list.ResourceBarrierTransition(renderer.Composite.Native, ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess);

            PixMarker.Begin(list, "Mesh blend");
            var cs = _cs!;
            cs.SetPushConstant("DepthTex", renderer.Depth.BindlessIndex);
            cs.SetPushConstant("EntityIdTex", renderer.EntityIdBuffer.BindlessIndex);
            cs.SetPushConstant("SlotMaterials", TransformBuffer.Instance.MaterialSrvIndex);
            cs.SetPushConstant("Materials", Material.MaterialsBufferIndex);
            cs.SetPushConstant("AlbedoUAV", renderer.Albedo.UavIndex);
            cs.SetPushConstant("NormalUAV", renderer.Normals.UavIndex);
            cs.SetPushConstant("DataUAV", renderer.Data.UavIndex);
            cs.SetPushConstant("TileReachUAV", _reach!.UavIndex);
            cs.SetPushConstant("ArgsUAV", _args!.UavIndex);
            cs.SetPushConstant("ListUAV", _list!.UavIndex);
            cs.SetPushConstant("KeptUAV", renderer.Composite.UavIndex);
            cs.SetPushConstant("KeptNormalsUAV", _keptNormals!.UavIndex);
            cs.SetPushConstant("TilesX", (uint)_tilesX);
            cs.SetPushConstant("TilesY", (uint)_tilesY);
            cs.SetPushConstant("MaxPixels", (uint)Math.Clamp(Engine.Settings.MeshBlendMaxPixels, 8, 256));
            cs.SetParam("PixelsPerMetre", 0.5f * height * camera.Projection.M22);
            cs.SetPushConstant("DebugView", Engine.Settings.DebugVisualizationMode == DebugVizMode.MeshBlend ? 1u : 0u);

            _reach.Transition(list, ResourceStates.UnorderedAccess);
            _args.Transition(list, ResourceStates.UnorderedAccess);
            _list.Transition(list, ResourceStates.UnorderedAccess);
            _keptNormals.Transition(list, ResourceStates.UnorderedAccess);

            // The seams and the tiles they reach
            cs.Dispatch(_kSeams, list, (uint)_tilesX, (uint)_tilesY);
            _reach.UAVBarrier(list);
            _args.UAVBarrier(list);
            cs.Dispatch(_kActive, list, (uint)((_tilesX + 7) / 8), (uint)((_tilesY + 7) / 8));
            _reach.UAVBarrier(list);
            _list.UAVBarrier(list);
            _args.Transition(list, ResourceStates.IndirectArgument);

            // What may change keeps what it was; then (all of it kept) the blend, in place
            cs.BindKernel(_kKeep, list);
            list.ExecuteIndirect(device.DispatchSignature, 1, _args.Native, 0, null, 0);
            list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
            cs.BindKernel(_kBlend, list);
            list.ExecuteIndirect(device.DispatchSignature, 1, _args.Native, 0, null, 0);
            PixMarker.End(list);

            list.ResourceBarrierTransition(renderer.Albedo.Native, ResourceStates.UnorderedAccess, ResourceStates.RenderTarget);
            list.ResourceBarrierTransition(renderer.Normals.Native, ResourceStates.UnorderedAccess, ResourceStates.RenderTarget);
            list.ResourceBarrierTransition(renderer.Data.Native, ResourceStates.UnorderedAccess, ResourceStates.RenderTarget);
            list.ResourceBarrierTransition(renderer.EntityIdBuffer.Native, shaderRead, ResourceStates.RenderTarget);
            list.ResourceBarrierTransition(renderer.Depth.Native, shaderRead, ResourceStates.DepthWrite);
            list.ResourceBarrierTransition(renderer.Composite.Native, ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
            return true;
        }

        public void Dispose()
        {
            _reach?.Dispose();
            _args?.Dispose();
            _list?.Dispose();
            _keptNormals?.Dispose();
        }
    }
}
