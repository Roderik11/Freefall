using System;
using Vortice.Direct3D12;
using Vortice.Mathematics;
using Vortice.Direct3D;

namespace Freefall.Graphics
{
    public class CommandList : IDisposable
    {
        private readonly ID3D12GraphicsCommandList _native;
        private readonly ID3D12CommandAllocator _allocator;
        private bool _disposed;

        public ID3D12GraphicsCommandList Native => _native;

        public CommandList(GraphicsDevice device)
        {
            _allocator = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct);
            _native = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, _allocator, null);
            _native.Close();
        }

        public void Reset()
        {
            _allocator.Reset();
            _native.Reset(_allocator, null);
        }

        public void Close()
        {
            _native.Close();
        }

        public void ResourceBarrierTransition(ID3D12Resource resource, ResourceStates stateBefore, ResourceStates stateAfter)
        {
            if (stateBefore == stateAfter) return;
            _native.ResourceBarrier(ResourceBarrier.BarrierTransition(resource, stateBefore, stateAfter));
        }

        public void SetRenderTargets(CpuDescriptorHandle rtvHandle, CpuDescriptorHandle? dsvHandle)
        {
            _native.OMSetRenderTargets(rtvHandle, dsvHandle);
        }

        public void SetViewport(Viewport viewport)
        {
            _native.RSSetViewport(viewport);
        }

        public void SetScissorRect(RectI scissorRect)
        {
            _native.RSSetScissorRect(scissorRect);
        }

        public void ClearRenderTargetView(CpuDescriptorHandle rtvHandle, Color4 color)
        {
            _native.ClearRenderTargetView(rtvHandle, color);
        }

        public void ClearDepthStencilView(CpuDescriptorHandle dsvHandle, ClearFlags clearFlags, float depth, byte stencil)
        {
            _native.ClearDepthStencilView(dsvHandle, clearFlags, depth, stencil);
        }

        public void SetPipelineState(PipelineState pipelineState)
        {
            _native.SetPipelineState(pipelineState.Native);
        }

        public void SetGraphicsRootSignature(ID3D12RootSignature rootSignature)
        {
            _native.SetGraphicsRootSignature(rootSignature);
        }

        public void SetPrimitiveTopology(PrimitiveTopology topology)
        {
            _native.IASetPrimitiveTopology(topology);
        }

        public void DrawInstanced(int vertexCountPerInstance, int instanceCount, int startVertexLocation, int startInstanceLocation)
        {
            _native.DrawInstanced((uint)vertexCountPerInstance, (uint)instanceCount, (uint)startVertexLocation, (uint)startInstanceLocation);
        }

        public void DrawIndexedInstanced(int indexCountPerInstance, int instanceCount, int startIndexLocation, int baseVertexLocation, int startInstanceLocation)
        {
            _native.DrawIndexedInstanced((uint)indexCountPerInstance, (uint)instanceCount, (uint)startIndexLocation, baseVertexLocation, (uint)startInstanceLocation);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _native?.Dispose();
                _allocator?.Dispose();
                _disposed = true;
            }
        }
    }
}
