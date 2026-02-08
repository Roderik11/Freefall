using System;
using Vortice.Direct3D12;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12; // Ensure static import

namespace Freefall.Graphics
{
    public class PipelineState : IDisposable
    {
        private ID3D12PipelineState _pipelineState;
        private ID3D12RootSignature _rootSignature;
        public ID3D12PipelineState Native => _pipelineState;
        public ID3D12RootSignature RootSignature => _rootSignature;

        public PipelineState(GraphicsDevice device, ID3D12RootSignature rootSignature, GraphicsPipelineStateDescription description)
        {
            _rootSignature = rootSignature;
            description.RootSignature = rootSignature;
            
            try
            {
                _pipelineState = device.NativeDevice.CreateGraphicsPipelineState(description);
            }
            catch (Exception ex)
            {
                Debug.Log($"[PipelineState] PSO Creation FAILED!");
                Debug.Log($"  VS Bytecode: {description.VertexShader.Length} bytes");
                Debug.Log($"  PS Bytecode: {description.PixelShader.Length} bytes");
                Debug.Log($"  RTV Count: {description.RenderTargetFormats.Length}");
                for (int i = 0; i < description.RenderTargetFormats.Length; i++)
                    Debug.Log($"    RTV[{i}]: {description.RenderTargetFormats[i]}");
                Debug.Log($"  DSV Format: {description.DepthStencilFormat}");
                Debug.Log($"  PrimitiveTopology: {description.PrimitiveTopologyType}");
                Debug.Log($"  SampleMask: {description.SampleMask}");
                throw;
            }
        }

        public static ID3D12RootSignature CreateRootSignature(GraphicsDevice device, RootSignatureDescription description)
        {
            Debug.Log($"Creating Root Signature with version 1.0...");
            return device.NativeDevice.CreateRootSignature(description, RootSignatureVersion.Version10);
        }

        public void Dispose()
        {
            _pipelineState?.Dispose();
            _rootSignature?.Dispose();
        }
    }
}
