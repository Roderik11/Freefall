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
        /// <summary>
        /// Description for creating a mesh shader PSO. Mirrors GraphicsPipelineStateDescription
        /// but for the AS → MS → PS pipeline.
        /// </summary>
        public class MeshShaderPSODescription
        {
            public ReadOnlyMemory<byte> AmplificationShader;
            public ReadOnlyMemory<byte> MeshShader;
            public ReadOnlyMemory<byte> PixelShader;
            public RasterizerDescription RasterizerState;
            public BlendDescription BlendState;
            public DepthStencilDescription DepthStencilState;
            public Format[] RenderTargetFormats = Array.Empty<Format>();
            public Format DepthStencilFormat;
            public SampleDescription SampleDescription = new(1, 0);
            public uint SampleMask = uint.MaxValue;
        }

        /// <summary>
        /// Create a PSO for mesh shader pipeline (AS → MS → PS).
        /// Uses the generic pipeline stream API (ID3D12Device2.CreatePipelineState).
        /// </summary>
        public unsafe PipelineState(GraphicsDevice device, ID3D12RootSignature rootSignature, MeshShaderPSODescription desc)
        {
            _rootSignature = rootSignature;

            try
            {
                // Pin shader bytecodes so they survive the CreatePipelineState call
                fixed (byte* msPtr = desc.MeshShader.Span)
                fixed (byte* asPtr = desc.AmplificationShader.Span)
                fixed (byte* psPtr = desc.PixelShader.Span)
                {
                    // Build pipeline stream as sequential sub-objects
                    var stream = new MeshShaderPipelineStream();
                    stream.RootSignature = new PipelineStateSubObjectTypeRootSignature(rootSignature);
                    stream.MeshShader = new PipelineStateSubObjectTypeMeshShader(desc.MeshShader.Span);
                    if (desc.AmplificationShader.Length > 0)
                        stream.AmplificationShader = new PipelineStateSubObjectTypeAmplificationShader(desc.AmplificationShader.Span);
                    stream.PixelShader = new PipelineStateSubObjectTypePixelShader(desc.PixelShader.Span);
                    stream.BlendState = new PipelineStateSubObjectTypeBlend(desc.BlendState);
                    stream.RasterizerState = new PipelineStateSubObjectTypeRasterizer(desc.RasterizerState);
                    stream.DepthStencilState = new PipelineStateSubObjectTypeDepthStencil(desc.DepthStencilState);
                    stream.RenderTargetFormats = new PipelineStateSubObjectTypeRenderTargetFormats(desc.RenderTargetFormats);
                    stream.DepthStencilFormat = new PipelineStateSubObjectTypeDepthStencilFormat(desc.DepthStencilFormat);
                    stream.SampleDescription = new PipelineStateSubObjectTypeSampleDescription(desc.SampleDescription);
                    stream.SampleMask = new PipelineStateSubObjectTypeSampleMask(desc.SampleMask);

                    // ID3D12Device2 is required for pipeline stream PSO creation (mesh shaders)
                    using var device2 = device.NativeDevice.QueryInterface<ID3D12Device2>();
                    _pipelineState = device2.CreatePipelineState(stream);
                }
            }
            catch (Exception ex)
            {
                Debug.Log($"[PipelineState] Mesh Shader PSO Creation FAILED: {ex.Message}");
                Debug.Log($"  MS Bytecode: {desc.MeshShader.Length} bytes");
                Debug.Log($"  AS Bytecode: {desc.AmplificationShader.Length} bytes");
                Debug.Log($"  PS Bytecode: {desc.PixelShader.Length} bytes");
                Debug.Log($"  RTV Count: {desc.RenderTargetFormats.Length}");
                Debug.Log($"  DSV Format: {desc.DepthStencilFormat}");
                throw;
            }
        }

        /// <summary>
        /// Pipeline state stream layout for mesh shader PSOs.
        /// Sub-objects must be laid out sequentially — D3D12 reads them as a byte stream.
        /// </summary>
        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        private struct MeshShaderPipelineStream
        {
            public PipelineStateSubObjectTypeRootSignature RootSignature;
            public PipelineStateSubObjectTypeMeshShader MeshShader;
            public PipelineStateSubObjectTypeAmplificationShader AmplificationShader;
            public PipelineStateSubObjectTypePixelShader PixelShader;
            public PipelineStateSubObjectTypeBlend BlendState;
            public PipelineStateSubObjectTypeRasterizer RasterizerState;
            public PipelineStateSubObjectTypeDepthStencil DepthStencilState;
            public PipelineStateSubObjectTypeRenderTargetFormats RenderTargetFormats;
            public PipelineStateSubObjectTypeDepthStencilFormat DepthStencilFormat;
            public PipelineStateSubObjectTypeSampleDescription SampleDescription;
            public PipelineStateSubObjectTypeSampleMask SampleMask;
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
