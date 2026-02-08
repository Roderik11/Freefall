using System;
using System.Numerics;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;
using Freefall.Assets;
using Freefall.Components;
using Freefall.Base;

namespace Freefall.Graphics
{
    public class ForwardRenderer : RenderPipeline
    {
        private List<Entity> _entities = new List<Entity>();
        
        // Keep these for resource management/disposal
        private Effect? _effect;
        private PipelineState? _sharedPipeline;

        public ForwardRenderer(GraphicsDevice device)
        {
            // Device is now global via Engine.Device, but we can keep the constructor signature for now or refactor it.
            // For now, let's keep it compatible with RenderView which passes it.
        }

        public override void Initialize(int width, int height)
        {
            // Root Signature with root CBVs for transforms and SRV table for textures
            var rootParameters = new RootParameter[]
            {
                new RootParameter(RootParameterType.ConstantBufferView, new RootDescriptor(0, 0), ShaderVisibility.Vertex),
                new RootParameter(RootParameterType.ConstantBufferView, new RootDescriptor(1, 0), ShaderVisibility.Vertex),
                new RootParameter(new RootDescriptorTable(new DescriptorRange(DescriptorRangeType.ShaderResourceView, 1, 0)), ShaderVisibility.Pixel),
            };

            var staticSamplers = new StaticSamplerDescription[]
            {
                new StaticSamplerDescription(ShaderVisibility.Pixel, 0, 0)
                {
                    Filter = Filter.MinMagMipLinear,
                    AddressU = TextureAddressMode.Wrap,
                    AddressV = TextureAddressMode.Wrap,
                    AddressW = TextureAddressMode.Wrap,
                    MipLODBias = 0,
                    MaxAnisotropy = 1,
                    ComparisonFunction = ComparisonFunction.Never,
                    BorderColor = StaticBorderColor.TransparentBlack,
                    MinLOD = 0,
                    MaxLOD = float.MaxValue,
                }
            };

            var rootSignatureDesc = new RootSignatureDescription(RootSignatureFlags.AllowInputAssemblerInputLayout, rootParameters, staticSamplers);
            var rootSignature = PipelineState.CreateRootSignature(Engine.Device, rootSignatureDesc);
            Debug.Log($"Root Signature created successfully: {rootSignature != null}");

            // Load Effect/Shaders
            string shaderPath = "Basic";
            try 
            {
                _effect = new Effect(shaderPath);
            }
            catch (Exception ex)
            {
                Debug.Log($"Failed to load effect: {ex.Message}");
                throw;
            }

            var technique = _effect.Techniques.Find(t => t.Name == "Standard");
            if (technique == null) throw new Exception("Technique 'Standard' not found in Basic.fx");
            
            var pass = technique.Passes[0];
            var vs = pass.VertexShader;
            var ps = pass.PixelShader;
            if (vs == null || ps == null) throw new Exception("Failed to load shaders from effect pass");

            // Input Layout
            InputElementDescription[] inputElementDescs = new[]
            {
                new InputElementDescription("POSITION", 0, Format.R32G32B32_Float, 0, 0, InputClassification.PerVertexData, 0),
                new InputElementDescription("NORMAL", 0, Format.R32G32B32_Float, 0, 1, InputClassification.PerVertexData, 0),
                new InputElementDescription("TEXCOORD", 0, Format.R32G32_Float, 0, 2, InputClassification.PerVertexData, 0),
            };

            // PSO
            var rasterizerDesc = new RasterizerDescription
            {
                CullMode = CullMode.None,
                FillMode = FillMode.Solid,
                FrontCounterClockwise = false,
                DepthBias = 0,
                DepthBiasClamp = 0.0f,
                SlopeScaledDepthBias = 0.0f,
                DepthClipEnable = true,
                MultisampleEnable = false,
                AntialiasedLineEnable = false,
                ForcedSampleCount = 0,
                ConservativeRaster = ConservativeRasterizationMode.Off
            };
            
            var psoDesc = new GraphicsPipelineStateDescription()
            {
                RootSignature = rootSignature,
                VertexShader = vs.Bytecode,
                PixelShader = ps.Bytecode,
                StreamOutput = new StreamOutputDescription(),
                BlendState = BlendDescription.Opaque,
                SampleMask = uint.MaxValue,
                RasterizerState = rasterizerDesc,
                DepthStencilState = DepthStencilDescription.Default,
                InputLayout = new InputLayoutDescription(inputElementDescs),
                IndexBufferStripCutValue = IndexBufferStripCutValue.Disabled,
                PrimitiveTopologyType = PrimitiveTopologyType.Triangle,
                DepthStencilFormat = Format.D32_Float,
                SampleDescription = new SampleDescription(1, 0),
                NodeMask = 0,
                Flags = PipelineStateFlags.None
            };
            psoDesc.RenderTargetFormats[0] = Format.R8G8B8A8_UNorm;

            _sharedPipeline = new PipelineState(Engine.Device, rootSignature, psoDesc);

            // Load Assets using Engine.Assets
            string texturePath = @"D:\Projects\2024\ProjectXYZ\Resources\Tree Prototypes\Oak\Oak_Trees\Textures\Oak_Dif.png";
            var texture = Engine.Assets.LoadTexture(texturePath);

            string modelPath = @"D:\Projects\2024\ProjectXYZ\Resources\Tree Prototypes\Oak\Oak_Trees\Oak_01.fbx";
            var mesh = Engine.Assets.LoadMesh(modelPath);

            // Create Objects
            CreateEntity(new Vector3(0, 0, 0), texture, mesh);
            CreateEntity(new Vector3(300, 0, 0), texture, mesh); // Second tree further away
        }

        private void CreateEntity(Vector3 position, Texture texture, Mesh mesh)
        {
            var material = new Material(_sharedPipeline!, _effect!.Techniques[0].Passes[0].VertexShader!, _effect!.Techniques[0].Passes[0].PixelShader!, Engine.Device);
            material.SetParameter("Tint", new Vector4(1.0f));
            material.SetTexture("Albedo", texture);

            var entity = new Entity("Tree");
            entity.Transform.Position = position;
            
            var renderer = entity.AddComponent<MeshRenderer>();
            renderer.Mesh = mesh;
            renderer.Material = material;

            _entities.Add(entity);
        }

        public override void Resize(int width, int height)
        {
        }

        public override void Clear(Camera camera)
        {
            var commandList = camera.Target.CommandList;
            commandList.ClearDepthStencilView(camera.Target.DepthBufferTarget, ClearFlags.Depth, 1.0f, 0);
        }

        public override void Render(Camera camera, ID3D12GraphicsCommandList list)
        {
            float time = (float)Time.TotalTime;
            
            list.IASetPrimitiveTopology(PrimitiveTopology.TriangleList);

            foreach (var entity in _entities)
            {
                // Simple animation for testing
                entity.Transform.SetRotation(0, time * 0.3f, 0);
                
                var renderer = entity.GetComponent<MeshRenderer>();
                if (renderer != null && renderer.Mesh != null && renderer.Material != null)
                {
                    var mb = new MaterialBlock();
                    mb.SetParameter("World", entity.Transform.WorldMatrix);
                    CommandBuffer.Enqueue(RenderPass.Opaque, renderer.Mesh, renderer.Material, mb);
                }
            }

            camera.SetShaderParams();
            CommandBuffer.Execute(RenderPass.Opaque, list, Engine.Device);
        }

        public override void Dispose()
        {
            // Engine.Assets and Engine.Device are disposed by Engine.Shutdown
            _sharedPipeline?.Dispose();
            _effect?.Dispose();
            
            // Clearning entities list - components destroy logic would go here
            _entities.Clear();
        }
    }
}
