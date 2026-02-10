using System;
using System.Numerics;
using System.Collections.Generic;
using Vortice.Direct3D12;
using Vortice.Direct3D12.Shader;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Material texture indices for bindless lookup - matches HLSL MaterialData layout
    /// </summary>
    public struct MaterialData
    {
        public uint AlbedoIdx;
        public uint NormalIdx;
        public uint RoughnessIdx;
        public uint MetallicIdx;
        public uint EmissiveIdx;
        public uint AOIdx;
        public uint Padding0;
        public uint Padding1;
    }
    
    /// <summary>
    /// Maps a shader pass to a RenderPass enum (Apex pattern)
    /// </summary>
    public class ShaderPass
    {
        public RenderPass RenderPass;
        public EffectPass EffectPass = null!;
        public int Index;
    }

    /// <summary>
    /// Parses technique passes and maps them to RenderPass enum by name (Apex pattern)
    /// </summary>
    public class SubShader
    {
        public List<ShaderPass> ShaderPasses = new();

        public SubShader(EffectTechnique technique)
        {
            for (int i = 0; i < technique.Passes.Count; i++)
            {
                var pass = technique.Passes[i];
                Enum.TryParse<RenderPass>(pass.Name, out var type);

                ShaderPasses.Add(new ShaderPass
                {
                    EffectPass = pass,
                    RenderPass = type,
                    Index = i
                });
            }
        }
    }

    public class Material : IDisposable
    {

        
        // Per-pass PSOs (Apex pattern for multi-pass rendering like shadow maps)
        private Dictionary<int, PipelineState> _passPipelineStates = new();
        private Dictionary<int, PipelineState> _passWireframePSOs = new();
        private bool _isFullscreenPass = false; // Skip wireframe for post-process passes
        private Dictionary<string, ConstantBuffer> _constantBuffers = new();
        private Dictionary<string, Texture> _textures = new();
        
        // Material ID indirection system
        public int MaterialID { get; private set; }
        private static int _nextMaterialID = 0;
        private static List<MaterialData> _allMaterials = new();
        private static readonly object _materialIdLock = new();
        private static ID3D12Resource? _materialsBuffer;
        private static IntPtr _materialsBufferPtr;
        private static bool _materialsBufferDirty = true;
        
        // SubShaders parsed from Effect techniques (Apex pattern)
        private readonly List<SubShader> _subShaders = new();
        
        public PipelineState PipelineState => _passPipelineStates.TryGetValue(Pass, out var pso) ? pso : _passPipelineStates[0];
        public IEnumerable<ConstantBuffer> ConstantBuffers => _constantBuffers.Values;
        public Effect? Effect { get; private set; }
        
        // Pass/Technique selection (Apex pattern)
        public int Pass { get; set; } = 0;
        public int Technique { get; set; } = 0;
        
        public List<ShaderPass> GetPasses() => _subShaders.Count > Technique ? _subShaders[Technique].ShaderPasses : new();

        /// <summary>True if the active pass has Hull+Domain shaders (requires patch topology).</summary>
        public bool HasTessellation
        {
            get
            {
                var passes = GetPasses();
                if (Pass < passes.Count)
                    return passes[Pass].EffectPass.HullShader != null;
                return false;
            }
        }
        
        /// <summary>
        /// Set active pass by RenderPass enum (Apex pattern)
        /// </summary>
        public void SetPass(RenderPass pass)
        {
            int i = 0;
            foreach (var p in GetPasses())
            {
                if (p.RenderPass == pass)
                {
                    Pass = i;
                    return;
                }
                i++;
            }
        }
        
        public RenderPass GetPass(int index) => GetPasses()[index].RenderPass;
        
        /// <summary>
        /// Check if material has a specific render pass defined
        /// </summary>
        public bool HasPass(RenderPass pass)
        {
            foreach (var p in GetPasses())
            {
                if (p.RenderPass == pass) return true;
            }
            return false;
        }
        
        private Dictionary<string, int> _textureSlots = new();

        public Material(Effect effect) : this(effect, Freefall.Engine.Device) { }

        public Material(Effect effect, GraphicsDevice device)
        {
            Debug.Log($"[Material] Creating Material for Effect: {effect.Name}");
            Effect = effect;
            
            // Populate SubShaders from Effect techniques (Apex pattern)
            foreach (var technique in effect.Techniques)
                _subShaders.Add(new SubShader(technique));
            
            // Assign unique MaterialID and register in global materials list (thread-safe)
            lock (_materialIdLock)
            {
                MaterialID = _nextMaterialID++;
                _allMaterials.Add(new MaterialData()); // Will be populated by SetTexture
                _materialsBufferDirty = true;
            }
            
            var renderState = effect.RenderState;

            // Determine Render Target count and formats from shader metadata
            Format[] rtvFormats = renderState.RenderTargetCount switch
            {
                4 => new[] { Format.R8G8B8A8_UNorm, Format.R16G16B16A16_SNorm, Format.R8G8B8A8_UNorm, Format.R32_Float }, // GBuffer + Depth
                3 => new[] { Format.R8G8B8A8_UNorm, Format.R16G16B16A16_SNorm, Format.R8G8B8A8_UNorm }, // GBuffer
                _ => new[] { Format.R8G8B8A8_UNorm } // Standard single target
            };
            Debug.Log($"[Material] RTV Formats Count: {rtvFormats.Length}");

            // In bindless, we use the Global Root Signature from the device
            var rootSignature = device.GlobalRootSignature;

            // Input Layout (Bindless = Empty)
            // We fetch vertices manually in VS using SV_VertexID

            // Rasterizer State
            var rasterizerDesc = RasterizerDescription.CullNone; // Debug: Disable culling

            // Blend State - use shader metadata, fall back to Effect's property
            BlendDescription blendDesc = renderState.BlendMode switch
            {
                "Additive" => new BlendDescription(Blend.One, Blend.One),
                "AlphaBlend" => BlendDescription.AlphaBlend,
                _ => effect.BlendState == BlendState.Additive 
                    ? new BlendDescription(Blend.One, Blend.One)
                    : effect.BlendState == BlendState.AlphaBlend 
                        ? BlendDescription.AlphaBlend 
                        : BlendDescription.Opaque
            };

            // Depth Stencil State - use shader metadata
            DepthStencilDescription depthDesc;
            Format depthFormat;
            
            if (!renderState.DepthTest && !renderState.DepthWrite)
            {
                // No depth (fullscreen quads, post-process)
                depthDesc = DepthStencilDescription.None;
                depthFormat = Format.Unknown;
            }
            else if (!renderState.DepthWrite)
            {
                // Read only (skybox, transparent)
                depthDesc = new DepthStencilDescription
                {
                    DepthEnable = true,
                    DepthWriteMask = DepthWriteMask.Zero,
                    DepthFunc = ComparisonFunction.LessEqual,
                    StencilEnable = false
                };
                depthFormat = Format.D32_Float;
            }
            else
            {
                // Full depth (standard geometry)
                depthDesc = DepthStencilDescription.Default;
                depthFormat = Format.D32_Float;
            }


            var psoDescBase = new GraphicsPipelineStateDescription()
            {
                RootSignature = rootSignature,
                InputLayout = default, 
                RasterizerState = rasterizerDesc,
                BlendState = blendDesc,
                DepthStencilState = depthDesc,
                PrimitiveTopologyType = PrimitiveTopologyType.Triangle,
                RenderTargetFormats = rtvFormats,
                DepthStencilFormat = depthFormat,
                SampleDescription = new SampleDescription(1, 0),
                SampleMask = uint.MaxValue,
                Flags = PipelineStateFlags.None
            };

            // Create PSO for each pass (Apex multi-pass pattern for shadow maps, etc.)
            var firstTechnique = effect.Techniques[0];
            for (int passIndex = 0; passIndex < firstTechnique.Passes.Count; passIndex++)
            {
                var currentPass = firstTechnique.Passes[passIndex];
                
                var psoDesc = psoDescBase;
                psoDesc.VertexShader = currentPass.VertexShader?.Bytecode ?? default;
                psoDesc.PixelShader = currentPass.PixelShader?.Bytecode ?? default;
                psoDesc.HullShader = currentPass.HullShader?.Bytecode ?? default;
                psoDesc.DomainShader = currentPass.DomainShader?.Bytecode ?? default;
                
                // Tessellation requires Patch topology type in PSO
                if (currentPass.HullShader != null)
                    psoDesc.PrimitiveTopologyType = PrimitiveTopologyType.Patch;
                
                // Shadow pass: depth-only rendering (0 RTVs, D32 DSV, depth write enabled)
                // Use the parsed RenderPass enum instead of string comparison (Apex pattern)
                var passType = _subShaders[0].ShaderPasses[passIndex].RenderPass;
                if (passType == RenderPass.Shadow)
                {
                    psoDesc.RenderTargetFormats = Array.Empty<Format>();
                    psoDesc.DepthStencilFormat = Format.D32_Float;
                    psoDesc.DepthStencilState = DepthStencilDescription.Default;
                    var shadowRaster = new RasterizerDescription(CullMode.None, FillMode.Solid);
                    shadowRaster.DepthBias = 4000;
                    shadowRaster.SlopeScaledDepthBias = 2.0f;
                    shadowRaster.DepthBiasClamp = 0.0f;
                    psoDesc.RasterizerState = shadowRaster;
                }
                
                _passPipelineStates[passIndex] = new PipelineState(device, rootSignature, psoDesc);
                
                // Create wireframe PSO variant
                psoDesc.RasterizerState = new RasterizerDescription(CullMode.None, FillMode.Wireframe);
                _passWireframePSOs[passIndex] = new PipelineState(device, rootSignature, psoDesc);
                
                // Reflection for constant buffers (from first pass for simplicity)
                if (passIndex == 0)
                {
                    if (currentPass.VertexShader?.Reflection != null) CreateConstantBuffers(device, currentPass.VertexShader.Reflection);
                    if (currentPass.PixelShader?.Reflection != null) CreateConstantBuffers(device, currentPass.PixelShader.Reflection);
                    if (currentPass.HullShader?.Reflection != null) CreateConstantBuffers(device, currentPass.HullShader.Reflection);
                    if (currentPass.DomainShader?.Reflection != null) CreateConstantBuffers(device, currentPass.DomainShader.Reflection);
                }
            }
            
            // Detect fullscreen passes (no depth = post-process, don't wireframe)
            _isFullscreenPass = !renderState.DepthTest && !renderState.DepthWrite;
        }

        public Material(PipelineState pipelineState, Shader vertexShader, Shader pixelShader, GraphicsDevice device)
        {
            _passPipelineStates[0] = pipelineState; // Custom PSO provided for pass 0

            // Reflect and Create Constant Buffers
            if (vertexShader.Reflection != null)
                CreateConstantBuffers(device, vertexShader.Reflection);
            
            if (pixelShader.Reflection != null)
                CreateConstantBuffers(device, pixelShader.Reflection);
        }
        
        private void CreateConstantBuffers(GraphicsDevice device, ID3D12ShaderReflection reflection)
        {
            for (uint i = 0; i < reflection.Description.ConstantBuffers; i++)
            {
                var buffer = reflection.GetConstantBufferByIndex(i);
                var desc = buffer.Description;
                
                if (!_constantBuffers.ContainsKey(desc.Name))
                {
                    var cb = new ConstantBuffer(device, buffer);
                    
                    // Assign slot based on name (matches root signature layout)
                    cb.Slot = desc.Name switch
                    {
                        "SceneConstants" => 1,
                        "ObjectConstants" => 2,
                        "terrain" => 2,
                        "landscape" => 2,
                        "tiling" => 3,
                        _ => -1 // Not bound automatically (e.g. PushConstants)
                    };
                    
                    _constantBuffers.Add(desc.Name, cb);
                }
                else
                {
                    // Merge variables from additional shader stages.
                    // DXC optimizes out unused variables per-stage, so the DS may
                    // reference cbuffer members (e.g., MaxHeight) that the VS doesn't.
                    _constantBuffers[desc.Name].MergeVariables(buffer);
                }
            }

            for (uint i = 0; i < reflection.Description.BoundResources; i++)
            {
                var desc = reflection.GetResourceBindingDescription(i);
                if (desc.Type == Vortice.Direct3D.ShaderInputType.Texture || 
                    desc.Type == Vortice.Direct3D.ShaderInputType.Structured ||
                    desc.Type == Vortice.Direct3D.ShaderInputType.ByteAddress)
                {
                    _textureSlots[desc.Name] = (int)desc.BindPoint;
                }
            }
        }


        public void SetParameter<T>(string name, T value) where T : unmanaged
        {
            foreach (var cb in _constantBuffers.Values)
                cb.SetParameter(name, value);
        }

        public void SetParameter<T>(string name, T[] values) where T : unmanaged
        {
            foreach (var cb in _constantBuffers.Values)
            {
                cb.SetParameterArray(name, values);
            }
        }

        public void SetTexture(string name, Texture texture)
        {
            if (texture == null) return;
            _textures[name] = texture;
            UpdateMaterialData(name, texture.BindlessIndex);
        }

        private Dictionary<string, uint> _bindlessIndices = new();

        public void SetTextureIndex(string name, uint index)
        {
            _bindlessIndices[name] = index;
        }
        
        /// <summary>
        /// Update this material's data in the global MaterialsBuffer when textures change
        /// </summary>
        private void UpdateMaterialData(string name, uint bindlessIdx)
        {
            lock (_materialIdLock)
            {
                if (MaterialID < 0 || MaterialID >= _allMaterials.Count) return;
                
                var data = _allMaterials[MaterialID];
                switch (name)
                {
                    case "Albedo":
                    case "AlbedoTex":
                        data.AlbedoIdx = bindlessIdx;
                        break;
                    case "Normal":
                    case "NormalTex":
                        data.NormalIdx = bindlessIdx;
                        break;
                    case "Roughness":
                        data.RoughnessIdx = bindlessIdx;
                        break;
                    case "Metallic":
                        data.MetallicIdx = bindlessIdx;
                        break;
                    case "Emissive":
                        data.EmissiveIdx = bindlessIdx;
                        break;
                    case "AO":
                        data.AOIdx = bindlessIdx;
                        break;
                }
                _allMaterials[MaterialID] = data;
                _materialsBufferDirty = true;
            }
        }



        public void Apply(CommandList commandList, GraphicsDevice device, MaterialBlock? block = null)
        {
             Apply(commandList.Native, device, block);
        }

        public void Apply(ID3D12GraphicsCommandList commandList, GraphicsDevice device, MaterialBlock? block = null)
        {
            // Apply Material Block Overrides if present
            block?.Apply(this);

            // Bind Pipeline - use selected pass's PSO, wireframe variant if enabled
            var pso = _passPipelineStates.TryGetValue(Pass, out var selectedPso) ? selectedPso : _passPipelineStates[0];
            if (Engine.Settings.TerrainWireframe && !_isFullscreenPass && _passWireframePSOs.TryGetValue(Pass, out var wireframePso))
            {
                pso = wireframePso;
            }
            commandList.SetPipelineState(pso.Native);
            commandList.SetGraphicsRootSignature(device.GlobalRootSignature);

            // Apply MaterialBlock and commit all constant buffers (Apex pattern)
            var effectBlock = Effect!.GetMaterialBlock();
            foreach (var cb in _constantBuffers.Values)
            {
                if (cb.Slot >= 0)
                {
                    // 1. Apply global MasterEffect parameters (CameraInverse, etc.)
                    effectBlock.Apply(cb);
                    // 2. Apply per-draw overrides AFTER MasterEffect (Cascades, LightSpaces, etc.)
                    //    This ensures per-draw values take priority over global defaults
                    block?.Apply(cb);
                    cb.Commit();
                    commandList.SetGraphicsRootConstantBufferView((uint)cb.Slot, cb.GpuAddress);
                }
            }

            // Keep block applied to Material for property access
            block?.Apply(this);

            // Data-driven push constants from shader resource bindings
            // Iterate bindings parsed from shader #define patterns
            foreach (var binding in Effect.ResourceBindings)
            {
                uint index = 0;
                
                // Try explicit bindless indices first (set via SetTextureIndex)
                if (_bindlessIndices.TryGetValue(binding.Name, out var bindlessIdx))
                {
                    index = bindlessIdx;
                }
                // Then try textures (set via SetTexture)
                else if (_textures.TryGetValue(binding.Name, out var tex))
                {
                    index = tex.BindlessIndex;
                }
                
                commandList.SetGraphicsRoot32BitConstant(0, index, (uint)binding.Slot);
            }
            
            // Slot 14: Materials buffer bindless index (for GET_MATERIAL macro)
            // Slots 2-13 are used by mesh rendering, so Materials uses 14 to avoid collision
            commandList.SetGraphicsRoot32BitConstant(0, _materialsBufferBindlessIndex, 14);
            
            // NOTE: SetDescriptorHeaps is now called ONCE per frame in RenderView.Prepare
            // Calling it here per-draw was extremely expensive
        }
        
        /// <summary>
        /// Bind the shared MaterialsBuffer to root slot. Call once per frame after materials are set up.
        /// Returns the bindless index for the Materials buffer (to be set as push constant slot 10).
        /// </summary>
        private static uint _materialsBufferBindlessIndex = 0;
        private static int _materialsBufferCapacity = 0;
        
        public static uint BindMaterialsBuffer(ID3D12GraphicsCommandList commandList, GraphicsDevice device)
        {
            // Take snapshot under lock to avoid racing with background Material creation
            MaterialData[] snapshot;
            int materialCount;
            bool dirty;
            lock (_materialIdLock)
            {
                materialCount = _allMaterials.Count;
                dirty = _materialsBufferDirty;
                snapshot = _allMaterials.ToArray();
            }
            
            if (materialCount == 0) return 0;
            
            int bufferSize = materialCount * System.Runtime.InteropServices.Marshal.SizeOf<MaterialData>();
            
            // Create or recreate buffer only if we need more capacity
            bool needsRecreate = _materialsBuffer == null || materialCount > _materialsBufferCapacity;
            
            if (needsRecreate)
            {
                // Use power of 2 growth strategy for capacity
                int newCapacity = Math.Max(32, materialCount);
                newCapacity = (int)Math.Pow(2, Math.Ceiling(Math.Log2(newCapacity)));
                
                int newBufferSize = newCapacity * System.Runtime.InteropServices.Marshal.SizeOf<MaterialData>();
                
                _materialsBuffer?.Dispose();
                _materialsBuffer = device.NativeDevice.CreateCommittedResource(
                    new HeapProperties(HeapType.Upload),
                    HeapFlags.None,
                    ResourceDescription.Buffer((ulong)newBufferSize),
                    ResourceStates.GenericRead);
                
                _materialsBufferCapacity = newCapacity;
                
                unsafe
                {
                    void* ptr;
                    _materialsBuffer.Map(0, null, &ptr);
                    _materialsBufferPtr = (IntPtr)ptr;
                }
                
                // Allocate bindless index and create SRV for materials buffer
                if (_materialsBufferBindlessIndex == 0)
                {
                    _materialsBufferBindlessIndex = device.AllocateBindlessIndex();
                }
                device.CreateStructuredBufferSRV(
                    _materialsBuffer,
                    (uint)newCapacity,
                    (uint)System.Runtime.InteropServices.Marshal.SizeOf<MaterialData>(),
                    _materialsBufferBindlessIndex);
                    
                // Force data update after recreate
                dirty = true;
            }
            
            // Update buffer data if dirty
            if (dirty)
            {
                unsafe
                {
                    fixed (MaterialData* src = snapshot)
                    {
                        Buffer.MemoryCopy(src, (void*)_materialsBufferPtr, bufferSize, bufferSize);
                    }
                }
                lock (_materialIdLock) { _materialsBufferDirty = false; }
            }
            
            return _materialsBufferBindlessIndex;
        }
        
        /// <summary>
        /// Get the bindless index for the Materials buffer (for push constant slot 10).
        /// </summary>
        public static uint MaterialsBufferIndex => _materialsBufferBindlessIndex;

        public void Dispose()
        {
            foreach (var cb in _constantBuffers.Values)
                cb.Dispose();
        }
        

    }

    public interface IParameterTarget
    {
        void SetParameter<T>(string name, T value) where T : unmanaged;
        void SetTexture(string name, Texture texture);
    }
    
    // Extensions to make Material implement IParameterTarget explicitly or implicitly
    // Since Material already has SetParameter/SetTexture, we can align it easily.
    
    public abstract class ParameterValue
    {
        public string Name = string.Empty;
        /// <summary>
        /// Push constant slot for per-instance GPU buffer binding.
        /// -1 = not yet resolved. Resolved automatically at enqueue time from shader resource bindings.
        /// </summary>
        public int PushConstantSlot = -1;
        public abstract void SetToTarget(Material target);
        public abstract void SetToTarget(ConstantBuffer cb);
        
        /// <summary>Copy raw bytes of the value into dest at the given offset. Returns bytes written.</summary>
        public virtual int CopyToStaging(byte[] dest, int offset) => 0;
        /// <summary>Number of array elements (for ArrayParameterValue). 0 for scalars/textures.</summary>
        public virtual int GetElementCount() => 0;
        /// <summary>Byte stride per array element. 0 for scalars/textures.</summary>
        public virtual int GetElementStride() => 0;
    }

    public sealed class ParameterValue<T> : ParameterValue where T : unmanaged
    {
        public T Value;
        public override void SetToTarget(Material target) => target.SetParameter(Name, Value);
        public override void SetToTarget(ConstantBuffer cb) => cb.SetParameter(Name, Value);
        
        public override int GetElementCount() => 1;
        public override int GetElementStride() => System.Runtime.InteropServices.Marshal.SizeOf<T>();
        public override int CopyToStaging(byte[] dest, int offset)
        {
            int size = System.Runtime.InteropServices.Marshal.SizeOf<T>();
            unsafe
            {
                fixed (T* ptr = &Value)
                {
                    System.Runtime.InteropServices.Marshal.Copy((IntPtr)ptr, dest, offset, size);
                }
            }
            return size;
        }
    }
    
    // For object types like Texture (not unmanaged)
    public sealed class TextureParameterValue : ParameterValue
    {
        public Texture Value = null!;
        public override void SetToTarget(Material target) => target.SetTexture(Name, Value);
        public override void SetToTarget(ConstantBuffer cb) { } // Textures don't apply to CBs
    }
    
    // For array types like Matrix4x4[], Vector4[]
    public sealed class ArrayParameterValue<T> : ParameterValue where T : unmanaged
    {
        public T[] Value = null!;
        public override void SetToTarget(Material target) { } // Arrays don't apply to materials directly
        public override void SetToTarget(ConstantBuffer cb) => cb.SetParameterArray(Name, Value);
        
        public override int GetElementCount() => Value?.Length ?? 0;
        public override int GetElementStride() => System.Runtime.InteropServices.Marshal.SizeOf<T>();
        public override int CopyToStaging(byte[] dest, int offset)
        {
            if (Value == null || Value.Length == 0) return 0;
            var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(Value.AsSpan());
            bytes.CopyTo(dest.AsSpan(offset));
            return bytes.Length;
        }
    }

    public class MaterialBlock
    {
        private Dictionary<int, ParameterValue> _parameters = new();

        public Dictionary<int, ParameterValue> Parameters => _parameters;

        public void Clear()
        {
            _parameters.Clear();
        }

        public void SetParameter<T>(string name, T value) where T : unmanaged
        {
            int hash = name.GetHashCode();
            if (!_parameters.TryGetValue(hash, out var param))
            {
                param = new ParameterValue<T> { Name = name, Value = value };
                _parameters.Add(hash, param);
            }
            else if (param is ParameterValue<T> typedParam)
            {
                typedParam.Value = value;
            }
        }

        public void SetTexture(string name, Texture value)
        {
            int hash = name.GetHashCode();
            if (!_parameters.TryGetValue(hash, out var param))
            {
                param = new TextureParameterValue { Name = name, Value = value };
                _parameters.Add(hash, param);
            }
            else if (param is TextureParameterValue typedParam)
            {
                typedParam.Value = value;
            }
        }
        
        // Support for unmanaged array types like Matrix4x4[], Vector4[]
        public void SetParameterArray<T>(string name, T[] value) where T : unmanaged
        {
            int hash = name.GetHashCode();
            if (!_parameters.TryGetValue(hash, out var param))
            {
                param = new ArrayParameterValue<T> { Name = name, Value = value };
                _parameters.Add(hash, param);
            }
            else if (param is ArrayParameterValue<T> typedParam)
            {
                typedParam.Value = value;
            }
        }
        
        public T? GetValue<T>(string name)
        {
            return GetValue<T>(name.GetHashCode());
        }
        
        public T? GetValue<T>(int hash)
        {
            if (_parameters.TryGetValue(hash, out var param))
            {
                // Use reflection to get the Value field
                var valueField = param.GetType().GetField("Value");
                if (valueField != null)
                {
                    var value = valueField.GetValue(param);
                    if (value is T result)
                        return result;
                }
            }
            return default;
        }
        
        public T[]? GetArrayValue<T>(int hash) where T : unmanaged
        {
            if (_parameters.TryGetValue(hash, out var param) && param is ArrayParameterValue<T> typed)
                return typed.Value;
            return null;
        }
        
        public bool TryGetParameter<T>(string name, out T value) where T : unmanaged
        {
            int hash = name.GetHashCode();
            if (_parameters.TryGetValue(hash, out var param) && param is ParameterValue<T> typedParam)
            {
                value = typedParam.Value;
                return true;
            }
            value = default;
            return false;
        }

        public void Apply(Material target)
        {
            foreach (var param in _parameters.Values)
                param.SetToTarget(target);
        }
        
        /// <summary>
        /// Apply all parameters to a ConstantBuffer (Apex pattern)
        /// </summary>
        public void Apply(ConstantBuffer cb)
        {
            foreach (var param in _parameters.Values)
                param.SetToTarget(cb);
        }
    }
}
