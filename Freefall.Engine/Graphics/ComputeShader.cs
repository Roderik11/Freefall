using System;
using System.Collections.Generic;
using System.IO;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// Compute pipeline with Effect-like named parameter binding.
    /// Parses push constant slots from #define XxxIdx GET_INDEX(N) patterns in the HLSL source,
    /// and discovers constant buffers from shader reflection — same infrastructure as Effect/Material.
    /// </summary>
    public class ComputeShader : IDisposable
    {
        private readonly string _filename;
        private readonly string _entryPoint;
        private ID3D12PipelineState? _pso;
        private readonly uint[] _constants = new uint[32]; // Root signature slot 0 capacity
        private bool _disposed;

        // Named parameter → push constant slot (from FXParser.ParseResourceBindings)
        private Dictionary<int, int> _resourceSlots = new();
        private List<ShaderResourceBinding> _resourceBindings = new();

        // Constant buffers discovered from shader reflection
        private Dictionary<string, ConstantBuffer> _constantBuffers = new();

        /// <summary>The compiled pipeline state object.</summary>
        public ID3D12PipelineState PSO => _pso ?? throw new InvalidOperationException("ComputeShader not yet compiled.");

        /// <summary>
        /// Create a compute shader from a file in the Shaders directory.
        /// PSO compilation is deferred until first Dispatch.
        /// </summary>
        /// <param name="filename">HLSL filename relative to Resources/Shaders (e.g. "decoration_prepass.hlsl")</param>
        /// <param name="entryPoint">Compute shader entry point name</param>
        public ComputeShader(string filename, string entryPoint)
        {
            _filename = filename;
            _entryPoint = entryPoint;
        }

        // ────────────── Named Push Constants ──────────────

        /// <summary>Set a uint push constant by name (resolved via #define XxxIdx GET_INDEX(N)).</summary>
        public void Set(string name, uint value)
        {
            EnsureCompiled();
            int slot = ResolveSlot(name);
            if (slot >= 0) _constants[slot] = value;
        }

        /// <summary>Set a float push constant by name.</summary>
        public void Set(string name, float value)
        {
            EnsureCompiled();
            int slot = ResolveSlot(name);
            if (slot >= 0) _constants[slot] = BitConverter.SingleToUInt32Bits(value);
        }

        /// <summary>Set a push constant to a Texture's bindless SRV index.</summary>
        public void Set(string name, Texture texture)
        {
            EnsureCompiled();
            int slot = ResolveSlot(name);
            if (slot >= 0) _constants[slot] = texture.BindlessIndex;
        }

        /// <summary>Set a push constant to a GraphicsBuffer's SRV index (for read access).</summary>
        public void SetSRV(string name, GraphicsBuffer buffer)
        {
            EnsureCompiled();
            int slot = ResolveSlot(name);
            if (slot >= 0) _constants[slot] = buffer.SrvIndex;
        }

        /// <summary>Set a push constant to a GraphicsBuffer's UAV index (for write access).</summary>
        public void SetUAV(string name, GraphicsBuffer buffer)
        {
            EnsureCompiled();
            int slot = ResolveSlot(name);
            if (slot >= 0) _constants[slot] = buffer.UavIndex;
        }

        // ────────────── Raw Push Constants (slot-based, fallback) ──────────────

        /// <summary>Set a uint push constant at the given slot.</summary>
        public void SetUint(int slot, uint value)
        {
            _constants[slot] = value;
        }

        /// <summary>Set a float push constant at the given slot.</summary>
        public void SetFloat(int slot, float value)
        {
            _constants[slot] = BitConverter.SingleToUInt32Bits(value);
        }

        /// <summary>Set a push constant from a GraphicsBuffer's SRV index.</summary>
        public void SetSRV(int slot, GraphicsBuffer buffer)
        {
            _constants[slot] = buffer.SrvIndex;
        }

        /// <summary>Set a push constant from a GraphicsBuffer's UAV index.</summary>
        public void SetUAV(int slot, GraphicsBuffer buffer)
        {
            _constants[slot] = buffer.UavIndex;
        }

        // ────────────── Constant Buffer Parameters ──────────────

        /// <summary>Set a constant buffer parameter by name (discovered from shader reflection).</summary>
        public void SetParam<T>(string name, T value) where T : unmanaged
        {
            EnsureCompiled();
            foreach (var cb in _constantBuffers.Values)
                cb.SetParameter(name, value);
        }

        /// <summary>Set a constant buffer array parameter by name.</summary>
        public void SetParamArray<T>(string name, T[] values) where T : unmanaged
        {
            EnsureCompiled();
            foreach (var cb in _constantBuffers.Values)
                cb.SetParameterArray(name, values);
        }

        // ────────────── Dispatch ──────────────

        /// <summary>
        /// Compile PSO (if needed), bind root signature + descriptor heap + push constants + cbuffers, and dispatch.
        /// </summary>
        public void Dispatch(ID3D12GraphicsCommandList cmd, uint groupsX, uint groupsY = 1, uint groupsZ = 1)
        {
            EnsureCompiled();

            var device = Engine.Device;

            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            cmd.SetPipelineState(_pso!);

            // Upload push constants
            for (int i = 0; i < _constants.Length; i++)
            {
                if (_constants[i] != 0)
                    cmd.SetComputeRoot32BitConstant(0, _constants[i], (uint)i);
            }

            // Bind constant buffers
            foreach (var cb in _constantBuffers.Values)
            {
                if (cb.Slot >= 0)
                {
                    cb.Commit();
                    cmd.SetComputeRootConstantBufferView((uint)cb.Slot, cb.GpuAddress);
                }
            }

            cmd.Dispatch(groupsX, groupsY, groupsZ);
        }

        // ────────────── Name Resolution ──────────────

        private int ResolveSlot(string name)
        {
            int hash = name.GetHashCode();
            if (_resourceSlots.TryGetValue(hash, out int slot))
                return slot;

            Debug.LogWarning("ComputeShader", $"[{_filename}] Unknown push constant: '{name}'");
            return -1;
        }

        // ────────────── Compilation ──────────────

        private void EnsureCompiled()
        {
            if (_pso != null) return;

            string basePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Resources", "Shaders");
            string fullPath = Path.Combine(basePath, _filename);
            string source = File.ReadAllText(fullPath);

            var shader = new Shader(source, _entryPoint, "cs_6_6", basePath);
            _pso = Engine.Device.CreateComputePipelineState(shader.Bytecode);

            // Parse push constant resource bindings (same as Effect)
            _resourceBindings = FXParser.ParseResourceBindings(source);
            foreach (var binding in _resourceBindings)
                _resourceSlots[binding.Name.GetHashCode()] = binding.Slot;

            // Discover constant buffers from shader reflection
            if (shader.Reflection != null)
            {
                var device = Engine.Device;
                var desc = shader.Reflection.Description;
                for (uint i = 0; i < desc.ConstantBuffers; i++)
                {
                    var cbReflection = shader.Reflection.GetConstantBufferByIndex(i);
                    var cbDesc = cbReflection.Description;

                    // Skip the PushConstants cbuffer (that's root parameter 0, not a real cbuffer)
                    if (cbDesc.Name == "PushConstants") continue;

                    if (!_constantBuffers.ContainsKey(cbDesc.Name))
                    {
                        var cb = new ConstantBuffer(device, cbReflection);

                        // Map cbuffer name to root signature slot (same convention as Material)
                        cb.Slot = cbDesc.Name switch
                        {
                            "SceneConstants" => 1,
                            "ObjectConstants" => 2,
                            "FrustumPlanes" => 1,
                            _ => -1
                        };

                        _constantBuffers.Add(cbDesc.Name, cb);
                        Debug.Log("[ComputeShader]", $"Discovered cbuffer '{cbDesc.Name}' -> slot {cb.Slot}");
                    }
                }
            }

            shader.Dispose();
            Debug.Log("[ComputeShader]", $"Compiled: {_filename} ({_entryPoint}), {_resourceBindings.Count} bindings, {_constantBuffers.Count} cbuffers");
        }

        // ────────────── Dispose ──────────────

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _pso?.Dispose();
            foreach (var cb in _constantBuffers.Values)
                cb.Dispose();
        }
    }
}
