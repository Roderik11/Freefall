using System;
using System.Collections.Generic;
using System.IO;
using Vortice.Direct3D12;
using Vortice.Direct3D12.Shader;

namespace Freefall.Graphics
{
    /// <summary>
    /// Compute pipeline with multi-kernel support and Effect-like named parameters.
    /// A single ComputeShader can hold multiple kernels (entry points) from the same HLSL file.
    /// Push constant slot mappings are file-level (from #define XxxIdx GET_INDEX(N)).
    /// Each kernel has its own constant values array.
    /// Use SetKernel to select the active kernel, then Set/Dispatch operate on it.
    /// </summary>
    public class ComputeShader : IDisposable
    {
        private readonly string _filename;
        private readonly string? _defaultEntryPoint;
        private bool _disposed;
        private bool _compiled;

        // Per-kernel data
        private struct Kernel
        {
            public string Name;
            public ID3D12PipelineState PSO;
            public uint[] Constants;
        }

        private readonly List<Kernel> _kernels = new();
        private int _activeKernel;

        // File-level: named parameter → push constant slot (from FXParser.ParseResourceBindings)
        private Dictionary<int, int> _resourceSlots = new();
        private List<ShaderResourceBinding> _resourceBindings = new();

        // Constant buffers discovered from shader reflection
        private Dictionary<string, ConstantBuffer> _constantBuffers = new();

        // Cached source and base path for lazy kernel compilation
        private string? _cachedSource;
        private string? _basePath;

        /// <summary>
        /// Create a compute shader from an HLSL file.
        /// Kernels are auto-discovered from #pragma kernel directives.
        /// </summary>
        public ComputeShader(string filename)
        {
            _filename = filename;
            _defaultEntryPoint = null;
        }

        /// <summary>
        /// Create a single-kernel compute shader (convenience).
        /// The entry point is used as fallback if no #pragma kernel directives exist.
        /// </summary>
        public ComputeShader(string filename, string entryPoint)
        {
            _filename = filename;
            _defaultEntryPoint = entryPoint;
        }

        // ────────────── Kernel Management ──────────────

        /// <summary>
        /// Look up a kernel by entry point name. Returns kernel index.
        /// All kernels declared via #pragma kernel are auto-compiled on load.
        /// </summary>
        public int FindKernel(string entryPoint)
        {
            EnsureSourceLoaded();

            for (int i = 0; i < _kernels.Count; i++)
            {
                if (_kernels[i].Name == entryPoint)
                    return i;
            }

            throw new KeyNotFoundException($"[{_filename}] Kernel '{entryPoint}' not found. Declared kernels: {string.Join(", ", GetKernelNames())}");
        }

        /// <summary>
        /// Set the active kernel by name. All subsequent Set/Dispatch calls operate on this kernel.
        /// </summary>
        public void SetKernel(string entryPoint)
        {
            _activeKernel = FindKernel(entryPoint);
        }

        /// <summary>
        /// Set the active kernel by index. All subsequent Set/Dispatch calls operate on this kernel.
        /// </summary>
        public void SetKernel(int kernel)
        {
            EnsureSourceLoaded();
            if (kernel < 0 || kernel >= _kernels.Count)
                throw new ArgumentOutOfRangeException(nameof(kernel));
            _activeKernel = kernel;
        }

        /// <summary>Get all compiled kernel names.</summary>
        public IEnumerable<string> GetKernelNames()
        {
            EnsureSourceLoaded();
            foreach (var k in _kernels)
                yield return k.Name;
        }

        /// <summary>Get the PSO for a specific kernel (for manual pipeline binding).</summary>
        public ID3D12PipelineState GetPSO(int kernel) => _kernels[kernel].PSO;

        // ────────────── Push Constants (operate on active kernel) ──────────────

        /// <summary>Set a uint push constant by name on the active kernel.</summary>
        public void Set(string name, uint value)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name);
            if (slot >= 0) _kernels[_activeKernel].Constants[slot] = value;
        }

        /// <summary>Set a float push constant by name on the active kernel.</summary>
        public void Set(string name, float value)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name);
            if (slot >= 0) _kernels[_activeKernel].Constants[slot] = BitConverter.SingleToUInt32Bits(value);
        }

        /// <summary>Set a Texture's bindless SRV index on the active kernel.</summary>
        public void Set(string name, Texture texture)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name);
            if (slot >= 0) _kernels[_activeKernel].Constants[slot] = texture.BindlessIndex;
        }

        /// <summary>Set a GraphicsBuffer's SRV index on the active kernel.</summary>
        public void SetSRV(string name, GraphicsBuffer buffer)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name);
            if (slot >= 0) _kernels[_activeKernel].Constants[slot] = buffer.SrvIndex;
        }

        /// <summary>Set a GraphicsBuffer's UAV index on the active kernel.</summary>
        public void SetUAV(string name, GraphicsBuffer buffer)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name);
            if (slot >= 0) _kernels[_activeKernel].Constants[slot] = buffer.UavIndex;
        }

        // ────────────── Constant Buffer Parameters (shared across all kernels) ──────────────

        /// <summary>Set a constant buffer parameter by name.</summary>
        public void SetParam<T>(string name, T value) where T : unmanaged
        {
            EnsureSourceLoaded();
            foreach (var cb in _constantBuffers.Values)
                cb.SetParameter(name, value);
        }

        /// <summary>Set a constant buffer array parameter by name.</summary>
        public void SetParamArray<T>(string name, T[] values) where T : unmanaged
        {
            EnsureSourceLoaded();
            foreach (var cb in _constantBuffers.Values)
                cb.SetParameterArray(name, values);
        }

        // ────────────── Dispatch (operates on active kernel) ──────────────

        /// <summary>
        /// Prepare the compute pipeline. Binds root signature and descriptor heaps.
        /// Call once before a sequence of SetKernel/Set/Dispatch calls.
        /// External cbuffer bindings (e.g., frustum planes) should be done AFTER Begin.
        /// </summary>
        public void Begin(ID3D12GraphicsCommandList cmd)
        {
            EnsureSourceLoaded();
            var device = Engine.Device;
            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
        }

        /// <summary>
        /// Dispatch the active kernel. Sets PSO and pushes constants.
        /// Call Begin() once before the first Dispatch in a sequence.
        /// </summary>
        public void Dispatch(ID3D12GraphicsCommandList cmd, uint groupsX, uint groupsY = 1, uint groupsZ = 1)
        {
            EnsureSourceLoaded();
            if (_kernels.Count == 0)
                throw new InvalidOperationException($"No kernels compiled for {_filename}.");

            var k = _kernels[_activeKernel];

            cmd.SetPipelineState(k.PSO);

            // Upload push constants for the active kernel
            for (int i = 0; i < k.Constants.Length; i++)
            {
                if (k.Constants[i] != 0)
                    cmd.SetComputeRoot32BitConstant(0, k.Constants[i], (uint)i);
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

        // ────────────── Internal ──────────────

        private int ResolveSlot(string name)
        {
            int hash = name.GetHashCode();
            if (_resourceSlots.TryGetValue(hash, out int slot))
                return slot;

            Debug.LogWarning("ComputeShader", $"[{_filename}] Unknown push constant: '{name}'");
            return -1;
        }

        private void EnsureSourceLoaded()
        {
            if (_compiled) return;
            _compiled = true;

            _basePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Resources", "Shaders");
            string fullPath = Path.Combine(_basePath, _filename);
            _cachedSource = File.ReadAllText(fullPath);

            // Parse file-level resource bindings (same as Effect)
            _resourceBindings = FXParser.ParseResourceBindings(_cachedSource);
            foreach (var binding in _resourceBindings)
                _resourceSlots[binding.Name.GetHashCode()] = binding.Slot;

            // Discover kernels from #pragma kernel directives
            var kernelNames = KernelParser.ParseKernels(_cachedSource);

            // If no #pragma kernel found but constructor specified an entry point, use that
            if (kernelNames.Count == 0 && _defaultEntryPoint != null)
                kernelNames.Add(_defaultEntryPoint);

            // Compile all kernels (like Effect compiles all passes)
            foreach (var name in kernelNames)
                CompileKernel(name);

            Debug.Log("[ComputeShader]", $"Loaded {_filename}: {_kernels.Count} kernels, {_resourceBindings.Count} bindings");
        }

        private void CompileKernel(string entryPoint)
        {
            var shader = new Shader(_cachedSource!, entryPoint, "cs_6_6", _basePath);
            var pso = Engine.Device.CreateComputePipelineState(shader.Bytecode);

            // Discover cbuffers from first kernel's reflection (shared across kernels)
            if (_kernels.Count == 0 && shader.Reflection != null)
                DiscoverConstantBuffers(shader.Reflection);

            shader.Dispose();

            int index = _kernels.Count;
            _kernels.Add(new Kernel
            {
                Name = entryPoint,
                PSO = pso,
                Constants = new uint[32]
            });

            Debug.Log("[ComputeShader]", $"  Compiled kernel [{index}]: {entryPoint}");
        }

        private void DiscoverConstantBuffers(ID3D12ShaderReflection reflection)
        {
            var device = Engine.Device;
            var desc = reflection.Description;
            for (uint i = 0; i < desc.ConstantBuffers; i++)
            {
                var cbReflection = reflection.GetConstantBufferByIndex(i);
                var cbDesc = cbReflection.Description;

                // Skip PushConstants (root parameter 0)
                if (cbDesc.Name == "PushConstants") continue;

                if (!_constantBuffers.ContainsKey(cbDesc.Name))
                {
                    var cb = new ConstantBuffer(device, cbReflection);
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

        // ────────────── Dispose ──────────────

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var k in _kernels)
                k.PSO?.Dispose();
            foreach (var cb in _constantBuffers.Values)
                cb.Dispose();
        }
    }
}
