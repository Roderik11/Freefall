using System;
using System.Collections.Generic;
using System.IO;
using Vortice.Direct3D12;
using Vortice.Direct3D12.Shader;

namespace Freefall.Graphics
{
    /// <summary>
    /// Compute pipeline with multi-kernel support and per-kernel push constant parameters.
    /// Each kernel has its own parameter set (like EffectParameter / MaterialBlock).
    /// Push constant slot mappings are file-level (from #define XxxIdx GET_INDEX(N)).
    /// Use Set(kernel, name, value) to write per-kernel params, Dispatch(kernel, ...) to execute.
    /// </summary>
    public class ComputeShader : IDisposable
    {
        private readonly string _filename;
        private readonly string? _defaultEntryPoint;
        private bool _disposed;
        private bool _compiled;

        private const int MaxPushConstants = 32;

        // Per-kernel data
        private struct Kernel
        {
            public string Name;
            public ID3D12PipelineState PSO;
            public uint[] Constants;   // Per-kernel push constant values (32 slots)
        }

        private readonly List<Kernel> _kernels = new();

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

            throw new InvalidOperationException(
                $"Kernel '{entryPoint}' not found in {_filename}. " +
                $"Available: {string.Join(", ", GetKernelNames())}");
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

        // ────────────── Unified SetParam (push constant → cbuffer fallback) ──────────────

        /// <summary>Set a uint value by name on a specific kernel. Tries push constants first, then cbuffer.</summary>
        public void SetParam(int kernel, string name, uint value)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name, silent: true);
            if (slot >= 0) _kernels[kernel].Constants[slot] = value;
            else SetCBParam(name, value);
        }

        /// <summary>Set a float value by name on a specific kernel. Tries push constants first, then cbuffer.</summary>
        public void SetParam(int kernel, string name, float value)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name, silent: true);
            if (slot >= 0) _kernels[kernel].Constants[slot] = BitConverter.SingleToUInt32Bits(value);
            else SetCBParam(name, value);
        }

        /// <summary>Set a Texture's bindless SRV index on a specific kernel.</summary>
        public void SetParam(int kernel, string name, Texture texture)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name);
            if (slot >= 0) _kernels[kernel].Constants[slot] = texture.BindlessIndex;
        }

        /// <summary>Set a GraphicsBuffer's SRV index on a specific kernel.</summary>
        public void SetSRV(int kernel, string name, GraphicsBuffer buffer)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name);
            if (slot >= 0) _kernels[kernel].Constants[slot] = buffer.SrvIndex;
        }

        /// <summary>Set a GraphicsBuffer's UAV index on a specific kernel.</summary>
        public void SetUAV(int kernel, string name, GraphicsBuffer buffer)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name);
            if (slot >= 0) _kernels[kernel].Constants[slot] = buffer.UavIndex;
        }

        // ────────────── SetParam on ALL kernels ──────────────

        /// <summary>Set a uint value on ALL kernels. Tries push constants first, then cbuffer.</summary>
        public void SetParam(string name, uint value)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name, silent: true);
            if (slot >= 0)
                for (int i = 0; i < _kernels.Count; i++)
                    _kernels[i].Constants[slot] = value;
            else SetCBParam(name, value);
        }

        /// <summary>Set a float value on ALL kernels. Tries push constants first, then cbuffer.</summary>
        public void SetParam(string name, float value)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name, silent: true);
            if (slot >= 0)
            {
                uint bits = BitConverter.SingleToUInt32Bits(value);
                for (int i = 0; i < _kernels.Count; i++)
                    _kernels[i].Constants[slot] = bits;
            }
            else SetCBParam(name, value);
        }

        /// <summary>Set a Texture's bindless SRV index on ALL kernels.</summary>
        public void SetParam(string name, Texture texture)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name);
            if (slot >= 0)
                for (int i = 0; i < _kernels.Count; i++)
                    _kernels[i].Constants[slot] = texture.BindlessIndex;
        }

        /// <summary>Set a GraphicsBuffer's SRV index on ALL kernels.</summary>
        public void SetSRV(string name, GraphicsBuffer buffer)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name);
            if (slot >= 0)
                for (int i = 0; i < _kernels.Count; i++)
                    _kernels[i].Constants[slot] = buffer.SrvIndex;
        }

        /// <summary>Set a GraphicsBuffer's UAV index on ALL kernels.</summary>
        public void SetUAV(string name, GraphicsBuffer buffer)
        {
            EnsureSourceLoaded();
            int slot = ResolveSlot(name);
            if (slot >= 0)
                for (int i = 0; i < _kernels.Count; i++)
                    _kernels[i].Constants[slot] = buffer.UavIndex;
        }

        /// <summary>Set a typed value by name (vectors, structs, etc.). Targets cbuffers.</summary>
        public void SetParam<T>(string name, T value) where T : unmanaged
        {
            EnsureSourceLoaded();
            SetCBParam(name, value);
        }

        /// <summary>Set a constant buffer array parameter by name.</summary>
        public void SetArray<T>(string name, T[] values) where T : unmanaged
        {
            EnsureSourceLoaded();
            foreach (var cb in _constantBuffers.Values)
                cb.SetParameterArray(name, values);
        }

        private void SetCBParam<T>(string name, T value) where T : unmanaged
        {
            foreach (var cb in _constantBuffers.Values)
                cb.SetParameter(name, value);
        }

        // ────────────── Dispatch (explicit kernel index) ──────────────

        /// <summary>
        /// Dispatch a specific kernel. Binds root signature, descriptor heaps,
        /// PSO, pushes ALL push constants from that kernel's parameter set,
        /// binds constant buffers, and dispatches.
        /// </summary>
        public void Dispatch(int kernel, ID3D12GraphicsCommandList cmd, uint groupsX, uint groupsY = 1, uint groupsZ = 1)
        {
            EnsureSourceLoaded();
            if (kernel < 0 || kernel >= _kernels.Count)
                throw new ArgumentOutOfRangeException(nameof(kernel));

            // Set PSO for this kernel (caller handles root sig + descriptor heaps + cbuffers)
            var k = _kernels[kernel];
            cmd.SetPipelineState(k.PSO);

            // Push ALL constants unconditionally from this kernel's array
            for (int i = 0; i < MaxPushConstants; i++)
                cmd.SetComputeRoot32BitConstant(0, k.Constants[i], (uint)i);

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

        // ────────────── Slot Resolution ──────────────

        private int ResolveSlot(string name, bool silent = false)
        {
            int hash = name.GetHashCode();
            if (_resourceSlots.TryGetValue(hash, out int slot))
                return slot;

            if (!silent)
                Debug.LogAlways($"[ComputeShader] Warning: parameter '{name}' not found in {_filename}");
            return -1;
        }

        // ────────────── Lazy Compilation ──────────────

        private void EnsureSourceLoaded()
        {
            if (_compiled) return;
            _compiled = true;

            // Resolve shader path
            string resourcesPath = Path.Combine(AppContext.BaseDirectory, "Resources", "Shaders");
            string fullPath = Path.Combine(resourcesPath, _filename);

            if (!File.Exists(fullPath))
            {
                fullPath = Path.Combine(AppContext.BaseDirectory, _filename);
                if (!File.Exists(fullPath))
                    throw new FileNotFoundException($"Compute shader not found: {_filename}");
            }

            _basePath = Path.GetDirectoryName(fullPath);
            _cachedSource = File.ReadAllText(fullPath);

            // Parse resource bindings (file-level: #define XxxIdx GET_INDEX(N))
            _resourceBindings = FXParser.ParseResourceBindings(_cachedSource);
            foreach (var binding in _resourceBindings)
                _resourceSlots[binding.Name.GetHashCode()] = binding.Slot;

            // Discover and compile kernels from #pragma kernel directives
            var kernelNames = KernelParser.ParseKernels(_cachedSource);

            if (kernelNames.Count == 0 && _defaultEntryPoint != null)
                kernelNames.Add(_defaultEntryPoint);

            foreach (var name in kernelNames)
                CompileKernel(name);

            _cachedSource = null;  // Free source after all kernels compiled
        }

        private void CompileKernel(string entryPoint)
        {
            var device = Engine.Device;
            var shader = new Shader(_cachedSource!, entryPoint, "cs_6_6");

            var pso = device.CreateComputePipelineState(shader.Bytecode);

            // Discover constant buffers from first kernel's reflection
            if (_kernels.Count == 0 && shader.Reflection != null)
                DiscoverConstantBuffers(shader.Reflection);

            _kernels.Add(new Kernel
            {
                Name = entryPoint,
                PSO = pso,
                Constants = new uint[MaxPushConstants]
            });

            shader.Dispose();
        }

        private void DiscoverConstantBuffers(ID3D12ShaderReflection reflection)
        {
            var device = Engine.Device;

            for (uint i = 0; i < reflection.Description.ConstantBuffers; i++)
            {
                var cbReflection = reflection.GetConstantBufferByIndex(i);
                if (cbReflection.Description.Name == "PushConstants") continue;

                var bindDesc = reflection.GetResourceBindingDescByName(cbReflection.Description.Name);

                // Skip cbuffers at b0-b2 — these are externally managed by the caller
                // (e.g. TerrainRenderer binds frustum planes, Hi-Z params, terrain params)
                if (bindDesc.BindPoint <= 2) continue;

                int rootSlot = bindDesc.BindPoint switch
                {
                    4 => 4,  // b4 → root slot 4 (TextureIndices / custom cbuffers)
                    _ => -1
                };

                var cb = new ConstantBuffer(device, cbReflection);
                cb.Slot = rootSlot;
                _constantBuffers[cbReflection.Description.Name] = cb;
            }
        }

        // ────────────── Dispose ──────────────

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            foreach (var k in _kernels)
                k.PSO?.Dispose();
            _kernels.Clear();

            foreach (var cb in _constantBuffers.Values)
                cb.Dispose();
            _constantBuffers.Clear();
        }
    }
}
