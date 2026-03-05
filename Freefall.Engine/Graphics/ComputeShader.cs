using System;
using System.IO;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// Wraps a compute pipeline state with push constant management and dispatch helpers.
    /// Compiles from HLSL source file on first use (lazy PSO creation).
    /// </summary>
    public class ComputeShader : IDisposable
    {
        private readonly string _filename;
        private readonly string _entryPoint;
        private ID3D12PipelineState? _pso;
        private readonly uint[] _constants = new uint[32]; // Match root signature slot 0 capacity
        private bool _disposed;

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

        // ────────────── Push Constants ──────────────

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

        // ────────────── Dispatch ──────────────

        /// <summary>
        /// Compile PSO (if needed), bind root signature + descriptor heap + push constants, and dispatch.
        /// </summary>
        public void Dispatch(ID3D12GraphicsCommandList cmd, uint groupsX, uint groupsY = 1, uint groupsZ = 1)
        {
            EnsureCompiled();

            var device = Engine.Device;

            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            cmd.SetPipelineState(_pso!);

            // Upload all push constants
            for (int i = 0; i < _constants.Length; i++)
            {
                if (_constants[i] != 0)
                    cmd.SetComputeRoot32BitConstant(0, _constants[i], (uint)i);
            }

            cmd.Dispatch(groupsX, groupsY, groupsZ);
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
            shader.Dispose();

            Debug.Log("[ComputeShader]", $"Compiled: {_filename} ({_entryPoint})");
        }

        // ────────────── Dispose ──────────────

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _pso?.Dispose();
        }
    }
}
