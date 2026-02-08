using System;
using System.Collections.Generic;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// A batch of draws sharing the same PSO, rendered via ExecuteIndirect.
    /// </summary>
    public class DrawBatch : IDisposable
    {
        public PipelineState PSO { get; }
        public Material Material { get; }
        
        /// <summary>GPU buffer containing IndirectDrawCommand structs</summary>
        public ID3D12Resource? IndirectArgsBuffer { get; private set; }
        
        /// <summary>GPU buffer containing atomic draw count (written by compute shader)</summary>
        public ID3D12Resource? DrawCountBuffer { get; private set; }
        
        /// <summary>Maximum number of draws this batch can hold</summary>
        public int MaxDrawCount { get; private set; }
        
        /// <summary>Current number of registered draws (before culling)</summary>
        public int RegisteredDrawCount => _draws.Count;
        
        private readonly List<DrawInstance> _draws = new();
        private bool _buffersDirty = true;
        
        public DrawBatch(PipelineState pso, Material material)
        {
            PSO = pso;
            Material = material;
        }
        
        /// <summary>
        /// Register a draw instance for this batch.
        /// </summary>
        public void AddDraw(DrawInstance draw)
        {
            _draws.Add(draw);
            _buffersDirty = true;
        }
        
        /// <summary>
        /// Clear all registered draws (call at start of frame).
        /// </summary>
        public void Clear()
        {
            _draws.Clear();
            _buffersDirty = true;
        }
        
        /// <summary>
        /// Get all registered draws for upload to GPU instance buffer.
        /// </summary>
        public ReadOnlySpan<DrawInstance> GetDraws() => _draws.ToArray();
        
        /// <summary>
        /// Allocate/resize GPU buffers if needed.
        /// </summary>
        public void EnsureBuffers(GraphicsDevice device, int capacity)
        {
            if (MaxDrawCount >= capacity && IndirectArgsBuffer != null)
                return;
                
            // Dispose old buffers
            IndirectArgsBuffer?.Dispose();
            DrawCountBuffer?.Dispose();
            
            MaxDrawCount = Math.Max(capacity, 1024); // Minimum 1024 draws
            
            // Indirect args buffer (UAV for compute, indirect args for draw)
            var argsSize = (ulong)(MaxDrawCount * IndirectDrawSizes.IndirectCommandSize);
            IndirectArgsBuffer = device.NativeDevice.CreateCommittedResource(
                HeapType.Default,
                ResourceDescription.Buffer(argsSize, ResourceFlags.AllowUnorderedAccess),
                ResourceStates.Common
            );
            
            // Draw count buffer (4 bytes, atomic counter)
            DrawCountBuffer = device.NativeDevice.CreateCommittedResource(
                HeapType.Default,
                ResourceDescription.Buffer(4, ResourceFlags.AllowUnorderedAccess),
                ResourceStates.Common
            );
            
            _buffersDirty = false;
        }
        
        public void Dispose()
        {
            IndirectArgsBuffer?.Dispose();
            DrawCountBuffer?.Dispose();
        }
    }
}
