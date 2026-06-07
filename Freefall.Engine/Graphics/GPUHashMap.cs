using System;
using Vortice.Direct3D12;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>
    /// GPU hash map with open addressing for sparse radiance cascade tile storage.
    /// Each entry is 8 bytes: uint key (packed grid coords + level) + uint poolIndex.
    /// Empty sentinel = 0xFFFFFFFF in key.
    /// </summary>
    public class GPUHashMap : IDisposable
    {
        private readonly GraphicsBuffer _entries;
        private readonly GraphicsBuffer _counter;
        private bool _disposed;

        public uint EntriesSrvIndex => _entries.SrvIndex;
        public uint EntriesUavIndex => _entries.UavIndex;
        public uint CounterUavIndex => _counter.UavIndex;
        public int Capacity { get; }
        public uint CapacityMask => (uint)(Capacity - 1);

        private GPUHashMap(GraphicsBuffer entries, GraphicsBuffer counter, int capacity)
        {
            _entries = entries;
            _counter = counter;
            Capacity = capacity;
        }

        /// <summary>
        /// Create a GPU hash map with the given capacity (must be power of two).
        /// </summary>
        public static GPUHashMap Create(int capacityPow2)
        {
            if (capacityPow2 <= 0 || (capacityPow2 & (capacityPow2 - 1)) != 0)
                throw new ArgumentException("Capacity must be a power of two.", nameof(capacityPow2));

            var entries = GraphicsBuffer.CreateStructured(capacityPow2, 8, srv: true, uav: true);
            var counter = GraphicsBuffer.CreateRaw(1, uav: true, clearable: true);

            return new GPUHashMap(entries, counter, capacityPow2);
        }

        /// <summary>
        /// Clear all entries to empty sentinel (0xFFFFFFFF) and reset counter to zero.
        /// </summary>
        public void Clear(ID3D12GraphicsCommandList cmd)
        {
            _entries.Transition(cmd, ResourceStates.UnorderedAccess);
            _counter.Transition(cmd, ResourceStates.UnorderedAccess);

            _entries.ClearUAV(cmd, new Int4(unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF), unchecked((int)0xFFFFFFFF)));
            _counter.ClearUAV(cmd, new Int4(0, 0, 0, 0));

            _entries.UAVBarrier(cmd);
            _counter.UAVBarrier(cmd);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _entries?.Dispose();
            _counter?.Dispose();
        }
    }
}
