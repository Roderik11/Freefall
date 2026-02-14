using System;
using System.Collections.Generic;

namespace Freefall.Graphics
{
    /// <summary>
    /// Thread-safe slot allocator for per-instance GPU data.
    /// Decoupled from specific buffer implementations — SceneBuffers uses this
    /// to assign unique slots shared across all GPUBuffer<T> channels.
    /// </summary>
    public class RenderSlotAllocator
    {
        private readonly Stack<int> _freeSlots = new();
        private int _nextSlot;
        private int _activeCount;
        private readonly object _lock = new();

        /// <summary>Number of currently allocated (live) slots.</summary>
        public int ActiveCount => _activeCount;

        /// <summary>High-water mark — maximum slot index ever issued plus one.</summary>
        public int HighWaterMark => _nextSlot;

        /// <summary>
        /// Allocate a new slot. Thread-safe.
        /// </summary>
        public int Allocate()
        {
            lock (_lock)
            {
                int slot = _freeSlots.Count > 0 ? _freeSlots.Pop() : _nextSlot++;
                _activeCount++;
                return slot;
            }
        }

        /// <summary>
        /// Release a slot for reuse. Thread-safe.
        /// </summary>
        public void Release(int slot)
        {
            lock (_lock)
            {
                _freeSlots.Push(slot);
                _activeCount--;
            }
        }
    }
}
