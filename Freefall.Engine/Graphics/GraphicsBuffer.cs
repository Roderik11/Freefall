using System;
using System.Runtime.CompilerServices;
using Vortice.Direct3D12;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>
    /// General-purpose GPU buffer with auto-managed bindless descriptors and resource state tracking.
    /// Wraps a single ID3D12Resource (default or upload heap) with optional SRV and UAV.
    /// For triple-buffered upload buffers with dirty tracking, use GPUBuffer&lt;T&gt; instead.
    /// </summary>
    public class GraphicsBuffer : IDisposable
    {
        private readonly GraphicsDevice _device;
        private readonly ID3D12Resource _resource;
        private readonly CpuDescriptorHandle _clearCpuHandle;
        private readonly bool _hasClearHandle;
        private ResourceStates _currentState;
        private bool _disposed;

        /// <summary>Native D3D12 resource.</summary>
        public ID3D12Resource Native => _resource;

        /// <summary>Bindless SRV index (0 if not created).</summary>
        public uint SrvIndex { get; }

        /// <summary>Bindless UAV index (0 if not created).</summary>
        public uint UavIndex { get; }

        /// <summary>Number of elements in the buffer.</summary>
        public int ElementCount { get; }

        /// <summary>Byte stride per element.</summary>
        public int Stride { get; }

        /// <summary>Total size in bytes.</summary>
        public int SizeInBytes => ElementCount * Stride;

        /// <summary>Current resource state for transition tracking.</summary>
        public ResourceStates CurrentState => _currentState;

        private GraphicsBuffer(
            GraphicsDevice device,
            ID3D12Resource resource,
            int elementCount,
            int stride,
            uint srvIndex,
            uint uavIndex,
            ResourceStates initialState,
            CpuDescriptorHandle clearCpuHandle,
            bool hasClearHandle)
        {
            _device = device;
            _resource = resource;
            ElementCount = elementCount;
            Stride = stride;
            SrvIndex = srvIndex;
            UavIndex = uavIndex;
            _currentState = initialState;
            _clearCpuHandle = clearCpuHandle;
            _hasClearHandle = hasClearHandle;
        }

        // ────────────── State Management ──────────────

        /// <summary>
        /// Emit a resource barrier if current state differs from target. No-op if already in target state.
        /// </summary>
        public void Transition(ID3D12GraphicsCommandList cmd, ResourceStates target)
        {
            if (_currentState == target) return;
            cmd.ResourceBarrierTransition(_resource, _currentState, target);
            _currentState = target;
        }

        /// <summary>
        /// Emit a UAV barrier for read-after-write synchronization on this buffer.
        /// </summary>
        public void UAVBarrier(ID3D12GraphicsCommandList cmd)
        {
            cmd.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_resource)));
        }

        /// <summary>
        /// Clear the UAV to the specified value. Requires UAV to have been created.
        /// </summary>
        public void ClearUAV(ID3D12GraphicsCommandList cmd, Int4 clearValue)
        {
            if (UavIndex == 0 || !_hasClearHandle)
                throw new InvalidOperationException("GraphicsBuffer has no UAV or was not created with ClearUAV support.");

            cmd.ClearUnorderedAccessViewUint(
                _device.GetGpuHandle(UavIndex),
                _clearCpuHandle,
                _resource,
                clearValue);
        }

        // ────────────── Factory Methods ──────────────

        /// <summary>
        /// Create a GPU-only (default heap) structured buffer with optional SRV and/or UAV.
        /// </summary>
        public static GraphicsBuffer CreateStructured<T>(
            int count,
            bool srv = false,
            bool uav = false,
            ResourceStates initialState = ResourceStates.Common) where T : unmanaged
        {
            var device = Engine.Device;
            int stride = Unsafe.SizeOf<T>();
            int size = count * stride;

            var resource = device.CreateDefaultBuffer(size);

            uint srvIndex = 0;
            if (srv)
            {
                srvIndex = device.AllocateBindlessIndex();
                device.CreateStructuredBufferSRV(resource, (uint)count, (uint)stride, srvIndex);
            }

            uint uavIndex = 0;
            CpuDescriptorHandle clearHandle = default;
            bool hasClear = false;
            if (uav)
            {
                uavIndex = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(resource, (uint)count, (uint)stride, uavIndex);

                // Create non-shader-visible CPU handle for ClearUnorderedAccessViewUint
                clearHandle = device.AllocateClearHandle(resource, (uint)count, (uint)stride);
                hasClear = true;
            }

            return new GraphicsBuffer(device, resource, count, stride, srvIndex, uavIndex, initialState, clearHandle, hasClear);
        }

        /// <summary>
        /// Create an upload heap structured buffer with SRV (for CPU→GPU data like constant arrays).
        /// </summary>
        public static GraphicsBuffer CreateUpload<T>(int count) where T : unmanaged
        {
            var device = Engine.Device;
            int stride = Unsafe.SizeOf<T>();
            int size = Math.Max(256, count * stride);

            var resource = device.CreateUploadBuffer(size);

            uint srvIndex = device.AllocateBindlessIndex();
            device.CreateStructuredBufferSRV(resource, (uint)count, (uint)stride, srvIndex);

            return new GraphicsBuffer(device, resource, count, stride, srvIndex, 0,
                ResourceStates.GenericRead, default, false);
        }

        /// <summary>
        /// Map the buffer for CPU write access. Only valid for upload buffers.
        /// </summary>
        public unsafe T* Map<T>() where T : unmanaged
        {
            void* ptr;
            _resource.Map(0, null, &ptr);
            return (T*)ptr;
        }

        /// <summary>
        /// Unmap the buffer after CPU writes.
        /// </summary>
        public void Unmap()
        {
            _resource.Unmap(0);
        }

        /// <summary>
        /// Upload a span of data to an upload buffer via Map/Unmap.
        /// </summary>
        public unsafe void Upload<T>(ReadOnlySpan<T> data) where T : unmanaged
        {
            void* ptr;
            _resource.Map(0, null, &ptr);
            data.CopyTo(new Span<T>(ptr, data.Length));
            _resource.Unmap(0);
        }

        // ────────────── Dispose ──────────────

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _resource?.Dispose();

            if (SrvIndex > 0)
                _device.ReleaseBindlessIndex(SrvIndex);
            if (UavIndex > 0)
                _device.ReleaseBindlessIndex(UavIndex);
        }
    }
}
