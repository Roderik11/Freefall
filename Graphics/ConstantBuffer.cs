using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.Direct3D12.Shader;

namespace Freefall.Graphics
{
    public class ConstantBuffer : IDisposable
    {
        private ID3D12Resource[] _buffers;
        private IntPtr[] _mappedDatas;
        private int _size;
        private Dictionary<int, int> _parameters = new();
        private bool[] _isDirty = new bool[CommandBuffer.FrameCount];
        private byte[] _shadowBuffer; // CPU side copy for partial updates

        /// <summary>Name of the constant buffer (e.g., "SceneConstants")</summary>
        public string Name { get; }
        
        /// <summary>Root signature slot this buffer binds to</summary>
        public int Slot { get; set; } = -1;

        public ID3D12Resource Native => _buffers[Engine.FrameIndex % CommandBuffer.FrameCount];
        public ulong GpuAddress => _buffers[Engine.FrameIndex % CommandBuffer.FrameCount].GPUVirtualAddress;

        public ConstantBuffer(GraphicsDevice device, ID3D12ShaderReflectionConstantBuffer reflectionBuffer)
        {
            var desc = reflectionBuffer.Description;
            Name = desc.Name;
            _size = (int)desc.Size;
            
            // Align to 256 bytes (CBV requirement)
            _size = (int)((desc.Size + 255) & ~255);

            _buffers = new ID3D12Resource[CommandBuffer.FrameCount];
            _mappedDatas = new IntPtr[CommandBuffer.FrameCount];

            for (int i = 0; i < CommandBuffer.FrameCount; i++)
            {
                _buffers[i] = device.NativeDevice.CreateCommittedResource(
                    new HeapProperties(HeapType.Upload),
                    HeapFlags.None,
                    ResourceDescription.Buffer((ulong)_size),
                    ResourceStates.GenericRead,
                    null);

                unsafe
                {
                    void* pData;
                    _buffers[i].Map(0, null, &pData);
                    _mappedDatas[i] = (IntPtr)pData;
                }
            }
            _shadowBuffer = new byte[_size];

            // Parse Variables
            for (uint i = 0; i < desc.VariableCount; i++)
            {
                var variable = reflectionBuffer.GetVariableByIndex(i);
                var variableDesc = variable.Description;
                _parameters[variableDesc.Name.GetHashCode()] = (int)variableDesc.StartOffset;
                Debug.Log($"[CBReflect] {Name}: var='{variableDesc.Name}' offset={variableDesc.StartOffset} size={variableDesc.Size} hash={variableDesc.Name.GetHashCode()}");
            }
        }

        public ConstantBuffer(GraphicsDevice device, string name, int size)
        {
            Name = name;
            _size = (int)((size + 255) & ~255);
            _buffers = new ID3D12Resource[CommandBuffer.FrameCount];
            _mappedDatas = new IntPtr[CommandBuffer.FrameCount];

            for (int i = 0; i < CommandBuffer.FrameCount; i++)
            {
                _buffers[i] = device.NativeDevice.CreateCommittedResource(
                    new HeapProperties(HeapType.Upload),
                    HeapFlags.None,
                    ResourceDescription.Buffer((ulong)_size),
                    ResourceStates.GenericRead,
                    null);

                unsafe
                {
                    void* pData;
                    _buffers[i].Map(0, null, &pData);
                    _mappedDatas[i] = (IntPtr)pData;
                }
            }
            _shadowBuffer = new byte[_size];
            _parameters[name.GetHashCode()] = 0;
        }

        public void SetParameter<T>(string name, T value) where T : unmanaged
        {
             SetParameter(name.GetHashCode(), value);
        }

        public void SetParameter<T>(int hash, T value) where T : unmanaged
        {
            if (_parameters.TryGetValue(hash, out int offset))
            {
                // Write to shadow buffer
                var span = MemoryMarshal.AsBytes(MemoryMarshal.CreateSpan(ref value, 1));
                if (offset + span.Length <= _size)
                {
                    span.CopyTo(_shadowBuffer.AsSpan(offset));
                    for(int i=0; i<CommandBuffer.FrameCount; i++) _isDirty[i] = true;
                }
            }
        }


        public void SetParameterArray<T>(string name, T[] values) where T : unmanaged
        {
            SetParameterArray(name.GetHashCode(), values);
        }

        public void SetParameterArray<T>(int hash, T[] values) where T : unmanaged
        {
            if (values == null || values.Length == 0) return;
            
            if (_parameters.TryGetValue(hash, out int offset))
            {
                // Write array to shadow buffer
                var span = MemoryMarshal.AsBytes(values.AsSpan());
                if (offset + span.Length <= _size)
                {
                    span.CopyTo(_shadowBuffer.AsSpan(offset));
                    for(int i=0; i<CommandBuffer.FrameCount; i++) _isDirty[i] = true;
                }
            }
        }

        public void Commit()
        {
            int frameIndex = Engine.FrameIndex % CommandBuffer.FrameCount;
            
            if (_isDirty[frameIndex])
            {
                unsafe
                {
                    Marshal.Copy(_shadowBuffer, 0, _mappedDatas[frameIndex], _size);
                }
                _isDirty[frameIndex] = false;
            }
        }

        public void Dispose()
        {
            if (_buffers != null)
            {
                for (int i = 0; i < _buffers.Length; i++)
                {
                    _buffers[i].Unmap(0);
                    _buffers[i].Dispose();
                }
            }
        }
    }
}
