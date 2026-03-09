using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.Direct3D12.Shader;

namespace Freefall.Graphics
{
    /// <summary>
    /// Abstract base for a shader constant buffer parameter discovered via reflection.
    /// Modeled after Apex's EffectParameter pattern.
    /// </summary>
    public abstract class EffectParameter
    {
        public string Name { get; set; }
        public string TypeName { get; set; }
        public int Offset { get; set; }
        public int Size { get; set; }
        public uint Elements { get; set; }
        public ConstantBuffer ConstantBuffer { get; set; }

        public abstract object GetValue();
        public abstract Type GetValueType();
        public abstract void Apply();

        /// <summary>
        /// Create the appropriately-typed EffectParameter from shader reflection metadata.
        /// TypeName is the HLSL type name (e.g. "float", "float3", "float4x4").
        /// </summary>
        public static EffectParameter Create(string name, string typeName, int offset, int size, uint elements, ConstantBuffer cb)
        {
            EffectParameter param = typeName switch
            {
                "bool"      => new BoolParameter(),
                "int"       => new IntParameter(),
                "uint"      => new UIntParameter(),
                "float"     => new FloatParameter(),
                "float2"    => new Vector2Parameter(),
                "float3"    => new Vector3Parameter(),
                "float4"    => new Vector4Parameter(),
                "float4x4"  => new MatrixParameter(),
                _           => new FloatParameter() // fallback for unknown types
            };
            param.Name = name;
            param.TypeName = typeName;
            param.Offset = offset;
            param.Size = size;
            param.Elements = elements;
            param.ConstantBuffer = cb;
            return param;
        }
    }

    public abstract class EffectParameter<T> : EffectParameter
    {
        private bool _changed;
        private T _value;

        public override Type GetValueType() => typeof(T);
        public override object GetValue() => _value;

        public T Value
        {
            get => _value;
            set { _changed = true; _value = value; }
        }

        public override void Apply()
        {
            if (!_changed) return;
            Commit();
            _changed = false;
        }

        protected abstract void Commit();
    }

    public class BoolParameter : EffectParameter<bool>
    {
        protected override void Commit() => ConstantBuffer.SetParameter(Name, Value);
    }

    public class IntParameter : EffectParameter<int>
    {
        protected override void Commit() => ConstantBuffer.SetParameter(Name, Value);
    }

    public class UIntParameter : EffectParameter<uint>
    {
        protected override void Commit() => ConstantBuffer.SetParameter(Name, Value);
    }

    public class FloatParameter : EffectParameter<float>
    {
        protected override void Commit() => ConstantBuffer.SetParameter(Name, Value);
    }

    public class Vector2Parameter : EffectParameter<System.Numerics.Vector2>
    {
        protected override void Commit() => ConstantBuffer.SetParameter(Name, Value);
    }

    public class Vector3Parameter : EffectParameter<System.Numerics.Vector3>
    {
        protected override void Commit() => ConstantBuffer.SetParameter(Name, Value);
    }

    public class Vector4Parameter : EffectParameter<System.Numerics.Vector4>
    {
        protected override void Commit() => ConstantBuffer.SetParameter(Name, Value);
    }

    public class MatrixParameter : EffectParameter<System.Numerics.Matrix4x4>
    {
        protected override void Commit() => ConstantBuffer.SetParameter(Name, Value);
    }

    public class ConstantBuffer : IDisposable
    {
        private ID3D12Resource[] _buffers;
        private IntPtr[] _mappedDatas;
        private int _size;
        private Dictionary<int, int> _parameters = new();
        private List<EffectParameter> _parameterList = new();
        private bool[] _isDirty = new bool[CommandBuffer.FrameCount];
        private byte[] _shadowBuffer; // CPU side copy for partial updates

        /// <summary>Name of the constant buffer (e.g., "SceneConstants")</summary>
        public string Name { get; }
        
        /// <summary>Root signature slot this buffer binds to</summary>
        public int Slot { get; set; } = -1;

        /// <summary>All parameters discovered from shader reflection.</summary>
        public IReadOnlyList<EffectParameter> Parameters => _parameterList;

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
                var varType = variable.VariableType;
                string typeName = varType.Description.Name;
                uint elements = varType.Description.ElementCount;
                _parameterList.Add(EffectParameter.Create(variableDesc.Name, typeName, (int)variableDesc.StartOffset, (int)variableDesc.Size, elements, this));
                //Debug.Log($"[CBReflect] {Name}: var='{variableDesc.Name}' offset={variableDesc.StartOffset} size={variableDesc.Size} hash={variableDesc.Name.GetHashCode()}");
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

        /// <summary>
        /// Merge variable offsets from another shader stage's reflection of the same cbuffer.
        /// DXC optimizes out unused variables per-stage, so a variable used only in the DS
        /// won't appear in the VS reflection. This ensures all stages' variables are registered.
        /// </summary>
        public void MergeVariables(ID3D12ShaderReflectionConstantBuffer reflectionBuffer)
        {
            var desc = reflectionBuffer.Description;
            for (uint i = 0; i < desc.VariableCount; i++)
            {
                var variable = reflectionBuffer.GetVariableByIndex(i);
                var variableDesc = variable.Description;
                int hash = variableDesc.Name.GetHashCode();
                if (!_parameters.ContainsKey(hash))
                {
                    _parameters[hash] = (int)variableDesc.StartOffset;
                    var varType = variable.VariableType;
                    string typeName = varType.Description.Name;
                    uint elements = varType.Description.ElementCount;
                    _parameterList.Add(EffectParameter.Create(variableDesc.Name, typeName, (int)variableDesc.StartOffset, (int)variableDesc.Size, elements, this));
                    //Debug.Log($"[CBMerge] {Name}: merged var='{variableDesc.Name}' offset={variableDesc.StartOffset} from additional stage");
                }
            }
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
                    fixed (byte* src = _shadowBuffer)
                        Unsafe.CopyBlock((void*)_mappedDatas[frameIndex], src, (uint)_size);
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
