using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Global registry of mesh/part metadata for GPU-driven rendering.
    /// Each unique mesh+part combination gets a stable ID at load time.
    /// The GPU looks up buffer indices from this registry instead of per-frame templates.
    /// </summary>
    public static class MeshRegistry
    {
        /// <summary>
        /// Per mesh/part metadata. Must match shader MeshPartEntry exactly.
        /// 72 bytes = 18 uints, same size as IndirectDrawCommand for easy GPU access.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct MeshPartEntry
        {
            public uint PosBufferIdx;
            public uint NormBufferIdx;
            public uint UVBufferIdx;
            public uint IndexBufferIdx;
            public uint BaseIndex;
            public uint VertexCount;
            public uint BoneWeightsBufferIdx;
            public uint NumBones;
            // Local-space bounding sphere (center + radius) for GPU culling
            public float BoundsCenterX;
            public float BoundsCenterY;
            public float BoundsCenterZ;
            public float BoundsRadius;
            // Padding to match IndirectDrawCommand size (72 bytes = 18 uints)
            public uint Reserved4;
            public uint Reserved5;
            public uint Reserved6;
            public uint Reserved7;
            public uint Reserved8;
            public uint Reserved9;
        }

        public const int MaxMeshParts = 4096;
        public const int EntrySize = 72; // 18 uints

        private static readonly Dictionary<(int meshId, int partIndex), int> _idMap = new();
        private static readonly List<MeshPartEntry> _entries = new();
        private static ID3D12Resource? _buffer;
        private static uint _srvIndex;
        private static bool _dirty = true;
        private static readonly object _lock = new object();

        public static uint SrvIndex => _srvIndex;
        public static int Count => _entries.Count;

        /// <summary>
        /// Register a mesh part and get its stable ID.
        /// Call this when a mesh is loaded, not per-frame.
        /// </summary>
        public static int Register(Mesh mesh, int partIndex)
        {
            var key = (mesh.GetInstanceId(), partIndex);
            
            lock (_lock)
            {
                if (_idMap.TryGetValue(key, out int existingId))
                    return existingId;

                if (_entries.Count >= MaxMeshParts)
                    throw new InvalidOperationException($"MeshRegistry exceeded max capacity of {MaxMeshParts}");

                var part = mesh.MeshParts[partIndex];
                var bounds = part.BoundingSphere;
                var entry = new MeshPartEntry
                {
                    PosBufferIdx = mesh.PosBufferIndex,
                    NormBufferIdx = mesh.NormBufferIndex,
                    UVBufferIdx = mesh.UVBufferIndex,
                    IndexBufferIdx = mesh.IndexBufferIndex,
                    BaseIndex = (uint)part.BaseIndex,
                    VertexCount = (uint)part.NumIndices,
                    BoneWeightsBufferIdx = mesh.BoneWeightBufferIndex,
                    NumBones = (uint)(mesh.Bones?.Length ?? 0),
                    BoundsCenterX = bounds.X,
                    BoundsCenterY = bounds.Y,
                    BoundsCenterZ = bounds.Z,
                    BoundsRadius = bounds.W,
                };

                int id = _entries.Count;
                _entries.Add(entry);
                _idMap[key] = id;
                _dirty = true;


                if (Engine.FrameIndex < 10)
                    Debug.Log("MeshRegistry", $"Registered mesh {mesh.Name} part {partIndex} as ID {id}");

                return id;
            }
        }

        /// <summary>
        /// Upload registry to GPU if modified. Call once per frame before rendering.
        /// </summary>
        public static void Upload(GraphicsDevice device)
        {
            if (!_dirty || _entries.Count == 0)
                return;

            // Create or resize buffer if needed
            if (_buffer == null)
            {
                int bufferSize = MaxMeshParts * EntrySize;
                _buffer = device.CreateUploadBuffer(bufferSize);
                _srvIndex = device.AllocateBindlessIndex();
                
                var srvDesc = new ShaderResourceViewDescription
                {
                    Format = Format.Unknown,
                    ViewDimension = ShaderResourceViewDimension.Buffer,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Buffer = new BufferShaderResourceView
                    {
                        FirstElement = 0,
                        NumElements = MaxMeshParts,
                        StructureByteStride = EntrySize,
                        Flags = BufferShaderResourceViewFlags.None
                    }
                };
                device.NativeDevice.CreateShaderResourceView(_buffer, srvDesc, device.GetCpuHandle(_srvIndex));
                
                Debug.Log("MeshRegistry", $"Created registry buffer, SRV index {_srvIndex}");
            }

            // Take snapshot under lock to avoid racing with background Register calls
            MeshPartEntry[] snapshot;
            int count;
            lock (_lock)
            {
                count = _entries.Count;
                snapshot = _entries.ToArray();
                _dirty = false;
            }

            // Upload snapshot (outside lock — GPU upload can be slow)
            unsafe
            {
                void* pData;
                _buffer.Map(0, null, &pData);
                var span = new Span<MeshPartEntry>(pData, count);
                for (int i = 0; i < count; i++)
                    span[i] = snapshot[i];
                _buffer.Unmap(0);
            }
        }

        /// <summary>
        /// Clear the registry. Call when unloading all content.
        /// </summary>
        public static void Clear()
        {
            _entries.Clear();
            _idMap.Clear();
            _dirty = true;
        }

        /// <summary>
        /// Dispose GPU resources.
        /// </summary>
        public static void Dispose()
        {
            _buffer?.Dispose();
            _buffer = null;
            if (_srvIndex != 0)
            {
                Engine.Device?.ReleaseBindlessIndex(_srvIndex);
                _srvIndex = 0;
            }
            Clear();
        }
    }
}
