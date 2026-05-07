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
            public uint TanBufferIdx;
            public uint Reserved5;
            public uint Reserved6;
            public uint Reserved7;
            public uint Reserved8;
            public uint Reserved9;
        }

        public const int MaxMeshParts = 65536;
        public const int EntrySize = 72; // 18 uints

        private static readonly Dictionary<(int meshId, int partIndex), int> _idMap = new();
        private static readonly List<MeshPartEntry> _entries = new();
        private static readonly Stack<int> _freeSlots = new();
        private static ID3D12Resource? _buffer;
        private static uint _srvIndex;
        private static bool _dirty = true;
        private static readonly Lock _lock = new();

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
                    TanBufferIdx = mesh.TanBufferIndex,
                };

                if (_idMap.TryGetValue(key, out int existingId))
                {
                    // Refresh the entry — dynamic meshes (gizmos) change
                    // NumIndices/buffer indices every frame.
                    _entries[existingId] = entry;
                    _dirty = true;
                    return existingId;
                }

                if (_freeSlots.Count > 0)
                {
                    int id = _freeSlots.Pop();
                    _entries[id] = entry;
                    _idMap[key] = id;
                    _dirty = true;
                    return id;
                }

                if (_entries.Count >= MaxMeshParts)
                    throw new InvalidOperationException($"MeshRegistry exceeded max capacity of {MaxMeshParts}");

                int newId = _entries.Count;
                _entries.Add(entry);
                _idMap[key] = newId;
                _dirty = true;

                if (Engine.FrameIndex < 10)
                    Debug.Log("MeshRegistry", $"Registered mesh {mesh.Name} part {partIndex} as ID {newId}");

                return newId;
            }
        }

        /// <summary>
        /// Unregister all parts of a mesh, freeing their registry slots for reuse.
        /// Call from Mesh.Dispose().
        /// </summary>
        public static void Unregister(Mesh mesh)
        {
            lock (_lock)
            {
                int meshId = mesh.GetInstanceId();
                var keysToRemove = new List<(int, int)>();
                foreach (var kv in _idMap)
                {
                    if (kv.Key.meshId == meshId)
                    {
                        keysToRemove.Add(kv.Key);
                        _freeSlots.Push(kv.Value);
                        // Zero out the entry so GPU doesn't reference stale data
                        _entries[kv.Value] = default;
                    }
                }
                foreach (var key in keysToRemove)
                    _idMap.Remove(key);

                if (keysToRemove.Count > 0)
                    _dirty = true;
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
            _freeSlots.Clear();
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
