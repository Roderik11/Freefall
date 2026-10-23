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
        /// 72 bytes = 18 uints.
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
            public uint Reserved3;
            // Local-space bounding sphere (center + radius) for GPU culling
            public float BoundsCenterX;
            public float BoundsCenterY;
            public float BoundsCenterZ;
            public float BoundsRadius;
            // Reserved fields for 72-byte struct alignment
            public uint TanBufferIdx;
            // The mesh's index buffer for indexed indirect draws (IndexedDrawCommand: its GPU address and size)
            public uint IndexBufferAddressLo;
            public uint IndexBufferAddressHi;
            public uint IndexBufferBytes;
            public uint Reserved8;
            public uint Reserved9;
        }

        /// <summary>
        /// LOD chain data for a mesh part, in a buffer parallel to the MeshPartEntry buffer (same IDs).
        /// Only the culling compute pass reads it. Must match shader MeshPartLod exactly (32 bytes).
        /// "K" values are normalised squared distances, see Mesh.GetPartLod. An all-zero entry means
        /// "no LOD": always drawn, never size-culled.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct MeshPartLod
        {
            // Mesh-local centre shared by all parts of the mesh, so they make the same LOD decision
            public float CenterX;
            public float CenterY;
            public float CenterZ;
            public float CullK;      // beyond this the mesh is too small on screen (0 = never)
            public float NearK;      // the part is not drawn closer than this (0 = no limit)
            public float FarK;       // beyond this NextPartId takes over (0 = no limit)
            public uint NextPartId;  // MeshPartId of the next LOD's part + 1 (0 = none: not drawn beyond FarK)
            public uint Reserved;
        }

        public const int MaxMeshParts = 65536;
        public const int EntrySize = 72; // 18 uints
        public const int LodEntrySize = 32;

        private static readonly Dictionary<(int meshId, int partIndex), int> _idMap = new();
        private static readonly List<MeshPartEntry> _entries = new();
        private static readonly List<MeshPartLod> _lodEntries = new(); // parallel to _entries
        private static readonly Stack<int> _freeSlots = new();
        private static ID3D12Resource? _buffer;
        private static ID3D12Resource? _lodBuffer;
        private static uint _srvIndex;
        private static uint _lodSrvIndex;
        private static bool _dirty = true;
        private static readonly Lock _lock = new();

        // Entries changed since the last upload, as one range (Upload copies only these). Editor gizmos
        // re-register their proxy meshes every frame, so a full copy each time is what this avoids.
        private static int _dirtyMin = int.MaxValue;
        private static int _dirtyMax = -1;

        private static void MarkDirty(int id)
        {
            _dirty = true;
            if (id < _dirtyMin) _dirtyMin = id;
            if (id > _dirtyMax) _dirtyMax = id;
        }

        private static void MarkAllDirty()
        {
            _dirty = true;
            _dirtyMin = 0;
            _dirtyMax = int.MaxValue;
        }

        private static bool SameBytes<T>(in T a, in T b) where T : unmanaged =>
            MemoryMarshal.AsBytes(new ReadOnlySpan<T>(in a)).SequenceEqual(MemoryMarshal.AsBytes(new ReadOnlySpan<T>(in b)));

        public static uint SrvIndex => _srvIndex;
        /// <summary>SRV of the MeshPartLod buffer (indexed by MeshPartId).</summary>
        public static uint LodSrvIndex => _lodSrvIndex;
        public static int Count => _entries.Count;

        /// <summary>
        /// Register (or refresh) every part of a mesh, including its LOD chain. Returns the part IDs.
        /// </summary>
        public static int[] RegisterMesh(Mesh mesh)
        {
            lock (_lock)
            {
                int count = mesh.MeshParts.Count;
                var ids = new int[count];
                for (int i = 0; i < count; i++)
                    ids[i] = Register(mesh, i);

                // A chain link needs the next part's ID, which may only exist after the loop above
                if (mesh.LODs.Count > 0)
                    for (int i = 0; i < count; i++)
                    {
                        var lod = BuildLodEntry(mesh, i);
                        if (SameBytes(lod, _lodEntries[ids[i]])) continue;
                        _lodEntries[ids[i]] = lod;
                        MarkDirty(ids[i]);
                    }

                return ids;
            }
        }

        private static MeshPartLod BuildLodEntry(Mesh mesh, int partIndex)
        {
            mesh.GetPartLod(partIndex, out var center, out float cullK, out float nearK, out float farK, out int nextPartIndex);

            uint next = 0;
            if (nextPartIndex >= 0 && _idMap.TryGetValue((mesh.GetInstanceId(), nextPartIndex), out int nextId))
                next = (uint)nextId + 1;

            return new MeshPartLod
            {
                CenterX = center.X,
                CenterY = center.Y,
                CenterZ = center.Z,
                CullK = cullK,
                NearK = nearK,
                FarK = farK,
                NextPartId = next,
            };
        }

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
                    Reserved3 = 0,
                    BoundsCenterX = bounds.X,
                    BoundsCenterY = bounds.Y,
                    BoundsCenterZ = bounds.Z,
                    BoundsRadius = bounds.W,
                    TanBufferIdx = mesh.TanBufferIndex,
                    IndexBufferAddressLo = (uint)mesh.IndexBufferView.BufferLocation,
                    IndexBufferAddressHi = (uint)(mesh.IndexBufferView.BufferLocation >> 32),
                    IndexBufferBytes = mesh.IndexBufferView.SizeInBytes,
                };
                var lodEntry = BuildLodEntry(mesh, partIndex);

                if (_idMap.TryGetValue(key, out int existingId))
                {
                    // Refresh the entry — dynamic meshes (gizmos) change
                    // NumIndices/buffer indices every frame, often to the same values.
                    if (SameBytes(entry, _entries[existingId]) && SameBytes(lodEntry, _lodEntries[existingId]))
                        return existingId;

                    _entries[existingId] = entry;
                    _lodEntries[existingId] = lodEntry;
                    MarkDirty(existingId);
                    return existingId;
                }

                if (_freeSlots.Count > 0)
                {
                    int id = _freeSlots.Pop();
                    _entries[id] = entry;
                    _lodEntries[id] = lodEntry;
                    _idMap[key] = id;
                    MarkDirty(id);
                    return id;
                }

                if (_entries.Count >= MaxMeshParts)
                    throw new InvalidOperationException($"MeshRegistry exceeded max capacity of {MaxMeshParts}");

                int newId = _entries.Count;
                _entries.Add(entry);
                _lodEntries.Add(lodEntry);
                _idMap[key] = newId;
                MarkDirty(newId);

                if (Engine.FrameIndex < 10)
                    Debug.Log("MeshRegistry", $"Registered mesh {mesh.Name} part {partIndex} as ID {newId}");

                return newId;
            }
        }

        /// <summary>
        /// Unregister all parts of a mesh, freeing their registry slots for reuse.
        /// Call from Mesh.Dispose().
        /// </summary>
        public static void Unregister(Mesh mesh) => Unregister(mesh, 0);

        /// <summary>
        /// Unregister the parts of a mesh with index &gt;= <paramref name="firstPart"/>
        /// (a hot-reloaded mesh that lost parts).
        /// </summary>
        public static void Unregister(Mesh mesh, int firstPart)
        {
            lock (_lock)
            {
                int meshId = mesh.GetInstanceId();
                var keysToRemove = new List<(int, int)>();
                foreach (var kv in _idMap)
                {
                    if (kv.Key.meshId == meshId && kv.Key.partIndex >= firstPart)
                    {
                        keysToRemove.Add(kv.Key);
                        _freeSlots.Push(kv.Value);
                        // Zero out the entry so GPU doesn't reference stale data
                        _entries[kv.Value] = default;
                        _lodEntries[kv.Value] = default;
                        MarkDirty(kv.Value);
                    }
                }
                foreach (var key in keysToRemove)
                    _idMap.Remove(key);
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
                lock (_lock) MarkAllDirty(); // a new buffer gets every entry
            }

            if (_lodBuffer == null)
            {
                _lodBuffer = device.CreateUploadBuffer(MaxMeshParts * LodEntrySize);
                _lodSrvIndex = device.AllocateBindlessIndex();

                var lodSrvDesc = new ShaderResourceViewDescription
                {
                    Format = Format.Unknown,
                    ViewDimension = ShaderResourceViewDimension.Buffer,
                    Shader4ComponentMapping = ShaderComponentMapping.Default,
                    Buffer = new BufferShaderResourceView
                    {
                        FirstElement = 0,
                        NumElements = MaxMeshParts,
                        StructureByteStride = LodEntrySize,
                        Flags = BufferShaderResourceViewFlags.None
                    }
                };
                device.NativeDevice.CreateShaderResourceView(_lodBuffer, lodSrvDesc, device.GetCpuHandle(_lodSrvIndex));
                lock (_lock) MarkAllDirty();
            }

            // Copy the changed range straight from the lists, under the lock (background Register calls).
            // Only a few entries change per frame, so this is short; no snapshot arrays.
            lock (_lock)
            {
                int count = _entries.Count;
                int first = Math.Max(_dirtyMin, 0);
                int end = Math.Min(_dirtyMax, count - 1) + 1;
                _dirty = false;
                _dirtyMin = int.MaxValue;
                _dirtyMax = -1;
                if (end <= first) return;

                var entries = CollectionsMarshal.AsSpan(_entries)[first..end];
                var lods = CollectionsMarshal.AsSpan(_lodEntries)[first..end];

                unsafe
                {
                    void* pData;
                    _buffer.Map(0, null, &pData);
                    entries.CopyTo(new Span<MeshPartEntry>((MeshPartEntry*)pData + first, entries.Length));
                    _buffer.Unmap(0);

                    void* pLodData;
                    _lodBuffer.Map(0, null, &pLodData);
                    lods.CopyTo(new Span<MeshPartLod>((MeshPartLod*)pLodData + first, lods.Length));
                    _lodBuffer.Unmap(0);
                }
            }
        }

        /// <summary>
        /// Clear the registry. Call when unloading all content.
        /// </summary>
        public static void Clear()
        {
            _entries.Clear();
            _lodEntries.Clear();
            _idMap.Clear();
            _freeSlots.Clear();
            MarkAllDirty();
        }

        /// <summary>
        /// Dispose GPU resources.
        /// </summary>
        public static void Dispose()
        {
            _buffer?.Dispose();
            _buffer = null;
            _lodBuffer?.Dispose();
            _lodBuffer = null;
            lock (_lock) MarkAllDirty(); // whatever buffer comes next needs everything
            if (_srvIndex != 0)
            {
                Engine.Device?.ReleaseBindlessIndex(_srvIndex);
                _srvIndex = 0;
            }
            if (_lodSrvIndex != 0)
            {
                Engine.Device?.ReleaseBindlessIndex(_lodSrvIndex);
                _lodSrvIndex = 0;
            }
            Clear();
        }
    }
}
