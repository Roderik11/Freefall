using System;
using System.Threading;
using System.Collections.Generic;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;
using System.Numerics;
using static Vortice.Direct3D12.D3D12;
using Freefall.Assets;
using Freefall.Animation;

namespace Freefall.Graphics
{
    /// <summary>Patch edge stitching type for Terrain.</summary>
    public enum PatchType
    {
        Default,
        N, E, S, W,
        NE, NW, SE, SW,
        Center
    }

    [Serializable]
    public class MeshPart
    {
        public string Name = string.Empty;
        public bool Enabled = true;
        public int BaseVertex;
        public int BaseIndex;
        public int NumIndices;
        public int MaterialSlot;  // Index into MeshRenderer.Materials[]
        public BoundingBox BoundingBox;
        public Vector4 BoundingSphere; // Local-space bounding sphere (center.xyz, radius)
    }

    /// <summary>
    /// A single LOD level, referencing a subset of MeshParts by index.
    /// </summary>
    public class MeshLOD
    {
        public int[] MeshPartIndices;

        /// <summary>
        /// Screen size (bounding sphere diameter as a fraction of viewport height) below which
        /// this LOD takes over from the previous one. Derived from triangle counts by
        /// Mesh.ComputeLODScreenSizes; LOD 0 is always float.MaxValue.
        /// </summary>
        [NonSerialized]
        public float ScreenSize = float.MaxValue;
    }

    [AssetTypeAlias("MeshData")]
    public partial class Mesh : Asset, IDisposable
    {
        private static volatile int _instanceCount;
        private readonly int _instanceId = Interlocked.Increment(ref _instanceCount);
        public int GetInstanceId() => _instanceId;
        
        public List<MeshPart> MeshParts { get; set; } = new List<MeshPart>();

        // LOD chain: each MeshLOD references a subset of MeshParts by index.
        // Populated by ModelImporter when sub-meshes have LOD naming conventions.
        public List<MeshLOD> LODs { get; set; } = new List<MeshLOD>();

        /// <summary>
        /// Indices of MeshParts not belonging to any LOD level.
        /// Drawn alongside the active LOD. Null if empty or no LODs.
        /// </summary>
        public int[]? NonLodPartIndices { get; set; }

        /// <summary>
        /// Per-mesh LOD distance bias. Default 1.0.
        /// Higher values keep high-detail LODs visible longer.
        /// Multiplied with global Engine.Settings.LODScale.
        /// </summary>
        [ValueRange(0.1f, 10.0f)]
        public float LODBias
        {
            get => _lodBias;
            set
            {
                if (_lodBias == value) return;
                _lodBias = value;
                // The bias is folded into the GPU LOD chain (MeshRegistry), so refresh the entries.
                if (_meshPartIds != null) RegisterMeshParts();
            }
        }
        private float _lodBias = 1.0f;

        public bool IsDynamic { get; set; }

        /// <summary>
        /// Screen size at which LOD 0 is considered to have exactly the triangle density it needs.
        /// Every lower LOD switches in where it reaches that same density, so the chain adapts to
        /// how many LODs a mesh has and how strongly each one is reduced.
        /// </summary>
        public const float LODReferenceScreenSize = 0.5f;

        /// <summary>Objects smaller than this fraction of viewport height are not drawn.</summary>
        public const float LODCullScreenSize = 0.006f;

        // A LOD must switch in at least this much below the previous one, even if it barely
        // reduces the triangle count (or the chain is not monotonic).
        private const float LODMinStep = 0.8f;

        // ...and at most this much below it. Billboard/impostor LODs have almost no triangles
        // (their detail is in the texture), so the density formula alone would put them at the
        // cull size and they would never show.
        private const float LODMaxStep = 0.5f;

        private const float LODCullScreenSizeSq = LODCullScreenSize * LODCullScreenSize;

        /// <summary>
        /// Derive MeshLOD.ScreenSize for every LOD: ScreenSize[i] = reference * sqrt(tris[i] / tris[0]).
        /// </summary>
        public void ComputeLODScreenSizes()
        {
            if (LODs.Count == 0) return;

            long baseIndices = CountLODIndices(LODs[0]);
            LODs[0].ScreenSize = float.MaxValue;

            float previous = LODReferenceScreenSize;
            for (int i = 1; i < LODs.Count; i++)
            {
                long indices = CountLODIndices(LODs[i]);
                float ratio = baseIndices > 0 ? MathF.Min(1f, (float)indices / baseIndices) : 1f;
                float size = LODReferenceScreenSize * MathF.Sqrt(ratio);
                size = Math.Clamp(size, previous * LODMaxStep, i == 1 ? previous : previous * LODMinStep);
                LODs[i].ScreenSize = size;
                previous = size;
            }
        }

        private long CountLODIndices(MeshLOD lod)
        {
            long count = 0;
            if (lod.MeshPartIndices != null)
                foreach (var idx in lod.MeshPartIndices)
                    if (idx < MeshParts.Count)
                        count += MeshParts[idx].NumIndices;
            return count;
        }

        /// <summary>
        /// CPU early-out for renderers that still enqueue every frame: true if the sphere is below
        /// LODCullScreenSize for Camera.Main. The GPU culler applies the same test (MeshPartLod.CullK);
        /// this only saves the enqueue and the per-instance upload for objects it would reject anyway.
        /// Squared space, no sqrt or division. Goes away once draws are GPU-resident.
        /// </summary>
        public static bool IsBelowCullSize(in BoundingSphere sphere)
        {
            var cam = Components.Camera.Main;
            if (cam == null) return false;

            float distanceSq = Vector3.DistanceSquared(sphere.Center, cam.Position);
            float scale = Engine.Settings.LODScale;
            return sphere.Radius * sphere.Radius * cam.FoVFactor * scale * scale < LODCullScreenSizeSq * distanceSq;
        }

        /// <summary>
        /// Part indices a renderer submits each frame: the head of every LOD chain plus the non-LOD
        /// parts (all parts if the mesh has no LOD chain). LOD selection happens on the GPU: the
        /// culling pass walks from each head to the part of the active LOD (see MeshRegistry.MeshPartLod).
        /// </summary>
        public int[] DrawPartIndices
        {
            get
            {
                // Runtime-built meshes fill MeshParts without calling ComputeNonLodPartIndices
                var draw = _drawPartIndices;
                if (draw == null || (LODs.Count == 0 && draw.Length != MeshParts.Count))
                {
                    ComputeLODChain();
                    draw = _drawPartIndices!;
                }
                return draw;
            }
        }
        private int[]? _drawPartIndices;

        // Per part: first/last LOD index containing it and the part that takes over at the next LOD
        // (same material slot), or -1. Null if the mesh has no LOD chain.
        private int[]? _partLodFirst;
        private int[]? _partLodLast;
        private int[]? _partLodNext;

        /// <summary>
        /// Build the per-part LOD chain: each part links to the part with the same material slot in
        /// the next LOD that has one. Parts nothing links to are the chain heads.
        /// </summary>
        private void ComputeLODChain()
        {
            int partCount = MeshParts.Count;

            if (LODs.Count == 0)
            {
                _partLodFirst = _partLodLast = _partLodNext = null;
                var all = new int[partCount];
                for (int i = 0; i < partCount; i++) all[i] = i;
                _drawPartIndices = all;
                return;
            }

            var first = new int[partCount];
            var last = new int[partCount];
            var next = new int[partCount];
            Array.Fill(first, -1);
            Array.Fill(last, -1);
            Array.Fill(next, -1);

            for (int lod = 0; lod < LODs.Count; lod++)
            {
                var indices = LODs[lod].MeshPartIndices;
                if (indices == null) continue;
                foreach (var idx in indices)
                {
                    if (idx >= partCount) continue;
                    if (first[idx] < 0) first[idx] = lod;
                    last[idx] = lod;
                }
            }

            var isTarget = new bool[partCount];
            for (int lod = 0; lod < LODs.Count; lod++)
            {
                var indices = LODs[lod].MeshPartIndices;
                if (indices == null) continue;
                foreach (var p in indices)
                {
                    if (p >= partCount || first[p] != lod) continue;
                    int slot = MeshParts[p].MaterialSlot;

                    for (int l = last[p] + 1; l < LODs.Count && next[p] < 0; l++)
                    {
                        var candidates = LODs[l].MeshPartIndices;
                        if (candidates == null) continue;
                        foreach (var q in candidates)
                        {
                            if (q >= partCount || first[q] != l || isTarget[q]) continue;
                            if (MeshParts[q].MaterialSlot != slot) continue;
                            next[p] = q;
                            isTarget[q] = true;
                            break;
                        }
                    }
                }
            }

            var draw = new List<int>();
            for (int i = 0; i < partCount; i++)
                if (first[i] >= 0 && !isTarget[i])
                    draw.Add(i);
            if (NonLodPartIndices != null)
                draw.AddRange(NonLodPartIndices);

            _partLodFirst = first;
            _partLodLast = last;
            _partLodNext = next;
            _drawPartIndices = draw.ToArray();
        }

        /// <summary>
        /// LOD data for one part's MeshRegistry entry. Distances are stored as "k" values:
        /// k = distanceSq / (Camera.FoVFactor * LODScale² * instanceScale²), so a screen-size threshold T
        /// becomes k = radius² * LODBias² / T². The mesh radius and bias are folded in here.
        /// </summary>
        /// <param name="center">Mesh-local centre every part measures distance to, so they switch together.</param>
        /// <param name="cullK">Beyond this k the mesh is smaller than LODCullScreenSize. 0 = never.</param>
        /// <param name="nearK">The part is not drawn closer than this. 0 = no limit.</param>
        /// <param name="farK">Beyond this the next part takes over. 0 = no limit.</param>
        /// <param name="nextPartIndex">Part of the next LOD, or -1.</param>
        internal void GetPartLod(int partIndex, out Vector3 center, out float cullK, out float nearK, out float farK, out int nextPartIndex)
        {
            var box = BoundingBox;
            center = (box.Min + box.Max) * 0.5f;
            float radiusSq = Vector3.DistanceSquared(box.Max, center);

            cullK = radiusSq / LODCullScreenSizeSq;
            nearK = 0;
            farK = 0;
            nextPartIndex = -1;

            var first = _partLodFirst;
            var last = _partLodLast;
            var next = _partLodNext;
            if (first == null || last == null || next == null) return;
            if (partIndex >= first.Length || first[partIndex] < 0) return;

            float biased = radiusSq * _lodBias * _lodBias;
            int a = first[partIndex];
            int b = last[partIndex];
            if (a > 0 && a < LODs.Count)
            {
                float t = LODs[a].ScreenSize;
                nearK = biased / (t * t);
            }
            if (b + 1 < LODs.Count)
            {
                float t = LODs[b + 1].ScreenSize;
                farK = biased / (t * t);
            }
            nextPartIndex = next[partIndex];
        }

        /// <summary>
        /// Compute NonLodPartIndices from LODs and MeshParts.
        /// Call after both LODs and MeshParts are populated.
        /// </summary>
        public void ComputeNonLodPartIndices()
        {
            ComputeLODScreenSizes();

            if (LODs.Count == 0 || MeshParts.Count == 0)
            {
                NonLodPartIndices = null;
                ComputeLODChain();
                return;
            }

            var inLod = new bool[MeshParts.Count];
            foreach (var lod in LODs)
                if (lod.MeshPartIndices != null)
                    foreach (var idx in lod.MeshPartIndices)
                        if (idx < inLod.Length)
                            inLod[idx] = true;

            var nonLod = new List<int>();
            for (int i = 0; i < inLod.Length; i++)
                if (!inLod[i] && MeshParts[i].Enabled)
                    nonLod.Add(i);

            NonLodPartIndices = nonLod.Count > 0 ? nonLod.ToArray() : null;
            ComputeLODChain();
        }
        
        // Buffers
        private ID3D12Resource _posBuffer = null!;
        private VertexBufferView _posView;
        private ID3D12Resource _normBuffer = null!;
        private VertexBufferView _normView;
        private ID3D12Resource _uvBuffer = null!;
        private VertexBufferView _uvView;
        private ID3D12Resource? _tanBuffer;

        private int _vertexCount;
        private ID3D12Resource _indexBuffer = null!;
        public IndexBufferView IndexBufferView => _indexBufferView;
        private IndexBufferView _indexBufferView;
        private int _indexCount;

        // Bindless Indices
        public uint PosBufferIndex { get; internal set; }
        public uint NormBufferIndex { get; internal set; }
        public uint UVBufferIndex { get; internal set; }
        public uint TanBufferIndex { get; internal set; }
        public uint IndexBufferIndex { get; internal set; }
        
        public BoundingBox BoundingBox { get; set; }
        
        // CPU-side data retained for physics cooking
        public Vector3[]? Positions { get; private set; }
        public uint[]? CpuIndices { get; private set; }
        
        public Vector4 LocalBoundingSphere
        {
            get
            {
                var center = BoundingBox.Center;
                var radius = (BoundingBox.Max - center).Length();
                return new Vector4(center, radius);
            }
        }

        // Skeleton / Animation
        /// <summary>The skeleton asset this mesh was imported with (for retargeting).</summary>
        public Animation.Skeleton Skeleton { get; set; }

        /// <summary>Bone data. Prefers Skeleton asset, falls back to inline MeshPacker data.</summary>
        public Bone[]? Bones => Skeleton?.Bones;

        public BoneWeight[]? BoneWeights { get; set; }
        public Matrix4x4 RootRotation { get; set; } = Matrix4x4.Identity;
        
        private ID3D12Resource? _boneWeightBuffer;
        public uint BoneWeightBufferIndex { get; private set; }

        /// <summary>
        /// Pre-cooked PhysX triangle mesh. Populated during asset loading
        /// so that RigidBody.Awake() doesn't need to cook on the main thread.
        /// </summary>
        [System.Text.Json.Serialization.JsonIgnore]
        public PhysX.TriangleMesh? CookedTriMesh { get; set; }

        /// <summary>
        /// Cook the physics triangle mesh on the calling thread (background) and cache it.
        /// Thread-safe: each call creates its own Cooking instance.
        /// </summary>
        public void CookPhysicsMesh()
        {
            if (Positions == null || CpuIndices == null)
                return;

            var triangles = System.Array.ConvertAll(CpuIndices, i => (int)i);

            var cooking = Freefall.Base.PhysicsWorld.Physics.CreateCooking();
            var desc = new PhysX.TriangleMeshDesc()
            {
                Flags = (PhysX.MeshFlag)0,
                Triangles = triangles,
                Points = Positions
            };

            var stream = new System.IO.MemoryStream();
            cooking.CookTriangleMesh(desc, stream);

            stream.Position = 0;
            CookedTriMesh = Freefall.Base.PhysicsWorld.Physics.CreateTriangleMesh(stream);
        }

        private int[]? _meshPartIds;

        public Mesh() { }

        public Mesh(GraphicsDevice device, Vector3[] positions, Vector3[] normals, Vector2[] uvs, uint[] indices)
        {
            CreateBuffers(device, positions, normals, null, uvs, indices);
            MarkReady();
        }

        public Mesh(GraphicsDevice device, MeshData data)
        {
            CreateBuffers(device, data.Positions, data.Normals, data.Tangents, data.UVs, data.Indices);
            MeshParts.AddRange(data.Parts);
            BoundingBox = data.BoundingBox;
            if (data.LODs != null && data.LODs.Count > 0)
                LODs.AddRange(data.LODs);
            ComputeNonLodPartIndices();
            
            //if (data.Bones != null)
            //    Bones = data.Bones;
            
            if (data.BoneWeights != null && data.BoneWeights.Length > 0)
            {
                BoneWeights = data.BoneWeights;
                CreateBoneWeightBuffer(device);
            }

            MarkReady();
        }

        // Async Compatible Factory
        public static Mesh CreateAsync(GraphicsDevice device, MeshData data)
        {
            var mesh = new Mesh();
            mesh.MeshParts.AddRange(data.Parts);
            mesh.BoundingBox = data.BoundingBox;
            mesh.BoneWeights = data.BoneWeights;
            if (data.LODs != null && data.LODs.Count > 0)
                mesh.LODs.AddRange(data.LODs);
            mesh.ComputeNonLodPartIndices();
            
            // For meshes, because they are structured buffers, we still create the Committed Resource on Main Thread
            // But we skip the *Upload* part here?
            // Wait, StreamingManager is built for Textures mostly.
            // For Meshes, we can use the same ring buffer logic, but we need 'RecordBufferUpload'.
            
            // To be safe and fast for this iteration:
            // We kept the synchronous buffer creation for Meshes in Stage 3 of the SceneLoader plan.
            // i.e., "Quick call to CreateCommittedResource".
            // So we can reuse CreateBuffers but we need to CHANGE it to NOT do the CopyQueueWait.
            
            mesh.CreateBuffersAsync(device, data.Positions, data.Normals, data.Tangents, data.UVs, data.Indices);
            
            // Bone weights?
            if (data.BoneWeights != null && data.BoneWeights.Length > 0)
            {
                mesh.CreateBoneWeightBuffer(device);
            }


            return mesh;
        }

        private void CreateBuffers(GraphicsDevice device, Vector3[] positions, Vector3[] normals, Vector4[] tangents, Vector2[] uvs, uint[] indices)
        {
            // Legacy Synchronous Path
             _vertexCount = positions.Length;
             
            // Retain CPU-side data for physics cooking
            Positions = positions;
            CpuIndices = indices;

            _posBuffer = CreateBuffer(device, positions);
            _posView = new VertexBufferView { BufferLocation = _posBuffer.GPUVirtualAddress, SizeInBytes = (uint)(positions.Length * 12), StrideInBytes = 12 };

            _normBuffer = CreateBuffer(device, normals);
            _normView = new VertexBufferView { BufferLocation = _normBuffer.GPUVirtualAddress, SizeInBytes = (uint)(normals.Length * 12), StrideInBytes = 12 };

            _uvBuffer = CreateBuffer(device, uvs);
            _uvView = new VertexBufferView { BufferLocation = _uvBuffer.GPUVirtualAddress, SizeInBytes = (uint)(uvs.Length * 8), StrideInBytes = 8 };

            PosBufferIndex = device.AllocateBindlessIndex();
            CreateStructuredBufferSRV(device, _posBuffer, (uint)positions.Length, 12, PosBufferIndex);

            NormBufferIndex = device.AllocateBindlessIndex();
            CreateStructuredBufferSRV(device, _normBuffer, (uint)normals.Length, 12, NormBufferIndex);

            UVBufferIndex = device.AllocateBindlessIndex();
            CreateStructuredBufferSRV(device, _uvBuffer, (uint)uvs.Length, 8, UVBufferIndex);

            // Tangent buffer (float4, 16 bytes per vertex)
            if (tangents != null && tangents.Length > 0)
            {
                _tanBuffer = CreateBuffer(device, tangents);
                TanBufferIndex = device.AllocateBindlessIndex();
                CreateStructuredBufferSRV(device, _tanBuffer, (uint)tangents.Length, 16, TanBufferIndex);
            }

            if (indices != null)
            {
                _indexCount = indices.Length;
                _indexBuffer = CreateBuffer(device, indices); // Note: CreateBuffer<uint>
                 _indexBufferView = new IndexBufferView { BufferLocation = _indexBuffer.GPUVirtualAddress, SizeInBytes = (uint)(indices.Length * 4), Format = Format.R32_UInt };
                 
                IndexBufferIndex = device.AllocateBindlessIndex();
                CreateStructuredBufferSRV(device, _indexBuffer, (uint)indices.Length, 4, IndexBufferIndex);
            }
        }
        
        private void CreateBuffersAsync(GraphicsDevice device, Vector3[] positions, Vector3[] normals, Vector4[] tangents, Vector2[] uvs, uint[] indices)
        {
            // 1. Create GPU Resources (Fast, no upload)
            _vertexCount = positions.Length;
            
            // Retain CPU-side data for physics cooking
            Positions = positions;
            CpuIndices = indices;
            _posBuffer = device.CreateDefaultBuffer(positions.Length * 12, ResourceFlags.None);
            _normBuffer = device.CreateDefaultBuffer(normals.Length * 12, ResourceFlags.None);
            _uvBuffer = device.CreateDefaultBuffer(uvs.Length * 8, ResourceFlags.None);
            
            // 2. Initialize Views and SRVs
             _posView = new VertexBufferView { BufferLocation = _posBuffer.GPUVirtualAddress, SizeInBytes = (uint)(positions.Length * 12), StrideInBytes = 12 };
             _normView = new VertexBufferView { BufferLocation = _normBuffer.GPUVirtualAddress, SizeInBytes = (uint)(normals.Length * 12), StrideInBytes = 12 };
             _uvView = new VertexBufferView { BufferLocation = _uvBuffer.GPUVirtualAddress, SizeInBytes = (uint)(uvs.Length * 8), StrideInBytes = 8 };

            PosBufferIndex = device.AllocateBindlessIndex();
            CreateStructuredBufferSRV(device, _posBuffer, (uint)positions.Length, 12, PosBufferIndex);

            NormBufferIndex = device.AllocateBindlessIndex();
            CreateStructuredBufferSRV(device, _normBuffer, (uint)normals.Length, 12, NormBufferIndex);

            UVBufferIndex = device.AllocateBindlessIndex();
            CreateStructuredBufferSRV(device, _uvBuffer, (uint)uvs.Length, 8, UVBufferIndex);

            // Tangent buffer (float4, 16 bytes per vertex)
            if (tangents != null && tangents.Length > 0)
            {
                _tanBuffer = device.CreateDefaultBuffer(tangents.Length * 16, ResourceFlags.None);
                TanBufferIndex = device.AllocateBindlessIndex();
                CreateStructuredBufferSRV(device, _tanBuffer, (uint)tangents.Length, 16, TanBufferIndex);
                StreamingManager.Instance.EnqueueBufferUpload(_tanBuffer, tangents);
            }
            
            // 3. Queue Uploads
            StreamingManager.Instance.EnqueueBufferUpload(_posBuffer, positions);
            StreamingManager.Instance.EnqueueBufferUpload(_normBuffer, normals);
            StreamingManager.Instance.EnqueueBufferUpload(_uvBuffer, uvs);
            
            if (indices != null)
            {
                _indexCount = indices.Length;
                _indexBuffer = device.CreateDefaultBuffer(indices.Length * 4, ResourceFlags.None);
                _indexBufferView = new IndexBufferView { BufferLocation = _indexBuffer.GPUVirtualAddress, SizeInBytes = (uint)(indices.Length * 4), Format = Format.R32_UInt };
                
                IndexBufferIndex = device.AllocateBindlessIndex();
                CreateStructuredBufferSRV(device, _indexBuffer, (uint)indices.Length, 4, IndexBufferIndex);
                
                StreamingManager.Instance.EnqueueBufferUpload(_indexBuffer, indices);
            }

            // Set fence logic? 
            // Since we enqueue multiple buffers, we need the *last* fence.
            // The Streaming Manager could return a "Task" or "Fence" for the batch?
        }
        
        // Helper
        private static ID3D12Resource CreateBuffer<T>(GraphicsDevice device, T[] data) where T : unmanaged
        {
             int size = data.Length * System.Runtime.InteropServices.Marshal.SizeOf<T>();
             var buffer = device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Default), HeapFlags.None, ResourceDescription.Buffer((ulong)size), ResourceStates.Common, null);
             
             // Synchronous Upload
             var uploadBuffer = device.CreateUploadBuffer(data);
             using (var cmd = device.NativeDevice.CreateCommandAllocator(CommandListType.Copy))
             using (var list = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Copy, cmd))
             {
                 list.CopyResource(buffer, uploadBuffer);
                 list.Close();
                 device.CopyQueueSubmitAndWait(list);
             }
             uploadBuffer.Dispose();
             return buffer;
        }

        private void CreateStructuredBufferSRV(GraphicsDevice device, ID3D12Resource resource, uint numElements, uint stride, uint bindlessIndex)
        {
            var srvDesc = new ShaderResourceViewDescription
            {
                Format = Format.Unknown,
                ViewDimension = ShaderResourceViewDimension.Buffer,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Buffer = new BufferShaderResourceView { FirstElement = 0, NumElements = numElements, StructureByteStride = stride }
            };
            device.NativeDevice.CreateShaderResourceView(resource, srvDesc, device.GetCpuHandle(bindlessIndex));
        }

        // Methods related to MeshRegistry, BoneWeights, Draw() etc. preserved...
        public int GetMeshPartId(int partIndex) { if (_meshPartIds == null) return -1; return _meshPartIds.Length > partIndex ? _meshPartIds[partIndex] : -1; }
        public void RegisterMeshParts() { if (MeshParts.Count == 0) return; _meshPartIds = MeshRegistry.RegisterMesh(this); }
        
        /// <summary>
        /// Bumped by ReplaceGeometry. Renderers compare it to invalidate bounds cached from this instance.
        /// </summary>
        public int GeometryVersion { get; private set; }

        /// <summary>
        /// Hot reload: take over <paramref name="source"/>'s GPU buffers, parts, LODs and bounds so every
        /// renderer referencing this instance draws the new geometry. <paramref name="source"/> receives the
        /// old buffers and is disposed after the frames in flight retire; it must be a freshly created,
        /// never-registered mesh whose uploads have completed (StreamingManager.Flush). Main thread only.
        /// </summary>
        public void ReplaceGeometry(Mesh source)
        {
            (_posBuffer, source._posBuffer) = (source._posBuffer, _posBuffer);
            (_normBuffer, source._normBuffer) = (source._normBuffer, _normBuffer);
            (_uvBuffer, source._uvBuffer) = (source._uvBuffer, _uvBuffer);
            (_tanBuffer, source._tanBuffer) = (source._tanBuffer, _tanBuffer);
            (_indexBuffer, source._indexBuffer) = (source._indexBuffer, _indexBuffer);
            (_boneWeightBuffer, source._boneWeightBuffer) = (source._boneWeightBuffer, _boneWeightBuffer);
            (_posView, source._posView) = (source._posView, _posView);
            (_normView, source._normView) = (source._normView, _normView);
            (_uvView, source._uvView) = (source._uvView, _uvView);
            (_indexBufferView, source._indexBufferView) = (source._indexBufferView, _indexBufferView);

            // Bindless indices travel with their buffers: the old ones are released when source is disposed.
            (PosBufferIndex, source.PosBufferIndex) = (source.PosBufferIndex, PosBufferIndex);
            (NormBufferIndex, source.NormBufferIndex) = (source.NormBufferIndex, NormBufferIndex);
            (UVBufferIndex, source.UVBufferIndex) = (source.UVBufferIndex, UVBufferIndex);
            (TanBufferIndex, source.TanBufferIndex) = (source.TanBufferIndex, TanBufferIndex);
            (IndexBufferIndex, source.IndexBufferIndex) = (source.IndexBufferIndex, IndexBufferIndex);
            (BoneWeightBufferIndex, source.BoneWeightBufferIndex) = (source.BoneWeightBufferIndex, BoneWeightBufferIndex);

            _vertexCount = source._vertexCount;
            _indexCount = source._indexCount;
            Positions = source.Positions;
            CpuIndices = source.CpuIndices;
            BoneWeights = source.BoneWeights;
            Skeleton = source.Skeleton ?? Skeleton;

            // Swap list references rather than mutating: nothing else holds the old lists.
            MeshParts = source.MeshParts;
            LODs = source.LODs;
            NonLodPartIndices = source.NonLodPartIndices;
            _drawPartIndices = source._drawPartIndices;
            _partLodFirst = source._partLodFirst;
            _partLodLast = source._partLodLast;
            _partLodNext = source._partLodNext;
            BoundingBox = source.BoundingBox;

            // Stale collision data; the next Collider cooks from the new Positions.
            CookedTriMesh = null;

            // Refresh registry entries in place (same MeshPartIds, new buffer indices/ranges/bounds),
            // then free the ids of parts that no longer exist.
            int oldPartCount = _meshPartIds?.Length ?? 0;
            if (MeshParts.Count > 0)
                RegisterMeshParts();
            else
                _meshPartIds = null;
            if (oldPartCount > MeshParts.Count)
                MeshRegistry.Unregister(this, MeshParts.Count);

            GeometryVersion++;
            Engine.Device.DeferDispose(source);
        }

        public int IndexCount => _indexCount;
        public int VertexCount => _vertexCount;
        
        /// <summary>
        /// Raw indexed draw. Does NOT set any push constants — caller is responsible for all root signature state.
        /// </summary>
        public void DrawIndexed(ID3D12GraphicsCommandList commandList)
        {
             if (_indexCount > 0)
             {
                 commandList.IASetIndexBuffer(_indexBufferView);
                 commandList.DrawIndexedInstanced((uint)_indexCount, 1, 0, 0, 0);
             }
             else
             {
                 commandList.DrawInstanced((uint)_vertexCount, 1, 0, 0);
             }
        }
        
        /// <summary>
        /// Raw instanced indexed draw. Does NOT set any push constants — caller is responsible for all root signature state.
        /// </summary>
        public void DrawIndexedInstanced(ID3D12GraphicsCommandList commandList, int instanceCount)
        {
             if (_indexCount > 0) 
             { 
                 commandList.IASetIndexBuffer(_indexBufferView); 
                 commandList.DrawIndexedInstanced((uint)_indexCount, (uint)instanceCount, 0, 0, 0); 
             }
             else 
             { 
                 commandList.DrawInstanced((uint)_vertexCount, (uint)instanceCount, 0, 0); 
             }
        }
        
        /// <summary>
        /// Backwards-compatible Draw (single instance). Does NOT set push constants.
        /// </summary>
        public void Draw(ID3D12GraphicsCommandList commandList) => DrawIndexed(commandList);
        
        public void CreateBoneWeightBuffer(GraphicsDevice device)
        {
            if (BoneWeights == null || BoneWeights.Length == 0) return;
            
            _boneWeightBuffer = CreateBuffer(device, BoneWeights);
            BoneWeightBufferIndex = device.AllocateBindlessIndex();
            CreateStructuredBufferSRV(device, _boneWeightBuffer, (uint)BoneWeights.Length,
                (uint)System.Runtime.InteropServices.Marshal.SizeOf<BoneWeight>(), BoneWeightBufferIndex);
        }

        public void Dispose()
        {
            var device = Engine.Device;

            // Free MeshRegistry slots so they can be reused
            MeshRegistry.Unregister(this);

            _posBuffer?.Dispose();
            _normBuffer?.Dispose();
            _uvBuffer?.Dispose();
            _indexBuffer?.Dispose();
            _boneWeightBuffer?.Dispose();
            _tanBuffer?.Dispose();

            // Release bindless descriptor slots
            if (device != null)
            {
                if (PosBufferIndex > 0) device.ReleaseBindlessIndex(PosBufferIndex);
                if (NormBufferIndex > 0) device.ReleaseBindlessIndex(NormBufferIndex);
                if (UVBufferIndex > 0) device.ReleaseBindlessIndex(UVBufferIndex);
                if (IndexBufferIndex > 0) device.ReleaseBindlessIndex(IndexBufferIndex);
                if (BoneWeightBufferIndex > 0) device.ReleaseBindlessIndex(BoneWeightBufferIndex);
                if (TanBufferIndex > 0) device.ReleaseBindlessIndex(TanBufferIndex);
            }
        }

        // ============================================================
        // Terrain Patch Meshes — ported from Apex Mesh.Grid.cs
        // 33×33 grid, vertices centered at origin (-16..+16)
        // Shader expects: uv = (pos.xz + 16) * (1/32)
        // ============================================================

        private const int PatchNum = 33;
        private const float PatchOffset = (float)(PatchNum - 1) / 2; // 16

        private static void GeneratePatchVertices(out Vector3[] verts, out Vector3[] norms, out Vector2[] uvs)
        {
            int count = PatchNum * PatchNum;
            verts = new Vector3[count];
            norms = new Vector3[count];
            uvs = new Vector2[count];

            for (int z = 0; z < PatchNum; z++)
            {
                for (int x = 0; x < PatchNum; x++)
                {
                    int index = z * PatchNum + x;
                    verts[index] = new Vector3(x - PatchOffset, 0, z - PatchOffset);
                    norms[index] = Vector3.UnitY;
                    uvs[index] = new Vector2(x, z) / (PatchNum - 1);
                }
            }
        }

        private static Mesh BuildPatch(GraphicsDevice device, uint[] indices)
        {
            GeneratePatchVertices(out var verts, out var norms, out var uvs);
            var mesh = new Mesh(device, verts, norms, uvs, indices);
            // XZ: vertices span [-PatchOffset, +PatchOffset] = [-16, +16]
            // Y: 0 in mesh space; actual height range set by Terrain.Awake via SetPatchBounds()
            mesh.BoundingBox = new BoundingBox(
                new Vector3(-PatchOffset, 0, -PatchOffset),
                new Vector3(PatchOffset, 0, PatchOffset));
            mesh.MeshParts.Add(new MeshPart { NumIndices = indices.Length, BoundingBox = mesh.BoundingBox, BoundingSphere = mesh.LocalBoundingSphere });
            return mesh;
        }

        public static Mesh CreatePatch(GraphicsDevice device)
        {
            var indices = new uint[(PatchNum - 1) * (PatchNum - 1) * 6];
            int id = 0;

            for (int y = 0; y < PatchNum - 1; y++)
            {
                for (int x = 0; x < PatchNum - 1; x++)
                {
                    uint v0 = (uint)(x + y * PatchNum);
                    uint v1 = (uint)((x + 1) + y * PatchNum);
                    uint v2 = (uint)(x + (y + 1) * PatchNum);
                    uint v3 = (uint)((x + 1) + (y + 1) * PatchNum);

                    indices[id++] = v2; indices[id++] = v1; indices[id++] = v0;
                    indices[id++] = v2; indices[id++] = v3; indices[id++] = v1;
                }
            }

            return BuildPatch(device, indices);
        }

        public static Mesh CreateCube(GraphicsDevice device, float size)
        {
            float s = size * 0.5f;
            // 24 verts (4 per face, unique normals)
            Vector3[] verts = {
                // Front face (Z-)
                new(-s, s,-s), new( s, s,-s), new(-s,-s,-s), new( s,-s,-s),
                // Back face (Z+)
                new( s, s, s), new(-s, s, s), new( s,-s, s), new(-s,-s, s),
                // Top face (Y+)
                new(-s, s, s), new( s, s, s), new(-s, s,-s), new( s, s,-s),
                // Bottom face (Y-)
                new(-s,-s,-s), new( s,-s,-s), new(-s,-s, s), new( s,-s, s),
                // Left face (X-)
                new(-s, s, s), new(-s, s,-s), new(-s,-s, s), new(-s,-s,-s),
                // Right face (X+)
                new( s, s,-s), new( s, s, s), new( s,-s,-s), new( s,-s, s),
            };
            Vector3[] norms = {
                -Vector3.UnitZ, -Vector3.UnitZ, -Vector3.UnitZ, -Vector3.UnitZ,
                 Vector3.UnitZ,  Vector3.UnitZ,  Vector3.UnitZ,  Vector3.UnitZ,
                 Vector3.UnitY,  Vector3.UnitY,  Vector3.UnitY,  Vector3.UnitY,
                -Vector3.UnitY, -Vector3.UnitY, -Vector3.UnitY, -Vector3.UnitY,
                -Vector3.UnitX, -Vector3.UnitX, -Vector3.UnitX, -Vector3.UnitX,
                 Vector3.UnitX,  Vector3.UnitX,  Vector3.UnitX,  Vector3.UnitX,
            };
            Vector2[] uvs = {
                // Front face
                new(0,0), new(1,0), new(0,1), new(1,1),
                // Back face
                new(0,0), new(1,0), new(0,1), new(1,1),
                // Top face
                new(0,0), new(1,0), new(0,1), new(1,1),
                // Bottom face
                new(0,0), new(1,0), new(0,1), new(1,1),
                // Left face
                new(0,0), new(1,0), new(0,1), new(1,1),
                // Right face
                new(0,0), new(1,0), new(0,1), new(1,1),
            };
            uint[] indices = {
                0,1,2, 2,1,3,     // Front
                4,5,6, 6,5,7,     // Back
                8,9,10, 10,9,11,  // Top
                12,13,14, 14,13,15, // Bottom
                16,17,18, 18,17,19, // Left
                20,21,22, 22,21,23, // Right
            };
            var mesh = new Mesh(device, verts, norms, uvs, indices);
            mesh.BoundingBox = new BoundingBox(new Vector3(-s, -s, -s), new Vector3(s, s, s));
            mesh.MeshParts.Add(new MeshPart { NumIndices = indices.Length, BoundingBox = mesh.BoundingBox, BoundingSphere = mesh.LocalBoundingSphere });
            return mesh;
        }
        
        public static Mesh CreateSphere(GraphicsDevice device, float radius, int slices, int stacks)
        {
            int vertCount = (slices + 1) * (stacks + 1);
            var verts = new Vector3[vertCount];
            var norms = new Vector3[vertCount];
            var uvs = new Vector2[vertCount];
            int vi = 0;
            for (int stack = 0; stack <= stacks; stack++)
            {
                float phi = MathF.PI * stack / stacks;
                for (int slice = 0; slice <= slices; slice++)
                {
                    float theta = 2 * MathF.PI * slice / slices;
                    float x = MathF.Sin(phi) * MathF.Cos(theta);
                    float y = MathF.Cos(phi);
                    float z2 = MathF.Sin(phi) * MathF.Sin(theta);
                    norms[vi] = new Vector3(x, y, z2);
                    verts[vi] = norms[vi] * radius;
                    uvs[vi] = new Vector2((float)slice / slices, (float)stack / stacks);
                    vi++;
                }
            }
            int idxCount = slices * stacks * 6;
            var indices = new uint[idxCount];
            int ii = 0;
            for (int stack = 0; stack < stacks; stack++)
            {
                for (int slice = 0; slice < slices; slice++)
                {
                    uint a = (uint)(stack * (slices + 1) + slice);
                    uint b = a + 1;
                    uint c = (uint)((stack + 1) * (slices + 1) + slice);
                    uint d = c + 1;
                    indices[ii++] = a; indices[ii++] = b; indices[ii++] = c;
                    indices[ii++] = b; indices[ii++] = d; indices[ii++] = c;
                }
            }
            var mesh = new Mesh(device, verts, norms, uvs, indices);
            mesh.BoundingBox = new BoundingBox(new Vector3(-radius, -radius, -radius), new Vector3(radius, radius, radius));
            mesh.MeshParts.Add(new MeshPart { NumIndices = indices.Length, BoundingBox = mesh.BoundingBox, BoundingSphere = mesh.LocalBoundingSphere });
            return mesh;
        }

        public static Mesh CreateQuad(GraphicsDevice device)
        {
            float size = 1.0f;
            Vector3[] verts = {
                new(-size, size, 0), new(size, size, 0),
                new(-size,-size, 0), new(size,-size, 0)
            };
            Vector3[] norms = { Vector3.UnitZ, Vector3.UnitZ, Vector3.UnitZ, Vector3.UnitZ };
            Vector2[] uvs = { new(0,0), new(1,0), new(0,1), new(1,1) };
            uint[] indices = { 0, 1, 2, 2, 1, 3 };
            var mesh = new Mesh(device, verts, norms, uvs, indices);
            mesh.BoundingBox = new BoundingBox(new Vector3(-size, -size, 0), new Vector3(size, size, 0));
            mesh.MeshParts.Add(new MeshPart { NumIndices = indices.Length, BoundingBox = mesh.BoundingBox, BoundingSphere = mesh.LocalBoundingSphere });
            return mesh;
        }

        /// <summary>
        /// Create a cylinder along the Y axis from y=0 to y=height.
        /// </summary>
        public static Mesh CreateCylinder(GraphicsDevice device, float radius, float height, int slices)
        {
            int vertCount = (slices + 1) * 2;
            var verts = new Vector3[vertCount];
            var norms = new Vector3[vertCount];
            var uvs = new Vector2[vertCount];

            for (int i = 0; i <= slices; i++)
            {
                float theta = 2 * MathF.PI * i / slices;
                float x = MathF.Cos(theta) * radius;
                float z = MathF.Sin(theta) * radius;
                var normal = Vector3.Normalize(new Vector3(x, 0, z));

                // Bottom ring
                verts[i] = new Vector3(x, 0, z);
                norms[i] = normal;
                uvs[i] = new Vector2((float)i / slices, 0);

                // Top ring
                verts[i + slices + 1] = new Vector3(x, height, z);
                norms[i + slices + 1] = normal;
                uvs[i + slices + 1] = new Vector2((float)i / slices, 1);
            }

            var indices = new uint[slices * 6];
            int ii = 0;
            for (int i = 0; i < slices; i++)
            {
                uint bl = (uint)i;
                uint br = (uint)(i + 1);
                uint tl = (uint)(i + slices + 1);
                uint tr = (uint)(i + slices + 2);
                indices[ii++] = bl; indices[ii++] = tl; indices[ii++] = br;
                indices[ii++] = br; indices[ii++] = tl; indices[ii++] = tr;
            }

            var mesh = new Mesh(device, verts, norms, uvs, indices);
            mesh.BoundingBox = new BoundingBox(new Vector3(-radius, 0, -radius), new Vector3(radius, height, radius));
            mesh.MeshParts.Add(new MeshPart { NumIndices = indices.Length, BoundingBox = mesh.BoundingBox, BoundingSphere = mesh.LocalBoundingSphere });
            return mesh;
        }

        /// <summary>
        /// Create a cone along the Y axis from y=0 (base) to y=height (tip).
        /// </summary>
        public static Mesh CreateCone(GraphicsDevice device, float radius, float height, int slices)
        {
            // Vertices: base ring + tip + base center
            int vertCount = slices + 2;
            var verts = new Vector3[vertCount];
            var norms = new Vector3[vertCount];
            var uvs = new Vector2[vertCount];

            // Slope normal factor
            float slopeLen = MathF.Sqrt(radius * radius + height * height);
            float ny = radius / slopeLen;
            float nr = height / slopeLen;

            for (int i = 0; i < slices; i++)
            {
                float theta = 2 * MathF.PI * i / slices;
                float x = MathF.Cos(theta);
                float z = MathF.Sin(theta);
                verts[i] = new Vector3(x * radius, 0, z * radius);
                norms[i] = Vector3.Normalize(new Vector3(x * nr, ny, z * nr));
                uvs[i] = new Vector2((float)i / slices, 0);
            }

            // Tip vertex
            verts[slices] = new Vector3(0, height, 0);
            norms[slices] = Vector3.UnitY;
            uvs[slices] = new Vector2(0.5f, 1);

            // Base center
            verts[slices + 1] = Vector3.Zero;
            norms[slices + 1] = -Vector3.UnitY;
            uvs[slices + 1] = new Vector2(0.5f, 0);

            // Side triangles + base triangles
            var indices = new uint[slices * 6];
            int ii = 0;
            uint tipIdx = (uint)slices;
            uint centerIdx = (uint)(slices + 1);
            for (int i = 0; i < slices; i++)
            {
                uint cur = (uint)i;
                uint next = (uint)((i + 1) % slices);
                // Side
                indices[ii++] = cur; indices[ii++] = tipIdx; indices[ii++] = next;
                // Base
                indices[ii++] = next; indices[ii++] = centerIdx; indices[ii++] = cur;
            }

            var mesh = new Mesh(device, verts, norms, uvs, indices);
            mesh.BoundingBox = new BoundingBox(new Vector3(-radius, 0, -radius), new Vector3(radius, height, radius));
            mesh.MeshParts.Add(new MeshPart { NumIndices = indices.Length, BoundingBox = mesh.BoundingBox, BoundingSphere = mesh.LocalBoundingSphere });
            return mesh;
        }

        /// <summary>
        /// Load an OBJ file from disk. Supports v/vn/vt/f (v/vt/vn format) and g groups as MeshParts.
        /// Ported from Apex OBJReader.
        /// </summary>
        public static Mesh LoadOBJ(GraphicsDevice device, string path, float scale = 1f)
        {
            var culture = System.Globalization.CultureInfo.InvariantCulture;
            string data = System.IO.File.ReadAllText(path);
            string[] lines = data.Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries);

            var positions = new List<Vector3>();
            var normals = new List<Vector3>();
            var uvs = new List<Vector2>();

            var vertexList = new List<Vector3>();
            var normalList = new List<Vector3>();
            var uvList = new List<Vector2>();
            var indexList = new List<uint>();

            var parts = new List<MeshPart>();
            var part = new MeshPart();
            parts.Add(part);

            int newVertices = 0;
            int baseIndex = 0;
            var hashMap = new Dictionary<string, int>();

            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);

            foreach (string line in lines)
            {
                if (line.Length == 0) continue;
                string[] cl = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                if (line.StartsWith("v "))
                {
                    var pos = new Vector3(
                        float.Parse(cl[1], culture),
                        float.Parse(cl[2], culture),
                        float.Parse(cl[3], culture)) * scale;
                    positions.Add(pos);
                    min = Vector3.Min(min, pos);
                    max = Vector3.Max(max, pos);
                }
                else if (line.StartsWith("vn "))
                {
                    normals.Add(new Vector3(
                        float.Parse(cl[1], culture),
                        float.Parse(cl[2], culture),
                        float.Parse(cl[3], culture)));
                }
                else if (line.StartsWith("vt "))
                {
                    uvs.Add(new Vector2(
                        float.Parse(cl[1], culture),
                        float.Parse(cl[2], culture)));
                }
                else if (line.StartsWith("g "))
                {
                    baseIndex += part.NumIndices;
                    part = new MeshPart { BaseIndex = baseIndex, Name = line.Substring(2) };
                    parts.Add(part);
                }
                else if (line.StartsWith("f "))
                {
                    int num = cl.Length - 1;
                    var vertHash = new List<string>(num);

                    for (int i = 1; i < cl.Length; i++)
                    {
                        string[] tri = cl[i].Split('/');
                        int pv = int.Parse(tri[0]) - 1;
                        int pvt = tri.Length > 1 && tri[1].Length > 0 ? int.Parse(tri[1]) - 1 : 0;
                        int pvn = tri.Length > 2 ? int.Parse(tri[2]) - 1 : 0;

                        string hash = $"{pv}_{pvt}_{pvn}";
                        vertHash.Add(hash);

                        if (!hashMap.ContainsKey(hash))
                        {
                            hashMap[hash] = newVertices;
                            vertexList.Add(positions[pv]);
                            normalList.Add(pvn < normals.Count ? normals[pvn] : Vector3.UnitY);
                            uvList.Add(pvt < uvs.Count ? uvs[pvt] : Vector2.Zero);
                            newVertices++;
                        }
                    }

                    if (num == 3)
                    {
                        indexList.Add((uint)hashMap[vertHash[2]]);
                        indexList.Add((uint)hashMap[vertHash[0]]);
                        indexList.Add((uint)hashMap[vertHash[1]]);
                        part.NumIndices += 3;
                    }
                    else if (num == 4)
                    {
                        indexList.Add((uint)hashMap[vertHash[2]]);
                        indexList.Add((uint)hashMap[vertHash[3]]);
                        indexList.Add((uint)hashMap[vertHash[0]]);
                        indexList.Add((uint)hashMap[vertHash[2]]);
                        indexList.Add((uint)hashMap[vertHash[0]]);
                        indexList.Add((uint)hashMap[vertHash[1]]);
                        part.NumIndices += 6;
                    }
                    else if (num > 4)
                    {
                        for (int r = 0; r < num - 2; r++)
                        {
                            indexList.Add((uint)hashMap[vertHash[0]]);
                            indexList.Add((uint)hashMap[vertHash[r + 1]]);
                            indexList.Add((uint)hashMap[vertHash[r + 2]]);
                            part.NumIndices += 3;
                        }
                    }
                }
            }

            // Remove empty parts
            parts.RemoveAll(p => p.NumIndices <= 0);

            var mesh = new Mesh(device, vertexList.ToArray(), normalList.ToArray(), uvList.ToArray(), indexList.ToArray());
            mesh.BoundingBox = new BoundingBox(min, max);
            mesh.MeshParts.AddRange(parts);
            return mesh;
        }

    }
}
