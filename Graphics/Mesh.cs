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

    public class MeshPart
    {
        public string Name = string.Empty;
        public bool Enabled = true;
        public int BaseVertex;
        public int BaseIndex;
        public int NumIndices;
        public BoundingBox BoundingBox;
    }

    public partial class Mesh : Asset, IDisposable
    {
        private static volatile int _instanceCount;
        private readonly int _instanceId = Interlocked.Increment(ref _instanceCount);
        public int GetInstanceId() => _instanceId;
        
        public List<MeshPart> MeshParts { get; set; } = new List<MeshPart>();
        
        // Buffers
        private ID3D12Resource _posBuffer = null!;
        private VertexBufferView _posView;
        private ID3D12Resource _normBuffer = null!;
        private VertexBufferView _normView;
        private ID3D12Resource _uvBuffer = null!;
        private VertexBufferView _uvView;

        private int _vertexCount;
        private ID3D12Resource _indexBuffer = null!;
        public IndexBufferView IndexBufferView => _indexBufferView;
        private IndexBufferView _indexBufferView;
        private int _indexCount;

        // Bindless Indices
        public uint PosBufferIndex { get; private set; }
        public uint NormBufferIndex { get; private set; }
        public uint UVBufferIndex { get; private set; }
        public uint IndexBufferIndex { get; private set; }
        
        public BoundingBox BoundingBox { get; set; }
        
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
        public Bone[]? Bones { get; set; }
        public BoneWeight[]? BoneWeights { get; set; }
        public Matrix4x4 RootRotation { get; set; } = Matrix4x4.Identity;
        
        private ID3D12Resource? _boneWeightBuffer;
        public uint BoneWeightBufferIndex { get; private set; }

        private int[]? _meshPartIds;

        public Mesh() { }

        public Mesh(GraphicsDevice device, Vector3[] positions, Vector3[] normals, Vector2[] uvs, uint[] indices)
        {
            CreateBuffers(device, positions, normals, uvs, indices);
            MarkReady();
        }

        public Mesh(GraphicsDevice device, MeshData data)
        {
            CreateBuffers(device, data.Positions, data.Normals, data.UVs, data.Indices);
            MeshParts.AddRange(data.Parts);
            BoundingBox = data.BoundingBox;
            
            if (data.Bones != null)
                Bones = data.Bones;
            
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
            mesh.Bones = data.Bones;
            mesh.BoneWeights = data.BoneWeights;
            
            // For meshes, because they are structured buffers, we still create the Committed Resource on Main Thread
            // But we skip the *Upload* part here?
            // Wait, StreamingManager is built for Textures mostly.
            // For Meshes, we can use the same ring buffer logic, but we need 'RecordBufferUpload'.
            
            // To be safe and fast for this iteration:
            // We kept the synchronous buffer creation for Meshes in Stage 3 of the SceneLoader plan.
            // i.e., "Quick call to CreateCommittedResource".
            // So we can reuse CreateBuffers but we need to CHANGE it to NOT do the CopyQueueWait.
            
            mesh.CreateBuffersAsync(device, data.Positions, data.Normals, data.UVs, data.Indices);
            
            // Bone weights?
            if (data.BoneWeights != null && data.BoneWeights.Length > 0)
            {
                 // mesh.CreateBoneWeightBufferAsync(device);
            }

            return mesh;
        }

        private void CreateBuffers(GraphicsDevice device, Vector3[] positions, Vector3[] normals, Vector2[] uvs, uint[] indices)
        {
            // Legacy Synchronous Path
             _vertexCount = positions.Length;

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

            if (indices != null)
            {
                _indexCount = indices.Length;
                _indexBuffer = CreateBuffer(device, indices); // Note: CreateBuffer<uint>
                 _indexBufferView = new IndexBufferView { BufferLocation = _indexBuffer.GPUVirtualAddress, SizeInBytes = (uint)(indices.Length * 4), Format = Format.R32_UInt };
                 
                IndexBufferIndex = device.AllocateBindlessIndex();
                CreateStructuredBufferSRV(device, _indexBuffer, (uint)indices.Length, 4, IndexBufferIndex);
            }
        }
        
        private void CreateBuffersAsync(GraphicsDevice device, Vector3[] positions, Vector3[] normals, Vector2[] uvs, uint[] indices)
        {
            // 1. Create GPU Resources (Fast, no upload)
            _vertexCount = positions.Length;
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
        public int GetMeshPartId(int partIndex) { if (_meshPartIds == null) return -1; return _meshPartIds[partIndex]; }
        public void RegisterMeshParts() { if (MeshParts.Count == 0) return; _meshPartIds = new int[MeshParts.Count]; for (int i = 0; i < MeshParts.Count; i++) _meshPartIds[i] = MeshRegistry.Register(this, i); }
        
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
        public void BindPoseDifference(Mesh source)
        {
            for (int i = 0; i < source.Bones.Length; i++)
            {
                float len1 = source.Bones[i].BindPoseMatrix.Translation.Length();
                float len2 = Bones[i].BindPoseMatrix.Translation.Length();
                float scaleFactor = len2 / len1;
                Bones[i].ScaleFactor = scaleFactor > 0 ? scaleFactor : 1;
            }
        }
        public void Dispose() { _posBuffer?.Dispose(); _normBuffer?.Dispose(); _uvBuffer?.Dispose(); _indexBuffer?.Dispose(); }

        public static Mesh Quad { get; set; }

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
            mesh.MeshParts.Add(new MeshPart { NumIndices = indices.Length });
            // XZ: vertices span [-PatchOffset, +PatchOffset] = [-16, +16]
            // Y: 0 in mesh space; actual height range set by Terrain.Awake via SetPatchBounds()
            mesh.BoundingBox = new BoundingBox(
                new Vector3(-PatchOffset, 0, -PatchOffset),
                new Vector3(PatchOffset, 0, PatchOffset));
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

        public static Mesh PatchW(GraphicsDevice device)
        {
            var indices = new uint[(PatchNum - 1) * (PatchNum - 1) * 6];
            int id = 0;
            int flipy = 0;

            for (int y = 0; y < PatchNum - 1; y++)
            {
                flipy = 1 - flipy;
                for (int x = 0; x < PatchNum - 1; x++)
                {
                    int v0 = x + y * PatchNum;
                    int v1 = (x + 1) + y * PatchNum;
                    int v2 = x + (y + 1) * PatchNum;
                    int v3 = (x + 1) + (y + 1) * PatchNum;

                    if (x == 0)
                    {
                        if (flipy == 1) v2 = v0;
                        else v0 = x + (y - 1) * PatchNum;
                    }

                    indices[id++] = (uint)v2; indices[id++] = (uint)v1; indices[id++] = (uint)v0;
                    indices[id++] = (uint)v2; indices[id++] = (uint)v3; indices[id++] = (uint)v1;
                }
            }
            return BuildPatch(device, indices);
        }

        public static Mesh PatchE(GraphicsDevice device)
        {
            var indices = new uint[(PatchNum - 1) * (PatchNum - 1) * 6];
            int id = 0;
            int flipy = 0;

            for (int y = 0; y < PatchNum - 1; y++)
            {
                flipy = 1 - flipy;
                for (int x = 0; x < PatchNum - 1; x++)
                {
                    int v0 = x + y * PatchNum;
                    int v1 = (x + 1) + y * PatchNum;
                    int v2 = x + (y + 1) * PatchNum;
                    int v3 = (x + 1) + (y + 1) * PatchNum;

                    if (x == PatchNum - 2)
                    {
                        if (flipy == 1) v3 = (x + 1) + (y + 2) * PatchNum;
                        else v1 = v3;
                    }

                    indices[id++] = (uint)v2; indices[id++] = (uint)v1; indices[id++] = (uint)v0;
                    indices[id++] = (uint)v2; indices[id++] = (uint)v3; indices[id++] = (uint)v1;
                }
            }
            return BuildPatch(device, indices);
        }

        public static Mesh PatchN(GraphicsDevice device)
        {
            var indices = new uint[(PatchNum - 1) * (PatchNum - 1) * 6];
            int id = 0;
            int flipx = 0;

            for (int y = 0; y < PatchNum - 1; y++)
            {
                for (int x = 0; x < PatchNum - 1; x++)
                {
                    flipx = 1 - flipx;
                    int v0 = x + y * PatchNum;
                    int v1 = (x + 1) + y * PatchNum;
                    int v2 = x + (y + 1) * PatchNum;
                    int v3 = (x + 1) + (y + 1) * PatchNum;

                    if (y == PatchNum - 2)
                    {
                        if (flipx == 1) v3 = (x + 2) + (y + 1) * PatchNum;
                        else v2 = (x + 1) + (y + 1) * PatchNum;
                    }

                    indices[id++] = (uint)v2; indices[id++] = (uint)v1; indices[id++] = (uint)v0;
                    indices[id++] = (uint)v2; indices[id++] = (uint)v3; indices[id++] = (uint)v1;
                }
            }
            return BuildPatch(device, indices);
        }

        public static Mesh PatchS(GraphicsDevice device)
        {
            var indices = new uint[(PatchNum - 1) * (PatchNum - 1) * 6];
            int id = 0;
            int flipx = 0;

            for (int y = 0; y < PatchNum - 1; y++)
            {
                for (int x = 0; x < PatchNum - 1; x++)
                {
                    flipx = 1 - flipx;
                    int v0 = x + y * PatchNum;
                    int v1 = (x + 1) + y * PatchNum;
                    int v2 = x + (y + 1) * PatchNum;
                    int v3 = (x + 1) + (y + 1) * PatchNum;

                    if (y == 0)
                    {
                        if (flipx == 1) v1 = v0;
                        else v0 = (x - 1) + y * PatchNum;
                    }

                    indices[id++] = (uint)v2; indices[id++] = (uint)v1; indices[id++] = (uint)v0;
                    indices[id++] = (uint)v2; indices[id++] = (uint)v3; indices[id++] = (uint)v1;
                }
            }
            return BuildPatch(device, indices);
        }

        public static Mesh PatchNW(GraphicsDevice device)
        {
            var indices = new uint[(PatchNum - 1) * (PatchNum - 1) * 6];
            int id = 0;
            int flipy = 0;
            int flipx = 0;

            for (int y = 0; y < PatchNum - 1; y++)
            {
                flipy = 1 - flipy;
                for (int x = 0; x < PatchNum - 1; x++)
                {
                    flipx = 1 - flipx;
                    int v0 = x + y * PatchNum;
                    int v1 = (x + 1) + y * PatchNum;
                    int v2 = x + (y + 1) * PatchNum;
                    int v3 = (x + 1) + (y + 1) * PatchNum;

                    if (x == 0)
                    {
                        if (flipy == 1) v2 = v0;
                        else v0 = x + (y - 1) * PatchNum;

                        if (y == PatchNum - 2)
                        {
                            if (flipx == 1) v3 = (x + 2) + (y + 1) * PatchNum;
                            else v2 = (x + 1) + (y + 1) * PatchNum;
                        }
                    }
                    else if (y == PatchNum - 2)
                    {
                        if (flipx == 1) v3 = (x + 2) + (y + 1) * PatchNum;
                        else v2 = (x + 1) + (y + 1) * PatchNum;
                    }

                    indices[id++] = (uint)v2; indices[id++] = (uint)v1; indices[id++] = (uint)v0;
                    indices[id++] = (uint)v2; indices[id++] = (uint)v3; indices[id++] = (uint)v1;
                }
            }
            return BuildPatch(device, indices);
        }

        public static Mesh PatchNE(GraphicsDevice device)
        {
            var indices = new uint[(PatchNum - 1) * (PatchNum - 1) * 6];
            int id = 0;
            int flipy = 0;
            int flipx = 0;

            for (int y = 0; y < PatchNum - 1; y++)
            {
                flipy = 1 - flipy;
                for (int x = 0; x < PatchNum - 1; x++)
                {
                    flipx = 1 - flipx;
                    int v0 = x + y * PatchNum;
                    int v1 = (x + 1) + y * PatchNum;
                    int v2 = x + (y + 1) * PatchNum;
                    int v3 = (x + 1) + (y + 1) * PatchNum;

                    if (x == PatchNum - 2)
                    {
                        if (flipy == 1) v3 = (x + 1) + (y + 2) * PatchNum;
                        else v1 = v3;

                        if (y == PatchNum - 2)
                        {
                            if (flipx == 1) v3 = (x + 2) + (y + 1) * PatchNum;
                            else v2 = (x + 1) + (y + 1) * PatchNum;
                        }
                    }
                    else if (y == PatchNum - 2)
                    {
                        if (flipx == 1) v3 = (x + 2) + (y + 1) * PatchNum;
                        else v2 = (x + 1) + (y + 1) * PatchNum;
                    }

                    indices[id++] = (uint)v2; indices[id++] = (uint)v1; indices[id++] = (uint)v0;
                    indices[id++] = (uint)v2; indices[id++] = (uint)v3; indices[id++] = (uint)v1;
                }
            }
            return BuildPatch(device, indices);
        }

        public static Mesh PatchSW(GraphicsDevice device)
        {
            var indices = new uint[(PatchNum - 1) * (PatchNum - 1) * 6];
            int id = 0;
            int flipy = 0;
            int flipx = 0;

            for (int y = 0; y < PatchNum - 1; y++)
            {
                flipy = 1 - flipy;
                for (int x = 0; x < PatchNum - 1; x++)
                {
                    flipx = 1 - flipx;
                    int v0 = x + y * PatchNum;
                    int v1 = (x + 1) + y * PatchNum;
                    int v2 = x + (y + 1) * PatchNum;
                    int v3 = (x + 1) + (y + 1) * PatchNum;

                    if (x == 0)
                    {
                        if (flipy == 1) v2 = v0;
                        else v0 = x + (y - 1) * PatchNum;

                        if (y == 0)
                        {
                            if (flipx == 1) v1 = v0;
                            else v0 = (x - 1) + y * PatchNum;
                        }
                    }
                    else if (y == 0)
                    {
                        if (flipx == 1) v1 = v0;
                        else v0 = (x - 1) + y * PatchNum;
                    }

                    indices[id++] = (uint)v2; indices[id++] = (uint)v1; indices[id++] = (uint)v0;
                    indices[id++] = (uint)v2; indices[id++] = (uint)v3; indices[id++] = (uint)v1;
                }
            }
            return BuildPatch(device, indices);
        }

        public static Mesh PatchSE(GraphicsDevice device)
        {
            var indices = new uint[(PatchNum - 1) * (PatchNum - 1) * 6];
            int id = 0;
            int flipy = 0;
            int flipx = 0;

            for (int y = 0; y < PatchNum - 1; y++)
            {
                flipy = 1 - flipy;
                for (int x = 0; x < PatchNum - 1; x++)
                {
                    flipx = 1 - flipx;
                    int v0 = x + y * PatchNum;
                    int v1 = (x + 1) + y * PatchNum;
                    int v2 = x + (y + 1) * PatchNum;
                    int v3 = (x + 1) + (y + 1) * PatchNum;

                    if (x == PatchNum - 2)
                    {
                        if (flipy == 1) v3 = (x + 1) + (y + 2) * PatchNum;
                        else v1 = v3;

                        if (y == 0)
                        {
                            if (flipx == 1) v1 = v0;
                            else v0 = (x - 1) + y * PatchNum;
                        }
                    }
                    else if (y == 0)
                    {
                        if (flipx == 1) v1 = v0;
                        else v0 = (x - 1) + y * PatchNum;
                    }

                    indices[id++] = (uint)v2; indices[id++] = (uint)v1; indices[id++] = (uint)v0;
                    indices[id++] = (uint)v2; indices[id++] = (uint)v3; indices[id++] = (uint)v1;
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
            Vector2 z = Vector2.Zero;
            Vector2[] uvs = {
                z, z, z, z,  z, z, z, z,  z, z, z, z,
                z, z, z, z,  z, z, z, z,  z, z, z, z,
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
            mesh.MeshParts.Add(new MeshPart { NumIndices = indices.Length });
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
                    indices[ii++] = a; indices[ii++] = c; indices[ii++] = b;
                    indices[ii++] = b; indices[ii++] = c; indices[ii++] = d;
                }
            }
            var mesh = new Mesh(device, verts, norms, uvs, indices);
            mesh.BoundingBox = new BoundingBox(new Vector3(-radius, -radius, -radius), new Vector3(radius, radius, radius));
            mesh.MeshParts.Add(new MeshPart { NumIndices = indices.Length });
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
            mesh.MeshParts.Add(new MeshPart { NumIndices = indices.Length });
            return mesh;
        }


    }
}
