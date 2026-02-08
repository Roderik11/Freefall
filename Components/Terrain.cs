using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;
using Freefall.Graphics;
using Freefall.Base;

namespace Freefall.Components
{
    public class Terrain : Freefall.Base.Component, IUpdate, IDraw
    {
        public static bool Wireframe { get; set; } = false;
        
        public class TextureLayer
        {
            public Texture Diffuse;
            public Texture Normals;
            public Vector2 Tiling = Vector2.One;
        }

        public enum PatchType
        {
            Default,
            N, E, S, W,
            NE, NW, SE, SW
        }

        public class NodePayload
        {
            public PatchType Type;
        }

        [StructLayout(LayoutKind.Sequential, Pack = 16)]
        struct TerrainPatch
        {
            public Matrix4x4 Transform;
            public Vector4 Rect;
            public Vector2 Level;
            public Vector2 Padding;
        }

        class Bucket
        {
            public int[] drawCounts = new int[CommandBuffer.FrameCount];
            public Mesh mesh;
            public ID3D12Resource[] instanceBuffers = new ID3D12Resource[CommandBuffer.FrameCount];
            public ID3D12Resource[] uploadBuffers = new ID3D12Resource[CommandBuffer.FrameCount];
            public int[] bufferSizes = new int[CommandBuffer.FrameCount];
            public BufferTexture?[] textures = new BufferTexture[CommandBuffer.FrameCount];
            public bool[] isDirty = new bool[CommandBuffer.FrameCount]; // Track if buffer needs update
            private TerrainPatch[][] arrays = new TerrainPatch[CommandBuffer.FrameCount][];

            public Bucket(Mesh mesh)
            {
                this.mesh = mesh;
                for (int i = 0; i < CommandBuffer.FrameCount; i++)
                {
                    arrays[i] = new TerrainPatch[128];
                    isDirty[i] = true; // Initially dirty
                }
            }

             public void Add(Matrix4x4 matrix, Vector4 rect, int level, int frameIndex)
            {
                if (drawCounts[frameIndex] >= arrays[frameIndex].Length)
                {
                    Array.Resize(ref arrays[frameIndex], arrays[frameIndex].Length * 2);
                }

                arrays[frameIndex][drawCounts[frameIndex]] = new TerrainPatch
                {
                    Transform = matrix,
                    Rect = rect,
                    Level = new Vector2(level, 0),
                    Padding = Vector2.Zero
                };
                drawCounts[frameIndex]++;
                isDirty[frameIndex] = true; // Mark as dirty when data changes
            }

            public void Clear(int frameIndex)
            {
                drawCounts[frameIndex] = 0;
                isDirty[frameIndex] = true; // Mark as dirty when cleared
            }

            /// <summary>
            /// Draw this bucket's mesh instanced, iterating MeshParts with proper BaseIndex/BaseVertex.
            /// Matches Apex Terrain.Bucket.Draw pattern — the only difference is bindless descriptor heap
            /// instead of bound vertex buffers.
            /// </summary>
            public void Draw(ID3D12GraphicsCommandList commandList, int drawCount)
            {
                if (drawCount == 0) return;

                commandList.IASetIndexBuffer(mesh.IndexBufferView);

                foreach (var part in mesh.MeshParts)
                {
                    commandList.DrawIndexedInstanced(
                        (uint)part.NumIndices,
                        (uint)drawCount,
                        (uint)part.BaseIndex,
                        part.BaseVertex,
                        0);
                }
            }

            public void UpdateBuffer(GraphicsDevice device, ID3D12GraphicsCommandList commandList, int frameIndex)
            {
                int count = drawCounts[frameIndex];
                if (count == 0) return;
                
                // Skip update if data hasn't changed
                if (!isDirty[frameIndex]) return;
                
                int requiredSize = count * Marshal.SizeOf<TerrainPatch>();
                
                // Only recreate if current buffer is too small
                if (instanceBuffers[frameIndex] == null || bufferSizes[frameIndex] < requiredSize)
                {
                    instanceBuffers[frameIndex]?.Dispose();
                    uploadBuffers[frameIndex]?.Dispose();
                    textures[frameIndex]?.Dispose();

                    // Grow slightly to avoid frequent reallocations
                    bufferSizes[frameIndex] = (int)(requiredSize * 1.5f);
                    if (bufferSizes[frameIndex] < 1024) bufferSizes[frameIndex] = 1024; // Min size

                    instanceBuffers[frameIndex] = device.NativeDevice.CreateCommittedResource(
                        new HeapProperties(HeapType.Default),
                        HeapFlags.None,
                        ResourceDescription.Buffer((ulong)bufferSizes[frameIndex]),
                        ResourceStates.CopyDest,
                        null);

                    uploadBuffers[frameIndex] = device.NativeDevice.CreateCommittedResource(
                        new HeapProperties(HeapType.Upload),
                        HeapFlags.None,
                        ResourceDescription.Buffer((ulong)bufferSizes[frameIndex]),
                        ResourceStates.GenericRead,
                        null);
                    
                    textures[frameIndex] = new BufferTexture(device, instanceBuffers[frameIndex], bufferSizes[frameIndex] / Marshal.SizeOf<TerrainPatch>(), Marshal.SizeOf<TerrainPatch>());
                }

                unsafe
                {
                    if (uploadBuffers[frameIndex] == null) return;
                    void* pData;
                    uploadBuffers[frameIndex].Map(0, null, &pData);
                    fixed(void* pSrc = arrays[frameIndex])
                    {
                        System.Runtime.CompilerServices.Unsafe.CopyBlock(pData, pSrc, (uint)requiredSize);
                    }
                    uploadBuffers[frameIndex].Unmap(0);
                }

                commandList.ResourceBarrierTransition(instanceBuffers[frameIndex], ResourceStates.NonPixelShaderResource | ResourceStates.PixelShaderResource, ResourceStates.CopyDest);
                commandList.CopyBufferRegion(instanceBuffers[frameIndex], 0, uploadBuffers[frameIndex], 0, (ulong)requiredSize);
                commandList.ResourceBarrierTransition(instanceBuffers[frameIndex], ResourceStates.CopyDest, ResourceStates.NonPixelShaderResource | ResourceStates.PixelShaderResource);
            
                // Clear dirty flag after update
                isDirty[frameIndex] = false;
            }
        }

        private Frustum frustum;
        private float sizeFactor;

        private readonly int patchSize = 32;
        private QuadTreeNode quadtree = null!;
        private readonly Dictionary<PatchType, Mesh> patches = new Dictionary<PatchType, Mesh>();
        private readonly List<QuadTreeNode> nodesToRender = new List<QuadTreeNode>();
        private readonly Dictionary<PatchType, Bucket> buckets = new Dictionary<PatchType, Bucket>();

        public Texture Heightmap = null!;
        public Material Material = null!;

        public Vector2 TerrainSize = new Vector2(4096 * 2, 4096 * 2);
        public float MaxHeight = 1024;

        public List<TextureLayer> Layers = new List<TextureLayer>();

        // Splatmap textures - individual for loading, then bundled into array
        public Texture?[] ControlMaps = new Texture?[4]; // 4 control maps, each RGBA = 4 layers = 16 total
        
        // Texture Arrays for efficient GPU sampling (like Spark)
        private Texture? ControlMapsArray; // Texture2DArray with 4 slices
        private Texture? DiffuseMapsArray; // Texture2DArray with up to 16 slices
        private Texture? NormalMapsArray;  // Texture2DArray with up to 16 slices
        
        private Vector4[] LayerTiling = new Vector4[32];
        
        // Cached list to avoid per-frame allocations
        private readonly List<BucketSnapshot> _drawSnapshotsCache = new List<BucketSnapshot>();

        public float[,] HeightField { get; set; } = null!;

        protected override void Awake()
        {
// Quadtree will be initialized in Update to ensure TerrainSize is correct

            var device = Engine.Device;

            patches.Add(PatchType.Default, Mesh.CreatePatch(device));
            patches.Add(PatchType.N, Mesh.PatchN(device));
            patches.Add(PatchType.E, Mesh.PatchE(device));
            patches.Add(PatchType.S, Mesh.PatchS(device));
            patches.Add(PatchType.W, Mesh.PatchW(device));
            patches.Add(PatchType.NW, Mesh.PatchNW(device));
            patches.Add(PatchType.NE, Mesh.PatchNE(device));
            patches.Add(PatchType.SW, Mesh.PatchSW(device));
            patches.Add(PatchType.SE, Mesh.PatchSE(device));
            
            Debug.Log("[Terrain] Awake");
        }

        private bool _initialized = false;

        private void CreateControlMaps(GraphicsDevice device)
        {
            // Dispose old textures
            for (int i = 0; i < ControlMaps.Length; i++)
            {
                ControlMaps[i]?.Dispose();
                ControlMaps[i] = null;
            }

            var filenames = new[] {
                "Resources/Terrain/Terrain_splatmap_0.dds",
                "Resources/Terrain/Terrain_splatmap_1.dds",
                "Resources/Terrain/Terrain_splatmap_2.dds",
                "Resources/Terrain/Terrain_splatmap_3.dds"
            };

            for (int i = 0; i < filenames.Length && i < ControlMaps.Length; i++)
            {
                if (System.IO.File.Exists(filenames[i]))
                {
                    ControlMaps[i] = new Texture(device, filenames[i]);
                    Debug.Log($"[Terrain] Loaded Splatmap {filenames[i]}: {ControlMaps[i].Native.Description.Width}x{ControlMaps[i].Native.Description.Height} {ControlMaps[i].Native.Description.Format}");
                }
                else
                {
                    Debug.LogError("Terrain", $"Splatmap not found: {filenames[i]}");
                    // Fallback to black for missing files
                    ControlMaps[i] = Texture.CreateFromData(device, 1, 1, new byte[] {0,0,0,0}); 
                }
            }
            Debug.Log($"[Terrain] Loaded {ControlMaps.Length} individual ControlMaps (bindless)");
        }

        private void SetupLayerTiling()
        {
            // Calculate tiling values and create texture arrays
            var diffuseList = new List<Texture>();
            var normalList = new List<Texture>();
            
            for (int i = 0; i < Layers.Count && i < LayerTiling.Length; i++)
            {
                var layer = Layers[i];
                // "Tiling" in Spark means "Size of one tile in meters".
                // So repeats = TerrainSize / TileSize.
                if (layer.Tiling.X != 0 && layer.Tiling.Y != 0)
                     LayerTiling[i] = new Vector4(TerrainSize.X / layer.Tiling.X, TerrainSize.Y / layer.Tiling.Y, 0, 0);
                else
                     LayerTiling[i] = Vector4.One;
                
                if (layer.Diffuse != null) diffuseList.Add(layer.Diffuse);
                if (layer.Normals != null) normalList.Add(layer.Normals);
            }
            
            // Create texture arrays for efficient GPU sampling
            var device = Engine.Device;
            if (diffuseList.Count > 0)
            {
                DiffuseMapsArray = Texture.CreateTexture2DArray(device, diffuseList);
                Debug.Log($"[Terrain] Created DiffuseMapsArray with {diffuseList.Count} slices");
            }
            if (normalList.Count > 0)
            {
                NormalMapsArray = Texture.CreateTexture2DArray(device, normalList);
                Debug.Log($"[Terrain] Created NormalMapsArray with {normalList.Count} slices");
            }
            
            // Create control map array
            var controlList = new List<Texture>();
            for (int i = 0; i < ControlMaps.Length; i++)
            {
                if (ControlMaps[i] != null) controlList.Add(ControlMaps[i]!);
            }
            if (controlList.Count > 0)
            {
                ControlMapsArray = Texture.CreateTexture2DArray(device, controlList);
                Debug.Log($"[Terrain] Created ControlMapsArray with {controlList.Count} slices");
            }
        }

        public void Update()
        {
            if (!_initialized)
            {
                var extent = TerrainSize / 2;
                quadtree = new QuadTreeNode(new Vector3(extent.X, 0, extent.Y), new Vector3(extent.X, MaxHeight * 10, extent.Y));
                sizeFactor = quadtree.Extents.X / patchSize;

                if (Layers.Count > 0)
                {
                    var device = Engine.Device;
                    CreateControlMaps(device);
                    SetupLayerTiling();
                }
                _initialized = true;
            }

            if (Camera.Main == null) 
            {
                // Debug.Log("[Terrain] Camera.Main is null");
                return;
            }
            // Debug.Log("[Terrain] Update");
            quadtree.Update(Camera.Main.Transform.Position - Transform.Position);

            
            // Draw() is now called via IDraw interface during Draw phase
        }

        public void Draw()
        {
            var totalSw = System.Diagnostics.Stopwatch.StartNew();
            
            int frameIndex = Engine.FrameIndex % CommandBuffer.FrameCount;

            var quadtreeSw = System.Diagnostics.Stopwatch.StartNew();
            nodesToRender.Clear();
            frustum = new Frustum(Camera.Main.ViewProjection);
            quadtree.Render(CheckPatchVisible);
            nodesToRender.Sort((a, b) => b.Depth.CompareTo(a.Depth));
            quadtreeSw.Stop();

            var clearSw = System.Diagnostics.Stopwatch.StartNew();
            foreach (var pair in buckets)
                pair.Value.Clear(frameIndex);
            clearSw.Stop();

            // Reuse cached list to avoid per-frame allocation
            _drawSnapshotsCache.Clear();
            var device = Engine.Device;

            var prepareSw = System.Diagnostics.Stopwatch.StartNew();
            foreach (var node in nodesToRender)
            {
                PrepareNode(node);
                RenderPatch(node, frameIndex);
            }
            prepareSw.Stop();

            var snapshotSw = System.Diagnostics.Stopwatch.StartNew();
            // We'll perform buffer updates inside the draw callback on the main command list
            // to ensure proper synchronization with the draw calls.
            foreach (var pair in buckets)
            {
                if (pair.Value.drawCounts[frameIndex] > 0)
                {
                    _drawSnapshotsCache.Add(new BucketSnapshot { Bucket = pair.Value, DrawCount = pair.Value.drawCounts[frameIndex] });
                }
            }
            snapshotSw.Stop();
            
            totalSw.Stop();
            
            if (Engine.FrameIndex % 60 == 0)
            {
                Debug.Log($"[Terrain.Draw] Total: {totalSw.Elapsed.TotalMilliseconds:F2}ms | QuadTree: {quadtreeSw.Elapsed.TotalMilliseconds:F2}ms | Clear: {clearSw.Elapsed.TotalMilliseconds:F2}ms | Prepare: {prepareSw.Elapsed.TotalMilliseconds:F2}ms | Snapshot: {snapshotSw.Elapsed.TotalMilliseconds:F2}ms");
            }
            
            // Capture reference for lambda (avoid closure allocation)
            var snapshots = _drawSnapshotsCache;

            CommandBuffer.Enqueue(RenderPass.Opaque, (commandList) =>
            {
                var updateSw = System.Diagnostics.Stopwatch.StartNew();
                // First pass: update all buffers
                foreach (var snapshot in snapshots)
                {
                    snapshot.Bucket.UpdateBuffer(device, commandList, frameIndex);
                }
                updateSw.Stop();
                
                var drawSw = System.Diagnostics.Stopwatch.StartNew();
                DrawInternal(commandList, frameIndex, snapshots);
                drawSw.Stop();
                
                if (Engine.FrameIndex % 60 == 0)
                {
                    Debug.Log($"[Terrain.Render] UpdateBuffer: {updateSw.Elapsed.TotalMilliseconds:F2}ms | DrawInternal: {drawSw.Elapsed.TotalMilliseconds:F2}ms");
                }
            });
        }

        struct BucketSnapshot
        {
            public Bucket Bucket;
            public int DrawCount;
        }

        private void DrawInternal(ID3D12GraphicsCommandList commandList, int frameIndex, List<BucketSnapshot> snapshots)
        {
            if (Camera.Main == null || Camera.Main.Transform == null || Transform == null) return;
            if (snapshots.Count == 0) return;

            // Set shared Material Properties ONCE (not per bucket)
            Material.SetParameter("Time", Time.TotalTime);
            Material.SetParameter("View", Camera.Main.View);
            Material.SetParameter("Projection", Camera.Main.Projection);
            Material.SetParameter("ViewProjection", Camera.Main.View * Camera.Main.Projection);

            // Terrain cbuffer parameters
            Material.SetParameter("CameraPos", Camera.Main.Position);
            Material.SetParameter("HeightTexel", Heightmap != null ? 1.0f / Heightmap.Native.Description.Width : 1.0f / 1024.0f);
            Material.SetParameter("MaxHeight", MaxHeight);
            Material.SetParameter("TerrainSize", TerrainSize);

            // Bind textures ONCE - use unified GET_INDEX approach
            if (Heightmap != null) Material.SetTexture("HeightTex", Heightmap);
            if (ControlMapsArray != null) Material.SetTexture("ControlMaps", ControlMapsArray);
            if (DiffuseMapsArray != null) Material.SetTexture("DiffuseMaps", DiffuseMapsArray);
            if (NormalMapsArray != null) Material.SetTexture("NormalMaps", NormalMapsArray);

            // Layer tiling cbuffer
            Material.SetParameter("LayerTiling", LayerTiling);

            // Apply material state ONCE before all draws
            // This sets PSO, root signature, constant buffers, texture indices
            Material.Apply(commandList, Engine.Device);

            // Set topology — must be done explicitly before terrain draws.
            // The batch pipeline sets this later (line 334 of CommandBuffer.cs),
            // but terrain runs as a custom lambda BEFORE that.
            commandList.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleList);

            // Draw Buckets - only update per-bucket push constants
            foreach (var snapshot in snapshots)
            {
                var bucket = snapshot.Bucket;
                int drawCount = snapshot.DrawCount;

                // Update only per-bucket push constants (PosBuffer and TerrainData indices)
                // Slot 0: Push constants - we only need to update indices 0 and 1
                uint posBufferIdx = bucket.mesh.PosBufferIndex;
                uint terrainDataIdx = bucket.textures[frameIndex]?.BindlessIndex ?? 0;
                
                // Use SetGraphicsRoot32BitConstant for minimal overhead
                commandList.SetGraphicsRoot32BitConstant(0, posBufferIdx, 0);
                commandList.SetGraphicsRoot32BitConstant(0, terrainDataIdx, 1);
                
                bucket.Draw(commandList, drawCount);
            }
        }

        private void PrepareNode(QuadTreeNode node)
        {
            if (node.Payload == null) node.Payload = new NodePayload();
            NodePayload payload = node.Payload as NodePayload;
            payload.Type = GetPatchType(node);
        }

        private void CheckPatchVisible(QuadTreeNode node)
        {
            if (frustum.Intersects(node.Bounds) == ContainmentType.Disjoint)
                return;

            nodesToRender.Add(node);
        }

        private void RenderPatch(QuadTreeNode node, int frameIndex)
        {
            if (node.Payload is not NodePayload payload) return;

            float s = 2 / (float)Math.Pow(2, node.Depth) * sizeFactor;
            Matrix4x4 translation = Matrix4x4.CreateTranslation(node.Position);
            Matrix4x4 scale = Matrix4x4.CreateScale(s, 1f, s);
            Matrix4x4 m = scale * translation;

            Vector4 rect = Vector4.Zero;
            // Match Spark's UV calculation exactly
            rect.X = node.Bounds.Min.X / (quadtree.Extents.X * 2);
            rect.Y = node.Bounds.Max.Z / (quadtree.Extents.Z * 2);
            rect.Z = node.Bounds.Max.X / (quadtree.Extents.X * 2);
            rect.W = node.Bounds.Min.Z / (quadtree.Extents.Z * 2);

            if(!buckets.TryGetValue(payload.Type, out var bucket))
            {
                bucket = new Bucket(patches[payload.Type]);
                buckets.Add(payload.Type, bucket);
            }

            bucket.Add(m * Transform.WorldMatrix, rect, QuadTreeNode.MaxDepth - (node.Depth + 1), frameIndex);
            // Debug.Log($"[Terrain] Patch Added to {payload.Type}, Count: {bucket.drawCount}");
        }

        private PatchType GetPatchType(QuadTreeNode node)
        {
             if (node.Location == 0) // SW
            {
                var south = node.Neighbors[0];
                var west = node.Neighbors[3];

                bool left = west != null && west.Depth < node.Depth;
                bool bottom = south != null && south.Depth < node.Depth;

                if (left && bottom) return PatchType.SW;
                if (left) return PatchType.W;
                if (bottom) return PatchType.S;
            }
            else if (node.Location == 1) // SE
            {
                var south = node.Neighbors[0];
                var east = node.Neighbors[1];

                bool right = east != null && east.Depth == node.Depth - 1;
                bool bottom = south != null && south.Depth == node.Depth - 1;

                if (right && bottom) return PatchType.SE;
                if (right) return PatchType.E;
                if (bottom) return PatchType.S;
            }
            else if (node.Location == 2) // NW
            {
                var north = node.Neighbors[2];
                var west = node.Neighbors[3];

                bool left = west != null && west.Depth < node.Depth;
                bool top = north != null && north.Depth < node.Depth;

                if (left && top) return PatchType.NW;
                if (left) return PatchType.W;
                if (top) return PatchType.N;
            }
            else if (node.Location == 3) // NE
            {
                var north = node.Neighbors[2];
                var east = node.Neighbors[1];

                bool right = east != null && east.Depth == node.Depth - 1;
                bool top = north != null && north.Depth == node.Depth - 1;

                if (right && top) return PatchType.NE;
                if (right) return PatchType.E;
                if (top) return PatchType.N;
            }

            return PatchType.Default;
        }

        public float GetHeight(Vector3 position)
        {
            if (HeightField == null) return Transform?.Position.Y ?? 0;

            if (Transform == null) return 0;
            position -= Transform.Position;

            var dimx = HeightField.GetLength(0) - 1;
            var dimy = HeightField.GetLength(1) - 1;

            var fx = (dimx + 1) / TerrainSize.X;
            var fy = (dimy + 1) / TerrainSize.Y;

            position.X *= fx;
            position.Z *= fy;

            int x = (int)Math.Floor(position.X);
            int z = (int)Math.Floor(position.Z);

            if (x < 0 || x > dimx) return Transform.Position.Y;
            if (z < 0 || z > dimy) return Transform.Position.Y;

            float xf = position.X - x;
            float zf = position.Z - z;

            float h1 = HeightField[x, z];
            float h2 = HeightField[Math.Min(dimx, x + 1), z];
            float h3 = HeightField[x, Math.Min(dimy, z + 1)];
            
            float height = h1;
            height += (h2 - h1) * xf;
            height += (h3 - h1) * zf;

            return Transform.Position.Y + height * MaxHeight;
        }
    }
}
