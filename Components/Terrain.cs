using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Mathematics;
using Freefall.Graphics;
using Freefall.Base;

namespace Freefall.Components
{
    public class Terrain : Freefall.Base.Component, IUpdate, IDraw, IHeightProvider
    {
        public static bool Wireframe { get; set; } = false;
        
        public class TextureLayer
        {
            public Texture Diffuse;
            public Texture Normals;
            public Vector2 Tiling = Vector2.One;
        }

        // PatchType enum is now in Freefall.Graphics namespace (shared with Landscape)

        public class NodePayload
        {
            public PatchType Type;
        }

        /// <summary>
        /// Per-instance terrain patch data — sent to GPU via the generic per-instance buffer system.
        /// Transform is in the global TransformBuffer (standard path).
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 16)]
        public struct TerrainPatchData
        {
            public Vector4 Rect;
            public Vector2 Level;
            public Vector2 Padding;
        }

        private Frustum frustum;
        private float sizeFactor;

        private readonly int patchSize = 32;
        private QuadTreeNode quadtree = null!;
        private readonly Dictionary<PatchType, Mesh> patches = new Dictionary<PatchType, Mesh>();
        private readonly List<QuadTreeNode> nodesToRender = new List<QuadTreeNode>();

        public Texture Heightmap = null!;
        public Material Material = null!;

        public Vector2 TerrainSize = new Vector2(4096 * 2, 4096 * 2);
        public float MaxHeight = 1024;

        public List<TextureLayer> Layers = new List<TextureLayer>();

        // Splatmap textures - individual for loading, then bundled into array
        public Texture?[] ControlMaps = new Texture?[4]; // 4 control maps, each RGBA = 4 layers = 16 total
        
        // Texture Arrays for efficient GPU sampling
        private Texture? ControlMapsArray;
        private Texture? DiffuseMapsArray;
        private Texture? NormalMapsArray;
        
        private Vector4[] LayerTiling = new Vector4[32];
        
        // Pooled TransformSlots and MaterialBlocks for terrain patches
        private const int MaxTerrainPatches = 1024;
        private int[] _transformSlots = null!;
        private MaterialBlock[] _blockPool = null!;
        private int _patchCount; // how many patches are active this frame

        public float[,] HeightField { get; set; } = null!;

        protected override void Awake()
        {
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
            
            // Pre-allocate pooled TransformSlots and MaterialBlocks
            _transformSlots = new int[MaxTerrainPatches];
            _blockPool = new MaterialBlock[MaxTerrainPatches];
            for (int i = 0; i < MaxTerrainPatches; i++)
            {
                _transformSlots[i] = TransformBuffer.Instance!.AllocateSlot();
                _blockPool[i] = new MaterialBlock();
            }
            
            // Set correct local-space Y bounds for occlusion culling.
            // Vertex shader displaces Y by [0, MaxHeight]; world matrix has Y scale = 1.
            foreach (var patch in patches.Values)
            {
                var bb = patch.BoundingBox;
                patch.BoundingBox = new BoundingBox(bb.Min, new Vector3(bb.Max.X, MaxHeight, bb.Max.Z));
            }

            // Initialize quadtree
            var extent = TerrainSize / 2;
            quadtree = new QuadTreeNode(new Vector3(extent.X, 0, extent.Y), new Vector3(extent.X, MaxHeight * 10, extent.Y));
            sizeFactor = quadtree.Extents.X / patchSize;

            // Load control maps and setup layer tiling
            if (Layers.Count > 0)
            {
                CreateControlMaps(device);
                SetupLayerTiling();
            }

            Debug.Log("[Terrain] Awake");
        }

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
                    ControlMaps[i] = Texture.CreateFromData(device, 1, 1, new byte[] {0,0,0,0}); 
                }
            }
            Debug.Log($"[Terrain] Loaded {ControlMaps.Length} individual ControlMaps (bindless)");
        }

        private void SetupLayerTiling()
        {
            var diffuseList = new List<Texture>();
            var normalList = new List<Texture>();
            
            for (int i = 0; i < Layers.Count && i < LayerTiling.Length; i++)
            {
                var layer = Layers[i];
                if (layer.Tiling.X != 0 && layer.Tiling.Y != 0)
                     LayerTiling[i] = new Vector4(TerrainSize.X / layer.Tiling.X, TerrainSize.Y / layer.Tiling.Y, 0, 0);
                else
                     LayerTiling[i] = Vector4.One;
                
                if (layer.Diffuse != null) diffuseList.Add(layer.Diffuse);
                if (layer.Normals != null) normalList.Add(layer.Normals);
            }
            
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
            if (Camera.Main == null) return;
            quadtree.Update(Camera.Main.Transform.Position - Transform.Position);
        }

        public void Draw()
        {
            if (Camera.Main == null) return;
            
            // Quadtree traversal — collect visible patches
            nodesToRender.Clear();
            frustum = new Frustum(Camera.Main.ViewProjection);
            quadtree.Render(CheckPatchVisible);
            nodesToRender.Sort((a, b) => b.Depth.CompareTo(a.Depth));
            
            // Set shared material params once
            Material.SetParameter("CameraPos", Camera.Main.Position);
            Material.SetParameter("HeightTexel", Heightmap != null ? 1.0f / Heightmap.Native.Description.Width : 1.0f / 1024.0f);
            Material.SetParameter("MaxHeight", MaxHeight);
            Material.SetParameter("TerrainSize", TerrainSize);
            Material.SetParameter("LayerTiling", LayerTiling);
            
            // Bind textures to Material (pushed to slots 17-20 by Material.Apply)
            if (Heightmap != null) Material.SetTexture("HeightTex", Heightmap);
            if (ControlMapsArray != null) Material.SetTexture("ControlMaps", ControlMapsArray);
            if (DiffuseMapsArray != null) Material.SetTexture("DiffuseMaps", DiffuseMapsArray);
            if (NormalMapsArray != null) Material.SetTexture("NormalMaps", NormalMapsArray);
            
            // Enqueue each visible patch through the standard InstanceBatch pipeline
            _patchCount = 0;
            foreach (var node in nodesToRender)
            {
                PrepareNode(node);
                RenderPatch(node);
            }
            
            if (Engine.FrameIndex % 60 == 0)
            {
                Debug.Log($"[Terrain] Patches: {_patchCount} | Nodes: {nodesToRender.Count}");
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
            //if (frustum.Intersects(node.Bounds) == ContainmentType.Disjoint)
            //    return;

            nodesToRender.Add(node);
        }

        private void RenderPatch(QuadTreeNode node)
        {
            if (node.Payload is not NodePayload payload) return;
            if (_patchCount >= MaxTerrainPatches) return; // safety cap

            float s = 2 / (float)Math.Pow(2, node.Depth) * sizeFactor;
            Matrix4x4 translation = Matrix4x4.CreateTranslation(node.Position);
            Matrix4x4 scale = Matrix4x4.CreateScale(s, 1f, s);
            Matrix4x4 patchWorld = scale * translation * Transform.WorldMatrix;

            Vector4 rect = Vector4.Zero;
            rect.X = node.Bounds.Min.X / (quadtree.Extents.X * 2);
            rect.Y = node.Bounds.Max.Z / (quadtree.Extents.Z * 2);
            rect.Z = node.Bounds.Max.X / (quadtree.Extents.X * 2);
            rect.W = node.Bounds.Min.Z / (quadtree.Extents.Z * 2);

            int idx = _patchCount++;
            
            // Write transform to global buffer
            int slot = _transformSlots[idx];
            TransformBuffer.Instance!.SetTransform(slot, patchWorld);
            
            // Set per-instance patch data on pooled MaterialBlock
            var block = _blockPool[idx];
            block.Clear();
            block.SetParameter("TerrainData", new TerrainPatchData
            {
                Rect = rect,
                Level = new Vector2(QuadTreeNode.MaxDepth - (node.Depth + 1), 0),
                Padding = Vector2.Zero
            });
            
            // Enqueue through standard pipeline — terrain gets GPU culling, batching, shadows for free
            Mesh patchMesh = patches[payload.Type];
            CommandBuffer.Enqueue(patchMesh, 0, Material, block, slot);
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
