using System;
using System.Collections.Generic;
using System.Numerics;
using Vortice.Mathematics;
using Vortice.Direct3D12; // For PrimitiveTopology
using Freefall.Graphics;
using Freefall.Base;

namespace Freefall.Components
{
    public class TerrainDecorator : Component, IUpdate
    {
        private Mesh GenerateMesh(float size, GraphicsDevice device)
        {
            // generate cloud of points
            int count = CountPerCell;
            var verts = new Vector3[count];
            var sizes = new Vector3[count];
            var colors = new Vector4[count];

            var rng = new Random(123); // Consistent seed for now

            for (int i = 0; i < count; i++)
            {
                var colorRnd = (float)rng.NextDouble() * (RandomColor.Y - RandomColor.X) + RandomColor.X;
                var rnd = (float)rng.NextDouble() * (RandomSize.Y - RandomSize.X) + RandomSize.X;
                var width = BaseSize.X * rnd;
                var height = BaseSize.Y * rnd;

                var uv = rng.Next(0, 3);
                colors[i] = new Vector4(1,1,1,1) * colorRnd;
                sizes[i] = new Vector3(width, height, uv);

                float x = (float)rng.NextDouble() * 2 * size - size;
                float z = (float)rng.NextDouble() * 2 * size - size;
                verts[i] = new Vector3(x, 0, z);
            }

            Vector3 min = new Vector3(-size, -size, -size);
            Vector3 max = new Vector3(size, size, size);

            // In Freefall, Mesh constructor takes simple arrays. We might need a special Mesh type for Points or custom buffer logic.
            // Standard Mesh expects IndexBuffer. 
            // For PointList topology, we can skip IndexBuffer or create a dummy one.
            
            // HACK: Create a dummy index buffer 0..count
            var indices = new uint[count];
            for(uint i=0; i<count; i++) indices[i] = i;

            var uvs = new Vector2[count]; // Dummy UVs

            var mesh = new Mesh(device, verts, sizes, uvs, indices); // Using normals slot for sizes? HACK. 
            // Better to change Mesh definition to support custom layouts, but for now reuse Normals.
            
            // mesh.Topology = PrimitiveTopology.PointList; // Need to expose Topology on Mesh
            mesh.BoundingBox = new BoundingBox(min, max);
            
            mesh.MeshParts.Add(new MeshPart
            {
                Name = "terrainDecoratorCell",
                NumIndices = count,
                BaseIndex = 0,
                BaseVertex = 0,
                Enabled = true
            });

            return mesh;
        }

        public class Cell
        {
            public long Hash;
            public bool IsOutOfRange;

            private Mesh mesh;
            private BoundingBox bounds;
          
            private readonly MaterialBlock materialBlock = new MaterialBlock();
            private readonly Vector3[] points = new Vector3[8];

            public void Generate(Mesh mesh, Vector3 position)
            {
                this.mesh = mesh;
                var matrix = Matrix4x4.CreateScale(Vector3.One) * Matrix4x4.CreateTranslation(position);
                materialBlock.SetParameter("World", matrix);

                // bounds update...
                // mesh.BoundingBox.GetCorners()...
                // Simplified bounds
                bounds = new BoundingBox(position + mesh.BoundingBox.Min, position + mesh.BoundingBox.Max);
            }

            public void SetId(int x, int y)
            {
                Hash = PerfectlyHashThem(x, y);
            }

            public void Draw(Material material, ID3D12GraphicsCommandList commandList)
            {
                // Frustum cull check here if needed
                
                // Set per-instance data (MaterialBlock)
                material.Apply(commandList, Engine.Device, materialBlock);

                // Force PointList topology for this draw
                commandList.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.PointList);
                
                mesh.Draw(commandList);
                
                // Restore topology? Usually Renderer resets it.
            }
        }

        public static long PerfectlyHashThem(int a, int b)
        {
            var A = (ulong)(a >= 0 ? 2 * (long)a : -2 * (long)a - 1);
            var B = (ulong)(b >= 0 ? 2 * (long)b : -2 * (long)b - 1);
            var C = (long)((A >= B ? A * A + A + B : A + B * B) / 2);
            return a < 0 && b < 0 || a >= 0 && b >= 0 ? C : -C - 1;
        }


        public int CountPerCell = 10000;
        public float Range = 128;
        public float CellSize = 16;

        public Vector2 BaseSize = new Vector2(0.2f, 0.4f);
        public Vector2 RandomSize = new Vector2(3, 4);
        public Vector2 RandomColor = new Vector2(.8f, 1f);

        public Material Material = null!;
        public Terrain Terrain = null!;

        private Dictionary<long, Cell> patches = new Dictionary<long, Cell>();
        private List<Cell> cells = new List<Cell>();
        private Mesh cellMesh = null!;

        protected override void Awake()
        {
            cellMesh = GenerateMesh(CellSize, Engine.Device);
        }

        public void Update()
        {
            if (Camera.Main == null || Terrain == null) return;

            // Simple update loop logic similar to original...
            // Omitted for brevity in initial port, can add if "grass" is required for current goal.
            // For now, let's just make sure it compiles.
        }

        public void Draw(ID3D12GraphicsCommandList commandList)
        {
            // Draw cells
        }
    }
}
