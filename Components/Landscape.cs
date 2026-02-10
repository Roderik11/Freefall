using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Graphics;
using Freefall.Base;

namespace Freefall.Components
{
    public class Landscape : Freefall.Base.Component, IUpdate, IDraw, IHeightProvider
    {
        /// <summary>
        /// Per-instance data for each landscape patch — matches HLSL LandscapePatchData.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 16)]
        public struct LandscapePatchData
        {
            public Vector4 Rect;       // UV rect into heightmap (xy=min, zw=max)
            public Vector2 Level;      // x=LOD level, y=morph factor
            public float RingScale;    // world scale of this ring
            public float Padding;
        }

        // Configuration
        public int RingCount = 6;
        public int GridResolution = 32;        // cells per ring side (ring 0)
        public int PatchResolution = 8;        // cells per patch side
        public float MaxHeight = 1024f;
        public Vector2 TerrainSize = new Vector2(4096 * 2, 4096 * 2);
        public float CellSize = 2f;

        // Resources
        public Texture Heightmap = null!;
        public Material Material = null!;
        public float[,] HeightField { get; set; } = null!;

        // Texture layers
        public class TextureLayer
        {
            public Texture Diffuse;
            public Texture Normals;
            public Vector2 Tiling = Vector2.One;
        }

        public List<TextureLayer> Layers = new();
        public Texture?[] ControlMaps = new Texture?[4];

        private Texture? ControlMapsArray;
        private Texture? DiffuseMapsArray;
        private Texture? NormalMapsArray;
        private Vector4[] LayerTiling = new Vector4[32];

        // Patch-based drawing
        private Dictionary<PatchType, Mesh> _patchMeshes = null!;
        private int[] _transformSlots = null!;
        private MaterialBlock[] _blockPool = null!;
        private int _maxPatches;
        private int _patchCount;

        protected override void Awake()
        {
            var device = Engine.Device;

            // Create shared patch mesh variants (9 types with edge stitching)
            _patchMeshes = Mesh.CreateLandscapePatches(device, PatchResolution);

            // Set Y bounds for height displacement on all patch meshes
            foreach (var mesh in _patchMeshes.Values)
            {
                var bb = mesh.BoundingBox;
                mesh.BoundingBox = new Vortice.Mathematics.BoundingBox(
                    bb.Min, new Vector3(bb.Max.X, MaxHeight, bb.Max.Z));
            }

            // All rings use the same patchesPerRow; ring 0 is solid, ring 1+ is a frame with center hole
            int patchesPerRow = GridResolution / PatchResolution;            // e.g. 32/8 = 4
            int ring0Patches = patchesPerRow * patchesPerRow;               // 16
            int holeSize = patchesPerRow / 2;                               // 2
            int framePatchesPerRing = ring0Patches - holeSize * holeSize;   // 16 - 4 = 12
            _maxPatches = ring0Patches + framePatchesPerRing * (RingCount - 1);

            // Pre-allocate pooled TransformSlots and MaterialBlocks
            _transformSlots = new int[_maxPatches];
            _blockPool = new MaterialBlock[_maxPatches];
            for (int i = 0; i < _maxPatches; i++)
            {
                _transformSlots[i] = TransformBuffer.Instance!.AllocateSlot();
                _blockPool[i] = new MaterialBlock();
            }

            // Setup texture layers
            if (Layers.Count > 0)
                SetupLayerTiling();

            Debug.Log($"[Landscape] Awake: {RingCount} rings, grid {GridResolution}, patch {PatchResolution}, max patches {_maxPatches}");
        }

        private void SetupLayerTiling()
        {
            var device = Engine.Device;
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

            if (diffuseList.Count > 0)
                DiffuseMapsArray = Texture.CreateTexture2DArray(device, diffuseList);
            if (normalList.Count > 0)
                NormalMapsArray = Texture.CreateTexture2DArray(device, normalList);

            // Load control maps
            var filenames = new[] {
                "Resources/Terrain/Terrain_splatmap_0.dds",
                "Resources/Terrain/Terrain_splatmap_1.dds",
                "Resources/Terrain/Terrain_splatmap_2.dds",
                "Resources/Terrain/Terrain_splatmap_3.dds"
            };

            for (int i = 0; i < filenames.Length && i < ControlMaps.Length; i++)
            {
                if (System.IO.File.Exists(filenames[i]))
                    ControlMaps[i] = new Texture(Engine.Device, filenames[i]);
                else
                    ControlMaps[i] = Texture.CreateFromData(Engine.Device, 1, 1, new byte[] { 0, 0, 0, 0 });
            }

            var controlList = new List<Texture>();
            foreach (var c in ControlMaps) if (c != null) controlList.Add(c);
            if (controlList.Count > 0)
                ControlMapsArray = Texture.CreateTexture2DArray(Engine.Device, controlList);
        }

        public void Update()
        {
            // Nothing to update — patch positions computed in Draw based on camera
        }

        public void Draw()
        {
            if (Material == null || Heightmap == null) return;

            var camera = Camera.Main;
            if (camera == null) return;

            Vector3 camPos = camera.Entity.Transform.Position;
            float heightTexel = 1.0f / Heightmap.Native.Description.Width;

            // Set material constants
            Material.SetParameter("CameraPos", camPos);
            Material.SetParameter("HeightTexel", heightTexel);
            Material.SetParameter("MaxHeight", MaxHeight);
            Material.SetParameter("TerrainSize", TerrainSize);
            var terrainOriginPos = Transform?.Position ?? Vector3.Zero;
            Material.SetParameter("TerrainOrigin", new Vector2(terrainOriginPos.X, terrainOriginPos.Z));
            Material.SetParameter("LayerTiling", LayerTiling);

            // Bind textures
            if (Heightmap != null) Material.SetTexture("HeightTex", Heightmap);
            if (ControlMapsArray != null) Material.SetTexture("ControlMaps", ControlMapsArray);
            if (DiffuseMapsArray != null) Material.SetTexture("DiffuseMaps", DiffuseMapsArray);
            if (NormalMapsArray != null) Material.SetTexture("NormalMaps", NormalMapsArray);

            // All rings share a single snap center (finest cell size).
            // The grid snaps to cell boundaries so height sampling stays stable.
            // The fractional remainder (SnapOffset) is passed to the shader,
            // which offsets vertex world positions AFTER height displacement.
            // This makes the terrain slide smoothly with the camera while
            // heights remain rock-solid between snaps.
            // At snap boundaries: oldSnap + cellSize == newSnap + 0 → continuous.
            float baseCellSize = CellSize;
            float snapX = MathF.Floor(camPos.X / baseCellSize) * baseCellSize;
            float snapZ = MathF.Floor(camPos.Z / baseCellSize) * baseCellSize;

            // Fractional remainder: smooth offset applied in the shader
            float snapOffsetX = camPos.X - snapX;
            float snapOffsetZ = camPos.Z - snapZ;
            Material.SetParameter("SnapOffset", new Vector2(snapOffsetX, snapOffsetZ));

            Vector3 entityPos = Transform?.Position ?? Vector3.Zero;
            _patchCount = 0;

            for (int ring = 0; ring < RingCount; ring++)
            {
                EmitRingPatches(ring, snapX, snapZ, entityPos, baseCellSize);
            }
        }

        private void EmitRingPatches(int ring, float snapX, float snapZ, Vector3 entityPos, float baseCellSize)
        {
            int patchesPerRow = GridResolution / PatchResolution; // e.g. 4
            float ringCellSize = baseCellSize * (1 << ring);
            float meshScale = GridResolution * ringCellSize;
            int holeMin, holeMax;

            if (ring == 0)
            {
                holeMin = holeMax = -1; // no hole
            }
            else
            {
                // Static hole: center (patchesPerRow/2)² patches = previous ring's footprint
                int holeSize = patchesPerRow / 2;               // 2
                holeMin = (patchesPerRow - holeSize) / 2;       // 1
                holeMax = holeMin + holeSize;                    // 3
            }

            float originX = snapX - meshScale * 0.5f;
            float originZ = snapZ - meshScale * 0.5f;
            float patchWorldSize = meshScale / patchesPerRow;

            // Determine which patches are on the OUTER boundary (border coarser ring)
            bool isOutermost = (ring == RingCount - 1);

            for (int pz = 0; pz < patchesPerRow; pz++)
            {
                for (int px = 0; px < patchesPerRow; px++)
                {
                    // Skip inner hole patches (rings 1+)
                    if (holeMin >= 0 && px >= holeMin && px < holeMax && pz >= holeMin && pz < holeMax)
                        continue;

                    if (_patchCount >= _maxPatches) return;

                    // Determine stitching type — outer boundary patches need collapsed edges
                    // to match the next coarser ring's vertex spacing
                    bool collapseW = !isOutermost && px == 0;
                    bool collapseE = !isOutermost && px == patchesPerRow - 1;
                    bool collapseS = !isOutermost && pz == 0;
                    bool collapseN = !isOutermost && pz == patchesPerRow - 1;

                    PatchType type = GetPatchType(collapseW, collapseE, collapseN, collapseS);
                    Mesh patchMesh = _patchMeshes[type];

                    // Patch world-space position and scale
                    float patchOriginX = originX + px * patchWorldSize;
                    float patchOriginZ = originZ + pz * patchWorldSize;

                    Matrix4x4 world = Matrix4x4.CreateScale(patchWorldSize, 1f, patchWorldSize) *
                                      Matrix4x4.CreateTranslation(patchOriginX, 0, patchOriginZ) *
                                      Transform.WorldMatrix;

                    // Heightmap UV rect for this patch
                    float uvMinX = (patchOriginX - entityPos.X) / TerrainSize.X;
                    float uvMinZ = (patchOriginZ - entityPos.Z) / TerrainSize.Y;
                    float uvMaxX = (patchOriginX + patchWorldSize - entityPos.X) / TerrainSize.X;
                    float uvMaxZ = (patchOriginZ + patchWorldSize - entityPos.Z) / TerrainSize.Y;

                    int idx = _patchCount++;
                    int slot = _transformSlots[idx];
                    TransformBuffer.Instance!.SetTransform(slot, world);

                    var block = _blockPool[idx];
                    block.Clear();
                    block.SetParameter("LandscapeData", new LandscapePatchData
                    {
                        Rect = new Vector4(uvMinX, uvMinZ, uvMaxX, uvMaxZ),
                        Level = new Vector2(ring, 0),
                        RingScale = (float)(1 << ring),
                        Padding = 0
                    });

                    CommandBuffer.Enqueue(patchMesh, 0, Material, block, slot);
                }
            }
        }

        private static PatchType GetPatchType(bool w, bool e, bool n, bool s)
        {
            if (w && e && n && s) return PatchType.Center;
            if (w && n) return PatchType.NW;
            if (w && s) return PatchType.SW;
            if (e && n) return PatchType.NE;
            if (e && s) return PatchType.SE;
            if (w) return PatchType.W;
            if (e) return PatchType.E;
            if (n) return PatchType.N;
            if (s) return PatchType.S;
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
