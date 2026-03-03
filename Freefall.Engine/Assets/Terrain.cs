using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text.Json.Serialization;
using Freefall.Graphics;

namespace Freefall.Assets
{
    public enum DecoratorMode { Mesh, Billboard, Cross }

    /// <summary>
    /// Terrain asset — holds all resource data (heightmap, layers, splatmaps).
    /// GPU rendering logic and Material live on TerrainRenderer (Component).
    /// </summary>
    public class Terrain : Asset
    {
        // ── Public resource data ──
        public Texture Heightmap;
        public Vector2 TerrainSize = new(1700, 1700);
        public float MaxHeight = 600;
        public List<TextureLayer> Layers;

        // ── Splatmaps (public for YAML serialization) ──
        public List<Texture> ControlMaps = new();

        // ── Ground Coverage ──
        public List<Decoration> Decorations = new();
        public List<Texture> DecoMaps = new();
        
        [ValueRange(1, 20)]
        public float DecorationDensity = 1.0f;

        public float DecorationRadius = 100f;

        [JsonIgnore] public int DecorationVersion { get; private set; }
        public void InvalidateDecorations() => DecorationVersion++;

        // ── HeightField — built internally from Heightmap ──
        [JsonIgnore]
        public float[,] HeightField { get; private set; }

        /// <summary>
        /// Pre-cooked PhysX height field. Populated during asset loading from the
        /// collision subasset so that RigidBody.Awake() doesn't need to cook on the main thread.
        /// </summary>
        [JsonIgnore]
        public PhysX.HeightField CookedHeightField { get; private set; }

        /// <summary>
        /// Set the pre-cooked PhysX height field. Called by TerrainLoader.
        /// </summary>
        internal void SetCookedHeightField(PhysX.HeightField hf) => CookedHeightField = hf;

        /// <summary>
        /// Builds the CPU-side height field from the current Heightmap texture.
        /// Must be called after Heightmap is assigned (e.g. during loading).
        /// </summary>
        public void BuildHeightField(string heightmapPath)
        {
            HeightField = Texture.ReadHeightField(heightmapPath);
        }

        // ── Texture Layer ──
        [Serializable]
        public class TextureLayer
        {
            public Texture Diffuse;
            public Texture Normals;
            public Vector2 Tiling = Vector2.One;
        }

        // ── Ground Coverage Decoration ──
        /// <summary>
        /// A single decoration entry. References a StaticMesh or Texture for rendering,
        /// plus an optional DensityMap (grayscale) that controls placement density.
        /// At runtime, all unique DensityMaps are packed into a Texture2DArray.
        /// </summary>
        [Serializable]
        public class Decoration
        {
            public DecoratorMode Mode = DecoratorMode.Mesh;
            public StaticMesh Mesh;       // Mesh mode: geometry + LODs + material
            public Texture Texture;       // Billboard/Cross mode: alpha-tested texture

            /// <summary>
            /// Optional density map (grayscale). When set, auto-resolves to a DecoMaps array
            /// slice at runtime. Without a density map, the decorator renders everywhere.
            /// </summary>
            public Texture DensityMap;

            /// <summary>Instances per square meter.</summary>
            [ValueRange(1, 20)]
            public float Density = 2.0f;

            public float Weight = 1.0f;
            public Vector2 HeightRange = new(0.3f, 0.6f);
            public Vector2 WidthRange = new(0.2f, 0.4f);

            /// <summary>Root rotation applied to mesh vertices (euler degrees).</summary>
            public Vector3 RootRotation = new(-90, 0, 0);

            /// <summary>Blend factor for aligning to terrain slope (0=upright, 1=fully aligned).</summary>
            [ValueRange(-1, 1)]
            public float SlopeBias = 0.0f;
        }
    }
}
