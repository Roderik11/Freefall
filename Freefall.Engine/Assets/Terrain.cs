using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text.Json.Serialization;
using Freefall.Graphics;

namespace Freefall.Assets
{
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
    }
}
