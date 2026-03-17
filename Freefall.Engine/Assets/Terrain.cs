using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;
using System.ComponentModel;
using Freefall.Base;
using Freefall.Graphics;
using Freefall.Reflection;
using Vortice.DXGI;

namespace Freefall.Assets
{
    public enum DecoratorMode { Mesh, Billboard, Cross }

    /// <summary>
    /// Terrain asset — holds all resource data (heightmap, layers, splatmaps).
    /// GPU rendering logic and Material live on TerrainRenderer (Component).
    /// </summary>
    public class Terrain : Asset
    {
        /// <summary>
        /// Resolution of the baked heightmap (power of 2). Configurable per-terrain.
        /// </summary>
        public int HeightmapResolution = 1024;

        /// <summary>
        /// Non-destructive height layer stack. Composited bottom-to-top via GPU bake.
        /// </summary>
        public List<HeightLayer> HeightLayers = new();

        /// <summary>
        /// GPU-baked heightmap texture (runtime only, regenerated from HeightLayers).
        /// </summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public Texture BakedHeightmap;

        /// <summary>
        /// Stamp groups — decal-like height placements grouped by brush texture.
        /// Baked after HeightLayers, one GPU pass per group.
        /// </summary>
        public List<StampGroup> StampGroups = new();

        /// <summary>
        /// The final heightmap: baked result if available, otherwise the first ImportHeightLayer source.
        /// All consumers (renderer, physics, decorators) should read this.
        /// </summary>
        [Reflection.DontSerialize]
        [JsonIgnore]
        public Texture Heightmap
        {
            get
            {
                if (BakedHeightmap != null) return BakedHeightmap;

                // Fallback: first enabled ImportHeightLayer source
                foreach (var layer in HeightLayers)
                {
                    if (layer is ImportHeightLayer import && import.Enabled && import.Source != null)
                        return import.Source;
                }

                return null;
            }
        }

        // ── Terrain dimensions ──
        public Vector2 TerrainSize = new(1700, 1700);
        public float MaxHeight = 600;
        public List<TextureLayer> Layers;

        /// <summary>
        /// Migration only: catches the old flat ControlMaps list during deserialization.
        /// Used directly by TerrainRenderer for RGBA-packed GPU splatmap array.
        /// Will be replaced by per-layer R16 ControlMaps after channel-split migration.
        /// </summary>
        [FormerlySerializedAs("ControlMaps")]
        [DontSerialize]
        public List<Texture> _legacyControlMaps;

        // ── Ground Coverage ──
        public List<Decoration> Decorations = new();
        
        [ValueRange(0.1f, 10)]
        public float DecorationDensity = 0.1f;

        public float DecorationRadius = 100f;

        public bool DrawDetail = true;

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

        /// <summary>
        /// Sets the CPU-side height field directly from readback data.
        /// Called by TerrainRenderer after GPU heightmap readback.
        /// </summary>
        internal void SetHeightField(float[,] heights) => HeightField = heights;

        // ── Texture Layer ──
        [Serializable]
        public class TextureLayer
        {
            public Texture Diffuse;
            public Texture Normals;
            public Vector2 Tiling = Vector2.One;

            /// <summary>Splatmap controlling where this layer paints (R16, hidden subasset).</summary>
            /// 
            [Browsable(false)]
            public Texture ControlMap;

            /// <summary>Staging: raw R16 bytes loaded from cache, consumed by GPU upload.</summary>
            [Reflection.DontSerialize]
            [JsonIgnore]
            public byte[] PendingControlMapBytes;
        }

        // ── Ground Coverage Decoration ──
        /// <summary>
        /// A single decoration entry. References a StaticMesh or Texture for rendering,
        /// plus an optional ControlMap (grayscale) that controls placement density.
        /// At runtime, all unique ControlMaps are packed into a Texture2DArray.
        /// </summary>
        [Serializable]
        public class Decoration
        {
            public DecoratorMode Mode = DecoratorMode.Mesh;
            public StaticMesh Mesh;       // Mesh mode: geometry + LODs + material
            public Texture Texture;       // Billboard/Cross mode: alpha-tested texture

            /// <summary>
            /// Controls placement density (R16, hidden subasset).
            /// Without a control map, the decorator renders everywhere.
            /// </summary>
            [FormerlySerializedAs("DensityMap")]
            [Browsable(false)]
            public Texture ControlMap;

            /// <summary>Staging: raw R16 bytes loaded from cache, consumed by GPU upload.</summary>
            [Reflection.DontSerialize]
            [JsonIgnore]
            public byte[] PendingControlMapBytes;

            /// <summary>Instances per square meter.</summary>
            [ValueRange(.1f, 10)]
            public float Density = 1.0f;

            [ValueRange(0f, 1f)]
            public float Weight = 1.0f;
            public Vector2 HeightRange = new(0.3f, 0.6f);
            public Vector2 WidthRange = new(0.2f, 0.4f);

            /// <summary>Root rotation applied to mesh vertices (euler degrees).</summary>
            public Vector3 RootRotation = new(-90, 0, 0);

            /// <summary>Blend factor for aligning to terrain slope (0=upright, 1=fully aligned).</summary>
            [ValueRange(-1, 1)]
            public float SlopeBias = 0.0f;

            /// <summary>Tint color for "healthy" instances (multiplicative). Alpha=0 means no tinting.</summary>
            public Vector4 HealthyColor = new(1, 1, 1, 1);

            /// <summary>Tint color for "dry" instances (multiplicative). Alpha=0 means no tinting.</summary>
            public Vector4 DryColor = new(1, 1, 1, 1);

            /// <summary>World-space noise frequency for healthy/dry blend. 0 = uniform healthy color.</summary>
            [ValueRange(0f, 1f)]
            public float NoiseSpread = 1.0f;
        }
    }
}
