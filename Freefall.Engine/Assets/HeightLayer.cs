using Freefall.Graphics;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Numerics;

namespace Freefall.Assets
{
    /// <summary>
    /// Blend mode for compositing height operations.
    /// </summary>
    public enum HeightBlendMode { Set, Add, Max, Lerp }

    // ── Height Layers (broad compositional operations, few per terrain) ──

    /// <summary>
    /// Base class for non-destructive terrain height layers.
    /// Layers compose bottom-to-top via GPU bake into a final heightmap.
    /// </summary>
    [Serializable]
    public abstract class HeightLayer
    {
        public bool Enabled = true;
        [ValueRange(0f, 1f)]
        public float Opacity = 1.0f;
        public HeightBlendMode BlendMode = HeightBlendMode.Set;
    }

    /// <summary>
    /// Imports a full heightmap texture as a base layer.
    /// Migration path: existing Heightmap GUID becomes Source on this layer.
    /// </summary>
    [Serializable]
    public class ImportHeightLayer : HeightLayer
    {
        public Texture Source;
    }

    /// <summary>
    /// Procedural noise layer. Generates height from GPU simplex/perlin noise.
    /// Parameters control the fractal characteristics of the generated terrain.
    /// </summary>
    [Serializable]
    public class NoiseHeightLayer : HeightLayer
    {
        public NoiseType Type = NoiseType.Simplex;

        /// <summary>Number of noise octaves (detail layers). More = finer detail, slower.</summary>
        [ValueRange(1, 12)]
        public int Octaves = 6;

        /// <summary>Base frequency. Lower = broader features.</summary>
        [ValueRange(0.001f, 10f)]
        public float Frequency = 0.5f;

        /// <summary>Amplitude scale for the noise output.</summary>
        [ValueRange(0f, 1f)]
        public float Amplitude = 0.3f;

        /// <summary>Per-octave frequency multiplier (lacunarity).</summary>
        [ValueRange(1f, 4f)]
        public float Lacunarity = 2.0f;

        /// <summary>Per-octave amplitude decay.</summary>
        [ValueRange(0f, 1f)]
        public float Persistence = 0.5f;

        /// <summary>World-space offset for panning the noise pattern.</summary>
        public Vector2 Offset = Vector2.Zero;

        /// <summary>Seed for noise permutation.</summary>
        public int Seed = 0;
    }

    public enum NoiseType { Simplex, Perlin, Ridged, Billow }

    /// <summary>
    /// GPU erosion simulation layer. Reads height produced by layers below,
    /// runs iterative hydraulic and/or thermal erosion, writes eroded result.
    /// Heavy compute — cached after first bake, re-runs only when parameters change.
    /// </summary>
    [Serializable]
    public class ErosionHeightLayer : HeightLayer
    {
        public ErosionMode Mode = ErosionMode.Hydraulic;

        /// <summary>Number of simulation iterations. More = deeper erosion, slower bake.</summary>
        [ValueRange(1, 500)]
        public int Iterations = 100;

        // ── Hydraulic erosion ──

        /// <summary>Rain amount per iteration (water deposited).</summary>
        [ValueRange(0f, 0.1f)]
        public float RainRate = 0.01f;

        /// <summary>Max sediment a unit of water can carry.</summary>
        [ValueRange(0f, 1f)]
        public float SedimentCapacity = 0.1f;

        /// <summary>Fraction of sediment deposited when water slows.</summary>
        [ValueRange(0f, 1f)]
        public float DepositionRate = 0.3f;

        /// <summary>Fraction of terrain dissolved per step.</summary>
        [ValueRange(0f, 0.1f)]
        public float DissolutionRate = 0.01f;

        /// <summary>Water evaporation rate per iteration.</summary>
        [ValueRange(0f, 1f)]
        public float Evaporation = 0.05f;

        // ── Thermal erosion ──

        /// <summary>Maximum stable slope angle in degrees. Steeper slopes collapse.</summary>
        [ValueRange(0f, 90f)]
        public float TalusAngle = 40f;

        /// <summary>Fraction of excess material moved per thermal iteration.</summary>
        [ValueRange(0f, 1f)]
        public float ThermalRate = 0.5f;

        public int Seed = 0;
    }

    public enum ErosionMode { Hydraulic, Thermal, Both }

    /// <summary>
    /// Hand-painted height layer. A GPU R16_Float texture (ControlMap) is painted
    /// by compute shader brush operations. The baker composites it with other layers.
    /// BlendMode is typically Add (positive = raise, negative = lower).
    /// </summary>
    [Serializable]
    public class PaintHeightLayer : HeightLayer
    {
        /// <summary>
        /// GPU R16_Float control map texture painted by the brush compute shader.
        /// Created on first brush stroke; composited during bake.
        /// Hidden subasset — GUID stored in YAML, data in cache.
        /// </summary>
        [Browsable(false)]
        public Texture ControlMap;

        /// <summary>
        /// CPU-side pixel data loaded from cache, awaiting GPU upload.
        /// TerrainRenderer checks this on first bake and calls TerrainBaker.UploadControlMap().
        /// </summary>
        [NonSerialized]
        public byte[] PendingControlMapBytes;

        public PaintHeightLayer()
        {
            BlendMode = HeightBlendMode.Add;
        }
    }

    // ── Stamps (decal-like height placements, grouped by brush texture) ──

    /// <summary>
    /// A single stamp placement — position, radius, strength, falloff, rotation.
    /// Lightweight: no texture reference (owned by the parent StampGroup).
    /// </summary>
    [Serializable]
    public class StampInstance
    {
        /// <summary>Terrain-space UV position [0..1]</summary>
        public Vector2 Position = new(0.5f, 0.5f);

        /// <summary>Stamp radius in terrain-space UV [0..1]</summary>
        [ValueRange(0f, 1f)]
        public float Radius = 0.1f;

        /// <summary>Stamp height strength</summary>
        [ValueRange(0f, 1f)]
        public float Strength = 1.0f;

        /// <summary>Falloff exponent (1 = linear, 2 = smooth)</summary>
        [ValueRange(1f, 2f)]
        public float Falloff = 2.0f;

        /// <summary>Rotation in degrees</summary>
        [ValueRange(0f, 360f)]
        public float Rotation = 0;
    }

    /// <summary>
    /// A group of stamp instances sharing the same brush texture and blend mode.
    /// Each click in the editor appends a StampInstance to the active group.
    /// GPU bake dispatches one pass per group with a StructuredBuffer of instances.
    /// </summary>
    [Serializable]
    public class StampGroup
    {
        public string Name = "Stamp Group";
        public Texture Brush;
        public HeightBlendMode BlendMode = HeightBlendMode.Add;
        [ValueRange(0f, 1f)]
        public float Opacity = 1.0f;
        public bool Enabled = true;
        public List<StampInstance> Instances = new();
    }
}
