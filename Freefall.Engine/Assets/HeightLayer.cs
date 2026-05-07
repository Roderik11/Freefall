using Freefall.Graphics;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Numerics;

using static Freefall.Assets.TerrainDirtyFlags;

namespace Freefall.Assets
{
    /// <summary>
    /// Blend mode for compositing height operations.
    /// </summary>
    public enum HeightBlendMode { Set, Add, Max, Lerp, Min, Flatten }

    // ── Height Layers (broad compositional operations, few per terrain) ──

    /// <summary>
    /// Base class for non-destructive terrain height layers.
    /// Layers compose bottom-to-top via GPU bake into a final heightmap.
    /// </summary>
    [Serializable]
    public abstract class HeightLayer
    {
        [DirtyFlag(HeightAll)]
        public bool Enabled = true;

        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 1f)]
        public float Opacity = 1.0f;

        [DirtyFlag(HeightAll)]
        public HeightBlendMode BlendMode = HeightBlendMode.Set;
    }

    /// <summary>
    /// Imports a full heightmap texture as a base layer.
    /// Migration path: existing Heightmap GUID becomes Source on this layer.
    /// </summary>
    [Serializable]
    public class ImportHeightLayer : HeightLayer
    {
        [DirtyFlag(HeightAll)]
        public Texture Source;
    }

    /// <summary>
    /// Procedural noise layer. Generates height from GPU simplex/perlin noise.
    /// Parameters control the fractal characteristics of the generated terrain.
    /// </summary>
    [Serializable]
    public class NoiseHeightLayer : HeightLayer
    {
        [DirtyFlag(HeightAll)]
        public NoiseType Type = NoiseType.Simplex;

        /// <summary>Number of noise octaves (detail layers). More = finer detail, slower.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(1, 12)]
        public int Octaves = 6;

        /// <summary>Base frequency. Lower = broader features.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0.01f, 5f)]
        public float Frequency = 0.5f;

        /// <summary>Amplitude scale for the noise output.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 1f)]
        public float Amplitude = 0.3f;

        /// <summary>Per-octave frequency multiplier (lacunarity).</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(1f, 4f)]
        public float Lacunarity = 2.0f;

        /// <summary>Per-octave amplitude decay.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 1f)]
        public float Persistence = 0.5f;

        /// <summary>World-space offset for panning the noise pattern.</summary>
        [DirtyFlag(HeightAll)]
        public Vector2 Offset = Vector2.Zero;

        /// <summary>Seed for noise permutation.</summary>
        [DirtyFlag(HeightAll)]
        public int Seed = 0;

        // ── Terracing ──

        /// <summary>Number of terrace steps. 0 = disabled (smooth noise).</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0, 64)]
        public int TerraceSteps = 0;

        /// <summary>Smoothness of terrace transitions. 0 = sharp shelves, 1 = rounded.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 1f)]
        public float TerraceSmoothness = 0.3f;

        // ── Spatial Mask (radial falloff) ──

        /// <summary>UV center of the mask [0..1]. Default = center of terrain.</summary>
        [DirtyFlag(HeightAll)]
        public Vector2 MaskCenter = new(0.5f, 0.5f);

        /// <summary>Mask radius in UV-space. 0 = disabled (full terrain).</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 1f)]
        public float MaskRadius = 0f;

        /// <summary>Mask edge falloff. Higher = sharper edges.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0.01f, 8f)]
        public float MaskFalloff = 2f;
    }

    public enum NoiseType { Simplex, Perlin, Ridged, Billow }

    /// <summary>
    /// Noise-based erosion filter layer. Applies the runevision Advanced Terrain
    /// Erosion Filter on top of the accumulated height from layers below.
    /// Single GPU dispatch — produces crisp branching gullies via Phacelle noise.
    /// </summary>
    [Serializable]
    public class ErosionHeightLayer : HeightLayer
    {
        /// <summary>Overall horizontal + vertical scale of erosion features.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0.01f, 1f)]
        public float Scale = 0.15f;

        /// <summary>Erosion magnitude. Higher = deeper gullies and sharper ridges.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0.01f, 1f)]
        public float Strength = 0.22f;

        /// <summary>Gully visibility weight. 0 = sharp peaks only, 1 = full gullies.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 1f)]
        public float GullyWeight = 0.5f;

        /// <summary>Detail level. Lower values restrict fine gullies to steep slopes.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0.1f, 5f)]
        public float Detail = 1.5f;

        /// <summary>Number of gully octaves. More = finer branching detail.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(1, 8)]
        public int Octaves = 5;

        /// <summary>Frequency multiplier per octave.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(1.5f, 4f)]
        public float Lacunarity = 2.0f;

        /// <summary>Amplitude decay per octave.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0.1f, 1f)]
        public float Gain = 0.5f;

        // ── Edge rounding ──

        /// <summary>Rounding of ridges (mountain crests). 0 = sharp, higher = softer.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 2f)]
        public float RidgeRounding = 0.1f;

        /// <summary>Rounding of creases (valley bottoms). 0 = sharp V-shaped, higher = smoother.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 2f)]
        public float CreaseRounding = 0f;

        // ── Advanced ──

        /// <summary>Phacelle cell size relative to stripe width. ~0.7 is a good default.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0.3f, 1.5f)]
        public float CellScale = 0.7f;

        /// <summary>Normalization of gully magnitudes. Higher = more consistent ridges, risk of loop artifacts.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 1f)]
        public float Normalization = 0.5f;

        /// <summary>Slope onset threshold — how quickly erosion ramps up with slope.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0.1f, 5f)]
        public float SlopeOnset = 1.25f;

        /// <summary>Assumed slope magnitude for gully directions (overrides actual gradient).</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 2f)]
        public float AssumedSlope = 0.7f;

        /// <summary>Amount to use assumed slope vs actual gradient. 0 = actual, 1 = fully assumed.</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 1f)]
        public float AssumedSlopeAmount = 1.0f;
    }

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

    // ── Stamps (decal-like height placements) ──

    /// <summary>
    /// A single stamp placement — self-contained with brush, blend mode, and all parameters.
    /// The baker groups stamps by (Brush, BlendMode, Opacity) at dispatch time.
    /// </summary>
    [Serializable]
    public class Stamp
    {
        public bool Enabled = true;

        [DirtyFlag(HeightAll)]
        public Texture Brush;

        [DirtyFlag(HeightAll)]
        public HeightBlendMode BlendMode = HeightBlendMode.Add;

        /// <summary>Stamp radius in terrain-space UV [0..1]</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 2f)]
        public float Radius = 0.5f;

        /// <summary>Stamp height strength</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 1f)]
        public float Strength = 0.5f;

        /// <summary>Terrain-space UV position [0..1]</summary>
        [DirtyFlag(HeightAll)]
        [Browsable(false)]
        public Vector2 Position = new(0.5f, 0.5f);

        [ValueRange(0f, 1f)]
        public float PositionX
        {
            get => Position.X;
            set => Position = new Vector2(value, Position.Y);
        }

        [ValueRange(0f, 1f)]
        public float PositionY
        {
            get => Position.Y;
            set => Position = new Vector2(Position.X, value);
        }

        /// <summary>Rotation in degrees</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 360f)]
        public float Rotation = 0;

        /// <summary>Fraction of radius at full strength before falloff begins (0 = fully soft, 1 = no fade)</summary>
        [DirtyFlag(HeightAll)]
        [ValueRange(0f, 1f)]
        public float Falloff = 0.5f;
    }
}
