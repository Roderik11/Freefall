using System.Numerics;
using Freefall.Base;
using Vortice.Mathematics;
using Category = System.ComponentModel.CategoryAttribute;

namespace Freefall.Components
{
    /// <summary>How a generated height field combines with the height built by lower-priority stamps.</summary>
    public enum HeightBlendMode { Set, Add, Max, Lerp, Min }

    public enum NoiseType { Simplex, Perlin, Ridged, Billow }

    /// <summary>
    /// Procedural terrain relief: fractal noise composited into the heightmap at this stamp's Priority.
    /// Global covers the terrain; Local fades the noise out over Radius + Falloff around the entity
    /// (a spline on the entity is ignored). A global noise stamp with BlendMode Set at the lowest
    /// priority is the classic generated base terrain.
    /// </summary>
    [Icon("icon_heightstamp.png")]
    public class HeightNoiseStamp : TerrainStamp
    {
        [Category("Blend")]
        public HeightBlendMode BlendMode = HeightBlendMode.Add;

        [ValueRange(0f, 1f)]
        public float Opacity = 1.0f;

        [Category("Noise")]
        public NoiseType Type = NoiseType.Simplex;

        /// <summary>Number of noise octaves (detail layers). More = finer detail, slower.</summary>
        [ValueRange(1, 12)]
        public int Octaves = 6;

        /// <summary>Base frequency. Lower = broader features.</summary>
        [ValueRange(0.01f, 5f)]
        public float Frequency = 0.5f;

        /// <summary>Amplitude of the noise, as a fraction of the terrain's MaxHeight.</summary>
        [ValueRange(0f, 1f)]
        public float Amplitude = 0.3f;

        /// <summary>Per-octave frequency multiplier (lacunarity).</summary>
        [ValueRange(1f, 4f)]
        public float Lacunarity = 2.0f;

        /// <summary>Per-octave amplitude decay.</summary>
        [ValueRange(0f, 1f)]
        public float Persistence = 0.5f;

        /// <summary>Offset for panning the noise pattern (terrain UV units).</summary>
        public Vector2 Offset = Vector2.Zero;

        /// <summary>Seed for the noise permutation (separate from the edge-noise seed).</summary>
        public int Seed = 0;

        /// <summary>Number of terrace steps. 0 = disabled (smooth noise).</summary>
        [Category("Terracing")]
        [ValueRange(0, 64)]
        public int TerraceSteps = 0;

        /// <summary>Smoothness of terrace transitions. 0 = sharp shelves, 1 = rounded.</summary>
        [ValueRange(0f, 1f)]
        public float TerraceSmoothness = 0.3f;

        protected override Color4 GizmoColor => new Color4(0.55f, 0.85f, 0.35f, 1f);
    }

    /// <summary>
    /// Noise-based erosion filter (runevision Advanced Terrain Erosion Filter): carves branching gullies
    /// into the height built by lower-priority stamps. Single GPU dispatch.
    ///
    /// Always covers the whole terrain; the stamp's shape is ignored. Put it above the stamps that
    /// build the relief and below roads and building pads, which should stay smooth.
    /// </summary>
    [Icon("icon_heightstamp.png")]
    public class HeightErosionStamp : TerrainStamp
    {
        [Category("Blend")]
        public HeightBlendMode BlendMode = HeightBlendMode.Set;

        [ValueRange(0f, 1f)]
        public float Opacity = 1.0f;

        /// <summary>Overall horizontal + vertical scale of erosion features.</summary>
        [Category("Erosion")]
        [ValueRange(0.01f, 1f)]
        public float Scale = 0.15f;

        /// <summary>Erosion magnitude. Higher = deeper gullies and sharper ridges.</summary>
        [ValueRange(0.01f, 1f)]
        public float Strength = 0.22f;

        /// <summary>Gully visibility weight. 0 = sharp peaks only, 1 = full gullies.</summary>
        [ValueRange(0f, 1f)]
        public float GullyWeight = 0.5f;

        /// <summary>Detail level. Lower values restrict fine gullies to steep slopes.</summary>
        [ValueRange(0.1f, 5f)]
        public float Detail = 1.5f;

        /// <summary>Number of gully octaves. More = finer branching detail.</summary>
        [ValueRange(1, 8)]
        public int Octaves = 5;

        /// <summary>Frequency multiplier per octave.</summary>
        [ValueRange(1.5f, 4f)]
        public float Lacunarity = 2.0f;

        /// <summary>Amplitude decay per octave.</summary>
        [ValueRange(0.1f, 1f)]
        public float Gain = 0.5f;

        /// <summary>Rounding of ridges (mountain crests). 0 = sharp, higher = softer.</summary>
        [Category("Rounding")]
        [ValueRange(0f, 2f)]
        public float RidgeRounding = 0.1f;

        /// <summary>Rounding of creases (valley bottoms). 0 = sharp V-shaped, higher = smoother.</summary>
        [ValueRange(0f, 2f)]
        public float CreaseRounding = 0f;

        /// <summary>Phacelle cell size relative to stripe width. ~0.7 is a good default.</summary>
        [Category("Advanced")]
        [ValueRange(0.3f, 1.5f)]
        public float CellScale = 0.7f;

        /// <summary>Normalization of gully magnitudes. Higher = more consistent ridges, risk of loop artifacts.</summary>
        [ValueRange(0f, 1f)]
        public float Normalization = 0.5f;

        /// <summary>Slope onset threshold: how quickly erosion ramps up with slope.</summary>
        [ValueRange(0.1f, 5f)]
        public float SlopeOnset = 1.25f;

        /// <summary>Assumed slope magnitude for gully directions (overrides actual gradient).</summary>
        [ValueRange(0f, 2f)]
        public float AssumedSlope = 0.7f;

        /// <summary>Amount to use assumed slope vs actual gradient. 0 = actual, 1 = fully assumed.</summary>
        [ValueRange(0f, 1f)]
        public float AssumedSlopeAmount = 1.0f;

        public HeightErosionStamp()
        {
            Shape = StampShape.Global;
        }

        protected override Color4 GizmoColor => new Color4(0.75f, 0.55f, 0.3f, 1f);
    }
}
