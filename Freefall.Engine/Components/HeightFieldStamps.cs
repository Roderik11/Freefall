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
    /// Global covers the terrain; Local fades the noise out over Radius + Falloff around the entity, or, when the
    /// entity has a Spline, along it as any spline stamp's shape: an open spline's corridor or a closed one's area,
    /// Radius and Falloff scaled by the spline's Widths (since 2026-11-07: pools and bars along a river's bed, on the
    /// river's own entity). A global noise stamp with BlendMode Set at the lowest priority is the classic generated
    /// base terrain.
    ///
    /// The pattern lies in world metres (since 2026-11-07; it was laid over the terrain's UV), so it stays where it is
    /// whatever the terrain's size and a page of the terrain bakes it by itself (TERRAIN_PAGES_PLAN.md stage 3).
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

        /// <summary>
        /// The width of the broadest features, in metres: a cell of the first octave. Each octave after it is
        /// Lacunarity times finer. Eldermoor's rolling ground is 170.67 (it had Frequency 1.5 over 4096 m).
        /// </summary>
        [ValueRange(10f, 5000f)]
        public float FeatureSize = 500f;

        /// <summary>Amplitude of the noise, as a fraction of the terrain's MaxHeight.</summary>
        [ValueRange(0f, 1f)]
        public float Amplitude = 0.3f;

        /// <summary>
        /// The noise around 0 (-Amplitude .. +Amplitude) instead of 0 .. Amplitude, so Add digs as much as it raises:
        /// pools and bars along a river's bed. Off, Add only ever raises (by about half the Amplitude on average:
        /// fractal noise averages its octaves towards the middle).
        /// </summary>
        public bool Centred = false;

        /// <summary>Per-octave frequency multiplier (lacunarity).</summary>
        [ValueRange(1f, 4f)]
        public float Lacunarity = 2.0f;

        /// <summary>Per-octave amplitude decay.</summary>
        [ValueRange(0f, 1f)]
        public float Persistence = 0.5f;

        /// <summary>Metres the pattern is moved by, along world X and Z.</summary>
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
}
