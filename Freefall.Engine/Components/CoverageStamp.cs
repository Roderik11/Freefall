using System.Collections.Generic;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Category = System.ComponentModel.CategoryAttribute;

namespace Freefall.Components
{
    /// <summary>
    /// Base for stamps that place something on the terrain surface (<see cref="SplatStamp"/>,
    /// <see cref="DecoStamp"/>). Adds the filter: a per-texel weight that multiplies the shape weight,
    /// so a stamp can say "here, but only on slopes" or "here, but only where the grass layer is".
    ///
    /// A global stamp with a filter is how terrain-wide rules are written: rock on cliffs, sand below
    /// the waterline, grass blades wherever the grass layer shows.
    ///
    /// The defaults pass everything, so an unfiltered stamp costs nothing extra.
    /// </summary>
    public abstract class CoverageStamp : TerrainStamp
    {
        /// <summary>Terrain height the stamp applies in, normalized 0..1 of the terrain's MaxHeight.</summary>
        [Category("Filter")]
        public Vector2 HeightRange = new(0, 1);

        /// <summary>Width of the fade at both ends of HeightRange (normalized 0..1).</summary>
        [ValueRange(0f, 0.5f)]
        public float HeightBlend = 0.05f;

        /// <summary>Slope the stamp applies on, in degrees (0 = flat, 90 = cliff).</summary>
        public Vector2 SlopeRange = new(0, 90);

        /// <summary>Width of the fade at both ends of SlopeRange (degrees).</summary>
        [ValueRange(0f, 30f)]
        public float SlopeBlend = 5.0f;

        /// <summary>
        /// Apply only where one of these layers shows: the weight is multiplied by the strongest of them,
        /// measured as what is visible after the layers painted over it. Empty = no requirement.
        /// </summary>
        public List<TerrainLayer> RequireLayers = [];

        /// <summary>
        /// Stay away from these layers: the weight is multiplied by (1 - the strongest of them), measured as
        /// what was painted there, even if something else was painted on top. Empty = no exclusion.
        /// </summary>
        public List<TerrainLayer> ExcludeLayers = [];

        /// <summary>True if any filter term can reject a texel, i.e. the GPU has to evaluate it.</summary>
        public bool HasFilter =>
            HeightRange.X > 0f || HeightRange.Y < 1f || SlopeRange.X > 0f || SlopeRange.Y < 90f ||
            RequireLayers is { Count: > 0 } || ExcludeLayers is { Count: > 0 };
    }
}
