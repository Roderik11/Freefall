using Freefall.Assets;
using Freefall.Base;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// Places a <see cref="Biome"/>: its layer and ground cover rules apply inside this stamp's shape (a circle, a
    /// spline's area or corridor, or the whole terrain when Global), faded by its falloff and edge noise, at its
    /// Priority among the other paint. The stamp's filter limits every rule, and each rule's own filter narrows that
    /// (<see cref="StampFilter.Both"/>): a Global biome stamp above some height is a biome derived from the relief.
    ///
    /// Where two biomes meet, the one baked later (higher Priority, then the stable id) paints over the other across
    /// its falloff. SplatStamps and DecoStamps above it are the local fixes: a road, a village green.
    /// </summary>
    [Icon("icon_splatstamp.png")]
    public class BiomeStamp : CoverageStamp
    {
        /// <summary>The biome this stamp places. A stamp without one does nothing.</summary>
        [System.ComponentModel.Category("Biome")]
        public Biome Biome;

        protected override Color4 GizmoColor => new Color4(0.45f, 0.8f, 0.35f, 1f); // green
    }
}
