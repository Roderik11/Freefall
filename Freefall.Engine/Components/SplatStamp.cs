using System.Collections.Generic;
using Freefall.Assets;
using Freefall.Base;
using Vortice.Mathematics;

namespace Freefall.Components
{
    public enum SplatOp
    {
        /// <summary>Paint the layer over whatever is there.</summary>
        Paint,
        /// <summary>Take the layer away, revealing what lies beneath it.</summary>
        Remove,
        /// <summary>Scale the layer's weight by Strength.</summary>
        Multiply,
    }

    /// <summary>
    /// Non-destructive splat stamp: places a <see cref="TerrainLayer"/> on the terrain within the stamp zone.
    ///
    /// Stamps composite in ascending Priority and a later stamp paints over an earlier one, whatever
    /// their layers. The layers a terrain renders (its palette) are the layers its splat stamps reference.
    /// </summary>
    [Icon("icon_splatstamp.png")]
    public class SplatStamp : CoverageStamp
    {
        /// <summary>The ground material this stamp places. A stamp without a layer does nothing.</summary>
        [System.ComponentModel.Category("Splat")]
        public TerrainLayer Layer;

        public SplatOp Op = SplatOp.Paint;

        /// <summary>Paint: opacity at full weight. Remove: how much is taken away. Multiply: the factor.</summary>
        [ValueRange(0f, 1f)]
        public float Strength = 1f;

        /// <summary>
        /// What this painted ground is (project-defined <see cref="Tag"/> assets, e.g. "Forest Floor"). Scatter that
        /// keeps clear of stamps (PCG ExcludeStamps) can name tags it does not mind: roadside props ignore a forest's
        /// leaf litter, meadow flowers do not.
        /// </summary>
        public List<Tag> Tags = [];

        protected override Color4 GizmoColor => new Color4(0.9f, 0.6f, 0.2f, 1f); // orange
    }
}
