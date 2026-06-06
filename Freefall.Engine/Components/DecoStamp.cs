using Freefall.Base;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// Non-destructive decoration stamp. Controls decoration density
    /// within the stamp zone (suppress or boost).
    /// </summary>
    public class DecoStamp : TerrainStamp
    {
        /// <summary>
        /// Decoration density multiplier within the stamp zone.
        /// 0 = fully suppress, 1 = no change, >1 = boost density.
        /// </summary>
        [ValueRange(0f, 2f)]
        public float Density = 0f;

        protected override Color4 GizmoColor => new Color4(0.5f, 0.7f, 1f, 1f); // blue
    }
}
