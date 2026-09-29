using Freefall.Base;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// Non-destructive splat stamp. Paints or removes a terrain splat layer
    /// within the stamp zone.
    /// </summary>
    [Icon("icon_splatstamp.png")]
    public class SplatStamp : TerrainStamp
    {
        /// <summary>Terrain layer index to paint.</summary>
        [System.ComponentModel.Category("Splat")]
        public int SplatLayerIndex = 0;

        /// <summary>Paint strength (0-1) at full weight.</summary>
        [ValueRange(0f, 1f)]
        public float Strength = 1f;

        protected override Color4 GizmoColor => new Color4(0.9f, 0.6f, 0.2f, 1f); // orange
    }
}
