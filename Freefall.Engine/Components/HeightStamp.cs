using Freefall.Base;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// Non-destructive height stamp. Flattens terrain to match the stamp shape
    /// (entity Y + HeightOffset, or spline path height).
    /// InvertShape negates the displacement for trenches/riverbeds.
    /// </summary>
    public class HeightStamp : TerrainStamp
    {
        /// <summary>
        /// Height offset from the entity/spline position (world units).
        /// </summary>
        public float HeightOffset = 0f;

        /// <summary>
        /// Invert the height displacement. Instead of flattening TO the target height,
        /// push terrain in the opposite direction (trenches, riverbeds).
        /// </summary>
        public bool InvertShape = false;

        protected override Color4 GizmoColor => new Color4(0.3f, 0.9f, 0.3f, 1f); // green
    }
}
