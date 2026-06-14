using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// Non-destructive height stamp. Flattens terrain to match the stamp shape
    /// (entity Y + HeightOffset, or spline path height).
    /// When a Heightmap is assigned, applies a spatially-varying height pattern
    /// scaled by Strength, with rotation from the entity transform.
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

        /// <summary>
        /// Optional heightmap texture. When set, the stamp applies this height pattern
        /// instead of flattening to a uniform height. Sampled in the stamp's local space
        /// with rotation from the entity transform.
        /// </summary>
        public Texture Heightmap;

        /// <summary>
        /// Height strength in world units. Scales the heightmap values.
        /// At Strength=100 and heightmap value=1.0, terrain is pushed 100 units above TargetHeight.
        /// Ignored when no Heightmap is assigned.
        /// </summary>
        [ValueRange(0f, 600f)]
        public float Strength = 50f;

        protected override Color4 GizmoColor => new Color4(0.3f, 0.9f, 0.3f, 1f); // green
    }
}
