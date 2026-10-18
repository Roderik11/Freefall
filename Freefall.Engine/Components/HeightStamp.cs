using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// Non-destructive height stamp. Flattens terrain to match the stamp shape
    /// (entity Y + HeightOffset, or spline path height).
    /// When a Heightmap is assigned, applies a spatially-varying height pattern
    /// scaled by Strength, with rotation from the entity transform. InvertShape makes it dig.
    ///
    /// A radial stamp's heightmap spans its Radius + Falloff. A spline stamp's spans everything the spline
    /// reaches and turns with it, so the spline's shape is the stamp's falloff: on a closed spline the texture
    /// applies in full over the area and fades out across the Falloff outside the outline (a lake's depths).
    ///
    /// With Shape = Global and a Heightmap, the heightmap is stretched over the whole terrain and
    /// replaces what lower-priority stamps built (height = heightmap * Strength, plus the entity's Y and
    /// HeightOffset): that is how an imported heightmap becomes the base of a terrain. A global stamp
    /// without a heightmap does nothing.
    /// </summary>
    [Icon("icon_heightstamp.png")]
    public class HeightStamp : TerrainStamp
    {
        /// <summary>
        /// How the stamp's height combines with what lower-priority stamps built, inside the stamp zone.
        /// Set flattens to it (roads, plots). Add raises the ground by HeightOffset plus the heightmap,
        /// wherever the entity sits vertically (a hill dropped onto existing relief). Max / Min only
        /// raise / only lower to it (a mountain that never digs, a basin that never fills).
        /// Ignored by a global stamp, which always replaces.
        /// </summary>
        [System.ComponentModel.Category("Height")]
        public HeightBlendMode BlendMode = HeightBlendMode.Set;

        /// <summary>
        /// Height offset from the entity/spline position (world units).
        /// </summary>
        public float HeightOffset = 0f;

        /// <summary>
        /// Turn the heightmap's relief upside down: it digs below the target height instead of rising above
        /// it (lake depths, craters). No effect without a Heightmap.
        /// </summary>
        public bool InvertShape = false;

        /// <summary>
        /// Optional heightmap texture. When set, the stamp applies this height pattern
        /// instead of flattening to a uniform height. Sampled in the stamp's local space
        /// with rotation from the entity transform, stretched over the stamp's footprint
        /// (a circle's Radius + Falloff, a spline's whole reach).
        /// </summary>
        public Texture Heightmap;

        /// <summary>
        /// Height strength in world units. Scales the heightmap values.
        /// At Strength=100 and heightmap value=1.0, terrain is pushed 100 units above TargetHeight
        /// (below it with InvertShape). Ignored when no Heightmap is assigned.
        /// </summary>
        [ValueRange(0f, 600f)]
        public float Strength = 50f;

        protected override Color4 GizmoColor => new Color4(0.3f, 0.9f, 0.3f, 1f); // green
    }
}
