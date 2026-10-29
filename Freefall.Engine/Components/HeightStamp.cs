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
    /// A closed spline is a mask with its falloff inside. On the stamp's own entity the stamp covers the outline
    /// grown by Radius and fades in over Falloff from that edge: nothing reaches past it. A round stamp (no spline
    /// of its own) on a child of an entity with a closed spline is confined to that outline the same way: its own
    /// circle and texture, faded inward over its Falloff (see <see cref="MaskSpline"/>). Move it around under the
    /// spline and it never leaves the shape. Open splines are corridors with the falloff outside, as before.
    ///
    /// A radial stamp's heightmap spans its Radius + Falloff; a spline stamp's spans the spline's area (closed) or
    /// corridor plus falloff (open) and turns with it.
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
        /// Closed spline only: the stamp covers everything outside the outline instead of inside, fading in over
        /// Falloff outward from it (Radius moves that edge into the outline). A seabed dug around an island's shelf:
        /// one outline along the shelf edge, BlendMode Min, the deep as its height. Its Heightmap spans the terrain.
        /// </summary>
        public bool Outside = false;

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

        /// <summary>
        /// Ridge and furrow, as a field ploughed the same way year after year: the ground rises in long ridges this far
        /// apart (metres, crest to crest) with furrows between, faded out over Falloff towards the edge, the headland
        /// the plough team turned on. They read from far off as stripes of light and shade. 0 = none. With BlendMode
        /// Add and no HeightOffset the stamp only ploughs: the ridges' mean height is the ground's.
        /// </summary>
        [System.ComponentModel.Category("Ridges")]
        [ValueRange(0f, 20f)]
        public float RidgeSpacing = 0f;

        /// <summary>Height from the bottom of a furrow to the crest of a ridge (m).</summary>
        [ValueRange(0f, 2f)]
        public float RidgeHeight = 0.4f;

        /// <summary>
        /// The ridges' direction in degrees, turned from the spline's longest control edge (0 = along the field's
        /// length), or from world +X without a spline.
        /// </summary>
        [ValueRange(-180f, 180f)]
        public float RidgeAngle = 0f;

        /// <summary>
        /// How far the ridges swing to the side towards the field's ends (m): the reverse S of a plough team that began
        /// to turn before the headland. 0 = straight.
        /// </summary>
        [ValueRange(0f, 20f)]
        public float RidgeBend = 2f;

        /// <summary>
        /// The closed spline that confines this stamp: the parent entity's, when this is a round local stamp with no
        /// spline of its own. Null otherwise.
        /// </summary>
        public Spline? MaskSpline
        {
            get
            {
                if (IsGlobal || IsSplineMode) return null;
                var spline = Transform?.Parent?.Entity?.GetComponent<Spline>();
                return spline is { Closed: true } && spline.Points.Count >= 3 ? spline : null;
            }
        }

        // A closed spline of its own is a mask: the falloff lies inside the edge
        protected override bool FalloffInsideClosedSpline => true;

        protected override bool ClosedSplineOutside => Outside;

        protected override Spline? GizmoMaskSpline => MaskSpline;

        protected override Color4 GizmoColor => new Color4(0.3f, 0.9f, 0.3f, 1f); // green
    }
}
