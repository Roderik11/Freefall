using System.Numerics;
using Description = System.ComponentModel.DescriptionAttribute;
using Category = System.ComponentModel.CategoryAttribute;
using Freefall.Base;

namespace Freefall.Components
{
    /// <summary>
    /// A ribbon left behind by the entity: one continuous strip through the places it has been,
    /// turned to face the camera, that changes width and colour from the entity to its far end
    /// (an arrow's streak, the arc of a blade). The component is data only:
    /// <see cref="Graphics.TrailSystem"/> records the path of every trail and draws them together.
    /// A trail whose entity is destroyed fades out where it is.
    /// </summary>
    [Icon("icon_particle.png")]
    public class TrailRenderer : Component
    {
        [Category("Trail")]
        [Description("Seconds a point of the trail lasts: the trail is as long as the way the entity went in that time")]
        [ValueRange(0.05f, 10f)]
        public float Lifetime = 0.5f;

        [Description("Leave a trail. Off: nothing is added and what is there fades out")]
        public bool Emitting = true;

        [Description("Meters the entity moves before the trail gets another point. Smaller follows a curve more closely")]
        [ValueRange(0.01f, 5f)]
        public float PointSpacing = 0.1f;

        [Category("Appearance")]
        [Description("Width at the entity (meters)")]
        [ValueRange(0f, 5f)]
        public float StartWidth = 0.05f;

        [Description("Width at the far end (meters)")]
        [ValueRange(0f, 5f)]
        public float EndWidth = 0f;

        // Colours are premultiplied, as a particle's: RGB is what the ribbon adds to the picture, alpha how
        // much it hides of what is behind it. Alpha 0 is additive; to fade out, RGB has to go to 0 as well.

        [Description("RGBA color at the entity. RGB is added to the picture, alpha hides what is behind: alpha 0 = additive")]
        public Vector4 ColorStart = new(1f, 1f, 1f, 0.5f);

        [Description("RGBA color at the far end. Fading out needs RGB at 0 too")]
        public Vector4 ColorEnd = Vector4.Zero;

        [Description("Off (default): the ribbon reads as light, RGB × Intensity its luminance in nits shown as a light source " +
                     "(Atmosphere.EmissiveExposure): an arrow's or a blade's streak stays readable at night (~4000). On: lit like a " +
                     "surface by the sky and the sun, RGB × Intensity its albedo: grey by day, dark at night")]
        public bool Lit = false;

        [Description("Multiplier on RGB: the luminance in nits, or with Lit the albedo (keep RGB × Intensity at most 1)")]
        [ValueRange(0f, 200000f)]
        public float Intensity = 1f;

        [Category("Blade")]
        [Description("A blade's trail: the ribbon is the sheet a line sweeps, from the entity to this point in the entity's own space (the entity at the tip of a blade, this towards its grip), bright at the entity's edge and fading to the other. It does not turn to the camera and the widths are not used. Zero = an ordinary trail")]
        public Vector3 BladeOffset;

        /// <summary>Index of this trail's entry in <see cref="Graphics.TrailSystem"/>; -1 while it has none.</summary>
        internal int SystemIndex = -1;
    }
}
