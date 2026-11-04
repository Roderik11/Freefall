using Vortice.Mathematics;
using Description = System.ComponentModel.DescriptionAttribute;
using Category = System.ComponentModel.CategoryAttribute;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Components
{
    /// <summary>
    /// A tube left behind by the entity, shaded by a vfx layers material (vfx_layers.fx) along the way it went: the fire
    /// an arrow or a bolt trails, a ball of flame's, an afterburner's. As long as the way the entity went in its Lifetime;
    /// its material's heat, erosion, smoke, flare and textures go by age along it, as they go by V along a vfx tube, so
    /// the head is the nose and the far end what has burnt out. The component is data only:
    /// <see cref="Graphics.TubeTrailSystem"/> records the path of every tube trail and draws them together. A trail
    /// whose entity is destroyed, or that stops emitting, burns out where it is.
    /// </summary>
    [Icon("icon_particle.png")]
    public class TubeTrail : Component
    {
        [Category("Tube Trail")]
        [Description("A vfx layers material (vfx_layers.fx, Shape 1: a tube), made along the trail as it is along a vfx tube's V: its source at the entity, its end where the trail has run out")]
        public Material Material;

        [Description("Seconds a place on the trail lasts: the trail is as long as the way the entity went in that time (at 30 m/s, 0.09 is 2.7 m)")]
        [ValueRange(0.02f, 10f)]
        public float Lifetime = 0.1f;

        [Description("Leave a trail. Off: nothing is added and what is there burns out")]
        public bool Emitting = true;

        [Description("Meters the entity moves before the trail gets another place. Smaller follows a curve more closely (the tube is drawn smooth through them)")]
        [ValueRange(0.01f, 5f)]
        public float PointSpacing = 0.1f;

        [Description("Its radius in meters, before the material's Flare (which widens it along its length)")]
        [ValueRange(0f, 5f)]
        public float Radius = 0.14f;

        [Description("Meters of trail the material's V spans once (its layers repeat by their tilings within it): the size of its flames along it. They stay where they were made, so they run back along the tube at the entity's own speed")]
        [ValueRange(0.1f, 20f)]
        public float TextureSpan = 2.6f;

        [Category("Look")]
        [Description("Multiplies its colour")]
        public Color3 Color = new Color3(1f, 1f, 1f);

        [Description("Multiplies its light and its smoke (a spell's Glow sets it)")]
        [ValueRange(0f, 4f)]
        public float Intensity = 1f;

        [Description("Added to the material's erosion threshold: raise it to burn the whole away")]
        [ValueRange(-1f, 1f)]
        public float Erode;

        [Description("Multiplies its heat (the value looked up in the ramp): below 1 it cools from white through red")]
        [ValueRange(0f, 4f)]
        public float Heat = 1f;

        /// <summary>Index of this trail's entry in <see cref="Graphics.TubeTrailSystem"/>; -1 while it has none.</summary>
        internal int SystemIndex = -1;
    }
}
