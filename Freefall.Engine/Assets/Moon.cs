using Freefall.Graphics;
using Vortice.Mathematics;
using Category = System.ComponentModel.CategoryAttribute;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Assets
{
    /// <summary>
    /// A moon of the world: its orbit, its light and its disc in the sky. The sky lists the world's moons
    /// (<c>SkyboxRenderer.Moons</c>). Each falls behind the sun by a whole day every <see cref="PeriodDays"/>, so it
    /// rises later each night and runs from new (beside the sun) through full (opposite it) and back; the sun lights
    /// the half of the disc that faces it. There is one directional light: the sun by day, at night the moon that
    /// gives the most light (its illuminance × the lit part of its disc through the air, nothing below the horizon).
    /// The brightest one also lights the night sky and its haze (Graphics.Atmosphere).
    /// </summary>
    [CreateAsset("Moon")]
    public class Moon : Asset
    {
        [Category("Orbit")]
        [Description("Days from one full moon to the next")]
        [ValueRange(1f, 100f)]
        public float PeriodDays = 29.5f;

        [Description("Where in its cycle the moon is on day 0: 0 = new, 0.5 = full")]
        [ValueRange(0f, 1f)]
        public float PhaseOffset = 0.5f;

        [Description("Compass heading of moonrise in degrees (0=+X, 90=+Z), like the sky's SunAzimuthAngle")]
        [ValueRange(0f, 360f)]
        public float Azimuth = 30f;

        [Description("Highest angle above the horizon in degrees, reached 12 h after its rise")]
        [ValueRange(5f, 89f)]
        public float MaxElevation = 50f;

        [Category("Light")]
        public Color3 LightColor = new Color3(0.33f, 0.5f, 1.0f);

        [Description("Illuminance in lux at full moon, above the atmosphere (our moon: about 0.25), scaled by the phase; the air dims it like the sun's, nothing below the horizon. Its hue only: LightColor keeps its brightness")]
        [ValueRange(0f, 2f)]
        public float LightIntensity = 0.22f;

        [Category("Disc")]
        [Description("Surface of the full disc as seen from the ground; empty = a plain grey disc")]
        public Texture Texture;

        [Description("Relief of the disc, a tangent-space normal map in the same view as Texture (red = east, green = up on the disc): craters and maria catch the light along the line between day and night; empty = a smooth sphere")]
        public Texture NormalMap;

        [Description("Apparent diameter in degrees (our moon: 0.5)")]
        [ValueRange(0.2f, 20f)]
        public float Size = 3f;

        public Color3 Tint = new Color3(1f, 1f, 1f);

        [Description("Brightness of the disc against the light it gives: 1 = a disc this size whose light is exactly LightIntensity")]
        [ValueRange(0f, 10f)]
        public float Brightness = 1.5f;
    }
}
