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
        [ValueRange(0f, 20f)]
        public float Brightness = 1.5f;

        [Category("Broken")]
        [Description("Something passed it in a glancing blow and took one side apart: the crust there broken into plates carried off along the blow, cracks across the face, the dark core showing, chunks, grit and dust trailing after. Drawn into a texture (Graphics.BrokenMoons) and again only when the sun has moved across it")]
        public bool Broken;

        [Description("Where the middle of the broken side lies, in degrees round the disc as seen: 0 = its right, 90 = its top")]
        [ValueRange(0f, 360f)]
        public float BreakBearing = 30f;

        [Description("How far the broken side turns towards us from the rim, in degrees: 0 = on the rim, 90 = the middle of the disc")]
        [ValueRange(0f, 90f)]
        public float BreakTilt = 35f;

        [Description("Size of the broken side: its angular radius on the sphere, in degrees")]
        [ValueRange(15f, 90f)]
        public float BreakSize = 55f;

        [Description("The way the blow carried what it broke, in degrees round the disc as seen: 0 = to the right, 90 = up")]
        [ValueRange(0f, 360f)]
        public float BlowBearing = 35f;

        [Description("How far the blow's way turns towards us, in degrees (0 = across the sky)")]
        [ValueRange(-60f, 60f)]
        public float BlowTilt = 15f;

        [Description("Plates the broken side split into")]
        [ValueRange(4, 40)]
        public int Plates = 18;

        [Description("How far the furthest plates have flown, in moon radii")]
        [ValueRange(0f, 4f)]
        public float Spread = 2.2f;

        [Description("Thickness of the crust the plates are made of, in moon radii")]
        [ValueRange(0.03f, 0.25f)]
        public float CrustThickness = 0.1f;

        [Description("Small chunks thrown further along the blow")]
        [ValueRange(0, 48)]
        public int Chunks = 20;

        [Description("Cracks across the face: 0 = none, 1 = round the break with a few reaching out, 2 = across most of it")]
        [ValueRange(0f, 2f)]
        public float Cracks = 1f;

        [Description("The dust plume out of the break: 0 = none")]
        [ValueRange(0f, 3f)]
        public float Dust = 1f;

        [Description("Grit trailing after the plates: 0 = none")]
        [ValueRange(0f, 3f)]
        public float Grit = 1f;

        [Description("Albedo of the broken rock (the sides of the plates, the chunks, the grit), against the surface texture's average of 1")]
        public Color3 InnerColor = new Color3(0.42f, 0.40f, 0.39f);

        [Description("Albedo of the core under the crust, where the plates have gone")]
        public Color3 CoreColor = new Color3(0.16f, 0.15f, 0.15f);

        [Description("Albedo of the dust")]
        public Color3 DustColor = new Color3(0.32f, 0.30f, 0.29f);

        [Description("Picks the plates' and chunks' layout")]
        public int Seed = 1;
    }
}
