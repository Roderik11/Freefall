using System;
using System.Numerics;
using Description = System.ComponentModel.DescriptionAttribute;
using Category = System.ComponentModel.CategoryAttribute;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Components
{
    public enum ParticleRenderMode
    {
        Forward,      // Renders to Composite after composition — gets fog automatically
        Transparent   // Renders to GBuffer in transparent pass
    }

    /// <summary>Where particles are spawned, relative to the emitter transform.</summary>
    public enum EmissionShape
    {
        Point,       // Single point at the emitter origin
        Sphere,      // Inside (or on the surface of) a sphere of ShapeRadius
        Hemisphere,  // Upper half of a sphere (local +Y)
        Circle,      // Flat disc in the local XZ plane
        Box,         // Box of ShapeExtents (half sizes)
        Cone,        // Disc of ShapeRadius; velocity fans outward by ConeAngle
        Surf,        // Where the ocean's surf breaks, within ShapeExtents.XZ of the camera (SurfMist; particle_compute.hlsl)
        Ground       // On whatever the sky sees within ShapeExtents.XZ of the emitter: the highest mesh (a roof, a street, a
                     // crown) or the terrain (the sky captures' top-down view); rain's splashes
    }

    /// <summary>How the initial velocity direction is chosen.</summary>
    public enum EmitDirectionMode
    {
        Directional, // EmitDirection (local space) with random cone spread of SpreadAngle
        Radial,      // Outward from the emitter origin through the spawn position
        Random       // Uniformly random direction
    }

    /// <summary>How particle quads are oriented.</summary>
    public enum ParticleBillboardMode
    {
        CameraFacing,      // Classic billboard, always faces the camera
        VelocityStretched, // Quad's up axis follows velocity, length grows with speed (rain, sparks)
        Upright            // Stands along world up (flames): a camera-facing quad swings its foot off the fire when
                           // seen from above or near the screen edge. With RotationRange 0 it turns about the axis to
                           // face the camera; above 0 every card keeps its own random angle about it (crossed cards)
    }

    /// <summary>What particles collide against.</summary>
    public enum ParticleCollisionMode
    {
        None,
        Plane,   // Infinite horizontal plane at PlaneHeight (world Y)
        Depth    // Screen-space collision against the depth GBuffer (opaque geometry)
    }

    public enum ParticleCollisionResponse
    {
        Kill,    // Particle dies on contact
        Bounce   // Reflect velocity around the surface normal, scaled by Bounciness
    }

    /// <summary>
    /// A GPU particle emitter. The component is data only: <see cref="ParticleSystem"/> gathers every
    /// emitter each frame and simulates and draws them together from one shared pool. The pool space an
    /// emitter gets follows from EmitRate x Lifetime, so there is no particle limit to set.
    /// Disabling the component stops emission; particles already alive finish their life.
    /// </summary>
    [Icon("icon_particle.png")]
    public class ParticleEmitter : Component
    {
        // ── Emission ──

        [Category("Emission")]
        [Description("Particles emitted per second")]
        [ValueRange(0f, 20000f)]
        public float EmitRate = 100f;

        [Category("Emission")]
        [Description("Particle lifetime in seconds")]
        [ValueRange(0.1f, 30f)]
        public float Lifetime = 2.0f;

        [Category("Emission")]
        [Description("Random lifetime variation (0 = exact, 1 = 0..2x lifetime)")]
        [ValueRange(0f, 1f)]
        public float LifetimeRandomness = 0.2f;

        // ── Shape ──

        [Category("Shape")]
        [Description("Volume particles are spawned in (local to the emitter transform)")]
        public EmissionShape Shape = EmissionShape.Point;

        [Category("Shape")]
        [Description("Radius for Sphere / Hemisphere / Circle / Cone shapes")]
        [ValueRange(0f, 100f)]
        public float ShapeRadius = 1.0f;

        [Category("Shape")]
        [Description("Half-extents for the Box shape")]
        public Vector3 ShapeExtents = new(1, 1, 1);

        [Category("Shape")]
        [Description("Cone shape only: angle (degrees) the velocity fans outward from the local up axis")]
        [ValueRange(0f, 89f)]
        public float ConeAngle = 25f;

        [Category("Shape")]
        [Description("Spawn only on the surface/edge of the shape instead of filling its volume")]
        public bool EmitFromShell = false;

        [Category("Shape")]
        [Description(@"Surf shape: mist over the broken water and behind the crests instead of spray off the breaking lips.
         Either way particles are born only where a wave breaks (EmitRate counts the slots respawned, each trying up to 16 points around the camera)")]
        public bool SurfMist = false;

        [Category("Shape")]
        [Description(@"Box shape: a particle that leaves the box sideways comes back in on the far side (rain and snow round
         the camera: flakes blown tens of metres in their fall stay in view, whatever the wind)")]
        public bool WrapInBox = false;

        // ── Motion ──

        [Category("Motion")]
        [Description("How the initial velocity direction is chosen")]
        public EmitDirectionMode DirectionMode = EmitDirectionMode.Directional;

        [Category("Motion")]
        [Description("Base emission direction in emitter-local space (Directional mode)")]
        public Vector3 EmitDirection = new(0, 1, 0);

        [Category("Motion")]
        [Description("Random cone half-angle (degrees) around the emission direction. 0 = exact, 180 = any direction")]
        [ValueRange(0f, 180f)]
        public float SpreadAngle = 15f;

        [Category("Motion")]
        [Description("Min/max initial speed (m/s), randomized per particle")]
        public Vector2 SpeedRange = new(1.5f, 2.5f);

        [Category("Motion")]
        [Description("Gravity force applied each frame")]
        public Vector3 Gravity = new(0, -9.81f, 0);

        [Category("Motion")]
        [Description("Air resistance. Pulls velocity toward Wind; with gravity this yields a terminal velocity")]
        [ValueRange(0f, 20f)]
        public float Drag = 0f;

        [Category("Motion")]
        [Description("Air velocity (world space). Only has an effect when Drag > 0")]
        public Vector3 Wind = Vector3.Zero;

        [Description("Share of the weather's wind (EnvironmentController) added to Wind. Like Wind it needs Drag > 0")]
        [ValueRange(0f, 1f)]
        public float WeatherWind = 0f;

        [Description("Swirling force that changes over space and time (m/s²): smoke curls, embers wander")]
        [ValueRange(0f, 50f)]
        public float Turbulence = 0f;

        [Description("Size of the swirls (meters)")]
        [ValueRange(0.05f, 20f)]
        public float TurbulenceSize = 1f;

        [Description("Share of the emitter's own movement its live particles take part in. 0 = they stay where they were let go (the smoke a carried torch leaves behind), 1 = they move along with it (the torch's flame). Sprites that move along and have a WindLean also lean away from the way the emitter goes, as in a wind")]
        [ValueRange(0f, 1f)]
        public float FollowEmitter = 0f;

        // ── Collision ──

        [Category("Collision")]
        [Description("What particles collide against")]
        public ParticleCollisionMode CollisionMode = ParticleCollisionMode.None;

        [Category("Collision")]
        [Description("What happens on contact")]
        public ParticleCollisionResponse CollisionResponse = ParticleCollisionResponse.Kill;

        [Category("Collision")]
        [Description("Plane mode: world-space Y of the collision plane")]
        public float PlaneHeight = 0f;

        [Category("Collision")]
        [Description("Depth mode: how far behind a surface (meters) still counts as a hit")]
        [ValueRange(0.01f, 5f)]
        public float CollisionThickness = 0.5f;

        [Category("Collision")]
        [Description("Bounce: velocity kept after impact (0 = stop, 1 = perfect bounce)")]
        [ValueRange(0f, 1f)]
        public float Bounciness = 0.3f;

        [Category("Collision")]
        [Description(@"Rain and snow: a particle dies where something covers it from the sky (roofs, crowns, overhangs: the sky
         captures' top-down view; the terrain's surface over a cave), on screen or off, and none is born there")]
        public bool StopUnderCover = false;

        // ── Appearance ──

        [Category("Appearance")]
        [Description("Particle size at birth (meters)")]
        [ValueRange(0f, 20f)]
        public float StartSize = 0.3f;

        [Category("Appearance")]
        [Description("Particle size at death (meters)")]
        [ValueRange(0f, 20f)]
        public float EndSize = 0.3f;

        [Category("Appearance")]
        [Description("Random size variation applied to both start and end (0 = exact, 1 = 0..2x)")]
        [ValueRange(0f, 1f)]
        public float SizeRandomness = 0.3f;

        [Category("Appearance")]
        [Description("Quad height / width. 1 = square")]
        [ValueRange(0.05f, 20f)]
        public float Aspect = 1f;

        // Colours are premultiplied: RGB is what the particle adds to the picture, alpha how much it hides
        // of what is behind it. Alpha 0 is additive (fire, sparks); to fade out, RGB has to go to 0 as well.

        [Category("Appearance")]
        [Description("RGBA color at birth. RGB is added to the picture, alpha hides what is behind: alpha 0 = additive")]
        public Vector4 ColorStart = Vector4.One;

        [Category("Appearance")]
        [Description("RGBA color at death. Fading out needs RGB at 0 too")]
        public Vector4 ColorEnd = new(1, 1, 1, 0);

        [Description("Lit like a surface by the sky and the sun (smoke, rain, snow, dust): RGB × Intensity is its albedo, grey " +
                     "by day and dark at night. Off: the particle gives light (flames, embers, sparks) and RGB × Intensity is " +
                     "its luminance in nits (a flame ~40000), shown as a light source (Atmosphere.EmissiveExposure)")]
        public bool Lit = true;

        [Description("Lit: water drops (rain). A drop shows the sky above and behind it through itself, like a tiny lens: it " +
                     "takes the sky's light from above in full, not the average with the dark ground below that a speck of " +
                     "dust takes; RGB × Intensity is how much of it comes through (~1). Diffuse-lit at 0.6 the rain vanished " +
                     "against dark walls")]
        public bool Water = false;

        [Description("Lit: how much of the sun's and moon's light the particles throw on along its way, so they glow with the " +
                     "light behind them and dim with it behind you (droplets: spray, mist, rain ~0.7; thin smoke ~0.3). " +
                     "0 = the same from every side")]
        [ValueRange(0f, 0.9f)]
        public float ForwardScatter = 0f;

        [Description("Lit: share of the light scattered evenly, as it is when it bounces around inside a dense puff (smoke, " +
                     "cloud: 0.4). 0 for particles that each scatter the light once, on their own: rain drops (the forward " +
                     "scattering alone, ForwardScatter ~0.87 for water)")]
        [ValueRange(0f, 1f)]
        public float MultipleScattering = 0.4f;

        [Description("Lit: inside the near fog volume (VolumetricFogRange) the particles are density in the air instead of " +
                     "sprites, lit as the air is: through the sun's and moon's shadows, by the sky where it is seen and by " +
                     "every point light through its shadow (a fire lights its smoke from inside). Past the volume they are " +
                     "sprites. Soft, at the volume's resolution: smoke, steam, dust clouds")]
        public bool Volumetric = false;

        [Description("Multiplier on RGB: with Lit, of the albedo (at most 1 for a particle that is one thing, a puff; more " +
                     "for a few particles standing in for many, as rain streaks do for the drops round them); without, " +
                     "the luminance in nits")]
        [ValueRange(0f, 200000f)]
        public float Intensity = 1f;

        [Description("Share of its lifetime a particle takes to appear (0 = at once)")]
        [ValueRange(0f, 1f)]
        public float FadeIn = 0f;

        [Category("Appearance")]
        [Description("Maximum rotation speed (radians/sec); above 0 particles also start at a random angle. Upright quads rotate about the up axis")]
        [ValueRange(0f, 10f)]
        public float RotationRange = 1.0f;

        [Category("Appearance")]
        public Texture? ParticleTexture;

        // ── Flipbook ──

        [Category("Flipbook")]
        [Description("Number of frames in the flipbook atlas (0 = no animation)")]
        public int FlipbookFrameCount = 0;

        [Category("Flipbook")]
        [Description("Flipbook playback speed (frames/sec)")]
        public float FlipbookAnimSpeed = 10f;

        [Category("Flipbook")]
        [Description("Number of columns in the flipbook atlas")]
        public int FlipbookColumns = 1;

        [Category("Flipbook")]
        [Description("Number of rows in the flipbook atlas")]
        public int FlipbookRows = 1;

        [Description("Each particle starts at a random frame, so the sprites of one emitter do not animate in step")]
        public bool FlipbookRandomStart = false;

        [Description("Cross-fade from frame to frame instead of stepping")]
        public bool FlipbookBlend = false;

        // ── Rendering ──

        [Category("Rendering")]
        [Description("Which render pass to use")]
        public ParticleRenderMode RenderMode = ParticleRenderMode.Forward;

        [Category("Rendering")]
        [Description("Quad orientation")]
        public ParticleBillboardMode BillboardMode = ParticleBillboardMode.CameraFacing;

        [Category("Rendering")]
        [Description("VelocityStretched: extra quad length per m/s of speed (meters)")]
        [ValueRange(0f, 1f)]
        public float StretchFactor = 0.05f;

        [Description("CameraFacing / Upright: the sprite leans with the wind instead of being carried by it. Its top shifts by this many meters per m/s of Wind (plus the WeatherWind share), the bottom stays. For flames; keep Drag at 0")]
        [ValueRange(0f, 0.5f)]
        public float WindLean = 0f;

        [Description("CameraFacing / Upright: where the quad sits on its particle, in quad widths (X) and heights (Y). (0, 0) = centred on it, (0, 0.5) = standing on it: the foot of a flame stays on a torch's head however the torch is turned")]
        public Vector2 Pivot = Vector2.Zero;

        [Category("Rendering")]
        [Description("Fade particles near opaque surfaces")]
        public bool SoftParticles = true;

        [Category("Rendering")]
        [Description("Depth range for soft particle fade (meters)")]
        [ValueRange(0.01f, 10f)]
        public float SoftRange = 0.5f;

        // ── Runtime overrides (not serialized, not shown) ──
        // Set by systems like EnvironmentController so the authored EmitRate / Wind stay intact in the scene file.

        /// <summary>Multiplier on EmitRate applied at runtime. 0 stops emission without touching EmitRate.</summary>
        [Freefall.Reflection.DontSerialize, System.ComponentModel.Browsable(false)]
        public float EmitRateScale = 1f;

        /// <summary>When set, replaces Wind for the simulation without overwriting the authored value.</summary>
        [Freefall.Reflection.DontSerialize, System.ComponentModel.Browsable(false)]
        public Vector3? WindOverride;

        /// <summary>Index of this emitter's entry in <see cref="Graphics.ParticleSystem"/>; -1 while it has none.</summary>
        internal int SystemIndex = -1;

        // A disabled emitter whose particles are gone sleeps in the system: it has to hear of it
        protected override void OnEnabledChanged() => Systems.Get<Graphics.ParticleSystem>()?.EnabledChanged(this);
    }
}
