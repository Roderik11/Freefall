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
        Cone         // Disc of ShapeRadius; velocity fans outward by ConeAngle
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
        VelocityStretched  // Quad's up axis follows velocity, length grows with speed (rain, sparks)
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

        [Category("Appearance")]
        [Description("RGBA color at birth")]
        public Vector4 ColorStart = Vector4.One;

        [Category("Appearance")]
        [Description("RGBA color at death")]
        public Vector4 ColorEnd = new(1, 1, 1, 0);

        [Category("Appearance")]
        [Description("Maximum rotation speed (radians/sec)")]
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
    }
}
