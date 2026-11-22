// GPU Particle System — data shared by particle_compute.hlsl and particle_draw.fx.
// All emitters live in one pool: every emitter owns a contiguous run of 256-slot chunks.

#ifndef PARTICLE_COMMON_HLSLI
#define PARTICLE_COMMON_HLSLI

#define PARTICLE_CHUNK_SIZE   256
#define PARTICLE_CHUNK_SHIFT  8
#define PARTICLE_NO_EMITTER   0xFFFFFFFF

struct ParticleCore        // 32 bytes — simulation state
{
    float3 Position;
    float  Age;
    float3 Velocity;
    float  Lifetime;       // 0 = slot never used; Age >= Lifetime = dead
};

struct ParticleVisual      // 16 bytes — per-particle variation, everything else comes from the emitter row
{
    float SizeScale;       // random factor on the emitter's start/end size
    float Rotation;        // radians
    float RotationSpeed;   // radians/sec
    float Random;          // 0..1, fixed for the particle's life (flipbook start frame)
};

// One row per emitter, rewritten by the CPU every frame (ParticleSystem.EmitterRow in C#).
// Laid out in 16-byte rows so no vector straddles a boundary and the C# struct maps 1:1.
struct ParticleEmitter     // 432 bytes
{
    float3 Position;       float  ParticleRadius;      // largest half-extent of one particle quad
    float3 EmitDirection;  float  SpreadCos;           // emitter-local, normalized / cos(spread half-angle)
    float3 Gravity;        float  Lifetime;
    float3 ShapeExtents;   float  ShapeRadius;
    float3 AxisX;          float  ConeTan;             // emitter world-space basis
    float3 AxisY;          float  LifetimeRandomness;
    float3 AxisZ;          float  SizeRandomness;
    float3 Wind;           float  Drag;
    float2 SpeedRange;     float  RotationRange;       float Bounciness;
    uint   Shape;          uint   DirectionMode;       uint  EmitFromShell;      uint  CollisionMode;
    uint   CollisionResponse; float PlaneHeight;       float CollisionThickness; uint  RandomSeed;
    // Ring: slots [FirstSlot, FirstSlot + Capacity). This frame respawns EmitCount slots from EmitStart.
    uint   FirstSlot;      uint   Capacity;            uint  EmitStart;          uint  EmitCount;
    float4 ColorStart;
    float4 ColorEnd;
    float2 SizeStartEnd;   float  Aspect;              float StretchFactor;
    uint   TextureIdx;     uint   FlipbookCols;        uint  FlipbookRows;       uint  FlipbookFrameCount;
    float  FlipbookAnimSpeed; uint BillboardMode;      uint  SoftEnabled;        float SoftRange;
    float3 BoundsCenter;   float  BoundsRadius;        // everything this emitter can have alive
    uint   RenderMode;     uint   Flags;               float Intensity;          float FadeInRate;   // brightness on RGB / 1 / (share of life to appear), 0 = appears at once
    float  Turbulence;     float  TurbulenceFrequency; float WindLean;           float Follow;       // m/s² / 1 / swirl size / m of lean at the sprite's top per m/s of Wind / FollowEmitter
    float3 Carry;          float  PivotX;              // how far live particles move along with the emitter this frame (FollowEmitter)
    float3 LeanWind;       float  PivotY;              // the wind a sprite leans in: Wind less the emitter's own motion it follows / quad offset in its sizes
    float  ForwardScatter; float  MultipleScattering;  float PyroFire;           float PyroTemperature; // lit: Henyey-Greenstein g of the light (ParticleLight) / its even share (ParticlePhase) / pyro: the fire's nits and kelvin at its brightest
    uint   MotionTextureIdx; float MotionStrength;     uint  Jets;               float JetSpreadCos; // the flipbook's motion atlas (0 = none) / its full-scale motion in shares of a frame / a burst in jets: their count (bits 0-15, 0 = none) and slowest share x 255 (bits 16-23) / cos of the spread round a jet
    float3 Inherit;        uint   Places;              // world m/s a particle born now starts with on top of its own (InheritVelocity × the emitter's velocity) / its bursts' places this frame (BurstAt): first in bits 0-23, count in 24-31, 0 = none
    // Mesh mode: the mesh's centre (it turns about it, and it is where the particle is) / its run of the mesh lists (the
    // camera's at MeshList, the cascades' at MeshList x PARTICLE_MESH_CASCADES)
    float3 MeshCenter;     uint   MeshList;
    // Mesh mode: the most indices of the parts it draws (the draws' vertex count) / 1 / (share of life it shrinks away in at
    // the end), 0 = never
    uint   MeshVertices;   float  ShrinkRate;          uint  MeshPad0;           uint  MeshPad1;
};

// A burst let go at a place of its own this frame (ParticleEmitter.BurstAt; ParticleSystem.PlaceRow): the emission's slots
// [Begin, End) are born as if the emitter stood there, turned by Rotation, with Velocity (already × InheritVelocity) added
struct BurstPlace          // 48 bytes
{
    float3 Position;       uint   Begin;
    float4 Rotation;       // quaternion xyzw
    float3 Velocity;       uint   End;
};

// ParticleEmitter.RenderMode
#define PARTICLE_MODE_FORWARD      0
#define PARTICLE_MODE_TRANSPARENT  1
#define PARTICLE_MODE_COUNT        2   // the sprite modes, each a draw list of PoolSlots entries
#define PARTICLE_MODE_MESH         2   // each particle a mesh in the G-buffer and the sun's cascades (gbuffer.fx pass Particles)

// Mesh mode: entries an emitter's particle may take in its shadow list (one per cascade it falls in)
#define PARTICLE_MESH_CASCADES     8
// Mesh mode: uints of indirect arguments per emitter row: DrawInstanced for the camera, then for the cascades
#define PARTICLE_MESH_ARGS         8

// The tumble of a mesh particle: a random turn it is born with and an axis it spins about at RotationSpeed (rad/s), both
// from its Random, so nothing is stored for them. A quaternion (xyzw) at its age.
uint ParticleMeshHash(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float ParticleMeshRand(uint seed) { return float(ParticleMeshHash(seed)) / 4294967295.0; }

float4 ParticleMeshQuatMul(float4 a, float4 b)
{
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

float3 ParticleMeshRotate(float4 q, float3 v)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float4 ParticleMeshTurn(ParticleVisual vis, float age)
{
    uint seed = asuint(vis.Random) * 0x9E3779B9u + 0x7F4A7C15u;
    // Born at any turn (Shoemake: uniform over all rotations)
    float u1 = ParticleMeshRand(seed), u2 = ParticleMeshRand(seed + 1u) * 6.283185307, u3 = ParticleMeshRand(seed + 2u) * 6.283185307;
    float a = sqrt(1.0 - u1), b = sqrt(u1);
    float4 born = float4(a * sin(u2), a * cos(u2), b * sin(u3), b * cos(u3));
    // Spinning about an axis of its own, fixed in the world
    float z = ParticleMeshRand(seed + 3u) * 2.0 - 1.0, phi = ParticleMeshRand(seed + 4u) * 6.283185307;
    float r = sqrt(saturate(1.0 - z * z));
    float3 axis = float3(r * cos(phi), r * sin(phi), z);
    float halfAngle = 0.5 * (vis.Rotation + vis.RotationSpeed * age);
    float s, c;
    sincos(halfAngle, s, c);
    return ParticleMeshQuatMul(float4(axis * s, c), born);
}

// A mesh particle's scale at age t (share of its life): its size from StartSize to EndSize (the mesh's own size at 1),
// grown in over FadeIn and shrunk away over the end of its life (ShrinkRate)
float ParticleMeshScale(ParticleEmitter e, ParticleVisual vis, float t)
{
    float scale = lerp(e.SizeStartEnd.x, e.SizeStartEnd.y, t) * vis.SizeScale;
    if (e.FadeInRate > 0.0) scale *= saturate(t * e.FadeInRate);
    if (e.ShrinkRate > 0.0) scale *= saturate((1.0 - t) * e.ShrinkRate);
    return scale;
}

// ParticleEmitter.Flags
#define PARTICLE_FLAG_RESET            1   // the emitter's slots hold someone else's data: kill them
#define PARTICLE_FLAG_FLIPBOOK_RANDOM  2   // each particle starts at its own frame
#define PARTICLE_FLAG_FLIPBOOK_BLEND   4   // cross-fade from frame to frame
#define PARTICLE_FLAG_LIT              8   // Forward mode: the colour is an albedo lit by the sky and the sun (smoke, rain); else a luminance in nits
#define PARTICLE_FLAG_SURF_MIST       16   // shape Surf: mist over the surf's broken water, else spray off its breaking lips
#define PARTICLE_FLAG_VOLUME          32   // Lit + Volumetric: density in the near fog volume inside its range, sprites past it
#define PARTICLE_FLAG_WRAP            64   // shape Box + WrapInBox: a particle leaving the box sideways comes in on the far side
#define PARTICLE_FLAG_COVER          128   // StopUnderCover: dies (and is not born) where the sky is covered over it
#define PARTICLE_FLAG_WATER          256   // Lit + Water: a drop shows the sky above it through itself (rain)
#define PARTICLE_FLAG_PYRO           512   // the texture is a pyro bake: R fire, G smoke coverage, B smoke key light, A fire glow in the smoke

#endif
