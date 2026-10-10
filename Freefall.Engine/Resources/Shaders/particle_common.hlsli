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
    float _pad;
};

// One row per emitter, rewritten by the CPU every frame (ParticleSystem.EmitterRow in C#).
// Laid out in 16-byte rows so no vector straddles a boundary and the C# struct maps 1:1.
struct ParticleEmitter     // 304 bytes
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
    uint   RenderMode;     uint   Flags;               uint  _pad0;              uint  _pad1;
};

// ParticleEmitter.RenderMode
#define PARTICLE_MODE_FORWARD      0
#define PARTICLE_MODE_TRANSPARENT  1
#define PARTICLE_MODE_COUNT        2

// ParticleEmitter.Flags
#define PARTICLE_FLAG_RESET   1   // the emitter's slots hold someone else's data: kill them

#endif
