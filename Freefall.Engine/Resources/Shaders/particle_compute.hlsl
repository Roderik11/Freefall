// GPU Particle System — Compute Pipeline
// Dead-list + Alive-list (ping-pong) pool management
// All particle simulation is GPU-resident; CPU only provides emitter config.

#pragma kernel CSInit
#pragma kernel CSEmit
#pragma kernel CSSimulate
#pragma kernel CSBuildDrawArgs

// ────────────── Data Structures ──────────────

struct ParticleCore        // 32 bytes — hot simulation data
{
    float3 Position;       // 12
    float  Age;            //  4
    float3 Velocity;       // 12
    float  Lifetime;       //  4
};

struct ParticleVisual      // 48 bytes — cold render data (written on emit, read on draw)
{
    float2 SizeStartEnd;   //  8  — lerp(start, end, age/lifetime)
    float4 ColorStart;     // 16  — RGBA at birth
    float  Rotation;       //  4  — current rotation angle (radians)
    float  RotationSpeed;  //  4  — radians/sec
    uint   FlipbookFrame;  //  4  — starting frame index
    uint   FlipbookCount;  //  4  — total frames (0 = no animation)
    float  AnimSpeed;      //  4  — flipbook frames/sec
    float  _pad0;          //  4  — 16-byte alignment
};

// ────────────── Push Constants ──────────────

cbuffer PushConstants : register(b3)
{
    uint ParticleCoreUAVIdx;    // 0  RWStructuredBuffer<ParticleCore>
    uint ParticleVisualUAVIdx;  // 1  RWStructuredBuffer<ParticleVisual>
    uint DeadListUAVIdx;        // 2  RWStructuredBuffer<uint>
    uint AliveListReadIdx;      // 3  StructuredBuffer<uint> (current frame read)
    uint AliveListWriteUAVIdx;  // 4  RWStructuredBuffer<uint> (current frame write)
    uint CountersUAVIdx;        // 5  RWStructuredBuffer<uint> [DeadCount, AliveRead, AliveWrite, EmitCount]
    uint DrawArgsUAVIdx;        // 6  RWStructuredBuffer<uint> [VertexCount, InstanceCount, StartVertex, StartInstance]
    uint MaxParticles;          // 7
};

// ────────────── Emitter Parameters ──────────────

cbuffer EmitterParams : register(b1)
{
    float3 EmitterPosition;
    float  DeltaTime;
    float3 EmitDirection;       // emitter-local, normalized
    float  SpreadCos;           // cos(spread half-angle)
    float3 Gravity;
    float  LifetimeParam;
    float2 SizeStartEnd;        // base start / end size
    float  RotationRange;
    uint   RandomSeed;
    float4 ColorStart;
    float4 ColorEnd;
    float  FlipbookFrameCount;
    float  FlipbookAnimSpeed;
    float2 SpeedRange;          // min/max initial speed
    float3 ShapeExtents;        // box half-extents
    float  ShapeRadius;
    uint   Shape;               // EmissionShape enum
    uint   DirectionMode;       // EmitDirectionMode enum
    uint   EmitFromShell;       // 1 = surface only
    float  ConeTan;             // tan(cone angle)
    float4 AxisX;               // emitter world-space basis (xyz), w unused
    float4 AxisY;
    float4 AxisZ;
    float3 Wind;                // air velocity (world)
    float  Drag;                // per-second pull toward Wind
    float  LifetimeRandomness;  // 0..1
    float  SizeRandomness;      // 0..1
    uint   CollisionMode;       // ParticleCollisionMode enum
    uint   CollisionResponse;   // ParticleCollisionResponse enum
    float  PlaneHeight;
    float  CollisionThickness;
    float  Bounciness;
    uint   DepthTexIdx;         // linear view-space depth GBuffer (0 = none)
    uint   NormalTexIdx;        // world-space normal GBuffer (0 = none)
    float3 _collPad;
    row_major float4x4 ViewProjection; // for depth collision projection
};

#define COLL_NONE   0
#define COLL_PLANE  1
#define COLL_DEPTH  2

#define RESP_KILL   0
#define RESP_BOUNCE 1

// Must match C# EmissionShape / EmitDirectionMode
#define SHAPE_POINT      0
#define SHAPE_SPHERE     1
#define SHAPE_HEMISPHERE 2
#define SHAPE_CIRCLE     3
#define SHAPE_BOX        4
#define SHAPE_CONE       5

#define DIR_DIRECTIONAL  0
#define DIR_RADIAL       1
#define DIR_RANDOM       2

// ────────────── RNG (PCG Hash) ──────────────

uint pcg_hash(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand01(uint seed)
{
    return float(pcg_hash(seed)) / 4294967295.0;
}

float rand_range(float lo, float hi, uint seed)
{
    return lerp(lo, hi, rand01(seed));
}

float3 rand_direction(uint seed)
{
    float z = rand01(seed) * 2.0 - 1.0;
    float a = rand01(seed + 1u) * 6.283185307;
    float r = sqrt(1.0 - z * z);
    return float3(r * cos(a), r * sin(a), z);
}

// Uniform random direction inside a cone of half-angle acos(cosMax) around +Z.
float3 rand_cone_z(float cosMax, uint seed)
{
    float z = lerp(cosMax, 1.0, rand01(seed));
    float a = rand01(seed + 1u) * 6.283185307;
    float r = sqrt(saturate(1.0 - z * z));
    return float3(r * cos(a), r * sin(a), z);
}

// Orthonormal basis around an arbitrary unit axis (Duff et al.)
void basis_from_axis(float3 n, out float3 t, out float3 b)
{
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

// Random direction within a cone of half-angle acos(cosMax) around 'axis'.
float3 rand_cone(float3 axis, float cosMax, uint seed)
{
    float3 t, b;
    basis_from_axis(axis, t, b);
    float3 d = rand_cone_z(cosMax, seed);
    return t * d.x + b * d.y + axis * d.z;
}

// Uniform point on a unit disc (XZ plane), or on its rim when shell != 0.
float3 rand_disc_xz(uint seed, uint shell)
{
    float a = rand01(seed) * 6.283185307;
    float r = shell != 0 ? 1.0 : sqrt(rand01(seed + 1u));
    return float3(r * cos(a), 0.0, r * sin(a));
}

// Emitter-local → world rotation
float3 local_to_world(float3 v)
{
    return AxisX.xyz * v.x + AxisY.xyz * v.y + AxisZ.xyz * v.z;
}

// Sample a spawn offset (emitter-local) and shape-derived direction hint.
// 'shapeDir' is the natural outward direction for the shape (used by Cone / Radial).
void sample_shape(uint seed, out float3 localPos, out float3 shapeDir)
{
    localPos = 0;
    shapeDir = float3(0, 1, 0);

    [branch]
    switch (Shape)
    {
        case SHAPE_SPHERE:
        {
            float3 d = rand_direction(seed);
            float r = EmitFromShell != 0 ? 1.0 : pow(rand01(seed + 2u), 1.0 / 3.0);
            localPos = d * r * ShapeRadius;
            shapeDir = d;
            break;
        }
        case SHAPE_HEMISPHERE:
        {
            float3 d = rand_direction(seed);
            d.y = abs(d.y);
            float r = EmitFromShell != 0 ? 1.0 : pow(rand01(seed + 2u), 1.0 / 3.0);
            localPos = d * r * ShapeRadius;
            shapeDir = d;
            break;
        }
        case SHAPE_CIRCLE:
        {
            float3 d = rand_disc_xz(seed, EmitFromShell);
            localPos = d * ShapeRadius;
            shapeDir = float3(0, 1, 0);
            break;
        }
        case SHAPE_BOX:
        {
            float3 u = float3(rand01(seed), rand01(seed + 1u), rand01(seed + 2u)) * 2.0 - 1.0;
            if (EmitFromShell != 0)
            {
                // Snap one random axis to a face
                uint axis = pcg_hash(seed + 3u) % 3u;
                float sgn = rand01(seed + 4u) < 0.5 ? -1.0 : 1.0;
                if (axis == 0) u.x = sgn; else if (axis == 1) u.y = sgn; else u.z = sgn;
            }
            localPos = u * ShapeExtents;
            shapeDir = float3(0, 1, 0);
            break;
        }
        case SHAPE_CONE:
        {
            // Spawn on a disc, velocity fans outward proportionally to distance from centre
            float3 d = rand_disc_xz(seed, EmitFromShell);
            localPos = d * ShapeRadius;
            shapeDir = normalize(float3(d.x * ConeTan, 1.0, d.z * ConeTan));
            break;
        }
        default: // SHAPE_POINT
            break;
    }
}

// ────────────── CSInit ──────────────
// Fill dead-list with [0..MaxParticles-1], set DeadCount = MaxParticles.
// Dispatch: ceil(MaxParticles / 256)

[numthreads(256, 1, 1)]
void CSInit(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    if (idx >= MaxParticles) return;

    RWStructuredBuffer<uint> DeadList = ResourceDescriptorHeap[DeadListUAVIdx];
    DeadList[idx] = idx;

    // Clear particle data
    RWStructuredBuffer<ParticleCore> Particles = ResourceDescriptorHeap[ParticleCoreUAVIdx];
    ParticleCore p = (ParticleCore)0;
    p.Age = -1.0; // mark as dead
    Particles[idx] = p;

    // First thread writes initial counters
    if (idx == 0)
    {
        RWStructuredBuffer<uint> Counters = ResourceDescriptorHeap[CountersUAVIdx];
        Counters[0] = MaxParticles; // DeadCount
        Counters[1] = 0;           // AliveCountRead
        Counters[2] = 0;           // AliveCountWrite
        Counters[3] = 0;           // EmitCount (CPU sets per frame)
    }
}

// ────────────── CSSimulate ──────────────
// Read from AliveListRead, integrate, write survivors to AliveListWrite.
// Dead particles are returned to the DeadList.
// Dispatch: ceil(AliveCountRead / 256)

[numthreads(256, 1, 1)]
void CSSimulate(uint3 dtid : SV_DispatchThreadID)
{
    RWStructuredBuffer<uint> Counters = ResourceDescriptorHeap[CountersUAVIdx];
    uint aliveCount = Counters[1]; // AliveCountRead

    uint idx = dtid.x;
    if (idx >= aliveCount) return;

    StructuredBuffer<uint> AliveListRead = ResourceDescriptorHeap[AliveListReadIdx];
    RWStructuredBuffer<ParticleCore> Particles = ResourceDescriptorHeap[ParticleCoreUAVIdx];
    RWStructuredBuffer<ParticleVisual> Visuals = ResourceDescriptorHeap[ParticleVisualUAVIdx];

    uint slot = AliveListRead[idx];
    ParticleCore p = Particles[slot];

    // Age the particle
    p.Age += DeltaTime;

    bool dead = p.Age >= p.Lifetime;

    if (!dead)
    {
        // Forces: drag pulls velocity toward the air velocity (Wind), then gravity.
        // With Drag > 0 this converges on a terminal velocity of Wind + Gravity / Drag.
        if (Drag > 0.0)
            p.Velocity += (Wind - p.Velocity) * saturate(Drag * DeltaTime);
        p.Velocity += Gravity * DeltaTime;

        float3 newPos = p.Position + p.Velocity * DeltaTime;

        // ── Collision ──
        bool hit = false;
        float3 hitNormal = float3(0, 1, 0);

        [branch]
        if (CollisionMode == COLL_PLANE)
        {
            if (newPos.y <= PlaneHeight && p.Velocity.y < 0.0)
            {
                hit = true;
                newPos.y = PlaneHeight;
            }
        }
        else if (CollisionMode == COLL_DEPTH && DepthTexIdx != 0)
        {
            float4 clip = mul(float4(newPos, 1.0), ViewProjection);
            if (clip.w > 0.0)
            {
                float2 ndc = clip.xy / clip.w;
                float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
                if (all(uv >= 0.0) && all(uv <= 1.0))
                {
                    Texture2D<float> DepthTex = ResourceDescriptorHeap[DepthTexIdx];
                    uint w, h;
                    DepthTex.GetDimensions(w, h);
                    int2 px = int2(uv * float2(w, h));
                    float sceneDepth = DepthTex.Load(int3(px, 0));

                    // 0 = sky. Particle is behind the surface but within thickness → hit.
                    float behind = clip.w - sceneDepth;
                    if (sceneDepth > 0.001 && behind > 0.0 && behind < CollisionThickness)
                    {
                        hit = true;
                        if (NormalTexIdx != 0)
                        {
                            Texture2D NormalTex = ResourceDescriptorHeap[NormalTexIdx];
                            float3 n = NormalTex.Load(int3(px, 0)).xyz;
                            float nl = length(n);
                            if (nl > 0.1) hitNormal = n / nl;
                        }
                        // Push back out along the normal so we don't re-hit next frame
                        newPos += hitNormal * behind;
                    }
                }
            }
        }

        if (hit)
        {
            if (CollisionResponse == RESP_BOUNCE)
            {
                float vn = dot(p.Velocity, hitNormal);
                if (vn < 0.0)
                    p.Velocity = (p.Velocity - hitNormal * vn * 2.0) * Bounciness;
            }
            else
            {
                dead = true;
            }
        }

        p.Position = newPos;
    }

    if (dead)
    {
        // Dead — return slot to dead list
        RWStructuredBuffer<uint> DeadList = ResourceDescriptorHeap[DeadListUAVIdx];
        uint deadIdx;
        InterlockedAdd(Counters[0], 1, deadIdx); // increment DeadCount
        DeadList[deadIdx] = slot;
    }
    else
    {
        // Update rotation
        ParticleVisual v = Visuals[slot];
        v.Rotation += v.RotationSpeed * DeltaTime;
        Visuals[slot] = v;

        Particles[slot] = p;

        RWStructuredBuffer<uint> AliveListWrite = ResourceDescriptorHeap[AliveListWriteUAVIdx];
        uint writeIdx;
        InterlockedAdd(Counters[2], 1, writeIdx); // increment AliveCountWrite
        AliveListWrite[writeIdx] = slot;
    }
}

// ────────────── CSEmit ──────────────
// Consume from dead-list, initialize new particles, append to alive-list.
// Dispatch: ceil(EmitCount / 64)

[numthreads(64, 1, 1)]
void CSEmit(uint3 dtid : SV_DispatchThreadID)
{
    RWStructuredBuffer<uint> Counters = ResourceDescriptorHeap[CountersUAVIdx];
    uint emitCount = Counters[3];

    uint idx = dtid.x;
    if (idx >= emitCount) return;

    // Consume from dead list (atomically decrement)
    uint oldDeadCount;
    InterlockedAdd(Counters[0], 0xFFFFFFFF, oldDeadCount); // -1

    // Pool exhausted. Several threads can race past zero here, so the value we saw may already be
    // "negative" (wrapped, e.g. 0xFFFFFFFF), not just 0. Reject anything that is not a valid 1..MaxParticles
    // count and give the decrement back — every over-popping thread does this, so the counter always
    // returns to exactly 0. (Checking only == 0 let a wrapped value through, which read DeadList out of
    // bounds and left the counter permanently corrupt: the emitter then went dark until re-init.)
    if ((int)oldDeadCount <= 0 || oldDeadCount > MaxParticles)
    {
        InterlockedAdd(Counters[0], 1, oldDeadCount);
        return;
    }

    RWStructuredBuffer<uint> DeadList = ResourceDescriptorHeap[DeadListUAVIdx];
    uint slot = DeadList[oldDeadCount - 1];

    // Build per-particle random seed
    uint seed = pcg_hash(RandomSeed + idx * 7919u + slot * 6271u);

    // Initialize particle
    RWStructuredBuffer<ParticleCore> Particles = ResourceDescriptorHeap[ParticleCoreUAVIdx];
    // Spawn position from the emission shape (emitter-local)
    float3 localPos, shapeDir;
    sample_shape(seed + 100u, localPos, shapeDir);

    // Initial direction (emitter-local)
    float3 localDir;
    [branch]
    if (DirectionMode == DIR_RANDOM)
    {
        localDir = rand_direction(seed + 200u);
    }
    else if (DirectionMode == DIR_RADIAL)
    {
        // Outward from the origin; fall back to the shape's natural direction at the centre
        float len = length(localPos);
        float3 radial = len > 1e-5 ? localPos / len : shapeDir;
        localDir = rand_cone(radial, SpreadCos, seed + 200u);
    }
    else // DIR_DIRECTIONAL
    {
        // Cone shape steers the base direction outward; other shapes use EmitDirection
        float3 baseDir = (Shape == SHAPE_CONE) ? shapeDir : EmitDirection;
        localDir = rand_cone(baseDir, SpreadCos, seed + 200u);
    }

    float speed = rand_range(SpeedRange.x, SpeedRange.y, seed + 4u);

    ParticleCore p;
    p.Position = EmitterPosition + local_to_world(localPos);
    p.Age = 0.0;
    p.Velocity = local_to_world(localDir) * speed;
    p.Lifetime = max(0.01, LifetimeParam * rand_range(1.0 - LifetimeRandomness, 1.0 + LifetimeRandomness, seed + 3u));
    Particles[slot] = p;

    // Initialize visual
    RWStructuredBuffer<ParticleVisual> Visuals = ResourceDescriptorHeap[ParticleVisualUAVIdx];
    ParticleVisual v;
    // One random scale factor applied to both start and end so the size curve keeps its shape
    float sizeScale = rand_range(1.0 - SizeRandomness, 1.0 + SizeRandomness, seed + 5u);
    v.SizeStartEnd = SizeStartEnd * sizeScale;
    v.ColorStart = ColorStart;
    v.Rotation = (RotationRange > 0) ? rand_range(0, 6.283185, seed + 7u) : 0.0;
    v.RotationSpeed = rand_range(-RotationRange, RotationRange, seed + 8u);
    v.FlipbookFrame = 0;
    v.FlipbookCount = (uint)FlipbookFrameCount;
    v.AnimSpeed = FlipbookAnimSpeed;
    v._pad0 = 0;
    Visuals[slot] = v;

    // Append to alive write list
    RWStructuredBuffer<uint> AliveListWrite = ResourceDescriptorHeap[AliveListWriteUAVIdx];
    uint writeIdx;
    InterlockedAdd(Counters[2], 1, writeIdx);
    AliveListWrite[writeIdx] = slot;
}

// ────────────── CSBuildDrawArgs ──────────────
// Write DrawInstancedArguments from alive count.
// Also swap counters for next frame.
// Dispatch: (1, 1, 1)

[numthreads(1, 1, 1)]
void CSBuildDrawArgs(uint3 dtid : SV_DispatchThreadID)
{
    RWStructuredBuffer<uint> Counters = ResourceDescriptorHeap[CountersUAVIdx];
    RWStructuredBuffer<uint> DrawArgs = ResourceDescriptorHeap[DrawArgsUAVIdx];

    uint aliveWrite = Counters[2];

    // Write indirect draw args: DrawInstanced(6 verts per quad, aliveCount instances, 0, 0)
    DrawArgs[0] = 6;           // VertexCountPerInstance
    DrawArgs[1] = aliveWrite;  // InstanceCount
    DrawArgs[2] = 0;           // StartVertexLocation
    DrawArgs[3] = 0;           // StartInstanceLocation

    // Prepare counters for next frame:
    // AliveCountRead = current AliveCountWrite (simulation will read this)
    // AliveCountWrite = 0 (will be incremented by next frame's simulate + emit)
    Counters[1] = aliveWrite;
    Counters[2] = 0;
    Counters[3] = 0; // Clear emit count (CPU sets it next frame)
}
