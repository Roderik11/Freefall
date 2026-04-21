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
    float3 EmitVelocity;
    float  VelocityRandomness;
    float3 Gravity;
    float  LifetimeParam;
    float2 SizeRange;
    float  RotationRange;
    uint   RandomSeed;
    float4 ColorStart;
    float4 ColorEnd;
    float  FlipbookFrameCount;
    float  FlipbookAnimSpeed;
    float2 _emitPad;
};

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

    if (p.Age >= p.Lifetime)
    {
        // Dead — return slot to dead list
        RWStructuredBuffer<uint> DeadList = ResourceDescriptorHeap[DeadListUAVIdx];
        uint deadIdx;
        InterlockedAdd(Counters[0], 1, deadIdx); // increment DeadCount
        DeadList[deadIdx] = slot;
    }
    else
    {
        // Alive — integrate and write to next alive list
        p.Velocity += Gravity * DeltaTime;
        p.Position += p.Velocity * DeltaTime;

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

    if (oldDeadCount == 0)
    {
        // No free slots — undo the decrement
        InterlockedAdd(Counters[0], 1, oldDeadCount);
        return;
    }

    RWStructuredBuffer<uint> DeadList = ResourceDescriptorHeap[DeadListUAVIdx];
    uint slot = DeadList[oldDeadCount - 1];

    // Build per-particle random seed
    uint seed = pcg_hash(RandomSeed + idx * 7919u + slot * 6271u);

    // Initialize particle
    RWStructuredBuffer<ParticleCore> Particles = ResourceDescriptorHeap[ParticleCoreUAVIdx];
    ParticleCore p;
    p.Position = EmitterPosition;
    p.Age = 0.0;
    p.Velocity = EmitVelocity + rand_direction(seed) * VelocityRandomness * length(EmitVelocity + 0.001);
    p.Lifetime = LifetimeParam * rand_range(0.8, 1.2, seed + 3u);
    Particles[slot] = p;

    // Initialize visual
    RWStructuredBuffer<ParticleVisual> Visuals = ResourceDescriptorHeap[ParticleVisualUAVIdx];
    ParticleVisual v;
    v.SizeStartEnd = float2(
        rand_range(SizeRange.x, SizeRange.y, seed + 5u),
        rand_range(SizeRange.x * 0.5, SizeRange.y * 1.5, seed + 6u)
    );
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
