// GPU Particle System — Compute Pipeline
// One pool for every emitter (see particle_common.hlsli). Two dispatches per rendered view:
//   CSCull      one thread per emitter: frustum + Hi-Z test of its bounds -> Visibility[emitter]
//   CSSimulate  one group per pool chunk: respawn / integrate / collide every particle, then append
//               the live ones of visible emitters to the draw list of their render mode
// There is no dead list: each emitter's slots form a ring and the CPU says which slots respawn.

#pragma kernel CSCull
#pragma kernel CSSimulate

#include "particle_common.hlsli"
#include "ocean_surf.hlsli"

// The renderer's frustum constants for this view (root slot 1). Layout = cull_instances.hlsl.
cbuffer FrustumPlanes : register(b0)
{
    float4 Plane0;
    float4 Plane1;
    float4 Plane2;
    float4 Plane3;
    float4 Plane4;
    float4 Plane5;
    row_major float4x4 OcclusionProjection; // previous frame's VP: matches the Hi-Z pyramid
    uint HiZSrvIdx;        // 0 = disabled
    float2 HiZSize;
    uint HiZMipCount;
    float NearPlane;
    uint CullStatsUAVIdx;
    uint DebugMode;
    float ProjScale;
    float3 CameraPosition;
    uint SortDirection;
};

// Written once per frame by the view that simulates (root slot 2)
cbuffer ParticleFrame : register(b1)
{
    row_major float4x4 CollisionViewProjection; // VP the depth GBuffer was rendered with
};

cbuffer PushConstants : register(b3)
{
    uint EmittersIdx;            //  0  StructuredBuffer<ParticleEmitter>
    uint ChunkMapIdx;            //  1  StructuredBuffer<uint>: chunk -> emitter row, PARTICLE_NO_EMITTER when free
    uint ParticleCoreUAVIdx;     //  2  RWStructuredBuffer<ParticleCore>
    uint ParticleVisualUAVIdx;   //  3  RWStructuredBuffer<ParticleVisual>
    uint VisibilityUAVIdx;       //  4  RWStructuredBuffer<uint>, one per emitter row
    uint DrawListUAVIdx;         //  5  RWStructuredBuffer<uint>, PoolSlots entries per render mode
    uint DrawArgsUAVIdx;         //  6  RWStructuredBuffer<uint>, 4 per render mode (DrawInstanced args)
    uint EmitterCount;           //  7
    uint PoolSlots;              //  8  slots in the pool = size of one draw list
    uint SimulateFlag;           //  9  0 = another view already advanced this frame: only build draw lists
    float DeltaTime;             // 10
    uint DepthTexIdx;            // 11  linear view-space depth GBuffer (0 = none)
    uint NormalTexIdx;           // 12  world-space normal GBuffer (0 = none)
    float SimTime;               // 13  seconds, moves the turbulence field
    uint OceanDataIdx;           // 14  StructuredBuffer<OceanData>, the ocean drawn this frame (shape Surf), 0 = none
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
#define SHAPE_SURF       6

#define DIR_DIRECTIONAL  0
#define DIR_RADIAL       1
#define DIR_RANDOM       2

// ────────────── Culling helpers (same tests as cull_instances.hlsl) ──────────────

bool IsVisible(float3 center, float radius)
{
    float4 planes[6] = { Plane0, Plane1, Plane2, Plane3, Plane4, Plane5 };

    for (uint i = 0; i < 6; i++)
    {
        if (dot(planes[i].xyz, center) + planes[i].w > radius)
            return false;
    }
    return true;
}

// True if the sphere is fully behind the previous frame's depth pyramid
bool IsOccluded(float3 worldCenter, float worldRadius)
{
    if (HiZSrvIdx == 0) return false;

    float4 clipCenter = mul(float4(worldCenter, 1.0), OcclusionProjection);
    // Camera inside or behind the sphere: cannot be tested
    if (clipCenter.w - worldRadius <= NearPlane) return false;

    float3 ndc = clipCenter.xyz / clipCenter.w;
    float2 uv = ndc.xy * float2(0.5, -0.5) + 0.5;
    if (any(uv < 0.0) || any(uv > 1.0)) return false;

    Texture2D<float> hiZ = ResourceDescriptorHeap[HiZSrvIdx];

    float w, h, levels;
    hiZ.GetDimensions(0, w, h, levels);

    float projScale = abs(OcclusionProjection._m11);
    projScale = projScale < 0.001 ? 1.0 : projScale;
    float screenRadius = (worldRadius * projScale / clipCenter.w) * h * 0.5;

    float mipLevel = min(ceil(log2(max(screenRadius * 2, 1.0))), levels - 1.0);
    uint mip = (uint)mipLevel;

    float mipW, mipH, unused;
    hiZ.GetDimensions(mip, mipW, mipH, unused);
    float2 mipSize = float2(mipW, mipH);

    int2 baseCoord = int2(uv * mipSize - 0.5);
    int2 maxCoord = int2(mipSize) - 1;

    float d0 = hiZ.Load(int3(clamp(baseCoord,             int2(0, 0), maxCoord), mip));
    float d1 = hiZ.Load(int3(clamp(baseCoord + int2(1, 0), int2(0, 0), maxCoord), mip));
    float d2 = hiZ.Load(int3(clamp(baseCoord + int2(0, 1), int2(0, 0), maxCoord), mip));
    float d3 = hiZ.Load(int3(clamp(baseCoord + int2(1, 1), int2(0, 0), maxCoord), mip));

    float sampledDepth = max(max(d0, d1), max(d2, d3));
    return clipCenter.w - worldRadius > sampledDepth;
}

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

// ────────────── Turbulence ──────────────

float lattice(int3 p, uint seed)
{
    return rand01(((uint)p.x * 73856093u) ^ ((uint)p.y * 19349663u) ^ ((uint)p.z * 83492791u) ^ seed);
}

// Gradient of 3D value noise (quintic interpolation, analytic derivatives)
float3 noise_gradient(float3 x, uint seed)
{
    float3 cell = floor(x);
    float3 w = x - cell;
    int3 i = (int3)cell;

    float3 u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
    float3 du = 30.0 * w * w * (w * (w - 2.0) + 1.0);

    float a = lattice(i, seed);
    float b = lattice(i + int3(1, 0, 0), seed);
    float c = lattice(i + int3(0, 1, 0), seed);
    float d = lattice(i + int3(1, 1, 0), seed);
    float e = lattice(i + int3(0, 0, 1), seed);
    float f = lattice(i + int3(1, 0, 1), seed);
    float g = lattice(i + int3(0, 1, 1), seed);
    float h = lattice(i + int3(1, 1, 1), seed);

    float k1 = b - a;
    float k2 = c - a;
    float k3 = e - a;
    float k4 = a - b - c + d;
    float k5 = a - c - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;

    return du * float3(k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
                       k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
                       k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y);
}

// Curl of three noise fields: a flow without sources or sinks, so particles swirl instead of bunching up
float3 curl_noise(float3 x)
{
    float3 g1 = noise_gradient(x, 0x1F123BB5u);
    float3 g2 = noise_gradient(x, 0x5F356495u);
    float3 g3 = noise_gradient(x, 0x9E3779B9u);
    return float3(g3.y - g2.z, g1.z - g3.x, g2.x - g1.y);
}

// ────────────── Emission ──────────────

// Emitter-local → world rotation
float3 local_to_world(ParticleEmitter e, float3 v)
{
    return e.AxisX * v.x + e.AxisY * v.y + e.AxisZ * v.z;
}

// Sample a spawn offset (emitter-local) and shape-derived direction hint.
// 'shapeDir' is the natural outward direction for the shape (used by Cone / Radial).
void sample_shape(ParticleEmitter e, uint seed, out float3 localPos, out float3 shapeDir)
{
    localPos = 0;
    shapeDir = float3(0, 1, 0);

    [branch]
    switch (e.Shape)
    {
        case SHAPE_SPHERE:
        {
            float3 d = rand_direction(seed);
            float r = e.EmitFromShell != 0 ? 1.0 : pow(rand01(seed + 2u), 1.0 / 3.0);
            localPos = d * r * e.ShapeRadius;
            shapeDir = d;
            break;
        }
        case SHAPE_HEMISPHERE:
        {
            float3 d = rand_direction(seed);
            d.y = abs(d.y);
            float r = e.EmitFromShell != 0 ? 1.0 : pow(rand01(seed + 2u), 1.0 / 3.0);
            localPos = d * r * e.ShapeRadius;
            shapeDir = d;
            break;
        }
        case SHAPE_CIRCLE:
        {
            float3 d = rand_disc_xz(seed, e.EmitFromShell);
            localPos = d * e.ShapeRadius;
            break;
        }
        case SHAPE_BOX:
        {
            float3 u = float3(rand01(seed), rand01(seed + 1u), rand01(seed + 2u)) * 2.0 - 1.0;
            if (e.EmitFromShell != 0)
            {
                // Snap one random axis to a face
                uint axis = pcg_hash(seed + 3u) % 3u;
                float sgn = rand01(seed + 4u) < 0.5 ? -1.0 : 1.0;
                if (axis == 0) u.x = sgn; else if (axis == 1) u.y = sgn; else u.z = sgn;
            }
            localPos = u * e.ShapeExtents;
            break;
        }
        case SHAPE_CONE:
        {
            // Spawn on a disc, velocity fans outward proportionally to distance from centre
            float3 d = rand_disc_xz(seed, e.EmitFromShell);
            localPos = d * e.ShapeRadius;
            shapeDir = normalize(float3(d.x * e.ConeTan, 1.0, d.z * e.ConeTan));
            break;
        }
        default: // SHAPE_POINT
            break;
    }
}

// Shape Surf: spray and mist where the ocean's surf breaks (ocean_surf.hlsli, the same waves the ocean draws). A
// candidate point is picked around the camera (ShapeExtents.xz half sizes) and the sea there decides whether anything
// is born: lip spray where a broken wave's crest is (its roller), thrown ahead along the wave's heading, a little up;
// mist (PARTICLE_FLAG_SURF_MIST) over the broken water and behind the crests, drifting slowly. Nothing is born where
// no wave breaks: no clock, no emitters along the shore. False = no candidate was used (the slot stays dead).
// A slot tries SURF_TRIES points: the surf is a thin band of the square (a breaker's roller a few metres wide), and with
// one try a slot was used so rarely that a 40 m breaker threw some 25 drops a second. Land and open sea fail the first,
// cheap test; only points in the coastal water sample the waves.
#define SURF_TRIES 16
bool spawn_surf(ParticleEmitter e, uint seed, inout ParticleCore p)
{
    if (OceanDataIdx == 0) return false;
    StructuredBuffer<OceanData> oceans = ResourceDescriptorHeap[OceanDataIdx];
    OceanData ocean = oceans[0];
    if (ocean.ShoreWarpSRV == 0) return false;
    bool mist = (e.Flags & PARTICLE_FLAG_SURF_MIST) != 0;

    float2 xz = 0;
    OceanShore shore = (OceanShore)0;
    SurfWave wave = (SurfWave)0;
    float3 d = 0;
    float height = 0;
    float breaking = 0;
    bool found = false;
    [loop]
    for (uint attempt = 0; attempt < SURF_TRIES && !found; attempt++)
    {
        uint s = seed + attempt * 7u;
        xz = CameraPosition.xz + (float2(rand01(s), rand01(s + 1u)) * 2.0 - 1.0) * e.ShapeExtents.xz;
        shore = ShoreAt(ocean, xz);
        if (shore.shoreDepth > 1e3 || shore.waterDepth < SURF_FOAM_END.x) continue;

        d = SampleDisplacement(ocean, xz, ocean.DisplacementAtten, distance(CameraPosition.xz, xz), shore.waveAtten,
                               shore.shoreDepth, shore.surfUV, true, wave);
        breaking = SurfBreaking(wave, shore.shoreDepth) * smoothstep(SURF_FOAM_END.x, SURF_FOAM_END.y, shore.shoreDepth);
        float2 weight = wave.envelope / max(max(wave.envelope.x, wave.envelope.y), 1e-6);
        float2 rt0 = SurfRollerTrail(wave.phase.x) * weight.x;
        float2 rt1 = SurfRollerTrail(wave.phase.y) * weight.y;
        float roller = max(rt0.x, rt1.x), trail = max(rt0.y, rt1.y);
        // Spray tears off the lip, the very top of the crest (phase 0, about a tenth of a radian either way), not off
        // the whole white roller down the face: born across the roller it was a band of puffs over the broken water
        float lip = max(exp(-wave.phase.x * wave.phase.x / 0.02) * weight.x, exp(-wave.phase.y * wave.phase.y / 0.02) * weight.y);
        // The bigger the breaker, the more it throws: a knee-high wave hardly any
        height = 2.0 * SurfEnvelope(wave);
        float chance = breaking * (mist ? max(trail, 0.5 * roller) : lip) * saturate(height / 1.5);
        found = rand01(s + 2u) < chance;
    }
    if (!found) return false;

    float slopeScale;
    float crest = DepthLimit(d.y, shore.shoreDepth, slopeScale);
    float2 heading = SurfHeading(wave);
    float celerity = sqrt(9.81 * max(shore.shoreDepth, 0.3));       // the breaker's speed in shallow water
    float3 velocity;
    if (mist)
    {
        // Mist rises off the broken water and drifts on with it, slowly
        float forward = celerity * 0.3;
        velocity = float3(heading.x * forward, rand_range(e.SpeedRange.x, e.SpeedRange.y, seed + 4u), heading.y * forward);
    }
    else
    {
        // The lip is thrown ahead of the wave, faster than the crest runs, and only a little up (about 10 to 30 degrees
        // over the heading); SpeedRange adds to it, more for a bigger breaker. Until 2026-10-19 it went mostly up
        // (SpeedRange straight up, a fraction of the crest's speed forward): "should launch in wave direction, not up"
        float speed = celerity * rand_range(1.0, 1.4, seed + 3u)
                    + rand_range(e.SpeedRange.x, e.SpeedRange.y, seed + 4u) * saturate(0.3 + height / 3.0);
        float lift = rand_range(0.18, 0.55, seed + 6u);
        velocity = normalize(float3(heading.x, lift, heading.y)) * speed;
    }
    float speed = length(velocity);

    // Where the ocean draws this point (its DS): pushed sideways no further than the water is deep, and on a breaking
    // wave leant forward by SURF_LEAN x the crest's height. Without the lean the spray rose up to 1.6 crest heights
    // behind the drawn lip
    float2 side = d.xz * min(1.0, max(shore.shoreDepth, 0.05) / max(length(d.xz), 1e-4));
    side += heading * (SURF_LEAN * breaking * max(crest, 0.0));
    // xz is the world's (x, z): float3(xz + side, y) made it (x, z, y), every drop born ~500 m up with its height as z
    p.Position = float3(xz.x + side.x, ocean.OceanPlaneY + crest + (mist ? 0.3 : 0.1), xz.y + side.y);
    p.Velocity = rand_cone(velocity / max(speed, 1e-4), e.SpreadCos, seed + 5u) * speed;
    return true;
}

void spawn(ParticleEmitter e, uint seed, out ParticleCore p, out ParticleVisual v)
{
    [branch]
    if (e.Shape == SHAPE_SURF)
    {
        p.Age = 0.0;
        p.Position = e.Position;
        p.Velocity = 0.0;
        p.Lifetime = max(0.01, e.Lifetime * rand_range(1.0 - e.LifetimeRandomness, 1.0 + e.LifetimeRandomness, seed + 3u));
        if (!spawn_surf(e, seed + 300u, p))
            p.Lifetime = 0.0;                                       // nothing breaking here: the slot stays dead
        v.SizeScale = rand_range(1.0 - e.SizeRandomness, 1.0 + e.SizeRandomness, seed + 5u);
        v.Rotation = (e.RotationRange > 0) ? rand_range(0, 6.283185, seed + 7u) : 0.0;
        v.RotationSpeed = rand_range(-e.RotationRange, e.RotationRange, seed + 8u);
        v.Random = rand01(seed + 9u);
        return;
    }

    float3 localPos, shapeDir;
    sample_shape(e, seed + 100u, localPos, shapeDir);

    float3 localDir;
    [branch]
    if (e.DirectionMode == DIR_RANDOM)
    {
        localDir = rand_direction(seed + 200u);
    }
    else if (e.DirectionMode == DIR_RADIAL)
    {
        // Outward from the origin; fall back to the shape's natural direction at the centre
        float len = length(localPos);
        float3 radial = len > 1e-5 ? localPos / len : shapeDir;
        localDir = rand_cone(radial, e.SpreadCos, seed + 200u);
    }
    else // DIR_DIRECTIONAL
    {
        // Cone shape steers the base direction outward; other shapes use EmitDirection
        float3 baseDir = (e.Shape == SHAPE_CONE) ? shapeDir : e.EmitDirection;
        localDir = rand_cone(baseDir, e.SpreadCos, seed + 200u);
    }

    float speed = rand_range(e.SpeedRange.x, e.SpeedRange.y, seed + 4u);

    p.Position = e.Position + local_to_world(e, localPos);
    p.Age = 0.0;
    p.Velocity = local_to_world(e, localDir) * speed;
    p.Lifetime = max(0.01, e.Lifetime * rand_range(1.0 - e.LifetimeRandomness, 1.0 + e.LifetimeRandomness, seed + 3u));

    // One random scale factor for start and end size so the size curve keeps its shape
    v.SizeScale = rand_range(1.0 - e.SizeRandomness, 1.0 + e.SizeRandomness, seed + 5u);
    v.Rotation = (e.RotationRange > 0) ? rand_range(0, 6.283185, seed + 7u) : 0.0;
    v.RotationSpeed = rand_range(-e.RotationRange, e.RotationRange, seed + 8u);
    v.Random = rand01(seed + 9u);
}

// ────────────── Simulation ──────────────

// Integrate one live particle. Returns false when it died this step.
bool integrate(ParticleEmitter e, inout ParticleCore p)
{
    p.Age += DeltaTime;
    if (p.Age >= p.Lifetime) return false;

    // Forces: drag pulls velocity toward the air velocity (Wind), then gravity.
    // With Drag > 0 this converges on a terminal velocity of Wind + Gravity / Drag.
    if (e.Drag > 0.0)
        p.Velocity += (e.Wind - p.Velocity) * saturate(e.Drag * DeltaTime);
    p.Velocity += e.Gravity * DeltaTime;

    // Turbulence: the field drifts upward over time, so a particle standing still is stirred too
    [branch]
    if (e.Turbulence > 0.0)
    {
        float3 at = p.Position * e.TurbulenceFrequency - float3(0.0, SimTime * 0.7, 0.0);
        p.Velocity += curl_noise(at) * (e.Turbulence * DeltaTime);
    }

    // Carried along by a moving emitter (FollowEmitter): the flame of a torch goes where the torch goes
    float3 newPos = p.Position + e.Carry + p.Velocity * DeltaTime;

    bool hit = false;
    float3 hitNormal = float3(0, 1, 0);

    [branch]
    if (e.CollisionMode == COLL_PLANE)
    {
        if (newPos.y <= e.PlaneHeight && p.Velocity.y < 0.0)
        {
            hit = true;
            newPos.y = e.PlaneHeight;
        }
    }
    else if (e.CollisionMode == COLL_DEPTH && DepthTexIdx != 0)
    {
        float4 clip = mul(float4(newPos, 1.0), CollisionViewProjection);
        if (clip.w > 0.0)
        {
            float2 ndc = clip.xy / clip.w;
            float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            if (all(uv >= 0.0) && all(uv <= 1.0))
            {
                Texture2D<float> DepthTex = ResourceDescriptorHeap[DepthTexIdx];
                uint w, h;
                DepthTex.GetDimensions(w, h);
                int2 px = min(int2(uv * float2(w, h)), int2(w, h) - 1);
                float sceneDepth = DepthTex.Load(int3(px, 0));

                // 0 = sky. Particle is behind the surface but within thickness → hit.
                float behind = clip.w - sceneDepth;
                if (sceneDepth > 0.001 && behind > 0.0 && behind < e.CollisionThickness)
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
        if (e.CollisionResponse != RESP_BOUNCE)
            return false;

        float vn = dot(p.Velocity, hitNormal);
        if (vn < 0.0)
            p.Velocity = (p.Velocity - hitNormal * vn * 2.0) * e.Bounciness;
    }

    p.Position = newPos;
    return true;
}

// ────────────── CSCull ──────────────
// Dispatch: ceil(EmitterCount / 64)

[numthreads(64, 1, 1)]
void CSCull(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;

    // First thread starts this view's draw lists: DrawInstanced(6 verts per quad, 0 instances)
    if (idx == 0)
    {
        RWStructuredBuffer<uint> DrawArgs = ResourceDescriptorHeap[DrawArgsUAVIdx];
        for (uint mode = 0; mode < PARTICLE_MODE_COUNT; mode++)
        {
            DrawArgs[mode * 4 + 0] = 6;
            DrawArgs[mode * 4 + 1] = 0;
            DrawArgs[mode * 4 + 2] = 0;
            DrawArgs[mode * 4 + 3] = 0;
        }
    }

    if (idx >= EmitterCount) return;

    StructuredBuffer<ParticleEmitter> Emitters = ResourceDescriptorHeap[EmittersIdx];
    ParticleEmitter e = Emitters[idx];

    bool visible = IsVisible(e.BoundsCenter, e.BoundsRadius)
        && !IsOccluded(e.BoundsCenter, e.BoundsRadius);

    RWStructuredBuffer<uint> Visibility = ResourceDescriptorHeap[VisibilityUAVIdx];
    Visibility[idx] = visible ? 1u : 0u;
}

// ────────────── CSSimulate ──────────────
// Dispatch: one group per pool chunk, up to the highest chunk in use.

[numthreads(PARTICLE_CHUNK_SIZE, 1, 1)]
void CSSimulate(uint3 gid : SV_GroupID, uint3 dtid : SV_DispatchThreadID)
{
    StructuredBuffer<uint> ChunkMap = ResourceDescriptorHeap[ChunkMapIdx];
    uint emitterIdx = ChunkMap[gid.x];
    if (emitterIdx >= EmitterCount) return; // free chunk

    StructuredBuffer<ParticleEmitter> Emitters = ResourceDescriptorHeap[EmittersIdx];
    ParticleEmitter e = Emitters[emitterIdx];

    uint slot = dtid.x;
    uint local = slot - e.FirstSlot;
    if (local >= e.Capacity) return;

    RWStructuredBuffer<ParticleCore> Particles = ResourceDescriptorHeap[ParticleCoreUAVIdx];
    ParticleCore p = Particles[slot];
    bool alive = p.Age < p.Lifetime;

    if (SimulateFlag != 0)
    {
        // The ring window [EmitStart, EmitStart + EmitCount) respawns this frame
        uint ringPos = local >= e.EmitStart ? local - e.EmitStart : local + e.Capacity - e.EmitStart;

        if (ringPos < e.EmitCount)
        {
            ParticleVisual v;
            spawn(e, pcg_hash(e.RandomSeed + local * 7919u + slot * 6271u), p, v);

            RWStructuredBuffer<ParticleVisual> Visuals = ResourceDescriptorHeap[ParticleVisualUAVIdx];
            Visuals[slot] = v;
            Particles[slot] = p;
            alive = p.Age < p.Lifetime;     // a Surf candidate where nothing breaks is born dead
        }
        else if ((e.Flags & PARTICLE_FLAG_RESET) != 0)
        {
            if (p.Lifetime != 0.0)
            {
                p.Lifetime = 0.0;
                Particles[slot] = p;
            }
            alive = false;
        }
        else if (alive)
        {
            alive = integrate(e, p);
            if (!alive) p.Age = p.Lifetime;
            Particles[slot] = p;
        }
    }

    if (!alive) return;

    RWStructuredBuffer<uint> Visibility = ResourceDescriptorHeap[VisibilityUAVIdx];
    if (Visibility[emitterIdx] == 0 || !IsVisible(p.Position, e.ParticleRadius)) return;

    RWStructuredBuffer<uint> DrawArgs = ResourceDescriptorHeap[DrawArgsUAVIdx];
    uint index;
    InterlockedAdd(DrawArgs[e.RenderMode * 4 + 1], 1, index);

    RWStructuredBuffer<uint> DrawList = ResourceDescriptorHeap[DrawListUAVIdx];
    DrawList[e.RenderMode * PoolSlots + index] = slot;
}
