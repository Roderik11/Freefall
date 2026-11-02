// Cloth on the GPU (Components/Cloth.cs): a skinned piece of clothing whose lower part hangs free.
//
// One thread group per cloth, all of a frame in one dispatch. A particle is a vertex of the mesh's LOD 0 (vertices at
// the same place, a seam in the UVs, are one particle). Each frame:
//   1. every particle's place as the bones would put it (skinned like gbuffer_skinned.fx, then into the world);
//   2. per substep: the place a step on (Verlet: its motion damped, gravity, the air's drag towards the wind), then
//      passes over its links (edges keep their length, links across two triangles keep it from folding), each pass
//      every particle moved by the mean of what its links ask (Jacobi, from one buffer into the other), kept within
//      its MaxAway of its bones' place and out of the body's parts (round cones, Cloth's ClothFrame);
//   3. every vertex's place and normal (from the triangles round it) in the renderer's object space, for the skinned
//      vertex shader to draw instead of skinning it.
// A particle with MaxAway 0 is where its bones put it. Positions are in the world: gravity and wind are.

#pragma kernel CSCloth

#define THREADS 1024

cbuffer PushConstants : register(b3)
{
    uint ParticlesIdx;      //  0  StructuredBuffer<ClothParticle>
    uint LinksIdx;          //  1  StructuredBuffer<ClothLink>
    uint VerticesIdx;       //  2  StructuredBuffer<uint4>: particle, first of its triangles, how many, winding sign
    uint VertexTrisIdx;     //  3  StructuredBuffer<uint4>: a triangle round a vertex as particles (xyz)
    uint MaxAwayIdx;        //  4  StructuredBuffer<float4>, per particle: how far it may leave its bones, its tether (the held particle nearest it, -1 none) and the tether's length
    uint StateUAVIdx;       //  5  RWStructuredBuffer<float4>: 5 ranges of ParticleCount (STATE_*)
    uint OutputUAVIdx;      //  6  RWStructuredBuffer<float4>: per vertex its place and its normal
    uint FrameIdx;          //  7  StructuredBuffer<ClothFrame>, one
    uint BonesIdx;          //  8  StructuredBuffer<row_major float4x4>: the Animator's bone buffer, 0 = bind pose
    uint ParticleCount;     //  9
    uint VertexCount;       // 10
};

struct ClothParticle
{
    float3 Rest;            // in the mesh's space, as its vertex buffer has it
    uint First;             // its first link
    float4 BoneIds;
    float4 Weights;
    uint Count;             // how many links
    uint3 _pad;
};

struct ClothLink
{
    uint Other;
    float Rest;             // its length at rest
    uint Bend;              // 1 = across two triangles, 0 = an edge
    uint _pad;
};

struct ClothFrame
{
    row_major float4x4 World;       // object -> world, this frame
    row_major float4x4 ToObject;    // world -> object
    float3 Gravity;
    float Dt;                       // a substep's seconds
    float3 Wind;
    float Drag;
    float Damping;
    float Stiffness;
    float Bending;
    float Thickness;
    uint Substeps;
    uint Iterations;
    uint PartCount;
    uint Reset;                     // 1 = start where the bones put everything
    float Backstop;                 // how far it may sink towards the body from where the bones put it
    float3 _pad;
    float4 PartFrom[16];            // a body part's axis: its end and its radius there, in the world
    float4 PartTo[16];
};

// The state's ranges
#define STATE_A 0       // place
#define STATE_B 1       // the other place buffer (a pass reads one and writes the other)
#define STATE_PREV 2    // place a step ago
#define STATE_TARGET 3  // where the bones put it last frame
#define STATE_NEXT 4    // ... and this frame
#define STATE_BACK 5    // x: the body part nearest where the bones put it this substep, y: how near its axis it may come

// The nearest place on a body part's axis, and how far along it (0..1)
float3 OnAxis(ClothFrame frame, uint part, float3 x, out float t)
{
    float3 a = frame.PartFrom[part].xyz, b = frame.PartTo[part].xyz;
    float3 ab = b - a;
    t = saturate(dot(x - a, ab) / max(dot(ab, ab), 1e-8));
    return a + ab * t;
}

float3 SkinnedWorld(ClothParticle particle, ClothFrame frame)
{
    float3 pos = particle.Rest;
    if (BonesIdx != 0u)
    {
        StructuredBuffer<row_major float4x4> bones = ResourceDescriptorHeap[BonesIdx];
        float4 skinned = mul(float4(pos, 1), bones[(uint)particle.BoneIds.x]) * particle.Weights.x
                       + mul(float4(pos, 1), bones[(uint)particle.BoneIds.y]) * particle.Weights.y
                       + mul(float4(pos, 1), bones[(uint)particle.BoneIds.z]) * particle.Weights.z
                       + mul(float4(pos, 1), bones[(uint)particle.BoneIds.w]) * particle.Weights.w;
        pos = skinned.xyz;
    }
    return mul(float4(pos, 1), frame.World).xyz;
}

// Out of the body's parts: the nearest place on a part's axis, the part's radius there and the cloth's thickness
float3 KeepOut(float3 x, ClothFrame frame)
{
    for (uint i = 0; i < frame.PartCount; i++)
    {
        float3 a = frame.PartFrom[i].xyz, b = frame.PartTo[i].xyz;
        float3 ab = b - a;
        float t = saturate(dot(x - a, ab) / max(dot(ab, ab), 1e-8));
        float3 closest = a + ab * t;
        float radius = lerp(frame.PartFrom[i].w, frame.PartTo[i].w, t) + frame.Thickness;
        float3 away = x - closest;
        float distanceSq = dot(away, away);
        if (distanceSq < radius * radius && distanceSq > 1e-12)
            x = closest + away * (radius * rsqrt(distanceSq));
    }
    return x;
}

[numthreads(THREADS, 1, 1)]
void CSCloth(uint3 threadId : SV_GroupThreadID)
{
    StructuredBuffer<ClothFrame> frames = ResourceDescriptorHeap[FrameIdx];
    StructuredBuffer<ClothParticle> particles = ResourceDescriptorHeap[ParticlesIdx];
    StructuredBuffer<ClothLink> links = ResourceDescriptorHeap[LinksIdx];
    StructuredBuffer<float4> limits = ResourceDescriptorHeap[MaxAwayIdx];
    RWStructuredBuffer<float4> state = ResourceDescriptorHeap[StateUAVIdx];
    ClothFrame frame = frames[0];

    uint tid = threadId.x;
    uint count = ParticleCount;

    // 1. Where the bones put each particle this frame
    for (uint i1 = tid; i1 < count; i1 += THREADS)
    {
        uint p = i1;
        float3 next = SkinnedWorld(particles[p], frame);
        state[STATE_NEXT * count + p] = float4(next, 0);
        if (frame.Reset != 0u)
        {
            state[STATE_A * count + p] = float4(next, 0);
            state[STATE_PREV * count + p] = float4(next, 0);
            state[STATE_TARGET * count + p] = float4(next, 0);
        }
    }
    DeviceMemoryBarrierWithGroupSync();

    float dt = frame.Dt;
    for (uint s = 0; s < frame.Substeps; s++)
    {
        float share = (s + 1.0) / frame.Substeps;

        // 2a. A step on
        for (uint i2 = tid; i2 < count; i2 += THREADS)
        {
            uint p = i2;
            float3 target = lerp(state[STATE_TARGET * count + p].xyz, state[STATE_NEXT * count + p].xyz, share);
            float3 x;
            float3 previous;
            if (limits[p].x <= 0.0)
            {
                x = target;
                previous = target;
            }
            else
            {
                x = state[STATE_A * count + p].xyz;
                float3 motion = (x - state[STATE_PREV * count + p].xyz) * (1.0 - frame.Damping);
                float3 velocity = motion / max(dt, 1e-5);
                float3 pull = frame.Gravity + frame.Drag * (frame.Wind - velocity);
                previous = x;
                x += motion + pull * dt * dt;
            }
            state[STATE_A * count + p] = float4(x, 0);
            state[STATE_PREV * count + p] = float4(previous, 0);

            // Its backstop: the body part whose axis is nearest where the bones put it, and that distance less how far
            // it may sink. Where the clothes under it stand out from the body's parts (mail), it stays over them.
            float nearest = 1e20;
            uint nearestPart = 0xFFFFFFFFu;
            for (uint i = 0; i < frame.PartCount; i++)
            {
                float t;
                float3 c = OnAxis(frame, i, target, t);
                float d = dot(target - c, target - c);
                if (d < nearest) { nearest = d; nearestPart = i; }
            }
            float reach = nearestPart != 0xFFFFFFFFu ? sqrt(nearest) - frame.Backstop : 0.0;
            state[STATE_BACK * count + p] = float4(nearestPart != 0xFFFFFFFFu ? (float)nearestPart : -1.0, reach, 0, 0);
        }
        DeviceMemoryBarrierWithGroupSync();

        // 2b. Passes over the links, from A into B and back (an even number: it ends in A)
        uint passes = (frame.Iterations + 1u) & ~1u;
        for (uint it = 0; it < passes; it++)
        {
            uint from = (it & 1u) == 0u ? STATE_A : STATE_B;
            uint to = (it & 1u) == 0u ? STATE_B : STATE_A;
            for (uint i3 = tid; i3 < count; i3 += THREADS)
            {
                uint p = i3;
                float3 target = lerp(state[STATE_TARGET * count + p].xyz, state[STATE_NEXT * count + p].xyz, share);
                float4 limit = limits[p];
                float most = limit.x;
                float3 x = state[from * count + p].xyz;
                if (most <= 0.0)
                {
                    state[to * count + p] = float4(target, 0);
                    continue;
                }

                ClothParticle particle = particles[p];
                float3 move = 0;
                float weight = 0;
                for (uint l = 0; l < particle.Count; l++)
                {
                    ClothLink link = links[particle.First + l];
                    float3 other = state[from * count + link.Other].xyz;
                    float3 d = x - other;
                    float length = sqrt(max(dot(d, d), 1e-12));
                    // A pinned neighbour does not give: this particle takes the whole correction
                    float give = limits[link.Other].x <= 0.0 ? 1.0 : 0.5;
                    float k = link.Bend != 0u ? frame.Bending : frame.Stiffness;
                    move -= d * ((length - link.Rest) / length) * give * k;
                    weight += 1.0;
                }
                if (weight > 0.0)
                    x += move * (1.5 / weight);

                // No further from the held particle nearest it than at rest (and a little): the edges alone let a
                // flap stretch out in a steady pull
                if (limit.y >= 0.0)
                {
                    float3 anchor = state[from * count + (uint)limit.y].xyz;
                    float3 reach = x - anchor;
                    float reachSq = dot(reach, reach);
                    if (reachSq > limit.z * limit.z)
                        x = anchor + reach * (limit.z * rsqrt(reachSq));
                }

                // Within its reach of where the bones put it, and out of the body
                float3 off = x - target;
                float offSq = dot(off, off);
                if (offSq > most * most)
                    x = target + off * (most * rsqrt(offSq));
                x = KeepOut(x, frame);

                // No nearer its backstop part's axis than the bones put it, less the Backstop
                float4 back = state[STATE_BACK * count + p];
                if (back.x >= 0.0 && back.y > 0.0)
                {
                    uint backPart = min((uint)back.x, frame.PartCount - 1u);
                    float t;
                    float3 c = OnAxis(frame, backPart, x, t);
                    float3 away = x - c;
                    float awaySq = dot(away, away);
                    if (awaySq < back.y * back.y && awaySq > 1e-12)
                        x = c + away * (back.y * rsqrt(awaySq));
                }

                state[to * count + p] = float4(x, 0);
            }
            DeviceMemoryBarrierWithGroupSync();
        }
    }

    // This frame's bones' places are the last frame's for the next
    for (uint i4 = tid; i4 < count; i4 += THREADS)
        state[STATE_TARGET * count + i4] = state[STATE_NEXT * count + i4];
    DeviceMemoryBarrierWithGroupSync();

    // 3. Every vertex: its particle's place and the normal of the triangles round it, in object space. A vertex of
    //    another LOD in the range is left w = 0: the vertex shader skins it.
    StructuredBuffer<uint4> vertices = ResourceDescriptorHeap[VerticesIdx];
    StructuredBuffer<uint4> vertexTris = ResourceDescriptorHeap[VertexTrisIdx];
    RWStructuredBuffer<float4> output = ResourceDescriptorHeap[OutputUAVIdx];
    for (uint v = tid; v < VertexCount; v += THREADS)
    {
        uint4 vertex = vertices[v];
        if (vertex.x == 0xFFFFFFFFu)
        {
            output[v * 2] = 0;
            output[v * 2 + 1] = 0;
            continue;
        }

        float3 x = state[STATE_A * count + vertex.x].xyz;
        float3 normal = 0;
        for (uint t = 0; t < vertex.z; t++)
        {
            uint3 tri = vertexTris[vertex.y + t].xyz;
            float3 a = state[STATE_A * count + tri.x].xyz;
            float3 b = state[STATE_A * count + tri.y].xyz;
            float3 c = state[STATE_A * count + tri.z].xyz;
            normal += cross(b - a, c - a);
        }
        if (vertex.w != 0u) normal = -normal;
        normal = mul(normal, (float3x3)frame.ToObject);
        float lengthSq = dot(normal, normal);
        normal = lengthSq > 1e-20 ? normal * rsqrt(lengthSq) : float3(0, 1, 0);

        output[v * 2] = float4(mul(float4(x, 1), frame.ToObject).xyz, 1);
        output[v * 2 + 1] = float4(normal, 0);
    }
}
