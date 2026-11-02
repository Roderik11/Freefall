// Cloth on the GPU (Graphics/ClothSystem.cs, Components/Cloth.cs): clothing on a skinned character, or a banner or a
// flag on a static mesh.
//
// One thread group per cloth, every cloth of a frame in one dispatch: the group's row (SV_GroupID) names its buffers,
// its ranges of the shared pools and its settings for this frame. A particle is a vertex of the mesh's LOD 0 (vertices
// at the same place, a seam in the UVs, are one particle). Each frame:
//   1. every particle's held place (skinned like gbuffer_skinned.fx, or as it is for a static mesh, then into the world);
//   2. per substep: the place a step on (Verlet: its motion damped, gravity, the air pressing on its face by how square
//      the wind meets it, the wind wavering by place and time), then passes over its links (edges keep their length,
//      links across two triangles keep it from folding), each pass every particle moved by the mean of what its links
//      ask (Jacobi, from one buffer into the other), kept within its tether of the nearest held particle and its reach
//      of its held place, out of the body's parts (round cones, the row's parts) and not sunk towards them;
//   3. every LOD 0 vertex's place and normal (from the triangles round it, turned the way the mesh's own normal faces
//      at the first frame) in the renderer's object space; then every vertex of the other LODs: its own held place
//      moved by as much as its particle was, with that particle's normal.
// The skinned vertex shader draws these instead of skinning. A particle with MaxAway 0 is held. Positions are in the
// world: gravity and wind are. Every buffer index comes from the group's row, the same for all its threads.

#pragma kernel CSCloth

#define THREADS 256
#define MOST_PARTS 16

cbuffer PushConstants : register(b3)
{
    uint RowsIdx;           //  0  StructuredBuffer<ClothRow>: one per group
    uint PartsIdx;          //  1  StructuredBuffer<float4>: the body parts' axes, two a part (end, radius there)
    uint StateUAVIdx;       //  2  RWStructuredBuffer<float4>: the state pool, 6 ranges of ParticleCount a cloth (STATE_*)
    uint OutputUAVIdx;      //  3  RWStructuredBuffer<float4>: the output pool, per vertex its place (w 1) and its normal (w: which way it faces, +1/-1)
    uint TimeBits;          //  4  seconds (asfloat), for the wind's wavering
};

struct ClothRow
{
    row_major float4x4 World;       // object -> world, this frame
    row_major float4x4 ToObject;    // world -> object
    float3 Wind;
    float Dt;                       // a substep's seconds
    float Drag;
    float Damping;
    float Stiffness;
    float Bending;
    float Thickness;
    float Backstop;                 // how far it may sink towards the body from its held place
    float Flutter;                  // how much the wind wavers, 0..1
    float Streams;                  // share of Drag the wind pulls with whichever way the cloth faces
    uint Substeps;
    uint Iterations;
    uint Reset;                     // 1 = start where everything is held
    uint PartFirst;                 // its body parts in the parts buffer (two float4 a part)
    uint PartCount;
    uint ParticleCount;
    uint VertexCount;
    uint VertexFirst;               // the mesh's vertex the range starts at
    uint StateBase;                 // its ranges of the pools (float4s)
    uint OutputBase;
    uint ParticlesIdx;              // StructuredBuffer<ClothParticle>
    uint LinksIdx;                  // StructuredBuffer<ClothLink>
    uint VerticesIdx;               // StructuredBuffer<uint4>: particle, first of its triangles, how many, flags (FOLLOWS)
    uint VertexTrisIdx;             // StructuredBuffer<uint4>: a triangle round a vertex as particles (xyz)
    uint LimitsIdx;                 // StructuredBuffer<float4>, per particle: how far it may leave its held place, its tether (the held particle nearest it, -1 none) and the tether's length
    uint BonesIdx;                  // StructuredBuffer<row_major float4x4>: the Animator's bone buffer, 0 = as the mesh is
    uint MeshPositionsIdx;          // the mesh's buffers: positions, normals, bone weights (0 = none)
    uint MeshNormalsIdx;
    uint MeshWeightsIdx;
    uint _pad;
};

struct ClothParticle
{
    float3 Rest;            // in the mesh's space, as its vertex buffer has it
    uint First;             // its first link
    float4 BoneIds;
    float4 Weights;
    uint Count;             // how many links
    uint Vertex;            // a vertex of LOD 0 it is (in the range): where its normal is
    uint2 _pad;
};

struct ClothLink
{
    uint Other;
    float Rest;             // its length at rest
    uint Bend;              // 1 = across two triangles, 0 = an edge
    uint _pad;
};

struct BoneWeight
{
    float4 BoneIDs;
    float4 Weights;
};

static const float3 Gravity = float3(0, -9.81, 0);

// The state's ranges
#define STATE_A 0       // place
#define STATE_B 1       // the other place buffer (a pass reads one and writes the other)
#define STATE_PREV 2    // place a step ago
#define STATE_TARGET 3  // its held place last frame (after the substeps: this frame's)
#define STATE_NEXT 4    // ... this frame
#define STATE_BACK 5    // x: the body part nearest its held place this substep, y: how near its axis it may come

// Vertices.w
#define FOLLOWS 2u

// The group's body parts, read once
groupshared float4 PartFrom[MOST_PARTS];
groupshared float4 PartTo[MOST_PARTS];

// The nearest place on a body part's axis
float3 OnAxis(uint part, float3 x)
{
    float3 a = PartFrom[part].xyz, b = PartTo[part].xyz;
    float3 ab = b - a;
    float t = saturate(dot(x - a, ab) / max(dot(ab, ab), 1e-8));
    return a + ab * t;
}

// A point of the mesh (its space) skinned as the vertex shader skins it, then into the world
float3 HeldWorld(float3 pos, float4 boneIds, float4 weights, ClothRow row)
{
    if (row.BonesIdx != 0u)
    {
        StructuredBuffer<row_major float4x4> bones = ResourceDescriptorHeap[row.BonesIdx];
        float4 skinned = mul(float4(pos, 1), bones[(uint)boneIds.x]) * weights.x
                       + mul(float4(pos, 1), bones[(uint)boneIds.y]) * weights.y
                       + mul(float4(pos, 1), bones[(uint)boneIds.z]) * weights.z
                       + mul(float4(pos, 1), bones[(uint)boneIds.w]) * weights.w;
        pos = skinned.xyz;
    }
    return mul(float4(pos, 1), row.World).xyz;
}

// A direction of the mesh skinned the same way, in object space
float3 HeldNormal(float3 normal, float4 boneIds, float4 weights, ClothRow row)
{
    if (row.BonesIdx == 0u)
        return normal;
    StructuredBuffer<row_major float4x4> bones = ResourceDescriptorHeap[row.BonesIdx];
    return mul(normal, (float3x3)bones[(uint)boneIds.x]) * weights.x
         + mul(normal, (float3x3)bones[(uint)boneIds.y]) * weights.y
         + mul(normal, (float3x3)bones[(uint)boneIds.z]) * weights.z
         + mul(normal, (float3x3)bones[(uint)boneIds.w]) * weights.w;
}

// Out of the body's parts: the nearest place on a part's axis, the part's radius there and the cloth's thickness
float3 KeepOut(float3 x, uint partCount, float thickness)
{
    for (uint i = 0; i < partCount; i++)
    {
        float3 a = PartFrom[i].xyz, b = PartTo[i].xyz;
        float3 ab = b - a;
        float t = saturate(dot(x - a, ab) / max(dot(ab, ab), 1e-8));
        float3 closest = a + ab * t;
        float radius = lerp(PartFrom[i].w, PartTo[i].w, t) + thickness;
        float3 away = x - closest;
        float distanceSq = dot(away, away);
        if (distanceSq < radius * radius && distanceSq > 1e-12)
            x = closest + away * (radius * rsqrt(distanceSq));
    }
    return x;
}

// The wind's wavering at a place, about 1 (smooth in place and time, a few metres and a second or so across)
float Waver(float3 p, float time)
{
    float a = sin(dot(p, float3(1.7, 2.3, 1.1)) + time * 3.1);
    float b = sin(dot(p, float3(-2.9, 1.3, 2.7)) + time * 5.3);
    float c = sin(dot(p, float3(0.7, -1.9, -3.3)) + time * 7.7);
    return 1.0 + 0.5 * a + 0.3 * b * a + 0.2 * c;
}

[numthreads(THREADS, 1, 1)]
void CSCloth(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID)
{
    StructuredBuffer<ClothRow> rows = ResourceDescriptorHeap[RowsIdx];
    ClothRow row = rows[groupId.x];

    StructuredBuffer<ClothParticle> particles = ResourceDescriptorHeap[row.ParticlesIdx];
    StructuredBuffer<ClothLink> links = ResourceDescriptorHeap[row.LinksIdx];
    StructuredBuffer<float4> limits = ResourceDescriptorHeap[row.LimitsIdx];
    RWStructuredBuffer<float4> state = ResourceDescriptorHeap[StateUAVIdx];
    RWStructuredBuffer<float4> output = ResourceDescriptorHeap[OutputUAVIdx];
    float time = asfloat(TimeBits);

    uint tid = threadId.x;
    uint count = row.ParticleCount;
    uint partCount = min(row.PartCount, (uint)MOST_PARTS);
    uint s0 = row.StateBase;
    uint o0 = row.OutputBase;

    if (tid < partCount)
    {
        StructuredBuffer<float4> parts = ResourceDescriptorHeap[PartsIdx];
        PartFrom[tid] = parts[row.PartFirst + tid * 2u];
        PartTo[tid] = parts[row.PartFirst + tid * 2u + 1u];
    }

    // 1. Where each particle is held this frame
    for (uint i1 = tid; i1 < count; i1 += THREADS)
    {
        uint p = i1;
        ClothParticle particle = particles[p];
        float3 next = HeldWorld(particle.Rest, particle.BoneIds, particle.Weights, row);
        state[s0 + STATE_NEXT * count + p] = float4(next, 0);
        if (row.Reset != 0u)
        {
            state[s0 + STATE_A * count + p] = float4(next, 0);
            state[s0 + STATE_PREV * count + p] = float4(next, 0);
            state[s0 + STATE_TARGET * count + p] = float4(next, 0);
        }
    }
    AllMemoryBarrierWithGroupSync();        // the parts in groupshared too

    float dt = row.Dt;
    float3x3 toWorld = (float3x3)row.World;
    for (uint s = 0; s < row.Substeps; s++)
    {
        float share = (s + 1.0) / row.Substeps;

        // 2a. A step on
        for (uint i2 = tid; i2 < count; i2 += THREADS)
        {
            uint p = i2;
            float3 target = lerp(state[s0 + STATE_TARGET * count + p].xyz, state[s0 + STATE_NEXT * count + p].xyz, share);
            float3 x;
            float3 previous;
            if (limits[p].x <= 0.0)
            {
                x = target;
                previous = target;
            }
            else
            {
                x = state[s0 + STATE_A * count + p].xyz;
                float3 motion = (x - state[s0 + STATE_PREV * count + p].xyz) * (1.0 - row.Damping);
                float3 velocity = motion / max(dt, 1e-5);

                // The air: the wind (wavering) against the cloth's own motion, pressing on its face by how square it
                // meets it (its normal from the last frame), and Streams of that whichever way it faces
                float3 wind = row.Wind * lerp(1.0, Waver(target, time), row.Flutter);
                float3 relative = wind - velocity;
                float3 normal = output[o0 + particles[p].Vertex * 2u + 1u].xyz;
                float3 pull = Gravity + relative * (row.Streams * row.Drag);
                float normalSq = dot(normal, normal);
                if (normalSq > 1e-8)
                {
                    float3 n = normalize(mul(normal, toWorld));
                    pull += n * (dot(relative, n) * row.Drag);
                }

                previous = x;
                x += motion + pull * dt * dt;
            }
            state[s0 + STATE_A * count + p] = float4(x, 0);
            state[s0 + STATE_PREV * count + p] = float4(previous, 0);

            // Its backstop: the body part whose axis is nearest its held place, and that distance less how far it may
            // sink. Where the clothes under it stand out from the body's parts (mail), it stays over them.
            float nearest = 1e20;
            uint nearestPart = 0xFFFFFFFFu;
            for (uint i = 0; i < partCount; i++)
            {
                float3 c = OnAxis(i, target);
                float d = dot(target - c, target - c);
                if (d < nearest) { nearest = d; nearestPart = i; }
            }
            float reach = nearestPart != 0xFFFFFFFFu ? sqrt(nearest) - row.Backstop : 0.0;
            state[s0 + STATE_BACK * count + p] = float4(nearestPart != 0xFFFFFFFFu ? (float)nearestPart : -1.0, reach, 0, 0);
        }
        DeviceMemoryBarrierWithGroupSync();

        // 2b. Passes over the links, from A into B and back (an even number: it ends in A)
        uint passes = (row.Iterations + 1u) & ~1u;
        for (uint it = 0; it < passes; it++)
        {
            uint from = s0 + ((it & 1u) == 0u ? STATE_A : STATE_B) * count;
            uint to = s0 + ((it & 1u) == 0u ? STATE_B : STATE_A) * count;
            for (uint i3 = tid; i3 < count; i3 += THREADS)
            {
                uint p = i3;
                float3 target = lerp(state[s0 + STATE_TARGET * count + p].xyz, state[s0 + STATE_NEXT * count + p].xyz, share);
                float4 limit = limits[p];
                float most = limit.x;
                float3 x = state[from + p].xyz;
                if (most <= 0.0)
                {
                    state[to + p] = float4(target, 0);
                    continue;
                }

                ClothParticle particle = particles[p];
                float3 move = 0;
                float weight = 0;
                for (uint l = 0; l < particle.Count; l++)
                {
                    ClothLink link = links[particle.First + l];
                    float3 other = state[from + link.Other].xyz;
                    float3 d = x - other;
                    float length = sqrt(max(dot(d, d), 1e-12));
                    // A held neighbour does not give: this particle takes the whole correction
                    float give = limits[link.Other].x <= 0.0 ? 1.0 : 0.5;
                    float k = link.Bend != 0u ? row.Bending : row.Stiffness;
                    move -= d * ((length - link.Rest) / length) * give * k;
                    weight += 1.0;
                }
                if (weight > 0.0)
                    x += move * (1.5 / weight);

                // No further from the held particle nearest it than at rest (and a little): the edges alone let a
                // flap stretch out in a steady pull
                if (limit.y >= 0.0)
                {
                    float3 anchor = state[from + (uint)limit.y].xyz;
                    float3 tether = x - anchor;
                    float tetherSq = dot(tether, tether);
                    if (tetherSq > limit.z * limit.z)
                        x = anchor + tether * (limit.z * rsqrt(tetherSq));
                }

                // Within its reach of its held place, and out of the body
                float3 off = x - target;
                float offSq = dot(off, off);
                if (offSq > most * most)
                    x = target + off * (most * rsqrt(offSq));
                x = KeepOut(x, partCount, row.Thickness);

                // No nearer its backstop part's axis than its held place, less the Backstop
                float4 back = state[s0 + STATE_BACK * count + p];
                if (back.x >= 0.0 && back.y > 0.0 && partCount > 0u)
                {
                    uint backPart = min((uint)back.x, partCount - 1u);
                    float3 c = OnAxis(backPart, x);
                    float3 away = x - c;
                    float awaySq = dot(away, away);
                    if (awaySq < back.y * back.y && awaySq > 1e-12)
                        x = c + away * (back.y * rsqrt(awaySq));
                }

                state[to + p] = float4(x, 0);
            }
            DeviceMemoryBarrierWithGroupSync();
        }
    }

    // This frame's held places are the last frame's for the next
    for (uint i4 = tid; i4 < count; i4 += THREADS)
        state[s0 + STATE_TARGET * count + i4] = state[s0 + STATE_NEXT * count + i4];
    DeviceMemoryBarrierWithGroupSync();

    // 3. Every vertex in object space. LOD 0's first: its particle's place and the normal of the triangles round it,
    //    turned the way the mesh's own normal faces (worked out at the first frame and kept in the normal's w). A vertex
    //    the cloth does not hold is left w = 0: the vertex shader draws it as it is.
    StructuredBuffer<uint4> vertices = ResourceDescriptorHeap[row.VerticesIdx];
    StructuredBuffer<uint4> vertexTris = ResourceDescriptorHeap[row.VertexTrisIdx];
    for (uint v = tid; v < row.VertexCount; v += THREADS)
    {
        uint4 vertex = vertices[v];
        if (vertex.x == 0xFFFFFFFFu)
        {
            output[o0 + v * 2] = 0;
            output[o0 + v * 2 + 1] = 0;
            continue;
        }
        if ((vertex.w & FOLLOWS) != 0u)
            continue;

        float3 x = state[s0 + STATE_A * count + vertex.x].xyz;
        float3 normal = 0;
        for (uint t = 0; t < vertex.z; t++)
        {
            uint3 tri = vertexTris[vertex.y + t].xyz;
            float3 a = state[s0 + STATE_A * count + tri.x].xyz;
            float3 b = state[s0 + STATE_A * count + tri.y].xyz;
            float3 c = state[s0 + STATE_A * count + tri.z].xyz;
            normal += cross(b - a, c - a);
        }
        normal = mul(normal, (float3x3)row.ToObject);

        float facing = output[o0 + v * 2 + 1].w;
        if (row.Reset != 0u || facing == 0.0)
        {
            // At rest the triangles' normal and the mesh's own agree or not: how this vertex's triangles are wound
            facing = 1.0;
            if (row.MeshNormalsIdx != 0u)
            {
                StructuredBuffer<float3> meshNormals = ResourceDescriptorHeap[row.MeshNormalsIdx];
                float3 own = meshNormals[row.VertexFirst + v];
                if (row.MeshWeightsIdx != 0u)
                {
                    StructuredBuffer<BoneWeight> meshWeights = ResourceDescriptorHeap[row.MeshWeightsIdx];
                    BoneWeight bw = meshWeights[row.VertexFirst + v];
                    own = HeldNormal(own, bw.BoneIDs, bw.Weights, row);
                }
                facing = dot(normal, own) < 0.0 ? -1.0 : 1.0;
            }
        }
        normal *= facing;
        float lengthSq = dot(normal, normal);
        normal = lengthSq > 1e-20 ? normal * rsqrt(lengthSq) : float3(0, 1, 0);

        output[o0 + v * 2] = float4(mul(float4(x, 1), row.ToObject).xyz, 1);
        output[o0 + v * 2 + 1] = float4(normal, facing);
    }
    DeviceMemoryBarrierWithGroupSync();

    //    Then the other LODs': its own held place, moved by as much as its particle has moved from its held place, and
    //    that particle's normal, turned the way its own normal faces
    if (row.MeshPositionsIdx == 0u)
        return;
    StructuredBuffer<float3> meshPositions = ResourceDescriptorHeap[row.MeshPositionsIdx];
    for (uint f = tid; f < row.VertexCount; f += THREADS)
    {
        uint4 vertex = vertices[f];
        if (vertex.x == 0xFFFFFFFFu || (vertex.w & FOLLOWS) == 0u)
            continue;

        float4 boneIds = 0, weights = float4(1, 0, 0, 0);
        if (row.MeshWeightsIdx != 0u)
        {
            StructuredBuffer<BoneWeight> meshWeights = ResourceDescriptorHeap[row.MeshWeightsIdx];
            BoneWeight bw = meshWeights[row.VertexFirst + f];
            boneIds = bw.BoneIDs;
            weights = bw.Weights;
        }
        float3 held = HeldWorld(meshPositions[row.VertexFirst + f], boneIds, weights, row);
        float3 moved = state[s0 + STATE_A * count + vertex.x].xyz - state[s0 + STATE_TARGET * count + vertex.x].xyz;
        float3 x = held + moved;

        uint lead = particles[vertex.x].Vertex;
        float3 normal = output[o0 + lead * 2 + 1].xyz;
        float facing = output[o0 + f * 2 + 1].w;
        if (row.Reset != 0u || facing == 0.0)
        {
            // Its own normal and its lead's, both at rest: the other side of a two-sided piece faces the other way
            facing = 1.0;
            if (row.MeshNormalsIdx != 0u)
            {
                StructuredBuffer<float3> meshNormals = ResourceDescriptorHeap[row.MeshNormalsIdx];
                facing = dot(meshNormals[row.VertexFirst + f], meshNormals[row.VertexFirst + lead]) < 0.0 ? -1.0 : 1.0;
            }
        }

        output[o0 + f * 2] = float4(mul(float4(x, 1), row.ToObject).xyz, 1);
        output[o0 + f * 2 + 1] = float4(normal * facing, facing);
    }
}
