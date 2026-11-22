// Mesh particles (ParticleEmitter RenderMode Mesh): the vertex shaders of gbuffer.fx's passes Particles and
// Shadow_Particles. Included by gbuffer.fx after its VSOutput, ShadowVSOutput and push constants, so a mesh particle is
// shaded by the same pixel shader as any mesh of that material: lit, shadowed, decals and weather on it, its own motion.
//
// Each instance is a pool slot from the emitter's list (ParticleSystem: the camera's built in the opaque pass's mesh phase,
// the cascades' by CSMeshShadow); the slot's chunk names the emitter row, which carries the mesh's centre, sizes and
// shrink. The particle stands where the mesh's centre is, turned by its tumble (particle_common.hlsli ParticleMeshTurn)
// and scaled by its size (ParticleMeshScale). Every part of an emitter is drawn with the emitter's longest vertex count:
// the corners past a part's own are collapsed.
//
// Push constants (gbuffer.fx): MeshPartId (2) the part, InstanceBaseOffset (3) the emitter's first entry in the list,
// SortedIndicesIdx (5) the camera's list, MeshRegistryIdx (6), MaterialsIdx (7), ParticleCoreIdx (9), ParticleVisualIdx
// (10), ParticleEmittersIdx (11), ParticleChunkMapIdx (12), ParticleMaterialId (13), ParticlePartVertices (15),
// IndexedDraw (17, 0: pulled), ParticleDeltaTime (18), ParticlePartPick (19, MeshVariants: each particle one part, about
// its own centre), ExpansionBufferIdx (20) the cascades' list, CascadeBufferSRVIdx (21).

#ifndef PARTICLE_MESH_HLSLI
#define PARTICLE_MESH_HLSLI

#include "particle_common.hlsli"

struct ParticleMeshPose
{
    float3 Position;        // the mesh's centre in the world
    float4 Turn;            // quaternion xyzw
    float  Scale;
    float3 Center;          // the mesh's centre in its own space
};

ParticleMeshPose ParticleMeshPoseOf(uint slot, float before)
{
    StructuredBuffer<uint> chunkMap = ResourceDescriptorHeap[ParticleChunkMapIdx];
    StructuredBuffer<ParticleEmitter> emitters = ResourceDescriptorHeap[ParticleEmittersIdx];
    StructuredBuffer<ParticleCore> cores = ResourceDescriptorHeap[ParticleCoreIdx];
    StructuredBuffer<ParticleVisual> visuals = ResourceDescriptorHeap[ParticleVisualIdx];
    ParticleEmitter e = emitters[chunkMap[slot >> PARTICLE_CHUNK_SHIFT]];
    ParticleCore core = cores[slot];
    ParticleVisual vis = visuals[slot];

    // before > 0: where it was that long ago (last frame, for the motion vectors)
    float age = max(core.Age - before, 0.0);
    ParticleMeshPose pose;
    pose.Position = core.Position - core.Velocity * (core.Age - age);
    pose.Turn = ParticleMeshTurn(vis, age);
    pose.Scale = ParticleMeshScale(e, vis, saturate(age / max(core.Lifetime, 0.001)));
    pose.Center = e.MeshCenter;
    return pose;
}

float3 ParticleMeshWorld(ParticleMeshPose pose, float3 local)
{
    return pose.Position + ParticleMeshRotate(pose.Turn, (local - pose.Center) * pose.Scale);
}

// MeshVariants (ParticlePartPick): whether this draw's part is the one the particle picked (each part's draw takes every
// particle of the emitter, and drops the others here, before anything else is read)
bool ParticleMeshPicked(uint slot)
{
    if (ParticlePartPick == 0u)
        return true;
    StructuredBuffer<ParticleVisual> visuals = ResourceDescriptorHeap[ParticleVisualIdx];
    uint picked = ParticleMeshHash(asuint(visuals[slot].Random) ^ 0x51ED270Bu) % (ParticlePartPick >> 16);
    return picked + 1u == (ParticlePartPick & 0xFFFFu);
}

// The pose a particle is drawn in by this part: a variant turns about its own part's centre, a whole mesh about the mesh's
ParticleMeshPose ParticleMeshPoseFor(uint slot, MeshPartEntry part, float before)
{
    ParticleMeshPose pose = ParticleMeshPoseOf(slot, before);
    if (ParticlePartPick != 0u)
        pose.Center = part.LocalBounds.xyz;
    return pose;
}

VSOutput VS_Particles(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output = (VSOutput)0;
    output.Position = float4(0.0, 0.0, 0.0, 1.0);          // dropped: a triangle of one point

    StructuredBuffer<uint> list = ResourceDescriptorHeap[SortedIndicesIdx];
    uint slot = list[InstanceBaseOffset + instanceID];
    // Past this part's corners (the draw runs to the emitter's longest part), or another of its parts is this particle
    if (primitiveVertexID >= ParticlePartVertices || !ParticleMeshPicked(slot))
        return output;

    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];
    // IndexedDraw is read (0 here) so the effect's batches keep drawing indexed (Effect.DrawsIndexed)
    uint vertexID = MeshVertexIndex(part, primitiveVertexID, primitiveVertexID, IndexedDraw);

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[part.NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];
    float3 pos = positions[vertexID];
    float3 norm = normals[vertexID];
    float3 push = norm;
    if (part.PushNormBufferIdx != 0)
    {
        StructuredBuffer<float3> pushNormals = ResourceDescriptorHeap[part.PushNormBufferIdx];
        push = pushNormals[vertexID];
    }

    ParticleMeshPose pose = ParticleMeshPoseFor(slot, part, 0.0);
    float3 world = ParticleMeshWorld(pose, pos);
    output.WorldPos = float4(world, 1.0);
    output.Position = mul(mul(output.WorldPos, View), Projection);

    // Where it was last frame: back along its velocity, its tumble and size a step younger
    float3 before = ParticleMeshWorld(ParticleMeshPoseFor(slot, part, asfloat(ParticleDeltaTime)), pos);
    output.MotionNow = MotionClip(world);
    output.MotionBefore = PrevMotionClip(before);

    output.Normal = ParticleMeshRotate(pose.Turn, norm);
    output.PushNormal = ParticleMeshRotate(pose.Turn, push);
    output.TexCoord = uvs[vertexID];
    output.TexCoord.y = 1 - output.TexCoord.y;
    output.MaterialID = ParticleMaterialId;
    output.Depth = output.Position.w;
    // No entity: never picked; decals take it as a static mesh (class 0)
    output.TransformSlot = 0;
    output.MeshPartIdx = 0;
    output.LodFade = 0;
    return output;
}

ShadowVSOutput VS_ParticlesShadow(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    ShadowVSOutput output;
    output.Position = float4(0.0, 0.0, 0.0, 1.0);
    output.TexCoord = float2(0.0, 0.0);
    output.MaterialID = 0;
    output.RTIndex = 0;
    output.ClipDistance = 1;

    StructuredBuffer<uint> expansion = ResourceDescriptorHeap[ExpansionBufferIdx];
    uint cascade, slot;
    DecodeShadowEntry(expansion[InstanceBaseOffset + instanceID], false, cascade, slot);
    if (primitiveVertexID >= ParticlePartVertices || !ParticleMeshPicked(slot))
        return output;

    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];
    uint vertexID = MeshVertexIndex(part, primitiveVertexID, primitiveVertexID, IndexedDraw);
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    float3 world = ParticleMeshWorld(ParticleMeshPoseFor(slot, part, 0.0), positions[vertexID]);
    output.Position = ShadowClipPosition(float4(world, 1.0), cascade, false, CascadeBufferSRVIdx, output.RTIndex, output.ClipDistance);
    return output;
}

#endif
