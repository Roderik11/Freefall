// GPU Frustum Culling with Hierarchical Prefix Sum Stream Compaction
// 5-pass pipeline for proper large-array support:
// 1. CSVisibility - frustum test, write visibility flags
// 2. CSLocalScan - local prefix sum per block, output block sums
// 3. CSBlockScan - scan the block sums
// 4. CSGlobalScatter - add block prefix, scatter to compacted output
// 5. CSMain - generate indirect draw commands

#pragma kernel CSVisibility
#pragma kernel CSHistogram
#pragma kernel CSLocalScan
#pragma kernel CSBlockScan
#pragma kernel CSGlobalScatter
#pragma kernel CSSortIndirection
#pragma kernel CSBitonicSort
#pragma kernel CSHistogramPrefixSum
#pragma kernel CSMain
#pragma kernel CSClear
#pragma kernel CSComputeDepthKeys
#pragma kernel CSBitonicSortPairs
#pragma kernel CSWriteSortedIndices
#pragma kernel CSVisibilityShadow
#pragma kernel CSVisibilityCascades
#pragma kernel CSHistogramCascades
#pragma kernel CSScatterCascades
#pragma kernel CSCompactCommands
#pragma kernel CSVisibilityLocalShadows
#pragma kernel CSHistogramLocalShadows
#pragma kernel CSScatterLocalShadows

#include "local_shadows.hlsli"

// Frustum planes + Hi-Z occlusion parameters (root slot 1 -> register b0)
cbuffer FrustumPlanes : register(b0)
{
    float4 Plane0;
    float4 Plane1;
    float4 Plane2;
    float4 Plane3;
    float4 Plane4;
    float4 Plane5;
    // Hi-Z occlusion culling parameters
    row_major float4x4 OcclusionProjection; // Projection matrix for sphere-to-screen
    uint HiZSrvIdx;        // Bindless SRV index of Hi-Z pyramid (0 = disabled)
    float2 HiZSize;        // Mip 0 dimensions
    uint HiZMipCount;      // Number of mip levels in pyramid
    float NearPlane;       // Camera near plane for depth margin calculation
    uint CullStatsUAVIdx;  // UAV index for cull stats buffer (0=disabled)
    uint DebugMode;        // Debug visualization mode (unused by culler, kept for layout compatibility)
    float ProjScale;       // Projection._m22 = cot(fovY/2) for sphere-to-screen size
    // Depth sorting parameters
    float3 CameraPosition; // Camera world position for distance computation
    uint SortDirection;    // 0 = front-to-back (opaque), 1 = back-to-front (transparent)
};

// Shadow cascade data (StructuredBuffer via push constant — used by rendering shaders)
struct CascadeData
{
    float4 Planes[6];           // frustum planes
    row_major float4x4 VP;      // current frame VP
    row_major float4x4 PrevVP;  // previous frame VP (for Hi-Z)
    float4 SplitDistances;      // X=near, Y=far
};

// CascadeBufferSRVIdx and ShadowHiZSrvIdx are aliased in cbuffer PushConstants below

// Shadow cascade frustum planes (cbuffer — used by culler for frustum tests)
cbuffer ShadowCascadePlanes : register(b1)
{
    float4 CascadePlane0[6];
    float4 CascadePlane1[6];
    float4 CascadePlane2[6];
    float4 CascadePlane3[6];
};

// Push constants for bindless buffer indices (root slot 0 -> register b3)
cbuffer PushConstants : register(b3)
{
    uint MeshRegistryIdx;           // slot 0  — SRV: global mesh/part registry
    uint OutputBufferUAVIdx;        // slot 1  — UAV: output commands
    uint PrevVisibilityIdx;         // slot 2  — SRV: previous visibility / shadow Hi-Z (aliased)
    uint InstanceMultiplierIdx;     // slot 3  — DrawInstanceCount multiplier (1=normal, 4=shadow)
    uint DescriptorBufferIdx;       // slot 4  — SRV: per-instance InstanceDescriptor buffer
    uint VisibleIndicesUAVIdx;      // slot 5  — UAV: output visible indices (compacted)
    uint CounterBufferUAVIdx;       // slot 6  — UAV: per-subbatch visible counts
    uint SubBatchCountIdx;          // slot 7  — Total number of subbatches
    uint CounterOffsetIdx;          // slot 8  — Offset into counter buffer
    uint VisibleIndicesSRVIdx;      // slot 9  — SRV: visible indices for rendering
    uint GlobalTransformsIdx;       // slot 10 — SRV: global transform buffer
    uint VisibilityFlagsUAVIdx;     // slot 11 — UAV: visibility flags (1=visible, 0=not)
    uint PrefixSumUAVIdx;           // slot 12 — UAV: prefix sum buffer
    uint TotalInstancesIdx;         // slot 13 — Total instance count
    uint BlockSumsUAVIdx;           // slot 14 — UAV: block sums for hierarchical scan
    uint NumBlocksIdx;              // slot 15 — Number of blocks in scan
    uint SubbatchStartIdx;          // slot 16 — Starting index of current subbatch
    uint SubbatchCountIdx;          // slot 17 — Number of instances in current subbatch
    uint DrawCountUAVIdx;            // slot 18 — UAV: non-empty draw command count for ExecuteIndirect
    uint IndirectionUAVIdx;         // slot 19 — UAV: maps group-order -> draw-order index
    uint SortSizeIdx;               // slot 20 — Block size for this merge step
    uint SortStrideIdx;             // slot 21 — Stride within block
    uint SortCountIdx;              // slot 22 — Total count to sort; the cascade chain: its cascade count (ShadowCascadeCount)
    uint SubbatchIdsIdx;            // slot 23 — SRV: per-instance subbatch index
    uint CombinedShadowVisUAVIdx;   // slot 24 — UAV: combined shadow visibility; CSVisibility: the first instance its transform cache re-reads every frame (CacheRefreshFrom)
    uint MaterialsBufferIdx;        // slot 25 — SRV: materials array buffer
    uint HistogramUAVIdx;           // slot 26 — UAV: histogram buffer (count per MeshPartId)
    uint UniquePartCountIdx;        // slot 27 — Max unique MeshPartIds in registry
    uint CascadeMaskUAVIdx;         // slot 28 — UAV: per-instance cascade mask (a bit per cascade); the camera's CSVisibility: its transform cache (VisibilityCacheIdx)
    uint ExpansionUAVIdx;           // slot 29 — UAV: expansion buffer for (instanceIdx, cascadeIdx); CSVisibility: the slots uploaded this tick (ChangedSlotsIdx)
    float ShadowMinCasterClip;      // slot 30 — shadow cascades 1+: smallest caster diameter in clip units (0 = off)
    uint CascadeBufferSRVIdx;       // slot 31 — SRV: StructuredBuffer<CascadeData> / shadow cascade idx (aliased)
    uint SortKeysUAVIdx;            // slot 32 — UAV: depth sort keys buffer (uint2 per entry)
    uint MeshLodRegistryIdx;        // slot 33 — SRV: MeshPartLod buffer, parallel to the mesh registry
    float LodCameraX;               // slot 34 — main camera position for LOD selection. Set for shadow
    float LodCameraY;               // slot 35   passes too, so shadows use the main view's LOD.
    float LodCameraZ;               // slot 36   (separate floats: a float3 would not pack into these slots)
    float LodFactor;                // slot 37 — Camera.FoVFactor * LODScale² (0 = no valid camera: LOD-managed instances are not drawn)
    uint InstanceBoundsUAVIdx;      // slot 38 — UAV: InstanceBounds per instance; CSVisibility writes, CSVisibilityCascades/LocalShadows read (0 = none)
    uint ShadowReceiversUAVIdx;     // slot 39 — UAV: the sun's receiver tiles + stats (shadow_receivers.hlsli); CSVisibilityCascades only (0 = none); CSVisibility: the changed records (ChangedRecordsIdx)
};

#include "shadow_cascades.hlsli"
#include "shadow_receivers.hlsli"

// Convenience aliases for named push constants (matching old #define usage)
#define OutputBufferIdx       OutputBufferUAVIdx
#define InstanceMultiplier    InstanceMultiplierIdx
#define VisibilityFlagsIdx    VisibilityFlagsUAVIdx
#define CounterBufferIdx      CounterBufferUAVIdx
#define SubBatchCount         SubBatchCountIdx
#define CounterOffset         CounterOffsetIdx
#define TotalInstances        TotalInstancesIdx
#define PrefixSumIdx          PrefixSumUAVIdx
#define BlockSumsIdx          BlockSumsUAVIdx
#define NumBlocks             NumBlocksIdx
#define IndirectionIdx        IndirectionUAVIdx
#define SubbatchStart         SubbatchStartIdx
#define SubbatchCount         SubbatchCountIdx
#define SortSize              SortSizeIdx
#define SortStride            SortStrideIdx
#define SortCount             SortCountIdx
#define HistogramIdx          HistogramUAVIdx
#define UniquePartCount       UniquePartCountIdx
#define VisibleIndicesUAVIdx  VisibleIndicesUAVIdx
#define CombinedShadowVisIdx  CombinedShadowVisUAVIdx
#define ExpansionUAVIdx       ExpansionUAVIdx
#define CascadeMaskUAVIdx     CascadeMaskUAVIdx
#define ShadowHiZSrvIdx       PrevVisibilityIdx
#define ShadowCascadeIdx      CascadeBufferSRVIdx
// The cascade chain never sorts: slot 22 holds its cascade count (clamped: a stale value must not run away)
#define ShadowCascadeCount    min(SortCountIdx, SHADOW_MAX_CASCADES)
// ... and slot 32 holds 1 + this batch's row of caster stats in the receivers buffer (0 = not counted; the sun's only)
#define CasterStatsRow        SortKeysUAVIdx
// The camera chain (InstanceBatch.Cull) never reads the shadow slots 24/28/29/39, so CSVisibility takes them over
#define CacheRefreshFrom      CombinedShadowVisUAVIdx
#define VisibilityCacheIdx    CascadeMaskUAVIdx
#define ChangedSlotsIdx       ExpansionUAVIdx
#define ChangedRecordsIdx     ShadowReceiversUAVIdx
// The camera chain's LOD crossfade (InstanceBatch.PrepareCull; every other chain leaves slot 8 at 0, and nothing
// reads slots 14, 15 or 17 without it)
#define LodFadeStateIdx       CounterOffsetIdx              // slot 8  — UAV: LodFadeState per instance (0 = no fades)
#define LodTime               asfloat(BlockSumsUAVIdx)      // slot 14 — seconds (Time.TotalTime)
#define LodFadeRate           asfloat(NumBlocksIdx)         // slot 15 — 1 / the fade's seconds (0 = switch at once: a camera cut)
#define LodFadeFrom           SubbatchCountIdx              // slot 17 — the persistent records, the only ones that keep a state
// Indexed draws (InstanceBatch.DrawsIndexed; nothing sets slot 12 otherwise, and CSLocalScan, which read it, is never
// dispatched): UAV the commands as indexed draws (IndexedDrawCommand; 0 = none). The camera chain's CSMain writes a part's
// command there (counted in the draw count's [1]) unless the part has a mirrored instance in view; the shadow chains'
// CSCompactCommands write every packed command there too, at the same slot.
#define IndexedCommandsIdx    PrefixSumUAVIdx               // slot 12
// The camera chain only (CSVisibility, CSMain in packing mode): UAV a uint per mesh part, the last MirrorCull in which a
// visible instance of the part was mirrored (its transform turns the triangles round, which only the pulled vertex
// shaders undo); 0 = none. The shadow chains use these slots for their cascades and never pack in CSMain.
#define MirrorPartsIdx        asuint(ShadowMinCasterClip)   // slot 30
#define MirrorCull            CascadeBufferSRVIdx           // slot 31 — this culling's number (never 0)

#include "lod_fade.hlsli"

// Per-subbatch instance range (extended with MeshPartId)
struct InstanceRange
{
    uint StartInstance;
    uint InstanceCount;
    uint MeshPartId;       // Index into MeshRegistry
    uint Reserved;         // Padding for 16-byte alignment
};

// MeshRegistry entry - matches C# MeshPartEntry exactly (72 bytes = 18 uints)
struct MeshPartEntry
{
    uint PosBufferIdx;
    uint NormBufferIdx;
    uint UVBufferIdx;
    uint IndexBufferIdx;
    uint BaseIndex;
    uint VertexCount;
    uint BoneWeightsBufferIdx;
    uint PushNormBufferIdx;
    // Local-space bounding sphere (center + radius) for GPU culling
    float4 LocalBounds;  // xyz = center, w = radius
    // Reserved fields for padding to match C# layout
    uint Reserved4;
    uint IndexBufferAddressLo;  // the mesh's index buffer (IndexedDrawCommand)
    uint IndexBufferAddressHi;
    uint IndexBufferBytes;
    uint Reserved8;
    uint Reserved9;
};

// LOD chain data per mesh part - matches C# MeshRegistry.MeshPartLod exactly (32 bytes).
// Indexed by MeshPartId. "K" values are normalised squared distances:
//   k = distanceSq / (LodFactor * instanceScale²)
// with the mesh radius, screen-size thresholds and LODBias folded in on the CPU (Mesh.GetPartLod).
struct MeshPartLod
{
    float3 Center;      // mesh-local centre shared by all parts, so they make the same decision
    float CullK;        // beyond this the mesh is too small on screen (0 = never)
    float NearK;        // the part is not drawn closer than this (0 = no limit)
    float FarK;         // beyond this NextPartId takes over (0 = no limit)
    uint NextPartId;    // MeshPartId of the next LOD's part + 1 (0 = none)
    uint Reserved;
};

// High bit of a subbatch ID: the instance was submitted as a LOD chain head and the culler
// resolves the part to draw (DrawBucket.LodManagedBit). Without it the ID is drawn as-is.
#define LOD_MANAGED_BIT 0x80000000u
#define INVALID_PART    0xFFFFFFFFu

//-----------------------------------------------------------------------------
// LOD selection: resolve a per-instance subbatch ID to the MeshPartId to draw.
// Returns INVALID_PART if nothing should be drawn (too small, or no part at this LOD).
// Depends only on the main camera (LodCamera*, LodFactor), never on the frustum being
// culled against, so shadow passes resolve the same part as the main view.
//
// The visibility kernels store the result as (partId + 1) in the visibility flags
// (0 = culled); CSHistogram / CSGlobalScatter / CSComputeDepthKeys read it from there.
//-----------------------------------------------------------------------------

// The part of the chain whose head is partId (its entry: lod) at k
uint LodChainPart(StructuredBuffer<MeshPartLod> lodRegistry, uint partId, MeshPartLod lod, float k)
{
    if (lod.CullK > 0.0 && k > lod.CullK)
        return INVALID_PART;

    // Walk the chain to the part of the active LOD
    for (uint i = 0; i < 8; i++)
    {
        if (k < lod.NearK)
            return INVALID_PART;
        if (lod.FarK == 0.0 || k < lod.FarK)
            return partId;
        if (lod.NextPartId == 0)
            return INVALID_PART;

        partId = lod.NextPartId - 1;
        lod = lodRegistry[partId];
    }
    return partId;
}

// k of an instance (see MeshPartLod) by the chain head's entry
float LodK(MeshPartLod head, row_major float4x4 world, float maxScaleSq)
{
    float3 toCamera = mul(float4(head.Center, 1.0), world).xyz - float3(LodCameraX, LodCameraY, LodCameraZ);
    return dot(toCamera, toCamera) / max(LodFactor * maxScaleSq, 1e-12);
}

uint ResolveLodPart(uint subbatchId, row_major float4x4 world, float maxScaleSq)
{
    uint partId = subbatchId & ~LOD_MANAGED_BIT;
    if ((subbatchId & LOD_MANAGED_BIT) == 0)
        return partId;

    // No valid main camera this frame (none exists, or it has not updated its projection yet, as in
    // the frames right after entering/leaving play mode): draw nothing. Falling back to the chain
    // head would draw every instance at LOD 0 at once, which is enough to hang the GPU on a forest.
    if (LodFactor <= 0.0)
        return INVALID_PART;

    StructuredBuffer<MeshPartLod> lodRegistry = ResourceDescriptorHeap[MeshLodRegistryIdx];
    MeshPartLod lod = lodRegistry[partId];
    return LodChainPart(lodRegistry, partId, lod, LodK(lod, world, maxScaleSq));
}

// k a part (or none) keeps beyond the edges of its band once it is drawn: about 5 % of the distance, so a camera
// resting at a switch distance does not start a crossfade every frame
#define LOD_HYSTERESIS 1.1

// ResolveLodPart for an instance that drew 'held' last frame (INVALID_PART: nothing): held stays while k is within
// LOD_HYSTERESIS of its band, that is while it is still the part at k × or ÷ LOD_HYSTERESIS. "Nothing" holds the same
// way, so the records of one tree that hand over at one distance (the last leaves LOD, the billboard) keep doing so
// at the same frame. A LOD-managed id and a valid camera (LodFactor > 0) only.
uint ResolveLodPartHeld(uint subbatchId, row_major float4x4 world, float maxScaleSq, uint held)
{
    uint partId = subbatchId & ~LOD_MANAGED_BIT;
    StructuredBuffer<MeshPartLod> lodRegistry = ResourceDescriptorHeap[MeshLodRegistryIdx];
    MeshPartLod lod = lodRegistry[partId];
    float k = LodK(lod, world, maxScaleSq);

    uint part = LodChainPart(lodRegistry, partId, lod, k);
    if (part != held && (LodChainPart(lodRegistry, partId, lod, k * LOD_HYSTERESIS) == held
                      || LodChainPart(lodRegistry, partId, lod, k / LOD_HYSTERESIS) == held))
        return held;
    return part;
}

// Per-instance descriptor (matches C# InstanceDescriptor exactly: 20 bytes = 5 uints)
struct InstanceDescriptor
{
    uint TransformSlot;
    uint MaterialId;
    uint CustomDataIdx;
    uint MeshPartIdx;
    uint BoneBufferIdx;
};

// An instance's world bounding sphere and LOD-resolved part as the camera's CSVisibility found them, for the
// shadow culling (InstanceBatch.instanceBoundsBuffers, 20 bytes). Part = INVALID_PART: not drawn at this LOD.
// Skinned instances carry the radius widened as the shadow passes need it.
struct InstanceBounds
{
    float3 Center;
    float Radius;
    uint Part;
};

void WriteInstanceBounds(uint instanceIdx, float3 center, float radius, uint part)
{
    if (InstanceBoundsUAVIdx == 0)
        return;
    RWStructuredBuffer<InstanceBounds> bounds = ResourceDescriptorHeap[InstanceBoundsUAVIdx];
    InstanceBounds b;
    b.Center = center;
    b.Radius = radius;
    b.Part = part;
    bounds[instanceIdx] = b;
}

// A shadow caster's world sphere and LOD part (returned; INVALID_PART = not drawn at this LOD) from its descriptor,
// transform, LOD chain and mesh registry entry: what CSVisibility does before it writes InstanceBounds.
uint GatherShadowCasterSphere(uint instanceIdx, out float3 worldCenter, out float worldRadius)
{
    worldCenter = 0;
    worldRadius = 0;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufferIdx];
    StructuredBuffer<row_major float4x4> transforms = ResourceDescriptorHeap[GlobalTransformsIdx];
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    StructuredBuffer<uint> subbatchIds = ResourceDescriptorHeap[SubbatchIdsIdx];

    row_major float4x4 world = transforms[descriptors[instanceIdx].TransformSlot];
    float maxScaleSq = max(dot(world[0].xyz, world[0].xyz),
                       max(dot(world[1].xyz, world[1].xyz),
                           dot(world[2].xyz, world[2].xyz)));

    // Same LOD as the main view (resolved from the main camera, not the light)
    uint meshPartId = ResolveLodPart(subbatchIds[instanceIdx], world, maxScaleSq);
    if (meshPartId == INVALID_PART)
        return INVALID_PART;

    float4 localSphere = meshRegistry[meshPartId].LocalBounds;
    worldCenter = mul(float4(localSphere.xyz, 1.0), world).xyz;
    worldRadius = localSphere.w * sqrt(maxScaleSq);
    if (descriptors[instanceIdx].BoneBufferIdx != 0)
        worldRadius *= 1.5; // skinned: the bind-pose bounds do not hold the animated pose
    return meshPartId;
}

// The same from the camera's visibility pass this frame (InstanceBounds, 20 sequential bytes) when slot 38 is set,
// else gathered. The gather cost about as much as the camera's own visibility pass (2026-10-16: 0.22 ms for the
// 76k-instance batch in the point light culling, 85% of it). Both resolve the main camera's LOD and widen skinned
// radii ×1.5, so either way the result is the same.
uint ShadowCasterSphere(uint instanceIdx, out float3 worldCenter, out float worldRadius)
{
    if (InstanceBoundsUAVIdx == 0)
        return GatherShadowCasterSphere(instanceIdx, worldCenter, worldRadius);

    RWStructuredBuffer<InstanceBounds> bounds = ResourceDescriptorHeap[InstanceBoundsUAVIdx];
    InstanceBounds b = bounds[instanceIdx];
    worldCenter = b.Center;
    worldRadius = b.Radius;
    return b.Part;
}

// Must match C# IndirectDrawCommand exactly (24 bytes = 6 uints)
struct IndirectDrawCommand
{
    // Root constants (slots 2-3, written by command signature)
    uint MeshPartId;            // Index into MeshRegistry
    uint InstanceBaseOffset;    // Base offset in sorted indices
    // D3D12_DRAW_INSTANCED_ARGUMENTS
    uint VertexCountPerInstance;
    uint DrawInstanceCount;
    uint StartVertexLocation;
    uint StartInstanceLocation;
};

// The same command as an indexed draw (C# IndexedDrawCommand, 48 bytes; GraphicsDevice.IndexedCommandSignature): the
// mesh's index buffer view, then D3D12_DRAW_INDEXED_ARGUMENTS. The vertex shader gets the vertex index in SV_VertexID.
struct IndexedDrawCommand
{
    uint MeshPartId;
    uint InstanceBaseOffset;
    uint IndexBufferAddressLo;  // D3D12_INDEX_BUFFER_VIEW
    uint IndexBufferAddressHi;
    uint IndexBufferBytes;
    uint IndexFormat;
    uint IndexCountPerInstance; // D3D12_DRAW_INDEXED_ARGUMENTS
    uint DrawInstanceCount;
    uint StartIndexLocation;
    int BaseVertexLocation;
    uint StartInstanceLocation;
    uint Padding;
};

#define DXGI_FORMAT_R32_UINT 42

// A part's draw as an indexed command: its indices from its mesh's index buffer, through the input assembler
IndexedDrawCommand MakeIndexedCommand(uint meshPartId, uint instanceBaseOffset, uint indexCount, uint instanceCount,
                                      MeshPartEntry entry)
{
    IndexedDrawCommand cmd;
    cmd.MeshPartId = meshPartId;
    cmd.InstanceBaseOffset = instanceBaseOffset;
    cmd.IndexBufferAddressLo = entry.IndexBufferAddressLo;
    cmd.IndexBufferAddressHi = entry.IndexBufferAddressHi;
    cmd.IndexBufferBytes = entry.IndexBufferBytes;
    cmd.IndexFormat = DXGI_FORMAT_R32_UINT;
    cmd.IndexCountPerInstance = indexCount;
    cmd.DrawInstanceCount = instanceCount;
    cmd.StartIndexLocation = entry.BaseIndex;
    cmd.BaseVertexLocation = 0;     // the indices are the mesh's own vertex indices
    cmd.StartInstanceLocation = 0;
    cmd.Padding = 0;
    return cmd;
}

//-----------------------------------------------------------------------------
// Frustum culling helper
//-----------------------------------------------------------------------------
bool IsVisible(float3 center, float radius)
{
    float4 planes[6] = { Plane0, Plane1, Plane2, Plane3, Plane4, Plane5 };
    
    for (uint i = 0; i < 6; i++)
    {
        float4 plane = planes[i];
        float dist = dot(plane.xyz, center) + plane.w;
        if (dist > radius)
            return false;
    }
    return true;
}

//-----------------------------------------------------------------------------
// Hi-Z occlusion culling helper
// Projects bounding sphere to screen, picks mip, samples depth pyramid
// Returns true if the sphere is FULLY occluded (behind solid geometry)
// NOTE: Only called for instances that already passed frustum culling
//-----------------------------------------------------------------------------
bool IsOccluded(float3 worldCenter, float worldRadius)
{
    if (HiZSrvIdx == 0) return false; // Hi-Z disabled
    
    // Project sphere center to clip space
    float4 clipCenter = mul(float4(worldCenter, 1.0), OcclusionProjection);
    float3 ndc = clipCenter.xyz / clipCenter.w;
    float2 uv = ndc.xy * float2(0.5, -0.5) + 0.5;
    
    // Off-screen center — can't test, assume visible
    if (any(uv < 0.0) || any(uv > 1.0)) return false;
    
    Texture2D<float> hiZ = ResourceDescriptorHeap[HiZSrvIdx];
    
    // Robustness: Get dimensions directly from accessibility
    float w, h, levels;
    hiZ.GetDimensions(0, w, h, levels);
    float2 mip0Size = float2(w, h);
    
    // Project sphere radius to screen pixels for mip selection.
    // For standard perspective, _m00 * mip0Size.x == _m11 * mip0Size.y,
    // so vertical-only calculation is sufficient.
    float projScale = abs(OcclusionProjection._m11);
    projScale = projScale < 0.001 ? 1.0 : projScale;
    float screenRadius = (worldRadius * projScale / clipCenter.w) * mip0Size.y * 0.5;
    
    // Pick mip where the sphere covers ~1 texel, then add +1 to widen
    // the 4-tap footprint beyond the sphere. This prevents false occlusion
    // when a nearby occluder fills the sphere's footprint but the object
    // extends beyond it (common with large terrain patches behind buildings).
    float mipLevel = ceil(log2(max(screenRadius * 2, 1.0)));
    mipLevel = min(mipLevel, levels - 1.0f);
    
    uint mip = (uint)mipLevel;
    // Must query actual GPU mip dimensions — mip0 / 2^mip diverges from
    // the GPU's floor(prev/2) chain on NPOT textures, causing right-side bias.
    float mipW, mipH, unused;
    hiZ.GetDimensions(mip, mipW, mipH, unused);
    float2 mipSize = float2(mipW, mipH);
    
    // 4-tap sampling for stability
    // Sample the 2x2 neighborhood around the center to catch edges
    float2 texCoordFloat = uv * mipSize - 0.5;
    int2 baseCoord = int2(texCoordFloat);
    
    // Clamp coordinates
    int2 maxCoord = int2(mipSize) - 1;
    
    // Fetch 4 samples
    float d0 = hiZ.Load(int3(clamp(baseCoord, int2(0,0), maxCoord), mip));
    float d1 = hiZ.Load(int3(clamp(baseCoord + int2(1,0), int2(0,0), maxCoord), mip));
    float d2 = hiZ.Load(int3(clamp(baseCoord + int2(0,1), int2(0,0), maxCoord), mip));
    float d3 = hiZ.Load(int3(clamp(baseCoord + int2(1,1), int2(0,0), maxCoord), mip));
    
    // Conservative test: take the MAX depth of the footprint
    // If ANY part of the footprint is "sky" (MaxFloat) or deep, the max will reflect that.
    // For the object to be occluded, it must be behind ALL sampled depths.
    float sampledDepth = max(max(d0, d1), max(d2, d3));
    
    // Conservative test: use the sphere's NEAREST point to the camera
    // Only occlude if even the closest part of the sphere is behind the occluder
    // Both values are view-space Z (clip.w) — linear, good precision at all distances
    float sphereNearestDepth = clipCenter.w - worldRadius;
    return sphereNearestDepth > sampledDepth;
}


//-----------------------------------------------------------------------------
// Shadow cascade frustum culling helper
// Tests a bounding sphere against the specified shadow cascade's frustum
//-----------------------------------------------------------------------------
// Whether a caster is big enough to draw into a cascade: its bounding sphere's diameter covers at least
// ShadowMinCasterClip of the cascade's clip space (EngineSettings.ShadowMinCasterTexels * 2 / resolution).
// Used for the cascades after the first: a small thing far out shades a texel or two, which screen-space
// shadows already give it. Negative: the same test with -ShadowMinCasterClip for every view (the sky captures,
// SkyCapture.cs, whose view is no cascade).
bool CoversShadowTexels(float worldRadius, uint cascadeIndex)
{
    float minClip = abs(ShadowMinCasterClip);
    if (minClip <= 0.0) return true;
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    row_major float4x4 vp = cascades[cascadeIndex].VP;
    float clipPerMetre = length(float3(vp._11, vp._21, vp._31)); // orthographic: the same for every point
    return 2.0 * worldRadius * clipPerMetre >= minClip;
}

bool IsVisibleShadow(float3 center, float radius, uint cascadeIndex)
{
    // Read planes from CascadeData structured buffer (GPU-computed, no CPU readback)
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    
    [unroll]
    for (uint i = 0; i < 6; i++)
    {
        float dist = dot(cascades[cascadeIndex].Planes[i].xyz, center) + cascades[cascadeIndex].Planes[i].w;
        if (dist > radius)
            return false;
    }
    return true;
}

//-----------------------------------------------------------------------------
// Shadow Hi-Z occlusion culling helper
// Projects bounding sphere into previous frame's shadow space for the given
// cascade and checks against the shadow Hi-Z pyramid (min reduction).
// Returns true if the sphere is fully behind existing shadow geometry.
//
// Shadow maps use standard Z: near=0, far=1.0. Pyramid stores min() → nearest.
// Occluded when sphere's farthest depth > min pyramid depth at that location.
//-----------------------------------------------------------------------------
bool IsOccludedShadow(float3 worldCenter, float worldRadius, uint cascadeIdx)
{
    if (ShadowHiZSrvIdx == 0) return false;
    
    // Read previous frame's VP from cascade buffer
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    row_major float4x4 vp = cascades[cascadeIdx].PrevVP;
    
    // Project sphere center into shadow clip space
    float4 clipCenter = mul(float4(worldCenter, 1.0), vp);
    float3 ndc = clipCenter.xyz / clipCenter.w;
    float2 uv = ndc.xy * float2(0.5, -0.5) + 0.5;
    
    // Off-screen → can't test, assume visible
    if (any(uv < 0.0) || any(uv > 1.0)) return false;
    
    // Shadow depth in standard Z [0,1]
    float shadowDepth = ndc.z;
    
    // Sphere's farthest point from light (deeper into shadow map)
    float4 offsetPoint = mul(float4(worldCenter, 1.0) + float4(0, 0, worldRadius, 0), vp);
    float depthExtent = abs(offsetPoint.z / offsetPoint.w - shadowDepth);
    float sphereFarDepth = shadowDepth + depthExtent;
    
    Texture2DArray<float> shadowHiZ = ResourceDescriptorHeap[ShadowHiZSrvIdx];
    
    float w, h, elements, levels;
    shadowHiZ.GetDimensions(0, w, h, elements, levels);
    float2 mip0Size = float2(w, h);
    
    // Orthographic projection: screen-space radius from VP scale
    float projRadiusX = worldRadius * abs(vp._m00);
    float projRadiusY = worldRadius * abs(vp._m11);
    float screenRadius = max(projRadiusX * mip0Size.x, projRadiusY * mip0Size.y) * 0.5;
    
    float mipLevel = ceil(log2(max(screenRadius * 2.0, 1.0)));
    mipLevel = min(mipLevel, levels - 1.0);
    
    uint mip = (uint)mipLevel;
    float2 mipSize = max(float2(1,1), mip0Size / (float)(1u << mip));
    
    // 4-tap neighborhood
    float2 texCoordFloat = uv * mipSize - 0.5;
    int2 baseCoord = int2(texCoordFloat);
    int2 maxCoord = int2(mipSize) - 1;
    
    float d0 = shadowHiZ.Load(int4(clamp(baseCoord,             int2(0,0), maxCoord), cascadeIdx, mip));
    float d1 = shadowHiZ.Load(int4(clamp(baseCoord + int2(1,0), int2(0,0), maxCoord), cascadeIdx, mip));
    float d2 = shadowHiZ.Load(int4(clamp(baseCoord + int2(0,1), int2(0,0), maxCoord), cascadeIdx, mip));
    float d3 = shadowHiZ.Load(int4(clamp(baseCoord + int2(1,1), int2(0,0), maxCoord), cascadeIdx, mip));
    
    float pyramidDepth = max(max(d0, d1), max(d2, d3));
    
    float sphereNearDepth = shadowDepth - depthExtent;
    return sphereNearDepth > pyramidDepth;
}

//-----------------------------------------------------------------------------
// Receiver mask (ShadowReceivers.cs, slot 39): whether a caster can darken anything the camera sees through this
// cascade. Under its footprint, widened by the shadow filter's reach, some receiver must lie deeper (farther from the
// sun) than the caster's nearest point; a tile without receivers holds 0. The footprint is read from the pyramid level
// whose cells are at least as wide as it, so it touches at most 2x2 of them.
//-----------------------------------------------------------------------------
bool ShadesReceivers(RWStructuredBuffer<uint> receivers, float3 worldCenter, float worldRadius, uint cascadeIdx, float marginUV)
{
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    row_major float4x4 vp = cascades[cascadeIdx].VP;

    float4 clip = mul(float4(worldCenter, 1.0), vp);    // orthographic: w = 1
    float casterNear = clip.z - worldRadius * length(float3(vp._13, vp._23, vp._33));

    float2 uv = clip.xy * float2(0.5, -0.5) + 0.5;
    float radiusUV = 0.5 * worldRadius * length(float3(vp._11, vp._21, vp._31)) + marginUV;
    float2 lo = saturate(uv - radiusUV) * RECEIVER_TILES;
    float2 hi = saturate(uv + radiusUV) * RECEIVER_TILES;

    float extent = max(hi.x - lo.x, hi.y - lo.y);
    uint level = min((uint)ceil(log2(max(extent, 1.0))), RECEIVER_LEVELS - 1u);
    uint size = RECEIVER_TILES >> level;
    uint2 a = min((uint2)lo >> level, size - 1u);
    uint2 b = min((uint2)hi >> level, size - 1u);

    uint base = ReceiverLevelBase(cascadeIdx, level);
    uint deepest = max(max(receivers[base + a.y * size + a.x], receivers[base + a.y * size + b.x]),
                       max(receivers[base + b.y * size + a.x], receivers[base + b.y * size + b.x]));
    return deepest != 0 && casterNear <= asfloat(deepest);
}

// Adds this wave's cascade entries, and those shading no receiver, to the receiver stats (entries and triangles)
void CountReceiverEntries(RWStructuredBuffer<uint> receivers, uint cascadeIdx, bool entry, bool shades, uint triangles)
{
    bool cut = entry && !shades;
    uint entries = WaveActiveCountBits(entry);
    uint entriesCut = WaveActiveCountBits(cut);
    uint tris = WaveActiveSum(entry ? triangles : 0u);
    uint trisCut = WaveActiveSum(cut ? triangles : 0u);
    if (WaveIsFirstLane() && entries > 0)
    {
        InterlockedAdd(receivers[RECEIVER_ENTRIES + cascadeIdx], entries);
        InterlockedAdd(receivers[RECEIVER_ENTRIES_CUT + cascadeIdx], entriesCut);
        InterlockedAdd(receivers[RECEIVER_TRIS + cascadeIdx], tris);
        InterlockedAdd(receivers[RECEIVER_TRIS_CUT + cascadeIdx], trisCut);
    }
}

//-----------------------------------------------------------------------------
// The camera's transform cache (InstanceBatch.visibilityCacheBuffer; CSVisibility, slots 24/28/29/39). The transforms,
// descriptors and subbatch ids live in upload heaps, so gathering them for every instance every frame read ~88 bytes
// per instance across PCIe: all of CSVisibility's cost (2026-10-16, Eldermoor's 163k instances: 0.40-0.61 ms; with the
// cache 0.18 ms while it still read the 24 bytes of descriptor and subbatch id to check each entry, 0.031-0.034 ms now).
// The cache keeps what the gather found per instance, in video memory and instance order, and nothing but the cache
// is read for an instance unless the C# or the transform upload says it changed:
//   - its transform slot was uploaded this tick (ChangedSlotsIdx: TransformBuffer's bit per slot, 0 = none was),
//   - its persistent record changed since the last culling (ChangedRecordsIdx: the batch's bit per instance, 0 = none:
//     added, swap-removed into this index, skinned or no longer),
//   - it is at or past CacheRefreshFrom: this frame's enqueued draws (PersistentCount), or every instance (0) when the
//     cache is new, the batch missed a tick or a compute shader wrote its records.
// Everything after the gather runs as before on the same values, so the flags and InstanceBounds come out identical.
//-----------------------------------------------------------------------------
#define VISIBILITY_CACHE_WRITTEN     1u     // Flags: 0 = never written (a new buffer is zeroed)
#define VISIBILITY_CACHE_SKINNED     2u

struct VisibilityCacheEntry     // 64 bytes (InstanceBatch.VisibilityCacheStride)
{
    float3 Row0;                // world[0..3].xyz: the affine rows; mul(float4(p, 1), world).xyz reads nothing else
    uint TransformSlot;
    float3 Row1;
    float MaxScaleSq;
    float3 Row2;
    uint SubbatchId;            // as staged: chain head part | LOD_MANAGED_BIT
    float3 Row3;
    uint Flags;
};

// One bit of a bit-per-index buffer (bitsIdx = 0: no bit is set)
bool BitSet(uint bitsIdx, uint index)
{
    if (bitsIdx == 0)
        return false;
    StructuredBuffer<uint> bits = ResourceDescriptorHeap[bitsIdx];
    return ((bits[index >> 5] >> (index & 31u)) & 1u) != 0;
}

// What CSVisibility needs of an instance besides the mesh registry: its world matrix, largest axis scale squared,
// staged subbatch id and whether it is skinned. From the transform cache when slot 28 is set (gathered and stored when
// stale), else gathered through the descriptor's transform slot as before.
void CameraCullInput(uint instanceIdx, out row_major float4x4 world, out float maxScaleSq, out uint subbatchId, out bool skinned)
{
    if (VisibilityCacheIdx != 0 && instanceIdx < CacheRefreshFrom)
    {
        RWStructuredBuffer<VisibilityCacheEntry> cache = ResourceDescriptorHeap[VisibilityCacheIdx];
        VisibilityCacheEntry entry = cache[instanceIdx];
        if (entry.Flags != 0 && !BitSet(ChangedRecordsIdx, instanceIdx) && !BitSet(ChangedSlotsIdx, entry.TransformSlot))
        {
            world = float4x4(float4(entry.Row0, 0.0), float4(entry.Row1, 0.0), float4(entry.Row2, 0.0), float4(entry.Row3, 1.0));
            maxScaleSq = entry.MaxScaleSq;
            subbatchId = entry.SubbatchId;
            skinned = (entry.Flags & VISIBILITY_CACHE_SKINNED) != 0;
            return;
        }
    }

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufferIdx];
    StructuredBuffer<row_major float4x4> transforms = ResourceDescriptorHeap[GlobalTransformsIdx];
    StructuredBuffer<uint> subbatchIds = ResourceDescriptorHeap[SubbatchIdsIdx];

    // Data is stored in add-order, instanceIdx == drawIdx (identity mapping)
    InstanceDescriptor descriptor = descriptors[instanceIdx];
    world = transforms[descriptor.TransformSlot];
    subbatchId = subbatchIds[instanceIdx];   // the meshPartId set in CommandBuffer.Enqueue, LOD-managed or not
    skinned = descriptor.BoneBufferIdx != 0;

    // Largest axis scale, squared. The LOD distance uses it as is; only the radius needs the sqrt.
    maxScaleSq = max(dot(world[0].xyz, world[0].xyz),
                 max(dot(world[1].xyz, world[1].xyz),
                     dot(world[2].xyz, world[2].xyz)));

    if (VisibilityCacheIdx != 0)
    {
        RWStructuredBuffer<VisibilityCacheEntry> cache = ResourceDescriptorHeap[VisibilityCacheIdx];
        VisibilityCacheEntry entry;
        entry.Row0 = world[0].xyz;
        entry.TransformSlot = descriptor.TransformSlot;
        entry.Row1 = world[1].xyz;
        entry.MaxScaleSq = maxScaleSq;
        entry.Row2 = world[2].xyz;
        entry.SubbatchId = subbatchId;
        entry.Row3 = world[3].xyz;
        entry.Flags = VISIBILITY_CACHE_WRITTEN | (skinned ? VISIBILITY_CACHE_SKINNED : 0u);
        cache[instanceIdx] = entry;
    }
}

//-----------------------------------------------------------------------------
// LOD crossfade (the camera chain, LodFadeStateIdx != 0: batches of effects that declare LodFade). A persistent
// instance whose part changes draws both parts for 1 / LodFadeRate seconds: the new one on a share of the pixels that
// grows from nothing to all, the old one on the rest (LodFadeClip in common.fx: the same blue noise for both, so they
// cover each pixel once). CSVisibility steps the state; the compaction (CSHistogram, CSGlobalScatter) adds an entry for
// the part faded from, and tags both entries with the share (lod_fade.hlsli). The shadows draw one part and switch at
// the middle of the fade (InstanceBounds.Part).
// The state is kept per instance in video memory (one buffer, written by the GPU only, in queue order), and starts
// over when the record changed (ChangedRecordsIdx: a swap-remove moved another instance into the index) or was never
// written (a new buffer is zeroed).
//-----------------------------------------------------------------------------
#define LOD_FADE_WRITTEN    0x80000000u
#define LOD_FADE_PART_MASK  0x00FFFFFFu

struct LodFadeState     // 16 bytes (InstanceBatch.LodFadeStride)
{
    uint Part;          // the part drawn + 1 (0 = none) | LOD_FADE_WRITTEN
    uint From;          // the part it fades from + 1 (0 = none); Part's when no fade runs
    float Start;        // LodTime the fade began
    uint Word;          // this frame's fade for the compaction: the share (1..127) << 24 | the part faded from + 1 when
                        // the camera sees it; 0 = no fade
};

// The fade word CSVisibility left for an instance this frame (0 = no fade)
uint LodFadeWord(uint instanceIdx)
{
    if (LodFadeStateIdx == 0 || instanceIdx >= LodFadeFrom)
        return 0;
    RWStructuredBuffer<LodFadeState> states = ResourceDescriptorHeap[LodFadeStateIdx];
    return states[instanceIdx].Word;
}

// The instance's part this frame by its fade state, which it steps: a switch (with hysteresis) starts a fade, none
// starts on a new record or a camera cut (LodFadeRate 0), and a fade that has run its time ends. fromPart and share:
// the part faded from (INVALID_PART: nothing) and the new part's share in 128ths (0 = no fade runs).
uint StepLodFade(inout LodFadeState state, uint instanceIdx, uint subbatchId, row_major float4x4 world, float maxScaleSq,
                 out uint fromPart, out uint share)
{
    bool fresh = (state.Part & LOD_FADE_WRITTEN) == 0 || BitSet(ChangedRecordsIdx, instanceIdx);
    uint was = state.Part & ~LOD_FADE_WRITTEN;
    uint part;
    if (fresh)
        part = ResolveLodPart(subbatchId, world, maxScaleSq);
    else
        part = ResolveLodPartHeld(subbatchId, world, maxScaleSq, was - 1);
    uint code = part + 1;   // INVALID_PART + 1 = 0: none

    if (fresh || LodFadeRate <= 0.0)
        state.From = code;
    else if (code != was)
    {
        // From the part drawn so far. A fade still running ends there; one that turns back (to the part it came
        // from) carries on from the share it had, the other way round.
        float back = code == state.From ? 1.0 - saturate((LodTime - state.Start) * LodFadeRate) : 0.0;
        state.From = was;
        state.Start = LodTime - back / LodFadeRate;
    }
    state.Part = code | LOD_FADE_WRITTEN;

    float fade = saturate((LodTime - state.Start) * LodFadeRate);
    if (fade >= 1.0)
        state.From = code;

    fromPart = state.From - 1;
    share = state.From != code ? clamp((uint)(fade * 128.0 + 0.5), 1u, 127u) : 0u;
    return part;
}

void StoreLodFade(uint instanceIdx, LodFadeState state, uint word)
{
    RWStructuredBuffer<LodFadeState> states = ResourceDescriptorHeap[LodFadeStateIdx];
    state.Word = word;
    states[instanceIdx] = state;
}

void ClearLodFadeWord(uint instanceIdx)
{
    RWStructuredBuffer<LodFadeState> states = ResourceDescriptorHeap[LodFadeStateIdx];
    states[instanceIdx].Word = 0;
}

//-----------------------------------------------------------------------------
// Pass 1: CSVisibility - Write visibility flags (1 if visible, 0 if not)
// Also counts visible instances per subbatch for command generation
//-----------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void CSVisibility(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint instanceIdx = dispatchThreadId.x;

    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    RWStructuredBuffer<uint> visibilityFlags = ResourceDescriptorHeap[VisibilityFlagsIdx];

    // Bounds check - write 0 for out-of-range to clear stale visibility flags
    if (instanceIdx >= TotalInstances)
    {
        visibilityFlags[instanceIdx] = 0;
        return;
    }

    row_major float4x4 world;
    float maxScaleSq;
    uint subbatchId;
    bool skinned;
    CameraCullInput(instanceIdx, world, maxScaleSq, subbatchId, skinned);

    // The part to draw, and while a crossfade runs the part it fades from and the new part's share (0 = no fade).
    // Without a valid camera the state waits: ResolveLodPart draws nothing then.
    uint subBatch, fromPart = INVALID_PART, share = 0;
    bool fades = LodFadeStateIdx != 0 && instanceIdx < LodFadeFrom && (subbatchId & LOD_MANAGED_BIT) != 0 && LodFactor > 0.0;
    LodFadeState fadeState = (LodFadeState)0;
    if (fades)
    {
        RWStructuredBuffer<LodFadeState> fadeStates = ResourceDescriptorHeap[LodFadeStateIdx];
        fadeState = fadeStates[instanceIdx];
        subBatch = StepLodFade(fadeState, instanceIdx, subbatchId, world, maxScaleSq, fromPart, share);
    }
    else
    {
        subBatch = ResolveLodPart(subbatchId, world, maxScaleSq);
        // The compaction reads every persistent record's fade word: clear the one a LOD-managed record left in this
        // index before another took its place, and all of them while there is no camera (the states wait)
        if (LodFadeStateIdx != 0 && instanceIdx < LodFadeFrom && (LodFactor <= 0.0 || BitSet(ChangedRecordsIdx, instanceIdx)))
            ClearLodFadeWord(instanceIdx);
    }

    // The camera tests the sphere of the part drawn, or of the part fading out when there is none
    uint testPart = subBatch != INVALID_PART ? subBatch : fromPart;
    if (testPart == INVALID_PART)
    {
        visibilityFlags[instanceIdx] = 0;
        WriteInstanceBounds(instanceIdx, 0, 0, INVALID_PART);
        if (fades)
            StoreLodFade(instanceIdx, fadeState, 0);
        return;
    }

    // Look up local bounding sphere from mesh registry (persistent, not per-instance)
    float4 localSphere = meshRegistry[testPart].LocalBounds;

    // Transform sphere to world space
    float3 worldCenter = mul(float4(localSphere.xyz, 1.0), world).xyz;
    float worldRadius = localSphere.w * sqrt(maxScaleSq);

    // For the shadow culling, which draws one part: the one faded from until the middle of a crossfade.
    // Skinned: the bind-pose bounds do not hold the animated pose.
    uint shadowPart = share != 0 && share < 64 ? fromPart : subBatch;
    if (shadowPart == testPart)
        WriteInstanceBounds(instanceIdx, worldCenter, skinned ? worldRadius * 1.5 : worldRadius, shadowPart);
    else if (shadowPart == INVALID_PART)
        WriteInstanceBounds(instanceIdx, 0, 0, INVALID_PART);
    else
    {
        float4 shadowSphere = meshRegistry[shadowPart].LocalBounds;
        float shadowRadius = shadowSphere.w * sqrt(maxScaleSq);
        WriteInstanceBounds(instanceIdx, mul(float4(shadowSphere.xyz, 1.0), world).xyz,
                            skinned ? shadowRadius * 1.5 : shadowRadius, shadowPart);
    }

    // Frustum test — save result for stats (avoid redundant re-evaluation)
    bool frustumVisible = IsVisible(worldCenter, worldRadius);
    bool hiZOccluded = false;
    
    // Hi-Z occlusion test (only for frustum-visible instances)
    // Small distance-proportional radius floor prevents false positives when
    // per-part AABB centers fall in gaps of open geometry (fences, bridges).
    if (frustumVisible)
    {
        float4 hiZClip = mul(float4(worldCenter, 1.0), OcclusionProjection);
        float hiZRadius = max(worldRadius, hiZClip.w * 0.01);
        hiZOccluded = IsOccluded(worldCenter, hiZRadius);
    }
    
    bool visible = frustumVisible && !hiZOccluded;

    // Write visibility flag: 0 = culled, otherwise resolved MeshPartId + 1 (INVALID_PART + 1 = 0: only fading out)
    visibilityFlags[instanceIdx] = visible ? (subBatch + 1u) : 0u;

    // A mirrored instance in view keeps its parts on the pulled draw (MirrorPartsIdx): every part it draws, both of a
    // crossfade
    if (visible && MirrorPartsIdx != 0 && determinant((float3x3)world) < 0.0)
    {
        RWStructuredBuffer<uint> mirrorParts = ResourceDescriptorHeap[MirrorPartsIdx];
        if (subBatch != INVALID_PART)
            mirrorParts[subBatch] = MirrorCull;
        if (share != 0 && fromPart != INVALID_PART)
            mirrorParts[fromPart] = MirrorCull;
    }

    // And the crossfade's word for the compaction
    if (fades)
        StoreLodFade(instanceIdx, fadeState, share != 0 ? (share << 24) | (visible ? fromPart + 1u : 0u) : 0u);

    // Update cull stats — wave-aggregated to reduce atomic contention
    if (CullStatsUAVIdx != 0)
    {
        RWByteAddressBuffer stats = ResourceDescriptorHeap[CullStatsUAVIdx];
        
        // Count matching threads in this wave, only lane 0 does the atomic
        uint visibleCount = WaveActiveCountBits(visible);
        uint occludedCount = WaveActiveCountBits(frustumVisible && hiZOccluded);
        
        if (WaveIsFirstLane())
        {
            if (visibleCount > 0)
                stats.InterlockedAdd(0, visibleCount);    // [0] = frustum+HiZ visible
            if (occludedCount > 0)
                stats.InterlockedAdd(4, occludedCount);   // [1] = passed frustum, failed Hi-Z
        }
    }
}

//-----------------------------------------------------------------------------
// Pass 1b: CSHistogram - Count instances per MeshPartId
// Runs once per instance, atomically increments histogram[meshPartId]
// Result: histogram[k] = count of visible instances with MeshPartId == k
//-----------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CSHistogram(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint instanceIdx = dispatchThreadId.x;
    if (instanceIdx >= TotalInstances)
        return;
    
    // Read visibility flag from the visibility pass (0 = culled, else LOD-resolved MeshPartId + 1)
    RWStructuredBuffer<uint> visibilityFlags = ResourceDescriptorHeap[VisibilityFlagsIdx];
    uint flag = visibilityFlags[instanceIdx];
    // A crossfade's part faded from + 1, when the camera sees it (0 = none)
    uint from = LodFadeWord(instanceIdx) & LOD_FADE_PART_MASK;

    // Atomically increment histogram counter for this MeshPartId
    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramIdx];
    if (flag != 0)
        InterlockedAdd(histogram[flag - 1], 1);
    if (from != 0)
        InterlockedAdd(histogram[from - 1], 1);
}

//-----------------------------------------------------------------------------
// Pass 2: CSLocalScan - Local prefix sum within each block
// Uses Blelloch scan algorithm. Outputs block sums for hierarchical propagation.
// Each block processes 512 elements (256 threads, 2 elements each)
//-----------------------------------------------------------------------------
#define BLOCK_SIZE 512
groupshared uint sharedData[BLOCK_SIZE];

[numthreads(256, 1, 1)]
void CSLocalScan(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    uint tid = groupThreadId.x;
    uint blockOffset = groupId.x * BLOCK_SIZE;
    
    RWStructuredBuffer<uint> flags = ResourceDescriptorHeap[VisibilityFlagsIdx];
    RWStructuredBuffer<uint> prefixSum = ResourceDescriptorHeap[PrefixSumIdx];
    RWStructuredBuffer<uint> blockSums = ResourceDescriptorHeap[BlockSumsIdx];
    
    // Load two elements per thread
    uint idx1 = blockOffset + tid;
    uint idx2 = blockOffset + tid + 256;
    
    // Flags hold MeshPartId + 1, not 1: count them, don't sum them
    sharedData[tid] = (idx1 < TotalInstances && flags[idx1] != 0) ? 1 : 0;
    sharedData[tid + 256] = (idx2 < TotalInstances && flags[idx2] != 0) ? 1 : 0;
    
    // Up-sweep (reduce) phase
    uint offset = 1;
    for (uint d = BLOCK_SIZE >> 1; d > 0; d >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (tid < d)
        {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            sharedData[bi] += sharedData[ai];
        }
        offset <<= 1;
    }
    
    // Save block sum and clear last element for exclusive scan
    if (tid == 0)
    {
        blockSums[groupId.x] = sharedData[BLOCK_SIZE - 1];
        sharedData[BLOCK_SIZE - 1] = 0;
    }
    
    // Down-sweep phase
    for (uint d2 = 1; d2 < BLOCK_SIZE; d2 <<= 1)
    {
        offset >>= 1;
        GroupMemoryBarrierWithGroupSync();
        if (tid < d2)
        {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            uint t = sharedData[ai];
            sharedData[ai] = sharedData[bi];
            sharedData[bi] += t;
        }
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    // Write local prefix sum results
    if (idx1 < TotalInstances)
        prefixSum[idx1] = sharedData[tid];
    if (idx2 < TotalInstances)
        prefixSum[idx2] = sharedData[tid + 256];
}

//-----------------------------------------------------------------------------
// Pass 3: CSBlockScan - Scan the block sums (single workgroup for simplicity)
// Assumes <= 512 blocks (supports up to 512 * 512 = 262144 instances)
//-----------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void CSBlockScan(uint3 groupThreadId : SV_GroupThreadID)
{
    uint tid = groupThreadId.x;
    
    RWStructuredBuffer<uint> blockSums = ResourceDescriptorHeap[BlockSumsIdx];
    
    // Load block sums (2 per thread)
    sharedData[tid] = (tid < NumBlocks) ? blockSums[tid] : 0;
    sharedData[tid + 256] = ((tid + 256) < NumBlocks) ? blockSums[tid + 256] : 0;
    
    // Up-sweep
    uint offset = 1;
    for (uint d = BLOCK_SIZE >> 1; d > 0; d >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (tid < d)
        {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            sharedData[bi] += sharedData[ai];
        }
        offset <<= 1;
    }
    
    if (tid == 0)
        sharedData[BLOCK_SIZE - 1] = 0;
    
    // Down-sweep
    for (uint d2 = 1; d2 < BLOCK_SIZE; d2 <<= 1)
    {
        offset >>= 1;
        GroupMemoryBarrierWithGroupSync();
        if (tid < d2)
        {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            uint t = sharedData[ai];
            sharedData[ai] = sharedData[bi];
            sharedData[bi] += t;
        }
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    // Write scanned block sums back
    if (tid < NumBlocks)
        blockSums[tid] = sharedData[tid];
    if ((tid + 256) < NumBlocks)
        blockSums[tid + 256] = sharedData[tid + 256];
}

//-----------------------------------------------------------------------------
// Pass 4: CSGlobalScatter - Scatter visible instances to per-MeshPartId compacted output
// Uses histogram prefix sum (counters buffer) for base offsets
// Atomically increments counters to get unique output positions
// MUST run AFTER CSHistogramPrefixSum so counters contains StartInstance offsets
//
// OUTPUT: one uint per visible instance = original instanceIdx
// The VS uses this index to look up all per-instance data from input buffers.
//-----------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void CSGlobalScatter(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupId : SV_GroupID)
{
    uint instanceIdx = dispatchThreadId.x;
    if (instanceIdx >= TotalInstances)
        return;
    
    RWStructuredBuffer<uint> visibilityFlags = ResourceDescriptorHeap[VisibilityFlagsIdx];
    RWStructuredBuffer<uint> counters = ResourceDescriptorHeap[CounterBufferIdx];
    RWStructuredBuffer<uint> visibleOut = ResourceDescriptorHeap[VisibleIndicesUAVIdx];

    // 0 = culled, else LOD-resolved MeshPartId + 1
    uint flag = visibilityFlags[instanceIdx];

    // A crossfading instance has an entry for each part, both tagged with the new part's share (lod_fade.hlsli)
    uint word = LodFadeWord(instanceIdx);
    uint entry = word != 0 ? instanceIdx | DRAW_FADE_BIT | ((word >> 24) << DRAW_FADE_SHIFT) : instanceIdx;

    uint outputIdx;
    if (flag >= 1)
    {
        uint meshPartId = flag - 1;
        InterlockedAdd(counters[meshPartId], 1, outputIdx);
        visibleOut[outputIdx] = entry;
    }

    uint from = word & LOD_FADE_PART_MASK;
    if (from != 0)
    {
        InterlockedAdd(counters[from - 1], 1, outputIdx);
        visibleOut[outputIdx] = entry | DRAW_FADE_OUT_BIT;
    }
}

//-----------------------------------------------------------------------------
// Pass 4b: CSSortIndirection - Sort indirection buffer per-subbatch by TransformSlot
// Uses bitonic sort. Ensures deterministic order regardless of CPU push order.
// Called multiple times with different SortSize/SortStride parameters for full sort.
// SubbatchStart/SubbatchCount define the range to sort within.
//
// BITONIC SORT ALGORITHM:
// - For each merge phase (blockSize = 2, 4, 8, ..., N):
//   - For each stride (stride = blockSize/2, blockSize/4, ..., 1):
//     - Compare pairs of elements 'stride' apart
//     - Direction alternates based on position within block to create bitonic sequence
//-----------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void CSSortIndirection(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    
    RWStructuredBuffer<uint> indirection = ResourceDescriptorHeap[IndirectionIdx];
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufferIdx];
    
    uint stride = SortStride;     // Distance between elements to compare
    uint blockSize = SortSize;    // Size of current bitonic block
    
    // Each thread compares one pair of elements
    // Thread idx compares elements at positions:
    //   left  = floor(idx / stride) * stride * 2 + idx % stride
    //   right = left + stride
    uint leftRelative = ((idx / stride) * stride * 2) + (idx % stride);
    uint rightRelative = leftRelative + stride;
    
    // Check bounds for each element
    bool leftValid = leftRelative < SortCount;
    bool rightValid = rightRelative < SortCount;
    
    // If BOTH are out of bounds, nothing to do
    if (!leftValid && !rightValid)
        return;
    
    // Add subbatch offset for actual buffer access
    uint leftIdx = SubbatchStart + leftRelative;
    uint rightIdx = SubbatchStart + rightRelative;
    
    // Determine sort direction within bitonic block:
    // - First half of each bitonic block: ascending
    // - Second half: descending
    bool ascending = ((leftRelative / blockSize) % 2) == 0;
    
    // Read indirection values - use MAX_UINT for out-of-bounds
    uint leftIndir = leftValid ? indirection[leftIdx] : 0xFFFFFFFF;
    uint rightIndir = rightValid ? indirection[rightIdx] : 0xFFFFFFFF;
    
    // Get sort keys (TransformSlot from descriptor) - stable keys assigned at entity creation
    // Use infinity (MAX_UINT) for out-of-bounds elements so they sort to the end
    uint leftKey = (leftValid && leftIndir != 0xFFFFFFFF) ? descriptors[leftIndir].TransformSlot : 0xFFFFFFFF;
    uint rightKey = (rightValid && rightIndir != 0xFFFFFFFF) ? descriptors[rightIndir].TransformSlot : 0xFFFFFFFF;
    
    // Compare and swap based on direction
    // ascending=true: we want leftKey <= rightKey, so swap if leftKey > rightKey
    // ascending=false: we want leftKey >= rightKey, so swap if leftKey < rightKey
    bool shouldSwap = ascending ? (leftKey > rightKey) : (leftKey < rightKey);
    
    // Only perform swap if BOTH elements are within bounds
    // (if one is out of bounds, the infinity key ensures no swap is needed)
    if (shouldSwap && leftValid && rightValid)
    {
        indirection[leftIdx] = rightIndir;
        indirection[rightIdx] = leftIndir;
    }
}

//-----------------------------------------------------------------------------
// Pass 5: CSBitonicSort - Sort visible indices by value for deterministic order
// This pass is called multiple times with different parameters for full sort.
// Uses SortSize, SortStride, SortCount from Indices[5] (defined above).
//-----------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void CSBitonicSort(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    
    RWStructuredBuffer<uint> data = ResourceDescriptorHeap[VisibleIndicesUAVIdx];
    
    // Each thread handles one comparison
    uint halfSize = SortStride;
    uint blockSize = SortSize;
    
    // Calculate pair indices for this thread
    uint blockIdx = idx / halfSize;
    uint offsetInBlock = idx % halfSize;
    
    uint leftIdx = blockIdx * blockSize + offsetInBlock;
    uint rightIdx = leftIdx + halfSize;
    
    // Bounds check
    if (rightIdx >= SortCount)
        return;
    
    // Determine sort direction (ascending for even blocks, descending for odd)
    bool ascending = ((leftIdx / blockSize) % 2) == 0;
    
    uint left = data[leftIdx];
    uint right = data[rightIdx];
    
    // Compare and swap if needed
    if ((left > right) == ascending)
    {
        data[leftIdx] = right;
        data[rightIdx] = left;
    }
}

//-----------------------------------------------------------------------------
// Pass 5a: CSHistogramPrefixSum - Parallel prefix sum of histogram
// 3-phase scan: (1) serial chunk prefix, (2) Blelloch scan of partials,
// (3) propagate block offsets. Supports up to MaxMeshParts (4096).
// Input: histogram[k] = visible count for MeshPartId k
// Output: counters[k] = exclusive prefix sum = StartInstance offset
//-----------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void CSHistogramPrefixSum(uint3 groupThreadId : SV_GroupThreadID)
{
    uint tid = groupThreadId.x;
    
    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramIdx];
    RWStructuredBuffer<uint> counters = ResourceDescriptorHeap[CounterBufferIdx];
    
    // Phase 1: Each thread serially prefix-sums its chunk
    uint elemsPerThread = (UniquePartCount + 255) / 256;
    uint myStart = tid * elemsPerThread;
    uint myEnd = min(myStart + elemsPerThread, UniquePartCount);
    
    uint localSum = 0;
    for (uint i = myStart; i < myEnd; i++)
    {
        uint count = histogram[i];
        counters[i] = localSum;
        localSum += count;
    }
    
    // Phase 2: Blelloch exclusive prefix sum of 256 partial sums
    sharedData[tid] = localSum;
    sharedData[tid + 256] = 0;
    
    uint offset = 1;
    for (uint d = BLOCK_SIZE >> 1; d > 0; d >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (tid < d)
        {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            sharedData[bi] += sharedData[ai];
        }
        offset <<= 1;
    }
    
    if (tid == 0) sharedData[BLOCK_SIZE - 1] = 0;
    
    for (uint d2 = 1; d2 < BLOCK_SIZE; d2 <<= 1)
    {
        offset >>= 1;
        GroupMemoryBarrierWithGroupSync();
        if (tid < d2)
        {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            uint t = sharedData[ai];
            sharedData[ai] = sharedData[bi];
            sharedData[bi] += t;
        }
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    // Phase 3: Add block prefix to each element
    uint blockPrefix = sharedData[tid];
    for (uint j = myStart; j < myEnd; j++)
    {
        counters[j] += blockPrefix;
    }
}

//-----------------------------------------------------------------------------
// Pass 5b: CSMain - Generate final indirect draw commands from MeshRegistry
// Fully GPU-driven: iterates over MeshRegistry, reads counts from histogram.
// Two modes:
//   DrawCountUAVIdx != 0: Packs non-empty commands to front via atomic append
//                         for count-buffer-driven ExecuteIndirect (opaque path).
//   DrawCountUAVIdx == 0: Fixed-index writes at commands[meshPartId] with zeroed
//                         empty commands (shadow path — downstream passes read by index).
//-----------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint meshPartId = dispatchThreadId.x;
    if (meshPartId >= UniquePartCount)
        return;
    
    // Read visible count and offset from histogram/prefix-sum
    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramIdx];
    RWStructuredBuffer<uint> counters = ResourceDescriptorHeap[CounterBufferIdx];
    
    uint visibleCount = histogram[meshPartId];
    
    RWStructuredBuffer<IndirectDrawCommand> output = ResourceDescriptorHeap[OutputBufferIdx];
    
    // Skip MeshPartIds with no visible instances
    if (visibleCount == 0)
    {
        // Fixed-index mode: write zeroed command (shadow pipeline reads by index)
        if (DrawCountUAVIdx == 0)
        {
            IndirectDrawCommand emptyCmd = (IndirectDrawCommand)0;
            output[meshPartId] = emptyCmd;
        }
        return;
    }
    
    // After CSGlobalScatter, counters[k] has been incremented by visibleCount
    // Original StartInstance = counters[k] - visibleCount
    uint counterVal = counters[meshPartId];
    uint baseOffset = (counterVal >= visibleCount) ? (counterVal - visibleCount) : 0;
    
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry entry = meshRegistry[meshPartId];

    // The camera chain's indexed draws (IndexedCommandsIdx): every part without a mirrored instance in view, packed
    // and counted in the draw count's [1]; the rest stay pulled
    if (DrawCountUAVIdx != 0 && IndexedCommandsIdx != 0)
    {
        bool mirrored = false;
        if (MirrorPartsIdx != 0)
        {
            RWStructuredBuffer<uint> mirrorParts = ResourceDescriptorHeap[MirrorPartsIdx];
            mirrored = mirrorParts[meshPartId] == MirrorCull;
        }
        if (!mirrored)
        {
            RWStructuredBuffer<uint> indexedCount = ResourceDescriptorHeap[DrawCountUAVIdx];
            uint indexedSlot;
            InterlockedAdd(indexedCount[1], 1, indexedSlot);
            RWStructuredBuffer<IndexedDrawCommand> indexedOut = ResourceDescriptorHeap[IndexedCommandsIdx];
            indexedOut[indexedSlot] = MakeIndexedCommand(meshPartId, baseOffset, entry.VertexCount,
                visibleCount * max(1u, InstanceMultiplier), entry);
            return;
        }
    }

    // Determine output slot
    uint outputSlot;
    if (DrawCountUAVIdx != 0)
    {
        // Atomic append: pack non-empty commands at the front of the buffer
        RWStructuredBuffer<uint> drawCount = ResourceDescriptorHeap[DrawCountUAVIdx];
        InterlockedAdd(drawCount[0], 1, outputSlot);
    }
    else
    {
        // Fixed-index: shadow pipeline reads commands[meshPartId]
        outputSlot = meshPartId;
    }
    
    IndirectDrawCommand cmd;
    cmd.MeshPartId = meshPartId;
    cmd.InstanceBaseOffset = baseOffset;
    cmd.VertexCountPerInstance = entry.VertexCount;
    cmd.DrawInstanceCount = visibleCount * max(1u, InstanceMultiplier);
    cmd.StartVertexLocation = 0;
    cmd.StartInstanceLocation = 0;
    
    output[outputSlot] = cmd;
}

//-----------------------------------------------------------------------------
// CSClear: Zero out counter buffer and histogram
//-----------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CSClear(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    
    RWStructuredBuffer<uint> counters = ResourceDescriptorHeap[CounterBufferIdx];
    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramIdx];
    
    if (idx < SubBatchCount)
        counters[idx] = 0;
    
    if (idx < UniquePartCount)
        histogram[idx] = 0;
}

//-----------------------------------------------------------------------------
// CSVisibilityShadow: Write visibility flags for shadow cascade culling
// Similar to CSVisibility but tests against ShadowCascadePlanes instead
// Uses ShadowCascadeIdx to select which cascade's frustum to test against
//-----------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void CSVisibilityShadow(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint instanceIdx = dispatchThreadId.x;
    
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufferIdx];
    StructuredBuffer<row_major float4x4> transforms = ResourceDescriptorHeap[GlobalTransformsIdx];
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    StructuredBuffer<uint> subbatchIds = ResourceDescriptorHeap[SubbatchIdsIdx];
    RWStructuredBuffer<uint> visibilityFlags = ResourceDescriptorHeap[VisibilityFlagsIdx];
    
    // Bounds check - write 0 for out-of-range
    if (instanceIdx >= TotalInstances)
    {
        visibilityFlags[instanceIdx] = 0;
        return;
    }
    
    // Get transform
    uint transformSlot = descriptors[instanceIdx].TransformSlot;
    row_major float4x4 world = transforms[transformSlot];

    // Largest axis scale, squared. The LOD distance uses it as is; only the radius needs the sqrt.
    float maxScaleSq = max(dot(world[0].xyz, world[0].xyz),
                       max(dot(world[1].xyz, world[1].xyz),
                           dot(world[2].xyz, world[2].xyz)));

    // Same LOD as the main view (resolved from the main camera, not the light)
    uint meshPartId = ResolveLodPart(subbatchIds[instanceIdx], world, maxScaleSq);
    if (meshPartId == INVALID_PART)
    {
        visibilityFlags[instanceIdx] = 0;
        return;
    }

    // Bounding sphere from mesh registry, transformed to world space
    float4 localSphere = meshRegistry[meshPartId].LocalBounds;
    float3 worldCenter = mul(float4(localSphere.xyz, 1.0), world).xyz;
    float worldRadius = localSphere.w * sqrt(maxScaleSq);

    // For skinned meshes (BoneBufferIdx != 0), the bounding sphere is static (bind pose)
    // and doesn't account for animation. Inflate it to prevent culling when limbs move outside.
    if (descriptors[instanceIdx].BoneBufferIdx != 0)
    {
        worldRadius *= 1.5;
    }

    // Frustum test against selected shadow cascade
    bool visible = IsVisibleShadow(worldCenter, worldRadius, ShadowCascadeIdx);

    // Write visibility flag: 0 = culled, otherwise resolved MeshPartId + 1
    visibilityFlags[instanceIdx] = visible ? (meshPartId + 1u) : 0u;
}

//-----------------------------------------------------------------------------
// Shadow LOD per cascade (EngineSettings.ShadowLodPerCascade, DirectionalLight.UploadCascadeBuffer). A cascade draws a
// caster at the detail the camera would draw it at from the distance where one screen pixel spans one of the
// cascade's texels: its SplitDistances.z holds that distance squared (0 = the camera's part, as the sky captures and
// the cover map keep). Never finer than the camera's part and never culled (the chain's last part at most). The steps
// along the part's LOD chain sit beside the cascade bits in the cascade mask word, 3 bits per cascade.
//-----------------------------------------------------------------------------
#define CASCADE_MASK_BITS   0xFFu       // bit c = cascade c (SHADOW_MAX_CASCADES of them)
#define CASCADE_LOD_SHIFT   8u          // then cascade c's LOD steps at CASCADE_LOD_SHIFT + 3c

// How many steps along its LOD chain from 'part' cascade c takes, for a caster whose largest axis scale is maxScaleSq
// squared
uint CascadeLodSteps(StructuredBuffer<CascadeData> cascades, StructuredBuffer<MeshPartLod> lodRegistry, uint part,
                     float maxScaleSq, uint c)
{
    float distanceSq = cascades[c].SplitDistances.z;
    if (distanceSq <= 0.0 || LodFactor <= 0.0)
        return 0;
    float k = distanceSq / max(LodFactor * maxScaleSq, 1e-12);   // LodK at that distance

    uint steps = 0;
    for (; steps < 7; steps++)
    {
        MeshPartLod lod = lodRegistry[part];
        if (lod.FarK == 0.0 || k < lod.FarK || lod.NextPartId == 0)
            break;
        part = lod.NextPartId - 1;
    }
    return steps;
}

// The part cascade c draws: the camera's 'part' and the steps the cascade mask word holds for c
uint CascadePart(StructuredBuffer<MeshPartLod> lodRegistry, uint part, uint maskWord, uint c)
{
    uint steps = (maskWord >> (CASCADE_LOD_SHIFT + 3u * c)) & 7u;
    for (uint i = 0; i < steps; i++)
        part = lodRegistry[part].NextPartId - 1;
    return part;
}

// Adds this wave's instances to its batch's row of caster stats (EngineSettings.ShadowCasterStats): how many cascades
// each is drawn into, and the triangles per cascade. 'mask' is the cascades it is drawn into (after the receiver test),
// 'inBox' whether any cascade's box held it.
void CountCasterStats(RWStructuredBuffer<uint> receivers, uint row, bool inBox, uint mask, uint part, uint lodWord,
                      StructuredBuffer<MeshPartEntry> meshRegistry, StructuredBuffer<MeshPartLod> lodRegistry, uint cascadeCount)
{
    uint base = RECEIVER_CASTER_BASE + min(row, RECEIVER_CASTER_ROWS - 1u) * RECEIVER_CASTER_WORDS;

    uint total = 0;
    uint once = 0;
    for (uint c = 0; c < cascadeCount; c++)
    {
        bool entry = (mask & (1u << c)) != 0;
        uint tris = entry ? meshRegistry[CascadePart(lodRegistry, part, lodWord, c)].VertexCount / 3u : 0u;
        if (entry && total == 0)
            once = tris;
        total += tris;
        uint entries = WaveActiveCountBits(entry);
        uint waveTris = WaveActiveSum(tris);
        if (WaveIsFirstLane() && entries > 0)
        {
            InterlockedAdd(receivers[base + CASTER_ENTRIES + c], entries);
            InterlockedAdd(receivers[base + CASTER_TRIS + c], waveTris);
        }
    }

    uint drawnIn = countbits(mask & CASCADE_MASK_BITS);
    for (uint n = 0; n <= cascadeCount; n++)
    {
        bool here = inBox && drawnIn == n;
        uint count = WaveActiveCountBits(here);
        uint waveTris = WaveActiveSum(here ? total : 0u);
        if (WaveIsFirstLane() && count > 0)
        {
            InterlockedAdd(receivers[base + CASTER_BY_COUNT + n], count);
            InterlockedAdd(receivers[base + CASTER_TRIS_BY_COUNT + n], waveTris);
        }
    }

    uint waveOnce = WaveActiveSum(once);
    if (WaveIsFirstLane() && waveOnce > 0)
        InterlockedAdd(receivers[base + CASTER_TRIS_ONCE], waveOnce);
}

//-----------------------------------------------------------------------------
// CSVisibilityCascades: every cascade's visibility of an instance in one dispatch (ShadowCascadeCount of them,
// slot 22: the sun's cascades in use this frame, or the sky captures' one view).
// Outputs:
//   - Combined visibility flag (the camera's part + 1 when any cascade draws it)
//   - Cascade mask word per instance: bit c = cascade c, and each cascade's LOD steps (CascadeLodSteps), which
//     CSHistogramCascades / CSScatterCascades turn into the parts' entries
//-----------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void CSVisibilityCascades(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint instanceIdx = dispatchThreadId.x;
    
    // Output buffers
    RWStructuredBuffer<uint> combinedVis = ResourceDescriptorHeap[CombinedShadowVisIdx];
    RWStructuredBuffer<uint> cascadeMasks = ResourceDescriptorHeap[CascadeMaskUAVIdx];
    
    // Bounds check - write 0 for out-of-range
    if (instanceIdx >= TotalInstances)
    {
        combinedVis[instanceIdx] = 0;
        cascadeMasks[instanceIdx] = 0;
        return;
    }
    
    // World sphere and LOD part, found once for all cascades (the camera's InstanceBounds when slot 38 is set)
    float3 worldCenter;
    float worldRadius;
    uint meshPartId = ShadowCasterSphere(instanceIdx, worldCenter, worldRadius);
    if (meshPartId == INVALID_PART)
    {
        combinedVis[instanceIdx] = 0;
        cascadeMasks[instanceIdx] = 0;
        return;
    }

    // Every cascade's frustum + shadow Hi-Z occlusion; past the first cascade, small casters are left out
    uint cascadeCount = ShadowCascadeCount;
    uint mask = 0;
    for (uint c = 0; c < cascadeCount; c++)
    {
        bool big = (c == 0 && ShadowMinCasterClip >= 0.0) || CoversShadowTexels(worldRadius, c);
        if (big && IsVisibleShadow(worldCenter, worldRadius, c) && !IsOccludedShadow(worldCenter, worldRadius, c))
            mask |= 1u << c;
    }

    // Each cascade's part (CascadeLodSteps): the camera's part, or coarser where the cascade's texels are coarser than
    // the camera's pixels on it. The scale from the sphere: the part's radius times it (skinned: 1.5 times more, which
    // only takes fewer steps).
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    StructuredBuffer<MeshPartLod> lodRegistry = ResourceDescriptorHeap[MeshLodRegistryIdx];
    StructuredBuffer<CascadeData> cascadeData = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    uint lodWord = 0;
    if (mask != 0 && MeshLodRegistryIdx != 0 && lodRegistry[meshPartId].FarK > 0.0)
    {
        float localRadius = meshRegistry[meshPartId].LocalBounds.w;
        float scale = localRadius > 0.0 ? worldRadius / localRadius : 1.0;
        for (uint c = 0; c < cascadeCount; c++)
            if ((mask & (1u << c)) != 0)
                lodWord |= CascadeLodSteps(cascadeData, lodRegistry, meshPartId, scale * scale, c) << (CASCADE_LOD_SHIFT + 3u * c);
    }

    // Receiver mask (slot 39): the entries whose caster shades nothing the camera sees through that cascade are
    // counted (RECEIVER_MODE_COUNT), and left out with RECEIVER_MODE_CULL
    if (ShadowReceiversUAVIdx != 0)
    {
        RWStructuredBuffer<uint> receivers = ResourceDescriptorHeap[ShadowReceiversUAVIdx];
        float marginUV = asfloat(receivers[RECEIVER_MARGIN]);

        uint shades = 0;
        for (uint c = 0; c < min(cascadeCount, RECEIVER_MAX_CASCADES); c++)
        {
            bool entry = (mask & (1u << c)) != 0;
            bool shadesHere = entry && ShadesReceivers(receivers, worldCenter, worldRadius, c, marginUV);
            // The triangles of the part the cascade draws (VertexCount = indices)
            uint triangles = entry ? meshRegistry[CascadePart(lodRegistry, meshPartId, lodWord, c)].VertexCount / 3u : 0u;
            CountReceiverEntries(receivers, c, entry, shadesHere, triangles);
            if (shadesHere)
                shades |= 1u << c;
        }

        bool inAny = mask != 0;
        bool cutFromAll = inAny && shades == 0;
        uint instances = WaveActiveCountBits(inAny);
        uint instancesCut = WaveActiveCountBits(cutFromAll);
        if (WaveIsFirstLane() && instances > 0)
        {
            InterlockedAdd(receivers[RECEIVER_INSTANCES], instances);
            InterlockedAdd(receivers[RECEIVER_INSTANCES_CUT], instancesCut);
        }

        if (receivers[RECEIVER_MODE] == RECEIVER_MODE_CULL)
            mask = shades;

        if (CasterStatsRow != 0)
            CountCasterStats(receivers, CasterStatsRow - 1u, inAny, mask, meshPartId, lodWord, meshRegistry, lodRegistry,
                             min(cascadeCount, RECEIVER_MAX_CASCADES));
    }

    // Combined visibility: 0 = in no cascade, otherwise the camera's MeshPartId + 1
    combinedVis[instanceIdx] = mask != 0 ? (meshPartId + 1u) : 0u;
    cascadeMasks[instanceIdx] = mask | lodWord;
}

//-----------------------------------------------------------------------------
// The cascade chain's compaction, as the point light chain's: an entry per (instance, cascade), counted under the
// part that cascade draws (CascadePart: the camera's, or coarser with ShadowLodPerCascade), so each part's entries
// sit together whatever cascade they are for. CSHistogramCascades counts them, CSHistogramPrefixSum places the parts,
// CSScatterCascades writes the entries (cascade and instance as shadow_cascades.hlsli packs them) straight into the
// expansion buffer (capacity x SHADOW_MAX_CASCADES entries: every instance in every cascade fits), and CSMain makes
// the commands at fixed part indices, base and count already in entries. Consecutive cascades of an instance that
// draw the same part (the steps only grow with the cascade) take one atomic.
//
// Input push constants: VisibilityFlagsIdx (the camera's part + 1, CSVisibilityCascades' combined flag),
// CascadeMaskUAVIdx (the cascade mask words), HistogramIdx, CounterBufferIdx, ExpansionUAVIdx, MeshLodRegistryIdx,
// TotalInstances.
//-----------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CSHistogramCascades(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint instanceIdx = dispatchThreadId.x;
    if (instanceIdx >= TotalInstances)
        return;

    RWStructuredBuffer<uint> visibilityFlags = ResourceDescriptorHeap[VisibilityFlagsIdx];
    uint flag = visibilityFlags[instanceIdx];
    if (flag == 0)
        return;

    RWStructuredBuffer<uint> cascadeMasks = ResourceDescriptorHeap[CascadeMaskUAVIdx];
    uint word = cascadeMasks[instanceIdx];
    StructuredBuffer<MeshPartLod> lodRegistry = ResourceDescriptorHeap[MeshLodRegistryIdx];
    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramIdx];

    uint mask = word & CASCADE_MASK_BITS;
    uint runPart = INVALID_PART, run = 0;
    while (mask != 0)
    {
        uint c = firstbitlow(mask);
        mask &= mask - 1;
        uint part = CascadePart(lodRegistry, flag - 1, word, c);
        if (part != runPart)
        {
            if (run != 0)
                InterlockedAdd(histogram[runPart], run);
            runPart = part;
            run = 0;
        }
        run++;
    }
    if (run != 0)
        InterlockedAdd(histogram[runPart], run);
}

[numthreads(256, 1, 1)]
void CSScatterCascades(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint instanceIdx = dispatchThreadId.x;
    if (instanceIdx >= TotalInstances)
        return;

    RWStructuredBuffer<uint> visibilityFlags = ResourceDescriptorHeap[VisibilityFlagsIdx];
    uint flag = visibilityFlags[instanceIdx];
    if (flag == 0)
        return;

    RWStructuredBuffer<uint> cascadeMasks = ResourceDescriptorHeap[CascadeMaskUAVIdx];
    uint word = cascadeMasks[instanceIdx];
    StructuredBuffer<MeshPartLod> lodRegistry = ResourceDescriptorHeap[MeshLodRegistryIdx];
    RWStructuredBuffer<uint> counters = ResourceDescriptorHeap[CounterBufferIdx];
    RWStructuredBuffer<uint> expansion = ResourceDescriptorHeap[ExpansionUAVIdx];
    uint instanceBits = instanceIdx & CASCADE_ENTRY_INSTANCE_MASK;

    uint mask = word & CASCADE_MASK_BITS;
    while (mask != 0)
    {
        // The run of cascades from the lowest left that draw its part
        uint part = CascadePart(lodRegistry, flag - 1, word, firstbitlow(mask));
        uint runBits = 0;
        uint rest = mask;
        while (rest != 0)
        {
            uint c = firstbitlow(rest);
            if (CascadePart(lodRegistry, flag - 1, word, c) != part)
                break;
            runBits |= 1u << c;
            rest &= rest - 1;
        }
        mask &= ~runBits;

        uint pos;
        InterlockedAdd(counters[part], countbits(runBits), pos);
        while (runBits != 0)
        {
            uint c = firstbitlow(runBits);
            runBits &= runBits - 1;
            expansion[pos++] = (c << CASCADE_ENTRY_SHIFT) | instanceBits;
        }
    }
}

//-----------------------------------------------------------------------------
// CSCompactCommands: Pack non-empty draw commands from source to destination.
// Runs after CSMain's fixed-index commands (shadow pipelines). Reads commands from
// OutputBufferIdx (source), atomically appends non-empty ones to a SEPARATE
// destination buffer (VisibleIndicesSRVIdx repurposed as dest UAV for this pass),
// and writes total count to DrawCountUAV for ExecuteIndirect's count buffer.
// Must use separate source/dest to avoid read-write race conditions.
//-----------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CSCompactCommands(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint meshPartId = dispatchThreadId.x;
    if (meshPartId >= UniquePartCount)
        return;
    
    RWStructuredBuffer<IndirectDrawCommand> source = ResourceDescriptorHeap[OutputBufferIdx];
    
    IndirectDrawCommand cmd = source[meshPartId];
    if (cmd.DrawInstanceCount == 0)
        return;
    
    RWStructuredBuffer<uint> drawCount = ResourceDescriptorHeap[DrawCountUAVIdx];
    uint slot;
    InterlockedAdd(drawCount[0], 1, slot);
    
    RWStructuredBuffer<IndirectDrawCommand> dest = ResourceDescriptorHeap[VisibleIndicesSRVIdx];
    dest[slot] = cmd;

    // A batch that draws indexed (InstanceBatch.DrawsIndexed) gets the same command as an indexed draw at the same slot.
    // The shadow vertex shaders never swap corners, so a mirrored instance needs no pulled draw here.
    if (IndexedCommandsIdx != 0)
    {
        StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
        RWStructuredBuffer<IndexedDrawCommand> indexedDest = ResourceDescriptorHeap[IndexedCommandsIdx];
        indexedDest[slot] = MakeIndexedCommand(cmd.MeshPartId, cmd.InstanceBaseOffset, cmd.VertexCountPerInstance,
            cmd.DrawInstanceCount, meshRegistry[cmd.MeshPartId]);
    }
}

//-----------------------------------------------------------------------------
// Point light shadows (LocalShadows.cs, InstanceBatch.CullLocalShadowAll)
//
// The cascade pipeline with three differences:
//  - views are point light cube faces, grouped by light: up to 128, so the per-instance view mask is
//    a uint4 (LocalShadowMaskIdx, a buffer of the local pass's own);
//  - the histogram counts an instance once per view it is in, so the prefix sum sizes the expansion
//    exactly (the cascades reserve SHADOW_MAX_CASCADES entries per instance, which a cube face set can exceed);
//  - the scatter writes the expansion entries itself; CSMain then makes commands whose base and count
//    are already in expansion entries (no patch pass).
// Order: CSVisibilityLocalShadows → CSHistogramLocalShadows → CSHistogramPrefixSum →
//        CSScatterLocalShadows → CSMain (fixed index) → CSCompactCommands.
//
// Push constants beyond the cascade path's, on slots that path leaves unused:
//-----------------------------------------------------------------------------
#define LocalShadowViewsIdx          CascadeBufferSRVIdx   // 31: SRV LocalShadowView[]
#define LocalShadowLightsIdx         SubbatchStartIdx      // 16: SRV LocalShadowLight[]
#define LocalShadowCountsIdx         SubbatchCountIdx      // 17: UAV the selection's counts, [0] = lights (local_shadow_select.hlsl)
#define LocalShadowMaskIdx           CascadeMaskUAVIdx     // 28: UAV uint4 view mask per instance
#define LocalShadowExpansionCapacity IndirectionUAVIdx     // 19: entries the expansion buffer holds

uint ViewCount(uint4 mask)
{
    return countbits(mask.x) + countbits(mask.y) + countbits(mask.z) + countbits(mask.w);
}

bool InLocalShadowView(LocalShadowView v, float3 center, float radius)
{
    [unroll]
    for (uint i = 0; i < 6; i++)
    {
        if (dot(v.Planes[i].xyz, center) + v.Planes[i].w > radius)
            return false;
    }
    return true;
}

[numthreads(256, 1, 1)]
void CSVisibilityLocalShadows(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint instanceIdx = dispatchThreadId.x;
    if (instanceIdx >= TotalInstances)
        return;

    RWStructuredBuffer<uint> combinedVis = ResourceDescriptorHeap[CombinedShadowVisIdx];
    RWStructuredBuffer<uint4> masks = ResourceDescriptorHeap[LocalShadowMaskIdx];

    // The sphere and LOD part (the camera's InstanceBounds when slot 38 is set)
    float3 worldCenter;
    float worldRadius;
    uint meshPartId = ShadowCasterSphere(instanceIdx, worldCenter, worldRadius);
    if (meshPartId == INVALID_PART)
    {
        combinedVis[instanceIdx] = 0;
        masks[instanceIdx] = 0;
        return;
    }

    StructuredBuffer<LocalShadowLight> lights = ResourceDescriptorHeap[LocalShadowLightsIdx];
    StructuredBuffer<LocalShadowView> views = ResourceDescriptorHeap[LocalShadowViewsIdx];

    RWStructuredBuffer<uint> shadowCounts = ResourceDescriptorHeap[LocalShadowCountsIdx];
    uint lightCount = shadowCounts[0];

    uint4 mask = 0;
    for (uint l = 0; l < lightCount; l++)
    {
        LocalShadowLight light = lights[l];

        // Out of the light's reach: none of its faces
        float3 d = worldCenter - light.Position;
        float reach = light.Range + worldRadius;
        if (dot(d, d) > reach * reach)
            continue;

        for (uint f = 0; f < 6; f++)
        {
            uint v = light.FaceView[f];
            if (v >= LOCAL_SHADOW_MAX_VIEWS || !InLocalShadowView(views[v], worldCenter, worldRadius))
                continue;
            uint bit = 1u << (v & 31u);
            uint word = v >> 5;
            if (word == 0)      mask.x |= bit;
            else if (word == 1) mask.y |= bit;
            else if (word == 2) mask.z |= bit;
            else                mask.w |= bit;
        }
    }

    // 0 = culled, otherwise the LOD-resolved MeshPartId + 1 (as the other visibility kernels)
    combinedVis[instanceIdx] = any(mask != 0) ? (meshPartId + 1u) : 0u;
    masks[instanceIdx] = mask;
}

// histogram[part] += the number of views each visible instance is in
[numthreads(64, 1, 1)]
void CSHistogramLocalShadows(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint instanceIdx = dispatchThreadId.x;
    if (instanceIdx >= TotalInstances)
        return;

    RWStructuredBuffer<uint> visibilityFlags = ResourceDescriptorHeap[VisibilityFlagsIdx];
    uint flag = visibilityFlags[instanceIdx];
    if (flag == 0)
        return;

    RWStructuredBuffer<uint4> masks = ResourceDescriptorHeap[LocalShadowMaskIdx];
    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramIdx];
    InterlockedAdd(histogram[flag - 1], ViewCount(masks[instanceIdx]));
}

// After CSHistogramPrefixSum: counters[part] = the part's first expansion entry. Appends one entry per view
// (view << 25 | instance) and leaves counters[part] = start + count, which is what CSMain expects.
[numthreads(256, 1, 1)]
void CSScatterLocalShadows(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint instanceIdx = dispatchThreadId.x;
    if (instanceIdx >= TotalInstances)
        return;

    RWStructuredBuffer<uint> visibilityFlags = ResourceDescriptorHeap[VisibilityFlagsIdx];
    uint flag = visibilityFlags[instanceIdx];
    if (flag == 0)
        return;

    RWStructuredBuffer<uint4> masks = ResourceDescriptorHeap[LocalShadowMaskIdx];
    uint4 mask = masks[instanceIdx];

    RWStructuredBuffer<uint> counters = ResourceDescriptorHeap[CounterBufferIdx];
    uint pos;
    InterlockedAdd(counters[flag - 1], ViewCount(mask), pos);

    // Past the capacity an entry is dropped. The draw still counts it and reads 0 there (instance 0 in
    // view 0): it needs the batch's instances in more than 4 faces each on average.
    RWStructuredBuffer<uint> expansion = ResourceDescriptorHeap[ExpansionUAVIdx];
    uint instanceBits = instanceIdx & LOCAL_SHADOW_INSTANCE_MASK;
    [unroll]
    for (uint word = 0; word < 4; word++)
    {
        uint bits = mask[word];
        while (bits != 0)
        {
            uint v = firstbitlow(bits) + word * 32;
            bits &= bits - 1;
            if (pos < LocalShadowExpansionCapacity)
                expansion[pos] = (v << LOCAL_SHADOW_VIEW_SHIFT) | instanceBits;
            pos++;
        }
    }
}

//-----------------------------------------------------------------------------
// GPU Depth Sorting: Sort visible instances by camera distance
// Runs AFTER CSGlobalScatter, BEFORE CSMain.
// Three-pass pipeline:
//   1. CSComputeDepthKeys: Pack (compositeKey, instanceIdx) into uint2 buffer
//      compositeKey = (meshPartId << 20) | (quantizedDistance & 0xFFFFF)
//   2. CSBitonicSortPairs: Bitonic sort on uint2 by .x key (ascending)
//   3. CSWriteSortedIndices: Unpack sorted .y values back to visibleIndices
//
// Sort direction is baked into the key:
//   front-to-back (opaque):      distBits >> 12 (ascending = nearest first)
//   back-to-front (transparent): ~(distBits >> 12) (ascending = farthest first)
//
// The meshPartId in the upper bits preserves subbatch grouping automatically.
//-----------------------------------------------------------------------------

[numthreads(256, 1, 1)]
void CSComputeDepthKeys(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    
    RWStructuredBuffer<uint> visibleIndices = ResourceDescriptorHeap[VisibleIndicesUAVIdx];
    RWStructuredBuffer<uint2> sortKeys = ResourceDescriptorHeap[SortKeysUAVIdx];
    
    if (idx >= TotalInstances)
    {
        sortKeys[idx] = uint2(0xFFFFFFFF, 0xFFFFFFFF);
        return;
    }
    
    uint entry = visibleIndices[idx];

    // Padding entries (0xFFFFFFFF) sort to end
    if (entry == 0xFFFFFFFF)
    {
        sortKeys[idx] = uint2(0xFFFFFFFF, 0xFFFFFFFF);
        return;
    }

    // A crossfade tags the entries (lod_fade.hlsli); they keep the tags through the sort
    uint instanceIdx = LodFadeStateIdx != 0 ? DrawEntryIndex(entry) : entry;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufferIdx];
    StructuredBuffer<row_major float4x4> transforms = ResourceDescriptorHeap[GlobalTransformsIdx];
    RWStructuredBuffer<uint> visibilityFlags = ResourceDescriptorHeap[VisibilityFlagsIdx];

    uint transformSlot = descriptors[instanceIdx].TransformSlot;
    row_major float4x4 world = transforms[transformSlot];
    
    // Extract world position from translation row
    float3 worldPos = float3(world[3][0], world[3][1], world[3][2]);
    float dist = distance(worldPos, CameraPosition);
    
    // asuint(float) sorts correctly for positive floats (IEEE 754)
    // Top 20 bits give ~1mm precision at 1m, ~2m at 2km — plenty for draw ordering
    uint distBits = asuint(dist);
    uint distQuantized = distBits >> 12;
    
    // Back-to-front: invert distance bits so ascending sort = farthest first
    if (SortDirection == 1)
        distQuantized = (~distQuantized) & 0xFFFFF;
    
    // LOD-resolved MeshPartId + 1 (visible instances only reach here, so the flag is non-zero); the entry of a
    // crossfade's part faded from: that part's
    uint meshPartId = (LodFadeStateIdx != 0 && (entry & DRAW_FADE_OUT_BIT) != 0)
        ? (LodFadeWord(instanceIdx) & LOD_FADE_PART_MASK) - 1
        : visibilityFlags[instanceIdx] - 1;

    // Composite key: meshPartId in upper 12 bits, distance in lower 20 bits
    uint compositeKey = (meshPartId << 20) | (distQuantized & 0xFFFFF);
    sortKeys[idx] = uint2(compositeKey, entry);
}

//-----------------------------------------------------------------------------
// CSBitonicSortPairs: Sort uint2 entries by .x key (ascending)
// Called multiple times with different SortSize/SortStride for full sort.
// SortSize = outer block size (determines ascending/descending direction)
// SortStride = current comparison stride (determines pair spacing)
//-----------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void CSBitonicSortPairs(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    
    RWStructuredBuffer<uint2> data = ResourceDescriptorHeap[SortKeysUAVIdx];
    
    uint halfSize = SortStride;
    uint pairBlock = halfSize * 2; // Local pair block size (NOT SortSize)
    
    // Calculate pair indices: pairs are spaced by halfSize within blocks of pairBlock
    uint blockIdx = idx / halfSize;
    uint offsetInBlock = idx % halfSize;
    
    uint leftIdx = blockIdx * pairBlock + offsetInBlock;
    uint rightIdx = leftIdx + halfSize;
    
    if (rightIdx >= SortCount)
        return;
    
    // Direction based on OUTER block size (SortSize), not local pair block
    bool ascending = ((leftIdx / SortSize) % 2) == 0;
    
    uint2 left = data[leftIdx];
    uint2 right = data[rightIdx];
    
    if ((left.x > right.x) == ascending)
    {
        data[leftIdx] = right;
        data[rightIdx] = left;
    }
}

//-----------------------------------------------------------------------------
// CSWriteSortedIndices: Write sorted instance indices back to visibleIndices
//-----------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void CSWriteSortedIndices(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    if (idx >= TotalInstances)
        return;
    
    RWStructuredBuffer<uint2> sortKeys = ResourceDescriptorHeap[SortKeysUAVIdx];
    RWStructuredBuffer<uint> visibleIndices = ResourceDescriptorHeap[VisibleIndicesUAVIdx];
    
    visibleIndices[idx] = sortKeys[idx].y;
}
