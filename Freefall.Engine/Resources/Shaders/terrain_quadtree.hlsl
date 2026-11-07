// GPU-Driven Quadtree Terrain — Restricted Quadtree, walked from the root (TERRAIN_PAGES_PLAN.md stage 2, 2026-11-07)
// Pass 1 (CSWalk): one group walks the tree level by level, testing only the children of nodes that split, so the
//   cost follows what is drawn and not the size of the terrain; then applies the neighbour rule (restricted quadtree:
//   ≤1 level difference) and walks again by the final flags into a list of leaves.
// Pass 2 (CSEmitLeaves): the leaves in view, with inline Hi-Z occlusion culling, as patches.
// Pass 3 (CSBuildDrawArgs): Single-thread pass that reads the append counter and writes
//   DrawInstanced indirect arguments for ExecuteIndirect.
// The shadow pass (CSEmitLeavesShadow) takes the same leaves, per cascade.
// Until this date every pass swept every node of the tree (CSMarkSplits): 21,845 at depth 7, 1.4 million at depth 10.
// The walk drew the same patches at depths 7, 8 and 10; on an RTX 4080 it costs 0.054 ms at depth 10 against the
// sweep's 0.044 (the sweep's cost hardly grows with the nodes there), and is kept because its cost follows what is
// drawn, which a smaller GPU and a deeper tree feel, and because stage 4's split by residency fits a walk from the root.

// Must match Terrain.TerrainPatchData in C#
struct TerrainPatchData
{
    float4 Rect;        // (minU, minV, maxU, maxV) in [0,1] terrain-space
    float2 Level;       // (lodLevel, 0)
    float2 Padding;
};

// Must match InstanceDescriptor in C#
struct InstanceDescriptor
{
    uint TransformSlot;
    uint MaterialId;
    uint CustomDataIdx;
    uint MeshPartIdx;
    uint BoneBufferIdx;
};

#pragma kernel CSWalk
#pragma kernel CSEmitLeaves
#pragma kernel CSBuildMinMaxMip
#pragma kernel CSBuildDrawArgs
#pragma kernel CSEmitLeavesShadow

// Push constants (root parameter 0, register b3) — bindless indices only
cbuffer PushConstants : register(b3)
{
    uint OutputDescriptorsUAVIdx;   // slot 0  — UAV: InstanceDescriptor buffer
    uint OutputSpheresUAVIdx;       // slot 1  — UAV: BoundingSphere buffer
    uint OutputSubbatchIdsUAVIdx;   // slot 2  — UAV: subbatch ID buffer
    uint OutputTerrainDataUAVIdx;   // slot 3  — UAV: TerrainPatchData buffer
    uint CounterUAVIdx;             // slot 4  — UAV: atomic counter
    uint CascadeIdxUAVIdx;          // slot 5  — UAV: cascade index buffer (shadow)
    uint SplitFlagsUAVIdx;          // slot 6  — UAV: split flags buffer
    uint HeightRangeSRVIdx;         // slot 7  — SRV: height range mip pyramid
    uint BuildMipIdx;               // slot 8  — uint: current mip for CSBuildMinMaxMip
    uint MipInputSRVIdx;            // slot 9  — SRV: previous mip (CSBuildMinMaxMip)
    uint MipOutputUAVIdx;           // slot 10 — UAV: current mip output (CSBuildMinMaxMip)
    uint IndirectArgsUAVIdx;        // slot 11 — UAV: indirect draw args
    uint VertexCountIdx;            // slot 12 — uint: mesh index count (CSBuildDrawArgs)
    uint CascadeCountIdx;           // slot 13 — uint: number of cascades (shadow)
    uint CascadeBufferSRVIdx;       // slot 14 — SRV: StructuredBuffer<CascadeData> (shadow)
    uint WalkUAVIdx;                // slot 15 — UAV: the walk's lists and its leaves (CSWalk)
};

// ── Frustum planes (root slot 1, register b0) ──
cbuffer FrustumPlanes : register(b0)
{
    float4 Planes[6];
};

// ── Hi-Z occlusion parameters (root slot 2, register b1) ──
cbuffer HiZParams : register(b1)
{
    row_major float4x4 OcclusionProjection;
    uint HiZSrvIdx;
    float2 HiZSize;
    uint HiZMipCount;
    float NearPlane;
    uint CullStatsUAVIdx;
    uint FrustumDebugMode;
    float _hiZPad;
};

// ── Terrain parameters (root slot 3, register b2) ──
cbuffer TerrainParams : register(b2)
{
    float3 CameraPos;      float MaxHeight;
    float3 RootCenter;     uint  MaxDepth;
    float3 RootExtents;    uint  TotalNodes;
    float2 TerrainSize;    float PixelErrorThreshold;  float ScreenHeight;
    float  TanHalfFov;     uint  TransformSlot;         uint  MaterialId;     uint  MeshPartId;
    uint   MaxPatches;     uint  TerrainPagesIdx;        uint  FrameTag;    uint WalkCapacity;   // FrameTag: MarkSplit; TerrainPagesIdx: terrain_pages.hlsli
    uint   ShadowReceiversIdx;  uint3 _terrainPad2;     // shadow pass: this frame's sun shadow receivers, 0 = no test
};

#include "shadow_receivers.hlsli"
#include "terrain_pages.hlsli"

// Sampler for height texture reads
SamplerState sampHeightCS : register(s2);

// UAVs accessed via ResourceDescriptorHeap (bindless)
// Output buffers are written via atomic append

// ─────────────────────────────────────────────────────────────────────
// Frustum + Hi-Z occlusion helpers (adapted from cull_instances.hlsl)
// Frustum test: sphere vs 6 frustum planes.
// ─────────────────────────────────────────────────────────────────────
bool IsFrustumVisible(float3 center, float radius)
{
    for (uint i = 0; i < 6; i++)
    {
        float dist = dot(Planes[i].xyz, center) + Planes[i].w;
        if (dist > radius)
            return false;
    }
    return true;
}

// Hi-Z occlusion test: project bounding sphere, pick mip, sample depth pyramid.
// Returns true if the sphere is FULLY behind solid geometry (should be culled).
bool IsTerrainOccluded(float3 worldCenter, float worldRadius)
{
    if (HiZSrvIdx == 0) return false; // Hi-Z disabled

    // Project sphere center to clip space using previous-frame VP
    float4 clipCenter = mul(float4(worldCenter, 1.0), OcclusionProjection);
    float3 ndc = clipCenter.xyz / clipCenter.w;
    float2 uv = ndc.xy * float2(0.5, -0.5) + 0.5;

    // Off-screen center — can't test, assume visible
    if (any(uv < 0.0) || any(uv > 1.0)) return false;

    Texture2D<float> hiZ = ResourceDescriptorHeap[HiZSrvIdx];

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
    // the 4-tap footprint beyond the sphere. Prevents false occlusion
    // when a nearby occluder fills the sphere's footprint.
    float mipLevel = ceil(log2(max(screenRadius * 2, 1.0)));
    mipLevel = min(mipLevel, levels - 1.0f);

    uint mip = (uint)mipLevel;
    float mipW, mipH, unused;
    hiZ.GetDimensions(mip, mipW, mipH, unused);
    float2 mipSize = float2(mipW, mipH);

    // 4-tap sampling for stability
    float2 texCoordFloat = uv * mipSize - 0.5;
    int2 baseCoord = int2(texCoordFloat);
    int2 maxCoord = int2(mipSize) - 1;

    float d0 = hiZ.Load(int3(clamp(baseCoord,              int2(0,0), maxCoord), mip));
    float d1 = hiZ.Load(int3(clamp(baseCoord + int2(1,0),  int2(0,0), maxCoord), mip));
    float d2 = hiZ.Load(int3(clamp(baseCoord + int2(0,1),  int2(0,0), maxCoord), mip));
    float d3 = hiZ.Load(int3(clamp(baseCoord + int2(1,1),  int2(0,0), maxCoord), mip));

    // Conservative: take MAX depth of footprint (farthest view-space Z)
    // View-space Z (Position.w) is NOT affected by reverse-Z — larger Z = farther.
    float sampledDepth = max(max(d0, d1), max(d2, d3));

    // Use sphere's nearest point to camera (clip.w - radius)
    float sphereNearestDepth = clipCenter.w - worldRadius;
    return sphereNearestDepth > sampledDepth; // sphere farther than farthest depth → occluded
}

// Compute the flat index where a given depth level starts in the complete quadtree.
// Level L starts at (4^L - 1) / 3
uint LevelStart(uint depth)
{
    // 4^depth via bit shift: 1 << (2*depth)
    return ((1u << (2u * depth)) - 1u) / 3u;
}


// Given a depth and local index, compute the node's center and extents in terrain space.
// The local index encodes the quadrant path as pairs of 2 bits.
// At each level, the quadrant is: 0=NW(-x,-z), 1=NE(+x,-z), 2=SW(-x,+z), 3=SE(+x,+z)
void ComputeNodeBounds(uint depth, uint localIdx, float3 rootCenter, float3 rootExtents,
                       out float3 center, out float3 extents)
{
    center = rootCenter;
    extents = rootExtents;
    
    // Walk the quadrant path from root to this node
    for (uint d = 0; d < depth; d++)
    {
        extents *= 0.5;
        
        // Extract quadrant for this level from the local index
        // The bits are stored MSB-first: level 1's quadrant is in the highest 2 bits
        uint shift = (depth - 1 - d) * 2;
        uint quadrant = (localIdx >> shift) & 3;
        
        float3 offset = float3(0, 0, 0);
        // 0=NW(-x,-z), 1=NE(+x,-z), 2=SW(-x,+z), 3=SE(+x,+z)
        offset.x = (quadrant & 1) ? extents.x : -extents.x;
        offset.z = (quadrant & 2) ? extents.z : -extents.z;
        
        center += offset;
    }
}

// Load precomputed height range from mip pyramid using exact integer coordinates.
// Mip level = maxDepth - depth: depth 0 (root) uses coarsest mip, maxDepth (leaves) uses mip 0.
// Uses Load() instead of SampleLevel because R32G32_Float doesn't support bilinear filtering.
float2 SampleHeightRange(uint depth, uint localIdx, uint maxDepth)
{
    Texture2D<float2> heightRange = ResourceDescriptorHeap[HeightRangeSRVIdx];
    
    // Decode grid position (inlined — DecodeGridPos is defined later in the file)
    uint ix = 0, iz = 0;
    for (uint d = 0; d < depth; d++)
    {
        uint shift = (depth - 1 - d) * 2;
        uint quadrant = (localIdx >> shift) & 3;
        ix = (ix << 1) | (quadrant & 1u);
        iz = (iz << 1) | ((quadrant >> 1) & 1u);
    }
    
    uint mipLevel = maxDepth - depth;
    return heightRange.Load(int3(ix, iz, mipLevel));
}

// Screen-space error split check.
// Geometric error = height range from mip pyramid (max error if this node isn't split).
// Project geometric error to screen pixels; split if error exceeds threshold.
// This naturally adapts to both distance AND terrain roughness in one metric.
bool ShouldSplit(uint depth, uint localIdx, float3 nodeCenter, float3 nodeExtents,
                 float3 cameraPos, uint maxDepth)
{
    // Sample height range from precomputed mip pyramid (already in world units —
    // CSBuildMinMaxMip multiplies by MaxHeight when building mip 0)
    float2 minmax = SampleHeightRange(depth, localIdx, maxDepth);

    // Geometric error: height variation in world units
    float geometricError = minmax.y - minmax.x;

    // Distance from the camera to the node's nearest point (its footprint over its height range), not to its centre:
    // standing at the edge of a large, nearly flat node put its centre half a node away, so it stayed coarse right
    // under the camera and the ground popped there when the threshold went up (2026-10-24, the gravel road at
    // Eldermoor's bindstone: 10 m triangles under the feet at 128 px)
    float3 boxMin = float3(nodeCenter.x - nodeExtents.x, minmax.x, nodeCenter.z - nodeExtents.z);
    float3 boxMax = float3(nodeCenter.x + nodeExtents.x, minmax.y, nodeCenter.z + nodeExtents.z);
    float dist = max(distance(cameraPos, clamp(cameraPos, boxMin, boxMax)), 0.001);

    // Project geometric error to screen pixels:
    //   screenError = (geometricError × screenHeight) / (2 × dist × tan(fov/2))
    float screenError = (geometricError * ScreenHeight) / (2.0 * dist * TanHalfFov);

    return screenError > PixelErrorThreshold;
}

// Compute UV rect for a node using exact integer arithmetic — no FP accumulation.
// Decodes X/Z grid position directly from the quadrant path encoded in localIdx.
// Returns (minX, maxZ, maxX, minZ) to match the existing rect layout.
float4 ComputeRectExact(uint depth, uint localIdx)
{
    // Decode X and Z grid coordinates from the quadrant path.
    // Each level contributes 1 bit to X (bit 0 of quadrant) and 1 bit to Z (bit 1).
    // Bits are stored MSB-first: level 1's quadrant is in the highest 2 bits.
    uint ix = 0, iz = 0;
    for (uint d = 0; d < depth; d++)
    {
        uint shift = (depth - 1 - d) * 2;
        uint quadrant = (localIdx >> shift) & 3;
        ix = (ix << 1) | (quadrant & 1u);        // bit 0 = X
        iz = (iz << 1) | ((quadrant >> 1) & 1u);  // bit 1 = Z
    }

    // Node size is exactly 1 / 2^depth — exact binary fraction
    float size = 1.0 / (float)(1u << depth);

    float minX = (float)ix * size;
    float minZ = (float)iz * size;
    float maxX = minX + size;
    float maxZ = minZ + size;

    // (minX, maxZ, maxX, minZ) — Z swapped to match existing layout
    return float4(minX, maxZ, maxX, minZ);
}

// ─────────────────────────────────────────────────────────────────────
// Split flag helpers
// ─────────────────────────────────────────────────────────────────────

// Mark a node as split for this evaluation: its flag holds this evaluation's tag (FrameTag, never 0), so a flag an
// earlier frame wrote reads as not split and the buffer is never cleared (it was, every frame, until 2026-11-07: 5.6 MB
// at depth 10). Every writer stores the same value, so no atomic is needed.
void MarkSplit(uint depth, uint localIdx)
{
    if (depth > MaxDepth) return;

    uint flatIdx = LevelStart(depth) + localIdx;
    RWByteAddressBuffer splitFlags = ResourceDescriptorHeap[SplitFlagsUAVIdx];
    splitFlags.Store(flatIdx * 4, FrameTag);
}

bool IsSplit(RWByteAddressBuffer splitFlags, uint depth, uint localIdx)
{
    return splitFlags.Load((LevelStart(depth) + localIdx) * 4) == FrameTag;
}

// Encode (ix, iz) grid coordinates back to localIdx at given depth.
uint EncodeLocalIdx(uint depth, uint ix, uint iz)
{
    uint localIdx = 0;
    for (uint d = 0; d < depth; d++)
    {
        uint shift = (depth - 1 - d) * 2;
        uint bx = (ix >> (depth - 1 - d)) & 1u;
        uint bz = (iz >> (depth - 1 - d)) & 1u;
        localIdx |= (bx | (bz << 1)) << shift;
    }
    return localIdx;
}

// Decode localIdx into (ix, iz) grid coordinates at given depth.
void DecodeGridPos(uint depth, uint localIdx, out uint ix, out uint iz)
{
    ix = 0;
    iz = 0;
    for (uint d = 0; d < depth; d++)
    {
        uint shift = (depth - 1 - d) * 2;
        uint quadrant = (localIdx >> shift) & 3;
        ix = (ix << 1) | (quadrant & 1u);
        iz = (iz << 1) | ((quadrant >> 1) & 1u);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Pass 1: the walk (TERRAIN_PAGES_PLAN.md stage 2)
// ─────────────────────────────────────────────────────────────────────

// The walk's buffer (WalkUAV, raw): a header of 4 uints, then four lists of WalkCapacity entries: two that take turns
// as one level's nodes and the next's, the nodes the screen error split, the leaves. An entry is a node packed as
// (depth << 28) | localIdx, so depth ≤ 14.
static const uint WALK_LEAVES = 0;          // header: leaves listed
static const uint WALK_SPLITS = 4;          // header: nodes the screen error split, listed
static const uint WALK_LIST_SPLITS = 2;
static const uint WALK_LIST_LEAVES = 3;
uint WalkList(uint list) { return 16 + list * WalkCapacity * 4; }
uint PackNode(uint depth, uint localIdx) { return (depth << 28) | localIdx; }
void UnpackNode(uint entry, out uint depth, out uint localIdx) { depth = entry >> 28; localIdx = entry & 0x0FFFFFFFu; }

// The neighbour rule (restricted quadtree): a node the screen error splits needs its four neighbours at its depth to
// exist, so that its children never border a leaf two levels coarser; so their parents split. The sweep (until
// 2026-11-07, ForceNeighborExist) marked a neighbour's whole ancestry, but above the parent it has split already: by this same
// rule applied to the node's own parent, which split too, since a parent's screen error is never below its child's
// (its height range holds the child's, its nearest point is no further).
void MarkNeighbourParent(uint depth, int nix, int niz)
{
    if (depth == 0) return;
    uint gridSize = 1u << depth;
    if (nix < 0 || (uint)nix >= gridSize || niz < 0 || (uint)niz >= gridSize) return;
    MarkSplit(depth - 1, EncodeLocalIdx(depth - 1, (uint)nix >> 1, (uint)niz >> 1));
}

#define WALK_THREADS 1024
groupshared uint gsListed[2];   // entries in the two level lists
groupshared uint gsSplits;
groupshared uint gsLeaves;

// One group walks the tree from the root, level by level: a few thousand nodes, where the sweep tested every node of
// the tree. Three steps: the screen-error test, top-down, visiting only the children of nodes that split (the same
// nodes split as when every node is tested, for the reason above); the neighbour rule over the nodes it split; the tree
// again from the root by the final flags, into the list of leaves the emission passes take. A level that would
// overflow the lists keeps its nodes as leaves.
[numthreads(WALK_THREADS, 1, 1)]
void CSWalk(uint gi : SV_GroupIndex)
{
    RWByteAddressBuffer walk = ResourceDescriptorHeap[WalkUAVIdx];
    RWByteAddressBuffer splitFlags = ResourceDescriptorHeap[SplitFlagsUAVIdx];
    uint cap = WalkCapacity;
    uint maxDepth = MaxDepth;

    // 1. The screen-error test, top-down
    if (gi == 0)
    {
        walk.Store(WalkList(0), PackNode(0, 0));
        gsListed[0] = 1;
        gsListed[1] = 0;
        gsSplits = 0;
        gsLeaves = 0;
    }
    AllMemoryBarrierWithGroupSync();

    uint cur = 0;
    [loop] for (uint depth = 0; depth < maxDepth; depth++)
    {
        uint count = min(gsListed[cur], cap);
        uint next = cur ^ 1u;
        for (uint i = gi; i < count; i += WALK_THREADS)
        {
            uint d, localIdx;
            UnpackNode(walk.Load(WalkList(cur) + i * 4), d, localIdx);
            float3 center, extents;
            ComputeNodeBounds(depth, localIdx, RootCenter, RootExtents, center, extents);
            if (!ShouldSplit(depth, localIdx, center, extents, CameraPos, maxDepth))
                continue;

            uint slot;
            InterlockedAdd(gsListed[next], 4u, slot);
            if (slot >= cap)
                continue;
            MarkSplit(depth, localIdx);
            uint s;
            InterlockedAdd(gsSplits, 1u, s);
            if (s < cap)
                walk.Store(WalkList(WALK_LIST_SPLITS) + s * 4, PackNode(depth, localIdx));
            for (uint q = 0; q < 4; q++)
                walk.Store(WalkList(next) + (slot + q) * 4, PackNode(depth + 1, (localIdx << 2) | q));
        }
        AllMemoryBarrierWithGroupSync();
        if (gi == 0)
            gsListed[cur] = 0;
        AllMemoryBarrierWithGroupSync();
        cur = next;
    }

    // 2. The neighbour rule
    uint splits = min(gsSplits, cap);
    for (uint i = gi; i < splits; i += WALK_THREADS)
    {
        uint depth, localIdx;
        UnpackNode(walk.Load(WalkList(WALK_LIST_SPLITS) + i * 4), depth, localIdx);
        uint ix, iz;
        DecodeGridPos(depth, localIdx, ix, iz);
        MarkNeighbourParent(depth, (int)ix - 1, (int)iz);
        MarkNeighbourParent(depth, (int)ix + 1, (int)iz);
        MarkNeighbourParent(depth, (int)ix, (int)iz - 1);
        MarkNeighbourParent(depth, (int)ix, (int)iz + 1);
    }
    AllMemoryBarrierWithGroupSync();

    // 3. The leaves, by the final flags
    if (gi == 0)
    {
        walk.Store(WalkList(0), PackNode(0, 0));
        gsListed[0] = 1;
        gsListed[1] = 0;
    }
    AllMemoryBarrierWithGroupSync();

    cur = 0;
    [loop] for (uint level = 0; level <= maxDepth; level++)
    {
        uint count = min(gsListed[cur], cap);
        uint next = cur ^ 1u;
        for (uint i = gi; i < count; i += WALK_THREADS)
        {
            uint d, localIdx;
            UnpackNode(walk.Load(WalkList(cur) + i * 4), d, localIdx);
            if (level < maxDepth && IsSplit(splitFlags, level, localIdx))
            {
                uint slot;
                InterlockedAdd(gsListed[next], 4u, slot);
                if (slot < cap)
                {
                    for (uint q = 0; q < 4; q++)
                        walk.Store(WalkList(next) + (slot + q) * 4, PackNode(level + 1, (localIdx << 2) | q));
                    continue;
                }
            }
            uint leaf;
            InterlockedAdd(gsLeaves, 1u, leaf);
            if (leaf < cap)
                walk.Store(WalkList(WALK_LIST_LEAVES) + leaf * 4, PackNode(level, localIdx));
        }
        AllMemoryBarrierWithGroupSync();
        if (gi == 0)
            gsListed[cur] = 0;
        AllMemoryBarrierWithGroupSync();
        cur = next;
    }

    if (gi == 0)
    {
        walk.Store(WALK_LEAVES, min(gsLeaves, cap));
        walk.Store(WALK_SPLITS, splits);
    }
}

// Which sides of a leaf border a coarser one: the neighbour at its depth does not exist (its parent did not split).
// Bits: 0 = S (-Z), 1 = E (+X), 2 = N (+Z), 3 = W (-X)
uint StitchMask(RWByteAddressBuffer splitFlags, uint depth, uint localIdx)
{
    if (depth == 0) return 0;
    uint ix, iz;
    DecodeGridPos(depth, localIdx, ix, iz);
    uint gridSize = 1u << depth;
    uint mask = 0;
    if (iz > 0 && !IsSplit(splitFlags, depth - 1, EncodeLocalIdx(depth, ix, iz - 1) >> 2)) mask |= 1u;
    if (ix + 1 < gridSize && !IsSplit(splitFlags, depth - 1, EncodeLocalIdx(depth, ix + 1, iz) >> 2)) mask |= 2u;
    if (iz + 1 < gridSize && !IsSplit(splitFlags, depth - 1, EncodeLocalIdx(depth, ix, iz + 1) >> 2)) mask |= 4u;
    if (ix > 0 && !IsSplit(splitFlags, depth - 1, EncodeLocalIdx(depth, ix - 1, iz) >> 2)) mask |= 8u;
    return mask;
}

// The leaf a thread of the emission passes takes: the walk's leaf number 'id', if there are that many
bool TakeLeaf(uint id, out uint depth, out uint localIdx)
{
    depth = 0;
    localIdx = 0;
    RWByteAddressBuffer walk = ResourceDescriptorHeap[WalkUAVIdx];
    if (id >= walk.Load(WALK_LEAVES)) return false;
    UnpackNode(walk.Load(WalkList(WALK_LIST_LEAVES) + id * 4), depth, localIdx);
    return true;
}

// ─────────────────────────────────────────────────────────────────────
// Pass 2: Emit leaves using restriction-enforced split flags
// ─────────────────────────────────────────────────────────────────────

[numthreads(256, 1, 1)]
void CSEmitLeaves(uint3 dtid : SV_DispatchThreadID)
{
    RWByteAddressBuffer splitFlags = ResourceDescriptorHeap[SplitFlagsUAVIdx];
    uint depth, localIdx;
    if (!TakeLeaf(dtid.x, depth, localIdx))
        return;

    uint maxDepth = MaxDepth;
    float3 rootCenter = RootCenter;
    float3 rootExtents = RootExtents;

    // Compute this node's bounds for frustum + Hi-Z tests
    float3 nodeCenter, nodeExtents;
    ComputeNodeBounds(depth, localIdx, rootCenter, rootExtents, nodeCenter, nodeExtents);

    // Tight bounding sphere from height range mip pyramid
    float2 minmax = SampleHeightRange(depth, localIdx, maxDepth);
    float xzRadius = length(float2(nodeExtents.x, nodeExtents.z));
    float heightHalfRange = (minmax.y - minmax.x) * 0.5;
    float sphereRadius = sqrt(xzRadius * xzRadius + heightHalfRange * heightHalfRange);
    float sphereY = (minmax.x + minmax.y) * 0.5;
    float3 sphereCenter = float3(nodeCenter.x, sphereY, nodeCenter.z);

    // Frustum cull: skip patches entirely outside the camera frustum
    if (!IsFrustumVisible(sphereCenter, sphereRadius))
        return;

    // Hi-Z occlusion cull: skip patches fully behind solid geometry.
    float4 sphereClip = mul(float4(sphereCenter, 1.0), OcclusionProjection);
    float biasedRadius = max(sphereRadius, sphereClip.w * 0.01);
    if (IsTerrainOccluded(sphereCenter, biasedRadius))
        return;
    
    // Atomic append to output — get slot index
    RWByteAddressBuffer counterBuffer = ResourceDescriptorHeap[CounterUAVIdx];
    uint patchIdx;
    counterBuffer.InterlockedAdd(0, 1, patchIdx);
    
    // Safety: don't write past buffer capacity
    uint maxPatches = MaxPatches;
    if (patchIdx >= maxPatches)
        return;
    
    // Write InstanceDescriptor
    RWStructuredBuffer<InstanceDescriptor> outDescriptors = ResourceDescriptorHeap[OutputDescriptorsUAVIdx];
    InstanceDescriptor desc;
    desc.TransformSlot = TransformSlot;
    desc.MaterialId = MaterialId;
    desc.CustomDataIdx = 0;
    desc.MeshPartIdx = 0;
    desc.BoneBufferIdx = 0;
    outDescriptors[patchIdx] = desc;
    
    // Write BoundingSphere (already computed above for culling)
    RWStructuredBuffer<float4> outSpheres = ResourceDescriptorHeap[OutputSpheresUAVIdx];
    outSpheres[patchIdx] = float4(sphereCenter, sphereRadius);
    
    // Write SubbatchId (all terrain patches use the same mesh)
    RWStructuredBuffer<uint> outSubbatchIds = ResourceDescriptorHeap[OutputSubbatchIdsUAVIdx];
    outSubbatchIds[patchIdx] = MeshPartId;
    
    // ── Stitch mask: the sides that border a coarser leaf ──
    uint stitchMask = StitchMask(splitFlags, depth, localIdx);

    // Write TerrainPatchData
    RWStructuredBuffer<TerrainPatchData> outTerrainData = ResourceDescriptorHeap[OutputTerrainDataUAVIdx];
    TerrainPatchData patchData;
    patchData.Rect = ComputeRectExact(depth, localIdx);
    // Level.x = LOD level (for debug), Level.y = stitch mask (4 bits)
    patchData.Level = float2((float)(maxDepth - depth), (float)stitchMask);
    patchData.Padding = float2(0, 0);
    outTerrainData[patchIdx] = patchData;
}

// ─────────────────────────────────────────────────────────────────────
// One-time: Build height range mip pyramid (RG32F texture)
// Dispatched once per mip, from mip 0 (finest) up to coarsest.
// Mip 0 reads heightmap texels; higher mips downsample from previous mip.
// ─────────────────────────────────────────────────────────────────────

[numthreads(8, 8, 1)]
void CSBuildMinMaxMip(uint3 dtid : SV_DispatchThreadID)
{
    uint buildMip = BuildMipIdx;
    
    RWTexture2D<float2> output = ResourceDescriptorHeap[MipOutputUAVIdx];
    
    // Get output dimensions
    uint outW, outH;
    output.GetDimensions(outW, outH);
    
    if (dtid.x >= outW || dtid.y >= outH)
        return;
    
    if (buildMip == 0)
    {
        // Mip 0: the range of every height sample a vertex of this node can interpolate between. Sample i sits at
        // i / (N - 1) in terrain UV, one on every vertex (terrain_grid.hlsli), so the node's uv square holds samples
        // uvMin (N - 1) .. uvMax (N - 1): a leaf's 33 x 33, exactly the samples its patch's vertices read. Until
        // 2026-11-07 the samples sat at (i + 0.5) / N and this took floor(uvMin N - 0.5) .. ceil(uvMax N - 0.5); until
        // 2026-10-24 a 4x4 grid of samples over the ~32x32: a crest between them stood outside the node's bounding
        // sphere, which the frustum, Hi-Z and shadow receiver tests all trust. Built once per height bake, so reading
        // them all costs nothing per frame.
        // The samples come from the terrain's pages (terrain_pages.hlsli, TERRAIN_PAGES_PLAN.md 4a)
        TerrainPagesParams pages = LoadTerrain(TerrainPagesIdx);
        float maxH = MaxHeight;
        int cells = (int)pages.Cells;

        float2 uvMin = float2(dtid.xy) / float2(outW, outH);
        float2 uvMax = float2(dtid.xy + 1) / float2(outW, outH);
        int2 lo = clamp(int2(floor(uvMin * cells)), 0, cells);
        int2 hi = clamp(int2(ceil(uvMax * cells)), 0, cells);

        float hMin = 1e30;
        float hMax = -1e30;
        for (int ty = lo.y; ty <= hi.y; ty++)
        {
            for (int tx = lo.x; tx <= hi.x; tx++)
            {
                float h = TerrainSample01(pages, int2(tx, ty)) * maxH;
                hMin = min(hMin, h);
                hMax = max(hMax, h);
            }
        }

        output[dtid.xy] = float2(hMin, hMax);
    }
    else
    {
        // Mip N>0: downsample from previous mip (min of mins, max of maxes)
        Texture2D<float2> input = ResourceDescriptorHeap[MipInputSRVIdx];
        
        uint2 srcBase = dtid.xy * 2;
        float2 a = input[srcBase + uint2(0, 0)];
        float2 b = input[srcBase + uint2(1, 0)];
        float2 c = input[srcBase + uint2(0, 1)];
        float2 d = input[srcBase + uint2(1, 1)];
        
        float hMin = min(min(a.x, b.x), min(c.x, d.x));
        float hMax = max(max(a.y, b.y), max(c.y, d.y));
        
        output[dtid.xy] = float2(hMin, hMax);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Pass 3: Build DrawInstanced indirect arguments from counter
// Single-thread dispatch (1,1,1). Reads the atomic counter and writes
// a D3D12_DRAW_INSTANCED_ARGUMENTS struct for ExecuteIndirect.
// ─────────────────────────────────────────────────────────────────────

[numthreads(1, 1, 1)]
void CSBuildDrawArgs(uint3 dtid : SV_DispatchThreadID)
{
    // Read the patch count from the atomic counter
    RWByteAddressBuffer counterBuffer = ResourceDescriptorHeap[CounterUAVIdx];
    uint patchCount = counterBuffer.Load(0);
    
    // Clamp to buffer capacity
    patchCount = min(patchCount, MaxPatches);
    
    // Write DrawIndexedInstanced args: (IndexCountPerInstance, InstanceCount, StartIndex, BaseVertex, StartInstance).
    // Indexed, the hardware reuses a grid vertex its neighbouring triangles share; pulled through the index buffer
    // (DrawInstanced until 2026-10-24) every corner ran the vertex shader: 6144 invocations per patch for 1089 vertices.
    RWByteAddressBuffer argsBuffer = ResourceDescriptorHeap[IndirectArgsUAVIdx];
    argsBuffer.Store(0, VertexCountIdx);     // IndexCountPerInstance (mesh index count from C#)
    argsBuffer.Store(4, patchCount);         // InstanceCount = number of visible patches
    argsBuffer.Store(8, 0u);                 // StartIndexLocation = 0
    argsBuffer.Store(12, 0u);                // BaseVertexLocation = 0
    argsBuffer.Store(16, 0u);                // StartInstanceLocation = 0
}
// ─────────────────────────────────────────────────────────────────────
// Shadow cascade data (StructuredBuffer via push constant, shared with cull_instances.hlsl)
// ─────────────────────────────────────────────────────────────────────
struct CascadeData
{
    float4 Planes[6];           // frustum planes
    row_major float4x4 VP;      // current frame VP
    row_major float4x4 PrevVP;  // previous frame VP (for Hi-Z)
    float4 SplitDistances;      // X=near, Y=far
};

// Shadow-specific: CascadeIdxUAVIdx = GET_INDEX(5), CascadeCountIdx = GET_INDEX(13), CascadeBufferSRVIdx = GET_INDEX(14)

// Whether a patch can shade anything the camera sees through cascade 'cascadeIdx': seen from the sun, a receiver of
// that cascade (shadow_receivers.hlsli) lies at or behind the patch's nearest point under its footprint, widened by the
// shadow filter's reach. As cull_instances.hlsl ShadesReceivers, with the VP from terrain-local space (C# puts it in
// this cascade buffer), so the patch's local sphere goes in as it is. Before 2026-10-24 terrain patches skipped this
// test: every cascade drew every patch in its box, the fine patches round the camera in the far cascades too, where
// nothing visible reads them.
bool PatchShadesReceivers(RWStructuredBuffer<uint> receivers, row_major float4x4 vp, float3 center, float radius,
                          uint cascadeIdx, float marginUV)
{
    float4 clip = mul(float4(center, 1.0), vp);         // orthographic: w = 1
    float casterNear = clip.z - radius * length(float3(vp._13, vp._23, vp._33));

    float2 uv = clip.xy * float2(0.5, -0.5) + 0.5;
    float radiusUV = 0.5 * radius * length(float3(vp._11, vp._21, vp._31)) + marginUV;
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

// Test a bounding sphere against one cascade's 6 frustum planes
bool IsCascadeVisible(float3 center, float radius, uint cascadeIdx)
{
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    CascadeData cascade = cascades[cascadeIdx];
    [unroll]
    for (uint i = 0; i < 6; i++)
    {
        float4 plane = cascade.Planes[i];
        float dist = dot(plane.xyz, center) + plane.w;
        if (dist > radius)
            return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────
// Shadow Pass: Emit leaves with per-cascade frustum culling
// For each leaf node, tests against each cascade's frustum individually.
// Emits compact (patch, cascadeIdx) pairs — only entries that intersect
// a cascade are written. Counter gives final instance count directly.
// No Hi-Z culling for shadows.
//
// Additional push constants beyond normal CSEmitLeaves:
//   Slot 9  (CascadeIdxUAV) = output cascadeIdx per entry (uint)
//   Slot 30 (CascadeCount)  = number of active cascades
// Cascade frustum planes bound at root slot 2 → register(b1).
// ─────────────────────────────────────────────────────────────────────

[numthreads(256, 1, 1)]
void CSEmitLeavesShadow(uint3 dtid : SV_DispatchThreadID)
{
    // The leaves of this frame's walk (CSWalk ran in the opaque pass, before the shadow maps)
    RWByteAddressBuffer splitFlags = ResourceDescriptorHeap[SplitFlagsUAVIdx];
    uint depth, localIdx;
    if (!TakeLeaf(dtid.x, depth, localIdx))
        return;

    uint maxDepth = MaxDepth;
    float3 rootCenter = RootCenter;
    float3 rootExtents = RootExtents;

    // Compute this node's bounds
    float3 nodeCenter, nodeExtents;
    ComputeNodeBounds(depth, localIdx, rootCenter, rootExtents, nodeCenter, nodeExtents);

    // Tight bounding sphere from height range mip pyramid (already in world units)
    float2 minmax = SampleHeightRange(depth, localIdx, maxDepth);
    float xzRadius = length(float2(nodeExtents.x, nodeExtents.z));
    float heightHalfRange = (minmax.y - minmax.x) * 0.5;
    float sphereRadius = sqrt(xzRadius * xzRadius + heightHalfRange * heightHalfRange);
    float sphereY = (minmax.x + minmax.y) * 0.5;
    float3 sphereCenter = float3(nodeCenter.x, sphereY, nodeCenter.z);

    // Test against each cascade frustum and emit per-cascade entries
    // Planes are in terrain-local space (C# transforms them), sphere is also local
    uint cascadeCount = CascadeCountIdx;
    uint maxPatches = MaxPatches;

    RWByteAddressBuffer counterBuffer = ResourceDescriptorHeap[CounterUAVIdx];
    RWStructuredBuffer<InstanceDescriptor> outDescriptors = ResourceDescriptorHeap[OutputDescriptorsUAVIdx];
    RWStructuredBuffer<float4> outSpheres = ResourceDescriptorHeap[OutputSpheresUAVIdx];
    RWStructuredBuffer<uint> outSubbatchIds = ResourceDescriptorHeap[OutputSubbatchIdsUAVIdx];
    RWStructuredBuffer<TerrainPatchData> outTerrainData = ResourceDescriptorHeap[OutputTerrainDataUAVIdx];
    RWStructuredBuffer<uint> outCascadeIdx = ResourceDescriptorHeap[CascadeIdxUAVIdx];

    // Precompute shared patch data
    uint transformSlot = TransformSlot;
    uint materialId = MaterialId;
    uint meshPartId = MeshPartId;

    // Stitch mask (as CSEmitLeaves)
    uint stitchMask = StitchMask(splitFlags, depth, localIdx);

    float4 patchRect = ComputeRectExact(depth, localIdx);

    // The receiver test (EngineSettings.ShadowReceiverCulling On)
    RWStructuredBuffer<uint> receivers = ResourceDescriptorHeap[ShadowReceiversIdx];
    StructuredBuffer<CascadeData> cascadeData = ResourceDescriptorHeap[CascadeBufferSRVIdx];
    bool testReceivers = ShadowReceiversIdx != 0 && receivers[RECEIVER_MODE] == RECEIVER_MODE_CULL;
    float marginUV = testReceivers ? asfloat(receivers[RECEIVER_MARGIN]) : 0.0;

    for (uint c = 0; c < cascadeCount; c++)
    {
        if (!IsCascadeVisible(sphereCenter, sphereRadius, c))
            continue;
        if (testReceivers && !PatchShadesReceivers(receivers, cascadeData[c].VP, sphereCenter, sphereRadius, c, marginUV))
            continue;

        uint patchIdx;
        counterBuffer.InterlockedAdd(0, 1, patchIdx);

        if (patchIdx >= maxPatches)
            return; // buffer full, stop entirely

        // Write InstanceDescriptor
        InstanceDescriptor desc;
        desc.TransformSlot = transformSlot;
        desc.MaterialId = materialId;
        desc.CustomDataIdx = 0;
        desc.MeshPartIdx = 0;
        desc.BoneBufferIdx = 0;
        outDescriptors[patchIdx] = desc;

        // Write BoundingSphere
        outSpheres[patchIdx] = float4(sphereCenter, sphereRadius);

        // Write SubbatchId
        outSubbatchIds[patchIdx] = meshPartId;

        // Write TerrainPatchData
        TerrainPatchData patch;
        patch.Rect = patchRect;
        // The mask as a value, as CSEmitLeaves writes it: VS_Shadow converts it back with (uint). Stored as
        // asfloat(stitchMask) until 2026-10-24, a denormal that converts to 0: no shadow patch was ever stitched, so
        // the shadow maps' terrain had T-junction cracks at every LOD seam
        patch.Level = float2((float)depth, (float)stitchMask);
        patch.Padding = float2(0, 0);
        outTerrainData[patchIdx] = patch;

        // Write cascade index for this entry
        outCascadeIdx[patchIdx] = c;
    }
}

