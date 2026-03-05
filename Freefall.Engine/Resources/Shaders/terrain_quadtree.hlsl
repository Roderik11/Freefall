// GPU-Driven Quadtree Terrain — Restricted Quadtree (Three-Pass)
// Pass 1 (CSMarkSplits): Each thread evaluates its node, marks split flags, and forces
//   spatial neighbors to split (restricted quadtree constraint: ≤1 level difference).
// Pass 2 (CSEmitLeaves): Reads enforced split flags and emits visible leaf patches
//   with inline Hi-Z occlusion culling.
// Pass 3 (CSBuildDrawArgs): Single-thread pass that reads the append counter and writes
//   DrawInstanced indirect arguments for ExecuteIndirect.

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
};

#pragma kernel CSMarkSplits
#pragma kernel CSEmitLeaves
#pragma kernel CSBuildMinMaxMip
#pragma kernel CSBuildDrawArgs
#pragma kernel CSEmitLeavesShadow

// Push constants (root parameter 0, register b3) — bindless indices only
cbuffer PushConstants : register(b3)
{
    uint4 Indices[8];
};

#define GET_INDEX(i) Indices[i/4][i%4]

// ── Per-dispatch bindless indices (push constants) ──
#define OutputDescriptorsUAVIdx  GET_INDEX(0)    // UAV: InstanceDescriptor buffer
#define OutputSpheresUAVIdx     GET_INDEX(1)    // UAV: BoundingSphere buffer
#define OutputSubbatchIdsUAVIdx GET_INDEX(2)    // UAV: subbatch ID buffer
#define OutputTerrainDataUAVIdx GET_INDEX(3)    // UAV: TerrainPatchData buffer
#define CounterUAVIdx           GET_INDEX(4)    // UAV: atomic counter
#define CascadeIdxUAVIdx        GET_INDEX(5)    // UAV: cascade index buffer (shadow)
#define SplitFlagsUAVIdx        GET_INDEX(6)    // UAV: split flags buffer
#define HeightRangeSRVIdx       GET_INDEX(7)    // SRV: height range mip pyramid
#define BuildMipIdx             GET_INDEX(8)    // uint: current mip for CSBuildMinMaxMip
#define MipInputSRVIdx          GET_INDEX(9)    // SRV: previous mip (CSBuildMinMaxMip)
#define MipOutputUAVIdx         GET_INDEX(10)   // UAV: current mip output (CSBuildMinMaxMip)
#define IndirectArgsUAVIdx      GET_INDEX(11)   // UAV: indirect draw args
#define VertexCountIdx          GET_INDEX(12)   // uint: mesh index count (CSBuildDrawArgs)
#define CascadeCountIdx         GET_INDEX(13)   // uint: number of cascades (shadow)
#define CascadeBufferSRVIdx     GET_INDEX(14)   // SRV: StructuredBuffer<CascadeData> (shadow)

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
    uint   MaxPatches;     uint  HeightTexIdx;           uint2 _terrainPad;
};

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

    // Project sphere radius to screen pixels for mip selection
    float projScale = OcclusionProjection._m11;
    projScale = abs(projScale) < 0.001 ? 1.0 : projScale;
    float projRadius = (worldRadius * projScale) / clipCenter.w;
    float screenRadius = projRadius * mip0Size.y * 0.5;

    // Pick mip where sphere covers ~2 texels (conservative)
    float mipLevel = ceil(log2(max(screenRadius * 2.0, 1.0)));
    mipLevel = min(mipLevel, levels - 1.0f);

    uint mip = (uint)mipLevel;
    float2 mipSize = max(float2(1,1), mip0Size / (float)(1u << mip));

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

// Decompose a flat node index into (depth, localIndex within that depth level).
// Returns depth, sets localIdx.
uint DecomposeIndex(uint flatIndex, out uint localIdx)
{
    // Walk levels to find which depth this index belongs to
    uint depth = 0;
    uint levelSize = 1; // 4^0 = 1
    uint levelOffset = 0;
    
    uint maxDepth = MaxDepth;
    
    for (uint d = 0; d <= maxDepth; d++)
    {
        if (flatIndex < levelOffset + levelSize)
        {
            depth = d;
            localIdx = flatIndex - levelOffset;
            return depth;
        }
        levelOffset += levelSize;
        levelSize *= 4;
    }
    
    // Should not reach here if flatIndex < TotalNodes
    localIdx = 0;
    return maxDepth;
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

    // Bounding sphere center uses actual height midpoint (world units)
    float3 sphereCenter = nodeCenter;
    sphereCenter.y = (minmax.x + minmax.y) * 0.5;

    // Distance from camera to node center
    float dist = max(distance(cameraPos, sphereCenter), 0.001);

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

// Mark a node as split in the split flags buffer (atomic OR).
void MarkSplit(uint depth, uint localIdx)
{
    if (depth > MaxDepth) return;
    
    uint flatIdx = LevelStart(depth) + localIdx;
    RWByteAddressBuffer splitFlags = ResourceDescriptorHeap[SplitFlagsUAVIdx];
    
    uint dummy;
    splitFlags.InterlockedOr(flatIdx * 4, 1u, dummy);
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

// Ensure a neighbor EXISTS in the tree by forcing its ancestry to split.
// Does NOT force the neighbor itself to split — it becomes a leaf at 'depth',
// which is at most 1 level coarser than our children at depth+1.
// The per-vertex CDLOD morph handles this 1-level transition seamlessly.
void ForceNeighborExist(uint depth, int neighborIx, int neighborIz)
{
    uint gridSize = 1u << depth;
    
    if (neighborIx < 0 || (uint)neighborIx >= gridSize) return;
    if (neighborIz < 0 || (uint)neighborIz >= gridSize) return;
    
    uint nix = (uint)neighborIx;
    uint niz = (uint)neighborIz;
    
    // Force entire ancestry chain so the neighbor exists as a leaf
    for (uint d = depth; d > 0; d--)
    {
        nix >>= 1;
        niz >>= 1;
        MarkSplit(d - 1, EncodeLocalIdx(d - 1, nix, niz));
    }
}

// ─────────────────────────────────────────────────────────────────────
// Pass 1: Mark split decisions + ensure neighbor existence
// Standard CDLOD with neighbor ancestry enforcement to prevent
// multi-level gaps. Neighbors are never forced to split themselves.
// ─────────────────────────────────────────────────────────────────────

[numthreads(256, 1, 1)]
void CSMarkSplits(uint3 dtid : SV_DispatchThreadID)
{
    uint nodeIndex = dtid.x;
    
    if (nodeIndex >= TotalNodes)
        return;
    
    uint maxDepth = MaxDepth;
    float3 cameraPos = CameraPos;
    float3 rootCenter = RootCenter;
    float3 rootExtents = RootExtents;
    
    // Decompose flat index into depth level and local index
    uint localIdx;
    uint depth = DecomposeIndex(nodeIndex, localIdx);
    
    // Can't split at max depth
    if (depth >= MaxDepth)
        return;
    
    // Compute this node's bounds
    float3 nodeCenter, nodeExtents;
    ComputeNodeBounds(depth, localIdx, rootCenter, rootExtents, nodeCenter, nodeExtents);
    
    // Screen-space error split check
    if (!ShouldSplit(depth, localIdx, nodeCenter, nodeExtents, cameraPos, maxDepth))
        return;
    
    // Mark this node as split
    MarkSplit(depth, localIdx);
    
    // Decode grid position for ancestry + neighbor enforcement
    uint ix, iz;
    DecodeGridPos(depth, localIdx, ix, iz);
    
    // Mark full ancestry so this node exists in the tree
    uint ax = ix, az = iz;
    for (uint d = depth; d > 0; d--)
    {
        ax >>= 1;
        az >>= 1;
        MarkSplit(d - 1, EncodeLocalIdx(d - 1, ax, az));
    }
    
    // Ensure cardinal neighbors exist at this depth (ancestry only).
    // This prevents multi-level gaps: our children at depth+1 will be
    // adjacent to these neighbors at depth — a 1-level diff that the
    // CDLOD morph handles naturally.
    ForceNeighborExist(depth, (int)ix - 1, (int)iz);
    ForceNeighborExist(depth, (int)ix + 1, (int)iz);
    ForceNeighborExist(depth, (int)ix,     (int)iz - 1);
    ForceNeighborExist(depth, (int)ix,     (int)iz + 1);
}

// ─────────────────────────────────────────────────────────────────────
// Pass 2: Emit leaves using restriction-enforced split flags
// ─────────────────────────────────────────────────────────────────────

[numthreads(256, 1, 1)]
void CSEmitLeaves(uint3 dtid : SV_DispatchThreadID)
{
    uint nodeIndex = dtid.x;
    
    if (nodeIndex >= TotalNodes)
        return;
    
    uint maxDepth = MaxDepth;
    float3 rootCenter = RootCenter;
    float3 rootExtents = RootExtents;
    
    // Decompose flat index into depth level and local index
    uint localIdx;
    uint depth = DecomposeIndex(nodeIndex, localIdx);
    
    // Read split flags
    RWByteAddressBuffer splitFlags = ResourceDescriptorHeap[SplitFlagsUAVIdx];
    uint mySplitFlag = splitFlags.Load(nodeIndex * 4);
    
    // A node is a visible leaf if:
    //   - Its parent is split (so this node exists) AND
    //   - It is NOT split itself, OR it's at max depth
    
    bool iAmLeaf;
    
    if (depth == 0)
    {
        // Root node: leaf only if it's not split
        iAmLeaf = (mySplitFlag == 0);
    }
    else
    {
        // Check parent's split flag
        uint parentLocalIdx = localIdx >> 2;
        uint parentFlatIdx = LevelStart(depth - 1) + parentLocalIdx;
        uint parentSplitFlag = splitFlags.Load(parentFlatIdx * 4);
        
        bool parentIsSplit = (parentSplitFlag != 0);
        bool iAmSplit = (mySplitFlag != 0);
        
        iAmLeaf = parentIsSplit && (!iAmSplit || depth == maxDepth);
    }
    
    if (!iAmLeaf)
        return;

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
    // Apply a distance-proportional bias (same as cull_instances.hlsl CSVisibility) to
    // compensate for camera movement between the previous-frame VP and the current frame.
    // Without this, flat patches (small heightHalfRange) have almost no depth margin and
    // get falsely occluded whenever the camera moves even slightly.
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
    outDescriptors[patchIdx] = desc;
    
    // Write BoundingSphere (already computed above for culling)
    RWStructuredBuffer<float4> outSpheres = ResourceDescriptorHeap[OutputSpheresUAVIdx];
    outSpheres[patchIdx] = float4(sphereCenter, sphereRadius);
    
    // Write SubbatchId (all terrain patches use the same mesh)
    RWStructuredBuffer<uint> outSubbatchIds = ResourceDescriptorHeap[OutputSubbatchIdsUAVIdx];
    outSubbatchIds[patchIdx] = MeshPartId;
    
    // ── Compute stitch mask ──
    // Check each cardinal neighbor: if the neighbor at the same depth does NOT exist
    // (parent wasn't split), the neighbor is coarser → set stitch bit.
    // Bits: 0=S(-Z), 1=E(+X), 2=N(+Z), 3=W(-X)
    uint ix, iz;
    DecodeGridPos(depth, localIdx, ix, iz);
    uint gridSize = 1u << depth;
    uint stitchMask = 0;

    if (depth > 0)
    {
        // South neighbor (iz - 1)
        if (iz > 0)
        {
            uint nIdx = LevelStart(depth) + EncodeLocalIdx(depth, ix, iz - 1);
            uint nParentLocal = (EncodeLocalIdx(depth, ix, iz - 1)) >> 2;
            uint nParentFlat = LevelStart(depth - 1) + nParentLocal;
            uint nParentSplit = splitFlags.Load(nParentFlat * 4);
            if (nParentSplit == 0) stitchMask |= 1u;  // neighbor is coarser
        }

        // East neighbor (ix + 1)
        if (ix + 1 < gridSize)
        {
            uint nParentLocal = (EncodeLocalIdx(depth, ix + 1, iz)) >> 2;
            uint nParentFlat = LevelStart(depth - 1) + nParentLocal;
            uint nParentSplit = splitFlags.Load(nParentFlat * 4);
            if (nParentSplit == 0) stitchMask |= 2u;
        }

        // North neighbor (iz + 1)
        if (iz + 1 < gridSize)
        {
            uint nParentLocal = (EncodeLocalIdx(depth, ix, iz + 1)) >> 2;
            uint nParentFlat = LevelStart(depth - 1) + nParentLocal;
            uint nParentSplit = splitFlags.Load(nParentFlat * 4);
            if (nParentSplit == 0) stitchMask |= 4u;
        }

        // West neighbor (ix - 1)
        if (ix > 0)
        {
            uint nParentLocal = (EncodeLocalIdx(depth, ix - 1, iz)) >> 2;
            uint nParentFlat = LevelStart(depth - 1) + nParentLocal;
            uint nParentSplit = splitFlags.Load(nParentFlat * 4);
            if (nParentSplit == 0) stitchMask |= 8u;
        }
    }

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
        // Mip 0: sample from the source heightmap
        // Each output texel covers a region of the heightmap; sample a 4x4 grid
        Texture2D<float4> heightTex = ResourceDescriptorHeap[HeightTexIdx];
        float maxH = MaxHeight;
        
        float2 uvMin = float2(dtid.xy) / float2(outW, outH);
        float2 uvMax = float2(dtid.xy + 1) / float2(outW, outH);
        
        float hMin = 1e30;
        float hMax = -1e30;
        
        const uint SAMPLES = 4;
        for (uint sy = 0; sy < SAMPLES; sy++)
        {
            for (uint sx = 0; sx < SAMPLES; sx++)
            {
                float2 t = (float2(sx, sy) + 0.5) / (float)SAMPLES;
                float2 uv = lerp(uvMin, uvMax, t);
                float h = heightTex.SampleLevel(sampHeightCS, uv, 0).r * maxH;
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
    
    // Write DrawInstanced args: (VertexCountPerInstance, InstanceCount, StartVertex, StartInstance)
    RWByteAddressBuffer argsBuffer = ResourceDescriptorHeap[IndirectArgsUAVIdx];
    argsBuffer.Store(0, VertexCountIdx);     // VertexCountPerInstance (mesh index count from C#)
    argsBuffer.Store(4, patchCount);         // InstanceCount = number of visible patches
    argsBuffer.Store(8, 0u);                 // StartVertexLocation = 0
    argsBuffer.Store(12, 0u);                // StartInstanceLocation = 0
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
    uint nodeIndex = dtid.x;

    if (nodeIndex >= TotalNodes)
        return;

    uint maxDepth = MaxDepth;
    float3 cameraPos = CameraPos;
    float3 rootCenter = RootCenter;
    float3 rootExtents = RootExtents;

    // Decompose flat index into depth level and local index
    uint localIdx;
    uint depth = DecomposeIndex(nodeIndex, localIdx);

    // Read split flags
    RWByteAddressBuffer splitFlags = ResourceDescriptorHeap[SplitFlagsUAVIdx];
    uint flatIdx = LevelStart(depth) + localIdx;
    uint isSplit = splitFlags.Load(flatIdx * 4);

    // A leaf is a node that is NOT split (and exists in the tree)
    if (depth > 0)
    {
        uint parentLocal = localIdx >> 2;
        uint parentFlat = LevelStart(depth - 1) + parentLocal;
        uint parentSplit = splitFlags.Load(parentFlat * 4);
        if (parentSplit == 0) return; // parent not split → this node doesn't exist
    }

    if (isSplit != 0 && depth < maxDepth)
        return; // interior node, not a leaf

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

    // Stitch mask (same as CSEmitLeaves)
    uint ix, iz;
    DecodeGridPos(depth, localIdx, ix, iz);
    uint gridSize = 1u << depth;
    uint stitchMask = 0;
    if (depth > 0)
    {
        if (iz > 0)
        {
            uint nParentLocal = (EncodeLocalIdx(depth, ix, iz - 1)) >> 2;
            uint nParentFlat = LevelStart(depth - 1) + nParentLocal;
            if (splitFlags.Load(nParentFlat * 4) == 0) stitchMask |= 1u;
        }
        if (ix + 1 < gridSize)
        {
            uint nParentLocal = (EncodeLocalIdx(depth, ix + 1, iz)) >> 2;
            uint nParentFlat = LevelStart(depth - 1) + nParentLocal;
            if (splitFlags.Load(nParentFlat * 4) == 0) stitchMask |= 2u;
        }
        if (iz + 1 < gridSize)
        {
            uint nParentLocal = (EncodeLocalIdx(depth, ix, iz + 1)) >> 2;
            uint nParentFlat = LevelStart(depth - 1) + nParentLocal;
            if (splitFlags.Load(nParentFlat * 4) == 0) stitchMask |= 4u;
        }
        if (ix > 0)
        {
            uint nParentLocal = (EncodeLocalIdx(depth, ix - 1, iz)) >> 2;
            uint nParentFlat = LevelStart(depth - 1) + nParentLocal;
            if (splitFlags.Load(nParentFlat * 4) == 0) stitchMask |= 8u;
        }
    }

    float4 patchRect = ComputeRectExact(depth, localIdx);

    for (uint c = 0; c < cascadeCount; c++)
    {
        if (!IsCascadeVisible(sphereCenter, sphereRadius, c))
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
        outDescriptors[patchIdx] = desc;

        // Write BoundingSphere
        outSpheres[patchIdx] = float4(sphereCenter, sphereRadius);

        // Write SubbatchId
        outSubbatchIds[patchIdx] = meshPartId;

        // Write TerrainPatchData
        TerrainPatchData patch;
        patch.Rect = patchRect;
        patch.Level = float2((float)depth, asfloat(stitchMask));
        patch.Padding = float2(0, 0);
        outTerrainData[patchIdx] = patch;

        // Write cascade index for this entry
        outCascadeIdx[patchIdx] = c;
    }
}

