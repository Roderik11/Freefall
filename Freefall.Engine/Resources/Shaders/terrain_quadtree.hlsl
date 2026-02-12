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

// Push constants (root parameter 0, register b3)
// Laid out as uint4 Indices[8]; — 32 dwords total
//   Slot 0: OutputDescriptorsUAV
//   Slot 1: OutputSpheresUAV
//   Slot 2: OutputSubbatchIdsUAV
//   Slot 3: OutputTerrainDataUAV
//   Slot 4: CounterUAV
//   Slot 5: MaxDepth
//   Slot 6: TransformSlot
//   Slot 7: MaterialId
//   Slot 8: MeshPartId
//   Slot 9: (unused, was LODRange0)
//   Slot 10: TotalNodes
//   Slot 11: HeightTexIdx
//   Slot 12-14: CameraPos (float3)
//   Slot 15-17: RootCenter (float3)
//   Slot 18-20: RootExtents (float3)
//   Slot 21: MaxPatches (capacity of output buffers)
//   Slot 22: SplitFlagsUAV
//   Slot 23: MaxHeight
//   Slot 24: HeightRangeSRV (full mip pyramid, RG32F, for runtime sampling)
//   Slot 25: TerrainSize.x
//   Slot 26: TerrainSize.y
//   Slot 27: BuildMip (current mip level for CSBuildMinMaxMip)
//   Slot 28: PixelErrorThreshold (float bits) — max screen-space error before split
//   Slot 29: ScreenHeight (float bits) / MipInputSRV (overwritten per-pass)
//   Slot 30: TanHalfFov (float bits) / MipOutputUAV (overwritten per-pass)
//   Slot 31: IndirectArgsUAV (target UAV for CSBuildDrawArgs output)

cbuffer PushConstants : register(b3)
{
    uint4 Indices[8];
};

// Frustum planes + Hi-Z occlusion parameters (root slot 1 -> register b0)
// Shared layout with cull_instances.hlsl — same constant buffer is used by both.
cbuffer FrustumPlanes : register(b0)
{
    float4 Plane0;
    float4 Plane1;
    float4 Plane2;
    float4 Plane3;
    float4 Plane4;
    float4 Plane5;
    // Hi-Z occlusion culling parameters
    row_major float4x4 OcclusionProjection; // ViewProjection for sphere-to-screen
    uint HiZSrvIdx;        // Bindless SRV index of Hi-Z pyramid (0 = disabled)
    float2 HiZSize;        // Mip 0 dimensions
    uint HiZMipCount;      // Number of mip levels in pyramid
    float NearPlane;       // Camera near plane
    uint CullStatsUAVIdx;  // UAV for cull stats (unused by terrain)
    uint FrustumDebugMode; // Debug visualization mode
    float _frustumPad1;
};

// Accessors for push constants
uint GetOutputDescriptorsUAV()  { return Indices[0].x; }
uint GetOutputSpheresUAV()      { return Indices[0].y; }
uint GetOutputSubbatchIdsUAV()  { return Indices[0].z; }
uint GetOutputTerrainDataUAV()  { return Indices[0].w; }
uint GetCounterUAV()            { return Indices[1].x; }
uint GetMaxDepth()              { return Indices[1].y; }
uint GetTransformSlot()         { return Indices[1].z; }
uint GetMaterialId()            { return Indices[1].w; }
uint GetMeshPartId()            { return Indices[2].x; }
uint GetTotalNodes()            { return Indices[2].z; }
uint GetHeightTexIdx()          { return Indices[2].w; }
uint GetMaxPatches()            { return Indices[5].y; }
uint GetSplitFlagsUAV()         { return Indices[5].z; }
float GetMaxHeight()            { return asfloat(Indices[5].w); }
uint GetHeightRangeSRV()        { return Indices[6].x; }
float2 GetTerrainSize()         { return float2(asfloat(Indices[6].y), asfloat(Indices[6].z)); }
uint GetBuildMip()              { return Indices[6].w; }
float GetPixelErrorThreshold()  { return asfloat(Indices[7].x); }
float GetScreenHeight()         { return asfloat(Indices[7].y); }
float GetTanHalfFov()           { return asfloat(Indices[7].z); }
uint GetMipInputSRV()           { return Indices[7].y; }
uint GetMipOutputUAV()          { return Indices[7].z; }
uint GetIndirectArgsUAV()       { return Indices[7].w; }

// We need to decode camera position and root center/extents from push constants.
// Since float3 spans across uint4 boundaries, decode manually:
// Sampler for height texture reads (static sampler s2 = ClampedBilinear)
SamplerState sampHeightCS : register(s2);

float3 DecodeCameraPos()
{
    return float3(asfloat(Indices[3].x), asfloat(Indices[3].y), asfloat(Indices[3].z));
}

float3 DecodeRootCenter()
{
    return float3(asfloat(Indices[3].w), asfloat(Indices[4].x), asfloat(Indices[4].y));
}

float3 DecodeRootExtents()
{
    return float3(asfloat(Indices[4].z), asfloat(Indices[4].w), asfloat(Indices[5].x));
}

// UAVs accessed via ResourceDescriptorHeap (bindless)
// Output buffers are written via atomic append

// ─────────────────────────────────────────────────────────────────────
// Frustum + Hi-Z occlusion helpers (adapted from cull_instances.hlsl)
// Frustum test: sphere vs 6 frustum planes.
// ─────────────────────────────────────────────────────────────────────
bool IsFrustumVisible(float3 center, float radius)
{
    float4 planes[6] = { Plane0, Plane1, Plane2, Plane3, Plane4, Plane5 };
    for (uint i = 0; i < 6; i++)
    {
        float dist = dot(planes[i].xyz, center) + planes[i].w;
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
    
    uint maxDepth = GetMaxDepth();
    
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
    Texture2D<float2> heightRange = ResourceDescriptorHeap[GetHeightRangeSRV()];
    
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
    float screenHeight = GetScreenHeight();
    float tanHalfFov = GetTanHalfFov();
    float screenError = (geometricError * screenHeight) / (2.0 * dist * tanHalfFov);

    return screenError > GetPixelErrorThreshold();
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
    if (depth > GetMaxDepth()) return;
    
    uint flatIdx = LevelStart(depth) + localIdx;
    RWByteAddressBuffer splitFlags = ResourceDescriptorHeap[GetSplitFlagsUAV()];
    
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
    uint totalNodes = GetTotalNodes();
    
    if (nodeIndex >= totalNodes)
        return;
    
    uint maxDepth = GetMaxDepth();
    float3 cameraPos = DecodeCameraPos();
    float3 rootCenter = DecodeRootCenter();
    float3 rootExtents = DecodeRootExtents();
    
    // Decompose flat index into depth level and local index
    uint localIdx;
    uint depth = DecomposeIndex(nodeIndex, localIdx);
    
    // Can't split at max depth
    if (depth >= maxDepth)
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
    uint totalNodes = GetTotalNodes();
    
    if (nodeIndex >= totalNodes)
        return;
    
    uint maxDepth = GetMaxDepth();
    float3 rootCenter = DecodeRootCenter();
    float3 rootExtents = DecodeRootExtents();
    float maxHeight = rootExtents.y;
    
    // Decompose flat index into depth level and local index
    uint localIdx;
    uint depth = DecomposeIndex(nodeIndex, localIdx);
    
    // Read split flags
    RWByteAddressBuffer splitFlags = ResourceDescriptorHeap[GetSplitFlagsUAV()];
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

    // Hi-Z occlusion cull: skip patches fully behind solid geometry
    if (IsTerrainOccluded(sphereCenter, sphereRadius))
        return;
    
    // Atomic append to output — get slot index
    RWByteAddressBuffer counterBuffer = ResourceDescriptorHeap[GetCounterUAV()];
    uint patchIdx;
    counterBuffer.InterlockedAdd(0, 1, patchIdx);
    
    // Safety: don't write past buffer capacity
    uint maxPatches = GetMaxPatches();
    if (patchIdx >= maxPatches)
        return;
    
    // Write InstanceDescriptor
    RWStructuredBuffer<InstanceDescriptor> outDescriptors = ResourceDescriptorHeap[GetOutputDescriptorsUAV()];
    InstanceDescriptor desc;
    desc.TransformSlot = GetTransformSlot();
    desc.MaterialId = GetMaterialId();
    desc.CustomDataIdx = 0;
    outDescriptors[patchIdx] = desc;
    
    // Write BoundingSphere (already computed above for culling)
    RWStructuredBuffer<float4> outSpheres = ResourceDescriptorHeap[GetOutputSpheresUAV()];
    outSpheres[patchIdx] = float4(sphereCenter, sphereRadius);
    
    // Write SubbatchId (all terrain patches use the same mesh)
    RWStructuredBuffer<uint> outSubbatchIds = ResourceDescriptorHeap[GetOutputSubbatchIdsUAV()];
    outSubbatchIds[patchIdx] = GetMeshPartId();
    
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
    RWStructuredBuffer<TerrainPatchData> outTerrainData = ResourceDescriptorHeap[GetOutputTerrainDataUAV()];
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
    uint buildMip = GetBuildMip();
    
    RWTexture2D<float2> output = ResourceDescriptorHeap[GetMipOutputUAV()];
    
    // Get output dimensions
    uint outW, outH;
    output.GetDimensions(outW, outH);
    
    if (dtid.x >= outW || dtid.y >= outH)
        return;
    
    if (buildMip == 0)
    {
        // Mip 0: sample from the source heightmap
        // Each output texel covers a region of the heightmap; sample a 4x4 grid
        Texture2D<float4> heightTex = ResourceDescriptorHeap[GetHeightTexIdx()];
        float maxH = GetMaxHeight();
        
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
        Texture2D<float2> input = ResourceDescriptorHeap[GetMipInputSRV()];
        
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
    RWByteAddressBuffer counterBuffer = ResourceDescriptorHeap[GetCounterUAV()];
    uint patchCount = counterBuffer.Load(0);
    
    // Clamp to buffer capacity
    patchCount = min(patchCount, GetMaxPatches());
    
    // Write DrawInstanced args: (VertexCountPerInstance, InstanceCount, StartVertex, StartInstance)
    // VertexCountPerInstance = total index count of the terrain patch mesh.
    // C# repurposes Indices[7].y (MipInputSRV) to pass the mesh index count before this dispatch.
    RWByteAddressBuffer argsBuffer = ResourceDescriptorHeap[GetIndirectArgsUAV()];
    argsBuffer.Store(0, GetMipInputSRV());   // VertexCountPerInstance (mesh index count from C#)
    argsBuffer.Store(4, patchCount);         // InstanceCount = number of visible patches
    argsBuffer.Store(8, 0u);                 // StartVertexLocation = 0
    argsBuffer.Store(12, 0u);                // StartInstanceLocation = 0
}
