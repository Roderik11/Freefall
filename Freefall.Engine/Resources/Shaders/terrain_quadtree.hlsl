// GPU-Driven Quadtree Terrain — Restricted Quadtree (Two-Pass)
// Pass 1 (CSMarkSplits): Each thread evaluates its node, marks split flags, and forces
//   spatial neighbors to split (restricted quadtree constraint: ≤1 level difference).
// Pass 2 (CSEmitLeaves): Reads enforced split flags and emits visible leaf patches.

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
//   Slot 9: LODRange0 (as float bits) — standard CDLOD base range
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
//   Slot 28: AdaptiveStrength (float bits) — how much roughness affects LOD range
//   Slot 29: MipInputSRV (source mip for CSBuildMinMaxMip)
//   Slot 30: MipOutputUAV (target mip for CSBuildMinMaxMip)

cbuffer PushConstants : register(b3)
{
    uint4 Indices[8];
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
float GetLODRange0()            { return asfloat(Indices[2].y); }
uint GetTotalNodes()            { return Indices[2].z; }
uint GetHeightTexIdx()          { return Indices[2].w; }
uint GetMaxPatches()            { return Indices[5].y; }
uint GetSplitFlagsUAV()         { return Indices[5].z; }
float GetMaxHeight()            { return asfloat(Indices[5].w); }
uint GetHeightRangeSRV()        { return Indices[6].x; }
float2 GetTerrainSize()         { return float2(asfloat(Indices[6].y), asfloat(Indices[6].z)); }
uint GetBuildMip()              { return Indices[6].w; }
float GetAdaptiveStrength()     { return asfloat(Indices[7].x); }
uint GetMipInputSRV()           { return Indices[7].y; }
uint GetMipOutputUAV()          { return Indices[7].z; }

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

// Standard CDLOD split check with per-node bounding sphere.
// Uses precomputed height range mip pyramid for tight bounding sphere distance test.
// Formula: split if distance(camera, sphere_center) - sphere_radius < LODRange[level]
bool ShouldSplit(uint depth, uint localIdx, float3 nodeCenter, float3 nodeExtents,
                 float3 cameraPos, uint maxDepth, float lodRange0)
{
    // Standard CDLOD LOD range for this node's level
    uint level = maxDepth - depth;
    float range = lodRange0 * (float)(1u << level);

    // Sample height range from precomputed mip pyramid
    float2 minmax = SampleHeightRange(depth, localIdx, maxDepth);

    // Bounding sphere center uses actual height midpoint
    float3 sphereCenter = nodeCenter;
    sphereCenter.y = (minmax.x + minmax.y) * 0.5;

    // Tight bounding sphere radius from XZ extent + height range
    float xzHalfDiag = length(float2(nodeExtents.x, nodeExtents.z));
    float heightHalfRange = (minmax.y - minmax.x) * 0.5;
    float sphereRadius = sqrt(xzHalfDiag * xzHalfDiag + heightHalfRange * heightHalfRange);

    float3 cam = cameraPos;
    // Flatten Y when camera is within terrain height range (ground-level view)
    if (cam.y < minmax.y)
    {
        cam.y = sphereCenter.y;
    }

    // Adaptive splitting: rough terrain splits at greater distances
    float roughness = (minmax.y - minmax.x) / max(GetMaxHeight(), 0.001);
    float adaptiveFactor = lerp(1.0, GetAdaptiveStrength(), saturate(roughness));
    float effectiveRange = range / adaptiveFactor;

    float dist = distance(cam, sphereCenter);
    return (dist - sphereRadius) < effectiveRange;
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
    float lodRange0 = GetLODRange0();
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
    
    // Standard CDLOD split check (with bounding sphere)
    if (!ShouldSplit(depth, localIdx, nodeCenter, nodeExtents, cameraPos, maxDepth, lodRange0))
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
    
    // Atomic append to output — get slot index
    RWByteAddressBuffer counterBuffer = ResourceDescriptorHeap[GetCounterUAV()];
    uint patchIdx;
    counterBuffer.InterlockedAdd(0, 1, patchIdx);
    
    // Safety: don't write past buffer capacity
    uint maxPatches = GetMaxPatches();
    if (patchIdx >= maxPatches)
        return;
    
    // Compute this node's bounds for bounding sphere
    float3 nodeCenter, nodeExtents;
    ComputeNodeBounds(depth, localIdx, rootCenter, rootExtents, nodeCenter, nodeExtents);
    
    // Write InstanceDescriptor
    RWStructuredBuffer<InstanceDescriptor> outDescriptors = ResourceDescriptorHeap[GetOutputDescriptorsUAV()];
    InstanceDescriptor desc;
    desc.TransformSlot = GetTransformSlot();
    desc.MaterialId = GetMaterialId();
    desc.CustomDataIdx = 0;
    outDescriptors[patchIdx] = desc;
    
    // Write BoundingSphere — tight per-node bounds from precomputed height range mip pyramid
    RWStructuredBuffer<float4> outSpheres = ResourceDescriptorHeap[GetOutputSpheresUAV()];
    float2 minmax = SampleHeightRange(depth, localIdx, maxDepth);
    float xzRadius = length(float2(nodeExtents.x, nodeExtents.z));
    float heightHalfRange = (minmax.y - minmax.x) * 0.5;
    float sphereRadius = sqrt(xzRadius * xzRadius + heightHalfRange * heightHalfRange);
    float sphereY = (minmax.x + minmax.y) * 0.5;
    outSpheres[patchIdx] = float4(nodeCenter.x, sphereY, nodeCenter.z, sphereRadius);
    
    // Write SubbatchId (all terrain patches use the same mesh)
    RWStructuredBuffer<uint> outSubbatchIds = ResourceDescriptorHeap[GetOutputSubbatchIdsUAV()];
    outSubbatchIds[patchIdx] = GetMeshPartId();
    
    // Write TerrainPatchData
    RWStructuredBuffer<TerrainPatchData> outTerrainData = ResourceDescriptorHeap[GetOutputTerrainDataUAV()];
    TerrainPatchData patchData;
    patchData.Rect = ComputeRectExact(depth, localIdx);
    // LOD level output (morph is computed per-vertex in the vertex shader via CDLOD)
    patchData.Level = float2((float)(maxDepth - depth), 0);
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
