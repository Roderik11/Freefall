// GPU Frustum Culling with Hierarchical Prefix Sum Stream Compaction
// 5-pass pipeline for proper large-array support:
// 1. CSVisibility - frustum test, write visibility flags
// 2. CSLocalScan - local prefix sum per block, output block sums
// 3. CSBlockScan - scan the block sums
// 4. CSGlobalScatter - add block prefix, scatter to compacted output
// 5. CSMain - generate indirect draw commands

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
    float _pad1;
};

// Shadow cascade frustum planes (root slot 2 -> register b1)
// 6 planes per cascade × 4 cascades = 24 float4s
cbuffer ShadowCascadePlanes : register(b1)
{
    float4 ShadowPlanes[24];
};

// Push constants for bindless buffer indices (root slot 0 -> register b3)
cbuffer PushConstants : register(b3)
{
    uint4 Indices[8];
};

// Root constant mappings
#define MeshRegistryIdx       Indices[0].x   // SRV: global mesh/part registry (replaces templates)
#define OutputBufferIdx       Indices[0].y   // UAV: output commands
#define PrevVisibilityIdx     Indices[0].z   // SRV: previous frame's visibility flags (feedback loop breaker)
#define _Unused0w             Indices[0].w   // (was InstanceRangesIdx, now unused)

#define DescriptorBufferIdx   Indices[1].x   // SRV: per-instance InstanceDescriptor buffer (TransformSlot, MaterialId, CustomDataIdx)
#define VisibleIndicesUAVIdx  Indices[1].y   // UAV: output visible indices (compacted)
#define CounterBufferIdx      Indices[1].z   // UAV: per-subbatch visible counts
#define SubBatchCount         Indices[1].w   // Total number of subbatches

#define CounterOffset         Indices[2].x   // Offset into counter buffer
#define VisibleIndicesSRVIdx  Indices[2].y   // SRV: visible indices for rendering
#define GlobalTransformsIdx   Indices[2].z   // SRV: global transform buffer
#define VisibilityFlagsIdx    Indices[2].w   // UAV: visibility flags (1=visible, 0=not)

#define PrefixSumIdx          Indices[3].x   // UAV: prefix sum buffer
#define TotalInstances        Indices[3].y   // Total instance count
#define BlockSumsIdx          Indices[3].z   // UAV: block sums for hierarchical scan
#define NumBlocks             Indices[3].w   // Number of blocks in scan

#define IndirectionIdx        Indices[4].w   // UAV: maps group-order -> draw-order index (writable for sort)

// Per-subbatch sort parameters (used during CSSortIndirection pass)
#define SubbatchStart         Indices[4].x   // Starting index of current subbatch
#define SubbatchCount         Indices[4].y   // Number of instances in current subbatch

// Sort parameters (used during CSSortIndirection and CSBitonicSort passes)
#define SortSize              Indices[5].x   // Block size for this merge step
#define SortStride            Indices[5].y   // Stride within block
#define SortCount             Indices[5].z   // Total count to sort (subbatch count or visible count)
#define SubbatchIdsIdx        Indices[5].w   // SRV: per-instance subbatch index

// Per-batch indices (set by CPU for each batch)
#define ShadowVisFlags0Idx    Indices[6].x   // UAV: shadow cascade 0 visibility flags (CSVisibilityShadow4)
#define MaterialsBufferIdx    Indices[6].y   // SRV: materials array buffer
#define HistogramIdx          Indices[6].z   // UAV: histogram buffer (count per MeshPartId)
#define UniquePartCount       Indices[6].w   // Max unique MeshPartIds in registry
#define ShadowVisFlags1Idx    Indices[7].x   // UAV: shadow cascade 1 visibility flags (CSVisibilityShadow4)
#define ShadowVisFlags2Idx    Indices[7].y   // UAV: shadow cascade 2 visibility flags (CSVisibilityShadow4)
#define BoneBufferIdx         Indices[7].z   // SRV: per-batch bone matrices buffer (0 for static batches)
#define ShadowCascadeIdx      Indices[7].w   // Shadow cascade index (0-3) / cascade 3 vis flags UAV

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
    uint NumBones;
    // Local-space bounding sphere (center + radius) for GPU culling
    float4 LocalBounds;  // xyz = center, w = radius
    // Reserved fields for padding to match IndirectDrawCommand size
    uint Reserved4;
    uint Reserved5;
    uint Reserved6;
    uint Reserved7;
    uint Reserved8;
    uint Reserved9;
};

// Per-instance descriptor (matches C# InstanceDescriptor exactly: 12 bytes = 3 uints)
struct InstanceDescriptor
{
    uint TransformSlot;
    uint MaterialId;
    uint CustomDataIdx;
};

// Must match C# IndirectDrawCommand exactly (72 bytes = 18 uints)
struct IndirectDrawCommand
{
    uint DescriptorBufIdx;
    uint Reserved0;
    uint SortedIndicesBufIdx;
    uint BoneWeightsBufIdx;
    uint BonesBufIdx;
    uint IndexBufIdx;
    uint BaseIndex;
    uint PosBufIdx;
    uint NormBufIdx;
    uint UVBufIdx;
    uint NumBones;
    uint InstanceBaseOffset;
    uint MaterialsIdx;
    uint GlobalTransformBufIdx;
    uint VertexCountPerInstance;
    uint DrawInstanceCount;
    uint StartVertexLocation;
    uint StartInstanceLocation;
};

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
    
    // Project sphere radius to screen pixels for mip selection
    // Fix: Include projection scale (cot(fov/2) == _m11) for correct screen size
    float projScale = OcclusionProjection._m11; 
    // Protect against zero projection
    projScale = abs(projScale) < 0.001 ? 1.0 : projScale;
    
    float projRadius = (worldRadius * projScale) / clipCenter.w;
    float screenRadius = projRadius * mip0Size.y * 0.5;
    
    // Pick mip level where the sphere covers ~2 texels (more conservative than 1)
    // This ensures we sample a mip where the object is significant
    float mipLevel = ceil(log2(max(screenRadius * 2.0, 1.0)));
    mipLevel = min(mipLevel, levels - 1.0f);
    
    uint mip = (uint)mipLevel;
    float2 mipSize = max(float2(1,1), mip0Size / (float)(1u << mip));
    
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
bool IsVisibleShadow(float3 center, float radius, uint cascadeIndex)
{
    uint planeOffset = cascadeIndex * 6;
    
    [unroll]
    for (uint i = 0; i < 6; i++)
    {
        float4 plane = ShadowPlanes[planeOffset + i];
        float dist = dot(plane.xyz, center) + plane.w;
        if (dist > radius)
            return false;
    }
    return true;
}

//-----------------------------------------------------------------------------
// Pass 1: CSVisibility - Write visibility flags (1 if visible, 0 if not)
// Also counts visible instances per subbatch for command generation
//-----------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void CSVisibility(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint instanceIdx = dispatchThreadId.x;
    
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufferIdx];
    StructuredBuffer<row_major float4x4> transforms = ResourceDescriptorHeap[GlobalTransformsIdx];
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    StructuredBuffer<uint> subbatchIds = ResourceDescriptorHeap[SubbatchIdsIdx];
    RWStructuredBuffer<uint> visibilityFlags = ResourceDescriptorHeap[VisibilityFlagsIdx];
    RWStructuredBuffer<uint> counters = ResourceDescriptorHeap[CounterBufferIdx];
    
    // Bounds check - write 0 for out-of-range to clear stale visibility flags
    if (instanceIdx >= TotalInstances)
    {
        visibilityFlags[instanceIdx] = 0;
        return;
    }
    
    // subbatchIds stores meshPartId directly (set in CommandBuffer.Enqueue)
    uint subBatch = subbatchIds[instanceIdx];
    
    // Data is stored in add-order, instanceIdx == drawIdx (identity mapping)
    uint transformSlot = descriptors[instanceIdx].TransformSlot;
    
    // Look up local bounding sphere from mesh registry (persistent, not per-instance)
    float4 localSphere = meshRegistry[subBatch].LocalBounds;
    
    // Transform sphere to world space
    row_major float4x4 world = transforms[transformSlot];
    float3 worldCenter = mul(float4(localSphere.xyz, 1.0), world).xyz;
    
    // Scale radius by maximum axis scale
    float3 scale = float3(
        length(world[0].xyz),
        length(world[1].xyz),
        length(world[2].xyz)
    );
    float maxScale = max(scale.x, max(scale.y, scale.z));
    float worldRadius = localSphere.w * maxScale;
    
    // Frustum test
    bool visible = IsVisible(worldCenter, worldRadius);
    
    // Hi-Z occlusion test (only for frustum-visible instances)
    // Small distance-proportional radius floor prevents false positives when
    // per-part AABB centers fall in gaps of open geometry (fences, bridges).
    if (visible)
    {
        float4 hiZClip = mul(float4(worldCenter, 1.0), OcclusionProjection);
        float hiZRadius = max(worldRadius, hiZClip.w * 0.01);
        if (IsOccluded(worldCenter, hiZRadius))
            visible = false;
    }
    
    // Write visibility flag: 0 = culled, 1 = visible
    uint flag = visible ? 1u : 0u;
    visibilityFlags[instanceIdx] = flag;
    
    // Update cull stats (atomic, across all threads)
    if (CullStatsUAVIdx != 0)
    {
        RWByteAddressBuffer stats = ResourceDescriptorHeap[CullStatsUAVIdx];
        if (flag == 1)
            stats.InterlockedAdd(0, 1);    // [0] = frustum+HiZ visible
        else if (IsVisible(worldCenter, worldRadius))
            stats.InterlockedAdd(4, 1);    // [1] = passed frustum, failed Hi-Z
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
    
    // Read visibility flag from CSVisibility output (1=visible)
    RWStructuredBuffer<uint> visibilityFlags = ResourceDescriptorHeap[VisibilityFlagsIdx];
    uint flag = visibilityFlags[instanceIdx];
    
    if (flag == 0)
        return;
    
    // Read MeshPartId for this instance (stored in subbatchIds, which is actually MeshPartId)
    StructuredBuffer<uint> meshPartIds = ResourceDescriptorHeap[SubbatchIdsIdx];
    uint meshPartId = meshPartIds[instanceIdx];
    
    // Atomically increment histogram counter for this MeshPartId
    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramIdx];
    InterlockedAdd(histogram[meshPartId], 1);
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
    
    sharedData[tid] = (idx1 < TotalInstances) ? flags[idx1] : 0;
    sharedData[tid + 256] = (idx2 < TotalInstances) ? flags[idx2] : 0;
    
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
    StructuredBuffer<uint> subbatchIds = ResourceDescriptorHeap[SubbatchIdsIdx];
    RWStructuredBuffer<uint> counters = ResourceDescriptorHeap[CounterBufferIdx];
    RWStructuredBuffer<uint> visibleOut = ResourceDescriptorHeap[VisibleIndicesUAVIdx];
    
    uint flag = visibilityFlags[instanceIdx];
    if (flag >= 1)
    {
        uint meshPartId = subbatchIds[instanceIdx];
        
        uint outputIdx;
        InterlockedAdd(counters[meshPartId], 1, outputIdx);
        
        visibleOut[outputIdx] = instanceIdx;
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
// Pass 5a: CSHistogramPrefixSum - Prefix sum of histogram for StartInstance offsets
// Runs once per MeshPartId slot (O(K) where K = registry size)
// Input: histogram[k] = visible count for MeshPartId k
// Output: counters[k] = exclusive prefix sum = StartInstance offset for MeshPartId k
//-----------------------------------------------------------------------------
[numthreads(1, 1, 1)]
void CSHistogramPrefixSum(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramIdx];
    RWStructuredBuffer<uint> counters = ResourceDescriptorHeap[CounterBufferIdx];
    
    // Sequential prefix sum — reads directly from histogram buffer
    // No groupshared size limit, supports up to MaxMeshParts (4096)
    uint sum = 0;
    for (uint i = 0; i < UniquePartCount; i++)
    {
        uint count = histogram[i];
        counters[i] = sum;  // Exclusive prefix sum -> StartInstance offset
        sum += count;
    }
}

//-----------------------------------------------------------------------------
// Pass 5b: CSMain - Generate final indirect draw commands from MeshRegistry
// Fully GPU-driven: iterates over MeshRegistry, reads counts from histogram
// Generates one command per MeshPartId with non-zero visible count
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
    
    // Skip MeshPartIds with no visible instances
    if (visibleCount == 0)
    {
        RWStructuredBuffer<IndirectDrawCommand> output = ResourceDescriptorHeap[OutputBufferIdx];
        IndirectDrawCommand emptyCmd = (IndirectDrawCommand)0;
        output[meshPartId] = emptyCmd;
        return;
    }
    
    // After CSGlobalScatter, counters[k] has been incremented by visibleCount
    // Original StartInstance = counters[k] - visibleCount
    uint counterVal = counters[meshPartId];
    uint baseOffset = (counterVal >= visibleCount) ? (counterVal - visibleCount) : 0;
    
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry entry = meshRegistry[meshPartId];
    
    RWStructuredBuffer<IndirectDrawCommand> output = ResourceDescriptorHeap[OutputBufferIdx];
    
    IndirectDrawCommand cmd;
    cmd.DescriptorBufIdx = DescriptorBufferIdx;
    cmd.Reserved0 = 0;
    cmd.SortedIndicesBufIdx = VisibleIndicesSRVIdx;
    cmd.BoneWeightsBufIdx = entry.BoneWeightsBufferIdx;
    cmd.BonesBufIdx = BoneBufferIdx;
    cmd.IndexBufIdx = entry.IndexBufferIdx;
    cmd.BaseIndex = entry.BaseIndex;
    cmd.PosBufIdx = entry.PosBufferIdx;
    cmd.NormBufIdx = entry.NormBufferIdx;
    cmd.UVBufIdx = entry.UVBufferIdx;
    cmd.NumBones = entry.NumBones;
    cmd.InstanceBaseOffset = baseOffset;
    cmd.MaterialsIdx = MaterialsBufferIdx;
    cmd.GlobalTransformBufIdx = GlobalTransformsIdx;
    cmd.VertexCountPerInstance = entry.VertexCount;
    cmd.DrawInstanceCount = visibleCount;
    cmd.StartVertexLocation = 0;
    cmd.StartInstanceLocation = 0;
    
    output[meshPartId] = cmd;
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
    
    // Get transform and bounding sphere from mesh registry
    uint transformSlot = descriptors[instanceIdx].TransformSlot;
    uint meshPartId = subbatchIds[instanceIdx];
    float4 localSphere = meshRegistry[meshPartId].LocalBounds;
    
    // Transform sphere to world space
    row_major float4x4 world = transforms[transformSlot];
    float3 worldCenter = mul(float4(localSphere.xyz, 1.0), world).xyz;
    
    // Scale radius by maximum axis scale
    float3 scale = float3(
        length(world[0].xyz),
        length(world[1].xyz),
        length(world[2].xyz)
    );
    float maxScale = max(scale.x, max(scale.y, scale.z));
    float worldRadius = localSphere.w * maxScale;
    
    // For skinned meshes (BoneBufferIdx != 0), the bounding sphere is static (bind pose)
    // and doesn't account for animation. Inflate it to prevent culling when limbs move outside.
    if (BoneBufferIdx != 0)
    {
        worldRadius *= 1.5;
    }
    
    // Frustum test against selected shadow cascade
    bool visible = IsVisibleShadow(worldCenter, worldRadius, ShadowCascadeIdx);
    
    // Write visibility flag
    visibilityFlags[instanceIdx] = visible ? 1 : 0;
}

//-----------------------------------------------------------------------------
// CSVisibilityShadow4: Unified 4-cascade shadow visibility pass
// Tests all 4 cascade frustums per instance in a single dispatch.
// Reads each instance's transform ONCE (not 4×), writes to 4 separate
// visibility flag buffers via push constant UAV indices.
//-----------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void CSVisibilityShadow4(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint instanceIdx = dispatchThreadId.x;
    
    // Get all 4 output buffers
    RWStructuredBuffer<uint> visFlags0 = ResourceDescriptorHeap[ShadowVisFlags0Idx];
    RWStructuredBuffer<uint> visFlags1 = ResourceDescriptorHeap[ShadowVisFlags1Idx];
    RWStructuredBuffer<uint> visFlags2 = ResourceDescriptorHeap[ShadowVisFlags2Idx];
    RWStructuredBuffer<uint> visFlags3 = ResourceDescriptorHeap[ShadowCascadeIdx]; // Reuses slot 31
    
    // Bounds check - write 0 for out-of-range
    if (instanceIdx >= TotalInstances)
    {
        visFlags0[instanceIdx] = 0;
        visFlags1[instanceIdx] = 0;
        visFlags2[instanceIdx] = 0;
        visFlags3[instanceIdx] = 0;
        return;
    }
    
    // Load instance data ONCE (instead of 4× in separate dispatches)
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufferIdx];
    StructuredBuffer<row_major float4x4> transforms = ResourceDescriptorHeap[GlobalTransformsIdx];
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    StructuredBuffer<uint> subbatchIds = ResourceDescriptorHeap[SubbatchIdsIdx];
    
    uint transformSlot = descriptors[instanceIdx].TransformSlot;
    uint meshPartId = subbatchIds[instanceIdx];
    float4 localSphere = meshRegistry[meshPartId].LocalBounds;
    
    // Transform sphere to world space (done ONCE)
    row_major float4x4 world = transforms[transformSlot];
    float3 worldCenter = mul(float4(localSphere.xyz, 1.0), world).xyz;
    
    float3 scale = float3(
        length(world[0].xyz),
        length(world[1].xyz),
        length(world[2].xyz)
    );
    float maxScale = max(scale.x, max(scale.y, scale.z));
    float worldRadius = localSphere.w * maxScale;
    
    // Inflate for skinned meshes (same as CSVisibilityShadow)
    if (BoneBufferIdx != 0)
        worldRadius *= 1.5;
    
    // Test all 4 cascade frustums
    visFlags0[instanceIdx] = IsVisibleShadow(worldCenter, worldRadius, 0) ? 1 : 0;
    visFlags1[instanceIdx] = IsVisibleShadow(worldCenter, worldRadius, 1) ? 1 : 0;
    visFlags2[instanceIdx] = IsVisibleShadow(worldCenter, worldRadius, 2) ? 1 : 0;
    visFlags3[instanceIdx] = IsVisibleShadow(worldCenter, worldRadius, 3) ? 1 : 0;
}
