// Grass Placement Compute Shader
// One thread per grid cell. Evaluates placement, culls, appends survivors.
// Output consumed by DrawInstancedIndirect.

// Inline push constants (can't #include from string-compiled shader)
struct PushConstantsData { uint4 indices[8]; };
ConstantBuffer<PushConstantsData> PushConstants : register(b3);
#define GET_INDEX(i) PushConstants.indices[i/4][i%4]

// Push constants layout (same root parameter 0 as grass.fx)
#define ChannelHeadersIdx   GET_INDEX(0)
#define DecoratorSlotsIdx   GET_INDEX(1)
#define LODTableIdx         GET_INDEX(2)
#define MeshRegistryIdx     GET_INDEX(3)
#define HeightmapIdx        GET_INDEX(4)
#define TerrainSizeX        asfloat(GET_INDEX(5))
#define TerrainSizeY        asfloat(GET_INDEX(6))
#define MaxHeight           asfloat(GET_INDEX(7))
#define CamPosX             asfloat(GET_INDEX(8))
#define CamPosY             asfloat(GET_INDEX(9))
#define CamPosZ             asfloat(GET_INDEX(10))
#define TerrainOriginX      asfloat(GET_INDEX(11))
#define TerrainOriginY      asfloat(GET_INDEX(12))
#define TerrainOriginZ      asfloat(GET_INDEX(13))
#define MaterialsBufferIdx  GET_INDEX(14)
#define CellSize            asfloat(GET_INDEX(15))
#define DecoRadius          asfloat(GET_INDEX(16))
#define TotalDensity        asfloat(GET_INDEX(17))
#define ControlMapsIdx      GET_INDEX(18)
#define FrustumPlanesIdx    GET_INDEX(19)
// CS-specific slots
#define GrassInstanceUAV    GET_INDEX(20)
#define GrassCounterUAV     GET_INDEX(21)
#define CellsPerSide        GET_INDEX(22)

// ─── GPU structs ───────────────────────────────────────────────────────────

struct ChannelHeader { uint StartIndex; uint Count; };

struct DecoratorSlot
{
    float Density;
    float MinH, MaxH;
    float MinW, MaxW;
    uint LODCount;
    uint LODTableOffset;
    float3x3 RootMat;
    float SlopeBias;
    uint ControlMapIndex;
    uint ControlChannel;
};

struct LODEntry { uint MeshPartId; float MaxDistance; uint MaterialId; uint _pad; };

struct MeshPartEntry
{
    uint PosBufferIdx, NormBufferIdx, UVBufferIdx, IndexBufferIdx;
    uint BaseIndex, VertexCount;
    uint BoneWeightsBufferIdx, NumBones;
    float BoundsCenterX, BoundsCenterY, BoundsCenterZ, BoundsRadius;
    uint Reserved4, Reserved5, Reserved6, Reserved7, Reserved8, Reserved9;
};

// Per-instance output — read by the VS (36 bytes)
struct GrassInstance
{
    float3 Position;        // world-space base position
    float  Scale;           // height scale
    float  Rotation;        // Y-axis rotation angle (radians)
    float  Width;           // width scale
    uint   MeshPartId;      // index into MeshRegistry
    uint   MaterialId;      // material index
    uint   DecoratorSlotIdx; // index into DecoratorSlots for RootMat lookup
};

// ─── PCG Hash ──────────────────────────────────────────────────────────────

uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint pcg2d(uint2 v)
{
    v = v * uint2(1664525u, 1013904223u) + uint2(1013904223u, 1664525u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v.x ^= v.x >> 16u;
    v.y ^= v.y >> 16u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    return v.x ^ v.y;
}

float hash21(float2 p)
{
    uint2 ip = uint2(asuint(p.x), asuint(p.y));
    return float(pcg2d(ip)) / 4294967295.0;
}

float2 hash22(float2 p)
{
    uint2 ip = uint2(asuint(p.x), asuint(p.y));
    uint h0 = pcg2d(ip);
    uint h1 = pcg(h0);
    return float2(float(h0), float(h1)) / 4294967295.0;
}

// ─── Samplers ──────────────────────────────────────────────────────────────

SamplerState HeightSampler : register(s0);
SamplerState ClampSampler  : register(s2);

// ─── CSPlacement ───────────────────────────────────────────────────────────
// One thread per grid cell. Evaluates density, heightmap, control map,
// frustum/distance cull, LOD. Survivors appended to output buffer.

[numthreads(8, 8, 1)]
void CSPlacement(uint3 dtid : SV_DispatchThreadID)
{
    uint cps = CellsPerSide;
    uint cellX = dtid.x;
    uint cellZ = dtid.y;
    if (cellX >= cps || cellZ >= cps) return;

    float cs = CellSize;
    float range = DecoRadius;
    float3 camPos = float3(CamPosX, CamPosY, CamPosZ);

    // World position — snap grid to cell boundaries for stable positions
    float gridStartX = floor((camPos.x - range) / cs) * cs;
    float gridStartZ = floor((camPos.z - range) / cs) * cs;
    float wx = gridStartX + (cellX + 0.5) * cs;
    float wz = gridStartZ + (cellZ + 0.5) * cs;

    // Seed based on integer cell coords (world-stable)
    float2 seed = float2(floor(wx / cs), floor(wz / cs));

    // Terrain bounds check
    float2 texelUV = float2(
        (wx - TerrainOriginX) / TerrainSizeX,
        (wz - TerrainOriginY) / TerrainSizeY
    );
    if (texelUV.x < 0 || texelUV.x > 1 || texelUV.y < 0 || texelUV.y > 1) return;

    // Distance cull
    float2 cellCenter = float2(wx, wz);
    float tileDist = distance(float2(camPos.x, camPos.z), cellCenter);
    if (tileDist >= range) return;

    // Frustum cull (bounding sphere at ground level)
    if (FrustumPlanesIdx != 0)
    {
        StructuredBuffer<float4> planes = ResourceDescriptorHeap[FrustumPlanesIdx];
        float groundMid = TerrainOriginZ + MaxHeight * 0.5;
        float verticalRadius = MaxHeight * 0.5;
        float3 sphereCenter = float3(wx, groundMid, wz);
        float sphereRadius = sqrt(cs * cs * 0.5 + verticalRadius * verticalRadius);

        // Test 5 planes: skip far (distance handles it)
        for (uint pi = 0; pi < 5; pi++)
        {
            float4 plane = planes[pi];
            if (dot(plane.xyz, sphereCenter) + plane.w > sphereRadius) return;
        }
    }

    // Load decorator data
    StructuredBuffer<ChannelHeader> headers = ResourceDescriptorHeap[ChannelHeadersIdx];
    StructuredBuffer<DecoratorSlot> slots   = ResourceDescriptorHeap[DecoratorSlotsIdx];
    StructuredBuffer<LODEntry>      lodTbl  = ResourceDescriptorHeap[LODTableIdx];

    ChannelHeader hdr = headers[0];
    if (hdr.Count == 0) return;

    // Independent per-decorator spawn
    float cellArea = cs * cs;
    uint winner = 0xFFFF;
    float bestHash = 2.0;

    for (uint s = 0; s < hdr.Count && s < 32; s++)
    {
        DecoratorSlot candidate = slots[hdr.StartIndex + s];
        float threshold = candidate.Density * cellArea;
        float h = hash21(seed + float2(s * 127.1, s * 311.7));
        if (h < threshold && h < bestHash)
        {
            bestHash = h;
            winner = s;
        }
    }
    if (winner == 0xFFFF) return;

    uint slotIdx = hdr.StartIndex + winner;
    DecoratorSlot slot = slots[slotIdx];

    // Heightmap sample
    Texture2D heightmap = ResourceDescriptorHeap[HeightmapIdx];
    float2 jitter = hash22(seed + 73.1) - 0.5;
    float2 jitteredUV = float2(
        (wx + jitter.x * cs - TerrainOriginX) / TerrainSizeX,
        (wz + jitter.y * cs - TerrainOriginY) / TerrainSizeY
    );
    float h = heightmap.SampleLevel(ClampSampler, jitteredUV, 0).r * MaxHeight + TerrainOriginZ;

    float3 instancePos = float3(wx + jitter.x * cs, h, wz + jitter.y * cs);

    // Control map scale/cull
    float ctrlScale = 1.0;
    if (ControlMapsIdx != 0)
    {
        Texture2DArray controlMaps = ResourceDescriptorHeap[ControlMapsIdx];
        float2 ctrlUV = float2(
            (instancePos.x - TerrainOriginX) / TerrainSizeX,
            1.0 - (instancePos.z - TerrainOriginY) / TerrainSizeY
        );
        float4 ctrlSample = controlMaps.SampleLevel(ClampSampler, float3(ctrlUV, slot.ControlMapIndex), 0);
        float channels[4] = { ctrlSample.r, ctrlSample.g, ctrlSample.b, ctrlSample.a };
        ctrlScale = channels[slot.ControlChannel];
        ctrlScale = smoothstep(0.15, 0.5, ctrlScale);
        if (ctrlScale < 0.01) return;
    }

    // Instance random scale and rotation
    float instanceScale = lerp(slot.MinH, slot.MaxH, hash21(seed + 33.7)) * ctrlScale;
    float instanceRot = hash21(seed + 51.3) * 6.2831853;
    float instanceWidth = lerp(slot.MinW, slot.MaxW, hash21(seed + 77.9));

    // LOD selection
    float distToCamera = distance(instancePos, camPos);
    uint meshPartId = 0;
    uint materialId = 0;
    bool lodFound = false;
    for (uint li = 0; li < slot.LODCount && li < 4; li++)
    {
        LODEntry lod = lodTbl[slot.LODTableOffset + li];
        if (distToCamera < lod.MaxDistance)
        {
            meshPartId = lod.MeshPartId;
            materialId = lod.MaterialId;
            lodFound = true;
            break;
        }
    }
    if (!lodFound) return;
    
    // Distance fade: smoothly shrink to zero in the last 25% of range
    float fadeStart = range * 0.75;
    float fadeFactor = smoothstep(0.0, 1.0, (range - distToCamera) / (range - fadeStart));
    instanceScale *= fadeFactor;
    instanceWidth *= fadeFactor;

    // Append to output buffer
    RWStructuredBuffer<GrassInstance> outBuffer = ResourceDescriptorHeap[GrassInstanceUAV];
    RWByteAddressBuffer counter = ResourceDescriptorHeap[GrassCounterUAV];

    uint idx;
    counter.InterlockedAdd(0, 1, idx);

    // Safety: don't overflow the buffer
    if (idx >= cps * cps) return;

    // Track actual max vertex count among survivors (offset 4 in counter buffer)
    StructuredBuffer<MeshPartEntry> meshReg = ResourceDescriptorHeap[MeshRegistryIdx];
    uint vertCount = meshReg[meshPartId].VertexCount;
    uint dummy;
    counter.InterlockedMax(4, vertCount, dummy);

    GrassInstance gi;
    gi.Position = instancePos;
    gi.Scale = instanceScale;
    gi.Rotation = instanceRot;
    gi.Width = instanceWidth;
    gi.MeshPartId = meshPartId;
    gi.MaterialId = materialId;
    gi.DecoratorSlotIdx = slotIdx;
    outBuffer[idx] = gi;
}

// ─── CSBuildDrawArgs ───────────────────────────────────────────────────────
// Single-thread: reads append counter + actual max vertex count, writes DrawInstancedIndirect args.

[numthreads(1, 1, 1)]
void CSBuildDrawArgs(uint3 dtid : SV_DispatchThreadID)
{
    RWByteAddressBuffer counter = ResourceDescriptorHeap[GrassCounterUAV];
    uint instanceCount = counter.Load(0);
    uint maxVertexCount = counter.Load(4);  // actual max from InterlockedMax

    // Clamp to buffer capacity
    uint cps = CellsPerSide;
    instanceCount = min(instanceCount, cps * cps);

    RWByteAddressBuffer argsBuffer = ResourceDescriptorHeap[GET_INDEX(24)]; // DrawArgsUAV
    argsBuffer.Store(0, maxVertexCount);      // VertexCountPerInstance (actual, not global max)
    argsBuffer.Store(4, instanceCount);       // InstanceCount
    argsBuffer.Store(8, 0u);                  // StartVertexLocation
    argsBuffer.Store(12, 0u);                 // StartInstanceLocation
}
