// Sparse 3D Radiance Cascades — compute shader
// Per-cascade angular scaling with interval storage and front-to-back merging
// Kernels: CSMark, CSPrepareIndirect, CSTrace0..3, CSShade

#pragma kernel CSMark
#pragma kernel CSPrepareIndirect
#pragma kernel CSTrace0
#pragma kernel CSTrace1
#pragma kernel CSTrace2
#pragma kernel CSTrace3
#pragma kernel CSMerge0
#pragma kernel CSMerge1
#pragma kernel CSMerge2
#pragma kernel CSShade
#pragma kernel CSComposeGI

// ─────────────────────── Push Constants ───────────────────────

cbuffer PushConstants : register(b3)
{
    // Per-level hash map entries (UAV for Mark, SRV for Shade)
    uint HEntries0Idx;
    uint HEntries1Idx;
    uint HEntries2Idx;
    uint HEntries3Idx;
    // Per-level hash map counters (UAV)
    uint HCounter0Idx;
    uint HCounter1Idx;
    uint HCounter2Idx;
    uint HCounter3Idx;
    // Per-level tile pools (UAV)
    uint TPool0Idx;
    uint TPool1Idx;
    uint TPool2Idx;
    uint TPool3Idx;
    // Per-level tile info (UAV)
    uint TInfo0Idx;
    uint TInfo1Idx;
    uint TInfo2Idx;
    uint TInfo3Idx;
    // Globals
    uint HCapMaskIdx;
    uint ScreenWIdx;
    uint ScreenHIdx;
    uint DepthGBufIdx;
    uint NormalTexIdx;
    uint AlbedoTexIdx;
    uint LightTexIdx;
    uint GIOutputIdx;
    uint RCIntensityIdx;
    uint CurLevelIdx;
    uint IndArgsIdx;
    uint RCDebugModeIdx; // 0=off, 1=tile lookup, 2=radiance per level
    uint DataTexIdx;     // GBuffer data (roughness, metalness, AO)
    uint LightBufUavIdx; // LightBuffer UAV for GI composition
    uint PrevGIIdx;      // Previous frame GI (SRV) for temporal blend
    uint HiZTexIdx;      // Hi-Z depth pyramid (all mips) for coarse-level traces
};

#include "common.fx"

SamplerState LinearSampler : register(s0);

// ─────────────────────── Constants ───────────────────────

#define CASCADE_COUNT     4
#define BASE_OCT_RES      4
#define BASE_GRID_SIZE    0.25
#define BASE_INTERVAL     0.25
#define TRACE_STEPS_BASE  16     // L0 steps; doubles per level
#define TILE_SCREEN_SIZE  8.0
#define SHADE_OCT_RES     6      // hemisphere integral resolution (6×6 = 36 dirs)
#define HASH_EMPTY        0xFFFFFFFF
#define POOL_INVALID      0xFFFFFFFF
#define PI                3.14159265

// Per-level max tiles (must match C# MaxTilesPerLevel)
uint MaxTilesForLevel(uint level)
{
    // L0: 262144, L1: 65536, L2: 16384, L3: 4096
    return 262144u >> (level * 2);
}

// ─────────────────────── Per-Level Helpers ───────────────────────

uint GetHEntriesIdx(uint level)
{
    if (level == 0) return HEntries0Idx;
    if (level == 1) return HEntries1Idx;
    if (level == 2) return HEntries2Idx;
    return HEntries3Idx;
}

uint GetHCounterIdx(uint level)
{
    if (level == 0) return HCounter0Idx;
    if (level == 1) return HCounter1Idx;
    if (level == 2) return HCounter2Idx;
    return HCounter3Idx;
}

uint GetTPoolIdx(uint level)
{
    if (level == 0) return TPool0Idx;
    if (level == 1) return TPool1Idx;
    if (level == 2) return TPool2Idx;
    return TPool3Idx;
}

uint GetTInfoIdx(uint level)
{
    if (level == 0) return TInfo0Idx;
    if (level == 1) return TInfo1Idx;
    if (level == 2) return TInfo2Idx;
    return TInfo3Idx;
}

uint OctResForLevel(uint level)  { return (uint)BASE_OCT_RES << level; }
uint DirsForLevel(uint level)    { uint r = OctResForLevel(level); return r * r; }
float GridSizeForLevel(uint level) { return BASE_GRID_SIZE * float(1u << level); }

float2 GetIntervalRange(uint level)
{
    float end   = BASE_INTERVAL * float(1u << (2u * level));
    float start = (level == 0) ? 0.0 : BASE_INTERVAL * float(1u << (2u * (level - 1u)));
    return float2(start, end);
}

// ─────────────────────── Hash Map ───────────────────────

struct HashEntry
{
    uint key;
    uint poolIndex;
};

uint PackKey(int3 gridCoord)
{
    // Bias into unsigned range: -512..+511 → 0..1023
    uint3 biased = uint3(gridCoord + int3(512, 512, 512));
    return (biased.x & 0x3FF)
         | ((biased.y & 0x3FF) << 10)
         | ((biased.z & 0x3FF) << 20);
}

uint HashSlot(uint key)
{
    return key * 2654435761u;
}

uint HashMapInsert(RWStructuredBuffer<HashEntry> entries, RWByteAddressBuffer counter,
                   uint capacityMask, uint key, float3 surfacePos, uint level)
{
    uint slot = HashSlot(key) & capacityMask;
    
    [loop]
    for (uint probe = 0; probe < 128; probe++)
    {
        uint idx = (slot + probe) & capacityMask;
        
        uint existing;
        InterlockedCompareExchange(entries[idx].key, HASH_EMPTY, key, existing);
        
        if (existing == HASH_EMPTY)
        {
            uint poolIdx;
            counter.InterlockedAdd(0, 1, poolIdx);
            
            // Reject if pool is full for this level
            if (poolIdx >= MaxTilesForLevel(level))
                return POOL_INVALID;
            
            entries[idx].poolIndex = poolIdx;
            
            // Store surface position (absolute world space)
            RWStructuredBuffer<float4> tileInfo = ResourceDescriptorHeap[GetTInfoIdx(level)];
            tileInfo[poolIdx] = float4(surfacePos, 0);
            
            return poolIdx;
        }
        else if (existing == key)
        {
            return entries[idx].poolIndex;
        }
    }
    
    return POOL_INVALID;
}

uint HashMapLookup(StructuredBuffer<HashEntry> entries, uint capacityMask, uint key)
{
    uint slot = HashSlot(key) & capacityMask;
    
    [loop]
    for (uint probe = 0; probe < 128; probe++)
    {
        uint idx = (slot + probe) & capacityMask;
        HashEntry entry = entries[idx];
        
        if (entry.key == HASH_EMPTY)
            return POOL_INVALID;
        
        if (entry.key == key)
            return entry.poolIndex;
    }
    
    return POOL_INVALID;
}

// ─────────────────────── Octahedral Mapping ───────────────────────

float2 OctEncode(float3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0)
    {
        float2 wrapped = (1.0 - abs(n.yx)) * float2(n.x >= 0 ? 1 : -1, n.y >= 0 ? 1 : -1);
        n.xy = wrapped;
    }
    return n.xy * 0.5 + 0.5;
}

float3 OctDecode(float2 oct)
{
    oct = oct * 2.0 - 1.0;
    float3 n = float3(oct, 1.0 - abs(oct.x) - abs(oct.y));
    if (n.z < 0)
    {
        float2 wrapped = (1.0 - abs(n.yx)) * float2(n.x >= 0 ? 1 : -1, n.y >= 0 ? 1 : -1);
        n.xy = wrapped;
    }
    return normalize(n);
}

float3 GetDirection(uint ix, uint iy, uint octRes)
{
    float2 uv = (float2(ix, iy) + 0.5) / float(octRes);
    return OctDecode(uv);
}

// ─────────────────────── R11G11B10 Packing ───────────────────────

uint PackR11G11B10(float3 rgb)
{
    rgb = max(rgb, 0);
    uint r = f32tof16(rgb.r);
    uint g = f32tof16(rgb.g);
    uint b = f32tof16(rgb.b);
    r = (r >> 4) & 0x7FF;
    g = (g >> 4) & 0x7FF;
    b = (b >> 5) & 0x3FF;
    return r | (g << 11) | (b << 22);
}

float3 UnpackR11G11B10(uint packed)
{
    uint r = (packed & 0x7FF) << 4;
    uint g = ((packed >> 11) & 0x7FF) << 4;
    uint b = ((packed >> 22) & 0x3FF) << 5;
    return float3(f16tof32(r), f16tof32(g), f16tof32(b));
}

// ─────────────────────── World Position ───────────────────────

float3 ReconstructWorldPos(uint2 px)
{
    Texture2D<float> depthBuf = ResourceDescriptorHeap[DepthGBufIdx];
    float linearDepth = depthBuf.Load(int3(px, 0)).r;
    
    float2 uv = (float2(px) + 0.5) / float2(ScreenWIdx, ScreenHIdx);
    float2 ndc = float2(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    
    float3 viewRay = float3(ndc.x / Projection._11, ndc.y / Projection._22, 1.0);
    float3 viewPos = viewRay * linearDepth;
    
    return mul(float4(viewPos, 0), ViewInverse).xyz;
}

// ─────────────────────── Tile Interval Read/Write ───────────────────────
// Each direction stores 2 uints: R11G11B10 radiance + f16 transmittance

void WriteTileInterval(RWByteAddressBuffer pool, uint tileIdx, uint dirIdx,
                       uint dirsPerTile, float3 radiance, float transmittance)
{
    uint offset = (tileIdx * dirsPerTile + dirIdx) * 8; // 2 uints × 4 bytes
    pool.Store(offset, PackR11G11B10(radiance));
    pool.Store(offset + 4, f32tof16(transmittance));
}

void ReadTileInterval(RWByteAddressBuffer pool, uint tileIdx, uint dirIdx,
                      uint dirsPerTile, out float3 radiance, out float transmittance)
{
    uint offset = (tileIdx * dirsPerTile + dirIdx) * 8;
    radiance = UnpackR11G11B10(pool.Load(offset));
    transmittance = f16tof32(pool.Load(offset + 4));
}

// ─────────────────────── Screen-Space Ray March ───────────────────────

void TraceInterval(float3 worldOrigin, float3 worldDir, float maxDist, uint level,
                   out float3 hitRadiance, out float hitTransmittance)
{
    hitRadiance = float3(0, 0, 0);
    hitTransmittance = 1.0;
    
    Texture2D<float> depthBuf = ResourceDescriptorHeap[DepthGBufIdx];
    float3 viewFwd = float3(View._13, View._23, View._33);
    
    // Scale steps per level: L0=16, L1=32, L2=64, L3=128
    uint traceSteps = TRACE_STEPS_BASE << level;
    float stepSize = maxDist / float(traceSteps);
    
    [loop]
    for (uint i = 0; i < traceSteps; i++)
    {
        float t = (float(i) + 0.5) * stepSize;
        float3 rayPos = worldOrigin + worldDir * t;
        
        float4 clip = mul(float4(rayPos, 1.0), CameraRelativeVP);
        if (clip.w <= 0.0) continue;
        
        float2 uv = (clip.xy / clip.w) * float2(0.5, -0.5) + 0.5;
        
        if (any(uv < 0.0) || any(uv > 1.0)) continue;
        
        float sceneDepth = depthBuf.SampleLevel(LinearSampler, uv, 0).r;
        if (sceneDepth <= 0.0) continue;
        
        float rayDepth = dot(rayPos, viewFwd);
        float penetration = rayDepth - sceneDepth;
        
        if (penetration > -stepSize * 1.5 && penetration < stepSize * 3.0)
        {
            // Skip self-intersection: first 4 steps check grazing angle
            if (i < 4)
            {
                Texture2D normalTex = ResourceDescriptorHeap[NormalTexIdx];
                float3 hitNormal = normalTex.SampleLevel(LinearSampler, uv, 0).xyz;
                if (abs(dot(hitNormal, worldDir)) < 0.15) continue;
            }
            
            Texture2D albedoTex = ResourceDescriptorHeap[AlbedoTexIdx];
            float4 albedoData = albedoTex.SampleLevel(LinearSampler, uv, 0);
            
            Texture2D lightTex = ResourceDescriptorHeap[LightTexIdx];
            float3 surfaceLight = lightTex.SampleLevel(LinearSampler, uv, 0).rgb;
            
            hitRadiance = surfaceLight * (1.0 / PI) + albedoData.rgb * albedoData.a;
            hitTransmittance = 0.0;
            return;
        }
    }
    
    hitTransmittance = 1.0;
}

// ─────────────────────── CSMark ───────────────────────
// Insert tiles at ALL cascade levels for each visible pixel

// Max distance from camera for each cascade level.
// Screen-space tile size stays roughly constant across distances.
float LevelMaxDist(uint level)
{
    // L0: 16m, L1: 32m, L2: 64m, L3: unlimited
    return BASE_GRID_SIZE * float(1u << level) * 64.0;
}

[numthreads(8, 8, 1)]
void CSMark(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= ScreenWIdx || dtid.y >= ScreenHIdx)
        return;
    
    Texture2D DepthGBuf = ResourceDescriptorHeap[DepthGBufIdx];
    float depth = DepthGBuf.Load(int3(dtid.xy, 0)).r;
    
    if (depth == 0.0)
        return;
    
    float3 worldPos = ReconstructWorldPos(dtid.xy);
    float3 absWorldPos = worldPos + CamPos;
    float distFromCam = length(worldPos);
    
    Texture2D NormalTex = ResourceDescriptorHeap[NormalTexIdx];
    float3 normal = NormalTex.Load(int3(dtid.xy, 0)).xyz;
    
    // Insert tile at each cascade level within its distance limit
    [unroll]
    for (uint level = 0; level < CASCADE_COUNT; level++)
    {
        if (distFromCam > LevelMaxDist(level)) continue;
        
        float gridSize = GridSizeForLevel(level);
        int3 gridCoord = int3(floor(absWorldPos / gridSize));
        uint key = PackKey(gridCoord);
        
        // XZ and Y: deterministic cell center
        float3 probePos = (float3(gridCoord) + 0.5) * gridSize;
        
        RWStructuredBuffer<HashEntry> entries = ResourceDescriptorHeap[GetHEntriesIdx(level)];
        RWByteAddressBuffer counter = ResourceDescriptorHeap[GetHCounterIdx(level)];
        
        HashMapInsert(entries, counter, HCapMaskIdx, key, probePos, level);
    }
}

// ─────────────────────── CSPrepareIndirect ───────────────────────
// Reads the current level's counter and writes indirect dispatch args

[numthreads(1, 1, 1)]
void CSPrepareIndirect(uint3 dtid : SV_DispatchThreadID)
{
    RWByteAddressBuffer counter = ResourceDescriptorHeap[GetHCounterIdx(CurLevelIdx)];
    RWByteAddressBuffer args = ResourceDescriptorHeap[IndArgsIdx];
    
    uint tileCount = counter.Load(0);
    uint maxTiles = MaxTilesForLevel(CurLevelIdx);
    tileCount = min(tileCount, maxTiles);
    
    args.Store(0, tileCount);
    args.Store(4, 1u);
    args.Store(8, 1u);
}

// ─────────────────────── CSTrace (Per-Level) ───────────────────────

void TraceLevel(uint level, uint2 octXY, uint tileIdx)
{
    uint octRes = OctResForLevel(level);
    uint dirsPerTile = octRes * octRes;
    uint dirIdx = octXY.y * octRes + octXY.x;
    
    // Read stored surface position (absolute world space)
    StructuredBuffer<float4> tileInfo = ResourceDescriptorHeap[GetTInfoIdx(level)];
    float3 surfacePos = tileInfo[tileIdx].xyz;
    
    // Convert to camera-relative
    float3 origin = surfacePos - CamPos;
    
    // Direction from 2D octahedral coords
    float3 dir = GetDirection(octXY.x, octXY.y, octRes);
    
    // Self-intersection bias: skip the first half grid-cell along the ray direction.
    // This pushes the trace start past the probe's own surface without depending
    // on screen-space normal sampling (which causes per-frame jitter).
    float gridSize = GridSizeForLevel(level);
    float bias = gridSize * 0.5;
    
    // Trace this level's interval: bias shifts the origin forward but we
    // keep the end point at range.y to avoid gaps between cascade levels.
    float2 range = GetIntervalRange(level);
    float3 rayOrigin = origin + dir * (range.x + bias);
    float maxDist = max(0.0, range.y - range.x - bias);
    
    float3 hitRadiance;
    float hitTransmittance;
    
    if (RCDebugModeIdx >= 3)
    {
        // Diagnostic: constant radiance to all tiles — tests merge/lookup correctness
        hitRadiance = float3(1, 1, 1);
        hitTransmittance = 0.0;
    }
    else
    {
        TraceInterval(rayOrigin, dir, maxDist, level, hitRadiance, hitTransmittance);
    }
    
    // Store interval (radiance + transmittance)
    RWByteAddressBuffer pool = ResourceDescriptorHeap[GetTPoolIdx(level)];
    WriteTileInterval(pool, tileIdx, dirIdx, dirsPerTile, hitRadiance, hitTransmittance);
}

// Level 0: octRes=4 → 4×4 = 16 dirs
[numthreads(4, 4, 1)]
void CSTrace0(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    TraceLevel(0, gtid.xy, gid.x);
}

// Level 1: octRes=8 → 8×8 = 64 dirs
[numthreads(8, 8, 1)]
void CSTrace1(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    TraceLevel(1, gtid.xy, gid.x);
}

// Level 2: octRes=16 → 16×16 = 256 dirs
[numthreads(16, 16, 1)]
void CSTrace2(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    TraceLevel(2, gtid.xy, gid.x);
}

// Level 3: octRes=32 → 32×32 = 1024 dirs
[numthreads(32, 32, 1)]
void CSTrace3(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    TraceLevel(3, gtid.xy, gid.x);
}

// ─────────────────────── Cascade Merge ───────────────────────
// Merges source level (N+1) into target level (N) with spatial + angular interpolation.
// Run top-down: L3→L2, L2→L1, L1→L0.
// After merge, L0 contains full radiance from all cascade levels.

void MergeLevel(uint targetLevel, uint2 octXY, uint tileIdx)
{
    uint targetOctRes = OctResForLevel(targetLevel);
    if (octXY.x >= targetOctRes || octXY.y >= targetOctRes) return;
    
    uint targetDirsPerTile = targetOctRes * targetOctRes;
    uint dirIdx = octXY.y * targetOctRes + octXY.x;
    
    // Read own (target) interval
    RWByteAddressBuffer targetPool = ResourceDescriptorHeap[GetTPoolIdx(targetLevel)];
    float3 myRad;
    float myTrans;
    ReadTileInterval(targetPool, tileIdx, dirIdx, targetDirsPerTile, myRad, myTrans);
    
    // If target is fully opaque, higher levels can't contribute
    if (myTrans < 0.001) return;
    
    // Read target probe position (absolute world space)
    StructuredBuffer<float4> targetInfo = ResourceDescriptorHeap[GetTInfoIdx(targetLevel)];
    float3 probePos = targetInfo[tileIdx].xyz;
    
    // Source level
    uint srcLevel = targetLevel + 1;
    uint srcOctRes = OctResForLevel(srcLevel);
    uint srcDirsPerTile = srcOctRes * srcOctRes;
    float srcGridSize = GridSizeForLevel(srcLevel);
    
    // Map probe position to source grid for spatial interpolation
    float3 srcCellPos = probePos / srcGridSize - 0.5;
    int3 srcBase = int3(floor(srcCellPos));
    float3 srcFrac = srcCellPos - float3(srcBase);
    
    // Map direction to source octahedral grid for angular interpolation
    float3 dir = GetDirection(octXY.x, octXY.y, targetOctRes);
    float2 octUV = OctEncode(dir);
    float2 srcDirPos = octUV * float(srcOctRes) - 0.5;
    int2 srcDirBase = int2(floor(srcDirPos));
    float2 srcDirFrac = srcDirPos - float2(srcDirBase);
    
    StructuredBuffer<HashEntry> srcEntries = ResourceDescriptorHeap[GetHEntriesIdx(srcLevel)];
    RWByteAddressBuffer srcPool = ResourceDescriptorHeap[GetTPoolIdx(srcLevel)];
    
    float3 interpRad = 0;
    float interpTrans = 0;
    float totalWeight = 0;
    
    // 8 spatial neighbors at source level
    [unroll]
    for (uint s = 0; s < 8; s++)
    {
        int3 srcOffset = int3(s & 1, (s >> 1) & 1, (s >> 2) & 1);
        int3 srcCoord = srcBase + srcOffset;
        uint srcKey = PackKey(srcCoord);
        uint srcTileIdx = HashMapLookup(srcEntries, HCapMaskIdx, srcKey);
        if (srcTileIdx == POOL_INVALID) continue;
        
        float3 sw = float3(
            (s & 1) ? srcFrac.x : (1.0 - srcFrac.x),
            ((s >> 1) & 1) ? srcFrac.y : (1.0 - srcFrac.y),
            ((s >> 2) & 1) ? srcFrac.z : (1.0 - srcFrac.z)
        );
        float spatialWeight = sw.x * sw.y * sw.z;
        
        // 4 angular neighbors (bilinear in octahedral space)
        [unroll]
        for (uint a = 0; a < 4; a++)
        {
            int2 angOffset = int2(a & 1, (a >> 1) & 1);
            int2 angCoord = clamp(srcDirBase + angOffset, int2(0, 0),
                                  int2(srcOctRes - 1, srcOctRes - 1));
            uint srcDirIdx = angCoord.y * srcOctRes + angCoord.x;
            
            float2 aw = float2(
                (a & 1) ? srcDirFrac.x : (1.0 - srcDirFrac.x),
                ((a >> 1) & 1) ? srcDirFrac.y : (1.0 - srcDirFrac.y)
            );
            float angularWeight = aw.x * aw.y;
            
            float3 sRad;
            float sTrans;
            ReadTileInterval(srcPool, srcTileIdx, srcDirIdx, srcDirsPerTile, sRad, sTrans);
            
            float w = spatialWeight * angularWeight;
            interpRad += sRad * w;
            interpTrans += sTrans * w;
            totalWeight += w;
        }
    }
    
    if (totalWeight > 0.001)
    {
        interpRad /= totalWeight;
        interpTrans /= totalWeight;
        
        // Front-to-back compose: target interval first, then source (farther)
        float3 newRad = myRad + myTrans * interpRad;
        float newTrans = myTrans * interpTrans;
        
        WriteTileInterval(targetPool, tileIdx, dirIdx, targetDirsPerTile, newRad, newTrans);
    }
}

// Merge L1→L0: target octRes=4, thread group 4×4
[numthreads(4, 4, 1)]
void CSMerge0(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    MergeLevel(0, gtid.xy, gid.x);
}

// Merge L2→L1: target octRes=8, thread group 8×8
[numthreads(8, 8, 1)]
void CSMerge1(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    MergeLevel(1, gtid.xy, gid.x);
}

// Merge L3→L2: target octRes=16, thread group 16×16
[numthreads(16, 16, 1)]
void CSMerge2(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    MergeLevel(2, gtid.xy, gid.x);
}

// ─────────────────────── CSShade ───────────────────────
// Merge all cascade levels front-to-back, cosine-weighted hemisphere integral

[numthreads(8, 8, 1)]
void CSShade(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= ScreenWIdx || dtid.y >= ScreenHIdx)
        return;
    
    RWTexture2D<float4> GIOutput = ResourceDescriptorHeap[GIOutputIdx];
    
    Texture2D DepthGBuf = ResourceDescriptorHeap[DepthGBufIdx];
    float depth = DepthGBuf.Load(int3(dtid.xy, 0)).r;
    
    if (depth == 0.0)
    {
        GIOutput[dtid.xy] = float4(0, 0, 0, 0);
        return;
    }
    
    float3 worldPos = ReconstructWorldPos(dtid.xy);
    float3 absWorldPos = worldPos + CamPos;
    Texture2D NormalTex = ResourceDescriptorHeap[NormalTexIdx];
    float3 normal = NormalTex.Load(int3(dtid.xy, 0)).xyz;
    
    // Phase 1: Find finest available cascade level with valid tiles
    uint shadeLevel = CASCADE_COUNT; // sentinel = none found
    uint tileCorners[8];
    float3 lerpW;
    
    [loop]
    for (uint lvl = 0; lvl < CASCADE_COUNT; lvl++)
    {
        float gridSize = GridSizeForLevel(lvl);
        float3 cellPos = absWorldPos / gridSize - 0.5;
        int3 baseCoord = int3(floor(cellPos));
        float3 w = cellPos - float3(baseCoord);
        
        StructuredBuffer<HashEntry> entries = ResourceDescriptorHeap[GetHEntriesIdx(lvl)];
        
        bool anyValid = false;
        uint corners[8];
        [unroll]
        for (uint c = 0; c < 8; c++)
        {
            int3 coord = baseCoord + int3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
            uint key = PackKey(coord);
            corners[c] = HashMapLookup(entries, HCapMaskIdx, key);
            if (corners[c] != POOL_INVALID) anyValid = true;
        }
        
        if (anyValid)
        {
            shadeLevel = lvl;
            lerpW = w;
            [unroll] for (uint cc = 0; cc < 8; cc++) tileCorners[cc] = corners[cc];
            break;
        }
    }
    
    if (shadeLevel >= CASCADE_COUNT)
    {
        GIOutput[dtid.xy] = float4(0, 0, 0, 0);
        return;
    }
    
    // ─── Debug: tile lookup visualization ───
    if (RCDebugModeIdx == 1)
    {
        uint found = 0;
        [unroll]
        for (uint cc = 0; cc < 8; cc++)
            if (tileCorners[cc] != POOL_INVALID) found++;
        // Encode level in color: L0=green, L1=cyan, L2=blue, L3=magenta
        float3 levelColor = float3(0, 1, 0);
        if (shadeLevel == 1) levelColor = float3(0, 1, 1);
        else if (shadeLevel == 2) levelColor = float3(0, 0, 1);
        else if (shadeLevel == 3) levelColor = float3(1, 0, 1);
        GIOutput[dtid.xy] = float4(levelColor * (found / 8.0), 1.0);
        return;
    }
    
    // ─── Debug: raw radiance in normal direction (mode 2) ───
    if (RCDebugModeIdx == 2)
    {
        if (tileCorners[0] != POOL_INVALID)
        {
            uint octResDbg = OctResForLevel(shadeLevel);
            float2 octUV = OctEncode(normal);
            uint ix = clamp(uint(octUV.x * float(octResDbg)), 0, octResDbg - 1);
            uint iy = clamp(uint(octUV.y * float(octResDbg)), 0, octResDbg - 1);
            uint dirIdx = iy * octResDbg + ix;
            uint dirsPerTile = octResDbg * octResDbg;
            float3 iRad; float iTrans;
            RWByteAddressBuffer pool = ResourceDescriptorHeap[GetTPoolIdx(shadeLevel)];
            ReadTileInterval(pool, tileCorners[0], dirIdx, dirsPerTile, iRad, iTrans);
            GIOutput[dtid.xy] = float4(iRad, 1.0);
        }
        else
            GIOutput[dtid.xy] = float4(0, 0, 0, 1);
        return;
    }
    
    // ─── Debug: per-level radiance in normal dir (mode 4) ───
    if (RCDebugModeIdx == 4)
    {
        StructuredBuffer<HashEntry> entries2 = ResourceDescriptorHeap[GetHEntriesIdx(2)];
        StructuredBuffer<HashEntry> entries3 = ResourceDescriptorHeap[GetHEntriesIdx(3)];
        
        float gridSize2 = GridSizeForLevel(2);
        int3 coord2 = int3(floor(absWorldPos / gridSize2));
        uint key2 = PackKey(coord2);
        uint tile2 = HashMapLookup(entries2, HCapMaskIdx, key2);
        
        float gridSize3 = GridSizeForLevel(3);
        int3 coord3 = int3(floor(absWorldPos / gridSize3));
        uint key3 = PackKey(coord3);
        uint tile3 = HashMapLookup(entries3, HCapMaskIdx, key3);
        
        float l0trans = 0;
        float l2sum = 0;
        float l3sum = 0;
        
        if (tileCorners[0] != POOL_INVALID)
        {
            uint octRes0 = OctResForLevel(shadeLevel);
            float2 octUV = OctEncode(normal);
            uint ix = clamp(uint(octUV.x * float(octRes0)), 0, octRes0 - 1);
            uint iy = clamp(uint(octUV.y * float(octRes0)), 0, octRes0 - 1);
            uint dirIdx = iy * octRes0 + ix;
            float3 r; float t;
            RWByteAddressBuffer pool = ResourceDescriptorHeap[GetTPoolIdx(shadeLevel)];
            ReadTileInterval(pool, tileCorners[0], dirIdx, octRes0 * octRes0, r, t);
            l0trans = t;
        }
        
        if (tile2 != POOL_INVALID)
        {
            uint octRes2 = OctResForLevel(2);
            uint dirs2 = octRes2 * octRes2;
            RWByteAddressBuffer pool2 = ResourceDescriptorHeap[GetTPoolIdx(2)];
            for (uint d = 0; d < dirs2; d++)
            {
                float3 r; float t;
                ReadTileInterval(pool2, tile2, d, dirs2, r, t);
                l2sum += dot(r, float3(0.333, 0.333, 0.333));
            }
        }
        
        if (tile3 != POOL_INVALID)
        {
            uint octRes3 = OctResForLevel(3);
            uint dirs3 = octRes3 * octRes3;
            RWByteAddressBuffer pool3 = ResourceDescriptorHeap[GetTPoolIdx(3)];
            for (uint d = 0; d < min(dirs3, 256u); d++)
            {
                float3 r; float t;
                ReadTileInterval(pool3, tile3, d, dirs3, r, t);
                l3sum += dot(r, float3(0.333, 0.333, 0.333));
            }
        }
        
        GIOutput[dtid.xy] = float4(l0trans, l2sum, l3sum, 1.0);
        return;
    }
    
    // Phase 2: Hemisphere integral using finest available level
    float3 irradiance = 0;
    
    uint octRes = OctResForLevel(shadeLevel);
    uint dirsPerTile = octRes * octRes;
    RWByteAddressBuffer pool = ResourceDescriptorHeap[GetTPoolIdx(shadeLevel)];
    
    [loop]
    for (uint dy = 0; dy < SHADE_OCT_RES; dy++)
    {
        [loop]
        for (uint dx = 0; dx < SHADE_OCT_RES; dx++)
        {
            float3 dir = GetDirection(dx, dy, SHADE_OCT_RES);
            float NdotD = saturate(dot(normal, dir));
            if (NdotD <= 0.0) continue;
            
            // Map query direction to the shade level's octahedral grid
            float2 octUV = OctEncode(dir);
            uint ix = clamp(uint(octUV.x * float(octRes)), 0, octRes - 1);
            uint iy = clamp(uint(octUV.y * float(octRes)), 0, octRes - 1);
            uint dirIdx = iy * octRes + ix;
            
            // Trilinear interpolation over 8 surrounding tiles
            float3 interpRad = 0;
            float totalWeight = 0;
            
            [unroll]
            for (uint c = 0; c < 8; c++)
            {
                if (tileCorners[c] == POOL_INVALID) continue;
                
                float3 cw = float3(
                    (c & 1) ? lerpW.x : (1.0 - lerpW.x),
                    ((c >> 1) & 1) ? lerpW.y : (1.0 - lerpW.y),
                    ((c >> 2) & 1) ? lerpW.z : (1.0 - lerpW.z)
                );
                float cornerWeight = cw.x * cw.y * cw.z;
                
                float3 cRad;
                float cTrans;
                ReadTileInterval(pool, tileCorners[c], dirIdx, dirsPerTile, cRad, cTrans);
                
                interpRad += cRad * cornerWeight;
                totalWeight += cornerWeight;
            }
            
            if (totalWeight > 0.001)
                irradiance += (interpRad / totalWeight) * NdotD;
        }
    }
    
    // Normalize: 4*PI / total_directions (uniform solid angle per direction)
    irradiance *= (4.0 * PI / float(SHADE_OCT_RES * SHADE_OCT_RES));
    
    float intensity = asfloat(RCIntensityIdx);
    GIOutput[dtid.xy] = float4(irradiance * intensity, 1.0);
}

// ─────────────────────── CSComposeGI ───────────────────────
// Applies GI to LightBuffer as a separate pass.
// Runs AFTER all direct lighting, so the trace next frame reads direct-only.

[numthreads(8, 8, 1)]
void CSComposeGI(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= ScreenWIdx || dtid.y >= ScreenHIdx) return;
    
    Texture2D<float4> giTex = ResourceDescriptorHeap[GIOutputIdx];
    float3 gi = giTex.Load(int3(dtid.xy, 0)).rgb;
    
    if (dot(gi, gi) < 0.0001) return; // skip if no GI
    
    Texture2D<float4> albedoTex = ResourceDescriptorHeap[AlbedoTexIdx];
    float3 albedo = albedoTex.Load(int3(dtid.xy, 0)).rgb;
    
    Texture2D<float4> dataTex = ResourceDescriptorHeap[DataTexIdx];
    float4 data = dataTex.Load(int3(dtid.xy, 0));
    float metal = saturate(data.g);
    float ao = saturate(data.b);
    
    float3 kd = (1.0 - metal); // simplified diffuse (no Fresnel for indirect)
    
    RWTexture2D<float4> lightBuf = ResourceDescriptorHeap[LightBufUavIdx];
    float3 existing = lightBuf[dtid.xy].rgb;
    lightBuf[dtid.xy] = float4(existing + kd * albedo * gi * ao, 1.0);
}
