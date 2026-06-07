// Sparse 3D Radiance Cascades — compute shader
// Per-cascade angular scaling with interval storage and front-to-back merging
// Kernels: CSMark, CSPrepareIndirect, CSTrace0..3, CSShade

#pragma kernel CSMark
#pragma kernel CSPrepareIndirect
#pragma kernel CSTrace0
#pragma kernel CSTrace1
#pragma kernel CSTrace2
#pragma kernel CSTrace3
#pragma kernel CSShade

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
};

#include "common.fx"

SamplerState LinearSampler : register(s0);

// ─────────────────────── Constants ───────────────────────

#define CASCADE_COUNT     4
#define BASE_OCT_RES      4
#define BASE_GRID_SIZE    0.5
#define BASE_INTERVAL     0.5
#define TRACE_STEPS       48
#define TILE_SCREEN_SIZE  8.0
#define SHADE_OCT_RES     6      // hemisphere integral resolution (6×6 = 36 dirs)
#define HASH_EMPTY        0xFFFFFFFF
#define POOL_INVALID      0xFFFFFFFF
#define PI                3.14159265

// Per-level max tiles (must match C# MaxTilesPerLevel)
uint MaxTilesForLevel(uint level)
{
    // L0: 65536, L1: 16384, L2: 4096, L3: 1024
    return 65536u >> (level * 2);
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
    return (asuint(gridCoord.x) & 0x3FF)
         | ((asuint(gridCoord.y) & 0x3FF) << 10)
         | ((asuint(gridCoord.z) & 0x3FF) << 20);
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

void TraceInterval(float3 worldOrigin, float3 worldDir, float maxDist,
                   out float3 hitRadiance, out float hitTransmittance)
{
    hitRadiance = float3(0, 0, 0);
    hitTransmittance = 1.0;
    
    Texture2D<float> depthBuf = ResourceDescriptorHeap[DepthGBufIdx];
    float3 viewFwd = float3(View._13, View._23, View._33);
    
    float stepSize = maxDist / float(TRACE_STEPS);
    
    // Get depth buffer dimensions for Load
    uint dbW, dbH;
    depthBuf.GetDimensions(dbW, dbH);
    
    [loop]
    for (int i = 0; i < TRACE_STEPS; i++)
    {
        float t = (float(i) + 0.5) * stepSize;
        float3 rayPos = worldOrigin + worldDir * t;
        
        float4 clip = mul(float4(rayPos, 1.0), CameraRelativeVP);
        if (clip.w <= 0.0) continue;
        
        float2 uv = (clip.xy / clip.w) * float2(0.5, -0.5) + 0.5;
        
        if (any(uv < 0.0) || any(uv > 1.0)) continue;
        
        int2 depthCoord = int2(uv * float2(dbW, dbH));
        float sceneDepth = depthBuf.Load(int3(depthCoord, 0)).r;
        
        if (sceneDepth <= 0.0) continue;
        
        float rayDepth = dot(rayPos, viewFwd);
        float penetration = rayDepth - sceneDepth;
        
        if (penetration > -stepSize * 1.5 && penetration < stepSize * 3.0)
        {
            Texture2D albedoTex = ResourceDescriptorHeap[AlbedoTexIdx];
            float4 albedoData = albedoTex.SampleLevel(LinearSampler, uv, 0);
            
            Texture2D lightTex = ResourceDescriptorHeap[LightTexIdx];
            float3 surfaceLight = lightTex.SampleLevel(LinearSampler, uv, 0).rgb;
            
            hitRadiance = albedoData.rgb * surfaceLight * (1.0 / PI) + albedoData.rgb * albedoData.a;
            hitTransmittance = 0.0;
            return;
        }
    }
    
    hitTransmittance = 1.0;
}

// ─────────────────────── CSMark ───────────────────────
// Insert tiles at ALL cascade levels for each visible pixel

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
    
    // Insert tile at every cascade level — with distance culling
    // Near-field levels (L0) don't need tiles for distant surfaces
    [unroll]
    for (uint level = 0; level < CASCADE_COUNT; level++)
    {
        // Only mark tiles within useful range: intervalEnd × 32
        float maxRange = GetIntervalRange(level).y;
        float markRadius = maxRange * 32.0;
        if (distFromCam > markRadius) continue;
        
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
    
    // Trace this level's interval only
    float2 range = GetIntervalRange(level);
    float3 rayOrigin = origin + dir * range.x;
    float maxDist = range.y - range.x;
    
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
        TraceInterval(rayOrigin, dir, maxDist, hitRadiance, hitTransmittance);
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
    
    // Phase 1: Look up 8 surrounding tile indices at all levels for trilinear interpolation
    // Each level has 8 corner tiles and 3 interpolation weights
    uint tileCorners[CASCADE_COUNT][8];  // [level][corner]
    float3 lerpWeights[CASCADE_COUNT];    // trilinear weights per level
    
    [unroll]
    for (uint lvl = 0; lvl < CASCADE_COUNT; lvl++)
    {
        float gridSize = GridSizeForLevel(lvl);
        float3 cellPos = absWorldPos / gridSize - 0.5; // shift so interpolation is centered on cell centers
        int3 baseCoord = int3(floor(cellPos));
        float3 frac3 = cellPos - float3(baseCoord);
        lerpWeights[lvl] = frac3;
        
        StructuredBuffer<HashEntry> entries = ResourceDescriptorHeap[GetHEntriesIdx(lvl)];
        
        // 8 corner lookups (2×2×2)
        [unroll]
        for (uint c = 0; c < 8; c++)
        {
            int3 offset = int3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
            int3 coord = baseCoord + offset;
            uint key = PackKey(coord);
            tileCorners[lvl][c] = HashMapLookup(entries, HCapMaskIdx, key);
        }
    }
    
    // ─── Debug: tile lookup visualization ───
    if (RCDebugModeIdx == 1)
    {
        float3 debugColor = 0;
        float3 levelColors[4] = {
            float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1), float3(1, 1, 0)
        };
        [unroll]
        for (uint d = 0; d < CASCADE_COUNT; d++)
        {
            uint found = 0;
            [unroll]
            for (uint cc = 0; cc < 8; cc++)
                if (tileCorners[d][cc] != POOL_INVALID) found++;
            if (found > 0)
                debugColor += levelColors[d] * (float(found) / 8.0);
        }
        GIOutput[dtid.xy] = float4(debugColor, 1.0);
        return;
    }
    
    // ─── Debug modes 2-4 use single-tile lookup (first corner) ───
    // (Keep existing debug modes working with minimal changes)
    if (RCDebugModeIdx >= 2 && RCDebugModeIdx <= 4)
    {
        uint tileIdx[CASCADE_COUNT];
        [unroll]
        for (uint dd = 0; dd < CASCADE_COUNT; dd++)
            tileIdx[dd] = tileCorners[dd][0];
        
        if (RCDebugModeIdx == 2)
        {
            float3 levelColors2[4] = {
                float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1), float3(1, 1, 0)
            };
            float3 debugColor2 = 0;
            [unroll]
            for (uint level2 = 0; level2 < CASCADE_COUNT; level2++)
            {
                if (tileIdx[level2] == POOL_INVALID) continue;
                uint octRes2 = OctResForLevel(level2);
                float2 octUV2 = OctEncode(normal);
                uint ix2 = clamp(uint(octUV2.x * float(octRes2)), 0, octRes2 - 1);
                uint iy2 = clamp(uint(octUV2.y * float(octRes2)), 0, octRes2 - 1);
                uint dirIdx2 = iy2 * octRes2 + ix2;
                uint dirsPerTile2 = octRes2 * octRes2;
                float3 iRad; float iTrans;
                RWByteAddressBuffer pool2 = ResourceDescriptorHeap[GetTPoolIdx(level2)];
                ReadTileInterval(pool2, tileIdx[level2], dirIdx2, dirsPerTile2, iRad, iTrans);
                float lum2 = dot(iRad, float3(0.299, 0.587, 0.114));
                debugColor2 += levelColors2[level2] * saturate(lum2);
            }
            GIOutput[dtid.xy] = float4(debugColor2, 1.0);
            return;
        }
        
        if (RCDebugModeIdx == 4)
        {
            float3 totalRad4 = 0;
            uint totalHits4 = 0;
            [unroll]
            for (uint lv4 = 0; lv4 < CASCADE_COUNT; lv4++)
            {
                if (tileIdx[lv4] == POOL_INVALID) continue;
                uint octRes4 = OctResForLevel(lv4);
                uint dpt4 = octRes4 * octRes4;
                RWByteAddressBuffer pool4 = ResourceDescriptorHeap[GetTPoolIdx(lv4)];
                for (uint d4 = 0; d4 < min(dpt4, 64u); d4++)
                {
                    float3 iR4; float iT4;
                    ReadTileInterval(pool4, tileIdx[lv4], d4, dpt4, iR4, iT4);
                    if (iT4 < 0.5) { totalRad4 += iR4; totalHits4++; }
                }
            }
            if (totalHits4 > 0)
                GIOutput[dtid.xy] = float4(totalRad4 / float(totalHits4), 1.0);
            else
                GIOutput[dtid.xy] = float4(0, 0, 0, 1);
            return;
        }
    }
    
    // Phase 2: Hemisphere integral with trilinear-interpolated probe data
    float3 irradiance = 0;
    
    [loop]
    for (uint dy = 0; dy < SHADE_OCT_RES; dy++)
    {
        [loop]
        for (uint dx = 0; dx < SHADE_OCT_RES; dx++)
        {
            float3 dir = GetDirection(dx, dy, SHADE_OCT_RES);
            float NdotD = saturate(dot(normal, dir));
            if (NdotD <= 0.0) continue;
            
            // Merge intervals front-to-back across cascade levels
            // For each level, trilinearly interpolate between 8 surrounding tiles
            float3 L = float3(0, 0, 0);
            float beta = 1.0;
            
            [unroll]
            for (uint level = 0; level < CASCADE_COUNT; level++)
            {
                uint octRes = OctResForLevel(level);
                float2 octUV = OctEncode(dir);
                uint ix = clamp(uint(octUV.x * float(octRes)), 0, octRes - 1);
                uint iy = clamp(uint(octUV.y * float(octRes)), 0, octRes - 1);
                uint dirIdx = iy * octRes + ix;
                uint dirsPerTile = octRes * octRes;
                
                RWByteAddressBuffer pool = ResourceDescriptorHeap[GetTPoolIdx(level)];
                float3 w = lerpWeights[level];
                
                // Trilinear interpolation over 8 corner tiles
                float3 interpRad = 0;
                float interpTrans = 0;
                float totalWeight = 0;
                
                [unroll]
                for (uint c = 0; c < 8; c++)
                {
                    if (tileCorners[level][c] == POOL_INVALID) continue;
                    
                    float3 cw = float3(
                        (c & 1) ? w.x : (1.0 - w.x),
                        ((c >> 1) & 1) ? w.y : (1.0 - w.y),
                        ((c >> 2) & 1) ? w.z : (1.0 - w.z)
                    );
                    float cornerWeight = cw.x * cw.y * cw.z;
                    
                    float3 cRad;
                    float cTrans;
                    ReadTileInterval(pool, tileCorners[level][c], dirIdx, dirsPerTile,
                                   cRad, cTrans);
                    
                    interpRad += cRad * cornerWeight;
                    interpTrans += cTrans * cornerWeight;
                    totalWeight += cornerWeight;
                }
                
                if (totalWeight > 0.001)
                {
                    interpRad /= totalWeight;
                    interpTrans /= totalWeight;
                    
                    L += beta * interpRad;
                    beta *= interpTrans;
                    if (beta < 0.001) break;
                }
            }
            
            irradiance += L * NdotD;
        }
    }
    
    // Normalize: 4*PI / total_directions (uniform solid angle per direction)
    irradiance *= (4.0 * PI / float(SHADE_OCT_RES * SHADE_OCT_RES));
    
    float intensity = asfloat(RCIntensityIdx);
    GIOutput[dtid.xy] = float4(irradiance * intensity, 1.0);
}
