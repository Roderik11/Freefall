// ════════════════════════════════════════════════════════════════════════════
// Single-Pass Hi-Z Depth Pyramid Generator (SPD)
// ════════════════════════════════════════════════════════════════════════════
//
// Generates the full mip chain of a Hi-Z depth pyramid in a single dispatch.
// Each thread group covers a 64x64 tile of the source and progressively
// reduces through groupshared memory, writing each mip level to its UAV.
// An atomic counter ensures the last surviving group reduces the final mips.
//
// Uses max() for reduction: farthest depth wins (inverse-Z compatible).
// Sky/empty pixels (depth <= 0) are promoted to FLT_MAX so tiles
// containing sky become un-occludable.
//
// Push constant layout (register b3, Indices[0..7]):
//   [0].x = SourceSrvIdx    — GBuffer depth (Texture2D<float>)
//   [0].y = CounterUavIdx   — RWByteAddressBuffer for atomic counter
//   [0].z = SourceWidth     — Source mip 0 width
//   [0].w = SourceHeight    — Source mip 0 height
//   [1].x = MipCount        — Total mip levels in pyramid
//   [1].y = NumGroupsX      — Dispatch grid width (for last-group logic)
//   [1].z = NumGroupsY      — Dispatch grid height
//   [1].w = Mip0UavIdx      — UAV for mip 0
//   [2].xyzw = mip UAV indices 1-4
//   [3].xyzw = mip UAV indices 5-8
//   [4].xyzw = mip UAV indices 9-12

cbuffer PushConstants : register(b3)
{
    uint4 Indices[8];
};

#define SourceSrvIdx   Indices[0].x
#define CounterUavIdx  Indices[0].y
#define SourceWidth    Indices[0].z
#define SourceHeight   Indices[0].w
#define MipCount       Indices[1].x
#define NumGroupsX     Indices[1].y
#define NumGroupsY     Indices[1].z

#define FLT_MAX 3.402823466e+38

// Mip UAV indices packed into push constants
uint GetMipUavIdx(uint mip)
{
    // mip 0 is at [1].w, mips 1-4 at [2].xyzw, mips 5-8 at [3].xyzw, mips 9-12 at [4].xyzw
    if (mip == 0) return Indices[1].w;
    mip -= 1; // now 0-based for remaining mips
    uint slot = 2 + (mip / 4);
    uint comp = mip % 4;
    return Indices[slot][comp];
}

// ── Groupshared storage ────────────────────────────────────────────────────
// 32×32 = 1024 entries: each thread loads a 2×2 block from source, reduces
// to 1 value, and stores in LDS. Then we progressively halve through LDS.
// We only need 16×16=256 entries for the second reduction and beyond,
// but allocate 1024 for the initial 32×32 load.
groupshared float gs_depth[32][32];

// ── Helpers ────────────────────────────────────────────────────────────────

float LoadSource(uint2 coord)
{
    Texture2D<float> src = ResourceDescriptorHeap[SourceSrvIdx];
    float d = src[coord];
    return d <= 0 ? FLT_MAX : d;
}

float Reduce4(float a, float b, float c, float d)
{
    return max(max(a, b), max(c, d));
}

void WriteMip(uint mip, uint2 coord, float value)
{
    if (mip >= MipCount) return;
    RWTexture2D<float> outMip = ResourceDescriptorHeap[GetMipUavIdx(mip)];
    outMip[coord] = value;
}

// ══════════════════════════════════════════════════════════════════════════
// Main kernel: CSSinglePassDownsample
// 256 threads per group, each group covers 64×64 source texels.
//
// Mip derivation within a single group:
//   Mip 0: 32×32 — each of 256 threads loads 2×2 from source, reduces to 1
//   Mip 1: 16×16 — 256 threads, each reduces 2×2 from LDS
//   Mip 2: 8×8   — 64 threads active
//   Mip 3: 4×4   — 16 threads active
//   Mip 4: 2×2   — 4 threads active
//   Mip 5: 1×1   — 1 thread active
//
// After all groups finish mip 5, the last group (via atomic counter)
// loads mip 5 values from all groups and reduces mips 6+.
// ══════════════════════════════════════════════════════════════════════════

[numthreads(256, 1, 1)]
void CSSinglePassDownsample(uint3 gid : SV_GroupID, uint gidx : SV_GroupIndex)
{
    // Map flat thread index to 2D within 32×32 tile
    // Use an interleaved pattern to keep 2×2 quads spatially coherent
    uint2 localCoord = uint2(gidx % 16, gidx / 16);  // 16×16 block

    // Base coordinate in source texture for this group's 64×64 tile
    uint2 groupBase = gid.xy * 64;

    // ════════════════════════════════════════════════════════════════════
    // Phase 1: Load source → Mip 0 (32×32 per group)
    // Each thread loads 4 source texels (2×2) and reduces to 1 value.
    // We need to cover 32×32 = 1024 values but only have 256 threads,
    // so each thread does 4 loads covering a 2×2 block.
    // ════════════════════════════════════════════════════════════════════

    // Thread covers 4 quadrants of the 32×32 space
    // Quadrant 0: top-left 16×16
    {
        uint2 localXY = localCoord;
        uint2 srcBase = groupBase + localXY * 2;

        float d0 = (srcBase.x < SourceWidth && srcBase.y < SourceHeight)
            ? LoadSource(srcBase + uint2(0, 0)) : FLT_MAX;
        float d1 = (srcBase.x + 1 < SourceWidth && srcBase.y < SourceHeight)
            ? LoadSource(srcBase + uint2(1, 0)) : FLT_MAX;
        float d2 = (srcBase.x < SourceWidth && srcBase.y + 1 < SourceHeight)
            ? LoadSource(srcBase + uint2(0, 1)) : FLT_MAX;
        float d3 = (srcBase.x + 1 < SourceWidth && srcBase.y + 1 < SourceHeight)
            ? LoadSource(srcBase + uint2(1, 1)) : FLT_MAX;

        float val = Reduce4(d0, d1, d2, d3);
        gs_depth[localXY.y][localXY.x] = val;

        // Write mip 0
        uint2 mipCoord = gid.xy * 32 + localXY;
        WriteMip(0, mipCoord, val);
    }

    // Quadrant 1: top-right 16×16
    {
        uint2 localXY = localCoord + uint2(16, 0);
        uint2 srcBase = groupBase + localXY * 2;

        float d0 = (srcBase.x < SourceWidth && srcBase.y < SourceHeight)
            ? LoadSource(srcBase + uint2(0, 0)) : FLT_MAX;
        float d1 = (srcBase.x + 1 < SourceWidth && srcBase.y < SourceHeight)
            ? LoadSource(srcBase + uint2(1, 0)) : FLT_MAX;
        float d2 = (srcBase.x < SourceWidth && srcBase.y + 1 < SourceHeight)
            ? LoadSource(srcBase + uint2(0, 1)) : FLT_MAX;
        float d3 = (srcBase.x + 1 < SourceWidth && srcBase.y + 1 < SourceHeight)
            ? LoadSource(srcBase + uint2(1, 1)) : FLT_MAX;

        float val = Reduce4(d0, d1, d2, d3);
        gs_depth[localXY.y][localXY.x] = val;

        uint2 mipCoord = gid.xy * 32 + localXY;
        WriteMip(0, mipCoord, val);
    }

    // Quadrant 2: bottom-left 16×16
    {
        uint2 localXY = localCoord + uint2(0, 16);
        uint2 srcBase = groupBase + localXY * 2;

        float d0 = (srcBase.x < SourceWidth && srcBase.y < SourceHeight)
            ? LoadSource(srcBase + uint2(0, 0)) : FLT_MAX;
        float d1 = (srcBase.x + 1 < SourceWidth && srcBase.y < SourceHeight)
            ? LoadSource(srcBase + uint2(1, 0)) : FLT_MAX;
        float d2 = (srcBase.x < SourceWidth && srcBase.y + 1 < SourceHeight)
            ? LoadSource(srcBase + uint2(0, 1)) : FLT_MAX;
        float d3 = (srcBase.x + 1 < SourceWidth && srcBase.y + 1 < SourceHeight)
            ? LoadSource(srcBase + uint2(1, 1)) : FLT_MAX;

        float val = Reduce4(d0, d1, d2, d3);
        gs_depth[localXY.y][localXY.x] = val;

        uint2 mipCoord = gid.xy * 32 + localXY;
        WriteMip(0, mipCoord, val);
    }

    // Quadrant 3: bottom-right 16×16
    {
        uint2 localXY = localCoord + uint2(16, 16);
        uint2 srcBase = groupBase + localXY * 2;

        float d0 = (srcBase.x < SourceWidth && srcBase.y < SourceHeight)
            ? LoadSource(srcBase + uint2(0, 0)) : FLT_MAX;
        float d1 = (srcBase.x + 1 < SourceWidth && srcBase.y < SourceHeight)
            ? LoadSource(srcBase + uint2(1, 0)) : FLT_MAX;
        float d2 = (srcBase.x < SourceWidth && srcBase.y + 1 < SourceHeight)
            ? LoadSource(srcBase + uint2(0, 1)) : FLT_MAX;
        float d3 = (srcBase.x + 1 < SourceWidth && srcBase.y + 1 < SourceHeight)
            ? LoadSource(srcBase + uint2(1, 1)) : FLT_MAX;

        float val = Reduce4(d0, d1, d2, d3);
        gs_depth[localXY.y][localXY.x] = val;

        uint2 mipCoord = gid.xy * 32 + localXY;
        WriteMip(0, mipCoord, val);
    }

    GroupMemoryBarrierWithGroupSync();

    // ════════════════════════════════════════════════════════════════════
    // Phase 2: Reduce through LDS → Mips 1-5
    // Each step halves the tile. All 256 threads participate for mip 1,
    // then threads drop off as the tile shrinks.
    // ════════════════════════════════════════════════════════════════════

    // Mip 1: 32×32 → 16×16 (all 256 threads)
    // IMPORTANT: read phase must complete before write phase to avoid data race
    // in gs_depth — reads and writes overlap in the [0..15][0..15] region.
    float mip1Val = 0;
    if (MipCount > 1)
    {
        uint2 src = localCoord * 2;
        mip1Val = Reduce4(
            gs_depth[src.y    ][src.x    ],
            gs_depth[src.y    ][src.x + 1],
            gs_depth[src.y + 1][src.x    ],
            gs_depth[src.y + 1][src.x + 1]
        );
    }
    GroupMemoryBarrierWithGroupSync();  // ensure ALL reads complete before writes
    if (MipCount > 1)
    {
        gs_depth[localCoord.y][localCoord.x] = mip1Val;
        WriteMip(1, gid.xy * 16 + localCoord, mip1Val);
    }
    GroupMemoryBarrierWithGroupSync();

    // Mip 2: 16×16 → 8×8 (64 threads)
    float mip2Val = 0;
    if (MipCount > 2 && gidx < 64)
    {
        uint2 xy = uint2(gidx % 8, gidx / 8);
        uint2 src = xy * 2;
        mip2Val = Reduce4(
            gs_depth[src.y    ][src.x    ],
            gs_depth[src.y    ][src.x + 1],
            gs_depth[src.y + 1][src.x    ],
            gs_depth[src.y + 1][src.x + 1]
        );
    }
    GroupMemoryBarrierWithGroupSync();
    if (MipCount > 2 && gidx < 64)
    {
        uint2 xy = uint2(gidx % 8, gidx / 8);
        gs_depth[xy.y][xy.x] = mip2Val;
        WriteMip(2, gid.xy * 8 + xy, mip2Val);
    }
    GroupMemoryBarrierWithGroupSync();

    // Mip 3: 8×8 → 4×4 (16 threads)
    float mip3Val = 0;
    if (MipCount > 3 && gidx < 16)
    {
        uint2 xy = uint2(gidx % 4, gidx / 4);
        uint2 src = xy * 2;
        mip3Val = Reduce4(
            gs_depth[src.y    ][src.x    ],
            gs_depth[src.y    ][src.x + 1],
            gs_depth[src.y + 1][src.x    ],
            gs_depth[src.y + 1][src.x + 1]
        );
    }
    GroupMemoryBarrierWithGroupSync();
    if (MipCount > 3 && gidx < 16)
    {
        uint2 xy = uint2(gidx % 4, gidx / 4);
        gs_depth[xy.y][xy.x] = mip3Val;
        WriteMip(3, gid.xy * 4 + xy, mip3Val);
    }
    GroupMemoryBarrierWithGroupSync();

    // Mip 4: 4×4 → 2×2 (4 threads)
    float mip4Val = 0;
    if (MipCount > 4 && gidx < 4)
    {
        uint2 xy = uint2(gidx % 2, gidx / 2);
        uint2 src = xy * 2;
        mip4Val = Reduce4(
            gs_depth[src.y    ][src.x    ],
            gs_depth[src.y    ][src.x + 1],
            gs_depth[src.y + 1][src.x    ],
            gs_depth[src.y + 1][src.x + 1]
        );
    }
    GroupMemoryBarrierWithGroupSync();
    if (MipCount > 4 && gidx < 4)
    {
        uint2 xy = uint2(gidx % 2, gidx / 2);
        gs_depth[xy.y][xy.x] = mip4Val;
        WriteMip(4, gid.xy * 2 + xy, mip4Val);
    }
    GroupMemoryBarrierWithGroupSync();

    // Mip 5: 2×2 → 1×1 (1 thread)
    if (MipCount > 5 && gidx == 0)
    {
        float val = Reduce4(
            gs_depth[0][0],
            gs_depth[0][1],
            gs_depth[1][0],
            gs_depth[1][1]
        );
        WriteMip(5, gid.xy, val);
    }
    // Mips 6+ are handled by separate small dispatches from C# (negligible cost)
}

// ══════════════════════════════════════════════════════════════════════════
// Per-mip downsample fallback (mips 6+)
// Simple 8×8 thread group, reads from previous mip SRV, writes to current mip UAV.
// Push constants: [0].x = InputSrvIdx, [0].y = OutputUavIdx, [0].z = OutputWidth, [0].w = OutputHeight
// ══════════════════════════════════════════════════════════════════════════

[numthreads(8, 8, 1)]
void CSDownsample(uint3 id : SV_DispatchThreadID)
{
    uint outW = Indices[0].z;
    uint outH = Indices[0].w;
    if (id.x >= outW || id.y >= outH) return;

    Texture2D<float> inputMip = ResourceDescriptorHeap[Indices[0].x];
    uint2 srcCoord = id.xy * 2;

    float d0 = inputMip[srcCoord + uint2(0, 0)];
    float d1 = inputMip[srcCoord + uint2(1, 0)];
    float d2 = inputMip[srcCoord + uint2(0, 1)];
    float d3 = inputMip[srcCoord + uint2(1, 1)];

    d0 = d0 <= 0 ? FLT_MAX : d0;
    d1 = d1 <= 0 ? FLT_MAX : d1;
    d2 = d2 <= 0 ? FLT_MAX : d2;
    d3 = d3 <= 0 ? FLT_MAX : d3;

    RWTexture2D<float> outputMip = ResourceDescriptorHeap[Indices[0].y];
    outputMip[id.xy] = max(max(d0, d1), max(d2, d3));
}

// ══════════════════════════════════════════════════════════════════════════
// Shadow Hi-Z: Single-pass SPD for one cascade slice (min reduction)
// Same push constant layout as CSSinglePassDownsample.
// Shadow maps use standard Z: near=0, far=1.0, clear to 1.0.
// max() reduction → farthest existing depth wins → only cull casters behind ALL existing geometry.
//
// Additional push constant:
//   [5].x = SliceIndex — which array slice to read from source
// ══════════════════════════════════════════════════════════════════════════

#define SliceIndex Indices[5].x

float LoadSourceShadow(uint2 coord)
{
    Texture2DArray<float> src = ResourceDescriptorHeap[SourceSrvIdx];
    return src[uint3(coord, SliceIndex)];
}

float ReduceShadow4(float a, float b, float c, float d)
{
    return max(max(a, b), max(c, d));
}

void WriteMipShadow(uint mip, uint2 coord, float value)
{
    if (mip >= MipCount) return;
    RWTexture2D<float> outMip = ResourceDescriptorHeap[GetMipUavIdx(mip)];
    outMip[coord] = value;
}

groupshared float gs_shadow[32][32];

[numthreads(256, 1, 1)]
void CSSinglePassDownsampleShadow(uint3 gid : SV_GroupID, uint gidx : SV_GroupIndex)
{
    uint2 localCoord = uint2(gidx % 16, gidx / 16);
    uint2 groupBase = gid.xy * 64;

    // Phase 1: Load source → Mip 0 (4 quadrants)
    // Quadrant 0
    {
        uint2 localXY = localCoord;
        uint2 srcBase = groupBase + localXY * 2;
        float d0 = (srcBase.x < SourceWidth && srcBase.y < SourceHeight) ? LoadSourceShadow(srcBase + uint2(0,0)) : 1.0;
        float d1 = (srcBase.x+1 < SourceWidth && srcBase.y < SourceHeight) ? LoadSourceShadow(srcBase + uint2(1,0)) : 1.0;
        float d2 = (srcBase.x < SourceWidth && srcBase.y+1 < SourceHeight) ? LoadSourceShadow(srcBase + uint2(0,1)) : 1.0;
        float d3 = (srcBase.x+1 < SourceWidth && srcBase.y+1 < SourceHeight) ? LoadSourceShadow(srcBase + uint2(1,1)) : 1.0;
        float val = ReduceShadow4(d0, d1, d2, d3);
        gs_shadow[localXY.y][localXY.x] = val;
        WriteMipShadow(0, gid.xy * 32 + localXY, val);
    }
    // Quadrant 1
    {
        uint2 localXY = localCoord + uint2(16, 0);
        uint2 srcBase = groupBase + localXY * 2;
        float d0 = (srcBase.x < SourceWidth && srcBase.y < SourceHeight) ? LoadSourceShadow(srcBase + uint2(0,0)) : 1.0;
        float d1 = (srcBase.x+1 < SourceWidth && srcBase.y < SourceHeight) ? LoadSourceShadow(srcBase + uint2(1,0)) : 1.0;
        float d2 = (srcBase.x < SourceWidth && srcBase.y+1 < SourceHeight) ? LoadSourceShadow(srcBase + uint2(0,1)) : 1.0;
        float d3 = (srcBase.x+1 < SourceWidth && srcBase.y+1 < SourceHeight) ? LoadSourceShadow(srcBase + uint2(1,1)) : 1.0;
        float val = ReduceShadow4(d0, d1, d2, d3);
        gs_shadow[localXY.y][localXY.x] = val;
        WriteMipShadow(0, gid.xy * 32 + localXY, val);
    }
    // Quadrant 2
    {
        uint2 localXY = localCoord + uint2(0, 16);
        uint2 srcBase = groupBase + localXY * 2;
        float d0 = (srcBase.x < SourceWidth && srcBase.y < SourceHeight) ? LoadSourceShadow(srcBase + uint2(0,0)) : 1.0;
        float d1 = (srcBase.x+1 < SourceWidth && srcBase.y < SourceHeight) ? LoadSourceShadow(srcBase + uint2(1,0)) : 1.0;
        float d2 = (srcBase.x < SourceWidth && srcBase.y+1 < SourceHeight) ? LoadSourceShadow(srcBase + uint2(0,1)) : 1.0;
        float d3 = (srcBase.x+1 < SourceWidth && srcBase.y+1 < SourceHeight) ? LoadSourceShadow(srcBase + uint2(1,1)) : 1.0;
        float val = ReduceShadow4(d0, d1, d2, d3);
        gs_shadow[localXY.y][localXY.x] = val;
        WriteMipShadow(0, gid.xy * 32 + localXY, val);
    }
    // Quadrant 3
    {
        uint2 localXY = localCoord + uint2(16, 16);
        uint2 srcBase = groupBase + localXY * 2;
        float d0 = (srcBase.x < SourceWidth && srcBase.y < SourceHeight) ? LoadSourceShadow(srcBase + uint2(0,0)) : 1.0;
        float d1 = (srcBase.x+1 < SourceWidth && srcBase.y < SourceHeight) ? LoadSourceShadow(srcBase + uint2(1,0)) : 1.0;
        float d2 = (srcBase.x < SourceWidth && srcBase.y+1 < SourceHeight) ? LoadSourceShadow(srcBase + uint2(0,1)) : 1.0;
        float d3 = (srcBase.x+1 < SourceWidth && srcBase.y+1 < SourceHeight) ? LoadSourceShadow(srcBase + uint2(1,1)) : 1.0;
        float val = ReduceShadow4(d0, d1, d2, d3);
        gs_shadow[localXY.y][localXY.x] = val;
        WriteMipShadow(0, gid.xy * 32 + localXY, val);
    }

    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Mip reductions through LDS (same split read→barrier→write pattern)
    // Mip 1: 32×32 → 16×16
    float m1 = 1.0;
    if (MipCount > 1) { uint2 s = localCoord * 2; m1 = ReduceShadow4(gs_shadow[s.y][s.x], gs_shadow[s.y][s.x+1], gs_shadow[s.y+1][s.x], gs_shadow[s.y+1][s.x+1]); }
    GroupMemoryBarrierWithGroupSync();
    if (MipCount > 1) { gs_shadow[localCoord.y][localCoord.x] = m1; WriteMipShadow(1, gid.xy*16 + localCoord, m1); }
    GroupMemoryBarrierWithGroupSync();

    // Mip 2: 16×16 → 8×8
    float m2 = 1.0;
    if (MipCount > 2 && gidx < 64) { uint2 xy = uint2(gidx%8, gidx/8); uint2 s = xy*2; m2 = ReduceShadow4(gs_shadow[s.y][s.x], gs_shadow[s.y][s.x+1], gs_shadow[s.y+1][s.x], gs_shadow[s.y+1][s.x+1]); }
    GroupMemoryBarrierWithGroupSync();
    if (MipCount > 2 && gidx < 64) { uint2 xy = uint2(gidx%8, gidx/8); gs_shadow[xy.y][xy.x] = m2; WriteMipShadow(2, gid.xy*8 + xy, m2); }
    GroupMemoryBarrierWithGroupSync();

    // Mip 3: 8×8 → 4×4
    float m3 = 1.0;
    if (MipCount > 3 && gidx < 16) { uint2 xy = uint2(gidx%4, gidx/4); uint2 s = xy*2; m3 = ReduceShadow4(gs_shadow[s.y][s.x], gs_shadow[s.y][s.x+1], gs_shadow[s.y+1][s.x], gs_shadow[s.y+1][s.x+1]); }
    GroupMemoryBarrierWithGroupSync();
    if (MipCount > 3 && gidx < 16) { uint2 xy = uint2(gidx%4, gidx/4); gs_shadow[xy.y][xy.x] = m3; WriteMipShadow(3, gid.xy*4 + xy, m3); }
    GroupMemoryBarrierWithGroupSync();

    // Mip 4: 4×4 → 2×2
    float m4 = 1.0;
    if (MipCount > 4 && gidx < 4) { uint2 xy = uint2(gidx%2, gidx/2); uint2 s = xy*2; m4 = ReduceShadow4(gs_shadow[s.y][s.x], gs_shadow[s.y][s.x+1], gs_shadow[s.y+1][s.x], gs_shadow[s.y+1][s.x+1]); }
    GroupMemoryBarrierWithGroupSync();
    if (MipCount > 4 && gidx < 4) { uint2 xy = uint2(gidx%2, gidx/2); gs_shadow[xy.y][xy.x] = m4; WriteMipShadow(4, gid.xy*2 + xy, m4); }
    GroupMemoryBarrierWithGroupSync();

    // Mip 5: 2×2 → 1×1
    if (MipCount > 5 && gidx == 0)
    {
        float val = ReduceShadow4(gs_shadow[0][0], gs_shadow[0][1], gs_shadow[1][0], gs_shadow[1][1]);
        WriteMipShadow(5, gid.xy, val);
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Shadow per-mip downsample fallback (mips 6+)
// Push constants: [0].x = InputSrvIdx, [0].y = OutputUavIdx, [0].z = Width, [0].w = Height
// Reads from Texture2DArray single-slice SRV, min() reduction.
// ══════════════════════════════════════════════════════════════════════════

[numthreads(8, 8, 1)]
void CSDownsampleShadow(uint3 id : SV_DispatchThreadID)
{
    uint outW = Indices[0].z;
    uint outH = Indices[0].w;
    if (id.x >= outW || id.y >= outH) return;

    // Input is a single-slice SRV (Texture2DArray with ArraySize=1)
    Texture2DArray<float> inputMip = ResourceDescriptorHeap[Indices[0].x];
    uint2 srcCoord = id.xy * 2;

    float d0 = inputMip[uint3(srcCoord + uint2(0,0), 0)];
    float d1 = inputMip[uint3(srcCoord + uint2(1,0), 0)];
    float d2 = inputMip[uint3(srcCoord + uint2(0,1), 0)];
    float d3 = inputMip[uint3(srcCoord + uint2(1,1), 0)];

    RWTexture2D<float> outputMip = ResourceDescriptorHeap[Indices[0].y];
    outputMip[id.xy] = max(max(d0, d1), max(d2, d3));
}
