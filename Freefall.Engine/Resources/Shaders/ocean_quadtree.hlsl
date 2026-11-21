// The sea's grid round the camera (Graphics/OceanQuadtree.cs, drawn by ocean.fx; .agent/knowledge/ocean_system/overview.md
// "The grid: a quadtree round the camera", 2026-11-21). The open sea is one plane as wide as its entity's scale, and one
// grid stretched over it gave TestWorld's 65 km sea 512 m cells: no tessellation goes past 64 segments an edge, so the
// waves next to a ship stood in 10 m facets. Here the sea is a quadtree over that square, split by distance alone (the
// sea is flat, so no height range is needed as the terrain's is): a node splits while the camera is nearer to it than
// SplitDistance times its size, down to MaxDepth, whose leaves have cells of about 2 m. Every leaf is drawn as
// OCEAN_LEAF_CELLS² cells of two triangle patches each, tessellated by their size on screen as before.
//
// One group walks the tree level by level from the root. A node is dropped with all it holds when its box (the waves'
// bound round it) lies outside the view or behind the opaque scene in this frame's Hi-Z, or wholly over land where the
// sea mask has no sea (the sea's reach, OceanQuadtree.BuildReach); the hull shader still culls each patch of a leaf.
//
// No split flags and no neighbour pass: with SplitDistance ≥ √2 two leaves sharing an edge are never more than one level
// apart. If a node of size s/2 splits, the camera is nearer to it than SplitDistance·s/2; a node of size s touching it is
// then nearer than that plus the small node's diagonal, √2·s/2, which is below SplitDistance·s, so it splits too.
// Whether a neighbour is coarser is then whether its parent splits, a function of the camera alone; the leaf snaps every
// second control point on that edge onto its neighbour's (ocean_patch.hlsli VS_Leaf), as the terrain's patches stitch.
#pragma kernel CSWalk

#include "ocean_leaf.hlsli"

cbuffer PushConstants : register(b3)
{
    uint ParamsSrvIdx;      // StructuredBuffer<OceanWalk>: one entry per view and frame in flight (OceanQuadtree)
    uint ParamsSlot;        // this walk's entry
    uint LeavesUAVIdx;      // RWStructuredBuffer<uint>: the leaves (PackOceanNode)
    uint ArgsUAVIdx;        // RWByteAddressBuffer: D3D12_DRAW_ARGUMENTS for the ocean's ExecuteIndirect
};

// OceanQuadtree.WalkParams, field for field
struct OceanWalk
{
    row_major float4x4 ViewProjection;
    float3 Camera;          // what the split is measured from (frozen with the frustum)
    float SeaLevel;
    float2 RootMin;         // the root square's corner (world xz)
    float RootSize;         // its side (m)
    uint MaxDepth;          // the leaves' deepest level (12 at most)
    float4 Extent;          // the ocean's own rectangle, min xz and max xz: nodes wholly outside are dropped
    float4 MaskRect;        // where the sea mask lies (the terrain): origin xz, size xz
    uint HiZSrv;            // this frame's Hi-Z pyramid (linear depth, max), 0 = none
    uint ReachSrv;          // StructuredBuffer<uint>: the sea's reach bits (REACH_*), 0 = sea everywhere
    uint Uniform;           // 1: the old fixed grid (EngineSettings.OceanUniformGrid): every node split to MaxDepth, none dropped
    uint MaxLeaves;
    float SplitDistance;    // a node splits while the camera is nearer than this times its size
    float Bound;            // how far the waves move the surface, every way (m): widens every node's box
    float2 _pad;
};

// The sea's reach: bit levels from REACH_RES² over the terrain down to 1, each bit set where any sea lies within that
// cell or the next (OceanQuadtree.BuildReach)
#define REACH_RES 256u
#define REACH_LEVELS 9

#define GROUP 256
#define LIST_CAP 2048

groupshared uint gNodes[2][LIST_CAP];
groupshared uint gCount[2];
groupshared uint gLeaves;

float NodeSize(OceanWalk w, uint depth)
{
    return w.RootSize / (float)(1u << depth);
}

bool ShouldSplit(OceanWalk w, uint depth, uint ix, uint iz)
{
    if (depth >= w.MaxDepth) return false;
    if (w.Uniform != 0) return true;
    float size = NodeSize(w, depth);
    float2 lo = w.RootMin + float2(ix, iz) * size;
    float2 nearest = clamp(w.Camera.xz, lo, lo + size);
    float3 d = float3(w.Camera.x - nearest.x, w.Camera.y - w.SeaLevel, w.Camera.z - nearest.y);
    float reach = w.SplitDistance * size;
    return dot(d, d) < reach * reach;
}

// The edges where the neighbour is a leaf a level coarser: its parent does not split (a sibling's parent is this node's
// own, which did). Past the root's edge there is nothing to meet
uint Stitch(OceanWalk w, uint depth, uint ix, uint iz)
{
    if (depth == 0 || w.Uniform != 0) return 0;
    uint n = 1u << depth;
    uint stitch = 0;
    if (ix > 0     && !ShouldSplit(w, depth - 1, (ix - 1) >> 1, iz >> 1)) stitch |= 1;
    if (ix + 1 < n && !ShouldSplit(w, depth - 1, (ix + 1) >> 1, iz >> 1)) stitch |= 2;
    if (iz > 0     && !ShouldSplit(w, depth - 1, ix >> 1, (iz - 1) >> 1)) stitch |= 4;
    if (iz + 1 < n && !ShouldSplit(w, depth - 1, ix >> 1, (iz + 1) >> 1)) stitch |= 8;
    return stitch;
}

uint ReachOffset(uint level)
{
    uint offset = 0;
    for (uint l = 0; l < level; l++)
    {
        uint res = REACH_RES >> l;
        offset += res * res;
    }
    return offset;
}

// Whether any sea lies under xz in [lo, hi] (undisplaced: what the pixel shader's OutsideSea reads). Past the terrain it
// is all sea
bool SeaWithin(OceanWalk w, float2 lo, float2 hi)
{
    if (w.ReachSrv == 0) return true;
    float2 u0 = (lo - w.MaskRect.xy) / w.MaskRect.zw;
    float2 u1 = (hi - w.MaskRect.xy) / w.MaskRect.zw;
    if (any(u0 < 0.0) || any(u1 > 1.0)) return true;

    // The level where the rectangle is at most a cell across: it covers at most 2 x 2 cells there
    float extent = max(u1.x - u0.x, u1.y - u0.y) * REACH_RES;
    uint level = min((uint)ceil(log2(max(extent, 1.0))), REACH_LEVELS - 1);
    uint res = REACH_RES >> level;
    uint2 c0 = min(uint2(u0 * res), res - 1);
    uint2 c1 = min(uint2(u1 * res), res - 1);
    uint offset = ReachOffset(level);
    StructuredBuffer<uint> reach = ResourceDescriptorHeap[w.ReachSrv];
    for (uint z = c0.y; z <= c1.y; z++)
    for (uint x = c0.x; x <= c1.x; x++)
    {
        uint bit = offset + z * res + x;
        if (reach[bit >> 5] & (1u << (bit & 31))) return true;
    }
    return false;
}

// Whether the box lies outside the view or behind the opaque scene: as ocean_patch.hlsli CullTriangle tests a patch
bool OutOfSight(OceanWalk w, float3 minP, float3 maxP)
{
    uint outside = 0x1F;
    float2 ndcMin = 1e30, ndcMax = -1e30;
    float nearestW = 1e30;
    bool reachesBehind = false;
    [unroll]
    for (uint i = 0; i < 8; i++)
    {
        float3 c = float3((i & 1) ? maxP.x : minP.x, (i & 2) ? maxP.y : minP.y, (i & 4) ? maxP.z : minP.z);
        float4 clip = mul(float4(c, 1.0), w.ViewProjection);

        uint beyond = 0;
        if (clip.x < -clip.w) beyond |= 1;
        if (clip.x >  clip.w) beyond |= 2;
        if (clip.y < -clip.w) beyond |= 4;
        if (clip.y >  clip.w) beyond |= 8;
        if (clip.w <= 0.0)    beyond |= 16;
        outside &= beyond;

        if (clip.w <= 0.01) reachesBehind = true;
        else
        {
            float2 ndc = clip.xy / clip.w;
            ndcMin = min(ndcMin, ndc);
            ndcMax = max(ndcMax, ndc);
            nearestW = min(nearestW, clip.w);
        }
    }

    if (outside != 0) return true;
    if (w.HiZSrv == 0 || reachesBehind) return false;

    // Hi-Z: linear view depth, max over each texel's footprint (sky = FLT_MAX). At the mip where the box's screen
    // rectangle spans at most one texel it touches at most 2 x 2 texels; hidden if its nearest point is behind them all
    ndcMin = max(ndcMin, -1.0);
    ndcMax = min(ndcMax, 1.0);
    float2 uvMin = float2(ndcMin.x, -ndcMax.y) * 0.5 + 0.5;
    float2 uvMax = float2(ndcMax.x, -ndcMin.y) * 0.5 + 0.5;

    Texture2D<float> hiZ = ResourceDescriptorHeap[w.HiZSrv];
    float tw, th, levels;
    hiZ.GetDimensions(0, tw, th, levels);
    float2 extent = (uvMax - uvMin) * float2(tw, th);
    uint mip = (uint)min(ceil(log2(max(max(extent.x, extent.y), 1.0))), levels - 1.0);

    float mw, mh, unused;
    hiZ.GetDimensions(mip, mw, mh, unused);
    int2 maxCoord = int2(mw, mh) - 1;
    int2 t0 = clamp(int2(uvMin * float2(mw, mh)), int2(0, 0), maxCoord);
    int2 t1 = clamp(int2(uvMax * float2(mw, mh)), int2(0, 0), maxCoord);
    float occluder = max(max(hiZ.Load(int3(t0.x, t0.y, mip)), hiZ.Load(int3(t1.x, t0.y, mip))),
                         max(hiZ.Load(int3(t0.x, t1.y, mip)), hiZ.Load(int3(t1.x, t1.y, mip))));
    return nearestW > occluder;
}

// Whether a node is left out with everything under it
bool Dropped(OceanWalk w, uint depth, uint ix, uint iz)
{
    if (w.Uniform != 0) return false;   // the old grid: only its patches were culled (hull shader)
    float size = NodeSize(w, depth);
    float2 lo = w.RootMin + float2(ix, iz) * size;
    float2 hi = lo + size;
    if (any(hi < w.Extent.xy) || any(lo > w.Extent.zw)) return true;
    if (!SeaWithin(w, lo, hi)) return true;
    float3 minP = float3(lo.x - w.Bound, w.SeaLevel - w.Bound, lo.y - w.Bound);
    float3 maxP = float3(hi.x + w.Bound, w.SeaLevel + w.Bound, hi.y + w.Bound);
    return OutOfSight(w, minP, maxP);
}

[numthreads(GROUP, 1, 1)]
void CSWalk(uint t : SV_GroupIndex)
{
    StructuredBuffer<OceanWalk> walks = ResourceDescriptorHeap[ParamsSrvIdx];
    OceanWalk w = walks[ParamsSlot];
    RWStructuredBuffer<uint> leaves = ResourceDescriptorHeap[LeavesUAVIdx];

    if (t == 0)
    {
        gNodes[0][0] = PackOceanNode(0, 0, 0, 0);
        gCount[0] = 1;
        gCount[1] = 0;
        gLeaves = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint cur = 0;
    for (uint depth = 0; depth <= w.MaxDepth; depth++)
    {
        uint count = min(gCount[cur], (uint)LIST_CAP);
        for (uint i = t; i < count; i += GROUP)
        {
            uint d, stitch, ix, iz;
            UnpackOceanNode(gNodes[cur][i], d, stitch, ix, iz);
            if (Dropped(w, depth, ix, iz)) continue;

            if (ShouldSplit(w, depth, ix, iz))
            {
                uint first;
                InterlockedAdd(gCount[cur ^ 1], 4, first);
                if (first + 4 <= LIST_CAP)
                {
                    [unroll]
                    for (uint c = 0; c < 4; c++)
                        gNodes[cur ^ 1][first + c] = PackOceanNode(depth + 1, 0, ix * 2 + (c & 1), iz * 2 + (c >> 1));
                    continue;
                }
                // No room for its children (never seen: a level holds a few hundred at most): drawn whole instead
            }

            uint at;
            InterlockedAdd(gLeaves, 1, at);
            if (at < w.MaxLeaves)
                leaves[at] = PackOceanNode(depth, Stitch(w, depth, ix, iz), ix, iz);
        }
        GroupMemoryBarrierWithGroupSync();
        if (t == 0) gCount[cur] = 0;
        cur ^= 1;
        GroupMemoryBarrierWithGroupSync();
        if (gCount[cur] == 0) break;
    }

    // Near first: the leaves were found level by level from the root, so the far sea's lie at the front. Turned round,
    // the waves round the eye draw first and the depth test spares the pixel shader some of the sea they hide (a low
    // view over a storm sea at TestWorld's ship bench: 0.1 ms of 2.5)
    AllMemoryBarrierWithGroupSync();
    uint found = min(gLeaves, w.MaxLeaves);
    for (uint i = t; i < found / 2; i += GROUP)
    {
        uint front = leaves[i];
        leaves[i] = leaves[found - 1 - i];
        leaves[found - 1 - i] = front;
    }

    if (t == 0)
    {
        RWByteAddressBuffer args = ResourceDescriptorHeap[ArgsUAVIdx];
        args.Store4(0, uint4(OCEAN_LEAF_VERTICES, found, 0, 0));
    }
}
