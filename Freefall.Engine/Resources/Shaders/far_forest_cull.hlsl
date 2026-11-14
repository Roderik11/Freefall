// The far forest's cull (Graphics/FarForest.cs), as the grass's spawn picks its plants each frame: by cell first, then
// the trees of the cells left.
//   CSResetCamera      a thread per kind: its camera draw command with no instances yet; thread 0: no cells yet
//   CSCullCells        a thread per cell: on the far tier's far side (one share for the whole cell), its box in view and
//                      not behind last frame's depth (Hi-Z, as grass_compute.hlsl IsCellOccluded): onto the cell list,
//                      which is the indirect dispatch of CSCullTrees (a group per cell)
//   CSCullTrees        a group per listed cell, its threads over its trees: in view and at least MinPixels across, a
//                      tree goes on its kind's visible list; the command's instance count is the list's length
//   CSResetShadow      as CSResetCamera, for the shadow commands and the shadow cell list
//   CSCullShadowCells  a thread per cell that casts (the far side past half way, common.fx FarTierCasts): the cascades
//                      its box touches as a mask, onto the shadow cell list
//   CSCullShadowTrees  a group per listed cell: an entry (far_forest.hlsli) for every cascade of the mask the tree
//                      touches; past the first cascade only trees covering ShadowMinCasterTexels texels (as
//                      InstanceBatch's shadow culling, CoversShadowTexels). The camera's thinning applies.
//   CSClampShadow      a thread per kind: a list that ran over its room draws only what it holds
// A draw command is far_forest.hlsli's FAR_COMMAND_UINTS (an indexed draw of the kind's part); a dispatch is 3 uints.

#pragma kernel CSResetCamera
#pragma kernel CSCullCells
#pragma kernel CSCullTrees
#pragma kernel CSResetShadow
#pragma kernel CSCullShadowCells
#pragma kernel CSCullShadowTrees
#pragma kernel CSClampShadow

cbuffer PushConstants : register(b3)
{
    uint TreesIdx;              // StructuredBuffer<FarTree>, by cell
    uint TreeCount;
    uint KindsIdx;              // StructuredBuffer<FarKind>
    uint KindCount;
    uint VisibleUAVIdx;         // RWStructuredBuffer<uint>: the camera's lists, a tree index each
    uint ArgsUAVIdx;            // RWByteAddressBuffer: the camera's draw commands, a kind each
    uint ShadowUAVIdx;          // RWStructuredBuffer<uint>: the shadow lists, (cascade << 28 | tree) each
    uint ShadowArgsUAVIdx;      // RWByteAddressBuffer: the shadow draw commands
    uint CascadeIdx;            // StructuredBuffer<CascadeData> (DirectionalLight.CurrentCascadeSrvIndex)
    uint CascadeCount;
    uint CellsIdx;              // StructuredBuffer<FarCell>
    uint CellCount;
    uint CellListUAVIdx;        // RWStructuredBuffer<uint>: the camera's cells left (cell index)
    uint CellArgsUAVIdx;        // RWByteAddressBuffer: CSCullTrees' dispatch (groups, 1, 1)
    uint ShadowCellListUAVIdx;  // RWStructuredBuffer<uint>: the shadow cells left (cell | cascade mask << 24)
    uint ShadowCellArgsUAVIdx;  // RWByteAddressBuffer: CSCullShadowTrees' dispatch
};

cbuffer FarForestView : register(b4)
{
    row_major float4x4 ViewProjection;
    row_major float4x4 OcclusionProjection;   // last frame's, which the Hi-Z pyramid was drawn with
    float3 CamPos;
    float ClipSlackX;           // clip units a metre of radius can reach across, at a view depth of one metre (P11 + 1)
    float ClipSlackY;
    float PixelsPerMetre;       // pixels a metre spans at a distance of one metre
    float MinPixels;            // a tree's radius on screen below this is not drawn
    float FarFrom;              // the far tier (Graphics.FarTier), as the shaders have it in SceneConstants; its cell is
    float FarBand;              // in each FarCell (the centre it fades by)
    float MinCasterClip;        // clip units a caster must span in the cascades after the first
    uint HiZTexture;            // the camera's Hi-Z pyramid (linear depth, farthest of 2x2); 0 = none
    float2 HiZSize;
    uint HiZMipCount;
    float ThinPixels;           // a crown narrower on screen than this is thinned (far_forest.hlsli); 0 = never
    float MaxGrow;              // how far a kept tree may grow
};

#include "far_forest.hlsli"

// As common.fx CascadeData (240 bytes)
struct CascadeData
{
    float4 Planes[6];
    row_major float4x4 VP;
    row_major float4x4 PrevVP;
    float4 SplitDistances;
};

// The far side's share of a cell, quantized as common.fx FarTierShows does (0..127)
uint FarShare(float2 cellCentre)
{
    if (FarBand <= 0.0)
        return 0u;
    float share = saturate((length(cellCentre - CamPos.xz) - FarFrom) / FarBand);
    return (uint)(share * 127.0 + 0.5);
}

// A sphere against the view's sides (clip space), in front of the camera
bool InView(float3 centre, float radius)
{
    float4 clip = mul(float4(centre, 1.0), ViewProjection);
    if (clip.w < -radius) return false;
    if (abs(clip.x) > clip.w + radius * ClipSlackX) return false;
    if (abs(clip.y) > clip.w + radius * ClipSlackY) return false;
    return true;
}

// Behind what last frame drew (grass_compute.hlsl IsCellOccluded): the sphere's nearest depth past the farthest depth
// of the pyramid texels under it
bool Occluded(float3 centre, float radius)
{
    if (HiZTexture == 0u) return false;
    float4 clipCenter = mul(float4(centre, 1.0), OcclusionProjection);
    if (clipCenter.w <= radius) return false;
    float3 ndc = clipCenter.xyz / clipCenter.w;
    float2 uv = ndc.xy * float2(0.5, -0.5) + 0.5;
    if (any(uv < 0.0) || any(uv > 1.0)) return false;

    Texture2D<float> hiZ = ResourceDescriptorHeap[HiZTexture];
    float projScale = abs(OcclusionProjection._m11);
    projScale = projScale < 0.001 ? 1.0 : projScale;
    float screenRadius = (radius * projScale / clipCenter.w) * HiZSize.y * 0.5;
    float mipLevel = min(ceil(log2(max(screenRadius * 2.0, 1.0))), float(HiZMipCount) - 1.0);
    uint mip = (uint)mipLevel;
    float mipW, mipH, levels;
    hiZ.GetDimensions(mip, mipW, mipH, levels);
    float2 mipSize = float2(mipW, mipH);
    int2 baseCoord = int2(uv * mipSize - 0.5);
    int2 maxCoord = int2(mipSize) - 1;
    float d0 = hiZ.Load(int3(clamp(baseCoord, int2(0, 0), maxCoord), mip));
    float d1 = hiZ.Load(int3(clamp(baseCoord + int2(1, 0), int2(0, 0), maxCoord), mip));
    float d2 = hiZ.Load(int3(clamp(baseCoord + int2(0, 1), int2(0, 0), maxCoord), mip));
    float d3 = hiZ.Load(int3(clamp(baseCoord + int2(1, 1), int2(0, 0), maxCoord), mip));
    float farthest = max(max(d0, d1), max(d2, d3));
    return clipCenter.w - radius > farthest;
}

void WriteCommand(RWByteAddressBuffer args, uint kind, FarKind k, uint baseOffset)
{
    uint at = kind * FAR_COMMAND_UINTS * 4u;
    args.Store(at + 0u, k.MeshPartId);
    args.Store(at + 4u, baseOffset);
    args.Store(at + 8u, k.IndexAddressLo);
    args.Store(at + 12u, k.IndexAddressHi);
    args.Store(at + 16u, k.IndexBytes);
    args.Store(at + 20u, 42u);                  // DXGI_FORMAT_R32_UINT
    args.Store(at + 24u, k.IndexCount);
    args.Store(at + 28u, 0u);                   // instances: the cull counts them
    args.Store(at + 32u, k.BaseIndex);
    args.Store(at + 36u, 0u);                   // base vertex: the indices are the mesh's own
    args.Store(at + 40u, 0u);
    args.Store(at + 44u, 0u);
}

void TreeSphere(FarTree t, FarKind k, out float3 centre, out float radius)
{
    float scale = FarTreeScale(t);
    centre = t.Position + float3(0.0, k.CentreY * scale, 0.0);
    radius = k.Radius * scale;
}

uint FarHash(uint v)
{
    v = v * 747796405u + 2891336453u;
    uint w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}

// Thinned out at this distance (far_forest.hlsli): its crown's width on screen against ThinPixels; which trees stay is
// fixed by where they stand, so nothing flickers. A kept one's growth, else 1
bool Thinned(FarTree t, FarKind k, float distance, out float grow)
{
    grow = 1.0;
    if (ThinPixels <= 0.0) return false;
    float width = 2.0 * k.HalfWidth * FarTreeScale(t) * PixelsPerMetre / distance;
    if (width >= ThinPixels) return false;
    float share = width / ThinPixels;
    share *= share;
    grow = min(rsqrt(max(share, 1e-6)), max(MaxGrow, 1.0));
    uint h = FarHash(asuint(t.Position.x) ^ FarHash(asuint(t.Position.z)));
    return (float)(h & 0xFFFFFFu) * (1.0 / 16777216.0) >= share;
}

// ─── The camera ──────────────────────────────────────────────────────────────

[numthreads(64, 1, 1)]
void CSResetCamera(uint3 id : SV_DispatchThreadID)
{
    if (id.x == 0u)
    {
        RWByteAddressBuffer cellArgs = ResourceDescriptorHeap[CellArgsUAVIdx];
        cellArgs.Store3(0u, uint3(0u, 1u, 1u));
    }
    if (id.x >= KindCount) return;
    StructuredBuffer<FarKind> kinds = ResourceDescriptorHeap[KindsIdx];
    RWByteAddressBuffer args = ResourceDescriptorHeap[ArgsUAVIdx];
    FarKind k = kinds[id.x];
    WriteCommand(args, id.x, k, k.VisibleBase);
}

[numthreads(64, 1, 1)]
void CSCullCells(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= CellCount) return;
    StructuredBuffer<FarCell> cells = ResourceDescriptorHeap[CellsIdx];
    FarCell c = cells[id.x];

    // Nearer than the far tier's band its instances draw it
    if (FarShare(c.Centre) == 0u) return;

    float3 centre = (c.Min + c.Max) * 0.5;
    float radius = length(c.Max - c.Min) * 0.5 * max(MaxGrow, 1.0);   // its trees may grow (thinning)
    if (!InView(centre, radius) || Occluded(centre, radius)) return;

    RWByteAddressBuffer cellArgs = ResourceDescriptorHeap[CellArgsUAVIdx];
    RWStructuredBuffer<uint> list = ResourceDescriptorHeap[CellListUAVIdx];
    uint slot;
    cellArgs.InterlockedAdd(0u, 1u, slot);
    list[slot] = id.x;
}

[numthreads(64, 1, 1)]
void CSCullTrees(uint3 group : SV_GroupID, uint gi : SV_GroupIndex)
{
    RWStructuredBuffer<uint> list = ResourceDescriptorHeap[CellListUAVIdx];   // read through its UAV (no SRV bound)
    StructuredBuffer<FarCell> cells = ResourceDescriptorHeap[CellsIdx];
    StructuredBuffer<FarTree> trees = ResourceDescriptorHeap[TreesIdx];
    StructuredBuffer<FarKind> kinds = ResourceDescriptorHeap[KindsIdx];
    RWByteAddressBuffer args = ResourceDescriptorHeap[ArgsUAVIdx];
    RWStructuredBuffer<uint> visible = ResourceDescriptorHeap[VisibleUAVIdx];
    FarCell c = cells[list[group.x]];

    for (uint i = gi; i < c.Count; i += 64u)
    {
        uint index = c.First + i;
        FarTree t = trees[index];
        uint kind = FarTreeKind(t);
        FarKind k = kinds[kind];
        float3 centre; float radius;
        TreeSphere(t, k, centre, radius);
        float distance = max(length(centre - CamPos), 1.0);
        float grow;
        if (Thinned(t, k, distance, grow)) continue;
        centre = t.Position + (centre - t.Position) * grow;
        radius *= grow;
        if (!InView(centre, radius)) continue;
        if (radius * PixelsPerMetre / distance < MinPixels) continue;

        uint slot;
        args.InterlockedAdd(kind * FAR_COMMAND_UINTS * 4u + FAR_COMMAND_INSTANCES, 1u, slot);
        visible[k.VisibleBase + slot] = index | (FarGrowBits(grow, 255u) << 24);
    }
}

// ─── The sun's cascades ──────────────────────────────────────────────────────

[numthreads(64, 1, 1)]
void CSResetShadow(uint3 id : SV_DispatchThreadID)
{
    if (id.x == 0u)
    {
        RWByteAddressBuffer cellArgs = ResourceDescriptorHeap[ShadowCellArgsUAVIdx];
        cellArgs.Store3(0u, uint3(0u, 1u, 1u));
    }
    if (id.x >= KindCount) return;
    StructuredBuffer<FarKind> kinds = ResourceDescriptorHeap[KindsIdx];
    RWByteAddressBuffer args = ResourceDescriptorHeap[ShadowArgsUAVIdx];
    FarKind k = kinds[id.x];
    WriteCommand(args, id.x, k, k.ShadowBase);
}

// A sphere against a cascade's square (orthographic: its reach in clip units is the same everywhere)
bool InCascade(row_major float4x4 vp, float3 centre, float radius, out float reachX)
{
    float4 clip = mul(float4(centre, 1.0), vp);
    reachX = radius * length(float3(vp._11, vp._21, vp._31));
    float reachY = radius * length(float3(vp._12, vp._22, vp._32));
    return abs(clip.x) <= 1.0 + reachX && abs(clip.y) <= 1.0 + reachY;
}

[numthreads(64, 1, 1)]
void CSCullShadowCells(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= CellCount) return;
    StructuredBuffer<FarCell> cells = ResourceDescriptorHeap[CellsIdx];
    FarCell c = cells[id.x];

    // The far side casts from half way through the band (the instances before)
    if (FarShare(c.Centre) < 64u) return;

    float3 centre = (c.Min + c.Max) * 0.5;
    float radius = length(c.Max - c.Min) * 0.5;
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeIdx];
    uint mask = 0u;
    for (uint s = 0; s < CascadeCount; s++)
    {
        float reach;
        if (InCascade(cascades[s].VP, centre, radius, reach))
            mask |= 1u << s;
    }
    if (mask == 0u) return;

    RWByteAddressBuffer cellArgs = ResourceDescriptorHeap[ShadowCellArgsUAVIdx];
    RWStructuredBuffer<uint> list = ResourceDescriptorHeap[ShadowCellListUAVIdx];
    uint slot;
    cellArgs.InterlockedAdd(0u, 1u, slot);
    list[slot] = id.x | (mask << 24);
}

[numthreads(64, 1, 1)]
void CSCullShadowTrees(uint3 group : SV_GroupID, uint gi : SV_GroupIndex)
{
    RWStructuredBuffer<uint> list = ResourceDescriptorHeap[ShadowCellListUAVIdx];   // read through its UAV
    StructuredBuffer<FarCell> cells = ResourceDescriptorHeap[CellsIdx];
    StructuredBuffer<FarTree> trees = ResourceDescriptorHeap[TreesIdx];
    StructuredBuffer<FarKind> kinds = ResourceDescriptorHeap[KindsIdx];
    StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[CascadeIdx];
    RWByteAddressBuffer args = ResourceDescriptorHeap[ShadowArgsUAVIdx];
    RWStructuredBuffer<uint> entries = ResourceDescriptorHeap[ShadowUAVIdx];
    uint packed = list[group.x];
    FarCell c = cells[packed & 0xFFFFFFu];
    uint mask = packed >> 24;

    for (uint i = gi; i < c.Count; i += 64u)
    {
        uint index = c.First + i;
        FarTree t = trees[index];
        uint kind = FarTreeKind(t);
        FarKind k = kinds[kind];
        float3 centre; float radius;
        TreeSphere(t, k, centre, radius);
        // The trees the camera keeps cast, grown as it draws them (thinned by its distance)
        float grow;
        if (Thinned(t, k, max(length(centre - CamPos), 1.0), grow)) continue;
        centre = t.Position + (centre - t.Position) * grow;
        radius *= grow;
        uint growBits = FarGrowBits(grow, 127u) << 22;
        for (uint s = 0; s < CascadeCount; s++)
        {
            if ((mask & (1u << s)) == 0u) continue;
            float reach;
            if (!InCascade(cascades[s].VP, centre, radius, reach)) continue;
            if (s > 0u && 2.0 * reach < MinCasterClip) continue;   // a texel or two: screen-space shadows carry it
            uint slot;
            args.InterlockedAdd(kind * FAR_COMMAND_UINTS * 4u + FAR_COMMAND_INSTANCES, 1u, slot);
            if (slot < k.ShadowCapacity)
                entries[k.ShadowBase + slot] = (s << 29) | growBits | index;
        }
    }
}

[numthreads(64, 1, 1)]
void CSClampShadow(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= KindCount) return;
    StructuredBuffer<FarKind> kinds = ResourceDescriptorHeap[KindsIdx];
    RWByteAddressBuffer args = ResourceDescriptorHeap[ShadowArgsUAVIdx];
    uint at = id.x * FAR_COMMAND_UINTS * 4u + FAR_COMMAND_INSTANCES;
    args.Store(at, min(args.Load(at), kinds[id.x].ShadowCapacity));
}
