// Decal cull (Graphics/DecalPass.cs, DECALS_PLAN.md B4), per view, right after the opaque pass:
//   CSClearDecals   the counters and the histogram for this view
//   CSCountDecals   a thread per registry slot: is the box in view and big enough on screen; a histogram of those by
//                   on-screen area
//   CSCullDecals    the same per slot; appends those in view to the visible list, the biggest first when there are more
//                   than it holds (the importance bands of point_light_cull.hlsl: which ones miss out is decided by their
//                   size, never by the order the GPU appends them in, which changes every frame and blinks)
//   CSSortDecals    one group: the visible list in the order decals_cs.hlsl applies them (priority, then registry
//                   slot: a stable key, so two overlapping decals never swap)
// Decals under 2 pixels across are left out, fading in over the next 4. Spawned decals (the ring at the registry's
// start, DECAL_SPAWNED) fade out by their age and are left out once it passes their lifetime.

#pragma kernel CSClearDecals
#pragma kernel CSCountDecals
#pragma kernel CSCullDecals
#pragma kernel CSSortDecals

cbuffer PushConstants : register(b3)
{
    uint RecordsIdx;            // StructuredBuffer<DecalRecord>: the registry
    uint RecordCount;           // registry slots to walk
    uint TransformsIdx;         // StructuredBuffer<row_major float4x4>: the global transform buffer
    uint FrameIdx;              // StructuredBuffer<DecalFrame>: [0]
    uint VisibleUAVIdx;         // RWStructuredBuffer<uint2>: (slot, asuint(fade)) of the decals in view, unordered
    uint CountUAVIdx;           // RWByteAddressBuffer: [0] = decals in the list, [1] = decals in view
    uint HistogramUAVIdx;       // RWStructuredBuffer<uint>[IMPORTANCE_BINS]: decals in view per area band
    uint SortedUAVIdx;          // RWStructuredBuffer<uint2>: the list in the order decals_cs.hlsl applies it
};

#include "decals.hlsli"

#define IMPORTANCE_BINS 64u

// A decal in view: its box's bounding sphere against the frustum, and its importance, the on-screen area (pixels²) of
// that sphere; fade toward the smallest
bool CullDecal(uint slot, DecalFrame frame, out DecalRecord r, out float importance, out float fade)
{
    StructuredBuffer<DecalRecord> records = ResourceDescriptorHeap[RecordsIdx];
    r = records[slot];
    importance = 0;
    fade = 0;
    if ((r.Flags & DECAL_ENABLED) == 0 || r.Opacity <= 0.0)
        return false;
    // A spawned one fades out over the last FadeOut seconds of its Lifetime
    float ageFade = 1.0;
    if ((r.Flags & DECAL_SPAWNED) != 0)
    {
        float left = r.Lifetime - (frame.Time - r.SpawnTime);
        if (left <= 0.0)
            return false;
        ageFade = saturate(left / max(r.FadeOut, 1e-3));
    }

    float4x4 m = DecalMatrix(TransformsIdx, r.TransformSlot);
    float radius = 0.5 * sqrt(dot(m[0].xyz, m[0].xyz) + dot(m[1].xyz, m[1].xyz) + dot(m[2].xyz, m[2].xyz));
    float3 p = m[3].xyz - frame.CameraPosition;
    [unroll]
    for (uint i = 0; i < 5; i++)
    {
        if (dot(frame.Planes[i].xyz, p) + frame.Planes[i].w < -radius)
            return false;
    }

    float radiusPixels = radius * frame.PixelsPerMetre / max(length(p), 0.01);
    if (radiusPixels < 1.0)
        return false;
    fade = saturate((radiusPixels - 1.0) / 2.0) * ageFade;
    importance = radiusPixels * radiusPixels;
    return true;
}

// Half-octave bands of the area: 1 px² (the smallest drawn) to ~2^30 (a decal over the camera) in the 64 bands
uint ImportanceBin(float importance)
{
    return (uint)clamp((int)floor(log2(importance) * 2.0) + 2, 0, (int)IMPORTANCE_BINS - 1);
}

[numthreads(64, 1, 1)]
void CSClearDecals(uint gi : SV_GroupIndex)
{
    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramUAVIdx];
    histogram[gi] = 0;
    if (gi == 0)
    {
        RWByteAddressBuffer countBuf = ResourceDescriptorHeap[CountUAVIdx];
        countBuf.Store2(0, uint2(0, 0));
    }
}

[numthreads(64, 1, 1)]
void CSCountDecals(uint3 dtid : SV_DispatchThreadID)
{
    uint slot = dtid.x;
    if (slot >= RecordCount)
        return;
    StructuredBuffer<DecalFrame> frames = ResourceDescriptorHeap[FrameIdx];
    DecalRecord r;
    float importance, fade;
    if (!CullDecal(slot, frames[0], r, importance, fade))
        return;
    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramUAVIdx];
    InterlockedAdd(histogram[ImportanceBin(importance)], 1);
    RWByteAddressBuffer countBuf = ResourceDescriptorHeap[CountUAVIdx];
    countBuf.InterlockedAdd(4, 1);
}

groupshared uint g_LowestBand;   // decals in lower bands are left out
groupshared uint g_FadedBand;    // decals in this band fade with their area (none: IMPORTANCE_BINS)

[numthreads(64, 1, 1)]
void CSCullDecals(uint3 dtid : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    if (gi == 0)
    {
        // Biggest bands first, as many as the list holds
        RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramUAVIdx];
        uint total = 0;
        for (uint b = 0; b < IMPORTANCE_BINS; b++)
            total += histogram[b];

        uint lowest = 0, faded = IMPORTANCE_BINS;
        if (total > MAX_VISIBLE_DECALS)
        {
            uint sum = 0;
            lowest = IMPORTANCE_BINS - 1;   // a top band over the cap on its own: the cap drops the rest
            for (int band = (int)IMPORTANCE_BINS - 1; band >= 0; band--)
            {
                if (sum + histogram[band] > MAX_VISIBLE_DECALS)
                {
                    lowest = min((uint)band + 1, IMPORTANCE_BINS - 1);
                    break;
                }
                sum += histogram[band];
            }
            faded = lowest;
        }
        g_LowestBand = lowest;
        g_FadedBand = faded;
    }
    GroupMemoryBarrierWithGroupSync();

    uint slot = dtid.x;
    if (slot >= RecordCount)
        return;
    StructuredBuffer<DecalFrame> frames = ResourceDescriptorHeap[FrameIdx];
    DecalRecord r;
    float importance, fade;
    if (!CullDecal(slot, frames[0], r, importance, fade))
        return;
    uint band = ImportanceBin(importance);
    if (band < g_LowestBand)
        return;
    if (band == g_FadedBand)
        fade *= saturate(frac(log2(importance) * 2.0));     // fades out toward the band's lower edge

    RWByteAddressBuffer countBuf = ResourceDescriptorHeap[CountUAVIdx];
    uint index;
    countBuf.InterlockedAdd(0, 1, index);
    if (index >= MAX_VISIBLE_DECALS)
        return;
    RWStructuredBuffer<uint2> visible = ResourceDescriptorHeap[VisibleUAVIdx];
    visible[index] = uint2(slot, asuint(fade));
}

groupshared uint g_Keys[MAX_VISIBLE_DECALS];

[numthreads(1024, 1, 1)]
void CSSortDecals(uint gi : SV_GroupIndex)
{
    RWByteAddressBuffer countBuf = ResourceDescriptorHeap[CountUAVIdx];
    uint count = min(countBuf.Load(0), MAX_VISIBLE_DECALS);
    if (count == 0)
        return;

    RWStructuredBuffer<uint2> visible = ResourceDescriptorHeap[VisibleUAVIdx];
    StructuredBuffer<DecalRecord> records = ResourceDescriptorHeap[RecordsIdx];

    // Keys: the priority in the top 4 bits, the registry slot in the next 16 (slots are handed out in the order decals
    // came, and stay theirs), the visible index in the low 12. Padded to a power of two with keys that sort last.
    uint n = count <= 1u ? 1u : 1u << (firstbithigh(count - 1u) + 1u);
    for (uint i = gi; i < n; i += 1024u)
    {
        uint key = 0xFFFFFFFFu;
        if (i < count)
        {
            uint slot = visible[i].x;
            uint priority = (records[slot].Flags >> DECAL_PRIORITY_SHIFT) & DECAL_PRIORITY_MASK;
            key = (priority << 28) | ((slot & 0xFFFFu) << 12) | i;
        }
        g_Keys[i] = key;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint k = 2u; k <= n; k <<= 1)
    {
        for (uint j = k >> 1; j > 0u; j >>= 1)
        {
            for (uint a = gi; a < n; a += 1024u)
            {
                uint b = a ^ j;
                if (b > a)
                {
                    uint ka = g_Keys[a], kb = g_Keys[b];
                    bool ascending = (a & k) == 0u;
                    if ((ka > kb) == ascending)
                    {
                        g_Keys[a] = kb;
                        g_Keys[b] = ka;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    RWStructuredBuffer<uint2> sorted = ResourceDescriptorHeap[SortedUAVIdx];
    for (uint s = gi; s < count; s += 1024u)
        sorted[s] = visible[g_Keys[s] & 0xFFFu];
}
