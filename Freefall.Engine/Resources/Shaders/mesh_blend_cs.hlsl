// Mesh blending: where a mesh cuts into the ground or into another mesh (a rock in the terrain, the cliff kit's rocks
// pushed into each other), near the seam it takes on the look of what it meets, so the cut reads as a soft join and not
// a hard line. Graphics/MeshBlendPass.cs runs it right after the opaque pass, before the decals and the weather:
//   CSSeams   a group per 16 × 16 tile, a thread per pixel: is the pixel's right or lower neighbour another surface that
//             meets it (another entity id, the two places close in 3D, so a cut and not an outline in front of something
//             farther), with a blend radius on either side? The tile's reach is the furthest on screen a pixel blending
//             toward one of its seams can lie, in pixels (0 = no seam). Group (0, 0) also starts the dispatch arguments.
//   CSActive  a thread per tile: listed (bit 31 of its reach) if a seam tile's reach comes to it. Every tile can be
//             listed: nothing is ever left out for want of room
//   CSKeep    a group per listed tile: the pixels of surfaces that blend (and so may change) keep what they were, their
//             albedo and AO in the Composite target (free until composition, which writes all of it) and their normal
//             and roughness in a buffer of a uint a pixel
//   CSBlend   a group per listed tile, a thread per pixel: a surface with a radius looks for the nearest seam along 8
//             directions, refines it and reads the surface across it where its own mirror image falls, as it was (kept if
//             that surface may change, else straight from the G-buffer, which nothing writes there); its albedo, normal,
//             roughness and AO go toward that surface's by its distance from the seam, written in place
// A surface blends by its own radius (its material's BlendRadius, read through the transform slot in EntityId's bits 8-29
// and TransformBuffer's per-slot material): against one without a radius (the terrain) it takes all of that one's look at
// the seam; when both have one they meet half way, so the two sides agree at the seam either way. Vegetation and
// characters (EntityId's classes 2 and 3) never take part. Its own metal, Albedo.a, Normals.w and Data.a stay.
// Docs: .agent/knowledge/rendering_engine/features/mesh_blending.md.

#pragma kernel CSSeams
#pragma kernel CSActive
#pragma kernel CSKeep
#pragma kernel CSBlend

cbuffer PushConstants : register(b3)
{
    uint DepthTexIdx;           // the hardware depth (reverse Z, 0 = sky)
    uint EntityIdTexIdx;        // R32_UInt: class in bits 30-31, transform slot in 8-29, mesh part in 0-7
    uint SlotMaterialsIdx;      // StructuredBuffer<uint>: per transform slot the MaterialID + 1 of its material, 0 = none
    uint MaterialsIdx;          // StructuredBuffer<MaterialData>
    uint AlbedoUAVIdx;          // the G-buffer: RGBA16F
    uint NormalUAVIdx;          // RGBA16 SNorm
    uint DataUAVIdx;            // RGBA8: roughness, metal, AO, flags
    uint TileReachUAVIdx;       // RWByteAddressBuffer: per tile its seams' reach in pixels (bits 0-30), listed (bit 31)
    uint ArgsUAVIdx;            // RWByteAddressBuffer: (listed tiles, 1, 1), the dispatch of CSKeep and CSBlend
    uint ListUAVIdx;            // RWByteAddressBuffer: the listed tiles, x | y << 16
    uint KeptUAVIdx;            // RWTexture2D<float4>: the Composite target: a blending surface's albedo rgb and AO as they were
    uint KeptNormalsUAVIdx;     // RWStructuredBuffer<uint>: per pixel (y × width + x) its normal and roughness as they were (Keep)
    uint TilesX;
    uint TilesY;
    uint MaxPixels;             // a blend reaches at most this far from its seam on screen
    float PixelsPerMetre;       // on screen at 1 m of view depth: 0.5 × height × Projection[1][1]
    uint DebugView;             // DebugVizMode.MeshBlend: the weight painted red into the albedo (full red at a seam)
};

#include "common.fx"

#define BLEND_TILE 16
#define BLEND_DIRECTIONS 8
#define BLEND_STEPS 4           // taps along a direction out to the reach
#define BLEND_REFINE 5          // halvings between the last tap on the surface and the first off it
#define BLEND_CANDIDATES 3      // directions refined, the nearest coarse hits first
#define TILE_LISTED 0x80000000u

static const float2 Directions[BLEND_DIRECTIONS] =
{
    float2(1.0, 0.0), float2(0.70710678, 0.70710678), float2(0.0, 1.0), float2(-0.70710678, 0.70710678),
    float2(-1.0, 0.0), float2(-0.70710678, -0.70710678), float2(0.0, -1.0), float2(0.70710678, -0.70710678)
};

// Whether a surface takes part: something drawn (0 = cleared: the sky), a mesh (class 0) or the terrain (class 1)
bool Takes(uint id)
{
    return id != 0u && (id >> 30) <= 1u;
}

// A surface's blend radius in metres: its transform slot's material's (0 = none)
float RadiusOf(uint id)
{
    StructuredBuffer<uint> slots = ResourceDescriptorHeap[SlotMaterialsIdx];
    uint material = slots[(id & 0x3FFFFFFFu) >> 8];   // past the buffer's end: 0
    if (material == 0u)
        return 0.0;
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    return materials[material - 1u].BlendRadius;
}

// Camera-relative place of the surface at a pixel (the hardware depth; 0 = sky)
float3 PlaceAt(Texture2D<float> depthTex, int2 px, out float depth)
{
    depth = depthTex.Load(int3(px, 0));
    return posFromDepth((float2(px) + 0.5) / ViewportSize, depth, CameraInverse).xyz;
}

// The view depth of a camera-relative place
float ViewDepthOf(float3 rel)
{
    return max(dot(rel, float3(View._13, View._23, View._33)), 1e-3);
}

// Whether two places on either side of a change of id are one cut (they meet) and not an outline in front of something
// farther: within half the larger radius, or within a few pixels' footprint where that is wider (ground at a grazing angle)
bool Meets(float3 a, float3 b, float radius)
{
    float3 d = a - b;
    float limit = max(0.5 * radius, 3.0 * ViewDepthOf(a) / PixelsPerMetre);
    return dot(d, d) < limit * limit;
}

// The pixel of a listed tile a thread works on
int2 ListedPixel(uint listed, uint2 gtid)
{
    RWByteAddressBuffer list = ResourceDescriptorHeap[ListUAVIdx];
    uint tile = list.Load(listed * 4u);
    return int2((tile & 0xFFFFu) * BLEND_TILE, (tile >> 16) * BLEND_TILE) + int2(gtid);
}

// Whether a pixel's tile is listed: only there can a pixel change
bool ListedAt(int2 px)
{
    RWByteAddressBuffer reach = ResourceDescriptorHeap[TileReachUAVIdx];
    uint2 tile = uint2(px) / BLEND_TILE;
    return (reach.Load((tile.y * TilesX + tile.x) * 4u) & TILE_LISTED) != 0u;
}

float2 OctEncode(float3 n)
{
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1e-6);
    float2 e = n.xy;
    if (n.z < 0.0)
        e = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return e;
}

float3 OctDecode(float2 e)
{
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-n.z);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

// A kept normal and roughness: octahedral 12 + 12 bits, roughness 8
uint PackKept(float3 normal, float roughness)
{
    uint2 q = (uint2)round((clamp(OctEncode(normal), -1.0, 1.0) * 0.5 + 0.5) * 4095.0);
    return q.x | (q.y << 12) | ((uint)round(saturate(roughness) * 255.0) << 24);
}

// ── Seams ──

groupshared uint g_Reach;

// The reach on screen (pixels) of a seam between pixel a and its neighbour b, 0 = none
uint SeamReach(Texture2D<uint> ids, Texture2D<float> depthTex, int2 a, uint idA, int2 b)
{
    if (any(b >= int2(ViewportSize)))
        return 0u;
    uint idB = ids.Load(int3(b, 0));
    if (idB == idA || !Takes(idA) || !Takes(idB))
        return 0u;
    float radius = max(RadiusOf(idA), RadiusOf(idB));
    if (radius <= 0.0)
        return 0u;
    float depthA, depthB;
    float3 pA = PlaceAt(depthTex, a, depthA), pB = PlaceAt(depthTex, b, depthB);
    if (depthA <= 0.0 || depthB <= 0.0 || !Meets(pA, pB, radius))
        return 0u;
    // A pixel that blends toward this seam lies within the radius of it in 3D, so it is no nearer the eye than the seam's
    // nearer side less the radius, and its own search (CSBlend) reaches no further on screen than the radius seen from
    // there. Taken from the seam's own depth, a pixel nearer the eye than the seam lay outside the listed tiles and its
    // band stopped at a tile's edge, shifting as the camera moved (user, 2026-11-24)
    float nearest = max(min(ViewDepthOf(pA), ViewDepthOf(pB)) - radius, 0.05);
    float onScreen = radius * PixelsPerMetre / nearest;
    return onScreen < 1.0 ? 0u : (uint)min(ceil(onScreen), (float)MaxPixels);
}

[numthreads(BLEND_TILE, BLEND_TILE, 1)]
void CSSeams(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint gi : SV_GroupIndex)
{
    if (gi == 0u)
    {
        g_Reach = 0u;
        if (all(gid.xy == 0u))
        {
            RWByteAddressBuffer args = ResourceDescriptorHeap[ArgsUAVIdx];
            args.Store3(0, uint3(0u, 1u, 1u));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    int2 px = int2(gid.xy * BLEND_TILE + gtid.xy);
    if (all(px < int2(ViewportSize)))
    {
        Texture2D<uint> ids = ResourceDescriptorHeap[EntityIdTexIdx];
        Texture2D<float> depthTex = ResourceDescriptorHeap[DepthTexIdx];
        uint id = ids.Load(int3(px, 0));
        uint reach = max(SeamReach(ids, depthTex, px, id, px + int2(1, 0)),
                         SeamReach(ids, depthTex, px, id, px + int2(0, 1)));
        if (reach > 0u)
            InterlockedMax(g_Reach, reach);
    }
    GroupMemoryBarrierWithGroupSync();

    if (gi == 0u)
    {
        RWByteAddressBuffer reachOut = ResourceDescriptorHeap[TileReachUAVIdx];
        reachOut.Store((gid.y * TilesX + gid.x) * 4u, g_Reach);
    }
}

// ── The tiles a seam reaches ──

[numthreads(8, 8, 1)]
void CSActive(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= TilesX || id.y >= TilesY)
        return;
    RWByteAddressBuffer reach = ResourceDescriptorHeap[TileReachUAVIdx];
    int span = (int)((MaxPixels + BLEND_TILE - 1u) / BLEND_TILE);
    bool listed = false;
    [loop] for (int dy = -span; dy <= span && !listed; dy++)
    {
        int ty = (int)id.y + dy;
        if (ty < 0 || ty >= (int)TilesY)
            continue;
        [loop] for (int dx = -span; dx <= span; dx++)
        {
            int tx = (int)id.x + dx;
            if (tx < 0 || tx >= (int)TilesX)
                continue;
            uint r = reach.Load(((uint)ty * TilesX + (uint)tx) * 4u) & ~TILE_LISTED;   // others may be marking theirs
            if (r == 0u)
                continue;
            // The pixels between the two tiles, at least
            float2 gap = max(float2(abs(dx), abs(dy)) - 1.0, 0.0) * BLEND_TILE;
            if (dot(gap, gap) <= (float)(r * r))
            {
                listed = true;
                break;
            }
        }
    }
    if (!listed)
        return;
    reach.InterlockedOr((id.y * TilesX + id.x) * 4u, TILE_LISTED);
    RWByteAddressBuffer args = ResourceDescriptorHeap[ArgsUAVIdx];
    uint at;
    args.InterlockedAdd(0, 1u, at);
    RWByteAddressBuffer list = ResourceDescriptorHeap[ListUAVIdx];
    list.Store(at * 4u, id.x | (id.y << 16));
}

// ── What may change keeps what it was ──

[numthreads(BLEND_TILE, BLEND_TILE, 1)]
void CSKeep(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    int2 px = ListedPixel(gid.x, gtid.xy);
    if (any(px >= int2(ViewportSize)))
        return;
    Texture2D<uint> ids = ResourceDescriptorHeap[EntityIdTexIdx];
    uint id = ids.Load(int3(px, 0));
    if (!Takes(id) || RadiusOf(id) <= 0.0)
        return;
    RWTexture2D<float4> albedoUav = ResourceDescriptorHeap[AlbedoUAVIdx];
    RWTexture2D<float4> normalUav = ResourceDescriptorHeap[NormalUAVIdx];
    RWTexture2D<float4> dataUav = ResourceDescriptorHeap[DataUAVIdx];
    RWTexture2D<float4> kept = ResourceDescriptorHeap[KeptUAVIdx];
    RWStructuredBuffer<uint> keptNormals = ResourceDescriptorHeap[KeptNormalsUAVIdx];
    float4 data = dataUav[px];
    kept[px] = float4(albedoUav[px].rgb, data.b);
    keptNormals[(uint)px.y * (uint)ViewportSize.x + (uint)px.x] = PackKept(normalUav[px].xyz, data.r);
}

// ── Blend ──

// A lattice point's random value (0..1) by an integer hash (PCG's output step): no trig, which loses its precision at
// the world's coordinates (decals_cs.hlsl HashCell)
float Hash3(int3 c)
{
    uint h = ((uint)c.x * 0x8DA6B343u) ^ ((uint)c.y * 0xD8163841u) ^ ((uint)c.z * 0xCB1AB31Fu);
    h = h * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    return h * (1.0 / 4294967296.0);
}

// Smooth value noise in world space (0..1): the band's edge breaks up with the surface, never with the screen
float ValueNoise(float3 p)
{
    float3 i = floor(p), f = p - i;
    float3 u = f * f * (3.0 - 2.0 * f);
    int3 c = int3(i);
    float a = lerp(Hash3(c), Hash3(c + int3(1, 0, 0)), u.x);
    float b = lerp(Hash3(c + int3(0, 1, 0)), Hash3(c + int3(1, 1, 0)), u.x);
    float d = lerp(Hash3(c + int3(0, 0, 1)), Hash3(c + int3(1, 0, 1)), u.x);
    float e = lerp(Hash3(c + int3(0, 1, 1)), Hash3(c + int3(1, 1, 1)), u.x);
    return lerp(lerp(a, b, u.y), lerp(d, e, u.y), u.z);
}

[numthreads(BLEND_TILE, BLEND_TILE, 1)]
void CSBlend(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    int2 px = ListedPixel(gid.x, gtid.xy);
    int2 size = int2(ViewportSize);
    if (any(px >= size))
        return;
    Texture2D<uint> ids = ResourceDescriptorHeap[EntityIdTexIdx];
    uint id = ids.Load(int3(px, 0));
    if (!Takes(id))
        return;
    float radius = RadiusOf(id);
    if (radius <= 0.0)
        return;
    Texture2D<float> depthTex = ResourceDescriptorHeap[DepthTexIdx];
    float depth;
    float3 p = PlaceAt(depthTex, px, depth);
    if (depth <= 0.0)
        return;
    float viewDepth = ViewDepthOf(p);
    float reach = min(radius * PixelsPerMetre / viewDepth, (float)MaxPixels);
    if (reach < 1.0)
        return;

    // Along each direction, the first tap off this surface (BLEND_STEPS + 1: none within the reach)
    uint first[BLEND_DIRECTIONS];
    [unroll] for (uint d = 0; d < BLEND_DIRECTIONS; d++)
    {
        first[d] = BLEND_STEPS + 1u;
        [loop] for (uint k = 1u; k <= BLEND_STEPS; k++)
        {
            int2 q = px + int2(round(Directions[d] * (reach * k / BLEND_STEPS)));
            if (any(q < 0) || any(q >= size))
                break;
            if (ids.Load(int3(q, 0)) != id)
            {
                first[d] = k;
                break;
            }
        }
    }

    // The nearest seam: the directions with the nearest coarse hits refined to the edge, kept if the surface across it
    // takes part and meets this one there
    float nearest = 1e30, share = 0.0;
    int2 across = px;
    bool acrossKept = false;
    [loop] for (uint c = 0; c < BLEND_CANDIDATES; c++)
    {
        uint pick = BLEND_DIRECTIONS, k = BLEND_STEPS + 1u;
        [unroll] for (uint d2 = 0; d2 < BLEND_DIRECTIONS; d2++)
        {
            if (first[d2] < k)
            {
                k = first[d2];
                pick = d2;
            }
        }
        if (pick == BLEND_DIRECTIONS)
            break;
        first[pick] = BLEND_STEPS + 1u;

        float2 dir = Directions[pick];
        float lo = reach * (k - 1u) / BLEND_STEPS, hi = reach * k / BLEND_STEPS;   // on the surface at lo, off it at hi
        [loop] for (uint r = 0; r < BLEND_REFINE; r++)
        {
            float mid = 0.5 * (lo + hi);
            if (ids.Load(int3(px + int2(round(dir * mid)), 0)) == id)
                lo = mid;
            else
                hi = mid;
        }
        int2 s0 = px + int2(round(dir * lo)), s1 = px + int2(round(dir * hi));
        uint other = ids.Load(int3(s1, 0));
        if (!Takes(other))
            continue;
        float otherRadius = RadiusOf(other);
        float depth0, depth1;
        float3 p0 = PlaceAt(depthTex, s0, depth0), p1 = PlaceAt(depthTex, s1, depth1);
        if (depth1 <= 0.0 || !Meets(p0, p1, max(radius, otherRadius)))
            continue;
        float distance = length(p - 0.5 * (p0 + p1));
        if (distance >= nearest)
            continue;
        nearest = distance;
        share = otherRadius > 0.0 ? 0.5 : 1.0;
        // Where this pixel's mirror image across the seam falls, while that is still the other surface; else just across
        int2 m = px + int2(round(dir * (lo + hi)));
        across = (any(m < 0) || any(m >= size) || ids.Load(int3(m, 0)) != other) ? s1 : m;
        // A surface that blends too may be changing there: read what it kept (CSKeep), which it did wherever it can
        // change (in a listed tile); anywhere else it stays as it is, and the G-buffer is read
        acrossKept = otherRadius > 0.0 && ListedAt(across);
    }
    if (nearest >= 1e29)
        return;

    // The weight: the share at the seam, none at the radius, the radius broken up along the band by noise on the surface.
    // Up close, where the reach on screen is capped, the band narrows with it instead of ending in a hard line
    float radiusHere = min(radius, MaxPixels * viewDepth / PixelsPerMetre);
    float3 world = p + CamPos;
    float n = 0.65 * ValueNoise(world / (0.45 * radius)) + 0.35 * ValueNoise(world / (0.15 * radius) + 17.3);
    float t = saturate(nearest / (radiusHere * lerp(0.45, 1.0, n)));
    float w = share * (1.0 - t * t * (3.0 - 2.0 * t));
    if (w < 1.0 / 256.0)
        return;

    RWTexture2D<float4> albedoUav = ResourceDescriptorHeap[AlbedoUAVIdx];
    RWTexture2D<float4> normalUav = ResourceDescriptorHeap[NormalUAVIdx];
    RWTexture2D<float4> dataUav = ResourceDescriptorHeap[DataUAVIdx];
    float4 albedo = albedoUav[px];
    float4 normal = normalUav[px];
    float4 data = dataUav[px];
    float3 otherAlbedo, otherNormal;
    float otherRoughness, otherAO;
    if (acrossKept)
    {
        RWTexture2D<float4> kept = ResourceDescriptorHeap[KeptUAVIdx];
        RWStructuredBuffer<uint> keptNormals = ResourceDescriptorHeap[KeptNormalsUAVIdx];
        float4 k = kept[across];
        uint v = keptNormals[(uint)across.y * (uint)size.x + (uint)across.x];
        otherAlbedo = k.rgb;
        otherAO = k.a;
        otherNormal = OctDecode(float2(v & 0xFFFu, (v >> 12) & 0xFFFu) / 4095.0 * 2.0 - 1.0);
        otherRoughness = (v >> 24) / 255.0;
    }
    else
    {
        float4 otherData = dataUav[across];
        otherAlbedo = albedoUav[across].rgb;
        otherNormal = normalUav[across].xyz;
        otherRoughness = otherData.r;
        otherAO = otherData.b;
    }

    albedo.rgb = DebugView != 0u ? lerp(albedo.rgb, float3(1.0, 0.02, 0.02), w / share)
                                 : lerp(albedo.rgb, otherAlbedo, w);
    float3 blended = lerp(normal.xyz, otherNormal, w);
    normal.xyz = dot(blended, blended) > 1e-6 ? normalize(blended) : normal.xyz;
    data.r = lerp(data.r, otherRoughness, w);
    data.b = lerp(data.b, otherAO, w);
    albedoUav[px] = albedo;
    normalUav[px] = normal;
    dataUav[px] = data;
}
