// Sun shadow receivers (ShadowReceivers.cs): where this frame's visible surfaces sample each cascade, as the deepest
// receiver light depth per tile, for the receiver-mask caster culling in CSVisibilityCascades (Bittner et al. 2011,
// "Shadow caster culling for efficient shadow mapping"). Layout in shadow_receivers.hlsli.
//
// CSOceanWaveBound  per texel of the ocean's displacement (one slice per band): the band's largest |vertical
//                   displacement|, for the ocean's wave band in CSReceiverTiles.
// CSReceiverTiles   per screen pixel: rebuilds its position from the linear depth, picks its cascade as
//                   light_directional_cs.hlsl does (and the next one inside the 10 % blend band), projects into that
//                   cascade and max-merges its light depth into the cascade's level-0 tile. Also marks where its view
//                   ray crosses the ocean's wave band in front of the opaque surface. Lanes that hit the same tile
//                   reduce first, so a wave issues one atomic per tile it touches (usually one or two).
// CSReceiverSpheres per instance of a forward batch (glass, water bodies) the camera saw last frame: its bounding
//                   sphere's footprint in every cascade its view depths can sample, at its far depth.
// CSReceiverLights  per visible point light: the five sun samples of local_shadow_select.hlsl SunVisibility.
// CSReceiverPyramid one group per cascade: max-reduces level 0 to 1×1 and counts the tiles with a receiver; group 0
//                   writes the header's mode and filter margin.

#pragma kernel CSOceanWaveBound
#pragma kernel CSReceiverTiles
#pragma kernel CSReceiverSpheres
#pragma kernel CSReceiverLights
#pragma kernel CSReceiverPyramid

cbuffer PushConstants : register(b3)
{
    uint DepthTexIdx;           // linear view depth (DepthGBuffer, 0 = sky)
    uint CascadesIdx;           // StructuredBuffer<CascadeData>: this frame's cascades (absolute-world VP, splits)
    uint ReceiversUAVIdx;       // RWStructuredBuffer<uint>: header + tile pyramids
    uint CascadeCountIdx;
    uint ScreenWidthIdx;
    uint ScreenHeightIdx;
    uint ModeIdx;               // RECEIVER_MODE_*: copied into the header for the culler
    float MarginUV;             // the shadow filter's reach in shadow-map UV: copied into the header
    uint OceanDisplacementIdx;  // Texture2DArray<float4>: the ocean's displacement per band (0 = no ocean drawn)
    uint OceanBandCountIdx;
    float OceanPlaneY;          // the ocean's rest height (world Y)
    float OceanMargin;       // metres added to the bands' summed wave heights (shore surge + slack)
    uint SphereBoundsIdx;       // RWStructuredBuffer<InstanceBounds>: a forward batch's bounds from last frame's culling
    uint SphereFlagsIdx;        // StructuredBuffer<uint>: its visibility flags then (0 = not seen)
    uint SphereCountIdx;        // its instances then
    uint VisibleLightsIdx;      // RWStructuredBuffer<PointLightData>: this frame's visible point lights (0 = none)
    uint VisibleLightCountIdx;  // RWByteAddressBuffer: [0] = visible lights
};

cbuffer Params : register(b4)
{
    float4 CameraPosition;      // xyz: world position of the camera
    float4 RayRight;            // xyz: view right / Projection._11
    float4 RayUp;               // xyz: view up / Projection._22
    float4 RayForward;          // xyz: view forward
};

#include "shadow_receivers.hlsli"
#include "point_lights.hlsli"

// The sun's cascades (common.fx's CascadeData; this file does not include common.fx)
struct SunCascade
{
    float4 Planes[6];
    row_major float4x4 VP;      // absolute world -> clip (orthographic)
    row_major float4x4 PrevVP;
    float4 SplitDistances;      // x = near, y = far (view depth)
};

// cull_instances.hlsl InstanceBounds (20 bytes)
struct InstanceBounds
{
    float3 Center;
    float Radius;
    uint Part;
};

#define NO_TILE      0xFFFFFFFFu
#define INVALID_PART 0xFFFFFFFFu

// Receiver depths are stored as asuint of a positive float; 0 stays "no receiver", so a receiver nearer the sun than
// the near plane still counts, as the smallest depth
uint DepthBits(float depth)
{
    return asuint(clamp(depth, 1e-6, 1.0));
}

// Word of the level-0 tile a cascade UV falls into
uint TileWord(uint cascadeIdx, float2 uv)
{
    uint2 tile = min(uint2(uv * RECEIVER_TILES), RECEIVER_TILES - 1u);
    return ReceiverLevelBase(cascadeIdx, 0) + tile.y * RECEIVER_TILES + tile.x;
}

// Cascade NDC of a world position (orthographic: w = 1, divided anyway)
float3 CascadeNdc(float3 worldPos, uint cascadeIdx)
{
    StructuredBuffer<SunCascade> cascades = ResourceDescriptorHeap[CascadesIdx];
    float4 clip = mul(float4(worldPos, 1.0), cascades[cascadeIdx].VP);
    return clip.xyz / clip.w;
}

// The cascade the lighting picks at a view depth: the first whose far split lies beyond it (NO_TILE past the last),
// and inside the last 10 % of that one's range the next one, which the lighting blends in (else NO_TILE)
void LightingCascades(float viewZ, out uint first, out uint second)
{
    StructuredBuffer<SunCascade> cascades = ResourceDescriptorHeap[CascadesIdx];
    first = NO_TILE;
    second = NO_TILE;
    for (uint i = 0; i < CascadeCountIdx; i++)
    {
        if (viewZ < cascades[i].SplitDistances.y)
        {
            first = i;
            break;
        }
    }
    if (first != NO_TILE && first + 1 < CascadeCountIdx)
    {
        float4 split = cascades[first].SplitDistances;
        if (split.y - viewZ < (split.y - split.x) * 0.1)
            second = first + 1;
    }
}

// Tile and depth of a world position in a cascade (NO_TILE: off its square, where the lighting reads nothing)
uint ReceiverTile(float3 worldPos, uint cascadeIdx, out uint depthBits)
{
    float3 ndc = CascadeNdc(worldPos, cascadeIdx);
    float2 uv = ndc.xy * float2(0.5, -0.5) + 0.5;
    depthBits = DepthBits(ndc.z);
    if (cascadeIdx == NO_TILE || any(uv < 0.0) || any(uv > 1.0))
        return NO_TILE;
    return TileWord(cascadeIdx, uv);
}

// Max-merges each lane's depth into its tile: the lanes sharing the first remaining lane's tile reduce together and
// one of them writes, until every lane is done.
void MergeIntoTile(RWStructuredBuffer<uint> receivers, uint tile, uint depthBits)
{
    [loop]
    while (tile != NO_TILE)
    {
        uint first = WaveReadLaneFirst(tile);
        if (first == tile)
        {
            uint deepest = WaveActiveMax(depthBits);
            if (WaveIsFirstLane())
                InterlockedMax(receivers[tile], deepest);
            tile = NO_TILE;
        }
    }
}

// ── CSOceanWaveBound ─────────────────────────────────────────────────────────
[numthreads(8, 8, 1)]
void CSOceanWaveBound(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    Texture2DArray<float4> displacement = ResourceDescriptorHeap[OceanDisplacementIdx];
    float w, h, slices;
    displacement.GetDimensions(w, h, slices);
    uint band = dispatchThreadId.z;

    float height = 0.0;
    if (dispatchThreadId.x < (uint)w && dispatchThreadId.y < (uint)h)
        height = abs(displacement.Load(int4(dispatchThreadId.xy, band, 0)).y);

    float highest = WaveActiveMax(height);
    if (WaveIsFirstLane())
    {
        RWStructuredBuffer<uint> receivers = ResourceDescriptorHeap[ReceiversUAVIdx];
        InterlockedMax(receivers[RECEIVER_WAVE_BOUND + band], asuint(highest));
    }
}

// ── CSReceiverTiles ──────────────────────────────────────────────────────────
[numthreads(8, 8, 1)]
void CSReceiverTiles(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 px = dispatchThreadId.xy;
    uint tiles[6] = { NO_TILE, NO_TILE, NO_TILE, NO_TILE, NO_TILE, NO_TILE };
    uint depths[6] = { 0, 0, 0, 0, 0, 0 };
    RWStructuredBuffer<uint> receivers = ResourceDescriptorHeap[ReceiversUAVIdx];

    if (px.x < ScreenWidthIdx && px.y < ScreenHeightIdx)
    {
        Texture2D<float> depthTex = ResourceDescriptorHeap[DepthTexIdx];
        float viewZ = depthTex.Load(int3(px, 0));

        float2 uv = (float2(px) + 0.5) / float2(ScreenWidthIdx, ScreenHeightIdx);
        float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
        float3 ray = ndc.x * RayRight.xyz + ndc.y * RayUp.xyz + RayForward.xyz;     // one metre of view depth

        // The opaque surface (0 = sky)
        if (viewZ > 0.0)
        {
            uint first, second;
            LightingCascades(viewZ, first, second);
            float3 worldPos = CameraPosition.xyz + viewZ * ray;
            if (first != NO_TILE)
                tiles[0] = ReceiverTile(worldPos, first, depths[0]);
            if (second != NO_TILE)
                tiles[1] = ReceiverTile(worldPos, second, depths[1]);
        }

        // The ocean, drawn forward after the shadows. Whatever the waves do, the surface this pixel sees lies on its view
        // ray inside the band OceanPlaneY ± (the bands' largest heights summed): mark where the ray enters and leaves the
        // band in front of the opaque surface, both at the deeper end's depth. The ends of the neighbouring pixels'
        // segments fall in between, so together they cover the band.
        if (OceanDisplacementIdx != 0)
        {
            float waveBound = OceanMargin;
            for (uint b = 0; b < min(OceanBandCountIdx, RECEIVER_MAX_BANDS); b++)
                waveBound += asfloat(receivers[RECEIVER_WAVE_BOUND + b]);

            StructuredBuffer<SunCascade> cascades = ResourceDescriptorHeap[CascadesIdx];
            float lastSplit = cascades[CascadeCountIdx - 1].SplitDistances.y;

            float tEnter = 0.0, tExit = lastSplit;
            float yLow = OceanPlaneY - waveBound, yHigh = OceanPlaneY + waveBound;
            if (abs(ray.y) > 1e-6)
            {
                float t0 = (yLow - CameraPosition.y) / ray.y;
                float t1 = (yHigh - CameraPosition.y) / ray.y;
                tEnter = max(min(t0, t1), 0.0);
                tExit = min(max(t0, t1), lastSplit);
            }
            else if (CameraPosition.y < yLow || CameraPosition.y > yHigh)
                tExit = -1.0;   // level ray outside the band
            if (viewZ > 0.0)
                tExit = min(tExit, viewZ);

            if (tEnter < tExit)
            {
                float3 enter = CameraPosition.xyz + tEnter * ray;
                float3 exit = CameraPosition.xyz + tExit * ray;
                float ends[2] = { tEnter, tExit };
                [unroll]
                for (uint e = 0; e < 2; e++)
                {
                    uint first, second;
                    LightingCascades(ends[e], first, second);
                    uint cascadeIdx[2] = { first, second };
                    [unroll]
                    for (uint k = 0; k < 2; k++)
                    {
                        if (cascadeIdx[k] == NO_TILE)
                            continue;
                        float3 ndcEnter = CascadeNdc(enter, cascadeIdx[k]);
                        float3 ndcExit = CascadeNdc(exit, cascadeIdx[k]);
                        float3 here = e == 0 ? ndcEnter : ndcExit;
                        float2 tileUV = here.xy * float2(0.5, -0.5) + 0.5;
                        if (any(tileUV < 0.0) || any(tileUV > 1.0))
                            continue;
                        tiles[2 + e * 2 + k] = TileWord(cascadeIdx[k], tileUV);
                        depths[2 + e * 2 + k] = DepthBits(max(ndcEnter.z, ndcExit.z));
                    }
                }
            }
        }
    }

    [unroll]
    for (uint i = 0; i < 6; i++)
        MergeIntoTile(receivers, tiles[i], depths[i]);
}

// ── CSReceiverSpheres ────────────────────────────────────────────────────────
// A forward instance's sphere marks every tile under its footprint at its far depth, in each cascade whose view depths
// the sphere reaches (a cascade's pixels lie between the split before it, less that one's 10 % blend band, and its own).
[numthreads(64, 1, 1)]
void CSReceiverSpheres(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint instanceIdx = dispatchThreadId.x;
    if (instanceIdx >= SphereCountIdx)
        return;
    StructuredBuffer<uint> flags = ResourceDescriptorHeap[SphereFlagsIdx];
    if (flags[instanceIdx] == 0)
        return;     // not seen last frame

    RWStructuredBuffer<InstanceBounds> bounds = ResourceDescriptorHeap[SphereBoundsIdx];
    InstanceBounds sphere = bounds[instanceIdx];
    if (sphere.Part == INVALID_PART || sphere.Radius <= 0.0)
        return;

    float viewZ = dot(sphere.Center - CameraPosition.xyz, RayForward.xyz);
    if (viewZ + sphere.Radius <= 0.0)
        return;

    StructuredBuffer<SunCascade> cascades = ResourceDescriptorHeap[CascadesIdx];
    RWStructuredBuffer<uint> receivers = ResourceDescriptorHeap[ReceiversUAVIdx];
    float nearest = -1e30;
    for (uint c = 0; c < CascadeCountIdx; c++)
    {
        float4 split = cascades[c].SplitDistances;
        bool reaches = viewZ + sphere.Radius >= nearest && viewZ - sphere.Radius < split.y;
        nearest = split.y - (split.y - split.x) * 0.1;      // the next cascade starts in this one's blend band
        if (!reaches)
            continue;

        row_major float4x4 vp = cascades[c].VP;
        float3 ndc = CascadeNdc(sphere.Center, c);
        float2 uv = ndc.xy * float2(0.5, -0.5) + 0.5;
        float radiusUV = 0.5 * sphere.Radius * length(float3(vp._11, vp._21, vp._31));
        if (any(uv + radiusUV < 0.0) || any(uv - radiusUV > 1.0))
            continue;
        uint depthBits = DepthBits(ndc.z + sphere.Radius * length(float3(vp._13, vp._23, vp._33)));

        uint2 lo = min(uint2(saturate(uv - radiusUV) * RECEIVER_TILES), RECEIVER_TILES - 1u);
        uint2 hi = min(uint2(saturate(uv + radiusUV) * RECEIVER_TILES), RECEIVER_TILES - 1u);
        uint base = ReceiverLevelBase(c, 0);
        for (uint y = lo.y; y <= hi.y; y++)
            for (uint x = lo.x; x <= hi.x; x++)
                InterlockedMax(receivers[base + y * RECEIVER_TILES + x], depthBits);
    }
}

// ── CSReceiverLights ─────────────────────────────────────────────────────────
// The point light selection's sun factor (local_shadow_select.hlsl SunVisibility): the light's centre and four points
// 0.75 m around it, each in the first cascade whose box holds it. A room light seen from outside keeps its shadow at
// noon only because the roof over it is in the map.
[numthreads(64, 1, 1)]
void CSReceiverLights(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    RWByteAddressBuffer countBuf = ResourceDescriptorHeap[VisibleLightCountIdx];
    if (dispatchThreadId.x >= min(countBuf.Load(0), MAX_VISIBLE_LIGHTS))
        return;
    RWStructuredBuffer<PointLightData> lights = ResourceDescriptorHeap[VisibleLightsIdx];
    float3 centre = lights[dispatchThreadId.x].Position + CameraPosition.xyz;
    RWStructuredBuffer<uint> receivers = ResourceDescriptorHeap[ReceiversUAVIdx];

    static const float3 offsets[5] =
    {
        float3(0, 0, 0), float3(0.75, 0, 0), float3(-0.75, 0, 0), float3(0, 0, 0.75), float3(0, 0, -0.75)
    };
    [unroll]
    for (uint t = 0; t < 5; t++)
    {
        float3 p = centre + offsets[t];
        for (uint c = 0; c < CascadeCountIdx; c++)
        {
            float3 ndc = CascadeNdc(p, c);
            if (any(abs(ndc.xy) > 1.0) || ndc.z < 0.0 || ndc.z > 1.0)
                continue;
            InterlockedMax(receivers[TileWord(c, ndc.xy * float2(0.5, -0.5) + 0.5)], DepthBits(ndc.z));
            break;
        }
    }
}

// ── CSReceiverPyramid ────────────────────────────────────────────────────────
groupshared uint g_Deepest[32 * 32];

[numthreads(32, 32, 1)]
void CSReceiverPyramid(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    uint cascadeIdx = groupId.x;
    uint2 cell = groupThreadId.xy;
    RWStructuredBuffer<uint> receivers = ResourceDescriptorHeap[ReceiversUAVIdx];

    if (cascadeIdx == 0 && cell.x == 0 && cell.y == 0)
    {
        receivers[RECEIVER_MODE] = ModeIdx;
        receivers[RECEIVER_MARGIN] = asuint(MarginUV);
        receivers[RECEIVER_CASCADES] = CascadeCountIdx;
    }

    // Level 1 from level 0's 2x2, counting the tiles with a receiver on the way
    uint base0 = ReceiverLevelBase(cascadeIdx, 0);
    uint2 t = cell * 2u;
    uint d00 = receivers[base0 + t.y * RECEIVER_TILES + t.x];
    uint d10 = receivers[base0 + t.y * RECEIVER_TILES + t.x + 1];
    uint d01 = receivers[base0 + (t.y + 1) * RECEIVER_TILES + t.x];
    uint d11 = receivers[base0 + (t.y + 1) * RECEIVER_TILES + t.x + 1];
    uint filled = (d00 != 0 ? 1u : 0u) + (d10 != 0 ? 1u : 0u) + (d01 != 0 ? 1u : 0u) + (d11 != 0 ? 1u : 0u);
    uint waveFilled = WaveActiveSum(filled);
    if (WaveIsFirstLane())
        InterlockedAdd(receivers[RECEIVER_TILE_COUNT + cascadeIdx], waveFilled);

    uint deepest = max(max(d00, d10), max(d01, d11));
    uint size = RECEIVER_TILES >> 1;
    receivers[ReceiverLevelBase(cascadeIdx, 1) + cell.y * size + cell.x] = deepest;
    g_Deepest[cell.y * 32 + cell.x] = deepest;
    GroupMemoryBarrierWithGroupSync();

    // Levels 2..6 in shared memory (stride 32), each from the one before
    for (uint level = 2; level < RECEIVER_LEVELS; level++)
    {
        size = RECEIVER_TILES >> level;
        bool active = cell.x < size && cell.y < size;
        if (active)
        {
            uint2 c = cell * 2u;
            deepest = max(max(g_Deepest[c.y * 32 + c.x], g_Deepest[c.y * 32 + c.x + 1]),
                          max(g_Deepest[(c.y + 1) * 32 + c.x], g_Deepest[(c.y + 1) * 32 + c.x + 1]));
        }
        GroupMemoryBarrierWithGroupSync();
        if (active)
        {
            g_Deepest[cell.y * 32 + cell.x] = deepest;
            receivers[ReceiverLevelBase(cascadeIdx, level) + cell.y * size + cell.x] = deepest;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
