// Light clusters (Graphics/LightClusters.cs, light_clusters_cs.hlsl): which of this frame's visible point lights reach a
// place in the view, for everything that is not an opaque surface: the air (the near fog volume), particles and ribbons.
// Built once per view after the point light shadow selection, read through SceneConstants.LightClustersIdx (the cluster
// buffer, ByteAddressBuffer) and ClusterLightsIdx (the lights in view-depth order, StructuredBuffer<PointLightData>).
//
// Z-binning (Drobot, "Improved culling for tiled and clustered rendering", SIGGRAPH 2017) instead of a light list per
// cluster: the lights are sorted by the view depth of their centres; each of LIGHT_ZBINS bins of view depth holds the
// first and last sorted light whose sphere reaches into it, and each 32-pixel screen tile a bitmask over the sorted lights
// whose sphere it sees. A place's lights are the tile's bits inside its bin's range. No list has a cap (a cap filled by
// atomic appends keeps whichever light the GPU happened to append first, and lights blink: point_lights.md "Hazards"),
// and the memory does not depend on how many lights crowd a cluster.
//
// Cluster buffer (uints):
//   [0..LC_HEADER)                    header (LC_* below): the lights, the view, this frame's point light and sun shadows
//                                     (built after both are drawn, so particles find them here)
//   [LC_BINS + bin]                   first sorted light | last << 16 (first > last: none)
//   [LC_LIGHT_BINS + light]           the light's first z-bin | last << 16 (written by the sort, read by the bins pass)
//   [LC_MASKS + tile * WORDS + word]  bit b of word w: sorted light w * 32 + b reaches the tile
#ifndef LIGHT_CLUSTERS_HLSLI
#define LIGHT_CLUSTERS_HLSLI

#include "point_lights.hlsli"
#include "local_shadows.hlsli"

#define LIGHT_CLUSTER_TILE_SHIFT    5u                          // 32-pixel tiles (LightClusters.TileShift)
#define LIGHT_CLUSTER_WORDS         (MAX_VISIBLE_LIGHTS / 32u)  // mask words a tile holds
#define LIGHT_ZBINS                 1024u                       // LightClusters.ZBins
#define LIGHT_ZBIN_NEAR             0.25                        // the first bin ends a step past this view depth (m)
#define LIGHT_ZBIN_OCTAVES          14.0                        // ... the last begins 2^14 times as far: 4 km

#define LC_COUNT            0u      // lights in the sorted list
#define LC_WORDS            1u      // mask words a tile uses (lights / 32, rounded up)
#define LC_TILES_X          2u
#define LC_WIDTH            3u      // the view's pixels: uv × this = the pixel
#define LC_HEIGHT           4u
#define LC_SHADOW_ATLAS     5u      // point light shadow atlas rendered this frame (Texture2DArray), 0 = none
#define LC_SHADOW_VIEWS     6u      // StructuredBuffer<LocalShadowView>
#define LC_SHADOW_LIGHTS    7u      // StructuredBuffer<LocalShadowLight>
#define LC_SUN_SHADOW_MAP   8u      // the directional light's cascades drawn this frame (Texture2DArray), 0 = none
#define LC_SUN_CASCADES     9u      // StructuredBuffer<CascadeData> (common.fx): absolute world → light clip, splits
#define LC_SUN_CASCADE_COUNT 10u
#define LC_SUN_DIRECTION    11u     // 3 floats (asuint): toward the body the cascades were drawn for
#define LC_HEADER           16u
#define LC_BINS             LC_HEADER
#define LC_LIGHT_BINS       (LC_BINS + LIGHT_ZBINS)
#define LC_MASKS            (LC_LIGHT_BINS + MAX_VISIBLE_LIGHTS)

// The z-bin of a view depth: bins grow by 2^(14/1024) (~1 %) each, from LIGHT_ZBIN_NEAR on; nearer is bin 0, farther
// the last
uint LightZBin(float viewDepth)
{
    float bin = log2(max(viewDepth, 1e-4) / LIGHT_ZBIN_NEAR) * (LIGHT_ZBINS / LIGHT_ZBIN_OCTAVES);
    return (uint)clamp(bin, 0.0, (float)(LIGHT_ZBINS - 1));
}

// A walk over the lights that reach one place: LightClusterBegin, then LightClusterNext until it returns false
struct LightClusterWalk
{
    uint MaskBase;      // byte offset of the tile's first mask word
    uint Word;          // the word being walked
    uint LastWord;
    uint Bits;          // its bits not walked yet
    uint Last;          // the last sorted light in the bin's range
};

uint LightClusterLastMask(uint last)
{
    uint b = last & 31u;
    return b == 31u ? 0xFFFFFFFFu : (1u << (b + 1u)) - 1u;
}

// uv: the place on screen (0..1, clamped); viewDepth: its view depth (m). False when no light reaches it.
bool LightClusterBegin(ByteAddressBuffer clusters, float2 uv, float viewDepth, out LightClusterWalk walk)
{
    walk = (LightClusterWalk)0;
    if (clusters.Load(LC_COUNT * 4) == 0)
        return false;
    uint range = clusters.Load((LC_BINS + LightZBin(viewDepth)) * 4);
    uint first = range & 0xFFFFu, last = range >> 16;
    if (first > last)
        return false;

    uint2 size = uint2(clusters.Load(LC_WIDTH * 4), clusters.Load(LC_HEIGHT * 4));
    uint2 px = min(uint2(saturate(uv) * float2(size)), size - 1u);
    uint2 tile = px >> LIGHT_CLUSTER_TILE_SHIFT;
    walk.MaskBase = (LC_MASKS + (tile.y * clusters.Load(LC_TILES_X * 4) + tile.x) * LIGHT_CLUSTER_WORDS) * 4;
    walk.Word = first >> 5;
    walk.LastWord = last >> 5;
    walk.Last = last;
    walk.Bits = clusters.Load(walk.MaskBase + walk.Word * 4) & (0xFFFFFFFFu << (first & 31u));
    if (walk.Word == walk.LastWord)
        walk.Bits &= LightClusterLastMask(last);
    return true;
}

// The next light (its index in the sorted list); false when the walk is done
bool LightClusterNext(ByteAddressBuffer clusters, inout LightClusterWalk walk, out uint light)
{
    light = 0;
    [loop] while (walk.Bits == 0)
    {
        if (walk.Word >= walk.LastWord)
            return false;
        walk.Word++;
        walk.Bits = clusters.Load(walk.MaskBase + walk.Word * 4);
        if (walk.Word == walk.LastWord)
            walk.Bits &= LightClusterLastMask(walk.Last);
    }
    uint b = firstbitlow(walk.Bits);
    walk.Bits &= walk.Bits - 1u;
    light = walk.Word * 32u + b;
    return true;
}

// A light's illuminance at a place (lux, colour included): inverse square with the Karis window to nothing at its range,
// as light_point_cs lights surfaces, but no closer than minDistance² (a place in the air or a particle can sit in the
// flame itself), and nothing inside its fitting (PointLightInner). toLight: from the place to the light. 0 past the range.
float3 PointLightIlluminance(PointLightData light, float3 toLight, float minDistanceSq)
{
    float d2 = dot(toLight, toLight);
    float x = d2 / (light.Range * light.Range);
    if (x >= 1.0)
        return 0;
    float window = 1.0 - x * x;
    return light.Color * (light.Intensity * window * window * PointLightInner(light.InnerRadius, d2) / max(d2, minDistanceSq));
}

// 1 = lit: a light's shadow at a place with no surface (the air, a particle), from the point light shadow atlas, one
// bilinear comparison a texel and a half toward the light (the jitter and the history, or the particle's size, do the
// filtering). 1 for a light without a shadow this frame. relPos: camera-relative.
float PointLightShadowAt(ByteAddressBuffer clusters, PointLightData light, float3 relPos, SamplerComparisonState cmp)
{
    uint atlasIdx = clusters.Load(LC_SHADOW_ATLAS * 4);
    if (light.ShadowLight == POINT_LIGHT_NO_SHADOW || atlasIdx == 0)
        return 1.0;
    StructuredBuffer<LocalShadowLight> shadowLights = ResourceDescriptorHeap[clusters.Load(LC_SHADOW_LIGHTS * 4)];
    StructuredBuffer<LocalShadowView> views = ResourceDescriptorHeap[clusters.Load(LC_SHADOW_VIEWS * 4)];
    float strength = shadowLights[light.ShadowLight].Strength;
    if (strength <= 0.0)
        return 1.0;

    float3 fromLight = relPos - light.Position;
    uint v = shadowLights[light.ShadowLight].FaceView[LocalShadowFace(fromLight)];
    if (v == LOCAL_SHADOW_NO_VIEW)
        return 1.0;
    LocalShadowView view = views[v];
    // Toward the light along the ray from it: the place stays in the same face
    float axisDistance = max(abs(fromLight.x), max(abs(fromLight.y), abs(fromLight.z)));
    float d2 = dot(fromLight, fromLight);
    float3 biased = relPos - fromLight * (1.5 * view.TexelPerMetre * axisDistance * rsqrt(max(d2, 1e-6)));

    float4 clip = mul(float4(biased, 1.0), view.SampleVP);
    float3 ndc = clip.xyz / clip.w;
    float2 uv = ndc.xy * float2(0.5, -0.5) + 0.5;
    float inset = 1.0 / (view.Tile.x * LOCAL_SHADOW_PAGE_SIZE);
    uv = clamp(uv, inset, 1.0 - inset);
    Texture2DArray<float> atlas = ResourceDescriptorHeap[atlasIdx];
    float lit = atlas.SampleCmpLevelZero(cmp, float3(uv * view.Tile.xy + view.Tile.zw, view.Page), ndc.z);
    return lerp(1.0, lit, strength);
}

#endif
