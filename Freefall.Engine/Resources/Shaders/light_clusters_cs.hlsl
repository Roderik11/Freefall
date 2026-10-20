// Light clusters: builds the z-bins and the tile masks of light_clusters.hlsli from this frame's visible point lights
// (point_light_cull.hlsl), once per view, after the point light shadow selection has written each light's shadow
// (Graphics/LightClusters.cs):
//   CSClusterSort   one group: sorts the visible lights by the view depth of their centres (bitonic, in group shared
//                   memory), writes them in that order and each light's first and last z-bin; the header
//   CSClusterBins   a thread per z-bin: the first and last sorted light whose sphere reaches into it
//   CSClusterTiles  a group per 32-pixel tile: a bit for every sorted light whose sphere the tile's frustum sees
// Docs: .agent/knowledge/rendering_engine/features/lighting/point_lights.md "Light clusters".

#pragma kernel CSClusterSort
#pragma kernel CSClusterBins
#pragma kernel CSClusterTiles

cbuffer PushConstants : register(b3)
{
    uint VisibleUAVIdx;         // RWStructuredBuffer<PointLightData>: this frame's visible lights (read)
    uint VisibleCountUAVIdx;    // RWByteAddressBuffer: [0] = lights in the list; 0 = no lights this frame
    uint FrameParamsIdx;        // StructuredBuffer<PointLightFrame>: the camera the lights were culled for
    uint LightsUAVIdx;          // RWStructuredBuffer<PointLightData>: the lights in view-depth order (written)
    uint ClustersUAVIdx;        // RWByteAddressBuffer: the cluster buffer (written)
    uint ViewWidth;             // the view's pixels
    uint ViewHeight;
    uint TilesX;                // 32-pixel tiles across
    uint ShadowAtlasIdx;        // for the header: the point light shadow atlas rendered this frame, 0 = none
    uint ShadowViewsIdx;
    uint ShadowLightsIdx;
    uint SunShadowMapIdx;       // ... the directional light's cascades drawn this frame, 0 = none
    uint SunCascadesIdx;
    uint SunCascadeCount;
    float SunDirectionX;        // ... toward the body they were drawn for
    float SunDirectionY;
    float SunDirectionZ;
};

#include "light_clusters.hlsli"

// View depth of a camera-relative place: clip w of the camera-relative view-projection
float ViewDepthOf(float4x4 vp, float3 p)
{
    return p.x * vp._14 + p.y * vp._24 + p.z * vp._34 + vp._44;
}

groupshared uint g_Keys[MAX_VISIBLE_LIGHTS];

[numthreads(1024, 1, 1)]
void CSClusterSort(uint gi : SV_GroupIndex)
{
    RWByteAddressBuffer clusters = ResourceDescriptorHeap[ClustersUAVIdx];
    uint count = 0;
    if (VisibleCountUAVIdx != 0)
    {
        RWByteAddressBuffer countBuf = ResourceDescriptorHeap[VisibleCountUAVIdx];
        count = min(countBuf.Load(0), MAX_VISIBLE_LIGHTS);
    }
    if (gi == 0)
    {
        clusters.Store(LC_COUNT * 4, count);
        clusters.Store(LC_WORDS * 4, (count + 31u) / 32u);
        clusters.Store(LC_TILES_X * 4, TilesX);
        clusters.Store(LC_WIDTH * 4, ViewWidth);
        clusters.Store(LC_HEIGHT * 4, ViewHeight);
        clusters.Store(LC_SHADOW_ATLAS * 4, ShadowAtlasIdx);
        clusters.Store(LC_SHADOW_VIEWS * 4, ShadowViewsIdx);
        clusters.Store(LC_SHADOW_LIGHTS * 4, ShadowLightsIdx);
        clusters.Store(LC_SUN_SHADOW_MAP * 4, SunShadowMapIdx);
        clusters.Store(LC_SUN_CASCADES * 4, SunCascadesIdx);
        clusters.Store(LC_SUN_CASCADE_COUNT * 4, SunCascadeCount);
        clusters.Store3(LC_SUN_DIRECTION * 4, asuint(float3(SunDirectionX, SunDirectionY, SunDirectionZ)));
    }
    if (count == 0)
        return;

    RWStructuredBuffer<PointLightData> visible = ResourceDescriptorHeap[VisibleUAVIdx];
    StructuredBuffer<PointLightFrame> frames = ResourceDescriptorHeap[FrameParamsIdx];
    float4x4 vp = frames[0].CameraRelativeVP;

    // Keys: the centre's view depth in the top 20 bits (a positive float's bits sort as the float does: 11 bits of
    // mantissa, 1/2048 of the depth), the visible index in the low 12. Padded to a power of two with keys that sort last.
    uint n = count <= 1u ? 1u : 1u << (firstbithigh(count - 1u) + 1u);
    for (uint i = gi; i < n; i += 1024u)
    {
        uint key = 0xFFFFFFFFu;
        if (i < count)
        {
            float z = max(ViewDepthOf(vp, visible[i].Position), 1e-3);
            key = ((asuint(z) >> 12) << 12) | i;
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

    RWStructuredBuffer<PointLightData> lights = ResourceDescriptorHeap[LightsUAVIdx];
    for (uint s = gi; s < count; s += 1024u)
    {
        PointLightData light = visible[g_Keys[s] & 0xFFFu];
        lights[s] = light;
        float z = ViewDepthOf(vp, light.Position);
        uint bins = 0xFFFFu;    // wholly behind the camera: in no bin
        if (z + light.Range > 0.0)
            bins = LightZBin(max(z - light.Range, 0.0)) | (LightZBin(z + light.Range) << 16);
        clusters.Store((LC_LIGHT_BINS + s) * 4, bins);
    }
}

groupshared uint g_LightBins[64];

[numthreads(64, 1, 1)]
void CSClusterBins(uint3 dtid : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    RWByteAddressBuffer clusters = ResourceDescriptorHeap[ClustersUAVIdx];
    uint count = clusters.Load(LC_COUNT * 4);
    uint bin = dtid.x;
    uint first = 0xFFFFu, last = 0u;
    for (uint base = 0; base < count; base += 64u)
    {
        g_LightBins[gi] = base + gi < count ? clusters.Load((LC_LIGHT_BINS + base + gi) * 4) : 0xFFFFu;
        GroupMemoryBarrierWithGroupSync();
        uint chunk = min(64u, count - base);
        for (uint i = 0; i < chunk; i++)
        {
            uint b = g_LightBins[i];
            if (bin >= (b & 0xFFFFu) && bin <= (b >> 16))
            {
                first = min(first, base + i);
                last = base + i;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (bin < LIGHT_ZBINS)
        clusters.Store((LC_BINS + bin) * 4, first | (last << 16));
}

groupshared uint g_Mask[LIGHT_CLUSTER_WORDS];

// A plane through the camera (camera-relative), normal in, of unit length
float4 TilePlane(float4 p)
{
    return p / length(p.xyz);
}

[numthreads(64, 1, 1)]
void CSClusterTiles(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    RWByteAddressBuffer clusters = ResourceDescriptorHeap[ClustersUAVIdx];
    uint count = clusters.Load(LC_COUNT * 4);
    uint words = (count + 31u) / 32u;
    for (uint w = gi; w < words; w += 64u)
        g_Mask[w] = 0;
    GroupMemoryBarrierWithGroupSync();

    if (count > 0)
    {
        // The tile's four side planes from the camera-relative view-projection's columns: ndc.x >= left is
        // clip.x - left × clip.w >= 0, and so on (planes through the camera: the projection has no translation there)
        StructuredBuffer<PointLightFrame> frames = ResourceDescriptorHeap[FrameParamsIdx];
        float4x4 vp = frames[0].CameraRelativeVP;
        float4 c0 = float4(vp._11, vp._21, vp._31, vp._41);
        float4 c1 = float4(vp._12, vp._22, vp._32, vp._42);
        float4 c3 = float4(vp._14, vp._24, vp._34, vp._44);
        float2 size = float2(ViewWidth, ViewHeight);
        float2 uv0 = float2(gid.xy << LIGHT_CLUSTER_TILE_SHIFT) / size;
        float2 uv1 = min(float2((gid.xy + 1u) << LIGHT_CLUSTER_TILE_SHIFT), size) / size;
        float4 planes[4];
        planes[0] = TilePlane(c0 - (uv0.x * 2.0 - 1.0) * c3);       // left
        planes[1] = TilePlane((uv1.x * 2.0 - 1.0) * c3 - c0);       // right
        planes[2] = TilePlane(c1 - (1.0 - uv1.y * 2.0) * c3);       // bottom
        planes[3] = TilePlane((1.0 - uv0.y * 2.0) * c3 - c1);       // top

        RWStructuredBuffer<PointLightData> lights = ResourceDescriptorHeap[LightsUAVIdx];
        for (uint s = gi; s < count; s += 64u)
        {
            PointLightData light = lights[s];
            bool inside = true;
            [unroll] for (uint p = 0; p < 4; p++)
                inside = inside && dot(planes[p].xyz, light.Position) + planes[p].w >= -light.Range;
            if (inside)
                InterlockedOr(g_Mask[s >> 5], 1u << (s & 31u));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint maskBase = (LC_MASKS + (gid.y * TilesX + gid.x) * LIGHT_CLUSTER_WORDS) * 4;
    for (uint w2 = gi; w2 < words; w2 += 64u)
        clusters.Store(maskBase + w2 * 4, g_Mask[w2]);
}
