// Point light cull (PointLightBuffer.cs, DeferredRenderer.CullPointLights)
//
// CSCountLights:  one thread per registry slot. Position from the light entity's world matrix in the global
//                 transform buffer, camera frustum; histogram of the lights in view by importance.
// CSCullLights:   the same per slot, flicker applied; appends the lights in view to this frame's visible list,
//                 the most important first when there are more than it holds: lights below the importance band
//                 that keeps the list within MAX_VISIBLE_LIGHTS are left out, those in that band fade with their
//                 importance. Which lights miss out is decided by their importance, not by the order the GPU
//                 happens to append them in (that order changes every frame: whole lights blinked, 2026-10-16).
//                 Runs after the opaque pass, before the point light shadows (their selection ranks this list).
// CSOccludeLights: one thread per visible light, in the lighting phase once this frame's Hi-Z pyramid exists:
//                 flags lights wholly behind it, which the tiled pass then skips.

#pragma kernel CSCountLights
#pragma kernel CSCullLights
#pragma kernel CSOccludeLights

cbuffer PushConstants : register(b3)
{
    uint RecordsIdx;            // StructuredBuffer<PointLightRecord>: the registry
    uint RecordCountIdx;        // registry slots to walk
    uint TransformsIdx;         // StructuredBuffer<row_major float4x4>: the global transform buffer
    uint FrameParamsIdx;        // StructuredBuffer<PointLightFrame>: [0]
    uint VisibleUAVIdx;         // RWStructuredBuffer<PointLightData>: the visible list
    uint VisibleCountUAVIdx;    // RWByteAddressBuffer: [0] = lights in the list, [1] = lights in view
    uint HiZTexIdx;             // CSOccludeLights: Hi-Z pyramid of this frame's linear depth
    uint HistogramUAVIdx;       // RWStructuredBuffer<uint>[IMPORTANCE_BINS]: lights in view per importance band
};

#include "point_lights.hlsli"

#define IMPORTANCE_BINS 64u

// A light in view and its importance: steady brightness × its on-screen radius² (the size of its pool of light).
// From the light's steady position, so flicker does not move it across a band and make it blink.
bool CullLight(uint slot, PointLightFrame frame, out PointLightRecord r, out float3 world, out float importance)
{
    StructuredBuffer<PointLightRecord> records = ResourceDescriptorHeap[RecordsIdx];
    r = records[slot];
    world = 0;
    importance = 0;
    if ((r.Flags & POINT_LIGHT_ENABLED) == 0 || r.Range <= 0.0 || r.Intensity <= 0.0)
        return false;

    StructuredBuffer<row_major float4x4> transforms = ResourceDescriptorHeap[TransformsIdx];
    world = mul(float4(0, 0, 0, 1), transforms[r.TransformSlot]).xyz;

    // Camera frustum (camera-relative, normals in)
    float3 p = world - frame.CameraPosition;
    [unroll]
    for (uint i = 0; i < 5; i++)
    {
        if (dot(frame.Planes[i].xyz, p) + frame.Planes[i].w < -r.Range)
            return false;
    }

    float radiusPixels = r.Range * frame.PixelsPerMetre / max(length(p), 0.5 * r.Range);
    importance = dot(r.Color, float3(0.2126, 0.7152, 0.0722)) * r.Intensity * radiusPixels * radiusPixels;
    return importance > 0.0;
}

// Half-octave bands: the position within one is frac(log2(importance) × 2)
uint ImportanceBin(float importance)
{
    return (uint)clamp((int)floor(log2(importance) * 2.0) + 16, 0, (int)IMPORTANCE_BINS - 1);
}

[numthreads(64, 1, 1)]
void CSCountLights(uint3 dtid : SV_DispatchThreadID)
{
    uint slot = dtid.x;
    if (slot >= RecordCountIdx)
        return;

    StructuredBuffer<PointLightFrame> frames = ResourceDescriptorHeap[FrameParamsIdx];
    PointLightRecord r;
    float3 world;
    float importance;
    if (!CullLight(slot, frames[0], r, world, importance))
        return;

    RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramUAVIdx];
    InterlockedAdd(histogram[ImportanceBin(importance)], 1);
    RWByteAddressBuffer countBuf = ResourceDescriptorHeap[VisibleCountUAVIdx];
    countBuf.InterlockedAdd(4, 1);
}

groupshared uint g_LowestBand;   // lights in lower bands are left out
groupshared uint g_FadedBand;    // lights in this band fade with their importance (none: IMPORTANCE_BINS)

[numthreads(64, 1, 1)]
void CSCullLights(uint3 dtid : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    if (gi == 0)
    {
        // Highest bands first, as many as the list holds
        RWStructuredBuffer<uint> histogram = ResourceDescriptorHeap[HistogramUAVIdx];
        uint total = 0;
        for (uint b = 0; b < IMPORTANCE_BINS; b++)
            total += histogram[b];

        uint lowest = 0, faded = IMPORTANCE_BINS;
        if (total > MAX_VISIBLE_LIGHTS)
        {
            uint sum = 0;
            lowest = IMPORTANCE_BINS - 1;   // a top band over the cap on its own: the cap drops the rest
            for (int band = (int)IMPORTANCE_BINS - 1; band >= 0; band--)
            {
                if (sum + histogram[band] > MAX_VISIBLE_LIGHTS)
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
    if (slot >= RecordCountIdx)
        return;

    StructuredBuffer<PointLightFrame> frames = ResourceDescriptorHeap[FrameParamsIdx];
    PointLightFrame frame = frames[0];
    PointLightRecord r;
    float3 world;
    float importance;
    if (!CullLight(slot, frame, r, world, importance))
        return;

    uint band = ImportanceBin(importance);
    if (band < g_LowestBand)
        return;

    float intensity = r.Intensity;
    if (band == g_FadedBand)
        intensity *= saturate(frac(log2(importance) * 2.0)); // fades out toward the band's lower edge
    if (r.Flicker > 0.0)
        ApplyFlicker(r, frame.Time, intensity);

    RWByteAddressBuffer countBuf = ResourceDescriptorHeap[VisibleCountUAVIdx];
    uint index;
    countBuf.InterlockedAdd(0, 1, index);
    if (index >= MAX_VISIBLE_LIGHTS)
        return;

    PointLightData light;
    light.Color = r.Color;
    light.Intensity = intensity;
    light.Position = world - frame.CameraPosition;
    light.Range = r.Range;
    light.ShadowLight = POINT_LIGHT_NO_SHADOW;
    light.Slot = slot;
    light.Flags = 0;
    light.Score = dot(r.Color, float3(0.2126, 0.7152, 0.0722)) * r.Intensity; // steady: no flicker in the rank

    RWStructuredBuffer<PointLightData> visible = ResourceDescriptorHeap[VisibleUAVIdx];
    visible[index] = light;
}

// Sphere wholly behind the Hi-Z pyramid (the test CSCullLights in light_point_cs.hlsl used to make)
[numthreads(64, 1, 1)]
void CSOccludeLights(uint3 dtid : SV_DispatchThreadID)
{
    RWByteAddressBuffer countBuf = ResourceDescriptorHeap[VisibleCountUAVIdx];
    uint count = min(countBuf.Load(0), MAX_VISIBLE_LIGHTS);
    uint i = dtid.x;
    if (i >= count || HiZTexIdx == 0)
        return;

    RWStructuredBuffer<PointLightData> visible = ResourceDescriptorHeap[VisibleUAVIdx];
    PointLightData light = visible[i];

    StructuredBuffer<PointLightFrame> frames = ResourceDescriptorHeap[FrameParamsIdx];
    float4x4 vp = frames[0].CameraRelativeVP;

    float4 clipCenter = mul(float4(light.Position, 1.0), vp);
    if (clipCenter.w <= 0)
        return; // centre behind the camera: cannot tell

    float3 ndc = clipCenter.xyz / clipCenter.w;
    float2 uv = ndc.xy * float2(0.5, -0.5) + 0.5;
    if (any(uv < 0.0) || any(uv > 1.0))
        return; // centre off screen: cannot tell

    Texture2D<float> hiZ = ResourceDescriptorHeap[HiZTexIdx];
    float w, h, levels;
    hiZ.GetDimensions(0, w, h, levels);

    float projScale = max(abs(vp._m11), 0.001);
    float screenRadius = (light.Range * projScale / clipCenter.w) * h * 0.5;

    float mipLevel = min(ceil(log2(max(screenRadius * 2.0, 1.0))), levels - 1.0);
    uint mip = (uint)mipLevel;

    float mipW, mipH, unused;
    hiZ.GetDimensions(mip, mipW, mipH, unused);
    float2 mipSize = float2(mipW, mipH);

    int2 baseCoord = int2(uv * mipSize - 0.5);
    int2 maxCoord = int2(mipSize) - 1;
    float d0 = hiZ.Load(int3(clamp(baseCoord,             int2(0, 0), maxCoord), mip));
    float d1 = hiZ.Load(int3(clamp(baseCoord + int2(1, 0), int2(0, 0), maxCoord), mip));
    float d2 = hiZ.Load(int3(clamp(baseCoord + int2(0, 1), int2(0, 0), maxCoord), mip));
    float d3 = hiZ.Load(int3(clamp(baseCoord + int2(1, 1), int2(0, 0), maxCoord), mip));

    float sampledDepth = max(max(d0, d1), max(d2, d3));
    if (clipCenter.w - light.Range > sampledDepth)
        visible[i].Flags = light.Flags | VISIBLE_LIGHT_OCCLUDED;
}
