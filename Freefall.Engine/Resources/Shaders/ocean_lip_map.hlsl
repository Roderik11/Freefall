// The surf's breaking lips round the camera, decided once per frame (C# LipMap): per texel the lip amount of the crest
// the point's strip belongs to, the phase's mean rate back to it and the crest's age, smoothed over a few metres. The
// ocean's PS, the lip pass (ocean_lip.fx) and the spray all read it (LipMapAt), so they agree on where a lip stands.
//   CSLipMapBuild   the crest search per texel (LipStripAmount) -> raw
//   CSLipMapSmooth  a 5 x 5 tent over raw -> the map: a jump in the crest search (two crests meeting, a front too
//                   steep for it) is a ramp a few metres wide instead of a step between two vertices

#pragma kernel CSLipMapBuild
#pragma kernel CSLipMapSmooth

#include "ocean_surf.hlsli"

cbuffer PushConstants : register(b3)
{
    uint OceanDataIdx;   // StructuredBuffer<OceanData>: this frame's, its LipMapOrigin the map's corner
    uint RawUAVIdx;      // RWTexture2D<float4>: amount, k, age
    uint RawSRVIdx;      // the same, to read
    uint MapUAVIdx;      // RWTexture2D<float4>
};

// The phase range a texel is decided over: wider than the lip's own (its skirt reaches 0.2 behind the crest and to 3.0
// ahead) by more than the smoothing's 2 m, so the smoothing finds the strip's amount at its ends, not the zeros past
// them. Ahead it ends at the trough (π), where the next wave's back begins
static const float2 LIP_MAP_PHASE = float2(-1.2, 3.1);

[numthreads(8, 8, 1)]
void CSLipMapBuild(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LIP_MAP_SIZE || id.y >= LIP_MAP_SIZE) return;
    StructuredBuffer<OceanData> oceans = ResourceDescriptorHeap[OceanDataIdx];
    OceanData ocean = oceans[0];
    RWTexture2D<float4> raw = ResourceDescriptorHeap[RawUAVIdx];

    float2 xz = ocean.LipMapOrigin + (float2(id.xy) + 0.5) * LIP_MAP_TEXEL;
    float2 centre = ocean.LipMapOrigin + 0.5 * LIP_MAP_SIZE * LIP_MAP_TEXEL;
    float4 result = 0.0;
    OceanShore shore = ShoreAt(ocean, xz);
    if (ocean.WarpOmega.x + ocean.WarpOmega.y > 0.0 && ocean.HeightmapSRV != 0 && shore.shoreDepth < 1e3)
    {
        SurfWave wave = LipWave(ocean, xz);
        float phase = LipPhase(wave);
        // As the lip pass's patch test had it: no lip where the wave is small for the depth or long past breaking
        float r = 2.0 * SurfEnvelope(wave) / max(shore.shoreDepth, 0.05);
        [branch]
        if (phase > LIP_MAP_PHASE.x && phase < LIP_MAP_PHASE.y && r > 0.1 && r < 3.0)
        {
            float2 crestXZ;
            SurfWave crestWave;
            float k, age;
            float amount = LipStripAmount(ocean, xz, wave, centre, crestXZ, crestWave, k, age);
            result = float4(amount, k, age, 0.0);
        }
    }
    raw[id.xy] = result;
}

[numthreads(8, 8, 1)]
void CSLipMapSmooth(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LIP_MAP_SIZE || id.y >= LIP_MAP_SIZE) return;
    Texture2D<float4> raw = ResourceDescriptorHeap[RawSRVIdx];
    RWTexture2D<float4> map = ResourceDescriptorHeap[MapUAVIdx];

    // A tent (1 2 3 2 1)²: the amount over every texel (none outside a strip), k and the age over the texels in one
    float amount = 0.0, kSum = 0.0, ageSum = 0.0, weight = 0.0;
    [unroll]
    for (int dy = -2; dy <= 2; ++dy)
    {
        [unroll]
        for (int dx = -2; dx <= 2; ++dx)
        {
            int2 p = clamp(int2(id.xy) + int2(dx, dy), 0, LIP_MAP_SIZE - 1);
            float3 v = raw.Load(int3(p, 0)).xyz;
            float w = (3.0 - abs(dx)) * (3.0 - abs(dy));
            amount += v.x * w;
            float valid = v.y > 0.0 ? w : 0.0;
            kSum += v.y * valid;
            ageSum += v.z * valid;
            weight += valid;
        }
    }
    map[id.xy] = weight > 0.0 ? float4(amount / 81.0, kSum / weight, ageSum / weight, 0.0) : float4(amount / 81.0, 0.0, 0.0, 0.0);
}
