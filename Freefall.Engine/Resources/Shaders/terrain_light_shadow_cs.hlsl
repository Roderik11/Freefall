// The terrain's shadow in the air, per light (god rays, ATMOSPHERE_PLAN open issue "the sun's glow in the haze is not
// shadowed"): over the terrain and a margin round it, the height below which the terrain hides the light,
//   H(p) = max over s > 0 of  h(p + s·d) − s·tan(e)        (d: the light's direction across the ground, e: its elevation)
// Air at p lower than H(p) lies in the terrain's shadow. Made by Graphics/TerrainLightShadow.cs when the light or the
// terrain changes: CSShadowInit takes the first step (one texel toward the light, the highest ground in it), then
// CSShadowStep doubles the reach each pass, M(k+1)(p) = max(M(k)(p), M(k)(p + 2^k·Δ·d) − 2^k·Δ·tan(e)), until it
// spans the map. A march per sample missed single peaks kilometres away (8 steps leapt from 970 m to 2150 m past one).
// Read by atmosphere_common.hlsli AtmTerrainLightVisibility (the aerial perspective volume, the near fog volume).

#pragma kernel CSShadowInit
#pragma kernel CSShadowStep

// What changes from dispatch to dispatch (the light, the reach) is pushed: the Params buffer holds one copy per frame
cbuffer PushConstants : register(b3)
{
    uint HeightmapIdx;          // the terrain's baked heightmap (R16)
    uint SourceTexIdx;          // M(k)
    uint TargetUAVIdx;          // M(k+1)
    uint LightDirXBits;         // asfloat: the light's direction across the ground, unit (texel x = world x,
    uint LightDirZBits;         //          texel y = world z)
    uint LightTanBits;          // asfloat: tan of the light's elevation
    uint StepTexelsBits;        // asfloat: this pass's reach, 2^k texels
};

cbuffer Params : register(b4)
{
    float2 MapOrigin;           // world xz of the map's corner
    float MapSize;              // metres the map spans each way
    uint MapResolution;         // texels each way
    float3 TerrainOrigin;       // the terrain's corner (y = its base)
    float TerrainMaxHeight;     // metres at 1
    float2 TerrainSize;         // metres it spans in x and z
    float2 _shadowPad;
};

#define LightDirXZ float2(asfloat(LightDirXBits), asfloat(LightDirZBits))
#define LightTan asfloat(LightTanBits)
#define StepTexels asfloat(StepTexelsBits)

#include "sky_visibility.hlsli"

SamplerState ShadowLinearClamp : register(s2);

#define NO_SHADOW -10000.0      // nothing toward the light

float HeightOrNone(TerrainCover t, float2 xz)
{
    float2 cell = (xz - t.Origin.xz) / t.Size;
    if (any(cell < 0.0) || any(cell > 1.0))
        return NO_SHADOW;
    return TerrainHeightAt(t, ShadowLinearClamp, xz);
}

[numthreads(8, 8, 1)]
void CSShadowInit(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= MapResolution))
        return;
    TerrainCover t;
    t.Heightmap = HeightmapIdx;
    t.MaxHeight = TerrainMaxHeight;
    t.Origin = TerrainOrigin;
    t.Size = TerrainSize;

    // The highest ground in the texel one step toward the light (the heightmap is finer than this map: its peaks must
    // not fall between the samples)
    float texel = MapSize / MapResolution;
    float2 p = MapOrigin + (id.xy + 0.5) * texel + LightDirXZ * texel;
    float h = NO_SHADOW;
    [unroll] for (int i = 0; i < 4; i++)
    {
        float2 offset = (float2(i & 1, i >> 1) - 0.5) * (texel * 0.5);
        h = max(h, HeightOrNone(t, p + offset));
    }
    RWTexture2D<float> target = ResourceDescriptorHeap[TargetUAVIdx];
    target[id.xy] = h > NO_SHADOW ? h - texel * LightTan : NO_SHADOW;
}

[numthreads(8, 8, 1)]
void CSShadowStep(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= MapResolution))
        return;
    Texture2D<float> source = ResourceDescriptorHeap[SourceTexIdx];
    RWTexture2D<float> target = ResourceDescriptorHeap[TargetUAVIdx];
    float m = source[id.xy];
    float2 uv = (id.xy + 0.5 + LightDirXZ * StepTexels) / MapResolution;
    if (all(uv >= 0.0) && all(uv <= 1.0))
    {
        float further = source.SampleLevel(ShadowLinearClamp, uv, 0);
        m = max(m, further - StepTexels * (MapSize / MapResolution) * LightTan);
    }
    target[id.xy] = m;
}
