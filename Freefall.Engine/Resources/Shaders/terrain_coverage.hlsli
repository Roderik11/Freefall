// terrain_coverage.hlsli — Shared by the splat bake (terrain_splat_bake.hlsl) and the decoration
// coverage bake (decoration_prepass.hlsl): the stamp descriptor both consume, its shape weight, its
// height/slope filter, and the height/slope sampling they filter on.
//
// Requires terrain_stamp_common.hlsli to be included first.

#ifndef TERRAIN_COVERAGE_HLSLI
#define TERRAIN_COVERAGE_HLSLI

// A SplatStamp or DecoStamp as the GPU sees it (matches C# CoverageStampGPU, 80 bytes)
struct CoverageStamp
{
    float2 Center;              // terrain UV center (radial)
    float  Radius;              // UV-space inner radius
    float  Falloff;             // UV-space falloff width
    uint   SplinePointOffset;   // into the spline point buffer (0xFFFFFFFF = radial)
    uint   SplinePointCount;    // 0 = radial, high bit = closed area
    float  NoiseFreq;           // edge noise frequency (0 = disabled)
    float  NoiseAmp;            // edge noise amplitude in UV space
    uint   NoiseSeed;
    uint   Flags;               // COVERAGE_* bits, operation in bits 8..15
    uint   Target;              // layer channel / decorator slot (0xFFFFFFFF = all)
    float  Strength;
    float  HeightMin, HeightMax, HeightBlend;   // normalized 0..1 of MaxHeight
    float  SlopeMin, SlopeMax, SlopeBlend;      // degrees
    uint   RequireMask;         // bit N = layer channel N must show here
    uint   ExcludeMask;         // bit N = layer channel N must not have been painted here
};

#define COVERAGE_GLOBAL 1u      // covers the whole terrain; shape fields are ignored
#define COVERAGE_FILTER 2u      // height/slope/layer terms can reject a texel

#define COVERAGE_ALL_TARGETS 0xFFFFFFFFu

uint CoverageOp(CoverageStamp s) { return (s.Flags >> 8) & 0xFFu; }

// Shape weight at a terrain UV (world-aligned, not control-map-flipped)
float CoverageShapeWeight(CoverageStamp s, float2 terrainUV, StructuredBuffer<StampSplinePoint> splinePoints)
{
    if (s.Flags & COVERAGE_GLOBAL) return 1.0;

    float nearestH, nearestHalfW;
    return EvaluateStampWeight(
        terrainUV, s.Center, s.Radius, s.Falloff,
        s.SplinePointOffset, s.SplinePointCount,
        s.NoiseFreq, s.NoiseAmp, s.NoiseSeed,
        splinePoints, nearestH, nearestHalfW);
}

// Ramp that is 0 below (edge - blend) and 1 from edge on. A zero blend is a hard step that includes the
// edge itself, so a range starting at 0 covers height 0 (smoothstep with equal edges is NaN there).
float CoverageRampUp(float edge, float blend, float x)
{
    if (blend <= 1e-6) return x >= edge ? 1.0 : 0.0;
    return saturate((x - (edge - blend)) / blend);
}

// Ramp that is 1 up to edge and 0 beyond (edge + blend)
float CoverageRampDown(float edge, float blend, float x)
{
    if (blend <= 1e-6) return x <= edge ? 1.0 : 0.0;
    return saturate(((edge + blend) - x) / blend);
}

float CoverageSmooth(float t) { return t * t * (3.0 - 2.0 * t); }

// Height/slope part of the filter (1 = passes)
float CoverageTerrainFilter(CoverageStamp s, float heightNorm, float slopeDeg)
{
    float m = 1.0;
    m *= CoverageSmooth(CoverageRampUp(s.SlopeMin, s.SlopeBlend, slopeDeg));
    m *= CoverageSmooth(CoverageRampDown(s.SlopeMax, s.SlopeBlend, slopeDeg));
    m *= CoverageSmooth(CoverageRampUp(s.HeightMin, s.HeightBlend, heightNorm));
    m *= CoverageSmooth(CoverageRampDown(s.HeightMax, s.HeightBlend, heightNorm));
    return m;
}

// Normalized height and slope in degrees at a terrain UV. The one definition of slope for terrain
// filters: central differences over two texels, as gputerrain.fx GetNormal.
void SampleTerrainHeightSlope(Texture2D heightTex, SamplerState samp, float2 terrainUV,
                              float maxHeight, float terrainSizeX,
                              out float heightNorm, out float slopeDeg)
{
    uint hmW, hmH;
    heightTex.GetDimensions(hmW, hmH);
    float texel = 1.0 / float(hmW);

    heightNorm = heightTex.SampleLevel(samp, terrainUV, 0).r;

    float hS = heightTex.SampleLevel(samp, terrainUV + float2(0, -texel), 0).r;
    float hW = heightTex.SampleLevel(samp, terrainUV + float2(-texel, 0), 0).r;
    float hE = heightTex.SampleLevel(samp, terrainUV + float2(texel, 0), 0).r;
    float hN = heightTex.SampleLevel(samp, terrainUV + float2(0, texel), 0).r;

    float heightScale = maxHeight / (2.0 * terrainSizeX * texel);

    float3 n = normalize(float3((hW - hE) * heightScale, 1.0, (hS - hN) * heightScale));
    slopeDeg = acos(saturate(n.y)) * (180.0 / 3.14159265);
}

#endif // TERRAIN_COVERAGE_HLSLI
