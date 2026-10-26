// terrain_coverage.hlsli — Shared by the splat bake (terrain_splat_bake.hlsl) and the decoration
// coverage bake (decoration_prepass.hlsl): the stamp descriptor both consume, its shape weight, its
// height/slope filter, and the height/slope sampling they filter on.
//
// Requires terrain_stamp_common.hlsli to be included first.

#ifndef TERRAIN_COVERAGE_HLSLI
#define TERRAIN_COVERAGE_HLSLI

// A SplatStamp or DecoStamp as the GPU sees it (matches C# CoverageStampGPU, 116 bytes)
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
    float  PatchSize;           // metres across, 0 = no patches
    float  PatchCoverage;       // about the share of area covered
    float  PatchSoftness;       // 0 = hard patch edge
    uint   PatchSeed;
    float4 Bounds;              // terrain UV (min x, min y, max x, max y) of all the shape can reach, falloff and
                                // edge noise included; nothing outside it is tested against the shape
    uint   Rows;                // deco: 1 + the stamp's DecoRow (grass_compute.hlsl), 0 = scattered
};

#define DECO_ROW_MARKER 254u    // a control texel channel with this slot names a DecoRow in its low byte

#define COVERAGE_GLOBAL 1u      // covers the whole terrain; shape fields are ignored
#define COVERAGE_FILTER 2u      // height/slope/layer terms can reject a texel

#define COVERAGE_ALL_TARGETS 0xFFFFFFFFu

uint CoverageOp(CoverageStamp s) { return (s.Flags >> 8) & 0xFFu; }

// Shape weight at a terrain UV (world-aligned, not control-map-flipped)
float CoverageShapeWeight(CoverageStamp s, float2 terrainUV, StructuredBuffer<StampSplinePoint> splinePoints)
{
    if (s.Flags & COVERAGE_GLOBAL) return 1.0;

    // Off the shape's box: a spline stamp's shape is a walk over every one of its segments (a forest outline has
    // over a thousand), and every texel of the map used to take it for every stamp (2026-10-26: 1.26 s of GPU for
    // one Eldermoor splat bake, near the 2 s at which Windows resets the device)
    if (any(terrainUV < s.Bounds.xy) || any(terrainUV > s.Bounds.zw)) return 0.0;

    float nearestH, nearestHalfW;
    return EvaluateStampWeight(
        terrainUV, s.Center, s.Radius, s.Falloff,
        s.SplinePointOffset, s.SplinePointCount,
        s.NoiseFreq, s.NoiseAmp, s.NoiseSeed,
        splinePoints, nearestH, nearestHalfW, false);
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

// Irregular patches (CoverageStamp.PatchSize): 1 inside a patch, 0 between them. Gradient noise at the patch size
// plus an octave at a third of it for ragged edges. The sum is close to normal with sigma 0.185 (measured over 400k
// samples); its logistic CDF turns it uniform, so PatchCoverage is the share of area at or above half weight.
float CoveragePatches(CoverageStamp s, float2 worldXZ)
{
    if (s.PatchSize <= 0) return 1.0;
    float2 seedOffset = float2(frac(s.PatchSeed * 0.0001234), frac(s.PatchSeed * 0.0007891)) * 1000.0;
    float2 p = worldXZ / s.PatchSize + seedOffset;
    float n = stamp_noise(p) + 0.35 * stamp_noise(p * 3.1 + 17.0);
    n = 1.0 / (1.0 + exp(-9.2 * n));                // 1.702 / 0.185
    float edge = 1.0 - s.PatchCoverage;
    float soft = max(s.PatchSoftness * 0.5, 0.01);
    return smoothstep(edge - soft, edge + soft, n);
}

// Height/slope/patch part of the filter (1 = passes). worldXZ: metres from the terrain origin.
float CoverageTerrainFilter(CoverageStamp s, float heightNorm, float slopeDeg, float2 worldXZ)
{
    float m = 1.0;
    m *= CoverageSmooth(CoverageRampUp(s.SlopeMin, s.SlopeBlend, slopeDeg));
    m *= CoverageSmooth(CoverageRampDown(s.SlopeMax, s.SlopeBlend, slopeDeg));
    m *= CoverageSmooth(CoverageRampUp(s.HeightMin, s.HeightBlend, heightNorm));
    m *= CoverageSmooth(CoverageRampDown(s.HeightMax, s.HeightBlend, heightNorm));
    m *= CoveragePatches(s, worldXZ);
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
