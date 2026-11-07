// terrain_height_bake.hlsl — Composites the terrain's height stamps into the terrain's height pool (R16_UNorm pages).
//
// TerrainBaker walks the height stamps in priority order and dispatches one kernel per step and page:
//   CS_InfluenceLayer  a run of HeightStamps (flatten / heightmap pattern, radial or spline)
//   CS_ImportLayer     a global HeightStamp with a heightmap (whole-terrain import)
//   CS_NoiseLayer      a HeightNoiseStamp
// CS_Clear zeroes the page first. CS_CompareReads is a check (TerrainBaker.CompareReads).
//
// Every kernel bakes a box of the terrain's grid (RegionX/Y/W/H), a page with its border (TERRAIN_PAGES_PLAN.md stages 3
// and 4a): a sample is a function of the stamps and of its integer index in the grid alone, never of the box it was
// baked in, so a page baked on its own equals the same samples of a whole bake to the bit, and a page's border equals
// its neighbour's samples. No kernel reads a neighbour. Sample s is written at pool texel s + (OutX, OutY) of OutSlice.
//
// Uses push constants (b3) matching engine convention; a step's parameters are in the plan's step buffer.

#pragma kernel CS_ImportLayer
#pragma kernel CS_Clear
#pragma kernel CS_NoiseLayer
#pragma kernel CS_InfluenceLayer
#pragma kernel CS_CompareReads

// Push constants (root parameter 0, register b3) — bindless indices + params
cbuffer PushConstants : register(b3)
{
    uint OutputIdx;          // 0  UAV: the height pool, RWTexture2DArray<float>
    uint StampBufIdx;        // 1  SRV: StructuredBuffer<HeightStampDescriptor>, every descriptor of the bake (influence)
    uint StampListIdx;       // 2  SRV: StructuredBuffer<uint>, the pages' stamp lists (influence)
    uint SplineBufIdx;       // 3  SRV: StructuredBuffer<StampSplinePoint> (influence)
    uint StepBufIdx;         // 4  SRV: StructuredBuffer<BakeStep>, the plan's steps (import, noise)
    uint StepIndex;          // 5  this dispatch's step in it
    uint ListStart;          // 6  this page's stamps of this run: list entries ListStart .. ListStart + ListCount
    uint ListCount;          // 7

    int  RegionX;            // 8  the box baked: its first sample in the terrain's grid (before the terrain, in a border)
    int  RegionY;            // 9
    uint RegionW;            // 10 and its size, in samples
    uint RegionH;            // 11
    int  OutX;               // 12 sample s is written at pool texel s + (OutX, OutY) of slice OutSlice
    int  OutY;               // 13
    uint OutSlice;           // 14

    uint GridSamples;        // 15 samples across the terrain (N): sample i at terrain UV i / (N - 1)
    float OriginX;           // 16 world X and Z of sample (0, 0), metres
    float OriginZ;           // 17
    float Spacing;           // 18 metres between samples (Terrain.VertexSpacing)

    float ClearValue;        // 19 what CS_Clear writes: 0 for a bake (TerrainBaker.CheckPageBake fills with another)
    uint NoiseLUTIdx;        // 20 SRV: the tileable noise LUT

    // ── CS_CompareReads (TerrainBaker.CompareReads): terrain_pages.hlsli's reads, for the CPU to check ──
    uint TerrainParamsIdx;   // 21 StructuredBuffer<TerrainPagesParams>
    uint CompareUAVIdx;      // 22 UAV: RWStructuredBuffer<float4> (height, 0, x, z)
    uint CompareCount;       // 23
};

// One step of the plan (C# TerrainBaker.HeightStepGPU), 96 bytes
struct BakeStep
{
    uint   BlendMode;          // 0=Set, 1=Add, 2=Max, 3=Lerp, 4=Min
    float  Opacity;            // [0..1]
    // Import (CS_ImportLayer)
    uint   Source;             // SRV: the imported heightmap
    float  ImportScale;        // the source's value times this, plus ImportBias (both normalized to MaxHeight)
    float  ImportBias;
    float2 ImportMin;          // world X and Z where the source's first sample lies, metres
    float2 ImportInvSize;      // 1 / the metres from its first sample to its last
    // Noise (CS_NoiseLayer)
    uint   NoiseType;          // 0=Simplex, 1=Perlin, 2=Ridged, 3=Billow
    uint   Octaves;            // number of fBm octaves
    float  Frequency;          // LUT units per metre: 1 / (16 x HeightNoiseStamp.FeatureSize)
    float  Amplitude;          // output amplitude, normalized to MaxHeight
    float  Lacunarity;         // per-octave frequency multiplier
    float  Persistence;        // per-octave amplitude decay
    float2 Offset;             // metres the pattern is panned by
    uint   Seed;               // seed for noise permutation
    uint   TerraceSteps;       // 0=disabled, N = number of terrace shelves
    float  TerraceSmoothness;  // 0=sharp, 1=fully rounded transitions
    float2 MaskCenter;         // world X and Z of the radial mask's centre, metres
    float  MaskRadius;         // metres, 0 = no mask (the whole terrain)
    float  MaskFalloff;        // mask edge falloff exponent
};

BakeStep LoadStep()
{
    StructuredBuffer<BakeStep> steps = ResourceDescriptorHeap[StepBufIdx];
    return steps[StepIndex];
}

SamplerState sampLinear : register(s0);

float Blend(float prev, float value, uint mode, float opacity)
{
    value *= opacity;
    switch (mode)
    {
        case 0: return value;                          // Set
        case 1: return prev + value;                   // Add
        case 2: return max(prev, value);               // Max
        case 3: return lerp(prev, value, opacity);     // Lerp
        case 4: return min(prev, value);               // Min
        default: return value;
    }
}

#include "terrain_grid.hlsli"
#include "terrain_pages.hlsli"

// The sample this thread bakes: its index in the terrain's grid (negative in the border before the terrain). False past
// the box.
bool RegionSample(uint3 dtid, out int2 sid)
{
    sid = int2(RegionX, RegionY) + int2(dtid.xy);
    return dtid.x < RegionW && dtid.y < RegionH;
}

// Where sample s of this page is written in the pool
uint3 OutTexel(int2 sid)
{
    return uint3(uint2(sid + int2(OutX, OutY)), OutSlice);
}

// The terrain UV of height sample (x, y): one sample on every vertex, sample i at i / (N - 1), the first and the last on
// the terrain's edges (terrain_grid.hlsli). Until 2026-11-07, (i + 0.5) / N, between the vertices.
float2 SampleTerrainUV(int2 sid)
{
    return float2(sid) / float(GridSamples - 1);
}

// The world X, Z of height sample (x, y), metres: from its integer index, so every box computes it alike
float2 SampleWorldXZ(int2 sid)
{
    return float2(OriginX, OriginZ) + float2(sid) * Spacing;
}

// ── CS_Clear: Zero the box (ClearValue) ──
[numthreads(8, 8, 1)]
void CS_Clear(uint3 dtid : SV_DispatchThreadID)
{
    int2 sid;
    if (!RegionSample(dtid, sid)) return;
    RWTexture2DArray<float> Output = ResourceDescriptorHeap[OutputIdx];
    Output[OutTexel(sid)] = ClearValue;
}

// ── CS_ImportLayer: Blend a heightmap laid over a rectangle of the world ──
// value = source * ImportScale + ImportBias (scale and bias, both normalized to MaxHeight). The source is read as a
// height map is: its first and last samples on the rectangle's edges (terrain_grid.hlsli), so one of the terrain's own
// size laid over the terrain imports sample for sample.
[numthreads(8, 8, 1)]
void CS_ImportLayer(uint3 dtid : SV_DispatchThreadID)
{
    int2 sid;
    if (!RegionSample(dtid, sid)) return;
    RWTexture2DArray<float> Output = ResourceDescriptorHeap[OutputIdx];
    BakeStep step = LoadStep();

    Texture2D<float> Source = ResourceDescriptorHeap[step.Source];
    uint sw, sh;
    Source.GetDimensions(sw, sh);
    float2 rectUV = (SampleWorldXZ(sid) - step.ImportMin) * step.ImportInvSize;
    float src = Source.SampleLevel(sampLinear, TerrainHeightUV(rectUV, float2(sw, sh)), 0) * step.ImportScale + step.ImportBias;

    uint3 at = OutTexel(sid);
    float prev = Output[at];
    Output[at] = Blend(prev, src, step.BlendMode, step.Opacity);
}

uint CompareHash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// ── CS_CompareReads: the height at a place through terrain_pages.hlsli, over the terrain and 3 m round it, for the CPU
// to compare with its own bilinear of the samples read back; every second place on a sample, where a vertex reads ──
[numthreads(64, 1, 1)]
void CS_CompareReads(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= CompareCount) return;
    TerrainPagesParams t = LoadTerrain(TerrainParamsIdx);
    uint h0 = CompareHash(dtid.x * 2u), h1 = CompareHash(dtid.x * 2u + 1u);
    float2 r = float2(h0 >> 8, h1 >> 8) / 16777216.0;
    float2 xz = t.Origin.xz - 3.0 + r * (t.Size + 6.0);
    if (dtid.x & 1u)
        xz = t.Origin.xz + round((xz - t.Origin.xz) / t.Spacing) * t.Spacing;

    RWStructuredBuffer<float4> o = ResourceDescriptorHeap[CompareUAVIdx];
    o[dtid.x] = float4(TerrainHeight01(t, xz), 0, xz);
}

// Sample the tileable noise LUT (R16G16_Float) at a given position.
// R and G hold two independent noise fields for per-octave decorrelation.
float sampleNoiseLUT(Texture2D<float2> lut, float2 p, uint octave)
{
    float2 s = lut.SampleLevel(sampLinear, p, 0);
    return (octave & 1) ? s.g : s.r;
}

// ── fBm accumulation with type-specific strategies ──
// Each noise type has a distinct accumulation to produce visually different terrain:
//   Simplex/Perlin: Standard fBm — smooth rolling hills
//   Ridged: Musgrave's ridged multifractal — sharp mountain ridges
//   Billow: Abs-folded fBm — puffy dome-like terrain
float fbmNoise(Texture2D<float2> lut, float2 p, uint type, uint octaves,
               float lacunarity, float persistence, float seed)
{
    // Seed-based initial rotation to make different seeds produce different patterns
    float seedAngle = seed * 1.9635;
    float cs = cos(seedAngle), sn = sin(seedAngle);
    float2x2 seedRot = float2x2(cs, -sn, sn, cs);
    p = mul(seedRot, p);

    // Per-octave UV transform helper
    #define OCTAVE_UV(i, freq) \
        float angle##i = 0.5 + (i) * 1.37; \
        float c##i = cos(angle##i), s##i = sin(angle##i); \
        float2 rp = float2(c##i * p.x - s##i * p.y, s##i * p.x + c##i * p.y) * (freq); \
        rp += float2((i) * 7.31, (i) * 11.17)

    float value = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    float maxAmp = 0.0;

    if (type <= 1) // Simplex / Perlin — standard fBm
    {
        for (uint i = 0; i < octaves && i < 12; i++)
        {
            OCTAVE_UV(i, freq);
            value += amp * sampleNoiseLUT(lut, rp, i);
            maxAmp += amp;
            amp *= persistence;
            freq *= lacunarity;
        }
        return value / maxAmp;
    }
    else if (type == 2) // Ridged multifractal
    {
        float weight = 1.0;
        const float offset = 1.0; // ridge offset
        const float gain = 2.0;   // sharpness

        for (uint i = 0; i < octaves && i < 12; i++)
        {
            OCTAVE_UV(i, freq);
            float n = sampleNoiseLUT(lut, rp, i);
            // Fold: create sharp ridges where noise crosses 0.5
            float signal = offset - abs(n * 2.0 - 1.0);
            signal *= signal; // sharpen ridges
            signal *= weight; // detail concentrates in valleys
            value += signal * amp;
            maxAmp += amp;
            // Next octave weight depends on current signal
            weight = saturate(signal * gain);
            amp *= persistence;
            freq *= lacunarity;
        }
        return value / maxAmp;
    }
    else // Billow (type == 3) — abs-folded gives dome shapes
    {
        for (uint i = 0; i < octaves && i < 12; i++)
        {
            OCTAVE_UV(i, freq);
            float n = sampleNoiseLUT(lut, rp, i);
            // Abs fold: creates rounded dome/pillow shapes
            float signal = abs(n * 2.0 - 1.0);
            value += amp * signal;
            maxAmp += amp;
            amp *= persistence;
            freq *= lacunarity;
        }
        return value / maxAmp;
    }

    #undef OCTAVE_UV
}

// ── CS_NoiseLayer: fractal noise in world metres (a pattern that stays where it is whatever the terrain's size) ──
[numthreads(8, 8, 1)]
void CS_NoiseLayer(uint3 dtid : SV_DispatchThreadID)
{
    int2 sid;
    if (!RegionSample(dtid, sid)) return;
    RWTexture2DArray<float> Output = ResourceDescriptorHeap[OutputIdx];
    BakeStep step = LoadStep();

    Texture2D<float2> NoiseLUT = ResourceDescriptorHeap[NoiseLUTIdx];

    float2 world = SampleWorldXZ(sid);
    float2 p = (world + step.Offset) * step.Frequency;

    float noise = fbmNoise(NoiseLUT, p, step.NoiseType, step.Octaves, step.Lacunarity, step.Persistence, (float)step.Seed);

    // ── Terracing post-process ──
    if (step.TerraceSteps > 0)
    {
        float steps = (float)step.TerraceSteps;
        float quantized = floor(noise * steps) / steps;
        float frac_part = frac(noise * steps);
        // Smooth-step between shelves for natural-looking ledges
        float smooth_frac = smoothstep(0.0, 1.0, frac_part) / steps;
        noise = lerp(quantized, quantized + smooth_frac, step.TerraceSmoothness);
    }

    float value = noise * step.Amplitude;

    // ── Spatial mask (radial falloff) ──
    if (step.MaskRadius > 0.0)
    {
        float2 delta = world - step.MaskCenter;
        float dist = length(delta);
        float mask = 1.0 - saturate(pow(dist / step.MaskRadius, step.MaskFalloff));
        value *= mask;
    }

    uint3 at = OutTexel(sid);
    float prev = Output[at];
    Output[at] = Blend(prev, value, step.BlendMode, step.Opacity);
}

// ═══════════════════════════════════════════════════════════════════════════
// CS_InfluenceLayer — Applies terrain influences (roads, rivers, buildings)
// ═══════════════════════════════════════════════════════════════════════════
//
// One run of HeightStamps over one page: the page's stamp list holds, in bake order, the descriptors whose bounds
// reach the page (TerrainBaker.BuildPageLists); this run's are entries ListStart .. ListStart + ListCount of it.

#include "terrain_stamp_common.hlsli"

// Per-stamp descriptor (matches C# HeightStampDescriptorGPU, 132 bytes)
struct HeightStampDescriptor
{
    float2 Center;             // terrain UV center (radial mode)
    float  Radius;             // UV-space inner radius
    float  Falloff;            // UV-space falloff width
    float  TargetHeight;       // normalized target height [0..1]
    uint   InvertShape;        // bit 0: the heightmap's relief digs instead of raising; bit 1: Outside (HeightStamp)
    uint   SplinePointOffset;  // offset into spline point buffer (0xFFFFFFFF = radial)
    uint   SplinePointCount;   // number of spline points (0 = radial, high bit = closed area)
    float  NoiseFreq;          // edge noise frequency (0 = disabled)
    float  NoiseAmp;           // edge noise amplitude in UV space
    uint   NoiseSeed;          // noise seed
    uint   HeightmapIdx;       // bindless SRV index (0 = no heightmap)
    float  HeightmapStrength;  // normalized strength (worldStrength / maxHeight)
    float  RotationSin;        // the heightmap's turn: UV delta → texture frame
    float  RotationCos;
    uint   BlendMode;          // 0=Set, 1=Add, 2=Max, 3=Lerp (as Set), 4=Min
    float2 HeightmapCenter;    // terrain UV of the heightmap's centre
    float2 HeightmapExtent;    // UV half-size of the heightmap along its turned axes
    uint   MaskPointOffset;    // a parent's closed spline masking a round stamp: its samples in the spline buffer
    uint   MaskPointCount;     // 0 = no mask
    float2 BoundsMin;          // terrain UV box outside which the weight is 0 (TerrainBaker.StampBounds)
    float2 BoundsMax;
    float2 RidgeAcross;        // ridge and furrow (HeightStamp.RidgeSpacing): unit vector across the ridges
    float  RidgeSpacing;       // UV between crests (0 = no ridges)
    float  RidgeHeight;        // normalized furrow-to-crest height
    float  RidgeBend;          // UV the ridges swing aside at the field's ends
    float2 RidgeAlong;         // the stamp's extent along the ridges: min, max of dot(uv, along)
};

// Ridge and furrow at a terrain UV, around 0 (the ridges' mean is the ground's): broad crowns and narrower furrows
// (sin^0.6, whose mean is 0.73), running along the field and swinging aside towards its ends in a reverse S as the
// plough team began to turn before the headland
float RidgeRelief(HeightStampDescriptor stamp, float2 uv)
{
    float2 along = float2(stamp.RidgeAcross.y, -stamp.RidgeAcross.x);
    float s = saturate((dot(uv, along) - stamp.RidgeAlong.x) / max(stamp.RidgeAlong.y - stamp.RidgeAlong.x, 1e-7));
    float bend = 2.0 * s - 1.0;
    float across = dot(uv, stamp.RidgeAcross) + stamp.RidgeBend * bend * bend * bend;
    float crown = pow(sin(3.14159265 * frac(across / stamp.RidgeSpacing)), 0.6);
    return stamp.RidgeHeight * (crown - 0.73);
}

[numthreads(8, 8, 1)]
void CS_InfluenceLayer(uint3 dtid : SV_DispatchThreadID)
{
    int2 sid;
    if (!RegionSample(dtid, sid)) return;
    RWTexture2DArray<float> Output = ResourceDescriptorHeap[OutputIdx];

    StructuredBuffer<HeightStampDescriptor> Stamps = ResourceDescriptorHeap[StampBufIdx];
    StructuredBuffer<uint> StampList = ResourceDescriptorHeap[StampListIdx];
    StructuredBuffer<StampSplinePoint> SplinePoints = ResourceDescriptorHeap[SplineBufIdx];

    uint3 at = OutTexel(sid);
    float2 uv = SampleTerrainUV(sid);
    float currentHeight = Output[at];
    bool modified = false;

    for (uint j = ListStart; j < ListStart + ListCount; j++)
    {
        HeightStampDescriptor stamp = Stamps[StampList[j]];
        // Out of the stamp's reach: skip it without walking its spline
        if (any(uv < stamp.BoundsMin) || any(uv > stamp.BoundsMax)) continue;

        // A closed spline of the stamp's own is a mask: its falloff lies inside the edge (insetFalloff), or the mask
        // is turned inside out (InvertShape bit 1, HeightStamp.Outside)
        float nearestH, nearestHalfW;
        float weight = EvaluateStampWeightEx(
            uv, stamp.Center, stamp.Radius, stamp.Falloff,
            stamp.SplinePointOffset, stamp.SplinePointCount,
            stamp.NoiseFreq, stamp.NoiseAmp, stamp.NoiseSeed,
            SplinePoints, nearestH, nearestHalfW, true, (stamp.InvertShape & 2u) != 0);

        if (weight <= 0) continue;

        // A round stamp under a closed spline is confined to it: the parent's outline, faded inward over the
        // stamp's own falloff (the mask's samples carry no width, so the outline is not grown)
        if (stamp.MaskPointCount != 0)
        {
            float maskH, maskHalfW;
            weight *= EvaluateStampWeight(
                uv, float2(0, 0), 0, stamp.Falloff,
                stamp.MaskPointOffset, stamp.MaskPointCount,
                stamp.NoiseFreq, stamp.NoiseAmp, stamp.NoiseSeed,
                SplinePoints, maskH, maskHalfW, true);
            if (weight <= 0) continue;
        }

        // Spline mode: use interpolated height from nearest spline point
        float targetH = (stamp.SplinePointCount > 0 && stamp.SplinePointOffset != 0xFFFFFFFF)
            ? nearestH : stamp.TargetHeight;

        // Heightmap: add spatially-varying height from texture
        if (stamp.HeightmapIdx != 0)
        {
            // Into the texture's frame: its centre, turned, [-extent, +extent] → [0, 1]. The extent is
            // Radius + Falloff for a radial stamp and the footprint of a spline stamp (TerrainBaker.FitHeightmapToSpline).
            float2 localDelta = uv - stamp.HeightmapCenter;
            float2 rotLocal = float2(
                localDelta.x * stamp.RotationCos + localDelta.y * stamp.RotationSin,
               -localDelta.x * stamp.RotationSin + localDelta.y * stamp.RotationCos
            );
            float2 hmUV = rotLocal / stamp.HeightmapExtent * 0.5 + 0.5;

            Texture2D<float> Heightmap = ResourceDescriptorHeap[stamp.HeightmapIdx];
            float relief = Heightmap.SampleLevel(sampLinear, hmUV, 0) * stamp.HeightmapStrength;

            // InvertShape turns the relief upside down: the texture digs (lake depths, craters)
            targetH += (stamp.InvertShape & 1u) ? -relief : relief;
        }

        // Ridge and furrow (HeightStamp.RidgeSpacing), faded towards the edge with the rest of the stamp
        if (stamp.RidgeSpacing > 0)
            targetH += RidgeRelief(stamp, uv);

        // Combine with the height built so far, then fade by the stamp weight
        float blended = targetH;                                             // Set
        if (stamp.BlendMode == 1)      blended = currentHeight + targetH;     // Add
        else if (stamp.BlendMode == 2) blended = max(currentHeight, targetH); // Max
        else if (stamp.BlendMode == 4) blended = min(currentHeight, targetH); // Min

        currentHeight += (blended - currentHeight) * weight;
        modified = true;
    }

    if (modified)
        Output[at] = currentHeight;
}
