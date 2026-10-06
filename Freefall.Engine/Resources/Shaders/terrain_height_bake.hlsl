// terrain_height_bake.hlsl — Composites the terrain's height stamps into a final R16_UNorm heightmap.
//
// TerrainBaker walks the height stamps in priority order and dispatches one kernel per step:
//   CS_InfluenceLayer  a run of HeightStamps (flatten / heightmap pattern, radial or spline)
//   CS_ImportLayer     a global HeightStamp with a heightmap (whole-terrain import)
//   CS_NoiseLayer      a HeightNoiseStamp
//   CS_ErosionFilter   a HeightErosionStamp
// CS_Clear zeroes the heightmap first.
//
// Uses push constants (b3) matching engine convention.

#pragma kernel CS_ImportLayer
#pragma kernel CS_Clear
#pragma kernel CS_NoiseLayer
#pragma kernel CS_ErosionFilter
#pragma kernel CS_InfluenceLayer

// Push constants (root parameter 0, register b3) — bindless indices + params
cbuffer PushConstants : register(b3)
{
    uint SourceIdx;       // slot 0 — SRV: source heightmap texture (import) / current heightmap (erosion)
    uint OutputIdx;       // slot 1 — UAV: output RWTexture2D<float>
    uint StampBufIdx;     // slot 2 — SRV: StructuredBuffer<HeightStampDescriptor> (influence)
    uint BlendMode;       // slot 3 — 0=Set, 1=Add, 2=Max, 3=Lerp, 4=Min
    float Opacity;        // slot 4 — step opacity [0..1]
    uint StampCount;      // slot 5 — number of stamp descriptors in this run (influence)
    float BrushRadius;    // slot 6 — reused per kernel, see the aliases below
    float BrushFalloff;   // slot 7 — reused per kernel
    float BrushTargetHeight; // slot 8 — import: height bias (normalized); erosion: assumed slope amount

    // Slots 9..19 are unused (they carried the paint brush raycast); kept so the slots below stay put.
    float _unused9;
    float _unused10;
    float _unused11;
    float _unused12;
    float _unused13;
    float _unused14;
    float _unused15;
    float _unused16;
    float _unused17;
    float _unused18;
    float _unused19;
    uint StampStart;        // slot 20 — first stamp descriptor of this run (influence)

    // ── Noise layer params (CS_NoiseLayer) ──
    uint NoiseType;       // slot 21 — 0=Simplex, 1=Perlin, 2=Ridged, 3=Billow
    uint Octaves;         // slot 22 — number of fBm octaves
    float Frequency;      // slot 23 — base frequency
    float Amplitude;      // slot 24 — output amplitude scale
    float Lacunarity;     // slot 25 — per-octave frequency multiplier
    float Persistence;    // slot 26 — per-octave amplitude decay
    float OffsetX;        // slot 27 — world-space noise offset X
    float OffsetY;        // slot 28 — world-space noise offset Y
    uint NoiseSeed;       // slot 29 — seed for noise permutation
    uint ErosionMode;     // slot 30 — 0=Hydraulic, 1=Thermal, 2=Both (reused as NoiseLUTIdx for CS_NoiseLayer)

    // ── Noise terrace + mask params (CS_NoiseLayer only) ──
    uint TerraceSteps;    // slot 31 — 0=disabled, N = number of terrace shelves
    float TerraceSmoothness; // slot 32 — 0=sharp, 1=fully rounded transitions
    float MaskCenterX;    // slot 33 — UV-space mask center X
    float MaskCenterY;    // slot 34 — UV-space mask center Y
    float MaskRadius;     // slot 35 — UV-space radius, 0=disabled (full terrain)
    float MaskFalloff;    // slot 36 — mask edge falloff exponent

    // ── Erosion params (CS_ErosionStep, reuses slots 23-28 since never concurrent) ──
    // RainRate      → slot 23 (Frequency)
    // SedimentCap   → slot 24 (Amplitude)
    // DepositionRate→ slot 25 (Lacunarity)
    // DissolutionRate→slot 26 (Persistence)
    // Evaporation   → slot 27 (OffsetX)
    // TalusAngle    → slot 28 (OffsetY)
    // ThermalRate   → slot 6  (BrushRadius)
    // Erosion aux UAVs: WaterIdx=slot 2 (StampBufIdx), SedimentIdx=slot 5 (StampCount)
    // PingPong source SRV: SourceIdx=slot 0
};

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

// ── CS_Clear: Zero the output heightmap ──
[numthreads(8, 8, 1)]
void CS_Clear(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;
    Output[dtid.xy] = 0;
}

// ── CS_ImportLayer: Blend a heightmap stretched over the whole terrain ──
// value = source * Amplitude + BrushTargetHeight (scale and bias, both normalized to MaxHeight)
[numthreads(8, 8, 1)]
void CS_ImportLayer(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    Texture2D<float> Source = ResourceDescriptorHeap[SourceIdx];
    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);
    float src = Source.SampleLevel(sampLinear, uv, 0) * Amplitude + BrushTargetHeight;

    float prev = Output[dtid.xy];
    Output[dtid.xy] = Blend(prev, src, BlendMode, Opacity);
}

#define NoiseLUTIdx ErosionMode

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

[numthreads(8, 8, 1)]
void CS_NoiseLayer(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    Texture2D<float2> NoiseLUT = ResourceDescriptorHeap[NoiseLUTIdx];

    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);
    float2 p = (uv + float2(OffsetX, OffsetY)) * Frequency;

    float noise = fbmNoise(NoiseLUT, p, NoiseType, Octaves, Lacunarity, Persistence, (float)NoiseSeed);

    // ── Terracing post-process ──
    if (TerraceSteps > 0)
    {
        float steps = (float)TerraceSteps;
        float quantized = floor(noise * steps) / steps;
        float frac_part = frac(noise * steps);
        // Smooth-step between shelves for natural-looking ledges
        float smooth_frac = smoothstep(0.0, 1.0, frac_part) / steps;
        noise = lerp(quantized, quantized + smooth_frac, TerraceSmoothness);
    }

    float value = noise * Amplitude;

    // ── Spatial mask (radial falloff) ──
    if (MaskRadius > 0.0)
    {
        float2 delta = uv - float2(MaskCenterX, MaskCenterY);
        float dist = length(delta);
        float mask = 1.0 - saturate(pow(dist / MaskRadius, MaskFalloff));
        value *= mask;
    }

    float prev = Output[dtid.xy];
    Output[dtid.xy] = Blend(prev, value, BlendMode, Opacity);
}

// ═══════════════════════════════════════════════════════════════════════════
// CS_ErosionFilter — Single-pass noise-based erosion filter
// ═══════════════════════════════════════════════════════════════════════════
//
// Ported from "Advanced Terrain Erosion Filter" by Rune Skovbo Johansen
// (https://www.shadertoy.com/view/wXcfWn) — MPL-2.0 licensed.
//
// Uses Phacelle Noise to generate gradient-aligned gullies that produce
// crisp branching patterns in a single dispatch (no iterative simulation).
//
// Slot reuse (erosion filter kernels, never concurrent with noise/stamps):
//   SourceIdx          (0)  → SRV: current baked heightmap
//   OutputIdx          (1)  → UAV: output heightmap
//   BrushRadius        (6)  → EF Scale
//   BrushFalloff       (7)  → EF AssumedSlopeValue
//   BrushTargetHeight  (8)  → EF AssumedSlopeAmount
//   Frequency          (23) → EF Strength
//   Amplitude          (24) → EF GullyWeight
//   Lacunarity         (25) → EF Detail
//   Persistence        (26) → EF Lacunarity
//   OffsetX            (27) → EF Gain
//   OffsetY            (28) → EF CellScale
//   NoiseSeed          (29) → EF Octaves (uint)
//   ErosionMode        (30) → EF Normalization (asfloat)
//   TerraceSteps       (31) → EF RidgeRounding (asfloat)
//   TerraceSmoothness  (32) → EF CreaseRounding
//   MaskCenterX        (33) → EF RoundingInputMult
//   MaskCenterY        (34) → EF RoundingOctaveMult
//   MaskRadius         (35) → EF OnsetInput
//   MaskFalloff        (36) → EF OnsetOctave

// Aliases for readability
#define EFScale           BrushRadius
#define EFAssumedVal      BrushFalloff
#define EFAssumedAmt      BrushTargetHeight
#define EFStrength        Frequency
#define EFGullyWeight     Amplitude
#define EFDetail          Lacunarity
#define EFLacunarity      Persistence
#define EFGain            OffsetX
#define EFCellScale       OffsetY
#define EFOctaves         NoiseSeed
#define EFNormalization   asfloat(ErosionMode)
#define EFRidgeRounding   asfloat(TerraceSteps)
#define EFCreaseRounding  TerraceSmoothness
#define EFRoundInputMul   MaskCenterX
#define EFRoundOctMul     MaskCenterY
#define EFOnsetInput      MaskRadius
#define EFOnsetOctave     MaskFalloff

#define EF_TAU 6.28318530717959

// ── Hash function (algebraic, no texture lookups) ──
float2 ef_hash(float2 x) {
    const float2 k = float2(0.3183099, 0.3678794);
    x = x * k + k.yx;
    return -1.0 + 2.0 * frac(16.0 * k * frac(x.x * x.y * (x.x + x.y)));
}

// ── Phacelle Noise: directional stripe noise aligned with input vector ──
// Produces cosine/sine wave pairs blended across Worley-like cells.
// Copyright (c) 2025 Rune Skovbo Johansen — MPL-2.0
float4 EF_PhacelleNoise(float2 p, float2 normDir, float freq, float offset, float normalization) {
    float2 sideDir = normDir.yx * float2(-1.0, 1.0) * freq * EF_TAU;
    offset *= EF_TAU;

    float2 pInt = floor(p);
    float2 pFrac = frac(p);
    float2 phaseDir = 0;
    float weightSum = 0;

    [unroll]
    for (int i = -1; i <= 2; i++) {
        [unroll]
        for (int j = -1; j <= 2; j++) {
            float2 gridOffset = float2(i, j);
            float2 gridPoint = pInt + gridOffset;
            float2 randomOffset = ef_hash(gridPoint) * 0.5;
            float2 vectorFromCellPoint = pFrac - gridOffset - randomOffset;

            float sqrDist = dot(vectorFromCellPoint, vectorFromCellPoint);
            float weight = exp(-sqrDist * 2.0);
            weight = max(0.0, weight - 0.01111);
            weightSum += weight;

            float waveInput = dot(vectorFromCellPoint, sideDir) + offset;
            phaseDir += float2(cos(waveInput), sin(waveInput)) * weight;
        }
    }

    float2 interpolated = phaseDir / weightSum;
    float magnitude = sqrt(dot(interpolated, interpolated));
    magnitude = max(1.0 - normalization, magnitude);
    return float4(interpolated / magnitude, sideDir);
}

// ── Helper functions ──

float ef_pow_inv(float t, float power) {
    return 1.0 - pow(1.0 - saturate(t), power);
}

float ef_ease_out(float t) {
    float v = 1.0 - saturate(t);
    return 1.0 - v * v;
}

float ef_smooth_start(float t, float smoothing) {
    if (t >= smoothing)
        return t - 0.5 * smoothing;
    return 0.5 * t * t / smoothing;
}

float2 ef_safe_normalize(float2 n) {
    float l = length(n);
    return (abs(l) > 1e-10) ? (n / l) : n;
}

// ── Advanced Terrain Erosion Filter ──
// Copyright (c) 2025 Rune Skovbo Johansen — MPL-2.0
//
// Returns float4(heightDelta, slopeDelta.xy, magnitude).
float4 EF_ErosionFilter(
    float2 p, float3 heightAndSlope, float fadeTarget,
    float strength, float gullyWeight, float detail,
    float4 rounding, float2 onset, float2 assumedSlope,
    float scale, uint octaves, float lacunarity,
    float gain, float cellScale, float normalization
) {
    strength *= scale;
    fadeTarget = clamp(fadeTarget, -1.0, 1.0);

    float3 inputHeightAndSlope = heightAndSlope;
    float freq = 1.0 / (scale * cellScale);
    float slopeLength = max(length(heightAndSlope.yz), 1e-10);
    float magnitude = 0.0;
    float roundingMult = 1.0;

    float roundingForInput = lerp(rounding.y, rounding.x, saturate(fadeTarget + 0.5)) * rounding.z;
    float combiMask = ef_ease_out(ef_smooth_start(slopeLength * onset.x, roundingForInput * onset.x));

    // Gully slope: mix of actual slope and assumed slope
    float2 gullySlope = lerp(heightAndSlope.yz,
        heightAndSlope.yz / slopeLength * assumedSlope.x, assumedSlope.y);

    for (uint i = 0; i < octaves && i < 8; i++) {
        float4 phacelle = EF_PhacelleNoise(p * freq, ef_safe_normalize(gullySlope),
            cellScale, 0.25, normalization);
        phacelle.zw *= -freq;
        float sloping = abs(phacelle.y);

        // Add normalized slope for gully direction (straight gullies technique)
        gullySlope += sign(phacelle.y) * phacelle.zw * strength * gullyWeight;

        // Gullies: height offset (x) and derivative (yz)
        float3 gullies = float3(phacelle.x, phacelle.y * phacelle.zw);
        // Fade towards fadeTarget based on combiMask
        float3 fadedGullies = lerp(float3(fadeTarget, 0, 0), gullies * gullyWeight, combiMask);
        heightAndSlope += fadedGullies * strength;
        magnitude += strength;

        // Update fadeTarget for next octave (stacked fading)
        fadeTarget = fadedGullies.x;

        // Update mask with this octave's ridge/crease contribution
        float roundingForOctave = lerp(rounding.y, rounding.x,
            saturate(phacelle.x + 0.5)) * roundingMult;
        float newMask = ef_ease_out(ef_smooth_start(sloping * onset.y,
            roundingForOctave * onset.y));
        combiMask = ef_pow_inv(combiMask, detail) * newMask;

        // Prepare next octave
        strength *= gain;
        freq *= lacunarity;
        roundingMult *= rounding.w;
    }

    float3 heightAndSlopeDelta = heightAndSlope - inputHeightAndSlope;
    return float4(heightAndSlopeDelta, magnitude);
}

// ── CS_ErosionFilter: Single-dispatch erosion filter applied to baked heightmap ──
[numthreads(8, 8, 1)]
void CS_ErosionFilter(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    Texture2D<float> Source = ResourceDescriptorHeap[SourceIdx];
    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);
    float2 texel = 1.0 / float2(w, h);

    // Sample height and compute gradient via finite differences
    float hC = Source.SampleLevel(sampLinear, uv, 0);
    float hL = Source.SampleLevel(sampLinear, uv - float2(texel.x, 0), 0);
    float hR = Source.SampleLevel(sampLinear, uv + float2(texel.x, 0), 0);
    float hD = Source.SampleLevel(sampLinear, uv - float2(0, texel.y), 0);
    float hU = Source.SampleLevel(sampLinear, uv + float2(0, texel.y), 0);
    float2 gradient = float2(hR - hL, hU - hD) * 0.5 / texel;

    // heightAndSlope: (height, dh/dx, dh/dy)
    float3 heightAndSlope = float3(hC, gradient);

    // Fade target: map height [0,1] → [-1,1] (valleys→peaks)
    float fadeTarget = clamp(hC * 2.0 - 1.0, -1.0, 1.0);

    // Rounding: (ridgeRounding, creaseRounding, inputMult, octaveMult)
    float4 rounding = float4(EFRidgeRounding, EFCreaseRounding, EFRoundInputMul, EFRoundOctMul);
    float2 onset = float2(EFOnsetInput, EFOnsetOctave);
    float2 assumedSlope = float2(EFAssumedVal, EFAssumedAmt);

    // Run the erosion filter
    float4 erosion = EF_ErosionFilter(
        uv, heightAndSlope, fadeTarget,
        EFStrength, EFGullyWeight, EFDetail,
        rounding, onset, assumedSlope,
        EFScale, EFOctaves, EFLacunarity,
        EFGain, EFCellScale, EFNormalization
    );

    // erosion.x = height delta, erosion.w = magnitude
    // Offset to preserve peaks/valleys: raise valleys, lower peaks
    float offset = -fadeTarget * erosion.w;
    float eroded = hC + erosion.x + offset;

    float prev = Output[dtid.xy];
    Output[dtid.xy] = Blend(prev, eroded, BlendMode, Opacity);
}

// ═══════════════════════════════════════════════════════════════════════════
// CS_InfluenceLayer — Applies terrain influences (roads, rivers, buildings)
// ═══════════════════════════════════════════════════════════════════════════
//
// Push constant slot reuse (influence kernel, never concurrent with noise/erosion):
//   OutputIdx     (1)  → UAV: output heightmap
//   StampBufIdx   (2)  → SRV: StructuredBuffer<HeightStampDescriptor> (every run of this bake)
//   StampCount    (5)  → number of descriptors in this run
//   StampStart    (20) → first descriptor of this run
//   BrushRadius   (6)  → SRV: StructuredBuffer<StampSplinePoint> (spline data)
//   NOTE: BrushRadius is repurposed as a uint (bindless index) via asuint/asfloat.
//         The kernel reads it as SplineBufIdx below.

// Alias push constant slots for readability
#define StampDescBufIdx  StampBufIdx
#define StampDescCount   StampCount
#define SplineBufIdx     asuint(BrushRadius)

#include "terrain_stamp_common.hlsli"

// Per-stamp descriptor (matches C# HeightStampDescriptorGPU, 64 bytes)
struct HeightStampDescriptor
{
    float2 Center;             // terrain UV center (radial mode)
    float  Radius;             // UV-space inner radius
    float  Falloff;            // UV-space falloff width
    float  TargetHeight;       // normalized target height [0..1]
    uint   InvertShape;        // 0 = flatten toward, 1 = push away
    uint   SplinePointOffset;  // offset into spline point buffer (0xFFFFFFFF = radial)
    uint   SplinePointCount;   // number of spline points (0 = radial, high bit = closed area)
    float  NoiseFreq;          // edge noise frequency (0 = disabled)
    float  NoiseAmp;           // edge noise amplitude in UV space
    uint   NoiseSeed;          // noise seed
    uint   HeightmapIdx;       // bindless SRV index (0 = no heightmap)
    float  HeightmapStrength;  // normalized strength (worldStrength / maxHeight)
    float  RotationSin;        // sin(entity Y rotation)
    float  RotationCos;        // cos(entity Y rotation)
    uint   BlendMode;          // 0=Set, 1=Add, 2=Max, 3=Lerp (as Set), 4=Min
};

[numthreads(8, 8, 1)]
void CS_InfluenceLayer(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    StructuredBuffer<HeightStampDescriptor> Stamps = ResourceDescriptorHeap[StampDescBufIdx];
    StructuredBuffer<StampSplinePoint> SplinePoints = ResourceDescriptorHeap[SplineBufIdx];

    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);
    float currentHeight = Output[dtid.xy];
    bool modified = false;

    for (uint i = StampStart; i < StampStart + StampDescCount; i++)
    {
        HeightStampDescriptor stamp = Stamps[i];

        float nearestH, nearestHalfW;
        float weight = EvaluateStampWeight(
            uv, stamp.Center, stamp.Radius, stamp.Falloff,
            stamp.SplinePointOffset, stamp.SplinePointCount,
            stamp.NoiseFreq, stamp.NoiseAmp, stamp.NoiseSeed,
            SplinePoints, nearestH, nearestHalfW);

        if (weight <= 0) continue;

        // Spline mode: use interpolated height from nearest spline point
        float targetH = (stamp.SplinePointCount > 0 && stamp.SplinePointOffset != 0xFFFFFFFF)
            ? nearestH : stamp.TargetHeight;

        // Heightmap: add spatially-varying height from texture
        if (stamp.HeightmapIdx != 0)
        {
            // Compute local UV within stamp footprint
            float2 localDelta = uv - stamp.Center;

            // Apply inverse rotation
            float2 rotLocal = float2(
                localDelta.x * stamp.RotationCos + localDelta.y * stamp.RotationSin,
               -localDelta.x * stamp.RotationSin + localDelta.y * stamp.RotationCos
            );

            // Map from [-extent, +extent] to [0, 1] for heightmap sampling
            float extent = stamp.Radius + stamp.Falloff;
            float2 hmUV = rotLocal / extent * 0.5 + 0.5;

            Texture2D<float> Heightmap = ResourceDescriptorHeap[stamp.HeightmapIdx];
            float hmValue = Heightmap.SampleLevel(sampLinear, hmUV, 0);
            targetH += hmValue * stamp.HeightmapStrength;
        }

        // Combine with the height built so far, then fade by the stamp weight
        float blended = targetH;                                             // Set
        if (stamp.BlendMode == 1)      blended = currentHeight + targetH;     // Add
        else if (stamp.BlendMode == 2) blended = max(currentHeight, targetH); // Max
        else if (stamp.BlendMode == 4) blended = min(currentHeight, targetH); // Min

        // Optionally invert the displacement
        float delta = blended - currentHeight;
        if (stamp.InvertShape) delta = -delta;
        currentHeight += delta * weight;
        modified = true;
    }

    if (modified)
        Output[dtid.xy] = currentHeight;
}
