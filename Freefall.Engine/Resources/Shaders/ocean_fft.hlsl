// ocean_fft.hlsl — LDS-based Stockham butterfly FFT + map assembly
// SM 6.6 bindless, push constants at b3

#pragma kernel CSHorizontalFFT
#pragma kernel CSVerticalFFT
#pragma kernel CSAssembleMaps
#pragma kernel CSClearTallies

#define PI 3.14159265358979323846
#define SIZE 512
#define LOG_SIZE 9

// The whitecap band's tallies, read back every frame (C# OceanFFT.ReadTallies): how squeezed its crests are (1 − J in
// SQUEEZE_BINS bins of SQUEEZE_STEP from SQUEEZE_FROM) and how much foam lies on it (FOAM_BINS bins over 0..1)
#define SQUEEZE_BINS 128
#define SQUEEZE_FROM -1.2
#define SQUEEZE_STEP 0.025
#define FOAM_BINS 16
#define TALLIES (SQUEEZE_BINS + FOAM_BINS)

cbuffer PushConstants : register(b3)
{
    uint InitialSpectrumIdx;   // RWTexture2DArray<float4> — H0
    uint SpectrumIdx;          // RWTexture2DArray<float4> — evolved spectrum
    uint SpectrumParamsIdx;    // (unused here, shared layout with spectrum)
    uint OceanConstantsIdx;    // StructuredBuffer<OceanConstants>
    uint DisplacementIdx;      // RWTexture2DArray<float4> — output displacement + foam
    uint SlopeIdx;             // RWTexture2DArray<float4> — output slope moments (sx, sz, sx², sz²)
    uint WhitecapTalliesIdx;   // RWStructuredBuffer<uint> — the whitecap band's tallies (TALLIES)
};

struct OceanConstants
{
    float FrameTime;
    float DeltaTime;
    float Gravity;
    float Depth;
    float RepeatTime;
    float LowCutoff;
    float HighCutoff;
    uint N;
    uint Seed;
    float2 Lambda;
    float FoamBias;
    float FoamDecayRate;
    float FoamThreshold;
    float FoamAdd;
    uint NumBands;
    uint LengthScalesSRV;
    uint NoWhitecapBands;  // bit per band whose crests make no foam (a swell from afar); 0 = every band foams
    uint WhitecapBand;     // the band whose whitecaps follow the wind (+1), tallied; 0 = none (FoamBias for every band)
    float CapSqueeze;      // its crests whiten past this squeeze (1 − J): the level a share of them reach (C# OceanFFT)
    float CapSpread;       // the spread of its squeeze; 0 = not tallied yet (FoamBias then)
    float _pad2;
};

// ═══════════════════════════════════════════════════════════════════════
// Complex arithmetic
// ═══════════════════════════════════════════════════════════════════════

float2 ComplexMult(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// ═══════════════════════════════════════════════════════════════════════
// LDS-based Stockham FFT — entire row in one dispatch
// Two ping-pong buffers in groupshared memory
// ═══════════════════════════════════════════════════════════════════════

groupshared float4 fftGroupBuffer[2][SIZE];

void ButterflyValues(uint step, uint index, out uint2 indices, out float2 twiddle)
{
    const float twoPi = 6.28318530718;
    uint b = SIZE >> (step + 1);
    uint w = b * (index / b);
    uint i = (w + index) % SIZE;
    sincos(-twoPi / SIZE * w, twiddle.y, twiddle.x);

    // Negate imaginary part for inverse FFT
    twiddle.y = -twiddle.y;
    indices = uint2(i, i + b);
}

float4 FFT(uint threadIndex, float4 input)
{
    fftGroupBuffer[0][threadIndex] = input;
    GroupMemoryBarrierWithGroupSync();
    bool flag = false;

    [unroll]
    for (uint step = 0; step < LOG_SIZE; ++step)
    {
        uint2 inputIndices;
        float2 twiddle;
        ButterflyValues(step, threadIndex, inputIndices, twiddle);

        float4 v = fftGroupBuffer[flag][inputIndices.y];
        fftGroupBuffer[!flag][threadIndex] = fftGroupBuffer[flag][inputIndices.x]
            + float4(ComplexMult(twiddle, v.xy), ComplexMult(twiddle, v.zw));

        flag = !flag;
        GroupMemoryBarrierWithGroupSync();
    }

    return fftGroupBuffer[flag][threadIndex];
}

// ═══════════════════════════════════════════════════════════════════════
// CSHorizontalFFT — process rows (all 8 slices per dispatch)
// ═══════════════════════════════════════════════════════════════════════

RWTexture2DArray<float4> _FourierTarget;

[numthreads(SIZE, 1, 1)]
void CSHorizontalFFT(uint3 id : SV_DispatchThreadID)
{
    RWTexture2DArray<float4> fourierTex = ResourceDescriptorHeap[SpectrumIdx];
    StructuredBuffer<OceanConstants> constants = ResourceDescriptorHeap[OceanConstantsIdx];
    uint numSlices = constants[0].NumBands * 2;

    [loop]
    for (uint i = 0; i < numSlices; ++i)
    {
        fourierTex[uint3(id.xy, i)] = FFT(id.x, fourierTex[uint3(id.xy, i)]);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CSVerticalFFT — process columns (transpose read, all 8 slices)
// ═══════════════════════════════════════════════════════════════════════

[numthreads(SIZE, 1, 1)]
void CSVerticalFFT(uint3 id : SV_DispatchThreadID)
{
    RWTexture2DArray<float4> fourierTex = ResourceDescriptorHeap[SpectrumIdx];
    StructuredBuffer<OceanConstants> constants = ResourceDescriptorHeap[OceanConstantsIdx];
    uint numSlices = constants[0].NumBands * 2;

    [loop]
    for (uint i = 0; i < numSlices; ++i)
    {
        fourierTex[uint3(id.yx, i)] = FFT(id.x, fourierTex[uint3(id.yx, i)]);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CSAssembleMaps — post-FFT: permute, pack displacement+foam, slopes
// ═══════════════════════════════════════════════════════════════════════

float4 Permute(float4 data, float3 id)
{
    return data * (1.0f - 2.0f * ((id.x + id.y) % 2));
}

groupshared uint gsTallies[TALLIES];

// Empties the tallies before CSAssembleMaps counts this frame's
[numthreads(64, 1, 1)]
void CSClearTallies(uint3 id : SV_DispatchThreadID)
{
    RWStructuredBuffer<uint> tallies = ResourceDescriptorHeap[WhitecapTalliesIdx];
    if (id.x < TALLIES) tallies[id.x] = 0;
}

[numthreads(8, 8, 1)]
void CSAssembleMaps(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    RWTexture2DArray<float4> spectrumTex = ResourceDescriptorHeap[SpectrumIdx];
    RWTexture2DArray<float4> displacementTex = ResourceDescriptorHeap[DisplacementIdx];
    RWTexture2DArray<float4> slopeTex = ResourceDescriptorHeap[SlopeIdx];
    StructuredBuffer<OceanConstants> constants = ResourceDescriptorHeap[OceanConstantsIdx];
    OceanConstants oc = constants[0];

    // The whitecap band's tallies are counted in the group first, then added to the buffer
    bool tally = oc.WhitecapBand != 0 && WhitecapTalliesIdx != 0;
    if (tally)
    {
        for (uint t = gi; t < TALLIES; t += 64) gsTallies[t] = 0;
        GroupMemoryBarrierWithGroupSync();
    }

    for (int i = 0; i < (int)oc.NumBands; ++i)
    {
        // Sign-correct the FFT output (center-shifted spectrum)
        float4 htildeDisplacement = Permute(spectrumTex[uint3(id.xy, i * 2)], id);
        float4 htildeSlope = Permute(spectrumTex[uint3(id.xy, i * 2 + 1)], id);

        float2 dxdz = htildeDisplacement.rg;
        float2 dydxz = htildeDisplacement.ba;
        float2 dyxdyz = htildeSlope.rg;
        float2 dxxdzz = htildeSlope.ba;

        // Jacobian for foam detection
        float jacobian = (1.0f + oc.Lambda.x * dxxdzz.x) * (1.0f + oc.Lambda.y * dxxdzz.y)
                        - oc.Lambda.x * oc.Lambda.y * dydxz.y * dydxz.y;

        // Final displacement with choppiness (lambda)
        float3 displacement = float3(oc.Lambda.x * dxdz.x, dydxz.x, oc.Lambda.y * dxdz.y);

        // Slopes of the displaced surface: the choppy displacement squeezes a crest together (1 + λ·∂D/∂x < 1), so the
        // same rise over a shorter run is steeper, and stretches a trough flatter. Floored where the crest folds over.
        // Until 2026-10-18 divided by 1 + |λ·∂D/∂x|, which flattened crests as well as troughs.
        float2 slopes = dyxdyz.xy / max(1.0 + dxxdzz * oc.Lambda, 0.2);

        // Temporal foam accumulation
        float foam = displacementTex[uint3(id.xy, i)].a;
        // Rates are authored per frame at 60 fps; scale by real time so foam lifetime is framerate independent
        float foamStep = oc.DeltaTime * 60.0;
        foam *= exp(-oc.FoamDecayRate * foamStep);
        foam = saturate(foam);

        // Whitecaps where the crest is squeezed. A band that makes no whitecaps of its own (NoWhitecapBands) stores how
        // squeezed its crests are right now instead (1 - J): ocean.fx lights those crests up inside the whitecaps of the
        // band that does, as their structure
        float squeeze = 1.0 - jacobian;
        if (((oc.NoWhitecapBands >> i) & 1u) != 0)
        {
            foam = saturate(squeeze);
        }
        else if ((uint)i + 1 == oc.WhitecapBand && oc.CapSpread > 0.0)
        {
            // The wind's whitecaps (since 2026-11-23): the crests squeezed past the level that the share the wind breaks
            // reach (CapSqueeze, from this band's own tallies), more the further past, in the band's own spread. A calm
            // sea's whitecaps are as white as a storm's, only fewer and smaller
            float past = squeeze - oc.CapSqueeze;
            if (past > 0.0)
                foam += oc.FoamAdd * (past / oc.CapSpread) * foamStep;
        }
        else
        {
            // Authored seas: J under FoamBias - FoamThreshold, more the deeper
            float biasedJacobian = max(0.0f, -(jacobian - oc.FoamBias));
            if (biasedJacobian > oc.FoamThreshold)
                foam += oc.FoamAdd * biasedJacobian * foamStep;
        }

        if (tally && (uint)i + 1 == oc.WhitecapBand)
        {
            int bin = clamp((int)floor((squeeze - SQUEEZE_FROM) / SQUEEZE_STEP), 0, SQUEEZE_BINS - 1);
            InterlockedAdd(gsTallies[bin], 1);
            InterlockedAdd(gsTallies[SQUEEZE_BINS + min((uint)(saturate(foam) * FOAM_BINS), FOAM_BINS - 1)], 1);
        }

        // Write output
        displacementTex[uint3(id.xy, i)] = float4(displacement, foam);
        // Slope moments: the mip chain averages all four, so a texel of any mip holds the mean slope and the mean
        // squared slope of the waves it covers: their variance (E[s²] − E[s]²) is the roughness a pixel sees there
        slopeTex[uint3(id.xy, i)] = float4(slopes, slopes * slopes);
    }

    if (tally)
    {
        GroupMemoryBarrierWithGroupSync();
        RWStructuredBuffer<uint> tallies = ResourceDescriptorHeap[WhitecapTalliesIdx];
        for (uint t = gi; t < TALLIES; t += 64)
            if (gsTallies[t] != 0) InterlockedAdd(tallies[t], gsTallies[t]);
    }
}
