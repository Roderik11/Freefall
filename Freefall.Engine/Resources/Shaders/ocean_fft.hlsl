// ocean_fft.hlsl — LDS-based Stockham butterfly FFT + map assembly
// SM 6.6 bindless, push constants at b3

#pragma kernel CSHorizontalFFT
#pragma kernel CSVerticalFFT
#pragma kernel CSAssembleMaps

#define PI 3.14159265358979323846
#define SIZE 512
#define LOG_SIZE 9

cbuffer PushConstants : register(b3)
{
    uint InitialSpectrumIdx;   // RWTexture2DArray<float4> — H0
    uint SpectrumIdx;          // RWTexture2DArray<float4> — evolved spectrum
    uint SpectrumParamsIdx;    // (unused here, shared layout with spectrum)
    uint OceanConstantsIdx;    // StructuredBuffer<OceanConstants>
    uint DisplacementIdx;      // RWTexture2DArray<float4> — output displacement + foam
    uint SlopeIdx;             // RWTexture2DArray<float2> — output slopes
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
    uint _pad0, _pad1;
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

[numthreads(8, 8, 1)]
void CSAssembleMaps(uint3 id : SV_DispatchThreadID)
{
    RWTexture2DArray<float4> spectrumTex = ResourceDescriptorHeap[SpectrumIdx];
    RWTexture2DArray<float4> displacementTex = ResourceDescriptorHeap[DisplacementIdx];
    RWTexture2DArray<float2> slopeTex = ResourceDescriptorHeap[SlopeIdx];
    StructuredBuffer<OceanConstants> constants = ResourceDescriptorHeap[OceanConstantsIdx];
    OceanConstants oc = constants[0];

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

        // Slopes normalized by displacement derivatives
        float2 slopes = dyxdyz.xy / (1 + abs(dxxdzz * oc.Lambda));

        // Temporal foam accumulation
        float foam = displacementTex[uint3(id.xy, i)].a;
        foam *= exp(-oc.FoamDecayRate);
        foam = saturate(foam);

        float biasedJacobian = max(0.0f, -(jacobian - oc.FoamBias));
        if (biasedJacobian > oc.FoamThreshold)
            foam += oc.FoamAdd * biasedJacobian;

        // Write output
        displacementTex[uint3(id.xy, i)] = float4(displacement, foam);
        slopeTex[uint3(id.xy, i)] = slopes;
    }
}
