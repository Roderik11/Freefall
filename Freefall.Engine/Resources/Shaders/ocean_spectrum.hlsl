// ocean_spectrum.hlsl — JONSWAP spectrum initialization + time evolution
// SM 6.6 bindless, push constants at b3

#pragma kernel CSInitSpectrum
#pragma kernel CSPackConjugate
#pragma kernel CSEvolveSpectrum

#define PI 3.14159265358979323846

cbuffer PushConstants : register(b3)
{
    uint InitialSpectrumIdx;   // RWTexture2DArray<float4> — H0 packed (rg=H0, ba=H0conj)
    uint SpectrumIdx;          // RWTexture2DArray<float4> — evolved displacement+slope spectra
    uint SpectrumParamsIdx;    // StructuredBuffer<SpectrumParameters> — 8 entries (dual per band)
    uint OceanConstantsIdx;    // StructuredBuffer<OceanConstants> — global params
};

struct SpectrumParameters
{
    float scale;
    float angle;
    float spreadBlend;
    float swell;
    float alpha;
    float peakOmega;
    float gamma;
    float shortWavesFade;
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
    uint LengthScalesSRV;  // bindless SRV to uint[] buffer
    uint _pad0, _pad1;
};

// ═══════════════════════════════════════════════════════════════════════
// Complex arithmetic
// ═══════════════════════════════════════════════════════════════════════

float2 ComplexMult(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

float2 EulerFormula(float x)
{
    return float2(cos(x), sin(x));
}

// ═══════════════════════════════════════════════════════════════════════
// Hashing & random
// ═══════════════════════════════════════════════════════════════════════

float hash(uint n)
{
    n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 0x789221U) + 0x1376312589U;
    return float(n & uint(0x7fffffffU)) / float(0x7fffffff);
}

float2 UniformToGaussian(float u1, float u2)
{
    float R = sqrt(-2.0f * log(u1));
    float theta = 2.0f * PI * u2;
    return float2(R * cos(theta), R * sin(theta));
}

// ═══════════════════════════════════════════════════════════════════════
// Dispersion
// ═══════════════════════════════════════════════════════════════════════

float Dispersion(float kMag, float gravity, float depth)
{
    return sqrt(gravity * kMag * tanh(min(kMag * depth, 20)));
}

float DispersionDerivative(float kMag, float gravity, float depth)
{
    float th = tanh(min(kMag * depth, 20));
    float ch = cosh(kMag * depth);
    return gravity * (depth * kMag / ch / ch + th) / Dispersion(kMag, gravity, depth) / 2.0f;
}

// ═══════════════════════════════════════════════════════════════════════
// JONSWAP Spectrum + Directional Spreading
// ═══════════════════════════════════════════════════════════════════════

float TMACorrection(float omega, float depth, float gravity)
{
    float omegaH = omega * sqrt(depth / gravity);
    if (omegaH <= 1.0f)
        return 0.5f * omegaH * omegaH;
    if (omegaH < 2.0f)
        return 1.0f - 0.5f * (2.0f - omegaH) * (2.0f - omegaH);
    return 1.0f;
}

float JONSWAP(float omega, SpectrumParameters spectrum, float gravity, float depth)
{
    float sigma = (omega <= spectrum.peakOmega) ? 0.07f : 0.09f;

    float r = exp(-(omega - spectrum.peakOmega) * (omega - spectrum.peakOmega)
              / 2.0f / sigma / sigma / spectrum.peakOmega / spectrum.peakOmega);

    float oneOverOmega = 1.0f / omega;
    float peakOmegaOverOmega = spectrum.peakOmega / omega;

    return spectrum.scale * TMACorrection(omega, depth, gravity)
        * spectrum.alpha * gravity * gravity
        * oneOverOmega * oneOverOmega * oneOverOmega * oneOverOmega * oneOverOmega
        * exp(-1.25f * peakOmegaOverOmega * peakOmegaOverOmega
              * peakOmegaOverOmega * peakOmegaOverOmega)
        * pow(abs(spectrum.gamma), r);
}

float ShortWavesFade(float kLength, SpectrumParameters spectrum)
{
    return exp(-spectrum.shortWavesFade * spectrum.shortWavesFade * kLength * kLength);
}

// ── Donelan-Banner directional spreading ──

float NormalizationFactor(float s)
{
    float s2 = s * s;
    float s3 = s2 * s;
    float s4 = s3 * s;
    if (s < 5)
        return -0.000564f * s4 + 0.00776f * s3 - 0.044f * s2 + 0.192f * s + 0.163f;
    else
        return -4.80e-08f * s4 + 1.07e-05f * s3 - 9.53e-04f * s2 + 5.90e-02f * s + 3.93e-01f;
}

float DonelanBannerBeta(float x)
{
    if (x < 0.95f) return 2.61f * pow(abs(x), 1.3f);
    if (x < 1.6f) return 2.28f * pow(abs(x), -1.3f);
    float p = -0.4f + 0.8393f * exp(-0.567f * log(x * x));
    return pow(10.0f, p);
}

float DonelanBanner(float theta, float omega, float peakOmega)
{
    float beta = DonelanBannerBeta(omega / peakOmega);
    float sech = 1.0f / cosh(beta * theta);
    return beta / 2.0f / tanh(beta * PI) * sech * sech;
}

float Cosine2s(float theta, float s)
{
    return NormalizationFactor(s) * pow(abs(cos(0.5f * theta)), 2.0f * s);
}

float SpreadPower(float omega, float peakOmega)
{
    if (omega > peakOmega)
        return 9.77f * pow(abs(omega / peakOmega), -2.5f);
    else
        return 6.97f * pow(abs(omega / peakOmega), 5.0f);
}

float DirectionSpectrum(float theta, float omega, SpectrumParameters spectrum)
{
    float s = SpreadPower(omega, spectrum.peakOmega)
            + 16 * tanh(min(omega / spectrum.peakOmega, 20)) * spectrum.swell * spectrum.swell;
    return lerp(2.0f / PI * cos(theta) * cos(theta),
                Cosine2s(theta - spectrum.angle, s),
                spectrum.spreadBlend);
}

// ═══════════════════════════════════════════════════════════════════════
// CSInitSpectrum — one-time: generate initial H0 spectrum
// ═══════════════════════════════════════════════════════════════════════

[numthreads(8, 8, 1)]
void CSInitSpectrum(uint3 id : SV_DispatchThreadID)
{
    RWTexture2DArray<float4> initialSpectrum = ResourceDescriptorHeap[InitialSpectrumIdx];
    StructuredBuffer<SpectrumParameters> spectrums = ResourceDescriptorHeap[SpectrumParamsIdx];
    StructuredBuffer<OceanConstants> constants = ResourceDescriptorHeap[OceanConstantsIdx];
    OceanConstants oc = constants[0];

    uint N = oc.N;
    uint seed = id.x + N * id.y + N + oc.Seed;

    StructuredBuffer<uint> lengthScales = ResourceDescriptorHeap[oc.LengthScalesSRV];

    [loop]
    for (uint i = 0; i < oc.NumBands; ++i)
    {
        float halfN = N / 2.0f;
        float deltaK = 2.0f * PI / (float)lengthScales[i];
        float2 K = (id.xy - halfN) * deltaK;
        float kLength = length(K);

        // Unique seed per band
        seed += i + hash(seed) * 10;
        float4 uniformRand = float4(hash(seed), hash(seed * 2), hash(seed * 3), hash(seed * 4));
        float2 gauss1 = UniformToGaussian(uniformRand.x, uniformRand.y);
        float2 gauss2 = UniformToGaussian(uniformRand.z, uniformRand.w);

        if (oc.LowCutoff <= kLength && kLength <= oc.HighCutoff)
        {
            float kAngle = atan2(K.y, K.x);
            float omega = Dispersion(kLength, oc.Gravity, oc.Depth);
            float dOmegadk = DispersionDerivative(kLength, oc.Gravity, oc.Depth);

            // Primary spectrum for this band
            float spectrum = JONSWAP(omega, spectrums[i * 2], oc.Gravity, oc.Depth)
                           * DirectionSpectrum(kAngle, omega, spectrums[i * 2])
                           * ShortWavesFade(kLength, spectrums[i * 2]);

            // Secondary spectrum (dual JONSWAP)
            if (spectrums[i * 2 + 1].scale > 0)
            {
                spectrum += JONSWAP(omega, spectrums[i * 2 + 1], oc.Gravity, oc.Depth)
                          * DirectionSpectrum(kAngle, omega, spectrums[i * 2 + 1])
                          * ShortWavesFade(kLength, spectrums[i * 2 + 1]);
            }

            initialSpectrum[uint3(id.xy, i)] = float4(
                float2(gauss2.x, gauss1.y) * sqrt(2 * spectrum * abs(dOmegadk) / kLength * deltaK * deltaK),
                0.0f, 0.0f
            );
        }
        else
        {
            initialSpectrum[uint3(id.xy, i)] = 0.0f;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CSPackConjugate — one-time: pack H0 + H0*(-k) into RGBA
// ═══════════════════════════════════════════════════════════════════════

[numthreads(8, 8, 1)]
void CSPackConjugate(uint3 id : SV_DispatchThreadID)
{
    RWTexture2DArray<float4> initialSpectrum = ResourceDescriptorHeap[InitialSpectrumIdx];
    StructuredBuffer<OceanConstants> constants = ResourceDescriptorHeap[OceanConstantsIdx];
    uint N = constants[0].N;

    [loop]
    for (uint i = 0; i < constants[0].NumBands; ++i)
    {
        float2 h0 = initialSpectrum[uint3(id.xy, i)].rg;
        float2 h0conj = initialSpectrum[uint3((N - id.x) % N, (N - id.y) % N, i)].rg;
        initialSpectrum[uint3(id.xy, i)] = float4(h0, h0conj.x, -h0conj.y);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CSEvolveSpectrum — per-frame: time-evolve spectrum for FFT
// ═══════════════════════════════════════════════════════════════════════

[numthreads(8, 8, 1)]
void CSEvolveSpectrum(uint3 id : SV_DispatchThreadID)
{
    RWTexture2DArray<float4> initialSpectrum = ResourceDescriptorHeap[InitialSpectrumIdx];
    RWTexture2DArray<float4> spectrumOut = ResourceDescriptorHeap[SpectrumIdx];
    StructuredBuffer<OceanConstants> constants = ResourceDescriptorHeap[OceanConstantsIdx];
    OceanConstants oc = constants[0];

    uint N = oc.N;
    StructuredBuffer<uint> lengthScales = ResourceDescriptorHeap[oc.LengthScalesSRV];

    [loop]
    for (int i = 0; i < (int)oc.NumBands; ++i)
    {
        float4 initialSignal = initialSpectrum[uint3(id.xy, i)];
        float2 h0 = initialSignal.xy;
        float2 h0conj = initialSignal.zw;

        float halfN = N / 2.0f;
        float2 K = (id.xy - halfN) * 2.0f * PI / (float)lengthScales[i];
        float kMag = length(K);
        float kMagRcp = (kMag < 0.0001f) ? 1.0f : rcp(kMag);

        // Quantized dispersion — uses full depth-dependent relation
        float w_0 = 2.0f * PI / oc.RepeatTime;
        float dispersion = floor(Dispersion(kMag, oc.Gravity, oc.Depth) / w_0) * w_0 * oc.FrameTime;

        float2 exponent = EulerFormula(dispersion);
        float2 htilde = ComplexMult(h0, exponent) + ComplexMult(h0conj, float2(exponent.x, -exponent.y));
        float2 ih = float2(-htilde.y, htilde.x);

        // Displacement spectra
        float2 displacementX = ih * K.x * kMagRcp;
        float2 displacementY = htilde;
        float2 displacementZ = ih * K.y * kMagRcp;

        // Derivative spectra (for slopes + Jacobian)
        float2 displacementX_dx = -htilde * K.x * K.x * kMagRcp;
        float2 displacementY_dx = ih * K.x;
        float2 displacementZ_dx = -htilde * K.x * K.y * kMagRcp;
        float2 displacementY_dz = ih * K.y;
        float2 displacementZ_dz = -htilde * K.y * K.y * kMagRcp;

        // Pack into 2 RGBA texels per band (numBands*2 slices total)
        float2 htildeDisplacementX = float2(displacementX.x - displacementZ.y, displacementX.y + displacementZ.x);
        float2 htildeDisplacementZ = float2(displacementY.x - displacementZ_dx.y, displacementY.y + displacementZ_dx.x);

        float2 htildeSlopeX = float2(displacementY_dx.x - displacementY_dz.y, displacementY_dx.y + displacementY_dz.x);
        float2 htildeSlopeZ = float2(displacementX_dx.x - displacementZ_dz.y, displacementX_dx.y + displacementZ_dz.x);

        spectrumOut[uint3(id.xy, i * 2)] = float4(htildeDisplacementX, htildeDisplacementZ);
        spectrumOut[uint3(id.xy, i * 2 + 1)] = float4(htildeSlopeX, htildeSlopeZ);
    }
}
