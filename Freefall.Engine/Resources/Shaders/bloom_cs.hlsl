// ============================================================================
// Bloom Compute Shader
// ============================================================================
// Multi-pass bloom using a mip pyramid:
//   1. CSBloomThreshold  — Extract bright pixels at half-res with Karis average
//                          for firefly suppression and soft thresholding.
//   2. CSBloomDownsample — Progressively downsample using a 13-tap filter
//                          (Jimenez 2014 / Call of Duty) to minimize aliasing.
//   3. CSBloomUpsample   — Progressively upsample with a 3x3 tent filter,
//                          additively blending into each finer mip level.
// ============================================================================

#pragma kernel CSBloomThreshold
#pragma kernel CSBloomDownsample
#pragma kernel CSBloomUpsample

// Engine-bound bilinear clamp sampler from the global root signature
SamplerState _LinearClamp : register(s2);

// Push constants — layout shared across all three kernels.
// The C# side sets different values per dispatch.
cbuffer PushConstants : register(b3)
{
    uint InputSrvIdx;
    uint OutputUAVIdx;
    uint OutputWidth;
    uint OutputHeight;
    uint Param0Bits;    // Threshold / InputTexelSize.x (reinterpreted as float)
    uint Param1Bits;    // SoftKnee  / InputTexelSize.y (reinterpreted as float)
    uint Param2Bits;    // unused    / BloomRadius      (reinterpreted as float)
    uint DepthSrvIdx;   // Depth buffer SRV (threshold only — sky mask)
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

float Luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

// Clamp NaN/Inf to zero — prevents propagation through the bloom mip chain
float3 SafeHDR(float3 c)
{
    return clamp(c, 0, 65504.0);
}

// ============================================================================
// Kernel 1: Bright-pass extraction with Karis average
// ============================================================================
// Each thread reads a 2x2 block from the full-res HDR source and writes one
// half-res output pixel. Soft thresholding provides a smooth falloff, and
// Karis averaging suppresses fireflies by weighting samples inversely with
// their luminance.
// ============================================================================

[numthreads(8, 8, 1)]
void CSBloomThreshold(uint3 id : SV_DispatchThreadID)
{
    uint halfWidth  = OutputWidth  / 2;
    uint halfHeight = OutputHeight / 2;

    if (id.x >= halfWidth || id.y >= halfHeight)
        return;

    float threshold = asfloat(Param0Bits);
    float softKnee  = asfloat(Param1Bits);

    Texture2D<float4> input  = ResourceDescriptorHeap[InputSrvIdx];
    RWTexture2D<float4> output = ResourceDescriptorHeap[OutputUAVIdx];

    // Load 2x2 block from the full-res source
    uint2 base = id.xy * 2;
    float3 s0 = SafeHDR(input.Load(int3(base + uint2(0, 0), 0)).rgb);
    float3 s1 = SafeHDR(input.Load(int3(base + uint2(1, 0), 0)).rgb);
    float3 s2 = SafeHDR(input.Load(int3(base + uint2(0, 1), 0)).rgb);
    float3 s3 = SafeHDR(input.Load(int3(base + uint2(1, 1), 0)).rgb);

    // Per-sample luminance
    float l0 = Luminance(s0);
    float l1 = Luminance(s1);
    float l2 = Luminance(s2);
    float l3 = Luminance(s3);

    // Karis average: weight each sample by 1 / (1 + luminance) to suppress fireflies
    float w0 = 1.0 / (1.0 + l0);
    float w1 = 1.0 / (1.0 + l1);
    float w2 = 1.0 / (1.0 + l2);
    float w3 = 1.0 / (1.0 + l3);
    float wSum = w0 + w1 + w2 + w3;

    float3 avg = (s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3) / wSum;

    // Sky mask: in reverse-Z, sky pixels have depth ≈ 0 (far plane).
    // Skip them so the sky doesn't bloom — only emissive/specular surfaces should.
    if (DepthSrvIdx != 0)
    {
        Texture2D<float> depthTex = ResourceDescriptorHeap[DepthSrvIdx];
        // Sample any corner of the 2x2 block — if it's sky, the whole block is sky
        float depth = depthTex.Load(int3(base, 0)).r;
        if (depth < 0.00001)
        {
            output[id.xy] = float4(0, 0, 0, 1);
            return;
        }
    }

    // Soft threshold
    float lum = Luminance(avg);
    float soft = max(0.0, lum - threshold + softKnee) / (2.0 * softKnee + 0.00001);
    soft = saturate(soft);
    soft = soft * soft;

    output[id.xy] = float4(avg * soft, 1.0);
}

// ============================================================================
// Kernel 2: 13-tap downsample (Jimenez 2014 / Call of Duty)
// ============================================================================
// Samples 13 bilinear taps in a cross+box pattern to produce a high-quality
// downsample that minimizes aliasing and pulsing artifacts.
//
//   a - b - c
//   - j - k -
//   d - e - f
//   - l - m -
//   g - h - i
//
// Weights sum to 1.0:
//   e (center)           : 0.125
//   a, c, g, i (corners) : 0.03125 each  (total 0.125)
//   b, d, f, h (edges)   : 0.0625  each  (total 0.25)
//   j, k, l, m (inner)   : 0.125   each  (total 0.5)
// ============================================================================

[numthreads(8, 8, 1)]
void CSBloomDownsample(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= OutputWidth || id.y >= OutputHeight)
        return;

    float2 texelSize = float2(asfloat(Param0Bits), asfloat(Param1Bits));

    Texture2D<float4> input  = ResourceDescriptorHeap[InputSrvIdx];
    RWTexture2D<float4> output = ResourceDescriptorHeap[OutputUAVIdx];

    float2 uv = (id.xy + 0.5) / float2(OutputWidth, OutputHeight);

    // 13-tap samples
    float3 a = input.SampleLevel(_LinearClamp, uv + float2(-2, -2) * texelSize, 0).rgb;
    float3 b = input.SampleLevel(_LinearClamp, uv + float2( 0, -2) * texelSize, 0).rgb;
    float3 c = input.SampleLevel(_LinearClamp, uv + float2( 2, -2) * texelSize, 0).rgb;
    float3 d = input.SampleLevel(_LinearClamp, uv + float2(-2,  0) * texelSize, 0).rgb;
    float3 e = input.SampleLevel(_LinearClamp, uv,                              0).rgb;
    float3 f = input.SampleLevel(_LinearClamp, uv + float2( 2,  0) * texelSize, 0).rgb;
    float3 g = input.SampleLevel(_LinearClamp, uv + float2(-2,  2) * texelSize, 0).rgb;
    float3 h = input.SampleLevel(_LinearClamp, uv + float2( 0,  2) * texelSize, 0).rgb;
    float3 i = input.SampleLevel(_LinearClamp, uv + float2( 2,  2) * texelSize, 0).rgb;
    float3 j = input.SampleLevel(_LinearClamp, uv + float2(-1, -1) * texelSize, 0).rgb;
    float3 k = input.SampleLevel(_LinearClamp, uv + float2( 1, -1) * texelSize, 0).rgb;
    float3 l = input.SampleLevel(_LinearClamp, uv + float2(-1,  1) * texelSize, 0).rgb;
    float3 m = input.SampleLevel(_LinearClamp, uv + float2( 1,  1) * texelSize, 0).rgb;

    float3 result = e * 0.125;
    result += (a + c + g + i) * 0.03125;
    result += (b + d + f + h) * 0.0625;
    result += (j + k + l + m) * 0.125;

    output[id.xy] = float4(result, 1.0);
}

// ============================================================================
// Kernel 3: 3x3 tent filter upsample with additive blend
// ============================================================================
// Reads the coarser mip through a bilinear sampler with a 3x3 tent kernel,
// then additively blends the result into the finer mip level (read-modify-write
// on the UAV, safe because each thread writes a unique pixel).
//
// Tent kernel (unnormalized, divide by 16):
//   1 2 1
//   2 4 2
//   1 2 1
// ============================================================================

[numthreads(8, 8, 1)]
void CSBloomUpsample(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= OutputWidth || id.y >= OutputHeight)
        return;

    float2 texelSize   = float2(asfloat(Param0Bits), asfloat(Param1Bits));
    float  bloomRadius = asfloat(Param2Bits);

    Texture2D<float4> input  = ResourceDescriptorHeap[InputSrvIdx];
    RWTexture2D<float4> output = ResourceDescriptorHeap[OutputUAVIdx];

    float2 uv = (id.xy + 0.5) / float2(OutputWidth, OutputHeight);
    float2 offset = texelSize * bloomRadius;

    float3 result = 0;
    result += input.SampleLevel(_LinearClamp, uv + float2(-1, -1) * offset, 0).rgb * 1.0;
    result += input.SampleLevel(_LinearClamp, uv + float2( 0, -1) * offset, 0).rgb * 2.0;
    result += input.SampleLevel(_LinearClamp, uv + float2( 1, -1) * offset, 0).rgb * 1.0;
    result += input.SampleLevel(_LinearClamp, uv + float2(-1,  0) * offset, 0).rgb * 2.0;
    result += input.SampleLevel(_LinearClamp, uv,                           0).rgb * 4.0;
    result += input.SampleLevel(_LinearClamp, uv + float2( 1,  0) * offset, 0).rgb * 2.0;
    result += input.SampleLevel(_LinearClamp, uv + float2(-1,  1) * offset, 0).rgb * 1.0;
    result += input.SampleLevel(_LinearClamp, uv + float2( 0,  1) * offset, 0).rgb * 2.0;
    result += input.SampleLevel(_LinearClamp, uv + float2( 1,  1) * offset, 0).rgb * 1.0;
    result /= 16.0;

    // Additive blend into the finer mip
    output[id.xy] = float4(output[id.xy].rgb + result, 1.0);
}
