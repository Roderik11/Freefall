// The composite scaled up to the view (DeferredRenderer.Upscale, EngineSettings.RenderScale below 1).
// Reads the pre-exposed HDR composite at the render size and writes the same light at the display size, which the
// bloom and the finalize pass then read in its place. Until a temporal upscaler takes this place (UPSCALING_PLAN.md),
// it is a plain spatial one: Catmull-Rom from nine bilinear taps, held between the darkest and brightest of the four
// nearest texels, since in HDR a bright edge beside a dark one (the sun's glint on a roof against the sky) rings far.

#pragma kernel CSUpscale

cbuffer PushConstants : register(b3)
{
    uint InputSrvIdx;      // the composite (render size)
    uint OutputUAVIdx;     // Upscaled (display size)
    uint InputWidth;
    uint InputHeight;
    uint OutputWidth;
    uint OutputHeight;
};

SamplerState LinearClampSampler : register(s2);

// What the finalize pass keeps of a pixel: no NaN, no negative light, nothing past half float
float3 Safe(float3 c)
{
    return min(max(c, 0.0), 65504.0);
}

float3 Tap(Texture2D<float4> input, float2 uv)
{
    return Safe(input.SampleLevel(LinearClampSampler, uv, 0).rgb);
}

[numthreads(8, 8, 1)]
void CSUpscale(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= OutputWidth || id.y >= OutputHeight)
        return;

    Texture2D<float4> input = ResourceDescriptorHeap[InputSrvIdx];
    RWTexture2D<float4> output = ResourceDescriptorHeap[OutputUAVIdx];

    float2 inputSize = float2(InputWidth, InputHeight);
    float2 uv = (float2(id.xy) + 0.5) / float2(OutputWidth, OutputHeight);

    // Catmull-Rom weights of the 4×4 texels round the sample, the middle two of each row and column merged into one
    // bilinear tap between them (exact: their weights have the same sign)
    float2 at = uv * inputSize;
    float2 centre1 = floor(at - 0.5) + 0.5;
    float2 f = at - centre1;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 uv0 = (centre1 - 1.0) / inputSize;
    float2 uv12 = (centre1 + w2 / w12) / inputSize;
    float2 uv3 = (centre1 + 2.0) / inputSize;

    float3 c = Tap(input, float2(uv0.x,  uv0.y))  * (w0.x  * w0.y)
             + Tap(input, float2(uv12.x, uv0.y))  * (w12.x * w0.y)
             + Tap(input, float2(uv3.x,  uv0.y))  * (w3.x  * w0.y)
             + Tap(input, float2(uv0.x,  uv12.y)) * (w0.x  * w12.y)
             + Tap(input, float2(uv12.x, uv12.y)) * (w12.x * w12.y)
             + Tap(input, float2(uv3.x,  uv12.y)) * (w3.x  * w12.y)
             + Tap(input, float2(uv0.x,  uv3.y))  * (w0.x  * w3.y)
             + Tap(input, float2(uv12.x, uv3.y))  * (w12.x * w3.y)
             + Tap(input, float2(uv3.x,  uv3.y))  * (w3.x  * w3.y);

    // Held between the four nearest texels: Catmull-Rom's negative lobes overshoot at every hard edge
    int2 last = int2(InputWidth, InputHeight) - 1;
    int2 p = int2(centre1 - 0.5);
    float3 n0 = Safe(input.Load(int3(clamp(p,               int2(0, 0), last), 0)).rgb);
    float3 n1 = Safe(input.Load(int3(clamp(p + int2(1, 0),  int2(0, 0), last), 0)).rgb);
    float3 n2 = Safe(input.Load(int3(clamp(p + int2(0, 1),  int2(0, 0), last), 0)).rgb);
    float3 n3 = Safe(input.Load(int3(clamp(p + int2(1, 1),  int2(0, 0), last), 0)).rgb);
    float3 lo = min(min(n0, n1), min(n2, n3));
    float3 hi = max(max(n0, n1), max(n2, n3));

    output[id.xy] = float4(clamp(c, lo, hi), 1.0);
}
