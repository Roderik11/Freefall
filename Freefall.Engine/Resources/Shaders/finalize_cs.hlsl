// Finalize Compute Shader — HDR → LDR conversion
// Reads the HDR Composite (which contains both deferred + forward content),
// adds bloom, tonemaps, applies vibrance/gamma/dithering, writes LDR output.
// Runs after all rendering and bloom are complete.

#pragma kernel CSFinalize

cbuffer PushConstants : register(b3)
{
    uint InputSrvIdx;       // HDR Composite SRV
    uint OutputUAVIdx;      // LDR output UAV
    uint ScreenWidth;
    uint ScreenHeight;
    uint BloomTexIdx;       // Bloom mip 0 SRV (0 = no bloom)
    uint BloomIntBits;      // Bloom strength (float reinterpreted as uint)
    uint TimeBits;          // Frame time for dither (float reinterpreted as uint)
};

SamplerState LinearClampSampler : register(s2);

[numthreads(8, 8, 1)]
void CSFinalize(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 px = dispatchThreadId.xy;
    
    if (px.x >= ScreenWidth || px.y >= ScreenHeight)
        return;
    
    Texture2D<float4> Input = ResourceDescriptorHeap[InputSrvIdx];
    RWTexture2D<float4> Output = ResourceDescriptorHeap[OutputUAVIdx];
    
    float3 finalColor = Input.Load(int3(px, 0)).rgb;
    // Sanitize NaN/Inf from HDR input (e.g. point light specular overflow)
    finalColor = max(finalColor, 0);
    finalColor = min(finalColor, 65504.0);
    
    // Add bloom (HDR, before tonemapping)
    if (BloomTexIdx != 0)
    {
        float bloomIntensity = asfloat(BloomIntBits);
        Texture2D BloomTex = ResourceDescriptorHeap[BloomTexIdx];
        // Bloom is half-res — sample with bilinear filtering for smooth upscale
        float2 uv = (float2(px) + 0.5) / float2(ScreenWidth, ScreenHeight);
        float3 bloom = BloomTex.SampleLevel(LinearClampSampler, uv, 0).rgb;
        bloom = max(bloom, 0);
        finalColor += bloom * bloomIntensity;
    }
    
    // ACES Filmic Tone Mapping (Narkowicz 2015 approximation)
    float3 x = finalColor;
    finalColor = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    finalColor = saturate(finalColor);
    
    // Vibrance boost: selectively saturate under-saturated pixels
    float luma = dot(finalColor, float3(0.2126, 0.7152, 0.0722));
    float currentSat = max(finalColor.r, max(finalColor.g, finalColor.b)) - min(finalColor.r, min(finalColor.g, finalColor.b));
    float vibranceAmount = (1.0 - currentSat) * 0.15;
    finalColor = lerp(float3(luma, luma, luma), finalColor, 1.0 + vibranceAmount);
    finalColor = saturate(finalColor);
    
    // Final Gamma Correction (Linear → sRGB)
    finalColor = pow(abs(finalColor), 1.0f / 2.2f);
    
    // Dithering — break up color banding in smooth gradients (sky)
    float Time = asfloat(TimeBits);
    float2 seed = float2(px) + float2(frac(Time * 0.1), frac(Time * 0.31));
    float noise1 = frac(sin(dot(seed, float2(12.9898, 78.233))) * 43758.5453);
    float noise2 = frac(sin(dot(seed, float2(39.3468, 11.135))) * 23564.2365);
    float dither = (noise1 + noise2 - 1.0) / 255.0;
    finalColor += dither;
    
    Output[px] = float4(finalColor, 1.0f);
}
