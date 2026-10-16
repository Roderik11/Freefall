// Shared by the bloom prefilter (bloom_cs.hlsl) and the finalize pass (finalize_cs.hlsl), which
// takes the same share back out of each pixel when the bloom spreads light instead of adding it.

// Share of a pixel's light that goes into the bloom pyramid: soft threshold on luminance.
float BloomShare(float luminance, float threshold, float knee)
{
    float soft = saturate(max(0.0, luminance - threshold + knee) / (2.0 * knee + 0.00001));
    return soft * soft;
}

// Reverse-Z: the sky is at depth 0. It stays out of the bloom.
bool BloomIsSky(float depth)
{
    return depth < 0.00001;
}
