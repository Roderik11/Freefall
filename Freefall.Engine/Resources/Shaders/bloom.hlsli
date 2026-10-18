// Shared by the bloom prefilter (bloom_cs.hlsl) and the finalize pass (finalize_cs.hlsl), which
// takes the same share back out of each pixel when the bloom spreads light instead of adding it.

// Share of a pixel's light that goes into the bloom pyramid: soft threshold on luminance.
float BloomShare(float luminance, float threshold, float knee)
{
    float soft = saturate(max(0.0, luminance - threshold + knee) / (2.0 * knee + 0.00001));
    return soft * soft;
}

// The light a pixel can give the bloom, softly capped at `cap` (pre-exposed luminance; 0 = no cap): c / (1 + L / cap).
// Daylight seen from a dark cave is thousands of times white; any share of it spread out was a blinding blob round the
// mouth. Capped, it glows like a lamp. The prefilter and the finalize's take-back use the same value, so the pixel
// gives up exactly what the bloom spreads.
float3 BloomSource(float3 c, float cap)
{
    if (cap <= 0.0)
        return c;
    float luminance = dot(c, float3(0.2126, 0.7152, 0.0722));
    return c / (1.0 + luminance / cap);
}

// Reverse-Z: the sky is at depth 0. It stays out of the bloom.
bool BloomIsSky(float depth)
{
    return depth < 0.00001;
}
