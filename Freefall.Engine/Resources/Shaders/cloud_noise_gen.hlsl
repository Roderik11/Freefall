// ──────────────────────────────────────────────────────────────
// Cloud Noise LUT Generator — dispatched once at startup
// Generates SEAMLESSLY TILEABLE 3D noise for cloud density.
//
// All noise functions wrap cell coordinates modulo the grid period,
// so the output texture tiles perfectly when sampled with Wrap addressing.
//
// Kernel: CSGenNoise — writes to a 128×128×128 RGBA8 volume
//   R: Perlin FBM     (connected cloud shapes)
//   G: Worley FBM     (billowy erosion, low freq)
//   B: Worley FBM     (detail erosion, higher freq)
//   A: Perlin-Worley  (pre-combined for convenience)
// ──────────────────────────────────────────────────────────────

#pragma kernel CSGenNoise

cbuffer PushConstants : register(b3)
{
    uint OutputUAVIdx;
    uint VolumeSize;     // 128
    uint _pad0;
    uint _pad1;
};

RWTexture3D<float4> OutputUAV : register(u0);

// ── Positive modulo (HLSL fmod can return negative) ──
float3 pmod(float3 x, float3 p)
{
    return x - p * floor(x / p);
}

// ── Hash: takes ALREADY-WRAPPED coordinates ──
float3 hash33(float3 p)
{
    p = float3(dot(p, float3(127.1, 311.7, 74.7)),
               dot(p, float3(269.5, 183.3, 246.1)),
               dot(p, float3(113.5, 271.9, 124.6)));
    return frac(sin(p) * 43758.5453123);
}

// ── Tileable Perlin gradient noise ──
// Period = number of grid cells before the pattern repeats

float3 fade(float3 t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

float gradientNoise(float3 p, float period)
{
    float3 per = float3(period, period, period);
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = fade(f);

    // Wrap the 8 corner coordinates for seamless tiling
    float3 i000 = pmod(i + float3(0,0,0), per);
    float3 i100 = pmod(i + float3(1,0,0), per);
    float3 i010 = pmod(i + float3(0,1,0), per);
    float3 i110 = pmod(i + float3(1,1,0), per);
    float3 i001 = pmod(i + float3(0,0,1), per);
    float3 i101 = pmod(i + float3(1,0,1), per);
    float3 i011 = pmod(i + float3(0,1,1), per);
    float3 i111 = pmod(i + float3(1,1,1), per);

    float n000 = dot(hash33(i000) * 2.0 - 1.0, f - float3(0,0,0));
    float n100 = dot(hash33(i100) * 2.0 - 1.0, f - float3(1,0,0));
    float n010 = dot(hash33(i010) * 2.0 - 1.0, f - float3(0,1,0));
    float n110 = dot(hash33(i110) * 2.0 - 1.0, f - float3(1,1,0));
    float n001 = dot(hash33(i001) * 2.0 - 1.0, f - float3(0,0,1));
    float n101 = dot(hash33(i101) * 2.0 - 1.0, f - float3(1,0,1));
    float n011 = dot(hash33(i011) * 2.0 - 1.0, f - float3(0,1,1));
    float n111 = dot(hash33(i111) * 2.0 - 1.0, f - float3(1,1,1));

    float nx00 = lerp(n000, n100, u.x);
    float nx10 = lerp(n010, n110, u.x);
    float nx01 = lerp(n001, n101, u.x);
    float nx11 = lerp(n011, n111, u.x);
    float nxy0 = lerp(nx00, nx10, u.y);
    float nxy1 = lerp(nx01, nx11, u.y);
    return lerp(nxy0, nxy1, u.z);
}

float perlinFBM(float3 p, int octaves, float baseFreq)
{
    float value = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < octaves; i++)
    {
        // Period doubles with frequency so each octave tiles correctly
        value += gradientNoise(p * freq, baseFreq * freq) * amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    return value * 0.5 + 0.5; // remap -1..1 to 0..1
}

// ── Tileable Worley noise ──
// Returns inverted distance (bright = inside cell puff)

float worley(float3 p, float period)
{
    float3 per = float3(period, period, period);
    float3 i = floor(p);
    float3 f = frac(p);

    float minDist = 1.0;
    for (int z = -1; z <= 1; z++)
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++)
    {
        float3 neighbor = float3(x, y, z);
        // Wrap the cell coordinate for seamless tiling
        float3 wrappedCell = pmod(i + neighbor, per);
        float3 featurePoint = hash33(wrappedCell);
        float3 diff = neighbor + featurePoint - f;
        float dist = dot(diff, diff);
        minDist = min(minDist, dist);
    }
    return 1.0 - sqrt(minDist); // inverted: bright inside puffs
}

float worleyFBM(float3 p, int octaves, float baseFreq)
{
    float value = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < octaves; i++)
    {
        value += worley(p * freq, baseFreq * freq) * amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    return value;
}

// ── Remap utility ──
float remap(float value, float lo, float hi, float newLo, float newHi)
{
    return newLo + saturate((value - lo) / (hi - lo)) * (newHi - newLo);
}

// ── Main kernel ──

[numthreads(4, 4, 4)]
void CSGenNoise(uint3 dtid : SV_DispatchThreadID)
{
    if (any(dtid >= VolumeSize)) return;

    float3 uvw = (float3(dtid) + 0.5) / float(VolumeSize);

    // Base frequency: how many noise cells across the texture
    // Lower = larger shapes, fewer tiles, more "cloudy"
    static const float BASE = 4.0;
    float3 p = uvw * BASE;

    // R: Perlin FBM — smooth, connected large-scale shapes
    float perlin = perlinFBM(p, 5, BASE);

    // G: Worley FBM at low frequency — for soft erosion
    float worLo = worleyFBM(p, 3, BASE);

    // B: Worley FBM at 2x frequency — finer detail erosion
    float worHi = worleyFBM(p * 2.0, 3, BASE * 2.0);

    // A: Perlin-Worley combined — Worley erodes Perlin edges
    // remap: where Worley is high (inside puffs), Perlin density is kept.
    //        where Worley is low (between puffs), Perlin is carved away.
    float pw = remap(perlin, worLo * 0.4, 1.0, 0.0, 1.0);

    RWTexture3D<float4> output = ResourceDescriptorHeap[OutputUAVIdx];
    output[dtid] = float4(perlin, worLo, worHi, pw);
}
