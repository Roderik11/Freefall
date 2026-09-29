// decoration_noise.hlsli — world-space noise for terrain decorators, shared by the decoration prepass (clump
// density, baked into the control texture) and the spawn kernel (per-instance height variation).
// Positions are terrain-local metres (0..TerrainSize), so both kernels see the same field.

#ifndef DECORATION_NOISE_HLSLI
#define DECORATION_NOISE_HLSLI

// Integer hash → unit gradient direction (no lattice-aligned artifacts, unlike value noise)
float2 DecoGradient(int2 cell)
{
    uint h = uint(cell.x) * 1597334677u ^ uint(cell.y) * 3812015801u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    float a = float(h) * (6.2831853 / 4294967296.0);
    return float2(cos(a), sin(a));
}

// 2D gradient (Perlin) noise with quintic fade, remapped to ~0..1
float DecoGradientNoise(float2 p)
{
    int2 i = int2(floor(p));
    float2 f = p - floor(p);
    float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float a = dot(DecoGradient(i),              f);
    float b = dot(DecoGradient(i + int2(1, 0)), f - float2(1, 0));
    float c = dot(DecoGradient(i + int2(0, 1)), f - float2(0, 1));
    float d = dot(DecoGradient(i + int2(1, 1)), f - float2(1, 1));
    float n = lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y); // ~[-0.7, 0.7]
    return saturate(n * 0.72 + 0.5);
}

// 3-octave fBm (0..1). 'salt' decorrelates decorators and the density vs. height fields.
float DecoFbm(float2 localPos, float scale, float salt)
{
    float2 p = localPos / max(scale, 0.5) + float2(salt * 17.31, salt * 9.77);
    return DecoGradientNoise(p) * 0.6
         + DecoGradientNoise(p * 2.13 + 3.7) * 0.28
         + DecoGradientNoise(p * 4.71 + 11.1) * 0.12;
}

// 0..1 clump mask: dense cores, bare gaps, soft ragged transitions
float DecoClusterMask(float2 localPos, float scale, uint slot)
{
    return smoothstep(0.36, 0.64, DecoFbm(localPos, scale, float(slot)));
}

// Height multiplier for a decorator: short patches ↔ lush patches, independent of the clump field.
// amount 0 = 1.0 everywhere; amount 1 = 0.4x .. 1.4x.
float DecoHeightFactor(float2 localPos, float scale, float amount, uint slot)
{
    float n = smoothstep(0.25, 0.75, DecoFbm(localPos, scale, float(slot) + 101.0));
    return lerp(1.0 - 0.6 * amount, 1.0 + 0.4 * amount, n);
}

#endif
