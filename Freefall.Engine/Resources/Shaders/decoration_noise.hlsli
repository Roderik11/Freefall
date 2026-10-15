// decoration_noise.hlsli — world-space noise for terrain decorators: the spawn kernel's clumps, height variation
// and edge offset (per instance), and the same edge offset in the terrain's ground tint (gputerrain.fx).
// Positions are terrain-local metres (0..TerrainSize), so both kernels see the same field.

#ifndef DECORATION_NOISE_HLSLI
#define DECORATION_NOISE_HLSLI

// Integer hash → unit gradient direction (no lattice-aligned artifacts, unlike value noise). Two 16-bit halves of the
// hash normalised, not cos/sin of an angle: the spawn kernel evaluates ~10 of these noises per candidate instance, and
// the trig made the spawn twice as expensive.
float2 DecoGradient(int2 cell)
{
    uint h = uint(cell.x) * 1597334677u ^ uint(cell.y) * 3812015801u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    float2 g = float2(h & 0xFFFFu, h >> 16) * (2.0 / 65535.0) - 1.0;
    return g * rsqrt(max(dot(g, g), 1e-4));
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

// Offset (terrain-local metres, mostly within +-0.5) for where a decorator looks up its coverage: wobbles the edge
// of a patch so it neither traces the 1 m control texels nor the height and slope lines its stamps follow. The spawn
// kernel and the ground tint under decorators (gputerrain.fx) use the same offset, so cards and tint share one edge.
float2 DecoEdgeWarp(float2 localPos)
{
    float2 p = localPos / 2.7;
    float2 q = localPos / 0.9;
    return float2(DecoGradientNoise(p + 31.7), DecoGradientNoise(p + float2(-12.9, 57.3))) * 1.1
         + float2(DecoGradientNoise(q + 5.3), DecoGradientNoise(q + float2(77.1, -3.9))) * 0.5
         - 0.8;
}

// The grain of a sward at 1.4..9 m: tussocks and denser or thinner clumps, darker and a little greener where dense,
// lighter and drier where thin. The ground under decorators (gputerrain.fx, the sward colour) and every plant on it
// (grass_compute.hlsl, per instance) take the same field, so from afar the meadow keeps the texture its plants have up
// close; without it the far meadow was one flat colour between the 18..240 m patches ("perfectly smooth", 2026-10-15).
// localPos: terrain-local metres.
float3 SwardGrain(float2 localPos)
{
    float g = (DecoGradientNoise(localPos / 9.0 + 211.3) - 0.5) * 1.0
            + (DecoGradientNoise(localPos / 3.5 + 37.9) - 0.5) * 0.6
            + (DecoGradientNoise(localPos / 1.4 + 91.7) - 0.5) * 0.4;        // about +-0.5
    float brightness = 1.0 + g * 0.55;                                        // ~0.75 .. 1.25
    float3 tone = lerp(float3(0.96, 1.0, 0.97), float3(1.04, 1.0, 0.97), saturate(g + 0.5));   // lush .. dry
    return brightness * tone;
}

// Height multiplier for a decorator: short patches ↔ lush patches, independent of the clump field.
// amount 0 = 1.0 everywhere; amount 1 = 0.4x .. 1.4x.
float DecoHeightFactor(float2 localPos, float scale, float amount, uint slot)
{
    float n = smoothstep(0.25, 0.75, DecoFbm(localPos, scale, float(slot) + 101.0));
    return lerp(1.0 - 0.6 * amount, 1.0 + 0.4 * amount, n);
}

#endif
