// ocean_noise.hlsl — Generate tileable Perlin + Worley FBM noise texture
// SM 6.6 bindless, push constants at b3
// Generates a 256×256 RGBA texture:
//   R = Perlin FBM (smooth, organic shapes for foam mask)
//   G = Worley FBM (multi-octave cellular for foam detail)
//   B = Perlin-Worley blend (combined organic + cellular)
//   A = High-freq Perlin FBM (edge breakup)

struct PushConstantsData { uint4 indices[8]; };
#define GET_INDEX(i) PushConstants.indices[i/4][i%4]
ConstantBuffer<PushConstantsData> PushConstants : register(b3);

#define OutputIdx GET_INDEX(0)
#define TexSize   GET_INDEX(1)

// ═══════════════════════════════════════════════════
// Hash functions
// ═══════════════════════════════════════════════════

float2 hash2(float2 p)
{
    float2 q = float2(dot(p, float2(127.1, 311.7)),
                      dot(p, float2(269.5, 183.3)));
    return frac(sin(q) * 43758.5453);
}

// ═══════════════════════════════════════════════════
// Perlin noise (tileable, quintic interpolation)
// ═══════════════════════════════════════════════════

float perlinNoise(float2 p, float period)
{
    float2 i = floor(p);
    float2 f = frac(p);
    float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    
    float2 g00 = hash2(fmod(i + float2(0, 0), period)) * 2.0 - 1.0;
    float2 g10 = hash2(fmod(i + float2(1, 0), period)) * 2.0 - 1.0;
    float2 g01 = hash2(fmod(i + float2(0, 1), period)) * 2.0 - 1.0;
    float2 g11 = hash2(fmod(i + float2(1, 1), period)) * 2.0 - 1.0;
    
    float n00 = dot(g00, f - float2(0, 0));
    float n10 = dot(g10, f - float2(1, 0));
    float n01 = dot(g01, f - float2(0, 1));
    float n11 = dot(g11, f - float2(1, 1));
    
    return lerp(lerp(n00, n10, u.x), lerp(n01, n11, u.x), u.y) * 0.5 + 0.5;
}

float perlinFBM(float2 p, float period, int octaves)
{
    float value = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < octaves; i++)
    {
        value += amp * perlinNoise(p * freq, period * freq);
        amp *= 0.5;
        freq *= 2.0;
    }
    return value;
}

// ═══════════════════════════════════════════════════
// Worley noise (tileable)
// ═══════════════════════════════════════════════════

float worleyNoise(float2 p, float period)
{
    float2 cell = floor(p);
    float2 f = frac(p);
    float minDist = 1e10;
    
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++)
    {
        float2 neighbor = float2(x, y);
        float2 wrapped = fmod(cell + neighbor + period, period);
        float2 pt = hash2(wrapped);
        float2 diff = neighbor + pt - f;
        float d = length(diff);
        minDist = min(minDist, d);
    }
    
    return minDist;
}

// Multi-octave Worley FBM — produces organic cellular patterns
// Each octave adds finer Worley detail, creating varied bubble sizes
float worleyFBM(float2 p, float basePeriod, int octaves)
{
    float value = 0.0;
    float amp = 0.625;          // stronger first octave
    float freq = 1.0;
    float periodScale = 1.0;
    
    for (int i = 0; i < octaves; i++)
    {
        float w = worleyNoise(p * freq, basePeriod * periodScale);
        value += amp * (1.0 - w); // invert: cell centers = 1, edges = 0
        amp *= 0.25;              // rapid falloff so fine detail is subtle
        freq *= 3.0;              // jump scale for visible multi-size effect
        periodScale *= 3.0;
    }
    return saturate(value);
}

// ═══════════════════════════════════════════════════
// Main kernel
// ═══════════════════════════════════════════════════

[numthreads(8, 8, 1)]
void CSGenerateNoise(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float4> output = ResourceDescriptorHeap[OutputIdx];
    uint size = TexSize;
    if (dtid.x >= size || dtid.y >= size) return;
    
    float2 uv = (float2(dtid.xy) + 0.5) / float(size);
    
    // R: Perlin FBM — smooth organic shapes for foam mask / shape placement
    //    5 cells across => ~51px per cell at 256×256
    float r = perlinFBM(uv * 5.0, 5.0, 5);
    
    // G: Worley FBM — multi-scale cellular for bubbly foam detail
    //    Base: 8 cells => ~32px per cell, then 24, 72 from octaves
    //    Multi-octave creates varied bubble sizes (large + small)
    float g = worleyFBM(uv * 8.0, 8.0, 3);
    
    // B: Perlin-Worley blend — organic shapes with cellular interior texture
    //    Best of both: smooth boundaries from Perlin, cellular detail from Worley
    float perlin = perlinFBM(uv * 4.0, 4.0, 4);
    float worley = worleyFBM(uv * 6.0, 6.0, 3);
    float b = saturate(perlin * 0.6 + worley * 0.4);
    
    // A: High-freq Perlin FBM — fine edge breakup and turbulence
    float a = perlinFBM(uv * 12.0, 12.0, 4);
    
    output[dtid.xy] = float4(r, g, b, a);
}
