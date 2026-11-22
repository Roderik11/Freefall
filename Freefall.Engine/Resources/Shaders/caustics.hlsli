// Caustics: the sunlight focused by the waves into a moving net of bright lines. Under the water on what lies below
// (underwater_cs.hlsl), and thrown up off the sea onto ships' hulls (light_directional_cs.hlsl HullWaterLight).
// Docs: .agent/knowledge/ocean_system/underwater.md, .agent/knowledge/engine_internals/ships.md
#ifndef CAUSTICS_HLSLI
#define CAUSTICS_HLSLI

float CausticHash(float2 p)
{
    return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

// Cells round points that sway on their own circles: the distance to the nearest border between two cells (F2 − F1), 0
// on the borders, where the net's lines are
float CausticCells(float2 p, float t)
{
    float2 i = floor(p), fp = frac(p);
    float d1 = 8.0, d2 = 8.0;
    [unroll] for (int y = -1; y <= 1; y++)
    [unroll] for (int x = -1; x <= 1; x++)
    {
        float2 g = float2(x, y);
        float h = CausticHash(i + g);
        float a = 6.2831853 * h + t * (0.6 + 0.5 * CausticHash(i + g + 7.1));
        float2 r = g + 0.5 + 0.35 * float2(cos(a), sin(a)) - fp;
        float d = dot(r, r);
        if (d < d1) { d2 = d1; d1 = d; }
        else if (d < d2) d2 = d;
    }
    return sqrt(d2) - sqrt(d1);
}

// The cells' space bent by slow waves, so their borders curve as the lines of light a rippled surface throws do (straight
// borders read as paving)
float2 CausticWarp(float2 p, float t)
{
    return p + 0.28 * float2(sin(p.y * 1.7 + t * 0.6) + 0.5 * sin(p.y * 3.1 - t * 0.9),
                             sin(p.x * 1.9 - t * 0.5) + 0.5 * sin(p.x * 2.7 + t * 0.8));
}

// Two nets of thin lines a little over a metre across, drifting apart, brightest where they cross; about CAUSTICS_MEAN on
// average. p: where the sunlight that reaches the place came in through the surface (world xz)
float Caustics(float2 p, float t)
{
    float2 pa = CausticWarp(p / 1.5, t);
    float2 pb = CausticWarp(p / 0.95 + 5.3, t * 1.2);
    float a = exp(-22.0 * CausticCells(pa, t * 0.6));
    float b = exp(-22.0 * CausticCells(pb, t * 0.8));
    return 0.8 * (a + b) + 1.5 * a * b;
}
#define CAUSTICS_MEAN 0.2

#endif
