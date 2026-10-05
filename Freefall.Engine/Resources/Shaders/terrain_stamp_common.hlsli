// terrain_stamp_common.hlsli — Shared stamp shape evaluation for terrain stamps
// Used by: terrain_height_bake.hlsl (CS_InfluenceLayer),
//          terrain_stamp_overlay.hlsl (CS_SplatStamp, CS_DecoStamp)

#ifndef TERRAIN_STAMP_COMMON_HLSLI
#define TERRAIN_STAMP_COMMON_HLSLI

// ── Spline point (matches C# StampSplinePointGPU, 16 bytes) ──
struct StampSplinePoint
{
    float2 UV;
    float  Height;
    float  HalfWidth;
};

// ── Noise ──

float2 stamp_hash(float2 p)
{
    p = float2(dot(p, float2(127.1, 311.7)),
               dot(p, float2(269.5, 183.3)));
    return -1.0 + 2.0 * frac(sin(p) * 43758.5453123);
}

float stamp_noise(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    float2 u = f * f * (3.0 - 2.0 * f);

    float n00 = dot(stamp_hash(i + float2(0, 0)), f - float2(0, 0));
    float n10 = dot(stamp_hash(i + float2(1, 0)), f - float2(1, 0));
    float n01 = dot(stamp_hash(i + float2(0, 1)), f - float2(0, 1));
    float n11 = dot(stamp_hash(i + float2(1, 1)), f - float2(1, 1));

    return lerp(lerp(n00, n10, u.x), lerp(n01, n11, u.x), u.y);
}

// ── Segment distance ──

float StampDistToSegment(float2 P, float2 A, float2 B, out float projT, out float2 nearest)
{
    float2 AB = B - A;
    float lenSq = dot(AB, AB);
    if (lenSq < 1e-10)
    {
        projT = 0;
        nearest = A;
        return length(P - A);
    }
    projT = saturate(dot(P - A, AB) / lenSq);
    nearest = A + projT * AB;
    return length(P - nearest);
}

// ── Stamp weight evaluation ──
// Computes the stamp weight [0..1] at UV position for a stamp shape.
// Parameters match the common shape fields shared by all stamp descriptor structs.
// Returns: weight (0 = outside, 1 = full effect)
// Out: nearestHeight, nearestHalfWidth (for spline mode height interpolation)

float EvaluateStampWeight(
    float2 uv,
    float2 center,
    float radius,
    float falloff,
    uint splinePointOffset,
    uint splinePointCount,
    float noiseFreq,
    float noiseAmp,
    uint noiseSeed,
    StructuredBuffer<StampSplinePoint> splinePoints,
    out float nearestHeight,
    out float nearestHalfWidth)
{
    float dist;
    nearestHeight = 0;
    nearestHalfWidth = radius;

    if (splinePointCount > 0 && splinePointOffset != 0xFFFFFFFF)
    {
        bool isClosed = (splinePointCount & 0x80000000) != 0;
        uint ptCount = splinePointCount & 0x7FFFFFFF;

        float bestDist = 1e10;
        float bestH = 0;
        float bestHalfW = radius;
        bool inside = false;

        uint segCount = isClosed ? ptCount : (ptCount - 1);

        for (uint s = 0; s < segCount; s++)
        {
            uint idx0 = splinePointOffset + s;
            uint idx1 = splinePointOffset + ((s + 1) % ptCount);
            StampSplinePoint p0 = splinePoints[idx0];
            StampSplinePoint p1 = splinePoints[idx1];

            float projT;
            float2 nearest;
            float d = StampDistToSegment(uv, p0.UV, p1.UV, projT, nearest);

            if (d < bestDist)
            {
                bestDist = d;
                bestH = lerp(p0.Height, p1.Height, projT);
                bestHalfW = lerp(p0.HalfWidth, p1.HalfWidth, projT);
            }

            if (isClosed)
            {
                float2 a = p0.UV, b = p1.UV;
                if ((a.y <= uv.y && b.y > uv.y) || (b.y <= uv.y && a.y > uv.y))
                {
                    float xHit = a.x + (uv.y - a.y) / (b.y - a.y) * (b.x - a.x);
                    if (uv.x < xHit)
                        inside = !inside;
                }
            }
        }

        if (ptCount == 1)
        {
            StampSplinePoint p0 = splinePoints[splinePointOffset];
            bestDist = length(uv - p0.UV);
            bestH = p0.Height;
            bestHalfW = p0.HalfWidth;
        }

        dist = (isClosed && inside) ? 0 : bestDist;
        nearestHeight = bestH;
        nearestHalfWidth = bestHalfW;

        // Per-point spline width (Spline.Widths): the samples carry radius * width, and the falloff
        // tapers with it, so a river's banks narrow together with its bed. 'radius' is still the
        // stamp's own radius here; a zero radius (closed area with no growth) leaves the falloff alone.
        if (radius > 1e-7)
            falloff *= bestHalfW / radius;
        radius = bestHalfW;
    }
    else
    {
        dist = length(uv - center);
    }

    float totalRadius = radius + falloff;

    // Early out with noise margin
    float noiseMargin = (noiseFreq > 0) ? noiseAmp : 0;
    if (dist >= totalRadius + noiseMargin) return 0;

    // Noise displacement on distance field
    if (noiseFreq > 0 && noiseAmp > 0)
    {
        float noiseScale = noiseFreq / max(totalRadius, 0.001);
        float2 noiseCoord = uv * noiseScale + float2(noiseSeed * 1.37, noiseSeed * 0.73);
        float n = stamp_noise(noiseCoord);
        dist += n * noiseAmp;
        dist = max(dist, 0);
    }

    if (dist >= totalRadius) return 0;

    // Smoothstep weight
    if (dist <= radius)
        return 1.0;

    float t = (dist - radius) / max(falloff, 0.001);
    return 1.0 - t * t * (3.0 - 2.0 * t);
}

#endif // TERRAIN_STAMP_COMMON_HLSLI
