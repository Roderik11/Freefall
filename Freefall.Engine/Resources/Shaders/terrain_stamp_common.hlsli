// terrain_stamp_common.hlsli — Shared stamp shape evaluation for terrain stamps
// Used by: terrain_height_bake.hlsl (CS_InfluenceLayer),
//          terrain_stamp_overlay.hlsl (CS_SplatStamp, CS_DecoStamp)

#ifndef TERRAIN_STAMP_COMMON_HLSLI
#define TERRAIN_STAMP_COMMON_HLSLI

// ── Spline point (matches C# StampSplinePointGPU, 20 bytes) ──
struct StampSplinePoint
{
    float2 UV;
    float  Height;
    float  HalfWidth;
    float  Fade;            // the stamp's start / end fade here (TerrainStamp.StartFade, EndFade): 1 = none
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
//
// insetFalloff (height stamps): a closed spline is a mask. Its area is the outline grown by radius, and the falloff
// lies INSIDE that edge: full effect deeper in than the falloff, nothing past the edge. Without it (coverage stamps)
// the area is full up to the grown edge and the falloff runs outside it. Circles and open splines are not affected.
//
// outsideClosed (with insetFalloff, HeightStamp.Outside): the mask is turned inside out. The area is everything beyond
// the outline (radius moves its edge into the outline) and the falloff runs outward from that edge.

// What a walk over a spline's segments gathers for a place: the nearest distance and the blend of height, width and fade
// round it. EvaluateStampWeightEx walks every segment; the page bakes walk only the segments that can reach their page
// (terrain_coverage.hlsli CoverageShapeWeightPage) and finish the same way (StampWeightAt).
//
// Height, width and fade come from the segments nearly as near as the nearest, weighted by ((nearest + e) / (d + e))^6,
// not from the nearest alone. Taken from the nearest segment, they jumped where another segment became the nearest: on the
// inside of every bend (the bisector) and between a meander's arms, where the two feet lie far apart along the spline.
// With per-point widths (a river widening seaward) that jump in reach left steps up to 6 m in the ground beside the Great
// River's valley (TestWorld, 2026-11-07). On the spline the nearest segment rules; one at 1.5x the distance weighs under
// 0.1, at 4x 0.00024. Built up in the one walk: when a nearer segment turns up, what was gathered is rescaled to it.
struct StampWalk
{
    float bestDist;
    float accW, accH, accHalfW, accFade;
};

StampWalk StampWalkStart()
{
    StampWalk w;
    w.bestDist = 1e10;
    w.accW = 0; w.accH = 0; w.accHalfW = 0; w.accFade = 0;
    return w;
}

// One segment (p0 to p1) into the walk
void StampWalkSegment(inout StampWalk w, float2 uv, StampSplinePoint p0, StampSplinePoint p1)
{
    const float blendEps = 1e-7;    // terrain UV, ~2 mm on 16 km: keeps the ratio defined on the spline itself

    float projT;
    float2 nearest;
    float d = StampDistToSegment(uv, p0.UV, p1.UV, projT, nearest);

    if (d < w.bestDist)
    {
        float r = (d + blendEps) / (w.bestDist + blendEps);
        float r2 = r * r;
        float rescale = r2 * r2 * r2;
        w.accW *= rescale; w.accH *= rescale; w.accHalfW *= rescale; w.accFade *= rescale;
        w.bestDist = d;
    }
    float q = (w.bestDist + blendEps) / (d + blendEps);
    float q2 = q * q;
    float wt = q2 * q2 * q2;
    w.accW += wt;
    w.accH += wt * lerp(p0.Height, p1.Height, projT);
    w.accHalfW += wt * lerp(p0.HalfWidth, p1.HalfWidth, projT);
    w.accFade += wt * lerp(p0.Fade, p1.Fade, projT);
}

// Where segment a-b crosses the row through uv, in terrain U, or false when it does not: half-open in V (the lower end
// counts, the upper does not), so a ray along the row counts a vertex once. 'precise': the page bakes split the ray from uv
// to +X at a U the CPU chose, and both sides of the split must see the same crossing as the whole ray
bool StampRowCrossing(float2 uv, float2 a, float2 b, out float xHit)
{
    xHit = 0;
    if (!((a.y <= uv.y && b.y > uv.y) || (b.y <= uv.y && a.y > uv.y))) return false;
    precise float t = (uv.y - a.y) / (b.y - a.y);
    precise float x = a.x + t * (b.x - a.x);
    xHit = x;
    return true;
}

// The weight at a place from what a walk found (a spline stamp), or from the distance to the centre (a circle)
float StampWeightAt(
    float2 uv,
    float2 center,
    float radius,
    float falloff,
    float noiseFreq,
    float noiseAmp,
    uint noiseSeed,
    bool spline,
    bool isClosed,
    bool inside,
    float bestDist,
    float bestH,
    float bestHalfW,
    float bestFade,
    out float nearestHeight,
    out float nearestHalfWidth,
    bool insetFalloff,
    bool outsideClosed)
{
    float dist;
    float outlineDist = 0;      // distance to the spline itself, inside or out
    bool insideClosed = false;
    bool closedMask = false;
    float fade = 1;             // an open spline's start / end fade at the nearest point (StampSplinePoint.Fade)
    nearestHeight = 0;
    nearestHalfWidth = radius;

    if (spline)
    {
        // An open spline that fades at an end: nothing of it there, its round cap past the last point included
        if (!isClosed)
        {
            fade = bestFade;
            if (fade <= 0) return 0;
        }

        dist = (isClosed && inside) ? 0 : bestDist;
        outlineDist = bestDist;
        insideClosed = isClosed && inside;
        closedMask = isClosed && insetFalloff;
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
    float noiseMargin = (noiseFreq > 0) ? noiseAmp : 0;

    if (closedMask)
    {
        // Signed distance to the outline (negative inside); the area ends at the outline grown by radius.
        // Outside: the sign turns, so the area is what lies beyond the outline.
        float sd = insideClosed ? -outlineDist : outlineDist;
        if (outsideClosed) sd = -sd;
        if (sd >= radius + noiseMargin) return 0;

        if (noiseFreq > 0 && noiseAmp > 0)
        {
            float noiseScale = noiseFreq / max(totalRadius, 0.001);
            float2 noiseCoord = uv * noiseScale + float2(noiseSeed * 1.37, noiseSeed * 0.73);
            sd += stamp_noise(noiseCoord) * noiseAmp;
        }

        float depth = radius - sd;   // how far inside the edge
        if (depth <= 0) return 0;
        float s = saturate(depth / max(falloff, 1e-6));
        return s * s * (3.0 - 2.0 * s);
    }

    // Early out with noise margin
    if (dist >= totalRadius + noiseMargin) return 0;

    // Noise displacement on distance field; the same noise breaks up a start / end fade (strongest halfway through
    // it, none where it is complete), so a path frays out instead of thinning evenly
    if (noiseFreq > 0 && noiseAmp > 0)
    {
        float noiseScale = noiseFreq / max(totalRadius, 0.001);
        float2 noiseCoord = uv * noiseScale + float2(noiseSeed * 1.37, noiseSeed * 0.73);
        float n = stamp_noise(noiseCoord);
        dist += n * noiseAmp;
        dist = max(dist, 0);
        fade = saturate(fade + 2.0 * n * fade * (1.0 - fade));
    }

    if (dist >= totalRadius) return 0;

    // Smoothstep weight
    if (dist <= radius)
        return fade;

    float t = (dist - radius) / max(falloff, 0.001);
    return (1.0 - t * t * (3.0 - 2.0 * t)) * fade;
}

float EvaluateStampWeightEx(
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
    out float nearestHalfWidth,
    bool insetFalloff,
    bool outsideClosed)
{
    if (splinePointCount > 0 && splinePointOffset != 0xFFFFFFFF)
    {
        bool isClosed = (splinePointCount & 0x80000000) != 0;
        uint ptCount = splinePointCount & 0x7FFFFFFF;

        StampWalk w = StampWalkStart();
        bool inside = false;   // closed: the parity of a ray from uv to +X
        uint segCount = isClosed ? ptCount : (ptCount - 1);
        for (uint s = 0; s < segCount; s++)
        {
            StampSplinePoint p0 = splinePoints[splinePointOffset + s];
            StampSplinePoint p1 = splinePoints[splinePointOffset + ((s + 1) % ptCount)];
            StampWalkSegment(w, uv, p0, p1);

            float xHit;
            if (isClosed && StampRowCrossing(uv, p0.UV, p1.UV, xHit) && uv.x < xHit)
                inside = !inside;
        }

        float bestDist = w.bestDist;
        float bestH = 0;
        float bestHalfW = radius;
        float bestFade = 1;
        if (w.accW > 0)
        {
            bestH = w.accH / w.accW;
            bestHalfW = w.accHalfW / w.accW;
            bestFade = w.accFade / w.accW;
        }

        if (ptCount == 1)
        {
            StampSplinePoint p0 = splinePoints[splinePointOffset];
            bestDist = length(uv - p0.UV);
            bestH = p0.Height;
            bestHalfW = p0.HalfWidth;
            bestFade = p0.Fade;
        }

        return StampWeightAt(uv, center, radius, falloff, noiseFreq, noiseAmp, noiseSeed, true, isClosed, inside,
            bestDist, bestH, bestHalfW, bestFade, nearestHeight, nearestHalfWidth, insetFalloff, outsideClosed);
    }
    return StampWeightAt(uv, center, radius, falloff, noiseFreq, noiseAmp, noiseSeed, false, false, false,
        0, 0, radius, 1, nearestHeight, nearestHalfWidth, insetFalloff, outsideClosed);
}

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
    out float nearestHalfWidth,
    bool insetFalloff)
{
    return EvaluateStampWeightEx(uv, center, radius, falloff, splinePointOffset, splinePointCount, noiseFreq, noiseAmp,
        noiseSeed, splinePoints, nearestHeight, nearestHalfWidth, insetFalloff, false);
}

#endif // TERRAIN_STAMP_COMMON_HLSLI
