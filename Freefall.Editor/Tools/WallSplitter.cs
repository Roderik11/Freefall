using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Base;

namespace Freefall.Editor.Tools
{
    /// <summary>
    /// Crossing point where a road intersects a wall polygon.
    /// </summary>
    public struct WallCrossing
    {
        /// <summary>World-space 2D position of the intersection.</summary>
        public Vector2 Position;

        /// <summary>Arc-length parameter along the wall perimeter.</summary>
        public float WallParameter;

        /// <summary>Road width at the crossing (scaled).</summary>
        public float RoadWidth;

        /// <summary>Wall tangent direction at crossing.</summary>
        public Vector2 WallTangent;
    }

    /// <summary>
    /// Result of splitting a wall polygon at road crossings.
    /// </summary>
    public class WallSplitResult
    {
        /// <summary>Open wall segments (each is a list of scaled 2D points).</summary>
        public List<List<Vector2>> Segments = new();

        /// <summary>Gate positions and orientations.</summary>
        public List<WallCrossing> Gates = new();
    }

    /// <summary>
    /// Splits a closed wall polygon into open segments where roads cross through.
    /// Each crossing creates a gap the width of the road, and a gate placement point.
    ///
    /// Watabou data shares vertices between roads and walls — roads pass through
    /// wall vertices rather than crossing mid-edge. Detection uses both vertex
    /// coincidence and segment intersection.
    /// </summary>
    public static class WallSplitter
    {
        private const float VertexSnapDist = 0.5f;

        /// <summary>
        /// Find all road-wall crossings and split the wall into open segments.
        /// If no roads cross the wall, returns null (caller should use the wall as-is).
        /// </summary>
        public static WallSplitResult Split(List<Vector2> wallPoints, List<WatabouPolyline> roads, float scale)
        {
            int n = wallPoints.Count;
            if (n < 3) return null;

            // Compute perimeter arc lengths
            var arcLen = new float[n + 1];
            arcLen[0] = 0;
            for (int i = 1; i <= n; i++)
                arcLen[i] = arcLen[i - 1] + Vector2.Distance(wallPoints[i % n], wallPoints[i - 1]);
            float totalLen = arcLen[n];

            // Build wall vertex set for fast lookup
            var crossings = new List<WallCrossing>();

            foreach (var road in roads)
            {
                float roadWidth = road.Width * scale;

                // Check each road point against wall vertices (shared vertex = crossing)
                for (int rp = 0; rp < road.Points.Count; rp++)
                {
                    var roadPt = road.Points[rp] * scale;

                    for (int wv = 0; wv < n; wv++)
                    {
                        if (Vector2.Distance(roadPt, wallPoints[wv]) > VertexSnapDist)
                            continue;

                        // Road point coincides with wall vertex — this is a crossing.
                        // Compute wall tangent as average of adjacent edges.
                        int prev = (wv - 1 + n) % n;
                        int next = (wv + 1) % n;
                        var tangent = Vector2.Normalize(wallPoints[next] - wallPoints[prev]);

                        crossings.Add(new WallCrossing
                        {
                            Position = wallPoints[wv],
                            WallParameter = arcLen[wv],
                            RoadWidth = roadWidth,
                            WallTangent = tangent
                        });
                        break; // one match per road point
                    }
                }

                // Also check segment-segment intersection (for roads that cross mid-edge)
                for (int re = 0; re < road.Points.Count - 1; re++)
                {
                    var r0 = road.Points[re] * scale;
                    var r1 = road.Points[re + 1] * scale;

                    for (int we = 0; we < n; we++)
                    {
                        int wNext = (we + 1) % n;
                        var w0 = wallPoints[we];
                        var w1 = wallPoints[wNext];

                        if (!SegmentIntersect(w0, w1, r0, r1, out float tWall, out _))
                            continue;

                        var pos = Vector2.Lerp(w0, w1, tWall);
                        float param = arcLen[we] + tWall * Vector2.Distance(w0, w1);

                        crossings.Add(new WallCrossing
                        {
                            Position = pos,
                            WallParameter = param,
                            RoadWidth = roadWidth,
                            WallTangent = Vector2.Normalize(w1 - w0)
                        });
                    }
                }
            }

            if (crossings.Count == 0) return null;

            // Deduplicate crossings that are very close on the perimeter
            crossings.Sort((a, b) => a.WallParameter.CompareTo(b.WallParameter));
            var deduped = new List<WallCrossing> { crossings[0] };
            for (int i = 1; i < crossings.Count; i++)
            {
                float dist = crossings[i].WallParameter - deduped[^1].WallParameter;
                if (dist > 1f) // more than 1 unit apart = separate crossing
                    deduped.Add(crossings[i]);
                else if (crossings[i].RoadWidth > deduped[^1].RoadWidth)
                    deduped[^1] = crossings[i]; // keep the wider road
            }
            crossings = deduped;

            Debug.Log($"[WallSplitter] Found {crossings.Count} crossings on wall ({n} pts)");

            // Build gap intervals
            var gaps = new List<(float start, float end)>();
            foreach (var c in crossings)
            {
                float half = c.RoadWidth * 0.5f;
                gaps.Add((Mod(c.WallParameter - half, totalLen),
                          Mod(c.WallParameter + half, totalLen)));
            }

            // Extract wall segments between gaps
            var result = new WallSplitResult();
            result.Gates.AddRange(crossings);

            for (int i = 0; i < gaps.Count; i++)
            {
                float segStart = gaps[i].end;
                float segEnd = gaps[(i + 1) % gaps.Count].start;

                var segment = ExtractSegment(wallPoints, arcLen, totalLen, segStart, segEnd);
                if (segment.Count >= 2)
                    result.Segments.Add(segment);
            }

            return result;
        }

        // ═══════════════════════════════════
        // ── Segment Extraction ──
        // ═══════════════════════════════════

        private static List<Vector2> ExtractSegment(
            List<Vector2> wallPoints, float[] arcLen, float totalLen,
            float segStart, float segEnd)
        {
            int n = wallPoints.Count;
            var result = new List<Vector2>();

            result.Add(PointAtParam(wallPoints, arcLen, totalLen, segStart));

            float traversal = segEnd - segStart;
            if (traversal <= 0) traversal += totalLen;

            int startEdge = FindEdge(arcLen, n, segStart);

            for (int step = 0; step < n; step++)
            {
                int vertIdx = (startEdge + 1 + step) % n;

                float vParam = arcLen[vertIdx];
                if (vertIdx <= startEdge) vParam += totalLen;

                float distFromStart = vParam - segStart;
                if (distFromStart < 0) distFromStart += totalLen;

                if (distFromStart >= traversal) break;

                result.Add(wallPoints[vertIdx]);
            }

            var endPt = PointAtParam(wallPoints, arcLen, totalLen, segEnd);
            if (result.Count == 0 || Vector2.Distance(result[^1], endPt) > 0.01f)
                result.Add(endPt);

            return result;
        }

        // ═══════════════════════════════════
        // ── Geometry Helpers ──
        // ═══════════════════════════════════

        private static int FindEdge(float[] arcLen, int n, float param)
        {
            param = Mod(param, arcLen[n]);
            for (int i = 0; i < n; i++)
            {
                if (param >= arcLen[i] && param <= arcLen[i + 1])
                    return i;
            }
            return n - 1;
        }

        private static Vector2 PointAtParam(List<Vector2> pts, float[] arcLen, float totalLen, float param)
        {
            int n = pts.Count;
            param = Mod(param, totalLen);
            int edge = FindEdge(arcLen, n, param);
            float edgeLen = arcLen[edge + 1] - arcLen[edge];
            if (edgeLen < 1e-6f) return pts[edge];
            float t = (param - arcLen[edge]) / edgeLen;
            return Vector2.Lerp(pts[edge], pts[(edge + 1) % n], t);
        }

        private static bool SegmentIntersect(Vector2 a0, Vector2 a1, Vector2 b0, Vector2 b1,
            out float tA, out float tB)
        {
            var dA = a1 - a0;
            var dB = b1 - b0;
            float denom = dA.X * dB.Y - dA.Y * dB.X;

            tA = tB = 0;
            if (MathF.Abs(denom) < 1e-8f) return false;

            var diff = b0 - a0;
            tA = (diff.X * dB.Y - diff.Y * dB.X) / denom;
            tB = (diff.X * dA.Y - diff.Y * dA.X) / denom;

            return tA > 0.001f && tA < 0.999f && tB > 0.001f && tB < 0.999f;
        }

        private static float Mod(float x, float m)
        {
            return ((x % m) + m) % m;
        }
    }
}
