using System;
using System.Numerics;

namespace Freefall.Components
{
    /// <summary>
    /// Cross-sections of an open spline, for anything built as a strip along it (RuntimeMesh roads and walls,
    /// WaterBody rivers): centre points, tangents, the horizontal right vector, arc length and the half-width
    /// to each side. All in the spline's local space.
    ///
    /// On the inside of a bend tighter than the strip is wide, cross-sections would cross each other and the
    /// strip would fold over itself. The half-width on that side is limited to the bend's radius there, so the
    /// inner edge pinches to a point instead.
    /// </summary>
    public sealed class SplineStrip
    {
        public readonly int Count;
        public readonly Vector3[] Points;
        public readonly Vector3[] Tangents;
        public readonly Vector3[] Rights;

        /// <summary>Requested half-width (width / 2 × the spline's per-point width), before bend limiting.</summary>
        public readonly float[] HalfWidths;
        public readonly float[] LeftHalfWidths;
        public readonly float[] RightHalfWidths;

        /// <summary>Distance along the centre line. Call RecomputeArcLengths after moving Points.</summary>
        public readonly float[] ArcLengths;

        private SplineStrip(int count)
        {
            Count = count;
            Points = new Vector3[count];
            Tangents = new Vector3[count];
            Rights = new Vector3[count];
            HalfWidths = new float[count];
            LeftHalfWidths = new float[count];
            RightHalfWidths = new float[count];
            ArcLengths = new float[count];
        }

        public Vector3 Left(int i) => Points[i] - Rights[i] * LeftHalfWidths[i];
        public Vector3 Right(int i) => Points[i] + Rights[i] * RightHalfWidths[i];

        /// <param name="width">Full width in meters where the spline's width is 1.</param>
        public static SplineStrip Sample(Spline spline, int segmentsPerSpan, float width)
        {
            int count = Math.Max(2, spline.SpanCount * segmentsPerSpan + 1);
            var strip = new SplineStrip(count);

            for (int i = 0; i < count; i++)
            {
                float t = (float)i / (count - 1);
                var fwd = spline.GetTangent(t);
                strip.Points[i] = spline.GetPoint(t);
                strip.Tangents[i] = fwd;
                strip.Rights[i] = Vector3.Normalize(new Vector3(-fwd.Z, 0, fwd.X));
                strip.HalfWidths[i] = width * 0.5f * spline.GetWidth(t);
            }

            strip.RecomputeArcLengths();
            strip.LimitInsideOfBends();
            return strip;
        }

        public void RecomputeArcLengths()
        {
            ArcLengths[0] = 0;
            for (int i = 1; i < Count; i++)
                ArcLengths[i] = ArcLengths[i - 1] + Vector3.Distance(Points[i], Points[i - 1]);
        }

        private void LimitInsideOfBends()
        {
            // Signed bend radius on the ground plane per cross-section: > 0 bends to the right.
            var radius = new float[Count];
            for (int i = 0; i < Count; i++)
            {
                radius[i] = float.PositiveInfinity;
                if (i == 0 || i == Count - 1) continue;

                var a = Horizontal(Tangents[i - 1]);
                var b = Horizontal(Tangents[i + 1]);
                float turn = MathF.Asin(Math.Clamp(a.X * b.Y - a.Y * b.X, -1f, 1f));
                if (MathF.Abs(turn) < 1e-4f) continue;

                float ds = Vector2.Distance(new Vector2(Points[i - 1].X, Points[i - 1].Z),
                                            new Vector2(Points[i + 1].X, Points[i + 1].Z));
                radius[i] = ds / turn;
            }

            // A cross-section also has to clear its neighbours', so take the tightest radius around it.
            const float Keep = 0.9f;
            for (int i = 0; i < Count; i++)
            {
                float left = HalfWidths[i], right = HalfWidths[i];
                for (int j = Math.Max(0, i - 2); j <= Math.Min(Count - 1, i + 2); j++)
                {
                    if (float.IsInfinity(radius[j])) continue;
                    if (radius[j] > 0) right = MathF.Min(right, radius[j] * Keep);
                    else left = MathF.Min(left, -radius[j] * Keep);
                }
                LeftHalfWidths[i] = left;
                RightHalfWidths[i] = right;
            }

            static Vector2 Horizontal(Vector3 v)
            {
                var h = new Vector2(v.X, v.Z);
                return h.LengthSquared() > 1e-12f ? Vector2.Normalize(h) : Vector2.UnitX;
            }
        }
    }
}
