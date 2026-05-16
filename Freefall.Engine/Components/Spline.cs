using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// Catmull-Rom spline component with interactive gizmo handles.
    /// Control points are in local space. Evaluate with GetPoint(t) / GetTangent(t).
    /// Supports open and closed loops.
    /// </summary>
    [Icon("icon_spline.png")]
    public class Spline : Component, ISceneGizmo
    {
        /// <summary>Local-space control points.</summary>
        public List<Vector3> Points = new()
        {
            new Vector3(0, 0, -5),
            new Vector3(0, 0, 0),
            new Vector3(0, 0, 5),
        };

        /// <summary>If true, the spline forms a closed loop.</summary>
        [System.ComponentModel.DefaultValue(false)]
        public bool Closed = false;

        /// <summary>Catmull-Rom tension. 0.5 = standard, 0 = loose, 1 = tight.</summary>
        [System.ComponentModel.DefaultValue(0.5f)]
        public float Tension = 0.5f;

        /// <summary>Number of line segments per span for gizmo drawing.</summary>
        [System.ComponentModel.DefaultValue(16)]
        [System.ComponentModel.Browsable(false)]
        public int Resolution = 16;

        /// <summary>Number of spans (segments between control points).</summary>
        public int SpanCount => Closed ? Points.Count : Math.Max(0, Points.Count - 1);

        /// <summary>Total number of evaluable points (spans * resolution).</summary>
        public int TotalSegments => SpanCount * Resolution;

        // ═══════════════════════════
        // ── Evaluation API ──
        // ═══════════════════════════

        /// <summary>
        /// Evaluate a point on the spline.
        /// t ∈ [0, 1] maps across the entire spline length.
        /// Returns local-space position.
        /// </summary>
        public Vector3 GetPoint(float t)
        {
            if (Points.Count < 2) return Points.Count > 0 ? Points[0] : Vector3.Zero;

            int spans = SpanCount;
            t = Math.Clamp(t, 0f, 1f) * spans;

            int span = (int)t;
            if (span >= spans) span = spans - 1;
            float local = t - span;

            GetControlPoints(span, out var p0, out var p1, out var p2, out var p3);
            return CatmullRom(p0, p1, p2, p3, local, Tension);
        }

        /// <summary>
        /// Evaluate the tangent (forward direction) at t ∈ [0, 1].
        /// Returns normalized local-space direction.
        /// </summary>
        public Vector3 GetTangent(float t)
        {
            if (Points.Count < 2) return Vector3.UnitZ;

            int spans = SpanCount;
            t = Math.Clamp(t, 0f, 1f) * spans;

            int span = (int)t;
            if (span >= spans) span = spans - 1;
            float local = t - span;

            GetControlPoints(span, out var p0, out var p1, out var p2, out var p3);
            return Vector3.Normalize(CatmullRomDerivative(p0, p1, p2, p3, local, Tension));
        }

        /// <summary>
        /// Evaluate a world-space point on the spline at t ∈ [0, 1].
        /// </summary>
        public Vector3 GetWorldPoint(float t)
        {
            var local = GetPoint(t);
            return Transform != null ? Vector3.Transform(local, Transform.Matrix) : local;
        }

        /// <summary>
        /// Approximate total arc length by sampling.
        /// </summary>
        public float GetLength(int samples = 64)
        {
            if (Points.Count < 2) return 0f;

            float length = 0f;
            Vector3 prev = GetPoint(0f);
            for (int i = 1; i <= samples; i++)
            {
                float t = (float)i / samples;
                Vector3 curr = GetPoint(t);
                length += Vector3.Distance(prev, curr);
                prev = curr;
            }
            return length;
        }

        /// <summary>
        /// Sample evenly-spaced points along the spline.
        /// Returns world-space positions.
        /// </summary>
        public List<Vector3> SampleEvenlySpaced(float spacing)
        {
            var result = new List<Vector3>();
            if (Points.Count < 2) return result;

            float totalLength = GetLength(128);
            if (totalLength < 0.01f) return result;

            int count = Math.Max(2, (int)(totalLength / spacing));
            for (int i = 0; i <= count; i++)
            {
                float t = (float)i / count;
                result.Add(GetWorldPoint(t));
            }
            return result;
        }

        // ═══════════════════
        // ── Gizmo ──
        // ═══════════════════

        public void DrawGizmos(GizmoContext ctx)
        {
            if (Points.Count < 2) return;

            // Draw the spline curve
            ctx.Color = new Color4(0.2f, 0.8f, 1f, 1f); // Cyan
            ctx.LineWidth = 2f;

            int spans = SpanCount;
            for (int s = 0; s < spans; s++)
            {
                GetControlPoints(s, out var p0, out var p1, out var p2, out var p3);
                Vector3 prev = p1;
                for (int i = 1; i <= Resolution; i++)
                {
                    float local = (float)i / Resolution;
                    Vector3 curr = CatmullRom(p0, p1, p2, p3, local, Tension);
                    ctx.DrawLine(prev, curr);
                    prev = curr;
                }
            }

            bool newPoint = false;
            int deletePoint = -1;

            // Draw interactive handles on each control point
            ctx.Color = new Color4(1f, 0.6f, 0.1f, 1f); // Orange
            ctx.LineWidth = 2f;
            for (int i = 0; i < Points.Count; i++)
            {
                var newPos = ctx.FreeMoveHandle(Points[i], out var clicked);
                if(clicked && Input.Shift)
                {
                    // append a new point after this one
                    newPoint = true;
                    break;
                }

                if (clicked && Input.Control)
                {
                    // delete this point
                    deletePoint = i;
                    break;
                }

                if (ctx.Changed)
                {
                    Points[i] = newPos;
                    MessageDispatcher.Send(EngineMsg.SplineChanged, this);
                }
            }

            if(newPoint)
            {
                // Insert a new point halfway between the last two
                int insertIndex = Points.Count - 1;
                Vector3 newPointPos = (Points[insertIndex] + Points[Math.Max(0, insertIndex - 1)]) * 0.5f;
                Points.Insert(insertIndex, newPointPos);
                MessageDispatcher.Send(EngineMsg.SplineChanged, this);
            }

            if(deletePoint != -1)
            {
                Points.RemoveAt(deletePoint);
                MessageDispatcher.Send(EngineMsg.SplineChanged, this);
            }

            // Draw tangent indicators at control points
            ctx.Color = new Color4(0.4f, 1f, 0.4f, 1f); // Green
            ctx.LineWidth = 1f;
            for (int i = 0; i < Points.Count; i++)
            {
                float t = (Points.Count > 1) ? (float)i / (Points.Count - 1) : 0f;
                if (Closed && i == Points.Count - 1) continue;
                var tangent = GetTangent(Closed ? t : Math.Clamp(t, 0.001f, 0.999f));
                ctx.DrawRay(Points[i], tangent, 0.5f);
            }
        }

        // ══════════════════════════════
        // ── Catmull-Rom Internals ──
        // ══════════════════════════════

        /// <summary>
        /// Get the 4 control points (p0, p1, p2, p3) for a given span index.
        /// Handles boundary clamping for open splines and wrapping for closed.
        /// </summary>
        private void GetControlPoints(int span, out Vector3 p0, out Vector3 p1, out Vector3 p2, out Vector3 p3)
        {
            int n = Points.Count;

            if (Closed)
            {
                p0 = Points[((span - 1) % n + n) % n];
                p1 = Points[span % n];
                p2 = Points[(span + 1) % n];
                p3 = Points[(span + 2) % n];
            }
            else
            {
                p0 = Points[Math.Max(0, span - 1)];
                p1 = Points[span];
                p2 = Points[Math.Min(n - 1, span + 1)];
                p3 = Points[Math.Min(n - 1, span + 2)];
            }
        }

        /// <summary>Catmull-Rom interpolation between p1 and p2.</summary>
        private static Vector3 CatmullRom(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t, float tension)
        {
            float t2 = t * t;
            float t3 = t2 * t;

            // Catmull-Rom matrix with tension parameter (alpha = tension)
            float a = tension;
            return p1
                + (-a * p0 + a * p2) * t
                + (2f * a * p0 + (a - 3f) * p1 + (3f - 2f * a) * p2 - a * p3) * t2
                + (-a * p0 + (2f - a) * p1 + (a - 2f) * p2 + a * p3) * t3;
        }

        /// <summary>Catmull-Rom first derivative (tangent).</summary>
        private static Vector3 CatmullRomDerivative(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t, float tension)
        {
            float t2 = t * t;
            float a = tension;

            return (-a * p0 + a * p2)
                + 2f * (2f * a * p0 + (a - 3f) * p1 + (3f - 2f * a) * p2 - a * p3) * t
                + 3f * (-a * p0 + (2f - a) * p1 + (a - 2f) * p2 + a * p3) * t2;
        }
    }
}
