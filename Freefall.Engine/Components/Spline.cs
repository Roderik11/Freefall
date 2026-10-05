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

        /// <summary>
        /// Width multiplier per control point (parallel to Points; missing entries count as 1).
        /// Everything built along the spline scales its own width by it: stamp radius and falloff,
        /// RuntimeMesh strips. A river that widens from spring to mouth is one spline.
        /// Edit in the scene view by dragging a point with Alt held.
        /// </summary>
        public List<float> Widths = new();

        /// <summary>If true, the spline forms a closed loop.</summary>
        [System.ComponentModel.DefaultValue(false)]
        public bool Closed = false;

        /// <summary>Catmull-Rom tension. 0.5 = standard, 0 = loose, 1 = tight.</summary>
        [System.ComponentModel.DefaultValue(0.5f)]
        [ValueRange(0.001f, 2f)]
        public float Tension = 0.5f;

        /// <summary>Number of line segments per span for gizmo drawing.</summary>
        [System.ComponentModel.DefaultValue(16)]
        [System.ComponentModel.Browsable(false)]
        public int Resolution = 16;

        /// <summary>Number of spans (segments between control points).</summary>
        public int SpanCount => Closed ? Points.Count : Math.Max(0, Points.Count - 1);

        /// <summary>Inspector / command-server edits (Points, Closed, Tension) must reach stamps, RuntimeMesh and PCG.</summary>
        public override void OnMemberChanged() => MessageDispatcher.Send(EngineMsg.SplineChanged, this);

        /// <summary>Total number of evaluable points (spans * resolution).</summary>
        public int TotalSegments => SpanCount * Resolution;

        /// <summary>
        /// Moves the entity's pivot to the XZ centre and average height of the control points (a Flat RuntimeMesh
        /// keeps its height — it is built at entity Y) without moving anything in the world: the control points and
        /// the entity's hand-made children are compensated. Makes splines authored far from their pivot (e.g. entity at
        /// the origin, world coordinates in the points) movable and rotatable by their transform.
        /// Returns the world-space distance the pivot moved (0 if it was already within <paramref name="tolerance"/>).
        /// </summary>
        public float CenterPivot(float tolerance = 0.01f)
        {
            var t = Transform;
            if (t == null || Points.Count == 0) return 0f;

            var world = t.Matrix;
            float minX = float.MaxValue, maxX = float.MinValue, minZ = float.MaxValue, maxZ = float.MinValue, sumY = 0f;
            foreach (var p in Points)
            {
                var w = Vector3.Transform(p, world);
                minX = MathF.Min(minX, w.X); maxX = MathF.Max(maxX, w.X);
                minZ = MathF.Min(minZ, w.Z); maxZ = MathF.Max(maxZ, w.Z);
                sumY += w.Y;
            }

            // A Flat RuntimeMesh is built at entity Y, so its pivot height is meaningful; everything else reads the
            // (compensated) world points, so the pivot also goes to their average height — not buried under terrain.
            var runtimeMesh = Entity?.GetComponent<RuntimeMesh>();
            bool keepHeight = runtimeMesh != null && runtimeMesh.HeightMode == RuntimeMeshHeightMode.Flat;

            var oldPivot = world.Translation;
            var shift = new Vector3((minX + maxX) * 0.5f - oldPivot.X,
                                    keepHeight ? 0f : sumY / Points.Count - oldPivot.Y,
                                    (minZ + maxZ) * 0.5f - oldPivot.Z);
            if (shift.Length() < tolerance) return 0f;

            // The same shift expressed in this entity's local frame (rotation/scale) and in its parent's frame.
            if (!Matrix4x4.Invert(world, out var worldInv)) return 0f;
            var localShift = Vector3.TransformNormal(shift, worldInv);
            var parentShift = shift;
            if (t.Parent != null && Matrix4x4.Invert(t.Parent.Matrix, out var parentInv))
                parentShift = Vector3.TransformNormal(shift, parentInv);

            for (int i = 0; i < Points.Count; i++)
                Points[i] -= localShift;

            // Keep hand-made children in place; generated output (PCG, DontSave) is rebuilt from the spline anyway.
            for (int i = 0; i < t.GetChildCount(); i++)
            {
                var child = t.GetChild(i);
                if (child?.Entity != null && (child.Entity.Flags & EntityFlags.DontSave) == 0)
                    child.Position -= localShift;
            }

            t.Position += parentShift;
            OnMemberChanged();
            return shift.Length();
        }

        // ═══════════════════════════
        // ── Width ──
        // ═══════════════════════════

        /// <summary>Width multiplier of control point <paramref name="index"/> (1 when not set).</summary>
        public float GetPointWidth(int index)
            => index >= 0 && index < Widths.Count ? Widths[index] : 1f;

        /// <summary>Set the width multiplier of a control point (pads the list with 1s as needed).</summary>
        public void SetPointWidth(int index, float width)
        {
            if (index < 0 || index >= Points.Count) return;
            while (Widths.Count <= index) Widths.Add(1f);
            Widths[index] = MathF.Max(0.02f, width);
        }

        /// <summary>True when at least one point has a width other than 1.</summary>
        public bool HasWidths
        {
            get
            {
                for (int i = 0; i < Widths.Count && i < Points.Count; i++)
                    if (MathF.Abs(Widths[i] - 1f) > 0.001f) return true;
                return false;
            }
        }

        /// <summary>Largest width multiplier along the spline (for bounds).</summary>
        public float MaxWidth
        {
            get
            {
                float max = 1f;
                for (int i = 0; i < Widths.Count && i < Points.Count; i++)
                    max = MathF.Max(max, Widths[i]);
                return max;
            }
        }

        /// <summary>
        /// Width multiplier at t ∈ [0, 1], eased between the two neighbouring control points
        /// (no overshoot: it never leaves the range of those two values).
        /// </summary>
        public float GetWidth(float t)
        {
            if (Widths.Count == 0 || Points.Count < 2) return 1f;

            int spans = SpanCount;
            t = Math.Clamp(t, 0f, 1f) * spans;
            int span = (int)t;
            if (span >= spans) span = spans - 1;
            float local = t - span;

            int n = Points.Count;
            float w0 = GetPointWidth(span % n);
            float w1 = GetPointWidth(Closed ? (span + 1) % n : Math.Min(n - 1, span + 1));
            float s = local * local * (3f - 2f * local);
            return w0 + (w1 - w0) * s;
        }

        /// <summary>
        /// Half-width in meters that a width of 1 stands for on this entity: the widest thing built
        /// along the spline. Only used to make Alt-dragging a point feel 1:1 (drag to where the edge
        /// should be).
        /// </summary>
        private float WidthReference()
        {
            float reference = 0f;
            if (Entity != null)
            {
                var height = Entity.GetComponent<HeightStamp>();
                if (height != null) reference = MathF.Max(reference, height.Radius);
                var splat = Entity.GetComponent<SplatStamp>();
                if (splat != null) reference = MathF.Max(reference, splat.Radius);
                var mesh = Entity.GetComponent<RuntimeMesh>();
                if (mesh != null) reference = MathF.Max(reference, mesh.Width * 0.5f);
            }
            return reference > 0.01f ? reference : 1f;
        }

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

            int deletePoint = -1;
            int insertPoint = -1;

            // Draw interactive handles on each control point
            ctx.Color = new Color4(1f, 0.6f, 0.1f, 1f); // Orange
            ctx.LineWidth = 2f;
            for (int i = 0; i < Points.Count; i++)
            {
                var newPos = ctx.FreeMoveHandle(Points[i], out var clicked);
                if(clicked && Input.Shift)
                {
                    // append a new point after this one
                    insertPoint = i;
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
                    if (Input.Alt)
                    {
                        // Alt-drag: the point stays put, the distance dragged away from it (on the
                        // ground plane) becomes its half-width.
                        var offset = newPos - Points[i];
                        float dragged = MathF.Sqrt(offset.X * offset.X + offset.Z * offset.Z);
                        SetPointWidth(i, dragged / WidthReference());
                    }
                    else
                    {
                        Points[i] = newPos;
                    }
                    MessageDispatcher.Send(EngineMsg.SplineChanged, this);
                }
            }

            if(insertPoint != -1)
            {
                // Insert a new point halfway between the last two
                Vector3 newPointPos = (Points[insertPoint] + Points[Math.Max(0, insertPoint - 1)]) * 0.5f;
                if (Widths.Count > insertPoint)
                    Widths.Insert(insertPoint, (GetPointWidth(insertPoint) + GetPointWidth(Math.Max(0, insertPoint - 1))) * 0.5f);
                Points.Insert(insertPoint, newPointPos);
                MessageDispatcher.Send(EngineMsg.SplineChanged, this);
            }

            if(deletePoint != -1)
            {
                if (Widths.Count > deletePoint) Widths.RemoveAt(deletePoint);
                Points.RemoveAt(deletePoint);
                MessageDispatcher.Send(EngineMsg.SplineChanged, this);
            }

            // Width bars: across the spline at every point that has its own width (all points while
            // Alt is held, so there is something to aim at before the first drag)
            if (HasWidths || Input.Alt)
            {
                ctx.Color = new Color4(1f, 0.35f, 0.75f, 1f); // Pink
                ctx.LineWidth = 2f;
                float reference = WidthReference();
                for (int i = 0; i < Points.Count; i++)
                {
                    float t = Closed ? (float)i / Points.Count : (float)i / (Points.Count - 1);
                    var tangent = GetTangent(Math.Clamp(t, 0.001f, 0.999f));
                    var across = new Vector3(-tangent.Z, 0, tangent.X);
                    if (across.LengthSquared() < 1e-8f) continue;
                    across = Vector3.Normalize(across) * (reference * GetPointWidth(i));
                    ctx.DrawLine(Points[i] - across, Points[i] + across);
                }
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
