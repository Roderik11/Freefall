using System;
using System.Numerics;
using System.Security.Cryptography;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;
using Category = System.ComponentModel.CategoryAttribute;

namespace Freefall.Components
{
    /// <summary>Where a terrain stamp applies.</summary>
    public enum StampShape
    {
        /// <summary>A circle around the entity, or the corridor/area of a sibling Spline.</summary>
        Local,
        /// <summary>The whole terrain the stamp's entity is parented under. Position and radius are ignored;
        /// what limits it is the stamp's filter (slope, height, layers).</summary>
        Global,
    }

    /// <summary>
    /// Abstract base for terrain stamp components (HeightStamp, SplatStamp, DecoStamp, ...).
    /// Provides shape definition (global, radial or spline corridor/area), edge noise,
    /// priority ordering, and gizmo visualization.
    ///
    /// Stamps are the only way terrain is authored: everything a terrain shows is the result of
    /// compositing the stamps in scope in ascending <see cref="Priority"/>.
    /// </summary>
    public abstract class TerrainStamp : Component, ISceneGizmo
    {
        [Category("Shape")]
        public StampShape Shape = StampShape.Local;

        /// <summary>
        /// Radius for radial stamps (when no Spline is present).
        /// Width for spline-based stamps (half-width on each side of the path).
        /// </summary>
        public float Radius = 4f;

        /// <summary>Falloff distance (blend from full effect to none). World units.</summary>
        public float Falloff = 3f;

        /// <summary>Evaluation priority. Higher values are applied later (overwrite lower).</summary>
        public int Priority = 0;

        /// <summary>Enable noise displacement on the falloff boundary for organic edges.</summary>
        [Category("Edge Noise")]
        public bool EnableNoise = false;

        /// <summary>Noise frequency relative to influence radius (bumps per radius).</summary>
        [ValueRange(0.5f, 20f)]
        public float NoiseFrequency = 3f;

        /// <summary>Noise amplitude in world units (how far edges displace).</summary>
        public float NoiseAmplitude = 2f;

        /// <summary>Noise seed for variation between stamps.</summary>
        public int NoiseSeed = RandomNumberGenerator.GetInt32(99999);

        // ═══════════════════════════════════════════
        // ── Runtime ──
        // ═══════════════════════════════════════════

        protected override void Awake()
        {
            Transform?.OnChanged += Transform_OnChanged;
        }

        public override void Destroy()
        {
            Transform?.OnChanged -= Transform_OnChanged;
        }

        private void Transform_OnChanged()
        {
            MessageDispatcher.Send(EngineMsg.StampChanged, this);
        }

        public override void OnMemberChanged()
        {
            _splineResolved = false;
            MessageDispatcher.Send(EngineMsg.StampChanged, this);
        }

        // ── Spline resolution ──

        /// <summary>Cached sibling spline (resolved lazily).</summary>
        private Spline _cachedSpline;
        private bool _splineResolved;

        /// <summary>True if this stamp covers the whole terrain it is parented under.</summary>
        public bool IsGlobal => Shape == StampShape.Global;

        /// <summary>True if this stamp follows a spline path.</summary>
        public bool IsSplineMode
        {
            get
            {
                if (IsGlobal) return false;
                ResolveSpline();
                return _cachedSpline != null;
            }
        }

        /// <summary>
        /// The terrain a global stamp applies to: the nearest TerrainRenderer on this entity or an ancestor.
        /// Null when there is none (the stamp then affects nothing).
        /// </summary>
        public TerrainRenderer FindOwningTerrain()
        {
            for (var t = Transform; t != null; t = t.Parent)
            {
                var renderer = t.Entity?.GetComponent<TerrainRenderer>();
                if (renderer != null) return renderer;
            }
            return null;
        }

        /// <summary>
        /// Whether this stamp takes part in the given terrain's bake. Local stamps apply to any terrain
        /// (the shape decides where); global stamps only to the terrain they are parented under.
        /// </summary>
        public bool AppliesTo(TerrainRenderer renderer)
        {
            if (!Enabled || Entity == null) return false;
            return !IsGlobal || FindOwningTerrain() == renderer;
        }

        /// <summary>
        /// Bake order: ascending Priority, ties broken by a stable per-component id so the result does not
        /// depend on the order components happen to sit in the cache.
        /// </summary>
        public static int CompareBakeOrder(TerrainStamp a, TerrainStamp b)
        {
            int byPriority = a.Priority.CompareTo(b.Priority);
            return byPriority != 0 ? byPriority : a.UID.CompareTo(b.UID);
        }

        /// <summary>Get the sibling Spline component, if any.</summary>
        public Spline GetSpline()
        {
            ResolveSpline();
            return _cachedSpline;
        }

        private void ResolveSpline()
        {
            if (_splineResolved) return;
            _splineResolved = true;
            _cachedSpline = Entity?.GetComponent<Spline>();
        }

        // ── Shape query API ──

        /// <summary>
        /// Compute the stamp weight at a world-space position.
        /// Returns 0..1 where 1 = full effect, 0 = outside stamp.
        /// </summary>
        public float GetWeight(Vector3 worldPos)
        {
            if (IsGlobal) return 1f;
            float distance = GetDistance(worldPos, out float width);
            float radius = Radius * width;
            float falloff = Falloff * width;
            if (distance >= radius + falloff) return 0f;
            if (distance <= radius) return 1f;
            float t = (distance - radius) / Math.Max(falloff, 0.001f);
            return 1f - SmoothStep(t);
        }

        /// <summary>
        /// Distance from worldPos to the edge of the stamp's full-strength core (negative inside).
        /// Unlike GetDistance(p) - Radius this follows the spline's per-point width.
        /// </summary>
        public float GetEdgeDistance(Vector3 worldPos)
        {
            float distance = GetDistance(worldPos, out float width);
            return distance - Radius * width;
        }

        /// <summary>
        /// Get the minimum distance from worldPos to the stamp shape.
        /// </summary>
        public float GetDistance(Vector3 worldPos) => GetDistance(worldPos, out _);

        /// <summary>
        /// As GetDistance, also returning the spline's width multiplier at the nearest point
        /// (1 for radial stamps): Radius and Falloff are scaled by it there.
        /// </summary>
        public float GetDistance(Vector3 worldPos, out float width)
        {
            width = 1f;
            if (IsSplineMode)
                return GetDistanceToSpline(worldPos, out width);

            var center = Transform?.WorldPosition ?? Vector3.Zero;
            float dx = worldPos.X - center.X;
            float dz = worldPos.Z - center.Z;
            return MathF.Sqrt(dx * dx + dz * dz);
        }

        /// <summary>
        /// Get world-space AABB covering the full stamp zone.
        /// </summary>
        public BoundingBox GetWorldBounds()
        {
            float extent = Radius + Falloff;

            if (IsSplineMode)
            {
                extent *= _cachedSpline.MaxWidth;
                var min = new Vector3(float.MaxValue);
                var max = new Vector3(float.MinValue);
                int samples = Math.Max(8, _cachedSpline.TotalSegments);
                for (int i = 0; i <= samples; i++)
                {
                    float t = (float)i / samples;
                    var p = _cachedSpline.GetWorldPoint(t);
                    min = Vector3.Min(min, p - new Vector3(extent));
                    max = Vector3.Max(max, p + new Vector3(extent));
                }
                return new BoundingBox(min, max);
            }

            var center = Transform?.WorldPosition ?? Vector3.Zero;
            return new BoundingBox(
                center - new Vector3(extent, extent * 2f, extent),
                center + new Vector3(extent, extent * 2f, extent));
        }

        // ── Spline helpers ──

        /// <summary>
        /// Get the target height at a world-space position.
        /// For radial mode, this is entity.Y + heightOffset.
        /// For spline mode, this is the spline's interpolated Y at nearest point + heightOffset.
        /// </summary>
        protected float GetTargetHeight(Vector3 worldPos, float heightOffset)
        {
            if (IsSplineMode)
            {
                float nearestT = FindNearestT(worldPos, 32);
                var splinePoint = _cachedSpline.GetWorldPoint(nearestT);
                return splinePoint.Y + heightOffset;
            }
            return (Transform?.WorldPosition.Y ?? 0f) + heightOffset;
        }

        private float GetDistanceToSpline(Vector3 worldPos, out float width)
        {
            float nearestT = FindNearestT(worldPos, 64);
            width = _cachedSpline.GetWidth(nearestT);

            // Closed splines are filled areas on the GPU (terrain_stamp_overlay / height bake): match that here
            if (_cachedSpline.Closed && IsInsideClosedSpline(worldPos))
                return 0f;

            var nearestPoint = _cachedSpline.GetWorldPoint(nearestT);
            float dx = worldPos.X - nearestPoint.X;
            float dz = worldPos.Z - nearestPoint.Z;
            return MathF.Sqrt(dx * dx + dz * dz);
        }

        private bool IsInsideClosedSpline(Vector3 worldPos)
        {
            const int samples = 64;
            bool inside = false;
            var prev = _cachedSpline.GetWorldPoint(1f);
            for (int i = 0; i < samples; i++)
            {
                var cur = _cachedSpline.GetWorldPoint((float)i / samples);
                if ((cur.Z > worldPos.Z) != (prev.Z > worldPos.Z) &&
                    worldPos.X < (prev.X - cur.X) * (worldPos.Z - cur.Z) / (prev.Z - cur.Z) + cur.X)
                    inside = !inside;
                prev = cur;
            }
            return inside;
        }

        private float FindNearestT(Vector3 worldPos, int samples)
        {
            float bestT = 0f;
            float bestDist = float.MaxValue;

            for (int i = 0; i <= samples; i++)
            {
                float t = (float)i / samples;
                var p = _cachedSpline.GetWorldPoint(t);
                float dx = worldPos.X - p.X;
                float dz = worldPos.Z - p.Z;
                float dist = dx * dx + dz * dz;
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestT = t;
                }
            }

            float step = 1f / samples;
            for (int iter = 0; iter < 4; iter++)
            {
                step *= 0.5f;
                float tA = Math.Max(0f, bestT - step);
                float tB = Math.Min(1f, bestT + step);

                var pA = _cachedSpline.GetWorldPoint(tA);
                var pB = _cachedSpline.GetWorldPoint(tB);

                float dA = (worldPos.X - pA.X) * (worldPos.X - pA.X) + (worldPos.Z - pA.Z) * (worldPos.Z - pA.Z);
                float dB = (worldPos.X - pB.X) * (worldPos.X - pB.X) + (worldPos.Z - pB.Z) * (worldPos.Z - pB.Z);

                bestT = dA < dB ? tA : tB;
            }

            return bestT;
        }

        private static float SmoothStep(float t)
        {
            t = Math.Clamp(t, 0f, 1f);
            return t * t * (3f - 2f * t);
        }

        // ═════════════════════════════════
        // ── Gizmo Visualization ──
        // ═════════════════════════════════

        public void DrawGizmos(GizmoContext ctx)
        {
            // A global stamp has no shape of its own to show
            if (IsGlobal) return;

            if (IsSplineMode)
            {
                DrawSplineGizmo(ctx);
                return;
            }

            DrawRadialGizmo(ctx);

            // The parent's outline confines the circle; the stamp is at full strength from Falloff inside it on
            // (unscaled by the spline's widths, as the bake's mask samples carry none)
            if (GizmoMaskSpline is { } mask && mask.Points.Count >= 2)
                DrawOutline(ctx, mask, 0f, Falloff > 0.01f ? -Falloff : null, scaleWithWidth: false);
        }

        private void DrawRadialGizmo(GizmoContext ctx)
        {
            ctx.Color = GizmoColor;
            ctx.LineWidth = 1.5f;
            ctx.DrawCircle(Vector3.Zero, Vector3.UnitY, Radius, 48);

            if (Falloff > 0.01f)
            {
                ctx.Color = new Color4(GizmoColor.R, GizmoColor.G, GizmoColor.B, 0.5f);
                ctx.LineWidth = 1f;
                ctx.DrawCircle(Vector3.Zero, Vector3.UnitY, Radius + Falloff, 48);
            }

            ctx.Color = new Color4(GizmoColor.R, GizmoColor.G, GizmoColor.B, 1f);
            Radius = ctx.RadiusHandle(Vector3.Zero, Radius);
        }

        private void DrawSplineGizmo(GizmoContext ctx)
        {
            var spline = _cachedSpline;
            if (spline == null || spline.Points.Count < 2) return;

            // Closed: an area whose edge is the outline grown by Radius. A height stamp's falloff lies inside that
            // edge, a coverage stamp's outside it (EvaluateStampWeight insetFalloff). Open: a corridor, falloff outside.
            float falloffLine = spline.Closed && FalloffInsideClosedSpline ? Radius - Falloff : Radius + Falloff;
            DrawOutline(ctx, spline, Radius, Falloff > 0.01f ? falloffLine : null, scaleWithWidth: true);
        }

        /// <summary>
        /// Lines along a spline at offsets from it (m): the edge (bright) and the falloff's other end (faint), with
        /// hatches between them at the control points. On a closed outline an offset is outward (negative = inside)
        /// and only that side is drawn; an open spline gets both sides.
        /// </summary>
        private void DrawOutline(GizmoContext ctx, Spline spline, float edge, float? falloffLine, bool scaleWithWidth)
        {
            int samples = spline.TotalSegments;
            var savedMatrix = ctx.Matrix;
            ctx.Matrix = Matrix4x4.Identity;

            // 'perp' is the left side of travel: the inside of a counter-clockwise outline
            float outward = spline.Closed && SignedAreaXZ(spline, samples) > 0f ? -1f : 1f;
            bool bothSides = !spline.Closed;
            var rotation = Matrix4x4.CreateFromQuaternion(spline.Transform?.Rotation ?? Quaternion.Identity);

            Vector3 Perp(float t)
            {
                var tangent = Vector3.Normalize(Vector3.TransformNormal(spline.GetTangent(t), rotation));
                return Vector3.Normalize(new Vector3(-tangent.Z, 0, tangent.X)) * outward;
            }

            Vector3 prevEdge = default, prevEdgeB = default, prevFall = default, prevFallB = default;
            for (int i = 0; i <= samples; i++)
            {
                float t = (float)i / samples;
                var point = spline.GetWorldPoint(t);
                var perp = Perp(t);
                float width = scaleWithWidth ? spline.GetWidth(t) : 1f;

                var edgeA = point + perp * (edge * width);
                var edgeB = point - perp * (edge * width);
                if (i > 0)
                {
                    ctx.Color = GizmoColor;
                    ctx.LineWidth = 1.5f;
                    ctx.DrawLine(prevEdge, edgeA);
                    if (bothSides) ctx.DrawLine(prevEdgeB, edgeB);
                }
                prevEdge = edgeA;
                prevEdgeB = edgeB;

                if (falloffLine is float fl)
                {
                    var fallA = point + perp * (fl * width);
                    var fallB = point - perp * (fl * width);
                    if (i > 0)
                    {
                        ctx.Color = new Color4(GizmoColor.R, GizmoColor.G, GizmoColor.B, 0.4f);
                        ctx.LineWidth = 1f;
                        ctx.DrawLine(prevFall, fallA);
                        if (bothSides) ctx.DrawLine(prevFallB, fallB);
                    }
                    prevFall = fallA;
                    prevFallB = fallB;
                }
            }

            // Cross-hatches at control points, across the band between the outline, the edge and the falloff line
            ctx.Color = new Color4(GizmoColor.R, GizmoColor.G, GizmoColor.B, 0.6f);
            ctx.LineWidth = 1f;
            float far = falloffLine ?? edge;
            for (int i = 0; i < spline.Points.Count; i++)
            {
                float t = spline.Points.Count > 1 ? (float)i / (spline.Points.Count - 1) : 0f;
                var point = spline.GetWorldPoint(t);
                var perp = Perp(Math.Clamp(t, 0.001f, 0.999f));
                float width = scaleWithWidth ? spline.GetPointWidth(i) : 1f;

                if (bothSides)
                {
                    float r = MathF.Max(MathF.Abs(edge), MathF.Abs(far)) * width;
                    ctx.DrawLine(point - perp * r, point + perp * r);
                }
                else
                {
                    float lo = MathF.Min(0f, MathF.Min(edge, far)) * width;
                    float hi = MathF.Max(0f, MathF.Max(edge, far)) * width;
                    ctx.DrawLine(point + perp * lo, point + perp * hi);
                }
            }

            ctx.Matrix = savedMatrix;
        }

        /// <summary>Shoelace area of the spline's world outline seen from above; positive = counter-clockwise in X/Z.</summary>
        private static float SignedAreaXZ(Spline spline, int samples)
        {
            float area = 0f;
            var prev = spline.GetWorldPoint(0f);
            for (int i = 1; i <= samples; i++)
            {
                var cur = spline.GetWorldPoint((float)i / samples);
                area += prev.X * cur.Z - cur.X * prev.Z;
                prev = cur;
            }
            return area * 0.5f;
        }

        /// <summary>Override in subclasses for distinct gizmo colors.</summary>
        protected virtual Color4 GizmoColor => new Color4(0.3f, 0.9f, 0.3f, 1f);

        /// <summary>True where a closed spline's falloff lies inside its edge (height stamps); coverage stamps fade outside it.</summary>
        protected virtual bool FalloffInsideClosedSpline => false;

        /// <summary>A closed spline confining this round stamp, drawn with it (HeightStamp.MaskSpline).</summary>
        protected virtual Spline? GizmoMaskSpline => null;
    }
}
