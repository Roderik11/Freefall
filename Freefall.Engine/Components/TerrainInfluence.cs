using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// Influence modes — what effect this influence has on the terrain.
    /// </summary>
    [Flags]
    public enum TerrainInfluenceMode
    {
        None = 0,
        /// <summary>Flatten terrain height to match the influence shape.</summary>
        FlattenHeight = 1 << 0,
        /// <summary>Paint a specific splat layer within the influence zone.</summary>
        PaintSplat = 1 << 1,
        /// <summary>Suppress decorations within the influence zone.</summary>
        SuppressDecorations = 1 << 2,
    }

    /// <summary>
    /// Non-destructive terrain influence. Attach to an entity to modify terrain
    /// height, splatmap, and/or decorations within the influence footprint.
    /// 
    /// Two shape modes:
    /// - Radial: uses entity position + Radius (for buildings, trees, etc.)
    /// - Spline: follows a sibling Spline component + Width (for roads, trails)
    /// 
    /// The TerrainRenderer queries all TerrainInfluence instances during its
    /// GPU bake passes to apply non-destructive modifications.
    /// </summary>
    public class TerrainInfluence : Component, ISceneGizmo
    {
        // ── Influence Mode ──

        /// <summary>What effects to apply.</summary>
        public TerrainInfluenceMode Mode = TerrainInfluenceMode.FlattenHeight
                                          | TerrainInfluenceMode.SuppressDecorations;

        // ── Shape ──

        /// <summary>
        /// Radius for radial influence (when no Spline is present).
        /// Width for spline-based influence (half-width on each side of the path).
        /// </summary>
        [ValueRange(0.1f, 200f)]
        public float Radius = 10f;

        /// <summary>Falloff distance (blend from full effect to none). World units.</summary>
        [ValueRange(0f, 50f)]
        public float Falloff = 3f;

        // ── Height Flatten ──

        /// <summary>
        /// Height offset from the entity/spline position.
        /// For spline mode, this offsets the flattened height from the spline point's Y.
        /// </summary>
        public float HeightOffset = 0f;

        // ── Splat Paint ──

        /// <summary>Layer index to paint within the influence zone.</summary>
        public int SplatLayerIndex = 0;

        /// <summary>Paint strength (0-1) at full influence.</summary>
        [ValueRange(0f, 1f)]
        public float SplatStrength = 1f;

        // ── Decoration Suppression ──

        /// <summary>Decoration suppression strength (0-1) at full influence.</summary>
        [ValueRange(0f, 1f)]
        public float DecoSuppression = 1f;

        // ═══════════════════════════════════════════
        // ── Runtime: Shape Query API ──
        // ═══════════════════════════════════════════

        /// <summary>Cached sibling spline (resolved lazily).</summary>
        private Spline _cachedSpline;
        private bool _splineResolved;

        /// <summary>True if this influence follows a spline path.</summary>
        public bool IsSplineMode
        {
            get
            {
                ResolveSpline();
                return _cachedSpline != null;
            }
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

        /// <summary>
        /// Compute the influence weight at a world-space position.
        /// Returns 0..1 where 1 = full effect, 0 = outside influence.
        /// </summary>
        public float GetWeight(Vector3 worldPos)
        {
            float distance = GetDistance(worldPos);
            if (distance >= Radius + Falloff) return 0f;
            if (distance <= Radius) return 1f;
            // Smooth falloff
            float t = (distance - Radius) / Math.Max(Falloff, 0.001f);
            return 1f - SmoothStep(t);
        }

        /// <summary>
        /// Get the target height at a world-space position.
        /// For radial mode, this is entity.Y + HeightOffset.
        /// For spline mode, this is spline's interpolated Y at nearest point + HeightOffset.
        /// </summary>
        public float GetTargetHeight(Vector3 worldPos)
        {
            if (IsSplineMode)
            {
                float nearestT = FindNearestT(worldPos, 32);
                var splinePoint = _cachedSpline.GetWorldPoint(nearestT);
                return splinePoint.Y + HeightOffset;
            }
            return (Transform?.WorldPosition.Y ?? 0f) + HeightOffset;
        }

        /// <summary>
        /// Get the minimum distance from worldPos to the influence shape.
        /// </summary>
        public float GetDistance(Vector3 worldPos)
        {
            if (IsSplineMode)
            {
                return GetDistanceToSpline(worldPos);
            }

            // Radial: distance from entity center (XZ plane)
            var center = Transform?.WorldPosition ?? Vector3.Zero;
            float dx = worldPos.X - center.X;
            float dz = worldPos.Z - center.Z;
            return MathF.Sqrt(dx * dx + dz * dz);
        }

        /// <summary>
        /// Get world-space AABB covering the full influence zone.
        /// Used for coarse culling before per-pixel evaluation.
        /// </summary>
        public BoundingBox GetWorldBounds()
        {
            float extent = Radius + Falloff;

            if (IsSplineMode)
            {
                // Walk the spline and expand bounds
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

            // Radial
            var center = Transform?.WorldPosition ?? Vector3.Zero;
            return new BoundingBox(
                center - new Vector3(extent, extent * 2f, extent),
                center + new Vector3(extent, extent * 2f, extent));
        }

        // ── Spline helpers ──

        private float GetDistanceToSpline(Vector3 worldPos)
        {
            // Find minimum XZ distance to spline by sampling
            float nearestT = FindNearestT(worldPos, 64);
            var nearestPoint = _cachedSpline.GetWorldPoint(nearestT);
            float dx = worldPos.X - nearestPoint.X;
            float dz = worldPos.Z - nearestPoint.Z;
            return MathF.Sqrt(dx * dx + dz * dz);
        }

        /// <summary>Find the parameter t that gives the nearest spline point (XZ distance).</summary>
        private float FindNearestT(Vector3 worldPos, int samples)
        {
            float bestT = 0f;
            float bestDist = float.MaxValue;

            // Coarse pass
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

            // Refine with binary search
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
            if (IsSplineMode)
                DrawSplineInfluence(ctx);
            else
                DrawRadialInfluence(ctx);
        }

        private void DrawRadialInfluence(GizmoContext ctx)
        {
            // Inner radius (full effect)
            ctx.Color = new Color4(0.3f, 0.9f, 0.3f, 1f);
            ctx.LineWidth = 1.5f;
            ctx.DrawCircle(Vector3.Zero, Vector3.UnitY, Radius, 48);

            // Outer radius (falloff edge)
            if (Falloff > 0.01f)
            {
                ctx.Color = new Color4(0.3f, 0.9f, 0.3f, 0.5f);
                ctx.LineWidth = 1f;
                ctx.DrawCircle(Vector3.Zero, Vector3.UnitY, Radius + Falloff, 48);
            }

            // Radius handle
            ctx.Color = new Color4(0.3f, 1f, 0.3f, 1f);
            Radius = ctx.RadiusHandle(Vector3.Zero, Radius);
        }

        private void DrawSplineInfluence(GizmoContext ctx)
        {
            var spline = _cachedSpline;
            if (spline == null || spline.Points.Count < 2) return;

            // Draw left and right offset curves showing the influence corridor
            int samples = spline.TotalSegments;
            var savedMatrix = ctx.Matrix;

            // We draw in world space for spline influence
            ctx.Matrix = Matrix4x4.Identity;

            ctx.Color = new Color4(0.3f, 0.9f, 0.3f, 1f);
            ctx.LineWidth = 1.5f;

            Vector3 prevLeft = Vector3.Zero, prevRight = Vector3.Zero;
            Vector3 prevOuterLeft = Vector3.Zero, prevOuterRight = Vector3.Zero;
            bool hasFalloff = Falloff > 0.01f;

            for (int i = 0; i <= samples; i++)
            {
                float t = (float)i / samples;
                var point = spline.GetWorldPoint(t);
                var tangent = spline.GetTangent(t);

                // Transform tangent to world space rotation
                if (Transform != null)
                {
                    var rotMatrix = Matrix4x4.CreateFromQuaternion(Transform.Rotation);
                    tangent = Vector3.TransformNormal(tangent, rotMatrix);
                }
                tangent = Vector3.Normalize(tangent);

                // Perpendicular in XZ plane
                var perp = Vector3.Normalize(new Vector3(-tangent.Z, 0, tangent.X));

                var left = point + perp * Radius;
                var right = point - perp * Radius;

                if (i > 0)
                {
                    ctx.Color = new Color4(0.3f, 0.9f, 0.3f, 1f);
                    ctx.LineWidth = 1.5f;
                    ctx.DrawLine(prevLeft, left);
                    ctx.DrawLine(prevRight, right);
                }

                // Outer falloff edges
                if (hasFalloff)
                {
                    var outerLeft = point + perp * (Radius + Falloff);
                    var outerRight = point - perp * (Radius + Falloff);

                    if (i > 0)
                    {
                        ctx.Color = new Color4(0.3f, 0.9f, 0.3f, 0.4f);
                        ctx.LineWidth = 1f;
                        ctx.DrawLine(prevOuterLeft, outerLeft);
                        ctx.DrawLine(prevOuterRight, outerRight);
                    }

                    prevOuterLeft = outerLeft;
                    prevOuterRight = outerRight;
                }

                prevLeft = left;
                prevRight = right;
            }

            // Draw cross-hatches at each control point
            ctx.Color = new Color4(0.3f, 0.9f, 0.3f, 0.6f);
            ctx.LineWidth = 1f;
            for (int i = 0; i < spline.Points.Count; i++)
            {
                float t = spline.Points.Count > 1 ? (float)i / (spline.Points.Count - 1) : 0f;
                var point = spline.GetWorldPoint(t);
                var tangent = spline.GetTangent(Math.Clamp(t, 0.001f, 0.999f));
                if (Transform != null)
                {
                    var rotMatrix = Matrix4x4.CreateFromQuaternion(Transform.Rotation);
                    tangent = Vector3.TransformNormal(tangent, rotMatrix);
                }
                tangent = Vector3.Normalize(tangent);
                var perp = Vector3.Normalize(new Vector3(-tangent.Z, 0, tangent.X));

                float outerR = Radius + Falloff;
                ctx.DrawLine(point - perp * outerR, point + perp * outerR);
            }

            ctx.Matrix = savedMatrix;
        }
    }
}
