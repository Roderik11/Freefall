using System;
using System.Numerics;
using System.Security.Cryptography;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// Abstract base for terrain stamp components (HeightStamp, SplatStamp, DecoStamp).
    /// Provides shape definition (radial or spline corridor/area), edge noise,
    /// priority ordering, and gizmo visualization.
    /// </summary>
    public abstract class TerrainStamp : Component, ISceneGizmo
    {
        // ── Shape ──

        /// <summary>
        /// Radius for radial stamps (when no Spline is present).
        /// Width for spline-based stamps (half-width on each side of the path).
        /// </summary>
        [ValueRange(0.1f, 20f)]
        public float Radius = 4f;

        /// <summary>Falloff distance (blend from full effect to none). World units.</summary>
        [ValueRange(0f, 20f)]
        public float Falloff = 3f;

        // ── Edge Noise (organic edge breakup) ──

        /// <summary>Enable noise displacement on the falloff boundary for organic edges.</summary>
        public bool EnableNoise = false;

        /// <summary>Noise frequency relative to influence radius (bumps per radius).</summary>
        [ValueRange(0.5f, 20f)]
        public float NoiseFrequency = 3f;

        /// <summary>Noise amplitude in world units (how far edges displace).</summary>
        [ValueRange(0f, 20f)]
        public float NoiseAmplitude = 2f;

        /// <summary>Noise seed for variation between stamps.</summary>
        public int NoiseSeed = RandomNumberGenerator.GetInt32(int.MaxValue);

        // ── Priority ──

        /// <summary>Evaluation priority. Higher values are applied later (overwrite lower).</summary>
        public int Priority = 0;

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

        /// <summary>True if this stamp follows a spline path.</summary>
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

        // ── Shape query API ──

        /// <summary>
        /// Compute the stamp weight at a world-space position.
        /// Returns 0..1 where 1 = full effect, 0 = outside stamp.
        /// </summary>
        public float GetWeight(Vector3 worldPos)
        {
            float distance = GetDistance(worldPos);
            if (distance >= Radius + Falloff) return 0f;
            if (distance <= Radius) return 1f;
            float t = (distance - Radius) / Math.Max(Falloff, 0.001f);
            return 1f - SmoothStep(t);
        }

        /// <summary>
        /// Get the minimum distance from worldPos to the stamp shape.
        /// </summary>
        public float GetDistance(Vector3 worldPos)
        {
            if (IsSplineMode)
                return GetDistanceToSpline(worldPos);

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

        private float GetDistanceToSpline(Vector3 worldPos)
        {
            float nearestT = FindNearestT(worldPos, 64);
            var nearestPoint = _cachedSpline.GetWorldPoint(nearestT);
            float dx = worldPos.X - nearestPoint.X;
            float dz = worldPos.Z - nearestPoint.Z;
            return MathF.Sqrt(dx * dx + dz * dz);
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
            if (IsSplineMode)
                DrawSplineGizmo(ctx);
            else
                DrawRadialGizmo(ctx);
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

            int samples = spline.TotalSegments;
            var savedMatrix = ctx.Matrix;
            ctx.Matrix = Matrix4x4.Identity;

            ctx.Color = GizmoColor;
            ctx.LineWidth = 1.5f;

            Vector3 prevLeft = Vector3.Zero, prevRight = Vector3.Zero;
            Vector3 prevOuterLeft = Vector3.Zero, prevOuterRight = Vector3.Zero;
            bool hasFalloff = Falloff > 0.01f;

            for (int i = 0; i <= samples; i++)
            {
                float t = (float)i / samples;
                var point = spline.GetWorldPoint(t);
                var tangent = spline.GetTangent(t);

                if (Transform != null)
                {
                    var rotMatrix = Matrix4x4.CreateFromQuaternion(Transform.Rotation);
                    tangent = Vector3.TransformNormal(tangent, rotMatrix);
                }
                tangent = Vector3.Normalize(tangent);

                var perp = Vector3.Normalize(new Vector3(-tangent.Z, 0, tangent.X));

                var left = point + perp * Radius;
                var right = point - perp * Radius;

                if (i > 0)
                {
                    ctx.Color = GizmoColor;
                    ctx.LineWidth = 1.5f;
                    ctx.DrawLine(prevLeft, left);
                    ctx.DrawLine(prevRight, right);
                }

                if (hasFalloff)
                {
                    var outerLeft = point + perp * (Radius + Falloff);
                    var outerRight = point - perp * (Radius + Falloff);

                    if (i > 0)
                    {
                        ctx.Color = new Color4(GizmoColor.R, GizmoColor.G, GizmoColor.B, 0.4f);
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

            // Cross-hatches at control points
            ctx.Color = new Color4(GizmoColor.R, GizmoColor.G, GizmoColor.B, 0.6f);
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

        /// <summary>Override in subclasses for distinct gizmo colors.</summary>
        protected virtual Color4 GizmoColor => new Color4(0.3f, 0.9f, 0.3f, 1f);
    }
}
