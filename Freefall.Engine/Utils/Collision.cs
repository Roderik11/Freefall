using System;
using System.Numerics;

namespace Freefall
{
    public static class Collision
    {
        private static bool IsZero(float value) => MathF.Abs(value) < 1e-6f;

        /// <summary>
        /// Determines whether a ray intersects a plane.
        /// Returns the distance along the ray to the intersection point.
        /// </summary>
        public static bool RayIntersectsPlane(ref Ray ray, ref Plane plane, out float distance)
        {
            float denom = Vector3.Dot(plane.Normal, ray.Direction);
            if (IsZero(denom))
            {
                distance = 0f;
                return false;
            }

            distance = (0f - plane.D - Vector3.Dot(plane.Normal, ray.Position)) / denom;
            return true;
        }

        /// <summary>
        /// Determines whether a ray intersects a plane.
        /// Returns the world-space intersection point.
        /// </summary>
        public static bool RayIntersectsPlane(ref Ray ray, ref Plane plane, out Vector3 point)
        {
            if (!RayIntersectsPlane(ref ray, ref plane, out float distance))
            {
                point = Vector3.Zero;
                return false;
            }

            point = ray.Position + ray.Direction * distance;
            return true;
        }

        /// <summary>
        /// Compute the signed clockwise angle (in radians) between two vectors
        /// projected onto a plane with the given normal.
        /// Used by the rotation gizmo to measure drag angle.
        /// </summary>
        public static float ClockwiseAngle(Vector3 a, Vector3 b, Vector3 n)
        {
            float dot = a.X * b.X + a.Y * b.Y + a.Z * b.Z;
            float det = a.X * b.Y * n.Z + b.X * n.Y * a.Z + n.X * a.Y * b.Z
                      - a.Z * b.Y * n.X - b.Z * n.Y * a.X - n.Z * a.Y * b.X;
            return MathF.Atan2(det, dot);
        }
    }
}
