using System;
using System.Numerics;

namespace Freefall.Animation
{
    /// <summary>
    /// Analytical two-bone IK solver (e.g. thigh → shin → foot). Finds where the mid joint
    /// (knee/elbow) goes from the lengths of the two bones and a pole that says on which side it
    /// bends, then the two rotations that put the bones there. Because the joint's place comes from
    /// the pole and not from how the chain is bent now, a chain held straight bends the right way.
    /// </summary>
    public static class TwoBoneIK
    {
        /// <summary>
        /// Solves a two-bone IK chain. All positions are in the same coordinate space.
        /// </summary>
        /// <param name="root">Position of the root joint (e.g. hip).</param>
        /// <param name="mid">Position of the mid joint (e.g. knee).</param>
        /// <param name="tip">Position of the end effector (e.g. foot/ankle).</param>
        /// <param name="target">Desired end-effector position.</param>
        /// <param name="poleTarget">Pole vector hint — the knee/elbow bends toward this.</param>
        /// <param name="soft">
        /// How far short of full length the chain starts to slow down, so that it comes to its full
        /// length gradually instead of snapping straight; never less than the chain is stretched
        /// already. 0: not at all.
        /// </param>
        /// <param name="rootCorrection">Rotation of the root bone about the root (hip).</param>
        /// <param name="midCorrection">Rotation of the lower bone about the mid joint where it ends up, after <paramref name="rootCorrection"/>.</param>
        /// <param name="newMid">Where the mid joint ends up.</param>
        /// <param name="newTip">Where the end effector ends up: the target, or as near as the chain gets.</param>
        /// <returns>True if a valid solution was found.</returns>
        public static bool Solve(
            Vector3 root, Vector3 mid, Vector3 tip,
            Vector3 target, Vector3 poleTarget, float soft,
            out Quaternion rootCorrection, out Quaternion midCorrection,
            out Vector3 newMid, out Vector3 newTip)
        {
            rootCorrection = Quaternion.Identity;
            midCorrection = Quaternion.Identity;
            newMid = mid;
            newTip = tip;

            float lenUpper = Vector3.Distance(root, mid);
            float lenLower = Vector3.Distance(mid, tip);

            Vector3 rootToTarget = target - root;
            float distToTarget = rootToTarget.Length();

            // Early out: the chain is degenerate
            if (distToTarget < 0.001f || lenUpper < 0.001f || lenLower < 0.001f)
                return false;

            Vector3 along = rootToTarget / distToTarget;

            float reach = Reach(distToTarget, Vector3.Distance(root, tip), lenUpper + lenLower, soft);
            reach = Math.Max(reach, Math.Abs(lenUpper - lenLower) + 0.001f);

            // ---- Where the mid joint goes: the triangle's height over the root → target line, on the pole's side ----
            float toFoot = (lenUpper * lenUpper - lenLower * lenLower + reach * reach) / (2f * reach);
            float height = MathF.Sqrt(MathF.Max(lenUpper * lenUpper - toFoot * toFoot, 0f));

            Vector3 side = Across(poleTarget - root, along);
            if (side.LengthSquared() < 1e-8f) side = Across(mid - root, along);
            if (side.LengthSquared() < 1e-8f)
                side = Across(MathF.Abs(along.X) < 0.9f ? Vector3.UnitX : Vector3.UnitY, along);
            side = Vector3.Normalize(side);

            newMid = root + along * toFoot + side * height;
            newTip = root + along * reach;

            // ---- The rotations that put the two bones there ----
            rootCorrection = RotationBetween(Vector3.Normalize(mid - root), Vector3.Normalize(newMid - root));

            Vector3 lower = Vector3.Transform(tip - mid, rootCorrection);
            midCorrection = RotationBetween(Vector3.Normalize(lower), Vector3.Normalize(newTip - newMid));

            return true;
        }

        /// <summary>
        /// How far from the root the chain gets for a target at <paramref name="distance"/>. Up to
        /// the point it starts to slow down it gets there; past that it closes on its
        /// <paramref name="length"/> and never quite reaches it.
        /// </summary>
        private static float Reach(float distance, float stretched, float length, float soft)
        {
            float start = Math.Max(length - soft, Math.Min(stretched, length));
            if (distance <= start) return distance;

            float left = length - start;
            return left > 1e-5f
                ? start + left * (1f - MathF.Exp(-(distance - start) / left))
                : length;
        }

        /// <summary>The longest a chain may be asked to stand for a target without slowing down (see <paramref name="soft"/> of <see cref="Solve"/>).</summary>
        public static float FreeReach(float stretched, float length, float soft)
            => Math.Max(length - soft, Math.Min(stretched, length));

        // The part of a vector across a direction of length 1
        private static Vector3 Across(Vector3 v, Vector3 along) => v - Vector3.Dot(v, along) * along;

        /// <summary>
        /// Computes the shortest rotation from direction a to direction b.
        /// Both vectors must be normalized.
        /// </summary>
        private static Quaternion RotationBetween(Vector3 a, Vector3 b)
        {
            float dot = Vector3.Dot(a, b);

            if (dot > 0.9999999f)
                return Quaternion.Identity;

            if (dot < -0.9999f)
            {
                // 180-degree rotation — pick a perpendicular axis
                Vector3 perp = MathF.Abs(a.X) < 0.9f
                    ? Vector3.Cross(a, Vector3.UnitX)
                    : Vector3.Cross(a, Vector3.UnitY);
                perp = Vector3.Normalize(perp);
                return Quaternion.CreateFromAxisAngle(perp, MathF.PI);
            }

            Vector3 axis = Vector3.Cross(a, b);
            float w = 1f + dot;
            return Quaternion.Normalize(new Quaternion(axis.X, axis.Y, axis.Z, w));
        }
    }
}
