using System;
using System.Numerics;

namespace Freefall.Animation
{
    /// <summary>
    /// Analytical two-bone IK solver (e.g. thigh → shin → foot).
    /// Uses the cosine rule to compute joint rotations that place the
    /// end-effector (tip) at a target position with a pole-vector hint
    /// to control the mid-joint (knee/elbow) orientation.
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
        /// <param name="poleTarget">Pole vector hint — the knee/elbow should point toward this.</param>
        /// <param name="rootCorrection">Output rotation correction for the root bone (hip).</param>
        /// <param name="midCorrection">Output rotation correction for the mid bone (knee).</param>
        /// <returns>True if a valid solution was found.</returns>
        public static bool Solve(
            Vector3 root, Vector3 mid, Vector3 tip,
            Vector3 target, Vector3 poleTarget,
            out Quaternion rootCorrection, out Quaternion midCorrection)
        {
            rootCorrection = Quaternion.Identity;
            midCorrection = Quaternion.Identity;

            float lenUpper = Vector3.Distance(root, mid);
            float lenLower = Vector3.Distance(mid, tip);
            float lenChain = lenUpper + lenLower;

            // Direction from root to target
            Vector3 rootToTarget = target - root;
            float distToTarget = rootToTarget.Length();

            // Early out: target unreachable or chain is degenerate
            if (distToTarget < 0.001f || lenUpper < 0.001f || lenLower < 0.001f)
                return false;

            // Clamp — if target is beyond reach, extend fully
            distToTarget = Math.Min(distToTarget, lenChain - 0.001f);

            // Current chain direction and target direction
            Vector3 rootToTip = tip - root;
            float rootToTipLen = rootToTip.Length();
            if (rootToTipLen < 0.001f)
                return false;

            Vector3 rootToTargetDir = Vector3.Normalize(rootToTarget);
            Vector3 rootToTipDir = Vector3.Normalize(rootToTip);

            // ---- Step 1: Cosine rule for the knee angle ----
            // Triangle sides: lenUpper, lenLower, distToTarget
            // Angle at the mid joint (knee)
            float cosAngleMid = (lenUpper * lenUpper + lenLower * lenLower - distToTarget * distToTarget)
                                / (2f * lenUpper * lenLower);
            cosAngleMid = Math.Clamp(cosAngleMid, -1f, 1f);
            float angleMid = MathF.Acos(cosAngleMid);

            // Current knee angle
            Vector3 midToRoot = Vector3.Normalize(root - mid);
            Vector3 midToTip = Vector3.Normalize(tip - mid);
            float cosCurrentMid = Vector3.Dot(midToRoot, midToTip);
            cosCurrentMid = Math.Clamp(cosCurrentMid, -1f, 1f);
            float currentAngleMid = MathF.Acos(cosCurrentMid);

            // Mid correction: rotate around the knee bend axis
            float midDelta = angleMid - currentAngleMid;
            if (MathF.Abs(midDelta) > 0.0001f)
            {
                Vector3 midAxis = Vector3.Cross(midToRoot, midToTip);
                if (midAxis.LengthSquared() < 0.0001f)
                {
                    // Fallback axis from pole target
                    midAxis = Vector3.Normalize(Vector3.Cross(midToRoot, poleTarget - mid));
                }
                else
                {
                    midAxis = Vector3.Normalize(midAxis);
                }

                midCorrection = Quaternion.CreateFromAxisAngle(midAxis, midDelta);
            }

            // ---- Step 2: Apply mid correction and recompute tip ----
            Vector3 newMidToTip = Vector3.Transform(tip - mid, midCorrection);
            Vector3 newTip = mid + newMidToTip;

            // ---- Step 3: Root rotation to aim the chain at the target ----
            Vector3 newRootToTip = newTip - root;
            float newLen = newRootToTip.Length();

            if (newLen > 0.001f)
            {
                Vector3 newRootToTipDir = Vector3.Normalize(newRootToTip);
                rootCorrection = RotationBetween(newRootToTipDir, rootToTargetDir);
            }

            // ---- Step 4: Pole target twist ----
            // After aiming, twist around the root→target axis so the knee points toward poleTarget
            Vector3 afterMid = root + Vector3.Transform(mid - root, rootCorrection);
            Vector3 chainAxis = rootToTargetDir;

            Vector3 currentPole = afterMid - root;
            currentPole -= Vector3.Dot(currentPole, chainAxis) * chainAxis;

            Vector3 desiredPole = poleTarget - root;
            desiredPole -= Vector3.Dot(desiredPole, chainAxis) * chainAxis;

            if (currentPole.LengthSquared() > 0.0001f && desiredPole.LengthSquared() > 0.0001f)
            {
                currentPole = Vector3.Normalize(currentPole);
                desiredPole = Vector3.Normalize(desiredPole);

                float poleDot = Math.Clamp(Vector3.Dot(currentPole, desiredPole), -1f, 1f);
                float poleAngle = MathF.Acos(poleDot);

                if (MathF.Abs(poleAngle) > 0.001f)
                {
                    Vector3 poleAxis = Vector3.Cross(currentPole, desiredPole);
                    if (poleAxis.LengthSquared() > 0.0001f)
                    {
                        poleAxis = Vector3.Normalize(poleAxis);
                        // Ensure rotation is around the chain axis direction
                        if (Vector3.Dot(poleAxis, chainAxis) < 0)
                            poleAngle = -poleAngle;

                        Quaternion poleTwist = Quaternion.CreateFromAxisAngle(chainAxis, poleAngle);
                        rootCorrection = poleTwist * rootCorrection;
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Computes the shortest rotation from direction a to direction b.
        /// Both vectors must be normalized.
        /// </summary>
        private static Quaternion RotationBetween(Vector3 a, Vector3 b)
        {
            float dot = Vector3.Dot(a, b);

            if (dot > 0.9999f)
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
