using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graph;

namespace Freefall.PCG
{
    public enum SamplingMode
    {
        /// <summary>Sample at fixed distance intervals along the spline curve.</summary>
        EvenSpacing,

        /// <summary>Walk control point edges directly, subdividing by Spacing.</summary>
        PerEdge
    }

    /// <summary>
    /// Samples points along a Spline component, producing a SamplePointSet
    /// with position and rotation (aligned to edge tangent, Y-up).
    /// 
    /// EvenSpacing: uses Catmull-Rom evaluation at fixed arc-length intervals.
    /// PerEdge: walks each control point pair as a straight segment, subdividing
    ///          by Spacing. Best for polygon-like splines (e.g., Watabou walls).
    /// </summary>
    [Category("Sampler")]
    public class SplineSampler : Node
    {
        /// <summary>
        /// Source spline. Injected by PCGComponent from the entity's Spline component.
        /// </summary>
        [System.ComponentModel.Browsable(false)]
        public Spline Spline;

        public SamplingMode Mode = SamplingMode.PerEdge;

        /// <summary>Distance between samples in meters.</summary>
        [ValueRange(0.5f, 50f)]
        public float Spacing = 5f;

        [Output]
        public SamplePointSet Output;

        public override void Process()
        {
            if (Spline == null || Spline.Points.Count < 2)
            {
                SetOutput("Output", SamplePointSet.Empty());
                return;
            }

            var result = Mode switch
            {
                SamplingMode.EvenSpacing => SampleEvenSpacing(Spline),
                SamplingMode.PerEdge => SamplePerEdge(Spline),
                _ => SamplePointSet.Empty()
            };

            SetOutput("Output", result);
            Debug.Log($"[SplineSampler] Produced {result.Count} samples (mode={Mode}, spacing={Spacing}m)");
        }

        /// <summary>
        /// Sample using Catmull-Rom evaluation at even arc-length intervals.
        /// </summary>
        private SamplePointSet SampleEvenSpacing(Spline spline)
        {
            float totalLength = spline.GetLength(256);
            if (totalLength < 0.01f) return SamplePointSet.Empty();

            int count = Math.Max(1, (int)(totalLength / Spacing));
            var positions = new List<Vector3>();
            var rotations = new List<Quaternion>();

            for (int i = 0; i < count; i++)
            {
                float t = (float)i / count;
                positions.Add(spline.GetPoint(t));
                var tangent = spline.GetTangent(t);
                rotations.Add(LookRotation(tangent, Vector3.UnitY));
            }

            return BuildResult(positions, rotations);
        }

        /// <summary>
        /// Walk each control point pair as a straight edge segment.
        /// Subdivides each edge by Spacing, producing evenly-spaced samples
        /// along each straight segment.
        /// </summary>
        private SamplePointSet SamplePerEdge(Spline spline)
        {
            int n = spline.Points.Count;
            int edgeCount = spline.Closed ? n : n - 1;

            var positions = new List<Vector3>();
            var rotations = new List<Quaternion>();

            for (int edge = 0; edge < edgeCount; edge++)
            {
                int next = (edge + 1) % n;
                var p0 = spline.Points[edge];
                var p1 = spline.Points[next];

                var edgeVec = p1 - p0;
                float edgeLen = edgeVec.Length();
                if (edgeLen < 0.01f) continue;

                var dir = edgeVec / edgeLen;
                var rot = LookRotation(dir, Vector3.UnitY);

                // How many segments fit on this edge
                int segments = Math.Max(1, (int)MathF.Round(edgeLen / Spacing));
                float actualSpacing = edgeLen / segments;

                for (int s = 0; s < segments; s++)
                {
                    float t = (s + 0.5f) * actualSpacing; // center of each segment
                    positions.Add(p0 + dir * t);
                    rotations.Add(rot);
                }
            }

            return BuildResult(positions, rotations);
        }

        private static SamplePointSet BuildResult(List<Vector3> positions, List<Quaternion> rotations)
        {
            int count = positions.Count;
            return new SamplePointSet
            {
                position = positions.ToArray(),
                extents = new Vector3[count],
                rotation = rotations.ToArray(),
                density = new float[count],
                tags = new string[count]
            };
        }

        /// <summary>
        /// Create a rotation that looks along 'forward' with 'up' as the up vector.
        /// Equivalent to Matrix4x4.CreateLookAt-style orientation.
        /// </summary>
        private static Quaternion LookRotation(Vector3 forward, Vector3 up)
        {
            forward = Vector3.Normalize(forward);
            if (forward.LengthSquared() < 0.001f) return Quaternion.Identity;

            var right = Vector3.Normalize(Vector3.Cross(up, forward));
            if (right.LengthSquared() < 0.001f)
            {
                // forward is parallel to up — pick an arbitrary right
                right = Vector3.UnitX;
            }
            var correctedUp = Vector3.Cross(forward, right);

            // Build rotation matrix → quaternion
            var m = new Matrix4x4(
                right.X, right.Y, right.Z, 0,
                correctedUp.X, correctedUp.Y, correctedUp.Z, 0,
                forward.X, forward.Y, forward.Z, 0,
                0, 0, 0, 1
            );
            return Quaternion.CreateFromRotationMatrix(m);
        }
    }
}
