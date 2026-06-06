using System;
using System.ComponentModel;
using System.Numerics;
using Freefall.Graph;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.PCG
{
    [Category("Terrain")]
    public class TerrainProjection : Node
    {
        [Input]
        public SamplePointSet Input;

        /// <summary>
        /// Local-to-world matrix. Injected by PCGComponent before execution.
        /// </summary>
        [Browsable(false)]
        public Matrix4x4 WorldMatrix = Matrix4x4.Identity;

        [Output]
        public SamplePointSet Output;

        public override void Process()
        {
            var points = GetInputValue<SamplePointSet>("Input");
            if (points == null || points.Count == 0)
            {
                SetOutput("Output", SamplePointSet.Empty());
                return;
            }

            var terrains = ComponentCache<TerrainRenderer>.All;
            if (terrains.Count == 0)
            {
                SetOutput("Output", points.Clone());
                return;
            }

            var terrain = terrains[0] as IHeightProvider;
            var result = points.Clone();
            result.normal = new Vector3[result.Count];
            const float delta = 0.5f;
            float localY = WorldMatrix.Translation.Y;

            for (int i = 0; i < result.Count; i++)
            {
                var localPos = result.position[i];
                var worldPos = Vector3.Transform(localPos, WorldMatrix);

                float h = terrain.GetHeight(worldPos);
                result.position[i] = new Vector3(localPos.X, h - localY, localPos.Z);

                float hL = terrain.GetHeight(new Vector3(worldPos.X - delta, 0, worldPos.Z));
                float hR = terrain.GetHeight(new Vector3(worldPos.X + delta, 0, worldPos.Z));
                float hD = terrain.GetHeight(new Vector3(worldPos.X, 0, worldPos.Z - delta));
                float hU = terrain.GetHeight(new Vector3(worldPos.X, 0, worldPos.Z + delta));

                var tangentX = new Vector3(delta * 2f, hR - hL, 0);
                var tangentZ = new Vector3(0, hU - hD, delta * 2f);
                result.normal[i] = Vector3.Normalize(Vector3.Cross(tangentZ, tangentX));
            }

            SetOutput("Output", result);
        }
    }

    [Category("Filter")]
    public class SlopeFilter : Node
    {
        [Input]
        public SamplePointSet Input;

        public float MaxSlopeAngle = 45f;

        [Output]
        public SamplePointSet Output;

        public override void Process()
        {
            var points = GetInputValue<SamplePointSet>("Input");
            if (points == null || points.Count == 0)
            {
                SetOutput("Output", SamplePointSet.Empty());
                return;
            }

            if (points.normal == null || points.normal.Length == 0)
            {
                SetOutput("Output", points);
                return;
            }

            float maxCos = MathF.Cos(MaxSlopeAngle * MathF.PI / 180f);
            SetOutput("Output", points.Filter(i => Vector3.Dot(points.normal[i], Vector3.UnitY) >= maxCos));
        }
    }

    [Category("Transform")]
    public class AlignToTerrain : Node
    {
        [Input]
        public SamplePointSet Input;

        [ValueRange(0f, 1f)]
        public float Blend = 1.0f;

        [Output]
        public SamplePointSet Output;

        public override void Process()
        {
            var points = GetInputValue<SamplePointSet>("Input");
            if (points == null || points.Count == 0)
            {
                SetOutput("Output", SamplePointSet.Empty());
                return;
            }

            if (points.normal == null || points.normal.Length == 0 || Blend <= 0f)
            {
                SetOutput("Output", points.Clone());
                return;
            }

            var result = points.Clone();

            for (int i = 0; i < result.Count; i++)
            {
                var normal = result.normal[i];
                var tilt = RotationFromTo(Vector3.UnitY, normal);
                tilt = Quaternion.Slerp(Quaternion.Identity, tilt, Blend);
                result.rotation[i] = Quaternion.Normalize(tilt * result.rotation[i]);
            }

            SetOutput("Output", result);
        }

        private static Quaternion RotationFromTo(Vector3 from, Vector3 to)
        {
            float dot = Vector3.Dot(from, to);
            if (dot >= 0.999999f) return Quaternion.Identity;
            if (dot <= -0.999999f)
            {
                // 180-degree rotation around any perpendicular axis
                var perp = MathF.Abs(from.X) < 0.9f
                    ? Vector3.Cross(from, Vector3.UnitX)
                    : Vector3.Cross(from, Vector3.UnitZ);
                perp = Vector3.Normalize(perp);
                return new Quaternion(perp, 0);
            }

            var axis = Vector3.Cross(from, to);
            var q = new Quaternion(axis, 1f + dot);
            return Quaternion.Normalize(q);
        }
    }
}
