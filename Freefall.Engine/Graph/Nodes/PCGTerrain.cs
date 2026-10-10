using System;
using System.Collections.Generic;
using System.ComponentModel;
using Freefall.Assets;
using System.Numerics;
using Freefall.Graph;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.PCG
{
    /// <summary>
    /// Nodes that compare sample points (local to the PCG entity) against world-space data — terrain heights, stamps,
    /// scene geometry. PCGComponent injects the entity's local-to-world matrix before execution, so a PCG entity
    /// works at any position/rotation instead of only at the origin.
    /// </summary>
    public interface IWorldSpaceNode
    {
        Matrix4x4 WorldMatrix { get; set; }
    }

    [Category("Terrain")]
    public class TerrainProjection : Node, IWorldSpaceNode
    {
        [Input]
        public SamplePointSet Input;

        /// <summary>
        /// Local-to-world matrix. Injected by PCGComponent before execution.
        /// </summary>
        [Browsable(false)] [Freefall.Reflection.DontSerialize]
        public Matrix4x4 WorldMatrix { get; set; } = Matrix4x4.Identity;

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

        /// <summary>Keep points at least this steep (degrees). 0 = no lower bound; e.g. 22 for rocks on hillsides.</summary>
        public float MinSlopeAngle = 0f;

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
            float minCos = MathF.Cos(MinSlopeAngle * MathF.PI / 180f);
            SetOutput("Output", points.Filter(i =>
            {
                float c = Vector3.Dot(points.normal[i], Vector3.UnitY);
                return c >= maxCos && (MinSlopeAngle <= 0f || c <= minCos);
            }));
        }
    }

    /// <summary>
    /// Keeps points whose world Y lies in [MinHeight, MaxHeight] — after TerrainProjection this is the terrain
    /// height, e.g. keep vegetation off beaches or below a tree line.
    /// </summary>
    [Category("Filter")]
    public class HeightFilter : Node, IWorldSpaceNode
    {
        [Input]
        public SamplePointSet Input;

        public float MinHeight = float.MinValue;
        public float MaxHeight = float.MaxValue;

        /// <summary>Local-to-world matrix of the PCG entity. Injected by PCGComponent.</summary>
        [Browsable(false)] [Freefall.Reflection.DontSerialize]
        public Matrix4x4 WorldMatrix { get; set; } = Matrix4x4.Identity;

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
            var m = WorldMatrix;
            SetOutput("Output", points.Filter(i =>
            {
                float y = Vector3.Transform(points.position[i], m).Y;
                return y >= MinHeight && y <= MaxHeight;
            }));
        }
    }

    /// <summary>
    /// Removes points covered by any SplatStamp in the scene (roads, fields, town ground, camps): the painted
    /// ground stays clear of scattered vegetation/rocks. Closed-spline stamps count as filled areas.
    /// Points are tested at their world position.
    /// </summary>
    [Category("Filter")]
    public class ExcludeStamps : Node, IWorldSpaceNode
    {
        /// <summary>Local-to-world matrix of the PCG entity. Injected by PCGComponent.</summary>
        [Browsable(false)] [Freefall.Reflection.DontSerialize]
        public Matrix4x4 WorldMatrix { get; set; } = Matrix4x4.Identity;

        [Input]
        public SamplePointSet Input;

        /// <summary>Extra clearance beyond each stamp's Radius, in meters (falloff zone is not excluded by default).</summary>
        [ValueRange(0f, 50f)]
        public float Margin = 1f;

        /// <summary>
        /// Also test the executing entity's own SplatStamp. For props offset beside a line (road lanterns, fences):
        /// they are dropped wherever they land inside the road itself, e.g. on the inside of a hairpin or at a junction.
        /// </summary>
        public bool IncludeOwnStamps = false;

        /// <summary>
        /// Stamps carrying any of these tags do not exclude anything here (e.g. "Forest Floor" for roadside props,
        /// which belong along a road through a forest although the forest paints its own ground).
        /// </summary>
        public List<Tag> IgnoreTags = new List<Tag>();

        [Output]
        public SamplePointSet Output;

        /// <summary>
        /// The executing PCG entity. Injected by PCGComponent: an area's own SplatStamp (e.g. a farmyard painted as
        /// trampled dirt) keeps other scatter out but must not exclude the area's own points (unless IncludeOwnStamps).
        /// </summary>
        [Browsable(false)]
        public Entity IgnoreEntity;

        public override void Process()
        {
            var points = GetInputValue<SamplePointSet>("Input");
            if (points == null || points.Count == 0)
            {
                SetOutput("Output", SamplePointSet.Empty());
                return;
            }

            var stamps = new System.Collections.Generic.List<(SplatStamp stamp, Vortice.Mathematics.BoundingBox bounds)>();
            foreach (var s in ComponentCache<SplatStamp>.All)
            {
                // Global stamps are terrain-wide rules (grass on flats, rock on cliffs), not painted ground to keep clear of
                if (s is not SplatStamp { Enabled: true, IsGlobal: false } splat || (!IncludeOwnStamps && splat.Entity == IgnoreEntity))
                    continue;
                if (IgnoreTags is { Count: > 0 } && splat.Tags is { Count: > 0 } && splat.Tags.Exists(t => t != null && IgnoreTags.Contains(t)))
                    continue;
                stamps.Add((splat, splat.GetWorldBounds()));
            }

            var m = WorldMatrix;
            SetOutput("Output", points.Filter(i =>
            {
                var p = Vector3.Transform(points.position[i], m);
                foreach (var (stamp, b) in stamps)
                {
                    if (p.X < b.Min.X - Margin || p.X > b.Max.X + Margin || p.Z < b.Min.Z - Margin || p.Z > b.Max.Z + Margin)
                        continue;
                    if (stamp.GetEdgeDistance(p) <= Margin) return false;
                }
                return true;
            }));
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
