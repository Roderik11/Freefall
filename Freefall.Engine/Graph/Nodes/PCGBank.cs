using System;
using System.ComponentModel;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graph;

namespace Freefall.PCG
{
    /// <summary>Where along a bank's fall line <see cref="BankFrame"/> puts each point.</summary>
    public enum BankAnchor
    {
        /// <summary>The bank's top edge, where the upper ground stops being flat.</summary>
        Crest,
        /// <summary>The bank's foot, where the lower ground starts.</summary>
        Foot,
        /// <summary>Between the two by <see cref="BankFrame.Share"/> (0 = crest, 1 = foot).</summary>
        Share
    }

    /// <summary>
    /// Finds the bank beside each sample of a line drawn along a bank, terrace edge or cliff line, and gives the point
    /// that bank's frame: local +Z down the fall line, the point moved to the crest, the foot or between them (plus a
    /// reach along the fall line), on the ground. The bank is the steep stretch of ground nearest the sample on the
    /// line across the spline (up to <see cref="Search"/> each way), so the spline can run along the top edge, the
    /// foot or the face itself. Optionally scales each point so a piece <see cref="FitHeight"/> tall spans the bank's
    /// height, and drops points whose bank is lower or higher than wanted (which faces go where).
    /// Rocks dressing a bank: big faces at Share ~0.5 fitted to its height, debris at the foot with a reach beyond it.
    /// </summary>
    [Category("Terrain")]
    public class BankFrame : Node, IWorldSpaceNode
    {
        [Input]
        public SamplePointSet Input;

        [Description("Where each point goes along the bank's fall line: its crest, its foot, or between them by Share")]
        public BankAnchor Anchor = BankAnchor.Share;

        [Description("With Anchor Share: 0 = the crest, 1 = the foot")]
        [ValueRange(0f, 1f)]
        public float Share = 0.5f;

        [Description("Metres the point moves on from the anchor along the fall line, the least (negative = uphill)")]
        public float ReachMin;

        [Description("Metres the point moves on from the anchor along the fall line, the most")]
        public float ReachMax;

        [Description("Above 1 crowds the reach towards ReachMin: debris thinning out away from the foot")]
        [ValueRange(0.1f, 8f)]
        public float ReachBias = 1f;

        [Description("Metres to look for the bank on each side of the spline, across it")]
        [ValueRange(2f, 100f)]
        public float Search = 30f;

        [Description("Ground steeper than this (degrees) counts as the bank")]
        [ValueRange(5f, 85f)]
        public float Steep = 28f;

        [Description("Keep only points whose bank is at least this high (m); 0 also keeps points with no bank found")]
        public float MinBankHeight;

        [Description("Keep only points whose bank is at most this high (m)")]
        public float MaxBankHeight = 1000f;

        [Description("0 = face straight across the spline, 1 = face the ground's own fall line on the bank")]
        [ValueRange(0f, 1f)]
        public float FollowSlope = 0.5f;

        [Description("0 = upright, 1 = tilted back to lie along the bank's slope (its height over its run). The scans are open at the top and back: upright on a 55 degree bank they leave an open wedge behind their top")]
        [ValueRange(0f, 1f)]
        public float Lean;

        [Description("Height of the piece above its pivot at scale 1 (m). Above 0, points are scaled so the piece spans the bank: (bank height + Over) / FitHeight, clamped to ScaleMin..ScaleMax, times the scale they came with")]
        public float FitHeight;

        [Description("With FitHeight: metres the fitted piece reaches above the bank's height")]
        public float Over;

        [Description("With FitHeight: the smallest fitted scale")]
        public float ScaleMin = 0.5f;

        [Description("With FitHeight: the largest fitted scale")]
        public float ScaleMax = 1.5f;

        [Description("With FitHeight: metres the fitted piece reaches below the bank's foot, so ground falling away in front of it leaves no gap under the face")]
        public float Under;

        [Description("With FitHeight: depth of the piece front to back at scale 1 (m), centred on its pivot. Above 0 the reach is measured from the piece's front face instead of its pivot, and the fitted scale is at least what makes its top reach Behind metres past the crest: a shallow piece on a bank with a long run is scaled up (uniformly) rather than leaving bare bank behind its top")]
        public float FitDepth;

        [Description("With FitDepth: metres the piece's top reaches past the crest into the ground above")]
        public float Behind = 0.5f;

        [Description("With FitHeight: hang the piece from the crest, its top Over above it, the rest going down as deep as its fitted scale takes it (a piece scaled up to reach the crest sinks below the foot). Use it without SeatOnGround. Off: the point stands on the ground")]
        public bool HangFromCrest;

        [Browsable(false)] [Freefall.Reflection.DontSerialize]
        public Matrix4x4 WorldMatrix { get; set; } = Matrix4x4.Identity;

        [Output]
        public SamplePointSet Output;

        public override void Process()
        {
            var points = GetInputValue<SamplePointSet>("Input");
            var terrains = ComponentCache<TerrainRenderer>.All;
            if (points == null || points.Count == 0 || terrains.Count == 0)
            {
                SetOutput("Output", points == null || points.Count == 0 ? SamplePointSet.Empty() : points.Clone());
                return;
            }

            var ground = (IHeightProvider)terrains[0];
            var m = WorldMatrix;
            Matrix4x4.Invert(m, out var inv);
            var rng = CreateRandom();

            var probe = new BankProbe(Search, Steep);

            var result = points.Clone();
            if (result.normal == null || result.normal.Length != result.Count) result.normal = new Vector3[result.Count];
            var keep = new bool[result.Count];

            for (int i = 0; i < result.Count; i++)
            {
                var p = Vector3.Transform(points.position[i], m);
                var right = Flat(Vector3.TransformNormal(Vector3.Transform(Vector3.UnitX, points.rotation[i]), m));
                if (right == Vector3.Zero) right = Vector3.UnitX;

                var bank = probe.Find(ground, p, right);
                var down = bank.Down;
                float tCrest = bank.CrestT, tFoot = bank.FootT, bankHeight = bank.Height;

                if (bankHeight > MaxBankHeight || (MinBankHeight > 0f && bankHeight < MinBankHeight))
                    continue;
                keep[i] = true;

                // Facing: across the spline, turned towards the ground's own fall line on the bank
                var facing = down;
                if (FollowSlope > 0f && bank.Found)
                {
                    var mid = p + down * ((tCrest + tFoot) * 0.5f);
                    float r = MathF.Max(1f, (tFoot - tCrest) * 0.5f);
                    var fall = Flat(new Vector3(
                        ground.GetHeight(mid - Vector3.UnitX * r) - ground.GetHeight(mid + Vector3.UnitX * r), 0f,
                        ground.GetHeight(mid - Vector3.UnitZ * r) - ground.GetHeight(mid + Vector3.UnitZ * r)));
                    if (fall != Vector3.Zero && Vector3.Dot(fall, down) > 0f)
                        facing = Flat(Vector3.Lerp(down, fall, FollowSlope));
                    if (facing == Vector3.Zero) facing = down;
                }

                float tAnchor = Anchor switch
                {
                    BankAnchor.Crest => tCrest,
                    BankAnchor.Foot => tFoot,
                    _ => tCrest + (tFoot - tCrest) * Share
                };
                float u = (float)rng.NextDouble();
                tAnchor += ReachMin + (ReachMax - ReachMin) * MathF.Pow(u, MathF.Max(ReachBias, 0.01f));

                // Lean back: the top towards the crest (local -Z), by a share of how far the bank is from vertical
                float lean = 0f;
                if (Lean > 0f && bank.Found && tFoot > tCrest)
                    lean = Lean * (MathF.PI * 0.5f - MathF.Atan2(bankHeight, tFoot - tCrest));

                // Fitted scale: the bank's height (plus Over and Under), and with FitDepth at least what takes the top
                // from the face at tAnchor back past the crest. Uniform: a piece scaled for depth grows taller too.
                float fit = 1f;
                if (FitHeight > 0f)
                {
                    float need = (bankHeight + Over + Under) / (FitHeight * MathF.Cos(lean));
                    if (FitDepth > 0f && bank.Found)
                        need = MathF.Max(need, (tAnchor - tCrest + Behind) / FitDepth);
                    fit = Math.Clamp(need, ScaleMin, MathF.Max(ScaleMin, ScaleMax));
                }
                // With FitDepth the reach places the front face: the pivot lies half the fitted depth behind it
                if (FitHeight > 0f && FitDepth > 0f)
                    tAnchor -= FitDepth * fit * 0.5f;

                var world = p + down * tAnchor;
                world.Y = FitHeight > 0f && HangFromCrest && bank.Found
                    ? bank.CrestHeight + Over - FitHeight * fit * MathF.Cos(lean)
                    : ground.GetHeight(world);
                result.position[i] = Vector3.Transform(world, inv);

                var facingLocal = Flat(Vector3.TransformNormal(facing, inv));
                result.rotation[i] = Quaternion.CreateFromAxisAngle(Vector3.UnitY, MathF.Atan2(facingLocal.X, facingLocal.Z))
                                     * Quaternion.CreateFromAxisAngle(Vector3.UnitX, -lean);
                result.normal[i] = Normal(ground, world);

                if (FitHeight > 0f)
                    result.extents[i] *= fit;
            }

            SetOutput("Output", result.Filter(i => keep[i]));
        }

        /// <summary>The horizontal part of v, normalised (zero when v is vertical or zero).</summary>
        public static Vector3 Flat(Vector3 v)
        {
            v.Y = 0f;
            float l = v.Length();
            return l < 1e-5f ? Vector3.Zero : v / l;
        }

        /// <summary>The ground's normal at a world point, from the heights half a metre round it.</summary>
        public static Vector3 Normal(IHeightProvider ground, Vector3 world)
        {
            const float d = 0.5f;
            float hL = ground.GetHeight(new Vector3(world.X - d, 0, world.Z));
            float hR = ground.GetHeight(new Vector3(world.X + d, 0, world.Z));
            float hD = ground.GetHeight(new Vector3(world.X, 0, world.Z - d));
            float hU = ground.GetHeight(new Vector3(world.X, 0, world.Z + d));
            return Vector3.Normalize(Vector3.Cross(new Vector3(0, hU - hD, d * 2f), new Vector3(d * 2f, hR - hL, 0)));
        }
    }

    /// <summary>
    /// Sets each point's height so a piece of the given footprint has no gap under it: the lowest ground under the
    /// footprint (turned and scaled with the point), less <see cref="Sink"/>. On a slope the uphill side goes into the
    /// ground instead of the downhill side hanging over it. Use after the point's final rotation and scale.
    /// </summary>
    [Category("Terrain")]
    public class SeatOnGround : Node, IWorldSpaceNode
    {
        [Input]
        public SamplePointSet Input;

        [Description("The piece's footprint at scale 1 (m): width along local X, depth along local Z, centred on the pivot")]
        public Vector2 Footprint = new Vector2(1f, 1f);

        [Description("Share of the footprint that must have ground under it: 1 = all of it (the lowest ground), 0.8 lets the outer tenth on each side overhang")]
        [ValueRange(0.1f, 1f)]
        public float Cover = 1f;

        [Description("Metres the pivot goes below that ground at scale 1 (times the point's Y scale)")]
        public float Sink;

        [Browsable(false)] [Freefall.Reflection.DontSerialize]
        public Matrix4x4 WorldMatrix { get; set; } = Matrix4x4.Identity;

        [Output]
        public SamplePointSet Output;

        private const int Samples = 5;

        public override void Process()
        {
            var points = GetInputValue<SamplePointSet>("Input");
            var terrains = ComponentCache<TerrainRenderer>.All;
            if (points == null || points.Count == 0 || terrains.Count == 0)
            {
                SetOutput("Output", points == null || points.Count == 0 ? SamplePointSet.Empty() : points.Clone());
                return;
            }

            var ground = (IHeightProvider)terrains[0];
            var m = WorldMatrix;
            Matrix4x4.Invert(m, out var inv);
            var result = points.Clone();
            float half = 0.5f * Math.Clamp(Cover, 0.1f, 1f);

            for (int i = 0; i < result.Count; i++)
            {
                var scale = result.extents[i];
                var lowest = float.MaxValue;
                for (int a = 0; a < Samples; a++)
                    for (int b = 0; b < Samples; b++)
                    {
                        float x = (a / (Samples - 1f) * 2f - 1f) * half * Footprint.X * scale.X;
                        float z = (b / (Samples - 1f) * 2f - 1f) * half * Footprint.Y * scale.Z;
                        var local = result.position[i] + Vector3.Transform(new Vector3(x, 0f, z), result.rotation[i]);
                        lowest = MathF.Min(lowest, ground.GetHeight(Vector3.Transform(local, m)));
                    }

                var world = Vector3.Transform(result.position[i], m);
                world.Y = lowest - Sink * scale.Y;
                result.position[i] = Vector3.Transform(world, inv);
            }

            SetOutput("Output", result);
        }
    }


    /// <summary>
    /// A prefab's size for placement nodes (project PCG nodes among them, e.g. BankWall in Freefall-Solid): the bounds of
    /// every mesh in it with its root at the origin. Measured once per prefab version by making an instance and
    /// destroying it again (main thread, as PCG runs), then cached.
    /// </summary>
    public static class PrefabBounds
    {
        private static readonly System.Collections.Generic.Dictionary<Freefall.Assets.Prefab, (byte[] yaml, Vector3 min, Vector3 max)> _bounds = new();

        public static (Vector3 min, Vector3 max)? Measure(Freefall.Assets.Prefab prefab)
        {
            lock (_bounds)
                if (_bounds.TryGetValue(prefab, out var c) && ReferenceEquals(c.yaml, prefab.SourceYaml))
                    return (c.min, c.max);
            var root = prefab.Instantiate();
            if (root == null) return null;
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            bool any = false;
            void Visit(Transform t)
            {
                foreach (var r in t.Entity.Components)
                {
                    if (r is not MeshRenderer mr || mr.Mesh == null) continue;
                    var bb = mr.Mesh.BoundingBox;
                    var w = t.WorldMatrix;
                    for (int i = 0; i < 8; i++)
                    {
                        var corner = new Vector3((i & 1) != 0 ? bb.Max.X : bb.Min.X, (i & 2) != 0 ? bb.Max.Y : bb.Min.Y,
                                                 (i & 4) != 0 ? bb.Max.Z : bb.Min.Z);
                        var p = Vector3.Transform(corner, w);
                        min = Vector3.Min(min, p);
                        max = Vector3.Max(max, p);
                        any = true;
                    }
                }
                for (int i = 0; i < t.Count; i++)
                    if (t.GetChild(i) is { } ch) Visit(ch);
            }
            root.Transform.Position = Vector3.Zero;
            Visit(root.Transform);
            root.Destroy();
            if (!any) return null;
            lock (_bounds) _bounds[prefab] = (prefab.SourceYaml, min, max);
            return (min, max);
        }
    }

    /// <summary>A bank found across a line by <see cref="BankProbe"/>: distances along <see cref="Down"/> from the probed point.</summary>
    public struct BankSection
    {
        /// <summary>Horizontal unit direction across the line towards the lower side.</summary>
        public Vector3 Down;
        public float CrestT, FootT;
        public float CrestHeight, FootHeight;
        public bool Found;
        public float Height => Found ? CrestHeight - FootHeight : 0f;
    }

    /// <summary>
    /// Finds the bank across a line: looks <see cref="Search"/> m to each side of a point, takes the side whose ground lies
    /// lower, and picks the steep run nearest the point (steeper than <see cref="Steep"/> degrees; flatter breaks of up to
    /// 2 m count as part of it). Its ends are the crest and the foot. Shared by the PCG node BankFrame and the project script Cliff
    /// component, so both read a bank the same way. Not thread-safe: one per caller (it keeps its height profile).
    /// </summary>
    public sealed class BankProbe
    {
        public const float Step = 0.5f;
        public readonly float Search;
        public readonly float Steep;

        private readonly int _steps;
        private readonly float[] _profile;
        private readonly float _tanSteepStep;

        public BankProbe(float search, float steepDegrees)
        {
            Search = MathF.Max(2f, search);
            Steep = Math.Clamp(steepDegrees, 1f, 89f);
            _steps = Math.Max(2, (int)MathF.Ceiling(Search / Step));
            _profile = new float[_steps * 2 + 1];
            _tanSteepStep = MathF.Tan(Steep * MathF.PI / 180f) * Step;
        }

        /// <summary>Ground height at distance t along the last probe's Down (t in [-Search, Search]).</summary>
        public float HeightAt(float t)
        {
            float f = Math.Clamp(t / Step + _steps, 0f, _profile.Length - 1.001f);
            int i = (int)f;
            return _profile[i] + (_profile[i + 1] - _profile[i]) * (f - i);
        }

        /// <param name="right">Horizontal unit vector across the line (the spline's right).</param>
        public BankSection Find(IHeightProvider ground, Vector3 p, Vector3 right)
        {
            // Downhill side: the side whose ground lies lower on average
            float side = 0f;
            for (int k = 1; k <= 3; k++)
            {
                float d = Search * k / 3f;
                side += ground.GetHeight(p - right * d) - ground.GetHeight(p + right * d);
            }
            var down = side >= 0f ? right : -right;

            for (int s = 0; s < _profile.Length; s++)
                _profile[s] = ground.GetHeight(p + down * ((s - _steps) * Step));

            int bestStart = -1, bestEnd = -1;
            float bestDist = float.MaxValue;
            for (int s = 0; s < _profile.Length - 1;)
            {
                if (_profile[s] - _profile[s + 1] <= _tanSteepStep) { s++; continue; }
                int start = s, end = s, gap = 0;
                for (int e = s; e < _profile.Length - 1; e++)
                {
                    if (_profile[e] - _profile[e + 1] > _tanSteepStep) { end = e; gap = 0; }
                    else if (++gap > 3) break;
                }
                float a = (start - _steps) * Step, b = (end + 1 - _steps) * Step;
                float dist = a > 0 ? a : b < 0 ? -b : 0f;
                if (dist < bestDist) { bestDist = dist; bestStart = start; bestEnd = end + 1; }
                s = end + 1;
            }

            var section = new BankSection { Down = down };
            if (bestStart >= 0)
            {
                section.Found = true;
                section.CrestT = (bestStart - _steps) * Step;
                section.FootT = (bestEnd - _steps) * Step;
                section.CrestHeight = _profile[bestStart];
                section.FootHeight = _profile[bestEnd];
            }
            else
            {
                section.CrestHeight = section.FootHeight = _profile[_steps];
            }
            return section;
        }
    }
}
