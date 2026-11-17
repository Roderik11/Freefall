using System;
using System.Collections.Generic;
using System.ComponentModel;
using Vortice.Mathematics;
using Freefall.Graph;
using System.Numerics;
using Freefall.Base;
using Freefall.Assets;
using Freefall.Components;
using Freefall.Graphics;

namespace Freefall.PCG
{
    public class SamplePointSet
    {
        public Vector3[] position;
        public Vector3[] extents;
        public Quaternion[] rotation;
        public float[] density;
        public string[] tags;
        public Vector3[] normal;

        public int Count => position != null ? position.Length : 0;

        /// <summary>
        /// Create an empty set.
        /// </summary>
        public static SamplePointSet Empty()
        {
            return new SamplePointSet
            {
                position = Array.Empty<Vector3>(),
                extents = Array.Empty<Vector3>(),
                rotation = Array.Empty<Quaternion>(),
                density = Array.Empty<float>(),
                tags = Array.Empty<string>(),
                normal = Array.Empty<Vector3>()
            };
        }

        /// <summary>
        /// Deep copy of this point set.
        /// </summary>
        public SamplePointSet Clone()
        {
            return new SamplePointSet
            {
                position = (Vector3[])position.Clone(),
                extents = (Vector3[])extents.Clone(),
                rotation = (Quaternion[])rotation.Clone(),
                density = (float[])density.Clone(),
                tags = tags != null ? (string[])tags.Clone() : Array.Empty<string>(),
                normal = normal != null ? (Vector3[])normal.Clone() : Array.Empty<Vector3>()
            };
        }

        /// <summary>
        /// Return a new set containing only points where the predicate is true.
        /// The predicate receives the index into the arrays.
        /// </summary>
        public SamplePointSet Filter(Func<int, bool> predicate)
        {
            var indices = new List<int>();
            for (int i = 0; i < Count; i++)
            {
                if (predicate(i))
                    indices.Add(i);
            }

            var result = new SamplePointSet
            {
                position = new Vector3[indices.Count],
                extents = new Vector3[indices.Count],
                rotation = new Quaternion[indices.Count],
                density = new float[indices.Count],
                tags = new string[indices.Count],
                normal = new Vector3[indices.Count]
            };

            for (int i = 0; i < indices.Count; i++)
            {
                int src = indices[i];
                result.position[i] = position[src];
                result.extents[i] = extents[src];
                result.rotation[i] = rotation[src];
                result.density[i] = density[src];
                if (tags != null && src < tags.Length)
                    result.tags[i] = tags[src];
                if (normal != null && src < normal.Length)
                    result.normal[i] = normal[src];
            }

            return result;
        }

        /// <summary>
        /// Concatenate two point sets into a new set.
        /// </summary>
        public static SamplePointSet Merge(SamplePointSet a, SamplePointSet b)
        {
            int countA = a?.Count ?? 0;
            int countB = b?.Count ?? 0;
            int total = countA + countB;

            var result = new SamplePointSet
            {
                position = new Vector3[total],
                extents = new Vector3[total],
                rotation = new Quaternion[total],
                density = new float[total],
                tags = new string[total],
                normal = new Vector3[total]
            };

            if (countA > 0)
            {
                Array.Copy(a.position, 0, result.position, 0, countA);
                Array.Copy(a.extents, 0, result.extents, 0, countA);
                Array.Copy(a.rotation, 0, result.rotation, 0, countA);
                Array.Copy(a.density, 0, result.density, 0, countA);
                if (a.tags != null) Array.Copy(a.tags, 0, result.tags, 0, Math.Min(a.tags.Length, countA));
                if (a.normal != null) Array.Copy(a.normal, 0, result.normal, 0, Math.Min(a.normal.Length, countA));
            }

            if (countB > 0)
            {
                Array.Copy(b.position, 0, result.position, countA, countB);
                Array.Copy(b.extents, 0, result.extents, countA, countB);
                Array.Copy(b.rotation, 0, result.rotation, countA, countB);
                Array.Copy(b.density, 0, result.density, countA, countB);
                if (b.tags != null) Array.Copy(b.tags, 0, result.tags, countA, Math.Min(b.tags.Length, countB));
                if (b.normal != null) Array.Copy(b.normal, 0, result.normal, countA, Math.Min(b.normal.Length, countB));
            }

            return result;
        }
    }

    [Category("Sampler")]
    public class Sampler : Node
    {
        [Output]
        public SamplePointSet Output;

        // distance between samples
        public int SampleDistance;
    }

    [Category("Math")]
    public class MakeVector : Node
    {
        [Input]
        public float X;

        [Input]
        public float Y;

        [Input]
        public float Z;

        [Output]
        public Vector3 Vector;
    }

    [Category("Tranform")]
    public class BoundsModifier : Node
    {
        [Input]
        public SamplePointSet Input;

        public Vector3 BoundsMin;
        public Vector3 BoundsMax;

        [Output]
        public SamplePointSet Output;
    }

    [Category("Tranform")]
    public class TranformPoints: Node
    {
        [Input]
        public SamplePointSet Input;

        public Vector3 OffsetMin;
        public Vector3 OffsetMax;

        public bool AbsoluteOffset;

        public Quaternion RotationMin = Quaternion.Identity;
        public Quaternion RotationMax = Quaternion.Identity;

        public bool AbsoluteRotation;

        public Vector3 ScaleMin = Vector3.One;
        public Vector3 ScaleMax = Vector3.One;

        public bool AbsoluteScale;

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

            var result = points.Clone();
            var rng = CreateRandom();

            for (int i = 0; i < result.Count; i++)
            {
                // Offset
                var offset = Lerp(OffsetMin, OffsetMax, (float)rng.NextDouble());
                if (AbsoluteOffset)
                {
                    result.position[i] += offset;
                }
                else
                {
                    // Local: rotate offset by the sample's existing orientation
                    result.position[i] += Vector3.Transform(offset, result.rotation[i]);
                }

                // Rotation
                var rot = Quaternion.Slerp(RotationMin, RotationMax, (float)rng.NextDouble());
                if (AbsoluteRotation)
                {
                    result.rotation[i] = rot;
                }
                else
                {
                    result.rotation[i] = Quaternion.Normalize(result.rotation[i] * rot);
                }

                // Scale (stored in extents)
                var scale = Lerp(ScaleMin, ScaleMax, (float)rng.NextDouble());
                if (AbsoluteScale)
                {
                    result.extents[i] = scale;
                }
                else
                {
                    result.extents[i] = new Vector3(
                        result.extents[i].X * scale.X,
                        result.extents[i].Y * scale.Y,
                        result.extents[i].Z * scale.Z);
                }
            }

            SetOutput("Output", result);
        }

        private static Vector3 Lerp(Vector3 a, Vector3 b, float t)
        {
            return new Vector3(
                a.X + (b.X - a.X) * t,
                a.Y + (b.Y - a.Y) * t,
                a.Z + (b.Z - a.Z) * t);
        }
    }

    [Category("Noise")]
    public class AttributeNoise : Node
    {
        [Input]
        public SamplePointSet Input;

        public float NoiseMin;
        public float MoiseMax;

        [Output]
        public SamplePointSet Output;
    }

    [Category("Filter")]
    public class DensityFilter : Node
    {
        [Input]
        public SamplePointSet Input;

        [ValueRange(0f, 1f)]
        public float LowerBound;
        [ValueRange(0f, 1f)]
        public float UpperBound = 1f;

        [Output]
        public SamplePointSet Output;

        /// <summary>Keeps points whose density lies in [LowerBound, UpperBound] (e.g. after DensityNoise).</summary>
        public override void Process()
        {
            var points = GetInputValue<SamplePointSet>("Input");
            if (points == null || points.Count == 0)
            {
                SetOutput("Output", SamplePointSet.Empty());
                return;
            }
            SetOutput("Output", points.Filter(i => points.density[i] >= LowerBound && points.density[i] <= UpperBound));
        }
    }

    /// <summary>
    /// Removes points that fall inside (the XZ footprint of) scene geometry: buildings, props, trees.
    /// Flat or huge renderers (street floors, pavement blocks, terrain-sized meshes, long walls whose AABB
    /// spans half the map) are ignored via MinHeight / MaxFootprint, as is this PCG component's own output
    /// and the output of PCG components that run after it (see PCGComponent.Sees).
    /// </summary>
    [Category("Filter")]
    public class ExcludeObstacles : Node, IWorldSpaceNode
    {
        [Input]
        public SamplePointSet Input;

        /// <summary>Local-to-world matrix of the PCG entity. Injected by PCGComponent.</summary>
        [Browsable(false)] [Freefall.Reflection.DontSerialize]
        public Matrix4x4 WorldMatrix { get; set; } = Matrix4x4.Identity;

        /// <summary>Extra clearance around each obstacle footprint, in meters.</summary>
        [ValueRange(0f, 10f)]
        public float Margin = 0.5f;

        /// <summary>Renderers flatter than this (world AABB height) are not obstacles.</summary>
        [ValueRange(0f, 20f)]
        public float MinHeight = 0.6f;

        /// <summary>Renderers whose AABB footprint exceeds this area (m²) are not obstacles.</summary>
        [ValueRange(1f, 100000f)]
        public float MaxFootprint = 3000f;

        [Output]
        public SamplePointSet Output;

        /// <summary>Spawned output of the executing PCGComponent. Injected before execution.</summary>
        [Browsable(false)]
        public Entity IgnoreRoot;

        /// <summary>The executing PCGComponent: decides which generated output counts. Injected before execution.</summary>
        [Browsable(false)] [Freefall.Reflection.DontSerialize]
        public PCGComponent Owner;

        public override void Process()
        {
            var points = GetInputValue<SamplePointSet>("Input");
            if (points == null || points.Count == 0)
            {
                SetOutput("Output", SamplePointSet.Empty());
                return;
            }

            // World positions once, and the area they cover: only obstacles reaching into it matter
            var m = WorldMatrix;
            var world = new Vector3[points.Count];
            var areaMin = new Vector2(float.MaxValue);
            var areaMax = new Vector2(float.MinValue);
            for (int i = 0; i < world.Length; i++)
            {
                world[i] = Vector3.Transform(points.position[i], m);
                var xz = new Vector2(world[i].X, world[i].Z);
                areaMin = Vector2.Min(areaMin, xz);
                areaMax = Vector2.Max(areaMax, xz);
            }

            var gatherScope = _gatherMarker.Auto();
            var boxes = new List<Vector4>(); // minX, minZ, maxX, maxZ, grown by the margin
            var cache = ObstacleCache.Current;

            // Authored content and what scripts build (houses, walls): always counts
            Gather(cache.Authored(), boxes, areaMin, areaMax);

            // Generated output counts only from components that run before this one, even while later output is in
            // the scene (PCGComponent.Sees)
            foreach (var pcg in ComponentCache<PCGComponent>.All)
            {
                if (pcg == Owner || pcg.Output is not { } output || IsUnder(output, IgnoreRoot)) continue;
                if (Owner != null && PCGComponent.CompareOrder(pcg, Owner) >= 0) continue;
                Gather(cache.Generated(pcg, output), boxes, areaMin, areaMax);
            }
            gatherScope.Dispose();

            using var filterScope = _filterMarker.Auto();
            SetOutput("Output", points.Filter(i =>
            {
                var p = world[i];
                foreach (var b in boxes)
                    if (p.X >= b.X && p.X <= b.Z && p.Z >= b.Y && p.Z <= b.W) return false;
                return true;
            }));
        }

        private static readonly ProfilerMarker _gatherMarker = ProfilerMarker.Get("ExcludeObstacles gather");
        private static readonly ProfilerMarker _filterMarker = ProfilerMarker.Get("ExcludeObstacles filter");

        /// <summary>The boxes of a set that reach into the area, grown by the margin, if they pass this node's filters.</summary>
        private void Gather(ObstacleCache.Set set, List<Vector4> boxes, Vector2 areaMin, Vector2 areaMax)
        {
            if (set.Boxes.Count == 0) return;
            // Nothing of the set comes near the area (a forest at the other end of the map)
            if (set.Max.X + Margin < areaMin.X || set.Min.X - Margin > areaMax.X
                || set.Max.Y + Margin < areaMin.Y || set.Min.Y - Margin > areaMax.Y) return;

            foreach (var o in set.Boxes)
            {
                if (o.Height < MinHeight) continue;
                if (o.Footprint > MaxFootprint) continue;

                var box = new Vector4(o.Min.X - Margin, o.Min.Y - Margin, o.Max.X + Margin, o.Max.Y + Margin);
                if (box.Z < areaMin.X || box.X > areaMax.X || box.W < areaMin.Y || box.Y > areaMax.Y) continue;
                boxes.Add(box);
            }
        }

        private static bool IsUnder(Entity entity, Entity root)
        {
            if (root == null) return false;
            for (var t = entity?.Transform; t != null; t = t.Parent)
                if (t.Entity == root) return true;
            return false;
        }
    }

    /// <summary>
    /// World boxes of the scene's renderers for ExcludeObstacles, worked out once per batch of PCG runs (a scheduler
    /// flush, "run all") instead of once per run: an Eldermoor load runs ExcludeObstacles ~140 times, and each run used
    /// to transform the boxes of all ~80,000 renderers again (1.7 s, then 1 s with a cheap reject first).
    /// Within a batch, what no PCG component generated stays where it is (scripts move things in their own updates,
    /// not during a flush). Generated output is kept per output container, and a run that regenerates puts out a new
    /// container. Outside a batch (a single Execute) every run gets a fresh cache: the same work as the old scan.
    /// </summary>
    internal sealed class ObstacleCache
    {
        internal readonly struct Box
        {
            public readonly Vector2 Min, Max;   // world XZ
            public readonly float Height, Footprint;
            public Box(Vector3 min, Vector3 max)
            {
                Min = new Vector2(min.X, min.Z);
                Max = new Vector2(max.X, max.Z);
                Height = max.Y - min.Y;
                Footprint = (max.X - min.X) * (max.Z - min.Z);
            }
        }

        internal sealed class Set
        {
            public readonly List<Box> Boxes = new();
            public Vector2 Min = new(float.MaxValue), Max = new(float.MinValue);   // XZ bounds of all boxes

            public void Add(MeshRenderer renderer)
            {
                if (renderer is not { Enabled: true, Mesh: { } mesh }) return;
                // A world cells' far mesh (Mesh.CellFade) stands in for a whole generator's placements: its box would be
                // the forest's or the bank's. What it stands for is in the generator's records.
                if (mesh.CellFade) return;
                Add(mesh, renderer.Entity.Transform.WorldMatrix);
            }

            /// <summary>A mesh's box under a world matrix: a renderer's, or a world cells' record's renderer.</summary>
            public void Add(Mesh mesh, in Matrix4x4 transform)
            {
                var bb = mesh.BoundingBox;
                var min = new Vector3(float.MaxValue);
                var max = new Vector3(float.MinValue);
                for (int c = 0; c < 8; c++)
                {
                    var corner = new Vector3((c & 1) != 0 ? bb.Max.X : bb.Min.X,
                                             (c & 2) != 0 ? bb.Max.Y : bb.Min.Y,
                                             (c & 4) != 0 ? bb.Max.Z : bb.Min.Z);
                    var p = Vector3.Transform(corner, transform);
                    min = Vector3.Min(min, p);
                    max = Vector3.Max(max, p);
                }

                var box = new Box(min, max);
                Boxes.Add(box);
                Min = Vector2.Min(Min, box.Min);
                Max = Vector2.Max(Max, box.Max);
            }
        }

        private static ObstacleCache? _batch;
        private static int _depth;

        private Set? _authored;
        private readonly Dictionary<Entity, Set> _generated = new();

        /// <summary>The batch's cache, or a fresh one outside a batch.</summary>
        public static ObstacleCache Current => _batch ?? new ObstacleCache();

        /// <summary>From here until the matching <see cref="EndBatch"/> (nested batches share the outer one) runs share one cache. Main thread.</summary>
        public static void BeginBatch()
        {
            if (_depth++ == 0) _batch = new ObstacleCache();
        }

        public static void EndBatch()
        {
            if (_depth > 0 && --_depth == 0) _batch = null;
        }

        /// <summary>
        /// Renderers no PCG component spawned, with the world cells' records of the spawners that are not PCG components
        /// (the kits' houses, walls and holds), each with the matrix its entity would have; the near cells' copies are left
        /// out, their records stand for them.
        /// </summary>
        public Set Authored()
        {
            if (_authored != null) return _authored;
            var set = new Set();
            foreach (var renderer in ComponentCache<MeshRenderer>.All)
                if (renderer.Enabled && renderer.Mesh != null && PCGComponent.GeneratorOf(renderer.Entity) == null
                    && !Freefall.World.WorldCells.IsCellCopy(renderer.Entity))
                    set.Add(renderer);
            Freefall.World.WorldCells.ForEachRenderer(owner => owner is not PCGComponent, (rd, world) =>
            {
                if (rd.Enabled) set.Add(rd.Mesh, world);
            });
            return _authored = set;
        }

        /// <summary>
        /// The renderers <paramref name="pcg"/> spawned into <paramref name="output"/> (not those of a PCG component nested
        /// in it), and with <see cref="PCGComponent.Cells"/> its records in the world cells, each with the matrix its
        /// entity would have: the same boxes whichever cells are spawned, and the same as with every placement an entity.
        /// The near cells' copies are left out, their records stand for them.
        /// </summary>
        public Set Generated(PCGComponent pcg, Entity output)
        {
            if (_generated.TryGetValue(output, out var set)) return set;
            set = new Set();
            foreach (var renderer in output.GetComponentsInChildren<MeshRenderer>())
                if (PCGComponent.GeneratorOf(renderer.Entity) == pcg && !Freefall.World.WorldCells.IsCellCopy(renderer.Entity))
                    set.Add(renderer);
            Freefall.World.WorldCells.ForEachRenderer(pcg, (rd, world) =>
            {
                if (rd.Enabled) set.Add(rd.Mesh, world);
            });
            _generated[output] = set;
            return set;
        }
    }

    [Category("Filter")]
    public class PointFilter : Node
    {
        [Input]
        public SamplePointSet Input;

        public float LowerBound;
        public float UpperBound;

        [Output]
        public SamplePointSet Output;
    }


    [Category("Noise")]
    public class DensityNoise : Node, IWorldSpaceNode
    {
        [Input]
        public SamplePointSet Input;

        [ValueRange(0f, 1f)]
        public float LowerBound;
        [ValueRange(0f, 1f)]
        public float UpperBound = 1f;

        /// <summary>World size of one noise cell in meters. Larger = broader clumps and gaps (a forest's clearings: 100-200).</summary>
        [ValueRange(0.1f, 400f)]
        public float Scale = 1f;

        /// <summary>
        /// Sample the noise at the points' world positions instead of their positions relative to the entity: areas
        /// that share a graph get patterns of their own, and none has a gap at its pivot (the noise is 0 at the origin).
        /// </summary>
        public bool WorldSpace;

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

            var result = points.Clone();
            float range = UpperBound - LowerBound;
            float invScale = 1f / MathF.Max(Scale, 0.001f);

            for (int i = 0; i < result.Count; i++)
            {
                var p = WorldSpace ? Vector3.Transform(result.position[i], WorldMatrix) : result.position[i];
                float n = HashNoise(p.X * invScale, p.Z * invScale);
                float value = LowerBound + n * range;
                result.density[i] *= value;
            }

            SetOutput("Output", result.Filter(i => result.density[i] > 0));
        }

        private static float HashNoise(float x, float z)
        {
            int ix = (int)MathF.Floor(x);
            int iz = (int)MathF.Floor(z);
            float fx = x - ix;
            float fz = z - iz;
            fx = fx * fx * (3f - 2f * fx);
            fz = fz * fz * (3f - 2f * fz);

            float a = Hash(ix, iz);
            float b = Hash(ix + 1, iz);
            float c = Hash(ix, iz + 1);
            float d = Hash(ix + 1, iz + 1);

            float ab = a + (b - a) * fx;
            float cd = c + (d - c) * fx;
            return ab + (cd - ab) * fz;
        }

        private static float Hash(int x, int z)
        {
            int h = x * 374761393 + z * 668265263;
            h = (h ^ (h >> 13)) * 1274126177;
            return (h & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        }
    }

    [Category("Filter")]
    public class Difference: Node
    {
        [Input]
        public SamplePointSet InputA;

        [Input]
        public SamplePointSet InputB;

        public DensityFunction DensityFunction = DensityFunction.Minimum;

        /// <summary>
        /// Metres round each point of B, times its scale, inside which A's points go (the forest's trees round an
        /// ancient one). 0 = only A's points that sit exactly on one of B's.
        /// </summary>
        public float Radius;

        [Output]
        public SamplePointSet Output;

        /// <summary>A's points that are not within Radius (times its scale) of a point of B, on the ground plane.</summary>
        public override void Process()
        {
            var a = GetInputValue<SamplePointSet>("InputA");
            var b = GetInputValue<SamplePointSet>("InputB");
            if (a == null || a.Count == 0)
            {
                SetOutput("Output", SamplePointSet.Empty());
                return;
            }
            if (b == null || b.Count == 0)
            {
                SetOutput("Output", a);
                return;
            }

            // B's points in a grid of cells as large as the largest clearance, so each of A's looks at 3 x 3 cells
            var reach = new float[b.Count];
            float maxReach = 0.01f;
            for (int j = 0; j < b.Count; j++)
            {
                var e = b.extents[j];
                reach[j] = MathF.Max(Radius * MathF.Max(e.X, e.Z), 0.01f);
                maxReach = MathF.Max(maxReach, reach[j]);
            }
            float invCell = 1f / maxReach;
            var grid = new Dictionary<long, List<int>>();
            for (int j = 0; j < b.Count; j++)
            {
                long key = ((long)(int)MathF.Floor(b.position[j].X * invCell) << 32) | (uint)(int)MathF.Floor(b.position[j].Z * invCell);
                if (!grid.TryGetValue(key, out var cell)) grid[key] = cell = new List<int>();
                cell.Add(j);
            }

            SetOutput("Output", a.Filter(i =>
            {
                var p = a.position[i];
                int cx = (int)MathF.Floor(p.X * invCell), cz = (int)MathF.Floor(p.Z * invCell);
                for (int dx = -1; dx <= 1; dx++)
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        if (!grid.TryGetValue(((long)(cx + dx) << 32) | (uint)(cz + dz), out var cell)) continue;
                        foreach (int j in cell)
                        {
                            float ex = b.position[j].X - p.X, ez = b.position[j].Z - p.Z;
                            if (ex * ex + ez * ez < reach[j] * reach[j]) return false;
                        }
                    }
                return true;
            }));
        }
    }

    /// <summary>
    /// Grows a small group round every input point: the point itself and up to MaxCount - 1 companions within Radius of
    /// it, none closer than Spacing to another of the group (a boulder with one or two smaller ones beside it, an ore
    /// outcrop of a few rocks). The group's size is drawn between MinCount and MaxCount, Bias above 1 making small groups
    /// commoner; a companion is scaled by CompanionScale (a share of the leader's scale, which also scales the radius and
    /// the spacing) and turned at random. Each group draws from a random of its own, made from the leader's position, so
    /// editing one part of an area does not reshuffle the groups elsewhere. Put it before the terrain projection and the
    /// filters: companions are projected and filtered like any point.
    /// </summary>
    [Category("Transform")]
    public class ClusterPoints : Node
    {
        [Input]
        public SamplePointSet Input;

        public int MinCount = 1;
        public int MaxCount = 4;

        /// <summary>Above 1, small groups are commoner than big ones (the size is MinCount + range x r^Bias).</summary>
        public float Bias = 1.4f;

        /// <summary>Metres from the leader a companion may stand (times the leader's scale).</summary>
        public float Radius = 4f;

        /// <summary>Least distance between two of a group, metres (times the leader's scale).</summary>
        public float Spacing = 1.8f;

        /// <summary>A companion's scale as a share of the leader's, drawn between X and Y.</summary>
        public Vector2 CompanionScale = new Vector2(0.45f, 0.85f);

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

            var pos = new List<Vector3>(); var ext = new List<Vector3>(); var rot = new List<Quaternion>();
            var den = new List<float>(); var tags = new List<string>(); var nrm = new List<Vector3>();
            int lo = Math.Max(1, MinCount), hi = Math.Max(lo, MaxCount);
            var group = new List<Vector3>(hi);

            for (int i = 0; i < points.Count; i++)
            {
                var p = points.position[i];
                var rng = new Random(PlaceSeed(Seed, (int)MathF.Floor(p.X * 8f), (int)MathF.Floor(p.Z * 8f)));
                float leaderScale = MathF.Max(points.extents[i].X, points.extents[i].Z);
                int n = lo + (int)MathF.Floor((hi - lo + 1) * MathF.Pow((float)rng.NextDouble(), MathF.Max(Bias, 0.01f)));
                n = Math.Min(n, hi);

                group.Clear();
                group.Add(p);
                Add(i, p, points.extents[i], points.rotation[i]);
                float radius = Radius * leaderScale, spacing = Spacing * leaderScale;
                for (int k = 1; k < n; k++)
                {
                    for (int attempt = 0; attempt < 12; attempt++)
                    {
                        float angle = (float)rng.NextDouble() * MathF.Tau;
                        float dist = spacing + (float)rng.NextDouble() * MathF.Max(radius - spacing, 0f);
                        var c = p + new Vector3(MathF.Cos(angle) * dist, 0f, MathF.Sin(angle) * dist);
                        bool clear = true;
                        foreach (var g in group)
                        {
                            float dx = g.X - c.X, dz = g.Z - c.Z;
                            if (dx * dx + dz * dz < spacing * spacing) { clear = false; break; }
                        }
                        if (!clear) continue;

                        float s = CompanionScale.X + (float)rng.NextDouble() * (CompanionScale.Y - CompanionScale.X);
                        var yaw = Quaternion.CreateFromAxisAngle(Vector3.UnitY, (float)rng.NextDouble() * MathF.Tau);
                        group.Add(c);
                        Add(i, c, points.extents[i] * s, Quaternion.Normalize(points.rotation[i] * yaw));
                        break;
                    }
                }
            }

            SetOutput("Output", new SamplePointSet
            {
                position = pos.ToArray(), extents = ext.ToArray(), rotation = rot.ToArray(), density = den.ToArray(),
                tags = tags.ToArray(), normal = nrm.ToArray()
            });

            void Add(int src, Vector3 p, Vector3 e, Quaternion r)
            {
                pos.Add(p); ext.Add(e); rot.Add(r); den.Add(points.density[src]);
                tags.Add(points.tags != null && src < points.tags.Length ? points.tags[src] : null);
                nrm.Add(points.normal != null && src < points.normal.Length ? points.normal[src] : Vector3.UnitY);
            }
        }

        // A leader's seed from the node's and its place (an eighth of a metre). The same in every process: until
        // 2026-11-08 it was HashCode.Combine, which .NET seeds anew in each process, so every forest graph that clusters
        // (Forest, SpruceForest, BeechForest) came out different on each editor start
        private static int PlaceSeed(int seed, int x, int z)
        {
            uint h = (uint)seed;
            h = Mix(h ^ (uint)x);
            h = Mix(h ^ (uint)z);
            return (int)h;

            // murmur3's finalizer, after a step of the golden ratio
            static uint Mix(uint v)
            {
                v += 0x9E3779B9u;
                v ^= v >> 16; v *= 0x85EBCA6Bu;
                v ^= v >> 13; v *= 0xC2B2AE35u;
                v ^= v >> 16;
                return v;
            }
        }
    }

    [Category("Density")]
    public class RemapDensity : Node
    {
        [Input]
        public SamplePointSet Input;

        public Vector2 OldRange;
        public Vector2 NewRange;

        [Output]
        public SamplePointSet Output;
    }

    [Category("Filter")]
    public class Union : Node
    {
        [Input]
        public SamplePointSet InputA;

        [Input]
        public SamplePointSet InputB;

        public DensityFunction DensityFunction = DensityFunction.Maximum;

        [Output]
        public SamplePointSet Output;

        /// <summary>Concatenates both inputs (e.g. the left and right verge of a road). Either input may be unconnected.</summary>
        public override void Process()
        {
            SetOutput("Output", SamplePointSet.Merge(
                GetInputValue<SamplePointSet>("InputA"),
                GetInputValue<SamplePointSet>("InputB")));
        }
    }

    [Category("Spawn")]
    public class SpawnMesh : Node
    {
        [Serializable]
        public class MeshElement
        {
            public Mesh Mesh;
            public float Weight = 1;
        }

        [Input]
        public SamplePointSet Input;

        public List<MeshElement> Meshes = new List<MeshElement>();
    }

    [Category("Spawn")]
    public class SpawnPrefab : Node
    {
        [Serializable]
        public class PrefabElement
        {
            public Prefab Prefab;
            public float Weight = 1;
        }

        [Input]
        public SamplePointSet Input;

        public List<PrefabElement> Entities = new List<PrefabElement>();

        /// <summary>With the generator's world cells (PCGComponent.Cells): metres from the camera where these placements'
        /// instances give way to their far form; 0 = the generator's Ring.</summary>
        [Description("With the generator's Cells on: metres from the camera where these placements' instances give way to their far form; 0 = the generator's Ring. Small things leave sooner anyway, where they are too small to draw.")]
        public float Ring;

        /// <summary>With the generator's world cells: these placements' far form, or the generator's.</summary>
        [Description("With the generator's Cells on: what stands in for these placements past their ring, or the generator's Far.")]
        public Freefall.World.SpawnFar Far = Freefall.World.SpawnFar.Generator;

        /// <summary>
        /// Parent entity for spawned instances. Set by PCGComponent before execution.
        /// </summary>
        [Browsable(false)]
        public Entity SpawnParent;

        /// <summary>
        /// Takes the placements instead of instances when the PCG component's entity has one (a cell spawner). Set by
        /// PCGComponent before execution.
        /// </summary>
        [Browsable(false)]
        public IPlacementSink? Sink;

        /// <summary>
        /// Closed spline only: every placement keeps its whole footprint inside the outline, <see cref="InsideMargin"/>
        /// in from it. A piece's footprint is the circle round its pivot that holds its mesh bounds, at the point's
        /// scale. Where the piece drawn does not fit, one of the list that fits is drawn by weight (a full oak where
        /// there is room, an apple nearer the edge); where none fits, the point is left out.
        /// </summary>
        [Description("Closed spline only: each placement keeps its whole footprint (the circle round its pivot that holds its bounds, at its scale) inside the outline. Where the piece drawn does not fit, one of the list that fits is drawn by weight; where none fits, the point is left out.")]
        public bool KeepInside;

        /// <summary>With KeepInside: how far inside the outline the footprints stay (metres).</summary>
        [Description("With Keep Inside: how far in from the outline the footprints stay (m), e.g. clear of a hedge round the area.")]
        [ValueRange(0f, 20f)]
        public float InsideMargin;

        /// <summary>The PCG entity's spline, for KeepInside. Set by PCGComponent before execution.</summary>
        [Browsable(false)]
        public Spline Spline;

        public override void Process()
        {
            var points = GetInputValue<SamplePointSet>("Input");
            if (points == null || points.Count == 0) return;
            if (Entities == null || Entities.Count == 0) return;

            // Compute total weight
            float totalWeight = 0;
            foreach (var e in Entities)
            {
                if (e.Prefab != null) totalWeight += e.Weight;
            }
            if (totalWeight <= 0) return;

            var rng = CreateRandom();
            var parent = SpawnParent ?? new Entity("PCG_Spawned");

            // KeepInside: each piece's footprint radius, measured once
            float[] footprint = null;
            IList<Vector3> outline = null;
            if (KeepInside && Spline is { Closed: true } && Spline.Points.Count >= 3)
            {
                outline = Spline.Points;
                footprint = new float[Entities.Count];
                for (int k = 0; k < Entities.Count; k++)
                    footprint[k] = Entities[k]?.Prefab is { } piece && PrefabBounds.Measure(piece) is { } b ? FootprintRadius(b.min, b.max) : 0f;
            }

            int left = 0;
            for (int i = 0; i < points.Count; i++)
            {
                // A point that names its prefab ("prefab=N", an index into Entities: BankWall) gets that one
                int drawn = PickIndex(rng, totalWeight);
                var prefab = drawn >= 0 ? Entities[drawn].Prefab : null;
                bool named = false;
                if (points.tags != null && i < points.tags.Length && points.tags[i] is { } t && t.StartsWith("prefab=")
                    && int.TryParse(t.AsSpan(7), out int chosen) && chosen >= 0 && chosen < Entities.Count
                    && Entities[chosen]?.Prefab != null)
                {
                    prefab = Entities[chosen].Prefab;
                    named = true;
                }
                if (prefab == null) continue;

                if (footprint != null && !named)
                {
                    var ext = points.extents[i];
                    float scale = MathF.Max(MathF.Abs(ext.X), MathF.Abs(ext.Z));
                    float room = RoomInside(points.position[i], outline) - InsideMargin;
                    if (footprint[drawn] * scale > room)
                    {
                        int fit = PickFitting(rng, footprint, scale, room);
                        if (fit < 0) { left++; continue; }
                        prefab = Entities[fit].Prefab;
                    }
                }

                if (Sink != null)
                {
                    if (Ring > 0f || Far != Freefall.World.SpawnFar.Generator)
                        Sink.Place(prefab, points.position[i], points.rotation[i], points.extents[i], Ring, Far);
                    else
                        Sink.Place(prefab, points.position[i], points.rotation[i], points.extents[i]);
                    continue;
                }

                var instance = prefab.Instantiate();
                if (instance == null) continue;

                instance.Transform.Parent = parent.Transform;
                instance.Transform.Position = points.position[i];
                instance.Transform.Rotation = points.rotation[i];
                instance.Transform.Scale = points.extents[i];
            }

            Debug.Log($"[SpawnPrefab] {(Sink != null ? "Placed" : "Spawned")} {points.Count - left} instances under '{parent.Name}'" +
                      (left > 0 ? $" ({left} left out: nothing fitted inside)" : ""));
        }

        // The same draw as before KeepInside existed (one roll per point), as an index into Entities
        private int PickIndex(Random rng, float totalWeight)
        {
            float roll = (float)rng.NextDouble() * totalWeight;
            float cumulative = 0;
            for (int k = 0; k < Entities.Count; k++)
            {
                if (Entities[k]?.Prefab == null) continue;
                cumulative += Entities[k].Weight;
                if (roll <= cumulative) return k;
            }
            for (int k = Entities.Count - 1; k >= 0; k--)
                if (Entities[k]?.Prefab != null) return k;
            return -1;
        }

        // A draw by weight among the pieces whose footprint fits the room; -1 if none does
        private int PickFitting(Random rng, float[] footprint, float scale, float room)
        {
            float total = 0;
            for (int k = 0; k < Entities.Count; k++)
                if (Entities[k]?.Prefab != null && footprint[k] * scale <= room) total += Entities[k].Weight;
            if (total <= 0) return -1;
            float roll = (float)rng.NextDouble() * total, cumulative = 0;
            int last = -1;
            for (int k = 0; k < Entities.Count; k++)
            {
                if (Entities[k]?.Prefab == null || footprint[k] * scale > room) continue;
                cumulative += Entities[k].Weight;
                last = k;
                if (roll <= cumulative) return k;
            }
            return last;
        }

        /// <summary>The circle round the pivot (XZ) that holds a footprint box: its centre's distance plus half its longer side.</summary>
        private static float FootprintRadius(Vector3 min, Vector3 max)
        {
            var centre = new Vector2((min.X + max.X) * 0.5f, (min.Z + max.Z) * 0.5f);
            return centre.Length() + 0.5f * MathF.Max(max.X - min.X, max.Z - min.Z);
        }

        /// <summary>How far a point lies inside a closed outline (XZ): its distance to the nearest edge, negative outside.</summary>
        private static float RoomInside(Vector3 p, IList<Vector3> outline)
        {
            int n = outline.Count;
            bool inside = false;
            float nearestSq = float.MaxValue;
            for (int i = 0, j = n - 1; i < n; j = i++)
            {
                var a = new Vector2(outline[i].X, outline[i].Z);
                var b = new Vector2(outline[j].X, outline[j].Z);
                if ((a.Y > p.Z) != (b.Y > p.Z) && p.X < (b.X - a.X) * (p.Z - a.Y) / (b.Y - a.Y) + a.X) inside = !inside;
                var ab = b - a;
                var q = new Vector2(p.X, p.Z);
                float t = Math.Clamp(Vector2.Dot(q - a, ab) / MathF.Max(ab.LengthSquared(), 1e-6f), 0f, 1f);
                nearestSq = MathF.Min(nearestSq, Vector2.DistanceSquared(q, a + ab * t));
            }
            float d = MathF.Sqrt(nearestSq);
            return inside ? d : -d;
        }
    }

    [Category("Filter")]
    public class SelfPruning : Node
    {
        [Input]
        public SamplePointSet Input;

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

            // Determine cell size from max extent across all points
            float maxExtent = 0;
            for (int i = 0; i < points.Count; i++)
            {
                var e = points.extents[i];
                float r = MathF.Max(e.X, MathF.Max(e.Y, e.Z));
                if (r > maxExtent) maxExtent = r;
            }

            float cellSize = maxExtent * 2f;
            if (cellSize <= 0) cellSize = 1f;
            float invCell = 1f / cellSize;

            var grid = new Dictionary<long, List<int>>();
            var accepted = new HashSet<int>();

            for (int i = 0; i < points.Count; i++)
            {
                var pos = points.position[i];
                var ext = points.extents[i];
                float ri = MathF.Max(ext.X, MathF.Max(ext.Y, ext.Z));

                int cx = (int)MathF.Floor(pos.X * invCell);
                int cz = (int)MathF.Floor(pos.Z * invCell);

                bool conflict = false;
                for (int dx = -1; dx <= 1 && !conflict; dx++)
                {
                    for (int dz = -1; dz <= 1 && !conflict; dz++)
                    {
                        long key = ((long)(cx + dx) << 32) | (uint)(cz + dz);
                        if (!grid.TryGetValue(key, out var cell)) continue;

                        for (int j = 0; j < cell.Count; j++)
                        {
                            int other = cell[j];
                            var oext = points.extents[other];
                            float rj = MathF.Max(oext.X, MathF.Max(oext.Y, oext.Z));
                            float dist = ri + rj;
                            var delta = points.position[other] - pos;
                            if (delta.X * delta.X + delta.Z * delta.Z < dist * dist)
                            {
                                conflict = true;
                                break;
                            }
                        }
                    }
                }

                if (!conflict)
                {
                    accepted.Add(i);
                    long myKey = ((long)cx << 32) | (uint)cz;
                    if (!grid.TryGetValue(myKey, out var myCell))
                    {
                        myCell = new List<int>();
                        grid[myKey] = myCell;
                    }
                    myCell.Add(i);
                }
            }

            SetOutput("Output", points.Filter(i => accepted.Contains(i)));
        }
    }

    public enum DensityFunction
    {
        Maximum,
        Minimum
    }
}
