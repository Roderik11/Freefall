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
        public float UpperBound;

        [Output]
        public SamplePointSet Output;
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
    public class DensityNoise : Node
    {
        [Input]
        public SamplePointSet Input;

        [ValueRange(0f, 1f)]
        public float LowerBound;
        [ValueRange(0f, 1f)]
        public float UpperBound = 1f;

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

            for (int i = 0; i < result.Count; i++)
            {
                float n = HashNoise(result.position[i].X, result.position[i].Z);
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

        [Output]
        public SamplePointSet Output;
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

        /// <summary>
        /// Parent entity for spawned instances. Set by PCGComponent before execution.
        /// </summary>
        [Browsable(false)]
        public Entity SpawnParent;

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

            for (int i = 0; i < points.Count; i++)
            {
                var prefab = PickWeighted(rng, totalWeight);
                if (prefab == null) continue;

                var instance = prefab.Instantiate();
                if (instance == null) continue;

                instance.Transform.Parent = parent.Transform;
                instance.Transform.Position = points.position[i];
                instance.Transform.Rotation = points.rotation[i];
                instance.Transform.Scale = points.extents[i];
            }

            Debug.Log($"[SpawnPrefab] Spawned {points.Count} instances under '{parent.Name}'");
        }

        private Prefab PickWeighted(Random rng, float totalWeight)
        {
            float roll = (float)rng.NextDouble() * totalWeight;
            float cumulative = 0;
            foreach (var e in Entities)
            {
                if (e.Prefab == null) continue;
                cumulative += e.Weight;
                if (roll <= cumulative) return e.Prefab;
            }
            return Entities[Entities.Count - 1].Prefab;
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
