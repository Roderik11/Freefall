using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;

namespace Freefall.Navigation
{
    /// <summary>
    /// Deterministic 64-bit hashing for bake inputs. The hashes are stored with the baked tiles,
    /// so System.HashCode (seeded per process) cannot be used.
    /// </summary>
    internal static class NavHash
    {
        public static ulong Mix(ulong h, ulong v)
        {
            ulong x = h * 0x9E3779B97F4A7C15UL + v;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
            return x ^ (x >> 31);
        }

        public static ulong Mix(ulong h, float v) => Mix(h, BitConverter.SingleToUInt32Bits(v));

        public static ulong Mix(ulong h, Vector3 v) => Mix(Mix(Mix(h, v.X), v.Y), v.Z);

        public static ulong Bytes(ulong h, ReadOnlySpan<byte> data)
        {
            var words = MemoryMarshal.Cast<byte, ulong>(data);
            foreach (ulong word in words)
                h = Mix(h, word);

            ulong tail = 0;
            for (int i = words.Length * sizeof(ulong); i < data.Length; i++)
                tail = (tail << 8) | data[i];

            return Mix(Mix(h, tail), (ulong)data.Length);
        }
    }

    /// <summary>A terrain's height data as the bake sees it.</summary>
    internal sealed class TerrainSource
    {
        public readonly float[,] Heights;
        public readonly Vector3 Position;
        public readonly Vector2 Size;
        public readonly float MaxHeight;
        public readonly ulong Hash;

        private readonly int _dimX;
        private readonly int _dimZ;

        public TerrainSource(float[,] heights, Vector3 position, Vector2 size, float maxHeight)
        {
            Heights = heights;
            Position = position;
            Size = size;
            MaxHeight = maxHeight;
            _dimX = heights.GetLength(0) - 1;
            _dimZ = heights.GetLength(1) - 1;

            ulong h = NavHash.Mix(NavHash.Mix(1, position), maxHeight);
            Hash = NavHash.Mix(NavHash.Mix(h, size.X), size.Y);
        }

        /// <summary>
        /// World height at a terrain-local position. Uses the collider's mapping (first and last
        /// sample sit on the terrain's edges), so the navmesh lies where characters actually stand.
        /// </summary>
        public float SampleHeight(float localX, float localZ)
        {
            float fx = Math.Clamp(localX / Size.X * _dimX, 0, _dimX);
            float fz = Math.Clamp(localZ / Size.Y * _dimZ, 0, _dimZ);

            int x0 = (int)fx, z0 = (int)fz;
            int x1 = Math.Min(x0 + 1, _dimX), z1 = Math.Min(z0 + 1, _dimZ);
            float tx = fx - x0, tz = fz - z0;

            float h0 = Heights[x0, z0] + (Heights[x1, z0] - Heights[x0, z0]) * tx;
            float h1 = Heights[x0, z1] + (Heights[x1, z1] - Heights[x0, z1]) * tx;

            return Position.Y + (h0 + (h1 - h0) * tz) * MaxHeight;
        }
    }

    /// <summary>
    /// The triangles of one mesh that take part in the bake (highest LOD only), cut into spatially
    /// compact chunks. Shared by every instance of the mesh, so a tile only has to transform the
    /// chunks of an instance that actually reach into it.
    /// </summary>
    internal sealed class MeshGeometry
    {
        public const int ChunkTriangles = 256;

        public struct Chunk
        {
            public int VertexStart, VertexCount;
            public int IndexStart, IndexCount;
            public Vector3 Center, Extents;
        }

        /// <summary>Chunk-local vertices, chunk after chunk.</summary>
        public Vector3[] Positions = [];

        /// <summary>Triangle indices, relative to the owning chunk's VertexStart.</summary>
        public int[] Indices = [];

        public Chunk[] Chunks = [];
        public Vector3 Center, Extents;
        public ulong Hash;

        private Vector3[]? _sourcePositions;
        private uint[]? _sourceIndices;
        private (int baseVertex, int start, int count)[]? _ranges;

        /// <summary>Capture what to bake from a mesh. Main thread; the heavy part is Prepare().</summary>
        public static MeshGeometry? FromMesh(Mesh mesh)
        {
            var positions = mesh.Positions;
            var indices = mesh.CpuIndices;
            if (positions == null || indices == null || positions.Length == 0 || indices.Length < 3)
                return null;

            var parts = mesh.MeshParts;
            var ranges = new List<(int, int, int)>();

            void AddPart(int partIndex)
            {
                if (partIndex < 0 || partIndex >= parts.Count) return;
                var part = parts[partIndex];
                if (part.Enabled && part.NumIndices >= 3)
                    ranges.Add((part.BaseVertex, part.BaseIndex, part.NumIndices));
            }

            if (parts.Count == 0)
            {
                ranges.Add((0, 0, indices.Length));
            }
            else if (mesh.LODs.Count > 0)
            {
                // Only the highest-detail geometry: LOD0 plus parts outside the LOD chain
                if (mesh.LODs[0].MeshPartIndices != null)
                    foreach (int partIndex in mesh.LODs[0].MeshPartIndices) AddPart(partIndex);
                if (mesh.NonLodPartIndices != null)
                    foreach (int partIndex in mesh.NonLodPartIndices) AddPart(partIndex);
            }
            else
            {
                for (int partIndex = 0; partIndex < parts.Count; partIndex++) AddPart(partIndex);
            }

            if (ranges.Count == 0) return null;

            return new MeshGeometry
            {
                _sourcePositions = positions,
                _sourceIndices = indices,
                _ranges = ranges.ToArray(),
            };
        }

        /// <summary>Hash the source data and build the chunks. Any thread, once.</summary>
        public void Prepare()
        {
            var positions = _sourcePositions!;
            var indices = _sourceIndices!;
            var ranges = _ranges!;
            _sourcePositions = null;
            _sourceIndices = null;
            _ranges = null;

            // ── Hash + triangle list ──
            ulong hash = NavHash.Bytes(1, MemoryMarshal.AsBytes(positions.AsSpan()));
            int capacity = 0;
            foreach (var (_, start, count) in ranges)
                capacity += Math.Max(0, Math.Min(start + count, indices.Length) - start);

            var triangles = new int[capacity];
            int n = 0;
            foreach (var (baseVertex, start, count) in ranges)
            {
                int end = Math.Min(start + count, indices.Length);
                if (start < 0 || end - start < 3) continue;

                hash = NavHash.Mix(hash, (ulong)baseVertex);
                hash = NavHash.Bytes(hash, MemoryMarshal.AsBytes(indices.AsSpan(start, end - start)));

                for (int i = start; i + 2 < end; i += 3)
                {
                    long a = baseVertex + (long)indices[i], b = baseVertex + (long)indices[i + 1], c = baseVertex + (long)indices[i + 2];
                    if (a >= positions.Length || b >= positions.Length || c >= positions.Length) continue;
                    if (a == b || b == c || a == c) continue;

                    triangles[n++] = (int)a;
                    triangles[n++] = (int)b;
                    triangles[n++] = (int)c;
                }
            }
            Hash = hash;

            int triCount = n / 3;
            if (triCount == 0) return;

            // ── Bounds ──
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            for (int i = 0; i < n; i++)
            {
                min = Vector3.Min(min, positions[triangles[i]]);
                max = Vector3.Max(max, positions[triangles[i]]);
            }
            Center = (min + max) * 0.5f;
            Extents = (max - min) * 0.5f;

            // ── Spatial order (Morton code of the centroid), so consecutive triangles are neighbours ──
            int[]? order = null;
            if (triCount > ChunkTriangles)
            {
                var size = Vector3.Max(max - min, new Vector3(1e-6f));
                var scale = new Vector3(1023f) / size;
                var keys = new uint[triCount];
                order = new int[triCount];

                for (int t = 0; t < triCount; t++)
                {
                    var centroid = (positions[triangles[t * 3]] + positions[triangles[t * 3 + 1]] + positions[triangles[t * 3 + 2]]) / 3f;
                    var q = (centroid - min) * scale;
                    keys[t] = Spread((uint)q.X) | (Spread((uint)q.Y) << 1) | (Spread((uint)q.Z) << 2);
                    order[t] = t;
                }
                Array.Sort(keys, order);
            }

            // ── Chunks, each with its own compact vertex list ──
            int chunkCount = (triCount + ChunkTriangles - 1) / ChunkTriangles;
            var chunks = new Chunk[chunkCount];
            var outPositions = new List<Vector3>(Math.Min(positions.Length, n));
            var outIndices = new int[n];

            // Output index of a source vertex. An entry below the current chunk's start belongs to an
            // earlier chunk (or is unset), so the array never needs clearing between chunks.
            var remap = new int[positions.Length];
            Array.Fill(remap, -1);

            int o = 0;
            for (int c = 0; c < chunkCount; c++)
            {
                int firstTri = c * ChunkTriangles;
                int lastTri = Math.Min(triCount, firstTri + ChunkTriangles);
                int vertexStart = outPositions.Count;
                int indexStart = o;
                var cmin = new Vector3(float.MaxValue);
                var cmax = new Vector3(float.MinValue);

                for (int t = firstTri; t < lastTri; t++)
                {
                    int src = (order != null ? order[t] : t) * 3;
                    for (int k = 0; k < 3; k++)
                    {
                        int v = triangles[src + k];
                        int mapped = remap[v];
                        if (mapped < vertexStart)
                        {
                            mapped = outPositions.Count;
                            remap[v] = mapped;
                            outPositions.Add(positions[v]);
                            cmin = Vector3.Min(cmin, positions[v]);
                            cmax = Vector3.Max(cmax, positions[v]);
                        }
                        outIndices[o++] = mapped - vertexStart;
                    }
                }

                chunks[c] = new Chunk
                {
                    VertexStart = vertexStart,
                    VertexCount = outPositions.Count - vertexStart,
                    IndexStart = indexStart,
                    IndexCount = o - indexStart,
                    Center = (cmin + cmax) * 0.5f,
                    Extents = (cmax - cmin) * 0.5f,
                };
            }

            Positions = outPositions.ToArray();
            Indices = outIndices;
            Chunks = chunks;
        }

        // Spread the low 10 bits of x so that two zero bits follow each one
        private static uint Spread(uint x)
        {
            x &= 0x3ff;
            x = (x ^ (x << 16)) & 0xff0000ff;
            x = (x ^ (x << 8)) & 0x0300f00f;
            x = (x ^ (x << 4)) & 0x030c30c3;
            x = (x ^ (x << 2)) & 0x09249249;
            return x;
        }

        /// <summary>World-space box around a local box under a transform.</summary>
        public static void TransformBounds(in Matrix4x4 m, Vector3 center, Vector3 extents, out Vector3 worldCenter, out Vector3 worldExtents)
        {
            worldCenter = Vector3.Transform(center, m);
            worldExtents = new Vector3(
                MathF.Abs(m.M11) * extents.X + MathF.Abs(m.M21) * extents.Y + MathF.Abs(m.M31) * extents.Z,
                MathF.Abs(m.M12) * extents.X + MathF.Abs(m.M22) * extents.Y + MathF.Abs(m.M32) * extents.Z,
                MathF.Abs(m.M13) * extents.X + MathF.Abs(m.M23) * extents.Y + MathF.Abs(m.M33) * extents.Z);
        }
    }

    internal struct MeshInstance
    {
        public int Geometry;
        public Matrix4x4 World;

        /// <summary>The transform flips the winding, so triangle normals point the other way.</summary>
        public bool Mirrored;

        public ulong Hash;
        public Vector3 Min, Max;
    }

    /// <summary>
    /// Everything a bake reads from the scene, captured on the main thread so the bake itself can
    /// run on worker threads. Holds references to mesh and height arrays, never copies of them.
    /// </summary>
    internal sealed class NavMeshBakeInput
    {
        public readonly List<TerrainSource> Terrains = [];
        public readonly List<MeshGeometry> Geometries = [];
        public MeshInstance[] Instances = [];

        public static NavMeshBakeInput Collect()
        {
            var input = new NavMeshBakeInput();

            foreach (var renderer in ComponentCache<TerrainRenderer>.All)
            {
                var terrain = renderer.Terrain;
                if (terrain?.HeightField == null || renderer.Transform == null) continue;
                if (terrain.HeightField.GetLength(0) < 2 || terrain.HeightField.GetLength(1) < 2) continue;

                input.Terrains.Add(new TerrainSource(
                    terrain.HeightField, renderer.Transform.Position, terrain.TerrainSize, terrain.MaxHeight));
            }

            // Mesh -> index into Geometries, or -1 for a mesh with nothing to bake
            var geometryOf = new Dictionary<Mesh, int>();
            var instances = new List<MeshInstance>();

            foreach (var renderer in ComponentCache<MeshRenderer>.All)
            {
                var mesh = renderer.Mesh;
                var entity = renderer.Entity;
                if (mesh == null || entity == null || !renderer.Enabled) continue;

                // Placement ghosts are not part of the scene yet
                if (renderer.ReplacementMaterial != null) continue;

                // Only static geometry: no RigidBody, or a static one
                var body = entity.GetComponentInParents<RigidBody>();
                if (body != null && !body.IsStatic) continue;

                if (!geometryOf.TryGetValue(mesh, out int geometry))
                {
                    var prepared = MeshGeometry.FromMesh(mesh);
                    geometry = prepared != null ? input.Geometries.Count : -1;
                    if (prepared != null) input.Geometries.Add(prepared);
                    geometryOf[mesh] = geometry;
                }
                if (geometry < 0) continue;

                var world = renderer.Transform?.Matrix ?? Matrix4x4.Identity;
                instances.Add(new MeshInstance
                {
                    Geometry = geometry,
                    World = world,
                    Mirrored = world.GetDeterminant() < 0,
                });
            }

            input.Instances = instances.ToArray();
            return input;
        }

        /// <summary>World bounds and hash of every instance. Needs the geometries prepared.</summary>
        public void PrepareInstances()
        {
            var instances = Instances;
            for (int i = 0; i < instances.Length; i++)
            {
                ref var instance = ref instances[i];
                var geometry = Geometries[instance.Geometry];

                MeshGeometry.TransformBounds(instance.World, geometry.Center, geometry.Extents, out var center, out var extents);
                instance.Min = center - extents;
                instance.Max = center + extents;

                var world = instance.World;
                ulong hash = NavHash.Bytes(geometry.Hash, MemoryMarshal.AsBytes(new ReadOnlySpan<Matrix4x4>(in world)));
                instance.Hash = hash;
            }
        }
    }
}
