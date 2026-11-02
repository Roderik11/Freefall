using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Freefall.Animation;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Graphics;
using Category = System.ComponentModel.CategoryAttribute;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Components
{
    /// <summary>
    /// A piece that moves as cloth, simulated on the GPU by <see cref="ClothSystem"/> (cloth.hlsl): clothing on a skinned
    /// character (it swings with the body's motion and falls against the body's parts: the Animator's body without the
    /// cloth, BodyShape), or a banner or a flag on a static mesh. The wind takes it by how square it meets it. What hangs
    /// free and how it moves is its <see cref="Settings"/>; <see cref="MaterialSlot"/> keeps it to one part of the mesh (a
    /// banner's cloth, not its bar).
    ///
    /// Goes on the entity of a SkinnedMeshRenderer, or of a MeshRenderer: a static mesh is drawn by a skinned twin the
    /// cloth makes (not saved), since only the skinned vertex shader reads the simulated places (struct Instance
    /// ClothVertices). Every LOD moves with it: LOD 0 is simulated, the others' vertices follow the particle nearest
    /// them. Simulated while it is in view and big enough on screen to be seen moving (and, on a character, while it is
    /// posed every frame); out of view it stands as it was last.
    /// </summary>
    [Icon("icon_skinnedmesh.png")]
    public class Cloth : Component
    {
        [Category("Cloth")]
        [Description("How it moves and what of it hangs free: a Cloth Settings asset, shared by the pieces that move alike. Empty = the defaults (a tabard's).")]
        public ClothSettings Settings;

        [Description("Only the parts of the mesh in this material slot are cloth; the rest (a banner's bar and irons) stays as it is. -1: all of it.")]
        public int MaterialSlot = -1;

        /// <summary>Simulated this frame or the last.</summary>
        [Freefall.Reflection.DontSerialize]
        public bool Simulating => SimulatedTick >= Engine.TickCount - 1;

        /// <summary>Particles its mesh's LOD 0 makes (vertices at one place are one).</summary>
        [Freefall.Reflection.DontSerialize]
        public int Particles => ClothMesh?.ParticleCount ?? 0;

        // ClothSystem's: its row in the system, the mesh made into cloth it simulates, the tick it was simulated last
        internal int SystemIndex = -1;
        internal ClothMesh? ClothMesh;
        internal long SimulatedTick = long.MinValue;

        private SkinnedMeshRenderer _renderer;
        private MeshRenderer _static;
        private Entity _twin;
        private Animator _animator;

        // A static mesh's materials made over for the skinned effect, once each
        private static readonly ConditionalWeakTable<Material, Material> s_twins = new();

        /// <summary>The skinned renderer it moves (FindRenderer first).</summary>
        internal SkinnedMeshRenderer Renderer => _renderer;

        /// <summary>The Animator that poses it, on a character; null on a static mesh.</summary>
        internal Animator Animator => _animator;

        public override void Destroy()
        {
            if (_twin != null && !_twin.Transform.IsDestroyed) _twin.Destroy();
            _twin = null;
            _renderer = null;
            if (_static != null && !_static.IsDestroyed) _static.DrawnElsewhere = false;
            _static = null;
            ClothMesh = null;
        }

        /// <summary>
        /// The skinned renderer the cloth moves: its own entity's, or for a static mesh a twin it makes under its entity
        /// (DontSave) with the mesh and its materials made over for the skinned effect, the static one drawing nothing
        /// meanwhile (PersistentRenderer.DrawnElsewhere, not saved).
        /// </summary>
        internal bool FindRenderer()
        {
            if (_renderer != null && !_renderer.IsDestroyed) return true;

            _renderer = Entity.GetComponent<SkinnedMeshRenderer>();
            if (_renderer != null)
            {
                _animator ??= Entity.GetComponentInParents<Animator>();
                return true;
            }

            _static ??= Entity.GetComponent<MeshRenderer>();
            var mesh = _static?.Mesh;
            if (mesh == null) return false;

            int slots = 0;
            foreach (var part in mesh.MeshParts) slots = Math.Max(slots, part.MaterialSlot + 1);
            var materials = new List<Material>(slots);
            for (int slot = 0; slot < slots; slot++) materials.Add(Twin(_static.GetMaterial(slot)));

            _twin = new Entity(Entity.Name + " (cloth)");
            _twin.Flags |= EntityFlags.DontSave;
            _twin.Transform.Parent = Transform;
            _renderer = _twin.AddComponent(new SkinnedMeshRenderer { Mesh = mesh, Materials = materials });
            _static.DrawnElsewhere = true;
            return true;
        }

        /// <summary>A material made over for the skinned effect: its textures and values (Tint, emissive, detail).</summary>
        private static Material Twin(Material material)
        {
            var skinned = InternalAssets.SkinnedEffect;
            if (material == null || skinned == null || material.Effect == skinned) return material;
            return s_twins.GetValue(material, m =>
            {
                var twin = new Material(skinned);
                foreach (var (name, texture) in m.Textures)
                    if (texture != null) twin.SetTexture(name, texture);
                foreach (var property in m.MaterialProperties)
                    if (twin.GetProperty(property.Name) is { } own && own.ValueType == property.ValueType) own.CopyFrom(property);
                return twin;
            });
        }
    }

    /// <summary>
    /// A mesh (or one material slot of it) made into cloth, once (cloth.hlsl): its LOD 0's vertices welded into
    /// particles (vertices at one place, seams in the UVs, are one), the links between them (the edges, and across two
    /// triangles to keep it from folding), for every vertex of LOD 0 its particle and the triangles round it (its
    /// normal), and for every vertex of the other LODs the particle nearest it, which it follows. Up on the GPU in
    /// upload buffers, written once, and made again when the mesh's geometry is (a reimport replaces it in the same Mesh:
    /// the old cloth's particles would be drawn over the new mesh's vertices).
    /// </summary>
    internal sealed class ClothMesh
    {
        public Mesh Mesh;
        public int Slot;
        public int Version;                 // the mesh's GeometryVersion it was made from
        public int ParticleCount, VertexFirst, VertexCount;
        public GraphicsBuffer Particles, Links, Vertices, VertexTris;

        // Each particle's place at rest (the mesh's space), and the hips' height, for what hangs free and the tethers
        private Vector3[] _rest;
        private float _hipsY;

        // Each particle's limits by the settings that decide them, shared by every cloth of this mesh with those values
        private readonly Dictionary<LimitsKey, GraphicsBuffer> _limits = new();

        private static readonly ConditionalWeakTable<Mesh, Dictionary<int, ClothMesh>> s_meshes = new();

        public static ClothMesh Of(Mesh mesh, int slot)
        {
            var bySlot = s_meshes.GetOrCreateValue(mesh);
            lock (bySlot)
            {
                if (bySlot.TryGetValue(slot, out var cloth) && cloth.Version == mesh.GeometryVersion) return cloth;
                cloth?.Release();
                bySlot[slot] = cloth = Build(mesh, slot);
                return cloth;
            }
        }

        // Its buffers, once the frames that read them are done
        private void Release()
        {
            var device = Engine.Device;
            device?.DeferDispose(Particles);
            device?.DeferDispose(Links);
            device?.DeferDispose(Vertices);
            device?.DeferDispose(VertexTris);
            foreach (var buffer in _limits.Values) device?.DeferDispose(buffer);
            _limits.Clear();
        }

        /// <summary>What decides a particle's limits (WriteLimits), from a Cloth Settings asset.</summary>
        public readonly record struct LimitsKey(ClothHold Held, float Anchor, Vector3 HeldToward, float HeldWithin, float Slope,
                                                float MostAway, float Tether)
        {
            public static LimitsKey Of(ClothSettings s) => new(s.Held, s.Anchor, s.HeldToward, s.HeldWithin, s.Slope, s.MostAway, s.Tether);
        }

        /// <summary>Its particles' limits for these settings: made the first time they are asked for.</summary>
        public GraphicsBuffer LimitsFor(LimitsKey key, ClothSettings settings)
        {
            if (_limits.TryGetValue(key, out var buffer)) return buffer;
            buffer = GraphicsBuffer.CreateUpload<Vector4>(Math.Max(1, ParticleCount), mapped: true);
            WriteLimits(buffer, settings);
            _limits[key] = buffer;
            return buffer;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct Particle
        {
            public Vector3 Rest;
            public uint First;
            public Vector4 BoneIds;
            public Vector4 Weights;
            public uint Count;
            public uint Vertex;   // a vertex of LOD 0 it is (relative to the range): where its normal is read from
            public uint Pad0, Pad1;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct Link
        {
            public uint Other;
            public float Rest;
            public uint Bend;
            public uint Pad;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct UInt4
        {
            public uint X, Y, Z, W;
        }

        // Vertices.W: a vertex of another LOD, which follows its particle
        private const uint Follows = 2u;

        private static ClothMesh Build(Mesh mesh, int slot)
        {
            var cloth = new ClothMesh { Mesh = mesh, Slot = slot, Version = mesh.GeometryVersion };
            var positions = mesh.Positions;
            var indices = mesh.CpuIndices;
            var weights = mesh.BoneWeights;
            if (positions == null || indices == null) return cloth;

            // LOD 0's parts are simulated, the other LODs' follow; of the one material slot only, when one is named
            var lod0 = new HashSet<int>();
            if (mesh.LODs.Count > 0)
            {
                foreach (int index in mesh.LODs[0].MeshPartIndices) lod0.Add(index);
                if (mesh.NonLodPartIndices != null)
                    foreach (int index in mesh.NonLodPartIndices) lod0.Add(index);
            }
            else
            {
                for (int i = 0; i < mesh.MeshParts.Count; i++) lod0.Add(i);
            }

            var triangles = new List<(int A, int B, int C)>();
            var others = new List<int>();
            int first = int.MaxValue, last = -1;
            for (int index = 0; index < mesh.MeshParts.Count; index++)
            {
                var part = mesh.MeshParts[index];
                if (!part.Enabled || (slot >= 0 && part.MaterialSlot != slot)) continue;
                bool simulated = lod0.Contains(index);
                int end = Math.Min(part.BaseIndex + part.NumIndices, indices.Length) - 2;
                for (int i = part.BaseIndex; i < end; i += 3)
                {
                    int a = (int)indices[i] + part.BaseVertex, b = (int)indices[i + 1] + part.BaseVertex, c = (int)indices[i + 2] + part.BaseVertex;
                    if ((uint)a >= (uint)positions.Length || (uint)b >= (uint)positions.Length || (uint)c >= (uint)positions.Length) continue;
                    if (simulated) triangles.Add((a, b, c));
                    else
                    {
                        others.Add(a);
                        others.Add(b);
                        others.Add(c);
                    }
                    first = Math.Min(first, Math.Min(a, Math.Min(b, c)));
                    last = Math.Max(last, Math.Max(a, Math.Max(b, c)));
                }
            }
            if (triangles.Count == 0) return cloth;

            // Vertices at one place (to half a millimetre) are one particle
            int vertexCount = last - first + 1;
            var particleOf = new int[vertexCount];
            Array.Fill(particleOf, -1);
            var byPlace = new Dictionary<(int, int, int), int>();
            var rest = new List<Vector3>();
            var bones = new List<BoneWeight>();
            var vertexOf = new List<int>();
            int Weld(int vertex)
            {
                int local = vertex - first;
                if (particleOf[local] >= 0) return particleOf[local];
                var p = positions[vertex];
                var key = ((int)MathF.Round(p.X * 2000f), (int)MathF.Round(p.Y * 2000f), (int)MathF.Round(p.Z * 2000f));
                if (!byPlace.TryGetValue(key, out int particle))
                {
                    particle = rest.Count;
                    byPlace[key] = particle;
                    rest.Add(p);
                    bones.Add(weights != null && vertex < weights.Length ? weights[vertex] : default);
                    vertexOf.Add(local);
                }
                particleOf[local] = particle;
                return particle;
            }

            var welded = new List<(int A, int B, int C, int Va, int Vb, int Vc)>(triangles.Count);
            foreach (var (a, b, c) in triangles)
            {
                int pa = Weld(a), pb = Weld(b), pc = Weld(c);
                if (pa == pb || pb == pc || pa == pc) continue;
                welded.Add((pa, pb, pc, a, b, c));
            }
            int count = rest.Count;

            // The edges, and across each edge two triangles share, their far corners
            var edges = new Dictionary<(int, int), List<int>>();
            void Edge(int a, int b, int opposite)
            {
                var key = a < b ? (a, b) : (b, a);
                if (!edges.TryGetValue(key, out var far)) edges[key] = far = new List<int>(2);
                far.Add(opposite);
            }
            foreach (var t in welded)
            {
                Edge(t.A, t.B, t.C);
                Edge(t.B, t.C, t.A);
                Edge(t.C, t.A, t.B);
            }

            var linksOf = new List<Link>[count];
            for (int i = 0; i < count; i++) linksOf[i] = new List<Link>(8);
            var linked = new HashSet<(int, int)>();
            void Join(int a, int b, bool bend)
            {
                var key = a < b ? (a, b) : (b, a);
                if (a == b || !linked.Add(key)) return;
                float length = Vector3.Distance(rest[a], rest[b]);
                linksOf[a].Add(new Link { Other = (uint)b, Rest = length, Bend = bend ? 1u : 0u });
                linksOf[b].Add(new Link { Other = (uint)a, Rest = length, Bend = bend ? 1u : 0u });
            }
            foreach (var ((a, b), _) in edges) Join(a, b, false);
            foreach (var (_, far) in edges)
                if (far.Count == 2) Join(far[0], far[1], true);

            // Each LOD 0 vertex's triangles, as particles
            var trianglesOf = new List<int>[vertexCount];
            for (int i = 0; i < welded.Count; i++)
            {
                var t = welded[i];
                (trianglesOf[t.Va - first] ??= new List<int>(6)).Add(i);
                (trianglesOf[t.Vb - first] ??= new List<int>(6)).Add(i);
                (trianglesOf[t.Vc - first] ??= new List<int>(6)).Add(i);
            }

            // The other LODs' vertices: the particle nearest each, by a grid of 5 cm cells over the particles
            var follows = new int[vertexCount];
            Array.Fill(follows, -1);
            if (others.Count > 0)
            {
                const float Cell = 0.05f;
                var grid = new Dictionary<(int, int, int), List<int>>();
                (int, int, int) CellOf(Vector3 p) => ((int)MathF.Floor(p.X / Cell), (int)MathF.Floor(p.Y / Cell), (int)MathF.Floor(p.Z / Cell));
                for (int i = 0; i < count; i++)
                {
                    var key = CellOf(rest[i]);
                    if (!grid.TryGetValue(key, out var list)) grid[key] = list = new List<int>(4);
                    list.Add(i);
                }
                foreach (int vertex in others)
                {
                    int local = vertex - first;
                    if (particleOf[local] >= 0 || follows[local] >= 0) continue;
                    var p = positions[vertex];
                    var (cx, cy, cz) = CellOf(p);
                    int best = -1;
                    float nearest = float.MaxValue;
                    for (int reach = 1; reach <= 8 && best < 0; reach *= 2)
                        for (int x = cx - reach; x <= cx + reach; x++)
                        for (int y = cy - reach; y <= cy + reach; y++)
                        for (int z = cz - reach; z <= cz + reach; z++)
                        {
                            if (!grid.TryGetValue((x, y, z), out var list)) continue;
                            foreach (int i in list)
                            {
                                float d = Vector3.DistanceSquared(p, rest[i]);
                                if (d < nearest) { nearest = d; best = i; }
                            }
                        }
                    if (best < 0)
                        for (int i = 0; i < count; i++)
                        {
                            float d = Vector3.DistanceSquared(p, rest[i]);
                            if (d < nearest) { nearest = d; best = i; }
                        }
                    follows[local] = best;
                }
            }

            // Up onto the GPU
            int linkCount = 0;
            foreach (var list in linksOf) linkCount += list.Count;
            int vertexTriCount = 0;
            foreach (var list in trianglesOf) vertexTriCount += list?.Count ?? 0;

            cloth.Particles = GraphicsBuffer.CreateUpload<Particle>(count, mapped: true);
            cloth.Links = GraphicsBuffer.CreateUpload<Link>(Math.Max(1, linkCount), mapped: true);
            cloth.Vertices = GraphicsBuffer.CreateUpload<UInt4>(vertexCount, mapped: true);
            cloth.VertexTris = GraphicsBuffer.CreateUpload<UInt4>(Math.Max(1, vertexTriCount), mapped: true);
            unsafe
            {
                var particleData = cloth.Particles.WritePtr<Particle>();
                var linkData = cloth.Links.WritePtr<Link>();
                int next = 0;
                for (int i = 0; i < count; i++)
                {
                    var w = bones[i];
                    particleData[i] = new Particle
                    {
                        Rest = rest[i], First = (uint)next, BoneIds = w.BoneIDs, Weights = w.Weights,
                        Count = (uint)linksOf[i].Count, Vertex = (uint)vertexOf[i],
                    };
                    foreach (var link in linksOf[i]) linkData[next++] = link;
                }

                var vertexData = cloth.Vertices.WritePtr<UInt4>();
                var triData = cloth.VertexTris.WritePtr<UInt4>();
                int nextTri = 0;
                for (int v = 0; v < vertexCount; v++)
                {
                    var list = trianglesOf[v];
                    if (particleOf[v] >= 0 && list != null)
                    {
                        vertexData[v] = new UInt4 { X = (uint)particleOf[v], Y = (uint)nextTri, Z = (uint)list.Count };
                        foreach (int t in list)
                        {
                            var tri = welded[t];
                            triData[nextTri++] = new UInt4 { X = (uint)tri.A, Y = (uint)tri.B, Z = (uint)tri.C };
                        }
                    }
                    else if (follows[v] >= 0) vertexData[v] = new UInt4 { X = (uint)follows[v], W = Follows };
                    else vertexData[v] = new UInt4 { X = uint.MaxValue };
                }
            }

            cloth._rest = rest.ToArray();
            cloth._hipsY = HipsHeight(mesh.Skeleton);
            cloth.ParticleCount = count;
            cloth.VertexFirst = first;
            cloth.VertexCount = vertexCount;
            return cloth;
        }

        // The hips' height in the rig's rest pose (the mesh's space: only X and Z of a flipped mesh are turned round)
        private static float HipsHeight(Skeleton skeleton)
        {
            var humanoid = skeleton != null ? HumanoidTemplate.Detect(skeleton) : null;
            if (humanoid == null || !humanoid.TryGetValue(HumanoidBone.Hips, out int hips)) return 1f;
            return Matrix4x4.Invert(skeleton.Bones[hips].OffsetMatrix, out var bind) ? bind.Translation.Y : 1f;
        }

        /// <summary>
        /// Each particle's limits, by the settings: how far it may leave its held place (by its distance from the hips'
        /// line, or from the held edge), and its tether: the held particle nearest it at rest and how far from it it may
        /// get (-1: none).
        /// </summary>
        private unsafe void WriteLimits(GraphicsBuffer buffer, ClothSettings settings)
        {
            var data = buffer.WritePtr<Vector4>();
            var most = new float[ParticleCount];

            // How far each particle is from what holds it
            var toward = settings.HeldToward.LengthSquared() > 1e-8f ? Vector3.Normalize(settings.HeldToward) : Vector3.UnitY;
            float edge = float.MinValue;
            if (settings.Held == ClothHold.Edge)
                for (int i = 0; i < ParticleCount; i++) edge = MathF.Max(edge, Vector3.Dot(_rest[i], toward));

            var held = new List<int>();
            for (int i = 0; i < ParticleCount; i++)
            {
                float away = settings.Held == ClothHold.Edge
                    ? edge - settings.HeldWithin - Vector3.Dot(_rest[i], toward)
                    : _hipsY + settings.Anchor - _rest[i].Y;
                most[i] = Math.Clamp(away * settings.Slope, 0f, settings.MostAway);
                if (most[i] <= 0f) held.Add(i);
            }

            for (int i = 0; i < ParticleCount; i++)
            {
                int tether = -1;
                float length = 0f;
                if (most[i] > 0f && settings.Tether > 0f)
                {
                    float nearest = float.MaxValue;
                    foreach (int h in held)
                    {
                        float d = Vector3.DistanceSquared(_rest[i], _rest[h]);
                        if (d < nearest) { nearest = d; tether = h; }
                    }
                    length = tether >= 0 ? MathF.Sqrt(nearest) * settings.Tether : 0f;
                }
                data[i] = new Vector4(most[i], tether, length, 0f);
            }
        }
    }
}
