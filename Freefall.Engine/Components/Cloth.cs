using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Freefall.Animation;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Direct3D12;
using Category = System.ComponentModel.CategoryAttribute;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Components
{
    /// <summary>
    /// A skinned piece of clothing whose lower part moves as cloth, simulated on the GPU (cloth.hlsl): it swings with
    /// the body's motion, falls against the body's parts (the Animator's body without the cloth, BodyShape) and the
    /// wind takes it. What hangs free and how it moves is its <see cref="Settings"/>. Goes on the entity of the
    /// SkinnedMeshRenderer; the skinned vertex shader draws the simulated places instead of skinning (struct Instance
    /// ClothVertices). Simulated while the character is posed every frame and within <see cref="SimulatedWithin"/> of
    /// the camera; further off it is plain skinning.
    /// </summary>
    [Icon("icon_skinnedmesh.png")]
    public class Cloth : Component
    {
        [Category("Cloth")]
        [Description("How it moves and what of it hangs free: a Cloth Settings asset, shared by the pieces that move alike. Empty = the defaults (a tabard's).")]
        public ClothSettings Settings;

        /// <summary>Metres from the camera within which it is simulated.</summary>
        public const float SimulatedWithin = 30f;

        // The most particles a cloth may have (one thread group does them all)
        private const int MostParticles = 16384;

        /// <summary>Being simulated this frame.</summary>
        [Freefall.Reflection.DontSerialize]
        public bool Simulating { get; private set; }

        /// <summary>Particles its mesh's LOD 0 makes (vertices at one place are one).</summary>
        [Freefall.Reflection.DontSerialize]
        public int Particles => _mesh?.ParticleCount ?? 0;

        private SkinnedMeshRenderer _renderer;
        private Animator _animator;
        private ClothMesh _mesh;
        private GraphicsBuffer _maxAway, _state, _output;
        private readonly GraphicsBuffer[] _frames = new GraphicsBuffer[3];
        private (float Anchor, float Slope, float Most, float Tether) _maxAwayFor;
        private long _simulatedTick = -10;
        private bool _shown;

        private static ComputeShader s_compute;
        private static int s_kernel;
        private static long s_tick = -1;

        public override void Destroy()
        {
            Hide();
            var device = Engine.Device;
            device?.DeferDispose(_maxAway);
            device?.DeferDispose(_state);
            device?.DeferDispose(_output);
            for (int i = 0; i < _frames.Length; i++)
            {
                device?.DeferDispose(_frames[i]);
                _frames[i] = null;
            }
            _maxAway = _state = _output = null;
            _mesh = null;
        }

        // ---- Every frame (DeferredRenderer, once a tick, before anything is drawn) ----

        /// <summary>Every cloth near enough, a dispatch each: their places for this frame's draws and shadows.</summary>
        internal static void SimulateAll(ID3D12GraphicsCommandList cmd)
        {
            if (s_tick == Engine.TickCount) return;
            s_tick = Engine.TickCount;

            var all = ComponentCache<Cloth>.All;
            if (all.Count == 0) return;

            bool bound = false;
            for (int i = 0; i < all.Count; i++)
            {
                var cloth = all[i];
                if (cloth == null || cloth.IsDestroyed) continue;
                try
                {
                    if (!cloth.Ready())
                    {
                        cloth.Hide();
                        continue;
                    }
                    if (!bound)
                    {
                        var device = Engine.Device;
                        cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
                        cmd.SetComputeRootSignature(device.GlobalRootSignature);
                        bound = true;
                    }
                    cloth.Simulate(cmd);
                }
                catch (Exception e)
                {
                    CallbackErrors.Report(cloth, nameof(Simulate), e);
                    cloth.Hide();
                }
            }
        }

        // Near enough and posed every frame, its mesh made into cloth and its buffers there
        private bool Ready()
        {
            _renderer ??= Entity.GetComponent<SkinnedMeshRenderer>();
            var mesh = _renderer?.Mesh;
            if (_renderer == null || mesh == null || !Enabled || !_renderer.Enabled) return false;

            _animator ??= Entity.GetComponentInParents<Animator>();
            if (_animator != null && _animator.PoseEvery != 1) return false;

            var camera = Camera.Main;
            if (camera == null || Vector3.DistanceSquared(camera.Position, Transform.WorldPosition) > SimulatedWithin * SimulatedWithin)
                return false;

            if (_mesh == null || _mesh.Mesh != mesh)
            {
                Destroy();
                _mesh = ClothMesh.Of(mesh);
            }
            if (_mesh == null || _mesh.ParticleCount == 0 || _mesh.ParticleCount > MostParticles) return false;

            if (s_compute == null)
            {
                s_compute = new ComputeShader("cloth.hlsl");
                s_kernel = s_compute.FindKernel("CSCloth");
            }

            _state ??= GraphicsBuffer.CreateStructured<Vector4>(_mesh.ParticleCount * 6, uav: true);
            _output ??= GraphicsBuffer.CreateStructured<Vector4>(_mesh.VertexCount * 2, srv: true, uav: true);
            for (int i = 0; i < _frames.Length; i++)
                _frames[i] ??= GraphicsBuffer.CreateUpload<FrameData>(1, mapped: true);

            var settings = Settings ?? ClothSettings.Default;
            var want = (settings.Anchor, settings.Slope, settings.MostAway, settings.Tether);
            if (_maxAway == null || want != _maxAwayFor)
            {
                _maxAway ??= GraphicsBuffer.CreateUpload<Vector4>(_mesh.ParticleCount, mapped: true);
                _mesh.WriteLimits(_maxAway, settings);
                _maxAwayFor = want;
            }
            return true;
        }

        private unsafe void Simulate(ID3D12GraphicsCommandList cmd)
        {
            var settings = Settings ?? ClothSettings.Default;
            var world = _renderer.Transform.Matrix;
            Matrix4x4.Invert(world, out var toObject);

            // Started over where the bones put it: the first frame, after a pause
            bool reset = Engine.TickCount - _simulatedTick > 2;
            _simulatedTick = Engine.TickCount;

            int substeps = Math.Clamp(settings.Substeps, 1, 8);
            float frameSeconds = Math.Clamp(Time.SmoothDelta, 1f / 240f, 1f / 30f);

            var frame = _frames[Engine.FrameIndex % _frames.Length];
            var data = frame.WritePtr<FrameData>();
            *data = new FrameData
            {
                World = world,
                ToObject = toObject,
                Gravity = new Vector3(0f, -9.81f, 0f),
                Dt = frameSeconds / substeps,
                Wind = Wind.Local,
                Drag = settings.Drag,
                Damping = settings.Damping,
                Stiffness = settings.Stiffness,
                Bending = settings.Bending,
                Thickness = settings.Thickness,
                Substeps = (uint)substeps,
                Iterations = (uint)Math.Clamp(settings.Iterations, 1, 32),
                Reset = reset ? 1u : 0u,
                Backstop = settings.Backstop,
            };

            // The body without the clothes that are cloth, round cones as thick on average as its oval parts
            uint parts = 0;
            if (_animator != null && _animator.TryGetBareBodyFrames(out var body, out var frames))
            {
                for (int i = 0; i < body.Parts.Length && parts < FrameData.MostParts; i++)
                {
                    var part = body.Parts[i];
                    float thick = MathF.Sqrt(part.Depth);
                    var from = Vector3.Transform(part.From, frames[i]);
                    var to = Vector3.Transform(part.To, frames[i]);
                    int at = (int)parts * 4;
                    data->PartFrom[at] = from.X; data->PartFrom[at + 1] = from.Y; data->PartFrom[at + 2] = from.Z;
                    data->PartFrom[at + 3] = part.RadiusFrom * thick;
                    data->PartTo[at] = to.X; data->PartTo[at + 1] = to.Y; data->PartTo[at + 2] = to.Z;
                    data->PartTo[at + 3] = part.RadiusTo * thick;
                    parts++;
                }
            }
            data->PartCount = parts;

            // Named without their "Idx" (ComputeShader drops it from the cbuffer's names)
            var cs = s_compute;
            cs.SetPushConstant("Particles", _mesh.Particles.SrvIndex);
            cs.SetPushConstant("Links", _mesh.Links.SrvIndex);
            cs.SetPushConstant("Vertices", _mesh.Vertices.SrvIndex);
            cs.SetPushConstant("VertexTris", _mesh.VertexTris.SrvIndex);
            cs.SetPushConstant("MaxAway", _maxAway.SrvIndex);
            cs.SetPushConstant("StateUAV", _state.UavIndex);
            cs.SetPushConstant("OutputUAV", _output.UavIndex);
            cs.SetPushConstant("Frame", frame.SrvIndex);
            cs.SetPushConstant("Bones", _renderer.BoneBufferIdx);
            cs.SetPushConstant("ParticleCount", (uint)_mesh.ParticleCount);
            cs.SetPushConstant("VertexCount", (uint)_mesh.VertexCount);

            _state.Transition(cmd, ResourceStates.UnorderedAccess);
            _output.Transition(cmd, ResourceStates.UnorderedAccess);
            cs.Dispatch(s_kernel, cmd, 1);
            _output.Transition(cmd, ResourceStates.NonPixelShaderResource);

            // The vertex shaders read the places from now on (not saved with the renderer: a bindless index)
            _renderer.WriteInstance("ClothVertices", new Vector4(_output.SrvIndex, 0f, 0f, 0f));
            _renderer.WriteInstance("ClothFirst", new Vector4(_mesh.VertexFirst, 0f, 0f, 0f));
            _renderer.WriteInstance("ClothCount", new Vector4(_mesh.VertexCount, 0f, 0f, 0f));
            _shown = true;
            Simulating = true;
        }

        // Plain skinning again
        private void Hide()
        {
            Simulating = false;
            if (!_shown) return;
            _shown = false;
            if (_renderer != null && !_renderer.IsDestroyed) _renderer.WriteInstance("ClothVertices", Vector4.Zero);
        }

        [StructLayout(LayoutKind.Sequential)]
        private unsafe struct FrameData
        {
            public const int MostParts = 16;

            public Matrix4x4 World;
            public Matrix4x4 ToObject;
            public Vector3 Gravity;
            public float Dt;
            public Vector3 Wind;
            public float Drag;
            public float Damping;
            public float Stiffness;
            public float Bending;
            public float Thickness;
            public uint Substeps;
            public uint Iterations;
            public uint PartCount;
            public uint Reset;
            public float Backstop;
            public float Pad0, Pad1, Pad2;
            public fixed float PartFrom[MostParts * 4];
            public fixed float PartTo[MostParts * 4];
        }
    }

    /// <summary>
    /// A mesh made into cloth, once per mesh (cloth.hlsl): its LOD 0's vertices welded into particles (vertices at one
    /// place, seams in the UVs, are one), the links between them (the edges, and across two triangles to keep it from
    /// folding), and for every vertex its particle and the triangles round it (for its normal). Up on the GPU in
    /// upload buffers, written once.
    /// </summary>
    internal sealed class ClothMesh
    {
        public Mesh Mesh;
        public int ParticleCount, VertexFirst, VertexCount;
        public GraphicsBuffer Particles, Links, Vertices, VertexTris;

        // Each particle's place in the rig's rest pose, and the hips' height, for what hangs free and the tethers
        private Vector3[] _rest;
        private float _hipsY;

        private static readonly ConditionalWeakTable<Mesh, ClothMesh> s_meshes = new();

        public static ClothMesh Of(Mesh mesh) => s_meshes.GetValue(mesh, Build);

        [StructLayout(LayoutKind.Sequential)]
        private struct Particle
        {
            public Vector3 Rest;
            public uint First;
            public Vector4 BoneIds;
            public Vector4 Weights;
            public uint Count;
            public uint Pad0, Pad1, Pad2;
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

        private static ClothMesh Build(Mesh mesh)
        {
            var cloth = new ClothMesh { Mesh = mesh };
            var positions = mesh.Positions;
            var indices = mesh.CpuIndices;
            var weights = mesh.BoneWeights;
            if (positions == null || indices == null || weights == null) return cloth;

            // LOD 0's triangles, as vertex numbers
            var parts = new List<int>();
            if (mesh.LODs.Count > 0)
            {
                parts.AddRange(mesh.LODs[0].MeshPartIndices);
                if (mesh.NonLodPartIndices != null) parts.AddRange(mesh.NonLodPartIndices);
            }
            else
            {
                for (int i = 0; i < mesh.MeshParts.Count; i++) parts.Add(i);
            }

            var triangles = new List<(int A, int B, int C)>();
            int first = int.MaxValue, last = -1;
            foreach (int index in parts)
            {
                if ((uint)index >= (uint)mesh.MeshParts.Count) continue;
                var part = mesh.MeshParts[index];
                if (!part.Enabled) continue;
                int end = Math.Min(part.BaseIndex + part.NumIndices, indices.Length) - 2;
                for (int i = part.BaseIndex; i < end; i += 3)
                {
                    int a = (int)indices[i] + part.BaseVertex, b = (int)indices[i + 1] + part.BaseVertex, c = (int)indices[i + 2] + part.BaseVertex;
                    if ((uint)a >= (uint)positions.Length || (uint)b >= (uint)positions.Length || (uint)c >= (uint)positions.Length) continue;
                    triangles.Add((a, b, c));
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
                    bones.Add(weights[vertex]);
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

            // Which way the triangles are wound: the normal most of them make points away from the line through the
            // middle of the body (cloth hangs round it)
            double outward = 0;
            foreach (var t in welded)
            {
                var a = rest[t.A];
                var normal = Vector3.Cross(rest[t.B] - a, rest[t.C] - a);
                var centre = (a + rest[t.B] + rest[t.C]) / 3f;
                outward += Vector3.Dot(normal, new Vector3(centre.X, 0f, centre.Z));
            }
            uint flip = outward < 0 ? 1u : 0u;

            // Each vertex's triangles, as particles
            var trianglesOf = new List<int>[vertexCount];
            for (int i = 0; i < welded.Count; i++)
            {
                var t = welded[i];
                (trianglesOf[t.Va - first] ??= new List<int>(6)).Add(i);
                (trianglesOf[t.Vb - first] ??= new List<int>(6)).Add(i);
                (trianglesOf[t.Vc - first] ??= new List<int>(6)).Add(i);
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
                    particleData[i] = new Particle { Rest = rest[i], First = (uint)next, BoneIds = w.BoneIDs, Weights = w.Weights, Count = (uint)linksOf[i].Count };
                    foreach (var link in linksOf[i]) linkData[next++] = link;
                }

                var vertexData = cloth.Vertices.WritePtr<UInt4>();
                var triData = cloth.VertexTris.WritePtr<UInt4>();
                int nextTri = 0;
                for (int v = 0; v < vertexCount; v++)
                {
                    var list = trianglesOf[v];
                    if (particleOf[v] < 0 || list == null)
                    {
                        vertexData[v] = new UInt4 { X = uint.MaxValue };
                        continue;
                    }
                    vertexData[v] = new UInt4 { X = (uint)particleOf[v], Y = (uint)nextTri, Z = (uint)list.Count, W = flip };
                    foreach (int t in list)
                    {
                        var tri = welded[t];
                        triData[nextTri++] = new UInt4 { X = (uint)tri.A, Y = (uint)tri.B, Z = (uint)tri.C };
                    }
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
        /// Each particle's limits, by the settings: how far it may leave its bones' place (by the line and the slope),
        /// and its tether: the held particle nearest it at rest and how far from it it may get (-1: none).
        /// </summary>
        public unsafe void WriteLimits(GraphicsBuffer buffer, ClothSettings settings)
        {
            var data = buffer.WritePtr<Vector4>();
            float line = _hipsY + settings.Anchor;
            var most = new float[ParticleCount];
            var held = new List<int>();
            for (int i = 0; i < ParticleCount; i++)
            {
                most[i] = Math.Clamp((line - _rest[i].Y) * settings.Slope, 0f, settings.MostAway);
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
