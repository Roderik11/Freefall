using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Vortice.Direct3D12;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>
    /// Simulates every <see cref="Cloth"/> on the GPU in one dispatch a frame (cloth.hlsl), from two shared pools.
    ///
    /// The state pool holds every cloth's particles (six ranges of its particle count: places, the place a step ago, its
    /// held places), the output pool every cloth's vertices this frame (place and normal), which the skinned vertex shader
    /// reads in place of skinning (struct Instance ClothVertices, ClothBase). A cloth gets its ranges the first time it is
    /// simulated and keeps them while it might be seen again; a pool grows by doubling and keeps what it held.
    ///
    /// Per frame: this system (UpdateGroup) picks the cloths to simulate: in view (its bounds, widened by how far it may
    /// leave them, against the main camera's frustum) and big enough on screen to be seen moving, with fewer substeps and
    /// passes the smaller it shows, the biggest first within a budget. Out of view a cloth stands as it was last, and
    /// carries on from there when it comes back (a character's starts again where its bones hold it). Too small for a
    /// while, it gives its ranges back and is drawn as the mesh is. Then, once a tick before anything draws
    /// (DeferredRenderer), <see cref="SimulateAll"/> writes a row per picked cloth (its transform, bones, body parts,
    /// settings, buffers and ranges), uploads the rows and dispatches one thread group per cloth.
    /// See .agent/knowledge/engine_internals/bodies.md "Cloth".
    /// </summary>
    [UpdateInEditor]
    public sealed class ClothSystem : EntitySystem
    {
        // Radius of its bounds on screen, in pixels: below the first it is not simulated, from the second on it gets all
        // its settings' substeps and passes, from the third half of them, below that one substep and a quarter
        private const float SimulatedFrom = 1.5f, FullFrom = 48f, HalfFrom = 12f;

        // Seconds unsimulated while too small to see before its ranges go back to the pools
        private const float ReleaseAfter = 5f;

        // The most work a frame (particles × substeps × passes): past it the smallest on screen wait
        private const long MostWork = 6_000_000;

        // The most particles one cloth may have
        private const int MostParticles = 16384;

        // The most body parts a cloth falls against (cloth.hlsl MOST_PARTS)
        private const int MostParts = 16;

        private const int FrameCount = 3;
        private const int PoolStart = 1 << 16;      // float4s
        private const int RangeGrain = 64;          // float4s

        /// <summary>One cloth's row of the frame's table. Must match struct ClothRow in cloth.hlsl (256 bytes).</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct ClothRow
        {
            public Matrix4x4 World;
            public Matrix4x4 ToObject;
            public Vector3 Wind;        public float Dt;
            public float Drag;          public float Damping;       public float Stiffness;     public float Bending;
            public float Thickness;     public float Backstop;      public float Flutter;       public float Streams;
            public uint Substeps;       public uint Iterations;     public uint Reset;          public uint PartFirst;
            public uint PartCount;      public uint ParticleCount;  public uint VertexCount;    public uint VertexFirst;
            public uint StateBase;      public uint OutputBase;     public uint Particles;      public uint Links;
            public uint Vertices;       public uint VertexTris;     public uint Limits;         public uint Bones;
            public uint MeshPositions;  public uint MeshNormals;    public uint MeshWeights;    public uint Pad;
        }

        /// <summary>Everything the system keeps per cloth.</summary>
        private sealed class Entry
        {
            public Cloth Cloth = null!;
            public ClothMesh? Mesh;
            public GraphicsBuffer? Limits;
            public ClothMesh.LimitsKey LimitsKey;

            public int State = -1, StateSize;       // its ranges of the pools (float4s); -1 = none
            public int Output = -1, OutputSize;
            public bool Fresh = true;               // its state holds nothing of its own yet: start where it is held

            public long SimulatedTick = long.MinValue;
            public float SimulatedAt;               // Time.TotalTime
            public Matrix4x4 LastWorld;

            // This frame's pick
            public int Substeps, Iterations;
            public float Size;                      // (radius on screen)² / its distance², for the order within the budget
            public long Work;

            // The renderer whose instance data names its output, and what it was told
            public PersistentRenderer? Shown;
            public uint ShownPool, ShownBase, ShownFirst, ShownCount;
        }

        private static ClothSystem? s_current;

        private readonly List<Entry> _entries = [];
        private readonly List<Entry> _picked = [];
        private static readonly Comparison<Entry> BySizeDescending = (a, b) => b.Size.CompareTo(a.Size);

        // Pools, and their free runs sorted by start
        private GraphicsBuffer? _state, _output;
        private readonly List<(int Start, int Count)> _stateFree = [], _outputFree = [];
        private int _stateCapacity, _outputCapacity;

        // The frame's table and body parts, a buffer per frame in flight
        private readonly GraphicsBuffer?[] _rows = new GraphicsBuffer?[FrameCount];
        private readonly GraphicsBuffer?[] _parts = new GraphicsBuffer?[FrameCount];
        private readonly int[] _rowCapacity = new int[FrameCount], _partCapacity = new int[FrameCount];

        private ComputeShader? _compute;
        private int _kernel;
        private long _tick = -1;

        private static readonly ProfilerMarker _simulateMarker = ProfilerMarker.Get("Cloth");

        /// <summary>Cloths simulated in the last frame, and their particles.</summary>
        public int SimulatedCount { get; private set; }
        public int SimulatedParticles { get; private set; }

        // ────────────── Registration ──────────────

        protected internal override void Initialize()
        {
            if (Unsafe.SizeOf<ClothRow>() != 256)
                throw new InvalidOperationException("ClothRow must match struct ClothRow in cloth.hlsl (256 bytes)");

            s_current = this;
            ComponentCache<Cloth>.Added += OnAdded;
            ComponentCache<Cloth>.Removed += OnRemoved;
            foreach (var cloth in ComponentCache<Cloth>.All)
                OnAdded(cloth);
        }

        protected internal override void Destroy()
        {
            ComponentCache<Cloth>.Added -= OnAdded;
            ComponentCache<Cloth>.Removed -= OnRemoved;
            if (s_current == this) s_current = null;

            var device = Engine.Device;
            device?.DeferDispose(_state);
            device?.DeferDispose(_output);
            for (int i = 0; i < FrameCount; i++)
            {
                device?.DeferDispose(_rows[i]);
                device?.DeferDispose(_parts[i]);
            }
        }

        private void OnAdded(Cloth cloth)
        {
            if (cloth.SystemIndex >= 0) return;
            cloth.SystemIndex = _entries.Count;
            _entries.Add(new Entry { Cloth = cloth });
        }

        private void OnRemoved(Cloth cloth)
        {
            int index = cloth.SystemIndex;
            if (index >= 0 && index < _entries.Count && _entries[index].Cloth == cloth)
                RemoveEntry(index);
        }

        private void RemoveEntry(int index)
        {
            var entry = _entries[index];
            Release(entry);
            entry.Cloth.SystemIndex = -1;
            entry.Cloth.SimulatedTick = long.MinValue;

            int last = _entries.Count - 1;
            if (index != last)
            {
                var moved = _entries[last];
                _entries[index] = moved;
                moved.Cloth.SystemIndex = index;
            }
            _entries.RemoveAt(last);
        }

        // Drawn as the mesh is again, its ranges back to the pools
        private void Release(Entry entry)
        {
            Hide(entry);
            Free(_stateFree, entry.State, entry.StateSize);
            Free(_outputFree, entry.Output, entry.OutputSize);
            entry.State = entry.Output = -1;
            entry.StateSize = entry.OutputSize = 0;
            entry.Fresh = true;
            entry.SimulatedTick = long.MinValue;
        }

        private static void Hide(Entry entry)
        {
            var shown = entry.Shown;
            entry.Shown = null;
            if (shown == null || shown.IsDestroyed) return;
            shown.ForgetInstance("ClothVertices");
            shown.ForgetInstance("ClothBase");
            shown.ForgetInstance("ClothFirst");
            shown.ForgetInstance("ClothCount");
        }

        // ────────────── Pick (UpdateGroup) ──────────────

        public override void Update()
        {
            _picked.Clear();

            var camera = Camera.Main;
            if (camera == null) return;
            var frustum = new Frustum(camera.ViewProjection);
            float pixelsPerSlope = camera.Projection.M22 * MathF.Max(camera.Target?.Height ?? 1080, 1) * 0.5f;
            var eye = camera.Position;
            float now = Time.TotalTime;
            long work = 0;

            for (int i = _entries.Count - 1; i >= 0; i--)
            {
                var entry = _entries[i];
                var cloth = entry.Cloth;

                // Destroyed this frame (Removed arrives at the next wake-up)
                if (cloth.IsDestroyed)
                {
                    RemoveEntry(i);
                    continue;
                }

                if (!Usable(entry))
                {
                    if (entry.Shown != null || entry.State >= 0) Release(entry);
                    continue;
                }

                // How big it shows: the radius of its bounds, widened by how far it may leave them, over its distance
                var settings = cloth.Settings ?? ClothSettings.Default;
                var bounds = entry.Cloth.Renderer.BoundingSphere;
                float radius = bounds.Radius + MathF.Max(0f, settings.MostAway);
                float distanceSq = MathF.Max(Vector3.DistanceSquared(eye, bounds.Center), 1e-4f);
                float onScreen = radius * radius * pixelsPerSlope * pixelsPerSlope;     // (pixels)² × distance²
                if (onScreen < SimulatedFrom * SimulatedFrom * distanceSq)
                {
                    if (entry.State >= 0 && now - entry.SimulatedAt > ReleaseAfter) Release(entry);
                    continue;
                }
                if (frustum.Intersects(new BoundingSphere(bounds.Center, radius)) == ContainmentType.Disjoint)
                    continue;

                int substeps = Math.Clamp(settings.Substeps, 1, 8), iterations = Math.Clamp(settings.Iterations, 1, 32);
                if (onScreen < HalfFrom * HalfFrom * distanceSq)
                {
                    substeps = 1;
                    iterations = Math.Max(2, iterations / 4);
                }
                else if (onScreen < FullFrom * FullFrom * distanceSq)
                {
                    substeps = Math.Max(1, substeps / 2);
                    iterations = Math.Max(2, iterations / 2);
                }
                entry.Substeps = substeps;
                entry.Iterations = iterations;
                entry.Size = onScreen / distanceSq;
                entry.Work = (long)entry.Mesh!.ParticleCount * substeps * ((iterations + 1) & ~1);
                work += entry.Work;
                _picked.Add(entry);
            }

            // Over the budget: the biggest on screen first
            if (work > MostWork)
            {
                _picked.Sort(BySizeDescending);
                long sum = 0;
                int keep = 0;
                while (keep < _picked.Count && sum + _picked[keep].Work <= MostWork) sum += _picked[keep++].Work;
                _picked.RemoveRange(keep, _picked.Count - keep);
            }
        }

        // Enabled, its renderer there (a static mesh's twin made now), on a character posed every frame, its mesh made
        // into cloth (again when the mesh was reimported), and its limits for its settings
        private bool Usable(Entry entry)
        {
            var cloth = entry.Cloth;
            if (!cloth.Enabled || !cloth.FindRenderer()) return false;
            var renderer = cloth.Renderer;
            var mesh = renderer.Mesh;
            if (mesh == null || !renderer.Enabled) return false;
            if (cloth.Animator != null && cloth.Animator.PoseEvery != 1) return false;

            var made = entry.Mesh;
            if (made == null || made.Mesh != mesh || made.Slot != cloth.MaterialSlot || made.Version != mesh.GeometryVersion)
            {
                var again = ClothMesh.Of(mesh, cloth.MaterialSlot);
                if (again != made)
                {
                    // Other particles and vertices: other ranges, and drawn as the mesh is until simulated
                    Release(entry);
                    entry.Mesh = again;
                    entry.Limits = null;
                    cloth.ClothMesh = again;
                }
            }
            made = entry.Mesh;
            if (made == null || made.ParticleCount == 0 || made.ParticleCount > MostParticles) return false;

            var settings = cloth.Settings ?? ClothSettings.Default;
            var key = ClothMesh.LimitsKey.Of(settings);
            if (entry.Limits == null || key != entry.LimitsKey)
            {
                entry.Limits = made.LimitsFor(key, settings);
                entry.LimitsKey = key;
            }
            return true;
        }

        // ────────────── Simulate (DeferredRenderer, once a tick) ──────────────

        /// <summary>
        /// The cloths picked this frame, one thread group each: their places for this frame's draws and shadows. Before
        /// InstanceData is uploaded: it tells the skinned vertex shader where each cloth's vertices are.
        /// </summary>
        internal static void SimulateAll(ID3D12GraphicsCommandList cmd) => s_current?.Simulate(cmd);

        private unsafe void Simulate(ID3D12GraphicsCommandList cmd)
        {
            if (_tick == Engine.TickCount) return;
            _tick = Engine.TickCount;

            using var _ = _simulateMarker.Auto(_picked.Count);
            SimulatedCount = 0;
            SimulatedParticles = 0;

            // Ranges for the newly picked; the pools grow first if they must (keeping what they hold)
            int stateWanted = 0, outputWanted = 0;
            for (int i = _picked.Count - 1; i >= 0; i--)
            {
                var entry = _picked[i];
                if (entry.Cloth.IsDestroyed || entry.Mesh == null || entry.Cloth.Renderer == null) { _picked.RemoveAt(i); continue; }
                if (entry.State < 0)
                {
                    stateWanted += Round(entry.Mesh.ParticleCount * 6);
                    outputWanted += Round(entry.Mesh.VertexCount * 2);
                }
            }
            EnsurePools(cmd, stateWanted, outputWanted);

            int count = _picked.Count;
            if (count > 0)
            {
                foreach (var entry in _picked)
                {
                    if (entry.State >= 0) continue;
                    entry.StateSize = Round(entry.Mesh!.ParticleCount * 6);
                    entry.OutputSize = Round(entry.Mesh.VertexCount * 2);
                    entry.State = Take(_stateFree, entry.StateSize);
                    entry.Output = Take(_outputFree, entry.OutputSize);
                    entry.Fresh = true;
                }

                int frame = Engine.FrameIndex % FrameCount;
                var rows = EnsureUpload<ClothRow>(_rows, _rowCapacity, frame, count);
                var parts = EnsureUpload<Vector4>(_parts, _partCapacity, frame, count * MostParts * 2);
                var rowData = rows.WritePtr<ClothRow>();
                var partData = parts.WritePtr<Vector4>();
                int partNext = 0;

                float frameSeconds = Math.Clamp(Time.SmoothDelta, 1f / 240f, 1f / 30f);
                var wind = Wind.Local;
                for (int i = 0; i < count; i++)
                {
                    var entry = _picked[i];
                    var cloth = entry.Cloth;
                    var renderer = cloth.Renderer;
                    var mesh = entry.Mesh!;
                    var settings = cloth.Settings ?? ClothSettings.Default;
                    var world = renderer.Transform.Matrix;
                    Matrix4x4.Invert(world, out var toObject);

                    // Started over where it is held: the first time, or after a pause in which a character's bones or a
                    // static piece moved (a flag that stood still carries on from where it was)
                    bool paused = entry.SimulatedTick < Engine.TickCount - 2;
                    bool reset = entry.Fresh || (paused && (cloth.Animator != null || world != entry.LastWorld));
                    entry.Fresh = false;
                    entry.SimulatedTick = Engine.TickCount;
                    entry.SimulatedAt = Time.TotalTime;
                    entry.LastWorld = world;

                    // The body without the clothes that are cloth, round cones as thick on average as its oval parts
                    uint partFirst = (uint)partNext, partCount = 0;
                    if (cloth.Animator != null && cloth.Animator.TryGetBareBodyFrames(out var body, out var frames))
                    {
                        for (int p = 0; p < body.Parts.Length && partCount < MostParts; p++)
                        {
                            var part = body.Parts[p];
                            float thick = MathF.Sqrt(part.Depth);
                            var from = Vector3.Transform(part.From, frames[p]);
                            var to = Vector3.Transform(part.To, frames[p]);
                            partData[partNext++] = new Vector4(from, part.RadiusFrom * thick);
                            partData[partNext++] = new Vector4(to, part.RadiusTo * thick);
                            partCount++;
                        }
                    }

                    rowData[i] = new ClothRow
                    {
                        World = world,
                        ToObject = toObject,
                        Wind = wind,
                        Dt = frameSeconds / entry.Substeps,
                        Drag = settings.Drag,
                        Damping = settings.Damping,
                        Stiffness = settings.Stiffness,
                        Bending = settings.Bending,
                        Thickness = settings.Thickness,
                        Backstop = settings.Backstop,
                        Flutter = settings.Flutter,
                        Streams = settings.Streams,
                        Substeps = (uint)entry.Substeps,
                        Iterations = (uint)entry.Iterations,
                        Reset = reset ? 1u : 0u,
                        PartFirst = partFirst,
                        PartCount = partCount,
                        ParticleCount = (uint)mesh.ParticleCount,
                        VertexCount = (uint)mesh.VertexCount,
                        VertexFirst = (uint)mesh.VertexFirst,
                        StateBase = (uint)entry.State,
                        OutputBase = (uint)entry.Output,
                        Particles = mesh.Particles.SrvIndex,
                        Links = mesh.Links.SrvIndex,
                        Vertices = mesh.Vertices.SrvIndex,
                        VertexTris = mesh.VertexTris.SrvIndex,
                        Limits = entry.Limits!.SrvIndex,
                        Bones = renderer.BoneBufferIdx,
                        MeshPositions = mesh.Mesh.PosBufferIndex,
                        MeshNormals = mesh.Mesh.NormBufferIndex,
                        MeshWeights = mesh.Mesh.BoneWeightBufferIndex,
                    };
                    cloth.SimulatedTick = Engine.TickCount;
                    SimulatedParticles += mesh.ParticleCount;
                }
                SimulatedCount = count;

                if (_compute == null)
                {
                    _compute = new ComputeShader("cloth.hlsl");
                    _kernel = _compute.FindKernel("CSCloth");
                }

                PixMarker.Begin(cmd, "Cloth");
                var device = Engine.Device;
                cmd.SetDescriptorHeaps(1, new[] { device.SrvHeap });
                cmd.SetComputeRootSignature(device.GlobalRootSignature);

                // Named without their "Idx" (ComputeShader drops it from the cbuffer's names)
                var cs = _compute;
                cs.SetPushConstant("Rows", rows.SrvIndex);
                cs.SetPushConstant("Parts", parts.SrvIndex);
                cs.SetPushConstant("StateUAV", _state!.UavIndex);
                cs.SetPushConstant("OutputUAV", _output!.UavIndex);
                cs.SetPushConstant("TimeBits", BitConverter.SingleToUInt32Bits(Time.TotalTime));

                _state.Transition(cmd, ResourceStates.UnorderedAccess);
                _output.Transition(cmd, ResourceStates.UnorderedAccess);
                cs.Dispatch(_kernel, cmd, (uint)count);
                _output.Transition(cmd, ResourceStates.NonPixelShaderResource);
                PixMarker.End(cmd);
            }

            // What the vertex shaders read: every cloth that has an output, named again when it or the pool changed
            for (int i = 0; i < _entries.Count; i++)
            {
                var entry = _entries[i];
                if (entry.Output < 0 || entry.SimulatedTick == long.MinValue || entry.Cloth.Renderer == null) continue;
                Show(entry, entry.Cloth.Renderer);
            }
        }

        private void Show(Entry entry, PersistentRenderer renderer)
        {
            if (entry.Shown != null && entry.Shown != renderer) Hide(entry);
            uint pool = _output!.SrvIndex, at = (uint)entry.Output;
            uint first = (uint)entry.Mesh!.VertexFirst, count = (uint)entry.Mesh.VertexCount;
            if (entry.Shown == renderer && entry.ShownPool == pool && entry.ShownBase == at && entry.ShownFirst == first && entry.ShownCount == count)
                return;
            renderer.WriteInstance("ClothVertices", new Vector4(pool, 0f, 0f, 0f));
            renderer.WriteInstance("ClothBase", new Vector4(at, 0f, 0f, 0f));
            renderer.WriteInstance("ClothFirst", new Vector4(first, 0f, 0f, 0f));
            renderer.WriteInstance("ClothCount", new Vector4(count, 0f, 0f, 0f));
            entry.Shown = renderer;
            entry.ShownPool = pool;
            entry.ShownBase = at;
            entry.ShownFirst = first;
            entry.ShownCount = count;
        }

        // ────────────── Pools ──────────────

        private static int Round(int float4s) => (float4s + RangeGrain - 1) / RangeGrain * RangeGrain;

        // Room for this much more in each pool: a pool that lacks it doubles, its old contents copied over
        private void EnsurePools(ID3D12GraphicsCommandList cmd, int stateWanted, int outputWanted)
        {
            if (_state == null)
            {
                _stateCapacity = Math.Max(PoolStart, NextPow2(stateWanted));
                _state = GraphicsBuffer.CreateStructured<Vector4>(_stateCapacity, srv: true, uav: true);
                _stateFree.Add((0, _stateCapacity));
            }
            else if (!Fits(_stateFree, stateWanted))
                Grow(cmd, ref _state, ref _stateCapacity, _stateFree, stateWanted);

            if (_output == null)
            {
                _outputCapacity = Math.Max(PoolStart, NextPow2(outputWanted));
                _output = GraphicsBuffer.CreateStructured<Vector4>(_outputCapacity, srv: true, uav: true);
                _outputFree.Add((0, _outputCapacity));
            }
            else if (!Fits(_outputFree, outputWanted))
                Grow(cmd, ref _output, ref _outputCapacity, _outputFree, outputWanted);
        }

        private static void Grow(ID3D12GraphicsCommandList cmd, ref GraphicsBuffer? pool, ref int capacity,
                                 List<(int Start, int Count)> free, int wanted)
        {
            int grown = capacity;
            while (!FitsAfterGrowth(free, capacity, grown, wanted)) grown *= 2;

            var bigger = GraphicsBuffer.CreateStructured<Vector4>(grown, srv: true, uav: true);
            pool!.Transition(cmd, ResourceStates.CopySource);
            bigger.Transition(cmd, ResourceStates.CopyDest);
            cmd.CopyBufferRegion(bigger.Native, 0, pool.Native, 0, (ulong)capacity * 16);
            Engine.Device.DeferDispose(pool);

            Free(free, capacity, grown - capacity);
            pool = bigger;
            capacity = grown;
        }

        // Whether the ranges for this many float4s fit, taken one cloth at a time first fit: surely when the largest
        // free run alone holds them all
        private static bool Fits(List<(int Start, int Count)> free, int wanted)
        {
            if (wanted <= 0) return true;
            int largest = 0;
            foreach (var run in free) largest = Math.Max(largest, run.Count);
            return largest >= wanted;
        }

        // The same once the pool has grown: the free run at its end joins the new room
        private static bool FitsAfterGrowth(List<(int Start, int Count)> free, int capacity, int grown, int wanted)
        {
            int tail = 0;
            if (free.Count > 0 && free[^1].Start + free[^1].Count == capacity) tail = free[^1].Count;
            return tail + grown - capacity >= wanted;
        }

        private static int NextPow2(int value)
        {
            int p = 1;
            while (p < value) p <<= 1;
            return p;
        }

        // First fit
        private static int Take(List<(int Start, int Count)> free, int size)
        {
            for (int i = 0; i < free.Count; i++)
            {
                var (start, count) = free[i];
                if (count < size) continue;
                if (count == size) free.RemoveAt(i);
                else free[i] = (start + size, count - size);
                return start;
            }
            throw new InvalidOperationException("ClothSystem: a pool has no room (EnsurePools missed it)");
        }

        // A run back, joined to its neighbours
        private static void Free(List<(int Start, int Count)> free, int start, int size)
        {
            if (start < 0 || size <= 0) return;
            int i = 0;
            while (i < free.Count && free[i].Start < start) i++;
            free.Insert(i, (start, size));
            if (i + 1 < free.Count && free[i].Start + free[i].Count == free[i + 1].Start)
            {
                free[i] = (free[i].Start, free[i].Count + free[i + 1].Count);
                free.RemoveAt(i + 1);
            }
            if (i > 0 && free[i - 1].Start + free[i - 1].Count == free[i].Start)
            {
                free[i - 1] = (free[i - 1].Start, free[i - 1].Count + free[i].Count);
                free.RemoveAt(i);
            }
        }

        // This frame's upload buffer, big enough: a frame's own, so the frames in flight keep theirs
        private static GraphicsBuffer EnsureUpload<T>(GraphicsBuffer?[] buffers, int[] capacities, int frame, int count) where T : unmanaged
        {
            var buffer = buffers[frame];
            if (buffer != null && capacities[frame] >= count) return buffer;
            Engine.Device.DeferDispose(buffer);
            int capacity = Math.Max(64, NextPow2(count));
            buffer = GraphicsBuffer.CreateUpload<T>(capacity, mapped: true);
            buffers[frame] = buffer;
            capacities[frame] = capacity;
            return buffer;
        }
    }
}
