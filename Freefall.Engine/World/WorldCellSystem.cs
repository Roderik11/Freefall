using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using static Freefall.World.WorldCells;

namespace Freefall.World
{
    /// <summary>
    /// Keeps the world cells' tiers by the viewers' and the camera's distance, within
    /// <see cref="EngineSettings.WorldCellsBudgetMs"/> a frame, and takes the far meshes built on workers.
    /// Every 0.25 s it looks at the active slices and the grid squares within reach of the viewers
    /// (<see cref="Vicinity.Viewers"/>, else the camera) and of the camera, never at the whole map:
    /// <list type="bullet">
    /// <item>a viewer within <see cref="EngineSettings.WorldCellsNear"/> (left 15 % further out): a slice's entities, nearest first;</item>
    /// <item>the camera within a group's ring (its kind's far tier distance and band, or where the GPU stops drawing it if
    /// nearer, <see cref="WorldCells.RingOf"/>) and a margin: the group's instances;</item>
    /// <item>beyond: the far form alone.</item>
    /// </list>
    /// A slice whose entities are made keeps its groups' instances two more frames, until the entities draw: they draw
    /// the same meshes, so the swap shows nothing.
    /// </summary>
    [UpdateInEditor]
    [UpdateAfter(typeof(ScriptGroup))]
    public sealed class WorldCellSystem : EntitySystem
    {
        /// <summary>Far meshes built on workers, made into the scene here, one a frame.</summary>
        internal static readonly ConcurrentQueue<FarBuilder.Result> Built = new();

        /// <summary>Groups whose instances go in a frame or two.</summary>
        internal static readonly List<Group> Dropping = new();

        private const float LookEvery = 0.25f;
        private const float RingMargin = 128f;   // the camera may move this far between two looks before a cell it fades in lacks instances
        private const float Leave = 1.15f;       // what is made goes again this much further out
        private const float CopiesPerMsAwake = 200f;   // copies whose colliders a cell's body makes in a millisecond when it wakes (play)

        /// <summary>
        /// Cells still to spawn or to make instances of, and far meshes still to make, after the last update: 0 when what
        /// the last look wanted is all there (the world map's bake waits for it). The look itself comes every 0.25 s.
        /// </summary>
        public static int Queued { get; private set; }

        /// <summary>Looks made so far. The world map's bake waits for one made after it moved the camera.</summary>
        public static int Looks { get; private set; }

        private static bool s_lookNow;

        /// <summary>
        /// The next update looks round the camera at once instead of at the next 0.25 s: after the camera jumps (the world
        /// map's bake moves it from chunk to chunk), so <see cref="Queued"/> speaks of the new place a frame later.
        /// </summary>
        public static void LookNow() => s_lookNow = true;

        private float _sinceLook = LookEvery;
        private bool _farTierOn;
        private int _stamp;
        private readonly List<(Slice Slice, float Distance)> _toSpawn = new();
        private readonly List<Slice> _toDespawn = new();
        private readonly List<(Group Group, float Distance)> _toInstance = new();
        private readonly List<Group> _toDrop = new();
        private readonly List<Vector3> _viewers = new();
        private readonly List<Slice> _candidates = new();

        // Where its frame goes, in the profiler
        private static readonly ProfilerMarker s_lookMarker = ProfilerMarker.Get("Cells look");
        private static readonly ProfilerMarker s_despawnMarker = ProfilerMarker.Get("Cells despawn");
        private static readonly ProfilerMarker s_dropMarker = ProfilerMarker.Get("Cells instances drop");
        private static readonly ProfilerMarker s_instanceMarker = ProfilerMarker.Get("Cells instances add");
        private static readonly ProfilerMarker s_spawnMarker = ProfilerMarker.Get("Cells spawn");

        public override void Update()
        {
            long start = Stopwatch.GetTimestamp();

            // The shaders' far tier is on while any owner keeps records
            if (OwnerCount > 0)
            {
                if (FarTier.HeldOffWithin > 0f) FarTier.Band = 0f;
                else if (!_farTierOn || FarTier.Band <= 0) FarTier.Band = FarTier.DefaultBand;
                _farTierOn = true;
            }
            else if (_farTierOn)
            {
                FarTier.Band = 0;
                _farTierOn = false;
            }

            if (Built.TryDequeue(out var result)) FarBuilder.Apply(result);
            ApplyMoves();   // spawners whose output moved: their records' cells where it stands now

            // Instances of slices spawned two frames ago go: their entities draw by now
            for (int i = Dropping.Count - 1; i >= 0; i--)
            {
                var g = Dropping[i];
                if (!g.Slice.Gone && g.DropInstancesIn > 0 && --g.DropInstancesIn == 0) RemoveInstances(g);
                if (g.Slice.Gone || g.DropInstancesIn <= 0)
                {
                    Dropping[i] = Dropping[^1];
                    Dropping.RemoveAt(Dropping.Count - 1);
                }
            }

            if (OwnerCount == 0) return;

            // Real time: the camera moves while the game clock stands still (a pause, slow motion, a stopped clock)
            _sinceLook += Time.RealDelta;
            if (_sinceLook >= LookEvery || s_lookNow)
            {
                _sinceLook = 0;
                s_lookNow = false;
                using (s_lookMarker.Auto()) Look();
                Looks++;
            }

            // Within the budget: what left first, then instances for the nearest, then entities for the nearest. At least
            // one thing a frame; a slice holds at most WorldCells.MaxSliceRecords, so one is never far past the budget.
            double budget = Engine.Settings.WorldCellsBudgetMs;
            int done = 0;
            long work = Stopwatch.GetTimestamp();

            using (s_despawnMarker.Auto())
                while (_toDespawn.Count > 0 && (done == 0 || Stopwatch.GetElapsedTime(work).TotalMilliseconds < budget))
                {
                    var slice = _toDespawn[^1];
                    _toDespawn.RemoveAt(_toDespawn.Count - 1);
                    if (slice.Gone || slice.Near || slice.Spawned == null) continue;
                    int draws = Despawn(slice);
                    if (draws > 0) { Stats.InstanceGroupsMade++; Stats.InstanceDrawsMade += draws; }
                    Stats.Despawned++;
                    done++;
                }
            using (s_dropMarker.Auto())
                while (_toDrop.Count > 0)
                {
                    var g = _toDrop[^1];
                    _toDrop.RemoveAt(_toDrop.Count - 1);
                    if (g.Slice.Gone || g.InRing || g.Draws == null) continue;
                    RemoveInstances(g);
                    Stats.InstanceGroupsDropped++;
                }
            int next = 0;
            using (s_instanceMarker.Auto())
                while (next < _toInstance.Count && (done == 0 || Stopwatch.GetElapsedTime(work).TotalMilliseconds < budget))
                {
                    var g = _toInstance[next++].Group;
                    if (g.Slice.Gone || !g.InRing || g.Draws != null || g.Slice.Spawned != null) continue;
                    long t0 = Stopwatch.GetTimestamp();
                    int draws = AddInstances(g);
                    Stats.InstanceMs += Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                    Stats.InstanceGroupsMade++;
                    Stats.InstanceDrawsMade += draws;
                    done++;
                }
            _toInstance.RemoveRange(0, next);
            next = 0;
            // In play a copy's collider is made when its cell's body wakes, the frame after (RigidBody.CreateActor, ~5 µs
            // a copy): copies taken from the parked ones cost next to nothing here, so dozens of cells fitted the budget
            // and their bodies woke together (62 ms, TestWorld's capital). So the copies are counted too.
            int copiesLeft = Engine.IsEditor ? int.MaxValue : (int)(budget * CopiesPerMsAwake);
            using (s_spawnMarker.Auto())
                while (next < _toSpawn.Count && (done == 0 || (copiesLeft > 0 && Stopwatch.GetElapsedTime(work).TotalMilliseconds < budget)))
                {
                    var slice = _toSpawn[next++].Slice;
                    if (slice.Gone || !slice.Near || slice.Spawned != null) continue;
                    copiesLeft -= slice.Records.Count;
                    long t0 = Stopwatch.GetTimestamp();
                    Spawn(slice);
                    Stats.SpawnMs += Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                    if (slice.Spawned == null) continue;
                    Stats.SpawnedEntities += slice.Records.Count;
                    Stats.Spawned++;
                    done++;
                    foreach (var g in slice.Groups)
                        if (g.Draws != null)
                        {
                            g.DropInstancesIn = 2;
                            Dropping.Add(g);
                        }
                }
            _toSpawn.RemoveRange(0, next);

            Stats.MaxFrameMs = Math.Max(Stats.MaxFrameMs, Stopwatch.GetElapsedTime(start).TotalMilliseconds);
            Queued = _toSpawn.Count + _toInstance.Count + Built.Count;
        }

        // Which slices want entities (a viewer within A) and which groups instances (the camera within their ring and a margin)
        private void Look()
        {
            _viewers.Clear();
            foreach (var v in Vicinity.Viewers) if (v != null) _viewers.Add(v.WorldPosition);
            var cam = Camera.Main;
            var camera = cam?.Transform?.WorldPosition;
            if (_viewers.Count == 0 && camera != null) _viewers.Add(camera.Value);
            if (_viewers.Count == 0) return;
            var eye = camera ?? _viewers[0];

            // Where the GPU stops drawing a mesh: its radius over the cull size × LODScale × √FoVFactor (cull_instances.hlsl)
            float cullFactor = cam != null ? Engine.Settings.LODScale * MathF.Sqrt(MathF.Max(cam.FoVFactor, 0f)) : 0f;

            float a = Engine.Settings.WorldCellsNear;
            float enter = a * a, leave = enter * Leave * Leave;
            // The grid round the camera as far as the far tier's ring; slices wanted from further away (drawn until the
            // GPU drops them: rocks, logs, a town) come from their owners' lists below, by distance
            float reach = MathF.Max(MathF.Max(FarTier.From + FarTier.Band, FarTier.From + FarTier.DefaultBand), FarTier.HeldOffWithin) + RingMargin;

            _toSpawn.Clear(); _toDespawn.Clear(); _toInstance.Clear(); _toDrop.Clear();
            _stamp++;
            _candidates.Clear();
            foreach (var slice in Active) { slice.Stamp = _stamp; _candidates.Add(slice); }
            int nearReach = (int)MathF.Ceiling(a * Leave / CellSize) + 1;
            foreach (var p in _viewers) Gather(p.X, p.Z, nearReach);
            Gather(eye.X, eye.Z, (int)MathF.Ceiling(reach * Leave / CellSize) + 1);
            foreach (var o in Owners)
                foreach (var slice in o.LongSlices)
                {
                    if (slice.Stamp == _stamp) continue;
                    float dx = slice.Cell.Centre.X - eye.X, dz = slice.Cell.Centre.Y - eye.Z;
                    float far = (slice.MaxRing + RingMargin) * Leave;
                    if (dx * dx + dz * dz > far * far) continue;
                    slice.Stamp = _stamp;
                    _candidates.Add(slice);
                }

            foreach (var slice in _candidates)
            {
                if (slice.Gone) { Active.Remove(slice); continue; }
                float d = float.MaxValue;
                foreach (var p in _viewers)
                {
                    float dx = MathF.Max(MathF.Max(slice.Min.X - p.X, 0f), p.X - slice.Max.X);
                    float dz = MathF.Max(MathF.Max(slice.Min.Y - p.Z, 0f), p.Z - slice.Max.Y);
                    d = MathF.Min(d, dx * dx + dz * dz);
                }
                bool near = slice.Near ? d < leave : d < enter;
                if (near && slice.Spawned == null) _toSpawn.Add((slice, d));
                else if (!near && slice.Spawned != null) _toDespawn.Add(slice);
                slice.Near = near;

                // By the cell's centre, as the shaders fade it; each group by its own ring
                float cx = slice.Cell.Centre.X - eye.X, cz = slice.Cell.Centre.Y - eye.Z, c = cx * cx + cz * cz;
                bool wanted = near || slice.Spawned != null;
                foreach (var g in slice.Groups)
                {
                    float ring = RingOf(g, cullFactor) + RingMargin;
                    float ringEnter = ring * ring;
                    bool inRing = g.InRing ? c < ringEnter * Leave * Leave : c < ringEnter;
                    g.InRing = inRing;
                    if (inRing && g.Draws == null && slice.Spawned == null) _toInstance.Add((g, c));
                    else if (!inRing && g.Draws != null) _toDrop.Add(g);
                    wanted |= inRing || g.Draws != null;
                }

                if (wanted) Active.Add(slice);
                else Active.Remove(slice);
            }
            _toSpawn.Sort((x, y) => x.Distance.CompareTo(y.Distance));
            _toInstance.Sort((x, y) => x.Distance.CompareTo(y.Distance));
        }

        // The slices in the grid squares within reach squares of (x, z), each once per look
        private void Gather(float x, float z, int reach)
        {
            var (cx, cz) = KeyOf(x, z);
            int x0 = Math.Max(cx - reach, KeyMin.X), x1 = Math.Min(cx + reach, KeyMax.X);
            int z0 = Math.Max(cz - reach, KeyMin.Z), z1 = Math.Min(cz + reach, KeyMax.Z);
            var cells = WorldCells.Cells;
            for (int kz = z0; kz <= z1; kz++)
                for (int kx = x0; kx <= x1; kx++)
                {
                    if (!cells.TryGetValue((kx, kz), out var cell)) continue;
                    foreach (var slice in cell.Slices)
                    {
                        if (slice.Stamp == _stamp) continue;
                        slice.Stamp = _stamp;
                        _candidates.Add(slice);
                    }
                }
        }
    }

    /// <summary>
    /// The world cells' drawing, once per view as the render group runs: the one far forest of every owner's trees, one
    /// cull and one draw for all of them (instances and near entities draw through the engine's batches).
    /// </summary>
    [UpdateInEditor]
    [UpdateInGroup(typeof(RenderGroup))]
    public sealed class WorldCellDrawSystem : EntitySystem
    {
        public override void Update() => Forest.Draw();
    }
}
