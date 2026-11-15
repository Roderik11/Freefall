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
    /// <item>a viewer within <see cref="EngineSettings.WorldCellsNear"/> (left 15 % further out): entities, nearest first;</item>
    /// <item>the camera within <see cref="FarTier.From"/> + <see cref="FarTier.Band"/> and a margin: instances;</item>
    /// <item>beyond: the far form alone.</item>
    /// </list>
    /// A slice whose entities are made keeps its instances two more frames, until the entities draw: they draw the same
    /// meshes, so the swap shows nothing.
    /// </summary>
    [UpdateInEditor]
    [UpdateAfter(typeof(ScriptGroup))]
    public sealed class WorldCellSystem : EntitySystem
    {
        /// <summary>Far meshes built on workers, made into the scene here, one a frame.</summary>
        internal static readonly ConcurrentQueue<FarBuilder.Result> Built = new();

        /// <summary>Slices whose instances go in a frame or two.</summary>
        internal static readonly List<Slice> Dropping = new();

        private const float LookEvery = 0.25f;
        private const float RingMargin = 128f;   // the camera may move this far between two looks before a cell it fades in lacks instances
        private const float Leave = 1.15f;       // what is made goes again this much further out

        private float _sinceLook = LookEvery;
        private bool _farTierOn;
        private int _stamp;
        private readonly List<(Slice Slice, float Distance)> _toSpawn = new();
        private readonly List<Slice> _toDespawn = new();
        private readonly List<(Slice Slice, float Distance)> _toInstance = new();
        private readonly List<Slice> _toDrop = new();
        private readonly List<Vector3> _viewers = new();
        private readonly List<Slice> _candidates = new();

        public override void Update()
        {
            long start = Stopwatch.GetTimestamp();

            // The shaders' far tier is on while any owner keeps records
            if (OwnerCount > 0)
            {
                if (!_farTierOn || FarTier.Band <= 0) FarTier.Band = FarTier.DefaultBand;
                _farTierOn = true;
            }
            else if (_farTierOn)
            {
                FarTier.Band = 0;
                _farTierOn = false;
            }

            if (Built.TryDequeue(out var result)) FarBuilder.Apply(result);

            // Instances of slices spawned two frames ago go: their entities draw by now
            for (int i = Dropping.Count - 1; i >= 0; i--)
            {
                var slice = Dropping[i];
                if (!slice.Gone && slice.DropInstancesIn > 0 && --slice.DropInstancesIn == 0) RemoveInstances(slice);
                if (slice.Gone || slice.DropInstancesIn <= 0)
                {
                    Dropping[i] = Dropping[^1];
                    Dropping.RemoveAt(Dropping.Count - 1);
                }
            }

            if (OwnerCount == 0) return;

            _sinceLook += Time.Delta;
            if (_sinceLook >= LookEvery)
            {
                _sinceLook = 0;
                Look();
            }

            // Within the budget: what left first, then instances for the nearest, then entities for the nearest
            double budget = Engine.Settings.WorldCellsBudgetMs;
            int done = 0;
            long work = Stopwatch.GetTimestamp();
            bool Within() => done == 0 || Stopwatch.GetElapsedTime(work).TotalMilliseconds < budget;

            while (_toDespawn.Count > 0 && Within())
            {
                var slice = _toDespawn[^1];
                _toDespawn.RemoveAt(_toDespawn.Count - 1);
                if (slice.Gone || slice.Near || slice.Spawned == null) continue;
                int draws = Despawn(slice);
                if (draws > 0) { Stats.InstanceSlicesMade++; Stats.InstanceDrawsMade += draws; }
                Stats.Despawned++;
                done++;
            }
            while (_toDrop.Count > 0)
            {
                var slice = _toDrop[^1];
                _toDrop.RemoveAt(_toDrop.Count - 1);
                if (slice.Gone || slice.InRing || slice.Group == null) continue;
                RemoveInstances(slice);
                Stats.InstanceSlicesDropped++;
            }
            int next = 0;
            while (next < _toInstance.Count && Within())
            {
                var slice = _toInstance[next++].Slice;
                if (slice.Gone || !slice.InRing || slice.Group != null || slice.Spawned != null) continue;
                long t0 = Stopwatch.GetTimestamp();
                int draws = AddInstances(slice);
                Stats.InstanceMs += Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                Stats.InstanceSlicesMade++;
                Stats.InstanceDrawsMade += draws;
                done++;
            }
            _toInstance.RemoveRange(0, next);
            next = 0;
            while (next < _toSpawn.Count && Within())
            {
                var slice = _toSpawn[next++].Slice;
                if (slice.Gone || !slice.Near || slice.Spawned != null) continue;
                long t0 = Stopwatch.GetTimestamp();
                Spawn(slice);
                Stats.SpawnMs += Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                if (slice.Spawned == null) continue;
                Stats.SpawnedEntities += slice.Records.Count;
                Stats.Spawned++;
                done++;
                if (slice.Group != null)
                {
                    slice.DropInstancesIn = 2;
                    Dropping.Add(slice);
                }
            }
            _toSpawn.RemoveRange(0, next);

            Stats.MaxFrameMs = Math.Max(Stats.MaxFrameMs, Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        }

        // Which slices want entities (a viewer within A) and which instances (the camera within the far tier, its band and a margin)
        private void Look()
        {
            _viewers.Clear();
            foreach (var v in Vicinity.Viewers) if (v != null) _viewers.Add(v.WorldPosition);
            var camera = Camera.Main?.Transform?.WorldPosition;
            if (_viewers.Count == 0 && camera != null) _viewers.Add(camera.Value);
            if (_viewers.Count == 0) return;
            var eye = camera ?? _viewers[0];

            float a = Engine.Settings.WorldCellsNear;
            float enter = a * a, leave = enter * Leave * Leave;
            float ring = FarTier.From + FarTier.Band + RingMargin;
            float ringEnter = ring * ring, ringLeave = ringEnter * Leave * Leave;

            _toSpawn.Clear(); _toDespawn.Clear(); _toInstance.Clear(); _toDrop.Clear();
            _stamp++;
            _candidates.Clear();
            foreach (var slice in Active) { slice.Stamp = _stamp; _candidates.Add(slice); }
            int nearReach = (int)MathF.Ceiling(a * Leave / CellSize) + 1;
            foreach (var p in _viewers) Gather(p.X, p.Z, nearReach);
            Gather(eye.X, eye.Z, (int)MathF.Ceiling(ring * Leave / CellSize) + 1);

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

                // By the cell's centre, as the shaders fade it
                float cx = slice.Cell.Centre.X - eye.X, cz = slice.Cell.Centre.Y - eye.Z, c = cx * cx + cz * cz;
                bool inRing = slice.InRing ? c < ringLeave : c < ringEnter;
                slice.InRing = inRing;
                if (inRing && slice.Group == null && slice.Spawned == null) _toInstance.Add((slice, c));
                else if (!inRing && slice.Group != null) _toDrop.Add(slice);

                if (near || inRing || slice.Spawned != null || slice.Group != null) Active.Add(slice);
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

    /// <summary>Draws the owners' far forests, once per view, as the render group runs.</summary>
    [UpdateInEditor]
    [UpdateInGroup(typeof(RenderGroup))]
    public sealed class WorldCellDrawSystem : EntitySystem
    {
        public override void Update()
        {
            if (OwnerCount == 0) return;
            foreach (var o in Owners)
                o.Forest?.Draw();
        }
    }
}
