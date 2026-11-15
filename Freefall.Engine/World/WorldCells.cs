using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;

namespace Freefall.World
{
    /// <summary>What stands in for an owner's placements past the far tier (<see cref="FarTier.From"/>).</summary>
    public enum FarForm
    {
        /// <summary>From each part's material: a tree's cards (gbuffer_crossmesh) as a far forest, CliffRock shells merged
        /// into a far mesh, nothing for the rest.</summary>
        Auto,
        /// <summary>The cards alone, as a far forest.</summary>
        Cards,
        /// <summary>Shells and cards merged into one far mesh.</summary>
        Merged,
        /// <summary>Nothing: the placements fade out at the far tier.</summary>
        None,
    }

    /// <summary>
    /// The world's cells (WORLD_CELLS_PLAN.md): one sparse grid of <see cref="FarTier.CellSize"/> squares for the whole
    /// world, holding every placement a spawner hands it as a record (a prefab and a transform). <see cref="WorldCellSystem"/>
    /// keeps what each cell needs: entities within <see cref="EngineSettings.WorldCellsNear"/> of a viewer, instances (each
    /// placement's LOD chains drawn without an entity) round the camera out to the far tier, and past it the owner's far
    /// form (<see cref="FarForm"/>).
    ///
    /// <para>An owner is the component that spawns: a <see cref="PCGComponent"/> with <see cref="PCGComponent.Cells"/>
    /// (through <see cref="Sink"/>), or a script that calls <see cref="Begin"/>, <see cref="Place"/> and <see cref="End"/>
    /// itself. Owners are known by their UID, so one recreated by a script reload finds its records: they are engine data
    /// and survive it. Records are kept in the owner's output frame, as the spawner placed them, so a near cell's copies
    /// are placed under its output exactly as an instantiating run would have placed them.</para>
    ///
    /// <para>A near cell is one container entity per owner ("Cell x,z" under the owner's output) with one static body:
    /// <see cref="PCGComponent.GeneratorOf"/> still finds the generator, and the cell is one PhysX actor.</para>
    ///
    /// Main thread.
    /// </summary>
    public static class WorldCells
    {
        /// <summary>One placement: its prefab (an index into the owner's list) and where it stands in the owner's output frame.</summary>
        internal struct Record
        {
            public int Prefab;
            public Vector3 Position;
            public Quaternion Rotation;
            public Vector3 Scale;
        }

        /// <summary>What a record costs: prefab index, position, rotation, scale.</summary>
        public const int RecordBytes = 44;

        internal sealed class Owner
        {
            public ulong Uid;
            public Component Component = null!;
            public Entity? Container;
            public FarForm Far;
            public Matrix4x4 ToWorld = Matrix4x4.Identity;
            public readonly List<Prefab> Prefabs = new();
            public readonly Dictionary<Prefab, int> PrefabIndex = new();
            public readonly List<Record> Records = new();
            public readonly List<Slice> Slices = new();
            public bool Open;                   // between Begin and End
            public int Generation;              // a far build of an earlier generation is dropped

            public FarForest? Forest;
            public Entity? FarEntity;
            public Mesh? FarMesh;
            public long FarVertices, FarTriangles, FarParts, FarVram, FarCardTriangles;
            public double FarBuildMs;
            public long InstanceDraws;

            public string Name => Component?.Entity?.Name ?? Uid.ToString();
            public bool Alive => Container != null && !Container.IsBeingDestroyed && Container.Transform != null;
        }

        /// <summary>One grid square: the slices of the owners with records in it.</summary>
        internal sealed class Cell
        {
            public (int X, int Z) Key;
            public Vector2 Centre;                      // world XZ, (Key + 0.5) × CellSize: what the shaders fade it by
            public readonly List<Slice> Slices = new(1);
        }

        /// <summary>One owner's records in one cell, and what is made of them.</summary>
        internal sealed class Slice
        {
            public Owner Owner = null!;
            public Cell Cell = null!;
            public readonly List<int> Records = new();
            public Vector2 Min = new(float.MaxValue), Max = new(float.MinValue);   // world XZ of its placements
            public bool Gone;                           // its owner ran again or was cleared
            public bool Near;                           // a viewer within A: its entities
            public bool InRing;                         // the camera within the far tier and its band: its instances (unless entities)
            public Entity? Spawned;
            public int DropInstancesIn = -1;            // frames until its instances go (its entities draw by then)
            public DrawGroup? Group;                    // its instances' draws
            public List<int>? Slots;                    // and their transform slots
            public int Draws;
            public int Stamp;                           // the look that last visited it
        }

        private static readonly Dictionary<ulong, Owner> s_owners = new();
        private static readonly Dictionary<(int, int), Cell> s_cells = new();

        /// <summary>Slices holding or wanting entities or instances: a look visits these and the squares round the viewers, never the whole map.</summary>
        internal static readonly HashSet<Slice> Active = new();

        internal static (int X, int Z) KeyMin = (int.MaxValue, int.MaxValue), KeyMax = (int.MinValue, int.MinValue);
        internal static IReadOnlyDictionary<(int, int), Cell> Cells => s_cells;
        internal static IEnumerable<Owner> Owners => s_owners.Values;

        /// <summary>Owners with records.</summary>
        public static int OwnerCount => s_owners.Count;

        /// <summary>Metres a side of a cell.</summary>
        public static float CellSize => FarTier.CellSize;

        public static (int X, int Z) KeyOf(float x, float z) => ((int)MathF.Floor(x / CellSize), (int)MathF.Floor(z / CellSize));

        // ── The API ──

        /// <summary>
        /// A spawner starts placing: whatever it placed before is let go of (entities, instances, far form), and what it
        /// places until <see cref="End"/> is kept in <paramref name="container"/>'s frame. Near cells' entities go under
        /// the container.
        /// </summary>
        public static void Begin(Component owner, Entity container, FarForm far = FarForm.Auto)
        {
            ArgumentNullException.ThrowIfNull(owner);
            ArgumentNullException.ThrowIfNull(container);
            if (!s_owners.TryGetValue(owner.UID, out var o))
                s_owners[owner.UID] = o = new Owner { Uid = owner.UID };
            else
                Drop(o);
            o.Component = owner;
            o.Container = container;
            o.Far = far;
            o.Open = true;
            o.Generation++;
            o.Records.Clear();
            o.Prefabs.Clear();
            o.PrefabIndex.Clear();
        }

        /// <summary>One placement, in the frame of the container given to <see cref="Begin"/>.</summary>
        public static void Place(Component owner, Prefab prefab, Vector3 position, Quaternion rotation, Vector3 scale)
        {
            if (prefab == null) return;
            if (!s_owners.TryGetValue(owner.UID, out var o) || !o.Open)
                throw new InvalidOperationException($"WorldCells.Place for '{owner.Entity?.Name}' outside Begin/End");
            if (!o.PrefabIndex.TryGetValue(prefab, out int index))
            {
                index = o.Prefabs.Count;
                o.Prefabs.Add(prefab);
                o.PrefabIndex[prefab] = index;
            }
            o.Records.Add(new Record { Prefab = index, Position = position, Rotation = rotation, Scale = scale });
        }

        /// <summary>The placing is done: the records go into their cells and the far form is made (the far mesh on workers).</summary>
        public static void End(Component owner)
        {
            if (!s_owners.TryGetValue(owner.UID, out var o) || !o.Open) return;
            o.Open = false;
            if (!o.Alive || o.Records.Count == 0)
            {
                if (o.Records.Count == 0) s_owners.Remove(o.Uid);
                return;
            }

            o.ToWorld = o.Container!.Transform.WorldMatrix;
            var byKey = new Dictionary<(int, int), Slice>();
            for (int i = 0; i < o.Records.Count; i++)
            {
                var w = Vector3.Transform(o.Records[i].Position, o.ToWorld);
                var key = KeyOf(w.X, w.Z);
                if (!byKey.TryGetValue(key, out var slice))
                {
                    if (!s_cells.TryGetValue(key, out var cell))
                    {
                        s_cells[key] = cell = new Cell { Key = key, Centre = new Vector2((key.Item1 + 0.5f) * CellSize, (key.Item2 + 0.5f) * CellSize) };
                        KeyMin = (Math.Min(KeyMin.X, key.Item1), Math.Min(KeyMin.Z, key.Item2));
                        KeyMax = (Math.Max(KeyMax.X, key.Item1), Math.Max(KeyMax.Z, key.Item2));
                    }
                    byKey[key] = slice = new Slice { Owner = o, Cell = cell };
                    cell.Slices.Add(slice);
                    o.Slices.Add(slice);
                }
                slice.Records.Add(i);
                slice.Min = Vector2.Min(slice.Min, new Vector2(w.X, w.Z));
                slice.Max = Vector2.Max(slice.Max, new Vector2(w.X, w.Z));
            }

            if (o.Far == FarForm.None) return;
            var shapes = o.Prefabs.Select(PrefabShapes.Of).ToArray();
            if (o.Far is FarForm.Auto or FarForm.Cards) BuildForest(o, shapes);
            bool mergeCards = o.Far == FarForm.Merged;
            if (o.Far is FarForm.Auto or FarForm.Merged
                && shapes.Any(parts => parts.Any(p => p.Kind == PrefabShapes.Kind.Shell || mergeCards && p.Kind == PrefabShapes.Kind.Card)))
                FarBuilder.Start(o, shapes, mergeCards);
        }

        /// <summary>Everything the owner placed goes: its cells' entities, instances, far form and records.</summary>
        public static void Clear(Component owner)
        {
            if (owner == null || !s_owners.Remove(owner.UID, out var o)) return;
            Drop(o);
        }

        /// <summary>Whether the owner has records.</summary>
        public static bool Has(Component owner) => owner != null && s_owners.ContainsKey(owner.UID);

        /// <summary>How many records the owner has.</summary>
        public static int RecordsOf(Component owner) => owner != null && s_owners.TryGetValue(owner.UID, out var o) ? o.Records.Count : 0;

        /// <summary>
        /// A <see cref="PCGComponent"/>'s placements into the cells (<see cref="PCGComponent.Cells"/>): the run's output
        /// container is the frame and the parent of the near cells.
        /// </summary>
        public sealed class Sink : IPlacementSink
        {
            private readonly Component _owner;
            private readonly Func<FarForm> _far;

            public Sink(Component owner, Func<FarForm> far)
            {
                _owner = owner;
                _far = far;
            }

            public void PlacementsBegin(Entity container) => Begin(_owner, container, _far());
            public void Place(Prefab prefab, Vector3 position, Quaternion rotation, Vector3 scale) => WorldCells.Place(_owner, prefab, position, rotation, scale);
            public void PlacementsEnd() => End(_owner);
        }

        // ── What a cell is made into ──

        private static void Drop(Owner o)
        {
            foreach (var slice in o.Slices)
            {
                slice.Gone = true;
                if (slice.Spawned != null && !slice.Spawned.IsBeingDestroyed) slice.Spawned.Destroy();
                slice.Spawned = null;
                RemoveInstances(slice);
                Active.Remove(slice);
                var cell = slice.Cell;
                cell.Slices.Remove(slice);
                if (cell.Slices.Count == 0) s_cells.Remove(cell.Key);
            }
            o.Slices.Clear();
            DropFar(o);
            o.Forest?.Dispose();
            o.Forest = null;
            o.Generation++;
            if (s_cells.Count == 0)
            {
                KeyMin = (int.MaxValue, int.MaxValue);
                KeyMax = (int.MinValue, int.MinValue);
            }
        }

        internal static void DropFar(Owner o)
        {
            if (o.FarEntity != null && !o.FarEntity.IsBeingDestroyed) o.FarEntity.Destroy();   // gone already when the output was destroyed
            o.FarEntity = null;
            if (o.FarMesh != null) Engine.Device.DeferDispose(o.FarMesh);   // its registry parts and buffers, once no frame draws it
            o.FarMesh = null;
            o.FarVertices = o.FarTriangles = o.FarParts = o.FarVram = o.FarCardTriangles = 0;
        }

        internal static Matrix4x4 LocalOf(in Record r) =>
            Matrix4x4.CreateScale(r.Scale) * Matrix4x4.CreateFromQuaternion(r.Rotation) * Matrix4x4.CreateTranslation(r.Position);

        private static readonly MaterialBlock NoParams = new();
        private static readonly Dictionary<Effect, int> s_instanceSlots = new();

        // An effect's shared per-object data slot, its defaults
        private static int InstanceSlotFor(Material material)
        {
            var effect = material.Effect;
            if (effect == null || effect.InstanceParams.Count == 0) return 0;
            if (s_instanceSlots.TryGetValue(effect, out int slot)) return slot;
            slot = InstanceData.Allocate();
            InstanceData.WriteDefaults(slot, effect.InstanceParams);
            s_instanceSlots[effect] = slot;
            return slot;
        }

        /// <summary>A slice's instances: each placement's LOD chains as its MeshRenderers draw them, a transform slot per
        /// renderer, no entity; flagged as the far tier's near side, so they give way to the far form past FarTier.From.</summary>
        internal static int AddInstances(Slice slice)
        {
            if (slice.Group != null || slice.Gone) return 0;
            var o = slice.Owner;
            slice.Group = new DrawGroup();
            slice.Slots = new List<int>(slice.Records.Count * 2);
            int draws = 0;
            foreach (int i in slice.Records)
            {
                var r = o.Records[i];
                var world = LocalOf(r) * o.ToWorld;
                foreach (var rd in PrefabShapes.RenderersOf(o.Prefabs[r.Prefab]))
                {
                    int slot = TransformBuffer.Instance.AllocateSlot();
                    TransformBuffer.Instance.SetTransform(slot, rd.ToRoot * world);
                    slice.Slots.Add(slot);
                    foreach (var (part, material) in rd.Chains)
                    {
                        CommandBuffer.AddPersistent(slice.Group, rd.Mesh, part, material, NoParams, slot, lodManaged: true,
                            instanceSlot: InstanceSlotFor(material), farTier: true);
                        draws++;
                    }
                }
            }
            slice.Draws = draws;
            o.InstanceDraws += draws;
            return draws;
        }

        internal static void RemoveInstances(Slice slice)
        {
            if (slice.Group == null) return;
            CommandBuffer.RemovePersistent(slice.Group);
            foreach (int slot in slice.Slots!) TransformBuffer.Instance.ReleaseSlot(slot);
            slice.Owner.InstanceDraws -= slice.Draws;
            slice.Draws = 0;
            slice.Group = null;
            slice.Slots = null;
            slice.DropInstancesIn = -1;
        }

        /// <summary>The slice's entities, under one container with a static body for their colliders.</summary>
        internal static void Spawn(Slice slice)
        {
            var o = slice.Owner;
            if (slice.Gone || slice.Spawned != null || !o.Alive) return;
            var root = new Entity($"Cell {slice.Cell.Key.X},{slice.Cell.Key.Z}");
            root.Transform.Parent = o.Container!.Transform;
            root.AddComponent<RigidBody>();
            foreach (int i in slice.Records)
            {
                var r = o.Records[i];
                var e = o.Prefabs[r.Prefab].Instantiate();
                if (e == null) continue;
                e.Transform.Parent = root.Transform;
                e.Transform.Position = r.Position;
                e.Transform.Rotation = r.Rotation;
                e.Transform.Scale = r.Scale;
            }
            slice.Spawned = root;
        }

        /// <summary>Its instances back first (when the camera's ring wants them), then its entities go. Returns the draws made.</summary>
        internal static int Despawn(Slice slice)
        {
            if (slice.Spawned == null) return 0;
            int draws = 0;
            slice.DropInstancesIn = -1;
            if (slice.InRing && slice.Group == null) draws = AddInstances(slice);
            if (!slice.Spawned.IsBeingDestroyed) slice.Spawned.Destroy();
            slice.Spawned = null;
            return draws;
        }

        /// <summary>Every placement's cards (its coarsest LOD in gbuffer_crossmesh) as a tree of the far forest: its world
        /// position, yaw and scales from the placement's frame and the renderer's place in the prefab.</summary>
        private static void BuildForest(Owner o, PrefabShapes.Part[][] shapes)
        {
            var kinds = new List<FarForest.Kind>();
            var kindOf = new Dictionary<(Mesh, int, Material), int>();
            var trees = new List<FarForest.Tree>();
            bool full = false;
            foreach (var r in o.Records)
            {
                Matrix4x4? world = null;
                foreach (var part in shapes[r.Prefab])
                {
                    if (part.Kind != PrefabShapes.Kind.Card || part.Material == null || part.Mesh == null) continue;
                    var key = (part.Mesh, part.PartIndex, part.Material);
                    if (!kindOf.TryGetValue(key, out int kind))
                    {
                        if (kinds.Count == 64) { full = true; continue; }
                        kind = kinds.Count;
                        kindOf[key] = kind;
                        kinds.Add(new FarForest.Kind { Mesh = part.Mesh, Part = part.PartIndex, Material = part.Material });
                    }
                    var m = part.ToRoot * (world ??= LocalOf(r) * o.ToWorld);
                    trees.Add(new FarForest.Tree
                    {
                        Position = m.Translation,
                        Yaw = MathF.Atan2(-m.M13, m.M11),
                        ScaleAcross = new Vector3(m.M11, m.M12, m.M13).Length(),
                        ScaleUp = new Vector3(m.M21, m.M22, m.M23).Length(),
                        Kind = kind,
                    });
                }
            }
            if (full) Debug.LogWarning("WorldCells", $"'{o.Name}': more than 64 kinds of cards; the rest have no far form");
            if (trees.Count > FarForest.MaxTrees)
            {
                Debug.LogWarning("WorldCells", $"'{o.Name}': {trees.Count:N0} far trees, more than a far forest holds ({FarForest.MaxTrees:N0}); the rest have no far form");
                trees.RemoveRange(FarForest.MaxTrees, trees.Count - FarForest.MaxTrees);
            }
            if (trees.Count > 0) o.Forest = new FarForest(kinds, trees.ToArray(), CellSize);
        }

        // ── Counts ──

        internal struct Counters
        {
            public double MaxFrameMs, SpawnMs, InstanceMs;
            public long Spawned, Despawned, SpawnedEntities, InstanceSlicesMade, InstanceSlicesDropped, InstanceDrawsMade;
        }

        internal static Counters Stats;

        /// <summary>The counters since the scene loaded (or the last reset): worst frame, spawns, instances made.</summary>
        public static void ResetCounters() => Stats = default;

        /// <summary>
        /// What the cells hold now, for engine_stats: records, cells and slices (an owner's records in one cell: the cell
        /// test's "cells"), what is made of them, the far forms, and the system's counters.
        /// </summary>
        public static object Describe()
        {
            long records = 0, slices = 0, spawned = 0, entities = 0, instanceSlices = 0, instancePlacements = 0, instanceDraws = 0;
            long trees = 0, forestVram = 0, farMeshes = 0, farVertices = 0, farTriangles = 0, farVram = 0;
            int forests = 0;
            foreach (var o in s_owners.Values)
            {
                records += o.Records.Count;
                slices += o.Slices.Count;
                instanceDraws += o.InstanceDraws;
                foreach (var s in o.Slices)
                {
                    if (s.Spawned != null) { spawned++; entities += s.Records.Count; }
                    if (s.Group != null) { instanceSlices++; instancePlacements += s.Records.Count; }
                }
                if (o.Forest != null) { forests++; trees += o.Forest.TreeCount; forestVram += o.Forest.VramBytes; }
                if (o.FarMesh != null) { farMeshes++; farVertices += o.FarVertices; farTriangles += o.FarTriangles; farVram += o.FarVram; }
            }
            var c = Stats;
            return new
            {
                owners = s_owners.Count,
                records,
                recordBytes = records * RecordBytes,
                cells = s_cells.Count,
                slices,
                activeSlices = Active.Count,
                near = new { slices = spawned, placements = entities },
                instances = new { slices = instanceSlices, placements = instancePlacements, draws = instanceDraws },
                farForests = new { forests, trees, vramBytes = forestVram },
                farMeshes = new { meshes = farMeshes, vertices = farVertices, triangles = farTriangles, vramBytes = farVram },
                system = new
                {
                    worstFrameMs = Math.Round(c.MaxFrameMs, 3),
                    slicesSpawned = c.Spawned,
                    slicesDespawned = c.Despawned,
                    usPerEntity = Math.Round(1000 * c.SpawnMs / Math.Max(1, c.SpawnedEntities), 2),
                    instanceSlicesMade = c.InstanceSlicesMade,
                    instanceSlicesDropped = c.InstanceSlicesDropped,
                    usPerInstanceDraw = Math.Round(1000 * c.InstanceMs / Math.Max(1, c.InstanceDrawsMade), 3),
                },
                farTier = new { from = FarTier.From, band = FarTier.Band, cellSize = FarTier.CellSize },
                byOwner = s_owners.Values.OrderByDescending(o => o.Records.Count).Take(40).Select(o => new
                {
                    name = o.Name,
                    records = o.Records.Count,
                    slices = o.Slices.Count,
                    far = o.Far.ToString(),
                    forestTrees = o.Forest?.TreeCount ?? 0,
                    farTriangles = o.FarTriangles,
                    farVramBytes = o.FarVram,
                    farBuildMs = Math.Round(o.FarBuildMs),
                }).ToArray(),
            };
        }
    }
}
