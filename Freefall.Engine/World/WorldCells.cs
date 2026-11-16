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
    /// <summary>What stands in for an owner's placements past their ring (the far tier).</summary>
    public enum FarForm
    {
        /// <summary>From each part's material: a tree's cards (gbuffer_crossmesh) as a far forest, CliffRock shells merged
        /// into a far mesh, nothing for the rest.</summary>
        Auto,
        /// <summary>The cards alone, as a far forest.</summary>
        Cards,
        /// <summary>Shells and cards merged into one far mesh.</summary>
        Merged,
        /// <summary>Nothing: the placements fade out at their ring.</summary>
        None,
    }

    /// <summary>A <c>SpawnPrefab</c> output's far form: the generator's, or one of its own.</summary>
    public enum SpawnFar
    {
        /// <summary>The generator's (<see cref="PCGComponent.Far"/>).</summary>
        Generator,
        Auto,
        Cards,
        Merged,
        None,
    }

    /// <summary>
    /// The world's cells (WORLD_CELLS_PLAN.md): one sparse grid of <see cref="FarTier.CellSize"/> squares for the whole
    /// world, holding every placement a spawner hands it as a record (a placement kind and a transform).
    /// <see cref="WorldCellSystem"/> keeps what each cell needs: entities within <see cref="EngineSettings.WorldCellsNear"/>
    /// of a viewer, instances (each placement's LOD chains drawn without an entity) round the camera out to each kind's
    /// ring, and past it the owner's far form (<see cref="FarForm"/>).
    ///
    /// <para>An owner is the component that spawns: a <see cref="PCGComponent"/> with <see cref="PCGComponent.Cells"/>
    /// (through <see cref="Sink"/>), or a script that calls <see cref="Begin"/>, <see cref="Place"/> and <see cref="End"/>
    /// itself. Owners are known by their UID, so one recreated by a script reload finds its records: they are engine data
    /// and survive it. Records are kept in the owner's output frame, as the spawner placed them, so a near cell's copies
    /// are placed under its output exactly as an instantiating run would have placed them.</para>
    ///
    /// <para><b>Kinds and rings</b> (stage 4): a record names a placement kind: its prefab, its ring (the distance its
    /// instances give way to the far form at, a far tier profile, <see cref="FarTier.ProfileFor"/>) and its far form. A
    /// kind's instances also leave the ring where the GPU stops drawing them anyway, too small on screen
    /// (<see cref="Mesh.LODCullScreenSize"/>, by the live camera's field of view and <c>LODScale</c>): small rocks and logs
    /// hold no instances where nobody could see them.</para>
    ///
    /// <para>A near cell is one container entity per owner ("Cell x,z" under the owner's output) with one static body:
    /// <see cref="PCGComponent.GeneratorOf"/> still finds the generator, and the cell is one PhysX actor.</para>
    ///
    /// Main thread.
    /// </summary>
    public static class WorldCells
    {
        /// <summary>One placement: its kind (an index into the owner's list) and where it stands in the owner's output frame.</summary>
        internal struct Record
        {
            public int Kind;
            public Vector3 Position;
            public Quaternion Rotation;
            public Vector3 Scale;
        }

        /// <summary>What a record costs: kind index, position, rotation, scale.</summary>
        public const int RecordBytes = 44;

        /// <summary>A prefab as an owner places it: with the ring and far form its placements take.</summary>
        internal sealed class Kind
        {
            public Prefab Prefab = null!;
            public float RingSet;               // the ring asked for (owner's or output's), metres; 0 = Auto
            public float Ring;                  // metres its far side starts at (the profile's distance), resolved at End
            public FarForm Far;
            public int Profile;                 // FarTier.ProfileFor(Ring)
            public float CullTerm;              // its renderers' largest radius / LODCullScreenSize: × scale × LODScale × √FoVFactor = where the GPU stops drawing it; 0 = never
            public float MaxScale;              // its placements' largest axis scale (in the output frame)
            public bool Coarse;                 // the ring is where its coarsest LOD starts, which its far form draws as it is
            public bool Stateful;               // its prefab keeps state of its own (IRecordState): its records have ids
            public MaterialSwap? Swap;          // materials it draws with in place of its prefab's (a house's palette)
            public bool NoFar;                  // nothing stands in for it far away: its instances draw until the GPU's size cull drops them, with no far tier fade
        }

        internal sealed class Owner
        {
            public ulong Uid;
            public Component Component = null!;
            public Entity? Container;
            public FarForm Far;
            public float Ring;                  // metres; 0 = FarTier.From
            public Matrix4x4 ToWorld = Matrix4x4.Identity;
            public float FrameScale = 1f;       // the output frame's largest axis scale
            public readonly List<Kind> Kinds = new();
            public readonly Dictionary<(Prefab, float, FarForm, MaterialSwap?), int> KindIndex = new();
            public readonly List<Record> Records = new();
            public readonly List<Slice> Slices = new();
            public readonly List<Entity> Live = new();           // live prefabs, made as entities under the container at once
            public readonly Dictionary<string, int> LiveBecause = new();   // the component that made them live, per type
            public bool Open;                   // between Begin and End
            public int Generation;              // a far build of an earlier generation is dropped
            public float MaxRing;               // the farthest any of its kinds reaches (from + band)
            public readonly List<Transform> Watched = new();  // its output and everything above it, whose moves it hears of
            public Action? OnMoved;
            public readonly List<Slice> LongSlices = new();   // slices whose ring reaches past the far tier's (WorldCellSystem looks at them by distance)
            public int StatefulRecords;         // records with an id (their prefab keeps state)
            public Dictionary<int, ulong>? IdOverrides;   // ids of records that stand on the spot of an earlier one of the same prefab

            public Entity? FarEntity;
            public Mesh? FarMesh;
            public long FarVertices, FarTriangles, FarParts, FarVram, FarCardTriangles;
            public double FarBuildMs;
            public long InstanceDraws;

            public string Name => Component?.Entity?.Name ?? Uid.ToString();
            public override string ToString() => Name;
            public bool Alive => Container != null && !Container.IsBeingDestroyed && Container.Transform != null;
        }

        /// <summary>One grid square: the slices of the owners with records in it.</summary>
        internal sealed class Cell
        {
            public (int X, int Z) Key;
            public Vector2 Centre;                      // world XZ, (Key + 0.5) × CellSize: what the shaders fade it by
            public readonly List<Slice> Slices = new(1);
        }

        /// <summary>One owner's records in one cell, and what is made of them: entities for all, instances per kind.</summary>
        internal sealed class Slice
        {
            public Owner Owner = null!;
            public Cell Cell = null!;
            public readonly List<int> Records = new();
            public readonly List<Group> Groups = new(1);
            public Vector2 Min = new(float.MaxValue), Max = new(float.MinValue);   // world XZ of its placements
            public bool Gone;                           // its owner ran again or was cleared
            public bool Near;                           // a viewer within A: its entities
            public Entity? Spawned;
            public List<(Entity Copy, int Record)>? Stateful;   // its copies that keep state, while it is spawned
            public float MaxRing;                       // the farthest any of its groups is wanted from (its kinds' rings)
            public int Stamp;                           // the look that last visited it
        }

        /// <summary>A slice's records of one kind: instanced together, while the camera is within the kind's ring.</summary>
        internal sealed class Group
        {
            public Slice Slice = null!;
            public int Kind;
            public readonly List<int> Records = new();
            public float MaxScale;                      // the largest axis scale of its records (in the output frame)
            public bool InRing;
            public int DropInstancesIn = -1;            // frames until its instances go (the slice's entities draw by then)
            public DrawGroup? Draws;                    // its instances' draws
            public List<int>? Slots;                    // and their transform slots
            public int DrawCount;
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
        /// the container. <paramref name="far"/> and <paramref name="ring"/> (metres; 0 = <see cref="FarTier.From"/>) are
        /// what its placements take unless a placement says otherwise.
        /// </summary>
        public static void Begin(Component owner, Entity container, FarForm far = FarForm.Auto, float ring = 0f)
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
            o.Ring = ring;
            o.Open = true;
            o.Generation++;
            o.Records.Clear();
            o.Kinds.Clear();
            o.KindIndex.Clear();
            o.LiveBecause.Clear();
        }

        /// <summary>
        /// One placement, in the frame of the container given to <see cref="Begin"/>, with the owner's ring and far form. A
        /// live prefab (one with a component a record cannot stand for: <see cref="PrefabShapes.LiveBecause"/>,
        /// <see cref="CellRecordAttribute"/>) is instantiated under the container at once, as without cells; the rest
        /// become records.
        /// </summary>
        public static void Place(Component owner, Prefab prefab, Vector3 position, Quaternion rotation, Vector3 scale)
            => Place(owner, prefab, position, rotation, scale, 0f, SpawnFar.Generator);

        /// <summary>One placement drawn with <paramref name="swap"/>'s materials in place of its prefab's (a house's palette;
        /// null = its own). A live prefab's copy gets the swap as well.</summary>
        public static void Place(Component owner, Prefab prefab, Vector3 position, Quaternion rotation, Vector3 scale, MaterialSwap? swap)
            => Place(owner, prefab, position, rotation, scale, 0f, SpawnFar.Generator, swap);

        /// <summary>
        /// Whether placements of <paramref name="prefab"/> become records (else they are live: an entity everywhere, as
        /// <see cref="Place(Component, Prefab, Vector3, Quaternion, Vector3)"/> makes them). For a spawner that keeps its live
        /// pieces itself, under entities of its own (a house's door leaves and lit windows under the house).
        /// </summary>
        public static bool IsRecord(Prefab prefab) => prefab != null && PrefabShapes.LiveBecause(prefab) == null;

        /// <summary>One placement with a ring (metres; 0 = the owner's) and a far form of its own.</summary>
        public static void Place(Component owner, Prefab prefab, Vector3 position, Quaternion rotation, Vector3 scale, float ring, SpawnFar far)
            => Place(owner, prefab, position, rotation, scale, ring, far, null);

        /// <summary>One placement with a ring, a far form and materials of its own.</summary>
        public static void Place(Component owner, Prefab prefab, Vector3 position, Quaternion rotation, Vector3 scale, float ring, SpawnFar far, MaterialSwap? swap)
        {
            if (prefab == null) return;
            if (!s_owners.TryGetValue(owner.UID, out var o) || !o.Open)
                throw new InvalidOperationException($"WorldCells.Place for '{owner.Entity?.Name}' outside Begin/End");
            if (PrefabShapes.LiveBecause(prefab) is { } because)
            {
                var e = prefab.Instantiate();
                if (e == null || !o.Alive) return;
                e.Transform.Parent = o.Container!.Transform;
                e.Transform.Position = position;
                e.Transform.Rotation = rotation;
                e.Transform.Scale = scale;
                swap?.ApplyTo(e);
                o.Live.Add(e);
                o.LiveBecause[because] = o.LiveBecause.GetValueOrDefault(because) + 1;
                return;
            }
            float kindRing = ring > 0f ? ring : o.Ring;   // 0 = Auto, resolved at End
            var kindFar = far == SpawnFar.Generator ? o.Far : (FarForm)((int)far - 1);
            var key = (prefab, kindRing, kindFar, swap);
            if (!o.KindIndex.TryGetValue(key, out int index))
            {
                index = o.Kinds.Count;
                o.Kinds.Add(new Kind { Prefab = prefab, RingSet = kindRing, Far = kindFar, CullTerm = CullTermOf(prefab), Stateful = PrefabShapes.IsStateful(prefab), Swap = swap });
                o.KindIndex[key] = index;
            }
            o.Kinds[index].MaxScale = MathF.Max(o.Kinds[index].MaxScale, MathF.Max(MathF.Abs(scale.X), MathF.Max(MathF.Abs(scale.Y), MathF.Abs(scale.Z))));
            o.Records.Add(new Record { Kind = index, Position = position, Rotation = rotation, Scale = scale });
        }

        /// <summary>
        /// The ring of a kind: the one asked for; with Auto, where its coarsest LOD starts when its far form draws that LOD
        /// as it is (a tree's cards in a far forest: nothing changes on screen where it hands over), else the far tier's
        /// own distance (merged shells, simplified, keep it). Worked out with the camera of the moment (its field of view,
        /// LODScale) and the kind's largest placement.
        /// </summary>
        private static void ResolveRing(Owner o, Kind k)
        {
            k.Coarse = false;
            k.NoFar = false;
            float ring = k.RingSet > 0f ? k.RingSet : FarTier.From;

            // Nothing stands in for it far away (no far form, or none of its parts are what its far form takes): it is drawn
            // as an entity would be, until the GPU's size cull drops it, wherever that is (logs, rocks, a town's pieces)
            if (k.RingSet <= 0f && !HasFarForm(k) && k.CullTerm > 0f)
            {
                float fovFactor = Camera.Main is { FoVFactor: > 0f } c ? c.FoVFactor : 5.8284f;   // 45° when there is no camera
                float cull = k.CullTerm * k.MaxScale * o.FrameScale * Engine.Settings.LODScale * MathF.Sqrt(fovFactor) * 1.1f;
                k.NoFar = true;
                k.Ring = MathF.Max(2f * CellSize, MathF.Ceiling(cull / 16f) * 16f);
                k.Profile = 0;
                return;
            }

            if (k.RingSet <= 0f && k.Far != FarForm.None)
            {
                float coarseTerm = 0f;
                bool covered = true, shells = false, any = false;
                foreach (var rd in PrefabShapes.RenderersOf(k.Prefab))
                {
                    if (!rd.Enabled || rd.Chains.Length == 0) continue;
                    any = true;
                    coarseTerm = MathF.Max(coarseTerm, rd.CoarseTerm);
                    shells |= rd.CoarseShells;
                    if (rd.CoarseOther || rd.CoarseTerm <= 0f
                        || rd.CoarseShells && k.Far is FarForm.Cards
                        || rd.CoarseCards && k.Far is not (FarForm.Auto or FarForm.Cards or FarForm.Merged))
                        covered = false;
                }
                // Only where the far form is the coarsest LOD itself (cards): merged shells are simplified and lose their
                // hidden faces, so they keep the far tier's distance, where they were judged
                if (any && covered && !shells && coarseTerm > 0f)
                {
                    float fovFactor = Camera.Main is { FoVFactor: > 0f } cam ? cam.FoVFactor : 5.8284f;   // 45° when there is no camera
                    // As the GPU switches, with room for its hysteresis
                    float coarse = coarseTerm * k.MaxScale * o.FrameScale * Engine.Settings.LODScale * MathF.Sqrt(fovFactor) * 1.1f;
                    coarse = MathF.Max(2f * CellSize, MathF.Ceiling(coarse / 16f) * 16f);
                    if (coarse < ring)
                    {
                        ring = coarse;
                        k.Coarse = true;
                    }
                }
            }
            k.Ring = ring;
            k.Profile = FarTier.ProfileFor(ring);
        }

        // How far from the camera a kind's instances are wanted at most: its ring and band, or where the GPU stops drawing it
        private static float ReachOf(Kind k) => k.NoFar ? k.Ring : FarTier.FromOf(k.Profile) + FarTier.BandOf(k.Profile);

        // Whether its far form takes any of its parts: cards (Auto, Cards, Merged) or CliffRock shells (Auto, Merged)
        private static bool HasFarForm(Kind k)
        {
            if (k.Far == FarForm.None) return false;
            foreach (var part in PrefabShapes.Of(k.Prefab))
                if (k.Far != FarForm.Cards || part.Kind == PrefabShapes.Kind.Card) return true;
            return false;
        }

        // The largest radius among a prefab's drawn renderers, in the prefab's frame, over the GPU's cull size: the factor
        // that gives the distance past which none of it is drawn (cull_instances.hlsl CullK)
        private static float CullTermOf(Prefab prefab)
        {
            float term = 0f;
            foreach (var rd in PrefabShapes.RenderersOf(prefab))
            {
                if (!rd.Enabled || rd.Chains.Length == 0) continue;
                var box = rd.Mesh.BoundingBox;
                float radius = Vector3.Distance(box.Max, (box.Min + box.Max) * 0.5f) * MaxAxisScale(rd.ToRoot);
                term = MathF.Max(term, radius / Mesh.LODCullScreenSize);
            }
            return term;
        }

        private static float MaxAxisScale(in Matrix4x4 m) => PrefabShapes.MaxAxisScale(m);

        /// <summary>The placing is done: the records go into their cells and the far form is made (the far mesh on workers).</summary>
        public static void End(Component owner)
        {
            if (!s_owners.TryGetValue(owner.UID, out var o) || !o.Open) return;
            o.Open = false;
            if (o.Alive) AssignIds(o);
            if (!o.Alive || o.Records.Count == 0)
            {
                if (o.Records.Count == 0 && o.Live.Count == 0) s_owners.Remove(o.Uid);
                return;
            }
            Build(o);
            Watch(o);
        }

        // ── Moving a spawner ──

        // Owners whose output (or something above it) moved since the last frame: re-sliced by the system
        private static readonly HashSet<Owner> s_moved = new();

        // The owner hears of its output and everything above it moving (Transform.OnChanged is a transform's own change)
        private static void Watch(Owner o)
        {
            Unwatch(o);
            o.OnMoved ??= () => s_moved.Add(o);
            for (var t = o.Container?.Transform; t != null; t = t.Parent)
            {
                t.OnChanged += o.OnMoved;
                o.Watched.Add(t);
            }
        }

        private static void Unwatch(Owner o)
        {
            foreach (var t in o.Watched) t.OnChanged -= o.OnMoved;
            o.Watched.Clear();
        }

        /// <summary>
        /// The owners whose output moved (a house dragged, its street's group turned): their records stay where they are in
        /// the output's frame, so their cells, instances and far forms are made again where the output now stands. Once a
        /// frame, from <see cref="WorldCellSystem"/>.
        /// </summary>
        internal static void ApplyMoves()
        {
            if (s_moved.Count == 0) return;
            var moved = s_moved.ToArray();
            s_moved.Clear();
            foreach (var o in moved)
            {
                if (o.Open || !o.Alive || o.Records.Count == 0 || !s_owners.TryGetValue(o.Uid, out var current) || current != o) continue;
                if (o.Container!.Transform.WorldMatrix == o.ToWorld) continue;   // its own child changed, or a change that moved nothing
                Unslice(o);
                Build(o);
                Watch(o);
            }
        }

        // The records into their cells where the output stands now, their rings resolved, and the far forms made
        private static void Build(Owner o)
        {
            o.ToWorld = o.Container!.Transform.WorldMatrix;
            o.FrameScale = MaxAxisScale(o.ToWorld);
            o.MaxRing = 0f;
            foreach (var k in o.Kinds)
            {
                ResolveRing(o, k);
                o.MaxRing = MathF.Max(o.MaxRing, ReachOf(k));
            }

            var byKey = new Dictionary<(int, int), Slice>();
            for (int i = 0; i < o.Records.Count; i++)
            {
                var r = o.Records[i];
                var w = Vector3.Transform(r.Position, o.ToWorld);
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

                Group? group = null;
                foreach (var g in slice.Groups) if (g.Kind == r.Kind) { group = g; break; }
                if (group == null) slice.Groups.Add(group = new Group { Slice = slice, Kind = r.Kind });
                group.Records.Add(i);
                group.MaxScale = MathF.Max(group.MaxScale, MathF.Max(MathF.Abs(r.Scale.X), MathF.Max(MathF.Abs(r.Scale.Y), MathF.Abs(r.Scale.Z))) * o.FrameScale);
            }

            // The slices wanted from further away than the far tier's ring: the system looks at them by distance, not
            // through the grid round the camera (which reaches only as far as the far tier)
            o.LongSlices.Clear();
            float shortReach = FarTier.From + FarTier.DefaultBand;
            foreach (var s in o.Slices)
            {
                s.MaxRing = 0f;
                foreach (var g in s.Groups) s.MaxRing = MathF.Max(s.MaxRing, ReachOf(o.Kinds[g.Kind]));
                if (s.MaxRing > shortReach) o.LongSlices.Add(s);
            }

            // Far forms: cards as a far forest per profile, shells (and Merged kinds' cards) in one far mesh
            var shapes = o.Kinds.Select(k => k.Far == FarForm.None ? Array.Empty<PrefabShapes.Part>() : PrefabShapes.Of(k.Prefab)).ToArray();
            if (o.Kinds.Any(k => k.Far is FarForm.Auto or FarForm.Cards)) BuildForests(o, shapes);
            bool merges = false;
            for (int k = 0; k < o.Kinds.Count && !merges; k++)
            {
                var far = o.Kinds[k].Far;
                foreach (var p in shapes[k])
                    if (p.Kind == PrefabShapes.Kind.Shell && far is FarForm.Auto or FarForm.Merged
                        || p.Kind == PrefabShapes.Kind.Card && far == FarForm.Merged) { merges = true; break; }
            }
            if (merges) FarBuilder.Start(o, shapes);
        }

        // ── Records with state (IRecordState, WorldState) ──

        /// <summary>
        /// A record's id: its owner's UID, its prefab's GUID and its position (in the owner's output frame) to the millimetre,
        /// so a re-run that places it again gives it the same id, and one stray point does not shift the others'. A record on
        /// the spot of an earlier one of the same prefab gets a counter (<see cref="AssignIds"/>).
        /// </summary>
        internal static ulong IdOf(Owner o, int record)
        {
            if (o.IdOverrides != null && o.IdOverrides.TryGetValue(record, out ulong id)) return id;
            var r = o.Records[record];
            var prefab = o.Kinds[r.Kind].Prefab;
            ulong h = Fnv(14695981039346656037UL, o.Uid);
            foreach (char c in string.IsNullOrEmpty(prefab.Guid) ? prefab.Name ?? "" : prefab.Guid) h = (h ^ c) * 1099511628211UL;
            h = Fnv(h, (ulong)(long)Math.Round(r.Position.X * 1000.0));
            h = Fnv(h, (ulong)(long)Math.Round(r.Position.Y * 1000.0));
            return Fnv(h, (ulong)(long)Math.Round(r.Position.Z * 1000.0));
        }

        private static ulong Fnv(ulong h, ulong v)
        {
            for (int i = 0; i < 8; i++)
            {
                h ^= (v >> (i * 8)) & 0xFF;
                h *= 1099511628211UL;
            }
            return h;
        }

        // The ids of the owner's records with state, a counter for those on the spot of an earlier one; what was kept for
        // records that are no longer placed (their placements moved, re-iding them) goes and is counted in the log
        private static void AssignIds(Owner o)
        {
            o.IdOverrides = null;
            o.StatefulRecords = 0;
            bool any = false;
            foreach (var k in o.Kinds) any |= k.Stateful;
            if (!any && WorldState.CountOf(o.Uid) == 0) return;
            var ids = new HashSet<ulong>();
            int twins = 0;
            for (int i = 0; i < o.Records.Count; i++)
            {
                if (!o.Kinds[o.Records[i].Kind].Stateful) continue;
                ulong id = IdOf(o, i);
                if (!ids.Add(id))
                {
                    ulong n = 1, other;
                    do other = Fnv(id, n++); while (!ids.Add(other));
                    (o.IdOverrides ??= new())[i] = other;
                    twins++;
                }
            }
            o.StatefulRecords = ids.Count;
            if (twins > 0)
                Debug.LogWarning("WorldCells", $"'{o.Name}': {twins} placements with state stand on the spot of another of the same prefab; they are told apart by their order");
            int orphans = WorldState.Prune(o.Uid, ids);
            if (orphans > 0)
                Debug.LogWarning("WorldCells", $"'{o.Name}': what was kept of {orphans} placements is gone: they are no longer placed where they were (an outline or the graph changed)");
        }

        private static readonly List<IRecordState> s_states = new();

        // The copy's components that keep state, in the copy's own order (the same for every copy of one prefab)
        private static void StatesOf(Entity copy, List<IRecordState> into)
        {
            into.Clear();
            var stack = new Stack<Transform>();
            stack.Push(copy.Transform);
            while (stack.Count > 0)
            {
                var t = stack.Pop();
                if (t.Entity != null)
                    foreach (var c in t.Entity.Components)
                        if (c is IRecordState s) into.Add(s);
                for (int i = t.Count - 1; i >= 0; i--) stack.Push(t.GetChild(i));
            }
        }

        // A copy just made reads what its record kept
        private static void LoadState(Owner o, Entity copy, int record)
        {
            if (!WorldState.TryGet(IdOf(o, record), out var states)) return;
            StatesOf(copy, s_states);
            for (int k = 0; k < s_states.Count && k < states.Length; k++)
            {
                if (states[k] == null) continue;
                try { s_states[k].Load(states[k]!); }
                catch (Exception ex) { Debug.LogWarning("WorldCells", $"{s_states[k].GetType().Name}.Load on '{copy.Name}' threw: {ex.Message}"); }
            }
            s_states.Clear();   // script components: held here they would keep a reloaded script assembly from unloading
        }

        // The slice's copies are about to go: what changed is kept, what is back to its prefab's is let go of
        private static void KeepState(Slice slice)
        {
            if (slice.Stateful == null) return;
            var o = slice.Owner;
            foreach (var (copy, record) in slice.Stateful)
            {
                StatesOf(copy, s_states);
                object?[]? kept = null;
                for (int k = 0; k < s_states.Count; k++)
                {
                    try
                    {
                        if (s_states[k].Changed) (kept ??= new object?[s_states.Count])[k] = s_states[k].Save();
                    }
                    catch (Exception ex) { Debug.LogWarning("WorldCells", $"{s_states[k].GetType().Name}.Save on '{copy.Name}' threw: {ex.Message}"); }
                }
                ulong id = IdOf(o, record);
                if (kept != null) WorldState.Put(id, o.Uid, kept);
                else WorldState.Remove(id);
            }
            s_states.Clear();   // script components: held here they would keep a reloaded script assembly from unloading
            slice.Stateful = null;
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
            private readonly Func<float> _ring;

            public Sink(Component owner, Func<FarForm> far, Func<float>? ring = null)
            {
                _owner = owner;
                _far = far;
                _ring = ring ?? (() => 0f);
            }

            public void PlacementsBegin(Entity container) => Begin(_owner, container, _far(), _ring());
            public void Place(Prefab prefab, Vector3 position, Quaternion rotation, Vector3 scale) => WorldCells.Place(_owner, prefab, position, rotation, scale);
            public void Place(Prefab prefab, Vector3 position, Quaternion rotation, Vector3 scale, float ring, SpawnFar far) => WorldCells.Place(_owner, prefab, position, rotation, scale, ring, far);
            public void PlacementsEnd() => End(_owner);
        }

        // ── What a cell is made into ──

        private static void Drop(Owner o)
        {
            Unwatch(o);
            Unslice(o);
            foreach (var e in o.Live)
                if (!e.IsBeingDestroyed) e.Destroy();   // gone already when the output was destroyed
            o.Live.Clear();
        }

        // What its records were made into goes (near copies, instances, far forms); the records and the live entities stay
        private static void Unslice(Owner o)
        {
            foreach (var slice in o.Slices)
            {
                slice.Gone = true;
                if (slice.Spawned != null)
                {
                    KeepState(slice);   // a re-run that places them again finds it under the same ids
                    s_spawned.Remove(slice.Spawned);
                    if (!slice.Spawned.IsBeingDestroyed) slice.Spawned.Destroy();
                }
                slice.Spawned = null;
                foreach (var g in slice.Groups) RemoveInstances(g);
                Active.Remove(slice);
                var cell = slice.Cell;
                cell.Slices.Remove(slice);
                if (cell.Slices.Count == 0) s_cells.Remove(cell.Key);
            }
            o.Slices.Clear();
            o.LongSlices.Clear();
            DropFar(o);
            Forest.Remove(o);
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

        /// <summary>
        /// How far from the camera (to the cell's centre, level) a group's instances are wanted: its kind's ring and band,
        /// or where the GPU stops drawing the largest of them if that is nearer (<paramref name="cullFactor"/> =
        /// LODScale × √FoVFactor of the camera; the cell's half diagonal added, since the GPU measures from the object).
        /// </summary>
        internal static float RingOf(Group g, float cullFactor)
        {
            var kind = g.Slice.Owner.Kinds[g.Kind];
            if (kind.NoFar)   // where the GPU stops drawing the largest of them, with the camera as it is now
                return cullFactor > 0f ? kind.CullTerm * g.MaxScale * cullFactor + 0.71f * CellSize : kind.Ring;
            float ring = FarTier.FromOf(kind.Profile) + FarTier.BandOf(kind.Profile);
            if (kind.CullTerm > 0f && cullFactor > 0f)
                ring = MathF.Min(ring, kind.CullTerm * g.MaxScale * cullFactor + 0.71f * CellSize);
            return ring;
        }

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

        /// <summary>A group's instances: each placement's LOD chains as its MeshRenderers draw them, a transform slot per
        /// renderer, no entity; flagged as the far tier's near side of the kind's profile, so they give way to the far
        /// form at its ring.</summary>
        internal static int AddInstances(Group g)
        {
            if (g.Draws != null || g.Slice.Gone) return 0;
            var o = g.Slice.Owner;
            var kind = o.Kinds[g.Kind];
            var renderers = PrefabShapes.RenderersOf(kind.Prefab);
            g.Draws = new DrawGroup();
            g.Slots = new List<int>(g.Records.Count * renderers.Length);
            int draws = 0;
            foreach (int i in g.Records)
            {
                var world = LocalOf(o.Records[i]) * o.ToWorld;
                foreach (var rd in renderers)
                {
                    if (!rd.Enabled || rd.Chains.Length == 0) continue;
                    int slot = TransformBuffer.Instance.AllocateSlot();
                    TransformBuffer.Instance.SetTransform(slot, rd.ToRoot * world);
                    g.Slots.Add(slot);
                    foreach (var (part, own) in rd.Chains)
                    {
                        var material = kind.Swap?.Map(own) ?? own;
                        CommandBuffer.AddPersistent(g.Draws, rd.Mesh, part, material, NoParams, slot, lodManaged: true,
                            instanceSlot: InstanceSlotFor(material), farTier: !kind.NoFar, farProfile: kind.Profile);
                        draws++;
                    }
                }
            }
            g.DrawCount = draws;
            o.InstanceDraws += draws;
            return draws;
        }

        internal static void RemoveInstances(Group g)
        {
            if (g.Draws == null) return;
            CommandBuffer.RemovePersistent(g.Draws);
            foreach (int slot in g.Slots!) TransformBuffer.Instance.ReleaseSlot(slot);
            g.Slice.Owner.InstanceDraws -= g.DrawCount;
            g.DrawCount = 0;
            g.Draws = null;
            g.Slots = null;
            g.DropInstancesIn = -1;
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
                var kind = o.Kinds[r.Kind];
                var e = kind.Prefab.Instantiate();
                if (e == null) continue;
                e.Transform.Parent = root.Transform;
                e.Transform.Position = r.Position;
                e.Transform.Rotation = r.Rotation;
                e.Transform.Scale = r.Scale;
                kind.Swap?.ApplyTo(e);
                if (kind.Stateful)
                {
                    (slice.Stateful ??= new()).Add((e, i));
                    LoadState(o, e, i);
                }
            }
            slice.Spawned = root;
            s_spawned.Add(root);
        }

        /// <summary>Its groups' instances back first (those the camera's ring wants), then its entities go. Returns the draws made.</summary>
        internal static int Despawn(Slice slice)
        {
            if (slice.Spawned == null) return 0;
            int draws = 0;
            foreach (var g in slice.Groups)
            {
                g.DropInstancesIn = -1;
                if (g.InRing && g.Draws == null) draws += AddInstances(g);
            }
            KeepState(slice);
            s_spawned.Remove(slice.Spawned);
            if (!slice.Spawned.IsBeingDestroyed) slice.Spawned.Destroy();
            slice.Spawned = null;
            return draws;
        }

        // The near cells' containers: what is under them is a copy of records, which readers of the scene take from the records
        private static readonly HashSet<Entity> s_spawned = new();

        /// <summary>
        /// Whether an entity is a near cell's copy of records (under a "Cell x,z" container): the navmesh bake and the
        /// PCG nodes that read generated output take the records instead, wherever the viewers are, and skip the copies so
        /// nothing counts twice.
        /// </summary>
        public static bool IsCellCopy(Entity? entity)
        {
            if (s_spawned.Count == 0) return false;
            for (var t = entity?.Transform; t != null; t = t.Parent)
                if (t.Entity != null && s_spawned.Contains(t.Entity)) return true;
            return false;
        }

        /// <summary>
        /// Every renderer of every record of every owner (by UID, records in order), with the world matrix its entity would
        /// have to the bit (<see cref="PrefabShapes.Renderer.WorldOf"/>): what the navmesh bake takes in place of entities.
        /// </summary>
        internal static void ForEachRenderer(Action<PrefabShapes.Renderer, Matrix4x4> visit)
        {
            foreach (var o in s_owners.Values.OrderBy(o => o.Uid))
                ForEachRenderer(o, visit);
        }

        /// <summary>Every renderer of an owner's records with its world matrix.</summary>
        internal static void ForEachRenderer(Owner o, Action<PrefabShapes.Renderer, Matrix4x4> visit)
        {
            if (o.Open) return;   // still placing: its records are not in cells yet
            foreach (var r in o.Records)
            {
                var root = LocalOf(r) * o.ToWorld;
                foreach (var rd in PrefabShapes.RenderersOf(o.Kinds[r.Kind].Prefab))
                    visit(rd, rd.WorldOf(root));
            }
        }

        /// <summary>
        /// For tools that compare scenes (scene_fingerprint): every enabled renderer of every record, with its owner and
        /// the world matrix its entity would have. Together with the entities that are not near cells' copies
        /// (<see cref="IsCellCopy"/>), it is the scene as it would be with every placement an entity.
        /// </summary>
        public static void VisitRecords(Action<Component, Mesh, Matrix4x4> visit)
        {
            foreach (var o in s_owners.Values.OrderBy(o => o.Uid))
                ForEachRenderer(o, (rd, world) =>
                {
                    if (rd.Enabled) visit(o.Component, rd.Mesh, world);
                });
        }

        /// <summary>Every renderer of the records of the owners <paramref name="takes"/> agrees to, with its world matrix.</summary>
        internal static void ForEachRenderer(Func<Component, bool> takes, Action<PrefabShapes.Renderer, Matrix4x4> visit)
        {
            foreach (var o in s_owners.Values.OrderBy(o => o.Uid))
                if (takes(o.Component)) ForEachRenderer(o, visit);
        }

        /// <summary>The records' renderers of a component that keeps its placements here (none if it does not).</summary>
        internal static void ForEachRenderer(Component owner, Action<PrefabShapes.Renderer, Matrix4x4> visit)
        {
            if (owner != null && s_owners.TryGetValue(owner.UID, out var o)) ForEachRenderer(o, visit);
        }

        /// <summary>
        /// The renderers of the records standing in the cells that reach into a ground box (a cell's margin round it, for
        /// pieces wider than their cell), of the owners <paramref name="takes"/> agrees to, with their world matrices.
        /// </summary>
        internal static void ForEachRendererIn(Vector2 min, Vector2 max, Func<Component, bool> takes, Action<PrefabShapes.Renderer, Matrix4x4> visit)
        {
            if (s_cells.Count == 0) return;
            var (x0, z0) = KeyOf(min.X, min.Y);
            var (x1, z1) = KeyOf(max.X, max.Y);
            x0 = Math.Max(x0 - 1, KeyMin.X); z0 = Math.Max(z0 - 1, KeyMin.Z);
            x1 = Math.Min(x1 + 1, KeyMax.X); z1 = Math.Min(z1 + 1, KeyMax.Z);
            if ((long)(x1 - x0 + 1) * (z1 - z0 + 1) > s_cells.Count)
            {
                // A box wider than the map's cells: every owner's records
                foreach (var o in s_owners.Values.OrderBy(o => o.Uid))
                    if (!o.Open && takes(o.Component)) ForEachRenderer(o, visit);
                return;
            }
            for (int kz = z0; kz <= z1; kz++)
                for (int kx = x0; kx <= x1; kx++)
                {
                    if (!s_cells.TryGetValue((kx, kz), out var cell)) continue;
                    foreach (var slice in cell.Slices)
                    {
                        var o = slice.Owner;
                        if (o.Open || !takes(o.Component)) continue;
                        foreach (int i in slice.Records)
                        {
                            var r = o.Records[i];
                            var root = LocalOf(r) * o.ToWorld;
                            foreach (var rd in PrefabShapes.RenderersOf(o.Kinds[r.Kind].Prefab))
                                visit(rd, rd.WorldOf(root));
                        }
                    }
                }
        }

        /// <summary>The far forest: every owner's far trees, culled and drawn once per view (<see cref="WorldCellDrawSystem"/>).</summary>
        internal static readonly FarForest Forest = new(CellSize);

        /// <summary>Every placement's cards (its coarsest LOD in gbuffer_crossmesh) of the kinds whose far form takes them
        /// (Auto, Cards) as a tree of the far forest, its kind the part, its material and the kind's profile: its world
        /// position, yaw and scales from the placement's frame and the renderer's place in the prefab.</summary>
        private static void BuildForests(Owner o, PrefabShapes.Part[][] shapes)
        {
            var kinds = new List<FarForest.Kind>();
            var kindOf = new Dictionary<FarForest.Kind, int>();
            var trees = new List<FarForest.Tree>();
            foreach (var r in o.Records)
            {
                var placed = o.Kinds[r.Kind];
                if (placed.Far is not (FarForm.Auto or FarForm.Cards)) continue;
                Matrix4x4? world = null;
                foreach (var part in shapes[r.Kind])
                {
                    if (part.Kind != PrefabShapes.Kind.Card || part.Material == null || part.Mesh == null) continue;
                    var key = new FarForest.Kind(part.Mesh, part.PartIndex, placed.Swap?.Map(part.Material) ?? part.Material, placed.Profile);
                    if (!kindOf.TryGetValue(key, out int kind))
                    {
                        kind = kinds.Count;
                        kindOf[key] = kind;
                        kinds.Add(key);
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
            if (trees.Count > 0) Forest.Set(o, kinds, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(trees));
            else Forest.Remove(o);
        }

        // ── Counts ──

        internal struct Counters
        {
            public double MaxFrameMs, SpawnMs, InstanceMs;
            public long Spawned, Despawned, SpawnedEntities, InstanceGroupsMade, InstanceGroupsDropped, InstanceDrawsMade;
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
            long records = 0, slices = 0, groups = 0, spawned = 0, entities = 0, instanceGroups = 0, instancePlacements = 0, instanceDraws = 0;
            long farMeshes = 0, farVertices = 0, farTriangles = 0, farVram = 0, live = 0;
            var liveBecause = new Dictionary<string, int>();
            foreach (var o in s_owners.Values)
            {
                live += o.Live.Count;
                foreach (var (type, n) in o.LiveBecause) liveBecause[type] = liveBecause.GetValueOrDefault(type) + n;
                records += o.Records.Count;
                slices += o.Slices.Count;
                instanceDraws += o.InstanceDraws;
                foreach (var s in o.Slices)
                {
                    groups += s.Groups.Count;
                    if (s.Spawned != null) { spawned++; entities += s.Records.Count; }
                    foreach (var g in s.Groups)
                        if (g.Draws != null) { instanceGroups++; instancePlacements += g.Records.Count; }
                }
                if (o.FarMesh != null) { farMeshes++; farVertices += o.FarVertices; farTriangles += o.FarTriangles; farVram += o.FarVram; }
            }
            var c = Stats;
            return new
            {
                owners = s_owners.Count,
                records,
                recordBytes = records * RecordBytes,
                // Placements of live prefabs, made as entities at once, and the component types that made them live
                live,
                liveBecause = liveBecause.OrderByDescending(p => p.Value).Select(p => $"{p.Key}: {p.Value}").ToArray(),
                // Records whose prefab keeps state (IRecordState), and how many of them have changed state kept (WorldState)
                withState = new { records = s_owners.Values.Sum(o => o.StatefulRecords), kept = WorldState.Count },
                cells = s_cells.Count,
                slices,
                groups,
                activeSlices = Active.Count,
                near = new { slices = spawned, placements = entities },
                instances = new { groups = instanceGroups, placements = instancePlacements, draws = instanceDraws },
                // The one far forest of every owner's trees
                farForest = new { owners = Forest.OwnerCount, trees = Forest.TreeCount, kinds = Forest.KindCount, cells = Forest.CellCount, vramBytes = Forest.VramBytes },
                farMeshes = new { meshes = farMeshes, vertices = farVertices, triangles = farTriangles, vramBytes = farVram },
                system = new
                {
                    worstFrameMs = Math.Round(c.MaxFrameMs, 3),
                    slicesSpawned = c.Spawned,
                    slicesDespawned = c.Despawned,
                    usPerEntity = Math.Round(1000 * c.SpawnMs / Math.Max(1, c.SpawnedEntities), 2),
                    instanceGroupsMade = c.InstanceGroupsMade,
                    instanceGroupsDropped = c.InstanceGroupsDropped,
                    usPerInstanceDraw = Math.Round(1000 * c.InstanceMs / Math.Max(1, c.InstanceDrawsMade), 3),
                },
                farTier = new { from = FarTier.From, band = FarTier.Band, cellSize = FarTier.CellSize, profiles = FarTier.ProfileCount },
                byOwner = s_owners.Values.OrderByDescending(o => o.Records.Count).Take(40).Select(o => new
                {
                    name = o.Name,
                    records = o.Records.Count,
                    live = o.Live.Count,
                    slices = o.Slices.Count,
                    far = o.Far.ToString(),
                    ring = o.Ring > 0f ? o.Ring : FarTier.From,
                    kinds = o.Kinds.Count,
                    instancedPlacements = o.Slices.Sum(s => s.Groups.Where(g => g.Draws != null).Sum(g => g.Records.Count)),
                    instanceDraws = o.InstanceDraws,
                    forestTrees = Forest.TreesOf(o),
                    farTriangles = o.FarTriangles,
                    farVramBytes = o.FarVram,
                    farBuildMs = Math.Round(o.FarBuildMs),
                }).ToArray(),
                // The kinds of the five largest owners: how far their instances reach with the camera as it is (the kind's
                // ring, or where the GPU stops drawing its largest placement), and how many of them are instanced now
                kinds = s_owners.Values.OrderByDescending(o => o.Records.Count).Take(5).SelectMany(o => o.Kinds.Select((k, i) =>
                {
                    float cullFactor = Camera.Main is { } cam ? Engine.Settings.LODScale * MathF.Sqrt(MathF.Max(cam.FoVFactor, 0f)) : 0f;
                    int count = 0, instanced = 0;
                    float maxScale = 0f;
                    foreach (var s in o.Slices)
                        foreach (var g in s.Groups)
                            if (g.Kind == i)
                            {
                                count += g.Records.Count;
                                if (g.Draws != null) instanced += g.Records.Count;
                                maxScale = MathF.Max(maxScale, g.MaxScale);
                            }
                    return new
                    {
                        owner = o.Name,
                        prefab = k.Prefab.Name,
                        records = count,
                        instanced,
                        ring = k.NoFar ? k.Ring : FarTier.FromOf(k.Profile),
                        coarse = k.Coarse,          // the ring is where its coarsest LOD starts (Auto)
                        noFar = k.NoFar,            // no far form: drawn until the GPU drops it, the ring is where that is
                        cullsAt = Math.Round(k.CullTerm * maxScale * cullFactor),
                        far = k.Far.ToString(),
                    };
                })).ToArray(),
            };
        }
    }
}
