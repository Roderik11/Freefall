using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Components;

namespace Freefall.Base
{
    /// <summary>How near a viewer something is (<see cref="Vicinity"/>).</summary>
    public enum Ring : byte
    {
        /// <summary>Beyond every viewer's middle ring.</summary>
        Far,

        /// <summary>Within a viewer's middle ring: near enough to be seen.</summary>
        Middle,

        /// <summary>Within a viewer's near ring: near enough for everything it can do.</summary>
        Near,
    }

    /// <summary>Something that wants to know how near a viewer it is (<see cref="Vicinity.Register"/>).</summary>
    public interface IVicinity
    {
        /// <summary>Its ring changed. Main thread, from the Vicinity's update; keep it short (queue the work).</summary>
        void RingChanged(Vicinity.Entry entry, Ring ring);
    }

    /// <summary>
    /// Rings around the viewers: what is near enough to be seen (middle) or near enough for everything it can do (near),
    /// for things that come and go as the player moves (NPCs, NPC_POPULATION_PLAN). Each registered thing has radii of
    /// its own and is told when its ring changes. It leaves a ring only a little further out than it enters it
    /// (<see cref="Slack"/>), so it does not flicker at the edge.
    ///
    /// Things are kept in a grid of <see cref="CellSize"/> cells by where they are; whoever moves one says so
    /// (<see cref="Move"/>), and only a crossing into another cell does anything. Every <see cref="EvaluateEvery"/>
    /// seconds the cells round the viewers (out to the widest middle ring) and the things already in a ring are
    /// looked at again, in squared distances: nothing further away is touched, however many there are.
    ///
    /// The viewers are the transforms game code adds (the player's character; every player's on a server). With none,
    /// the main camera is the viewer, which in the editor is the scene camera: authors see what is near where they look.
    ///
    /// Things are game objects: whoever registers them unregisters them before its code goes away (a script system in
    /// its Destroy), or the engine keeps the old script assembly alive.
    /// </summary>
    public static class Vicinity
    {
        /// <summary>Metres a side of a cell of the grid things are found by.</summary>
        public const float CellSize = 64f;

        /// <summary>Seconds between two looks at the rings.</summary>
        public const float EvaluateEvery = 0.25f;

        /// <summary>A ring is left this many times further out than it is entered.</summary>
        public const float Slack = 1.15f;

        /// <summary>A registered thing: where it is, its radii and its ring now.</summary>
        public sealed class Entry
        {
            public IVicinity Item { get; internal set; } = null!;
            public Vector3 Position { get; internal set; }
            public Ring Ring { get; internal set; }
            public float Near { get; internal set; }
            public float Middle { get; internal set; }
            internal float NearIn, NearOut, MiddleIn, MiddleOut;   // squared
            internal long Cell;
            internal int ActiveIndex = -1;
            internal int Looked;
            internal bool Registered;
        }

        private static readonly Dictionary<long, List<Entry>> s_cells = new();
        private static readonly List<Entry> s_active = new();          // in a ring other than Far
        private static readonly List<Transform> s_viewers = new();
        private static readonly List<Vector3> s_at = new();
        private static readonly List<(Entry Entry, Ring Ring)> s_changes = new();
        private static float s_widest;                                  // the widest middle ring's way out
        private static float s_sinceLook = EvaluateEvery;
        private static int s_look;
        private static int s_count;

        /// <summary>Things registered.</summary>
        public static int Count => s_count;

        /// <summary>Things in the middle or near ring.</summary>
        public static int Active => s_active.Count;

        /// <summary>
        /// The viewers game code has added; empty means the main camera.
        /// </summary>
        public static IReadOnlyList<Transform> Viewers => s_viewers;

        public static void AddViewer(Transform viewer)
        {
            if (viewer != null && !s_viewers.Contains(viewer)) s_viewers.Add(viewer);
            s_sinceLook = EvaluateEvery;
        }

        public static void RemoveViewer(Transform viewer)
        {
            s_viewers.Remove(viewer);
            s_sinceLook = EvaluateEvery;
        }

        /// <summary>
        /// Register a thing at a place with a near and a middle radius (metres; near no wider than middle). It starts
        /// Far and is told its ring at the next look.
        /// </summary>
        public static Entry Register(IVicinity item, Vector3 position, float near, float middle)
        {
            var entry = new Entry { Item = item, Position = position, Registered = true };
            SetRadii(entry, near, middle);
            entry.Cell = CellOf(position);
            CellList(entry.Cell).Add(entry);
            s_count++;
            s_sinceLook = EvaluateEvery;
            return entry;
        }

        /// <summary>Another near and middle radius for a registered thing.</summary>
        public static void SetRadii(Entry entry, float near, float middle)
        {
            middle = MathF.Max(middle, 0f);
            near = Math.Clamp(near, 0f, middle);
            entry.Near = near;
            entry.Middle = middle;
            entry.NearIn = near * near;
            entry.NearOut = near * near * Slack * Slack;
            entry.MiddleIn = middle * middle;
            entry.MiddleOut = middle * middle * Slack * Slack;
            s_widest = MathF.Max(s_widest, middle * Slack);
        }

        /// <summary>The thing is somewhere else now. Only a crossing into another cell costs anything.</summary>
        public static void Move(Entry entry, Vector3 position)
        {
            if (!entry.Registered) return;
            entry.Position = position;
            long cell = CellOf(position);
            if (cell == entry.Cell) return;
            if (s_cells.TryGetValue(entry.Cell, out var from)) from.Remove(entry);
            entry.Cell = cell;
            CellList(cell).Add(entry);
        }

        /// <summary>Forget a thing. It is not told it went Far.</summary>
        public static void Unregister(Entry entry)
        {
            if (entry == null || !entry.Registered) return;
            entry.Registered = false;
            if (s_cells.TryGetValue(entry.Cell, out var list)) list.Remove(entry);
            if (entry.ActiveIndex >= 0) RemoveActive(entry);
            entry.Ring = Ring.Far;
            s_count--;
        }

        /// <summary>Look at the rings at the next update, not after the usual wait (a viewer was put somewhere else).</summary>
        public static void LookSoon() => s_sinceLook = EvaluateEvery;

        /// <summary>Called by <see cref="VicinitySystem"/> every frame.</summary>
        internal static void Update()
        {
            s_sinceLook += Time.Delta;
            if (s_sinceLook < EvaluateEvery || s_count == 0) return;
            s_sinceLook = 0f;
            Look();
        }

        private static void Look()
        {
            s_at.Clear();
            for (int i = s_viewers.Count - 1; i >= 0; i--)
            {
                var viewer = s_viewers[i];
                if (viewer == null || viewer.IsDestroyed) { s_viewers.RemoveAt(i); continue; }
                s_at.Add(viewer.WorldPosition);
            }
            if (s_at.Count == 0 && Camera.Main?.Transform != null)
                s_at.Add(Camera.Main.Transform.WorldPosition);

            s_look++;
            s_changes.Clear();

            // What is in a ring already: it may have left it (backwards: one that leaves is swapped with the last)
            for (int i = s_active.Count - 1; i >= 0; i--)
                Judge(s_active[i]);

            // The cells round each viewer, out to the widest middle ring
            int reach = (int)MathF.Ceiling(s_widest / CellSize);
            foreach (var at in s_at)
            {
                int cx = (int)MathF.Floor(at.X / CellSize), cz = (int)MathF.Floor(at.Z / CellSize);
                for (int x = cx - reach; x <= cx + reach; x++)
                    for (int z = cz - reach; z <= cz + reach; z++)
                    {
                        if (!s_cells.TryGetValue(Key(x, z), out var list)) continue;
                        for (int i = 0; i < list.Count; i++) Judge(list[i]);
                    }
            }

            // Told after the looking, so that what they do (register, move, unregister) does not upset it
            foreach (var (entry, ring) in s_changes)
            {
                if (!entry.Registered || entry.Ring != ring) continue;
                try { entry.Item.RingChanged(entry, ring); }
                catch (Exception e) { CallbackErrors.Report(entry.Item, nameof(IVicinity.RingChanged), e); }
            }
            s_changes.Clear();
        }

        // Its ring by the nearest viewer, with slack on the way out
        private static void Judge(Entry entry)
        {
            if (entry.Looked == s_look || !entry.Registered) return;
            entry.Looked = s_look;

            float nearest = float.MaxValue;
            var p = entry.Position;
            foreach (var at in s_at)
            {
                float dx = p.X - at.X, dz = p.Z - at.Z;
                nearest = MathF.Min(nearest, dx * dx + dz * dz);
            }

            var was = entry.Ring;
            Ring ring;
            if (nearest <= entry.NearIn || (was == Ring.Near && nearest <= entry.NearOut)) ring = Ring.Near;
            else if (nearest <= entry.MiddleIn || (was != Ring.Far && nearest <= entry.MiddleOut)) ring = Ring.Middle;
            else ring = Ring.Far;
            if (ring == was) return;

            entry.Ring = ring;
            if (ring == Ring.Far) RemoveActive(entry);
            else if (entry.ActiveIndex < 0)
            {
                entry.ActiveIndex = s_active.Count;
                s_active.Add(entry);
            }
            s_changes.Add((entry, ring));
        }

        private static void RemoveActive(Entry entry)
        {
            int index = entry.ActiveIndex, last = s_active.Count - 1;
            if (index < 0) return;
            var moved = s_active[last];
            s_active[index] = moved;
            moved.ActiveIndex = index;
            s_active.RemoveAt(last);
            entry.ActiveIndex = -1;
        }

        private static List<Entry> CellList(long cell)
        {
            if (!s_cells.TryGetValue(cell, out var list)) s_cells[cell] = list = new List<Entry>();
            return list;
        }

        private static long CellOf(Vector3 p) => Key((int)MathF.Floor(p.X / CellSize), (int)MathF.Floor(p.Z / CellSize));

        private static long Key(int x, int z) => ((long)x << 32) | (uint)z;
    }

    /// <summary>Looks at the <see cref="Vicinity"/> rings, before the scripts, in the editor as in play.</summary>
    [UpdateInEditor]
    [UpdateBefore(typeof(ScriptGroup))]
    public sealed class VicinitySystem : EntitySystem
    {
        public override void Update() => Vicinity.Update();
    }
}
