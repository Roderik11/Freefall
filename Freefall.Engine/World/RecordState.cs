using System;
using System.Collections;
using System.Collections.Generic;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Serialization;

namespace Freefall.World
{
    /// <summary>
    /// A component with state of its own that may sit on a record's prefab (WORLD_CELLS_PLAN.md stage 5): a worked tree,
    /// later a door, a looted chest. Such a prefab is still a record (<see cref="PrefabShapes.IsRecordComponent"/>), with an
    /// id (<see cref="WorldCells"/>: its owner, prefab and position); while its cell is away its state is kept in
    /// <see cref="WorldState"/>, only if it changed, and a copy made again reads it back.
    /// <para>Time is game time (<see cref="Time.TotalTime"/>), resolved by whoever reads it: nothing runs while the cell is
    /// away, so a component keeps when something happened and works out what that means now (a tree full again after
    /// its refill time).</para>
    /// </summary>
    public interface IRecordState
    {
        /// <summary>It differs from what its prefab makes: worth keeping while its cell is away. False drops what was kept.</summary>
        bool Changed { get; }

        /// <summary>
        /// Its state, as the copy is about to go: plain data (lists copied, no references into scene objects), so it can
        /// be written to a save one day as the player's save is (the engine's YAML).
        /// </summary>
        object Save();

        /// <summary>Its state back, on a copy of the prefab just made for its record (before the copy first updates).</summary>
        void Load(object state);
    }

    /// <summary>
    /// What the world's records keep while their cells are away: record id → the states of the copy's
    /// <see cref="IRecordState"/> components (in the copy's own order, null for one that had not changed), only for records
    /// that changed. Kept in memory: it fixes "full again after walking 200 m away", not loads; a scene load clears it.
    /// <para>Each state is kept as the engine's YAML (as the player's save writes), never as the object itself: a script
    /// reload replaces the classes the states are made of, and an object of the old class would neither be read by the new
    /// code nor let the old assembly unload. The same text is what a save would write.</para>
    /// Main thread.
    /// </summary>
    public static class WorldState
    {
        private struct Entry
        {
            public ulong Owner;        // the spawner's UID: an owner's run prunes its own entries whose records went
            public string?[] States;   // YAML, null for a component that had not changed
        }

        private static readonly Dictionary<ulong, Entry> s_entries = new();
        private static readonly Dictionary<ulong, int> s_perOwner = new();
        private static bool s_listening;

        /// <summary>Records with state kept.</summary>
        public static int Count => s_entries.Count;

        /// <summary>Records of an owner with state kept.</summary>
        public static int CountOf(ulong owner) => s_perOwner.GetValueOrDefault(owner);

        /// <summary>What a record kept, read back into objects of today's classes (null where a component had not changed).</summary>
        internal static bool TryGet(ulong id, out object?[] states)
        {
            if (!s_entries.TryGetValue(id, out var e))
            {
                states = Array.Empty<object?>();
                return false;
            }
            states = new object?[e.States.Length];
            for (int i = 0; i < states.Length; i++) states[i] = Read(e.States[i]);
            return true;
        }

        /// <summary>A record's states (null where a component had not changed), written as YAML.</summary>
        internal static void Put(ulong id, ulong owner, object?[] states)
        {
            Listen();
            var text = new string?[states.Length];
            for (int i = 0; i < states.Length; i++) text[i] = Write(states[i]);
            if (s_entries.TryGetValue(id, out var old)) Forget(old.Owner);
            s_entries[id] = new Entry { Owner = owner, States = text };
            s_perOwner[owner] = s_perOwner.GetValueOrDefault(owner) + 1;
        }

        /// <summary>A state as the engine's YAML; null for null or one that cannot be written (logged).</summary>
        internal static string? Write(object? state)
        {
            if (state == null) return null;
            try { return new YAMLSerializer().Serialize(state); }
            catch (Exception ex)
            {
                Debug.LogWarning("WorldState", $"a {state.GetType().Name} could not be written: {ex.Message}");
                return null;
            }
        }

        /// <summary>A state read back from YAML into today's classes, its asset references loaded as the player's save
        /// loads them; null for null or text that cannot be read (logged).</summary>
        internal static object? Read(string? yaml)
        {
            if (yaml == null) return null;
            try
            {
                var state = new YAMLSerializer().Deserialize(yaml, out var stubs);
                foreach (var stub in stubs)
                {
                    Asset? asset = null;
                    try { asset = Engine.Assets.LoadByGuid(stub.Guid, stub.AssetType); }
                    catch (Exception ex) { Debug.LogWarning("WorldState", $"{stub.AssetType?.Name} {stub.Guid} could not be loaded: {ex.Message}"); }
                    if (stub.Field != null) stub.Field.SetValue(stub.Parent, asset);
                    else if (stub.Parent is IList list && stub.ListIndex >= 0 && stub.ListIndex < list.Count) list[stub.ListIndex] = asset;
                }
                return state;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("WorldState", $"a kept state could not be read: {ex.Message}");
                return null;
            }
        }

        internal static void Remove(ulong id)
        {
            if (s_entries.Remove(id, out var old)) Forget(old.Owner);
        }

        /// <summary>The owner's entries whose ids are not among <paramref name="ids"/> go (their placements moved or went);
        /// returns how many.</summary>
        internal static int Prune(ulong owner, HashSet<ulong> ids)
        {
            if (CountOf(owner) == 0) return 0;
            var gone = new List<ulong>();
            foreach (var (id, e) in s_entries)
                if (e.Owner == owner && !ids.Contains(id)) gone.Add(id);
            foreach (var id in gone) Remove(id);
            return gone.Count;
        }

        /// <summary>Everything kept goes (a scene load).</summary>
        public static void Clear()
        {
            s_entries.Clear();
            s_perOwner.Clear();
        }

        private static void Forget(ulong owner)
        {
            if (s_perOwner.TryGetValue(owner, out int n))
            {
                if (n <= 1) s_perOwner.Remove(owner);
                else s_perOwner[owner] = n - 1;
            }
        }

        private static void Listen()
        {
            if (s_listening) return;
            s_listening = true;
            SceneManager.LoadStarted += _ => Clear();
        }
    }
}
