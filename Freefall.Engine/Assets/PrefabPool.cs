using System.Collections.Generic;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Assets
{
    /// <summary>
    /// Bodies of one prefab, kept for reuse instead of destroyed: for things that come and go as a player moves (NPCs
    /// near a viewer, NPC_POPULATION_PLAN). A body is built once. A person built afresh costs about 1.1 ms on its
    /// first pose alone (retargeting set up, bone buffers created); taken from the pool it costs neither, since its
    /// Animator stays as it was.
    ///
    /// <see cref="Return"/> puts the body to sleep (<see cref="Dormancy"/>: its <see cref="IDormant"/> components
    /// told, every enabled component disabled) and hangs it under a hidden root that is not saved. <see cref="Rent"/>
    /// puts it under its new parent at its new place and wakes it. A new body comes from
    /// <see cref="Prefab.Instantiate"/>; its components wake in the next wake-up step as usual. A scene load destroys
    /// the root with everything else, and every pool forgets its bodies. A component whose registration lies outside
    /// the engine's own (a PhysX capsule a script made) has to give it up in <see cref="IDormant.OnSleep"/>.
    /// </summary>
    public sealed class PrefabPool
    {
        private static readonly Dictionary<Prefab, PrefabPool> s_pools = new();
        private static Entity? s_root;
        private static bool s_listening;

        private readonly Prefab _prefab;
        private readonly Stack<Dormancy> _free = new();

        private PrefabPool(Prefab prefab) => _prefab = prefab;

        /// <summary>The pool of a prefab's bodies (one per prefab).</summary>
        public static PrefabPool For(Prefab prefab)
        {
            if (!s_listening)
            {
                s_listening = true;
                SceneManager.LoadStarted += _ => Forget();
            }
            if (!s_pools.TryGetValue(prefab, out var pool))
                s_pools[prefab] = pool = new PrefabPool(prefab);
            return pool;
        }

        /// <summary>Bodies waiting in the pool.</summary>
        public int Free => _free.Count;

        /// <summary>Bodies out of the pool.</summary>
        public int Rented { get; private set; }

        /// <summary>Bodies waiting in every pool, and out of every pool.</summary>
        public static (int Free, int Rented) Totals()
        {
            int free = 0, rented = 0;
            foreach (var pool in s_pools.Values)
            {
                free += pool.Free;
                rented += pool.Rented;
            }
            return (free, rented);
        }

        /// <summary>
        /// A body: one returned earlier, or a new instance. Under <paramref name="parent"/> (null: at the top), at a
        /// place and turn in that parent's space.
        /// </summary>
        public Entity? Rent(Transform? parent, Vector3 position, Quaternion rotation)
        {
            while (_free.Count > 0)
            {
                var asleep = _free.Pop();
                var body = asleep.Root;
                if (body?.Transform == null || body.Transform.IsDestroyed) continue;

                Place(body, parent, position, rotation);
                asleep.Wake();
                Rented++;
                return body;
            }

            var made = _prefab.Instantiate();
            if (made == null) return null;
            Place(made, parent, position, rotation);
            Rented++;
            return made;
        }

        /// <summary>Put a body that <see cref="Rent"/> gave back in the pool.</summary>
        public void Return(Entity body)
        {
            if (body?.Transform == null || body.Transform.IsDestroyed) return;
            Rented--;
            var asleep = Dormancy.Sleep(body);
            body.Transform.Parent = Root().Transform;
            _free.Push(asleep);
        }

        /// <summary>Destroy the bodies waiting in this pool.</summary>
        public void Clear()
        {
            while (_free.Count > 0)
            {
                var body = _free.Pop().Root;
                if (body?.Transform != null && !body.Transform.IsDestroyed) body.Destroy();
            }
        }

        // A scene load destroys the root and every body with it
        private static void Forget()
        {
            foreach (var pool in s_pools.Values)
            {
                pool._free.Clear();
                pool.Rented = 0;
            }
            s_root = null;
        }

        private static Entity Root()
        {
            if (s_root == null || s_root.Transform == null || s_root.Transform.IsDestroyed)
                s_root = new Entity("Prefab Pool") { Flags = EntityFlags.HideAndDontSave };
            return s_root;
        }

        private static void Place(Entity body, Transform? parent, Vector3 position, Quaternion rotation)
        {
            body.Transform.Parent = parent;
            body.Transform.Position = position;
            body.Transform.Rotation = rotation;
        }
    }
}
