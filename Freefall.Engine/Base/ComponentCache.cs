using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;
using Freefall.Components;

namespace Freefall.Base
{
    public interface IComponentCache
    {
        Type ComponentType { get; }
        /// <summary>Components added since the last wake-up, waiting for their Early/Awake.</summary>
        int PendingWakeups { get; }
        /// <summary>Early() for the waiting components, until <paramref name="deadline"/> (a Stopwatch timestamp). True when none is left.</summary>
        bool Early(long deadline);
        /// <summary>The Removed/Added events, and Awake() for the waiting components that have had Early(), until <paramref name="deadline"/>. True when none is left.</summary>
        bool Awake(long deadline);
        void Update();
        void Draw();
    }

    public class ComponentCache<T> : IComponentCache where T : Component
    {
        public static readonly Type Type = typeof(T);
        internal static readonly List<T> List = [];
        public static IReadOnlyList<T> All => List;
        internal static readonly ComponentSet<IUpdate> UpdateList = new();
        internal static readonly ComponentSet<IDraw> DrawList = new();

        internal static readonly List<T> WakeupList = [];
        private static readonly List<T> RemovedList = [];

        /// <summary>
        /// Raised on the main thread in the wake-up step (ahead of the UpdateGroup), once per component after
        /// its Awake(). For systems that keep state per component. A system created later than the
        /// components it cares about walks <see cref="All"/> in Initialize(); it can then see a component
        /// there first and get Added for it afterwards, so its handler must tolerate that.
        /// A subscriber from a script assembly must unsubscribe in its Destroy().
        /// </summary>
        public static event Action<T>? Added;

        /// <summary>
        /// Raised in the same step for every component that left the cache after Added was raised for it
        /// (destroyed, removed, or detached to move to another entity; the latter is announced again).
        /// </summary>
        public static event Action<T>? Removed;

        private static readonly Dictionary<int, int> Indices = [];
        private static readonly Lock _lock = new ();

        private bool IsParallel;
        private bool HasUpdate;
        private bool HasDraw;

        // One timing per loop over this type's components (Update, Draw, wake-up), not per component
        private readonly ProfilerMarker _marker = ProfilerMarker.Get(Type.Name);

        public Type ComponentType => Type;

        public int PendingWakeups => WakeupList.Count - _awakeCursor;

        // How far Early() and Awake() got through WakeupList. A wake-up can stop at a deadline (a scene load wakes the
        // scene a slice per frame) and carry on from here; Awake() never passes Early().
        private static int _earlyCursor, _awakeCursor;

        static ComponentCache()
        {
            bool hasUpdate = typeof(IUpdate).IsAssignableFrom(Type);
            bool hasDraw = typeof(IDraw).IsAssignableFrom(Type);
            bool isParallel = typeof(IParallel).IsAssignableFrom(Type);

            // Always register — even pure-data components (no IUpdate/IDraw)
            // need the wakeup step so their Awake() fires.
            ScriptExecution.Add(new ComponentCache<T>
            {
                IsParallel = isParallel,
                HasUpdate = hasUpdate,
                HasDraw = hasDraw
            });
        }

        static void FastRemove(int index)
        {
            int last = List.Count - 1;
            if (index > last) return;

            var a = List[index];
            var b = List[last];

            Indices[b.Entity.Id] = index;
            Indices.Remove(a.Entity.Id);

            List[index] = b;
            List[last] = a;

            List.RemoveAt(last);

            if (a is IUpdate update) UpdateList.Remove(update);
            if (a is IDraw draw) DrawList.Remove(draw);

            if (a.Announced)
            {
                a.Announced = false;
                if (Removed != null) RemovedList.Add(a);
            }
        }

        public static T? Get(Entity entity)
        {
            lock (_lock)
            {
                if (Indices.TryGetValue(entity.Id, out int result))
                    return List[result];

                return null;
            }
        }

        public static bool Remove(Entity entity)
        {
            lock (_lock)
            {
                if (!Indices.TryGetValue(entity.Id, out int result))
                    return false;

                FastRemove(result);
                return true;
            }
        }

        public static T Add(Entity entity, T component)
        {
            lock (_lock)
            {
                int id = entity.Id;
                if (Indices.TryGetValue(id, out int result))
                    return List[result];

                if (component is IUpdate update)
                    UpdateList.Add(update);

                if (component is IDraw draw)
                    DrawList.Add(draw);

                Indices.Add(id, List.Count);
                component.Entity = entity;
                List.Add(component);
                WakeupList.Add(component);

                return component;
            }
        }

        public bool Early(long deadline)
        {
            lock (_lock)
            {
                if (_earlyCursor >= WakeupList.Count) return true;
                using var _ = _marker.Auto(WakeupList.Count - _earlyCursor);
                while (_earlyCursor < WakeupList.Count)
                {
                    WakeupList[_earlyCursor++].EarlyBird();
                    if (deadline != long.MaxValue && System.Diagnostics.Stopwatch.GetTimestamp() >= deadline) break;
                }
                return _earlyCursor >= WakeupList.Count;
            }
        }

        public bool Awake(long deadline)
        {
            lock (_lock)
            {
                if (RemovedList.Count == 0 && _awakeCursor >= _earlyCursor) return _awakeCursor >= WakeupList.Count;
                using var _ = _marker.Auto(_earlyCursor - _awakeCursor);

                if (RemovedList.Count > 0)
                {
                    foreach (var component in RemovedList)
                    {
                        try { Removed?.Invoke(component); }
                        catch (Exception e) { CallbackErrors.Report(component, $"{Type.Name} cache Removed handler", e); }
                    }

                    RemovedList.Clear();
                }

                // Only those that have had Early(): one added meanwhile (by another component's Awake) gets both later
                while (_awakeCursor < _earlyCursor)
                {
                    var component = WakeupList[_awakeCursor++];
                    component.WakeUp();

                    if (!component.IsDestroyed && !component.Announced)
                    {
                        component.Announced = true;
                        try { Added?.Invoke(component); }
                        catch (Exception e) { CallbackErrors.Report(component, $"{Type.Name} cache Added handler", e); }
                    }
                    if (deadline != long.MaxValue && System.Diagnostics.Stopwatch.GetTimestamp() >= deadline) break;
                }

                // Drop the woken ones; the cursors keep pointing at the same components
                if (_awakeCursor > 0)
                {
                    WakeupList.RemoveRange(0, _awakeCursor);
                    _earlyCursor -= _awakeCursor;
                    _awakeCursor = 0;
                }
                return WakeupList.Count == 0;
            }
        }

        public void Update()
        {
            if (!HasUpdate) return;

            lock (_lock)
            {
                if (UpdateList.Count == 0) return;
                using var _ = _marker.Auto(UpdateList.Count);

                // What one throws is reported and the rest still update (CallbackErrors). The outer catch is for the
                // list itself: a component that destroys its own entity in Update changes it under the foreach, and
                // the rest of this type then waits for the next frame.
                // A disabled component does not update (a body put away in a PrefabPool, NPC_POPULATION_PLAN)
                try
                {
                    if (IsParallel)
                        Parallel.ForEach(UpdateList.Collection, comp =>
                        {
                            if (comp is Component { Enabled: false }) return;
                            try { comp.Update(); }
                            catch (Exception e) { CallbackErrors.Report(comp, nameof(IUpdate.Update), e); }
                        });
                    else
                        foreach (var comp in UpdateList.Collection)
                        {
                            if (comp is Component { Enabled: false }) continue;
                            try { comp.Update(); }
                            catch (Exception e) { CallbackErrors.Report(comp, nameof(IUpdate.Update), e); }
                        }
                }
                catch (Exception e) { CallbackErrors.Report(null, $"{Type.Name} components' Update loop", e); }
            }
        }

        public void Draw()
        {
            if (!HasDraw || DrawList.Count == 0) return;
            using var _ = _marker.Auto(DrawList.Count);

            try
            {
                if (IsParallel)
                    Parallel.ForEach(DrawList.Collection, comp =>
                    {
                        if (comp is Component { Enabled: false }) return;
                        try { comp.Draw(); }
                        catch (Exception e) { CallbackErrors.Report(comp, nameof(IDraw.Draw), e); }
                    });
                else
                    foreach (var comp in DrawList.Collection)
                    {
                        if (comp is Component { Enabled: false }) continue;
                        try { comp.Draw(); }
                        catch (Exception e) { CallbackErrors.Report(comp, nameof(IDraw.Draw), e); }
                    }
            }
            catch (Exception e) { CallbackErrors.Report(null, $"{Type.Name} components' Draw loop", e); }
        }

        public override string ToString()
        {
            return typeof(T).Name;
        }
    }

}
