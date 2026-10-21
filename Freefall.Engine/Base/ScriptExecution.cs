using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Freefall.Components;

namespace Freefall.Base
{
    public static class ScriptExecution
    {
        internal static List<IComponentCache> list = new List<IComponentCache>();

        private static readonly ProfilerMarker _wakeUpMarker = ProfilerMarker.Get("Wake-up");

        internal static void Add(IComponentCache cache)
        {
            list.Add(cache);
        }

        /// <summary>Components of every type waiting for their Early/Awake (main thread).</summary>
        internal static int PendingWakeups()
        {
            int count = 0;
            for (int i = 0; i < list.Count; i++)
                count += list[i].PendingWakeups;
            return count;
        }

        /// <summary>
        /// Drop the caches of component types matching the predicate (script types whose assembly is being
        /// unloaded), so the dispatch list stops referencing ComponentCache&lt;OldType&gt;.
        /// Main thread only, never from inside Update/Draw.
        /// </summary>
        internal static int RemoveCaches(Func<Type, bool> predicate)
        {
            return list.RemoveAll(c => predicate(c.ComponentType));
        }

        /// <summary>
        /// Early()/Awake() for components added since the last frame, plus the caches' Added/Removed
        /// notifications. Runs ahead of the UpdateGroup.
        /// </summary>
        public static void WakeUp()
        {
            using var _ = _wakeUpMarker.Auto();
            Pass(long.MaxValue);
        }

        /// <summary>
        /// A scene load's wake-up slice: passes until nothing waits or <paramref name="deadline"/> (a Stopwatch
        /// timestamp) has passed, also for what the woken components add (the timber kit's pieces). Every component
        /// still gets Early() before any Awake() of its pass. True when nothing waits any more.
        /// </summary>
        internal static bool WakeUp(long deadline)
        {
            using var _ = _wakeUpMarker.Auto();
            while (PendingWakeups() > 0)
            {
                Pass(deadline);
                if (System.Diagnostics.Stopwatch.GetTimestamp() >= deadline) break;
            }
            return PendingWakeups() == 0;
        }

        // Early() for every type, then Awake(); only the deadline stops it early
        private static void Pass(long deadline)
        {
            for (int i = 0; i < list.Count; i++)
                if (!list[i].Early(deadline)) return;

            for (int i = 0; i < list.Count; i++)
            {
                list[i].Awake(deadline);
                if (System.Diagnostics.Stopwatch.GetTimestamp() >= deadline) return;
            }
        }

        /// <summary>IUpdate dispatch; called by <see cref="ScriptUpdateSystem"/>.</summary>
        public static void UpdateComponents()
        {
            for (int i = 0; i < list.Count; i++)
                list[i].Update();
        }

        /// <summary>IDraw dispatch; called by <see cref="ScriptDrawSystem"/>.</summary>
        public static void Draw()
        {
            for (int i = 0; i < list.Count; i++)
            {
                list[i].Draw();
            }
        }
    }
}
