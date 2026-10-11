using System;
using System.Collections.Generic;
using System.Threading;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.PCG
{
    /// <summary>
    /// Runs PCG regeneration. Everything that wants a component to regenerate (waking up, spline, graph, terrain
    /// heights, member edits, another component's new output) calls PCGComponent.Invalidate(), which queues it
    /// here; each queued component runs at most once per frame, and not at all while an edit is in progress
    /// (gizmo drag, inspector drag, a multi-step build) or while terrain heights are still on their way.
    ///
    /// Queued components always run in PCGComponent.CompareOrder, whatever order they were queued in, and a
    /// component that has run queues the later ones around it that read the scene. So a component is always
    /// run again after everything it depends on, and the scene ends up the same whichever change came first.
    ///
    /// PCGComponent.Execute() stays immediate and takes the component out of the queue.
    /// Main thread only, except BeginHold / EndHold.
    /// </summary>
    public static class PCGScheduler
    {
        private static readonly List<PCGComponent> _dirty = new();
        private static readonly HashSet<PCGComponent> _ranThisFlush = new();
        private static readonly List<Func<bool>> _holdConditions = new();
        private static int _holds;

        // Frames the queue has been waiting for terrain heights. Past the limit it stops waiting, until heights
        // do arrive: a terrain that is never drawn, or has nothing to bake, never reads any back.
        private static int _terrainWaitFrames;
        private static bool _terrainGaveUp;
        private const int MaxTerrainWaitFrames = 300;

        /// <summary>Components waiting to regenerate.</summary>
        public static int PendingCount => _dirty.Count;

        public static void Request(PCGComponent component)
        {
            if (!_dirty.Contains(component))
                _dirty.Add(component);
        }

        public static void Cancel(PCGComponent component) => _dirty.Remove(component);

        /// <summary>
        /// Keep queued components waiting for as long as the condition returns true (e.g. "a gizmo handle is
        /// being dragged"). The condition is dropped the first time it returns false.
        /// </summary>
        public static void HoldWhile(Func<bool> condition)
        {
            if (!_holdConditions.Contains(condition))
                _holdConditions.Add(condition);
        }

        /// <summary>Keep queued components waiting until the matching EndHold. Callable from any thread.</summary>
        public static void BeginHold() => Interlocked.Increment(ref _holds);

        public static void EndHold() => Interlocked.Decrement(ref _holds);

        /// <summary>
        /// A terrain of the scene has no CPU heights yet, or new ones coming (bake or readback outstanding).
        /// A graph run now would project onto flat ground or onto heights about to be replaced, so runs wait.
        /// </summary>
        public static bool TerrainHeightsPending => !_terrainGaveUp && AnyTerrainPending();

        private static bool AnyTerrainPending()
        {
            foreach (var terrain in ComponentCache<TerrainRenderer>.All)
                if (terrain.HeightsPending) return true;
            return false;
        }

        /// <summary>Once per frame: run what is queued unless an edit is still in progress.</summary>
        public static void Update()
        {
            for (int i = _holdConditions.Count - 1; i >= 0; i--)
                if (!_holdConditions[i]())
                    _holdConditions.RemoveAt(i);

            bool terrainPending = AnyTerrainPending();
            if (!terrainPending)
            {
                _terrainWaitFrames = 0;
                _terrainGaveUp = false;
            }

            if (_dirty.Count == 0) return;
            if (_holdConditions.Count > 0 || Volatile.Read(ref _holds) > 0) return;

            if (terrainPending && !_terrainGaveUp)
            {
                if (++_terrainWaitFrames <= MaxTerrainWaitFrames) return;

                _terrainGaveUp = true;
                Debug.LogWarning("PCG", "Terrain heights did not arrive; running PCG graphs without them.");
            }

            Flush();
        }

        /// <summary>
        /// Run everything queued now, ignoring edit holds (e.g. before showing a freshly loaded level).
        /// Components still waiting for terrain heights stay queued.
        /// </summary>
        public static void Flush()
        {
            if (_dirty.Count == 0) return;

            // A run queues the later components around it; those are picked up below, in order. A component
            // queued again after it ran here (a message sent by a run, missing terrain heights) waits for the
            // next frame, so this always ends.
            _ranThisFlush.Clear();
            while (true)
            {
                PCGComponent? next = null;
                foreach (var component in _dirty)
                {
                    if (_ranThisFlush.Contains(component)) continue;
                    if (next == null || PCGComponent.CompareOrder(component, next) < 0)
                        next = component;
                }
                if (next == null) break;

                _ranThisFlush.Add(next);
                _dirty.Remove(next);
                if (!next.IsDestroyed)
                    next.Execute();
            }
            _ranThisFlush.Clear();
        }

        /// <summary>
        /// Run every PCG component of the scene now, in order. <paramref name="executed"/> is called after each
        /// run with the time it took in milliseconds.
        /// </summary>
        public static void ExecuteAll(Action<PCGComponent, double>? executed = null)
        {
            var components = new List<PCGComponent>();
            foreach (var component in ComponentCache<PCGComponent>.All)
                if (!component.IsDestroyed && component.Graph != null)
                    components.Add(component);
            components.Sort(PCGComponent.CompareOrder);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            foreach (var component in components)
            {
                double start = sw.Elapsed.TotalMilliseconds;
                component.Execute();
                executed?.Invoke(component, sw.Elapsed.TotalMilliseconds - start);
            }
        }

        /// <summary>
        /// A component's output changed (it ran, or went away). Components that run after it, read the scene
        /// and work in the same area took the old output into account: queue them.
        /// </summary>
        internal static void OutputChanged(PCGComponent source)
        {
            bool bounded = source.TryGetRegion(out var min, out var max);

            foreach (var other in ComponentCache<PCGComponent>.All)
            {
                // A component that has not run yet depends on nothing; it runs when it is asked to
                if (other == source || other.IsDestroyed || !other.HasOutput || !other.ReadsScene) continue;
                if (PCGComponent.CompareOrder(source, other) >= 0) continue;

                if (bounded && other.TryGetRegion(out var otherMin, out var otherMax)
                    && (otherMax.X < min.X || otherMin.X > max.X || otherMax.Y < min.Y || otherMin.Y > max.Y))
                    continue;

                Request(other);
            }
        }
    }
}
