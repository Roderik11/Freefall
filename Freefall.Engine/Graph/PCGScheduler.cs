using System;
using System.Collections.Generic;
using System.Threading;
using Freefall.Components;

namespace Freefall.PCG
{
    /// <summary>
    /// Coalesces PCG regeneration. Change notifications (spline, graph, terrain heights, member edits) call
    /// PCGComponent.Invalidate(), which queues the component here; each queued component runs at most once per
    /// frame, and not at all while an edit is in progress (gizmo drag, inspector drag, a multi-step build).
    /// PCGComponent.Execute() stays immediate and takes the component out of the queue.
    /// Main thread only, except BeginHold / EndHold.
    /// </summary>
    public static class PCGScheduler
    {
        private static readonly List<PCGComponent> _dirty = new();
        private static readonly List<PCGComponent> _running = new();
        private static readonly List<Func<bool>> _holdConditions = new();
        private static int _holds;

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

        /// <summary>Once per frame: run what is queued unless an edit is still in progress.</summary>
        public static void Update()
        {
            for (int i = _holdConditions.Count - 1; i >= 0; i--)
                if (!_holdConditions[i]())
                    _holdConditions.RemoveAt(i);

            if (_dirty.Count == 0) return;
            if (_holdConditions.Count > 0 || Volatile.Read(ref _holds) > 0) return;

            Flush();
        }

        /// <summary>Run everything queued now, ignoring holds (e.g. before showing a freshly loaded level).</summary>
        public static void Flush()
        {
            if (_dirty.Count == 0) return;

            // Executing can queue further components (messages sent by the run); those wait for the next frame.
            _running.AddRange(_dirty);
            _dirty.Clear();

            foreach (var component in _running)
                if (!component.IsDestroyed)
                    component.Execute();

            _running.Clear();
        }
    }
}
