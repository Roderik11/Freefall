using System;
using System.Diagnostics;
using System.Runtime.ExceptionServices;
using System.Threading;
using Freefall.Components;

namespace Freefall
{
    /// <summary>
    /// A parallel for that allocates nothing per call, for per-frame work: Parallel.For made about 2 KB of garbage on the
    /// main thread every call (its replicating tasks, range manager and loop state), which the Animator poses showed every
    /// frame. The caller's thread takes part; helpers are one kept work item queued on the .NET thread pool, which queues
    /// an IThreadPoolWorkItem without a wrapper. Keep the body delegate in a field: a lambda that captures anything is a
    /// new delegate every call.
    ///
    /// One call at a time: a call made while another runs (from inside a body, or from another thread) runs its indices
    /// in order on the calling thread.
    ///
    /// The profiler hears of every call: the section open around it shows how many threads took part and the time they
    /// spent in its bodies together (Profiler.Parallel), so what runs threaded shows without marking it by hand.
    /// </summary>
    public static class ParallelWork
    {
        // Indices still to be claimed, counted down: a helper that comes late (its call already done) claims a negative
        // number and goes, and the next call sets the count anew. No call starts before every index of the last one has
        // been processed, so a valid claim always belongs to the body that is set.
        private static int _unclaimed = -1;
        private static int _unfinished;
        private static Action<int>? _body;
        private static Exception? _error;
        private static int _busy;
        private static readonly ManualResetEventSlim _done = new(false);

        // For the profiler: the threads that took part in this call, their time in its bodies together (Stopwatch
        // ticks), and those still adding theirs up after their last body
        private static int _threads;
        private static long _workTicks;
        private static int _adding;
        private static readonly Helper _helper = new();

        // Most helpers queued for one call, on top of the calling thread
        private static readonly int MaxHelpers = Math.Max(0, Environment.ProcessorCount - 1);

        private sealed class Helper : IThreadPoolWorkItem
        {
            public void Execute() => RunClaims();
        }

        /// <summary>
        /// Runs <paramref name="body"/> for every index in [0, <paramref name="count"/>), spread over the calling thread
        /// and thread pool helpers, in no order, and returns when all are done. The first exception a body throws is
        /// thrown again here once the rest are done.
        /// </summary>
        public static void For(int count, Action<int> body)
        {
            if (count <= 0) return;

            bool alone = count == 1 || MaxHelpers == 0;
            if (alone || Interlocked.CompareExchange(ref _busy, 1, 0) != 0)
            {
                long start = Stopwatch.GetTimestamp();
                for (int i = 0; i < count; i++)
                    body(i);
                // In order on this thread. A call inside another's body is part of that one's work, already counted
                if (alone) Profiler.Parallel(Stopwatch.GetTimestamp() - start, 1);
                return;
            }

            try
            {
                _body = body;
                _error = null;
                _unfinished = count;
                _threads = 0;
                _workTicks = 0;
                _done.Reset();
                Interlocked.Exchange(ref _unclaimed, count);   // publishes the above to whoever claims from here on

                int helpers = Math.Min(count - 1, MaxHelpers);
                for (int i = 0; i < helpers; i++)
                    ThreadPool.UnsafeQueueUserWorkItem(_helper, preferLocal: false);

                RunClaims();
                _done.Wait();

                // A helper adds its time just after its last body: wait for that (nanoseconds), then tell the profiler
                var spin = new SpinWait();
                while (Volatile.Read(ref _adding) != 0) spin.SpinOnce(-1);
                Profiler.Parallel(Volatile.Read(ref _workTicks), Volatile.Read(ref _threads));

                if (_error != null)
                    ExceptionDispatchInfo.Capture(_error).Throw();
            }
            finally
            {
                _body = null;
                Volatile.Write(ref _busy, 0);

                // What the bodies moved off the main thread has its OnChanged handlers fired here, on the caller's thread,
                // before the call returns: as if the work had been done in place (Transform.SetDirty queues them)
                if (Engine.OnMainThread) Transform.FireChangesFromWorkers();
            }
        }

        private static void RunClaims()
        {
            int index = Interlocked.Decrement(ref _unclaimed);
            if (index < 0) return;   // came late: nothing left

            // Counted before its first body, so the call cannot end before this thread has added its time
            Interlocked.Increment(ref _adding);
            long start = Stopwatch.GetTimestamp();
            do
            {
                try { _body!(index); }
                catch (Exception e) { Interlocked.CompareExchange(ref _error, e, null); }

                if (Interlocked.Decrement(ref _unfinished) == 0)
                    _done.Set();
            }
            while ((index = Interlocked.Decrement(ref _unclaimed)) >= 0);

            Interlocked.Add(ref _workTicks, Stopwatch.GetTimestamp() - start);
            Interlocked.Increment(ref _threads);
            Interlocked.Decrement(ref _adding);
        }
    }
}
