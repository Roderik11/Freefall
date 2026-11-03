using System;
using System.Runtime.ExceptionServices;
using System.Threading;

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

            if (count == 1 || MaxHelpers == 0 || Interlocked.CompareExchange(ref _busy, 1, 0) != 0)
            {
                for (int i = 0; i < count; i++)
                    body(i);
                return;
            }

            try
            {
                _body = body;
                _error = null;
                _unfinished = count;
                _done.Reset();
                Interlocked.Exchange(ref _unclaimed, count);   // publishes the above to whoever claims from here on

                int helpers = Math.Min(count - 1, MaxHelpers);
                for (int i = 0; i < helpers; i++)
                    ThreadPool.UnsafeQueueUserWorkItem(_helper, preferLocal: false);

                RunClaims();
                _done.Wait();

                if (_error != null)
                    ExceptionDispatchInfo.Capture(_error).Throw();
            }
            finally
            {
                _body = null;
                Volatile.Write(ref _busy, 0);
            }
        }

        private static void RunClaims()
        {
            int index;
            while ((index = Interlocked.Decrement(ref _unclaimed)) >= 0)
            {
                try { _body!(index); }
                catch (Exception e) { Interlocked.CompareExchange(ref _error, e, null); }

                if (Interlocked.Decrement(ref _unfinished) == 0)
                    _done.Set();
            }
        }
    }
}
