using System;
using System.Collections.Generic;
using System.Reflection;

namespace Freefall.Base
{
    /// <summary>
    /// Where what a component's callbacks throw goes (Early, Awake, Update, Draw, Destroy, OnAttached,
    /// OnEnabledChanged, the caches' Added and Removed handlers), and what a script's system throws in its Update: the
    /// error is logged and the loop goes on with the next one. One broken script must not take the editor, or the game,
    /// down with it (a script's file write that failed in Update closed the editor with the scene unsaved, 2026-10-23).
    ///
    /// Each distinct error (the type, the callback, the exception and its message) is logged once with the stack where it
    /// was thrown; repeats are counted and reported at 10, 100, 1000, ..., so a script that throws every frame does not
    /// flood the console. Thread-safe: IParallel components update on workers.
    /// </summary>
    public static class CallbackErrors
    {
        // By text, never by Type: a script type kept here would pin its assembly after a hot reload
        private static readonly Dictionary<string, long> _seen = [];

        // Errors whose messages differ every time (an index, a time) would grow the table without end
        private const int MaxKept = 1000;

        /// <summary>Errors reported since the last <see cref="Forget"/>, repeats included.</summary>
        public static long Count { get; private set; }

        /// <summary>
        /// Report what <paramref name="owner"/>'s <paramref name="callback"/> threw (owner: a component or a system).
        /// </summary>
        public static void Report(object? owner, string callback, Exception exception)
        {
            if (exception is TargetInvocationException { InnerException: { } inner }) exception = inner;

            var type = owner?.GetType();
            string key = $"{type?.FullName}.{callback}|{exception.GetType().FullName}|{exception.Message}";
            long times;
            lock (_seen)
            {
                if (_seen.Count >= MaxKept && !_seen.ContainsKey(key)) _seen.Clear();
                _seen.TryGetValue(key, out times);
                _seen[key] = ++times;
                Count++;
            }

            string who = type != null ? $"[{type.Name}] " : "";
            string where = owner is Component { Entity: { } entity } ? $" on '{entity.Name}'" : "";
            if (times == 1)
                Debug.LogException($"{who}{callback}{where} threw", exception);
            else if (IsPowerOfTen(times))
                Debug.LogError($"{who}{callback}", $"threw {times} times now: {exception.GetType().Name}: {exception.Message}");
        }

        /// <summary>Start counting afresh: a reloaded script's errors are logged in full again.</summary>
        public static void Forget()
        {
            lock (_seen)
            {
                _seen.Clear();
                Count = 0;
            }
        }

        private static bool IsPowerOfTen(long n)
        {
            while (n >= 10 && n % 10 == 0) n /= 10;
            return n == 1;
        }
    }
}
