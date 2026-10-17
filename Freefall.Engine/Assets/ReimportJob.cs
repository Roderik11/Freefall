using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace Freefall.Assets
{
    /// <summary>
    /// A forced reimport of a list of source files (AssetDatabase.ReimportFolder / ReimportFiles). It runs on the
    /// main thread, a few files per frame (AssetDatabase.TickReimports), so the editor keeps drawing, the console
    /// shows its progress, and it goes on when whoever started it stops waiting (an MCP call that timed out).
    /// </summary>
    public sealed class ReimportJob
    {
        private readonly List<string> _failed = new();
        private readonly Stopwatch _time = Stopwatch.StartNew();
        private readonly TaskCompletionSource<ReimportJob> _completion = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private double _lastReportSeconds;

        /// <summary>What was reimported: a folder relative to Assets ("Assets" for all of it), a file, or a caller's label.</summary>
        public string Label { get; }

        /// <summary>Source paths relative to Assets, in the order they are imported.</summary>
        public IReadOnlyList<string> Paths { get; }

        public int Total => Paths.Count;

        /// <summary>Files handled so far, failed ones included.</summary>
        public int Done { get; private set; }

        public int Imported => Done - _failed.Count;
        public int Failed => _failed.Count;
        public IReadOnlyList<string> FailedPaths => _failed;

        /// <summary>Loaded assets updated in place from the new cache files (hot reload).</summary>
        public int Reloaded { get; internal set; }

        /// <summary>Ended before its last file: the project was closed or another one opened.</summary>
        public bool Aborted { get; private set; }

        public TimeSpan Elapsed => _time.Elapsed;
        public bool IsCompleted => _completion.Task.IsCompleted;

        /// <summary>Completes when the job ends; continuations run on the thread pool.</summary>
        public Task<ReimportJob> Completion => _completion.Task;

        /// <summary>Raised on the main thread when the job ends.</summary>
        public event Action<ReimportJob> Completed;

        internal ReimportJob(string label, IReadOnlyList<string> paths)
        {
            Label = label;
            Paths = paths;
        }

        /// <summary>Progress in one line: "'Pack': 120/368 files, 1 failed, 34 s".</summary>
        public string Describe() =>
            $"'{Label}': {Done}/{Total} files, {Failed} failed, {Elapsed.TotalSeconds:F0} s";

        internal void Advance(string path, bool imported)
        {
            if (!imported) _failed.Add(path);
            Done++;
        }

        /// <summary>True once per interval, for the progress line.</summary>
        internal bool ShouldReport(double intervalSeconds)
        {
            double now = Elapsed.TotalSeconds;
            if (now - _lastReportSeconds < intervalSeconds) return false;
            _lastReportSeconds = now;
            return true;
        }

        internal string Summary()
        {
            var text = $"[AssetDatabase] Reimported '{Label}': {Imported} of {Total} files in {Elapsed.TotalSeconds:F1} s";
            if (Failed > 0)
                text += $", {Failed} failed: " + string.Join(", ", _failed.Take(10)) + (Failed > 10 ? ", ..." : "");
            if (Reloaded > 0)
                text += $", {Reloaded} loaded assets updated in place";
            return text;
        }

        internal void Complete(bool aborted = false)
        {
            Aborted = aborted;
            _time.Stop();
            try { Completed?.Invoke(this); }
            catch (Exception ex) { Debug.LogWarning("AssetDatabase", $"Reimport completion handler failed: {ex.Message}"); }
            _completion.TrySetResult(this);
        }
    }
}
