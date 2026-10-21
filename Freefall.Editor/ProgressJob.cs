using System.Diagnostics;
using Squid;

namespace Freefall.Editor
{
    /// <summary>
    /// A long editor job with the landing page's modal progress popup (ModalDialogs.ShowProgress), whoever started it: a
    /// button or MCP. The job is named steps, each weighted by its measured seconds for the bar. The status record is
    /// replaced on every change, so any thread reads a consistent one: the popup every frame, editor_status. One job
    /// runs at a time, and the popup keeps the user from starting another. The work runs on workers and reaches the
    /// main thread through Engine.RunOnMainThreadAsync slices (the pattern of ProjectOpening, editor_debugging/project_open.md).
    /// Publish, Scan and the texture caps use it.
    /// </summary>
    internal sealed class ProgressJob
    {
        /// <summary>Where a job stands: the title, the step i/n with what it is doing, the whole job 0..1.</summary>
        public sealed record Status(string Title, int StepIndex, int StepCount, string Step, string Detail, float Fraction, double Seconds);

        static ProgressJob? _running;

        /// <summary>The job running now, or null.</summary>
        public static ProgressJob? Running => Volatile.Read(ref _running);

        readonly string _tag;
        readonly string[] _steps;
        readonly double[] _weights;
        readonly Stopwatch _clock = Stopwatch.StartNew();
        readonly Desktop _desktop;
        readonly Window _popup;
        volatile Status _status;
        int _index;

        public string Title { get; }
        public Status Current => _status;
        public TimeSpan Elapsed => _clock.Elapsed;

        /// <summary>The steps that ran, with their seconds.</summary>
        public List<(string Step, double Seconds)> StepTimes { get; } = new();

        ProgressJob(string title, string tag, (string Name, double Weight)[] steps)
        {
            Title = title;
            _tag = tag;
            _steps = steps.Select(s => s.Name).ToArray();
            _weights = steps.Select(s => s.Weight).ToArray();
            _status = new Status(title, 0, _steps.Length, _steps.Length > 0 ? _steps[0] : "", "", 0f, 0);
            _desktop = Program.EditorUI;
            _popup = ModalDialogs.ShowProgress(_desktop, () => _status is { } s
                ? new ModalDialogs.Progress(s.Title, $"{s.Step}   ({s.StepIndex + 1}/{s.StepCount})", s.Detail, s.Fraction)
                : null);
        }

        /// <summary>
        /// Main thread: shows the popup. <paramref name="tag"/> starts the job's log lines ("[Publish] …"). Throws
        /// InvalidOperationException while another job runs.
        /// </summary>
        public static ProgressJob Start(string title, string tag, params (string Name, double Weight)[] steps)
        {
            if (Running is { } other) throw new InvalidOperationException($"Wait for it to finish: {other.Title} is running");
            var job = new ProgressJob(title, tag, steps);
            Volatile.Write(ref _running, job);
            return job;
        }

        /// <summary>Run a step (its work starts on the calling thread: a worker's Task.Run, or Engine.RunOnMainThreadAsync) and log its time.</summary>
        public async Task Step(string name, Func<Task> work)
        {
            int index = Array.IndexOf(_steps, name);
            if (index >= 0) _index = index;
            Publish(name, "", 0f);
            var watch = Stopwatch.StartNew();
            await work();
            StepTimes.Add((name, watch.Elapsed.TotalSeconds));
            Debug.LogAlways($"[{_tag}] {name}: {watch.Elapsed.TotalSeconds:0.0} s");
        }

        /// <summary>Progress within the current step (any thread): what it is doing, the share of the step done.</summary>
        public void Report(string detail, float fraction)
        {
            var status = _status;
            if (status != null) Publish(status.Step, detail, fraction);
        }

        /// <summary>A step's weight once its size is known (the packs by their bytes, a reimport by its files).</summary>
        public void SetWeight(string step, double weight)
        {
            int i = Array.IndexOf(_steps, step);
            if (i >= 0) _weights[i] = weight;
        }

        /// <summary>Worker: the job is done. Closes the popup and shows <paramref name="toast"/> (null: none).</summary>
        public Task Finish(string? toast) => Engine.RunOnMainThreadAsync(() =>
        {
            End();
            if (toast != null) Toast.Show(toast, 4f);
        });

        /// <summary>Worker: the job failed. Logs it, closes the popup and shows the reason in a message popup.</summary>
        public Task Fail(string title, Exception error)
        {
            Debug.LogWarning(_tag, $"{Title}: {error}");
            return Engine.RunOnMainThreadAsync(() =>
            {
                End();
                ModalDialogs.ShowMessage(_desktop, title, error.Message, ("OK", null));
            });
        }

        void End()
        {
            _popup.Close();
            Interlocked.CompareExchange(ref _running, null, this);
        }

        void Publish(string step, string detail, float stepFraction)
        {
            double total = 0, done = 0;
            for (int i = 0; i < _weights.Length; i++)
            {
                total += _weights[i];
                if (i < _index) done += _weights[i];
            }
            if (_index < _weights.Length)
                done += _weights[_index] * Math.Clamp(stepFraction, 0f, 1f);

            _status = new Status(Title, _index, _steps.Length, step, detail ?? "", total > 0 ? (float)(done / total) : 0f, _clock.Elapsed.TotalSeconds);
        }

        /// <summary>The exception a failed await threw, unwrapped from an AggregateException.</summary>
        public static Exception Unwrap(Exception ex) => ex is AggregateException agg ? agg.Flatten().InnerException ?? ex : ex;
    }
}
