using System;
using System.Threading.Tasks;

namespace Freefall.Base
{
    /// <summary>
    /// A long job under the editor's modal progress popup: named steps, each weighted for the bar, and a line of what it is
    /// doing. The editor's own (publish, scan) and the project's (the world map's bake) alike. Any thread, as the editor's
    /// ProgressJob is.
    /// </summary>
    public interface IEditorJob
    {
        /// <summary>How long it has run.</summary>
        TimeSpan Elapsed { get; }

        /// <summary>Runs a step (its work starts on the calling thread) and logs its time.</summary>
        Task Step(string name, Func<Task> work);

        /// <summary>Progress within the current step: what it is doing, the share of the step done.</summary>
        void Report(string detail, float fraction);

        /// <summary>The job is done: the popup closes, <paramref name="toast"/> shows (null: none).</summary>
        Task Finish(string? toast);

        /// <summary>The job failed: logged, the popup closes, the reason shows in a message popup.</summary>
        Task Fail(string title, Exception error);
    }

    /// <summary>
    /// Starts an <see cref="IEditorJob"/> from code that does not know the editor: project scripts. The editor sets
    /// <see cref="Starter"/> at startup; a published game has none.
    /// </summary>
    public static class EditorJobs
    {
        /// <summary>(title, log tag, steps) → the job; set by the editor.</summary>
        public static Func<string, string, (string Name, double Weight)[], IEditorJob>? Starter { get; set; }

        /// <summary>Whether a job can be started here (the editor).</summary>
        public static bool Available => Starter != null;

        /// <summary>
        /// A job with the popup, its steps in order (weights by their expected seconds). Main thread. Throws
        /// InvalidOperationException outside the editor, or while another job or a scene load runs.
        /// </summary>
        public static IEditorJob Start(string title, string tag, params (string Name, double Weight)[] steps)
            => Starter?.Invoke(title, tag, steps) ?? throw new InvalidOperationException("Only the editor runs long jobs");
    }
}
