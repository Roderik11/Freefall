using System;
using System.Collections.Generic;
using System.Globalization;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// The <see cref="Profiler"/> call tree: average and max ms over the last 'frames' recorded frames, or
    /// one frame ('slowest' = the slowest of them). Children sorted by time; sections under 'minMs' are left out.
    /// </summary>
    [CommandRoute("GET", "/api/debug/profiler")]
    public class GetProfilerCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var qs = CommandHelpers.ParseQueryString(context.Path);
            int frames = qs.TryGetValue("frames", out var f) && int.TryParse(f, out var fv) ? Math.Clamp(fv, 1, Profiler.HistoryLength) : 120;
            double minMs = qs.TryGetValue("minMs", out var m) && double.TryParse(m, NumberStyles.Float, CultureInfo.InvariantCulture, out var mv) ? mv : 0.05;
            bool slowest = qs.TryGetValue("slowest", out var s) && s == "true";

            long end = Profiler.FrameCount;
            long start = Math.Max(end - Math.Min(frames, Profiler.RecordedFrames), 0);
            int n = (int)(end - start);
            if (n == 0)
                return CommandResult.Json(new { frames = 0, note = "no frames recorded yet" });

            float sum = 0, max = 0;
            long alloc = 0, slowestFrame = start;
            int gcFrames = 0;
            for (long i = start; i < end; i++)
            {
                float ms = Profiler.FrameMs(i);
                sum += ms;
                if (ms > max) { max = ms; slowestFrame = i; }
                alloc += Profiler.FrameAllocatedBytes(i);
                if (Profiler.FrameCollections(i) != 0) gcFrames++;
            }

            long? frame = slowest ? slowestFrame : null;
            var gpu = Profiler.GpuRoot.Stats(n);

            return CommandResult.Json(new
            {
                frames = n,
                paused = Profiler.Paused,
                avgMs = Math.Round(sum / n, 3),
                maxMs = Math.Round(max, 3),
                // the primary command list on the GPU (first to last command), a few frames behind
                gpuAvgMs = Math.Round(gpu.AvgMs, 3),
                gpuMaxMs = Math.Round(gpu.MaxMs, 3),
                // the per-section allocKB below counts the main thread only
                allocatedKBPerFrameAllThreads = Math.Round(alloc / 1024.0 / n, 1),
                framesWithGC = gcFrames,
                frame = frame == null ? null : new
                {
                    index = frame.Value,
                    ms = Math.Round(Profiler.FrameMs(frame.Value), 3),
                    gpuTimed = Profiler.IsGpuRecorded(frame.Value),
                    gc = Profiler.FrameCollections(frame.Value) // bit 0..2 = gen0..gen2 collected
                },
                tree = Describe(Profiler.Root, n, frame, minMs),
                gpuTree = Describe(Profiler.GpuRoot, n, frame, minMs)
            });
        }

        private static Dictionary<string, object> Describe(ProfilerNode node, int frames, long? frame, double minMs)
        {
            var result = new Dictionary<string, object> { ["name"] = node.Name };
            if (frame is long index)
            {
                var sample = node.Sample(index);
                result["ms"] = Math.Round(sample.Ms, 3);
                if (sample.AllocBytes >= 52) result["allocKB"] = Math.Round(sample.AllocBytes / 1024.0, 1);
                if (sample.Items > 0) result["items"] = sample.Items;
                if (sample.Calls > 1) result["calls"] = sample.Calls;
            }
            else
            {
                var stats = node.Stats(frames);
                result["ms"] = Math.Round(stats.AvgMs, 3);
                result["maxMs"] = Math.Round(stats.MaxMs, 3);
                if (stats.AvgAllocBytes >= 52) result["allocKB"] = Math.Round(stats.AvgAllocBytes / 1024.0, 1);
                if (stats.AvgItems > 0) result["items"] = Math.Round(stats.AvgItems, 1);
                if (stats.AvgCalls > 1.01f) result["calls"] = Math.Round(stats.AvgCalls, 1);
            }

            var children = new List<(ProfilerNode Node, float Ms, float Max)>();
            foreach (var child in node.Children)
            {
                float ms, maxMs;
                if (frame is long i) ms = maxMs = child.Sample(i).Ms;
                else
                {
                    var stats = child.Stats(frames);
                    ms = stats.AvgMs;
                    maxMs = stats.MaxMs;
                }

                // A section that is cheap on average but spikes still shows
                if (ms >= minMs || maxMs >= minMs * 10) children.Add((child, ms, maxMs));
            }

            if (children.Count > 0)
            {
                children.Sort((a, b) => b.Ms.CompareTo(a.Ms));
                var list = new List<Dictionary<string, object>>(children.Count);
                foreach (var (child, _, _) in children)
                    list.Add(Describe(child, frames, frame, minMs));
                result["children"] = list;
            }

            return result;
        }
    }
}
