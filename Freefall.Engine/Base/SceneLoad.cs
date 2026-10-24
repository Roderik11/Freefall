using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using Freefall.Assets;
using Freefall.Components;
using Freefall.PCG;
using Freefall.Serialization;

namespace Freefall.Base
{
    /// <summary>
    /// A scene on its way in (<see cref="SceneManager.LoadAsync"/>, <see cref="SceneManager.LoadNow"/>). The load is one
    /// sequence of steps, each given a slice of the frame (<see cref="SceneManager.SliceBudgetMs"/>), so frames keep coming:
    /// clear the old scene, read the new one, wake its components, wait for the terrain's heights, generate (PCG, and
    /// wake what it put out), wait for background work components asked it to (<see cref="WaitFor"/>), then two frames
    /// with everything in place. While it runs the world is paused: no physics, navigation or component updates, and
    /// PCG runs only here. Rendering goes on (the terrain bakes in its draw, the UI shows the progress).
    /// <para>One unit of work is never cut: a component's Awake or one PCG graph takes as long as it takes, so the
    /// longest slice (in the summary) is the frame a load cannot go below without moving that work off the main thread.</para>
    /// </summary>
    public sealed class SceneLoad
    {
        public enum Step { Clearing, Reading, Waking, Terrain, Generating, Waiting, Finishing, Done }

        // Share of the bar per step, roughly what each takes on Eldermoor after a warm start
        private static readonly float[] Weights = { 0.10f, 0.10f, 0.35f, 0.03f, 0.35f, 0.04f, 0.03f };
        private const int TerrainWaitFrames = 300;
        private const int FinishFrames = 2;

        public SceneAsset? Scene { get; }
        public string Name { get; }

        /// <summary>The step it is in; <see cref="Step.Done"/> once finished, failed or superseded.</summary>
        public Step Current { get; private set; } = Step.Clearing;

        /// <summary>What it is doing, for a progress line ("Waking components: 1,234 left").</summary>
        public string Activity { get; private set; } = "Clearing the scene";

        public bool IsDone => Current == Step.Done;

        /// <summary>Why it stopped early (a step threw); null otherwise.</summary>
        public Exception? Error { get; private set; }

        /// <summary>The step that threw <see cref="Error"/>.</summary>
        public Step FailedIn { get; private set; } = Step.Done;

        /// <summary>Another load replaced it before it finished.</summary>
        public bool Superseded { get; private set; }

        /// <summary>The scene's entities as read from its file (not what PCG or scripts add), once read.</summary>
        public List<Entity> Entities => _read?.Entities ?? _empty;
        private static readonly List<Entity> _empty = new();

        /// <summary>Seconds since it started; once done, how long it took.</summary>
        public double Seconds => _doneSeconds ?? Stopwatch.GetElapsedTime(_start).TotalSeconds;
        private double? _doneSeconds;

        /// <summary>When it finished (also when it failed or was superseded). On the main thread.</summary>
        public event Action<SceneLoad>? Completed;

        /// <summary>
        /// The old scene is gone, the new one not read yet: for what must change in between (the editor switches play
        /// mode here, so the old scene is torn down in the mode it ran in). Subscribe right after LoadAsync: nothing runs
        /// before the next frame.
        /// </summary>
        public event Action<SceneLoad>? Cleared;

        /// <summary>The scene's entities have been read (just before <see cref="SceneManager.Loaded"/>); nothing is awake yet.</summary>
        public event Action<SceneLoad>? Read;

        /// <summary>The [SceneLoad] summary, once done.</summary>
        public string? Report { get; private set; }

        private readonly byte[] _yaml;
        private readonly long _start = Stopwatch.GetTimestamp();
        private readonly double[] _stepMs = new double[(int)Step.Done];      // main-thread time per step
        private readonly double[] _stepEndS = new double[(int)Step.Done];    // seconds since the start when each step ended
        private EntityManager.ClearJob? _clear;
        private EntitySerializer? _serializer;
        private EntitySerializer.Job? _read;
        private readonly HashSet<PCGComponent> _ran = new();
        private int _ranTotal;
        private bool _batchOpen;
        private int _terrainFrames, _finishFrames, _woken;
        private readonly List<(Task Task, string What)> _waits = new();
        private readonly SceneLoadReport _report = new();
        private double _longestSliceMs;
        private int _slices;

        private readonly bool _clearPersistent;

        internal SceneLoad(SceneAsset? scene, byte[] yaml, bool clearPersistent = false)
        {
            Scene = scene;
            Name = scene?.Name ?? "(unsaved scene)";
            _yaml = yaml;
            _clearPersistent = clearPersistent;
            _history.TryGetValue(HistoryKey, out _previous);
        }

        // When each step ended in the last load of the same scene that finished (this session): the bar's shares
        private static readonly Dictionary<string, double[]> _history = new();
        private readonly double[]? _previous;
        private string HistoryKey => Scene?.Guid ?? Name;
        private float _progress;

        /// <summary>Number of steps before <see cref="Step.Done"/>, for a "(3/7)".</summary>
        public const int StepCount = (int)Step.Done;

        /// <summary>A step's name for people.</summary>
        public static string Describe(Step step) => step switch
        {
            Step.Clearing => "Clearing the old scene",
            Step.Reading => "Reading the scene",
            Step.Waking => "Starting components",
            Step.Terrain => "Terrain",
            Step.Generating => "Generating",
            Step.Waiting => "Background work",
            Step.Finishing => "Finishing",
            _ => "Done",
        };

        /// <summary>
        /// 0..1 over all steps, for a loading bar: it never goes back (a step's count of work left can grow, the woken
        /// timber kit adds its pieces). Each step's share of the bar is the share of the time it took the last time this
        /// scene was loaded (this session), or a fixed guess before that.
        /// </summary>
        public float Progress
        {
            get
            {
                if (Current == Step.Done) return 1f;
                float now = StepStart(Current) + StepShare(Current) * Math.Clamp(StepProgress, 0f, 1f);
                _progress = Math.Max(_progress, Math.Min(now, 0.999f));
                return _progress;
            }
        }

        // Where a step starts on the bar, and how much of it it takes
        private float StepStart(Step step)
        {
            if (_previous != null && _previous[^1] > 0)
                return step == Step.Clearing ? 0f : (float)(_previous[(int)step - 1] / _previous[^1]);
            float start = 0f;
            for (int i = 0; i < (int)step; i++) start += Weights[i];
            return start;
        }

        private float StepShare(Step step) => _previous != null && _previous[^1] > 0
            ? (float)((_previous[(int)step] - (step == Step.Clearing ? 0 : _previous[(int)step - 1])) / _previous[^1])
            : Weights[(int)step];

        /// <summary>
        /// The load needs the world drawn this frame (<see cref="SceneManager.HideWorldWhileLoading"/>): the terrain bakes
        /// or keeps its heights in its draw (from the terrain step on: TerrainRenderer waits for every stamp to wake), and
        /// the last frames create what the first visible frame would.
        /// </summary>
        internal bool WantsWorldDrawn => Current switch
        {
            Step.Terrain or Step.Finishing => true,
            Step.Generating or Step.Waiting => PCGScheduler.TerrainHeightsPending,   // generated stamps rebake it
            _ => false,
        };

        private float StepProgress => Current switch
        {
            Step.Clearing => _clear?.Progress ?? 0f,
            Step.Reading => _read?.Progress ?? 0f,
            Step.Waking => Fraction(_woken, ScriptExecution.PendingWakeups()),
            Step.Terrain => (float)_terrainFrames / 10f,
            Step.Generating => Fraction(_ranTotal, PCGScheduler.PendingCount),
            Step.Waiting => Fraction(_waits.FindAll(w => w.Task.IsCompleted).Count, _waits.FindAll(w => !w.Task.IsCompleted).Count),
            Step.Finishing => (float)_finishFrames / FinishFrames,
            _ => 1f,
        };

        private static float Fraction(int done, int left) => done + left == 0 ? 0f : (float)done / (done + left);

        /// <summary>
        /// Make the load wait for background work of a component it woke (the navmesh's assembly on a worker): it finishes
        /// only once the task has. A task that fails is logged and not waited for again. Outside a load, nothing waits.
        /// </summary>
        public void WaitFor(Task task, string what)
        {
            if (task == null || IsDone) return;
            _waits.Add((task, what));
        }

        /// <summary>
        /// Run steps until done, <paramref name="deadline"/> (a Stopwatch timestamp) has passed, or the step
        /// <paramref name="stopBefore"/> is reached (LoadNow: clear and read, nothing more). Main thread.
        /// </summary>
        internal void Run(long deadline, Step stopBefore = Step.Done)
        {
            if (IsDone) return;
            long sliceStart = Stopwatch.GetTimestamp();
            try
            {
                while (!IsDone && Current < stopBefore)
                {
                    var step = Current;
                    long t0 = Stopwatch.GetTimestamp();
                    bool finished = RunStep(deadline);
                    _stepMs[(int)step] += Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                    if (!finished) break;

                    _stepEndS[(int)step] = Seconds;
                    Current = step + 1;
                    if (Current == Step.Done) Finish();
                    else if (Stopwatch.GetTimestamp() >= deadline) break;
                }
            }
            catch (Exception ex)
            {
                Error = ex;
                FailedIn = Current;
                Debug.LogError("SceneLoad", $"Loading '{Name}' stopped in {Current}: {ex}");
                Current = Step.Done;
                Finish();
            }

            double sliceMs = Stopwatch.GetElapsedTime(sliceStart).TotalMilliseconds;
            _slices++;
            if (sliceMs > _longestSliceMs) _longestSliceMs = sliceMs;
        }

        // One step's work; true when the step is finished
        private bool RunStep(long deadline)
        {
            switch (Current)
            {
                case Step.Clearing:
                    _clear ??= new EntityManager.ClearJob(_clearPersistent);
                    Activity = $"Clearing the scene ({_clear.Total:N0} entities)";
                    if (!_clear.Step(deadline)) return false;
                    Cleared?.Invoke(this);
                    return true;

                case Step.Reading:
                    if (_read == null)
                    {
                        _serializer = new EntitySerializer();
                        _read = _serializer.BeginLoad(_yaml);
                    }
                    Activity = _read.Activity;
                    if (!_read.Step(deadline)) return false;
                    Read?.Invoke(this);
                    SceneManager.OnRead(this, _read.Entities);
                    _report.Begin(Name, _stepMs[(int)Step.Clearing], _serializer!.LastTimings, _read.Entities.Count);
                    return true;

                case Step.Waking:
                {
                    int before = ScriptExecution.PendingWakeups();
                    bool done = ScriptExecution.WakeUp(deadline);
                    _woken += Math.Max(0, before - ScriptExecution.PendingWakeups());
                    Activity = done ? "Components awake" : $"Waking components ({ScriptExecution.PendingWakeups():N0} left)";
                    return done;
                }

                case Step.Terrain:
                    // The terrain bakes or keeps its heights in its first draw, which a frame has to come for
                    if (!PCGScheduler.TerrainHeightsPending) return true;
                    Activity = "Terrain heights";
                    if (++_terrainFrames < TerrainWaitFrames) return false;
                    PCGScheduler.StopWaitingForTerrain();
                    return true;

                case Step.Generating:
                {
                    if (!_batchOpen)
                    {
                        ObstacleCache.BeginBatch();
                        _batchOpen = true;
                    }
                    int ranBefore = _ran.Count;
                    bool queueDone = PCGScheduler.RunQueued(deadline, _ran);
                    _ranTotal += Math.Max(0, _ran.Count - ranBefore);
                    var last = PCGScheduler.LastRun;
                    Activity = queueDone ? "Waking generated content"
                             : $"Generating {_ranTotal}/{_ranTotal + PCGScheduler.PendingCount}" + (last?.Entity != null ? $": {last.Entity.Name}" : "");
                    if (!queueDone || Stopwatch.GetTimestamp() >= deadline) return false;

                    // What the graphs put out wakes here too, before the load counts as generated
                    if (!ScriptExecution.WakeUp(deadline)) return false;
                    if (PCGScheduler.PendingCount > 0) return false;   // woken output queued more PCG
                    ObstacleCache.EndBatch();
                    _batchOpen = false;
                    return true;
                }

                case Step.Waiting:
                {
                    foreach (var (task, what) in _waits)
                    {
                        if (!task.IsCompleted)
                        {
                            Activity = what;
                            return false;
                        }
                        if (task.IsFaulted)
                            Debug.LogWarning("SceneLoad", $"'{Name}': {what} failed: {task.Exception?.GetBaseException().Message}");
                    }
                    // Work that finished on the main thread may have added components or PCG requests
                    if (ScriptExecution.PendingWakeups() > 0 || PCGScheduler.PendingCount > 0)
                    {
                        Current = Step.Waking;
                        return false;
                    }
                    return true;
                }

                case Step.Finishing:
                    // A frame or two with everything in place, before the world counts as ready (the first draws of new
                    // content create pipelines and upload buffers)
                    Activity = "Finishing";
                    return ++_finishFrames > FinishFrames;
            }
            return true;
        }

        /// <summary>Another load replaces this one: it stops where it is.</summary>
        internal void Supersede()
        {
            if (IsDone) return;
            Superseded = true;
            Current = Step.Done;
            Finish();
        }

        private void Finish()
        {
            _doneSeconds = Seconds;
            if (_batchOpen)
            {
                ObstacleCache.EndBatch();
                _batchOpen = false;
            }

            var text = new StringBuilder();
            text.Append($"[SceneLoad] {Name} ");
            text.Append(Error != null ? $"failed after {Seconds:F1} s" : Superseded ? $"superseded after {Seconds:F1} s" : $"ready after {Seconds:F1} s");
            text.Append(" (main thread ms / done at s:");
            for (int i = 0; i < _stepMs.Length; i++)
                text.Append($" {(Step)i} {_stepMs[i]:F0}/{_stepEndS[i]:F1}");
            text.Append($"); {_slices} slices, longest {_longestSliceMs:F0} ms; PCG {_ranTotal} runs");
            if (_waits.Count > 0) text.Append($"; waited for {string.Join(", ", _waits.ConvertAll(w => w.What))}");
            Report = _report.Finish(text.ToString());
            Debug.LogAlways(Report);
            if (Error == null && !Superseded)
                _history[HistoryKey] = (double[])_stepEndS.Clone();

            SceneManager.OnDone(this);
            Completed?.Invoke(this);
        }

        /// <summary>Once per tick while it runs (SceneManager.Update): the frame's time and profiler sections, for the report.</summary>
        internal void Frame() => _report.Frame();
    }

    /// <summary>
    /// What a load set off, for its summary: the frames (longest, those over 100 ms) and the time in the profiler sections
    /// a load sets off, wherever they appear in the tree (a name nested in itself counts once), and the wake-up per type.
    /// Measures only.
    /// </summary>
    internal sealed class SceneLoadReport
    {
        private const double StallMs = 100;

        private static readonly string[] Sections =
        {
            "Wake-up", "PCG", "PCG graph", "PCG destroy output", "Prefab.Instantiate", "Prefab.ApplyTo",
            "Parse YAML", "Resolve references", "Resolve assets", "Hydrate prefabs",
            "TerrainRenderer", "Terrain texture arrays", "Terrain holes", "Terrain height plan", "Terrain palettes",
            "Terrain height readback", "Terrain heights changed", "Terrain splat plan", "Terrain deco plan",
            "Terrain heightmap upload", "RuntimeMesh generate", "RuntimeMesh upload", "Push normals",
            "NavMesh world", "NavMesh gizmo mesh", "ExcludeObstacles", "ExcludeObstacles gather", "ExcludeObstacles filter",
            "Deferred disposals", "Collider find body", "Collider actor", "Collider geometry", "Collider mesh cook", "Collider shape",
            "Terrain collider samples", "Terrain collider cook", "Terrain collider height field", "Render", "Physics",
        };
        private static readonly HashSet<string> SectionSet = new(Sections);

        private long _lastCheck = Stopwatch.GetTimestamp();
        private long _lastFrame = Profiler.FrameCount;
        private int _frames, _stalls;
        private double _longestMs, _stallMs;
        private readonly Dictionary<string, double> _sectionMs = new();
        private readonly Dictionary<string, double> _wakeMs = new();

        /// <summary>The scene is read: log what that took.</summary>
        public void Begin(string name, double clearMs, EntitySerializer.Timings t, int entities)
        {
            Debug.LogAlways($"[SceneLoad] {name}: cleared {clearMs:F0} ms, read (parse {t.ParseMs:F0}, references {t.ReferencesMs:F0}, " +
                            $"assets {t.AssetsMs:F0}, prefabs {t.PrefabsMs:F0}; {entities} entities)");
        }

        public void Frame()
        {
            long now = Stopwatch.GetTimestamp();
            double frameMs = (now - _lastCheck) * 1000.0 / Stopwatch.Frequency;
            _lastCheck = now;
            _frames++;
            if (frameMs > _longestMs) _longestMs = frameMs;
            if (frameMs > StallMs) { _stalls++; _stallMs += frameMs; }

            for (long f = _lastFrame; f < Profiler.FrameCount; f++)
                Accumulate(Profiler.Root, f, null);
            _lastFrame = Profiler.FrameCount;
        }

        private void Accumulate(ProfilerNode node, long frame, HashSet<string>? onPath)
        {
            foreach (var child in node.Children)
            {
                string name = child.Name;
                bool counted = SectionSet.Contains(name) && (onPath == null || !onPath.Contains(name));
                float ms = 0;
                if (counted || node.Name == "Wake-up")
                {
                    ms = child.Sample(frame).Ms;
                    if (ms <= 0) continue;   // did not run in that frame: nor did anything under it
                }

                if (counted) _sectionMs[name] = _sectionMs.GetValueOrDefault(name) + ms;
                if (node.Name == "Wake-up") _wakeMs[name] = _wakeMs.GetValueOrDefault(name) + ms;

                if (child.Children.Count == 0) continue;
                if (counted)
                {
                    onPath ??= new HashSet<string>();
                    onPath.Add(name);
                    Accumulate(child, frame, onPath);
                    onPath.Remove(name);
                }
                else
                {
                    Accumulate(child, frame, onPath);
                }
            }
        }

        public string Finish(string head)
        {
            var text = new StringBuilder(head);
            text.Append($"; {EntityManager.Entities.Count} entities; {_frames} frames, longest {_longestMs:F0} ms, {_stalls} over {StallMs:F0} ms = {_stallMs:F0} ms");

            var sections = new List<string>();
            foreach (var n in Sections)
                if (_sectionMs.TryGetValue(n, out var ms)) sections.Add($"{n} {ms:F0}");
            text.Append(". Sections (ms): ").Append(string.Join(", ", sections));

            var wake = new List<KeyValuePair<string, double>>(_wakeMs);
            wake.Sort((a, b) => b.Value.CompareTo(a.Value));
            text.Append(". Wake-up by type (ms): ")
                .Append(string.Join(", ", wake.GetRange(0, Math.Min(8, wake.Count)).ConvertAll(p => $"{p.Key} {p.Value:F0}")));
            return text.ToString();
        }
    }
}
