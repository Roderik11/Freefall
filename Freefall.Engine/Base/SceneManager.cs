using System;
using System.Collections.Generic;
using System.Diagnostics;
using Freefall.Assets;

namespace Freefall.Base
{
    /// <summary>
    /// Changes the scene. A load is a <see cref="SceneLoad"/>: steps that each get a slice of the frame, so frames keep
    /// coming (a loading screen, the editor's progress popup) and the world is paused until it is ready.
    /// Game code calls <see cref="LoadAsync"/> (or <see cref="Load"/>) with a SceneAsset it references (a field), and
    /// shows <see cref="Loading"/>'s progress. <see cref="LoadNow"/> clears and reads at once and leaves the rest to the
    /// following frames (editor, tools). Entities marked DontDestroy survive the change.
    /// </summary>
    public static class SceneManager
    {
        private static SceneAsset? _pending;

        /// <summary>The scene loaded last, or null. Set once its entities have been read.</summary>
        public static SceneAsset? Current { get; private set; }

        /// <summary>The load in progress, or null.</summary>
        public static SceneLoad? Loading { get; private set; }

        /// <summary>A load is in progress: the world is paused (no physics, navigation or component updates).</summary>
        public static bool IsLoading => Loading != null;

        /// <summary>Main-thread time a load may take per frame, in ms. A component's Awake or one PCG graph is never cut, so single frames can take longer.</summary>
        public static double SliceBudgetMs { get; set; } = 20;

        /// <summary>
        /// After a scene's entities have been read: the scene and its entities. Their Early/Awake have not run yet, and
        /// the load goes on (wake-up, terrain, PCG); <see cref="Ready"/> follows at the end. For what must point at the new
        /// scene while it loads (cameras at their view).
        /// </summary>
        public static event Action<SceneAsset?, List<Entity>>? Loaded;

        /// <summary>A load has finished: everything is awake and generated, the world runs again. Also after one that failed (see <see cref="SceneLoad.Error"/>).</summary>
        public static event Action<SceneLoad>? Ready;

        /// <summary>A load has started.</summary>
        public static event Action<SceneLoad>? LoadStarted;

        /// <summary>Replace the scene with <paramref name="scene"/>, starting at the next frame. Same as <see cref="LoadAsync"/>, for callers that need no handle.</summary>
        public static void Load(SceneAsset scene)
        {
            _pending = scene ?? throw new ArgumentNullException(nameof(scene));
        }

        /// <summary>
        /// Start replacing the scene with <paramref name="scene"/>: the steps run in slices from the next frame on. Safe
        /// from a component's update. A load still in progress is superseded. <paramref name="yaml"/> stands in for the
        /// scene's own YAML (the editor's play mode starts from the scene as it was in memory); with it,
        /// <paramref name="scene"/> may be null for a scene that is not an asset yet.
        /// </summary>
        public static SceneLoad LoadAsync(SceneAsset? scene, byte[]? yaml = null)
        {
            yaml ??= scene?.Yaml ?? throw new InvalidOperationException(
                scene == null ? "No scene and no YAML to load" : $"Scene '{scene.Name}' has no YAML");

            _pending = null;
            Loading?.Supersede();
            var load = new SceneLoad(scene, yaml);
            Loading = load;
            LoadStarted?.Invoke(load);
            return load;
        }

        /// <summary>
        /// Replace the scene with <paramref name="scene"/>: every entity not marked DontDestroy is destroyed and the scene's
        /// entities are created before this returns. The rest of the load (wake-up, terrain, PCG) follows over the next
        /// frames as <see cref="Loading"/>. Not from inside a component update: it destroys the entities being updated;
        /// use <see cref="LoadAsync"/> there.
        /// </summary>
        public static List<Entity> LoadNow(SceneAsset? scene, byte[]? yaml = null)
        {
            var load = LoadAsync(scene, yaml);
            load.Run(long.MaxValue, stopBefore: SceneLoad.Step.Waking);
            return load.Entities;
        }

        /// <summary>
        /// The [SceneLoad] summary of the last load that finished: main-thread time and end of each step, slices,
        /// frames and the profiler sections it set off. Null until one has.
        /// </summary>
        public static string? LastLoadReport { get; private set; }

        /// <summary>Same as <see cref="IsLoading"/>.</summary>
        public static bool LoadInProgress => IsLoading;

        /// <summary>Engine.Tick, before physics and Update: carry out a Load asked for last frame, and give the load in progress its slice.</summary>
        internal static void Update()
        {
            if (_pending is { } scene)
            {
                _pending = null;
                LoadAsync(scene);
            }

            var load = Loading;
            if (load == null) return;
            load.Frame();
            load.Run(Stopwatch.GetTimestamp() + (long)(SliceBudgetMs * Stopwatch.Frequency / 1000));
        }

        // SceneLoad: the entities are read
        internal static void OnRead(SceneLoad load, List<Entity> entities)
        {
            Current = load.Scene;
            Loaded?.Invoke(load.Scene, entities);
        }

        /// <summary>The load that finished last (also one that failed or was superseded), or null.</summary>
        public static SceneLoad? LastLoad { get; private set; }

        // SceneLoad: finished, failed or superseded
        internal static void OnDone(SceneLoad load)
        {
            LastLoad = load;
            LastLoadReport = load.Report;
            if (Loading == load) Loading = null;
            if (!load.Superseded) Ready?.Invoke(load);
        }
    }
}
