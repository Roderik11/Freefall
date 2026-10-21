using System;
using System.Collections.Generic;
using System.Diagnostics;
using Freefall.Assets;

namespace Freefall.Base
{
    /// <summary>
    /// Changes the scene. A load is a <see cref="SceneLoad"/>: steps that each get a slice of the frame, so frames keep
    /// coming (a loading screen, the editor's progress popup) and the world is paused until it is ready.
    /// <para><b>From game code:</b> <c>SceneManager.LoadAsync(scene)</c> (or <see cref="Load"/>, the same) with a
    /// SceneAsset the script references in a field, never by name. It may be called from a component's Update: the old
    /// scene goes from the next frame on. Then, in this order: <see cref="LoadStarted"/>; the old scene is destroyed
    /// (<see cref="SceneLoad.Cleared"/>); the new one is read (<see cref="Loaded"/>, nothing awake yet); its components
    /// wake, the terrain bakes, PCG generates; <see cref="Ready"/>, and the world runs again.</para>
    /// <para><b>While it loads</b> (<see cref="Loading"/> is not null) no component's Update runs, and no physics: a
    /// loading screen updates in its UI controls' own Update (Squid), which the host runs every frame, and reads
    /// <see cref="SceneLoad.Progress"/> (0..1, never going back), <see cref="SceneLoad.Name"/> and
    /// <see cref="SceneLoad.Scene"/>. Its entity is marked <see cref="EntityFlags.DontDestroy"/>, so the load does not
    /// destroy it: entities with that flag survive every scene change (the editor's Stop takes them too).</para>
    /// <para><see cref="LoadNow"/> clears and reads at once and leaves the rest to the following frames (editor, tools).</para>
    /// </summary>
    public static class SceneManager
    {
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

        /// <summary>
        /// The world is not drawn while a scene loads, only the UI over a cleared picture (a loading screen covers it
        /// anyway), except in the frames the load needs it drawn: the terrain bakes its heights in its draw, and the
        /// last frames before ready create the pipelines and buffers of what is new. Saves the drawing of a half-made
        /// world in every frame of the load. The player turns it on; the editor draws its viewports itself.
        /// </summary>
        public static bool HideWorldWhileLoading { get; set; }

        /// <summary>Engine.Tick: whether the main camera draws the world this frame.</summary>
        internal static bool DrawsWorld => Loading is not { } load || !HideWorldWhileLoading || load.WantsWorldDrawn;

        /// <summary>Replace the scene with <paramref name="scene"/>, starting at the next frame: <see cref="LoadAsync"/>.</summary>
        public static SceneLoad Load(SceneAsset scene) => LoadAsync(scene ?? throw new ArgumentNullException(nameof(scene)));

        /// <summary>
        /// Start replacing the scene with <paramref name="scene"/>: the steps run in slices from the next frame on. Safe
        /// from a component's update. A load still in progress is superseded. <paramref name="yaml"/> stands in for the
        /// scene's own YAML (the editor's play mode starts from the scene as it was in memory); with it,
        /// <paramref name="scene"/> may be null for a scene that is not an asset yet. With
        /// <paramref name="clearPersistent"/> the DontDestroy entities go too (the editor's Stop: the game is over).
        /// </summary>
        public static SceneLoad LoadAsync(SceneAsset? scene, byte[]? yaml = null, bool clearPersistent = false)
        {
            yaml ??= scene?.Yaml ?? throw new InvalidOperationException(
                scene == null ? "No scene and no YAML to load" : $"Scene '{scene.Name}' has no YAML");

            Loading?.Supersede();
            var load = new SceneLoad(scene, yaml, clearPersistent);
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

        /// <summary>Engine.Tick, before physics and Update: give the load in progress its slice.</summary>
        internal static void Update()
        {
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
