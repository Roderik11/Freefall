using System;
using System.Collections.Generic;
using Freefall.Assets;
using Freefall.Serialization;

namespace Freefall.Base
{
    /// <summary>
    /// Changes the scene. Game code calls <see cref="Load"/> with a SceneAsset it references (a field), and the
    /// scene is replaced at the start of the next frame, so a component may call it from its own update.
    /// <see cref="LoadNow"/> replaces it at once (editor, game start-up). Entities marked DontDestroy survive
    /// the change (EntityManager.ClearScene).
    /// </summary>
    public static class SceneManager
    {
        private static SceneAsset? _pending;

        /// <summary>The scene loaded last, or null.</summary>
        public static SceneAsset? Current { get; private set; }

        /// <summary>After a scene's entities exist (their Awake has run): the scene and its entities.</summary>
        public static event Action<SceneAsset?, List<Entity>>? Loaded;

        /// <summary>Replace the scene with <paramref name="scene"/> at the start of the next frame.</summary>
        public static void Load(SceneAsset scene)
        {
            _pending = scene ?? throw new ArgumentNullException(nameof(scene));
        }

        /// <summary>
        /// Replace the scene with <paramref name="scene"/> now: every entity not marked DontDestroy is destroyed,
        /// then the scene's entities are created. <paramref name="yaml"/> stands in for the scene's own YAML (the
        /// editor's play mode starts from the scene as it was in memory); with it, <paramref name="scene"/> may be
        /// null for a scene that is not an asset yet. Not from inside a component update: it destroys the entities
        /// being updated; use <see cref="Load"/> there.
        /// </summary>
        public static List<Entity> LoadNow(SceneAsset? scene, byte[]? yaml = null)
        {
            yaml ??= scene?.Yaml ?? throw new InvalidOperationException(
                scene == null ? "No scene and no YAML to load" : $"Scene '{scene.Name}' has no YAML");

            _pending = null;
            EntityManager.ClearScene();
            var entities = new EntitySerializer().LoadFromBytes(yaml);
            Current = scene;
            Loaded?.Invoke(scene, entities);
            return entities;
        }

        /// <summary>Engine.Tick: carry out a Load requested since the last frame.</summary>
        internal static void Update()
        {
            var scene = _pending;
            if (scene == null) return;
            _pending = null;
            LoadNow(scene);
        }
    }
}
