using System.IO;
using Freefall.Assets;
using Freefall.Assets.Packers;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads SceneAssets from cache (.scene files packed by ScenePacker). Only the YAML is read here;
    /// SceneManager creates the entities when the scene is loaded.
    /// </summary>
    [AssetLoader(typeof(SceneAsset), ".scene")]
    public class SceneLoader : IAssetLoader
    {
        private readonly ScenePacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, nameof(SceneData));
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for scene '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            SceneData data;
            using (var stream = File.OpenRead(cachePath))
                data = _packer.Read(stream);

            var scene = new SceneAsset
            {
                Yaml = data.Yaml,
                Name = name,
            };

            scene.MarkReady();
            return scene;
        }

        /// <summary>
        /// Hot reload: swap in the reimported YAML (a saved scene, a file replaced on disk), so the next load
        /// of this scene builds what the file now holds. The entities of a scene already loaded are not touched.
        /// </summary>
        public bool Reload(Asset existing, string cachePath, string name, AssetManager manager, string guid)
        {
            if (existing is not SceneAsset scene)
                return false;

            using (var stream = File.OpenRead(cachePath))
                scene.Yaml = _packer.Read(stream).Yaml;
            return true;
        }
    }
}
