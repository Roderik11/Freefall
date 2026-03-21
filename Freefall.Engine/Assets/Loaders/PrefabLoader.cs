using System.IO;
using Freefall.Assets;
using Freefall.Assets.Packers;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads Prefab assets from cache (.prefab files packed by PrefabPacker).
    /// Unpacks PrefabData (raw YAML bytes) and stores them on the Prefab
    /// for re-parsing on each Instantiate() call.
    /// </summary>
    [AssetLoader(typeof(Prefab))]
    public class PrefabLoader : IAssetLoader
    {
        private readonly PrefabPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, "PrefabData");
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for prefab '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            PrefabData data;
            using (var stream = File.OpenRead(cachePath))
                data = _packer.Read(stream);

            var prefab = new Prefab
            {
                SourceYaml = data.Yaml,
                Name = name,
                RootEntityName = name,
            };

            prefab.MarkReady();
            return prefab;
        }
    }
}
