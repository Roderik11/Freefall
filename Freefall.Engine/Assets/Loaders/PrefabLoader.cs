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
    [AssetLoader(typeof(Prefab), ".prefab")]
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

        /// <summary>
        /// Hot reload: swap in the reimported YAML so future Instantiate() calls use it.
        /// Already-placed instances are entities and keep their component state.
        /// </summary>
        public bool Reload(Asset existing, string cachePath, string name, AssetManager manager, string guid)
        {
            if (existing is not Prefab prefab)
                return false;

            using (var stream = File.OpenRead(cachePath))
                prefab.SourceYaml = _packer.Read(stream).Yaml;
            return true;
        }

        /// <summary>
        /// Save a Prefab's SourceYaml back to its source .prefab file,
        /// then re-import so the binary cache stays in sync.
        /// </summary>
        public void Save(Asset asset, string savePath)
        {
            var prefab = (Prefab)asset;
            if (prefab.SourceYaml == null || prefab.SourceYaml.Length == 0)
            {
                Debug.LogWarning("PrefabLoader", $"Cannot save prefab '{prefab.Name}': no SourceYaml");
                return;
            }

            File.WriteAllBytes(savePath, prefab.SourceYaml);
            Debug.Log($"[PrefabLoader] Saved: {savePath}");

            // Re-import to update the binary cache (may fail for newly created files)
            var assetsDir = Engine.Project?.AssetsDirectory;
            if (!string.IsNullOrEmpty(assetsDir) && savePath.StartsWith(assetsDir))
            {
                var relativePath = Path.GetRelativePath(assetsDir, savePath).Replace('\\', '/');
                try { AssetDatabase.ImportAssetByPath(relativePath); }
                catch { /* File not yet tracked — caller will Refresh + Import */ }
            }
        }
    }
}
