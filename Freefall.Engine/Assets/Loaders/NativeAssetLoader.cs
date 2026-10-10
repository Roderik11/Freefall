using System.IO;
using System.Text;
using Freefall.Assets.Packers;
using Freefall.Serialization;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// General-purpose loader for YAML-serialized .asset files.
    /// Handles any Asset type (NodeGraph, Animation, etc.).
    /// If the loaded asset implements IRebuildAfterLoad, calls
    /// RebuildAfterLoad() to resolve runtime references.
    /// </summary>
    [AssetLoader(typeof(Asset), ".asset")]
    public class NativeAssetLoader : IAssetLoader
    {
        private readonly AssetDefinitionPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, "AssetDefinitionData");
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for asset '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            AssetDefinitionData defData;
            using (var stream = File.OpenRead(cachePath))
                defData = _packer.Read(stream);

            var yaml = Encoding.UTF8.GetString(defData.YamlBytes);
            var asset = NativeImporter.LoadFromString(yaml, manager);

            if (asset == null)
                throw new InvalidDataException($"Failed to deserialize asset from cache: {name}");

            asset.Name = name;

            if (asset is IRebuildAfterLoad rebuildable)
                rebuildable.RebuildAfterLoad();

            asset.MarkReady();
            return asset;
        }

        /// <summary>
        /// Hot reload: read the reimported YAML into the loaded instance, so every component and asset that
        /// references it sees the new data. Its lists and nested objects are replaced; whoever built something
        /// from them (inspector controls, generated entities) hears about it through "AssetReloaded".
        /// </summary>
        public bool Reload(Asset existing, string cachePath, string name, AssetManager manager, string guid)
        {
            string yaml;
            using (var stream = File.OpenRead(cachePath))
                yaml = Encoding.UTF8.GetString(_packer.Read(stream).YamlBytes);

            // The reimport that follows saving this very instance: it already holds that state, and reading
            // it back would only swap the objects the inspector or a graph editor is still editing.
            if (NativeImporter.SaveToString(existing) == yaml)
                return false;

            if (!NativeImporter.LoadInto(existing, yaml, manager))
                return false;

            existing.Name = name;
            Freefall.Base.MessageDispatcher.Send("AssetReloaded", existing);
            return true;
        }

        public void Save(Asset asset, string savePath)
        {
            var yaml = NativeImporter.SaveToString(asset);
            File.WriteAllText(savePath, yaml, Encoding.UTF8);
            Debug.Log($"[NativeAssetLoader] Saved: {savePath}");
        }
    }
}
