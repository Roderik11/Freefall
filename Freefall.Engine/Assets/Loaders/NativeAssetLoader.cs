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

        public void Save(Asset asset, string savePath)
        {
            var yaml = NativeImporter.SaveToString(asset);
            File.WriteAllText(savePath, yaml, Encoding.UTF8);
            Debug.Log($"[NativeAssetLoader] Saved: {savePath}");
        }
    }
}
