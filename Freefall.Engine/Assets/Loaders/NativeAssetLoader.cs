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
            if (cachePath == null || !Artifacts.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for asset '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            AssetDefinitionData defData;
            using (var stream = Artifacts.Open(cachePath))
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
            using (var stream = Artifacts.Open(cachePath))
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

        // A member named …CacheFile holding "{guid}.{ext}": a file of the asset's own in the Cache root, which no meta lists
        // (the world map's tiles, WorldMap.TilesCacheFile)
        private static readonly System.Text.RegularExpressions.Regex CacheFileRef = new(
            @"^\s*\w*CacheFile:\s*""?([0-9a-fA-F]{32})\.(\w+)""?\s*$",
            System.Text.RegularExpressions.RegexOptions.Multiline);

        /// <summary>
        /// The Cache-root files the asset's members name (…CacheFile: "{guid}.{ext}"), read from its YAML: a published game
        /// packs them, and their GUIDs are no "unknown GUID" to the scan.
        /// </summary>
        public System.Collections.Generic.IEnumerable<ImplicitArtifact> GetImplicitDependencies(string guid)
        {
            var cachePath = AssetDatabase.ResolveCachePathByGuid(guid);
            if (cachePath == null || !Artifacts.Exists(cachePath)) yield break;

            AssetDefinitionData defData;
            using (var stream = Artifacts.Open(cachePath))
                defData = _packer.Read(stream);

            foreach (System.Text.RegularExpressions.Match m in CacheFileRef.Matches(Encoding.UTF8.GetString(defData.YamlBytes)))
            {
                var fileGuid = m.Groups[1].Value;
                var ext = m.Groups[2].Value;
                yield return new ImplicitArtifact(fileGuid, Path.Combine(AssetDatabase.Project.CacheDirectory, $"{fileGuid}.{ext}"), ext);
            }
        }

        public void Save(Asset asset, string savePath)
        {
            var yaml = NativeImporter.SaveToString(asset);
            File.WriteAllText(savePath, yaml, Encoding.UTF8);
            Debug.Log($"[NativeAssetLoader] Saved: {savePath}");
        }
    }
}
