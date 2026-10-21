using System.IO;
using Freefall.Assets.Packers;
using Freefall.Graphics;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads a project's .fx asset (EffectImporter): its cached text compiled by <see cref="Effect.FromAsset"/>, one
    /// effect per GUID. The engine's own effects are not loaded here: they are InternalAssets with fixed GUIDs.
    /// </summary>
    [AssetLoader(typeof(Effect), ".fx")]
    public class EffectLoader : IAssetLoader
    {
        private readonly EffectSourcePacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, nameof(EffectSourceData));
            if (cachePath == null || !Artifacts.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for effect '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            // A cache file is named by its GUID: no lookup by name
            var guid = sourceGuid ?? Path.GetFileNameWithoutExtension(cachePath);
            return Effect.FromAsset(guid, name, Read(cachePath).Source, SourcePathOf(guid));
        }

        /// <summary>
        /// The file was reimported (asset_refresh, a save that hot reload already compiled): compile the cached
        /// text if it is not the one running, then rebuild the materials using it.
        /// </summary>
        public bool Reload(Asset existing, string cachePath, string name, AssetManager manager, string guid)
        {
            if (existing is not Effect effect) return false;
            if (effect.TryReloadFrom(Read(cachePath).Source))
                Material.RebuildForEffect(effect);
            return true;
        }

        private EffectSourceData Read(string cachePath)
        {
            using var stream = Artifacts.Open(cachePath);
            return _packer.Read(stream);
        }

        // The .fx in Assets, where its own includes are; none when the project ships without its sources
        private static string SourcePathOf(string guid)
        {
            var path = AssetDatabase.GuidToFullPath(guid);
            return path != null && File.Exists(path) ? path : null;
        }
    }
}
