using System;

namespace Freefall.Assets
{
    /// <summary>
    /// Attribute marking a class as a runtime asset loader for a specific Asset type.
    /// Discovered by AssetManager at startup.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class AssetLoaderAttribute : Attribute
    {
        public Type AssetType { get; }

        public AssetLoaderAttribute(Type assetType)
        {
            AssetType = assetType;
        }
    }

    /// <summary>
    /// Runtime asset loader interface. Loads assets from cache files
    /// and produces GPU-ready runtime objects.
    ///
    /// Loaders are the runtime counterpart to Importers:
    ///   Importer: source file → engine data → cache (editor-only)
    ///   Loader:   cache → GPU runtime objects (both editor + game)
    /// </summary>
    public interface IAssetLoader
    {
        /// <summary>
        /// Load an asset by name. Resolves cache path via AssetDatabase,
        /// unpacks binary data via the appropriate packer, and creates
        /// the GPU-ready runtime object.
        /// </summary>
        /// <param name="name">Asset name (matches SubAsset name in meta files)</param>
        /// <param name="manager">AssetManager for resolving dependent assets by GUID</param>
        Asset Load(string name, AssetManager manager);

        /// <summary>
        /// Load an asset directly from a resolved cache file path.
        /// Default implementation falls back to name-based Load.
        /// </summary>
        Asset LoadFromCache(string cachePath, string name, AssetManager manager)
            => Load(name, manager);
    }
}
