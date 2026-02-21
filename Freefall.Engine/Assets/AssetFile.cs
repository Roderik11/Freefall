using System.IO;
using Freefall.Serialization;

namespace Freefall.Assets
{
    /// <summary>
    /// Reads and writes .asset files (YAML-serialized Asset definitions).
    /// Delegates to NativeImporter for all serialization.
    /// These are editor-created composite assets like StaticMesh, Terrain, etc.
    /// </summary>
    public static class AssetFile
    {
        /// <summary>
        /// Save an asset definition to a .asset YAML file.
        /// </summary>
        public static void Save(string path, Asset asset)
        {
            NativeImporter.Save(path, asset);
        }

        /// <summary>
        /// Load an asset definition from a .asset YAML file.
        /// Returns null if the type is unknown.
        /// </summary>
        public static Asset Load(string path)
        {
            return NativeImporter.Load(path);
        }
    }
}
