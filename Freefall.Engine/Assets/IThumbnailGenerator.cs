using System;

namespace Freefall.Assets
{
    /// <summary>
    /// Attribute marking a class as a thumbnail generator for a specific Asset type.
    /// The asset type should be the runtime type (e.g. typeof(Mesh), typeof(Prefab)).
    /// Discovered at startup via reflection (same pattern as AssetLoaderAttribute).
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public class ThumbnailGeneratorAttribute : Attribute
    {
        public Type AssetType { get; }

        public ThumbnailGeneratorAttribute(Type assetType)
        {
            AssetType = assetType;
        }
    }

    /// <summary>
    /// Interface for type-specific thumbnail generators.
    /// Implementations are stateless — the shared ThumbnailRenderer (GPU context)
    /// is passed in by the caller. Generators should not create GPU resources.
    /// </summary>
    public interface IThumbnailGenerator
    {
        /// <summary>
        /// Generate a thumbnail for the given asset GUID.
        /// The renderer parameter provides shared GPU rendering context.
        /// Returns true if the thumbnail was created successfully.
        /// </summary>
        bool Generate(string guid, AssetManager assets, object renderer);
    }
}
