using System;

namespace Freefall.Assets
{
    /// <summary>
    /// Marks a class as a custom asset serializer for a specific Asset type.
    /// Discovered via reflection by NativeImporter at startup.
    /// Same pattern as [AssetImporter] for importers.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class AssetSerializerAttribute : Attribute
    {
        public Type AssetType { get; }

        public AssetSerializerAttribute(Type assetType)
        {
            AssetType = assetType;
        }
    }

    /// <summary>
    /// Interface for custom .asset serializers that override the default
    /// YAMLSerializer for types requiring hand-written logic (e.g. Material).
    /// </summary>
    public interface IAssetSerializer
    {
        string Serialize(Asset asset);
        Asset Deserialize(string yaml);
    }
}
