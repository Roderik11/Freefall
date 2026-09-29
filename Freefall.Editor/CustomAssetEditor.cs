using System;

namespace Freefall.Editor
{
    /// <summary>
    /// Marks a class as a custom editor for a specific asset type.
    /// When a user double-clicks an asset in the browser, the editor
    /// discovers the matching custom editor and opens it.
    /// 
    /// The decorated class must implement ICustomAssetEditor.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class CustomAssetEditorAttribute : Attribute
    {
        public Type AssetType { get; }

        public CustomAssetEditorAttribute(Type assetType)
        {
            AssetType = assetType;
        }
    }

    /// <summary>
    /// Interface for custom asset editors opened via double-click in the asset browser.
    /// </summary>
    public interface ICustomAssetEditor
    {
        /// <summary>
        /// Open an asset for editing. The asset has already been loaded.
        /// </summary>
        void OpenAsset(Freefall.Assets.Asset asset);
    }
}
