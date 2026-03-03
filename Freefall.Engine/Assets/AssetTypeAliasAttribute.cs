using System;

namespace Freefall.Assets
{
    /// <summary>
    /// Declares alternate type name(s) that map to this Asset class.
    /// Used by the editor to resolve importer artifact type names
    /// (e.g. "MeshData", "DdsTextureData") to runtime types.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public class AssetTypeAliasAttribute : Attribute
    {
        public string Alias { get; }

        public AssetTypeAliasAttribute(string alias)
        {
            Alias = alias;
        }
    }
}
