using System;

namespace Freefall.Assets
{
    // Backward compatibility aliases — will be removed once all code is migrated.
    
    /// <summary>
    /// Deprecated: Use AssetImporterAttribute instead.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class AssetReaderAttribute : AssetImporterAttribute
    {
        public AssetReaderAttribute(params string[] extensions) : base(extensions) { }
    }
}
