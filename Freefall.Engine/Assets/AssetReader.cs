using System;

namespace Freefall.Assets
{
    /// <summary>
    /// Marks a class as an asset reader for specific file extensions.
    /// Like Apex's AssetReaderAttribute.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class AssetReaderAttribute : Attribute
    {
        public string[] Extensions { get; }

        public AssetReaderAttribute(params string[] extensions)
        {
            Extensions = extensions;
        }
    }

    /// <summary>
    /// Base class for asset readers that import assets from files.
    /// Like Apex's AssetReader<T>.
    /// </summary>
    public abstract class AssetReader<T> where T : Asset
    {
        public abstract T Import(string filepath);
    }
}
