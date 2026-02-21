using System;
using System.IO;

namespace Freefall.Assets
{
    /// <summary>
    /// Marks a class as an asset packer and specifies its cache file extension.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class AssetPackerAttribute : Attribute
    {
        public string CacheExtension { get; }

        public AssetPackerAttribute(string cacheExtension)
        {
            CacheExtension = cacheExtension;
        }
    }
    /// <summary>
    /// Base class for binary asset packers. Converts assets to/from fast binary format
    /// for Cache storage. Each packer includes a version header for format evolution.
    /// </summary>
    public abstract class AssetPacker<T>
    {
        /// <summary>
        /// Format version. Increment when the binary layout changes.
        /// </summary>
        public abstract int Version { get; }

        /// <summary>
        /// Write asset data (without version header).
        /// </summary>
        public abstract void Pack(BinaryWriter writer, T asset);

        /// <summary>
        /// Read asset data (without version header).
        /// </summary>
        public abstract T Unpack(BinaryReader reader, int version);

        /// <summary>
        /// Write version header + asset data.
        /// </summary>
        public void Write(Stream stream, T asset)
        {
            using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
            writer.Write(Version);
            Pack(writer, asset);
        }

        /// <summary>
        /// Read version header + asset data.
        /// </summary>
        public T Read(Stream stream)
        {
            using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
            int version = reader.ReadInt32();
            return Unpack(reader, version);
        }
    }
}
