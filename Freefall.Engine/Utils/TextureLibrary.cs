using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace Freefall
{
    /// <summary>
    /// Resolves raw texture paths (PNG/TGA) to pre-mipmapped DDS files
    /// from the Spark Library/Packed cache via Library/Meta GUID lookups.
    /// </summary>
    public static class TextureLibrary
    {
        private static readonly string MetaDir = @"D:\Projects\2024\ProjectXYZ\Library\Meta";
        private static readonly string PackedDir = @"D:\Projects\2024\ProjectXYZ\Library\Packed";
        private static readonly string ResourcesRoot = @"D:\Projects\2024\ProjectXYZ\Resources\";

        // Normalized relative path → GUID
        private static Dictionary<string, string> _pathToGuid;

        /// <summary>
        /// Parses all .meta files and builds the path→GUID lookup.
        /// Call once at startup or before first scene load.
        /// </summary>
        public static void Initialize()
        {
            _pathToGuid = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

            foreach (var metaFile in Directory.EnumerateFiles(MetaDir, "*.meta"))
            {
                try
                {
                    var json = File.ReadAllText(metaFile);
                    using var doc = JsonDocument.Parse(json);
                    var root = doc.RootElement;

                    // Only process texture assets
                    if (root.TryGetProperty("AssetReader", out var reader) &&
                        reader.TryGetProperty("$type", out var typeEl) &&
                        typeEl.GetString() == "!TextureReader" &&
                        root.TryGetProperty("Path", out var pathEl) &&
                        root.TryGetProperty("Guid", out var guidEl))
                    {
                        var relativePath = pathEl.GetString();
                        var guid = guidEl.GetString();

                        if (!string.IsNullOrEmpty(relativePath) && !string.IsNullOrEmpty(guid))
                        {
                            // Normalize to forward slashes for consistent lookup
                            var normalized = relativePath.Replace('/', '\\');
                            _pathToGuid[normalized] = guid;
                        }
                    }
                }
                catch
                {
                    // Skip malformed meta files
                }
            }

            Debug.Log($"[TextureLibrary] Initialized — {_pathToGuid.Count} texture mappings loaded.");
        }

        /// <summary>
        /// Given an absolute path to a raw texture (PNG/TGA), returns the path
        /// to the pre-mipmapped DDS in Library/Packed, or null if not found.
        /// </summary>
        public static string ResolvePackedDDS(string absoluteTexturePath)
        {
            if (_pathToGuid == null) return null;

            // Convert absolute path to relative path under Resources
            string relativePath;
            if (absoluteTexturePath.StartsWith(ResourcesRoot, StringComparison.OrdinalIgnoreCase))
            {
                relativePath = absoluteTexturePath.Substring(ResourcesRoot.Length);
            }
            else
            {
                return null;
            }

            // Normalize
            relativePath = relativePath.Replace('/', '\\');

            if (_pathToGuid.TryGetValue(relativePath, out var guid))
            {
                var ddsPath = Path.Combine(PackedDir, guid + ".dds");
                if (File.Exists(ddsPath))
                    return ddsPath;
            }

            return null;
        }
    }
}
