using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Freefall.Assets;
using Freefall.Reflection;

namespace Freefall.Serialization
{
    /// <summary>
    /// Generic YAML reader/writer for .asset files.
    /// Can serialize/deserialize any Asset subclass using YAMLSerializer + Reflector.
    /// Checks for custom [AssetSerializer] implementations first, then falls back
    /// to the generic YAMLSerializer for standard reflection-based serialization.
    /// </summary>
    public static class NativeImporter
    {
        // ThreadLocal: each thread gets its own serializer instance so DeferAssetLoading
        // doesn't race between concurrent LoadFromString calls (e.g. streaming scene load).
        [ThreadStatic] private static YAMLSerializer _threadSerializer;
        private static YAMLSerializer Serializer => _threadSerializer ??= new YAMLSerializer();

        // Custom serializers discovered via [AssetSerializer] attribute
        private static readonly Dictionary<Type, IAssetSerializer> _customSerializers = new();
        private static bool _initialized;

        /// <summary>
        /// Discover all custom [AssetSerializer] implementations via reflection.
        /// </summary>
        private static void EnsureInitialized()
        {
            if (_initialized) return;
            _initialized = true;

            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                try
                {
                    foreach (var type in assembly.GetTypes())
                    {
                        var attr = type.GetCustomAttribute<AssetSerializerAttribute>();
                        if (attr == null) continue;

                        if (!typeof(IAssetSerializer).IsAssignableFrom(type))
                        {
                            Debug.LogWarning("NativeImporter",
                                $"[AssetSerializer] on {type.Name} but it doesn't implement IAssetSerializer");
                            continue;
                        }

                        var instance = (IAssetSerializer)Activator.CreateInstance(type);
                        _customSerializers[attr.AssetType] = instance;
                        Debug.Log($"[NativeImporter] Registered custom serializer: {type.Name} → {attr.AssetType.Name}");
                    }
                }
                catch (ReflectionTypeLoadException) { }
            }
        }

        /// <summary>
        /// Find a custom serializer for the given asset type, checking the full hierarchy.
        /// </summary>
        private static IAssetSerializer FindSerializer(Type assetType)
        {
            EnsureInitialized();

            // Walk up the type hierarchy to find a registered serializer
            var type = assetType;
            while (type != null && type != typeof(object))
            {
                if (_customSerializers.TryGetValue(type, out var serializer))
                    return serializer;
                type = type.BaseType;
            }
            return null;
        }

        /// <summary>
        /// Save any Asset subclass to a .asset YAML file.
        /// Format: type name header followed by field mapping.
        /// </summary>
        public static void Save(string path, Asset asset)
        {
            var yaml = SaveToString(asset);
            File.WriteAllText(path, yaml, Encoding.UTF8);
        }

        /// <summary>
        /// Serialize an Asset to a YAML string (for caching / testing).
        /// </summary>
        public static string SaveToString(Asset asset)
        {
            var customSerializer = FindSerializer(asset.GetType());
            if (customSerializer != null)
                return customSerializer.Serialize(asset);

            return Serializer.Serialize(asset);
        }

        /// <summary>
        /// Load any Asset subclass from a .asset YAML file.
        /// Resolves the concrete type from the embedded type name.
        /// </summary>
        public static Asset Load(string path)
        {
            var yaml = File.ReadAllText(path, Encoding.UTF8);
            return LoadFromString(yaml);
        }

        /// <summary>
        /// Load an Asset from a YAML string.
        /// </summary>
        public static Asset LoadFromString(string yaml)
        {
            Serializer.DeferAssetLoading = true;
            try
            {
                // Peek at the type tag to find a custom serializer
                var typeName = PeekTypeName(yaml);
                if (typeName != null)
                {
                    var type = ResolveAssetType(typeName);
                    if (type != null)
                    {
                        var customSerializer = FindSerializer(type);
                        if (customSerializer != null)
                            return customSerializer.Deserialize(yaml);

                        // No custom serializer — reformat !Tag YAML into wrapped format
                        // that YAMLSerializer.Deserialize expects: "TypeName:\n  fields"
                        var wrapped = ReformatTagYaml(yaml, type.Name);
                        return Serializer.Deserialize(wrapped) as Asset;
                    }
                }

                return Serializer.Deserialize(yaml) as Asset;
            }
            finally { Serializer.DeferAssetLoading = false; }
        }

        /// <summary>
        /// Convert !Tag format YAML into wrapped TypeName: { fields } format.
        /// Input:  "!StaticMesh\nName: Cage_01_V2\nMesh: abc123"
        /// Output: "StaticMesh:\n  Name: Cage_01_V2\n  Mesh: abc123"
        /// </summary>
        private static string ReformatTagYaml(string yaml, string typeName)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"{typeName}:");
            foreach (var rawLine in yaml.Split('\n'))
            {
                var line = rawLine.TrimEnd('\r');
                // Skip empty lines and the !Tag line
                if (string.IsNullOrWhiteSpace(line)) continue;
                if (line.TrimStart().StartsWith("!")) continue;
                // Indent all content lines under the type wrapper
                sb.AppendLine("  " + line);
            }
            return sb.ToString();
        }

        /// <summary>
        /// Resolve a short or fully-qualified type name to a Type.
        /// Uses Reflector for proper assembly-wide type resolution.
        /// </summary>
        private static Type ResolveAssetType(string name)
        {
            if (name.StartsWith("!"))
                name = name.Substring(1);

            // Try fully qualified first
            var type = Reflector.GetType(name);
            if (type != null) return type;

            // Try with "Freefall." prefix (e.g. "Components.Transform" → "Freefall.Components.Transform")
            type = Reflector.GetType("Freefall." + name);
            if (type != null) return type;

            // Simple name (e.g. "StaticMesh") → scan all assemblies
            type = Reflector.FindTypeBySimpleName(name);
            return type;
        }

        /// <summary>
        /// Extract the type name from YAML header (e.g. "!Material" or "--- !StaticMesh").
        /// </summary>
        private static string PeekTypeName(string yaml)
        {
            // Look for !TypeName pattern in the first line(s)
            var lines = yaml.Split('\n', 3);
            foreach (var line in lines)
            {
                var trimmed = line.Trim();
                if (trimmed.StartsWith("!"))
                    return trimmed.Substring(1).Trim();
                if (trimmed.Contains("!"))
                {
                    var idx = trimmed.IndexOf('!');
                    var rest = trimmed.Substring(idx + 1).Trim();
                    if (rest.Length > 0)
                        return rest;
                }
            }
            return null;
        }
    }
}
