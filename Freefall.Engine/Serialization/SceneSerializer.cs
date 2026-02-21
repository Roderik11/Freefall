using System;
using System.Buffers;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Text;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Reflection;
using LiteYaml.Emitter;

namespace Freefall.Serialization
{
    /// <summary>
    /// Serializes entities and their components to .scene YAML files.
    /// Format: multi-document YAML, one document per entity.
    /// </summary>
    public class SceneSerializer
    {
        private static readonly Dictionary<string, IYAMLConverter> _converters;

        static SceneSerializer()
        {
            // Share the converter registry from YAMLSerializer
            _converters = YAMLSerializer.Converters;
        }

        /// <summary>
        /// Save entities to a .scene file.
        /// </summary>
        public void Save(string path, IEnumerable<Entity> entities)
        {
            var yaml = SaveToString(entities);
            File.WriteAllText(path, yaml, Encoding.UTF8);
        }

        /// <summary>
        /// Serialize entities to a YAML string (for testing/debugging).
        /// </summary>
        public string SaveToString(IEnumerable<Entity> entities)
        {
            var sb = new StringBuilder();

            foreach (var entity in entities)
            {
                sb.AppendLine("--- !Entity");
                SerializeEntity(entity, sb);
            }

            return sb.ToString();
        }

        private void SerializeEntity(Entity entity, StringBuilder sb)
        {
            sb.AppendLine($"Name: {entity.Name}");
            sb.AppendLine("Components:");

            foreach (var component in entity.Components)
            {
                var type = component.GetType();
                var typeName = ShortenTypeName(type.FullName!);
                sb.AppendLine($"  - {typeName}:");

                var mapping = Reflector.GetMapping(type);
                foreach (var field in mapping)
                {
                    SerializeField(field, component, sb);
                }
            }
        }

        private void SerializeField(Field field, object parent, StringBuilder sb)
        {
            if (!field.CanWrite) return;
            if (field.Ignored) return;

            // Skip component infrastructure fields
            if (field.Name == "Entity" || field.Name == "Enabled") return;

            var value = field.GetValue(parent);
            if (value == null) return;

            // Skip defaults
            var defaultValue = field.GetAttribute<DefaultValueAttribute>();
            if (defaultValue != null && Equals(value, defaultValue.Value)) return;

            // Asset reference → GUID
            if (typeof(Asset).IsAssignableFrom(field.Type))
            {
                if (value is Asset asset && !string.IsNullOrEmpty(asset.Guid))
                    sb.AppendLine($"      {field.Name}: {asset.Guid}");
                return;
            }

            // Converter (primitives, vectors, etc.)
            if (_converters.TryGetValue(field.Type.FullName!, out var converter))
            {
                var buf = new ArrayBufferWriter<byte>(64);
                var emitter = new Utf8YamlEmitter(buf);
                converter.Write(value, ref emitter);
                var yamlValue = Encoding.UTF8.GetString(buf.WrittenSpan).Trim();
                sb.AppendLine($"      {field.Name}: {yamlValue}");
                return;
            }

            // Enum → int
            if (field.Type.IsEnum)
            {
                sb.AppendLine($"      {field.Name}: {(int)value}");
                return;
            }
        }

        private static string ShortenTypeName(string fullName)
        {
            if (fullName.StartsWith("Freefall."))
            {
                var lastDot = fullName.LastIndexOf('.');
                return lastDot >= 0 ? fullName[(lastDot + 1)..] : fullName;
            }
            return fullName;
        }
    }
}
