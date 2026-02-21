using System;
using System.Buffers;
using System.Collections;
using System.ComponentModel;
using System.Numerics;
using System.Text;
using Freefall.Assets;
using Freefall.Reflection;
using LiteYaml;
using LiteYaml.Emitter;
using LiteYaml.Parser;

namespace Freefall.Serialization
{
    /// <summary>
    /// YAML serializer/deserializer for components and entities.
    /// Handles field-by-field serialization with type converters,
    /// asset references as GUIDs, enum-as-int, [DefaultValue] skip,
    /// [Serializable] nested objects, and list serialization.
    /// </summary>
    public class YAMLSerializer
    {
        /// <summary>
        /// When true, asset references create stub objects with GUID only (no LoadByGuid).
        /// Used during cache file deserialization to prevent recursive loading.
        /// </summary>
        public bool DeferAssetLoading { get; set; }
        private static readonly Dictionary<string, IYAMLConverter> _converters = new();
        internal static Dictionary<string, IYAMLConverter> Converters => _converters;

        private ArrayBufferWriter<byte> _bufferWriter;

        static YAMLSerializer()
        {
            _converters.Add(typeof(int).FullName!, new IntYAMLConverter());
            _converters.Add(typeof(float).FullName!, new FloatYAMLConverter());
            _converters.Add(typeof(double).FullName!, new DoubleYAMLConverter());
            _converters.Add(typeof(bool).FullName!, new BoolYAMLConverter());
            _converters.Add(typeof(string).FullName!, new StringYAMLConverter());
            _converters.Add(typeof(Vector2).FullName!, new Vector2YAMLConverter());
            _converters.Add(typeof(Vector3).FullName!, new Vector3YAMLConverter());
            _converters.Add(typeof(Vector4).FullName!, new Vector4YAMLConverter());
            _converters.Add(typeof(Quaternion).FullName!, new QuaternionYAMLConverter());
            _converters.Add(typeof(Vortice.Mathematics.Color3).FullName!, new Color3YAMLConverter());
        }

        /// <summary>
        /// Register a custom converter for a specific type.
        /// </summary>
        public static void RegisterConverter<T>(IYAMLConverter converter)
        {
            _converters[typeof(T).FullName!] = converter;
        }

        // ── Emitter lifecycle ─────────────────────────────────────

        /// <summary>
        /// Begin a new YAML emission buffer. Use for multi-document scene files.
        /// </summary>
        public Utf8YamlEmitter Begin(YamlEmitterOptions? options = null)
        {
            _bufferWriter = new ArrayBufferWriter<byte>(256);
            return new Utf8YamlEmitter(_bufferWriter, options);
        }

        /// <summary>
        /// Get the final YAML string from the emission buffer.
        /// </summary>
        public string ToString(in Utf8YamlEmitter emitter)
            => Encoding.UTF8.GetString(_bufferWriter.WrittenSpan);

        // ── Serialize ─────────────────────────────────────────────

        /// <summary>
        /// Serialize a single object to a standalone YAML string.
        /// Output: { TypeName: { field1: val, ... } }
        /// </summary>
        public string Serialize(object obj, YamlEmitterOptions? options = null)
        {
            if (obj == null) return null;

            var bufferWriter = new ArrayBufferWriter<byte>(256);
            var emitter = new Utf8YamlEmitter(bufferWriter, options);

            Serialize(obj, ref emitter);

            return Encoding.UTF8.GetString(bufferWriter.WrittenSpan);
        }

        /// <summary>
        /// Serialize an object into an existing emitter (for multi-doc scenes).
        /// Writes: { ShortTypeName: { fields... } }
        /// </summary>
        public void Serialize(object obj, ref Utf8YamlEmitter emitter)
        {
            if (obj == null) return;

            var type = obj.GetType();
            var typeName = ShortenTypeName(type.FullName!);

            emitter.BeginMapping();
            emitter.WriteString(typeName);
            SerializeObject(obj, ref emitter);
            emitter.EndMapping();
        }

        private void SerializeObject(object obj, ref Utf8YamlEmitter emitter)
        {
            if (obj == null)
            {
                emitter.WriteNull();
                return;
            }

            var type = obj.GetType();
            var mapping = Reflector.GetMapping(type);

            emitter.BeginMapping();
            foreach (var field in mapping)
                SerializeField(field, obj, ref emitter);
            emitter.EndMapping();
        }

        private void SerializeField(Field field, object parent, ref Utf8YamlEmitter emitter)
        {
            if (!field.CanWrite) return;
            if (field.Ignored) return;

            var value = field.GetValue(parent);
            if (value == null) return;

            var valueType = value.GetType();

            // ── Asset reference → GUID ──
            if (typeof(Asset).IsAssignableFrom(field.Type))
            {
                if (value is Asset asset && !string.IsNullOrEmpty(asset.Guid))
                {
                    emitter.WriteString(field.Name);
                    emitter.WriteString(asset.Guid);
                }
                return;
            }

            // ── List/Array ──
            if (typeof(IList).IsAssignableFrom(field.Type))
            {
                SerializeList(field, value as IList, ref emitter);
                return;
            }

            // ── Converter (primitives, vectors, etc.) ──
            if (_converters.TryGetValue(field.Type.FullName!, out var converter))
            {
                var defaultValue = field.GetAttribute<DefaultValueAttribute>();
                if (defaultValue == null || !Equals(value, defaultValue.Value))
                {
                    WriteFieldName(field, valueType, ref emitter);
                    converter.Write(value, ref emitter);
                }
                return;
            }

            // ── Enum → int ──
            if (field.Type.IsEnum)
            {
                var defaultValue = field.GetAttribute<DefaultValueAttribute>();
                if (defaultValue == null || !Equals(value, defaultValue.Value))
                {
                    WriteFieldName(field, valueType, ref emitter);
                    emitter.WriteInt32((int)value);
                }
                return;
            }

            // ── Nested [Serializable] object ──
            if (field.Type.HasAttribute<SerializableAttribute>(true))
            {
                WriteFieldName(field, valueType, ref emitter);
                SerializeObject(value, ref emitter);
                return;
            }
        }

        private void SerializeList(Field field, IList list, ref Utf8YamlEmitter emitter)
        {
            if (list == null || list.Count == 0) return;

            // Skip multidimensional arrays
            if (field.Type.IsArray && field.Type.GetArrayRank() > 1)
                return;

            var eltype = GetElementType(field.Type);
            if (eltype == null) return;

            // List of assets → list of GUIDs
            if (typeof(Asset).IsAssignableFrom(eltype))
            {
                emitter.WriteString(field.Name);
                emitter.BeginSequence();
                for (int i = 0; i < list.Count; i++)
                {
                    var asset = list[i] as Asset;
                    emitter.WriteString(asset?.Guid ?? "");
                }
                emitter.EndSequence();
                return;
            }

            // List of primitives/converters
            if (_converters.TryGetValue(eltype.FullName!, out var converter))
            {
                emitter.WriteString(field.Name);
                emitter.BeginSequence();
                for (int i = 0; i < list.Count; i++)
                    converter.Write(list[i], ref emitter);
                emitter.EndSequence();
                return;
            }

            // List of [Serializable] objects
            if (Reflector.GetAttribute<SerializableAttribute>(eltype) != null)
            {
                emitter.WriteString(field.Name);
                emitter.BeginSequence();
                for (int i = 0; i < list.Count; i++)
                {
                    SerializeElement(eltype, list[i], ref emitter);
                }
                emitter.EndSequence();
                return;
            }
        }

        private void SerializeElement(Type baseType, object element, ref Utf8YamlEmitter emitter)
        {
            if (element == null)
            {
                emitter.WriteNull();
                return;
            }

            var valueType = element.GetType();
            var mapping = Reflector.GetMapping(valueType);

            emitter.BeginMapping();
            emitter.WriteString(valueType.FullName!);
            emitter.BeginMapping();
            foreach (var field in mapping)
                SerializeField(field, element, ref emitter);
            emitter.EndMapping();
            emitter.EndMapping();
        }

        // ── Deserialize ───────────────────────────────────────────

        /// <summary>
        /// Deserialize a YAML string to a specific type T.
        /// Expects format: { TypeName: { fields... } }
        /// </summary>
        public T Deserialize<T>(string text)
        {
            var bytes = Encoding.UTF8.GetBytes(text);
            var parser = YamlParser.FromBytes(bytes);

            var type = typeof(T);
            var fields = Reflector.GetMapping(type);
            var instance = Activator.CreateInstance<T>();

            parser.SkipAfter(ParseEventType.DocumentStart);
            parser.SkipAfter(ParseEventType.MappingStart);
            parser.ReadScalarAsString(); // type name

            while (!parser.End)
                ReadNext(ref parser, instance, fields);

            return instance;
        }

        /// <summary>
        /// Deserialize a YAML string to an object, using the embedded type name.
        /// </summary>
        public object Deserialize(string text)
        {
            var bytes = Encoding.UTF8.GetBytes(text);
            var parser = YamlParser.FromBytes(bytes);

            parser.SkipAfter(ParseEventType.DocumentStart);
            parser.SkipAfter(ParseEventType.MappingStart);
            var typeName = parser.ReadScalarAsString();
            var fullName = ExpandTypeName(typeName);
            var type = Reflector.GetType(fullName);

            if (type == null) return null;

            var fields = Reflector.GetMapping(type);
            var instance = Activator.CreateInstance(type);

            while (parser.CurrentEventType != ParseEventType.MappingEnd && !parser.End)
                ReadNext(ref parser, instance, fields);

            return instance;
        }

        /// <summary>
        /// Deserialize a multi-document YAML stream (e.g. a .scene file).
        /// Each --- document produces one deserialized object.
        /// </summary>
        public List<object> DeserializeAll(byte[] bytes)
        {
            var text = Encoding.UTF8.GetString(bytes).TrimStart('\uFEFF');
            var results = new List<object>();
            var sb = new StringBuilder();

            foreach (var rawLine in text.Split('\n'))
            {
                var line = rawLine.TrimEnd('\r');
                if (line.StartsWith("---"))
                {
                    if (sb.Length > 0)
                    {
                        try
                        {
                            var obj = Deserialize(sb.ToString());
                            if (obj != null) results.Add(obj);
                        }
                        catch (Exception ex)
                        {
                            if (results.Count < 5)
                                Debug.Log($"[DeserializeAll] Error deserializing block:\n{sb}\nException: {ex}");
                        }
                        sb.Clear();
                    }
                    continue;
                }
                sb.AppendLine(line);
            }

            // Flush last block
            if (sb.Length > 0)
            {
                try
                {
                    var obj = Deserialize(sb.ToString());
                    if (obj != null) results.Add(obj);
                }
                catch { }
            }

            return results;
        }

        private object DeserializeNested(ref YamlParser parser, Type type)
        {
            if (parser.TryGetCurrentTag(out var tag))
            {
                var typename = ExpandTypeName(tag.ToString());
                type = Reflector.GetType(typename);
            }

            if (type == null)
            {
                parser.SkipCurrentNode();
                return null;
            }

            var fields = Reflector.GetMapping(type);
            var value = Activator.CreateInstance(type);

            while (parser.CurrentEventType != ParseEventType.MappingEnd)
            {
                if (parser.CurrentEventType == ParseEventType.MappingStart)
                {
                    parser.Read(); // consume MappingStart
                    continue;
                }

                if (parser.CurrentEventType == ParseEventType.Scalar)
                {
                    // Peek at the key — could be a type discriminator or a field name
                    var key = parser.ReadScalarAsString();

                    // Type discriminator: "Freefall.Assets.MeshElement" followed by inner mapping
                    if (key != null && key.Contains('.'))
                    {
                        var resolvedType = Reflector.GetType(key);
                        if (resolvedType != null && type.IsAssignableFrom(resolvedType))
                        {
                            type = resolvedType;
                            fields = Reflector.GetMapping(type);
                            value = Activator.CreateInstance(type);
                        }

                        // Read the inner mapping containing the actual fields
                        if (parser.CurrentEventType == ParseEventType.MappingStart)
                        {
                            parser.Read(); // consume MappingStart
                            while (parser.CurrentEventType != ParseEventType.MappingEnd)
                                ReadNext(ref parser, value, fields);
                            parser.Read(); // consume MappingEnd
                        }
                        else
                        {
                            parser.SkipCurrentNode();
                        }
                        continue;
                    }

                    // Regular field name — process value using ReadNode logic
                    if (key != null && fields.TryGetValue(key, out var field))
                    {
                        ReadNodeValue(ref parser, value, field);
                    }
                    else
                    {
                        parser.SkipCurrentNode();
                    }
                    continue;
                }

                // Fallback
                ReadNext(ref parser, value, fields);
            }

            return value;
        }

        private void ReadNext(ref YamlParser parser, object instance, Mapping fields)
        {
            switch (parser.CurrentEventType)
            {
                case ParseEventType.MappingStart:
                    parser.Read();
                    ReadNode(ref parser, instance, fields);
                    break;
                case ParseEventType.MappingEnd:
                    parser.Read();
                    break;
                case ParseEventType.SequenceEnd:
                    parser.Read();
                    break;
                case ParseEventType.Scalar:
                    ReadNode(ref parser, instance, fields);
                    break;
                case ParseEventType.DocumentStart:
                    parser.SkipAfter(ParseEventType.DocumentStart);
                    break;
                default:
                    parser.Read();
                    break;
            }
        }

        private void ReadNode(ref YamlParser parser, object parent, Mapping fields)
        {
            if (parser.CurrentEventType != ParseEventType.Scalar)
                throw new InvalidOperationException($"Expected scalar field name, got {parser.CurrentEventType}");

            var key = parser.ReadScalarAsString();

            if (!fields.TryGetValue(key, out var field))
            {
                parser.SkipCurrentNode();
                return;
            }

            ReadNodeValue(ref parser, parent, field);
        }

        /// <summary>
        /// Read and set a field value — key already consumed, parser is at the value token.
        /// </summary>
        private void ReadNodeValue(ref YamlParser parser, object parent, Field field)
        {
            var type = field.Type;
            if (parser.TryGetCurrentTag(out var tag))
            {
                var typename = ExpandTypeName(tag.ToString());
                var resolved = Reflector.GetType(typename);
                if (resolved != null) type = resolved;
            }

            if (type == null)
            {
                parser.SkipCurrentNode();
                return;
            }

            // ── Converter (primitives, vectors, etc.) ──
            if (_converters.TryGetValue(field.Type.FullName!, out var converter))
            {
                field.SetValue(parent, converter.Read(ref parser));
                return;
            }

            // ── Enum ──
            if (field.Type.IsEnum)
            {
                field.SetValue(parent, parser.ReadScalarAsInt32());
                return;
            }

            // ── List/Array ──
            if (typeof(IList).IsAssignableFrom(field.Type))
            {
                DeserializeList(ref parser, parent, field, type);
                return;
            }

            // ── Asset reference (GUID string or {guid: xxx}) ──
            if (typeof(Asset).IsAssignableFrom(field.Type))
            {
                string guidValue = null;

                if (parser.CurrentEventType == ParseEventType.Scalar)
                {
                    guidValue = parser.ReadScalarAsString();
                }
                else if (parser.CurrentEventType == ParseEventType.MappingStart)
                {
                    parser.Read();
                    while (parser.CurrentEventType != ParseEventType.MappingEnd && !parser.End)
                    {
                        var flowKey = parser.ReadScalarAsString();
                        if (flowKey == "guid")
                            guidValue = parser.ReadScalarAsString()?.Trim();
                        else
                            parser.SkipCurrentNode();
                    }
                    parser.Read();
                }
                else
                {
                    parser.SkipCurrentNode();
                    return;
                }

                if (string.IsNullOrEmpty(guidValue))
                    return;

                if (DeferAssetLoading)
                {
                    try
                    {
                        var stub = (Asset)Activator.CreateInstance(field.Type);
                        stub.Guid = guidValue;
                        field.SetValue(parent, stub);
                    }
                    catch { }
                    return;
                }

                if (Engine.Assets != null)
                {
                    try
                    {
                        if (guidValue.Length == 32 && System.Text.RegularExpressions.Regex.IsMatch(guidValue, "^[0-9a-fA-F]+$"))
                        {
                            var loadMethod = typeof(AssetManager).GetMethod("LoadByGuid")!.MakeGenericMethod(field.Type);
                            var asset = loadMethod.Invoke(Engine.Assets, new object[] { guidValue });
                            if (asset != null) field.SetValue(parent, asset);
                        }
                        else
                        {
                            var loadMethod = typeof(AssetManager).GetMethod("Load")!.MakeGenericMethod(field.Type);
                            var asset = loadMethod.Invoke(Engine.Assets, new object[] { guidValue });
                            if (asset != null) field.SetValue(parent, asset);
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning("YAMLSerializer", $"Failed to load {field.Type.Name} '{guidValue}': {ex.Message}");
                    }
                }
                return;
            }

            // ── Nested [Serializable] ──
            if (field.Type.HasAttribute<SerializableAttribute>(true))
            {
                var value = DeserializeNested(ref parser, field.Type);
                field.SetValue(parent, value);
                return;
            }

            // ── Unhandled ──
            parser.SkipCurrentNode();
        }

        private void DeserializeList(ref YamlParser parser, object parent, Field field, Type type)
        {
            var eltype = GetElementType(field.Type);
            if (eltype == null)
            {
                parser.SkipCurrentNode();
                return;
            }

            // Primitive/converter lists
            if (_converters.TryGetValue(eltype.FullName!, out var conv))
            {
                parser.Read(); // SequenceStart
                var list = (field.GetValue(parent) ?? Activator.CreateInstance(type)) as IList;

                while (parser.CurrentEventType != ParseEventType.SequenceEnd)
                {
                    var value = conv.Read(ref parser);
                    list!.Add(value);
                }

                field.SetValue(parent, list);
                return;
            }

            // Complex type lists
            if (parser.CurrentEventType != ParseEventType.SequenceStart)
            {
                parser.SkipCurrentNode();
                return;
            }

            var newlist = (field.GetValue(parent) ?? Activator.CreateInstance(type)) as IList;

            parser.Read(); // consume SequenceStart

            while (parser.CurrentEventType != ParseEventType.SequenceEnd && !parser.End)
            {
                if (parser.CurrentEventType == ParseEventType.MappingStart)
                {
                    // Untagged mapping — deserialize as the known element type
                    var deserialized = DeserializeNested(ref parser, eltype);
                    if (deserialized != null)
                        newlist!.Add(deserialized);
                }
                else if (parser.CurrentEventType == ParseEventType.Scalar)
                {
                    // Polymorphic: scalar is a type name tag, followed by a mapping
                    var elementTypeName = parser.ReadScalarAsString();
                    var fullName = ExpandTypeName(elementTypeName);
                    var serializedType = Reflector.GetType(fullName);

                    if (serializedType != null && eltype.IsAssignableFrom(serializedType))
                    {
                        var deserialized = DeserializeNested(ref parser, serializedType);
                        if (deserialized != null)
                            newlist!.Add(deserialized);
                    }
                }
                else
                {
                    parser.Read(); // skip unexpected events
                }
            }

            parser.Read(); // consume SequenceEnd

            field.SetValue(parent, newlist);
        }

        // ── Helpers ───────────────────────────────────────────────

        private void WriteFieldName(Field field, Type valueType, ref Utf8YamlEmitter emitter)
        {
            emitter.WriteString(field.Name);

            // Tag polymorphic fields so deserializer knows the concrete type
            if (valueType != field.Type)
            {
                var shortName = ShortenTypeName(valueType.FullName!);
                emitter.SetTag($"!{shortName}");
            }
        }

        private static Type GetElementType(Type listType)
        {
            if (listType.HasElementType)
                return listType.GetElementType();
            if (listType.GenericTypeArguments.Length > 0)
                return listType.GenericTypeArguments[0];
            if (listType.BaseType?.GenericTypeArguments.Length > 0)
                return listType.BaseType.GenericTypeArguments[0];
            return null;
        }

        /// <summary>
        /// Strip all namespace parts for Freefall types, keeping just the class name.
        /// "Freefall.Graphics.StaticMesh" → "StaticMesh"
        /// </summary>
        private static string ShortenTypeName(string fullName)
        {
            if (fullName.StartsWith("Freefall."))
            {
                var lastDot = fullName.LastIndexOf('.');
                return lastDot >= 0 ? fullName[(lastDot + 1)..] : fullName;
            }
            return fullName;
        }

        /// <summary>
        /// Expand a shortened type name back to full.
        /// "Components.Transform" → "Freefall.Components.Transform"
        /// </summary>
        private static string ExpandTypeName(string name)
        {
            if (name.StartsWith("!"))
                name = name.Substring(1);

            // Already fully qualified
            if (name.StartsWith("Freefall."))
                return name;

            // "Components.Transform" → try "Freefall.Components.Transform"
            var fqn = "Freefall." + name;
            if (Reflector.GetType(fqn) != null)
                return fqn;

            // Simple name like "StaticMesh" → scan all assemblies
            var type = Reflector.FindTypeBySimpleName(name);
            if (type != null)
                return type.FullName!;

            return fqn; // fallback
        }
    }
}
