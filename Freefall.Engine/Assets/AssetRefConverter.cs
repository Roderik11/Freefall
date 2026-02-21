using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Freefall.Assets
{
    /// <summary>
    /// JSON converter that serializes Asset references as GUID strings
    /// and deserializes GUID strings back to unresolved AssetRef markers.
    ///
    /// Write: writes asset.Guid as a JSON string
    /// Read:  creates an AssetRef placeholder with the GUID stored for later resolution
    ///
    /// Usage: Added to JsonSerializerOptions.Converters in AssetFile.
    /// </summary>
    public class AssetRefConverterFactory : JsonConverterFactory
    {
        public override bool CanConvert(Type typeToConvert)
        {
            return typeof(Asset).IsAssignableFrom(typeToConvert) && !typeToConvert.IsAbstract;
        }

        public override JsonConverter CreateConverter(Type typeToConvert, JsonSerializerOptions options)
        {
            var converterType = typeof(AssetRefConverter<>).MakeGenericType(typeToConvert);
            return (JsonConverter)Activator.CreateInstance(converterType);
        }
    }

    /// <summary>
    /// Typed converter for a specific Asset subclass.
    /// Serializes as GUID string, deserializes as unresolved placeholder.
    /// </summary>
    public class AssetRefConverter<T> : JsonConverter<T> where T : Asset, new()
    {
        public override T Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType == JsonTokenType.Null)
                return null;

            if (reader.TokenType == JsonTokenType.String)
            {
                var guid = reader.GetString();
                // Return a placeholder with the GUID stored for later resolution
                return new T { Guid = guid };
            }

            throw new JsonException($"Expected string (GUID) for asset reference, got {reader.TokenType}");
        }

        public override void Write(Utf8JsonWriter writer, T value, JsonSerializerOptions options)
        {
            if (value == null)
            {
                writer.WriteNullValue();
                return;
            }

            if (!string.IsNullOrEmpty(value.Guid))
            {
                writer.WriteStringValue(value.Guid);
            }
            else if (!string.IsNullOrEmpty(value.Name))
            {
                // Fallback: write name if GUID not set (editor convenience)
                writer.WriteStringValue($"name:{value.Name}");
            }
            else
            {
                writer.WriteNullValue();
            }
        }
    }
}
