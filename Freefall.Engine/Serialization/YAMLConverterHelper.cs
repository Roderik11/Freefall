using System;
using System.Globalization;
using System.Numerics;
using Vortice.Mathematics;

namespace Freefall.Serialization
{
    /// <summary>
    /// String-based parse/format for types supported by the YAML converter system.
    /// Used by MaterialProperty for serializing scalar material properties.
    /// </summary>
    public static class YAMLConverterHelper
    {
        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        public static bool TryParse<T>(string text, out T result)
        {
            result = default;
            if (string.IsNullOrWhiteSpace(text)) return false;
            text = text.Trim();

            // Handle legacy single-float format for Vector2 (e.g. DetailTiling: 5)
            if (typeof(T) == typeof(Vector2))
            {
                if (text.StartsWith("["))
                {
                    var parts = text.TrimStart('[').TrimEnd(']').Split(',');
                    if (parts.Length >= 2 &&
                        float.TryParse(parts[0].Trim(), NumberStyles.Float, Inv, out float x) &&
                        float.TryParse(parts[1].Trim(), NumberStyles.Float, Inv, out float y))
                    {
                        result = (T)(object)new Vector2(x, y);
                        return true;
                    }
                }
                else if (float.TryParse(text, NumberStyles.Float, Inv, out float f))
                {
                    result = (T)(object)new Vector2(f, f);
                    return true;
                }
                return false;
            }
            if (typeof(T) == typeof(Color3))
            {
                var parts = text.TrimStart('<').TrimEnd('>').Split(',');
                if (parts.Length >= 3 &&
                    float.TryParse(parts[0].Trim(), NumberStyles.Float, Inv, out float x) &&
                    float.TryParse(parts[1].Trim(), NumberStyles.Float, Inv, out float y) &&
                    float.TryParse(parts[2].Trim(), NumberStyles.Float, Inv, out float z))
                {
                    result = (T)(object)new Color3(x, y, z);
                    return true;
                }
                return false;
            }

            if (typeof(T) == typeof(Vector3))
            {
                var parts = text.TrimStart('[').TrimEnd(']').Split(',');
                if (parts.Length >= 3 &&
                    float.TryParse(parts[0].Trim(), NumberStyles.Float, Inv, out float x) &&
                    float.TryParse(parts[1].Trim(), NumberStyles.Float, Inv, out float y) &&
                    float.TryParse(parts[2].Trim(), NumberStyles.Float, Inv, out float z))
                {
                    result = (T)(object)new Vector3(x, y, z);
                    return true;
                }
                return false;
            }

            if (typeof(T) == typeof(float))
            {
                if (float.TryParse(text, NumberStyles.Float, Inv, out float f))
                {
                    result = (T)(object)f;
                    return true;
                }
                return false;
            }

            return false;
        }

        public static string Format<T>(T value)
        {
            return value switch
            {
                float f => f.ToString(Inv),
                Vector2 v => $"[{v.X.ToString(Inv)}, {v.Y.ToString(Inv)}]",
                Vector3 v => $"[{v.X.ToString(Inv)}, {v.Y.ToString(Inv)}, {v.Z.ToString(Inv)}]",
                _ => value.ToString()
            };
        }
    }
}
