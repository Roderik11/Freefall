using Squid;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using System.Text.Json.Serialization;
using Vortice.Mathematics;

namespace Freefall.Editor
{
    /// <summary>
    /// A named color entry for an asset type. Pure data class for JSON persistence.
    /// </summary>
    public class AssetTypeColor
    {
        public string Name { get; set; }
        public Color3 Color;

        public AssetTypeColor() { }
        public AssetTypeColor(string name, Color3 color) { Name = name; Color = color; }
    }

    public class SnapSettings
    {
        public bool SnapToGrid { get; set; } = false; // Whether to snap entity positions to a grid (editor-only)
        public Vector3 GridSnap { get; set; } = new Vector3(1, 1, 1);
        public float AngleSnap { get; set; } = 15.0f; // Snap rotation to increments of this angle in degrees
    }

    /// <summary>
    /// Editor-wide preferences, persisted as JSON.
    /// </summary>
    public class EditorPreferences
    {
        public static EditorPreferences Instance { get; private set; } = new();

        public List<AssetTypeColor> AssetTypeColors { get; set; } = new();
        public SnapSettings Snapping { get; set; } = new SnapSettings();

        /// <summary>
        /// Engine render/debug settings (Engine.Settings: fog, bloom, shadows, ...) as name → invariant string, so toggles
        /// made in the Settings panel or via settings_set survive editor restarts. Captured on Save, applied on Load.
        /// </summary>
        public Dictionary<string, string> EngineSettings { get; set; } = new();

        private static IEnumerable<System.Reflection.PropertyInfo> PersistedEngineSettings =>
            typeof(Freefall.EngineSettings).GetProperties(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance)
                .Where(p => p.CanRead && p.CanWrite &&
                            (p.PropertyType == typeof(bool) || p.PropertyType == typeof(int) || p.PropertyType == typeof(float) || p.PropertyType.IsEnum));

        private static void CaptureEngineSettings()
        {
            var settings = Engine.Settings;
            if (settings == null) return;
            Instance.EngineSettings = PersistedEngineSettings.ToDictionary(
                p => p.Name,
                p => Convert.ToString(p.GetValue(settings), System.Globalization.CultureInfo.InvariantCulture));
        }

        private static void ApplyEngineSettings()
        {
            var settings = Engine.Settings;
            if (settings == null || Instance.EngineSettings == null) return;
            foreach (var p in PersistedEngineSettings)
            {
                if (!Instance.EngineSettings.TryGetValue(p.Name, out var s)) continue;
                try
                {
                    object v = p.PropertyType.IsEnum
                        ? Enum.Parse(p.PropertyType, s, ignoreCase: true)
                        : Convert.ChangeType(s, p.PropertyType, System.Globalization.CultureInfo.InvariantCulture);
                    p.SetValue(settings, v);
                }
                catch { /* renamed/removed setting — keep the engine default */ }
            }
        }

        // Style management — maps type name → ControlStyle (not serialized)
        [JsonIgnore]
        private static Dictionary<string, ControlStyle> _skin;
        [JsonIgnore]
        private static readonly Dictionary<string, ControlStyle> _styles = new();

        private const string StylePrefix = "assetcolor_";

        /// <summary>
        /// Get the style name for an asset type (e.g. "assetcolor_MODEL").
        /// Ensures the entry and style exist.
        /// </summary>
        public string GetAssetStyleName(string typeName)
        {
            if (string.IsNullOrEmpty(typeName)) typeName = "UNKNOWN";
            typeName = typeName.ToUpperInvariant();
            GetAssetColor(typeName);
            return StylePrefix + typeName;
        }

        /// <summary>
        /// Get the color for an asset type. Auto-creates entry + style if missing.
        /// </summary>
        public Color3 GetAssetColor(string typeName)
        {
            if (string.IsNullOrEmpty(typeName)) typeName = "UNKNOWN";
            typeName = typeName.ToUpperInvariant();

            var entry = AssetTypeColors.FirstOrDefault(e =>
                string.Equals(e.Name, typeName, StringComparison.OrdinalIgnoreCase));

            if (entry != null)
            {
                EnsureStyle(entry);
                return entry.Color;
            }

            var color = GenerateColorFromName(typeName);
            entry = new AssetTypeColor(typeName, color);
            AssetTypeColors.Add(entry);
            EnsureStyle(entry);
            return color;
        }

        /// <summary>
        /// Register ControlStyles in the skin for all known asset type colors.
        /// Call once after EditorSkin.Apply and EditorPreferences.Load.
        /// </summary>
        public static void RegisterStyles(Desktop desktop)
        {
            _skin = desktop.Skin;
            foreach (var entry in Instance.AssetTypeColors)
                Instance.EnsureStyle(entry);
        }

        /// <summary>
        /// Scan the asset database and pre-register all discovered type names.
        /// Call after AssetDatabase is initialized and RegisterStyles has been called.
        /// </summary>
        public static void DiscoverAllTypes()
        {
            foreach (var meta in Assets.AssetDatabase.GetAllMeta())
            {
                // Importer type (e.g. "Freefall.Assets.Importers.ModelImporter" → "MODEL")
                var importerType = CleanImporterType(meta.ImporterType);
                if (!string.IsNullOrEmpty(importerType))
                    Instance.GetAssetColor(importerType);

                // Sub-asset types (e.g. "StaticMeshData" → "STATICMESH")
                if (meta.SubAssets != null)
                {
                    foreach (var sub in meta.SubAssets)
                    {
                        var subType = CleanTypeName(sub.AssetType ?? sub.Type);
                        if (!string.IsNullOrEmpty(subType))
                            Instance.GetAssetColor(subType);
                    }
                }
            }
        }

        /// <summary>
        /// Clean an artifact type name: strip "Data" suffix, uppercase.
        /// </summary>
        private static string CleanTypeName(string type)
        {
            if (string.IsNullOrEmpty(type)) return null;
            if (type.EndsWith("Data", StringComparison.Ordinal))
                type = type[..^4];
            return type.ToUpperInvariant();
        }

        /// <summary>
        /// Clean an importer type: extract last segment, strip "Importer" suffix, uppercase.
        /// </summary>
        private static string CleanImporterType(string importerType)
        {
            if (string.IsNullOrEmpty(importerType)) return null;
            var name = importerType;
            var dot = name.LastIndexOf('.');
            if (dot >= 0) name = name[(dot + 1)..];
            if (name.EndsWith("Importer", StringComparison.Ordinal))
                name = name[..^8];
            return name.ToUpperInvariant();
        }

        /// <summary>
        /// Sync an AssetTypeColor's Color3 value to its ControlStyle.BackColor.
        /// </summary>
        public static void SyncStyle(AssetTypeColor entry)
        {
            var styleName = StylePrefix + entry.Name;
            if (_styles.TryGetValue(styleName, out var style))
                style.BackColor = ColorInt.ARGB(1f, entry.Color.R, entry.Color.G, entry.Color.B);
        }

        private void EnsureStyle(AssetTypeColor entry)
        {
            if (_skin == null) return;

            var styleName = StylePrefix + entry.Name;
            if (_styles.ContainsKey(styleName)) return;

            var backColor = ColorInt.ARGB(1f, entry.Color.R, entry.Color.G, entry.Color.B);
            var style = new ControlStyle { BackColor = backColor };
            _skin[styleName] = style;
            _styles[styleName] = style;
        }

        // --- Color generation ---

        private static Color3 GenerateColorFromName(string name)
        {
            uint hash = 0;
            foreach (char c in name)
                hash = hash * 31 + c;

            float hue = (hash % 360) / 360f;
            return HsvToRgb(hue, 0.55f, 0.85f);
        }

        private static Color3 HsvToRgb(float h, float s, float v)
        {
            float c = v * s;
            float x = c * (1f - MathF.Abs((h * 6f) % 2f - 1f));
            float m = v - c;

            float r, g, b;
            int sector = (int)(h * 6f) % 6;

            switch (sector)
            {
                case 0: r = c; g = x; b = 0; break;
                case 1: r = x; g = c; b = 0; break;
                case 2: r = 0; g = c; b = x; break;
                case 3: r = 0; g = x; b = c; break;
                case 4: r = x; g = 0; b = c; break;
                default: r = c; g = 0; b = x; break;
            }

            return new Color3(r + m, g + m, b + m);
        }

        // --- JSON Persistence ---

        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            WriteIndented = true,
            PropertyNameCaseInsensitive = true,
            IncludeFields = true,
            Converters = { new Color3JsonConverter() }
        };

        private static string PrefsPath =>
            Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "Freefall", "EditorPreferences.json");

        public static void Load()
        {
            if (!File.Exists(PrefsPath))
            {
                Instance = new EditorPreferences();
                return;
            }

            try
            {
                var json = File.ReadAllText(PrefsPath);
                Instance = JsonSerializer.Deserialize<EditorPreferences>(json, JsonOptions) ?? new();
                ApplyEngineSettings();
            }
            catch (Exception ex)
            {
                Debug.LogWarning("EditorPreferences", $"Failed to load: {ex.Message}");
                Instance = new EditorPreferences();
            }
        }

        public static void Save()
        {
            try
            {
                CaptureEngineSettings();
                var dir = Path.GetDirectoryName(PrefsPath);
                Directory.CreateDirectory(dir);
                var json = JsonSerializer.Serialize(Instance, JsonOptions);
                File.WriteAllText(PrefsPath, json);
            }
            catch (Exception ex)
            {
                Debug.LogWarning("EditorPreferences", $"Failed to save: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// JSON converter for Vortice.Mathematics.Color3 (R, G, B floats).
    /// </summary>
    public class Color3JsonConverter : JsonConverter<Color3>
    {
        public override Color3 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException();

            float r = 0, g = 0, b = 0;
            while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
            {
                if (reader.TokenType != JsonTokenType.PropertyName) continue;
                var prop = reader.GetString();
                reader.Read();
                switch (prop)
                {
                    case "R": r = reader.GetSingle(); break;
                    case "G": g = reader.GetSingle(); break;
                    case "B": b = reader.GetSingle(); break;
                }
            }

            return new Color3(r, g, b);
        }

        public override void Write(Utf8JsonWriter writer, Color3 value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            writer.WriteNumber("R", MathF.Round(value.R, 4));
            writer.WriteNumber("G", MathF.Round(value.G, 4));
            writer.WriteNumber("B", MathF.Round(value.B, 4));
            writer.WriteEndObject();
        }
    }
}
