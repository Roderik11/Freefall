using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text;
using Freefall.Assets;
using Freefall.Graphics;

namespace Freefall.Assets.Serializers
{
    /// <summary>
    /// Custom serializer for Material .asset files.
    /// Material can't use generic YAMLSerializer because textures and parameters
    /// are stored in runtime dictionaries, not as public fields.
    ///
    /// Format:
    ///   !Material
    ///   Effect: <effect_guid>
    ///   Textures:
    ///     AlbedoTex: <texture_guid>
    ///     NormalTex: <texture_guid>
    ///   Parameters:
    ///     MyFloat: 1.5
    ///     MyColor: [1, 0.5, 0.2, 1]
    /// </summary>
    [AssetSerializer(typeof(Material))]
    public class MaterialSerializer : IAssetSerializer
    {
        public string Serialize(Asset asset)
        {
            var mat = (Material)asset;
            var sb = new StringBuilder();

            sb.AppendLine("!Material");

            // Effect reference
            if (mat.Effect != null && !string.IsNullOrEmpty(mat.Effect.Guid))
                sb.AppendLine($"Effect: {mat.Effect.Guid}");
            else if (mat.Effect != null)
                sb.AppendLine($"Effect: {mat.Effect.Name}");

            // Textures
            if (mat.Textures.Count > 0)
            {
                sb.AppendLine("Textures:");
                foreach (var (slot, texture) in mat.Textures)
                {
                    if (!string.IsNullOrEmpty(texture.Guid))
                        sb.AppendLine($"  {slot}: {texture.Guid}");
                    else
                        sb.AppendLine($"  {slot}: {texture.Name}");
                }
            }

            return sb.ToString();
        }

        public Asset Deserialize(string yaml)
        {
            // Parse the material definition into a lightweight descriptor.
            // The actual Material with GPU resources is created by MaterialLoader
            // when it resolves GUIDs. Here we just store the parsed data.
            var def = new MaterialDefinition();

            var lines = yaml.Split('\n');
            string currentSection = null;

            foreach (var rawLine in lines)
            {
                var line = rawLine.Trim();
                if (string.IsNullOrEmpty(line) || line.StartsWith("!")) continue;

                if (line == "Textures:")
                {
                    currentSection = "Textures";
                    continue;
                }
                if (line == "Parameters:")
                {
                    currentSection = "Parameters";
                    continue;
                }

                // Top-level key: value
                if (!line.StartsWith("  ") && line.Contains(':') && currentSection == null)
                {
                    var colonIdx = line.IndexOf(':');
                    var key = line.Substring(0, colonIdx).Trim();
                    var val = line.Substring(colonIdx + 1).Trim();

                    if (key == "Effect")
                        def.EffectRef = ParseRef(val);
                    else if (key != "Name" && key != "Textures" && key != "Parameters" && !string.IsNullOrEmpty(val))
                    {
                        // Flat format: texture refs at top level (e.g. "Roughness: 82865390...")
                        def.TextureRefs[key] = ParseRef(val);
                    }
                }

                // Indented entries (in a section)
                if (currentSection != null && line.Contains(':'))
                {
                    var colonIdx = line.IndexOf(':');
                    var key = line.Substring(0, colonIdx).Trim();
                    var val = line.Substring(colonIdx + 1).Trim();

                    if (currentSection == "Textures")
                        def.TextureRefs[key] = ParseRef(val);
                    else if (currentSection == "Parameters")
                        def.Parameters[key] = val;
                }
            }

            return def;
        }

        /// <summary>
        /// Parse a reference value: either {guid: xxx} or a plain name.
        /// </summary>
        private static string ParseRef(string value)
        {
            value = value.Trim();
            if (value.StartsWith("{guid:"))
            {
                var start = value.IndexOf(':') + 1;
                var end = value.IndexOf('}');
                return value.Substring(start, end - start).Trim();
            }
            return value;
        }
    }

    /// <summary>
    /// Lightweight deserialized Material definition — no GPU resources.
    /// Holds Effect/Texture references as GUIDs or names until MaterialLoader resolves them.
    /// </summary>
    public class MaterialDefinition : Asset
    {
        public string EffectRef { get; set; }
        public Dictionary<string, string> TextureRefs { get; set; } = new();
        public Dictionary<string, string> Parameters { get; set; } = new();
    }
}
