using System.IO;
using System.Text;
using Freefall.Assets.Packers;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports .asset files (YAML-serialized Asset definitions).
    /// Reads the raw YAML and stores it as-is for caching — no deserialization
    /// during import to avoid issues with types that need runtime resources.
    /// </summary>
    [AssetImporter(".asset")]
    public class AssetFileImporter : IImporter
    {
        public ImportResult Import(string filepath)
        {
            var result = new ImportResult();
            var name = Path.GetFileNameWithoutExtension(filepath);
            var yaml = File.ReadAllText(filepath, Encoding.UTF8);

            // Peek at the YAML type tag to get the concrete type name
            // (e.g. "!Material" → "Material", "!StaticMesh" → "StaticMesh")
            var typeName = PeekTypeName(yaml) ?? "Asset";

            result.Artifacts.Add(new ImportArtifact
            {
                Name = name,
                Type = typeName,
                Data = new AssetDefinitionData
                {
                    TypeName = typeName,
                    YamlBytes = Encoding.UTF8.GetBytes(yaml)
                }
            });

            return result;
        }

        /// <summary>
        /// Extract the type name from YAML content.
        /// Supports both !TypeName tag format and TypeName: wrapper format.
        /// </summary>
        private static string PeekTypeName(string yaml)
        {
            using var reader = new StringReader(yaml);
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                line = line.Trim();
                if (string.IsNullOrEmpty(line)) continue;

                // !Tag format: "!StaticMesh"
                if (line.StartsWith("!"))
                    return line.Substring(1).Trim();
                if (line.Contains("!"))
                {
                    var idx = line.IndexOf('!');
                    var rest = line.Substring(idx + 1).Trim();
                    if (rest.Length > 0) return rest;
                }

                // Wrapper format: "StaticMesh:" (first non-empty line ending with :)
                if (line.EndsWith(":") && !line.Contains(" "))
                    return line.Substring(0, line.Length - 1);

                break; // Only check the first meaningful line
            }
            return null;
        }
    }
}
