using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace Freefall.Editor.Tools
{
    /// <summary>
    /// Utility for searching a Unity project: GUID→path resolution,
    /// prefab lookup, and .mat parsing for texture slots.
    /// </summary>
    public class UnityProjectIndex
    {
        private readonly string _assetsRoot;
        private readonly Dictionary<string, string> _guidToPath = new();
        private readonly Dictionary<string, string> _prefabNameToPath = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, string> _fbxNameToPath = new(StringComparer.OrdinalIgnoreCase);

        // Unity → Freefall texture slot mapping
        private static readonly Dictionary<string, string> SlotMap = new()
        {
            { "_MainTex",          "Albedo" },
            { "_BumpMap",          "Normal" },
            { "_SpecGlossMap",     "Roughness" },
            { "_GlossMap",         "Roughness" },
            { "_MetallicGlossMap", "Metallic" },
            { "_OcclusionMap",     "AO" },
            { "_EmissionMap",      "Emissive" },
            { "_ParallaxMap",      "HeightTex" },
        };

        private static readonly Regex GuidRegex = new(@"guid:\s*([0-9a-f]{32})", RegexOptions.Compiled);

        public UnityProjectIndex(string unityAssetsRoot)
        {
            _assetsRoot = unityAssetsRoot;
        }

        /// <summary>
        /// Scan all .meta files to build GUID→path index,
        /// and index .prefab + .fbx files by basename.
        /// </summary>
        public void Build()
        {
            Debug.Log("[UnityProjectIndex] Building GUID index...");

            foreach (var metaFile in Directory.EnumerateFiles(_assetsRoot, "*.meta", SearchOption.AllDirectories))
            {
                // Read just the first few lines to find the guid line
                using var reader = new StreamReader(metaFile);
                for (int i = 0; i < 4 && !reader.EndOfStream; i++)
                {
                    var line = reader.ReadLine();
                    if (line != null && line.StartsWith("guid:"))
                    {
                        var match = GuidRegex.Match(line);
                        if (match.Success)
                        {
                            var assetPath = metaFile[..^5]; // strip .meta
                            _guidToPath[match.Groups[1].Value] = assetPath;
                        }
                        break;
                    }
                }
            }

            // Index prefab and fbx files by basename (case-insensitive)
            foreach (var file in Directory.EnumerateFiles(_assetsRoot, "*.prefab", SearchOption.AllDirectories))
            {
                var key = Path.GetFileNameWithoutExtension(file);
                _prefabNameToPath.TryAdd(key, file);
            }

            foreach (var file in Directory.EnumerateFiles(_assetsRoot, "*.fbx", SearchOption.AllDirectories))
            {
                var key = Path.GetFileNameWithoutExtension(file);
                _fbxNameToPath.TryAdd(key, file);
            }

            Debug.Log($"[UnityProjectIndex] Indexed {_guidToPath.Count} GUIDs, " +
                      $"{_prefabNameToPath.Count} prefabs, {_fbxNameToPath.Count} FBX files");
        }

        /// <summary>
        /// Resolve a Unity GUID to a file path, or null.
        /// </summary>
        public string ResolveGuid(string guid)
        {
            if (string.IsNullOrEmpty(guid) || guid.StartsWith("00000000000000000000"))
                return null;
            return _guidToPath.TryGetValue(guid, out var path) ? path : null;
        }

        /// <summary>
        /// Find a Unity .prefab file by name (case-insensitive).
        /// </summary>
        public string FindPrefab(string name)
        {
            return _prefabNameToPath.TryGetValue(name, out var path) ? path : null;
        }

        /// <summary>
        /// Find a Unity .fbx file by name (case-insensitive).
        /// </summary>
        public string FindFbx(string name)
        {
            return _fbxNameToPath.TryGetValue(name, out var path) ? path : null;
        }

        /// <summary>
        /// Extract all GUIDs referenced in a file (prefab, mat, etc).
        /// Returns distinct GUIDs.
        /// </summary>
        public static List<string> ExtractGuids(string filePath)
        {
            if (!File.Exists(filePath)) return new List<string>();

            var text = File.ReadAllText(filePath);
            var matches = GuidRegex.Matches(text);
            var result = new HashSet<string>();

            foreach (Match m in matches)
            {
                var guid = m.Groups[1].Value;
                if (!guid.StartsWith("00000000000000000000"))
                    result.Add(guid);
            }

            return result.ToList();
        }

        /// <summary>
        /// Extract mesh (.fbx/.obj/.dae) and material (.mat) file paths
        /// from a Unity prefab by resolving its GUIDs.
        /// </summary>
        public (List<string> meshPaths, List<string> matPaths) ExtractPrefabAssets(string prefabPath)
        {
            var meshPaths = new List<string>();
            var matPaths = new List<string>();

            var guids = ExtractGuids(prefabPath);
            foreach (var guid in guids)
            {
                var resolved = ResolveGuid(guid);
                if (resolved == null) continue;

                var ext = Path.GetExtension(resolved).ToLowerInvariant();
                switch (ext)
                {
                    case ".fbx" or ".obj" or ".dae":
                        if (!meshPaths.Contains(resolved))
                            meshPaths.Add(resolved);
                        break;
                    case ".mat":
                        if (!matPaths.Contains(resolved))
                            matPaths.Add(resolved);
                        break;
                    case ".prefab":
                        // Nested prefab — recurse
                        var (nested_m, nested_mat) = ExtractPrefabAssets(resolved);
                        foreach (var nm in nested_m)
                            if (!meshPaths.Contains(nm)) meshPaths.Add(nm);
                        foreach (var nmat in nested_mat)
                            if (!matPaths.Contains(nmat)) matPaths.Add(nmat);
                        break;
                }
            }

            return (meshPaths, matPaths);
        }

        /// <summary>
        /// Parse a Unity .mat file and return texture slot → file path mappings.
        /// Only returns slots we care about (from SlotMap).
        /// </summary>
        public Dictionary<string, string> ParseMaterial(string matPath)
        {
            var result = new Dictionary<string, string>();
            if (!File.Exists(matPath)) return result;

            var lines = File.ReadAllLines(matPath);
            string currentSlot = null;

            for (int i = 0; i < lines.Length; i++)
            {
                var line = lines[i].TrimStart();

                // Detect texture slot name (e.g., "- _MainTex:")
                if (line.StartsWith("- _"))
                {
                    var colonIdx = line.IndexOf(':');
                    if (colonIdx > 2)
                        currentSlot = line[2..colonIdx];
                    else
                        currentSlot = null;
                    continue;
                }

                // Detect texture reference within current slot
                if (currentSlot != null && line.StartsWith("m_Texture:"))
                {
                    var match = GuidRegex.Match(line);
                    if (match.Success && SlotMap.ContainsKey(currentSlot))
                    {
                        var texPath = ResolveGuid(match.Groups[1].Value);
                        if (texPath != null)
                            result[SlotMap[currentSlot]] = texPath;
                    }
                    currentSlot = null;
                }
            }

            return result;
        }

        /// <summary>
        /// Get a relative path from a full Unity asset path, stripped of pack prefixes.
        /// e.g., "D:\...\BIG_Environment_Pack_Reforged\Environment\Buildings\..." → "Buildings\..."
        /// </summary>
        public string GetStrippedRelPath(string fullPath)
        {
            var rel = fullPath;
            if (fullPath.StartsWith(_assetsRoot, StringComparison.OrdinalIgnoreCase))
                rel = fullPath[(_assetsRoot.Length)..].TrimStart('\\', '/');

            // Strip pack prefix
            var patterns = new[]
            {
                @"BIG_Environment_Pack_Reforged[\\/]Environment[\\/]",
                @"BIG_Environment_Pack_Reforged[\\/]"
            };

            foreach (var pattern in patterns)
            {
                var stripped = Regex.Replace(rel, "^" + pattern, "", RegexOptions.IgnoreCase);
                if (stripped != rel) return stripped;
            }

            return rel;
        }
    }
}
