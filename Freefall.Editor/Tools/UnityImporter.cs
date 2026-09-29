using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Diagnostics;
using Freefall.Assets;
using Freefall.Assets.Importers;
using Freefall.Graphics;
using Freefall.Serialization;
using Freefall.Components;
using Freefall.Base;
using System.Numerics;
using System.Text.Json;

namespace Freefall.Editor.Tools
{
    /// <summary>
    /// Per-pack ModelImporter settings configurable from the UnityImporterWindow.
    /// </summary>
    public class UnityImporterSettings
    {
        public float Scale = 1f;
        public bool ConvertUnits = true;
        public bool PreTransform = false;
        public bool Optimize = false;
        public bool LeftHanded = true;
        public bool FlipUVs = false;
        public bool FlipWinding = true;
        public bool CalculateTangents = true;
    }

    /// <summary>
    /// Imports Unity Asset Store packs (HDRP) into a Freefall project.
    ///
    /// Given a Unity source directory containing .meta, .mat, .prefab, .fbx, and texture files,
    /// this tool:
    ///   1. Discovers all assets via .meta GUIDs
    ///   2. Copies textures + FBX models into the Freefall Assets/ directory
    ///   3. Converts unsupported formats (EXR) to PNG
    ///   4. Creates Freefall .mat files with texture GUID references
    ///   5. Converts Unity prefabs to Freefall .prefab files (MeshRenderer-based)
    ///
    /// Usage:
    ///   UnityImporter.Import(@"D:\UnityPacks\MedievalTown", assetsRoot, "MedievalTown");
    /// </summary>
    public static class UnityImporter
    {
        // Unity HDRP ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ Freefall texture slot mapping
        private static readonly Dictionary<string, string> HdrpSlotMap = new()
        {
            { "_BaseColorMap",     "Albedo" },
            { "_BaseMap",          "Albedo" },    // URP / Standard shader
            { "_MainTex",          "Albedo" },
            { "_NormalMap",        "Normal" },
            { "_BumpMap",          "Normal" },
            { "_HeightMap",        "HeightTex" },
            { "_ParallaxMap",      "HeightTex" },
            { "_EmissiveColorMap", "Emissive" },
            { "_EmissionMap",      "Emissive" },
            { "_OcclusionMap",     "AO" },        // only used if texture isn't a packed ORMH
            // _MaskMap is packed HDRP (M/AO/Detail/Smoothness) -> skipped
            { "_MetallicGlossMap", "Metallic" },
        };

        // Unity material name → Freefall GUID mapping (populated during import)
        private static Dictionary<string, string> _materialNameToGuid = new(StringComparer.OrdinalIgnoreCase);


        // Texture slot priority (first match wins for each Freefall slot)
        private static readonly string[] AlbedoPriority = { "_BaseColorMap", "_BaseMap", "_MainTex" };

        // Importable texture extensions
        private static readonly HashSet<string> TextureExtensions = new(StringComparer.OrdinalIgnoreCase)
            { ".png", ".tga", ".jpg", ".jpeg", ".psd" };

        // Extensions that need conversion to PNG
        private static readonly HashSet<string> ConvertExtensions = new(StringComparer.OrdinalIgnoreCase)
            { ".exr", ".tif", ".tiff" };

        // Model extensions
        private static readonly HashSet<string> ModelExtensions = new(StringComparer.OrdinalIgnoreCase)
            { ".fbx", ".obj" };

        // Directories to skip
        private static readonly HashSet<string> SkipDirs = new(StringComparer.OrdinalIgnoreCase)
            { "Shaders", "VFX", "Scenes", "Animation", "Scripts" };

        /// <summary>
        /// Import a Unity asset pack into the Freefall project.
        /// </summary>
        /// <param name="unitySourceDir">Root of the Unity asset pack (contains .meta files)</param>
        /// <param name="assetsRoot">Freefall project Assets/ directory</param>
        /// <param name="packName">Subfolder name within Assets/ for this pack</param>
        /// <param name="onProgress">Optional callback for status updates (shown in UI)</param>
        /// <param name="settings">Optional ModelImporter settings override for this pack</param>
        // Diagnostic log file written during import for debugging
        private static string _diagLogPath;
        private static void DiagLog(string msg)
        {
            try { if (_diagLogPath != null) File.AppendAllText(_diagLogPath, msg + "\n"); } catch { }
        }

        public static void LoadScene(string path)
        {
            if (string.IsNullOrEmpty(path))
                return;

            var jsonText = File.ReadAllText(path);
            var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
            var exportData = JsonSerializer.Deserialize<PrefabExportData>(jsonText, options);
       
            foreach(var prefab in exportData.Prefabs)
                SpawnPrefab(prefab);

            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private static void SpawnPrefab(ExportEntity entity, Entity parent = null)
        {
            var guid = AssetDatabase.FindGuidByName(entity.Mesh, "Prefab");
            var prefab = Engine.Assets.LoadByGuid<Prefab>(guid);
            Entity instance = null;

            if (prefab != null)
                instance = prefab.Instantiate();

            if (instance == null && entity.Children.Count == 0)
                return;

            instance ??= new Entity(entity.Name);
            instance.Transform.Position = entity.Position.ToVector3();
            instance.Transform.Rotation = entity.Rotation.ToQuaternion();
            instance.Transform.Scale = entity.Scale.ToVector3();

            if (parent != null)
                instance.Transform.Parent = parent.Transform;

            foreach (var child in entity.Children)
                SpawnPrefab(child, instance);
        }


        public static void Import(string unitySourceDir, string assetsRoot, string packName, Action<string> onProgress = null, UnityImporterSettings settings = null)
        {
            _diagLogPath = Path.Combine(assetsRoot, "..", "unity_import_diag.log");
            try { File.WriteAllText(_diagLogPath, $"=== UnityImporter Diagnostic Log ===\n{DateTime.Now}\n\n"); } catch { }

            void Status(string msg) { Debug.Log(msg); onProgress?.Invoke(msg); DiagLog(msg); }
            Status($"[UnityImporter] Importing: {packName}");
            Status($"[UnityImporter] Source: {unitySourceDir}");

            Status("Discovering Unity assets...");
            var discovery = Discover(unitySourceDir);

            Status("Copying raw assets...");
            var copyMap = CopyAssets(unitySourceDir, assetsRoot, packName, discovery);

            Status("Importing into AssetDatabase...");
            Debug.Log("[UnityImporter] Phase 3: Refreshing AssetDatabase...");
            AssetDatabase.Refresh();
            Debug.Log($"[UnityImporter]   {AssetDatabase.GetAllPaths().Count()} total tracked assets");

            // Set ModelImporter override if settings provided
            if (settings != null)
            {
                ModelImporter.OverrideSettings = new ModelImporter
                {
                    Scale = settings.Scale,
                    PreTransform = settings.PreTransform,
                    FlipUVs = settings.FlipUVs,
                    FlipWinding = settings.FlipWinding,
                    CalculateTangents = settings.CalculateTangents,
                    LeftHanded = settings.LeftHanded,
                    ConvertUnits = settings.ConvertUnits,
                    Optimize = settings.Optimize
                };
            }

            try
            {

            Status("Harvesting Freefall GUIDs...");
            var guidMap = HarvestGuids(assetsRoot, copyMap, discovery);

            // Parse prefabs.json for exported material + prefab data
            var prefabJsonPath = Path.Combine(unitySourceDir, "prefabs.json");
            PrefabExportData exportData = null;
            if (File.Exists(prefabJsonPath))
            {
                try
                {
                    var jsonText = File.ReadAllText(prefabJsonPath);
                    var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                    exportData = JsonSerializer.Deserialize<PrefabExportData>(jsonText, options);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("UnityImporter", $"Failed to parse prefabs.json: {ex.Message}");
                }
            }
            var exportMaterials = exportData?.Materials?.Count > 0 ? exportData.Materials : null;

            // Phase 5: Create materials
            var materialGuidMap = new Dictionary<string, string>();

            if (exportMaterials != null)
            {
                Status($"Creating Freefall materials from JSON ({exportMaterials.Count} materials)...");
                CreateMaterialsFromJson(assetsRoot, packName, exportMaterials);
            }
            else
            {
                Status("Creating Freefall materials...");
                materialGuidMap = CreateMaterials(assetsRoot, packName, discovery, guidMap);
            }

            // Import all: textures, materials, FBX files
            // Priority ordering ensures textures → materials → models → prefabs
            Status("Importing all assets...");
            AssetDatabase.Refresh();
            AssetDatabase.ImportAllAsync(onProgress).GetAwaiter().GetResult();

            if (exportMaterials != null)
            {
                _materialNameToGuid = BuildMaterialNameToGuidMap(exportMaterials);
            }
            else
            {
                // Rebuild materialGuidMap with real GUIDs from AssetDatabase
                foreach (var (unityMatGuid, matRef) in discovery.Materials)
                {
                    var realGuid = FindMaterialGuid(assetsRoot, discovery, unityMatGuid);
                    if (realGuid != null)
                        materialGuidMap[unityMatGuid] = realGuid;
                }
            }

            // Phase 6: Convert Unity prefabs to Freefall prefabs
            if (File.Exists(prefabJsonPath))
            {
                Status("Creating Freefall prefabs from JSON...");
                Engine.RunOnMainThreadAsync(() =>
                    CreatePrefabsFromJson(prefabJsonPath, assetsRoot, packName, onProgress)
                ).GetAwaiter().GetResult();
            }
            else
            {
                Status("Skipping prefab creation (no prefabs.json found - run Export Prefabs in Unity first)");
            }

            // Final import pass
            Status("Final import pass...");
            AssetDatabase.Refresh();
            AssetDatabase.ImportAllAsync(onProgress).GetAwaiter().GetResult();
            Status("Import complete!");

            }
            finally
            {
                ModelImporter.OverrideSettings = null;
            }
        }

        private class DiscoveryResult
        {
            public Dictionary<string, AssetInfo> Textures = new();
            public Dictionary<string, AssetInfo> Models = new();
            public Dictionary<string, MaterialRef> Materials = new();
        }

        private class AssetInfo
        {
            public string RelPath;  // Relative to Unity source root
            public string Name;
            public List<string> MaterialGuids = new(); // Unity material GUIDs from FBX .meta externalObjects (DEPRECATED - use MaterialMap)
            public Dictionary<string, string> MaterialMap = new(); // Material name → Unity GUID from FBX .meta
            public float FileScale = 1f; // Scale from FBX .meta (useFileScale * globalScale)
        }

        private class MaterialRef
        {
            public string RelPath;
            public string Name;
            public Dictionary<string, string> TextureSlots = new(); // Unity slot name → Unity texture GUID
        }

        private static DiscoveryResult Discover(string sourceDir, Action<string> onProgress = null)
        {
            void Status(string msg) { Debug.Log(msg); onProgress?.Invoke(msg); }

            Status("[UnityImporter] Phase 1: Discovering Unity assets...");
            var result = new DiscoveryResult();

            foreach (var metaPath in Directory.EnumerateFiles(sourceDir, "*.meta", SearchOption.AllDirectories))
            {
                var companion = metaPath[..^5]; // Strip ".meta"
                if (!File.Exists(companion)) continue;


                var ext = Path.GetExtension(companion).ToLowerInvariant();
                var relDir = Path.GetRelativePath(sourceDir, Path.GetDirectoryName(companion)!);

                // Skip non-importable directories
                if (relDir.Split(Path.DirectorySeparatorChar).Any(d => SkipDirs.Contains(d)))
                    continue;

                var guid = ParseMetaGuid(metaPath);
                if (guid == null) continue;

                var relPath = Path.GetRelativePath(sourceDir, companion);
                var name = Path.GetFileNameWithoutExtension(companion);

                Status($"Discovered: {name}");

                if (TextureExtensions.Contains(ext) || ConvertExtensions.Contains(ext))
                {
                    result.Textures[guid] = new AssetInfo { RelPath = relPath, Name = name };
                }
                else if (ModelExtensions.Contains(ext))
                {
                    var modelInfo = new AssetInfo { RelPath = relPath, Name = name };
                    // Parse material assignments from FBX .meta externalObjects
                    // Parse material name→GUID map from FBX .meta
                    ParseFbxMetaMaterials(metaPath, modelInfo.MaterialMap);
                    // Parse file scale: useFileScale=1 + FBX cm convention → scale 100
                    modelInfo.FileScale = ParseFbxFileScale(metaPath);
                    result.Models[guid] = modelInfo;
                }
                else if (ext == ".mat")
                {
                    var slots = ParseUnityMaterial(companion);
                    result.Materials[guid] = new MaterialRef
                    {
                        RelPath = relPath,
                        Name = name,
                        TextureSlots = slots,
                    };
                }

            }

            return result;
        }

        private static string ParseMetaGuid(string metaPath)
        {
            try
            {
                foreach (var line in File.ReadLines(metaPath))
                {
                    if (line.StartsWith("guid:"))
                        return line[5..].Trim();
                }
            }
            catch { }
            return null;
        }

        private static Dictionary<string, string> ParseUnityMaterial(string matPath)
        {
            var slots = new Dictionary<string, string>();
            try
            {
                var content = File.ReadAllText(matPath);
                // Pattern: - _SlotName:\n    m_Texture: {fileID: X, guid: Y, type: Z}
                var matches = Regex.Matches(content,
                    @"- (\w+):\s*\n\s*m_Texture:\s*\{fileID:\s*(\d+),\s*guid:\s*([a-f0-9]+)");
                foreach (Match m in matches)
                {
                    var slotName = m.Groups[1].Value;
                    var fileId = int.Parse(m.Groups[2].Value);
                    var guid = m.Groups[3].Value;
                    if (fileId != 0 && HdrpSlotMap.ContainsKey(slotName))
                        slots.TryAdd(slotName, guid);
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("UnityImporter", $"Failed to parse material {matPath}: {ex.Message}");
            }
            return slots;
        }


        //  Phase 2: Copy + Convert
        /// <returns>Unity relative path -> Freefall relative path</returns>
        private static Dictionary<string, string> CopyAssets(
            string sourceDir, string assetsRoot, string packName, DiscoveryResult discovery)
        {
            Debug.Log("[UnityImporter] Phase 2: Copying raw assets (preserving folder structure)...");
            var copyMap = new Dictionary<string, string>();
            int copied = 0, converted = 0, skipped = 0;

            // Copy textures
            foreach (var (guid, info) in discovery.Textures)
            {
                var src = Path.Combine(sourceDir, info.RelPath);
                if (!File.Exists(src)) { skipped++; continue; }

                var ext = Path.GetExtension(src).ToLowerInvariant();
                bool needsConvert = ConvertExtensions.Contains(ext);

                // Preserve source folder structure: packName/original/relative/path
                var dstRel = Path.Combine(packName, info.RelPath);
                if (needsConvert)
                    dstRel = Path.ChangeExtension(dstRel, ".png");

                var dst = Path.Combine(assetsRoot, dstRel);
                if (!File.Exists(dst))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(dst)!);

                    if (needsConvert)
                    {
                        if (!ConvertWithTexconv(src, Path.GetDirectoryName(dst)!))
                        {
                            skipped++;
                            continue;
                        }
                        converted++;
                    }
                    else
                    {
                        File.Copy(src, dst);
                    }
                    copied++;
                }

                copyMap[info.RelPath] = dstRel;
            }

            // Copy models
            foreach (var (guid, info) in discovery.Models)
            {
                var src = Path.Combine(sourceDir, info.RelPath);
                if (!File.Exists(src)) { skipped++; continue; }

                var dstRel = Path.Combine(packName, info.RelPath);
                var dst = Path.Combine(assetsRoot, dstRel);

                if (!File.Exists(dst))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
                    File.Copy(src, dst);
                    copied++;
                }

                copyMap[info.RelPath] = dstRel;
            }

            Debug.Log($"[UnityImporter]   Copied: {copied}, Converted: {converted}, Skipped: {skipped}");
            return copyMap;
        }

        //  Phase 4: Harvest GUIDs
        /// <returns>Unity GUID -> Freefall GUID</returns>
        private static Dictionary<string, string> HarvestGuids(
            string assetsRoot, Dictionary<string, string> copyMap, DiscoveryResult discovery)
        {
            Debug.Log("[UnityImporter] Phase 4: Harvesting Freefall GUIDs...");
            var unityToFreefall = new Dictionary<string, string>();
            int found = 0, missing = 0;

            // Build reverse map: unity rel path -> unity guid
            var pathToUnityGuid = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var (guid, info) in discovery.Textures)
                pathToUnityGuid.TryAdd(info.RelPath, guid);
            foreach (var (guid, info) in discovery.Models)
                pathToUnityGuid.TryAdd(info.RelPath, guid);

            foreach (var (unityPath, freefallPath) in copyMap)
            {
                var normalizedPath = freefallPath.Replace('/', '\\').TrimStart('\\');
                var freefallGuid = AssetDatabase.PathToGuid(normalizedPath);
                if (freefallGuid != null && pathToUnityGuid.TryGetValue(unityPath, out var unityGuid))
                {
                    unityToFreefall[unityGuid] = freefallGuid;
                    found++;
                }
                else
                {
                    missing++;
                    if (freefallGuid == null)
                        Debug.LogWarning("UnityImporter", $"  HarvestGuids: AssetDatabase has no GUID for path '{normalizedPath}'");
                }
            }

            Debug.Log($"[UnityImporter]   Mapped: {found}, Missing: {missing}");
            return unityToFreefall;
        }

        //  Phase 5: Create Materials
        private static Dictionary<string, string> CreateMaterials(
            string assetsRoot, string packName, DiscoveryResult discovery,
            Dictionary<string, string> guidMap)
        {
            Debug.Log("[UnityImporter] Phase 5: Creating Freefall materials...");
            var materialGuidMap = new Dictionary<string, string>();
            int created = 0, skipped = 0;

            foreach (var (unityMatGuid, matRef) in discovery.Materials)
            {
                // Map Unity texture slots -> Freefall texture GUIDs
                var freefallSlots = new Dictionary<string, string>();

                // Process in priority order for Albedo
                string albedoGuid = null;
                foreach (var slot in AlbedoPriority)
                {
                    if (matRef.TextureSlots.TryGetValue(slot, out var texGuid) &&
                        guidMap.TryGetValue(texGuid, out var ffGuid))
                    {
                        albedoGuid = ffGuid;
                        break;
                    }
                }
                if (albedoGuid != null)
                {
                    freefallSlots["Albedo"] = albedoGuid;
                }
                else
                {
                    freefallSlots["Albedo"] = InternalAssets.Guids.DefaultDiffuse;

                    // Diagnostic: why is albedo missing?
                    bool hasAnyAlbedoSlot = false;
                    foreach (var slot in AlbedoPriority)
                    {
                        if (matRef.TextureSlots.TryGetValue(slot, out var texGuid))
                        {
                            hasAnyAlbedoSlot = true;
                            var texName = discovery.Textures.TryGetValue(texGuid, out var texInfo) ? texInfo.Name : "???";
                            bool inGuidMap = guidMap.ContainsKey(texGuid);
                            Debug.LogWarning("UnityImporter",
                                $"Material '{matRef.Name}': {slot} references texture '{texName}' (guid:{texGuid[..8]}...) " +
                                $"but {(inGuidMap ? "GUID mapped OK - unexpected" : "texture NOT in guidMap (not copied/discovered?)")}");
                            break;
                        }
                    }
                    if (!hasAnyAlbedoSlot)
                        Debug.LogWarning("UnityImporter",
                            $"Material '{matRef.Name}': No albedo texture slot found ({string.Join(", ", matRef.TextureSlots.Keys)})");
                }

                // Process remaining slots
                foreach (var (unitySlot, unityTexGuid) in matRef.TextureSlots)
                {
                    if (!HdrpSlotMap.TryGetValue(unitySlot, out var ffSlot)) continue;
                    if (ffSlot == "Albedo") continue; // Already handled above
                    if (freefallSlots.ContainsKey(ffSlot)) continue;
                    if (!guidMap.TryGetValue(unityTexGuid, out var ffTexGuid)) continue;

                    // Skip packed ORMH textures referenced via _OcclusionMap
                    if (unitySlot == "_OcclusionMap" && IsPackedTextureName(unityTexGuid, discovery))
                        continue;

                    freefallSlots[ffSlot] = ffTexGuid;
                }

                if (freefallSlots.Count == 0) { skipped++; continue; }

                // Write .mat file
                var matPath = Path.Combine(assetsRoot, packName, "Materials", matRef.Name + ".mat");
                Directory.CreateDirectory(Path.GetDirectoryName(matPath)!);

                var sb = new StringBuilder();
                sb.AppendLine("!Material");
                sb.AppendLine($"Name: {matRef.Name}");
                foreach (var (slot, guid) in freefallSlots)
                    sb.AppendLine($"{slot}: {guid}");

                File.WriteAllText(matPath, sb.ToString(), Encoding.UTF8);
                created++;

                // Pre-compute deterministic GUID for cross-referencing in Phase 6
                var relPath = Path.GetRelativePath(assetsRoot, matPath).Replace('/', '\\').TrimStart('\\');
                materialGuidMap[unityMatGuid] = DeterministicGuid(relPath);
            }

            Debug.Log($"[UnityImporter]   Created: {created}, Skipped: {skipped}");
            return materialGuidMap;
        }

        /// <summary>
        /// Create Freefall .mat files from authoritative Unity-exported material data.
        /// Uses texture filenames (matched in AssetDatabase) instead of fragile GUID chains.
        /// </summary>
        private static void CreateMaterialsFromJson(
            string assetsRoot, string packName, List<JsonMaterial> exportMaterials)
        {
            Debug.Log($"[UnityImporter] Phase 5 (JSON): Creating {exportMaterials.Count} materials from export data...");
            int created = 0, noAlbedo = 0;

            foreach (var em in exportMaterials)
            {
                var slots = new Dictionary<string, string>();

                AddTexSlot(slots, "Albedo", em.Albedo);
                AddTexSlot(slots, "Normal", em.Normal);
                AddTexSlot(slots, "Emissive", em.Emissive);
                AddTexSlot(slots, "AO", em.AO);
                AddTexSlot(slots, "Metallic", em.Metallic);
                AddTexSlot(slots, "HeightTex", em.Height);

                if (slots.Count == 0)
                    Debug.LogWarning("UnityImporter", $"Material '{em.Name}': No texture slots resolved. JSON has: Albedo='{em.Albedo}', Normal='{em.Normal}', Emissive='{em.Emissive}', AO='{em.AO}', Metallic='{em.Metallic}', Height='{em.Height}'");

                if (!slots.ContainsKey("Albedo"))
                {
                    slots["Albedo"] = InternalAssets.Guids.DefaultDiffuse;
                    if (!string.IsNullOrEmpty(em.Albedo))
                        Debug.LogWarning("UnityImporter", $"Material '{em.Name}': Albedo texture '{em.Albedo}' not found in AssetDatabase");
                    else
                        noAlbedo++;
                }

                var matPath = Path.Combine(assetsRoot, packName, "Materials", em.Name + ".mat");
                Directory.CreateDirectory(Path.GetDirectoryName(matPath)!);

                var sb = new StringBuilder();
                sb.AppendLine("!Material");
                sb.AppendLine($"Name: {em.Name}");
                foreach (var (slot, guid) in slots)
                    sb.AppendLine($"{slot}: {guid}");

                File.WriteAllText(matPath, sb.ToString(), Encoding.UTF8);
                created++;
            }

            Debug.Log($"[UnityImporter]   Created: {created}, No albedo texture: {noAlbedo}");
        }

        private static void AddTexSlot(Dictionary<string, string> slots, string slotName, string texFilename)
        {
            if (string.IsNullOrEmpty(texFilename)) return;
            var guid = FindTextureGuidByFilename(texFilename);
            if (guid != null)
                slots[slotName] = guid;
        }

        private static string FindTextureGuidByFilename(string filename)
        {
            // Try exact filename match, then extension fallbacks for converted textures
            var candidates = new[] { filename, Path.ChangeExtension(filename, ".png"), Path.ChangeExtension(filename, ".tga"), Path.ChangeExtension(filename, ".tif") };
            foreach (var candidate in candidates)
            {
                foreach (var path in AssetDatabase.GetAllPaths())
                {
                    if (Path.GetFileName(path).Equals(candidate, StringComparison.OrdinalIgnoreCase))
                    {
                        var guid = AssetDatabase.PathToGuid(path);
                        if (guid != null) return guid;
                    }
                }
            }
            Debug.LogWarning("UnityImporter", $"  FindTextureGuidByFilename: '{filename}' not found (tried: {string.Join(", ", candidates)})");
            DiagLog($"  TEX_MISS '{filename}' (tried: {string.Join(", ", candidates)})");
            return null;
        }

        /// <summary>
        /// Build material name → Freefall GUID map from imported .mat files.
        /// Used with JSON-exported material data for direct name-based matching.
        /// </summary>
        private static Dictionary<string, string> BuildMaterialNameToGuidMap(List<JsonMaterial> exportMaterials)
        {
            var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var em in exportMaterials)
            {
                var pattern = em.Name + ".mat";
                foreach (var path in AssetDatabase.GetAllPaths())
                {
                    if (Path.GetFileName(path).Equals(pattern, StringComparison.OrdinalIgnoreCase))
                    {
                        var guid = AssetDatabase.PathToGuid(path);
                        if (guid != null) { map[em.Name] = guid; break; }
                    }
                }
            }
            Debug.Log($"[UnityImporter]   Material name→GUID map: {map.Count}/{exportMaterials.Count} resolved");
            DiagLog($"MaterialNameToGuid: {map.Count}/{exportMaterials.Count} resolved");
            if (map.Count < exportMaterials.Count)
            {
                foreach (var em in exportMaterials)
                    if (!map.ContainsKey(em.Name))
                        DiagLog($"  MISSING .mat for: '{em.Name}'");
            }
            return map;
        }


        /// <summary>Extract material name from Assimp mesh part name like "SM_Door_LOD0 [Material.019]"</summary>
        private static string ExtractMaterialName(string partName)
        {
            if (string.IsNullOrEmpty(partName)) return null;
            int start = partName.LastIndexOf('[');
            int end = partName.LastIndexOf(']');
            if (start >= 0 && end > start)
                return partName.Substring(start + 1, end - start - 1);
            return null;
        }


        /// <summary>Parse externalObjects from FBX .meta to extract material name → GUID mappings.</summary>
        private static void ParseFbxMetaMaterials(string metaPath, Dictionary<string, string> materialNameToGuid)
        {
            try
            {
                var lines = File.ReadAllLines(metaPath);
                bool inExternalObjects = false;
                string currentName = null;

                for (int i = 0; i < lines.Length; i++)
                {
                    var line = lines[i];

                    if (line.TrimStart().StartsWith("externalObjects:"))
                    {
                        inExternalObjects = true;
                        continue;
                    }

                    if (!inExternalObjects) continue;

                    // End of externalObjects section
                    if (!line.StartsWith(" ") && !line.StartsWith("\t") && line.Length > 0 && !line.TrimStart().StartsWith("-"))
                    {
                        inExternalObjects = false;
                        continue;
                    }

                    // Capture material slot name
                    if (line.Contains("name:"))
                    {
                        var nameVal = line.Substring(line.IndexOf("name:") + 5).Trim();
                        currentName = nameVal;
                    }

                    // Capture GUID
                    if (line.Contains("second:") && line.Contains("guid:"))
                    {
                        var match = Regex.Match(line, @"guid:\s*([a-f0-9]+)");
                        if (match.Success && currentName != null)
                        {
                            materialNameToGuid.TryAdd(currentName, match.Groups[1].Value);
                        }
                        currentName = null;
                    }
                }
            }
            catch { }
        }

        /// <summary>Parse FBX .meta for the effective file scale.
        /// FBX files in cm (useFileScale=1, globalScale=1) need scale 100
        /// because Unity's FBX importer applies 0.01 to the root transform,
        /// then LOD renderers have scale 100 to compensate.</summary>
        private static float ParseFbxFileScale(string metaPath)
        {
            try
            {
                var lines = File.ReadAllLines(metaPath);
                bool useFileScale = false;
                float globalScale = 1f;
                bool inMeshes = false;

                foreach (var line in lines)
                {
                    var trimmed = line.TrimStart();
                    if (trimmed.StartsWith("meshes:")) { inMeshes = true; continue; }
                    if (inMeshes && !line.StartsWith(" ") && !line.StartsWith("\t") && trimmed.Length > 0)
                        inMeshes = false;

                    if (inMeshes)
                    {
                        if (trimmed.StartsWith("useFileScale:"))
                            useFileScale = trimmed.EndsWith("1");
                        else if (trimmed.StartsWith("globalScale:") && float.TryParse(
                            trimmed.Substring(12).Trim(), System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture, out var gs))
                            globalScale = gs;
                    }
                }

                // useFileScale=1 + globalScale=1 → renderer children use scale 100
                if (useFileScale && globalScale == 1f)
                    return 100f;

                return globalScale;
            }
            catch { return 1f; }
        }

        // JSON data classes matching Unity ExportTools output
        private class PrefabExportData
        {
            public List<ExportEntity> Prefabs { get; set; } = new();
            public List<JsonMaterial> Materials { get; set; } = new();
        }

        private class ExportEntity
        {
            public string Name { get; set; }
            public string Path { get; set; }
            public JsonVector3 Position { get; set; }
            public JsonQuaternion Rotation { get; set; }
            public JsonVector3 Scale { get; set; }
            public string Mesh { get; set; }
            public string MeshAlt { get; set; }
            public List<ExportEntity> Children { get; set; } = new();
            public List<string> Materials { get; set; } = new();
        }

        private class JsonMaterial
        {
            public string Name { get; set; }
            public string Albedo { get; set; }
            public string Normal { get; set; }
            public string Emissive { get; set; }
            public string AO { get; set; }
            public string Metallic { get; set; }
            public string Height { get; set; }
        }


        private class JsonVector3
        {
            public float x { get; set; }
            public float y { get; set; }
            public float z { get; set; }
            public Vector3 ToVector3() => new(x, y, z);
        }

        private class JsonQuaternion
        {
            public float x { get; set; }
            public float y { get; set; }
            public float z { get; set; }
            public float w { get; set; } = 1;
            public Quaternion ToQuaternion() => new(x, y, z, w);
        }

        /// <summary>
        /// Convert Unity prefabs to Freefall prefabs from JSON exported by Unity's ExportTools.
        /// Each exported prefab becomes a Freefall .prefab file with MeshRenderer entities.
        /// </summary>
        private static void CreatePrefabsFromJson(
            string jsonPath, string assetsRoot, string packName,
            Action<string> onProgress = null)
        {
            Debug.Log($"[UnityImporter] Phase 6: Creating Freefall prefabs from JSON...");
            Debug.Log($"[UnityImporter]   Source: {jsonPath}");

            var jsonText = File.ReadAllText(jsonPath);
            var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
            var data = JsonSerializer.Deserialize<PrefabExportData>(jsonText, options);

            if (data?.Prefabs == null || data.Prefabs.Count == 0)
            {
                Debug.Log("[UnityImporter]   No prefabs found in JSON");
                return;
            }

            int created = 0, skipped = 0;

            foreach (var exportPrefab in data.Prefabs)
            {
                var tempEntities = new List<Entity>();

                // Root entity
                var rootEntity = new Entity(exportPrefab.Name, register: false);
                tempEntities.Add(rootEntity);

                if (exportPrefab.Children.Count > 0)
                    CreateChildEntities(exportPrefab, rootEntity, tempEntities);

                // Single mesh prefab (no children, root has mesh)
                if (!string.IsNullOrEmpty(exportPrefab.Mesh))
                    AttachMeshRenderer(rootEntity, exportPrefab.Mesh, exportPrefab.Materials, exportPrefab.MeshAlt);

                if (tempEntities.Count <= 1 && rootEntity.GetComponent<MeshRenderer>() == null)
                {
                    foreach (var e in tempEntities) e.Destroy();
                    skipped++;
                    continue;
                }

                var serializer = new EntitySerializer();
                var yamlStr = serializer.SaveToString(tempEntities);

                foreach (var e in tempEntities)
                    e.Destroy();

                // Use exported Path if available, otherwise fall back to flat "Prefabs" folder
                string prefabPath;
                if (!string.IsNullOrEmpty(exportPrefab.Path))
                {
                    // Path is relative from pack root, e.g. "Chapel/Prefabs/Chapel.prefab"
                    // Strip the .prefab extension and replace with our name
                    var relDir = System.IO.Path.GetDirectoryName(exportPrefab.Path)?.Replace('\\', '/');
                    if (string.IsNullOrEmpty(relDir))
                        prefabPath = System.IO.Path.Combine(assetsRoot, packName, exportPrefab.Name + ".prefab");
                    else
                        prefabPath = System.IO.Path.Combine(assetsRoot, packName, relDir, exportPrefab.Name + ".prefab");
                }
                else
                {
                    prefabPath = System.IO.Path.Combine(assetsRoot, packName, "Prefabs", exportPrefab.Name + ".prefab");
                }
                Directory.CreateDirectory(Path.GetDirectoryName(prefabPath)!);
                File.WriteAllText(prefabPath, yamlStr, Encoding.UTF8);
                created++;

                onProgress?.Invoke($"Prefab: {exportPrefab.Name}");
            }

            Debug.Log($"[UnityImporter]   Prefabs created: {created}, Skipped: {skipped}");
        }

        private static void CreateChildEntities(
            ExportEntity exportNode, Entity parentEntity, List<Entity> allEntities)
        {
            foreach (var child in exportNode.Children)
            {
                var entity = new Entity(child.Name ?? "Part", register: false);
                entity.Transform.Parent = parentEntity.Transform;
                entity.Transform.Position = child.Position?.ToVector3() ?? Vector3.Zero;
                entity.Transform.Rotation = child.Rotation?.ToQuaternion() ?? Quaternion.Identity;
                entity.Transform.Scale = child.Scale?.ToVector3() ?? Vector3.One;

                if (!string.IsNullOrEmpty(child.Mesh))
                    AttachMeshRenderer(entity, child.Mesh, child.Materials, child.MeshAlt);

                allEntities.Add(entity);

                if (child.Children.Count > 0)
                    CreateChildEntities(child, entity, allEntities);
            }
        }

        /// <summary>
        /// Resolve a mesh name to its MeshData sub-asset GUID and attach a MeshRenderer.
        /// Loads the companion PrefabData sub-asset (built by ModelImporter.PostImport)
        /// to copy pre-resolved material overrides.
        /// </summary>
        private static void AttachMeshRenderer(Entity entity, string meshName, List<string> materialNames = null, string meshAlt = null)
        {
            // Try alt name first (internal mesh node name — more specific for multi-mesh FBX files)
            string meshGuid = null;
            if (!string.IsNullOrEmpty(meshAlt))
                meshGuid = AssetDatabase.ResolveGuidByName(meshAlt, nameof(MeshData));
            if (meshGuid == null)
                meshGuid = AssetDatabase.ResolveGuidByName(meshName, nameof(MeshData));
            if (meshGuid == null)
            {
                Debug.LogWarning("UnityImporter", $"Mesh not found: '{meshName}' for entity '{entity.Name}'");
                return;
            }

            var renderer = new MeshRenderer();
            renderer.Mesh = new Mesh { Guid = meshGuid };

            // Assign materials from Unity export data (per-entity material names)
            if (materialNames != null && materialNames.Count > 0 && _materialNameToGuid.Count > 0)
            {
                for (int i = 0; i < materialNames.Count; i++)
                {
                    var matName = materialNames[i];
                    if (string.IsNullOrEmpty(matName)) continue;

                    _materialNameToGuid.TryGetValue(matName, out var matGuid);
                    matGuid ??= InternalAssets.Guids.DefaultMaterial;

                    renderer.Materials.Add(new MaterialOverride
                    {
                        MaterialSlot = i,
                        Material = new Graphics.Material { Guid = matGuid }
                    });
                }
            }
            else
            {
                // Fallback: load materials from the companion PrefabData sub-asset
                var prefabGuid = AssetDatabase.ResolveGuidByName(meshName, nameof(PrefabData));
                if (prefabGuid != null)
                {
                    var cachePath = AssetDatabase.ResolveCachePathByGuid(prefabGuid, typeof(PrefabData));
                    if (cachePath != null && File.Exists(cachePath))
                    {
                        try
                        {
                            var packer = new Freefall.Assets.Packers.PrefabPacker();
                            using var stream = File.OpenRead(cachePath);
                            var prefabData = packer.Read(stream);

                            if (prefabData?.Yaml != null)
                            {
                                var serializer = new EntitySerializer();
                                var prefabEntities = serializer.LoadFromBytes(prefabData.Yaml);

                                if (prefabEntities?.Count > 0)
                                {
                                    var srcRenderer = prefabEntities[0].GetComponent<MeshRenderer>();
                                    if (srcRenderer != null)
                                        renderer.Materials = srcRenderer.Materials;
                                }

                                foreach (var e in prefabEntities)
                                    e.Destroy();
                            }
                        }
                        catch (Exception ex)
                        {
                            Debug.LogWarning("UnityImporter",
                                $"Failed to load PrefabData for '{meshName}': {ex.Message}");
                        }
                    }
                }
            }

            entity.AddComponent(renderer);
        }
       
        /// <summary>
        /// Strip Assimp-appended suffixes like " [Material.003]" from mesh part names.
        /// </summary>
        private static string StripAssimpSuffix(string name)
        {
            if (string.IsNullOrEmpty(name)) return name;
            int bracketIdx = name.IndexOf(" [");
            return bracketIdx >= 0 ? name.Substring(0, bracketIdx) : name;
        }

        /// <summary>
        /// Try to find a material's Freefall GUID by looking up its file path in AssetDatabase.
        /// </summary>
        private static string FindMaterialGuid(string assetsRoot, DiscoveryResult discovery, string unityMatGuid)
        {
            if (!discovery.Materials.TryGetValue(unityMatGuid, out var matRef)) return null;

            // Search for the .mat file we created
            var pattern = matRef.Name + ".mat";
            foreach (var path in AssetDatabase.GetAllPaths())
            {
                if (path.EndsWith(pattern, StringComparison.OrdinalIgnoreCase))
                {
                    var guid = AssetDatabase.PathToGuid(path);
                    if (guid != null) return guid;
                }
            }
            return null;
        }

        /// <summary>
        /// Convert a PSD/EXR file to PNG using texconv.exe.
        /// </summary>
        private static bool ConvertWithTexconv(string srcPath, string outputDir)
        {
            try
            {
                // texconv.exe is bundled with the editor
                var editorDir = AppDomain.CurrentDomain.BaseDirectory;
                var texconv = Path.Combine(editorDir, "texconv.exe");
                if (!File.Exists(texconv))
                    texconv = Path.Combine(Path.GetDirectoryName(typeof(UnityImporter).Assembly.Location)!, "texconv.exe");
                if (!File.Exists(texconv))
                {
                    Debug.LogWarning("UnityImporter", "texconv.exe not found");
                    return false;
                }

                Directory.CreateDirectory(outputDir);

                var psi = new ProcessStartInfo
                {
                    FileName = texconv,
                    Arguments = $"-ft PNG -y -o \"{outputDir}\" \"{srcPath}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                };

                using var proc = Process.Start(psi);
                proc!.WaitForExit(30000); // 30s timeout
                return proc.ExitCode == 0;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("UnityImporter", $"texconv failed for {Path.GetFileName(srcPath)}: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Returns true if the texture referenced by unityGuid has a name indicating
        /// it's a packed channel texture (ORMH, MaskMap, etc.) rather than a standalone AO map.
        /// </summary>
        private static bool IsPackedTextureName(string unityGuid, DiscoveryResult discovery)
        {
            if (!discovery.Textures.TryGetValue(unityGuid, out var info))
                return false;

            var name = Path.GetFileNameWithoutExtension(info.RelPath);

            // Common packed texture naming conventions
            return name.Contains("ORMH", StringComparison.OrdinalIgnoreCase)
                || name.Contains("_ORM", StringComparison.OrdinalIgnoreCase)
                || name.Contains("MaskMap", StringComparison.OrdinalIgnoreCase)
                || name.Contains("_Packed", StringComparison.OrdinalIgnoreCase)
                || name.Contains("_Pack", StringComparison.OrdinalIgnoreCase);
        }

        private static string DeterministicGuid(string path)
        {
            var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(path.ToLowerInvariant()));
            return new Guid(bytes.AsSpan(0, 16)).ToString("N");
        }
    }
}
