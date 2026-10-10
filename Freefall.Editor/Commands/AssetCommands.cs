using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Freefall.Assets;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// GET /api/asset/get?guid=X — read every public field/property of an asset (loads it if needed).
    /// Field names are the exact member names accepted by /api/asset/setproperty.
    /// </summary>
    [CommandRoute("GET", "/api/asset/get")]
    public class GetAssetCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var query = CommandHelpers.ParseQueryString(context.Path);
            if (!query.TryGetValue("guid", out var guid) || string.IsNullOrWhiteSpace(guid))
                return CommandResult.BadRequest("Query parameter 'guid' is required");

            var asset = CommandHelpers.FindOrLoadAsset(guid);
            if (asset == null)
                return CommandResult.NotFound($"Asset not found: {guid}");

            return CommandResult.Json(new
            {
                guid,
                type = asset.GetType().Name,
                name = asset.Name,
                path = AssetDatabase.GuidToPath(guid),
                fields = CommandHelpers.SerializeMembers(asset)
            });
        }
    }

    /// <summary>
    /// POST /api/assets/refresh — rescan the Assets folder and import new/changed files
    /// (for files written to disk outside the editor, e.g. generated textures or meshes).
    /// </summary>
    [CommandRoute("POST", "/api/assets/refresh")]
    public class RefreshAssetsCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            int before = AssetDatabase.GetAllPaths().Count();
            AssetDatabase.Refresh();
            AssetDatabase.ImportAll();
            int after = AssetDatabase.GetAllPaths().Count();

            // Update loaded meshes/textures/materials/prefabs/.asset files in place now (the engine tick would
            // otherwise do it next frame) so the response can report it.
            int reloaded = Engine.Assets.ReloadReimported();

            return CommandResult.Json(new { status = "refreshed", assetCount = after, added = after - before, reloaded });
        }
    }

    [CommandRoute("GET", "/api/assets/list")]
    public class ListAssetsCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            var qs = CommandHelpers.ParseQueryString(context.Path);
            var subPath = qs.TryGetValue("path", out var p) ? p : "";

            var assetsDir = Engine.Project.AssetsDirectory;
            var targetDir = string.IsNullOrEmpty(subPath)
                ? assetsDir
                : Path.Combine(assetsDir, subPath);

            if (!Directory.Exists(targetDir))
                return CommandResult.NotFound($"Directory not found: {subPath}");

            var folders = Directory.GetDirectories(targetDir)
                .Select(d => new DirectoryInfo(d))
                .Select(d => new { name = d.Name, type = "folder" })
                .ToArray();

            var files = Directory.GetFiles(targetDir)
                .Where(f => AssetDatabase.IsImportableExtension(Path.GetExtension(f)))
                .Select(f => new FileInfo(f))
                .Select(f => new
                {
                    name = f.Name,
                    type = "file",
                    extension = f.Extension,
                    sizeBytes = f.Length
                })
                .ToArray();

            return CommandResult.Json(new
            {
                path = subPath,
                folderCount = folders.Length,
                fileCount = files.Length,
                folders,
                files
            });
        }
    }

    /// <summary>
    /// Search assets by name, with optional type filter.
    /// GET /api/assets/search?query=rock&amp;type=staticmesh&amp;limit=20
    /// </summary>
    [CommandRoute("GET", "/api/assets/search")]
    public class SearchAssetsCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            var qs = CommandHelpers.ParseQueryString(context.Path);
            if (!qs.TryGetValue("query", out var query) || string.IsNullOrWhiteSpace(query))
                return CommandResult.BadRequest("Query parameter 'query' required");

            var limit = qs.TryGetValue("limit", out var limitStr) && int.TryParse(limitStr, out var l) ? l : 50;
            qs.TryGetValue("type", out var typeFilter);

            var results = new List<object>();
            var allPaths = AssetDatabase.GetAllPaths();

            foreach (var path in allPaths)
            {
                var fileName = Path.GetFileNameWithoutExtension(path);
                if (!fileName.Contains(query, StringComparison.OrdinalIgnoreCase))
                    continue;

                if (!string.IsNullOrEmpty(typeFilter))
                {
                    var ext = Path.GetExtension(path);
                    var extNoDot = ext.StartsWith(".") ? ext.Substring(1) : ext;
                    if (!extNoDot.Equals(typeFilter, StringComparison.OrdinalIgnoreCase))
                        continue;
                }

                var guid = AssetDatabase.PathToGuid(path);
                var meta = guid != null ? AssetDatabase.GetMeta(guid) : null;

                results.Add(new
                {
                    path,
                    name = fileName,
                    extension = Path.GetExtension(path),
                    guid,
                    importerType = meta?.ImporterType
                });

                if (results.Count >= limit)
                    break;
            }

            return CommandResult.Json(new
            {
                query,
                type = typeFilter,
                count = results.Count,
                results
            });
        }
    }

    /// <summary>
    /// List all registered importable asset types (extensions + importer names).
    /// GET /api/assets/types
    /// </summary>
    [CommandRoute("GET", "/api/assets/types")]
    public class ListAssetTypesCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var extensions = AssetDatabase.GetImportableExtensions()
                .OrderBy(e => e.extension)
                .Select(e => new { extension = e.extension, importerType = e.importerType })
                .ToArray();

            return CommandResult.Json(new { count = extensions.Length, extensions });
        }
    }

    /// <summary>
    /// Resolve an asset name to its GUID.
    /// GET /api/assets/resolve?name=castle_tower
    /// </summary>
    [CommandRoute("GET", "/api/assets/resolve")]
    public class ResolveAssetCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            var qs = CommandHelpers.ParseQueryString(context.Path);
            if (!qs.TryGetValue("name", out var name) || string.IsNullOrWhiteSpace(name))
                return CommandResult.BadRequest("Query parameter 'name' required");

            // Optional type filter: names are shared across types (a prefab and the FBX mesh it wraps).
            var guid = qs.TryGetValue("type", out var type) && !string.IsNullOrWhiteSpace(type)
                ? AssetDatabase.ResolveGuidByName(name, type)
                : AssetDatabase.ResolveGuidByName(name);
            if (guid == null)
                return CommandResult.NotFound($"No asset found with name '{name}'" + (type != null ? $" of type {type}" : ""));

            var friendlyName = AssetDatabase.ResolveFriendlyName(guid);
            var path = AssetDatabase.GuidToPath(guid);

            return CommandResult.Json(new
            {
                name = friendlyName,
                guid,
                path
            });
        }
    }

    /// <summary>
    /// Create a new asset on disk and import it.
    /// POST /api/asset/create
    /// Body: {"type":"Terrain", "name":"MyTerrain"}
    /// Optionally: {"path": "Pack/MyAsset.terrain"} to specify a save location.
    /// </summary>
    [CommandRoute("POST", "/api/asset/create")]
    public class CreateAssetCommand : EditorCommand
    {
        private static Dictionary<string, Type> _assetTypes;

        /// <summary>Forget the cached types after a script reload (it would pin the old script assembly).</summary>
        internal static void ResetTypeCache() => _assetTypes = null;

        private static Dictionary<string, Type> AssetTypes
        {
            get
            {
                if (_assetTypes != null) return _assetTypes;
                _assetTypes = new Dictionary<string, Type>(StringComparer.OrdinalIgnoreCase);
                foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
                {
                    if (ScriptCompiler.IsStale(asm)) continue;
                    try
                    {
                        foreach (var type in asm.GetTypes())
                            if (!type.IsAbstract && typeof(Asset).IsAssignableFrom(type)
                                && type.GetConstructor(Type.EmptyTypes) != null)
                                _assetTypes[type.Name] = type;
                    }
                    catch { }
                }
                return _assetTypes;
            }
        }

        public override CommandResult Execute(CommandContext context)
        {
            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required: {\"type\":\"Terrain\"}");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            string typeName = root.TryGetProperty("type", out var t) ? t.GetString() : null;
            if (string.IsNullOrEmpty(typeName))
                return CommandResult.BadRequest("'type' is required");

            if (!AssetTypes.TryGetValue(typeName, out var assetType))
                return CommandResult.BadRequest($"Unknown asset type '{typeName}'. Available: {string.Join(", ", AssetTypes.Keys)}");

            string name = root.TryGetProperty("name", out var n) ? n.GetString() : typeName;

            var projectDir = Engine.Project?.AssetsDirectory;
            if (string.IsNullOrEmpty(projectDir))
                return CommandResult.BadRequest("No project open");

            // Determine folder path (support custom path in body)
            string folderPath;
            if (root.TryGetProperty("path", out var pathEl) && !string.IsNullOrEmpty(pathEl.GetString()))
            {
                var customPath = Path.Combine(projectDir, pathEl.GetString());
                folderPath = Path.GetDirectoryName(customPath);
                name = Path.GetFileNameWithoutExtension(customPath);
            }
            else
            {
                folderPath = projectDir;
            }

            if (!string.IsNullOrEmpty(folderPath) && !Directory.Exists(folderPath))
                Directory.CreateDirectory(folderPath);

            try
            {
                var guid = AssetCreator.CreateAsset(assetType, folderPath, name);
                if (string.IsNullOrEmpty(guid))
                    return CommandResult.Error(500, $"Failed to create asset of type '{typeName}': {AssetCreator.LastError ?? "unknown error"}");

                var asset = Engine.Assets.LoadByGuid(guid, assetType);
                return CommandResult.Json(new { guid, type = assetType.Name, name = asset?.Name ?? name });
            }
            catch (Exception ex)
            {
                return CommandResult.Error(500, $"Failed to create asset: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Save an asset to disk by GUID.
    /// POST /api/asset/save
    /// Body: {"guid":"..."} or {"guid":"...", "path":"D:/custom/path.terrain"}
    /// </summary>
    [CommandRoute("POST", "/api/asset/save")]
    public class SaveAssetCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'guid'");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            string guid = root.TryGetProperty("guid", out var gp) ? gp.GetString() : null;
            if (string.IsNullOrEmpty(guid))
                return CommandResult.BadRequest("'guid' is required");

            var asset = Engine.Assets.FindByGuid(guid);
            if (asset == null)
                return CommandResult.NotFound($"Asset not found: {guid}");

            string savePath = root.TryGetProperty("path", out var pp) ? pp.GetString() : null;
            if (string.IsNullOrEmpty(savePath))
                savePath = asset.AssetPath;
            if (string.IsNullOrEmpty(savePath))
            {
                var projectDir = Engine.Project?.AssetsDirectory;
                if (string.IsNullOrEmpty(projectDir))
                    return CommandResult.BadRequest("No save path: asset has no registered path and no project is open");

                string ext = AssetManager.GetFileExtension(asset.GetType());
                savePath = Path.Combine(projectDir, (asset.Name ?? "asset") + ext);
            }

            try
            {
                Engine.Assets.SaveAsset(asset, savePath);
                return CommandResult.Json(new
                {
                    status = "saved",
                    path = savePath,
                    type = asset.GetType().Name,
                    name = asset.Name,
                    guid
                });
            }
            catch (Exception ex)
            {
                return CommandResult.Error(500, $"Failed to save asset: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Set one or more fields on a loaded asset by reflection, then save + reimport it
    /// so the Library cache matches the source file.
    /// POST /api/asset/setproperty
    /// Body: {"guid":"...", "property":"CloudCoverage", "value":0.8}
    ///   or: {"guid":"...", "values": {"CloudCoverage":0.8, "SkyTintColor":[0.5,0.6,0.7]}}
    /// Optional: "save": false to only change the in-memory asset.
    /// Value encoding is the same as entity setproperty (floats, vectors, enums, asset GUIDs; Color3 as [r,g,b]).
    /// </summary>
    [CommandRoute("POST", "/api/asset/setproperty")]
    public class SetAssetPropertyCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'guid' and 'property'/'value' or 'values'");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            string guid = root.TryGetProperty("guid", out var gp) ? gp.GetString() : null;
            if (string.IsNullOrEmpty(guid))
                return CommandResult.BadRequest("'guid' is required");

            var asset = CommandHelpers.FindOrLoadAsset(guid);
            if (asset == null)
                return CommandResult.NotFound($"Asset not found: {guid}");

            var assetType = asset.GetType();
            var applied = new Dictionary<string, object>();

            void Apply(string propertyName, JsonElement valueEl)
            {
                var field = Freefall.Reflection.Reflector.GetField(assetType, propertyName);
                if (field == null)
                    throw new InvalidOperationException($"Property '{propertyName}' not found on '{assetType.Name}'");
                object converted = SetPropertyCommand.ConvertJsonValue(valueEl, field.Type);
                field.SetValue(asset, converted);
                applied[propertyName] = CommandHelpers.SerializeValue(field.GetValue(asset));
            }

            try
            {
                if (root.TryGetProperty("values", out var values) && values.ValueKind == JsonValueKind.Object)
                {
                    foreach (var kv in values.EnumerateObject())
                        Apply(kv.Name, kv.Value);
                }
                else
                {
                    if (!root.TryGetProperty("property", out var pp) || !root.TryGetProperty("value", out var vp))
                        return CommandResult.BadRequest("Body must contain 'property' and 'value', or a 'values' object");
                    Apply(pp.GetString(), vp);
                }
            }
            catch (Exception ex)
            {
                return CommandResult.Error(400, $"Failed to set property: {ex.Message}");
            }

            // Terrain renders from GPU-side derived data; without this, layer/size/decoration edits stay invisible.
            if (asset is Terrain terrain)
                terrain.MarkForUpdate(TerrainDirtyFlags.All);

            bool save = !root.TryGetProperty("save", out var sp) || sp.GetBoolean();
            string savedPath = null;
            if (save)
            {
                try
                {
                    var relativePath = AssetDatabase.GuidToPath(guid);
                    if (relativePath != null && Engine.Project != null)
                    {
                        savedPath = Path.Combine(Engine.Project.AssetsDirectory, relativePath);
                        Engine.Assets.SaveAsset(asset, savedPath);
                        AssetDatabase.ImportAssetByPath(relativePath);
                        asset.ClearDirty();
                    }
                    else
                    {
                        asset.MarkDirty();
                    }
                }
                catch (Exception ex)
                {
                    return CommandResult.Error(500, $"Values applied in memory but save failed: {ex.Message}");
                }
            }
            else
            {
                asset.MarkDirty();
            }

            return CommandResult.Json(new { status = "set", guid, type = assetType.Name, name = asset.Name, applied, saved = savedPath });
        }
    }

    /// <summary>
    /// Import a Unity asset pack into the Freefall project.
    /// POST /api/tools/import-unity-pack
    /// Body: { "source": "D:/UnityPacks/MedievalTown", "pack": "MedievalTown" }
    /// </summary>
    [CommandRoute("POST", "/api/tools/import-unity-pack")]
    public class ImportUnityPackCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext ctx)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            string source = null, pack = null;
            try
            {
                using var doc = ctx.ParseBody();
                source = doc.RootElement.GetProperty("source").GetString();
                pack = doc.RootElement.GetProperty("pack").GetString();
            }
            catch { }

            if (string.IsNullOrEmpty(source) || string.IsNullOrEmpty(pack))
                return CommandResult.Error(400, "Required: source (Unity source dir), pack (pack name)");

            if (!Directory.Exists(source))
                return CommandResult.Error(404, $"Source directory not found: {source}");

            var assetsRoot = Engine.Project.AssetsDirectory;

            try
            {
                Tools.UnityImporter.Import(source, assetsRoot, pack);
                return CommandResult.Json(new { status = "complete", pack, assetsRoot });
            }
            catch (Exception ex)
            {
                return CommandResult.Error(500, $"Import failed: {ex.Message}");
            }
        }
    }
}
