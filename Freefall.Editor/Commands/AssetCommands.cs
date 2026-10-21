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

    /// <summary>
    /// Shared by the commands that start a forced reimport (AssetDatabase.ReimportFiles): the answers for a
    /// running job, a started one and a finished one.
    /// </summary>
    internal static class ReimportResults
    {
        public static CommandResult Busy(ReimportJob running) => CommandResult.Error(409,
            $"A reimport is running: {running.Describe()}. It goes on by itself (console_log shows its progress); " +
            "start another one when it is done.");

        public static object Started(ReimportJob job) => new
        {
            status = "started",
            path = job.Label,
            total = job.Total,
            note = "Importing a few files per frame; console_log shows the progress, the last line the result.",
        };

        public static object Finished(ReimportJob job) => new
        {
            status = job.Aborted ? "aborted" : "reimported",
            path = job.Label,
            total = job.Total,
            imported = job.Imported,
            failed = job.Failed,
            failedPaths = job.FailedPaths.Take(50).ToArray(),
            reloaded = job.Reloaded,
            seconds = Math.Round(job.Elapsed.TotalSeconds, 1),
        };

        /// <summary>Run 'start' on the main thread; if it started a job and the caller waits, answer when the job ends.</summary>
        public static async Task<CommandResult> StartAndWaitAsync(Func<(CommandResult result, ReimportJob job, bool wait)> start,
            Func<ReimportJob, object> finished)
        {
            var (result, job, wait) = await EditorCommandServer.Instance.RunOnMainThread(start);
            if (job == null || !wait) return result;

            // No time limit: a caller that gives up leaves the job running, it never depended on this wait
            return CommandResult.Json(finished(await job.Completion));
        }

        /// <summary>Extensions from a JSON array of strings, or null.</summary>
        public static string[] ParseExtensions(JsonElement root) =>
            root.TryGetProperty("extensions", out var e) && e.ValueKind == JsonValueKind.Array
                ? e.EnumerateArray().Select(x => x.GetString()).Where(x => !string.IsNullOrWhiteSpace(x)).ToArray()
                : null;

        public static bool ParseBool(JsonElement root, string name, bool fallback) =>
            root.TryGetProperty(name, out var b) && (b.ValueKind == JsonValueKind.True || b.ValueKind == JsonValueKind.False)
                ? b.GetBoolean() : fallback;
    }

    /// <summary>
    /// Reimport every tracked file under a folder of Assets (or one file), changed or not: for cache files that
    /// predate an importer change or new import settings (asset_refresh only imports files newer than their
    /// last import).
    /// POST /api/assets/reimport
    /// Body: {"path":"Medieval Mega Pack", "extensions":["fbx"], "wait":true}   ("" = all of Assets)
    /// The files import on the main thread, a few per frame (AssetDatabase.TickReimports): the editor keeps
    /// drawing and the console shows the progress. With wait the answer comes after the last file; a caller
    /// that times out leaves the reimport running, and a second call answers with its progress.
    /// </summary>
    [CommandRoute("POST", "/api/assets/reimport")]
    public class ReimportAssetsCommand : EditorCommand, IAsyncEditorCommand
    {
        // Synchronous fallback: starts the reimport and answers right away
        public override CommandResult Execute(CommandContext context) => Start(context).result;

        public Task<CommandResult> ExecuteAsync(CommandContext context)
            => ReimportResults.StartAndWaitAsync(() => Start(context), ReimportResults.Finished);

        private static (CommandResult result, ReimportJob job, bool wait) Start(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return (CommandResult.Error(409, "No project is open"), null, false);

            using var doc = context.ParseBody();
            var root = doc.RootElement;
            if (!root.TryGetProperty("path", out var p) || p.ValueKind != JsonValueKind.String)
                return (CommandResult.BadRequest("'path' is required: a folder or file relative to Assets, \"\" for all of Assets"), null, false);

            var path = p.GetString().Trim().TrimStart('/', '\\');
            var extensions = ReimportResults.ParseExtensions(root);
            bool wait = ReimportResults.ParseBool(root, "wait", true);

            if (AssetDatabase.ActiveReimport is { } running)
                return (ReimportResults.Busy(running), null, false);

            var fullPath = Path.Combine(Engine.Project.AssetsDirectory, path);
            if (!Directory.Exists(fullPath) && AssetDatabase.PathToGuid(path) == null)
                return (CommandResult.NotFound($"No folder or tracked file '{path}' in Assets"), null, false);

            if (AssetDatabase.GetPathsInFolder(path, extensions).Count == 0)
                return (CommandResult.NotFound($"No importable files in '{path}'"
                    + (extensions != null ? $" with extension {string.Join(", ", extensions)}" : "")), null, false);

            var job = AssetDatabase.ReimportFolder(path, extensions);
            return (CommandResult.Json(ReimportResults.Started(job)), job, wait);
        }
    }

    /// <summary>
    /// Import settings of a source file: its importer's members as the meta stores them.
    /// GET /api/asset/import-settings?path=Pack/Rock.fbx   (or the file's GUID, or a sub-asset's GUID)
    /// </summary>
    [CommandRoute("GET", "/api/asset/import-settings")]
    public class GetImportSettingsCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            var query = CommandHelpers.ParseQueryString(context.Path);
            if (!query.TryGetValue("path", out var path) || string.IsNullOrWhiteSpace(path))
                return CommandResult.BadRequest("Query parameter 'path' is required (file relative to Assets, or a GUID)");

            var guid = ImportSettings.ResolveSourceGuid(path);
            if (guid == null)
                return CommandResult.NotFound($"No tracked file or GUID '{path}'");

            var importer = AssetDatabase.GetImporter(guid);
            if (importer == null)
                return CommandResult.BadRequest($"'{AssetDatabase.GuidToPath(guid)}' has no importer");

            return CommandResult.Json(new
            {
                path = AssetDatabase.GuidToPath(guid),
                guid,
                importer = importer.GetType().Name,
                settings = ImportSettings.Read(importer),
            });
        }
    }

    /// <summary>
    /// Set import settings on one source file or on every tracked file under a folder, save them to the
    /// metas and (by default) reimport the changed files.
    /// POST /api/asset/import-settings
    /// Body: {"path":"Pack/Rocks", "values":{"LODPixelError":6}, "extensions":["fbx"], "reimport":true, "wait":true}
    /// In a folder only files whose importer has every given setting change (a texture has no LODPixelError);
    /// the rest are counted as skipped. Nothing is saved if a value does not convert. The reimport is a
    /// forced one (AssetDatabase.ReimportFiles), with the same progress and waiting as /api/assets/reimport.
    /// </summary>
    [CommandRoute("POST", "/api/asset/import-settings")]
    public class SetImportSettingsCommand : EditorCommand, IAsyncEditorCommand
    {
        // Synchronous fallback: saves, starts the reimport and answers right away
        public override CommandResult Execute(CommandContext context) => Start(context, out _).result;

        public Task<CommandResult> ExecuteAsync(CommandContext context)
        {
            object saved = null;
            return ReimportResults.StartAndWaitAsync(() => Start(context, out saved),
                job => new { status = "saved", saved, reimport = ReimportResults.Finished(job) });
        }

        private static (CommandResult result, ReimportJob job, bool wait) Start(CommandContext context, out object saved)
        {
            saved = null;
            if (!Program.IsProjectOpen)
                return (CommandResult.Error(409, "No project is open"), null, false);

            using var doc = context.ParseBody();
            var root = doc.RootElement;
            string path = root.TryGetProperty("path", out var p) ? p.GetString()?.Trim().TrimStart('/', '\\') : null;
            if (string.IsNullOrWhiteSpace(path))
                return (CommandResult.BadRequest("'path' is required: a file or folder relative to Assets, or a GUID"), null, false);
            if (!root.TryGetProperty("values", out var values) || values.ValueKind != JsonValueKind.Object
                || !values.EnumerateObject().Any())
                return (CommandResult.BadRequest("'values' is required: an object of setting name → value"), null, false);

            var extensions = ReimportResults.ParseExtensions(root);
            bool reimport = ReimportResults.ParseBool(root, "reimport", true);
            bool wait = ReimportResults.ParseBool(root, "wait", true);

            if (reimport && AssetDatabase.ActiveReimport is { } running)
                return (ReimportResults.Busy(running), null, false);

            // One file (path or GUID), or every tracked file under a folder
            List<string> guids;
            var fileGuid = ImportSettings.ResolveSourceGuid(path);
            bool single = fileGuid != null;
            if (single)
                guids = new List<string> { fileGuid };
            else if (Directory.Exists(Path.Combine(Engine.Project.AssetsDirectory, path)))
                guids = AssetDatabase.GetPathsInFolder(path, extensions).Select(AssetDatabase.PathToGuid).Where(g => g != null).ToList();
            else
                return (CommandResult.NotFound($"No tracked file, GUID or folder '{path}'"), null, false);

            // Apply in memory first: nothing is saved if a value fails to convert
            var changed = new List<(string guid, IImporter importer)>();
            var skipped = new Dictionary<string, int>();
            object applied = null;
            foreach (var guid in guids)
            {
                var importer = AssetDatabase.GetImporter(guid);
                var names = importer != null ? AssetDatabase.GetImporterSettingNames(importer) : new List<string>();
                var missing = values.EnumerateObject().Select(v => v.Name).Where(n => !names.Contains(n)).ToList();
                if (importer == null || missing.Count > 0)
                {
                    if (single)
                        return (CommandResult.BadRequest(importer == null
                            ? $"'{AssetDatabase.GuidToPath(guid)}' has no importer"
                            : $"Not a setting of {importer.GetType().Name}: {string.Join(", ", missing)}. Its settings: {string.Join(", ", names)}"),
                            null, false);
                    var key = importer?.GetType().Name ?? "no importer";
                    skipped[key] = skipped.GetValueOrDefault(key) + 1;
                    continue;
                }

                try
                {
                    applied = ImportSettings.Apply(importer, values);
                }
                catch (Exception ex)
                {
                    return (CommandResult.BadRequest($"{AssetDatabase.GuidToPath(guid)}: {ex.Message} (nothing saved)"), null, false);
                }
                changed.Add((guid, importer));
            }

            if (changed.Count == 0)
                return (CommandResult.BadRequest($"No file in '{path}' has the settings {string.Join(", ", values.EnumerateObject().Select(v => v.Name))}"
                    + (skipped.Count > 0 ? $" (importers there: {string.Join(", ", skipped.Select(s => $"{s.Key} ×{s.Value}"))})" : "")), null, false);

            foreach (var (guid, importer) in changed)
                AssetDatabase.SaveImporter(guid, importer);

            saved = new
            {
                changed = changed.Count,
                skipped = skipped.Count > 0 ? skipped : null,
                applied,
            };

            if (!reimport)
                return (CommandResult.Json(new { status = "saved", saved, note = "Not reimported: asset_reimport applies the settings" }), null, false);

            var label = single ? AssetDatabase.GuidToPath(changed[0].guid) : path.Replace('/', '\\').Trim('\\');
            var job = AssetDatabase.ReimportFiles(changed.Select(c => AssetDatabase.GuidToPath(c.guid)), label);
            return (CommandResult.Json(new { status = "saved", saved, reimport = ReimportResults.Started(job) }), job, wait);
        }
    }

    /// <summary>Reading and writing importer settings by member name (any IImporter, through reflection).</summary>
    internal static class ImportSettings
    {
        /// <summary>Source GUID for a path relative to Assets, a source GUID or a sub-asset GUID; null if none is tracked.</summary>
        public static string ResolveSourceGuid(string pathOrGuid) =>
            AssetDatabase.PathToGuid(pathOrGuid) ?? AssetDatabase.GetMeta(pathOrGuid)?.Guid;

        /// <summary>Every setting the meta stores, name → value, in the importer's order.</summary>
        public static Dictionary<string, object> Read(IImporter importer)
        {
            var settings = new Dictionary<string, object>();
            foreach (var name in AssetDatabase.GetImporterSettingNames(importer))
            {
                var field = Freefall.Reflection.Reflector.GetField(importer.GetType(), name);
                try { settings[name] = CommandHelpers.SerializeValue(field?.GetValue(importer)); }
                catch { settings[name] = "<error>"; }
            }
            return settings;
        }

        /// <summary>
        /// Set the given members on the importer and return them as read back. A TextureImporter is marked
        /// UserConfigured, as the texture inspector's Apply does: otherwise its import guesses sRGB and format
        /// from the file name again and overwrites what was set.
        /// </summary>
        public static Dictionary<string, object> Apply(IImporter importer, JsonElement values)
        {
            var type = importer.GetType();
            var applied = new Dictionary<string, object>();
            foreach (var kv in values.EnumerateObject())
            {
                var field = Freefall.Reflection.Reflector.GetField(type, kv.Name)
                    ?? throw new InvalidOperationException($"'{kv.Name}' not found on {type.Name}");
                field.SetValue(importer, SetPropertyCommand.ConvertJsonValue(kv.Value, field.Type));
                applied[kv.Name] = CommandHelpers.SerializeValue(field.GetValue(importer));
            }

            if (importer is Freefall.Assets.Importers.TextureImporter texture && !applied.ContainsKey(nameof(texture.UserConfigured)))
                texture.UserConfigured = true;

            return applied;
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

        /// <summary>The Asset class of this name (case-insensitive), or null.</summary>
        internal static Type FindAssetType(string name)
            => !string.IsNullOrEmpty(name) && AssetTypes.TryGetValue(name, out var type) ? type : null;

        /// <summary>Every Asset class a type name may be.</summary>
        internal static IEnumerable<string> AssetTypeNames => AssetTypes.Keys;

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
    /// Without 'path' the asset is written to its own source file (resolved by GUID through the AssetDatabase).
    /// SaveAsset imports a tracked file right after writing it, so the cache holds the saved state too; a
    /// 'path' the database does not track is only written.
    /// </summary>
    [CommandRoute("POST", "/api/asset/rename")]
    public class RenameAssetCommand : EditorCommand
    {
        /// <summary>
        /// POST /api/asset/rename  Body: {"guid":"...", "name":"Hunter's Roast"}
        /// Renames the asset's source file in its folder (the extension stays) and the loaded asset with it
        /// (AssetDatabase.RenameAsset, as the asset browser's rename does). The GUID stays, so every
        /// reference to it holds. An asset's name is its file's name: this is how a name is changed.
        /// </summary>
        public override CommandResult Execute(CommandContext context)
        {
            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required: {\"guid\", \"name\"}");

            using var doc = context.ParseBody();
            var root = doc.RootElement;
            string guid = root.TryGetProperty("guid", out var g) ? g.GetString() : null;
            string name = root.TryGetProperty("name", out var n) ? n.GetString()?.Trim() : null;
            if (string.IsNullOrEmpty(guid) || string.IsNullOrEmpty(name))
                return CommandResult.BadRequest("'guid' and 'name' are required");
            if (name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
                return CommandResult.BadRequest($"'{name}' cannot be a file name");

            string before = AssetDatabase.GuidToPath(guid);
            if (before == null)
                return CommandResult.NotFound($"No asset file with GUID {guid}");
            if (!AssetDatabase.RenameAsset(guid, name))
                return CommandResult.BadRequest($"Could not rename {before} to '{name}' (a file of that name there already? see console_log)");

            return CommandResult.Json(new { status = "renamed", guid, from = before, to = AssetDatabase.GuidToPath(guid) });
        }
    }

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

            // The asset's own file is the one the database tracks the GUID under. Not asset.AssetPath: a load by
            // GUID leaves it empty, or holding the path the .asset was last saved with (stale for a copied or
            // renamed file). And never a path made up from the name: that wrote a second file with the same
            // GUID into the Assets root.
            string savePath = root.TryGetProperty("path", out var pp) ? pp.GetString() : null;
            if (string.IsNullOrEmpty(savePath))
                savePath = AssetDatabase.GuidToFullPath(guid);
            if (string.IsNullOrEmpty(savePath))
                return CommandResult.BadRequest(
                    $"'{asset.Name}' ({asset.GetType().Name}) has no source file of its own (a sub-asset of an imported file, or a built-in asset): pass 'path' to write it somewhere");

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
    /// Open an asset in its custom editor tab, like a double-click in the asset browser.
    /// POST /api/asset/open
    /// Body: {"guid":"..."}
    /// </summary>
    [CommandRoute("POST", "/api/asset/open")]
    public class OpenAssetCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'guid'");

            using var doc = context.ParseBody();
            string guid = doc.RootElement.TryGetProperty("guid", out var gp) ? gp.GetString() : null;
            if (string.IsNullOrEmpty(guid))
                return CommandResult.BadRequest("'guid' is required");

            var assetType = AssetDatabase.GetAssetType(guid);
            if (assetType == null)
                return CommandResult.NotFound($"Asset not found: {guid}");
            // A scene opens in the scene view, like a double-click on it
            if (CustomAssetEditorRegistry.Find(assetType) == null && !typeof(SceneAsset).IsAssignableFrom(assetType))
                return CommandResult.BadRequest($"No custom editor for asset type '{assetType.Name}'");

            Freefall.Base.MessageDispatcher.Send(Msg.OpenAssetInEditor, (guid, assetType));
            return CommandResult.Json(new { status = "opened", guid, type = assetType.Name });
        }
    }

    /// <summary>
    /// Set one or more fields on a loaded asset by reflection, then save + reimport it
    /// so the Library cache matches the source file.
    /// POST /api/asset/setproperty
    /// Body: {"guid":"...", "property":"CloudCoverage", "value":0.8}
    ///   or: {"guid":"...", "values": {"CloudCoverage":0.8, "FogColor":[0.5,0.6,0.7]}}
    /// Optional: "save": false to only change the in-memory asset.
    /// Value encoding is the same as entity setproperty (floats, vectors, enums, asset GUIDs; Color3 as [r,g,b]).
    /// An asset reference may also be an asset's path relative to Assets with a leading '@'
    /// ("@Characters/Stats/Strength.asset"; '@@' for a string that really starts with '@'), and an element
    /// of a list of an abstract type names its class with "$type" ({"$type":"StatBonus", ...}).
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

            try
            {
                if (root.TryGetProperty("values", out var values) && values.ValueKind == JsonValueKind.Object)
                {
                    foreach (var kv in values.EnumerateObject())
                        applied[kv.Name] = ApplyValue(asset, kv.Name, kv.Value);
                }
                else
                {
                    if (!root.TryGetProperty("property", out var pp) || !root.TryGetProperty("value", out var vp))
                        return CommandResult.BadRequest("Body must contain 'property' and 'value', or a 'values' object");
                    applied[pp.GetString()] = ApplyValue(asset, pp.GetString(), vp);
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
                        // As an inspector edit does: dependents refresh on the dirty message (a terrain re-reads a
                        // layer's tint and tiling), which saving alone never sends
                        asset.MarkDirty();
                        savedPath = Path.Combine(Engine.Project.AssetsDirectory, relativePath);
                        Engine.Assets.SaveAsset(asset, savedPath);
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

        /// <summary>
        /// Sets one member of an asset from JSON (the encodings above, '@path' references included) and returns
        /// the value as it now reads. A material's shader properties count as members.
        /// </summary>
        internal static object ApplyValue(Asset asset, string propertyName, JsonElement valueEl)
        {
            var assetType = asset.GetType();
            valueEl = AssetPathReferences.Resolve(valueEl);
            var field = Freefall.Reflection.Reflector.GetField(assetType, propertyName);

            // A material's shader properties (Tint, EmissiveColor, EmissiveIntensity, DetailTiling…) are not C#
            // members: set their Value as the material inspector does
            if (field == null && asset is Freefall.Graphics.Material material)
            {
                foreach (var materialProperty in material.MaterialProperties)
                {
                    if (materialProperty.Name != propertyName) continue;
                    var valueField = new Freefall.Reflection.Field(materialProperty.GetType().GetProperty("Value")!);
                    valueField.SetValue(materialProperty, SetPropertyCommand.ConvertJsonValue(valueEl, valueField.Type));
                    return CommandHelpers.SerializeValue(valueField.GetValue(materialProperty));
                }
            }

            if (field == null)
                throw new InvalidOperationException($"Property '{propertyName}' not found on '{assetType.Name}'");
            object converted = SetPropertyCommand.ConvertJsonValue(valueEl, field.Type);
            field.SetValue(asset, converted);
            return CommandHelpers.SerializeValue(field.GetValue(asset));
        }
    }

    /// <summary>
    /// Asset references by path in JSON values: a string that starts with '@' is the path of an asset relative
    /// to Assets ("@Characters/Stats/Strength.asset"; the extension may be left off for a .asset) and becomes its
    /// GUID. "@@..." stands for a string that really starts with '@'.
    /// </summary>
    internal static class AssetPathReferences
    {
        public static JsonElement Resolve(JsonElement element)
        {
            if (!Mentions(element)) return element;
            var node = System.Text.Json.Nodes.JsonNode.Parse(element.GetRawText());
            node = Replace(node);
            using var doc = JsonDocument.Parse(node?.ToJsonString() ?? "null");
            return doc.RootElement.Clone();
        }

        /// <summary>The GUID of the asset at a path relative to Assets, or null.</summary>
        public static string GuidOf(string path)
        {
            if (string.IsNullOrEmpty(path)) return null;
            path = path.Replace('/', '\\').TrimStart('\\');
            return AssetDatabase.PathToGuid(path)
                ?? (string.IsNullOrEmpty(Path.GetExtension(path)) ? AssetDatabase.PathToGuid(path + ".asset") : null);
        }

        static bool Mentions(JsonElement element) => element.ValueKind switch
        {
            JsonValueKind.String => element.GetString()?.StartsWith('@') == true,
            JsonValueKind.Array => element.EnumerateArray().Any(Mentions),
            JsonValueKind.Object => element.EnumerateObject().Any(p => Mentions(p.Value)),
            _ => false,
        };

        static System.Text.Json.Nodes.JsonNode Replace(System.Text.Json.Nodes.JsonNode node)
        {
            switch (node)
            {
                case System.Text.Json.Nodes.JsonArray array:
                    for (int i = 0; i < array.Count; i++)
                    {
                        var replaced = Replace(array[i]);
                        if (!ReferenceEquals(replaced, array[i])) array[i] = replaced;
                    }
                    return array;

                case System.Text.Json.Nodes.JsonObject obj:
                    foreach (var key in obj.Select(p => p.Key).ToList())
                    {
                        var replaced = Replace(obj[key]);
                        if (!ReferenceEquals(replaced, obj[key])) obj[key] = replaced;
                    }
                    return obj;

                case System.Text.Json.Nodes.JsonValue value when value.TryGetValue(out string text) && text.StartsWith('@'):
                    if (text.StartsWith("@@", StringComparison.Ordinal))
                        return System.Text.Json.Nodes.JsonValue.Create(text[1..]);
                    return System.Text.Json.Nodes.JsonValue.Create(GuidOf(text[1..])
                        ?? throw new InvalidOperationException($"No asset at '{text[1..]}' (paths are relative to Assets)"));

                default:
                    return node;
            }
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
