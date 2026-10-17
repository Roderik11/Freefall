using System.ComponentModel;
using System.Linq;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Freefall.Editor.Mcp
{
    [McpServerToolType]
    public static class AssetTools
    {
        [McpServerTool(Name = "asset_search", Title = "Search assets", ReadOnly = true, OpenWorld = false)]
        [Description("Find assets whose file name contains 'query' (case-insensitive). Returns path, guid, extension and importer.")]
        public static Task<CallToolResult> Search(
            string query,
            [Description("Filter by extension without the dot, e.g. 'prefab', 'staticmesh', 'material', 'terrain', 'png'")] string? type = null,
            int limit = 50)
            => McpBridge.Get("/api/assets/search", ("query", query), ("type", type), ("limit", limit));

        [McpServerTool(Name = "asset_list", Title = "List asset folder", ReadOnly = true, OpenWorld = false)]
        [Description("Folders and importable files in one Assets sub-folder (not recursive).")]
        public static Task<CallToolResult> List([Description("Folder relative to Assets; empty = root")] string path = "")
            => McpBridge.Get("/api/assets/list", ("path", path));

        [McpServerTool(Name = "asset_resolve", Title = "Resolve asset name", ReadOnly = true, OpenWorld = false)]
        [Description("Exact asset (or sub-asset) name → guid and path.")]
        public static Task<CallToolResult> Resolve(string name)
            => McpBridge.Get("/api/assets/resolve", ("name", name));

        [McpServerTool(Name = "asset_refresh", Title = "Import new files", Idempotent = true, OpenWorld = false)]
        [Description("Rescan the Assets folder and import files that were added or changed on disk outside the editor " +
                     "(generated textures, meshes, copied packs). Needed before new files can be found or used. " +
                     "Changed meshes, textures, materials, prefabs and .asset files that are already loaded are hot-reloaded in place, " +
                     "so placed instances update without a scene reload ('reloaded' = count). A script component that generated " +
                     "entities from a reloaded .asset keeps its old output until it rebuilds, e.g. when one of its members is set again. " +
                     "Files that did not change are not imported again: asset_reimport forces that.")]
        public static Task<CallToolResult> Refresh() => McpBridge.Post("/api/assets/refresh");

        [McpServerTool(Name = "asset_reimport", Title = "Reimport folder", Idempotent = true, OpenWorld = false)]
        [Description("Reimport every importable file under a folder of Assets (or one file), changed or not: for cache files that predate " +
                     "an importer change (e.g. generated LODs) or new import settings, which asset_refresh skips (it imports only files newer " +
                     "than their last import). Files import on the editor's main thread a few per frame, so the editor keeps running and " +
                     "console_log shows the progress every few seconds. Loaded meshes, textures, materials and prefabs update in place. " +
                     "The call answers after the last file with imported/failed counts; if it times out the reimport goes on — " +
                     "calling asset_reimport again answers with the running job's progress, console_log's last line is the result. One at a time.")]
        public static Task<CallToolResult> Reimport(
            [Description("Folder or file relative to Assets, e.g. 'Medieval Mega Pack' or 'Rocks/Boulder.fbx'; '' = all of Assets")] string path,
            [Description("Only files with these extensions, e.g. ['fbx'] (default: every importable file)")] string[]? extensions = null,
            [Description("false = answer right after starting")] bool wait = true)
            => McpBridge.Post("/api/assets/reimport", new { path, extensions, wait });

        [McpServerTool(Name = "asset_get_import_settings", Title = "Get import settings", ReadOnly = true, OpenWorld = false)]
        [Description("Import settings of a source file (model, texture, audio, effect, typeface, ...): its importer type and every setting " +
                     "the meta stores, with names exact for asset_set_import_settings. Model files have e.g. GenerateLODs, LODRatios, " +
                     "LODMinTriangles, LODPixelError (generated LODs), Scale, SmoothingAngle; textures Format, sRGB, IsNormalMap.")]
        public static Task<CallToolResult> GetImportSettings(
            [Description("Source file relative to Assets, or its GUID or a sub-asset's GUID (e.g. a mesh's)")] string path)
            => McpBridge.Get("/api/asset/import-settings", ("path", path));

        [McpServerTool(Name = "asset_set_import_settings", Title = "Set import settings", Destructive = true, Idempotent = true, OpenWorld = false)]
        [Description("Set import settings on a source file, or on every file under a folder, save them to the meta and (by default) reimport " +
                     "those files; loaded assets update in place. Works for every importer; asset_get_import_settings lists the names. In a " +
                     "folder only files whose importer has every given setting change, the rest are counted as skipped (e.g. textures when " +
                     "setting LODPixelError). Nothing is saved if a value fails to convert. A texture's settings mark it UserConfigured, as " +
                     "the texture inspector's Apply does (otherwise its import guesses sRGB and format from the file name again). The " +
                     "reimport behaves as asset_reimport (progress in console_log, goes on if the call times out). Same value encodings as asset_set_properties.")]
        public static Task<CallToolResult> SetImportSettings(
            [Description("File or folder relative to Assets, or a file's GUID or a sub-asset's GUID")] string path,
            [Description("Setting name → value")] JsonObject values,
            [Description("In a folder: only files with these extensions, e.g. ['fbx']")] string[]? extensions = null,
            [Description("Reimport the changed files (false = only save; asset_reimport applies the settings later)")] bool reimport = true,
            [Description("Wait for the reimport to finish")] bool wait = true)
            => McpBridge.Post("/api/asset/import-settings", new JsonObject
            {
                ["path"] = path,
                ["values"] = values.DeepClone(),
                ["extensions"] = extensions != null ? new JsonArray(extensions.Select(e => (JsonNode?)JsonValue.Create(e)).ToArray()) : null,
                ["reimport"] = reimport,
                ["wait"] = wait,
            }.ToJsonString());

        [McpServerTool(Name = "asset_types", Title = "List asset types", ReadOnly = true, OpenWorld = false)]
        [Description("Every importable file extension and its importer.")]
        public static Task<CallToolResult> Types() => McpBridge.Get("/api/assets/types");

        [McpServerTool(Name = "asset_get", Title = "Get asset", ReadOnly = true, OpenWorld = false)]
        [Description("Read all public fields/properties of an asset (materials, presets, terrain settings, ...). Names are exact, for asset_set_properties.")]
        public static Task<CallToolResult> Get(string guid)
            => McpBridge.Get("/api/asset/get", ("guid", guid));

        [McpServerTool(Name = "asset_create", Title = "Create asset", OpenWorld = false)]
        [Description("Create, save and import a new asset of a C# Asset type (e.g. 'EnvironmentPreset', 'Material', 'Terrain'; an unknown type returns the list of valid names). Never overwrites: a numeric suffix is added if the name exists. Returns the new guid.")]
        public static Task<CallToolResult> Create(
            [Description("Asset class name (case-insensitive)")] string type,
            string? name = null,
            [Description("Path relative to Assets, e.g. 'Environment/Storm.asset' — its file name overrides 'name'; the extension is chosen automatically")] string? path = null)
            => McpBridge.Post("/api/asset/create", new { type, name, path });

        [McpServerTool(Name = "asset_set_properties", Title = "Set asset properties", Destructive = true, Idempotent = true, OpenWorld = false)]
        [Description("Set fields on an asset, then (by default) save it to disk and reimport so the editor's cached copy matches. " +
                     "On a Material this also sets its shader properties by name (Tint, EmissiveColor, EmissiveIntensity, DetailTiling, …; asset_get lists them under MaterialProperties). " +
                     "A file edited or regenerated on disk reaches the running editor with asset_refresh instead. " +
                     "Import settings of a source file (a model's generated LODs, a texture's format) are set with asset_set_import_settings. " +
                     "Same value encodings as entity_set_properties. Not atomic — earlier members stay set if a later one fails.")]
        public static Task<CallToolResult> SetProperties(
            string guid,
            [Description("Member name → value")] JsonObject values,
            [Description("Save + reimport after applying (false = only mark dirty in memory)")] bool save = true)
            => McpBridge.Post("/api/asset/setproperty", new JsonObject
            {
                ["guid"] = guid,
                ["values"] = values.DeepClone(),
                ["save"] = save,
            }.ToJsonString());

        [McpServerTool(Name = "asset_save", Title = "Save asset", Destructive = true, OpenWorld = false)]
        [Description("Write a loaded asset to its file and reimport it, so the editor's cached copy matches. " +
                     "Use after in-memory edits, e.g. terrain changes or asset_set_properties with save=false.")]
        public static Task<CallToolResult> Save(string guid, [Description("Override path; default is the asset's own file")] string? path = null)
            => McpBridge.Post("/api/asset/save", new { guid, path });

        [McpServerTool(Name = "asset_open", Title = "Open asset in its editor", Idempotent = true, OpenWorld = false)]
        [Description("Open an asset in its custom editor tab (node graphs, animation graphs, animation sets), like a double-click " +
                     "in the asset browser. Follow with screenshot target='editor' to look at it.")]
        public static Task<CallToolResult> Open(string guid)
            => McpBridge.Post("/api/asset/open", new { guid });

        [McpServerTool(Name = "import_unity_pack", Title = "Import Unity asset pack", OpenWorld = false)]
        [Description("Convert a Unity asset pack from disk into the project's Assets/<pack> folder. Runs synchronously and can take minutes; " +
                     "if the call times out the import is still running — watch console_log.")]
        public static Task<CallToolResult> ImportUnityPack(
            [Description("Unity project / pack source directory on disk")] string source,
            [Description("Pack name (target folder under Assets)")] string pack)
            => McpBridge.Post("/api/tools/import-unity-pack", new { source, pack });

        // --- Terrain (read-only queries; authoring is done with stamp components) ---

        [McpServerTool(Name = "terrain_info", Title = "Terrain info", ReadOnly = true, OpenWorld = false)]
        [Description("Terrain asset, size, max height, heightmap resolution, origin and world bounds.")]
        public static Task<CallToolResult> TerrainInfo() => McpBridge.Get("/api/terrain/info");

        [McpServerTool(Name = "terrain_height", Title = "Terrain heights", ReadOnly = true, OpenWorld = false)]
        [Description("Terrain surface height (world Y) at one or more ground points.")]
        public static Task<CallToolResult> TerrainHeight(PointXZ[] points)
            => McpBridge.Post("/api/terrain/heights", new { points });

        [McpServerTool(Name = "terrain_sample_area", Title = "Sample terrain area", ReadOnly = true, OpenWorld = false)]
        [Description("Height min/max/average, slope and flatness over a rectangle — for judging whether a spot suits a building, road, etc.")]
        public static Task<CallToolResult> TerrainSampleArea(
            PointXZ center,
            [Description("Rectangle size (x, z) in world units")] PointXZ size,
            [Description("Rectangle rotation about Y, degrees")] float rotation = 0,
            [Description("Samples per world unit (0.1–10)")] float density = 1)
            => McpBridge.Post("/api/terrain/sample", new { center, size, rotation, density });
    }
}
