using System.ComponentModel;
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
                     "Changed meshes, textures, materials and prefabs that are already loaded are hot-reloaded in place, " +
                     "so placed instances update without a scene reload ('reloaded' = count).")]
        public static Task<CallToolResult> Refresh() => McpBridge.Post("/api/assets/refresh");

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
                     "This is the only way to edit assets: editing the file on disk does not reach the running editor. " +
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
        [Description("Write a loaded asset to disk (no reimport). Use after in-memory edits, e.g. terrain changes.")]
        public static Task<CallToolResult> Save(string guid, [Description("Override path; default is the asset's own file")] string? path = null)
            => McpBridge.Post("/api/asset/save", new { guid, path });

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
