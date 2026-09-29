using System.ComponentModel;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Freefall.Editor.Mcp
{
    [Description("Area filter: a circle (center + radius) or a rectangle (min + max) on the ground plane")]
    public record AreaFilter(
        [property: Description("Circle center")] PointXZ? Center = null,
        [property: Description("Circle radius, default 100")] float? Radius = null,
        [property: Description("Rectangle min corner")] PointXZ? Min = null,
        [property: Description("Rectangle max corner")] PointXZ? Max = null);

    /// <summary>Content authoring tools: PCG graphs, materials, bulk scene queries/edits, prefab measurement.</summary>
    [McpServerToolType]
    public static class AuthoringTools
    {
        // --- PCG / node graphs ---

        [McpServerTool(Name = "graph_node_types", Title = "List graph node types", ReadOnly = true, OpenWorld = false)]
        [Description("Every graph node type (PCG samplers, filters, transforms, spawners, ...) with category, input/output port names " +
                     "and editable settings with their types and defaults. Read this before graph_edit.")]
        public static Task<CallToolResult> NodeTypes() => McpBridge.Get("/api/graph/nodetypes");

        [McpServerTool(Name = "graph_get", Title = "Get graph", ReadOnly = true, OpenWorld = false)]
        [Description("A node graph asset (e.g. PCGGraph): nodes with ids, types, settings and ports; connections as " +
                     "{from:{node,port}, to:{node,port}} (producer → consumer); and the entities whose PCGComponent uses it.")]
        public static Task<CallToolResult> GetGraph(string guid) => McpBridge.Get("/api/graph/get", ("guid", guid));

        [McpServerTool(Name = "graph_edit", Title = "Edit graph", Destructive = true, OpenWorld = false)]
        [Description("Apply edit ops to a node graph in order, then save + reimport it and (run=true) re-execute every PCGComponent " +
                     "using it; the result lists each user's spawn count. Ops (JSON objects): " +
                     "{op:'add', type:'DensityNoise', values:{...}, ref:'noise'} — 'ref' names the new node for later ops in the same call; " +
                     "{op:'set', node:5, values:{Scale:40}}; {op:'remove', node:5}; " +
                     "{op:'connect', from:3, fromPort:'Output', to:'noise', toPort:'Input'} (ports default Output→Input); " +
                     "{op:'disconnect', from:3, to:7}. Settings use the same encodings as entity_set_properties " +
                     "(lists of objects for SpawnPrefab.Entities: [{Prefab:'guid', Weight:1}]). Create a new graph with asset_create type 'PCGGraph'.")]
        public static Task<CallToolResult> EditGraph(
            string guid,
            [Description("Edit operations, applied in order")] JsonArray ops,
            [Description("Re-run PCG components that use this graph")] bool run = true)
            => McpBridge.Post("/api/graph/edit", new JsonObject
            {
                ["guid"] = guid,
                ["ops"] = ops.DeepClone(),
                ["run"] = run,
            }.ToJsonString());

        [McpServerTool(Name = "pcg_execute", Title = "Run PCG", OpenWorld = false)]
        [Description("Re-run the PCGComponent on an entity (or all PCG components when id is omitted) and report spawn counts and timing per component. " +
                     "PCG output regenerates on load and whenever its spline or graph changes; it is never saved into the scene.")]
        public static Task<CallToolResult> ExecutePcg([Description("Entity id with a PCGComponent; omit for all")] int? id = null)
            => McpBridge.Post("/api/pcg/execute", new { id });

        // --- Materials ---

        [McpServerTool(Name = "material_set_textures", Title = "Set material textures", Destructive = true, OpenWorld = false)]
        [Description("Assign texture slots on a Material by slot name → texture GUID (e.g. {Albedo:'...', Normal:'...', Metallic:'...'}), optionally its effect, " +
                     "then save + reimport. The result lists the material's slot names. Create the material first with asset_create type 'Material'.")]
        public static Task<CallToolResult> SetMaterialTextures(
            string guid,
            [Description("Slot name → texture GUID")] JsonObject textures,
            [Description("Effect GUID (optional)")] string? effect = null)
            => McpBridge.Post("/api/material/settextures", new JsonObject
            {
                ["guid"] = guid,
                ["textures"] = textures.DeepClone(),
                ["effect"] = effect,
            }.ToJsonString());

        // --- Scene queries & bulk edits ---

        [McpServerTool(Name = "scene_query", Title = "Query entities", ReadOnly = true, OpenWorld = false)]
        [Description("Entities matching all given filters, with world position, rotation, scale and prefab GUID, plus a per-name histogram. " +
                     "Top-level, hand-authored entities only unless topLevelOnly=false / includeGenerated=true (PCG output).")]
        public static Task<CallToolResult> Query(
            [Description("Name pattern with * wildcards")] string? name = null,
            [Description("Component type name")] string? component = null,
            [Description("Prefab GUID or name")] string? prefab = null,
            AreaFilter? area = null,
            bool topLevelOnly = true,
            bool includeGenerated = false,
            [Description("Only return count + histogram")] bool summaryOnly = false,
            int limit = 200)
            => McpBridge.Post("/api/scene/query", new { name, component, prefab, area, topLevelOnly, includeGenerated, summaryOnly, limit });

        [McpServerTool(Name = "entities_delete", Title = "Delete entities (bulk)", Destructive = true, OpenWorld = false)]
        [Description("Delete many entities at once: by 'ids' and/or the same filters as scene_query (name, component, prefab, area). " +
                     "Use dryRun=true first to see the count and per-name histogram. At least one of ids/filters is required.")]
        public static Task<CallToolResult> DeleteMany(
            int[]? ids = null,
            string? name = null,
            string? component = null,
            string? prefab = null,
            AreaFilter? area = null,
            bool topLevelOnly = true,
            bool dryRun = false)
            => McpBridge.Post("/api/scene/delete", new { ids, name, component, prefab, area, topLevelOnly, dryRun });

        [McpServerTool(Name = "entity_remove_component", Title = "Remove component", Destructive = true, OpenWorld = false)]
        [Description("Remove a component (by type name) from an entity. Transform cannot be removed.")]
        public static Task<CallToolResult> RemoveComponent(int id, string type)
            => McpBridge.Post($"/api/entity/{id}/removecomponent", new { type });

        [McpServerTool(Name = "prefab_measure", Title = "Measure prefabs", ReadOnly = true, OpenWorld = false)]
        [Description("Bounds (min/max/size, relative to the prefab pivot at the origin) of one or more prefabs without placing them in the scene. " +
                     "Use before choosing scale ranges for PCG spawners or placing modular pieces; empty=true means the prefab renders nothing.")]
        public static Task<CallToolResult> MeasurePrefabs(string[] guids)
            => McpBridge.Post("/api/prefab/measure", new { guids });
    }
}
