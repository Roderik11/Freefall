using System.ComponentModel;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Freefall.Editor.Mcp
{
    [McpServerToolType]
    public static class NavMeshTools
    {
        [McpServerTool(Name = "navmesh_bake", Title = "Bake navmesh", OpenWorld = false)]
        [Description("Bake the navmesh of the scene's NavMeshSurface from the terrain and static meshes. Runs in the background; " +
                     "only tiles whose surroundings changed since the last bake are rebuilt, so rebaking after a local edit is quick. " +
                     "Waits for the bake to finish (up to waitSeconds), then returns the result; on timeout the bake keeps running — poll navmesh_status. " +
                     "lastBake.changedTiles lists where the scene differs from the previous bake ([minX, minZ, maxX, maxZ] per tile, first 256). " +
                     "The baked data lives in the NavMesh asset: asset_save it (guid in the result) to keep it.")]
        public static async Task<CallToolResult> Bake(
            [Description("Rebuild every tile instead of only the changed ones")] bool rebuildAll = false,
            [Description("Max seconds to wait for the bake; 0 returns immediately")] int waitSeconds = 300)
        {
            var result = await McpBridge.Post("/api/navmesh/bake", new { rebuildAll });
            if (result.IsError == true) return result;

            var deadline = System.DateTime.UtcNow.AddSeconds(waitSeconds);
            while (IsBaking(result) && System.DateTime.UtcNow < deadline)
            {
                await Task.Delay(500);
                result = await McpBridge.Get("/api/navmesh/status");
                if (result.IsError == true) return result;
            }
            return result;
        }

        [McpServerTool(Name = "navmesh_status", Title = "Navmesh status", ReadOnly = true, OpenWorld = false)]
        [Description("Navmesh bake progress (stage, tiles done/total), the result of the last bake, and the NavMesh asset's poly count.")]
        public static Task<CallToolResult> Status() => McpBridge.Get("/api/navmesh/status");

        private static bool IsBaking(CallToolResult result)
            => JsonNode.Parse(EditorTools.TextOf(result))?["baking"]?.GetValue<bool>() == true;
    }
}
