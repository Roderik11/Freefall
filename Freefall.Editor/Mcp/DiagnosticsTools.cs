using System.ComponentModel;
using System.Threading.Tasks;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Freefall.Editor.Mcp
{
    [McpServerToolType]
    public static class DiagnosticsTools
    {
        [McpServerTool(Name = "scene_fingerprint", Title = "Scene fingerprint", ReadOnly = true, OpenWorld = false)]
        [Description("Per top-level entity: how many static meshes are under it and a hash of which mesh sits where " +
                     "('exact' = every bit of the transforms, 'coarse' = positions to the centimetre), plus totals. " +
                     "Take it twice — two loads of a scene, before and after pcg_execute — and compare entity by entity to find " +
                     "generated content (PCG, script generators) that is not reproducible. Equal 'exact' totals = identical scenes.")]
        public static Task<CallToolResult> SceneFingerprint() => McpBridge.Get("/api/scene/fingerprint");

        [McpServerTool(Name = "scene_ground_check", Title = "Ground check", ReadOnly = true, OpenWorld = false)]
        [Description("How every placed piece in a circle stands on the terrain: entities' meshes and the world cells' records " +
                     "(kit walls, towers, houses) alike. Lists pieces 'over a drop' (part of the footprint hangs over the ground: " +
                     "a tower half over a cliff), 'hovering' (all of it a little above the ground) and 'against a bank' (on the " +
                     "ground, but the ground just outside rises well above its footing: a house beside a raised ramp), grouped by " +
                     "their top entity, and the worst pieces with positions. Run it round every change to ground, walls, roads or " +
                     "buildings, not only where the work was. The ground is the terrain only: pieces on bridges or raised " +
                     "paving show as hovering.")]
        public static Task<CallToolResult> SceneGroundCheck(
            [Description("Centre of the circle (world x, z)")] PointXZ center,
            [Description("Radius in metres")] float radius = 150,
            [Description("How far a piece's bottom may stand above the ground before it counts as hanging (m)")] float tolerance = 0.3f,
            [Description("How far the ground just outside a piece may rise over its footing before it counts as a bank (m)")] float rise = 2.5f,
            [Description("How far outside the footprint the ground is read for banks (m)")] float ring = 3f,
            [Description("Pieces smaller than this on the ground, either way, are skipped (m)")] float minSize = 1.5f,
            [Description("How many of the worst pieces to list")] int limit = 60,
            [Description("Owners (their top entity) or meshes to leave out, by name with * wildcards, e.g. [\"*Cliffs*\", \"*Rocks*\"]: cliff rocks stand against banks on purpose")] string[]? ignore = null)
            => McpBridge.Post("/api/scene/ground-check", new { center, radius, tolerance, rise, ring, minSize, limit, ignore });
    }
}
