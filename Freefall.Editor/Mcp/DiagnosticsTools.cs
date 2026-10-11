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
    }
}
