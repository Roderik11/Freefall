using System.ComponentModel;
using System.Threading.Tasks;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Freefall.Editor.Mcp
{
    [McpServerToolType]
    public static class PublishTools
    {
        [McpServerTool(Name = "publish_scan", Title = "Scan a build profile", Idempotent = true, OpenWorld = false)]
        [Description("What a published game would pack: walks the references from a BuildProfile's scenes (and every asset under an " +
                     "AlwaysPublish folder) all the way down, and writes the build report next to the profile's output folder " +
                     "(every asset with its size and the chain that reached it). Returns the totals by root, type and folder, and the " +
                     "first warnings (unknown GUIDs, assets not imported). Nothing is published.")]
        public static Task<CallToolResult> Scan(
            [Description("The BuildProfile asset's GUID, or its path relative to Assets")] string profile,
            [Description("Also list the assets this editor has loaded that the scan did not reach. With the profile's scenes open " +
                         "(and played), that is what code loads by name or by a GUID written in it; the editor loads a few things " +
                         "for itself too (a selected asset, an open graph)")] bool compareLoaded = false)
            => McpBridge.Post("/api/publish/scan", new { profile, compareLoaded });
    }
}
