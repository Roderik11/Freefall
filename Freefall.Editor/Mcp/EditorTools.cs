using System;
using System.ComponentModel;
using System.Drawing;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Freefall.Base;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Freefall.Editor.Mcp
{
    [McpServerToolType]
    public static class EditorTools
    {
        [McpServerTool(Name = "editor_status", Title = "Editor status", ReadOnly = true, OpenWorld = false)]
        [Description("Whether a project is open, which project and scene, plus fps/entity count. Call this first to check the editor is reachable and ready.")]
        public static async Task<CallToolResult> Status()
        {
            var server = EditorCommandServer.Instance;
            var info = await server.RunOnMainThread(() => new JsonObject
            {
                ["projectOpen"] = Program.IsProjectOpen,
                ["project"] = Engine.Project == null ? null : new JsonObject
                {
                    ["name"] = Engine.Project.Name,
                    ["path"] = Engine.Project.RootDirectory,
                },
                ["scene"] = Program.EditorUI?.CurrentScenePath,
            });

            var stats = await server.DispatchAsync("GET", "/api/engine/stats", null);
            if (stats.StatusCode == 200 && stats.Body != null)
                info["stats"] = JsonNode.Parse(stats.Body);

            return Text(info);
        }

        [McpServerTool(Name = "project_list_recent", Title = "List recent projects", ReadOnly = true, OpenWorld = false)]
        [Description("Recently opened projects with their index, name and path (for project_open).")]
        public static Task<CallToolResult> ListRecentProjects() => McpBridge.Get("/api/project/recent");

        [McpServerTool(Name = "project_open", Title = "Open project", Destructive = false, OpenWorld = false)]
        [Description("Open a project from the landing page and wait until import finishes. No-op if a project is already open (the editor cannot switch projects without a restart).")]
        public static async Task<CallToolResult> OpenProject(
            [Description("Project root directory")] string? path = null,
            [Description("Index into project_list_recent, used when path is omitted")] int? recentIndex = null,
            [Description("Max seconds to wait for the import to finish")] int waitSeconds = 300)
        {
            if (path == null && recentIndex == null)
                return McpBridge.Error("Pass either 'path' or 'recentIndex'.");

            var start = await McpBridge.Post("/api/project/open", new { path, recent = recentIndex });
            if (start.IsError == true || TextOf(start).Contains("already_open"))
                return start;

            var server = EditorCommandServer.Instance;
            var deadline = DateTime.UtcNow.AddSeconds(waitSeconds);
            while (DateTime.UtcNow < deadline)
            {
                if (await server.RunOnMainThread(() => Program.IsProjectOpen))
                    return await Status();
                await Task.Delay(500);
            }
            return McpBridge.Error($"Project is still importing after {waitSeconds}s — call editor_status later.");
        }

        [McpServerTool(Name = "editor_shutdown", Title = "Close the editor", Destructive = true, OpenWorld = false)]
        [Description("Close the editor window (e.g. before rebuilding the locked editor DLL). Unsaved work may prompt or be lost.")]
        public static Task<CallToolResult> Shutdown() => McpBridge.Get("/api/editor/shutdown");

        [McpServerTool(Name = "screenshot", Title = "Screenshot", ReadOnly = true, OpenWorld = false)]
        [Description("Capture the editor as an image. Defaults to the whole editor window (UI panels, inspector, console included) " +
                     "downscaled to 1280px. Use target='viewport' for just the 3D scene view, and 'crop' (in the target's full-resolution " +
                     "pixels, reported in the result text) to zoom in on small details at full resolution. " +
                     "Takes ~1 s: the editor first shows a brief 'hands off the mouse' cue (not in the image) and freezes the camera. " +
                     "Pass cameraPosition/lookAt to move the editor camera first (same as camera_move) — one call per view.")]
        public static async Task<CallToolResult> Screenshot(
            [Description("'editor' = whole window incl. UI, 'viewport' = scene view only")] CaptureTarget target = CaptureTarget.Editor,
            [Description("Sub-rectangle to capture, in the target's full-resolution pixel space")] PixelRect? crop = null,
            [Description("Longest output edge in pixels; 0 = full resolution")] int maxSize = 1280,
            [Description("jpeg (small, default) or png (lossless, for thin lines/text)")] ScreenshotFormat format = ScreenshotFormat.Jpeg,
            [Description("Move the editor camera here first (world)")] Vec3? cameraPosition = null,
            [Description("Aim the camera at this world point first")] Vec3? lookAt = null)
        {
            if (cameraPosition != null || lookAt != null)
            {
                var moved = await McpBridge.Post("/api/camera/move", new { position = cameraPosition, lookAt });
                if (moved.IsError == true) return moved;
            }

            CaptureResult shot;
            try
            {
                Rectangle? rect = crop == null ? null : new Rectangle(crop.X, crop.Y, crop.Width, crop.Height);
                // Plays an on-screen "hands off the mouse" cue first; the cue is hidden before the capture.
                shot = await ScreenshotScheduler.RequestAsync(target, rect, maxSize, format == ScreenshotFormat.Png);
            }
            catch (Exception ex)
            {
                return McpBridge.Error($"Screenshot failed: {ex.Message}");
            }

            var region = shot.Region;
            var note = $"{target}: full size {shot.TargetWidth}x{shot.TargetHeight}; " +
                       $"captured window region x={region.X} y={region.Y} {region.Width}x{region.Height} → image {shot.Width}x{shot.Height}.";

            return new CallToolResult
            {
                Content =
                [
                    ImageContentBlock.FromBytes(shot.Data, shot.MimeType),
                    new TextContentBlock { Text = note },
                ],
            };
        }

        [McpServerTool(Name = "console_log", Title = "Read console log", ReadOnly = true, OpenWorld = false)]
        [Description("Recent editor console lines, oldest first. 'offset' skips that many of the newest lines (for paging back).")]
        public static Task<CallToolResult> ConsoleLog(int count = 50, int offset = 0)
            => McpBridge.Get("/api/console/log", ("count", count), ("offset", offset));

        [McpServerTool(Name = "console_clear", Title = "Clear console", Destructive = true, Idempotent = true, OpenWorld = false)]
        [Description("Clear the editor console log (useful before an action so console_log shows only its output).")]
        public static Task<CallToolResult> ConsoleClear() => McpBridge.Get("/api/console/clear");

        [McpServerTool(Name = "engine_stats", Title = "Engine stats", ReadOnly = true, OpenWorld = false)]
        [Description("Frame stats: fps, frame time, entity count, render counters (batches, draw calls, visible/occluded, grass and mesh instances), and the system tree in update order with each system's last update time in ms.")]
        public static Task<CallToolResult> EngineStats() => McpBridge.Get("/api/debug/stats");

        [McpServerTool(Name = "settings_get", Title = "Get engine settings", ReadOnly = true, OpenWorld = false)]
        [Description("All engine render/debug settings (shadows, fog, bloom, SSDM, radiance cascades, debug visualization, ...). Keys are the PascalCase names settings_set expects.")]
        public static async Task<CallToolResult> SettingsGet()
        {
            var result = await McpBridge.Get("/api/settings");
            if (result.IsError == true) return result;

            // The route lowercases only the first letter (lODScale); restore the real property names.
            var src = JsonNode.Parse(TextOf(result))!.AsObject();
            var dst = new JsonObject();
            foreach (var (key, value) in src)
                dst[char.ToUpperInvariant(key[0]) + key[1..]] = value?.DeepClone();
            return Text(dst);
        }

        [McpServerTool(Name = "settings_set", Title = "Set engine settings", Idempotent = true, OpenWorld = false)]
        [Description("Set engine settings by PascalCase name, e.g. {\"EnableBloom\": true, \"LODScale\": 1.5, \"DebugVisualizationMode\": \"CascadeColors\"}. " +
                     "bool/int/float/enum only; unknown keys are ignored, so check the 'changed' map in the result.")]
        public static Task<CallToolResult> SettingsSet(
            [Description("Setting name → value")] JsonObject values)
            => McpBridge.Post("/api/settings", values);

        // --- helpers ---

        internal static CallToolResult Text(JsonNode node) => new()
        {
            Content = [new TextContentBlock { Text = node.ToJsonString() }],
        };

        internal static string TextOf(CallToolResult result)
            => result.Content.Count > 0 && result.Content[0] is TextContentBlock t ? t.Text : "";
    }
}
