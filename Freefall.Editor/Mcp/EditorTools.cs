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
        [Description("Whether a project is open, which project and scene, plus fps/entity count. Call this first to check the editor is reachable and ready. " +
                     "While a project is opening, 'opening' says which step it is on and how far along it is (projectOpen is false until it is done); " +
                     "while a job with the progress popup runs (Publish, a build profile's Scan, texture caps), 'job' does the same.")]
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
                ["sceneGuid"] = Program.EditorUI?.CurrentScene?.Guid,
                // What Save Changed Assets would write (a terrain after a height stamp edit, a baked navmesh)
                ["changedAssets"] = new JsonArray(AssetCreator.ChangedAssets.Select(a => (JsonNode?)$"{a.GetType().Name} {a.Name}").ToArray()),
            });

            if (ProjectOpening.Current is { } opening)
                info["opening"] = OpeningJson(opening);
            if (ProgressJob.Running?.Current is { } job)
                info["job"] = JobJson(job);

            // A scene load in progress: its step, what it is doing, how far (calls that change the scene wait for it)
            var sceneLoad = await server.DispatchAsync("GET", "/api/scene/load-status", null);
            if (sceneLoad.StatusCode == 200 && sceneLoad.Body != null && JsonNode.Parse(sceneLoad.Body)?["loading"] is JsonObject loading)
                info["sceneLoad"] = loading.DeepClone();

            var stats = await server.DispatchAsync("GET", "/api/engine/stats", null);
            if (stats.StatusCode == 200 && stats.Body != null)
                info["stats"] = JsonNode.Parse(stats.Body);

            return Text(info);
        }

        [McpServerTool(Name = "project_list_recent", Title = "List recent projects", ReadOnly = true, OpenWorld = false)]
        [Description("Recently opened projects with their index, name and path (for project_open).")]
        public static Task<CallToolResult> ListRecentProjects() => McpBridge.Get("/api/project/recent");

        [McpServerTool(Name = "project_open", Title = "Open project", Destructive = false, OpenWorld = false)]
        [Description("Open a project from the landing page and wait until it is fully open (asset scan, import, scripts, editor UI). " +
                     "The editor stays responsive meanwhile: editor_status reports the step. If an open is already running, waits for that one. " +
                     "No-op if a project is already open (the editor cannot switch projects without a restart).")]
        public static async Task<CallToolResult> OpenProject(
            [Description("Project root directory")] string? path = null,
            [Description("Index into project_list_recent, used when path is omitted")] int? recentIndex = null,
            [Description("Max seconds to wait for the open to finish")] int waitSeconds = 300)
        {
            if (path == null && recentIndex == null)
                return McpBridge.Error("Pass either 'path' or 'recentIndex'.");

            var start = await McpBridge.Post("/api/project/open", new { path, recent = recentIndex });
            if (start.IsError == true || TextOf(start).Contains("already_open"))
                return start;

            // Started (or already running): the open completes with null once the editor is up, else with the error
            var completion = ProjectOpening.Completion;
            try
            {
                var error = await completion.WaitAsync(TimeSpan.FromSeconds(waitSeconds));
                if (error != null)
                    return McpBridge.Error($"Opening the project failed: {error}");
            }
            catch (TimeoutException)
            {
                var opening = ProjectOpening.Current;
                return McpBridge.Error($"The project is still opening after {waitSeconds}s" +
                                       (opening != null ? $" (step: {opening.Step}, {opening.Fraction:P0})" : "") +
                                       " — call editor_status later.");
            }
            return await Status();
        }

        internal static JsonObject JobJson(ProgressJob.Status s) => new()
        {
            ["title"] = s.Title,
            ["step"] = s.Step,
            ["stepIndex"] = s.StepIndex + 1,
            ["stepCount"] = s.StepCount,
            ["detail"] = s.Detail,
            ["fraction"] = Math.Round(s.Fraction, 3),
            ["seconds"] = Math.Round(s.Seconds, 1),
        };

        private static JsonObject OpeningJson(ProjectOpening.Status s) => new()
        {
            ["path"] = s.Path,
            ["step"] = s.Step,
            ["stepIndex"] = s.StepIndex + 1,
            ["stepCount"] = s.StepCount,
            ["detail"] = s.Detail,
            ["fraction"] = Math.Round(s.Fraction, 3),
            ["seconds"] = Math.Round(s.Seconds, 1),
        };

        [McpServerTool(Name = "editor_shutdown", Title = "Close the editor", Destructive = true, OpenWorld = false)]
        [Description("Close the editor window (e.g. before rebuilding the locked editor DLL). Unsaved work may prompt or be lost.")]
        public static Task<CallToolResult> Shutdown() => McpBridge.Get("/api/editor/shutdown");

        [McpServerTool(Name = "editor_window", Title = "Editor window size and state", OpenWorld = false)]
        [Description("Set the editor window's state ('normal', 'maximized', or 'fullscreen' = borderless over the whole screen) " +
                     "and/or its client size in pixels (width + height, normal state). No arguments = report only. Returns the " +
                     "window state, client size and the scene view's size, e.g. for measuring GPU cost at a given resolution.")]
        public static async Task<CallToolResult> EditorWindow(string? state = null, int width = 0, int height = 0)
        {
            if (state != null || (width > 0 && height > 0))
            {
                var set = await McpBridge.Post("/api/editor/window", new { state, width, height });
                if (set.IsError == true) return set;
                await Task.Delay(750); // applied after the frame; the swap chain resizes on the next tick
            }
            return await McpBridge.Get("/api/editor/window");
        }

        [McpServerTool(Name = "scene_view_maximize", Title = "Maximize the scene view", OpenWorld = false)]
        [Description("Show the scene view alone over the whole dock area (maximized: true) or put it back into the layout " +
                     "(false). Same as double-clicking its tab. Returns the window and scene view sizes.")]
        public static async Task<CallToolResult> SceneViewMaximize(bool maximized = true)
        {
            var result = await McpBridge.Post("/api/editor/scene-maximized", new { maximized });
            if (result.IsError == true) return result;
            await Task.Delay(300); // the viewport picks up its new size on the next layout
            return await McpBridge.Get("/api/editor/window");
        }

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
        [Description("Frame stats: fps, frame time, entity count, render counters (batches, draw calls, visible/occluded, grass and mesh instances), and the system tree in update order with each system's time in the last frame in ms. For where the frame's CPU time goes, use 'profiler'.")]
        public static Task<CallToolResult> EngineStats() => McpBridge.Get("/api/debug/stats");

        [McpServerTool(Name = "profiler", Title = "CPU profiler", ReadOnly = true, OpenWorld = false)]
        [Description("Where the frame's time goes. 'tree' is main-thread CPU time: engine tick phases, systems, each component type's " +
                     "Update/Draw/wake-up loop (items = components), render passes (command recording), editor UI; 'Wait for GPU' is the " +
                     "CPU waiting for the GPU. 'gpuTree' is GPU time per PIX region (G-buffer passes, shadows, lighting, post, editor UI) " +
                     "on the primary command list, from timestamp queries. Average and max ms over the last 'frames' frames (max 300), " +
                     "plus KB allocated on the main thread; 'slowest' shows the slowest of those frames instead, and 'frame' one frame by " +
                     "index (the result lists every frame over 100 ms in 'longFrames': a scene load's stalls, for instance). Sections under 'minMs' " +
                     "(unless they spike to 10x that) are left out. Also shown live in the editor's Profiler tab.")]
        public static Task<CallToolResult> ProfilerTree(int frames = 120, double minMs = 0.05, bool slowest = false,
            [Description("One recorded frame by index (from longFrames); -1 = none")] long frame = -1)
            => McpBridge.Get("/api/debug/profiler", ("frames", frames), ("minMs", minMs), ("slowest", slowest), ("frame", frame));

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

        [McpServerTool(Name = "play_mode", Title = "Play mode", OpenWorld = false)]
        [Description("Enter or leave play mode, and open or close the game's menu (what Escape toggles: the game's windows, a free " +
                     "cursor, no player control). No arguments = report only. By default this does NOT save the scene: it is kept in " +
                     "memory and Stop goes back to it, so the scene file is untouched (save=true writes it first, as the toolbar's Play " +
                     "does). Play and Stop are scene loads behind the editor's progress popup; this waits until the load is done " +
                     "(Play on Eldermoor ~12 s, most of it physics cooking; Stop ~5 s), unless wait=false. " +
                     "The game's UI is drawn over the scene viewport: screenshot target='viewport' shows it. Drive the game with " +
                     "play_input; read the game UI's controls with game_ui.")]
        public static async Task<CallToolResult> PlayMode(
            [Description("true = Play, false = Stop")] bool? play = null,
            [Description("true = game menu open, false = closed (play mode only)")] bool? menu = null,
            [Description("Play only: write the scene to its file first (the toolbar's way). Default false: kept in memory")] bool save = false,
            [Description("Wait until the game runs (Play) or the scene is back (Stop), and has settled")] bool wait = true,
            [Description("Max seconds to wait")] int waitSeconds = 120)
        {
            if (play == null && menu == null)
                return await McpBridge.Get("/api/playmode");

            var body = new JsonObject();
            if (play != null) body["play"] = play.Value;
            if (play == true) body["save"] = save;
            if (menu != null && play != true) body["menu"] = menu.Value;
            var result = await McpBridge.Post("/api/playmode", body);
            if (play == null || !wait || result.IsError == true) return result;

            // Play and Stop are scene loads (SceneLoad, in slices behind the editor's popup): done when the load is
            var done = await SceneTools.WaitForLoad(waitSeconds);
            if (done == null)
                return McpBridge.Error(play == true
                    ? $"Play started, but the scene was still loading after {waitSeconds}s; check play_mode later."
                    : $"Stop asked, but the scene was still loading after {waitSeconds}s; check play_mode later.");
            if (done["last"]?["failed"] is { } failed)
                return McpBridge.Error($"The scene load failed: {failed}");

            if (play == true && menu != null)
                return await McpBridge.Post("/api/playmode", new JsonObject { ["menu"] = menu.Value });
            return await McpBridge.Get("/api/playmode");
        }

        [McpServerTool(Name = "play_input", Title = "Play input into the game", OpenWorld = false)]
        [Description("Play input into the running game, as the player's hands would: keys, mouse buttons, mouse look, the game UI's " +
                     "cursor (a separate one; the user's real mouse is not moved), waits. steps is a list, each one of: " +
                     "{\"key\":\"W\",\"hold\":1.5} (tap without hold), {\"down\":\"W\"} / {\"up\":\"W\"}, {\"mouseDown\":\"left\"} / {\"mouseUp\":\"left\"}, " +
                     "{\"click\":\"left\",\"at\":[x,y]}, {\"drag\":{\"from\":[x,y],\"to\":[x,y],\"over\":0.3}}, {\"cursor\":[x,y]} (null = real mouse again), " +
                     "{\"move\":[x,y],\"over\":0.3}, {\"look\":[dx,dy],\"over\":0.5} (camera; ~10 counts a degree), {\"scroll\":120}, {\"wait\":0.5}. " +
                     "Keys by Freefall.Keys name (W A S D, Space, Escape, ShiftKey, E, F, Q, 1-9). Coordinates are pixels of the game's " +
                     "picture at the viewport's full size (screenshot target='viewport' reports it; game_ui gives control rects in the same " +
                     "pixels); the tool's cursor shows as a small cross in screenshots. Escape opens the game menu (or play_mode menu=true). " +
                     "A new call stops steps still playing (and lets go of what they held); after steps that finished, the cursor " +
                     "and held keys/buttons stay, so a drag can be held across calls (mouseDown, move, screenshot, then mouseUp). " +
                     "play_input_stop lets go of everything. Waits until the steps are done unless wait=false.")]
        public static async Task<CallToolResult> PlayInput(
            [Description("The steps, in order")] JsonArray steps,
            [Description("Wait until the steps are done")] bool wait = true)
        {
            var result = await McpBridge.Post("/api/play/input", new JsonObject { ["steps"] = steps.DeepClone() }.ToJsonString());
            if (!wait || result.IsError == true) return result;

            float seconds = 0f;
            try { seconds = JsonNode.Parse(TextOf(result))?["seconds"]?.GetValue<float>() ?? 0f; } catch { }
            var deadline = DateTime.UtcNow.AddSeconds(seconds + 30);
            await Task.Delay(TimeSpan.FromSeconds(Math.Min(seconds, 60)));
            while (DateTime.UtcNow < deadline)
            {
                var status = await McpBridge.Get("/api/play/input");
                if (status.IsError != true && JsonNode.Parse(TextOf(status))?["playing"]?.GetValue<bool>() == false)
                    return status;
                await Task.Delay(100);
            }
            return await McpBridge.Get("/api/play/input");
        }

        [McpServerTool(Name = "play_input_stop", Title = "Stop played input", OpenWorld = false)]
        [Description("Stop the steps play_input is playing, let go of every key and button they hold, and give the game UI the real mouse back.")]
        public static Task<CallToolResult> PlayInputStop() => McpBridge.Post("/api/play/input/stop", new JsonObject());

        [McpServerTool(Name = "game_ui", Title = "Game UI controls", ReadOnly = true, OpenWorld = false)]
        [Description("The game UI's controls that show (play mode): type, name, style, text, tag (what a control stands for: an item, " +
                     "a slot), rect [x,y,w,h] in pixels of the game's picture (what play_input takes), hot/selected, and path. " +
                     "filter = only those whose name, text, tag, style or type contains it (case-insensitive). all = hidden ones too.")]
        public static Task<CallToolResult> GameUITree(
            [Description("Substring to look for")] string? filter = null,
            [Description("Include hidden controls")] bool all = false)
            => McpBridge.Get("/api/play/ui", ("filter", filter), ("all", all ? "true" : null));

        // --- helpers ---

        internal static CallToolResult Text(JsonNode node) => new()
        {
            Content = [new TextContentBlock { Text = node.ToJsonString() }],
        };

        internal static string TextOf(CallToolResult result)
            => result.Content.Count > 0 && result.Content[0] is TextContentBlock t ? t.Text : "";
    }
}
