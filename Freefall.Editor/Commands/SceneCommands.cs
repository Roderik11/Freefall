using System;
using System.IO;
using System.Linq;
using Freefall.Assets;
using Freefall.Base;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// The scene load in progress (step i/n, what it is doing, 0..1, seconds), or none; and the last one that finished:
    /// its scene, whether it failed or was superseded, its [SceneLoad] summary, and with ?entities=true the entities read
    /// from its file (id, uid, name). MCP scene_load and play_mode wait on this.
    /// </summary>
    [CommandRoute("GET", "/api/scene/load-status")]
    public class SceneLoadStatusCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var qs = CommandHelpers.ParseQueryString(context.Path);
            bool withEntities = qs.TryGetValue("entities", out var e) && e == "true";
            var loading = SceneManager.Loading;
            var last = SceneManager.LastLoad;
            return CommandResult.Json(new
            {
                inProgress = loading != null,
                loading = loading == null ? null : Describe(loading),
                last = last == null ? null : new
                {
                    scene = last.Name,
                    guid = last.Scene?.Guid,
                    seconds = Math.Round(last.Seconds, 1),
                    failed = last.Error?.Message,
                    superseded = last.Superseded,
                    entityCount = last.Entities.Count,
                    entities = withEntities
                        ? last.Entities.Where(x => EntityManager.GetEntity(x.Id) == x).Select(x => new { id = x.Id, uid = x.UID.ToString(), name = x.Name }).ToArray()
                        : null,
                },
                lastReport = SceneManager.LastLoadReport,
            });
        }

        internal static object Describe(SceneLoad load) => new
        {
            scene = load.Name,
            step = SceneLoad.Describe(load.Current),
            stepIndex = (int)load.Current + 1,
            stepCount = SceneLoad.StepCount,
            activity = load.Activity,
            fraction = Math.Round(load.Progress, 3),
            seconds = Math.Round(load.Seconds, 1),
        };
    }

    [CommandRoute("POST", "/api/scene/load")]
    public class LoadSceneCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen || Program.EditorUI == null)
                return CommandResult.Error(409, "No project is open");
            if (Program.EditorUI.IsPlaying)
                return CommandResult.Error(409, "Stop play mode before loading a scene");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Request body required with 'guid' or 'path'");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            // 'guid', or 'path' (relative to Assets or absolute), which may also hold a scene's GUID
            string? guid = root.TryGetProperty("guid", out var guidProp) ? guidProp.GetString() : null;
            string? path = root.TryGetProperty("path", out var pathProp) ? pathProp.GetString() : null;
            if (string.IsNullOrEmpty(guid) && AssetDatabase.GetMeta(path ?? "") != null)
                guid = path;

            SceneAsset? scene;
            if (!string.IsNullOrEmpty(guid))
            {
                if (AssetDatabase.GetAssetType(guid) is not { } type || !typeof(SceneAsset).IsAssignableFrom(type))
                    return CommandResult.NotFound($"No scene asset with GUID {guid}");
                scene = Engine.Assets.LoadByGuid<SceneAsset>(guid);
            }
            else if (!string.IsNullOrEmpty(path))
            {
                scene = EditorDesktop.FindScene(path);
                if (scene == null)
                    return CommandResult.NotFound($"Scene file not found inside the project's Assets folder: {path}");
            }
            else
                return CommandResult.BadRequest("Body must contain 'guid' or 'path'");

            try
            {
                // Starts the load and returns: it runs in slices behind the progress popup (GET /api/scene/load-status)
                var load = Program.EditorUI.OpenScene(scene);
                path = Program.EditorUI.CurrentScenePath;
                load.Completed += done => MessageDispatcher.Send(Msg.SceneLoaded, new
                {
                    path,
                    entityCount = done.Entities.Count
                });

                return CommandResult.Json(new
                {
                    status = "loading",
                    guid = scene.Guid,
                    path,
                });
            }
            catch (InvalidOperationException ex)
            {
                return CommandResult.Error(409, ex.Message);
            }
            catch (Exception ex)
            {
                return CommandResult.Error(500, $"Failed to load scene: {ex.Message}");
            }
        }
    }

    [CommandRoute("POST", "/api/scene/new")]
    public class NewSceneCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen || Program.EditorUI == null)
                return CommandResult.Error(409, "No project is open");
            if (Program.EditorUI.IsPlaying)
                return CommandResult.Error(409, "Stop play mode before starting a new scene");

            try
            {
                // Starts the load and returns, as /api/scene/load does (GET /api/scene/load-status)
                Program.EditorUI.NewScene();
                return CommandResult.Json(new { status = "loading" });
            }
            catch (InvalidOperationException ex)
            {
                return CommandResult.Error(409, ex.Message);
            }
        }
    }

    [CommandRoute("POST", "/api/scene/save")]
    public class SaveSceneCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen || Program.EditorUI == null)
                return CommandResult.Error(409, "No project is open");
            if (Program.EditorUI.IsPlaying)
                return CommandResult.Error(409, "Stop play mode before saving the scene");

            string? path = null;

            if (!string.IsNullOrEmpty(context.Body))
            {
                using var doc = context.ParseBody();
                if (doc.RootElement.TryGetProperty("path", out var pathProp))
                    path = pathProp.GetString();
            }

            // Default to the scene that is currently open (same target as File > Save Scene)
            if (string.IsNullOrEmpty(path) && Program.EditorUI.CurrentScene == null)
                return CommandResult.BadRequest("No scene is open yet — pass 'path' (relative to Assets) to save as a new file");

            try
            {
                var scene = Program.EditorUI.SaveScene(string.IsNullOrEmpty(path) ? null : path);
                ProjectThumbnails.Capture();

                return CommandResult.Json(new
                {
                    status = "saved",
                    guid = scene.Guid,
                    path = Program.EditorUI.CurrentScenePath,
                    entityCount = EntityManager.Entities.Count
                });
            }
            catch (Exception ex)
            {
                return CommandResult.Error(500, $"Failed to save scene: {ex.Message}");
            }
        }
    }
}
