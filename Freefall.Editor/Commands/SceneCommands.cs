using System;
using System.IO;
using System.Linq;
using Freefall.Assets;
using Freefall.Base;

namespace Freefall.Editor.Commands
{
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
                var entities = Program.EditorUI.OpenScene(scene);
                path = Program.EditorUI.CurrentScenePath;
                MessageDispatcher.Send(Msg.SceneLoaded, new
                {
                    path,
                    entityCount = entities.Count
                });

                return CommandResult.Json(new
                {
                    status = "ok",
                    guid = scene.Guid,
                    path,
                    entityCount = entities.Count,
                    entities = entities.Select(e => new { id = e.Id, uid = e.UID.ToString(), name = e.Name }).ToArray()
                });
            }
            catch (Exception ex)
            {
                return CommandResult.Error(500, $"Failed to load scene: {ex.Message}");
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
