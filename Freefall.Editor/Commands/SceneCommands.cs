using System;
using System.IO;
using System.Linq;
using Freefall.Base;

namespace Freefall.Editor.Commands
{
    [CommandRoute("POST", "/api/scene/load")]
    public class LoadSceneCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Request body required with 'path'");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            if (!root.TryGetProperty("path", out var pathProp))
                return CommandResult.BadRequest("Body must contain 'path'");

            var path = pathProp.GetString();

            // Allow relative paths (resolved against project assets directory)
            if (!Path.IsPathRooted(path) && Engine.Project != null)
                path = Path.Combine(Engine.Project.AssetsDirectory, path);

            if (!File.Exists(path))
                return CommandResult.NotFound($"Scene file not found: {path}");

            try
            {
                // Clear existing scene entities (preserves DontDestroyOnLoad)
                EntityManager.ClearScene();
                var serializer = new Freefall.Serialization.EntitySerializer();
                var entities = serializer.Load(path);
                if (Program.EditorUI != null)
                    Program.EditorUI.CurrentScenePath = path;
                MessageDispatcher.Send(Msg.RefreshExplorer);
                MessageDispatcher.Send(Msg.SceneLoaded, new
                {
                    path,
                    entityCount = entities.Count
                });

                return CommandResult.Json(new
                {
                    status = "ok",
                    path,
                    entityCount = entities.Count,
                    entities = entities.Select(e => new { id = e.Id, name = e.Name }).ToArray()
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
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            string path = null;

            if (!string.IsNullOrEmpty(context.Body))
            {
                using var doc = context.ParseBody();
                if (doc.RootElement.TryGetProperty("path", out var pathProp))
                    path = pathProp.GetString();
            }

            // Default to the scene that is currently open (same target as File > Save Scene)
            if (string.IsNullOrEmpty(path))
                path = Program.EditorUI?.CurrentScenePath;

            if (string.IsNullOrEmpty(path))
                return CommandResult.BadRequest("No scene is open yet — pass 'path' (relative to Assets) to save as a new file");

            // Relative paths resolve against the project assets directory, like scene/load
            if (!Path.IsPathRooted(path) && Engine.Project != null)
                path = Path.Combine(Engine.Project.AssetsDirectory, path);

            try
            {
                var serializer = new Freefall.Serialization.EntitySerializer();
                var entities = EntityManager.Entities.ToArray();
                serializer.Save(path, entities);
                if (Program.EditorUI != null)
                    Program.EditorUI.CurrentScenePath = path;

                return CommandResult.Json(new
                {
                    status = "saved",
                    path,
                    entityCount = entities.Length
                });
            }
            catch (Exception ex)
            {
                return CommandResult.Error(500, $"Failed to save scene: {ex.Message}");
            }
        }
    }
}
