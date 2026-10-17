using System.IO;
using System.Linq;

namespace Freefall.Editor.Commands
{
    [CommandRoute("POST", "/api/project/open")]
    public class OpenProjectCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (Program.IsProjectOpen)
                return CommandResult.Json(new { status = "already_open", project = Engine.Project?.Name });

            // One open at a time: a second request joins the one in progress (project_open waits for it)
            if (ProjectOpening.Current is { } opening)
                return CommandResult.Json(new { status = "opening", path = opening.Path, step = opening.Step, fraction = opening.Fraction });

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Request body required with 'path'");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            string path = null;

            if (root.TryGetProperty("path", out var pathProp))
                path = pathProp.GetString();

            // Allow opening by recent project index
            if (path == null && root.TryGetProperty("recent", out var recentProp))
            {
                var index = recentProp.GetInt32();
                if (index >= 0 && index < RecentProjects.Entries.Count)
                    path = RecentProjects.Entries[index].Path;
                else
                    return CommandResult.BadRequest($"Recent project index {index} out of range (0..{RecentProjects.Entries.Count - 1})");
            }

            if (string.IsNullOrEmpty(path) || !Directory.Exists(path))
                return CommandResult.BadRequest($"Invalid project path: '{path}'");

            // Starts the open (workers + short main-thread slices) and returns at once; editor_status follows it
            if (Program.Landing == null || !ProjectOpening.Start(path, Program.Landing))
                return CommandResult.Error(409, "The editor cannot open a project now (not on the landing page)");

            return CommandResult.Json(new { status = "opening", path });
        }
    }

    [CommandRoute("GET", "/api/project/recent")]
    public class GetRecentProjectsCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var entries = RecentProjects.Entries.Select((e, i) => new
            {
                index = i,
                name = e.Name,
                path = e.Path,
                lastOpened = e.LastOpened
            }).ToArray();

            return CommandResult.Json(entries);
        }
    }
}
