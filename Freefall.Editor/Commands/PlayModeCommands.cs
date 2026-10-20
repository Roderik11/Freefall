using Freefall.UI;

namespace Freefall.Editor.Commands
{
    [CommandRoute("GET", "/api/playmode")]
    public class GetPlayModeCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var editor = Program.EditorUI;
            if (editor == null) return CommandResult.BadRequest("No project open");
            return CommandResult.Json(new { playing = editor.IsPlaying, menuOpen = GameUI.MenuOpen, uiRunning = GameUI.Running });
        }
    }

    /// <summary>
    /// Body {"play": true|false, "menu": true|false, "save": true|false}, all optional. play = the
    /// toolbar's Play / Stop (Play saves the scene first; with "save": false it keeps it in memory
    /// for Stop instead and leaves the file alone); menu = what Escape toggles in the game
    /// (windows, free cursor, no player control).
    /// </summary>
    [CommandRoute("POST", "/api/playmode")]
    public class SetPlayModeCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var editor = Program.EditorUI;
            if (editor == null) return CommandResult.BadRequest("No project open");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            if (root.TryGetProperty("play", out var play))
            {
                if (play.GetBoolean())
                {
                    // save: false keeps the scene in memory for Stop instead of writing it (tools)
                    bool save = !root.TryGetProperty("save", out var s) || s.GetBoolean();
                    var refused = editor.StartPlay(save);
                    if (refused != null && editor.IsPlaying == false)
                        return CommandResult.BadRequest(refused);
                }
                else
                    editor.StopPlay();
            }

            if (root.TryGetProperty("menu", out var menu))
            {
                if (!editor.IsPlaying) return CommandResult.BadRequest("Not playing: the game's menu exists in play mode only");
                GameUI.MenuOpen = menu.GetBoolean();
            }

            return CommandResult.Json(new { playing = editor.IsPlaying, menuOpen = GameUI.MenuOpen, uiRunning = GameUI.Running });
        }
    }
}
