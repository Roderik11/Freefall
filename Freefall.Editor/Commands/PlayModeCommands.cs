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
            return CommandResult.Json(new { playing = editor.IsPlaying, menuOpen = GameUI.MenuOpen });
        }
    }

    /// <summary>
    /// Body {"play": true|false, "menu": true|false}, both optional. play = the toolbar's Play / Stop
    /// (Play saves the scene first); menu = what Escape toggles in the game (windows, free cursor,
    /// no player control).
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
                    var refused = editor.StartPlay();
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

            return CommandResult.Json(new { playing = editor.IsPlaying, menuOpen = GameUI.MenuOpen });
        }
    }
}
