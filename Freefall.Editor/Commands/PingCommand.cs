namespace Freefall.Editor.Commands
{
    [CommandRoute("GET", "/api/ping")]
    public class PingCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            return CommandResult.Json(new { status = "ok" });
        }
    }
}
