namespace Freefall.Editor.Commands
{
    [CommandRoute("GET", "/api/editor/shutdown")]
    public class ShutdownCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            // Defer close to after the current frame completes
            // (form.Close() during ProcessCommands mid-RenderGui causes ObjectDisposedException)
            Program.Form.BeginInvoke(() => Program.Form.Close());
            return CommandResult.Json(new { status = "shutting_down" });
        }
    }
}
