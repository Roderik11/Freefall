using System;
using System.Drawing;
using System.Threading.Tasks;
using Freefall.Editor.Mcp;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// GET /api/screenshot?target=editor|viewport&amp;maxSize=0&amp;format=png|jpeg&amp;x=&amp;y=&amp;w=&amp;h=
    /// Defaults to the full-resolution editor window as PNG. Dispatched over HTTP it goes through
    /// ScreenshotScheduler (on-screen cue first, overlay excluded from the image).
    /// </summary>
    [CommandRoute("GET", "/api/screenshot")]
    public class ScreenshotCommand : EditorCommand, IAsyncEditorCommand
    {
        private record Args(CaptureTarget Target, Rectangle? Crop, int MaxSize, bool Png);

        /// <summary>Immediate capture without a cue (synchronous main-thread dispatch).</summary>
        public override CommandResult Execute(CommandContext context)
        {
            try
            {
                var a = Parse(context);
                return ToResult(ScreenCapture.Capture(a.Target, a.Crop, a.MaxSize, a.Png));
            }
            catch (Exception ex)
            {
                return CommandResult.Error(500, $"Screenshot failed: {ex.Message}");
            }
        }

        public async Task<CommandResult> ExecuteAsync(CommandContext context)
        {
            try
            {
                var a = Parse(context);
                return ToResult(await ScreenshotScheduler.RequestAsync(a.Target, a.Crop, a.MaxSize, a.Png));
            }
            catch (Exception ex)
            {
                return CommandResult.Error(500, $"Screenshot failed: {ex.Message}");
            }
        }

        private static Args Parse(CommandContext context)
        {
            var q = CommandHelpers.ParseQueryString(context.Path);
            var target = q.TryGetValue("target", out var t) && t.Equals("viewport", StringComparison.OrdinalIgnoreCase)
                ? CaptureTarget.Viewport : CaptureTarget.Editor;
            int maxSize = q.TryGetValue("maxSize", out var ms) && int.TryParse(ms, out var m) ? m : 0;
            bool png = !(q.TryGetValue("format", out var f) && f.StartsWith("jp", StringComparison.OrdinalIgnoreCase));

            Rectangle? crop = null;
            if (q.TryGetValue("w", out var ws) && q.TryGetValue("h", out var hs))
            {
                q.TryGetValue("x", out var xs);
                q.TryGetValue("y", out var ys);
                crop = new Rectangle(int.TryParse(xs, out var x) ? x : 0, int.TryParse(ys, out var y) ? y : 0,
                    int.Parse(ws), int.Parse(hs));
            }

            return new Args(target, crop, maxSize, png);
        }

        private static CommandResult ToResult(CaptureResult shot) => new()
        {
            StatusCode = 200,
            ContentType = shot.MimeType,
            BinaryBody = shot.Data
        };
    }
}
