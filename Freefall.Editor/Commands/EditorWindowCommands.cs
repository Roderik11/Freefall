using System;
using System.Windows.Forms;

namespace Freefall.Editor.Commands
{
    /// <summary>The editor window's state, client size and the scene view's size.</summary>
    [CommandRoute("GET", "/api/editor/window")]
    public class GetEditorWindowCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context) => CommandResult.Json(EditorWindowState.Describe());
    }

    /// <summary>
    /// Body {"state": "normal" | "maximized" | "fullscreen", "width": px, "height": px}, all optional. Fullscreen is
    /// borderless and covers the screen; width/height set the client size (normal state). Applied after the
    /// current frame (a resize mid-frame would recreate the swap chain under the recording command list).
    /// </summary>
    [CommandRoute("POST", "/api/editor/window")]
    public class SetEditorWindowCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            string? state = null;
            int width = 0, height = 0;
            if (!string.IsNullOrEmpty(context.Body))
            {
                using var doc = context.ParseBody();
                var root = doc.RootElement;
                if (root.TryGetProperty("state", out var s)) state = s.GetString()?.ToLowerInvariant();
                if (root.TryGetProperty("width", out var w)) width = w.GetInt32();
                if (root.TryGetProperty("height", out var h)) height = h.GetInt32();
            }

            if (state != null && state is not ("normal" or "maximized" or "fullscreen"))
                return CommandResult.BadRequest("state must be normal, maximized or fullscreen");

            var form = Program.Form;
            form.BeginInvoke(() =>
            {
                if (state == "fullscreen")
                {
                    // Borderless maximized covers the taskbar too; leave maximized first so the border change takes
                    form.WindowState = FormWindowState.Normal;
                    form.FormBorderStyle = FormBorderStyle.None;
                    form.WindowState = FormWindowState.Maximized;
                }
                else if (state != null)
                {
                    form.FormBorderStyle = FormBorderStyle.Sizable;
                    form.WindowState = state == "maximized" ? FormWindowState.Maximized : FormWindowState.Normal;
                }

                if (width > 0 && height > 0 && form.WindowState == FormWindowState.Normal)
                    form.ClientSize = new System.Drawing.Size(width, height);
            });

            return CommandResult.Json(new { status = "applying", state, width, height });
        }
    }

    /// <summary>Body {"maximized": true | false}: the scene view alone over the whole dock area, or back in its place.</summary>
    [CommandRoute("POST", "/api/editor/scene-maximized")]
    public class SceneMaximizedCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var ui = Program.EditorUI;
            if (ui?.DockArea == null || ui.Scene == null)
                return CommandResult.BadRequest("No editor layout (is a project open?)");

            bool maximized = true;
            if (!string.IsNullOrEmpty(context.Body))
            {
                using var doc = context.ParseBody();
                if (doc.RootElement.TryGetProperty("maximized", out var m)) maximized = m.GetBoolean();
            }

            ui.DockArea.Maximize(maximized ? ui.Scene : null);
            return CommandResult.Json(EditorWindowState.Describe());
        }
    }

    internal static class EditorWindowState
    {
        public static object Describe()
        {
            var form = Program.Form;
            var ui = Program.EditorUI;
            var viewport = ui?.SceneViewport;
            string state = form.FormBorderStyle == FormBorderStyle.None && form.WindowState == FormWindowState.Maximized
                ? "fullscreen"
                : form.WindowState.ToString().ToLowerInvariant();

            return new
            {
                state,
                clientWidth = form.ClientSize.Width,
                clientHeight = form.ClientSize.Height,
                sceneMaximized = ui?.DockArea?.Maximized != null && ui.DockArea.Maximized == ui.Scene,
                sceneViewWidth = viewport?.Size.x ?? 0,
                sceneViewHeight = viewport?.Size.y ?? 0,
            };
        }
    }
}
