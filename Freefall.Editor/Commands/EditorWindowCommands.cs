using System;
using System.Drawing;
using System.Runtime.InteropServices;
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
                    form.MaximumSize = Size.Empty;
                    form.WindowState = FormWindowState.Normal;
                    form.FormBorderStyle = FormBorderStyle.None;
                    form.WindowState = FormWindowState.Maximized;
                }
                else if (state != null)
                {
                    form.MaximumSize = Size.Empty;
                    form.FormBorderStyle = FormBorderStyle.Sizable;
                    form.WindowState = state == "maximized" ? FormWindowState.Maximized : FormWindowState.Normal;
                }

                if (width > 0 && height > 0 && form.WindowState == FormWindowState.Normal)
                    SetClientSize(form, width, height);
            });

            return CommandResult.Json(new { status = "applying", state, width, height });
        }

        [DllImport("user32.dll")]
        private static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int x, int y, int cx, int cy, uint flags);
        private const uint SWP_NOMOVE = 0x0002, SWP_NOZORDER = 0x0004, SWP_NOACTIVATE = 0x0010;

        /// <summary>
        /// Windows keeps a window within its maximum track size (about the screen plus the frame), and WinForms'
        /// ClientSize clamps to it too. A larger one (a scene view the screen's full height under the toolbar, for a
        /// capture) lifts MaximumSize, which WinForms answers WM_GETMINMAXINFO with, and is set by SetWindowPos.
        /// </summary>
        private static void SetClientSize(Form form, int width, int height)
        {
            var frame = form.Size - form.ClientSize;
            var size = new Size(width + frame.Width, height + frame.Height);
            var max = SystemInformation.MaxWindowTrackSize;
            if (size.Width <= max.Width && size.Height <= max.Height)
            {
                form.MaximumSize = Size.Empty;
                form.ClientSize = new Size(width, height);
                return;
            }

            form.MaximumSize = new Size(Math.Max(size.Width, max.Width), Math.Max(size.Height, max.Height));
            SetWindowPos(form.Handle, IntPtr.Zero, 0, 0, size.Width, size.Height, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
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
