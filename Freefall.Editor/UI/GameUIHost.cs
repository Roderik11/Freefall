using Squid;
using Freefall.UI;

namespace Freefall.Editor
{
    /// <summary>
    /// Plays host to the game's UI (<see cref="GameUI"/>) in play mode: a second Squid desktop laid
    /// over the scene viewport. The editor's desktop updates first; then the game's gets the mouse
    /// if the editor left it to the picture (nothing of the editor's, a menu or a tooltip, is on top
    /// of it there), and the viewport draws it over the picture inside the editor's own draw, so
    /// editor popups stay in front.
    /// </summary>
    internal static class GameUIHost
    {
        /// <summary>The viewport showing the game's picture this frame (the main camera's target).</summary>
        public static InnerViewport? Viewport;

        /// <summary>The mouse is on a control of the game's UI, not on the bare picture.</summary>
        public static bool MouseOverGameUI
        {
            get
            {
                if (!GameUI.Running) return false;
                var hot = GameUI.Desktop.HotControl;
                return hot != null && hot != GameUI.Desktop;
            }
        }

        /// <summary>After the editor's desktop has updated.</summary>
        public static void Update(Desktop editor)
        {
            var view = Viewport;
            if (Engine.IsEditor || view == null || view.Desktop != editor)
            {
                if (GameUI.Running) GameUI.Reset();
                editor.ShowCursor = true;
                return;
            }

            GameUI.Update();

            var desktop = GameUI.Desktop;
            var at = view.Location;
            var size = view.Size;
            desktop.Position = at;
            desktop.Size = size;
            Input.LockArea = new Vortice.Mathematics.RectI(at.x, at.y, size.x, size.y);

            // A locked cursor is the game's: nothing of either desktop follows it
            editor.ShowCursor = !Input.IsMouseLocked;

            bool mouseHere = !Input.IsMouseLocked && editor.HotControl == view;
            if (mouseHere)
            {
                desktop.Update();
                return;
            }

            // Elsewhere for the game: nothing of its gets hovered or clicked. The editor's frame is
            // done with the mouse by now, and the next one sets it anew.
            var mouse = Gui.MousePosition;
            int scroll = Gui.MouseScroll;
            Gui.SetMouse(-100000, -100000, 0);
            desktop.Update();
            Gui.SetMouse(mouse.x, mouse.y, scroll);
        }

        /// <summary>Inside the viewport's draw, over the picture.</summary>
        public static void Draw(InnerViewport view)
        {
            if (!GameUI.Running || view != Viewport) return;
            GameUI.Desktop.DrawNested();
        }
    }
}
