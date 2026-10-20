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

            // A tool holds the game's cursor (MCP play_input): the game's desktop gets that cursor and
            // the tool's buttons, with button states of its own, and the editor's are put back after
            if (Input.SimulatedCursor is { } simulated)
            {
                UpdateSimulated(desktop, at, simulated);
                return;
            }
            if (_simulatedLast.x != int.MinValue)
            {
                _simulatedLast = new Point(int.MinValue, int.MinValue);
                System.Array.Clear(_simulatedButtons);
            }

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

        // The game desktop's button states while a tool drives it, and where its cursor was last frame
        // (Squid takes the movement from the last position it was given, which is the editor's)
        private static readonly ButtonState[] _simulatedButtons = new ButtonState[5];
        private static readonly ButtonState[] _editorButtons = new ButtonState[5];
        private static Point _simulatedLast = new(int.MinValue, int.MinValue);

        private static void UpdateSimulated(Desktop desktop, Point at, System.Numerics.Vector2 cursor)
        {
            var mouse = Gui.MousePosition;
            int scroll = Gui.MouseScroll;
            Gui.GetButtonStates(_editorButtons);

            var here = new Point(at.x + (int)MathF.Round(cursor.X), at.y + (int)MathF.Round(cursor.Y));
            if (_simulatedLast.x == int.MinValue) _simulatedLast = here;
            Gui.SetMouse(_simulatedLast.x, _simulatedLast.y, 0);
            Gui.SetMouse(here.x, here.y, Input.SimulatedWheel);
            _simulatedLast = here;

            Gui.SetButtonStates(_simulatedButtons);
            Gui.SetButtons(Input.IsSimulatedButtonDown(0), Input.IsSimulatedButtonDown(1), Input.IsSimulatedButtonDown(2));
            desktop.Update();
            Gui.GetButtonStates(_simulatedButtons);

            Gui.SetButtonStates(_editorButtons);
            Gui.SetMouse(mouse.x, mouse.y, scroll);
        }

        /// <summary>Where a tool's cursor is over the game's UI, in its desktop's coordinates, or null (for the cursor mark).</summary>
        public static Point? SimulatedCursorOnDesktop
            => Input.SimulatedCursor is { } c && GameUI.Running
                ? new Point(GameUI.Desktop.Position.x + (int)MathF.Round(c.X), GameUI.Desktop.Position.y + (int)MathF.Round(c.Y))
                : null;

        /// <summary>Inside the viewport's draw, over the picture.</summary>
        public static void Draw(InnerViewport view)
        {
            if (!GameUI.Running || view != Viewport) return;
            GameUI.Desktop.DrawNested();

            // Where a tool's cursor is, so a screenshot shows what it points at: a small cross
            if (SimulatedCursorOnDesktop is { } c)
            {
                var renderer = Gui.Renderer;
                int white = ColorInt.ARGB(0.9f, 1f, 1f, 1f), dark = ColorInt.ARGB(0.8f, 0f, 0f, 0f);
                renderer.DrawBox(c.x - 7, c.y - 1, 15, 3, dark);
                renderer.DrawBox(c.x - 1, c.y - 7, 3, 15, dark);
                renderer.DrawBox(c.x - 6, c.y, 13, 1, white);
                renderer.DrawBox(c.x, c.y - 6, 1, 13, white);
            }
        }
    }
}
